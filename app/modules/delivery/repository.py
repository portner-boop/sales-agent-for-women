import json
from time import time
from typing import Any
from uuid import uuid4

from app.modules.delivery.entities import OutboxMessage
from app.modules.delivery.observations import project_research_delivery
from app.modules.delivery.schemas import OutboxDraft
from app.shared.entities import RevisionSnapshot, ScopeContext, StaleStateError
from app.shared.storage import (
    Database,
    Transaction,
    fetchall,
    fetchone,
    require_scope,
)


class DeliveryRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def prepare(
        self,
        tx: Transaction,
        snapshot: RevisionSnapshot,
        draft: OutboxDraft,
        now: float,
        proposal: dict[str, Any] | None = None,
        turn_id: str | None = None,
    ) -> str:
        await require_scope(tx, snapshot.scope)
        if draft.predecessor_id is not None:
            predecessor = await fetchone(
                tx,
                "SELECT * FROM outbox_messages WHERE id=? AND scope_id=? AND epoch=?",
                (draft.predecessor_id, snapshot.scope.scope_id, snapshot.scope.epoch),
            )
            if (
                predecessor is None
                or predecessor["kind"] != "reply"
                or not (
                    draft.kind == "reply"
                    or draft.kind == "service"
                    and (proposal or {}).get("origin")
                    in {"platform_redirect", "preview_closing"}
                )
                or turn_id is None
                or predecessor["turn_id"] != turn_id
                or predecessor["input_revision"] != snapshot.input_revision
                or predecessor["control_revision"] != snapshot.control_revision
                or predecessor["memory_revision"] != snapshot.memory_revision
            ):
                raise StaleStateError("Continuation requires its scoped reply")
        identifier = uuid4().hex
        await tx.execute(
            "INSERT OR IGNORE INTO outbox_messages(id,action_key,scope_id,epoch,input_revision,control_revision,memory_revision,turn_id,kind,text,proposal_json,deadline,created_at,predecessor_id,message_delay_seconds) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                identifier,
                draft.action_key,
                snapshot.scope.scope_id,
                snapshot.scope.epoch,
                snapshot.input_revision,
                snapshot.control_revision,
                snapshot.memory_revision,
                turn_id,
                draft.kind,
                draft.text,
                json.dumps(proposal or {}, ensure_ascii=False),
                draft.deadline,
                now + draft.delay_seconds if draft.predecessor_id is None else now,
                draft.predecessor_id,
                draft.delay_seconds,
            ),
        )
        row = await fetchone(
            tx,
            "SELECT id,status FROM outbox_messages WHERE action_key=?",
            (draft.action_key,),
        )
        assert row is not None
        if row["status"] in {"ready", "sending"} and (
            (proposal or {}).get("platform_transition") is True
            or (proposal or {}).get("origin") == "platform_redirect"
        ):
            await tx.execute(
                "UPDATE conversations SET invitation_state='planned' WHERE id=? AND epoch=? AND invitation_state='none'",
                (snapshot.scope.scope_id, snapshot.scope.epoch),
            )
        return str(row["id"])

    async def ready_ids(self, now: float) -> list[str]:
        async with self.database.read() as tx:
            rows = await fetchall(
                tx,
                "SELECT o.id FROM outbox_messages o LEFT JOIN outbox_messages p ON p.id=o.predecessor_id WHERE o.status='ready' AND (o.created_at<=? OR o.deadline<=?) AND (o.predecessor_id IS NULL OR p.status='delivered') ORDER BY o.created_at,o.rowid LIMIT 32",
                (now, now),
            )
            return [str(row["id"]) for row in rows]

    async def gate(
        self,
        tx: Transaction,
        identifier: str,
        now: float,
        can_reply: bool,
        kill_switch: bool,
    ) -> OutboxMessage | None:
        row = await fetchone(
            tx,
            "SELECT o.*,c.chat_id,c.language,c.control_state,c.epoch AS current_epoch,c.input_revision AS current_input,c.control_revision AS current_control,c.memory_revision AS current_memory FROM outbox_messages o JOIN conversations c ON c.id=o.scope_id WHERE o.id=? AND o.status='ready'",
            (identifier,),
        )
        if row is None:
            return None
        if row["predecessor_id"] is not None:
            predecessor = await fetchone(
                tx,
                "SELECT status FROM outbox_messages WHERE id=? AND scope_id=? AND epoch=?",
                (row["predecessor_id"], row["scope_id"], row["epoch"]),
            )
            if predecessor is None or predecessor["status"] in {
                "cancelled",
                "permanent_failure",
                "delivery_unknown",
            }:
                await tx.execute(
                    "UPDATE outbox_messages SET status='cancelled',error_code='PREDECESSOR_NOT_DELIVERED' WHERE id=?",
                    (identifier,),
                )
                await self.cancel_successor(tx, identifier)
                return None
            if predecessor["status"] != "delivered":
                return None
        proposal = json.loads(row["proposal_json"])
        stale = (
            row["epoch"] != row["current_epoch"]
            or row["control_revision"] != row["current_control"]
        )
        if row["kind"] == "reply" or row["turn_id"] is not None:
            stale = (
                stale
                or row["input_revision"] != row["current_input"]
                or row["memory_revision"] != row["current_memory"]
            )
        continuation_allowed = False
        deadline = float(row["deadline"])
        if row["predecessor_id"] is not None:
            continuation_allowed = (
                row["control_state"] == "enabled"
                and await fetchone(
                    tx,
                    "SELECT 1 FROM outgoing_dialogue_turns WHERE turn_id=? AND scope_id=? AND epoch=? AND outcome='delivered'",
                    (row["turn_id"], row["scope_id"], row["epoch"]),
                )
                is not None
            )
        if (
            now >= row["deadline"]
            and not stale
            and not kill_switch
            and (can_reply or continuation_allowed or row["kind"] == "service")
            and not proposal.get("hard_deadline")
            and row["attempts"] < 3
        ):
            await tx.execute(
                "UPDATE outbox_messages SET deadline=? WHERE id=?",
                (now + 30, identifier),
            )
            deadline = now + 30
        blocked = (
            kill_switch
            or stale
            or now >= deadline
            or (row["kind"] == "reply" and not can_reply and not continuation_allowed)
        )
        if blocked:
            await tx.execute(
                "UPDATE outbox_messages SET status='cancelled',error_code='SEND_GATE' WHERE id=?",
                (identifier,),
            )
            await self.cancel_successor(tx, identifier)
            await self._release_invitation(tx, row["scope_id"], row["epoch"])
            return None
        if now < row["created_at"]:
            return None
        text = row["text"]
        await tx.execute(
            "UPDATE outbox_messages SET status='sending',attempts=attempts+1 WHERE id=?",
            (identifier,),
        )
        return OutboxMessage(
            identifier,
            ScopeContext(row["scope_id"], row["epoch"]),
            row["chat_id"],
            row["turn_id"],
            row["kind"],
            text,
            proposal,
            row["attempts"] + 1,
            deadline,
            row["predecessor_id"],
        )

    async def action_scope(self, identifier: str) -> str | None:
        async with self.database.read() as tx:
            row = await fetchone(
                tx, "SELECT scope_id FROM outbox_messages WHERE id=?", (identifier,)
            )
            return str(row["scope_id"]) if row else None

    async def interrupt_send(self, identifier: str, now: float | None = None) -> None:
        async def interrupt(tx: Transaction) -> None:
            result = await tx.execute(
                "UPDATE outbox_messages SET status='delivery_unknown',error_code='SEND_INTERRUPTED' WHERE id=? AND status='sending'",
                (identifier,),
            )
            if result.rowcount:
                await self.cancel_successor(tx, identifier)
                await self._record_outcome(
                    tx,
                    identifier,
                    "delivery_unknown",
                    now if now is not None else time(),
                )
                scope = await fetchone(
                    tx,
                    "SELECT scope_id,epoch FROM outbox_messages WHERE id=?",
                    (identifier,),
                )
                assert scope is not None
                await self._release_invitation(tx, scope["scope_id"], scope["epoch"])

        await self.database.write(interrupt)

    async def outcome(
        self,
        tx: Transaction,
        message: OutboxMessage,
        status: str,
        now: float,
        external_id: int | None,
        code: str | None,
        retry_after: float,
    ) -> bool:
        stored_status = (
            "ready"
            if status == "retryable_failure" and message.attempts < 3
            else "permanent_failure" if status == "retryable_failure" else status
        )
        cursor = await tx.execute(
            "UPDATE outbox_messages SET status=?,created_at=?,external_message_id=?,error_code=? WHERE id=? AND status='sending'",
            (
                stored_status,
                now + retry_after if status == "retryable_failure" else now,
                external_id,
                code,
                message.id,
            ),
        )
        if not cursor.rowcount:
            return False
        if stored_status == "delivered":
            await tx.execute(
                "UPDATE outbox_messages SET created_at=?+COALESCE(message_delay_seconds,delay_seconds),deadline=CASE WHEN message_delay_seconds IS NOT NULL AND COALESCE(json_extract(proposal_json,'$.hard_deadline'),0)=0 THEN MAX(deadline,?+message_delay_seconds+30) ELSE deadline END WHERE predecessor_id=? AND status='ready'",
                (now, now, message.id),
            )
        elif stored_status in {"permanent_failure", "delivery_unknown"}:
            await self.cancel_successor(tx, message.id)
            await self._release_invitation(
                tx, message.scope.scope_id, message.scope.epoch
            )
        await self._record_outcome(tx, message.id, stored_status, now)
        return True

    async def _record_outcome(
        self, tx: Transaction, identifier: str, status: str, now: float
    ) -> None:
        row = await fetchone(
            tx,
            "SELECT o.* FROM outbox_messages o JOIN conversations c ON c.id=o.scope_id AND c.epoch=o.epoch WHERE o.id=?",
            (identifier,),
        )
        if row is None:
            return
        if status in {"delivered", "permanent_failure", "delivery_unknown"}:
            await tx.execute(
                "UPDATE outbox_messages SET proposal_json=json_set(proposal_json,'$.delivery_finished_at',COALESCE(json_extract(proposal_json,'$.delivery_finished_at'),?)) WHERE id=?",
                (now, identifier),
            )
        proposal = json.loads(row["proposal_json"])
        run_id = proposal.get("research_run_id")
        if isinstance(run_id, str):
            await project_research_delivery(tx, run_id)
        if (
            status in {"delivered", "delivery_unknown"}
            and row["kind"] == "reply"
            and row["turn_id"] is not None
            and proposal.get("origin") != "followup"
        ):
            cursor = await tx.execute(
                "INSERT OR IGNORE INTO outgoing_dialogue_turns(turn_id,scope_id,epoch,outcome,recorded_at) VALUES(?,?,?,?,?)",
                (
                    row["turn_id"],
                    row["scope_id"],
                    row["epoch"],
                    "delivered" if status == "delivered" else "unknown",
                    now,
                ),
            )
            if cursor.rowcount:
                await tx.execute(
                    "UPDATE conversations SET preview_reply_count=preview_reply_count+1,pending_outgoing_turn_id=CASE WHEN ?='delivered' THEN ? ELSE pending_outgoing_turn_id END WHERE id=? AND epoch=?",
                    (status, row["turn_id"], row["scope_id"], row["epoch"]),
                )
        invitation = (
            proposal.get("origin") == "platform_redirect"
            or proposal.get("platform_transition") is True
        )
        if status == "delivered" and invitation:
            await tx.execute(
                "UPDATE conversations SET invitation_state='delivered',preview_closed_at=COALESCE(preview_closed_at,?) WHERE id=? AND epoch=?",
                (now, row["scope_id"], row["epoch"]),
            )
        elif status == "delivery_unknown" and invitation:
            await tx.execute(
                "UPDATE conversations SET invitation_state='unknown' WHERE id=? AND epoch=? AND invitation_state!='delivered'",
                (row["scope_id"], row["epoch"]),
            )
        if row["turn_id"] is not None:
            turn_status = {
                "permanent_failure": "delivery_failed",
                "delivery_unknown": "delivery_unknown",
            }.get(status)
            if turn_status is not None:
                await tx.execute(
                    "UPDATE turns SET status=? WHERE id=? AND scope_id=? AND epoch=?",
                    (turn_status, row["turn_id"], row["scope_id"], row["epoch"]),
                )
            elif status == "delivered":
                await tx.execute(
                    "UPDATE turns SET status='delivered' WHERE id=? AND scope_id=? AND epoch=? AND NOT EXISTS(SELECT 1 FROM outbox_messages WHERE turn_id=? AND scope_id=? AND epoch=? AND status!='delivered')",
                    (
                        row["turn_id"],
                        row["scope_id"],
                        row["epoch"],
                        row["turn_id"],
                        row["scope_id"],
                        row["epoch"],
                    ),
                )

    async def _release_invitation(
        self, tx: Transaction, scope_id: str, epoch: int
    ) -> None:
        await tx.execute(
            "UPDATE conversations SET invitation_state='none' WHERE id=? AND epoch=? AND invitation_state='planned' AND NOT EXISTS(SELECT 1 FROM outbox_messages WHERE scope_id=? AND epoch=? AND status IN ('ready','sending') AND (json_extract(proposal_json,'$.origin')='platform_redirect' OR json_extract(proposal_json,'$.platform_transition')=1))",
            (scope_id, epoch, scope_id, epoch),
        )

    async def cancel_successor(self, tx: Transaction, identifier: str) -> None:
        await tx.execute(
            "WITH RECURSIVE descendants(id) AS (SELECT id FROM outbox_messages WHERE predecessor_id=? UNION SELECT o.id FROM outbox_messages o JOIN descendants d ON o.predecessor_id=d.id) UPDATE outbox_messages SET status='cancelled',error_code='PREDECESSOR_NOT_DELIVERED' WHERE id IN (SELECT id FROM descendants) AND status='ready'",
            (identifier,),
        )
        row = await fetchone(
            tx, "SELECT scope_id,epoch FROM outbox_messages WHERE id=?", (identifier,)
        )
        if row is not None:
            await self._release_invitation(tx, row["scope_id"], row["epoch"])

    async def cancel_pending(
        self, tx: Transaction, scope: ScopeContext, code: str
    ) -> None:
        await require_scope(tx, scope)
        await tx.execute(
            "UPDATE outbox_messages SET status='cancelled',error_code=? WHERE scope_id=? AND epoch=? AND status='ready' AND (kind='reply' OR turn_id IS NOT NULL)",
            (code, scope.scope_id, scope.epoch),
        )
        await self._release_invitation(tx, scope.scope_id, scope.epoch)

    async def cancel_followups(self, tx: Transaction, scope: ScopeContext) -> None:
        await require_scope(tx, scope)
        await tx.execute(
            "UPDATE outbox_messages SET status='cancelled',error_code='NEW_INPUT' WHERE scope_id=? AND epoch=? AND status='ready' AND json_extract(proposal_json,'$.origin')='followup'",
            (scope.scope_id, scope.epoch),
        )

    async def has_successor(self, tx: Transaction, identifier: str) -> bool:
        return (
            await fetchone(
                tx,
                "SELECT id FROM outbox_messages WHERE predecessor_id=? AND status IN ('ready','sending')",
                (identifier,),
            )
            is not None
        )

    async def next_ready_at(self, scope: ScopeContext) -> float | None:
        async with self.database.read() as tx:
            await require_scope(tx, scope)
            row = await fetchone(
                tx,
                "SELECT MIN(MIN(o.created_at,o.deadline)) AS due FROM outbox_messages o LEFT JOIN outbox_messages p ON p.id=o.predecessor_id WHERE o.scope_id=? AND o.epoch=? AND o.status='ready' AND (o.predecessor_id IS NULL OR p.status='delivered')",
                (scope.scope_id, scope.epoch),
            )
            return (
                float(row["due"])
                if row is not None and row["due"] is not None
                else None
            )

    async def waiting_parts(self, now: float) -> list[tuple[str, str, int]]:
        async with self.database.read() as tx:
            rows = await fetchall(
                tx,
                "SELECT o.id,o.scope_id,c.chat_id FROM outbox_messages o LEFT JOIN outbox_messages p ON p.id=o.predecessor_id JOIN conversations c ON c.id=o.scope_id WHERE o.status='ready' AND (o.predecessor_id IS NULL OR p.status='delivered') AND (o.kind='reply' OR o.turn_id IS NOT NULL) AND o.created_at>? AND o.deadline>? AND o.epoch=c.epoch AND o.input_revision=c.input_revision AND o.control_revision=c.control_revision AND o.memory_revision=c.memory_revision AND c.control_state='enabled' AND NOT EXISTS(SELECT 1 FROM runtime_controls WHERE kill_switch=1)",
                (now, now),
            )
            return [(row["id"], row["scope_id"], row["chat_id"]) for row in rows]

    async def recover(self, now: float | None = None) -> None:
        async def recover(tx: Transaction) -> None:
            sending = await fetchall(
                tx,
                "SELECT id FROM outbox_messages WHERE status IN ('sending','delivery_unknown')",
            )
            await tx.execute(
                "UPDATE outbox_messages SET status='delivery_unknown',error_code='PROCESS_INTERRUPTED' WHERE status='sending'"
            )
            await tx.execute(
                "WITH RECURSIVE blocked(id) AS (SELECT id FROM outbox_messages WHERE status IN ('cancelled','permanent_failure','delivery_unknown') UNION SELECT o.id FROM outbox_messages o JOIN blocked b ON o.predecessor_id=b.id) UPDATE outbox_messages SET status='cancelled',error_code='PREDECESSOR_NOT_DELIVERED' WHERE status='ready' AND id IN (SELECT id FROM blocked)"
            )
            for row in sending:
                await self._record_outcome(
                    tx,
                    row["id"],
                    "delivery_unknown",
                    now if now is not None else time(),
                )
            planned = await fetchall(
                tx,
                "SELECT id,epoch FROM conversations WHERE invitation_state='planned'",
            )
            for row in planned:
                await self._release_invitation(tx, row["id"], row["epoch"])

        await self.database.write(recover)

    async def purge(self, tx: Transaction, scope_id: str) -> None:
        await tx.execute("DELETE FROM outbox_messages WHERE scope_id=?", (scope_id,))

    async def kill_switch(self, enabled: bool) -> None:
        async def update(tx: Transaction) -> None:
            await tx.execute(
                "UPDATE runtime_controls SET kill_switch=? WHERE id=1", (int(enabled),)
            )
            if enabled:
                await tx.execute(
                    "UPDATE outbox_messages SET status='cancelled',error_code='KILL_SWITCH' WHERE status='ready'"
                )
                rows = await fetchall(
                    tx,
                    "SELECT id,epoch FROM conversations WHERE invitation_state='planned'",
                )
                for row in rows:
                    await self._release_invitation(tx, row["id"], row["epoch"])

        await self.database.write(update)

    async def is_killed(self, tx: Transaction) -> bool:
        row = await fetchone(tx, "SELECT kill_switch FROM runtime_controls WHERE id=1")
        return row is None or bool(row["kill_switch"])

    async def metrics(self) -> dict[str, int]:
        async with self.database.read() as tx:
            rows = await fetchall(
                tx,
                "SELECT status,COUNT(*) AS count FROM outbox_messages GROUP BY status",
            )
            return {f"outbox_{row['status']}": row["count"] for row in rows}
