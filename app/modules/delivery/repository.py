import json
from typing import Any
from uuid import uuid4

from pydantic import TypeAdapter, ValidationError

from app.modules.conversations.schemas import QuestionGoal
from app.modules.delivery.entities import OutboxMessage
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
                    and (proposal or {}).get("origin") == "platform_redirect"
                )
                or turn_id is None
                or predecessor["turn_id"] != turn_id
                or predecessor["input_revision"] != snapshot.input_revision
                or predecessor["control_revision"] != snapshot.control_revision
                or predecessor["memory_revision"] != snapshot.memory_revision
                or predecessor["deadline"] != draft.deadline
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
            tx, "SELECT id FROM outbox_messages WHERE action_key=?", (draft.action_key,)
        )
        assert row is not None
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
            "SELECT o.*,c.chat_id,c.language,c.control_state,c.purchase_state,c.epoch AS current_epoch,c.input_revision AS current_input,c.control_revision AS current_control,c.memory_revision AS current_memory FROM outbox_messages o JOIN conversations c ON c.id=o.scope_id WHERE o.id=? AND o.status='ready'",
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
        if proposal.get("question_goal") is not None:
            try:
                TypeAdapter(QuestionGoal).validate_python(proposal["question_goal"])
            except ValidationError:
                await tx.execute(
                    "UPDATE outbox_messages SET status='cancelled',error_code='INVALID_QUESTION_GOAL' WHERE id=?",
                    (identifier,),
                )
                await self.cancel_successor(tx, identifier)
                return None
        stale = (
            row["epoch"] != row["current_epoch"]
            or row["control_revision"] != row["current_control"]
        )
        if row["kind"] in {"reply", "offer"} or row["turn_id"] is not None:
            stale = (
                stale
                or row["input_revision"] != row["current_input"]
                or row["memory_revision"] != row["current_memory"]
            )
        blocked = (
            kill_switch
            or stale
            or now >= row["deadline"]
            or (row["kind"] == "reply" and not can_reply)
        )
        blocked = blocked or (
            row["kind"] == "offer"
            and (
                row["control_state"] != "enabled"
                or row["purchase_state"] == "confirmed"
            )
        )
        if blocked:
            await tx.execute(
                "UPDATE outbox_messages SET status='cancelled',error_code='SEND_GATE' WHERE id=?",
                (identifier,),
            )
            await self.cancel_successor(tx, identifier)
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
            row["deadline"],
            row["predecessor_id"],
        )

    async def action_scope(self, identifier: str) -> str | None:
        async with self.database.read() as tx:
            row = await fetchone(
                tx, "SELECT scope_id FROM outbox_messages WHERE id=?", (identifier,)
            )
            return str(row["scope_id"]) if row else None

    async def interrupt_send(self, identifier: str) -> None:
        async def interrupt(tx: Transaction) -> None:
            result = await tx.execute(
                "UPDATE outbox_messages SET status='delivery_unknown',error_code='SEND_INTERRUPTED' WHERE id=? AND status='sending'",
                (identifier,),
            )
            if result.rowcount:
                await self.cancel_successor(tx, identifier)

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
    ) -> None:
        stored_status = (
            "ready"
            if status == "retryable_failure" and message.attempts < 3
            else "permanent_failure" if status == "retryable_failure" else status
        )
        await tx.execute(
            "UPDATE outbox_messages SET status=?,created_at=?,external_message_id=?,error_code=? WHERE id=? AND status='sending'",
            (
                stored_status,
                now + retry_after if status == "retryable_failure" else now,
                external_id,
                code,
                message.id,
            ),
        )
        if stored_status == "delivered":
            await tx.execute(
                "UPDATE outbox_messages SET created_at=?+COALESCE(message_delay_seconds,delay_seconds),deadline=CASE WHEN message_delay_seconds IS NOT NULL THEN MAX(deadline,?+message_delay_seconds+30) ELSE deadline END WHERE predecessor_id=? AND status='ready'",
                (now, now, message.id),
            )
        elif stored_status in {"permanent_failure", "delivery_unknown"}:
            await self.cancel_successor(tx, message.id)

    async def cancel_successor(self, tx: Transaction, identifier: str) -> None:
        await tx.execute(
            "WITH RECURSIVE descendants(id) AS (SELECT id FROM outbox_messages WHERE predecessor_id=? UNION SELECT o.id FROM outbox_messages o JOIN descendants d ON o.predecessor_id=d.id) UPDATE outbox_messages SET status='cancelled',error_code='PREDECESSOR_NOT_DELIVERED' WHERE id IN (SELECT id FROM descendants) AND status='ready'",
            (identifier,),
        )

    async def cancel_pending(
        self, tx: Transaction, scope: ScopeContext, code: str
    ) -> None:
        await require_scope(tx, scope)
        await tx.execute(
            "UPDATE outbox_messages SET status='cancelled',error_code=? WHERE scope_id=? AND epoch=? AND status='ready' AND (kind IN ('reply','offer') OR turn_id IS NOT NULL)",
            (code, scope.scope_id, scope.epoch),
        )

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
                "SELECT o.id,o.scope_id,c.chat_id FROM outbox_messages o LEFT JOIN outbox_messages p ON p.id=o.predecessor_id JOIN conversations c ON c.id=o.scope_id WHERE o.status='ready' AND (o.predecessor_id IS NULL OR p.status='delivered') AND (o.kind IN ('reply','offer') OR o.turn_id IS NOT NULL) AND o.created_at>? AND o.deadline>? AND o.epoch=c.epoch AND o.input_revision=c.input_revision AND o.control_revision=c.control_revision AND o.memory_revision=c.memory_revision AND c.control_state='enabled' AND c.purchase_state!='confirmed' AND NOT EXISTS(SELECT 1 FROM runtime_controls WHERE kill_switch=1)",
                (now, now),
            )
            return [(row["id"], row["scope_id"], row["chat_id"]) for row in rows]

    async def recover(self) -> None:
        async def recover(tx: Transaction) -> None:
            await tx.execute(
                "UPDATE outbox_messages SET status='delivery_unknown',error_code='PROCESS_INTERRUPTED' WHERE status='sending'"
            )
            await tx.execute(
                "WITH RECURSIVE blocked(id) AS (SELECT id FROM outbox_messages WHERE status IN ('cancelled','permanent_failure','delivery_unknown') UNION SELECT o.id FROM outbox_messages o JOIN blocked b ON o.predecessor_id=b.id) UPDATE outbox_messages SET status='cancelled',error_code='PREDECESSOR_NOT_DELIVERED' WHERE status='ready' AND id IN (SELECT id FROM blocked)"
            )

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
