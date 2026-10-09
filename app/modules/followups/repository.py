from uuid import uuid4

from app.modules.conversations.entities import Turn
from app.modules.followups.entities import FollowupTicket
from app.shared.entities import ScopeContext, StaleStateError
from app.shared.storage import Database, Transaction, fetchall, fetchone, require_scope


class FollowupRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def schedule(
        self,
        tx: Transaction,
        scope: ScopeContext,
        anchor_id: str,
        now: float,
        run_at: float,
        expires_at: float,
    ) -> str | None:
        await require_scope(tx, scope)
        row = await fetchone(
            tx,
            "SELECT o.input_revision,o.control_revision,c.processed_sequence FROM outbox_messages o JOIN conversations c ON c.id=o.scope_id WHERE o.id=? AND o.scope_id=? AND o.epoch=? AND o.status='delivered' AND o.input_revision=c.input_revision AND o.control_revision=c.control_revision AND c.control_state='enabled' AND c.dialogue_mode NOT IN ('closing','service','repair') AND c.purchase_state!='confirmed' AND NOT EXISTS(SELECT 1 FROM runtime_controls WHERE kill_switch=1) AND NOT EXISTS(SELECT 1 FROM messages WHERE scope_id=c.id AND epoch=c.epoch AND pending=1) AND NOT EXISTS(SELECT 1 FROM memory_items WHERE scope_id=c.id AND epoch=c.epoch AND validity_status='current' AND kind='boundary' AND predicate IN ('proactive_permission','followup_permission','contact_permission') AND value_json='false')",
            (anchor_id, scope.scope_id, scope.epoch),
        )
        if row is None:
            return None
        existing = await fetchone(
            tx,
            "SELECT id FROM followups WHERE scope_id=? AND epoch=? AND input_revision=?",
            (scope.scope_id, scope.epoch, row["input_revision"]),
        )
        if existing is not None:
            return None
        identifier, turn_id = uuid4().hex, uuid4().hex
        await tx.execute(
            "INSERT INTO turns VALUES(?,?,?,'scheduled',?,?,?,?)",
            (
                turn_id,
                scope.scope_id,
                scope.epoch,
                row["input_revision"],
                row["processed_sequence"],
                row["processed_sequence"],
                now,
            ),
        )
        await tx.execute(
            "INSERT INTO followups(id,scope_id,epoch,input_revision,control_revision,anchor_id,turn_id,run_at,expires_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                identifier,
                scope.scope_id,
                scope.epoch,
                row["input_revision"],
                row["control_revision"],
                anchor_id,
                turn_id,
                run_at,
                expires_at,
            ),
        )
        return identifier

    async def start(
        self, scope: ScopeContext, identifier: str, now: float
    ) -> FollowupTicket | None:
        async def start(tx: Transaction) -> FollowupTicket | None:
            await require_scope(tx, scope)
            row = await fetchone(
                tx,
                "SELECT f.*,c.input_revision AS current_input,c.control_revision AS current_control,c.control_state,c.purchase_state,o.status AS anchor_status FROM followups f JOIN conversations c ON c.id=f.scope_id JOIN outbox_messages o ON o.id=f.anchor_id WHERE f.id=? AND f.scope_id=? AND f.epoch=? AND f.status IN ('pending','generating')",
                (identifier, scope.scope_id, scope.epoch),
            )
            if row is None:
                return None
            if row["run_at"] > now:
                raise ValueError("Follow-up job ran before its timer")
            killed = await fetchone(
                tx, "SELECT kill_switch FROM runtime_controls WHERE id=1"
            )
            blocked = await fetchone(
                tx,
                "SELECT id FROM memory_items WHERE scope_id=? AND epoch=? AND validity_status='current' AND kind='boundary' AND predicate IN ('proactive_permission','followup_permission','contact_permission') AND value_json='false' LIMIT 1",
                (scope.scope_id, scope.epoch),
            )
            valid = (
                now < row["expires_at"]
                and row["input_revision"] == row["current_input"]
                and row["control_revision"] == row["current_control"]
                and row["control_state"] == "enabled"
                and row["purchase_state"] != "confirmed"
                and row["anchor_status"] == "delivered"
                and killed is not None
                and not killed["kill_switch"]
                and blocked is None
            )
            if not valid:
                await self.finish(tx, identifier, "skipped", "PREFLIGHT")
                return None
            await tx.execute(
                "UPDATE followups SET status='generating' WHERE id=?", (identifier,)
            )
            await tx.execute(
                "UPDATE turns SET status='processing' WHERE id=?", (row["turn_id"],)
            )
            return FollowupTicket(
                identifier,
                Turn(row["turn_id"], scope, row["input_revision"], ()),
                row["control_revision"],
                row["anchor_id"],
                row["run_at"],
                row["expires_at"],
            )

        return await self.database.write(start)

    async def prepared(self, tx: Transaction, ticket: FollowupTicket) -> None:
        row = await fetchone(
            tx,
            "SELECT status FROM followups WHERE id=? AND scope_id=? AND epoch=?",
            (ticket.id, ticket.turn.scope.scope_id, ticket.turn.scope.epoch),
        )
        if row is None or row["status"] != "generating":
            raise StaleStateError("Follow-up was cancelled before preparation")
        await self.finish(tx, ticket.id, "prepared")

    async def finish(
        self, tx: Transaction, identifier: str, status: str, reason: str | None = None
    ) -> None:
        await tx.execute(
            "UPDATE followups SET status=?,reason=? WHERE id=? AND status IN ('pending','generating')",
            (status, reason, identifier),
        )
        if status in {"cancelled", "skipped"}:
            await tx.execute(
                "UPDATE turns SET status=? WHERE id=(SELECT turn_id FROM followups WHERE id=?) AND status IN ('scheduled','processing')",
                (status, identifier),
            )

    async def cancel(self, tx: Transaction, scope: ScopeContext) -> None:
        await require_scope(tx, scope)
        await tx.execute(
            "UPDATE turns SET status='cancelled' WHERE id IN (SELECT turn_id FROM followups WHERE scope_id=? AND epoch=? AND status IN ('pending','generating'))",
            (scope.scope_id, scope.epoch),
        )
        await tx.execute(
            "UPDATE followups SET status='cancelled',reason='NEW_INPUT' WHERE scope_id=? AND epoch=? AND status IN ('pending','generating')",
            (scope.scope_id, scope.epoch),
        )

    async def purge(self, tx: Transaction, scope_id: str) -> None:
        await tx.execute("DELETE FROM followups WHERE scope_id=?", (scope_id,))

    async def metrics(self) -> dict[str, int]:
        async with self.database.read() as tx:
            rows = await fetchall(
                tx, "SELECT status,COUNT(*) n FROM followups GROUP BY status"
            )
            outcomes = await fetchall(
                tx,
                "SELECT status,COUNT(*) n FROM outbox_messages WHERE json_extract(proposal_json,'$.origin')='followup' GROUP BY status",
            )
        return {
            **{f'followups_{row["status"]}': row["n"] for row in rows},
            **{f'followup_delivery_{row["status"]}': row["n"] for row in outcomes},
        }
