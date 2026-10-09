import json
from uuid import uuid4

from app.modules.commerce.entities import Offer
from app.modules.commerce.schemas import HandoffContext
from app.shared.entities import ScopeContext
from app.shared.storage import Database, Transaction, fetchone, require_scope


class CommerceRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def transition_allowed(self, tx: Transaction, scope: ScopeContext) -> bool:
        await require_scope(tx, scope)
        blocked = await fetchone(
            tx,
            "SELECT id FROM memory_items WHERE scope_id=? AND epoch=? AND subject_id='user_self' AND validity_status='current' AND kind='boundary' AND ((predicate='offer_refusal' AND value_json='true') OR (predicate='contact_permission' AND value_json='false')) LIMIT 1",
            (scope.scope_id, scope.epoch),
        )
        return blocked is None

    async def count_reply(self, tx: Transaction, scope: ScopeContext) -> None:
        await require_scope(tx, scope)
        await tx.execute(
            "UPDATE conversations SET preview_reply_count=preview_reply_count+1 WHERE id=? AND preview_closed_at IS NULL",
            (scope.scope_id,),
        )

    async def close_preview(
        self, tx: Transaction, scope: ScopeContext, now: float
    ) -> None:
        await require_scope(tx, scope)
        await tx.execute(
            "UPDATE conversations SET preview_closed_at=COALESCE(preview_closed_at,?) WHERE id=?",
            (now, scope.scope_id),
        )

    async def purge(self, tx: Transaction, scope_id: str) -> None:
        await tx.execute("DELETE FROM handoffs WHERE scope_id=?", (scope_id,))

    async def initialize(self, minutes: int) -> None:
        async def initialize(tx: Transaction) -> None:
            await tx.execute(
                "INSERT INTO offers VALUES('demo','Test package',?,1) ON CONFLICT(id) DO UPDATE SET duration_minutes=excluded.duration_minutes",
                (minutes,),
            )

        await self.database.write(initialize)

    async def offer(self) -> Offer:
        async with self.database.read() as tx:
            row = await fetchone(tx, "SELECT * FROM offers WHERE id='demo'")
            assert row is not None
            return Offer(row["id"], row["title"], row["duration_minutes"])

    async def confirm_mock(
        self,
        tx: Transaction,
        scope: ScopeContext,
        event_key: str,
        context: HandoffContext,
        now: float,
    ) -> bool:
        await require_scope(tx, scope)
        existing = await fetchone(
            tx, "SELECT id FROM purchases WHERE event_key=?", (event_key,)
        )
        if existing:
            return False
        identifier = uuid4().hex
        await tx.execute(
            "INSERT INTO purchases VALUES(?,?,'demo',?,1,?)",
            (identifier, scope.scope_id, event_key, now),
        )
        await tx.execute(
            "INSERT INTO handoffs VALUES(?,?,?,'requested',?,1)",
            (
                uuid4().hex,
                scope.scope_id,
                identifier,
                json.dumps({"simulation": True, "context": context.model_dump()}),
            ),
        )
        return True

    async def record_handoff(
        self, tx: Transaction, scope: ScopeContext, text: str
    ) -> None:
        await require_scope(tx, scope)
        row = await fetchone(
            tx,
            "SELECT payload_json FROM handoffs WHERE scope_id=? AND status='requested'",
            (scope.scope_id,),
        )
        if row is None:
            return
        payload = json.loads(row["payload_json"])
        payload["service_status"] = text
        await tx.execute(
            "UPDATE handoffs SET status='delivered',payload_json=? WHERE scope_id=? AND status='requested'",
            (json.dumps(payload), scope.scope_id),
        )
