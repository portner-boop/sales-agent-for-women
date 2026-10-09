import json
from uuid import uuid4

from app.modules.telegram.entities import IncomingEvent
from app.shared.entities import ScopeContext
from app.shared.storage import Database, Transaction, fetchone


class TelegramRepository:
    def __init__(self, database: Database, channel: str) -> None:
        self.database = database
        self.channel = channel

    async def offset(self) -> int:
        async with self.database.read() as tx:
            row = await fetchone(
                tx,
                "SELECT next_offset FROM poller_offsets WHERE channel_account_id=?",
                (self.channel,),
            )
            return int(row["next_offset"]) if row else 0

    async def exists(self, tx: Transaction, update_id: int) -> bool:
        return (
            await fetchone(
                tx,
                "SELECT id FROM inbox_events WHERE channel_account_id=? AND update_id=?",
                (self.channel, update_id),
            )
            is not None
        )

    async def record(
        self,
        tx: Transaction,
        event: IncomingEvent,
        scope: ScopeContext | None,
        now: float,
    ) -> None:
        await tx.execute(
            "INSERT INTO inbox_events(id,channel_account_id,update_id,scope_id,epoch,raw_json,status,reason,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                uuid4().hex,
                self.channel,
                event.update_id,
                scope.scope_id if scope else None,
                scope.epoch if scope else None,
                json.dumps(event.raw, ensure_ascii=False),
                "quarantine" if event.kind == "quarantine" else "accepted",
                event.reason,
                now,
            ),
        )

    async def advance_offset(self, tx: Transaction, offset: int) -> None:
        await tx.execute(
            "INSERT INTO poller_offsets VALUES(?,?) ON CONFLICT(channel_account_id) DO UPDATE SET next_offset=MAX(next_offset,excluded.next_offset)",
            (self.channel, offset),
        )

    async def purge(self, tx: Transaction, scope_id: str) -> None:
        await tx.execute(
            "UPDATE inbox_events SET raw_json='{}',status='deleted' WHERE scope_id=?",
            (scope_id,),
        )
