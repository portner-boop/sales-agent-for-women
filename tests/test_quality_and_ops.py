import asyncio

from app.application import Application
from app.modules.delivery.entities import SendResult
from app.shared.control import request_control
from app.shared.operations import ProcessLock, restore_backup
from app.shared.storage import Transaction
from tests.helpers import FakeSender, conversation, scalar, update


async def test_stop_attempts_cancel_inflight_sender_and_records_unknown(
    app: Application,
) -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    class HangingSender:
        async def send_text(self, chat_id: int, text: str) -> SendResult:
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            raise AssertionError

    await app.telegram.ingest([update(1, "hello")])
    await app.turn_worker.run_once()
    app.delivery.sender = HangingSender()
    task = asyncio.create_task(app.delivery.run_once())
    await asyncio.wait_for(entered.wait(), 2)
    await app.telegram.ingest([update(2, "/stop")])
    await task
    assert cancelled.is_set()
    assert (
        await scalar(
            app, "SELECT COUNT(*) FROM outbox_messages WHERE status='delivery_unknown'"
        )
        == 1
    )
    assert await scalar(app, "SELECT COUNT(*) FROM said_ledger") == 0


async def test_sender_uses_remaining_deadline(app: Application) -> None:
    class SlowSender:
        async def send_text(self, chat_id: int, text: str) -> SendResult:
            await asyncio.sleep(1)
            return SendResult("delivered", 999)

    await app.telegram.ingest([update(1, "hello")])
    await app.turn_worker.run_once()

    async def shorten(tx: Transaction) -> None:
        await tx.execute(
            "UPDATE outbox_messages SET deadline=?", (app.clock.now() + 0.01,)
        )

    await app.database.write(shorten)
    app.delivery.sender = SlowSender()
    await app.delivery.run_once()
    assert await scalar(app, "SELECT status FROM outbox_messages") == "delivery_unknown"


async def test_live_control_metrics_backup_and_kill_switch(app: Application) -> None:
    await app.start_controls()
    await app.telegram.ingest([update(1, "hello")])
    await app.turn_worker.run_once()
    result = await request_control(
        app.settings.storage.directory, {"command": "kill-switch", "enabled": True}
    )
    assert result == {"kill_switch": True}
    await app.drain()
    assert not app.sender.sent
    assert (
        await request_control(app.settings.storage.directory, {"command": "metrics"})
    )["outbox_cancelled"] == 1
    assert "path" in await request_control(
        app.settings.storage.directory, {"command": "backup"}
    )


async def test_restore_never_resends_snapshot_ready_action(app: Application) -> None:
    await app.telegram.ingest([update(1, "hello")])
    await app.turn_worker.run_once()
    backup = await app.memory.backup()
    await app.delivery.run_once()
    assert len(app.sender.sent) == 1
    await app.close()
    with ProcessLock(app.settings.storage.directory):
        restore_backup(backup, app.settings.storage.directory)
    restarted = await Application.open(app.settings, FakeSender(), app.model, app.clock)
    try:
        await restarted.drain()
        assert not restarted.sender.sent
        assert (await conversation(restarted)).control_state == "paused"
        assert (await conversation(restarted)).preview_reply_count == 1
        assert (
            await scalar(restarted, "SELECT status FROM outbox_messages")
            == "delivery_unknown"
        )
    finally:
        await restarted.close()
