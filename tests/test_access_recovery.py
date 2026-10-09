from app.application import Application
from app.modules.jobs.schemas import JobPayload
from app.shared.storage import Transaction
from tests.helpers import conversation, scalar, update


async def test_chat_continues_after_restart_and_long_pause(app: Application) -> None:
    await app.telegram.ingest([update(1, "Привет", user=3)])
    await app.drain()
    before = await conversation(app, 3)
    app.clock.advance(86400)
    await app.close()
    restarted = await Application.open(app.settings, app.sender, app.model, app.clock)
    try:
        await restarted.telegram.ingest([update(2, "Продолжим разговор", user=3)])
        await restarted.drain()
        current = await conversation(restarted, 3)
        assert current.id == before.id and current.epoch == before.epoch
        assert current.first_reply_at == before.first_reply_at
        assert restarted.commerce.can_reply(current)
        assert (
            await scalar(
                restarted, "SELECT COUNT(*) FROM said_ledger WHERE action_kind='reply'"
            )
            == 2
        )
    finally:
        await restarted.close()


async def test_removed_command_only_shows_four_commands(app: Application) -> None:
    await app.telegram.ingest([update(1, "Привет", user=3)])
    await app.drain()
    before = await conversation(app, 3)
    calls = len(app.model.calls)
    for number, command in enumerate(
        ["/purchase_test", "/whoami", "/offer", "/reset_test", "/forget_confirm"], 2
    ):
        await app.telegram.ingest([update(number, command, user=3)])
        await app.drain()
        assert app.sender.sent[-1][1] == (
            "Доступны /start — начать, /stop — остановить, /privacy — условия, /forget — удалить память."
        )
    current = await conversation(app, 3)
    assert current.epoch == before.epoch
    assert current.control_state == before.control_state
    assert current.purchase_state == before.purchase_state
    assert len(app.model.calls) == calls
    assert await scalar(app, "SELECT COUNT(*) FROM purchases") == 0


async def test_retry_of_skipped_input_preserves_original_and_gets_fresh_deadline(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "hello", user=3)])
    await app.drain()
    await app.telegram.ingest([update(2, "/stop", user=3)])
    await app.drain()
    await app.telegram.ingest([update(3, "coffee detail", user=3)])
    await app.drain()
    before = await conversation(app, 3)
    original = next(
        item
        for item in await app.conversations.messages(before.scope)
        if item.text == "coffee detail"
    )
    assert (
        await scalar(app, "SELECT status FROM turns WHERE id=?", (original.turn_id,))
        == "skipped"
    )
    app.clock.advance(600)
    await app.telegram.ingest([update(4, "/start", user=3)])
    await app.drain()

    async def retry(tx: Transaction) -> None:
        await tx.execute("UPDATE messages SET pending=1 WHERE id=?", (original.id,))
        await app.jobs.schedule(
            tx,
            "turn",
            "authorized-skipped-retry",
            app.clock.now(),
            app.clock.now(),
            before.scope,
            JobPayload(data={"retry_requested_at": app.clock.now()}),
        )

    await app.database.write(retry)
    await app.drain()
    current = next(
        item
        for item in await app.conversations.messages(before.scope)
        if item.id == original.id
    )
    assert current.received_at == original.received_at
    assert current.version == original.version
    assert current.text == original.text
    assert (
        await scalar(app, "SELECT COUNT(*) FROM agent_runs WHERE status='fallback'")
        == 0
    )
    assert (
        await scalar(app, "SELECT COUNT(*) FROM said_ledger WHERE action_kind='reply'")
        == 2
    )
