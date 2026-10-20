import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.types import Update

from app.application import Application
from app.modules.telegram.adapter import (
    DurablePoller,
    TelegramSender,
    configure_commands,
)
from tests.helpers import conversation, scalar, update


@pytest.mark.parametrize(
    "text,edited", [("hello", False), ("/start", False), ("corrected", True)]
)
async def test_poller_accepts_real_aiogram_updates(
    app: Application, text: str, edited: bool
) -> None:
    if edited:
        await app.telegram.ingest([update(1, "original", message_id=1)])
        await app.drain()
    raw = update(2, text, message_id=1, edited=edited)
    payload = raw["edited_message" if edited else "message"]
    payload["from"]["first_name"] = "Test"
    bot = AsyncMock(spec=Bot)
    bot.get_webhook_info.return_value = SimpleNamespace(url="")
    bot.get_updates.side_effect = [
        [Update.model_validate(raw)],
        asyncio.CancelledError(),
    ]
    with pytest.raises(asyncio.CancelledError):
        await DurablePoller(bot, app.telegram, 20).run()
    await app.drain()
    assert await app.telegram.repository.offset() == 3
    assert (
        await scalar(app, "SELECT COUNT(*) FROM inbox_events WHERE status='quarantine'")
        == 0
    )
    current = await conversation(app)
    assert current.external_user_id == 1
    if text == "/start":
        assert app.sender.sent[-1] == (
            1,
            "AI roleplay test: Marko is a fictional character who stays in character inside the scene. /start starts or resumes a conversation, /stop stops messages, /privacy shows data terms, /forget deletes memory.",
        )
        assert not app.model.calls
    else:
        messages = await app.conversations.messages(current.scope)
        message = next(item for item in messages if item.role == "user")
        assert message.text == text
        assert message.version == (2 if edited else 1)
        assert len(app.sender.sent) == (2 if edited else 1)


async def test_poller_accepts_real_membership_update_and_stops_chat(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "hello")])
    await app.drain()
    bot_user = {"id": 100, "is_bot": True, "first_name": "Test Bot"}
    raw = {
        "update_id": 2,
        "my_chat_member": {
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "Test"},
            "date": 1900000000,
            "old_chat_member": {"status": "member", "user": bot_user},
            "new_chat_member": {"status": "kicked", "user": bot_user, "until_date": 0},
        },
    }
    bot = AsyncMock(spec=Bot)
    bot.get_webhook_info.return_value = SimpleNamespace(url="")
    bot.get_updates.side_effect = [
        [Update.model_validate(raw)],
        asyncio.CancelledError(),
    ]
    with pytest.raises(asyncio.CancelledError):
        await DurablePoller(bot, app.telegram, 20).run()
    assert (await conversation(app)).control_state == "stopped"
    assert (
        await scalar(app, "SELECT COUNT(*) FROM inbox_events WHERE status='quarantine'")
        == 0
    )


@pytest.mark.parametrize("interrupt", [False, True])
async def test_typing_runs_during_generation_and_stops_afterwards(
    app: Application, interrupt: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    typing_started = asyncio.Event()
    bot = AsyncMock(spec=Bot)
    bot.id = 100

    async def send_action(**kwargs: object) -> bool:
        assert kwargs["chat_id"] == 1
        assert kwargs["action"] == "typing"
        typing_started.set()
        return True

    bot.send_chat_action.side_effect = send_action
    sender = TelegramSender(bot)
    typing = sender.typing(1)
    monkeypatch.setattr(sender, "typing", lambda chat_id: typing)
    app.sender = sender
    app.model.release = asyncio.Event()
    await app.telegram.ingest([update(1, "hello")])
    worker = asyncio.create_task(app.turn_worker.run_once())
    try:
        await asyncio.wait_for(typing_started.wait(), 2)
        await asyncio.wait_for(app.model.entered.wait(), 2)
        assert not worker.done()
        assert typing.running
        if interrupt:
            await app.telegram.ingest([update(2, "/stop")])
        else:
            app.model.release.set()
        await asyncio.wait_for(worker, 2)
        assert not typing.running
        await app.drain()
        assert await scalar(
            app, "SELECT COUNT(*) FROM said_ledger WHERE action_kind='reply'"
        ) == (0 if interrupt else 1)
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


async def test_telegram_menu_contains_only_four_commands() -> None:
    bot = AsyncMock(spec=Bot)
    await configure_commands(bot)
    commands = bot.set_my_commands.call_args.args[0]
    assert [item.command for item in commands] == ["start", "stop", "privacy", "forget"]
    bot.set_my_commands.assert_awaited_once()


@pytest.mark.parametrize(
    "language,identity_word",
    [("en", "fictional"), ("ru", "вымышленн"), ("es", "fictici"), ("pt", "fictíci")],
)
async def test_service_commands_explain_roleplay_without_model_calls(
    app: Application, language: str, identity_word: str
) -> None:
    for number, command in enumerate(["/start", "/privacy"], 1):
        await app.telegram.ingest([update(number, command, language=language)])
        await app.drain()
        text = app.sender.sent[-1][1]
        assert identity_word in text
        assert "Marko" in text or "Марко" in text
        assert "ИИ" in text or "AI" in text or "IA" in text
        assert "/stop" in text
    assert not app.model.calls
