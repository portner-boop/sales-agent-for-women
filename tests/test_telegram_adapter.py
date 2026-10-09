import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.types import Update

from app.application import Application
from app.modules.delivery.schemas import OutboxDraft
from app.modules.telegram.adapter import (
    DurablePoller,
    TelegramSender,
    configure_commands,
)
from app.shared.storage import Transaction
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
            "I'm Marko, a virtual companion. This is a test chat. /start starts or resumes a conversation, /stop stops messages, /privacy shows data terms, /forget deletes memory.",
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


async def test_interpreter_failure_respects_russian_input_with_english_profile(
    app: Application,
) -> None:
    app.model.responses["interpret"].append(ValueError("invalid response"))
    await app.telegram.ingest([update(1, "Привет. Хочется немного поболтать.", user=3)])
    await app.drain()
    assert app.sender.sent[0][1].startswith("Спасибо")
    assert (await conversation(app, 3)).first_reply_at is not None
    recovery = next(data for task, data in app.model.calls if task == "recover")
    assert recovery["APPLICATION_STATE"]["language"] == "ru"


async def test_interpreter_failure_keeps_prior_language_for_short_acknowledgement(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "Olá, vamos conversar.")])
    await app.drain()
    app.model.responses["interpret"].append(ValueError("invalid response"))
    app.model.responses["recover"].append(
        {"reply": "Certo, vamos continuar.", "dialogue_act": "acknowledge"}
    )
    await app.telegram.ingest([update(2, "ok")])
    await app.drain()
    assert app.sender.sent[-1] == (
        1,
        "Certo, vamos continuar",
    )


async def test_send_gate_rejects_long_question_goal_before_external_send(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "/start")])
    await app.drain()
    current = await conversation(app)
    sent = list(app.sender.sent)

    async def prepare(tx: Transaction) -> None:
        await app.delivery_repository.prepare(
            tx,
            current.snapshot,
            OutboxDraft(
                action_key="invalid-question-goal",
                kind="reply",
                text="Would you like a chat?",
                deadline=app.clock.now() + 30,
            ),
            app.clock.now(),
            proposal={"question_goal": "x" * 81},
        )

    await app.database.write(prepare)
    await app.drain()
    assert app.sender.sent == sent
    assert (
        await scalar(
            app,
            "SELECT error_code FROM outbox_messages WHERE action_key='invalid-question-goal'",
        )
        == "INVALID_QUESTION_GOAL"
    )


async def test_nonmutating_memory_operation_does_not_trigger_patch_validation(
    app: Application,
) -> None:
    app.model.responses["interpret"].append(
        {
            "intent": "chat",
            "dialogue_mode": "continuation",
            "language": "en",
            "memory_operations": [
                {"operation": "no_change", "predicate": "coffee", "sources": []}
            ],
        }
    )
    await app.telegram.ingest([update(1, "ok")])
    await app.drain()
    assert [task for task, _ in app.model.calls] == ["interpret", "generate"]
    assert await scalar(app, "SELECT COUNT(*) FROM memory_items") == 0
    assert (
        await scalar(app, "SELECT COUNT(*) FROM said_ledger WHERE action_kind='reply'")
        == 1
    )


async def test_telegram_menu_contains_only_four_commands() -> None:
    bot = AsyncMock(spec=Bot)
    await configure_commands(bot)
    commands = bot.set_my_commands.call_args.args[0]
    assert [item.command for item in commands] == ["start", "stop", "privacy", "forget"]
    bot.set_my_commands.assert_awaited_once()


async def test_native_telegram_typing_covers_the_continuation_pause(
    app: Application, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_reply_parts import prepare_pair

    first, _ = await prepare_pair(app)
    await app.delivery.send(first)
    bot = AsyncMock(spec=Bot)
    bot.id = 100
    started = asyncio.Event()

    async def send_action(**kwargs: object) -> bool:
        assert kwargs["chat_id"] == 1
        assert kwargs["action"] == "typing"
        started.set()
        return True

    bot.send_chat_action.side_effect = send_action
    bot.send_message.return_value = SimpleNamespace(message_id=100002)
    sender = TelegramSender(bot)
    typing = sender.typing(1)
    monkeypatch.setattr(sender, "typing", lambda chat_id: typing)
    app.delivery.sender = sender
    await app.delivery.run_once()
    await asyncio.wait_for(started.wait(), 1)
    assert typing.running
    bot.send_message.assert_not_awaited()
    app.clock.advance(4)
    await app.delivery.run_once()
    assert not typing.running
    bot.send_message.assert_awaited_once_with(1, "Что теперь на ужин?")
