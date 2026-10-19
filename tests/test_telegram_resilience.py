import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.exceptions import (
    TelegramForbiddenError,
    TelegramNetworkError,
    TelegramRetryAfter,
    TelegramServerError,
)
from aiogram.methods import GetUpdates, GetWebhookInfo, SendChatAction, SendMessage

from app.application import Application
from app.modules.telegram.adapter import DurablePoller, TelegramSender
from app.modules.telegram.typing import TelegramTyping
from tests.helpers import scalar, update


@pytest.mark.parametrize(
    "error_type", [TelegramNetworkError, TelegramServerError, TimeoutError]
)
async def test_typing_recovers_without_unobserved_background_error(
    error_type: type[Exception],
) -> None:
    bot = AsyncMock(spec=Bot)
    recovered = asyncio.Event()
    error = (
        TimeoutError()
        if error_type is TimeoutError
        else error_type(
            method=SendChatAction(chat_id=1, action="typing"), message="offline"
        )
    )
    attempts = 0

    async def action(**kwargs: object) -> bool:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise error
        recovered.set()
        return True

    bot.send_chat_action.side_effect = action
    indicator = TelegramTyping(bot, 1, interval=0.001)
    async with indicator:
        await asyncio.wait_for(recovered.wait(), 1)
        assert indicator.running
    assert not indicator.running
    assert indicator.task is None
    assert attempts >= 2


async def test_hung_typing_request_is_cancelled_without_blocking_exit() -> None:
    bot = AsyncMock(spec=Bot)
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def action(**kwargs: object) -> bool:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return True

    bot.send_chat_action.side_effect = action
    async with asyncio.timeout(1):
        async with TelegramTyping(bot, 1):
            await entered.wait()
    assert cancelled.is_set()


async def test_typing_timeout_is_retried_and_body_exception_is_preserved() -> None:
    bot = AsyncMock(spec=Bot)
    attempts = 0
    retried = asyncio.Event()

    async def action(**kwargs: object) -> bool:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            await asyncio.Event().wait()
        retried.set()
        return True

    bot.send_chat_action.side_effect = action
    indicator = TelegramTyping(bot, 1, interval=0.001, timeout=0.01)
    with pytest.raises(ValueError, match="main work"):
        async with indicator:
            await asyncio.wait_for(retried.wait(), 1)
            raise ValueError("main work")
    assert indicator.task is None


async def test_rejected_typing_does_not_reject_generated_answer(
    app: Application,
) -> None:
    bot = AsyncMock(spec=Bot)
    attempted = asyncio.Event()

    async def action(**kwargs: object) -> bool:
        attempted.set()
        raise TelegramForbiddenError(
            method=SendChatAction(chat_id=1, action="typing"),
            message="typing unavailable",
        )

    bot.send_chat_action.side_effect = action
    app.sender = TelegramSender(bot)
    app.model.release = asyncio.Event()
    await app.telegram.ingest([update(1, "hello")])
    worker = asyncio.create_task(app.turn_worker.run_once())
    try:
        await asyncio.wait_for(attempted.wait(), 1)
        await asyncio.wait_for(app.model.entered.wait(), 1)
        app.model.release.set()
        await asyncio.wait_for(worker, 2)
        await app.delivery.run_once()
        assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"
        assert await scalar(app, "SELECT status FROM outbox_messages") == "delivered"
        bot.send_chat_action.assert_awaited_once()
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.parametrize("phase", ["webhook", "updates"])
async def test_poller_recovers_read_errors_without_losing_offset(
    app: Application, phase: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    bot = AsyncMock(spec=Bot)
    method = GetWebhookInfo() if phase == "webhook" else GetUpdates()
    errors = [
        TelegramNetworkError(method=method, message="offline"),
        TimeoutError(),
        TelegramRetryAfter(method=method, message="slow down", retry_after=7),
    ]
    bot.get_webhook_info.return_value = SimpleNamespace(url="")
    bot.get_updates.side_effect = [[], asyncio.CancelledError()]
    if phase == "webhook":
        bot.get_webhook_info.side_effect = [*errors, SimpleNamespace(url="")]
    else:
        bot.get_updates.side_effect = [*errors, [], asyncio.CancelledError()]
    pauses = []
    original_sleep = asyncio.sleep

    async def pause(delay: float) -> None:
        pauses.append(delay)
        await original_sleep(0)

    monkeypatch.setattr("app.modules.telegram.adapter.asyncio.sleep", pause)
    original_offset = await app.telegram.repository.offset()
    with pytest.raises(asyncio.CancelledError):
        await DurablePoller(bot, app.telegram, 20).run()
    assert pauses == [1, 2, 7]
    assert await app.telegram.repository.offset() == original_offset
    assert all(
        call.kwargs["offset"] == original_offset
        for call in bot.get_updates.await_args_list
    )


async def test_existing_webhook_is_not_treated_as_transient(app: Application) -> None:
    bot = AsyncMock(spec=Bot)
    bot.get_webhook_info.return_value = SimpleNamespace(
        url="https://example.test/webhook"
    )
    with pytest.raises(RuntimeError, match="webhook"):
        await DurablePoller(bot, app.telegram, 20).run()
    bot.get_updates.assert_not_awaited()


async def test_uncertain_message_delivery_is_never_blindly_repeated() -> None:
    bot = AsyncMock(spec=Bot)
    bot.send_message.side_effect = TelegramNetworkError(
        method=SendMessage(chat_id=1, text="A reply"), message="connection lost"
    )
    result = await TelegramSender(bot).send_text(1, "A reply")
    assert result.status == "delivery_unknown"
    bot.send_message.assert_awaited_once_with(1, "A reply")


async def test_cancellation_of_main_work_also_stops_typing() -> None:
    bot = AsyncMock(spec=Bot)
    entered = asyncio.Event()
    indicator = TelegramTyping(bot, 1)

    async def work() -> None:
        async with indicator:
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(work())
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert indicator.task is None
