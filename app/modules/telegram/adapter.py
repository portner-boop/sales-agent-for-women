import asyncio
import logging

from aiogram import Bot
from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramNetworkError,
    TelegramRetryAfter,
    TelegramServerError,
)
from aiogram.types import BotCommand

from app.modules.delivery.entities import SendResult
from app.modules.telegram.service import TelegramService
from app.modules.telegram.typing import TelegramTyping

logger = logging.getLogger(__name__)


async def configure_commands(bot: Bot) -> None:
    await bot.set_my_commands(
        [
            BotCommand(command="start", description="Начать переписку"),
            BotCommand(command="stop", description="Остановить переписку"),
            BotCommand(command="privacy", description="Условия обработки данных"),
            BotCommand(command="forget", description="Удалить память"),
        ]
    )


class TelegramSender:
    def __init__(self, bot: Bot) -> None:
        self.bot = bot

    async def send_text(self, chat_id: int, text: str) -> SendResult:
        try:
            message = await self.bot.send_message(chat_id, text)
            return SendResult("delivered", message.message_id)
        except TelegramRetryAfter as error:
            return SendResult(
                "retryable_failure",
                error_code="RATE_LIMIT",
                retry_after=error.retry_after,
            )
        except (TelegramBadRequest, TelegramForbiddenError):
            return SendResult("permanent_failure", error_code="TELEGRAM_REJECTED")
        except (TelegramNetworkError, TelegramServerError, TimeoutError):
            return SendResult("delivery_unknown", error_code="TELEGRAM_RESULT_UNKNOWN")

    def typing(self, chat_id: int) -> TelegramTyping:
        return TelegramTyping(self.bot, chat_id)


class DurablePoller:
    def __init__(self, bot: Bot, ingress: TelegramService, poll_timeout: int) -> None:
        self.bot = bot
        self.ingress = ingress
        self.poll_timeout = poll_timeout

    async def run(self) -> None:
        webhook_checked = False
        failures = 0
        while True:
            offset = await self.ingress.repository.offset()
            phase = "updates" if webhook_checked else "webhook"
            try:
                if not webhook_checked:
                    webhook = await self.bot.get_webhook_info(request_timeout=10)
                    if webhook.url:
                        raise RuntimeError(
                            "A webhook is configured; remove it before polling without dropping pending updates"
                        )
                    webhook_checked = True
                    phase = "updates"
                updates = await self.bot.get_updates(
                    offset=offset,
                    timeout=self.poll_timeout,
                    allowed_updates=["message", "edited_message", "my_chat_member"],
                    request_timeout=self.poll_timeout + 10,
                )
                if updates:
                    await self.ingress.ingest(
                        [
                            update.model_dump(
                                mode="json", exclude_none=True, by_alias=True
                            )
                            for update in updates
                        ]
                    )
                failures = 0
            except (
                TelegramNetworkError,
                TelegramServerError,
                TelegramRetryAfter,
                TimeoutError,
            ) as error:
                delay = min(30, 2 ** min(failures, 5))
                if isinstance(error, TelegramRetryAfter):
                    delay = max(delay, error.retry_after)
                failures += 1
                logger.warning(
                    "poller_transient_error phase=%s offset=%s type=%s retry_after=%s",
                    phase,
                    offset,
                    type(error).__name__,
                    delay,
                )
                await asyncio.sleep(delay)
