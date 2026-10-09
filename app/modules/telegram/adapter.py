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
from aiogram.utils.chat_action import ChatActionSender

from app.modules.delivery.entities import SendResult
from app.modules.telegram.service import TelegramService

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

    def typing(self, chat_id: int) -> ChatActionSender:
        return ChatActionSender.typing(chat_id, self.bot)


class DurablePoller:
    def __init__(self, bot: Bot, ingress: TelegramService, poll_timeout: int) -> None:
        self.bot = bot
        self.ingress = ingress
        self.poll_timeout = poll_timeout

    async def run(self) -> None:
        webhook = await self.bot.get_webhook_info()
        if webhook.url:
            raise RuntimeError(
                "A webhook is configured; remove it before polling without dropping pending updates"
            )
        while True:
            offset = await self.ingress.repository.offset()
            try:
                updates = await self.bot.get_updates(
                    offset=offset,
                    timeout=self.poll_timeout,
                    allowed_updates=["message", "edited_message", "my_chat_member"],
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
            except (TelegramNetworkError, TelegramServerError):
                logger.warning("poller_transient_error offset=%s", offset)
                await asyncio.sleep(1)
