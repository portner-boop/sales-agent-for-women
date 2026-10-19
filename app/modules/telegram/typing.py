import asyncio
import logging
from types import TracebackType

from aiogram import Bot
from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramNetworkError,
    TelegramRetryAfter,
    TelegramServerError,
)

logger = logging.getLogger(__name__)


class TelegramTyping:
    def __init__(
        self, bot: Bot, chat_id: int, interval: float = 5, timeout: float = 3
    ) -> None:
        self.bot = bot
        self.chat_id = chat_id
        self.interval = interval
        self.timeout = timeout
        self.task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    async def __aenter__(self) -> "TelegramTyping":
        if self.task is not None:
            raise RuntimeError("Typing context already entered")
        self.task = asyncio.create_task(self._run())
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        task, self.task = self.task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _run(self) -> None:
        while True:
            delay = self.interval
            try:
                async with asyncio.timeout(self.timeout):
                    await self.bot.send_chat_action(
                        chat_id=self.chat_id,
                        action="typing",
                    )
            except TelegramRetryAfter as error:
                delay = max(delay, error.retry_after)
                logger.warning("typing_rate_limited retry_after=%s", delay)
            except (TelegramNetworkError, TelegramServerError, TimeoutError) as error:
                logger.warning("typing_transient_error type=%s", type(error).__name__)
            except (TelegramBadRequest, TelegramForbiddenError) as error:
                logger.warning("typing_rejected type=%s", type(error).__name__)
                return
            except Exception as error:
                logger.warning("typing_failed type=%s", type(error).__name__)
                return
            await asyncio.sleep(delay)
