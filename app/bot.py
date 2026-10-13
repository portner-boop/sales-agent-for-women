import argparse
import asyncio
import hashlib
import logging
import signal
from contextlib import AsyncExitStack
from pathlib import Path

from aiogram import Bot
from aiogram.client.session.aiohttp import AiohttpSession

from app.application import Application
from app.modules.telegram.adapter import (
    DurablePoller,
    TelegramSender,
    configure_commands,
)
from app.shared.config import Settings, load_settings


async def run_bot(settings: Settings) -> None:
    settings.require_credentials()
    bot = Bot(
        settings.telegram.token.get_secret_value(),
        session=AiohttpSession(proxy=settings.telegram.proxy_url or None),
    )
    async with AsyncExitStack() as stack:
        stack.push_async_callback(bot.session.close)
        sender = TelegramSender(bot)
        channel = hashlib.sha256(
            settings.telegram.token.get_secret_value().split(":")[0].encode()
        ).hexdigest()[:20]
        application = await Application.open(settings, sender, channel=channel)
        stack.push_async_callback(application.close)
        await configure_commands(bot)
        await application.start_controls()
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for name in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(name, stop.set)
            stack.callback(loop.remove_signal_handler, name)
        tasks = [
            asyncio.create_task(
                DurablePoller(
                    bot, application.telegram, settings.telegram.poll_timeout
                ).run()
            ),
            asyncio.create_task(application.control_worker.run()),
            asyncio.create_task(application.background_worker.run()),
            asyncio.create_task(application.delivery.run()),
            asyncio.create_task(application.maintenance()),
        ]
        tasks.extend(
            asyncio.create_task(application.turn_worker.run())
            for _ in range(settings.runtime.llm_concurrency)
        )
        waiter = asyncio.create_task(stop.wait())
        try:
            done, _ = await asyncio.wait(
                [*tasks, waiter], return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                if task is not waiter:
                    await task
        finally:
            for task in [*tasks, waiter]:
                task.cancel()
            await asyncio.gather(*tasks, waiter, return_exceptions=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog="bot.py")
    parser.add_argument("--settings", type=Path)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    try:
        asyncio.run(run_bot(load_settings(arguments.settings)))
    except (ValueError, RuntimeError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
