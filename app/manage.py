import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from app.application import Application
from app.modules.agent.entities import SchemaT
from app.modules.delivery.entities import SendResult
from app.modules.dialogue.runner import EvaluationOptions, run_evaluation
from app.shared.config import load_settings
from app.shared.control import request_control
from app.shared.operations import (
    ProcessLock,
    restore_backup,
    verify_backup,
)


class DisabledSender:
    async def send_text(self, chat_id: int, text: str) -> SendResult:
        raise RuntimeError("Transport is disabled in maintenance mode")


class InactiveModel:
    async def complete(
        self,
        task: str,
        rules: str,
        data: dict[str, Any],
        schema: type[SchemaT],
        deadline_seconds: float,
    ) -> SchemaT:
        raise RuntimeError("LLM is disabled in maintenance mode")


async def execute(arguments: argparse.Namespace) -> None:
    settings = load_settings(arguments.settings)
    if arguments.command in {"probe", "evaluate"}:
        live = arguments.command == "probe" or arguments.live
        if live:
            settings.require_credentials(telegram=False)
        result = await run_evaluation(
            EvaluationOptions(
                family="quality" if arguments.command == "probe" else arguments.cases,
                output=arguments.report,
                limit=1 if arguments.command == "probe" else arguments.limit,
                live=live,
                max_calls=2 if arguments.command == "probe" else arguments.max_calls,
                concurrency=(
                    1 if arguments.command == "probe" else arguments.concurrency
                ),
            ),
            settings=settings,
        )
        print(result.model_dump_json())
        return
    if arguments.command == "verify-backup":
        print(json.dumps(verify_backup(arguments.path)))
        return
    if arguments.command in {"metrics", "backup", "kill-switch"}:
        request = {"command": arguments.command}
        if arguments.command == "kill-switch":
            request["enabled"] = arguments.enabled == "on"
        response = await request_control(settings.storage.directory, request)
        if response is not None:
            print(json.dumps(response))
            return
    if arguments.command == "restore":
        with ProcessLock(settings.storage.directory):
            restore_backup(arguments.path, settings.storage.directory)
    application = await Application.open(settings, DisabledSender(), InactiveModel())
    try:
        if arguments.command == "backup":
            path = await application.memory.backup()
            print(path.resolve())
        elif arguments.command == "kill-switch":
            await application.delivery_repository.kill_switch(arguments.enabled == "on")
            print(json.dumps({"kill_switch": arguments.enabled == "on"}))
        else:
            print(json.dumps(await application.metrics()))
    finally:
        await application.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="manage.py")
    parser.add_argument("--settings", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("metrics", "backup"):
        commands.add_parser(name)
    for name in ("restore", "verify-backup"):
        commands.add_parser(name).add_argument("path", type=Path)
    commands.add_parser("kill-switch").add_argument("enabled", choices=("on", "off"))
    probe = commands.add_parser("probe")
    probe.add_argument("--report", type=Path, default=Path("reports/probe.jsonl"))
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument(
        "--cases",
        choices=(
            "quality",
            "conversations",
            "identity",
            "identity-chain",
            "languages",
            "all",
        ),
        default="quality",
    )
    evaluate.add_argument(
        "--report", type=Path, default=Path("reports/evaluation.jsonl")
    )
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--max-calls", type=int, default=2500)
    evaluate.add_argument("--concurrency", type=int, default=1)
    evaluate.add_argument("--live", action="store_true")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    try:
        asyncio.run(execute(parser.parse_args()))
    except (ValueError, RuntimeError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
