import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from app.application import Application
from app.evaluation import evaluate_cases
from app.modules.agent.entities import SchemaT
from app.modules.agent.gateway import GatewayModel
from app.modules.agent.prompts import INTERPRETER_RULES
from app.modules.agent.schemas import TurnInterpretation
from app.modules.delivery.entities import SendResult
from app.modules.memory.entities import CompactionBlock
from app.modules.memory.schemas import EpisodeSummary
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

    async def summarize(
        self, block: CompactionBlock, deadline_seconds: float, output_tokens: int
    ) -> EpisodeSummary:
        raise RuntimeError("LLM is disabled in maintenance mode")


async def execute(arguments: argparse.Namespace) -> None:
    settings = load_settings(arguments.settings)
    if arguments.command == "probe":
        settings.require_credentials(telegram=False)
        model = GatewayModel(settings.llm, 1)
        result = await model.complete(
            "interpret",
            INTERPRETER_RULES,
            {
                "CURRENT_INPUT": [
                    {
                        "id": "probe-message",
                        "version": 1,
                        "role": "user",
                        "text": "Hello, I would like a short chat.",
                    }
                ],
                "REFERENCE_MESSAGES": [],
                "RECENT_MESSAGES": [],
                "facts": [],
                "entities": [],
                "language": "en",
            },
            TurnInterpretation,
            15,
        )
        print(
            json.dumps(
                {
                    "model": settings.llm.model,
                    "base_url": settings.llm.base_url,
                    "structured_mode": settings.llm.structured_mode,
                    "result": result.model_dump(),
                }
            )
        )
        return
    if arguments.command == "evaluate":
        settings.require_credentials(telegram=False)
        evaluation_result = await evaluate_cases(
            settings,
            GatewayModel(settings.llm, settings.runtime.llm_concurrency),
            arguments.cases,
            arguments.report,
            arguments.case,
        )
        print(
            json.dumps(
                {
                    "report": str(arguments.report.resolve()),
                    "accepted": evaluation_result["accepted"],
                    "total": evaluation_result["total"],
                }
            )
        )
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
    for name in ("probe", "metrics", "backup"):
        commands.add_parser(name)
    for name in ("restore", "verify-backup"):
        commands.add_parser(name).add_argument("path", type=Path)
    commands.add_parser("kill-switch").add_argument("enabled", choices=("on", "off"))
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument(
        "--cases", type=Path, default=Path("docs/regression_cases.jsonl")
    )
    evaluate.add_argument(
        "--report", type=Path, default=Path("reports/gateway-evaluation.json")
    )
    evaluate.add_argument("--case")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    try:
        asyncio.run(execute(parser.parse_args()))
    except (ValueError, RuntimeError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
