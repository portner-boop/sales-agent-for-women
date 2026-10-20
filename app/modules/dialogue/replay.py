import argparse
import asyncio
import inspect
import json
import tempfile
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.application import Application
from app.modules.agent.entities import LanguageModel, SchemaT, current_llm_metrics
from app.modules.agent.gateway import GatewayModel
from app.modules.delivery.entities import SendResult
from app.modules.dialogue.prompts import rules_for_mode
from app.shared.clock import MutableClock
from app.shared.config import Settings, load_settings
from app.shared.entities import ScopeContext
from app.shared.storage import fetchall


class ReplayStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str | None = None
    wait_seconds: float | None = Field(default=None, gt=0)
    language: str = "en"

    @model_validator(mode="after")
    def one_action(self) -> "ReplayStep":
        if (self.message is None) == (self.wait_seconds is None):
            raise ValueError("Specify exactly one message or pause")
        return self


class ReplayCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    steps: list[ReplayStep] = Field(min_length=1)


@dataclass
class RecordingSender:
    next_id: int = 100000

    async def send_text(self, chat_id: int, text: str) -> SendResult:
        self.next_id += 1
        return SendResult("delivered", self.next_id)


class RecordingModel:
    def __init__(self, model: LanguageModel, limit: int) -> None:
        self.model, self.limit = model, limit
        self.attempts: list[dict[str, Any]] = []

    async def complete(
        self,
        task: str,
        rules: str,
        data: dict[str, Any],
        schema: type[SchemaT],
        deadline_seconds: float,
    ) -> SchemaT:
        if len(self.attempts) >= self.limit:
            raise RuntimeError("Replay model call budget exhausted")
        attempt: dict[str, Any] = {
            "task": task,
            "request": data,
            "rules_sha256": sha256(rules.encode()).hexdigest(),
        }
        self.attempts.append(attempt)
        started = time.perf_counter()
        try:
            result = await self.model.complete(
                task, rules, data, schema, deadline_seconds
            )
            attempt["draft"] = result.model_dump(mode="json")
            return result
        except BaseException as error:
            attempt["error"] = type(error).__name__
            raise
        finally:
            attempt["seconds"] = time.perf_counter() - started
            attempt["metrics"] = current_llm_metrics.get() or {}


def incoming(number: int, text: str, language: str, now: float) -> dict[str, Any]:
    message: dict[str, Any] = {
        "message_id": number,
        "date": int(now),
        "chat": {"id": 900001, "type": "private"},
        "from": {"id": 900001, "is_bot": False, "language_code": language},
        "text": text,
    }
    if text.startswith("/"):
        message["entities"] = [
            {"type": "bot_command", "offset": 0, "length": len(text.split()[0])}
        ]
    return {"update_id": number, "message": message}


async def deliver_pending(app: Application, clock: MutableClock) -> None:
    await app.drain()
    async with app.database.read() as tx:
        rows = await fetchall(tx, "SELECT id,epoch FROM conversations")
    scopes = [ScopeContext(row["id"], row["epoch"]) for row in rows]
    for _ in range(100):
        due_times = [
            due
            for scope in scopes
            if (due := await app.delivery_repository.next_ready_at(scope)) is not None
        ]
        if not due_times:
            return
        clock.advance(max(0, min(due_times) - clock.now()))
        await app.drain()
    raise RuntimeError("Replay delivery limit exceeded")


async def run_case(
    case: ReplayCase, settings: Settings, model: RecordingModel, report: dict[str, Any]
) -> dict[str, Any]:
    clock = MutableClock(time.time())
    with tempfile.TemporaryDirectory(prefix="marko-replay-") as directory:
        configured = settings.model_copy(
            update={
                "environment": "test",
                "storage": settings.storage.model_copy(
                    update={"directory": Path(directory)}
                ),
                "runtime": settings.runtime.model_copy(update={"coalesce_seconds": 0}),
            }
        )
        app = await Application.open(
            configured, RecordingSender(), model, clock, channel="isolated-replay"
        )
        delivered: set[str] = set()
        try:
            for index, step in enumerate(case.steps, 1):
                if len(model.attempts) >= model.limit:
                    report["status"] = "budget_exhausted"
                    break
                started = time.perf_counter()
                call_offset = len(model.attempts)
                if step.message is not None:
                    await app.telegram.ingest(
                        [incoming(index, step.message, step.language, clock.now())]
                    )
                else:
                    assert step.wait_seconds is not None
                    clock.advance(step.wait_seconds)
                await deliver_pending(app, clock)
                async with app.database.read() as tx:
                    rows = await fetchall(
                        tx,
                        "SELECT id,kind,text,status,turn_id,proposal_json FROM outbox_messages ORDER BY rowid",
                    )
                    followups = await fetchall(
                        tx, "SELECT status,reason,run_at FROM followups ORDER BY rowid"
                    )
                    states = await fetchall(
                        tx,
                        "SELECT preview_reply_count,completed_pair_count,invitation_state,preview_closed_at FROM conversations",
                    )
                actions = []
                for row in rows:
                    if row["id"] not in delivered and row["status"] == "delivered":
                        action = dict(row)
                        action["proposal"] = json.loads(action.pop("proposal_json"))
                        actions.append(action)
                        delivered.add(row["id"])
                report["steps"].append(
                    {
                        **step.model_dump(exclude_none=True),
                        "seconds": time.perf_counter() - started,
                        "actions": actions,
                        "attempts": model.attempts[call_offset:],
                        "followups": [dict(row) for row in followups],
                        "state": dict(states[0]) if states else {},
                    }
                )
                print(
                    case.id,
                    index,
                    json.dumps([a["text"] for a in actions], ensure_ascii=False),
                    flush=True,
                )
            report["metrics"] = await app.metrics()
        finally:
            await app.close()
        return report


async def run_replay(
    cases: list[ReplayCase],
    settings: Settings,
    model: LanguageModel,
    output: Path,
    *,
    max_calls: int = 100,
) -> dict[str, Any]:
    if not 1 <= max_calls <= 2500:
        raise ValueError("Replay budget must be between 1 and 2500")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        recorded = RecordingModel(model, max_calls)
        report: dict[str, Any] = {
            "model": settings.llm.model,
            "llm_parameters": {
                "structured_mode": settings.llm.structured_mode,
                "reasoning_effort": settings.llm.reasoning_effort,
                "verbosity": settings.llm.verbosity,
                "output_tokens": settings.llm.output_tokens,
            },
            "transport": "RecordingSender; no Telegram API calls",
            "clock": "MutableClock; simulated pauses, real model latency",
            "prompt_sha256": sha256(rules_for_mode().encode()).hexdigest(),
            "gateway_sha256": sha256(
                inspect.getsource(GatewayModel).encode()
            ).hexdigest(),
            "scenario_sha256": sha256(
                json.dumps([case.model_dump() for case in cases]).encode()
            ).hexdigest(),
            "call_limit": max_calls,
            "status": "interrupted",
            "cases": [],
        }
        try:
            for case in cases:
                if len(recorded.attempts) >= max_calls:
                    report["status"] = "budget_exhausted"
                    break
                case_report: dict[str, Any] = {"id": case.id, "steps": []}
                report["cases"].append(case_report)
                await run_case(case, settings, recorded, case_report)
                if case_report.get("status") == "budget_exhausted":
                    report["status"] = "budget_exhausted"
                    break
            else:
                report["status"] = "complete"
        finally:
            report["calls"] = len(recorded.attempts)
            report["attempts"] = recorded.attempts
            stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=Path, action="append", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-calls", type=int, default=100)
    parser.add_argument("--live", action="store_true", required=True)
    options = parser.parse_args()
    cases = [
        ReplayCase.model_validate(case)
        for path in options.scenario
        for case in json.loads(path.read_text(encoding="utf-8"))
    ]
    settings = load_settings()
    settings.require_credentials(telegram=False)
    asyncio.run(
        run_replay(
            cases,
            settings,
            GatewayModel(settings.llm, concurrency=1),
            options.report,
            max_calls=min(options.max_calls, settings.runtime.research_api_call_limit),
        )
    )


if __name__ == "__main__":
    main()
