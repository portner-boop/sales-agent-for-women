import argparse
import asyncio
import json
import os
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from pydantic import BaseModel, ConfigDict, Field

from app.application import Application
from app.evaluation import EvaluationSender, receive
from app.modules.agent.entities import SchemaT
from app.modules.agent.gateway import GatewayModel
from app.modules.memory.entities import CompactionBlock
from app.modules.memory.schemas import EpisodeSummary
from app.shared.config import load_settings
from app.shared.storage import fetchall


class DatasetStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario: str
    text: str
    user_id: int = Field(default=1, ge=1)


class RawResponseRecorder(BaseCallbackHandler):
    def __init__(self, event: ContextVar[dict[str, Any] | None]) -> None:
        self.event = event

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        event = self.event.get()
        if event is not None:
            event["raw_responses"] = [
                generation.message.model_dump(mode="json")
                for group in response.generations
                for generation in group
                if hasattr(generation, "message")
            ]


class DatasetModel:
    def __init__(self, gateway: GatewayModel) -> None:
        self.gateway = gateway
        self.events: list[dict[str, Any]] = []
        self.event: ContextVar[dict[str, Any] | None] = ContextVar(
            "dataset_event", default=None
        )
        self.gateway.model.callbacks = [RawResponseRecorder(self.event)]

    async def complete(
        self,
        task: str,
        rules: str,
        data: dict[str, Any],
        schema: type[SchemaT],
        deadline_seconds: float,
    ) -> SchemaT:
        event: dict[str, Any] = {
            "task": task,
            "rules": rules,
            "request": data,
            "schema": schema.model_json_schema(),
        }
        self.events.append(event)
        token = self.event.set(event)
        started = time.monotonic()
        try:
            result = await self.gateway.complete(
                task, rules, data, schema, deadline_seconds
            )
            event["result"] = result.model_dump(mode="json")
            return result
        except Exception as error:
            self.record_error(event, error)
            raise
        finally:
            event["seconds"] = time.monotonic() - started
            self.event.reset(token)

    async def summarize(
        self, block: CompactionBlock, deadline_seconds: float, output_tokens: int
    ) -> EpisodeSummary:
        event: dict[str, Any] = {
            "task": "summarize",
            "request": {
                "messages": [message.as_data() for message in block.messages],
                "entity_map": block.entity_map,
                "source_hash": block.source_hash,
            },
        }
        self.events.append(event)
        token = self.event.set(event)
        started = time.monotonic()
        try:
            result = await self.gateway.summarize(
                block, deadline_seconds, output_tokens
            )
            event["result"] = result.model_dump(mode="json")
            return result
        except Exception as error:
            self.record_error(event, error)
            raise
        finally:
            event["seconds"] = time.monotonic() - started
            self.event.reset(token)

    def record_error(self, event: dict[str, Any], error: Exception) -> None:
        event["exception_type"] = type(error).__name__
        event["exception"] = str(error).replace(
            self.gateway.settings.api_key.get_secret_value(), "<redacted>"
        )[:2000]
        completion = getattr(error, "completion", None) or getattr(
            error.__cause__, "completion", None
        )
        if isinstance(completion, BaseModel):
            event["raw_error_response"] = completion.model_dump(mode="json")


def append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


async def collect(arguments: argparse.Namespace) -> None:
    settings = load_settings()
    settings.require_credentials(telegram=False)
    steps = [
        DatasetStep.model_validate_json(line)
        for line in arguments.cases.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if arguments.directory.resolve() == settings.storage.directory.resolve():
        raise ValueError("Dataset must use an isolated storage directory")
    arguments.directory.mkdir(parents=True, exist_ok=False)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x", encoding="utf-8"):
        pass
    arguments.output.chmod(0o600)
    local = settings.model_copy(
        update={
            "environment": "test",
            "storage": settings.storage.model_copy(
                update={"directory": arguments.directory}
            ),
            "runtime": settings.runtime.model_copy(update={"coalesce_seconds": 0}),
            "memory": settings.memory.model_copy(
                update={"history_threshold_tokens": arguments.compaction_threshold}
            ),
        }
    )
    model = DatasetModel(GatewayModel(local.llm, local.runtime.llm_concurrency))
    sender = EvaluationSender()
    app = await Application.open(local, sender, model, channel="diagnostic-dataset")
    try:
        for index, step in enumerate(steps, start=1):
            event_start = len(model.events)
            attempt_start = len(sender.attempts)
            async with app.database.read() as tx:
                existing = await fetchall(tx, "SELECT id FROM agent_runs")
            run_ids = {row["id"] for row in existing}
            started = time.monotonic()
            error = None
            try:
                await receive(
                    app, sender, index, step.text, step.user_id, app.clock.now()
                )
            except Exception as failure:
                error = type(failure).__name__
            elapsed = time.monotonic() - started
            async with app.database.read() as tx:
                rows = await fetchall(
                    tx, "SELECT * FROM agent_runs ORDER BY started_at"
                )
                scopes = await fetchall(tx, "SELECT * FROM conversations")
                outbox = await fetchall(
                    tx, "SELECT * FROM outbox_messages ORDER BY rowid"
                )
                summaries = await fetchall(tx, "SELECT * FROM memory_summaries")
                coverage = await fetchall(tx, "SELECT * FROM summary_sources")
                jobs = await fetchall(
                    tx,
                    "SELECT kind,status,error_code,attempts FROM scheduled_jobs WHERE error_code IS NOT NULL",
                )
            memory = []
            agenda = []
            messages: list[dict[str, Any]] = []
            for scope_row in scopes:
                conversation = await app.conversations.get(scope_row["id"])
                memory.append(
                    {
                        "user_id": conversation.external_user_id,
                        "items": await app.memory.repository.items(
                            conversation.scope, history=True
                        ),
                    }
                )
                agenda.append(
                    {
                        "user_id": conversation.external_user_id,
                        "items": await app.conversations.agenda(conversation.scope),
                    }
                )
                messages.extend(
                    message.as_data()
                    for message in await app.conversations.messages(conversation.scope)
                )
            runs = [dict(row) for row in rows if row["id"] not in run_ids]
            turn_ids = {run["turn_id"] for run in runs}
            record = {
                "dataset_version": 1,
                "origin": "synthetic_dialogue",
                "phase": "post_fix",
                "step": index,
                **step.model_dump(),
                "model": local.llm.model,
                "reasoning_effort": local.llm.reasoning_effort,
                "base_url": local.llm.base_url,
                "compaction_threshold_tokens": arguments.compaction_threshold,
                "reply_pacing": {
                    "min_seconds": local.runtime.reply_pause_min_seconds,
                    "max_seconds": local.runtime.reply_pause_max_seconds,
                },
                "seconds_including_background": elapsed,
                "runs": runs,
                "events": model.events[event_start:],
                "send_attempts": sender.attempts[attempt_start:],
                "outbox_parts": [
                    dict(row) for row in outbox if row["turn_id"] in turn_ids
                ],
                "scopes": [dict(row) for row in scopes],
                "memory_after": memory,
                "agenda_after": agenda,
                "messages_after": messages,
                "summaries_after": [dict(row) for row in summaries],
                "coverage_after": [dict(row) for row in coverage],
                "jobs_with_errors": [dict(row) for row in jobs],
                "error": error,
                "database": str((arguments.directory / "app.db").resolve()),
                "human_rating": None,
            }
            append_record(arguments.output, record)
            print(
                json.dumps(
                    {
                        "step": index,
                        "scenario": step.scenario,
                        "statuses": [run["status"] for run in runs],
                        "replies": [
                            item["text"] for item in sender.attempts[attempt_start:]
                        ],
                        "summaries": len(summaries),
                        "error": error,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        await app.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--compaction-threshold", type=int, default=6000)
    asyncio.run(collect(parser.parse_args()))


if __name__ == "__main__":
    main()
