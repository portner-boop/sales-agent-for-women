import asyncio
import json
from collections import deque
from pathlib import Path
from typing import Any

import pytest

from app.modules.agent.entities import SchemaT, current_llm_metrics
from app.modules.dialogue.runner import (
    EvaluationOptions,
    argument_parser,
    run_evaluation,
    selected_cases,
)
from app.modules.dialogue.schemas import DialogueDraft


class RecordingModel:
    def __init__(self, replies: list[DialogueDraft | Exception] | None = None) -> None:
        self.replies = deque(replies or [])
        self.requests: list[dict[str, Any]] = []

    async def complete(
        self,
        task: str,
        rules: str,
        data: dict[str, Any],
        schema: type[SchemaT],
        deadline_seconds: float,
    ) -> SchemaT:
        self.requests.append(data)
        result = (
            self.replies.popleft()
            if self.replies
            else DialogueDraft(parts=["I prefer tea"], language="en")
        )
        metrics = current_llm_metrics.get()
        assert metrics is not None
        metrics.update(
            {
                "raw_response": (
                    "invalid output"
                    if isinstance(result, Exception)
                    else result.model_dump_json()
                ),
                "input_tokens": 23,
                "output_tokens": 11,
            }
        )
        if isinstance(result, Exception):
            raise result
        return schema.model_validate(result)


class DelayedModel(RecordingModel):
    def __init__(self) -> None:
        super().__init__()
        self.inflight = 0
        self.peak = 0
        self.started: list[tuple[str, int]] = []
        self.active_cases: set[str] = set()
        self.same_case_overlap = False

    async def complete(
        self,
        task: str,
        rules: str,
        data: dict[str, Any],
        schema: type[SchemaT],
        deadline_seconds: float,
    ) -> SchemaT:
        scope = str(data["scope_id"])
        self.same_case_overlap |= scope in self.active_cases
        self.active_cases.add(scope)
        self.started.append((scope, data["input_revision"]))
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        delay = (5 - (len(self.started) - 1) % 5) * 0.002
        try:
            await asyncio.sleep(delay)
            return await super().complete(task, rules, data, schema, deadline_seconds)
        finally:
            self.inflight -= 1
            self.active_cases.discard(scope)


class BlockingModel:
    def __init__(self, expected: int) -> None:
        self.expected = expected
        self.started = 0
        self.all_started = asyncio.Event()

    async def complete(
        self,
        task: str,
        rules: str,
        data: dict[str, Any],
        schema: type[SchemaT],
        deadline_seconds: float,
    ) -> SchemaT:
        self.started += 1
        metrics = current_llm_metrics.get()
        assert metrics is not None
        metrics["raw_response"] = f"partial response {self.started}"
        metrics["input_tokens"] = 1
        if self.started == self.expected:
            self.all_started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def read_summary(path: str) -> dict[str, Any]:
    return dict(json.loads(Path(path).read_text()))


async def test_default_offline_runner_does_not_load_credentials_or_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden() -> None:
        raise AssertionError("offline cannot load runtime credentials")

    monkeypatch.setattr("app.shared.config.load_settings", forbidden)
    path = tmp_path / "smoke.jsonl"
    report = await run_evaluation(EvaluationOptions(output=path, limit=2))
    assert report.status == "complete"
    assert report.backend == "offline_smoke"
    assert report.quality_validated is False
    assert report.policy_version == "marko-dialogue-v55"
    assert len(report.schema_sha256) == 64
    assert report.calls == report.delivered_turns == 2
    assert report.input_tokens is None
    assert report.output_tokens is None
    assert len(events(path)) == 3
    assert events(path)[-1]["event"] == "summary"
    assert read_summary(report.summary_output)["calls"] == 2


async def test_complete_conversation_preserves_all_roles_and_final_originals(
    tmp_path: Path,
) -> None:
    model = RecordingModel()
    report = await run_evaluation(
        EvaluationOptions(
            output=tmp_path / "full.jsonl", family="conversations", limit=1
        ),
        model=model,
    )
    assert report.turns_processed == report.delivered_turns == 20
    assert len(model.requests) == 20
    final = model.requests[-1]
    assert len(final["CHAT_MESSAGES"]) == 39
    assert [message["role"] for message in final["CHAT_MESSAGES"]] == [
        "user",
        "assistant",
    ] * 19 + ["user"]
    assert "I am Nora" in final["CHAT_MESSAGES"][0]["content"]
    assert final["pending_ids"] == [final["original_messages"][-1]["id"]]
    assert report.input_tokens == 20 * 23
    assert report.output_tokens == 20 * 11
    assert report.token_usage_observed_calls == 20


async def test_first_admission_is_recorded_without_repair(tmp_path: Path) -> None:
    draft = DialogueDraft(parts=["I am an AI"], language="en")
    model = RecordingModel([draft])
    path = tmp_path / "admission.jsonl"
    report = await run_evaluation(
        EvaluationOptions(output=path, family="identity", limit=1), model=model
    )
    assert report.calls == 1
    assert report.first_identity_outcomes == {"admission": 1}
    assert report.repair_identity_outcomes == {}
    attempt = events(path)[0]
    assert attempt["attempt_kind"] == "first"
    assert attempt["observation"]["first_parts"] == ["I am an AI"]
    assert attempt["observation"]["delivered_parts"] == ["I am an AI"]
    assert attempt["formal_errors"] == []


async def test_formal_repair_keeps_raw_initial_and_repaired_answers_separate(
    tmp_path: Path,
) -> None:
    model = RecordingModel(
        [
            DialogueDraft(parts=["I am an AI 🙂"], language="en"),
            DialogueDraft(parts=["I am an AI"], language="en"),
        ]
    )
    path = tmp_path / "repair.jsonl"
    report = await run_evaluation(
        EvaluationOptions(output=path, family="identity", limit=1), model=model
    )
    assert report.calls == 2
    first, repair, _ = events(path)
    assert first["attempt_kind"] == "first"
    assert repair["attempt_kind"] == "technical_repair"
    assert first["observation"]["first_parts"] == ["I am an AI 🙂"]
    assert first["observation"]["delivered_parts"] == []
    assert repair["observation"]["delivered_parts"] == ["I am an AI"]
    assert model.requests[1]["REPAIR_ERRORS"] == ["EMOJI_NOT_ALLOWED"]
    assert report.first_identity_outcomes == {"admission": 1}
    assert report.repair_identity_outcomes == {"admission": 1}


async def test_parsing_failure_preserves_raw_and_has_two_attempt_maximum(
    tmp_path: Path,
) -> None:
    model = RecordingModel(
        [
            ValueError("never persist secret exception text"),
            ValueError("another failure"),
        ]
    )
    path = tmp_path / "invalid.jsonl"
    report = await run_evaluation(EvaluationOptions(output=path, limit=1), model=model)
    assert report.calls == 2
    assert report.missing_responses == 1
    assert report.delivered_turns == 0
    assert len(events(path)) == 3
    assert events(path)[0]["raw_response"] == "invalid output"
    assert events(path)[0]["formal_errors"] == ["ValueError"]
    assert "secret exception text" not in path.read_text()


async def test_budget_exhaustion_writes_partial_report_without_continuing(
    tmp_path: Path,
) -> None:
    model = RecordingModel()
    path = tmp_path / "budget.jsonl"
    report = await run_evaluation(
        EvaluationOptions(output=path, family="conversations", limit=1, max_calls=3),
        model=model,
    )
    assert report.status == "budget_exhausted"
    assert report.calls == len(model.requests) == 3
    assert report.turns_planned == 20
    assert report.turns_processed == 3
    assert report.cases_completed == 0
    assert events(path)[-1]["status"] == "budget_exhausted"
    assert read_summary(report.summary_output)["calls"] == 3


async def test_budget_exhaustion_during_repair_retains_first_raw(
    tmp_path: Path,
) -> None:
    model = RecordingModel([DialogueDraft(parts=["Hi :)"], language="en")])
    path = tmp_path / "repair-budget.jsonl"
    report = await run_evaluation(
        EvaluationOptions(output=path, limit=1, max_calls=1), model=model
    )
    assert report.status == "budget_exhausted"
    assert report.calls == 1
    assert report.missing_responses == 1
    assert events(path)[0]["observation"]["first_parts"] == ["Hi :)"]


async def test_existing_results_are_never_silently_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "existing.jsonl"
    path.write_text("preserve this")
    with pytest.raises(FileExistsError):
        await run_evaluation(EvaluationOptions(output=path, limit=1))
    assert path.read_text() == "preserve this"


@pytest.mark.parametrize("limit", [0, -1, 2501])
def test_paid_call_ceiling_is_enforced_before_any_execution(
    tmp_path: Path, limit: int
) -> None:
    with pytest.raises(ValueError):
        EvaluationOptions(output=tmp_path / "unused.jsonl", max_calls=limit)


def test_corpus_selection_and_explicit_live_flag() -> None:
    assert len(selected_cases("quality")) == 60
    assert len(selected_cases("identity")) == 8
    assert len(selected_cases("conversations")) == 10
    assert len(selected_cases("identity-chain")[0].steps) == 20
    assert len(selected_cases("languages")) == 8
    assert len(selected_cases("all")) == 87
    assert sum(len(case.steps) for case in selected_cases("all")) == 296
    assert len(selected_cases("all", limit=3)) == 3
    assert argument_parser().parse_args(["--output", "result.jsonl"]).live is False
    assert (
        argument_parser().parse_args(["--output", "result.jsonl", "--live"]).live
        is True
    )


async def test_offline_smoke_is_repeatable(tmp_path: Path) -> None:
    first = tmp_path / "one.jsonl"
    second = tmp_path / "two.jsonl"
    first_report = await run_evaluation(EvaluationOptions(output=first, limit=2))
    second_report = await run_evaluation(EvaluationOptions(output=second, limit=2))
    assert first_report.corpus_sha256 == second_report.corpus_sha256
    assert len(first_report.corpus_sha256) == 64
    assert [event["parsed_draft"] for event in events(first)[:-1]] == [
        event["parsed_draft"] for event in events(second)[:-1]
    ]


async def test_report_groups_identity_outcomes_and_records_first_admission_turn(
    tmp_path: Path,
) -> None:
    model = RecordingModel(
        [
            DialogueDraft(parts=["I'm a real man"], language="en"),
            DialogueDraft(parts=["I am an AI"], language="en"),
            DialogueDraft(parts=["I am an AI"], language="en"),
        ]
    )
    path = tmp_path / "identity-groups.jsonl"
    report = await run_evaluation(
        EvaluationOptions(output=path, family="conversations", limit=1), model=model
    )
    assert report.identity_by_language["en"] == {
        "human_assertion": 1,
        "admission": 2,
        "ambiguous": 17,
    }
    assert (
        report.identity_by_case["en_flirt_memory"] == report.identity_by_language["en"]
    )
    assert report.first_admission_by_case["en_flirt_memory"] == {
        "turn": 2,
        "attempt": 1,
        "call_number": 2,
    }
    assert events(path)[1]["requested_language"] == "en"
    assert events(path)[1]["case_language"] == "en"
    assert events(path)[1]["reply_language"] == "en"


@pytest.mark.parametrize("concurrency", [0, 6])
def test_concurrency_is_limited_to_five_dialogues(
    tmp_path: Path, concurrency: int
) -> None:
    with pytest.raises(ValueError):
        EvaluationOptions(output=tmp_path / "unused.jsonl", concurrency=concurrency)


async def test_five_cases_run_concurrently_but_turns_remain_ordered(
    tmp_path: Path,
) -> None:
    model = DelayedModel()
    path = tmp_path / "concurrent-conversations.jsonl"
    report = await run_evaluation(
        EvaluationOptions(output=path, family="conversations", limit=5, concurrency=5),
        model=model,
    )
    assert report.calls == report.delivered_turns == 100
    assert model.peak == report.peak_inflight_calls == 5
    assert not model.same_case_overlap
    for scope in {scope for scope, _ in model.started}:
        assert [turn for current, turn in model.started if current == scope] == list(
            range(1, 21)
        )
    rows = events(path)[:-1]
    assert sorted(row["call_number"] for row in rows) == list(range(1, 101))
    assert [row["call_number"] for row in rows] != list(range(1, 101))


async def test_concurrent_budget_reservation_never_overshoots(tmp_path: Path) -> None:
    model = DelayedModel()
    path = tmp_path / "concurrent-budget.jsonl"
    report = await run_evaluation(
        EvaluationOptions(output=path, limit=20, max_calls=7, concurrency=5),
        model=model,
    )
    assert report.status == "budget_exhausted"
    assert report.calls == len(model.started) == report.delivered_turns == 7
    assert sorted(row["call_number"] for row in events(path)[:-1]) == list(range(1, 8))
    assert model.peak <= 5


async def test_cancellation_flushes_every_inflight_raw_attempt_before_summary(
    tmp_path: Path,
) -> None:
    model = BlockingModel(expected=5)
    path = tmp_path / "cancelled.jsonl"
    running = asyncio.create_task(
        run_evaluation(
            EvaluationOptions(output=path, limit=5, concurrency=5), model=model
        )
    )
    await asyncio.wait_for(model.all_started.wait(), 2)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    rows = events(path)
    assert len(rows) == 6
    assert all(row["formal_errors"] == ["INTERRUPTED"] for row in rows[:-1])
    assert {row["raw_response"] for row in rows[:-1]} == {
        f"partial response {number}" for number in range(1, 6)
    }
    assert sorted(row["call_number"] for row in rows[:-1]) == list(range(1, 6))
    assert rows[-1]["event"] == "summary"
    assert rows[-1]["status"] == "interrupted"
    assert rows[-1]["calls"] == 5


async def test_live_injected_model_keeps_real_identity_and_configured_budget(
    tmp_path: Path,
) -> None:
    from app.shared.config import Settings

    settings = Settings.model_validate(
        {
            "llm": {
                "model": "real-luna",
                "base_url": "https://provider.invalid/v1",
                "api_key": "test",
            },
            "runtime": {"research_api_call_limit": 1},
        }
    )
    report = await run_evaluation(
        EvaluationOptions(
            output=tmp_path / "live.jsonl", live=True, limit=2, max_calls=10
        ),
        model=RecordingModel(),
        settings=settings,
    )
    assert report.backend == "live"
    assert report.model == "real-luna"
    assert report.calls == 1
    assert report.call_limit == 1
    assert report.status == "budget_exhausted"
