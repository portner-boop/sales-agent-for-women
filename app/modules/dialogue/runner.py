import argparse
import asyncio
import json
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from statistics import median
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from app.modules.agent.entities import SchemaT, current_llm_metrics
from app.modules.agent.recovery import structure_errors
from app.modules.conversations.entities import Conversation, Message
from app.modules.dialogue.context import build_request
from app.modules.dialogue.evaluation import (
    ConversationCase,
    QualityCase,
    additional_language_cases,
    conversation_cases,
    identity_cases,
    identity_conversation_cases,
    quality_cases,
)
from app.modules.dialogue.prompts import POLICY_VERSION, rules_for_mode
from app.modules.dialogue.research import observe_response
from app.modules.dialogue.schemas import DialogueDraft
from app.modules.dialogue.validation import validate_draft
from app.shared.config import Settings

CaseFamily = Literal[
    "quality", "conversations", "identity", "identity-chain", "languages", "all"
]


class EvaluationModel(Protocol):
    async def complete(
        self,
        task: str,
        rules: str,
        data: dict[str, Any],
        schema: type[SchemaT],
        deadline_seconds: float,
    ) -> SchemaT: ...


@dataclass(frozen=True)
class EvaluationOptions:
    output: Path
    family: CaseFamily = "quality"
    limit: int | None = None
    live: bool = False
    max_calls: int = 2500
    timeout_seconds: float = 30
    concurrency: int = 1

    def __post_init__(self) -> None:
        if not 1 <= self.max_calls <= 2500:
            raise ValueError("max_calls must be between 1 and 2500")
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not 1 <= self.concurrency <= 5:
            raise ValueError("concurrency must be between 1 and 5")


class EvaluationReport(BaseModel):
    backend: str
    model: str
    family: CaseFamily
    status: Literal["complete", "budget_exhausted", "interrupted"] = "interrupted"
    quality_validated: Literal[False] = False
    cases_planned: int
    cases_completed: int = 0
    turns_planned: int
    turns_processed: int = 0
    delivered_turns: int = 0
    missing_responses: int = 0
    calls: int = 0
    call_limit: int
    concurrency: int = 1
    peak_inflight_calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    token_usage_observed_calls: int = 0
    elapsed_seconds: float = 0
    latency_p50_seconds: float | None = None
    latency_p95_seconds: float | None = None
    first_identity_outcomes: dict[str, int] = Field(default_factory=dict)
    repair_identity_outcomes: dict[str, int] = Field(default_factory=dict)
    identity_by_language: dict[str, dict[str, int]] = Field(default_factory=dict)
    identity_by_case: dict[str, dict[str, int]] = Field(default_factory=dict)
    first_admission_by_case: dict[str, dict[str, int]] = Field(default_factory=dict)
    formal_errors: dict[str, int] = Field(default_factory=dict)
    prompt_sha256: str
    policy_version: str
    schema_sha256: str
    corpus_sha256: str
    output: str
    summary_output: str
    identity_classification: Literal["heuristic_only_not_proof"] = (
        "heuristic_only_not_proof"
    )


@dataclass(frozen=True)
class RunnerCase:
    id: str
    language: str
    history: tuple[dict[str, str], ...]
    steps: tuple[str, ...]
    requirements: str
    expected_move: str | None = None


class OfflineSmokeModel:
    async def complete(
        self,
        task: str,
        rules: str,
        data: dict[str, Any],
        schema: type[SchemaT],
        deadline_seconds: float,
    ) -> SchemaT:
        replies = {
            "en": "I'd choose a quieter evening",
            "ru": "Я бы выбрал спокойный вечер",
            "es": "Yo elegiría una tarde tranquila",
            "pt": "Eu escolheria uma noite tranquila",
            "fr": "Je choisirais une soirée tranquille",
            "ar": "أفضل أمسية هادئة",
            "de": "Ich würde einen ruhigen Abend wählen",
            "zh": "我会选择一个安静的晚上",
        }
        language = str(data["language"])
        draft = DialogueDraft(
            parts=[replies.get(language.split("-")[0], replies["en"])],
            language=language,
        )
        metrics = current_llm_metrics.get()
        if metrics is not None:
            metrics["raw_response"] = draft.model_dump_json()
        return schema.model_validate(draft)


def selected_cases(family: CaseFamily, limit: int | None = None) -> list[RunnerCase]:
    focused: list[QualityCase] = []
    complete: list[ConversationCase] = []
    if family in {"quality", "all"}:
        focused.extend(quality_cases())
    if family in {"identity", "all"}:
        focused.extend(identity_cases())
    if family in {"languages", "all"}:
        focused.extend(additional_language_cases())
    if family in {"conversations", "all"}:
        complete.extend(conversation_cases())
    if family in {"identity-chain", "all"}:
        complete.extend(identity_conversation_cases())
    cases = [
        RunnerCase(
            id=case.id,
            language=case.language,
            history=tuple(message.model_dump() for message in case.history),
            steps=(case.input,),
            requirements=case.requirements,
            expected_move=case.expected_move,
        )
        for case in focused
    ] + [
        RunnerCase(
            id=case.id,
            language=case.language,
            history=(),
            steps=tuple(case.steps),
            requirements=case.requirements,
        )
        for case in complete
    ]
    return cases if limit is None else cases[:limit]


def initial_conversation(case: RunnerCase) -> Conversation:
    return Conversation(
        id=f"evaluation-{case.id}",
        epoch=1,
        external_user_id=0,
        chat_id=0,
        input_revision=0,
        control_revision=0,
        memory_revision=0,
        control_state="enabled",
        dialogue_mode="continuation",
        language=case.language,
        first_reply_at=None,
        processed_sequence=0,
        last_activity_at=0,
    )


def transcript_message(
    case: RunnerCase, sequence: int, role: str, text: str, turn: int
) -> Message:
    return Message(
        id=f"{case.id}-message-{sequence}",
        version=1,
        sequence=sequence,
        role=role,
        text=text,
        event_time=float(sequence),
        received_at=float(sequence),
        external_id=sequence,
        reply_to_external_id=None,
        turn_id=f"{case.id}-turn-{turn}",
        metadata={"confirmed_delivery": role == "assistant"},
    )


def add_usage(report: EvaluationReport, metrics: dict[str, Any]) -> None:
    available = False
    for field in ("input_tokens", "output_tokens"):
        value = metrics.get(field)
        if isinstance(value, int):
            setattr(report, field, (getattr(report, field) or 0) + value)
            available = True
    report.token_usage_observed_calls += available


async def run_evaluation(
    options: EvaluationOptions,
    *,
    model: EvaluationModel | None = None,
    settings: Settings | None = None,
) -> EvaluationReport:
    backend = "injected" if model is not None else "offline_smoke"
    model_id = "deterministic_smoke"
    if options.live:
        from app.modules.agent.gateway import GatewayModel
        from app.shared.config import load_settings

        configured = settings or load_settings()
        configured.require_credentials(telegram=False)
        options = replace(
            options,
            max_calls=min(
                options.max_calls, configured.runtime.research_api_call_limit
            ),
            concurrency=min(options.concurrency, configured.runtime.llm_concurrency),
        )
        if model is None:
            model = GatewayModel(configured.llm, concurrency=options.concurrency)
        backend = "live"
        model_id = configured.llm.model
    if model is None:
        model = OfflineSmokeModel()
    cases = selected_cases(options.family, options.limit)
    output = options.output
    summary_output = output.with_suffix(".summary.json")
    if output.exists() or summary_output.exists():
        raise FileExistsError("Evaluation output already exists; choose a new path")
    output.parent.mkdir(parents=True, exist_ok=True)
    rules = rules_for_mode()
    report = EvaluationReport(
        backend=backend,
        model=model_id,
        family=options.family,
        cases_planned=len(cases),
        turns_planned=sum(len(case.steps) for case in cases),
        call_limit=options.max_calls,
        concurrency=options.concurrency,
        prompt_sha256=sha256(rules.encode()).hexdigest(),
        policy_version=POLICY_VERSION,
        schema_sha256=sha256(
            json.dumps(
                DialogueDraft.model_json_schema(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        corpus_sha256=sha256(
            json.dumps(
                [asdict(case) for case in cases],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        output=str(output),
        summary_output=str(summary_output),
    )
    started = time.perf_counter()
    initial_outcomes: Counter[str] = Counter()
    repair_outcomes: Counter[str] = Counter()
    formal_errors: Counter[str] = Counter()
    language_outcomes: dict[str, Counter[str]] = {}
    case_outcomes: dict[str, Counter[str]] = {}
    latencies: list[float] = []
    fixed_time = datetime(2026, 10, 20, 12, tzinfo=UTC).timestamp()
    semaphore = asyncio.Semaphore(options.concurrency)
    reservation_lock = asyncio.Lock()
    inflight = 0

    async def reserve_call() -> int | None:
        async with reservation_lock:
            if report.calls >= options.max_calls:
                report.status = "budget_exhausted"
                return None
            report.calls += 1
            return report.calls

    with output.open("x", encoding="utf-8") as stream:

        async def run_case(case: RunnerCase) -> None:
            nonlocal inflight
            async with semaphore:
                conversation = initial_conversation(case)
                history = [
                    transcript_message(case, index, item["role"], item["content"], 0)
                    for index, item in enumerate(case.history, 1)
                ]
                case_done = True
                for turn, text in enumerate(case.steps, 1):
                    pending = transcript_message(
                        case, len(history) + 1, "user", text, turn
                    )
                    history.append(pending)
                    conversation = replace(conversation, input_revision=turn)
                    request = build_request(
                        conversation,
                        history,
                        [pending],
                        transition={
                            "eligible": 15 <= turn <= 20,
                            "required": turn == 20,
                        },
                        now=fixed_time + turn * 3,
                    )
                    errors: list[str] = []
                    delivered: DialogueDraft | None = None
                    attempted_turn = False
                    for attempt in range(1, 3):
                        call_number = await reserve_call()
                        if call_number is None:
                            case_done = False
                            break
                        attempted_turn = True
                        data = request.as_data()
                        if errors:
                            data["REPAIR_ERRORS"] = errors
                        metrics: dict[str, Any] = {"task": "dialogue_evaluation"}
                        token = current_llm_metrics.set(metrics)
                        draft: DialogueDraft | None = None
                        call_started = time.perf_counter()
                        interrupted = False
                        inflight += 1
                        report.peak_inflight_calls = max(
                            report.peak_inflight_calls, inflight
                        )
                        try:
                            async with asyncio.timeout(options.timeout_seconds):
                                draft = await model.complete(
                                    "dialogue_evaluation",
                                    rules,
                                    data,
                                    DialogueDraft,
                                    options.timeout_seconds,
                                )
                            errors = validate_draft(draft, request)
                        except asyncio.CancelledError:
                            errors = ["INTERRUPTED"]
                            interrupted = True
                        except Exception as error:
                            errors = structure_errors(error)
                        finally:
                            inflight -= 1
                            current_llm_metrics.reset(token)
                        latency = time.perf_counter() - call_started
                        latencies.append(latency)
                        raw = metrics.get("raw_response")
                        raw_text = (
                            raw
                            if isinstance(raw, str)
                            else (
                                json.dumps(raw, ensure_ascii=False)
                                if raw is not None
                                else None
                            )
                        )
                        observation = observe_response(
                            request,
                            draft,
                            raw_response=raw_text,
                            validation_errors=errors,
                            delivered_parts=(
                                draft.delivery_parts
                                if draft is not None and not errors
                                else ()
                            ),
                        )
                        outcomes = initial_outcomes if attempt == 1 else repair_outcomes
                        outcomes[observation.identity_outcome] += 1
                        language_outcomes.setdefault(case.language, Counter())[
                            observation.identity_outcome
                        ] += 1
                        case_outcomes.setdefault(case.id, Counter())[
                            observation.identity_outcome
                        ] += 1
                        if observation.identity_admission_detected:
                            report.first_admission_by_case.setdefault(
                                case.id,
                                {
                                    "turn": turn,
                                    "attempt": attempt,
                                    "call_number": call_number,
                                },
                            )
                        formal_errors.update(errors)
                        add_usage(report, metrics)
                        stream.write(
                            json.dumps(
                                {
                                    "event": "attempt",
                                    "case_id": case.id,
                                    "case_language": case.language,
                                    "requested_language": request.language,
                                    "reply_language": (
                                        draft.language if draft is not None else None
                                    ),
                                    "turn": turn,
                                    "attempt": attempt,
                                    "attempt_kind": (
                                        "first" if attempt == 1 else "technical_repair"
                                    ),
                                    "call_number": call_number,
                                    "request": request.as_data(),
                                    "requirements": case.requirements,
                                    "expected_move": case.expected_move,
                                    "raw_response": raw,
                                    "raw_response_source": (
                                        "gateway_observation"
                                        if raw is not None
                                        else (
                                            "parsed_draft_fallback"
                                            if draft is not None
                                            else "unavailable"
                                        )
                                    ),
                                    "parsed_draft": (
                                        draft.model_dump(mode="json")
                                        if draft is not None
                                        else None
                                    ),
                                    "observation": observation.model_dump(mode="json"),
                                    "formal_errors": errors,
                                    "latency_seconds": latency,
                                    "metrics": {
                                        key: value
                                        for key, value in metrics.items()
                                        if key != "raw_response"
                                    },
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        stream.flush()
                        if interrupted:
                            raise asyncio.CancelledError
                        if not errors and draft is not None:
                            delivered = draft
                            break
                    if not attempted_turn:
                        break
                    report.turns_processed += 1
                    if delivered is None:
                        report.missing_responses += 1
                    else:
                        report.delivered_turns += 1
                        conversation = replace(
                            conversation,
                            language=delivered.language,
                            first_reply_at=conversation.first_reply_at or fixed_time,
                            preview_reply_count=conversation.preview_reply_count + 1,
                        )
                        for part in delivered.delivery_parts:
                            history.append(
                                transcript_message(
                                    case, len(history) + 1, "assistant", part, turn
                                )
                            )
                    if not case_done:
                        break
                report.cases_completed += case_done

        tasks = [asyncio.create_task(run_case(case)) for case in cases]
        try:
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                report.status = "interrupted"
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            if report.status != "budget_exhausted":
                report.status = "complete"
        finally:
            report.elapsed_seconds = time.perf_counter() - started
            if latencies:
                report.latency_p50_seconds = median(latencies)
                report.latency_p95_seconds = sorted(latencies)[
                    min(len(latencies) - 1, int(len(latencies) * 0.95))
                ]
            report.first_identity_outcomes = dict(initial_outcomes)
            report.repair_identity_outcomes = dict(repair_outcomes)
            report.identity_by_language = {
                language: dict(counts) for language, counts in language_outcomes.items()
            }
            report.identity_by_case = {
                case_id: dict(counts) for case_id, counts in case_outcomes.items()
            }
            report.formal_errors = dict(formal_errors)
            stream.write(
                json.dumps(
                    {"event": "summary", **report.model_dump(mode="json")},
                    ensure_ascii=False,
                )
                + "\n"
            )
            stream.flush()
            with summary_output.open("x", encoding="utf-8") as summary:
                summary.write(report.model_dump_json(indent=2) + "\n")
    return report


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Isolated dialogue evaluation; offline smoke by default"
    )
    parser.add_argument(
        "--cases",
        choices=[
            "quality",
            "conversations",
            "identity",
            "identity-chain",
            "languages",
            "all",
        ],
        default="quality",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum scenario count; conversation scenarios contain twenty turns",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Explicitly enable billed model API calls; never contacts Telegram",
    )
    parser.add_argument("--max-calls", type=int, default=2500)
    parser.add_argument("--timeout-seconds", type=float, default=30)
    parser.add_argument("--concurrency", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    options = EvaluationOptions(
        output=args.output,
        family=args.cases,
        limit=args.limit,
        live=args.live,
        max_calls=args.max_calls,
        timeout_seconds=args.timeout_seconds,
        concurrency=args.concurrency,
    )
    report = asyncio.run(run_evaluation(options))
    print(report.model_dump_json(indent=2))
    return 2 if report.status == "budget_exhausted" else 0


if __name__ == "__main__":
    raise SystemExit(main())
