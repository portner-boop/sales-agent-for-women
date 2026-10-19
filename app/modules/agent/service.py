import asyncio
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import APIConnectionError, APIStatusError
from pydantic import ValidationError

from app.modules.agent.capabilities import (
    is_photo_request,
    is_standalone_photo_request,
    photo_language,
    photo_refusal,
)
from app.modules.agent.context import ContextBuilder
from app.modules.agent.entities import (
    BuiltContext,
    LanguageModel,
    SchemaT,
    current_llm_metrics,
)
from app.modules.agent.errors import (
    CallBudgetError,
    ModelOutputError,
    ReplyRejectedError,
)
from app.modules.agent.initiative import followup_eligibility
from app.modules.agent.language import (
    language_instruction,
    reply_language,
    reply_validation_instruction,
)
from app.modules.agent.policy import (
    RuleBundle,
    RuleSelector,
)
from app.modules.agent.prompts import (
    CATALOG,
    FOLLOWUP_RULES,
    FOLLOWUP_VALIDATOR_RULES,
    GENERATOR_CONTRACT,
    GENERATOR_RULES,
    INTERPRETER_RULES,
    MEMORY_VALIDATOR_RULES,
    PLATFORM_TRANSITION_RULES,
    RECOVERY_RULES,
    STRUCTURE_REPAIR_RULES,
    VALIDATOR_RULES,
    platform_turn_instruction,
)
from app.modules.agent.recovery import (
    activity_reply,
    can_retry_response,
    failure_code,
    retry_delay,
    structure_errors,
)
from app.modules.agent.repository import AgentRepository
from app.modules.agent.schemas import (
    ContinuingReplyProposal,
    FollowupProposal,
    RecoveryReplyProposal,
    ReplyProposal,
    SemanticVerdict,
    TurnInterpretation,
)
from app.modules.agent.segmentation import delivery_parts
from app.modules.agent.style import refine_day_question, strip_terminal_period
from app.modules.agent.style_plan import (
    apply_casual_imperfection,
    reduce_letter_imperfections,
    semantic_reply,
    supported_imperfection,
)
from app.modules.agent.time import standalone_time_reply
from app.modules.agent.validation import validate_reply
from app.modules.commerce.requests import (
    declines_platform_link,
    platform_request_language,
    requests_platform_info,
    requests_platform_link_again,
)
from app.modules.commerce.service import CommerceService
from app.modules.conversations.entities import Conversation, Turn
from app.modules.conversations.repository import ConversationRepository
from app.modules.conversations.service import ConversationService
from app.modules.delivery.repository import DeliveryRepository
from app.modules.delivery.schemas import OutboxDraft
from app.modules.delivery.timing import response_delays
from app.modules.followups.entities import FollowupTicket
from app.modules.jobs.entities import Job, JobRetryError
from app.modules.memory.service import MemoryService
from app.shared.config import RuntimeSettings
from app.shared.entities import (
    Clock,
    ContextOverflowError,
    EvidenceError,
    ScopeContext,
    StaleStateError,
)
from app.shared.storage import Transaction, fetchone

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    run_id: str
    turn_id: str
    scope_id: str
    epoch: int
    phase: str


@dataclass
class ActiveRun:
    turn: Turn
    conversation: Conversation
    deadline: float
    started: float
    response_received_at: float | None = None
    calls: int = 0
    retries: int = 0
    llm_events: list[dict[str, Any]] = field(default_factory=list)
    validation_events: list[dict[str, Any]] = field(default_factory=list)
    interpretation: TurnInterpretation | None = None
    interpreter_context: dict[str, Any] | None = None
    context: BuiltContext | None = None
    context_reference_time: float | None = None
    proposal: ReplyProposal | None = None
    followup: FollowupTicket | None = None
    on_prepared: Callable[[Transaction], Awaitable[None]] | None = None
    recovering: bool = False
    recovery_invitation_pending: bool = False
    rule_bundle: RuleBundle | None = None
    context_seconds: float = 0
    selection_seconds: float = 0
    pacing_seconds: tuple[float, ...] = ()


class AgentService:
    def __init__(
        self,
        repository: AgentRepository,
        conversations: ConversationRepository,
        memory: MemoryService,
        context_builder: ContextBuilder,
        delivery: DeliveryRepository,
        commerce: CommerceService,
        model: LanguageModel,
        clock: Clock,
        settings: RuntimeSettings,
        checkpointer: Any,
        on_preview_closed: (
            Callable[[Transaction, ScopeContext], Awaitable[None]] | None
        ) = None,
    ) -> None:
        self.repository = repository
        self.conversations = conversations
        self.conversation_service = ConversationService(conversations, clock)
        self.memory = memory
        self.builder = context_builder
        self.delivery = delivery
        self.commerce = commerce
        self.model = model
        self.clock = clock
        self.settings = settings
        self.on_preview_closed = on_preview_closed
        self.catalog = CATALOG
        self.rule_selector = RuleSelector(self.catalog, self.builder.counter.count)
        self.active: dict[str, ActiveRun] = {}
        self.tasks: dict[str, set[asyncio.Task[None]]] = defaultdict(set)
        self.locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        graph = StateGraph(GraphState)
        nodes = {
            "interpret": self._interpret,
            "resolve": self._resolve,
            "context": self._context,
            "generate": self._generate,
            "validate": self._validate,
            "prepare": self._prepare,
        }
        previous = START
        for name, node in nodes.items():
            graph.add_node(name, node)
            graph.add_edge(previous, name)
            previous = name
        graph.add_edge(previous, END)
        self.graph = graph.compile(checkpointer=checkpointer)

    def _remaining(self, run: ActiveRun) -> float:
        remaining = run.deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError("Turn deadline exhausted")
        if run.calls >= self.settings.llm_max_calls:
            raise CallBudgetError("Turn call budget exhausted")
        return remaining

    async def _call(
        self,
        run: ActiveRun,
        task: str,
        rules: str,
        data: dict[str, Any],
        schema: type[SchemaT],
    ) -> SchemaT:
        data = {**data, "POLICY_VERSION": self.catalog.version}
        if task in {"generate", "repair", "recover", "followup"}:
            bundle = run.rule_bundle or self.rule_selector.full(
                "recovery_without_interpretation"
            )
            rules = rules.replace(
                GENERATOR_RULES, GENERATOR_CONTRACT + "\n" + bundle.text
            )
            rules += language_instruction(data["APPLICATION_STATE"]["language"])
            rules += platform_turn_instruction(data.get("PLATFORM_TRANSITION", {}))
        elif task in {"validate", "validate_repair", "validate_recovery"}:
            rules += reply_validation_instruction(
                data["context"]["APPLICATION_STATE"]["language"]
            )
            rules += platform_turn_instruction(
                data["context"].get("PLATFORM_TRANSITION", {})
            )
        for attempt in range(2):
            current = await self.conversation_service.current(run.turn)
            async with self.repository.database.read() as tx:
                killed = await self.delivery.is_killed(tx)
            if killed or not self.commerce.can_reply(current):
                raise StaleStateError("Conversation stopped")
            if run.followup is not None and (
                current.snapshot.control_revision != run.followup.control_revision
                or self.clock.now() >= run.followup.expires_at
            ):
                raise StaleStateError("Follow-up expired or control changed")
            data = await self._fit_request(run, rules, data, schema)
            remaining = self._remaining(run)
            run.calls += 1
            metrics: dict[str, Any] = {"task": task, "attempt": attempt + 1}
            token = current_llm_metrics.set(metrics)
            started = asyncio.get_running_loop().time()
            delay = None
            try:
                timeout = min(remaining, self.settings.llm_attempt_seconds)
                async with asyncio.timeout(timeout):
                    result = await self.model.complete(
                        task, rules, data, schema, timeout
                    )
                if isinstance(result, ReplyProposal):
                    run.validation_events.append(
                        {"stage": task + "_proposal", "proposal": result.model_dump()}
                    )
                return result
            except Exception as error:
                metrics["error_code"] = failure_code(error)
                delay = retry_delay(error)
                if (
                    attempt == 1
                    or run.retries >= 2
                    or delay is None
                    or delay >= self.settings.llm_attempt_seconds
                    or run.calls >= self.settings.llm_max_calls
                    or run.deadline - asyncio.get_running_loop().time() <= delay + 0.1
                ):
                    raise
                run.retries += 1
                run.validation_events.append(
                    {"stage": "retry", "task": task, "error_code": failure_code(error)}
                )
                if isinstance(error, (ValidationError, ModelOutputError)):
                    data = {**data, "STRUCTURE_ERRORS": structure_errors(error)}
                    rules += STRUCTURE_REPAIR_RULES
            finally:
                metrics["seconds"] = asyncio.get_running_loop().time() - started
                run.llm_events.append(metrics)
                current_llm_metrics.reset(token)
            if delay:
                await asyncio.sleep(delay)
        raise AssertionError("Unreachable model attempt")

    async def _fit_request(
        self, run: ActiveRun, rules: str, data: dict[str, Any], schema: type[SchemaT]
    ) -> dict[str, Any]:
        def tokens(payload: dict[str, Any]) -> int:
            return (
                self.builder.counter.count(
                    {
                        "rules": rules,
                        "data": payload,
                        "schema": schema.model_json_schema(),
                    }
                )
                + 500
            )

        total = tokens(data)
        limit = self.builder.settings.context_tokens
        if total <= limit:
            return data
        if run.context is not None and ("context" in data or "CURRENT_INPUT" in data):
            baseline = (
                self.builder.counter.count(
                    {
                        "rules": GENERATOR_RULES,
                        "data": run.context.data,
                        "schema": self._reply_schema(run).model_json_schema(),
                    }
                )
                + 500
            )
            assert run.interpretation is not None
            old_keys = set(run.context.data)
            cited = run.proposal.used_source_message_ids if run.proposal else []
            known_ids = {item["id"] for item in run.context.manifest.messages}
            cited_memory = set(run.proposal.used_memory_ids if run.proposal else [])
            protected_facts = [
                item
                for item in run.context.data["MEMORY_DATA"]["facts"]
                if item["id"] in cited_memory
            ]
            run.interpretation = run.interpretation.model_copy(
                update={
                    "retrieval_predicates": list(
                        dict.fromkeys(
                            [item["predicate"] for item in protected_facts]
                            + run.interpretation.retrieval_predicates
                        )
                    ),
                    "retrieval_subject_ids": list(
                        dict.fromkeys(
                            [item["subject_id"] for item in protected_facts]
                            + run.interpretation.retrieval_subject_ids
                        )
                    ),
                    "referenced_message_ids": list(
                        dict.fromkeys(
                            run.interpretation.referenced_message_ids
                            + [
                                identifier
                                for identifier in cited
                                if identifier in known_ids
                            ]
                        )
                    ),
                }
            )
            reserve = max(0, total - baseline)
            for _ in range(3):
                await self._build_context(run, request_reserve_tokens=reserve)
                data = (
                    {**data, "context": run.context.data}
                    if "context" in data
                    else {
                        **run.context.data,
                        **{
                            key: value
                            for key, value in data.items()
                            if key not in old_keys
                        },
                    }
                )
                overflow = tokens(data) - limit
                if overflow <= 0:
                    return data
                reserve += overflow + 64
        if tokens(data) > limit:
            raise ContextOverflowError("LLM request exceeds its input budget")
        return data

    async def process(self, job: Job) -> None:
        if job.scope is None:
            raise ValueError("Turn job requires a scope")
        scope_id = job.scope.scope_id
        task = asyncio.create_task(self._process(job))
        self.tasks[scope_id].add(task)
        try:
            await task
        except asyncio.CancelledError:
            parent = asyncio.current_task()
            if parent is not None and parent.cancelling():
                raise
        finally:
            self.tasks[scope_id].discard(task)

    async def cancel_scope(self, scope_id: str) -> None:
        tasks = list(self.tasks.get(scope_id, set()))
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def process_followup(
        self,
        ticket: FollowupTicket,
        on_prepared: Callable[[Transaction], Awaitable[None]],
    ) -> bool:
        remaining = min(
            self.settings.turn_deadline_seconds,
            ticket.expires_at - self.clock.now(),
        )
        if remaining <= 0:
            return False
        try:
            async with asyncio.timeout(remaining):
                return await self._process_followup(ticket, on_prepared)
        except TimeoutError as error:
            logger.warning("followup_deadline id=%s", ticket.id)
            raise JobRetryError("FOLLOWUP_DEADLINE_EXCEEDED") from error

    async def _process_followup(
        self,
        ticket: FollowupTicket,
        on_prepared: Callable[[Transaction], Awaitable[None]],
    ) -> bool:
        async with self.locks[ticket.turn.scope.scope_id]:
            try:
                conversation = await self.conversation_service.current(ticket.turn)
            except StaleStateError:
                return False
            started = asyncio.get_running_loop().time()
            queued_seconds = max(0, self.clock.now() - ticket.run_at)
            limit = min(
                self.settings.turn_deadline_seconds,
                ticket.expires_at - self.clock.now(),
            )
            if limit <= 0:
                return False
            run = ActiveRun(
                ticket.turn,
                conversation,
                started + limit,
                started,
                response_received_at=self.clock.now(),
                interpretation=TurnInterpretation(
                    intent="continue_after_silence",
                    dialogue_mode="continuation",
                    language=conversation.language,
                ),
                followup=ticket,
                on_prepared=on_prepared,
            )
            identifier = await self.repository.start(ticket.turn, self.clock.now())
            self.active[identifier] = run
            state: GraphState = {
                "run_id": identifier,
                "turn_id": ticket.turn.id,
                "scope_id": ticket.turn.scope.scope_id,
                "epoch": ticket.turn.scope.epoch,
                "phase": "followup",
            }
            status, code = "skipped", None
            try:
                async with asyncio.timeout(limit):
                    await self._build_context(run)
                    assert run.context is not None
                    if run.context.data["DIALOGUE_PLAN"][
                        "waiting_question_message_ids"
                    ]:
                        run.validation_events.append(
                            {
                                "stage": "followup_skipped",
                                "reason": "LATEST_QUESTION_PENDING",
                            }
                        )
                        return False
                    if not run.context.data["DIALOGUE_PLAN"]["questions_allowed"]:
                        run.validation_events.append(
                            {
                                "stage": "followup_skipped",
                                "reason": "QUESTIONS_NOT_ALLOWED",
                            }
                        )
                        return False
                    run.proposal = await self._call(
                        run,
                        "followup",
                        GENERATOR_RULES + FOLLOWUP_RULES,
                        run.context.data,
                        FollowupProposal,
                    )
                    try:
                        await self._validate(state)
                    except ReplyRejectedError as error:
                        await self._recover(identifier, error)
                        if (
                            isinstance(run.proposal, FollowupProposal)
                            and run.proposal.should_send
                        ):
                            status = "prepared"
                        return status == "prepared"
                    if (
                        isinstance(run.proposal, FollowupProposal)
                        and run.proposal.should_send
                    ):
                        await self._prepare(state)
                        status = "prepared"
            except asyncio.CancelledError:
                status, code = "cancelled", "INTERRUPTED"
                raise
            except StaleStateError:
                status, code = "cancelled", "STALE_STATE"
            except Exception as error:
                code = failure_code(error)
                run.validation_events.append(
                    {"stage": "followup_failure", "error_code": code}
                )
                logger.warning("followup_skipped run=%s code=%s", identifier, code)
                raise JobRetryError(
                    code,
                    retry_after=retry_delay(error) or 0,
                    permanent=not can_retry_response(error),
                ) from error
            finally:
                self.active.pop(identifier, None)
                await self.repository.finish(
                    identifier,
                    status,
                    self.clock.now(),
                    run.context.manifest if run.context else None,
                    {
                        "origin": "followup",
                        "llm_calls": run.calls,
                        "llm_retries": run.retries,
                        "llm_events": run.llm_events,
                        "validation_events": run.validation_events,
                        "elapsed_seconds": asyncio.get_running_loop().time() - started,
                        "queued_seconds": queued_seconds,
                        "context_seconds": run.context_seconds,
                        "selection_seconds": run.selection_seconds,
                        "pacing_seconds": list(run.pacing_seconds),
                        "policy_version": self.catalog.version,
                    },
                    code,
                )
            return status == "prepared"

    async def _process(self, job: Job) -> None:
        if job.scope is None:
            raise ValueError("Turn job requires a scope")
        cutoff = self.clock.now() - 180
        async with self.repository.database.read() as tx:
            fresh = await fetchone(
                tx,
                "SELECT 1 FROM messages m JOIN message_versions v ON v.message_id=m.id AND v.version=m.current_version WHERE m.scope_id=? AND m.epoch=? AND m.pending=1 AND v.received_at>? LIMIT 1",
                (job.scope.scope_id, job.scope.epoch, cutoff),
            )
        if fresh is not None:
            await self.conversations.expire_pending(job.scope, cutoff)
        async with self.repository.database.read() as tx:
            row = await fetchone(
                tx,
                "SELECT MIN(v.received_at) received_at FROM messages m JOIN message_versions v ON v.message_id=m.id AND v.version=m.current_version WHERE m.scope_id=? AND m.epoch=? AND m.pending=1",
                (job.scope.scope_id, job.scope.epoch),
            )
        if row is None or row["received_at"] is None:
            return
        received_at = float(row["received_at"])
        remaining = 180 - max(0, self.clock.now() - received_at)
        waiting = asyncio.get_running_loop().time()
        if remaining <= 0 or job.attempts > 3:
            await self._process_locked(job, waiting, fallback_only=True)
            return
        try:
            async with asyncio.timeout(remaining):
                await self._process_locked(job, waiting)
        except TimeoutError:
            await self._process_locked(job, waiting, fallback_only=True)

    async def _process_locked(
        self, job: Job, waiting: float, fallback_only: bool = False
    ) -> None:
        assert job.scope is not None
        async with self.locks[job.scope.scope_id]:
            conversation = await self.conversations.get(job.scope.scope_id)
            if conversation.scope != job.scope:
                raise StaleStateError("Turn job belongs to an old generation")
            turn = await self.conversations.claim_turn(job.scope, self.clock.now())
            if turn is None:
                return
            platform_info_requested = await self._platform_info_requested(turn)
            async with self.repository.database.read() as tx:
                killed = await self.delivery.is_killed(tx)
            if killed or (
                not self.commerce.can_reply(conversation)
                and not (
                    platform_info_requested
                    and conversation.control_state == "enabled"
                    and conversation.purchase_state != "confirmed"
                )
            ):

                async def skip(tx: Transaction) -> None:
                    await self.conversations.finish_turn(tx, turn, "skipped")

                await self.repository.database.write(skip)
                return
            started = asyncio.get_running_loop().time()
            limit = self.settings.turn_deadline_seconds
            received_at = min(message.received_at for message in turn.messages)
            queued = max(0, self.clock.now() - received_at)
            limit = min(limit, max(0.001, 180 - queued))
            final_attempt = job.is_final_attempt or queued + limit >= 180
            run = ActiveRun(
                turn,
                conversation,
                started + limit,
                started,
                response_received_at=received_at,
            )
            identifier = await self.repository.start(turn, self.clock.now())
            self.active[identifier] = run
            status, code = "prepared", None
            retry_required = False
            retry_allowed = True
            retry_after = 0.0
            try:
                if fallback_only or queued >= 180:
                    status, code = "fallback", "RESPONSE_DEADLINE"
                    retry_required = not await self._fallback(
                        run, code, final_attempt=True
                    )
                    if retry_required:
                        status = "unanswered"
                else:
                    async with asyncio.timeout(limit):
                        if platform_info_requested:
                            await self._prepare_platform_info(run)
                        else:
                            await self.graph.ainvoke(
                                {
                                    "run_id": identifier,
                                    "turn_id": turn.id,
                                    "scope_id": turn.scope.scope_id,
                                    "epoch": turn.scope.epoch,
                                    "phase": "start",
                                },
                                {
                                    "configurable": {
                                        "thread_id": f"{turn.scope.scope_id}:{turn.scope.epoch}"
                                    }
                                },
                            )
            except StaleStateError:
                status, code = "cancelled", "STALE_STATE"
            except asyncio.CancelledError:
                status, code = "cancelled", "INTERRUPTED"
                raise
            except Exception as error:
                status, code = "fallback", failure_code(error)
                retry_allowed = can_retry_response(error)
                retry_after = retry_delay(error) or 0.0
                run.validation_events.append(
                    {
                        "stage": "failure",
                        "type": code,
                        "reason": (
                            str(error)[:256]
                            if type(error)
                            in {
                                ValueError,
                                EvidenceError,
                                ContextOverflowError,
                                ReplyRejectedError,
                            }
                            else None
                        ),
                    }
                )
                try:
                    if self._has_photo_request(run):
                        await self._fallback(
                            run, code, final_attempt or not retry_allowed
                        )
                        run.validation_events.append({"stage": "photo_failure_refusal"})
                        status, code = "prepared", None
                    elif isinstance(
                        error, (ValueError, EvidenceError, ReplyRejectedError)
                    ):
                        await self._recover(identifier, error)
                        status, code = "prepared", None
                    else:
                        retry_required = not await self._fallback(
                            run, code, final_attempt or not retry_allowed
                        )
                except asyncio.CancelledError:
                    status, code = "cancelled", "INTERRUPTED"
                    raise
                except StaleStateError:
                    status, code = "cancelled", "STALE_STATE"
                except Exception as recovery_error:
                    code = failure_code(recovery_error)
                    retry_allowed = can_retry_response(recovery_error)
                    retry_after = retry_delay(recovery_error) or 0.0
                    run.validation_events.append(
                        {"stage": "recovery_failure", "error_code": code}
                    )
                    retry_required = not await self._fallback(
                        run, code, final_attempt or not retry_allowed
                    )
                if retry_required:
                    status = (
                        "unanswered"
                        if final_attempt or not retry_allowed
                        else "deferred"
                    )
                if status in {"fallback", "deferred", "unanswered"}:
                    logger.warning("agent_failed run=%s code=%s", identifier, code)
            finally:
                self.active.pop(identifier, None)
                await self.repository.finish(
                    identifier,
                    status,
                    self.clock.now(),
                    run.context.manifest if run.context else None,
                    {
                        "llm_calls": run.calls,
                        "llm_retries": run.retries,
                        "llm_events": run.llm_events,
                        "validation_events": run.validation_events,
                        "elapsed_seconds": asyncio.get_running_loop().time() - started,
                        "queued_seconds": queued,
                        "context_seconds": run.context_seconds,
                        "selection_seconds": run.selection_seconds,
                        "pacing_seconds": list(run.pacing_seconds),
                        "lock_seconds": started - waiting,
                        "policy_version": self.catalog.version,
                        "generation_seconds": sum(
                            e["seconds"]
                            for e in run.llm_events
                            if e["task"] in {"generate", "followup"}
                        ),
                        "repair_seconds": sum(
                            e["seconds"]
                            for e in run.llm_events
                            if e["task"]
                            in {
                                "repair",
                                "recover",
                                "validate_repair",
                                "validate_recovery",
                            }
                        ),
                    },
                    code,
                )
            if retry_required:
                raise JobRetryError(
                    code or "RESPONSE_UNAVAILABLE",
                    retry_after,
                    final_attempt or not retry_allowed,
                )

    async def _platform_info_requested(self, turn: Turn) -> bool:
        direct = [
            message
            for message in turn.messages
            if not message.metadata.get("forwarded")
        ]
        if direct and declines_platform_link(direct[-1].text):
            return False
        if any(requests_platform_info(message.text) for message in direct):
            return True
        if not any(requests_platform_link_again(message.text) for message in direct):
            return False
        history = await self.conversations.messages(turn.scope)
        delivered = [message for message in history if message.role != "user"]
        return any(
            "behappy2day" in message.text.casefold() for message in delivered[-3:]
        )

    async def _prepare_platform_info(self, run: ActiveRun) -> None:
        async def prepare(tx: Transaction) -> None:
            current = await self.conversations.get(run.turn.scope.scope_id, tx)
            if (
                current.scope != run.turn.scope
                or current.input_revision != run.turn.input_revision
                or current.snapshot.control_revision
                != run.conversation.snapshot.control_revision
                or current.control_state != "enabled"
                or current.purchase_state == "confirmed"
                or await self.delivery.is_killed(tx)
            ):
                raise StaleStateError("Platform information request changed")
            language = platform_request_language(
                " ".join(message.text for message in run.turn.messages),
                current.language,
            )
            text = self.commerce.platform_info_text(language)
            now = self.clock.now()
            delay = self._reply_delays(run, (text,), current, now)[0]
            await tx.execute(
                "UPDATE conversations SET language=? WHERE id=? AND epoch=?",
                (language, current.id, current.epoch),
            )
            await self.delivery.prepare(
                tx,
                current.snapshot,
                OutboxDraft(
                    action_key=f"platform-info:{run.turn.id}",
                    kind="service",
                    text=text,
                    deadline=now + delay + 30,
                    delay_seconds=delay,
                ),
                now,
                {
                    "origin": "platform_info",
                    "reply": text,
                    "dialogue_act": "platform_info",
                    "question_goal": None,
                    "followup_allowed": False,
                },
                run.turn.id,
            )
            await self.conversations.finish_turn(tx, run.turn, "prepared")

        await self.repository.database.write(prepare)
        run.validation_events.append({"stage": "server_platform_info"})

    async def _interpret(self, state: GraphState) -> dict[str, str]:
        run = self.active[state["run_id"]]
        text = " ".join(message.text for message in run.turn.messages)
        if self._standalone_photo_request(run):
            run.interpretation = TurnInterpretation(
                intent="photo_request",
                dialogue_mode="continuation",
                language=photo_language(text, self._input_language(run)),
            )
            return {"phase": "interpreted"}
        data = await self.builder.interpreter_data(run.turn, run.conversation)
        run.interpreter_context = data
        if self.builder.interpreter_tokens(data) > self.builder.settings.context_tokens:
            raise ContextOverflowError("Interpreter context exceeds its budget")
        run.interpretation = await self._call(
            run, "interpret", INTERPRETER_RULES, data, TurnInterpretation
        )
        run.interpretation = run.interpretation.model_copy(
            update={
                "language": reply_language(
                    text, run.conversation.language, run.interpretation.language
                )
            }
        )
        return {"phase": "interpreted"}

    async def _resolve(self, state: GraphState) -> dict[str, str]:
        run = self.active[state["run_id"]]
        interpretation = run.interpretation
        assert interpretation is not None
        interpretation = interpretation.model_copy(
            update={
                "memory_operations": [
                    (
                        operation.model_copy(update={"kind": "conversation_glossary"})
                        if operation.operation == "add"
                        and operation.modality == "figurative"
                        and operation.kind in {"fact", "preference"}
                        else operation
                    )
                    for operation in interpretation.memory_operations
                    if operation.operation not in {"no_change", "needs_clarification"}
                ]
            }
        )
        run.interpretation = interpretation
        try:
            await self._resolve_memory(run, interpretation)
        except EvidenceError as error:
            run.validation_events.append(
                {
                    "stage": "memory_patch_discarded",
                    "error_code": failure_code(error),
                    "reason": str(error)[:256],
                }
            )
            run.interpretation = interpretation.model_copy(
                update={
                    "entities": [],
                    "memory_operations": [],
                    "agenda_operations": [],
                }
            )
            await self.memory.resolve(
                run.conversation.snapshot,
                run.turn.messages,
                [],
                [],
                [],
                interpretation.language,
                interpretation.dialogue_mode,
            )
        run.conversation = await self.conversation_service.current(run.turn)
        return {"phase": "resolved"}

    async def _resolve_memory(
        self, run: ActiveRun, interpretation: TurnInterpretation
    ) -> None:
        if (
            interpretation.memory_operations
            or interpretation.entities
            or interpretation.agenda_operations
        ):
            assert run.interpreter_context is not None
            operations = interpretation.memory_operations
            existing = [
                item
                for item in await self.memory.repository.items(run.turn.scope)
                if any(
                    item["id"] == operation.target_memory_id
                    or item["subject_id"] == operation.subject_id
                    and item["predicate"] == operation.predicate
                    for operation in operations
                )
            ]
            data = {
                "CURRENT_INPUT": run.interpreter_context["CURRENT_INPUT"],
                "REFERENCE_MESSAGES": run.interpreter_context["REFERENCE_MESSAGES"],
                "RECENT_MESSAGES": run.interpreter_context["RECENT_MESSAGES"],
                "existing_facts": existing,
                "patch": interpretation.model_dump(
                    include={"memory_operations", "entities", "agenda_operations"}
                ),
            }
            if (
                self.builder.counter.count(data) + 500
                > self.builder.settings.context_tokens
            ):
                raise EvidenceError("Memory validation input exceeds the budget")
            try:
                verdict = await self._call(
                    run,
                    "validate_memory",
                    MEMORY_VALIDATOR_RULES,
                    data,
                    SemanticVerdict,
                )
            except (
                ValidationError,
                ModelOutputError,
                ContextOverflowError,
                TimeoutError,
                APIConnectionError,
                APIStatusError,
            ) as error:
                if isinstance(error, APIStatusError) and not can_retry_response(error):
                    raise
                raise EvidenceError(
                    "Memory validation unavailable: " + failure_code(error)
                ) from error
            run.validation_events.append({"stage": "memory", **verdict.model_dump()})
            if verdict.decision != "accept":
                if interpretation.memory_operations or interpretation.entities:
                    raise EvidenceError("Memory patch failed semantic validation")
                interpretation = interpretation.model_copy(
                    update={"agenda_operations": []}
                )
                run.interpretation = interpretation
                run.validation_events.append({"stage": "agenda_patch_discarded"})
        await self.memory.resolve(
            run.conversation.snapshot,
            run.turn.messages,
            interpretation.entities,
            interpretation.memory_operations,
            interpretation.agenda_operations,
            interpretation.language,
            interpretation.dialogue_mode,
        )

    async def _context(self, state: GraphState) -> dict[str, str]:
        return await self._build_context(self.active[state["run_id"]])

    async def _build_context(
        self, run: ActiveRun, request_reserve_tokens: int = 0
    ) -> dict[str, str]:
        context_started = asyncio.get_running_loop().time()
        interpretation = run.interpretation
        assert interpretation is not None
        direct_ids = list(interpretation.referenced_message_ids)
        history = await self.conversations.messages(run.turn.scope)
        for message in run.turn.messages:
            if message.reply_to_external_id is not None:
                direct_ids.extend(
                    item.id
                    for item in history
                    if item.external_id == message.reply_to_external_id
                )
        interpretation = interpretation.model_copy(
            update={"referenced_message_ids": list(dict.fromkeys(direct_ids))}
        )
        run.interpretation = interpretation
        if interpretation.recall_scope in {"general", "all"}:
            items = await self.memory.repository.items(run.turn.scope)
            predicates = [
                item["predicate"]
                for item in items
                if interpretation.recall_scope == "all"
                or item["predicate"]
                in {
                    "preferred_name",
                    "name",
                    "home_city",
                    "city",
                    "occupation",
                    "children_count",
                    "has_children",
                    "has_pet",
                    "pet_name",
                    "pet_age",
                    "age_years",
                    "breed",
                }
            ]
            interpretation = interpretation.model_copy(
                update={
                    "retrieval_predicates": list(
                        dict.fromkeys(interpretation.retrieval_predicates + predicates)
                    )
                }
            )
            run.interpretation = interpretation
        view = await self.memory.retrieve(
            run.turn.scope,
            interpretation.retrieval_predicates,
            interpretation.retrieval_subject_ids,
            interpretation.retrieval_topic_kinds,
            interpretation.history_query,
            direct_ids,
        )
        selection_started = asyncio.get_running_loop().time()
        run.rule_bundle = self.rule_selector.select(
            interpretation,
            {
                "followup": run.followup is not None,
                "followup_enabled": self.settings.followup_enabled,
                "commerce": self.commerce.settings.preview_enabled,
            },
        )
        run.selection_seconds += asyncio.get_running_loop().time() - selection_started
        if run.context_reference_time is None:
            run.context_reference_time = self.clock.now()
        run.context = await self.builder.build(
            run.turn,
            interpretation,
            run.conversation,
            view,
            request_reserve_tokens=request_reserve_tokens,
            policy_selection={
                "version": run.rule_bundle.version,
                "ids": list(run.rule_bundle.ids),
            },
            reference_time=run.context_reference_time,
        )
        run.context.data["APPLICATION_STATE"][
            "followup_enabled"
        ] = self.settings.followup_enabled
        if self.commerce.settings.preview_enabled:
            run.context.data["ALLOWED_ACTIONS"] = ["none"]
        if run.followup is not None:
            run.context.data["TRIGGER"] = {
                "kind": "silence",
                "followup_id": run.followup.id,
                "seconds_after_timer": max(0, self.clock.now() - run.followup.run_at),
            }
            run.context.data["DIALOGUE_PLAN"]["requires_continuation"] = False
            run.context.manifest.token_count_by_section["rules_and_schema"] = (
                self.builder.counter.count(GENERATOR_RULES + FOLLOWUP_RULES)
                + self.builder.counter.count(FollowupProposal.model_json_schema())
                + 500
            )
        run.context.data["PLATFORM_TRANSITION"] = await self.commerce.transition_state(
            run.conversation
        )
        if run.followup is not None:
            run.context.data["PLATFORM_TRANSITION"] = {
                "eligible": False,
                "required": False,
            }
        if (
            run.context.data["PLATFORM_TRANSITION"]["eligible"]
            or run.context.data["PLATFORM_TRANSITION"]["required"]
        ):
            run.context.data["DIALOGUE_PLAN"]["requires_continuation"] = False
        if run.recovering:
            self._configure_recovery_plan(run)
        run.context.manifest.policy = run.rule_bundle.as_data()
        for section in (
            "POLICY_SELECTION",
            "STYLE_EXAMPLES",
            "APPLICATION_STATE",
            "DIALOGUE_PLAN",
            "TRIGGER",
            "PLATFORM_TRANSITION",
            "ALLOWED_ACTIONS",
        ):
            if section in run.context.data:
                run.context.manifest.token_count_by_section[section] = (
                    self.builder.counter.count(run.context.data[section])
                )
        run.context_seconds += asyncio.get_running_loop().time() - context_started
        return {"phase": "context_ready"}

    async def _generate(self, state: GraphState) -> dict[str, str]:
        run = self.active[state["run_id"]]
        assert run.context is not None
        if self._standalone_photo_request(run):
            run.proposal = ReplyProposal(
                reply=photo_refusal(run.conversation.language),
                dialogue_act="photo_refusal",
            )
            return {"phase": "generated"}
        run.proposal = await self._call(
            run,
            "generate",
            GENERATOR_RULES + PLATFORM_TRANSITION_RULES,
            run.context.data,
            (
                ContinuingReplyProposal
                if run.context.data["DIALOGUE_PLAN"]["requires_continuation"]
                else ReplyProposal
            ),
        )
        return {"phase": "generated"}

    async def _check_proposal(self, run: ActiveRun, stage: str, task: str) -> list[str]:
        assert run.context and run.proposal and run.interpretation
        if self._standalone_photo_request(run):
            run.proposal = ReplyProposal(
                reply=photo_refusal(run.conversation.language),
                dialogue_act="photo_refusal",
            )
            run.context.data["DIALOGUE_PLAN"]["requires_continuation"] = False
            run.validation_events.append({"stage": stage + "_photo_refusal"})
            return []
        if run.interpretation.dialogue_mode != "closing":
            user_text = " ".join(message.text for message in run.turn.messages)
            for field_name in ("reply", "continuation"):
                part = getattr(run.proposal, field_name)
                if part is None:
                    continue
                refined, question_changed = refine_day_question(part, user_text)
                if refined != part:
                    changes: dict[str, Any] = {field_name: refined}
                    if question_changed:
                        changes.update(
                            question_goal="how_day_is_going",
                            question_time_scope="today",
                            question_part=field_name,
                        )
                    run.proposal = run.proposal.model_copy(update=changes)
                    run.validation_events.append(
                        {"stage": stage + "_day_wording", "part": field_name}
                    )
        for field_name in ("reply", "continuation"):
            part = getattr(run.proposal, field_name)
            if part is None:
                continue
            normalized = strip_terminal_period(
                part.translate(str.maketrans("—–", "--"))
            )
            if normalized != part:
                run.proposal = run.proposal.model_copy(update={field_name: normalized})
                run.validation_events.append(
                    {"stage": stage + "_typography_normalized", "part": field_name}
                )
        transition = run.context.data.get("PLATFORM_TRANSITION", {})
        if transition.get("eligible") and (
            run.proposal.platform_transition or transition.get("required")
        ):
            if run.proposal.followup_allowed:
                run.validation_events.append(
                    {"stage": stage + "_invitation_followup_disabled"}
                )
            run.proposal = run.proposal.model_copy(
                update={
                    "platform_transition": True,
                    "requested_action": "none",
                    "dialogue_act": "platform_redirect",
                    "followup_allowed": False,
                }
            )
        if run.proposal.question_goal:
            stored = await self.conversations.question_state(
                run.turn.scope,
                run.proposal.question_goal,
                run.proposal.question_time_scope,
            )
            existing = {item["id"] for item in run.context.data["AGENDA"]}
            run.context.data["AGENDA"].extend(
                item for item in stored if item["id"] not in existing
            )
        if (
            run.proposal.casual_imperfection_used
            or run.proposal.casual_imperfection is not None
        ) and not supported_imperfection(
            run.proposal.casual_imperfection, run.proposal.full_reply
        ):
            run.validation_events.append(
                {
                    "stage": stage + "_style_metadata",
                    "reason": "UNSUPPORTED_IMPERFECTION",
                }
            )
            run.proposal = run.proposal.model_copy(
                update={"casual_imperfection_used": False, "casual_imperfection": None}
            )
        protected = self._protected_style_texts(run)
        if run.context.data["STYLE_PLAN"]["request_imperfection"]:
            relaxed = apply_casual_imperfection(run.proposal, protected)
            if relaxed is not run.proposal:
                assert relaxed.casual_imperfection is not None
                run.validation_events.append(
                    {
                        "stage": stage + "_casual_imperfection_applied",
                        **relaxed.casual_imperfection.model_dump(),
                    }
                )
                run.proposal = relaxed
        known_memory = set(run.context.manifest.memory_ids)
        known_sources = {message["id"] for message in run.context.manifest.messages}
        memory_ids = [
            identifier
            for identifier in run.proposal.used_memory_ids
            if identifier in known_memory
        ]
        source_ids = [
            identifier
            for identifier in run.proposal.used_source_message_ids
            if identifier in known_sources
        ]
        if (memory_ids or source_ids) and (
            memory_ids != run.proposal.used_memory_ids
            or source_ids != run.proposal.used_source_message_ids
        ):
            run.validation_events.append(
                {
                    "stage": stage + "_reference_metadata_normalized",
                    "discarded_memory_ids": len(run.proposal.used_memory_ids)
                    - len(memory_ids),
                    "discarded_source_ids": len(run.proposal.used_source_message_ids)
                    - len(source_ids),
                }
            )
            run.proposal = run.proposal.model_copy(
                update={
                    "used_memory_ids": memory_ids,
                    "used_source_message_ids": source_ids,
                }
            )
        errors = validate_reply(run.proposal, run.context)
        run.validation_events.append({"stage": stage + "_static", "errors": errors})
        if not errors:
            checked_proposal = semantic_reply(run.proposal, protected)
            verdict = await self._call(
                run,
                task,
                (
                    FOLLOWUP_VALIDATOR_RULES
                    if run.followup is not None
                    else VALIDATOR_RULES
                ),
                {
                    "reply": checked_proposal.model_dump(
                        exclude=(
                            {"followup_allowed"} if run.followup is not None else set()
                        )
                    ),
                    "context": run.context.data,
                },
                SemanticVerdict,
            )
            run.validation_events.append(
                {
                    "stage": stage + "_semantic",
                    "normalized_imperfection": checked_proposal is not run.proposal,
                    **verdict.model_dump(),
                }
            )
            if verdict.decision != "accept":
                errors = verdict.reason_codes or ["SEMANTIC_REJECTION"]
        return errors

    def _reply_schema(self, run: ActiveRun) -> type[ReplyProposal]:
        assert run.context is not None
        if run.followup is not None:
            return FollowupProposal
        if run.recovering:
            return RecoveryReplyProposal
        return (
            ContinuingReplyProposal
            if run.context.data["DIALOGUE_PLAN"]["requires_continuation"]
            else ReplyProposal
        )

    async def _validate(self, state: GraphState) -> dict[str, str]:
        run = self.active[state["run_id"]]
        assert run.context and run.proposal
        errors = await self._check_proposal(run, "draft", "validate")
        if errors:
            run.proposal = await self._call(
                run,
                "repair",
                GENERATOR_RULES
                + PLATFORM_TRANSITION_RULES
                + (FOLLOWUP_RULES if run.followup is not None else ""),
                {
                    **run.context.data,
                    "REJECTED_PROPOSAL": run.proposal.model_dump(),
                    "REPAIR_REASON_CODES": errors,
                    "REPAIR_VERDICT": run.validation_events[-1],
                },
                self._reply_schema(run),
            )
            errors = await self._check_proposal(run, "repair", "validate_repair")
            if errors:
                raise ReplyRejectedError("Repair rejected: " + ",".join(errors))
        return {"phase": "validated"}

    async def _recover(self, identifier: str, error: Exception) -> None:
        run = self.active[identifier]
        remaining = self._remaining(run)
        if self.settings.llm_max_calls - run.calls < 2:
            raise CallBudgetError("Recovery requires generation and independent check")
        run.recovering = True
        async with asyncio.timeout(remaining):
            run.conversation = await self.conversation_service.current(run.turn)
            if run.context is None:
                if run.interpretation is None:
                    run.interpretation = TurnInterpretation(
                        intent="clarify_current_input",
                        dialogue_mode="clarification",
                        language=self._input_language(run),
                        needs_clarification=True,
                    )
                await self._build_context(run)
            assert run.context is not None
            self._configure_recovery_plan(run)
            run.proposal = await self._call(
                run,
                "recover",
                GENERATOR_RULES
                + PLATFORM_TRANSITION_RULES
                + RECOVERY_RULES
                + (FOLLOWUP_RULES if run.followup is not None else ""),
                {
                    **run.context.data,
                    "RECOVERY_FAILURE": failure_code(error),
                    "REJECTED_PROPOSAL": (
                        run.proposal.model_dump() if run.proposal is not None else None
                    ),
                    "REJECTED_CHECKS": [
                        {
                            key: value
                            for key, value in event.items()
                            if key != "proposal"
                        }
                        for event in run.validation_events[-4:]
                        if "proposal" not in event
                    ],
                },
                self._reply_schema(run),
            )
            errors = await self._check_proposal(run, "recovery", "validate_recovery")
            if errors and self._remove_rejected_continuation(run, errors):
                errors = await self._check_proposal(
                    run, "recovery_answer", "validate_recovery"
                )
            if errors:
                raise ReplyRejectedError("Recovery rejected: " + ",".join(errors))
            if (
                isinstance(run.proposal, FollowupProposal)
                and not run.proposal.should_send
            ):
                run.validation_events.append({"stage": "followup_recovery_declined"})
                return
            await self._prepare(
                {
                    "run_id": identifier,
                    "turn_id": run.turn.id,
                    "scope_id": run.turn.scope.scope_id,
                    "epoch": run.turn.scope.epoch,
                    "phase": "recovered",
                }
            )
            run.validation_events.append({"stage": "recovered"})

    def _remove_rejected_continuation(self, run: ActiveRun, errors: list[str]) -> bool:
        assert run.proposal is not None
        if (
            run.followup is not None
            or run.proposal.continuation is None
            or run.calls >= self.settings.llm_max_calls
        ):
            return False
        verdict = run.validation_events[-1]
        quote = verdict.get("violating_text")
        rejected_question = (
            verdict.get("stage") == "recovery_static"
            and bool(errors)
            and set(errors)
            <= {
                "ALREADY_ANSWERED",
                "REPEATED_QUESTION",
                "QUESTION_NOT_ALLOWED",
                "INTERRUPTED_QUESTION_REPLACED",
                "SITE_INTERVIEW_QUESTION",
            }
            and run.proposal.question_part == "continuation"
        )
        checked_proposal = (
            semantic_reply(run.proposal)
            if verdict.get("normalized_imperfection")
            else run.proposal
        )
        rejected_continuation = (
            bool(errors)
            and set(errors) <= {"CONVERSATION.REPETITION", "UNRELATED_QUESTION"}
            and verdict.get("stage") == "recovery_semantic"
            and isinstance(quote, str)
            and bool(quote.strip())
            and checked_proposal.continuation is not None
            and quote in checked_proposal.continuation
            and quote not in checked_proposal.reply
        )
        if not (rejected_question or rejected_continuation):
            return False
        changes: dict[str, Any] = {"continuation": None}
        if run.proposal.question_part != "reply":
            changes.update(
                question_goal=None, question_part=None, question_thread_id=None
            )
        if (
            run.proposal.casual_imperfection is not None
            and run.proposal.casual_imperfection.written not in run.proposal.reply
        ):
            changes.update(casual_imperfection=None, casual_imperfection_used=False)
        run.proposal = run.proposal.model_copy(update=changes)
        assert run.context is not None
        run.context.data["DIALOGUE_PLAN"].update(
            initiative_due=False,
            requires_continuation=False,
            strategy="respond_in_context",
        )
        run.validation_events.append({"stage": "recovery_continuation_removed"})
        return True

    def _configure_recovery_plan(self, run: ActiveRun) -> None:
        assert run.context is not None
        plan = run.context.data["DIALOGUE_PLAN"]
        plan["requires_continuation"] = False
        if run.followup is None:
            transition = run.context.data.get("PLATFORM_TRANSITION", {})
            run.recovery_invitation_pending = run.recovery_invitation_pending or bool(
                transition.get("eligible")
                and (
                    transition.get("required")
                    or run.proposal is not None
                    and run.proposal.platform_transition
                )
            )
            run.context.data["PLATFORM_TRANSITION"] = {
                "eligible": False,
                "required": False,
                "server_invitation_pending": run.recovery_invitation_pending,
            }

    def _reply_delays(
        self,
        run: ActiveRun,
        parts: tuple[str, ...],
        conversation: Conversation,
        now: float,
    ) -> tuple[float, ...]:
        delays = response_delays(
            parts,
            run.response_received_at if run.response_received_at is not None else now,
            now,
            conversation.first_reply_at is None and run.followup is None,
            self.settings,
        )
        run.pacing_seconds = delays
        return delays

    def _protected_style_texts(self, run: ActiveRun) -> list[str]:
        assert run.interpretation is not None and run.context is not None
        return [
            entity.name for entity in run.interpretation.entities if entity.name
        ] + [
            str(fact["value"])
            for fact in run.context.data.get("MEMORY_DATA", {}).get("facts", [])
            if fact.get("value") is not None
        ]

    def _delivery_parts(self, run: ActiveRun) -> tuple[ReplyProposal, ...]:
        assert run.proposal is not None and run.context is not None
        proposal = run.proposal
        plan = run.context.data["STYLE_PLAN"]
        protected = self._protected_style_texts(run)
        dialogue = run.context.data["DIALOGUE_PLAN"]
        split_sentences = bool(
            run.followup is None
            and run.interpretation is not None
            and run.interpretation.dialogue_mode != "closing"
            and run.interpretation.recall_scope == "none"
            and not run.interpretation.detail_requested
            and not proposal.platform_transition
            and proposal.requested_action == "none"
            and dialogue["max_messages"] > 1
        )
        parts = []
        for index, part in enumerate(
            delivery_parts(proposal, split_sentences, protected)
        ):
            text = part.reply
            if (
                part.casual_imperfection is None
                and plan["request_imperfection"]
                and plan["target_message_interval"] == 1
                and not proposal.platform_transition
                and proposal.requested_action == "none"
            ):
                styled = apply_casual_imperfection(part, protected)
                if (
                    styled is not part
                    and styled.casual_imperfection is not None
                    and semantic_reply(styled, protected).reply == text
                ):
                    part = styled
                    run.validation_events.append(
                        {
                            "stage": "delivery_part_style",
                            "part": index,
                            "edit": styled.casual_imperfection.model_dump(),
                        }
                    )
            parts.append(reduce_letter_imperfections(part))
        return tuple(parts)

    async def _prepare(self, state: GraphState) -> dict[str, str]:
        run = self.active[state["run_id"]]
        assert run.proposal and run.context
        proposal = run.proposal
        followup_allowed, followup_reason = followup_eligibility(
            proposal, run.context.data
        )
        if self._standalone_photo_request(run):
            followup_allowed, followup_reason = False, "STANDALONE_PHOTO_BOUNDARY"
        run.validation_events.append(
            {
                "stage": "followup_eligibility",
                "model_allowed": proposal.followup_allowed,
                "allowed": followup_allowed,
                "reason": followup_reason,
                "dialogue_mode": run.context.data["APPLICATION_STATE"]["dialogue_mode"],
            }
        )
        proposal = proposal.model_copy(update={"followup_allowed": followup_allowed})
        run.proposal = proposal
        style_plan = {
            key: value
            for key, value in run.context.data["STYLE_PLAN"].items()
            if key != "visible_forms"
        }
        resume_question = run.context.data["DIALOGUE_PLAN"].get("resume_question")
        response_language = run.context.data["APPLICATION_STATE"]["language"]
        conversation = await self.conversation_service.current(run.turn)
        if (
            conversation.snapshot.control_revision
            != run.context.manifest.control_revision
            or conversation.memory_revision != run.context.manifest.memory_revision
        ):
            raise StaleStateError("Reply snapshot changed")
        kind = "offer" if run.proposal.requested_action == "show_offer" else "reply"
        styled_parts = self._delivery_parts(run)
        parts = (
            (await self.commerce.offer_text(conversation.language),)
            if kind == "offer"
            else tuple(part.reply for part in styled_parts)
        )

        async def prepare(tx: Transaction) -> None:
            current = await self.conversations.get(conversation.id, tx)
            if (
                current.snapshot != conversation.snapshot
                or not self.commerce.can_reply(current)
            ):
                raise StaleStateError("Reply changed before outbox commit")
            remaining = run.deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("Turn deadline expired before outbox commit")
            turn_state = await fetchone(
                tx, "SELECT status FROM turns WHERE id=?", (run.turn.id,)
            )
            if run.followup is None and (
                turn_state is None or turn_state["status"] != "running"
            ):
                raise StaleStateError("Turn no longer preparing")
            now = self.clock.now()
            inflight = await fetchone(
                tx,
                "SELECT COUNT(*) n FROM outbox_messages WHERE scope_id=? AND epoch=? AND kind='reply' AND status='sending' AND predecessor_id IS NULL AND turn_id IS NOT NULL AND COALESCE(json_extract(proposal_json,'$.origin'),'')!='followup'",
                (current.id, current.epoch),
            )
            assert inflight is not None
            allowed = (
                self.commerce.settings.preview_enabled
                and run.followup is None
                and not self.commerce.preview_closed(current)
                and current.preview_reply_count
                >= self.commerce.settings.preview_min_replies
                and await self.commerce.repository.transition_allowed(tx, current.scope)
            )
            limit_required = (
                allowed
                and current.preview_reply_count + inflight["n"]
                >= self.commerce.settings.preview_max_replies - 1
            )
            if (
                proposal.platform_transition
                or limit_required
                or allowed
                and run.recovery_invitation_pending
            ):
                if not allowed:
                    raise StaleStateError("Platform transition is not allowed")
                text = self.commerce.transition_text(
                    response_language, notice=proposal.platform_notice
                )
                delays = self._reply_delays(
                    run, (proposal.full_reply, text), current, now
                )
                deadline = now + sum(delays) + 30
                await self.commerce.repository.close_preview(
                    tx, current.scope, self.clock.now()
                )
                await self.delivery.cancel_pending(tx, current.scope, "PREVIEW_CLOSED")
                if self.on_preview_closed is not None:
                    await self.on_preview_closed(tx, current.scope)
                answer_id = await self.delivery.prepare(
                    tx,
                    current.snapshot,
                    OutboxDraft(
                        action_key=f"turn:{run.turn.id}",
                        kind="reply",
                        text=proposal.full_reply,
                        deadline=deadline,
                        delay_seconds=delays[0],
                    ),
                    now,
                    {
                        **proposal.model_dump(
                            exclude={"continuation", "question_part"}
                        ),
                        "origin": "platform_answer",
                        "reply": proposal.full_reply,
                        "followup_allowed": False,
                        "style_plan": style_plan,
                        "response_language": response_language,
                    },
                    run.turn.id,
                )
                await self.delivery.prepare(
                    tx,
                    current.snapshot,
                    OutboxDraft(
                        action_key=f"platform-preview:{current.id}:{current.epoch}",
                        kind="service",
                        text=text,
                        deadline=deadline,
                        predecessor_id=answer_id,
                        delay_seconds=delays[1],
                    ),
                    self.clock.now(),
                    {
                        "origin": "platform_redirect",
                        "transition_trigger": "limit" if limit_required else "agent",
                        "completed_replies": current.preview_reply_count,
                        "dialogue_act": "platform_redirect",
                        "reply": text,
                        "answer": proposal.full_reply,
                        "used_memory_ids": proposal.used_memory_ids,
                        "used_source_message_ids": proposal.used_source_message_ids,
                        "followup_allowed": False,
                        "response_language": response_language,
                    },
                    run.turn.id,
                )
                await self.conversations.finish_turn(tx, run.turn, "prepared")
                return
            delays = self._reply_delays(run, parts, current, now)
            deadline = now + 30 + sum(delays)
            if run.followup is not None:
                deadline = min(deadline, run.followup.expires_at)
                if now + sum(delays) >= deadline:
                    raise StaleStateError("Follow-up expired before outbox commit")
            predecessor_id = None
            for index, text in enumerate(parts):
                metadata: dict[str, Any] = (
                    {
                        **styled_parts[index].model_dump(
                            exclude={"continuation", "question_part"}
                        ),
                        "reply": text,
                    }
                    if kind != "offer"
                    else {"dialogue_act": "show_offer"}
                )
                if resume_question and run.followup is None:
                    metadata["resume_question_thread_id"] = resume_question["thread_id"]
                metadata["style_plan"] = style_plan
                metadata["response_language"] = response_language
                if len(parts) > 1 and not metadata.get("question_goal"):
                    metadata["dialogue_act"] = "reaction"
                if run.followup is not None:
                    metadata["origin"] = "followup"
                    metadata["followup_id"] = run.followup.id
                predecessor_id = await self.delivery.prepare(
                    tx,
                    current.snapshot,
                    OutboxDraft(
                        action_key=(
                            f"followup:{run.followup.id}"
                            if run.followup is not None
                            else f"turn:{run.turn.id}"
                        )
                        + (
                            f":continuation:{index}"
                            if index > 1
                            else ":continuation" if index else ""
                        ),
                        kind=kind,
                        text=text,
                        deadline=deadline,
                        predecessor_id=predecessor_id,
                        delay_seconds=delays[index],
                    ),
                    self.clock.now(),
                    metadata,
                    run.turn.id,
                )
            await self.conversations.finish_turn(tx, run.turn, "prepared")
            if run.on_prepared is not None:
                await run.on_prepared(tx)

        await self.repository.database.write(prepare)
        return {"phase": "prepared"}

    def _input_language(self, run: ActiveRun) -> str:
        if not run.turn.messages:
            return run.conversation.language
        return reply_language(
            run.turn.messages[-1].text,
            run.conversation.language,
            run.interpretation.language if run.interpretation is not None else None,
        )

    def _time_reply(self, run: ActiveRun) -> str | None:
        if run.followup is not None or len(run.turn.messages) != 1:
            return None
        message = run.turn.messages[0]
        if message.metadata.get("forwarded"):
            return None
        return standalone_time_reply(
            message.text, self.clock.now(), self._input_language(run)
        )

    def _standalone_photo_request(self, run: ActiveRun) -> bool:
        return (
            run.followup is None
            and all(
                not message.metadata.get("forwarded") for message in run.turn.messages
            )
            and is_standalone_photo_request(
                " ".join(message.text for message in run.turn.messages)
            )
        )

    def _has_photo_request(self, run: ActiveRun) -> bool:
        return any(
            not message.metadata.get("forwarded") and is_photo_request(message.text)
            for message in run.turn.messages
        )

    async def _fallback(self, run: ActiveRun, code: str, final_attempt: bool) -> bool:
        photo_requested = self._has_photo_request(run)
        language = self._input_language(run)
        direct_text = (
            run.turn.messages[0].text
            if len(run.turn.messages) == 1
            and not run.turn.messages[0].metadata.get("forwarded")
            else None
        )
        activity_text = activity_reply(direct_text, language) if direct_text else None

        async def prepare(tx: Transaction) -> bool:
            current = await self.conversations.get(run.turn.scope.scope_id, tx)
            if (
                current.scope != run.turn.scope
                or current.input_revision != run.turn.input_revision
                or current.snapshot.control_revision
                != run.conversation.snapshot.control_revision
                or await self.delivery.is_killed(tx)
            ):
                return True
            if self.commerce.can_reply(current):
                text = (
                    photo_refusal(
                        photo_language(
                            " ".join(message.text for message in run.turn.messages),
                            self._input_language(run),
                        )
                    )
                    if photo_requested
                    else self._time_reply(run) or activity_text
                )
                if text is None:
                    await self.conversations.finish_turn(
                        tx, run.turn, "unanswered" if final_attempt else "deferred"
                    )
                    return False
                now = self.clock.now()
                delay = self._reply_delays(run, (text,), current, now)[0]
                await self.delivery.prepare(
                    tx,
                    current.snapshot,
                    OutboxDraft(
                        action_key=f"fallback:{run.turn.id}",
                        kind="reply" if photo_requested else "service",
                        text=text,
                        deadline=now + delay + 30,
                        delay_seconds=delay,
                    ),
                    now,
                    (
                        {
                            "reply": text,
                            "dialogue_act": "photo_refusal",
                            "question_goal": None,
                            "followup_allowed": False,
                        }
                        if photo_requested
                        else {
                            "origin": "response_recovery",
                            "reply": text,
                            "dialogue_act": (
                                "current_activity" if activity_text else "unavailable"
                            ),
                            "question_goal": None,
                            "followup_allowed": False,
                            "error_code": code,
                        }
                    ),
                    turn_id=run.turn.id,
                )
            await self.conversations.finish_turn(tx, run.turn, "prepared")
            return True

        return await self.repository.database.write(prepare)
