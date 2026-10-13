import asyncio
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from app.modules.agent.capabilities import (
    is_contact_request,
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
from app.modules.agent.language import (
    language_instruction,
    reply_language,
    reply_validation_instruction,
)
from app.modules.agent.prompts import (
    FOLLOWUP_RULES,
    FOLLOWUP_VALIDATOR_RULES,
    GENERATOR_RULES,
    INTERPRETER_RULES,
    MEMORY_VALIDATOR_RULES,
    PLATFORM_TRANSITION_RULES,
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
from app.modules.agent.style import refine_day_question, strip_terminal_period
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
    proposal: ReplyProposal | None = None
    followup: FollowupTicket | None = None
    on_prepared: Callable[[Transaction], Awaitable[None]] | None = None
    recovering: bool = False


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
        if task in {"generate", "repair", "recover", "followup"}:
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
                    rules += "\nCorrect STRUCTURE_ERRORS. Return the complete JSON object matching JSON_SCHEMA, with every required field and no extra fields."
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
            await self._build_context(
                run, request_reserve_tokens=max(0, total - baseline) + 128
            )
            data = (
                {**data, "context": run.context.data}
                if "context" in data
                else {
                    **run.context.data,
                    **{
                        key: value for key, value in data.items() if key not in old_keys
                    },
                }
            )
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
                    run.proposal = await self._call(
                        run,
                        "followup",
                        GENERATOR_RULES + FOLLOWUP_RULES,
                        run.context.data,
                        FollowupProposal,
                    )
                    await self._validate(state)
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
                    },
                    code,
                )
            return status == "prepared"

    async def _process(self, job: Job) -> None:
        if job.scope is None:
            raise ValueError("Turn job requires a scope")
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
            limit = (
                self.settings.first_deadline_seconds
                if conversation.first_reply_at is None
                else self.settings.turn_deadline_seconds
            )
            received_at = min(message.received_at for message in turn.messages)
            retry_requested_at = job.payload.get("retry_requested_at")
            if retry_requested_at is None and (
                job.attempts > 1
                or any(message.turn_id is not None for message in turn.messages)
            ):
                retry_requested_at = self.clock.now()
            if retry_requested_at is not None:
                if (
                    not isinstance(retry_requested_at, (int, float))
                    or retry_requested_at > self.clock.now()
                ):
                    raise ValueError("Invalid retry request time")
                received_at = max(received_at, retry_requested_at)
            queued = max(0, self.clock.now() - received_at)
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
                            run, code, job.is_final_attempt or not retry_allowed
                        )
                        run.validation_events.append({"stage": "photo_failure_refusal"})
                        status, code = "prepared", None
                    elif isinstance(error, (ValueError, EvidenceError)):
                        await self._recover(identifier, error)
                        status, code = "prepared", None
                    else:
                        retry_required = not await self._fallback(
                            run, code, job.is_final_attempt or not retry_allowed
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
                        run, code, job.is_final_attempt or not retry_allowed
                    )
                if retry_required:
                    status = (
                        "unanswered"
                        if job.is_final_attempt or not retry_allowed
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
                    },
                    code,
                )
            if retry_required:
                raise JobRetryError(
                    code or "RESPONSE_UNAVAILABLE", retry_after, not retry_allowed
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
                    operation
                    for operation in interpretation.memory_operations
                    if operation.operation not in {"no_change", "needs_clarification"}
                ]
            }
        )
        run.interpretation = interpretation
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
                raise ContextOverflowError("Memory validation input exceeds the budget")
            verdict = await self._call(
                run, "validate_memory", MEMORY_VALIDATOR_RULES, data, SemanticVerdict
            )
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
        run.conversation = await self.conversation_service.current(run.turn)
        return {"phase": "resolved"}

    async def _context(self, state: GraphState) -> dict[str, str]:
        return await self._build_context(self.active[state["run_id"]])

    async def _build_context(
        self, run: ActiveRun, request_reserve_tokens: int = 0
    ) -> dict[str, str]:
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
        view = await self.memory.retrieve(
            run.turn.scope,
            interpretation.retrieval_predicates,
            interpretation.retrieval_subject_ids,
            interpretation.retrieval_topic_kinds,
            interpretation.history_query,
            direct_ids,
        )
        run.context = await self.builder.build(
            run.turn,
            interpretation,
            run.conversation,
            view,
            request_reserve_tokens=request_reserve_tokens,
        )
        run.context.data["APPLICATION_STATE"]["followup_enabled"] = (
            self.settings.followup_enabled and run.followup is None
        )
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
        for section in (
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
        if (
            run.followup is None
            and run.proposal.continuation is not None
            and run.proposal.question_goal is None
            and "?" not in run.proposal.full_reply
            and any(
                not message.metadata.get("forwarded")
                and is_contact_request(message.text)
                for message in run.turn.messages
            )
        ):
            run.proposal = run.proposal.model_copy(
                update={
                    "reply": run.proposal.reply.rstrip(". ")
                    + ". "
                    + run.proposal.continuation,
                    "continuation": None,
                    "question_part": None,
                    "followup_allowed": False,
                }
            )
            run.context.data["DIALOGUE_PLAN"]["requires_continuation"] = False
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
        run.proposal = run.proposal.model_copy(
            update={
                "reply": strip_terminal_period(run.proposal.reply),
                "continuation": (
                    strip_terminal_period(run.proposal.continuation)
                    if run.proposal.continuation is not None
                    else None
                ),
            }
        )
        transition = run.context.data.get("PLATFORM_TRANSITION", {})
        if transition.get("eligible") and (
            run.proposal.platform_transition or transition.get("required")
        ):
            run.proposal = run.proposal.model_copy(
                update={
                    "platform_transition": True,
                    "requested_action": "none",
                    "dialogue_act": "platform_redirect",
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
        errors = validate_reply(run.proposal, run.context)
        run.validation_events.append({"stage": stage + "_static", "errors": errors})
        if not errors:
            verdict = await self._call(
                run,
                task,
                VALIDATOR_RULES
                + (FOLLOWUP_VALIDATOR_RULES if run.followup is not None else ""),
                {"reply": run.proposal.model_dump(), "context": run.context.data},
                SemanticVerdict,
            )
            run.validation_events.append(
                {"stage": stage + "_semantic", **verdict.model_dump()}
            )
            if verdict.decision != "accept":
                errors = verdict.reason_codes or ["SEMANTIC_REJECTION"]
        return errors

    def _reply_schema(self, run: ActiveRun) -> type[ReplyProposal]:
        assert run.context is not None
        if run.recovering:
            return RecoveryReplyProposal
        if run.followup is not None:
            return FollowupProposal
        return (
            ContinuingReplyProposal
            if run.context.data["DIALOGUE_PLAN"]["requires_continuation"]
            else ReplyProposal
        )

    async def _validate(self, state: GraphState) -> dict[str, str]:
        run = self.active[state["run_id"]]
        assert run.context and run.proposal
        if isinstance(run.proposal, FollowupProposal) and not run.proposal.should_send:
            return {"phase": "skipped"}
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
            if (
                isinstance(run.proposal, FollowupProposal)
                and not run.proposal.should_send
            ):
                return {"phase": "skipped"}
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
                run.interpretation = TurnInterpretation(
                    intent="clarify_current_input",
                    dialogue_mode="clarification",
                    language=self._input_language(run),
                    needs_clarification=True,
                )
                await self._build_context(run)
            assert run.context is not None
            run.proposal = await self._call(
                run,
                "recover",
                GENERATOR_RULES
                + PLATFORM_TRANSITION_RULES
                + "\nRECOVERY: Answer the actual current input using the supplied originals and active boundaries. Correct the concrete rejected claim while preserving supported content, Marko's personality and the intended conversational move. A relevant question, continuation and followup_allowed=true remain valid in an open conversation; a previous failed draft is not a user request for silence. requested_action=none. Do not repeat the failed claim, invent facts, claim rejected memory was stored, or add a service-error apology. If evidence is insufficient, ask one useful clarification when allowed. The current platform contract still applies. This candidate will be independently checked.",
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
            if errors:
                raise ReplyRejectedError("Recovery rejected: " + ",".join(errors))
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

    def _reply_delays(
        self,
        run: ActiveRun,
        parts: tuple[str, ...],
        conversation: Conversation,
        now: float,
    ) -> tuple[float, ...]:
        return response_delays(
            parts,
            run.response_received_at if run.response_received_at is not None else now,
            now,
            conversation.first_reply_at is None and run.followup is None,
            self.settings,
        )

    async def _prepare(self, state: GraphState) -> dict[str, str]:
        run = self.active[state["run_id"]]
        assert run.proposal and run.context
        proposal = run.proposal
        conversation = await self.conversation_service.current(run.turn)
        if (
            conversation.snapshot.control_revision
            != run.context.manifest.control_revision
            or conversation.memory_revision != run.context.manifest.memory_revision
        ):
            raise StaleStateError("Reply snapshot changed")
        kind = "offer" if run.proposal.requested_action == "show_offer" else "reply"
        parts = (
            (await self.commerce.offer_text(conversation.language),)
            if kind == "offer"
            else proposal.parts
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
            now = self.clock.now()
            inflight = await fetchone(
                tx,
                "SELECT COUNT(*) n FROM outbox_messages WHERE scope_id=? AND epoch=? AND kind='reply' AND status='sending' AND predecessor_id IS NULL AND turn_id IS NOT NULL AND COALESCE(json_extract(proposal_json,'$.origin'),'')!='followup'",
                (current.id, current.epoch),
            )
            assert inflight is not None
            limit_required = (
                self.commerce.settings.preview_enabled
                and run.followup is None
                and current.preview_reply_count + inflight["n"]
                >= self.commerce.settings.preview_max_replies - 1
            )
            if proposal.platform_transition or limit_required:
                allowed = (
                    self.commerce.settings.preview_enabled
                    and current.preview_reply_count
                    >= self.commerce.settings.preview_min_replies
                    and await self.commerce.repository.transition_allowed(
                        tx, current.scope
                    )
                )
                if proposal.platform_transition and not allowed and not limit_required:
                    raise StaleStateError("Platform transition is not allowed")
                text = (
                    self.commerce.transition_text(
                        current.language, proposal.full_reply, proposal.platform_notice
                    )
                    if allowed
                    else self.commerce.farewell_text(
                        current.language, proposal.full_reply, proposal.platform_notice
                    )
                )
                origin = "platform_redirect" if allowed else "preview_closing"
                delay = self._reply_delays(run, (text,), current, now)[0]
                await self.commerce.repository.close_preview(
                    tx, current.scope, self.clock.now()
                )
                await self.delivery.cancel_pending(tx, current.scope, "PREVIEW_CLOSED")
                if self.on_preview_closed is not None:
                    await self.on_preview_closed(tx, current.scope)
                await self.delivery.prepare(
                    tx,
                    current.snapshot,
                    OutboxDraft(
                        action_key=f"platform-preview:{current.id}:{current.epoch}",
                        kind="service",
                        text=text,
                        deadline=now + delay + 30,
                        delay_seconds=delay,
                    ),
                    self.clock.now(),
                    {
                        "origin": origin,
                        "transition_trigger": "limit" if limit_required else "agent",
                        "completed_replies": current.preview_reply_count,
                        "dialogue_act": "platform_redirect",
                        "reply": text,
                        "answer": proposal.full_reply,
                        "used_memory_ids": proposal.used_memory_ids,
                        "used_source_message_ids": proposal.used_source_message_ids,
                        "followup_allowed": False,
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
                        **proposal.model_dump(
                            exclude={"continuation", "question_part"}
                        ),
                        "reply": text,
                    }
                    if kind != "offer"
                    else {"dialogue_act": "show_offer"}
                )
                if index != proposal.question_part_index:
                    metadata["question_goal"] = None
                if len(parts) == 2 and not metadata.get("question_goal"):
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
                        + (":continuation" if index else ""),
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
                        tx, run.turn, "skipped" if final_attempt else "deferred"
                    )
                    return False
                now = self.clock.now()
                await self.delivery.prepare(
                    tx,
                    current.snapshot,
                    OutboxDraft(
                        action_key=f"fallback:{run.turn.id}",
                        kind="reply" if photo_requested else "service",
                        text=text,
                        deadline=now + 30,
                    ),
                    self.clock.now(),
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
