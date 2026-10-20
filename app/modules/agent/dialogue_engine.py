import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from app.modules.agent.entities import LanguageModel, current_llm_metrics
from app.modules.agent.errors import ReplyRejectedError
from app.modules.agent.recovery import (
    can_retry_response,
    failure_code,
    retry_delay,
    structure_errors,
)
from app.modules.agent.repository import AgentRepository
from app.modules.commerce.service import CommerceService
from app.modules.conversations.entities import Conversation, Turn
from app.modules.conversations.repository import ConversationRepository
from app.modules.delivery.repository import DeliveryRepository
from app.modules.delivery.schemas import OutboxDraft
from app.modules.delivery.timing import DeliveryTiming
from app.modules.dialogue.context import build_request
from app.modules.dialogue.presentation import CHAT_RENDERING
from app.modules.dialogue.prompts import POLICY_VERSION, rules_for_mode
from app.modules.dialogue.research import observe_response
from app.modules.dialogue.schemas import (
    DialogueDraft,
    DialogueRequest,
    MemoryCandidate,
    response_schema,
)
from app.modules.dialogue.tokens import ContextBudget
from app.modules.dialogue.validation import (
    grounding_errors,
    memory_candidate_errors,
    validate_draft,
)
from app.modules.followups.entities import FollowupTicket
from app.modules.jobs.entities import Job
from app.modules.memory.schemas import MemoryOperation, SourceReference
from app.modules.memory.service import MemoryService
from app.shared.config import RuntimeSettings
from app.shared.entities import (
    Clock,
    ContextOverflowError,
    ScopeContext,
    StaleStateError,
)
from app.shared.storage import Transaction, fetchone


def service_notice(language: str, kind: str) -> str:
    notices = {
        "failure": {
            "ru": "Не удалось отправить ответ. Попробуй написать ещё раз",
            "en": "The reply could not be completed. Please try again",
            "es": "No se pudo completar la respuesta. Vuelve a intentarlo",
            "pt": "Não foi possível concluir a resposta. Tente novamente",
            "fr": "La réponse n’a pas pu être envoyée. Essaie de m’écrire à nouveau",
            "de": "Die Antwort konnte nicht gesendet werden. Schreib bitte noch einmal",
            "ar": "تعذر إرسال الرد. حاولي الكتابة مرة أخرى",
            "zh": "回复未能完成，请再发一条消息",
        },
        "overflow": {
            "ru": "Сообщение не помещается в контекст этой беседы. Пожалуйста, сократи его",
            "en": "This message exceeds the conversation context. Please shorten it",
            "es": "El mensaje supera el contexto de esta conversación. Acórtalo, por favor",
            "pt": "A mensagem excede o contexto desta conversa. Por favor, encurte-a",
            "fr": "Le message dépasse la capacité de cette conversation. Raccourcis-le, s’il te plaît",
            "de": "Die Nachricht ist für diesen Gesprächskontext zu lang. Bitte kürze sie",
            "ar": "الرسالة أطول من سعة هذه المحادثة. يرجى اختصارها",
            "zh": "消息超出了本次对话的容量，请缩短一些",
        },
        "invitation": {
            "ru": "Если хочешь продолжить, на BeHappy2Day общение платное. Перенос этой беседы пока проверяется в тестовом режиме",
            "en": "Further chatting on BeHappy2Day is paid. Transfer of this conversation is currently a test integration",
            "es": "Continuar en BeHappy2Day es de pago. La transferencia de esta conversación aún está en pruebas",
            "pt": "Continuar no BeHappy2Day é pago. A transferência desta conversa ainda está em testes",
            "fr": "La suite sur BeHappy2Day est payante. Le transfert de cette conversation est encore en phase de test",
            "de": "Die Fortsetzung auf BeHappy2Day ist kostenpflichtig. Die Übertragung dieses Gesprächs wird noch getestet",
            "ar": "مواصلة المحادثة على BeHappy2Day مدفوعة. نقل هذه المحادثة ما زال قيد الاختبار",
            "zh": "在 BeHappy2Day 继续聊天需要付费。本次对话的转移功能仍在测试中",
        },
        "closing": {
            "ru": "Пробная беседа здесь завершена",
            "en": "The trial conversation here has ended",
            "es": "La conversación de prueba aquí ha terminado",
            "pt": "A conversa de teste aqui terminou",
            "fr": "La conversation d’essai est terminée ici",
            "de": "Das Probegespräch hier ist beendet",
            "ar": "انتهت المحادثة التجريبية هنا",
            "zh": "这里的试聊已经结束",
        },
    }
    choices = notices[kind]
    return choices.get(language.split("-")[0], choices["en"])


@dataclass
class DialogueRun:
    turn: Turn
    conversation: Conversation
    identifier: str
    deadline: float
    started: float
    queued_seconds: float
    reaction_started_at: float
    task: asyncio.Task[Any] | None = None
    first_reply_target_at: float | None = None
    request: DialogueRequest | None = None
    draft: DialogueDraft | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    memory_errors: list[str] = field(default_factory=list)
    followup: FollowupTicket | None = None
    on_prepared: Callable[[Transaction], Awaitable[None]] | None = None


class DialogueState(TypedDict):
    run_id: str


class DialogueEngine:
    def __init__(
        self,
        repository: AgentRepository,
        conversations: ConversationRepository,
        memory: MemoryService,
        builder: ContextBudget,
        delivery: DeliveryRepository,
        commerce: CommerceService,
        model: LanguageModel,
        clock: Clock,
        settings: RuntimeSettings,
    ) -> None:
        self.repository, self.conversations, self.memory = (
            repository,
            conversations,
            memory,
        )
        self.builder, self.delivery, self.commerce = builder, delivery, commerce
        self.model, self.clock, self.settings = model, clock, settings
        self.locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.active: dict[str, DialogueRun] = {}
        graph = StateGraph(DialogueState)
        graph.add_node("generate", self._generate)
        graph.add_node("prepare", self._prepare)
        graph.add_edge(START, "generate")
        graph.add_edge("generate", "prepare")
        graph.add_edge("prepare", END)
        self.graph = graph.compile()

    async def cancel_obsolete(self, scope: ScopeContext, revision: int) -> None:
        tasks = {
            run.task
            for run in self.active.values()
            if run.turn.scope == scope
            and run.turn.input_revision < revision
            and run.task is not None
            and run.task is not asyncio.current_task()
        }
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def process(self, job: Job) -> None:
        if job.scope is None:
            raise ValueError("Dialogue requires scope")
        async with self.locks[job.scope.scope_id]:
            conversation = await self.conversations.get(job.scope.scope_id)
            if conversation.scope != job.scope:
                raise StaleStateError("Dialogue epoch changed")
            turn = await self.conversations.claim_turn(job.scope, self.clock.now())
            if turn is None:
                return
            if not self.commerce.can_reply(conversation):

                async def skip(tx: Transaction) -> None:
                    await self.conversations.finish_turn(tx, turn, "skipped")

                await self.repository.database.write(skip)
                return
            await self._execute(turn, conversation)

    async def followup(
        self,
        ticket: FollowupTicket,
        on_prepared: Callable[[Transaction], Awaitable[None]],
    ) -> bool:
        async with self.locks[ticket.turn.scope.scope_id]:
            current = await self.conversations.get(ticket.turn.scope.scope_id)
            if current.scope != ticket.turn.scope or not self.commerce.can_reply(
                current
            ):
                return False
            if (
                current.input_revision != ticket.turn.input_revision
                or current.control_revision != ticket.control_revision
                or self.clock.now() >= ticket.expires_at
            ):
                return False
            return await self._execute(ticket.turn, current, ticket, on_prepared)

    async def _execute(
        self,
        turn: Turn,
        conversation: Conversation,
        followup: FollowupTicket | None = None,
        on_prepared: Callable[[Transaction], Awaitable[None]] | None = None,
    ) -> bool:
        identifier = await self.repository.start(turn, self.clock.now())
        started = asyncio.get_running_loop().time()
        received = min(
            (message.received_at for message in turn.messages), default=self.clock.now()
        )
        queued = max(0, self.clock.now() - received)
        limit = min(
            self.settings.turn_deadline_seconds,
            self.settings.response_max_age_seconds - queued,
        )
        first_deadline = None
        if conversation.first_reply_at is None and followup is None:
            first_deadline = received + self.settings.first_deadline_seconds
        if followup is not None:
            limit = min(limit, followup.expires_at - self.clock.now())
        run = DialogueRun(
            turn,
            conversation,
            identifier,
            started + max(0, limit),
            started,
            queued,
            received,
            task=asyncio.current_task(),
            first_reply_target_at=first_deadline,
            followup=followup,
            on_prepared=on_prepared,
        )
        self.active[identifier] = run
        status, error_code = "prepared", None
        try:
            if limit <= 0:
                raise TimeoutError("Response age exceeded")
            async with asyncio.timeout(limit):
                await self._build(run)
                await self.graph.ainvoke({"run_id": identifier})
            if run.draft is not None and not run.draft.should_send:
                status = "skipped"
        except StaleStateError:
            status, error_code = "cancelled", "STALE_STATE"
        except asyncio.CancelledError:
            status, error_code = "cancelled", "INTERRUPTED"
            raise
        except Exception as error:
            status, error_code = "fallback", failure_code(error)
            if followup is None:
                await self._notice(
                    run,
                    (
                        "overflow"
                        if isinstance(error, ContextOverflowError)
                        else "failure"
                    ),
                )
            else:
                status = "skipped"
        finally:
            self.active.pop(identifier, None)
            await self.repository.finish(
                identifier,
                status,
                self.clock.now(),
                None,
                {
                    "engine_version": "v2",
                    "persona_mode": conversation.persona_mode,
                    "policy_version": POLICY_VERSION,
                    "llm_calls": len(run.events),
                    "llm_events": run.events,
                    "research_observations": run.observations,
                    "memory_errors": run.memory_errors,
                    "full_history_message_ids": (
                        [item.id for item in run.request.original_messages]
                        if run.request
                        else []
                    ),
                    "queued_seconds": queued,
                    "elapsed_seconds": asyncio.get_running_loop().time() - started,
                    "generation_seconds": sum(
                        event.get("seconds", 0) for event in run.events
                    ),
                    "origin": "followup" if followup else "dialogue",
                },
                error_code,
            )
        return status == "prepared"

    async def _current(self, run: DialogueRun) -> Conversation:
        current = await self.conversations.get(run.turn.scope.scope_id)
        async with self.repository.database.read() as tx:
            killed = await self.delivery.is_killed(tx)
        if (
            current.scope != run.turn.scope
            or current.input_revision != run.turn.input_revision
            or current.control_revision != run.conversation.control_revision
            or killed
            or not self.commerce.can_reply(current)
        ):
            raise StaleStateError("Dialogue state changed")
        if run.followup is not None and self.clock.now() >= run.followup.expires_at:
            raise StaleStateError("Follow-up expired")
        return current

    async def _build(self, run: DialogueRun) -> None:
        current = await self._current(run)
        history = await self.conversations.messages(current.scope)
        items = await self.memory.repository.items(current.scope)
        run.request = build_request(
            current,
            history,
            run.turn.messages,
            mode="research_human",
            followup=run.followup is not None,
            boundaries=[
                item
                for item in items
                if item["kind"] == "boundary" and item["validity_status"] == "current"
            ],
            transition=await self.commerce.transition_state(current),
            now=self.clock.now(),
        )
        if run.followup is not None:
            last_user_sequence = max(
                (message.sequence for message in history if message.role == "user"),
                default=0,
            )
            run.request.state["followup_opportunity"] = {
                "ordinal": run.followup.ordinal,
                "required": run.followup.requires_response,
                "maximum": self.settings.followup_max_count,
                "silence_started_at": run.followup.silence_started_at,
                "previous_messages": [
                    {"message_id": message.id}
                    for message in history
                    if message.sequence > last_user_sequence
                    and message.role == "assistant"
                    and message.metadata.get("origin") == "followup"
                    and message.metadata.get("confirmed_delivery")
                ],
            }
        run.request.state["facts"] = [
            item
            for item in items
            if item["kind"] != "boundary" and item["validity_status"] == "current"
        ]
        rules = rules_for_mode(
            run.request.mode,
            followup_ordinal=run.followup.ordinal if run.followup else 0,
        )
        if (
            self.builder.counter.count(
                {
                    "rules": rules,
                    "data": run.request.as_data(),
                    "schema": response_schema(run.request).model_json_schema(),
                }
            )
            + 1500
            > self.builder.settings.context_tokens
        ):
            raise ContextOverflowError(
                "Full dialogue exceeds configured context budget"
            )

    async def _generate(self, state: DialogueState) -> dict[str, str]:
        run = self.active[state["run_id"]]
        assert run.request is not None
        errors: list[str] = []
        for attempt in range(self.settings.dialogue_max_calls):
            await self._current(run)
            remaining = run.deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("Dialogue deadline")
            metrics: dict[str, Any] = {
                "task": "dialogue_repair" if attempt else "dialogue",
                "attempt": attempt + 1,
            }
            token = current_llm_metrics.set(metrics)
            started = asyncio.get_running_loop().time()
            draft = None
            try:
                data = run.request.as_data()
                if errors:
                    data["REPAIR_ERRORS"] = errors
                errors = []
                recovery_window = remaining - min(1, remaining * 0.1)
                attempt_seconds = (
                    recovery_window
                    if attempt
                    else min(recovery_window, self.settings.llm_attempt_seconds)
                )
                if attempt == 0 and run.first_reply_target_at is not None:
                    target_remaining = (
                        self.settings.first_deadline_seconds
                        - run.queued_seconds
                        - (asyncio.get_running_loop().time() - run.started)
                    )
                    if target_remaining > 0:
                        attempt_seconds = min(
                            attempt_seconds,
                            target_remaining - min(1, target_remaining * 0.1),
                        )
                metrics["budget_seconds"] = attempt_seconds
                draft = await self.model.complete(
                    metrics["task"],
                    rules_for_mode(
                        run.request.mode,
                        followup_ordinal=run.followup.ordinal if run.followup else 0,
                    ),
                    data,
                    response_schema(run.request),
                    attempt_seconds,
                )
                metrics["grounding_errors"] = grounding_errors(draft, run.request)
                errors = validate_draft(draft, run.request)
                if not errors:
                    run.draft = draft
                    return {}
                raise ReplyRejectedError(",".join(errors))
            except (StaleStateError, asyncio.CancelledError):
                raise
            except Exception as error:
                metrics["error_code"] = failure_code(error)
                errors = errors or structure_errors(error)
                if (
                    attempt + 1 >= self.settings.dialogue_max_calls
                    or not can_retry_response(error)
                ):
                    raise
                delay = retry_delay(error) or 0
                if delay >= run.deadline - asyncio.get_running_loop().time():
                    raise
                if delay:
                    await asyncio.sleep(delay)
            finally:
                metrics["seconds"] = asyncio.get_running_loop().time() - started
                metrics["validation_errors"] = list(errors)
                run.events.append(metrics)
                raw = metrics.get("raw_response")
                observation = observe_response(
                    run.request,
                    draft,
                    raw_response=raw if isinstance(raw, str) else None,
                    validation_errors=errors,
                )
                run.observations.append(observation.model_dump(mode="json"))
                current_llm_metrics.reset(token)
        raise ReplyRejectedError("Dialogue attempts exhausted")

    async def _project(self, run: DialogueRun) -> None:
        assert run.request is not None and run.draft is not None
        operations: list[MemoryOperation] = []
        if run.followup is None:
            sources = [
                SourceReference(
                    message_id=message.id,
                    message_version=message.version,
                    evidence_text=message.text,
                )
                for message in run.turn.messages
                if message.role == "user" and not message.metadata.get("forwarded")
            ]
            flags = {
                "question_permission": (
                    False
                    if run.draft.no_questions
                    else True if run.draft.allow_questions else None
                ),
                "followup_permission": (
                    False
                    if run.draft.no_proactive
                    else True if run.draft.allow_proactive else None
                ),
                "offer_refusal": True if run.draft.transition_blocked else None,
            }
            for predicate, value in flags.items():
                existing = next(
                    (
                        item
                        for item in run.request.boundaries
                        if item.get("subject_id") == "user_self"
                        and item.get("predicate") == predicate
                    ),
                    None,
                )
                if (
                    value is None
                    or not sources
                    or (existing is not None and existing.get("value") == value)
                ):
                    continue
                if (
                    existing is not None
                    or value is False
                    or predicate == "offer_refusal"
                ):
                    operations.append(
                        MemoryOperation(
                            operation="supersede" if existing is not None else "add",
                            target_memory_id=(
                                existing["id"] if existing is not None else None
                            ),
                            kind="boundary",
                            predicate=predicate,
                            value=value,
                            sources=sources,
                        )
                    )
        for raw_candidate in run.draft.memory_candidates:
            try:
                candidate = MemoryCandidate.model_validate(raw_candidate)
            except ValidationError as error:
                run.memory_errors.extend(structure_errors(error))
                continue
            errors = memory_candidate_errors(candidate, run.request)
            if errors:
                run.memory_errors.extend(errors)
                continue
            operations.append(
                MemoryOperation(
                    **candidate.model_dump(
                        exclude={"message_id", "message_version", "evidence_text"}
                    ),
                    sources=[
                        SourceReference(
                            message_id=candidate.message_id,
                            message_version=candidate.message_version,
                            evidence_text=candidate.evidence_text,
                        )
                    ],
                )
            )
        for operation in operations:
            try:
                current = await self._current(run)
                await self.memory.resolve(
                    current.snapshot,
                    run.turn.messages,
                    [operation],
                )
            except StaleStateError:
                raise
            except Exception as error:
                run.memory_errors.append(type(error).__name__)

    async def _prepare(self, state: DialogueState) -> dict[str, str]:
        run = self.active[state["run_id"]]
        assert run.draft is not None and run.request is not None
        if not run.draft.should_send:
            return {}
        await self._project(run)
        current = await self._current(run)
        draft = run.draft
        transition = await self.commerce.transition_state(current)
        invitation = bool(
            transition["eligible"]
            and (draft.suggest_transition or transition["required"])
            and not draft.transition_blocked
            and not draft.pause_requested
            and not draft.closing
            and not draft.no_proactive
            and run.followup is None
        )
        final = (
            run.followup is None
            and current.preview_reply_count + 1
            >= self.commerce.settings.preview_max_replies
        )

        rendered_parts = [CHAT_RENDERING.render(text) for text in draft.delivery_parts]
        timing = DeliveryTiming(self.settings.reply_timing)
        durations = [timing.duration(text) for text in rendered_parts]

        async def prepare(tx: Transaction) -> None:
            latest = await self.conversations.get(current.id, tx)
            if (
                latest.snapshot != current.snapshot
                or not self.commerce.can_reply(latest)
                or await self.delivery.is_killed(tx)
            ):
                raise StaleStateError("Dialogue changed before delivery")
            now = self.clock.now()
            elapsed = max(
                now - run.reaction_started_at,
                run.queued_seconds + asyncio.get_running_loop().time() - run.started,
            )
            predecessor = None
            expected_at = now
            for index, (rendered, duration) in enumerate(
                zip(rendered_parts, durations, strict=True)
            ):
                first = index == 0 and run.first_reply_target_at is not None
                if first:
                    duration = min(
                        duration, max(0, self.settings.first_deadline_seconds - 1)
                    )
                delay = max(0, duration - elapsed) if index == 0 else duration
                if run.followup is not None:
                    delay = min(delay, max(0, run.followup.expires_at - now - 1))
                expected_at += delay
                deadline = expected_at + 30
                if run.followup is not None:
                    deadline = min(deadline, run.followup.expires_at)
                predecessor = await self.delivery.prepare(
                    tx,
                    latest.snapshot,
                    OutboxDraft(
                        action_key=f"v2:{run.turn.id}:{index}",
                        kind="reply",
                        text=rendered,
                        deadline=deadline,
                        predecessor_id=predecessor,
                        delay_seconds=delay,
                    ),
                    now,
                    {
                        "timing": {
                            "characters": len(rendered),
                            "target_seconds": duration,
                            "delay_seconds": delay,
                            "elapsed_seconds": elapsed if index == 0 else 0,
                        },
                        "origin": "followup" if run.followup else "dialogue_v2",
                        "reply": rendered,
                        "rendering": CHAT_RENDERING.model_dump(),
                        "response_language": draft.language,
                        "conversation_action": draft.conversation_action,
                        "awaiting_user": draft.awaiting_user,
                        "pause_requested": draft.pause_requested,
                        "followup_allowed": not (
                            draft.closing
                            or draft.no_proactive
                            or draft.pause_requested
                            or final
                            or invitation
                        ),
                        "hard_deadline": False,
                        "first_reply_target_at": (
                            run.first_reply_target_at if first else None
                        ),
                        "followup_ordinal": run.followup.ordinal if run.followup else 0,
                        "silence_started_at": (
                            run.followup.silence_started_at if run.followup else None
                        ),
                        "policy_version": POLICY_VERSION,
                        "research_run_id": run.identifier,
                    },
                    run.turn.id,
                )
            if invitation or final:
                text = service_notice(
                    draft.language, "invitation" if invitation else "closing"
                )
                if invitation:
                    text += (
                        "\n"
                        + self.commerce.transition_text(draft.language).splitlines()[-1]
                    )
                await self.delivery.prepare(
                    tx,
                    latest.snapshot,
                    OutboxDraft(
                        action_key=f"v2-notice:{run.turn.id}",
                        kind="service",
                        text=text,
                        deadline=expected_at + 30,
                        predecessor_id=predecessor,
                    ),
                    now,
                    {
                        "origin": (
                            "platform_redirect" if invitation else "preview_closing"
                        ),
                        "platform_transition": invitation,
                        "reply": text,
                        "response_language": draft.language,
                        "conversation_action": draft.conversation_action,
                        "awaiting_user": draft.awaiting_user,
                        "pause_requested": draft.pause_requested,
                        "followup_allowed": False,
                    },
                    run.turn.id,
                )
            await tx.execute(
                "UPDATE conversations SET language=?,dialogue_mode=? WHERE id=?",
                (
                    draft.language,
                    "closing" if draft.closing or final else "continuation",
                    latest.id,
                ),
            )
            await self.conversations.finish_turn(tx, run.turn, "prepared")
            if run.on_prepared is not None:
                await run.on_prepared(tx)
            if asyncio.get_running_loop().time() >= run.deadline:
                raise TimeoutError("Dialogue deadline expired during outbox commit")

        await self.repository.database.write(prepare)
        return {}

    async def _notice(self, run: DialogueRun, kind: str) -> None:
        try:
            current = await self._current(run)
        except StaleStateError:
            return

        language = run.request.language if run.request else current.language

        async def prepare(tx: Transaction) -> None:
            latest = await self.conversations.get(current.id, tx)
            if latest.snapshot != current.snapshot or await self.delivery.is_killed(tx):
                return
            existing = await fetchone(
                tx,
                "SELECT id FROM outbox_messages WHERE turn_id=? AND scope_id=? AND epoch=? AND status IN ('ready','sending','delivered','delivery_unknown') LIMIT 1",
                (run.turn.id, current.id, current.epoch),
            )
            if existing is not None:
                return
            await self.delivery.prepare(
                tx,
                latest.snapshot,
                OutboxDraft(
                    action_key=f"v2-failure:{run.turn.id}",
                    kind="service",
                    text=service_notice(language, kind),
                    deadline=self.clock.now() + 30,
                ),
                self.clock.now(),
                {
                    "origin": "response_recovery",
                    "reply": service_notice(language, kind),
                    "followup_allowed": False,
                },
                run.turn.id,
            )
            await self.conversations.finish_turn(tx, run.turn, "prepared")

        await self.repository.database.write(prepare)
