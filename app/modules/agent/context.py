import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import tiktoken

from app.modules.agent.entities import BuiltContext
from app.modules.agent.persona import persona_projection
from app.modules.agent.policy import select_examples
from app.modules.agent.prompts import CATALOG, GENERATOR_RULES, INTERPRETER_RULES
from app.modules.agent.questions import resumable_question
from app.modules.agent.schemas import (
    ContextManifest,
    ContinuingReplyProposal,
    ReplyProposal,
    TurnInterpretation,
)
from app.modules.agent.style_plan import build_style_plan
from app.modules.agent.time import current_time_facts, resolve_current_times
from app.modules.conversations.entities import Conversation, Message, Turn
from app.modules.conversations.repository import ConversationRepository
from app.modules.memory.entities import MemoryView
from app.modules.memory.repository import MemoryRepository
from app.shared.config import MemorySettings
from app.shared.entities import Clock, ContextOverflowError


class TokenCounter:
    def __init__(self, model: str = "gpt-5.2") -> None:
        os.environ.setdefault(
            "TIKTOKEN_CACHE_DIR",
            str(Path(__file__).parents[2] / "resources" / "tokenizer"),
        )
        try:
            self.encoding = tiktoken.encoding_for_model(model.removeprefix("openai/"))
        except KeyError:
            self.encoding = tiktoken.get_encoding("o200k_base")

    def count(self, value: Any) -> int:
        text = (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, default=str)
        )
        return len(self.encoding.encode(text, disallowed_special=()))


def context_message(message: Message) -> dict[str, Any]:
    return {
        **message.as_data(),
        "metadata": {
            key: value
            for key, value in message.metadata.items()
            if key
            not in {
                "style_plan",
                "casual_imperfection",
                "casual_imperfection_used",
                "followup_allowed",
                "resume_question_thread_id",
            }
        },
    }


def context_memory_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in item.items()
        if key not in {"scope_id", "epoch", "recorded_at"}
    }


def recent_messages(messages: list[Message], count: int) -> list[Message]:
    positions = {
        message.turn_id or message.id: index for index, message in enumerate(messages)
    }
    turns = sorted(positions, key=positions.__getitem__)
    allowed = set(turns[-count:])
    return [
        message for message in messages if (message.turn_id or message.id) in allowed
    ]


class ContextBuilder:
    def __init__(
        self,
        conversations: ConversationRepository,
        memory: MemoryRepository,
        settings: MemorySettings,
        counter: TokenCounter,
        clock: Clock,
        model_id: str,
    ) -> None:
        self.conversations = conversations
        self.memory = memory
        self.settings = settings
        self.counter = counter
        self.clock = clock
        self.model_id = model_id

    def _fit_history(
        self,
        data: dict[str, Any],
        messages: list[Message],
        protected_ids: set[str],
        token_count: Callable[[], int],
    ) -> None:
        recent_ids = {message.id for message in recent_messages(messages, 1)}
        remaining_tokens = token_count()
        if remaining_tokens <= self.settings.context_tokens:
            return
        groups: dict[str, list[Message]] = {}
        for message in messages:
            groups.setdefault(message.turn_id or message.id, []).append(message)
        omitted_ids: set[str] = set()

        def remove_omitted() -> int:
            data["RECENT_MESSAGES"] = [
                message
                for message in data["RECENT_MESSAGES"]
                if message["id"] not in omitted_ids
            ]
            data["HISTORY_STATUS"]["omitted_uncompressed_messages"] += len(omitted_ids)
            omitted_ids.clear()
            return token_count()

        ordered_groups = sorted(
            groups.values(),
            key=lambda group: any(message.id in recent_ids for message in group),
        )
        supplied_originals = {
            (source["id"], source["version"], source["text"])
            for source in data.get("MEMORY_DATA", {}).get("sources", [])
        }
        history_only_ids = {
            message.id
            for message in messages
            if message.id in protected_ids
            and (message.id, message.version, message.text) not in supplied_originals
        }
        for group in ordered_groups:
            identifiers = {message.id for message in group}
            if identifiers & history_only_ids:
                continue
            omitted_ids.update(identifiers)
            remaining_tokens -= self.counter.count(
                [context_message(message) for message in group]
            )
            if remaining_tokens <= self.settings.context_tokens:
                remaining_tokens = remove_omitted()
                if remaining_tokens <= self.settings.context_tokens:
                    return
        if omitted_ids:
            remove_omitted()

    def _fit_agenda(
        self, data: dict[str, Any], topics: list[str], token_count: Callable[[], int]
    ) -> None:
        remaining_tokens = token_count()
        if remaining_tokens <= self.settings.context_tokens:
            return
        omitted_ids: set[str] = set()

        def remove_omitted() -> int:
            data["AGENDA"] = [
                item for item in data["AGENDA"] if item["id"] not in omitted_ids
            ]
            data["HISTORY_STATUS"]["omitted_agenda_items"] += len(omitted_ids)
            omitted_ids.clear()
            return token_count()

        for item in reversed(data["AGENDA"][1:]):
            resume = data.get("DIALOGUE_PLAN", {}).get("resume_question")
            if (
                (resume and item["id"] == resume["thread_id"])
                or item["topic_kind"] in topics
                or item["question_goal"] in topics
            ):
                continue
            omitted_ids.add(item["id"])
            remaining_tokens -= self.counter.count(item)
            if remaining_tokens <= self.settings.context_tokens:
                remaining_tokens = remove_omitted()
                if remaining_tokens <= self.settings.context_tokens:
                    return
        if omitted_ids:
            remove_omitted()

    def _fit_delivered(
        self,
        data: dict[str, Any],
        messages: list[Message],
        protected_ids: set[str],
        token_count: Callable[[], int],
    ) -> None:
        groups: dict[str, list[Message]] = {}
        for message in messages:
            groups.setdefault(message.turn_id or message.id, []).append(message)
        for group in list(groups.values())[:-1]:
            if token_count() <= self.settings.context_tokens:
                return
            identifiers = {message.id for message in group}
            if identifiers & protected_ids:
                continue
            data["RECENT_DELIVERED"] = [
                message
                for message in data["RECENT_DELIVERED"]
                if message["id"] not in identifiers
            ]
            data["HISTORY_STATUS"]["omitted_delivered_messages"] += len(identifiers)

    async def interpreter_data(
        self, turn: Turn, conversation: Conversation
    ) -> dict[str, Any]:
        history = await self.conversations.messages(turn.scope)
        current_ids = {message.id for message in turn.messages}
        text = " ".join(message.text for message in turn.messages).casefold()
        entities = await self.memory.entities(turn.scope)
        matched = [
            entity
            for entity in entities
            if entity["name"] and entity["name"].casefold() in text
        ]
        canonical_entities = [
            entity for entity in entities if entity["id"] in {"user_self", "persona"}
        ]
        selected_entities = list(
            {
                entity["id"]: entity
                for entity in canonical_entities + matched + entities[:50]
            }.values()
        )
        selected_ids = {entity["id"] for entity in matched}
        items = await self.memory.items(turn.scope)
        boundaries = [item for item in items if item["kind"] == "boundary"]
        facts = [
            {
                key: item[key]
                for key in (
                    "id",
                    "subject_id",
                    "predicate",
                    "value",
                    "qualifiers",
                    "modality",
                    "validity_status",
                )
            }
            for item in items
            if item["kind"] != "boundary"
        ]
        facts.sort(
            key=lambda item: (
                item["predicate"] not in {"preferred_name", "preferred_language"},
                item["subject_id"] not in selected_ids,
                item["predicate"] not in text
                and str(item["value"]).casefold() not in text,
            )
        )
        prior = recent_messages(
            [message for message in history if message.id not in current_ids],
            self.settings.recent_turns,
        )
        data: dict[str, Any] = {
            "POLICY_VERSION": CATALOG.version,
            "CURRENT_INPUT": [context_message(message) for message in turn.messages],
            "RECENT_MESSAGES": [context_message(message) for message in prior],
            "MEMORY": list(boundaries),
            "ENTITIES": selected_entities,
            "AGENDA": await self.conversations.context_agenda(turn.scope, []),
            "prior_language": conversation.language,
            "now_utc": self.clock.now(),
            "HISTORY_STATUS": {
                "omitted_uncompressed_messages": 0,
                "omitted_agenda_items": 0,
                "originals_preserved": True,
            },
        }
        references = [
            original
            for message in turn.messages
            if message.reply_to_external_id is not None
            for original in history
            if original.external_id == message.reply_to_external_id
        ]
        data["REFERENCE_MESSAGES"] = [
            context_message(message) for message in references
        ]

        def token_count() -> int:
            return self.interpreter_tokens(data)

        self._fit_agenda(data, [], token_count)
        while token_count() > self.settings.context_tokens:
            optional = next(
                (
                    entity
                    for entity in reversed(data["ENTITIES"])
                    if entity["id"] not in {"user_self", "persona"} | selected_ids
                ),
                None,
            )
            if optional is None:
                break
            data["ENTITIES"].remove(optional)
        self._fit_history(
            data, prior, {message.id for message in references}, token_count
        )
        for item in facts:
            if (
                self.interpreter_tokens({**data, "MEMORY": data["MEMORY"] + [item]})
                <= self.settings.context_tokens
            ):
                data["MEMORY"].append(item)
        if self.interpreter_tokens(data) > self.settings.context_tokens:
            raise ContextOverflowError("Mandatory interpreter input exceeds the budget")
        return data

    def interpreter_tokens(self, data: dict[str, Any]) -> int:
        return (
            self.counter.count(
                {
                    "rules": INTERPRETER_RULES,
                    "data": data,
                    "schema": TurnInterpretation.model_json_schema(),
                }
            )
            + 500
        )

    def dialogue_plan(
        self, turn: Turn, interpretation: TurnInterpretation, view: MemoryView
    ) -> dict[str, Any]:
        text = " ".join(message.text for message in turn.messages).strip().casefold()
        share_requested = not any(
            message.metadata.get("forwarded") for message in turn.messages
        ) and bool(
            re.search(
                r"(?:расскажи|говори).*?(?:не только спрашивай|не спрашивай)|"
                r"(?:tell me|share).*?(?:instead of asking|rather than asking)",
                text,
            )
        )
        questions_allowed = not share_requested and not any(
            item["predicate"] == "question_permission" and item["value"] is False
            for item in view.boundaries
        )
        return {
            "strategy": (
                "share_opinion_without_question"
                if share_requested
                else "respond_in_context"
            ),
            "requires_continuation": False,
            "reciprocal_input": bool(
                re.search(
                    r"(?:^|[.!?]\s*|\b)(?:а ты как|а у тебя как|как ты|"
                    r"how are you|how about you|and you|y tú|cómo estás|"
                    r"e você|como você está)\s*[?!.]*$",
                    text,
                )
            ),
            "questions_allowed": questions_allowed,
            "max_messages": 2,
        }

    async def build(
        self,
        turn: Turn,
        interpretation: TurnInterpretation,
        conversation: Conversation,
        view: MemoryView,
        request_reserve_tokens: int = 0,
        policy_selection: dict[str, Any] | None = None,
        reference_time: float | None = None,
    ) -> BuiltContext:
        if request_reserve_tokens < 0:
            raise ValueError("Request reserve cannot be negative")
        now = self.clock.now() if reference_time is None else reference_time
        history = await self.conversations.messages(turn.scope)
        input_ids = {message.id for message in turn.messages}
        prior = [message for message in history if message.id not in input_ids]
        covered = await self.memory.covered_ids(turn.scope)
        recent = recent_messages(prior, self.settings.recent_turns)
        recent_ids = {message.id for message in recent}
        mandatory_history = [
            message
            for message in prior
            if message.id not in covered or message.id in recent_ids
        ]
        agenda = await self.conversations.context_agenda(
            turn.scope,
            interpretation.retrieval_topic_kinds + interpretation.retrieval_predicates,
        )
        commitments = await self.conversations.commitments(turn.scope)
        fact_budget = 800
        available_facts = [context_memory_item(item) for item in view.facts]
        facts = [
            item
            for item in available_facts
            if item["predicate"] in interpretation.retrieval_predicates
        ]
        required_fact_ids = {item["id"] for item in facts}
        for item in available_facts:
            if item["id"] in required_fact_ids:
                continue
            if self.counter.count(facts + [item]) > fact_budget:
                continue
            facts.append(item)
        summaries: list[dict[str, Any]] = []
        for item in view.summaries:
            if (
                self.counter.count(summaries + [item])
                <= self.settings.summary_context_tokens
            ):
                summaries.append(item)
        sources: list[dict[str, Any]] = []
        current_originals = {
            (message.id, message.version, message.text) for message in turn.messages
        }
        for message in view.sources:
            if (message.id, message.version, message.text) in current_originals:
                continue
            if (
                self.counter.count(sources + [context_message(message)]) <= 1200
                or message.id in interpretation.referenced_message_ids
            ):
                sources.append(context_message(message))
        permissions = interpretation.requested_persona_fields
        time_facts, unresolved_times = resolve_current_times(
            interpretation.current_time_requests, turn.messages, now
        )
        if not interpretation.current_time_requests:
            time_facts = current_time_facts(
                " ".join(
                    message.text
                    for message in turn.messages
                    if not message.metadata.get("forwarded")
                ),
                now,
            )
        data: dict[str, Any] = {
            "PERSONA_PROJECTION": persona_projection(permissions, now),
            "CURRENT_TIME_FACTS": time_facts,
            "UNRESOLVED_TIME_REQUESTS": unresolved_times,
            "CHAT_CAPABILITIES": {"can_send_photos": False, "text_only": True},
            "APPLICATION_STATE": {
                "control": conversation.control_state,
                "purchase": conversation.purchase_state,
                "language": interpretation.language,
                "dialogue_mode": interpretation.dialogue_mode,
                "now_utc": now,
                "client_timezone": None,
            },
            "ACTIVE_BOUNDARIES": [
                context_memory_item(item) for item in view.boundaries
            ],
            "AGENDA": agenda,
            "COMMITMENTS": commitments,
            "MEMORY_DATA": {"facts": facts, "summaries": summaries, "sources": sources},
            "RECENT_MESSAGES": [
                context_message(message) for message in mandatory_history
            ],
            "CURRENT_INPUT": [context_message(message) for message in turn.messages],
            "DIALOGUE_PLAN": self.dialogue_plan(turn, interpretation, view),
            "HISTORY_STATUS": {
                "omitted_uncompressed_messages": 0,
                "omitted_agenda_items": 0,
                "omitted_delivered_messages": 0,
                "originals_preserved": True,
            },
            "ALLOWED_ACTIONS": ["none"]
            + (
                ["show_offer"]
                if interpretation.requested_business_action == "show_offer"
                and not any(
                    item["predicate"] == "offer_refusal" and item["value"]
                    for item in view.boundaries
                )
                else []
            ),
        }

        delivered = [
            message
            for message in prior
            if message.role == "assistant"
            and message.metadata.get("confirmed_delivery")
        ][-10:]
        data["RECENT_DELIVERED"] = [context_message(message) for message in delivered]
        data["QUESTION_EXCLUSIONS"] = [
            {
                "goal": item["question_goal"] or item["topic_kind"],
                "subject": item["subject_id"],
                "time_scope": item["question_time_scope"],
            }
            for item in agenda
            if item["status"] in {"answered", "declined", "awaiting_user"}
        ][:12]
        data["RECALL_REQUIRED"] = (
            [
                {
                    "id": item["id"],
                    "subject_id": item["subject_id"],
                    "predicate": item["predicate"],
                    "value": item["value"],
                    "qualifiers": item["qualifiers"],
                }
                for item in facts
                if item["id"] in required_fact_ids
            ]
            if interpretation.recall_scope != "none"
            else []
        )
        data["APPLICATION_STATE"]["recall_scope"] = interpretation.recall_scope
        data["APPLICATION_STATE"][
            "current_day_part_confirmed"
        ] = interpretation.current_day_part_confirmed
        data["RELATIONSHIP_STAGE"] = (
            "opening"
            if len([message for message in prior if message.role == "user"]) < 3
            and not covered
            else "established"
        )
        turns: dict[str, list[Message]] = {}
        for message in delivered:
            turns.setdefault(message.turn_id or message.id, []).append(message)
        consecutive_questions = 0
        for group in reversed(list(turns.values())):
            if any(message.metadata.get("question_goal") for message in group):
                consecutive_questions += 1
            else:
                break
        plan = data["DIALOGUE_PLAN"]
        latest_user_sequence = max(
            (message.sequence for message in history if message.role == "user"),
            default=0,
        )
        latest_turn = list(turns.values())[-1] if turns else []
        awaiting_question_ids = {
            item.get("last_question_message_id")
            for item in agenda
            if item["status"] == "awaiting_user"
        }
        plan["waiting_question_message_ids"] = [
            message.id
            for message in latest_turn
            if message.sequence > latest_user_sequence
            and (
                message.metadata.get("question_goal")
                or message.id in awaiting_question_ids
            )
        ]
        plan["consecutive_question_turns"] = consecutive_questions
        exception = (
            interpretation.needs_clarification
            or interpretation.user_requested_questions
        )
        if consecutive_questions >= 2 and not exception:
            plan["questions_allowed"] = False
        plan["initiative_due"] = bool(
            turn.messages
            and (
                interpretation.dialogue_mode in {"opening", "continuation", "playful"}
                or (
                    interpretation.dialogue_mode in {"repair", "service"}
                    and set(interpretation.policy_topics).intersection(
                        {"verification", "contact"}
                    )
                )
            )
            and interpretation.requested_business_action == "none"
            and not interpretation.detail_requested
            and not any(message.metadata.get("forwarded") for message in turn.messages)
            and plan["questions_allowed"]
            and not plan["waiting_question_message_ids"]
            and (consecutive_questions < 2 or exception)
        )
        resume = None
        if (
            plan["questions_allowed"]
            and not plan["waiting_question_message_ids"]
            and interpretation.dialogue_mode != "closing"
            and interpretation.requested_business_action == "none"
            and not interpretation.detail_requested
            and interpretation.recall_scope == "none"
            and not any(message.metadata.get("forwarded") for message in turn.messages)
        ):
            candidate = resumable_question(history, agenda)
            if candidate and (
                turn.messages
                and (
                    interpretation.question_interrupted
                    or interpretation.continue_conversation
                    and "verification" in interpretation.policy_topics
                    and not interpretation.memory_operations
                    and not interpretation.agenda_operations
                )
                or not turn.messages
                and latest_turn
                and latest_turn[-1].metadata.get("resume_question_thread_id")
                == candidate["thread_id"]
            ):
                resume = candidate
        if resume:
            plan["resume_question"] = resume
            data["QUESTION_EXCLUSIONS"] = [
                item
                for item in data["QUESTION_EXCLUSIONS"]
                if not (
                    item["goal"] == resume["goal"]
                    and item["subject"] == resume["subject_id"]
                    and item["time_scope"] == resume["time_scope"]
                )
            ]
        if not turn.messages:
            plan["strategy"] = (
                "wait_for_answer"
                if plan["waiting_question_message_ids"]
                else (
                    "ask_connected_question"
                    if plan["questions_allowed"]
                    else "share_opinion_without_question"
                )
            )
        elif resume:
            plan["strategy"] = "answer_then_resume_thread"
        elif plan["initiative_due"]:
            plan["strategy"] = "develop_conversation_naturally"
        plan["question_exception"] = exception
        plan["detail_requested"] = interpretation.detail_requested
        plan["casual_character_limit"] = (
            None
            if interpretation.detail_requested
            or interpretation.recall_scope == "all"
            or not set(interpretation.policy_topics).intersection({"social", "support"})
            else 350
        )
        casual = (
            interpretation.recall_scope == "none"
            and not interpretation.detail_requested
            and not interpretation.requested_persona_fields
            and not time_facts
            and not interpretation.needs_clarification
            and interpretation.dialogue_mode not in {"closing", "clarification"}
            and not set(interpretation.policy_topics).intersection(
                {"photo", "commerce"}
            )
        )
        data["STYLE_PLAN"] = build_style_plan(
            prior, casual, max_parts=plan["max_messages"] if turn.messages else 1
        )

        data["POLICY_SELECTION"] = policy_selection or {}
        data["STYLE_EXAMPLES"] = select_examples(
            interpretation.policy_topics + [interpretation.dialogue_mode],
            data["RELATIONSHIP_STAGE"],
            self.counter.count,
        )
        reply_schema = (
            ContinuingReplyProposal
            if data["DIALOGUE_PLAN"]["requires_continuation"]
            else ReplyProposal
        )

        def total() -> int:
            return (
                self.counter.count(
                    {
                        "rules": GENERATOR_RULES,
                        "data": data,
                        "schema": reply_schema.model_json_schema(),
                    }
                )
                + 500
                + request_reserve_tokens
            )

        if total() > self.settings.context_tokens:
            data["STYLE_EXAMPLES"] = []
        while total() > self.settings.context_tokens and summaries:
            summaries.pop()
        for item in list(reversed(facts)):
            if total() <= self.settings.context_tokens:
                break
            if item["id"] not in required_fact_ids:
                facts.remove(item)
        for source in list(reversed(sources)):
            if total() <= self.settings.context_tokens:
                break
            if source["id"] not in interpretation.referenced_message_ids:
                sources.remove(source)
        self._fit_history(
            data,
            mandatory_history,
            set(interpretation.referenced_message_ids),
            total,
        )
        self._fit_agenda(
            data,
            interpretation.retrieval_topic_kinds + interpretation.retrieval_predicates,
            total,
        )
        self._fit_delivered(
            data, delivered, set(interpretation.referenced_message_ids), total
        )
        if total() > self.settings.context_tokens:
            sections = {key: self.counter.count(value) for key, value in data.items()}
            raise ContextOverflowError(
                "Mandatory context exceeds the input budget: "
                f"total={total()} limit={self.settings.context_tokens} "
                f"reserve={request_reserve_tokens} sections={json.dumps(sections)}"
            )
        counts = {key: self.counter.count(value) for key, value in data.items()}
        counts["rules_and_schema"] = (
            self.counter.count(GENERATOR_RULES)
            + self.counter.count(reply_schema.model_json_schema())
            + 500
        )
        counts["request_reserve_tokens"] = request_reserve_tokens
        evidence_messages = [
            {"id": source["message_id"], "version": source["message_version"]}
            for item in facts + list(view.boundaries)
            for source in item["sources"]
        ]
        used_messages = list(
            {
                (message["id"], message["version"]): {
                    "id": message["id"],
                    "version": message["version"],
                }
                for message in data["CURRENT_INPUT"]
                + data["RECENT_MESSAGES"]
                + sources
                + evidence_messages
            }.values()
        )
        manifest = ContextManifest(
            policy_version=(policy_selection or {}).get("version", "unversioned"),
            scope_id=turn.scope.scope_id,
            epoch=turn.scope.epoch,
            input_revision=conversation.input_revision,
            control_revision=conversation.control_revision,
            memory_revision=conversation.memory_revision,
            model_id=self.model_id,
            messages=used_messages,
            memory_ids=[item["id"] for item in facts + list(view.boundaries)],
            summary_ids=[item["id"] for item in summaries],
            token_count_by_section=counts,
        )
        return BuiltContext(data, manifest)
