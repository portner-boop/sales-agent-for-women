import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import tiktoken

from app.modules.agent.entities import BuiltContext
from app.modules.agent.persona import persona_projection
from app.modules.agent.prompts import GENERATOR_RULES, INTERPRETER_RULES
from app.modules.agent.schemas import (
    ContextManifest,
    ContinuingReplyProposal,
    ReplyProposal,
    TurnInterpretation,
)
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
        for group in ordered_groups:
            identifiers = {message.id for message in group}
            if identifiers & protected_ids:
                continue
            omitted_ids.update(identifiers)
            remaining_tokens -= self.counter.count(
                [message.as_data() for message in group]
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
            if item["topic_kind"] in topics or item["question_goal"] in topics:
                continue
            omitted_ids.add(item["id"])
            remaining_tokens -= self.counter.count(item)
            if remaining_tokens <= self.settings.context_tokens:
                remaining_tokens = remove_omitted()
                if remaining_tokens <= self.settings.context_tokens:
                    return
        if omitted_ids:
            remove_omitted()

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
            "CURRENT_INPUT": [message.as_data() for message in turn.messages],
            "RECENT_MESSAGES": [message.as_data() for message in prior],
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
        data["REFERENCE_MESSAGES"] = [message.as_data() for message in references]

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
        reciprocal = bool(
            re.search(
                r"(?:^|[.!?]\s*|\b)(?:а ты как|а у тебя как|как ты|"
                r"how are you|how about you|and you|y tú|cómo estás|"
                r"e você|como você está)\s*[?!.]*$",
                text,
            )
        )
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
        requires_continuation = (
            (reciprocal or interpretation.continue_conversation)
            and not any(message.metadata.get("forwarded") for message in turn.messages)
            and questions_allowed
            and interpretation.dialogue_mode not in {"closing", "repair", "service"}
            and interpretation.requested_business_action == "none"
        )
        return {
            "strategy": (
                "share_opinion_without_question"
                if share_requested
                else (
                    "answer_then_connected_hook"
                    if requires_continuation
                    else "respond_in_context"
                )
            ),
            "requires_continuation": requires_continuation,
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
    ) -> BuiltContext:
        if request_reserve_tokens < 0:
            raise ValueError("Request reserve cannot be negative")
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
        facts = [
            item
            for item in view.facts
            if item["predicate"] in interpretation.retrieval_predicates
        ]
        required_fact_ids = {item["id"] for item in facts}
        for item in view.facts:
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
        for message in view.sources:
            if (
                self.counter.count(sources + [message.as_data()]) <= 1200
                or message.id in interpretation.referenced_message_ids
            ):
                sources.append(message.as_data())
        permissions = interpretation.requested_persona_fields
        time_facts, unresolved_times = resolve_current_times(
            interpretation.current_time_requests, turn.messages, self.clock.now()
        )
        if not interpretation.current_time_requests:
            time_facts = current_time_facts(
                " ".join(
                    message.text
                    for message in turn.messages
                    if not message.metadata.get("forwarded")
                ),
                self.clock.now(),
            )
        data: dict[str, Any] = {
            "PERSONA_PROJECTION": persona_projection(permissions, self.clock.now()),
            "CURRENT_TIME_FACTS": time_facts,
            "UNRESOLVED_TIME_REQUESTS": unresolved_times,
            "CHAT_CAPABILITIES": {"can_send_photos": False, "text_only": True},
            "APPLICATION_STATE": {
                "control": conversation.control_state,
                "purchase": conversation.purchase_state,
                "language": interpretation.language,
                "dialogue_mode": interpretation.dialogue_mode,
                "now_utc": self.clock.now(),
                "client_timezone": None,
            },
            "ACTIVE_BOUNDARIES": list(view.boundaries),
            "AGENDA": agenda,
            "COMMITMENTS": commitments,
            "MEMORY_DATA": {"facts": facts, "summaries": summaries, "sources": sources},
            "RECENT_MESSAGES": [message.as_data() for message in mandatory_history],
            "CURRENT_INPUT": [message.as_data() for message in turn.messages],
            "DIALOGUE_PLAN": self.dialogue_plan(turn, interpretation, view),
            "HISTORY_STATUS": {
                "omitted_uncompressed_messages": 0,
                "omitted_agenda_items": 0,
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
        self._fit_agenda(
            data,
            interpretation.retrieval_topic_kinds + interpretation.retrieval_predicates,
            total,
        )
        self._fit_history(
            data,
            mandatory_history,
            set(interpretation.referenced_message_ids),
            total,
        )
        if total() > self.settings.context_tokens:
            raise ContextOverflowError("Mandatory context exceeds the input budget")
        counts = {key: self.counter.count(value) for key, value in data.items()}
        counts["rules_and_schema"] = (
            self.counter.count(GENERATOR_RULES)
            + self.counter.count(reply_schema.model_json_schema())
            + 500
        )
        counts["request_reserve_tokens"] = request_reserve_tokens
        used_messages = list(
            {
                (message["id"], message["version"]): {
                    "id": message["id"],
                    "version": message["version"],
                }
                for message in data["CURRENT_INPUT"] + data["RECENT_MESSAGES"] + sources
            }.values()
        )
        manifest = ContextManifest(
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
