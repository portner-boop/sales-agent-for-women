from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any, Literal, cast

from app.modules.conversations.entities import Conversation, Message
from app.modules.dialogue.episodes import calendar_context
from app.modules.dialogue.languages import confident_language
from app.modules.dialogue.persona import DISCLOSURES, persona_projection
from app.modules.dialogue.prompts import POLICY_VERSION
from app.modules.dialogue.schemas import (
    ChatMessage,
    DialogueMode,
    DialogueRequest,
    OriginalMessage,
)


def original_message(message: Message) -> OriginalMessage:
    return OriginalMessage(
        id=message.id,
        version=message.version,
        sequence=message.sequence,
        role=cast(Literal["user", "assistant"], message.role),
        text=message.text,
        event_time=message.event_time,
        reply_to_external_id=message.reply_to_external_id,
        forwarded=bool(message.metadata.get("forwarded")),
        delivery_kind=(
            (
                "followup"
                if message.metadata.get("origin") == "followup"
                else "ordinary_reply"
            )
            if message.role == "assistant"
            else None
        ),
    )


def build_request(
    conversation: Conversation,
    history: Sequence[Message],
    pending: Sequence[Message],
    *,
    mode: DialogueMode = "research_human",
    followup: bool = False,
    boundaries: Sequence[Mapping[str, Any]] = (),
    transition: Mapping[str, Any] | None = None,
    now: float | None = None,
) -> DialogueRequest:
    current = {message.id: message for message in history}
    current.update({message.id: message for message in pending})
    ordered = sorted(current.values(), key=lambda item: item.sequence)
    last_assistant = next(
        (
            item
            for item in reversed(ordered)
            if item.role == "assistant" and item.metadata.get("confirmed_delivery")
        ),
        None,
    )
    last_control_revision = (
        last_assistant.metadata.get("delivery_control_revision")
        if last_assistant is not None
        else None
    )
    exchange_reopened = (
        isinstance(last_control_revision, int)
        and last_control_revision < conversation.control_revision
    )
    if exchange_reopened:
        last_assistant = None
    messages = [
        original_message(message)
        for message in ordered
        if message.role == "user"
        or message.role == "assistant"
        and message.metadata.get("confirmed_delivery")
    ]
    timestamp = now if now is not None else datetime.now(UTC).timestamp()
    persona = persona_projection((), timestamp)
    incoming_language = (
        confident_language(pending[-1].text) if pending and not followup else None
    )
    return DialogueRequest(
        scope_id=conversation.id,
        epoch=conversation.epoch,
        input_revision=conversation.input_revision,
        mode=mode,
        language=incoming_language or conversation.language,
        followup=followup,
        original_messages=messages,
        pending_ids=(
            []
            if followup
            else [message.id for message in pending if message.role == "user"]
        ),
        persona=persona,
        restricted_persona=deepcopy(DISCLOSURES),
        boundaries=[dict(item) for item in boundaries],
        transition=dict(transition or {}),
        state={
            "exchange_reopened": exchange_reopened,
            "service_events": [
                {"sequence": message.sequence, "text": message.text}
                for message in history
                if message.role == "service"
                and message.metadata.get("confirmed_delivery")
            ],
            "previous_assistant_turn": [
                item.text
                for item in ordered
                if last_assistant is not None
                and item.turn_id == last_assistant.turn_id
                and item.role == "assistant"
                and item.metadata.get("confirmed_delivery")
            ],
            "control": conversation.control_state,
            "previous_language": conversation.language,
            "language_source": (
                "incoming_text" if incoming_language else "session_fallback"
            ),
            "policy_version": POLICY_VERSION,
            "completed_reply_count": conversation.preview_reply_count,
            "preview_state": {
                "completed_replies": conversation.preview_reply_count,
                "completed_pairs": conversation.completed_pair_count,
                "pending_outgoing_turn_id": conversation.pending_outgoing_turn_id,
                "invitation_state": conversation.invitation_state,
                "closed_at": conversation.preview_closed_at,
            },
            "now_utc": timestamp,
            "calendar": calendar_context(timestamp),
            "text_only": True,
            "profile_photo_available": False,
            "visual_content_available": False,
            "full_history": True,
        },
        CHAT_MESSAGES=[
            ChatMessage(role=message.role, content=message.text) for message in messages
        ],
    )
