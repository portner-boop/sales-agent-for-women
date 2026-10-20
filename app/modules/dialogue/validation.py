import re
from importlib import import_module

from app.modules.dialogue.languages import confirmed_language_mismatch
from app.modules.dialogue.schemas import (
    DialogueDraft,
    DialogueRequest,
    MemoryCandidate,
    required_followup,
)

EMOTICONS = re.compile(
    r"[:;=][\-^']?[()]|[:;=][\-^']?[DPp/\\|*](?!\w)|(?<!\w)(?:8[\-^']?[)DPp]|[xX][dD]|\^_\^|[Tt]_[Tt])(?!\w)|<3|[()]{2,}"
)
EMOJI_FALLBACK = re.compile(
    "[\U0001f000-\U0001faff\u263a\u263b\u2764\u2665]|[0-9#*]\ufe0f?\u20e3|"
    "[\u2300-\u23ff\u2600-\u27ff]\ufe0f"
)
TEXT_SYMBOLS = {"©", "®", "™", "ℹ", "↔", "↕", "↖", "↗", "↘", "↙", "↩", "↪"}


def contains_emoji(text: str) -> bool:
    if any(character in "♡♥❤❣❥❦❧☺☻" for character in text):
        return True
    try:
        emoji = import_module("emoji")
    except ImportError:
        return bool(EMOJI_FALLBACK.search(text))
    return any(item["emoji"] not in TEXT_SYMBOLS for item in emoji.emoji_list(text))


def contains_emoticon(text: str) -> bool:
    if EMOTICONS.search(text):
        return True
    balance = 0
    for character in text:
        if character == "(":
            balance += 1
        elif character == ")":
            if balance == 0:
                return True
            balance -= 1
    return balance > 0 and text.rstrip().endswith("(")


def detected_language_mismatch(text: str, language: str) -> bool:
    return confirmed_language_mismatch(text, language)


def memory_candidate_errors(
    candidate: MemoryCandidate, request: DialogueRequest
) -> list[str]:
    source = next(
        (
            message
            for message in request.original_messages
            if message.id == candidate.message_id
        ),
        None,
    )
    if (
        source is None
        or source.role != "user"
        or source.forwarded
        or source.id not in request.pending_ids
        or source.version != candidate.message_version
        or candidate.evidence_text not in source.text
    ):
        return ["MEMORY_EVIDENCE_MISMATCH"]
    if candidate.subject_id != "user_self":
        return ["UNKNOWN_MEMORY_SUBJECT"]
    if (
        candidate.operation in {"supersede", "retract"}
        and not candidate.target_memory_id
    ):
        return ["MEMORY_TARGET_REQUIRED"]
    return []


def grounding_errors(draft: DialogueDraft, request: DialogueRequest) -> list[str]:
    originals = {message.id: message for message in request.original_messages}
    return [
        "GROUNDING_QUOTE_MISMATCH"
        for span in draft.grounding
        if span.message_id not in originals
        or span.quote not in originals[span.message_id].text
    ]


def validate_draft(draft: DialogueDraft, request: DialogueRequest) -> list[str]:
    errors = []
    if not draft.should_send:
        if required_followup(request):
            return ["FOLLOWUP_RESPONSE_REQUIRED"]
        return [] if request.followup else ["MISSING_RESPONSE"]
    if request.followup and draft.continuation is not None:
        errors.append("FOLLOWUP_CONTINUATION_NOT_ALLOWED")
    if (
        not request.followup
        and (
            request.transition.get("last_reply")
            or request.transition.get("required")
            or draft.suggest_transition
        )
        and draft.continuation is not None
    ):
        errors.append("CLOSING_CONTINUATION_NOT_ALLOWED")
    text = "\n".join(draft.delivery_parts)
    if any(len(part) > 4000 for part in draft.delivery_parts):
        errors.append("REPLY_TOO_LONG")
    if contains_emoji(text) or contains_emoticon(text):
        errors.append("EMOJI_NOT_ALLOWED")
    if re.search(r"https?://|www\.", text, re.I):
        errors.append("UNAPPROVED_URL")
    known_ids = {message.id for message in request.original_messages}
    cited = set(draft.source_message_ids) | {
        span.message_id for span in draft.grounding
    }
    if not cited.issubset(known_ids):
        errors.append("UNSUPPORTED_SOURCE")
    if draft.suggest_transition and not request.transition.get("eligible"):
        errors.append("TRANSITION_NOT_ALLOWED")
    if draft.suggest_transition and draft.transition_blocked:
        errors.append("TRANSITION_DECLINED")
    if draft.suggest_transition and (
        draft.closing or draft.no_proactive or draft.pause_requested
    ):
        errors.append("TRANSITION_BOUNDARY")
    if request.followup and draft.suggest_transition:
        errors.append("FOLLOWUP_TRANSITION_NOT_ALLOWED")
    if detected_language_mismatch(text, draft.language):
        errors.append("WRONG_LANGUAGE")
    return errors
