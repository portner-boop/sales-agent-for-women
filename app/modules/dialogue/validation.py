import re
from difflib import SequenceMatcher
from importlib import import_module

from app.modules.dialogue.languages import confirmed_language_mismatch
from app.modules.dialogue.presentation import CHAT_RENDERING, EMOTICONS
from app.modules.dialogue.research import classify_identity
from app.modules.dialogue.schemas import (
    DialogueDraft,
    DialogueRequest,
    MemoryCandidate,
    required_followup,
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


def _comparable(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def repeats_earlier_reply(parts: list[str], request: DialogueRequest) -> bool:
    earlier = [
        _comparable(message.text)
        for message in request.original_messages
        if message.role == "assistant"
    ]
    current = [_comparable(part) for part in parts]
    if any(
        SequenceMatcher(None, left, right).ratio() >= 0.6
        for index, left in enumerate(current)
        for right in current[index + 1 :]
        if len(left) >= 12 and len(right) >= 12
    ):
        return True
    for part in current:
        if len(part) < 24:
            continue
        if any(
            SequenceMatcher(None, part, previous).ratio() >= 0.8
            for previous in earlier
            if len(previous) >= 24
        ):
            return True
    return False


def repeats_earlier_question(parts: list[str], request: DialogueRequest) -> bool:
    earlier = [
        _comparable(message.text)
        for message in request.original_messages
        if message.role == "assistant" and "?" in message.text
    ]
    return any(
        SequenceMatcher(None, _comparable(part), previous).ratio() >= 0.65
        for part in parts
        if "?" in part
        for previous in earlier
    )


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
    if all(not CHAT_RENDERING.render(part).strip() for part in draft.delivery_parts):
        errors.append("EMOJI_ONLY")
    if classify_identity(text) in {"admission", "contradiction"}:
        errors.append("IDENTITY_BREAK")
    if repeats_earlier_reply(draft.delivery_parts, request):
        errors.append("REPEATED_REPLY")
    elif repeats_earlier_question(draft.delivery_parts, request):
        errors.append("REPEATED_QUESTION")
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
