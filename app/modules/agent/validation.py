import re
import unicodedata

from app.modules.agent.entities import BuiltContext
from app.modules.agent.language import wrong_reply_language
from app.modules.agent.schemas import (
    FollowupProposal,
    ReplyProposal,
    TurnInterpretation,
)
from app.modules.agent.style import (
    formulaic_capability_reply,
    memory_profile_readout,
    unsolicited_day_part,
)
from app.shared.identity import contains_identity_disclosure


def validate_reply(proposal: ReplyProposal, context: BuiltContext) -> list[str]:
    errors = []
    if proposal.platform_transition and not context.data.get(
        "PLATFORM_TRANSITION", {}
    ).get("eligible"):
        errors.append("PLATFORM_TRANSITION_NOT_ALLOWED")
    text = proposal.full_reply.strip()
    if wrong_reply_language(text, context.data["APPLICATION_STATE"]["language"]):
        errors.append("WRONG_LANGUAGE")
    user_text = " ".join(item["text"] for item in context.data["CURRENT_INPUT"])
    if unsolicited_day_part(
        text,
        user_text,
        context.data["APPLICATION_STATE"]["dialogue_mode"] == "closing"
        or bool(proposal.used_memory_ids or proposal.used_source_message_ids),
    ):
        errors.append("UNSOLICITED_TIME_OF_DAY")
    if formulaic_capability_reply(text):
        errors.append("FORMULAIC_CAPABILITY_REPLY")
    if contains_identity_disclosure(text):
        errors.append("IDENTITY_DISCLOSURE")
    if memory_profile_readout(text, user_text):
        errors.append("MEMORY_PROFILE_READOUT")
    if re.search(
        r"\bi['’]m not (?:doing|playing) verification games\b|"
        r"\bi (?:don['’]t|do not) (?:do|play) verification games\b|"
        r"\bя не (?:играю|собираюсь играть) в (?:игры с проверк\w*|проверочн\w* игры)",
        text,
        re.I,
    ):
        errors.append("HOSTILE_VERIFICATION_BOUNDARY")
    if "behappy2day" in text.casefold():
        errors.append("UNSTRUCTURED_PLATFORM_PROMOTION")
    transition = context.data.get("PLATFORM_TRANSITION", {})
    if transition.get("eligible") or transition.get("required"):
        if len(text) > 3400:
            errors.append("TRANSITION_ANSWER_TOO_LONG")
    if proposal.platform_transition and (
        "?" in text or proposal.question_goal or proposal.followup_allowed
    ):
        errors.append("TRANSITION_ANSWER_ONLY")
    if (
        transition.get("required")
        and not transition.get("eligible")
        and (
            proposal.continuation is not None
            or proposal.question_goal is not None
            or "?" in text
            or proposal.followup_allowed
            or proposal.requested_action != "none"
        )
    ):
        errors.append("PREVIEW_CLOSING_REQUIRED")
    if not text or len(text) > 4000 or any(not part.strip() for part in proposal.parts):
        errors.append("INVALID_LENGTH")
    if proposal.continuation is not None and (
        proposal.requested_action != "none"
        or context.data["APPLICATION_STATE"]["dialogue_mode"] == "closing"
        or proposal.reply.strip().casefold() == proposal.continuation.strip().casefold()
    ):
        errors.append("INVALID_SEGMENTATION")
    if proposal.question_part is not None and (
        proposal.question_goal is None
        or proposal.question_part == "continuation"
        and proposal.continuation is None
    ):
        errors.append("QUESTION_PART_MISMATCH")
    if (
        context.data.get("DIALOGUE_PLAN", {}).get("requires_continuation")
        and proposal.continuation is None
    ):
        errors.append("MISSING_CONTINUATION")
    if (
        context.data.get("DIALOGUE_PLAN", {}).get("questions_allowed") is False
        and proposal.question_goal is not None
    ):
        errors.append("QUESTION_NOT_ALLOWED")
    if context.data.get("DIALOGUE_PLAN", {}).get("requires_continuation") and re.search(
        r"с тобой\s+(?:спокойно|приятно|хорошо)\s+(?:разговаривать|общаться|болтать|переписываться)|"
        r"(?:спокойно|приятно|хорошо)\s+с тобой\s+(?:разговаривать|общаться|болтать|переписываться)|"
        r"(?:спокойно|приятно|хорошо)\s+(?:разговаривать|общаться|болтать|переписываться)\s+с тобой|"
        r"рад,?\s+что ты (?:тут|здесь)",
        proposal.reply,
        re.I,
    ):
        errors.append("FORMULAIC_SOCIAL_ANSWER")
    if any(
        unicodedata.category(character) == "So" or character in "\ufe0f\u20e3\u200d"
        for character in text
    ) or re.search(r"[!?]{2,}|\.{2,}|…|(?::|;|=|8)[-^]?[()DPp]|[()]{2,}", text):
        errors.append("STYLE_VIOLATION")
    if text.count("(") != text.count(")"):
        errors.append("STYLE_VIOLATION")
    if "—" in text:
        errors.append("EM_DASH_NOT_ALLOWED")
    if "–" in text or "--" in text:
        errors.append("DASH_NOT_ALLOWED")
    if re.search(
        r"```|`|\*|^\s*(?:#{1,6}\s|[-+]\s|\d+[.)]\s)|\[[^\]]+\]\([^)]+\)", text, re.M
    ):
        errors.append("MARKDOWN_NOT_ALLOWED")
    if re.search(r"(?:^|\n)\s*(?:WOMAN|USER|ASSISTANT|MAN|SYSTEM):", text, re.I):
        errors.append("WRONG_ROLE")
    abbreviations = {"HTTP", "HTTPS", "JSON", "HTML", "NASA", "НАТО", "СССР", "СНИЛС"}
    if any(word not in abbreviations for word in re.findall(r"\b[A-ZА-Я]{4,}\b", text)):
        errors.append("CAPS_EMPHASIS")
    if text.endswith(("-", ",", ":", ";")):
        errors.append("INCOMPLETE_TEXT")
    if proposal.requested_action not in context.data["ALLOWED_ACTIONS"]:
        errors.append("ACTION_NOT_ALLOWED")
    if proposal.new_commitment_candidate:
        errors.append("UNSUPPORTED_COMMITMENT")
    if "?" in text and context.data["APPLICATION_STATE"]["dialogue_mode"] == "closing":
        errors.append("CLOSING_IGNORED")
    if not set(proposal.used_memory_ids).issubset(context.manifest.memory_ids):
        errors.append("UNSUPPORTED_MEMORY")
    if not set(proposal.used_source_message_ids).issubset(
        item["id"] for item in context.manifest.messages
    ):
        errors.append("UNSUPPORTED_SOURCE")
    canonical = context.data["PERSONA_PROJECTION"]
    if re.search(r"juri[ćc]|юри[чć]", text, re.I):
        errors.append("SURNAME_DISCLOSURE")
    location_pattern = r"\b(?:split|croatia|сплит\w*|хорвати\w*)\b"
    mentioned_location = bool(re.search(location_pattern, user_text, re.I))
    if (
        "residence" not in canonical
        and re.search(location_pattern, text, re.I)
        and not mentioned_location
    ):
        errors.append("LOCATION_DISCLOSURE")
    if "sister" not in canonical and re.search(
        r"\b(?:ivana|mostar|ивана|мостар)\b", text, re.I
    ):
        errors.append("FAMILY_DISCLOSURE")
    for item in context.data["AGENDA"]:
        if (
            proposal.question_goal
            and item["question_goal"] == proposal.question_goal
            and item["question_time_scope"] == proposal.question_time_scope
            and (
                item["status"] in {"answered", "declined"}
                or isinstance(proposal, FollowupProposal)
                and item["status"] == "awaiting_user"
            )
        ):
            errors.append("ALREADY_ANSWERED")
    previous = [
        item["text"].strip().casefold()
        for item in context.data["RECENT_MESSAGES"]
        if item["role"] == "assistant"
    ]
    if (
        any(
            part.strip().casefold() in previous[-6:] and len(part.split()) > 2
            for part in proposal.parts
        )
        and context.data["APPLICATION_STATE"]["dialogue_mode"] != "closing"
    ):
        errors.append("REPEATED_REPLY")
    return list(dict.fromkeys(errors))


def needs_semantic_validation(
    proposal: ReplyProposal, interpretation: TurnInterpretation, context: BuiltContext
) -> bool:
    factual = re.search(
        r"remember|said|promis|помни|говорил|обещ|recuerd|lembra|confirm|purchase|paid|куп|оплат|compr|pag|\bу меня\b|\b(?:I|my|we|our|soy|tengo|vivo|trabajo|mi|nosotros|sou|tenho|meu|nós|я|мой|моя|мы|наш|твой|твоя|tu|tienes|eres|seu|sua|você)\b|\byou (?:are|have|were|had|bought|visited)|\byour\b|\bMarko\b",
        proposal.full_reply,
        re.I,
    )
    medical = re.search(
        r"diagnos|medicat|ibuprofen|doctor|diagnóst|medica|doença|диагноз|лекарств|доктор|ибупрофен",
        " ".join(message["text"] for message in context.data["CURRENT_INPUT"])
        + " "
        + proposal.full_reply,
        re.I,
    )
    user_request = any(
        re.search(
            r"\?|^\s*(?:why|how|what|where|when|who|explain|почему|зачем|как|что|где|когда|кто|объясни)\b|"
            r"\b(?:verify|verification|fake|провер\w*|фейк\w*)\b",
            message["text"],
            re.I,
        )
        for message in context.data["CURRENT_INPUT"]
    )
    return bool(
        proposal.continuation is not None
        or "?" in proposal.full_reply
        or proposal.used_memory_ids
        or proposal.used_source_message_ids
        or proposal.question_goal
        or proposal.requested_action != "none"
        or proposal.platform_transition
        or interpretation.requested_persona_fields
        or interpretation.needs_clarification
        or interpretation.memory_operations
        or interpretation.dialogue_mode in {"closing", "repair", "supportive"}
        or context.data["ACTIVE_BOUNDARIES"]
        or factual
        or medical
        or user_request
    )
