import json
import random
import re
from collections.abc import Sequence
from functools import cache
from hashlib import sha256
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.modules.agent.policy import POLICY_DIRECTORY
from app.modules.agent.schemas import ChatImperfection, ReplyProposal
from app.modules.conversations.entities import Message


class StylePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    imperfection_interval: int = Field(ge=1)
    interval_jitter: int = Field(default=1, ge=0, le=1)
    letter_typo_keep_probability: float = Field(default=1, ge=0, le=1)
    apostrophe_forms: tuple[str, ...] = ()
    visible_forms: dict[str, str] = Field(default_factory=dict)


@cache
def style_policy() -> StylePolicy:
    return StylePolicy.model_validate(
        json.loads((POLICY_DIRECTORY / "style.json").read_text(encoding="utf-8"))
    )


def visible_edit_count(edit: ChatImperfection) -> int:
    standard = re.findall(r"\w+(?:['’]\w+)*|\W+", edit.standard)
    written = re.findall(r"\w+(?:['’]\w+)*|\W+", edit.written)
    if len(standard) != len(written):
        return 0
    count = 0
    for before, after in zip(standard, written, strict=True):
        if before == after:
            continue
        omission = (
            before.casefold().replace("’", "'") in style_policy().apostrophe_forms
            and before.replace("'", "").replace("’", "") == after
        )
        if style_policy().visible_forms.get(before) != after and not omission:
            return 0
        count += 1
    return count if 1 <= count <= 2 else 0


def supported_imperfection(edit: ChatImperfection | None, text: str) -> bool:
    if edit is None or edit.written not in text:
        return False
    if style_policy().visible_forms.get(
        edit.standard
    ) == edit.written or visible_edit_count(edit):
        return any(
            editable_span(text, match.start(), match.end(), ())
            for match in re.finditer(
                r"(?<!\w)" + re.escape(edit.written) + r"(?!\w)", text
            )
        )
    for match in re.finditer(r"(?<=[^\W\d_])[,،] (?=[^\W\d_])", edit.standard):
        index = match.end() - 1
        if edit.written == edit.standard[:index] + edit.standard[index + 1 :]:
            return True
    for index, character in enumerate(edit.standard):
        if not 0 < index < len(edit.standard) - 1:
            continue
        before, after = edit.standard[:index], edit.standard[index + 1 :]
        if (
            character in {",", "،"}
            and before[-1].isalpha()
            and after[0].isspace()
            and edit.written == before + after
        ):
            return True
        if (
            character in {"'", "’"}
            and before[-1].isalpha()
            and after[0].isalpha()
            and edit.written == before + " " + after
        ):
            return True
    return False


def semantic_reply(
    proposal: ReplyProposal, protected_texts: Sequence[str] = ()
) -> ReplyProposal:
    edit = proposal.casual_imperfection
    if (
        not proposal.casual_imperfection_used
        or edit is None
        or not supported_imperfection(edit, proposal.full_reply)
        or proposal.full_reply.count(edit.written) != 1
    ):
        return proposal
    start = proposal.full_reply.index(edit.written)
    if not editable_span(
        proposal.full_reply, start, start + len(edit.written), protected_texts
    ) or not editable_span(edit.standard, 0, len(edit.standard), protected_texts):
        return proposal
    for field in ("reply", "continuation"):
        text = getattr(proposal, field)
        if text is not None and edit.written in text:
            return proposal.model_copy(
                update={
                    field: text.replace(edit.written, edit.standard, 1),
                    "casual_imperfection_used": False,
                    "casual_imperfection": None,
                }
            )
    return proposal


def editable_span(
    text: str, start: int, end: int, protected_texts: Sequence[str]
) -> bool:
    patterns = [
        r'"[^"\n]*"|“[^”\n]*”|«[^»\n]*»|`[^`\n]*`',
        r"(?<!\w)'[^'\n]+'(?!\w)",
        r"https?://\S+|www\.\S+|\S+@\S+",
    ]
    patterns.extend(
        r"(?<!\w)" + re.escape(value) + r"(?!\w)" for value in protected_texts if value
    )
    return not any(
        start < match.end() and end > match.start()
        for pattern in patterns
        for match in re.finditer(pattern, text, re.I)
    )


def reduce_letter_imperfections(proposal: ReplyProposal) -> ReplyProposal:
    edit = proposal.casual_imperfection
    probability = style_policy().letter_typo_keep_probability
    if edit is None or probability == 1:
        return proposal
    restored = semantic_reply(proposal)
    if restored is proposal:
        return proposal
    standard = re.findall(r"\w+(?:['’]\w+)*|\W+", edit.standard)
    written = re.findall(r"\w+(?:['’]\w+)*|\W+", edit.written)
    if len(standard) != len(written):
        return proposal
    for index, (before, after) in enumerate(zip(standard, written, strict=True)):
        if (
            before != after
            and style_policy().visible_forms.get(before) == after
            and random.random() >= probability
        ):
            written[index] = before
    reduced = "".join(written)
    if reduced == edit.written:
        return proposal
    if reduced == edit.standard:
        return restored
    field = "reply" if edit.written in proposal.reply else "continuation"
    text = getattr(proposal, field)
    return proposal.model_copy(
        update={
            field: text.replace(edit.written, reduced, 1),
            "casual_imperfection": ChatImperfection(
                standard=edit.standard, written=reduced
            ),
        }
    )


def apply_casual_imperfection(
    proposal: ReplyProposal, protected_texts: Sequence[str] = ()
) -> ReplyProposal:
    original = proposal
    if proposal.casual_imperfection_used:
        edit = proposal.casual_imperfection
        if (
            edit is None
            or style_policy().visible_forms.get(edit.standard) == edit.written
            or visible_edit_count(edit)
            or not supported_imperfection(edit, proposal.full_reply)
            or proposal.full_reply.count(edit.written) != 1
        ):
            return proposal
        for field in ("reply", "continuation"):
            text = getattr(proposal, field)
            if text is not None and edit.written in text:
                start = text.index(edit.written)
                if not editable_span(
                    text, start, start + len(edit.written), protected_texts
                ):
                    return original
                proposal = proposal.model_copy(
                    update={
                        field: text.replace(edit.written, edit.standard, 1),
                        "casual_imperfection_used": False,
                        "casual_imperfection": None,
                    }
                )
                break
    candidates: list[tuple[str, int, int, str, str]] = []
    apostrophe_candidates: list[tuple[str, int, int, str, str]] = []
    punctuation_candidates: list[tuple[str, int, int, str, str]] = []
    for field in ("reply", "continuation"):
        text = getattr(proposal, field)
        if text is None:
            continue
        for standard, written in style_policy().visible_forms.items():
            for match in re.finditer(
                r"(?<!\w)" + re.escape(standard) + r"(?!\w)", text
            ):
                if not editable_span(text, match.start(), match.end(), protected_texts):
                    continue
                candidates.append(
                    (field, match.start(), match.end(), standard, written)
                )
        for match in re.finditer(r"\b[^\W\d_]+['’][^\W\d_]+\b", text):
            standard = match.group()
            if standard.casefold().replace(
                "’", "'"
            ) in style_policy().apostrophe_forms and editable_span(
                text, match.start(), match.end(), protected_texts
            ):
                apostrophe_candidates.append(
                    (
                        field,
                        match.start(),
                        match.end(),
                        standard,
                        standard.replace("'", "").replace("’", ""),
                    )
                )
        if original.casual_imperfection_used:
            continue
        for match in re.finditer(r"[^\W\d_]+[,،] [^\W\d_]+", text):
            if not editable_span(text, match.start(), match.end(), protected_texts):
                continue
            standard = match.group()
            if len(standard) <= 80:
                punctuation_candidates.append(
                    (
                        field,
                        match.start(),
                        match.end(),
                        standard,
                        standard.replace(" ", "", 1),
                    )
                )
    candidates.extend(apostrophe_candidates)
    if candidates or punctuation_candidates:
        preferred = (
            apostrophe_candidates
            if apostrophe_candidates and random.random() < 0.5
            else candidates
        )
        field, start, end, standard, written = random.choice(
            preferred or punctuation_candidates
        )
        text = getattr(proposal, field)
        if candidates and len(text) >= 25 and random.random() < 0.8:
            second_candidates = [
                candidate
                for candidate in candidates
                if candidate[0] == field
                and (candidate[2] < start or candidate[1] > end)
                and " " not in standard
                and " " not in candidate[3]
                and max(end, candidate[2]) - min(start, candidate[1]) <= 80
                and max(end, candidate[2])
                - min(start, candidate[1])
                + len(written)
                - len(standard)
                + len(candidate[4])
                - len(candidate[3])
                <= 80
                and editable_span(
                    text,
                    min(start, candidate[1]),
                    max(end, candidate[2]),
                    protected_texts,
                )
            ]
            if second_candidates:
                other = random.choice(second_candidates)
                first, second = sorted(
                    [(field, start, end, standard, written), other],
                    key=lambda candidate: candidate[1],
                )
                start, end = first[1], second[2]
                standard = text[start:end]
                written = first[4] + text[first[2] : second[1]] + second[4]
        return proposal.model_copy(
            update={
                field: text[:start] + written + text[end:],
                "casual_imperfection_used": True,
                "casual_imperfection": ChatImperfection(
                    written=written, standard=standard
                ),
            }
        )
    if original.casual_imperfection_used:
        return original
    return relax_internal_punctuation(proposal, protected_texts)


def relax_internal_punctuation(
    proposal: ReplyProposal, protected_texts: Sequence[str] = ()
) -> ReplyProposal:
    if proposal.casual_imperfection_used:
        return proposal
    for field in ("reply", "continuation"):
        text = getattr(proposal, field)
        if text is None:
            continue
        for match in re.finditer(r"(?<=[^\W\d_])[,،](?=\s+[^\W\d_])", text):
            index = match.start()
            before = re.search(r"\S+$", text[:index])
            after = re.match(r"\s+\S+", text[index + 1 :])
            assert before is not None and after is not None
            start, end = before.start(), index + 1 + after.end()
            if not editable_span(text, start, end, protected_texts):
                continue
            standard = text[start:end]
            if len(standard) > 80:
                continue
            written = text[start:index] + text[index + 1 : end]
            return proposal.model_copy(
                update={
                    field: text[:index] + text[index + 1 :],
                    "casual_imperfection_used": True,
                    "casual_imperfection": ChatImperfection(
                        written=written, standard=standard
                    ),
                }
            )
        for match in re.finditer(r"\b[^\W\d_]+['’][^\W\d_]+\b", text):
            if not editable_span(text, match.start(), match.end(), protected_texts):
                continue
            standard = match.group()
            if (
                standard.casefold().replace("’", "'")
                not in style_policy().apostrophe_forms
            ):
                continue
            written = standard.replace("'", "").replace("’", "")
            return proposal.model_copy(
                update={
                    field: text[: match.start()] + written + text[match.end() :],
                    "casual_imperfection_used": True,
                    "casual_imperfection": ChatImperfection(
                        written=written, standard=standard
                    ),
                }
            )
    return proposal


def build_style_plan(
    messages: list[Message], eligible: bool, max_parts: int = 1
) -> dict[str, Any]:
    turns: dict[str, list[Message]] = {}
    delivered: list[Message] = []
    for message in messages:
        if message.role == "assistant" and message.metadata.get("confirmed_delivery"):
            delivered.append(message)
            turns.setdefault(message.turn_id or message.id, []).append(message)
    eligible_count = 0
    used_count = 0
    for parts in turns.values():
        included = any(
            part.metadata.get("style_plan", {}).get("eligible") for part in parts
        )
        used = any(part.metadata.get("casual_imperfection_used") for part in parts)
        eligible_count += included
        used_count += included and used
    policy = style_policy()
    last_used = next(
        (
            index
            for index in range(len(delivered) - 1, -1, -1)
            if delivered[index].metadata.get("casual_imperfection_used")
        ),
        -1,
    )
    anchor = (
        delivered[last_used].id
        if last_used >= 0
        else delivered[0].id if delivered else "first"
    )
    variation = (
        int.from_bytes(sha256(anchor.encode()).digest()[:4])
        % (2 * policy.interval_jitter + 1)
        - policy.interval_jitter
    )
    interval = max(1, policy.imperfection_interval + variation)
    since_used = len(delivered) - last_used - 1
    requested = bool(
        eligible
        and delivered
        and (interval == 1 or since_used >= 1)
        and since_used + max_parts >= interval
    )
    return {
        "eligible": eligible,
        "allow_imperfection": requested,
        "request_imperfection": requested,
        "maximum": 1,
        "delivered_eligible_turns": eligible_count,
        "delivered_imperfect_turns": used_count,
        "delivered_messages": len(delivered),
        "messages_since_imperfection": since_used,
        "target_message_interval": interval,
        "visible_forms": {},
    }
