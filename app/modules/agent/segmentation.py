import re
from collections.abc import Sequence

from app.modules.agent.schemas import ReplyProposal
from app.modules.agent.style import strip_terminal_period
from app.modules.agent.style_plan import editable_span, semantic_reply

ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "г",
    "ул",
    "д",
    "стр",
    "им",
    "т.д",
    "т.е",
    "т.п",
}


def sentence_parts(text: str, protected_texts: Sequence[str] = ()) -> tuple[str, ...]:
    if len(text) < 80:
        return (text,)
    boundaries = []
    for match in re.finditer(r"(?<=[.!?。！？؟])\s+(?=[^\W\d_])", text):
        end = match.start()
        if not editable_span(text, end - 1, match.end(), protected_texts):
            continue
        if text[end - 1] == ".":
            token = re.search(r"([\w.]+)\.$", text[:end])
            if token and (
                len(token[1]) == 1
                or token[1].casefold() in ABBREVIATIONS
                or token[1].isdigit()
                or "." in token[1]
            ):
                continue
        boundaries.append((end, match.end()))
    if not boundaries:
        return (text,)
    parts = []
    start = 0
    for end, next_start in boundaries:
        parts.append(text[start:end])
        start = next_start
    parts.append(text[start:])
    return tuple(parts)


def delivery_parts(
    proposal: ReplyProposal,
    split_sentences: bool,
    protected_texts: Sequence[str] = (),
) -> tuple[ReplyProposal, ...]:
    parts = []
    for index, text in enumerate(proposal.parts):
        edit = proposal.casual_imperfection
        if edit is not None and edit.written not in text:
            edit = None
        question_goal = (
            proposal.question_goal if index == proposal.question_part_index else None
        )
        part = proposal.model_copy(
            update={
                "reply": text,
                "continuation": None,
                "question_goal": question_goal,
                "question_part": "reply" if question_goal else None,
                "casual_imperfection": edit,
                "casual_imperfection_used": edit is not None,
            }
        )
        standard = semantic_reply(part, protected_texts)
        sentences = (
            sentence_parts(standard.reply, protected_texts)
            if split_sentences
            else (standard.reply,)
        )
        question_indices = [
            position
            for position, sentence in enumerate(sentences)
            if any(
                editable_span(sentence, match.start(), match.end(), protected_texts)
                for match in re.finditer(r"[?？؟]", sentence)
            )
        ]
        if len(sentences) == 1 or question_goal and len(question_indices) != 1:
            parts.append(part)
            continue
        for position, sentence in enumerate(sentences):
            is_question = question_goal is not None and position in question_indices
            parts.append(
                standard.model_copy(
                    update={
                        "reply": strip_terminal_period(sentence),
                        "question_goal": question_goal if is_question else None,
                        "question_part": "reply" if is_question else None,
                    }
                )
            )
    return tuple(parts)
