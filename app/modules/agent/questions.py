import re
import unicodedata
from typing import Any

from app.modules.agent.schemas import ReplyProposal
from app.modules.conversations.entities import Message

GOAL_ALIASES = {
    "how_old": "age",
    "age_years": "age",
    "years_old": "age",
    "type_of_dog": "breed",
    "dog_breed": "breed",
    "breed_of_dog": "breed",
    "pet_breed": "breed",
    "dogs": "dog",
    "cats": "cat",
}


def normalized_question(text: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def canonical_goal(goal: str) -> tuple[str, ...]:
    normalized = goal.casefold().replace("-", "_").replace(" ", "_")
    for source, target in GOAL_ALIASES.items():
        normalized = normalized.replace(source, target)
    return tuple(
        sorted(
            set(normalized.split("_"))
            - {"ask", "about", "and", "the", "get", "question", "what", "is"}
        )
    )


def resumable_question(
    history: list[Message], agenda: list[dict[str, Any]]
) -> dict[str, str] | None:
    delivered = [
        message
        for message in history
        if message.role == "assistant" and message.metadata.get("confirmed_delivery")
    ]
    last_question = next(
        (
            message
            for message in reversed(delivered[-10:])
            if message.metadata.get("question_goal")
        ),
        None,
    )
    if last_question is None or not any(
        message.role == "user" and message.sequence > last_question.sequence
        for message in history
    ):
        return None
    thread = next(
        (
            item
            for item in agenda
            if item["status"] == "awaiting_user"
            and item.get("last_question_message_id") == last_question.id
        ),
        None,
    )
    if thread is None:
        return None
    goal = thread["question_goal"] or thread["topic_kind"]
    asked = sum(
        canonical_goal(message.metadata.get("question_goal") or "")
        == canonical_goal(goal)
        and message.metadata.get("question_subject_id", "user_self")
        == thread["subject_id"]
        and message.metadata.get("question_time_scope", thread["question_time_scope"])
        == thread["question_time_scope"]
        for message in delivered
    )
    if asked != 1:
        return None
    return {
        "thread_id": thread["id"],
        "goal": goal,
        "subject_id": thread["subject_id"],
        "time_scope": thread["question_time_scope"],
    }


def question_errors(proposal: ReplyProposal, data: dict[str, Any]) -> list[str]:
    if not proposal.question_goal:
        return []
    errors = []
    goal = set(canonical_goal(proposal.question_goal))
    if (
        not goal.intersection(
            {"historical", "historic", "archaeological", "heritage", "tourist"}
        )
        and goal.intersection({"site", "website", "platform"})
        and goal.intersection(
            {
                "like",
                "likes",
                "liking",
                "enjoy",
                "enjoyment",
                "favorite",
                "favourite",
                "interest",
                "interests",
                "preference",
                "preferences",
                "experience",
                "opinion",
                "expectation",
                "expectations",
                "motivation",
                "reason",
                "joined",
                "joining",
                "registration",
                "tenure",
                "duration",
            }
        )
    ):
        errors.append("SITE_INTERVIEW_QUESTION")
    agenda = data.get("AGENDA", [])
    linked = next(
        (item for item in agenda if item["id"] == proposal.question_thread_id), None
    )
    if proposal.question_thread_id and linked is None:
        errors.append("UNKNOWN_QUESTION_THREAD")
    if linked and (
        linked["subject_id"] != proposal.question_subject_id
        or linked["question_time_scope"] != proposal.question_time_scope
        or canonical_goal(linked["question_goal"] or linked["topic_kind"])
        != canonical_goal(proposal.question_goal)
    ):
        errors.append("QUESTION_THREAD_MISMATCH")
    resume = data.get("DIALOGUE_PLAN", {}).get("resume_question")
    resuming = bool(
        resume
        and linked
        and linked["status"] == "awaiting_user"
        and linked["id"] == resume["thread_id"]
        and canonical_goal(proposal.question_goal) == canonical_goal(resume["goal"])
        and proposal.question_subject_id == resume["subject_id"]
        and proposal.question_time_scope == resume["time_scope"]
    )
    if resume and not resuming:
        errors.append("INTERRUPTED_QUESTION_REPLACED")
    if (
        resuming
        and data["DIALOGUE_PLAN"].get("strategy") == "answer_then_resume_thread"
        and proposal.question_part != "continuation"
    ):
        errors.append("RESUMED_QUESTION_REQUIRES_SECOND_MESSAGE")
    for item in agenda:
        same = (
            item.get("subject_id", "user_self") == proposal.question_subject_id
            and item["question_time_scope"] == proposal.question_time_scope
            and canonical_goal(item["question_goal"] or item["topic_kind"])
            == canonical_goal(proposal.question_goal)
        )
        if resuming and linked is item:
            continue
        if (same or linked is item) and item["status"] in {
            "answered",
            "declined",
            "awaiting_user",
        }:
            errors.append("ALREADY_ANSWERED")
    part = proposal.parts[proposal.question_part_index]
    if not resuming and any(
        normalized_question(part) == normalized_question(item["text"])
        for item in data.get("RECENT_DELIVERED", [])
    ):
        errors.append("REPEATED_QUESTION")
    return errors
