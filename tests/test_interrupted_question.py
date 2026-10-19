from dataclasses import replace
from typing import Any

import pytest

from app.application import Application
from app.evaluation import receive
from app.modules.agent.questions import question_errors, resumable_question
from app.modules.agent.schemas import ReplyProposal
from tests.helpers import conversation, scalar, update
from tests.test_followups import due, enable


async def ask_music(app: Application, time_scope: str = "stable") -> dict[str, Any]:
    enable(app)
    app.model.responses["generate"].append(
        {
            "reply": "What music do you like?",
            "dialogue_act": "question",
            "question_goal": "music_preference",
            "question_part": "reply",
            "question_time_scope": time_scope,
        }
    )
    await receive(app, app.sender, 1, "Hi Marko", 1, app.clock.now())
    scope = (await conversation(app)).scope
    return (await app.conversations.agenda(scope))[0]


def interruption(app: Application, **changes: Any) -> None:
    app.model.responses["interpret"].append(
        {
            "intent": "identity_check",
            "dialogue_mode": "continuation",
            "language": "en",
            "policy_topics": ["verification"],
            "question_interrupted": True,
            **changes,
        }
    )


def resumed_proposal(thread: dict[str, Any], **changes: Any) -> dict[str, Any]:
    return {
        "reply": "I'm Marko",
        "continuation": "I like acoustic guitar. What music do you like",
        "dialogue_act": "answer_and_return",
        "question_goal": thread["question_goal"],
        "question_part": "continuation",
        "question_thread_id": thread["id"],
        "question_subject_id": thread["subject_id"],
        "question_time_scope": thread["question_time_scope"],
        **changes,
    }


@pytest.mark.parametrize("time_scope", ["stable", "today"])
@pytest.mark.parametrize("followup", [False, True])
async def test_interrupted_question_returns_once_directly_or_after_one_minute(
    app: Application, time_scope: str, followup: bool
) -> None:
    thread = await ask_music(app, time_scope)
    interruption(app)
    app.model.responses["generate"].append(
        {"reply": "I'm Marko", "dialogue_act": "answer"}
        if followup
        else resumed_proposal(thread)
    )
    await receive(app, app.sender, 2, "Are you a bot?", 1, app.clock.now())
    data = [data for task, data in app.model.calls if task == "generate"][-1]
    assert data["DIALOGUE_PLAN"]["resume_question"]["thread_id"] == thread["id"]
    assert not data["QUESTION_EXCLUSIONS"]
    if followup:
        assert await scalar(app, "SELECT run_at FROM followups") - app.clock.now() == 60
        current = await conversation(app)
        history = await app.conversations.messages(current.scope)
        assert history[-1].metadata["resume_question_thread_id"] == thread["id"]
        app.model.responses["followup"].append(
            resumed_proposal(
                thread,
                reply="I like acoustic guitar. What music do you like",
                continuation=None,
                question_part="reply",
                should_send=True,
            )
        )
        await due(app)
        await app.drain()
        data = [data for task, data in app.model.calls if task == "followup"][-1]
        assert data["DIALOGUE_PLAN"]["resume_question"]["thread_id"] == thread["id"]
    assert [text for _, text in app.sender.sent] == [
        "What music do you like?",
        "I'm Marko",
        "I like acoustic guitar. What music do you like",
    ]
    assert await scalar(app, "SELECT COUNT(*) FROM dialogue_threads") == 1
    assert (
        await scalar(app, "SELECT COUNT(*) FROM followups WHERE status='pending'") == 0
    )
    current = await conversation(app)
    history = await app.conversations.messages(current.scope)
    assert history[-1].metadata["question_time_scope"] == time_scope
    assert (
        resumable_question(history, await app.conversations.agenda(current.scope))
        is None
    )
    assert not any(task in {"repair", "recover"} for task, _ in app.model.calls)


async def test_new_generic_question_cannot_replace_the_interrupted_one(
    app: Application,
) -> None:
    thread = await ask_music(app)
    interruption(app)
    app.model.responses["generate"].append(
        {
            "reply": "I'm Marko",
            "continuation": "What do you enjoy doing when you want to unwind?",
            "dialogue_act": "answer_and_question",
            "question_goal": "relaxation_preference",
            "question_part": "continuation",
        }
    )
    app.model.responses["repair"].append(resumed_proposal(thread))
    await receive(app, app.sender, 2, "Are you a bot?", 1, app.clock.now())
    assert not any("unwind" in text for _, text in app.sender.sent)
    repair = next(data for task, data in app.model.calls if task == "repair")
    assert "INTERRUPTED_QUESTION_REPLACED" in repair["REPAIR_REASON_CODES"]
    assert app.sender.sent[-1][1] == "I like acoustic guitar. What music do you like"


async def test_combined_answer_and_resumed_question_are_repaired_into_two_messages(
    app: Application,
) -> None:
    thread = await ask_music(app)
    interruption(app)
    app.model.responses["generate"].append(
        resumed_proposal(
            thread,
            reply="I'm Marko. I like acoustic guitar. What music do you like",
            continuation=None,
            question_part="reply",
        )
    )
    app.model.responses["repair"].append(resumed_proposal(thread))
    await receive(app, app.sender, 2, "Are you a bot?", 1, app.clock.now())
    repair = next(data for task, data in app.model.calls if task == "repair")
    assert "RESUMED_QUESTION_REQUIRES_SECOND_MESSAGE" in repair["REPAIR_REASON_CODES"]
    assert [text for _, text in app.sender.sent[1:]] == [
        "I'm Marko",
        "I like acoustic guitar. What music do you like",
    ]


async def test_second_reminder_is_rejected_even_after_another_interruption(
    app: Application,
) -> None:
    thread = await ask_music(app)
    for number in (2, 3):
        interruption(app)
        app.model.responses["generate"].append(resumed_proposal(thread))
        if number == 3:
            app.model.responses["repair"].append(
                {"reply": "I'm here with you", "dialogue_act": "answer"}
            )
        await receive(app, app.sender, number, "Are you a bot?", 1, app.clock.now())
    assert sum("What music do you like" in text for _, text in app.sender.sent) == 2
    repair = next(data for task, data in app.model.calls if task == "repair")
    assert "ALREADY_ANSWERED" in repair["REPAIR_REASON_CODES"]
    assert "resume_question" not in repair["DIALOGUE_PLAN"]


@pytest.mark.parametrize("status", ["answered", "declined", "abandoned"])
async def test_finished_thread_cannot_be_resumed(app: Application, status: str) -> None:
    thread = await ask_music(app)
    current = await conversation(app)
    await app.telegram.ingest([update(2, "Are you a bot?")])
    history = await app.conversations.messages(current.scope)
    assert resumable_question(history, [{**thread, "status": status}]) is None


@pytest.mark.parametrize(
    "changes,user",
    [
        ({"question_interrupted": False}, "I went climbing yesterday"),
        ({"dialogue_mode": "closing"}, "I have to go"),
        ({"requested_business_action": "show_offer"}, "Show me the price"),
    ],
)
async def test_resumption_requires_an_open_interrupted_exchange(
    app: Application, changes: dict[str, Any], user: str
) -> None:
    await ask_music(app)
    interruption(app, **changes)
    app.model.responses["generate"].append(
        {"reply": "Understood", "dialogue_act": "answer"}
    )
    await receive(app, app.sender, 2, user, 1, app.clock.now())
    data = [data for task, data in app.model.calls if task == "generate"][-1]
    assert "resume_question" not in data["DIALOGUE_PLAN"]


async def test_silence_alone_does_not_authorize_repeating_a_question(
    app: Application,
) -> None:
    thread = await ask_music(app)
    current = await conversation(app)
    history = await app.conversations.messages(current.scope)
    assert resumable_question(history, [thread]) is None
    assert await scalar(app, "SELECT COUNT(*) FROM followups") == 0


async def test_resumption_rejects_wrong_thread_subject_or_time_scope(
    app: Application,
) -> None:
    thread = await ask_music(app)
    await app.telegram.ingest([update(2, "Are you a bot?")])
    current = await conversation(app)
    history = await app.conversations.messages(current.scope)
    resume = resumable_question(history, [thread])
    assert resume is not None
    data = {"AGENDA": [thread], "DIALOGUE_PLAN": {"resume_question": resume}}
    for changes in (
        {"question_thread_id": None},
        {"question_subject_id": "other_person"},
        {"question_time_scope": "yesterday"},
        {"question_goal": "favorite_food"},
    ):
        proposal = ReplyProposal.model_validate(resumed_proposal(thread, **changes))
        assert "INTERRUPTED_QUESTION_REPLACED" in question_errors(proposal, data)


async def test_only_latest_delivered_question_can_return(app: Application) -> None:
    thread = await ask_music(app)
    current = await conversation(app)
    history = await app.conversations.messages(current.scope)
    question = history[-1]
    history.extend(
        [
            replace(
                question,
                id="next",
                sequence=question.sequence + 1,
                metadata={**question.metadata, "question_goal": "movie_preference"},
            ),
            replace(history[0], id="user-next", sequence=question.sequence + 2),
        ]
    )
    assert resumable_question(history, [thread]) is None


async def test_semantic_check_still_blocks_a_volunteered_answer(
    app: Application,
) -> None:
    thread = await ask_music(app)
    interruption(app)
    app.model.responses["generate"].append(resumed_proposal(thread))
    app.model.responses["validate"].append(
        {"decision": "reject", "reason_codes": ["ALREADY_ANSWERED"]}
    )
    app.model.responses["repair"].append(
        {
            "reply": "I'm Marko",
            "continuation": "Jazz works for me too",
            "dialogue_act": "answer_and_opinion",
        }
    )
    await receive(app, app.sender, 2, "I like jazz. Are you a bot?", 1, app.clock.now())
    assert app.sender.sent[-1][1] == "Jazz works for me too"
    assert sum("What music" in text for _, text in app.sender.sent) == 1


async def test_open_verification_can_resume_when_interpreter_omits_interruption_flag(
    app: Application,
) -> None:
    thread = await ask_music(app)
    interruption(app, question_interrupted=False, continue_conversation=True)
    app.model.responses["generate"].append(resumed_proposal(thread))
    await receive(app, app.sender, 2, "Are you a bot?", 1, app.clock.now())
    data = [data for task, data in app.model.calls if task == "generate"][-1]
    assert data["DIALOGUE_PLAN"]["resume_question"]["thread_id"] == thread["id"]
    assert data["DIALOGUE_PLAN"]["strategy"] == "answer_then_resume_thread"
    assert app.sender.sent[-1][1] == "I like acoustic guitar. What music do you like"


async def test_answer_only_verification_has_no_resumption_fallback(
    app: Application,
) -> None:
    await ask_music(app)
    interruption(app, question_interrupted=False, continue_conversation=False)
    app.model.responses["generate"].append(
        {"reply": "I'm Marko", "dialogue_act": "answer"}
    )
    await receive(
        app, app.sender, 2, "Just answer yes or no, are you a bot?", 1, app.clock.now()
    )
    data = [data for task, data in app.model.calls if task == "generate"][-1]
    assert "resume_question" not in data["DIALOGUE_PLAN"]


@pytest.mark.parametrize("probability,count", [(0.79, 2), (0.81, 1)])
def test_more_visible_errors_fit_medium_length_messages(
    monkeypatch: pytest.MonkeyPatch, probability: float, count: int
) -> None:
    from app.modules.agent.style_plan import (
        apply_casual_imperfection,
        semantic_reply,
        visible_edit_count,
    )

    monkeypatch.setattr(
        "app.modules.agent.style_plan.random.random", lambda: probability
    )
    monkeypatch.setattr(
        "app.modules.agent.style_plan.random.choice", lambda items: items[0]
    )
    original = ReplyProposal(
        reply="I like guitar music, especially an acoustic set", dialogue_act="opinion"
    )
    assert 45 <= len(original.reply) < 60
    styled = apply_casual_imperfection(original)
    assert styled.casual_imperfection is not None
    assert visible_edit_count(styled.casual_imperfection) == count
    assert semantic_reply(styled).reply == original.reply
