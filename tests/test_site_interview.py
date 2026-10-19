import pytest

from app.application import Application
from app.evaluation import receive
from app.modules.agent.questions import question_errors
from app.modules.agent.schemas import ReplyProposal
from tests.helpers import scalar
from tests.test_followups import due, enable


@pytest.mark.parametrize(
    "goal",
    [
        "site_enjoyment",
        "website_preferences",
        "platform_joining_reason",
        "site_usage_duration",
    ],
)
def test_site_interview_goals_are_blocked(goal: str) -> None:
    proposal = ReplyProposal(
        reply="What do you enjoy most about this site?",
        dialogue_act="question",
        question_goal=goal,
    )
    assert "SITE_INTERVIEW_QUESTION" in question_errors(proposal, {})


@pytest.mark.parametrize(
    "goal",
    [
        "food_preference",
        "music_interest",
        "website_error_message",
        "historical_site_interest",
        "archaeological_site_preferences",
    ],
)
def test_personal_questions_and_technical_clarification_remain_available(
    goal: str,
) -> None:
    proposal = ReplyProposal(
        reply="Which one?", dialogue_act="question", question_goal=goal
    )
    assert "SITE_INTERVIEW_QUESTION" not in question_errors(proposal, {})


@pytest.mark.parametrize("followup", [False, True])
async def test_site_interview_is_repaired_before_any_delivery(
    app: Application, followup: bool
) -> None:
    if followup:
        enable(app)
        app.model.responses["generate"].append(
            {"reply": "I like quiet old streets", "dialogue_act": "opinion"}
        )
        await receive(
            app,
            app.sender,
            1,
            "I enjoy exploring old neighborhoods",
            1,
            app.clock.now(),
        )
    bad = {
        "reply": "What do you enojy most abotu this site?",
        "dialogue_act": "question",
        "question_goal": "site_enjoyment",
        "casual_imperfection_used": True,
        "casual_imperfection": {
            "standard": "enjoy most about",
            "written": "enojy most abotu",
        },
    }
    good = (
        {
            "reply": "Do you prefer exploring on foot?",
            "dialogue_act": "question",
            "question_goal": "exploring_on_foot",
        }
        if followup
        else {
            "reply": "I like finding a small cafe on a quiet street",
            "dialogue_act": "opinion",
        }
    )
    app.model.responses["followup" if followup else "generate"].append(
        {**bad, **({"should_send": True} if followup else {})}
    )
    app.model.responses["repair"].append(
        {**good, **({"should_send": True} if followup else {})}
    )
    if followup:
        await due(app)
        await app.drain()
    else:
        await receive(
            app,
            app.sender,
            1,
            "I enjoy exploring old neighborhoods",
            1,
            app.clock.now(),
        )
    assert app.sender.sent[-1][1] == good["reply"]
    assert all("site" not in text for _, text in app.sender.sent)
    repair = next(data for task, data in app.model.calls if task == "repair")
    assert "SITE_INTERVIEW_QUESTION" in repair["REPAIR_REASON_CODES"]
    assert (
        await scalar(
            app,
            "SELECT COUNT(*) FROM dialogue_threads WHERE question_goal='site_enjoyment'",
        )
        == 0
    )


async def test_recovery_can_keep_personal_answer_and_drop_site_question(
    app: Application,
) -> None:
    for task in ("generate", "repair"):
        app.model.responses[task].append(
            {"reply": "I explored that street yesterday", "dialogue_act": "chat"}
        )
    for task in ("validate", "validate_repair"):
        app.model.responses[task].append(
            {"decision": "reject", "reason_codes": ["INVENTED_PERSONA_CIRCUMSTANCE"]}
        )
    app.model.responses["recover"].append(
        {
            "reply": "I like quiet old streets",
            "continuation": "What do you like about this site?",
            "dialogue_act": "chat",
            "question_goal": "site_preferences",
            "question_part": "continuation",
        }
    )
    await receive(
        app, app.sender, 1, "I enjoy exploring old neighborhoods", 1, app.clock.now()
    )
    assert [text for _, text in app.sender.sent] == ["I like quiet old streets"]
    assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"
