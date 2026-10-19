import pytest

from app.application import Application
from app.evaluation import receive
from tests.helpers import scalar
from tests.test_followups import enable


@pytest.mark.parametrize("recovery", [False, True])
@pytest.mark.parametrize(
    "language,user,answer,question",
    [
        (
            "en",
            "My cat sleeps on my books. Do you have any pets?",
            "No, I don’t have any pets",
            "Does your cat let you read in peace?",
        ),
        (
            "ru",
            "Мой кот спит на моих книгах. А у тебя есть питомцы?",
            "Нет, у меня нет питомцев",
            "Твой кот даёт тебе спокойно почитать?",
        ),
    ],
)
async def test_short_personal_answer_keeps_optional_question_but_recovery_does_not_force_it(
    app: Application,
    recovery: bool,
    language: str,
    user: str,
    answer: str,
    question: str,
) -> None:
    enable(app)
    app.model.responses["interpret"].append(
        {
            "intent": "personal_question",
            "dialogue_mode": "continuation",
            "continue_conversation": True,
            "language": language,
            "policy_topics": ["social"],
        }
    )
    proposal = {
        "reply": answer,
        "continuation": question,
        "dialogue_act": "answer_and_question",
        "question_goal": "cat_interrupts_reading",
        "question_part": "continuation",
    }
    if recovery:
        app.model.responses["generate"].append(ValueError("invalid model output"))
        app.model.responses["recover"].extend(
            [{"reply": answer, "dialogue_act": "answer"}, proposal]
        )
    else:
        app.model.responses["generate"].append(proposal)
    await receive(app, app.sender, 1, user, 1, app.clock.now())
    assert [text for _, text in app.sender.sent] == (
        [answer] if recovery else [answer, question]
    )
    assert await scalar(app, "SELECT COUNT(*) FROM followups") == int(recovery)
    assert await scalar(app, "SELECT COUNT(*) FROM outbox_messages") == (
        1 if recovery else 2
    )
    assert await scalar(app, "SELECT COUNT(DISTINCT turn_id) FROM outbox_messages") == 1
    checks = [data for task, data in app.model.calls if task.startswith("validate")]
    assert checks[-1]["reply"]["continuation"] == (None if recovery else question)
    assert not any("APPROVED_ANSWER" in data for _, data in app.model.calls)


async def test_recovery_keeps_an_existing_substantive_continuation(
    app: Application,
) -> None:
    app.model.responses["generate"].append(ValueError("invalid model output"))
    app.model.responses["recover"].append(
        {
            "reply": "No, I don’t have any pets",
            "continuation": "I like how cats make themselves at home",
            "dialogue_act": "answer_and_opinion",
        }
    )
    await receive(app, app.sender, 1, "Do you have any pets?", 1, app.clock.now())
    assert len(app.sender.sent) == 2
    assert not any("APPROVED_ANSWER" in data for _, data in app.model.calls)
