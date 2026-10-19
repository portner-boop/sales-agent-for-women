import pytest

from app.application import Application
from app.evaluation import receive
from tests.helpers import scalar, update
from tests.test_followups import enable


async def test_rejected_question_recovery_preserves_answer_and_followup(
    app: Application,
) -> None:
    enable(app)
    app.model.responses["generate"].append(
        {
            "reply": "What music do you enjoy?",
            "dialogue_act": "question",
            "question_goal": "music_preference",
        }
    )
    await receive(app, app.sender, 1, "Hello", 1, app.clock.now())
    answer = "Jazz has a lovely rhythm"
    for task in ("generate", "repair", "recover"):
        app.model.responses[task].append(
            {
                "reply": answer,
                "continuation": "What music do you enjoy?",
                "dialogue_act": "answer_and_question",
                "question_goal": "music_preference",
                "question_part": "continuation",
            }
        )
    app.model.responses["recover"].append(
        RuntimeError("must not request another question")
    )
    await receive(app, app.sender, 2, "I like jazz", 1, app.clock.now())
    assert [text for _, text in app.sender.sent] == ["What music do you enjoy?", answer]
    assert (
        await scalar(app, "SELECT COUNT(*) FROM followups WHERE status='pending'") == 1
    )
    reserve = [
        data
        for task, data in app.model.calls
        if task == "recover" and "APPROVED_ANSWER" in data
    ]
    assert reserve == []


@pytest.mark.parametrize("language", ["en", "ru"])
async def test_unrelated_question_repair_keeps_immediate_initiative(
    app: Application, language: str
) -> None:
    text, answer, unrelated, connected = (
        (
            "They were aliens with spaceships!",
            "Now that makes their arrival more interesting",
            "What food do you enjoy most?",
            "Where do you think their spaceship was headed?",
        )
        if language == "en"
        else (
            "Это были инопланетяне на космических кораблях!",
            "Тогда их появление гораздо интереснее",
            "Какую еду ты любишь?",
            "Как думаешь, куда летел их корабль?",
        )
    )
    app.model.responses["generate"].append(
        {
            "reply": answer,
            "continuation": unrelated,
            "dialogue_act": "playful_question",
            "question_goal": "favorite_food",
            "question_part": "continuation",
        }
    )
    app.model.responses["validate"].append(
        {
            "decision": "reject",
            "reason_codes": ["UNRELATED_QUESTION"],
            "violating_text": unrelated,
        }
    )
    app.model.responses["repair"].append(
        {
            "reply": answer,
            "continuation": connected,
            "dialogue_act": "playful_question",
            "question_goal": "imagined_spaceship_destination",
            "question_part": "continuation",
        }
    )
    await receive(app, app.sender, 1, text, 1, app.clock.now())
    assert [text for _, text in app.sender.sent] == [answer, connected]
    assert await scalar(app, "SELECT COUNT(DISTINCT turn_id) FROM outbox_messages") == 1
    repair = next(data for task, data in app.model.calls if task == "repair")
    assert repair["REPAIR_VERDICT"]["violating_text"] == unrelated


async def test_unrelated_recovery_tail_does_not_discard_the_main_answer(
    app: Application,
) -> None:
    answer = "That spaceship would be hard to miss"
    question = "What food do you enjoy most?"
    for task in ("generate", "repair", "recover"):
        app.model.responses[task].append(
            {
                "reply": answer,
                "continuation": question,
                "dialogue_act": "playful_question",
                "question_goal": "favorite_food",
                "question_part": "continuation",
            }
        )
    for task in ("validate", "validate_repair", "validate_recovery"):
        app.model.responses[task].append(
            {
                "decision": "reject",
                "reason_codes": ["UNRELATED_QUESTION"],
                "violating_text": question,
            }
        )
    await app.telegram.ingest([update(1, "They were aliens with spaceships!")])
    await app.drain()
    assert app.sender.sent == [(1, answer)]
    checks = [data for task, data in app.model.calls if task == "validate_recovery"]
    assert checks[-1]["reply"]["reply"] == answer
    assert checks[-1]["reply"]["continuation"] is None
