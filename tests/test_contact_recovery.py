import json

import pytest

from app.application import Application
from app.evaluation import receive
from tests.helpers import conversation, interpretation_operation, scalar, update
from tests.test_followups import due, enable

ANSWER = "I don’t share my number; I prefer getting to know each other here"
TAIL = "That honey cake sounds tempting, secret recipe and all"


@pytest.mark.parametrize("contact_verb", ["lets talk", "let's talk"])
async def test_normalized_rejected_tail_keeps_answer_and_later_followup(
    app: Application, contact_verb: str
) -> None:
    enable(app)
    for number, text, question, goal in [
        (1, "Hello", "What music do you enjoy?", "music_preference"),
        (2, "Jazz, and I enjoy baking", "What cakes do you bake?", "cake_preference"),
    ]:
        app.model.responses["generate"].append(
            {"reply": question, "dialogue_act": "question", "question_goal": goal}
        )
        await receive(app, app.sender, number, text, 1, app.clock.now())
    candidate = {
        "reply": ANSWER,
        "continuation": TAIL.replace("sounds", "soudns"),
        "dialogue_act": "boundary_contact",
        "casual_imperfection_used": True,
        "casual_imperfection": {"written": "soudns", "standard": "sounds"},
    }
    for task in ("generate", "repair", "recover"):
        app.model.responses[task].append(candidate)
    for task in ("validate", "validate_repair", "validate_recovery"):
        app.model.responses[task].append(
            {
                "decision": "reject",
                "reason_codes": ["CONVERSATION.REPETITION"],
                "violating_text": TAIL,
            }
        )
    await receive(
        app,
        app.sender,
        3,
        f"its a secret ;) ok then {contact_verb} on whatsapp, its more comfortable for me. whats your number?",
        1,
        app.clock.now(),
    )
    assert [text for _, text in app.sender.sent][2:] == [ANSWER]
    checks = [data for task, data in app.model.calls if task == "validate_recovery"]
    assert len(checks) == 2
    assert checks[0]["reply"]["continuation"] == TAIL
    assert checks[1]["reply"]["reply"] == ANSWER
    assert checks[1]["reply"]["continuation"] is None
    assert checks[1]["reply"]["casual_imperfection_used"] is False
    assert checks[1]["context"]["DIALOGUE_PLAN"]["questions_allowed"] is False
    assert (
        await scalar(app, "SELECT COUNT(*) FROM outbox_messages WHERE kind='service'")
        == 0
    )
    assert (
        await scalar(app, "SELECT status FROM agent_runs ORDER BY rowid DESC LIMIT 1")
        == "prepared"
    )
    assert (
        await scalar(app, "SELECT COUNT(*) FROM followups WHERE status='pending'") == 1
    )
    app.model.responses["followup"].append(
        {
            "reply": "Do you enjoy cooking savoury dishes too?",
            "question_goal": "savoury_cooking_interest",
            "dialogue_act": "question",
            "should_send": True,
        }
    )
    await due(app)
    await app.drain()
    assert app.sender.sent[-1] == (1, "Do you enjoy cooking savoury dishes too?")
    app.clock.advance(3600)
    await app.drain()
    assert len(app.sender.sent) == 4


async def test_forbidden_question_tail_is_removed_and_answer_checked(
    app: Application,
) -> None:
    enable(app)
    await app.telegram.ingest([update(1, "No questions. Can you share your WhatsApp?")])
    current = await conversation(app)
    original = (await app.conversations.messages(current.scope))[-1]
    app.model.responses["interpret"].append(
        interpretation_operation(original, "question_permission", False, "boundary")
    )
    for task in ("generate", "repair", "recover"):
        app.model.responses[task].append(
            {
                "reply": ANSWER,
                "continuation": "Do you enjoy cooking?",
                "dialogue_act": "boundary_contact",
                "question_goal": "cooking_interest",
                "question_part": "continuation",
            }
        )
    await app.drain()
    assert app.sender.sent == [(1, ANSWER)]
    checks = [data for task, data in app.model.calls if task == "validate_recovery"]
    assert len(checks) == 1
    assert checks[0]["reply"]["continuation"] is None
    assert checks[0]["reply"]["question_goal"] is None
    assert await scalar(app, "SELECT COUNT(*) FROM followups") == 0


@pytest.mark.parametrize("violation_in_main", [True, False])
async def test_recovery_never_delivers_an_unapproved_main_answer(
    app: Application, violation_in_main: bool
) -> None:
    candidate = {
        "reply": ANSWER,
        "continuation": TAIL,
        "dialogue_act": "boundary_contact",
    }
    for task in ("generate", "repair", "recover"):
        app.model.responses[task].append(candidate)
    for task in ("validate", "validate_repair", "validate_recovery"):
        app.model.responses[task].append(
            {
                "decision": "reject",
                "reason_codes": ["CONVERSATION.REPETITION"],
                "violating_text": ANSWER if violation_in_main else TAIL,
            }
        )
    app.model.responses["validate_recovery"].append(
        {
            "decision": "reject",
            "reason_codes": ["CORE.GROUNDING"],
            "violating_text": ANSWER,
        }
    )
    await app.telegram.ingest([update(1, "Let's talk on WhatsApp. Share your number?")])
    await app.turn_worker.run_once()
    assert not app.sender.sent
    assert await scalar(app, "SELECT COUNT(*) FROM outbox_messages") == 0
    metrics = json.loads(await scalar(app, "SELECT metrics_json FROM agent_runs"))
    trimmed = any(
        event["stage"] == "recovery_continuation_removed"
        for event in metrics["validation_events"]
    )
    assert trimmed is not violation_in_main
