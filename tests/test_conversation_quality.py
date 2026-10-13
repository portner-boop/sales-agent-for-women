from typing import Any

import pytest

from app.application import Application
from app.evaluation import receive
from app.modules.agent.entities import SchemaT
from tests.helpers import PLATFORM_INVITATION_EN, scalar, update
from tests.test_platform_transition import seed_count


@pytest.mark.parametrize(
    "question,candidate,answer,reason",
    [
        (
            "Can you give me your WhatsApp?",
            "I dont share any contact. Lets keep it here",
            "We’ve only just met. Let’s get to know each other here",
            "FORMULAIC_CAPABILITY_REPLY",
        ),
        (
            "I want to check that you are not fake",
            "I m not doing verification games",
            "Fair enough. What would you like to check?",
            "HOSTILE_VERIFICATION_BOUNDARY",
        ),
        (
            "I want to check that you are not fake",
            "Just trust me or leave",
            "I understand wanting to be sure",
            "HOSTILE_VERIFICATION_BOUNDARY",
        ),
        (
            "What music do you like?",
            "I couldn t give you a proper reply. Let's try again",
            "I like quiet music",
            "SERVICE_FAILURE_REPLY",
        ),
    ],
)
async def test_chat_spelling_does_not_bypass_quality_checks(
    app: Application, question: str, candidate: str, answer: str, reason: str
) -> None:
    app.model.responses["generate"].append(
        {"reply": candidate, "dialogue_act": "answer"}
    )
    app.model.responses["validate"].append(
        {"decision": "reject", "reason_codes": [reason]}
    )
    repair: dict[str, Any] = {"reply": answer, "dialogue_act": "answer"}
    if "?" in answer:
        repair.update(question_goal="verification_method", question_part="reply")
    app.model.responses["repair"].append(repair)
    await receive(app, app.sender, 1, question, 1, app.clock.now())
    assert app.sender.sent == [(1, answer)]
    request = next(data for task, data in app.model.calls if task == "repair")
    assert reason in request["REPAIR_REASON_CODES"]


@pytest.mark.parametrize(
    "answer",
    [
        "I m more into quiet music",
        "im more into quiet music",
        "I dont like noisy cafes",
        "Quiet music sounds good to me",
        "I like quiet music it helps me relax",
        "I like quiet music bettr",
    ],
)
async def test_readable_chat_imperfections_reach_delivery_unchanged(
    app: Application, answer: str
) -> None:
    app.model.responses["generate"].append({"reply": answer, "dialogue_act": "opinion"})
    await receive(app, app.sender, 1, "What music do you like?", 1, app.clock.now())
    assert app.sender.sent == [(1, answer)]
    assert await scalar(app, "SELECT text FROM outbox_messages") == answer
    assert not any(task in {"repair", "recover"} for task, _ in app.model.calls)


async def test_wrong_language_in_one_part_cannot_hide_in_the_other(
    app: Application,
) -> None:
    app.model.responses["generate"].append(
        {
            "reply": "Мне нравится спокойная музыка без слов, особенно тихая гитара",
            "continuation": "Quiet music sounds good",
            "dialogue_act": "opinion",
        }
    )
    answer = "Мне нравится тихая гитара"
    app.model.responses["validate"].append(
        {"decision": "reject", "reason_codes": ["WRONG_LANGUAGE"]}
    )
    app.model.responses["repair"].append({"reply": answer, "dialogue_act": "opinion"})
    await receive(app, app.sender, 1, "Какая музыка тебе нравится?", 1, app.clock.now())
    assert app.sender.sent == [(1, answer)]
    request = next(data for task, data in app.model.calls if task == "repair")
    assert "WRONG_LANGUAGE" in request["REPAIR_REASON_CODES"]


async def test_departure_is_repaired_before_platform_invitation(
    app: Application,
) -> None:
    await seed_count(app, 19)
    app.model.responses["generate"].append(
        {
            "reply": "I need to wrap up here",
            "dialogue_act": "answer",
            "platform_notice": PLATFORM_INVITATION_EN,
        }
    )
    answer = "Right now I’m reading your messages"
    app.model.responses["validate"].append(
        {"decision": "reject", "reason_codes": ["TRANSITION_IGNORED_QUESTION"]}
    )
    app.model.responses["repair"].append(
        {
            "reply": answer,
            "dialogue_act": "answer",
            "platform_notice": PLATFORM_INVITATION_EN,
        }
    )
    await app.telegram.ingest([update(2, "What’s new with you?")])
    await app.drain()
    assert app.sender.sent[-1][1] == app.commerce.transition_text("en", answer)
    request = next(data for task, data in app.model.calls if task == "repair")
    assert "TRANSITION_IGNORED_QUESTION" in request["REPAIR_REASON_CODES"]


async def test_language_excuse_is_checked_against_originals(
    app: Application, monkeypatch: pytest.MonkeyPatch
) -> None:
    complete = app.model.complete
    checked_rules: list[str] = []

    async def capture(
        task: str,
        rules: str,
        data: dict[str, Any],
        schema: type[SchemaT],
        deadline_seconds: float,
    ) -> SchemaT:
        if task == "validate":
            checked_rules.append(rules)
            assert data["context"]["APPLICATION_STATE"]["language"] == "en"
        return await complete(task, rules, data, schema, deadline_seconds)

    monkeypatch.setattr(app.model, "complete", capture)
    app.model.responses["generate"].append(
        {"reply": "My earlier messages were in Russian", "dialogue_act": "answer"}
    )
    app.model.responses["validate"].append(
        {"decision": "reject", "reason_codes": ["INVENTED_LANGUAGE_HISTORY"]}
    )
    answer = "Sorry, I switched languages by mistake"
    app.model.responses["repair"].append({"reply": answer, "dialogue_act": "repair"})
    await receive(
        app, app.sender, 1, "Why did you reply in Russian?", 1, app.clock.now()
    )
    assert app.sender.sent == [(1, answer)]
    assert len(checked_rules) == 1
    assert "Reject INVENTED_LANGUAGE_HISTORY" in checked_rules[0]


@pytest.mark.parametrize(
    "candidate",
    ["Here's what I know about you: you like quiet music", "Your profile: quiet music"],
)
async def test_profile_intro_is_repaired_without_dropping_the_fact(
    app: Application, candidate: str
) -> None:
    await receive(app, app.sender, 1, "I like quiet music", 1, app.clock.now())
    app.model.responses["generate"].append(
        {"reply": candidate, "dialogue_act": "recall"}
    )
    answer = "You like quiet music"
    app.model.responses["validate"].append(
        {"decision": "reject", "reason_codes": ["MEMORY_PROFILE_READOUT"]}
    )
    app.model.responses["repair"].append({"reply": answer, "dialogue_act": "recall"})
    await receive(
        app, app.sender, 2, "What do you remember about me?", 1, app.clock.now()
    )
    assert app.sender.sent[-1] == (1, answer)
    request = next(data for task, data in app.model.calls if task == "repair")
    assert "MEMORY_PROFILE_READOUT" in request["REPAIR_REASON_CODES"]
