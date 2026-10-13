import json

import pytest

from app.application import Application
from app.evaluation import receive
from app.modules.agent.style import strip_terminal_period
from tests.helpers import scalar, update


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Нормально.", "Нормально"),
        ("Coffee wins.", "Coffee wins"),
        ("Me gusta.", "Me gusta"),
        ("Gostei.", "Gostei"),
        ("Кофе с мёдом. Я бы попробовал.", "Кофе с мёдом. Я бы попробовал"),
        ("Цена 12.50.", "Цена 12.50"),
        ("Version 1.2.3.", "Version 1.2.3"),
        ("https://example.com/page", "https://example.com/page"),
        ("What about you?", "What about you?"),
        ("Really!", "Really!"),
        ("Maybe...", "Maybe..."),
        ("Maybe..", "Maybe.."),
        ("For example, e.g.", "For example, e.g."),
        ("In the U.S.", "In the U.S."),
        ("И т. д.", "И т. д."),
        ("  Good to hear.  ", "Good to hear"),
        ("", ""),
    ],
)
def test_phone_punctuation_preserves_meaning(text: str, expected: str) -> None:
    assert strip_terminal_period(text) == expected
    assert strip_terminal_period(expected) == expected


async def test_two_parts_and_recorded_proposals_match_actual_phone_text(
    app: Application,
) -> None:
    app.model.responses["generate"].append(
        {
            "reply": "Honey in coffee. I would try that.",
            "continuation": "Is that your usual coffee?",
            "dialogue_act": "reaction_and_question",
            "question_goal": "usual_coffee",
            "question_part": "continuation",
        }
    )
    await receive(
        app, app.sender, 1, "Black coffee with honey today.", 1, app.clock.now()
    )
    assert [text for _, text in app.sender.sent] == [
        "Honey in coffee. I would try that",
        "Is that your usual coffee?",
    ]
    events = json.loads(await scalar(app, "SELECT metrics_json FROM agent_runs"))[
        "validation_events"
    ]
    raw = next(e["proposal"] for e in events if e["stage"] == "generate_proposal")
    assert raw["reply"] == "Honey in coffee. I would try that."
    checked = next(
        data["reply"] for task, data in app.model.calls if task == "validate"
    )
    assert checked["reply"] == "Honey in coffee. I would try that"
    assert checked["continuation"] == "Is that your usual coffee?"
    assert (
        await scalar(app, "SELECT text FROM outbox_messages ORDER BY rowid LIMIT 1")
        == "Honey in coffee. I would try that"
    )
    assert [task for task, _ in app.model.calls] == [
        "interpret",
        "generate",
        "validate",
    ]


async def test_ellipsis_is_still_repaired(app: Application) -> None:
    app.model.responses["generate"].append(
        {"reply": "That is odd...", "dialogue_act": "reaction"}
    )
    app.model.responses["repair"].append(
        {"reply": "Honey is an interesting choice.", "dialogue_act": "reaction"}
    )
    await app.telegram.ingest([update(1, "Coffee with honey.")])
    await app.drain()
    assert app.sender.sent[-1][1] == "Honey is an interesting choice"
    repaired = next(data for task, data in app.model.calls if task == "repair")
    assert "STYLE_VIOLATION" in repaired["REPAIR_REASON_CODES"]


async def test_system_command_keeps_its_clear_punctuation(app: Application) -> None:
    await app.telegram.ingest([update(1, "/stop")])
    await app.drain()
    assert (
        app.sender.sent[-1][1]
        == "Chat stopped. Memory is retained. Use /start to resume."
    )
    assert not app.model.calls


async def test_followup_uses_the_same_phone_style(app: Application) -> None:
    from tests.test_followups import due, enable, seed

    enable(app)
    await seed(app)
    app.model.responses["followup"].append(
        {
            "reply": "Мне нравятся тихие парки.",
            "dialogue_act": "opinion",
            "should_send": True,
        }
    )
    await due(app)
    await app.drain()
    assert app.sender.sent[-1][1] == "Мне нравятся тихие парки"
    request = next(
        data for task, data in reversed(app.model.calls) if task == "validate"
    )
    assert request["reply"]["reply"] == "Мне нравятся тихие парки"
    assert await scalar(app, "SELECT status FROM followups") == "prepared"


async def test_recovery_uses_the_same_phone_style(app: Application) -> None:
    from app.shared.entities import EvidenceError

    app.model.responses["generate"].append(EvidenceError("Rejected claim"))
    app.model.responses["recover"].append(
        {"reply": "We can keep it simple.", "dialogue_act": "reaction"}
    )
    await app.telegram.ingest([update(1, "Keep it simple please.")])
    await app.drain()
    assert app.sender.sent[-1][1] == "We can keep it simple"
    checked = next(
        data for task, data in app.model.calls if task == "validate_recovery"
    )
    assert checked["reply"]["reply"] == "We can keep it simple"
