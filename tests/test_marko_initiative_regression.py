import asyncio
from typing import Any

import pytest

from app.application import Application
from app.modules.agent.entities import SchemaT
from app.shared.storage import fetchall
from tests.helpers import scalar, update
from tests.test_followups import due, enable
from tests.test_message_timing import paced_app

__all__ = ["paced_app"]


async def test_first_generation_can_finish_after_presentation_target(
    paced_app: Application, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = paced_app
    runtime = app.agent.settings.model_copy(
        update={"first_reply_max_seconds": 0.01, "first_deadline_seconds": 1}
    )
    app.agent.settings = runtime
    app.settings = app.settings.model_copy(update={"runtime": runtime})
    complete = app.model.complete

    async def slow(
        task: str,
        rules: str,
        data: dict[str, Any],
        schema: type[SchemaT],
        deadline_seconds: float,
    ) -> SchemaT:
        if task == "generate":
            await asyncio.sleep(0.02)
            app.clock.advance(0.02)
        return await complete(task, rules, data, schema, deadline_seconds)

    monkeypatch.setattr(app.model, "complete", slow)
    await app.telegram.ingest([update(1, "Hi Marko")])
    await app.drain()
    assert len(app.sender.sent) == 1
    assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"


@pytest.mark.parametrize("dialogue_mode", ["continuation", "repair"])
async def test_recovery_preserves_connected_question_and_followup(
    app: Application, dialogue_mode: str
) -> None:
    enable(app)
    app.model.responses["interpret"].append(
        {"intent": "chat", "dialogue_mode": dialogue_mode, "language": "en"}
    )
    app.model.responses["generate"].append(
        {"reply": "I visited that park yesterday", "dialogue_act": "answer"}
    )
    app.model.responses["validate"].append(
        {"decision": "reject", "reason_codes": ["INVENTED_PERSONA_CIRCUMSTANCE"]}
    )
    app.model.responses["repair"].append(
        {"reply": "I visited another park yesterday", "dialogue_act": "answer"}
    )
    app.model.responses["validate_repair"].append(
        {"decision": "reject", "reason_codes": ["INVENTED_PERSONA_CIRCUMSTANCE"]}
    )
    app.model.responses["recover"].append(
        {
            "reply": "I like parks with a quiet corner by the water. Crowded paths aren't really my thing",
            "continuation": "What did you like most about that park?",
            "dialogue_act": "opinion_and_question",
            "question_goal": "favorite_park_detail",
            "question_part": "continuation",
            "followup_allowed": True,
        }
    )
    await app.telegram.ingest([update(1, "I went to a lovely park")])
    await app.drain()
    app.clock.advance(4)
    await app.drain()
    assert len(app.sender.sent) == 2
    assert app.sender.sent[-1][1] == "What did you like most about that park?"
    assert (
        await scalar(app, "SELECT COUNT(*) FROM followups WHERE status='pending'") == 1
    )
    app.model.responses["followup"].append(
        {
            "reply": "A bench by the water beats a noisy terrace for me",
            "dialogue_act": "opinion",
            "should_send": True,
        }
    )
    await due(app)
    await app.drain()
    assert len(app.sender.sent) == 3, await scalar(
        app, "SELECT metrics_json FROM agent_runs ORDER BY rowid DESC LIMIT 1"
    )
    assert app.sender.sent[-1][1] == "A bench by the water beats a noisy terrace for me"
    assert [task for task, _ in app.model.calls][-2:] == ["followup", "validate"]


async def test_two_parts_arrive_before_followup_timer_starts(
    paced_app: Application,
) -> None:
    app = paced_app
    enable(app)
    app.model.responses["generate"].append(
        {
            "reply": "Jazz with forgotten lyrics sounds like a very human kind of concert. I like it when musicians can laugh at themselves",
            "continuation": "Did the audience join in, or did everyone pretend not to notice?",
            "dialogue_act": "reaction_and_question",
            "question_goal": "concert_audience_reaction",
            "question_part": "continuation",
            "followup_allowed": True,
        }
    )
    await app.telegram.ingest(
        [update(1, "I went to a jazz concert and the singer forgot half the lyrics")]
    )
    received_at = app.clock.now()
    await app.turn_worker.run_once()
    async with app.database.read() as tx:
        rows = await fetchall(
            tx, "SELECT id,text,created_at FROM outbox_messages ORDER BY rowid"
        )
    assert len(rows) == 2
    assert rows[0]["created_at"] - received_at <= 10
    app.clock.advance(rows[0]["created_at"] - app.clock.now())
    await app.delivery.run_once()
    assert len(app.sender.sent) == 1
    assert await scalar(app, "SELECT COUNT(*) FROM followups") == 0
    second_due = await scalar(
        app, "SELECT created_at FROM outbox_messages WHERE id=?", (rows[1]["id"],)
    )
    second_delay = 3 + len(rows[1]["text"]) / 3
    assert second_due - app.clock.now() == pytest.approx(second_delay)
    app.clock.advance(second_delay - 0.01)
    await app.drain()
    assert len(app.sender.sent) == 1
    assert await scalar(app, "SELECT COUNT(*) FROM followups") == 0
    app.clock.advance(0.01)
    await app.drain()
    assert len(app.sender.sent) == 2
    timer = await scalar(app, "SELECT run_at FROM followups WHERE status='pending'")
    assert 30 <= timer - app.clock.now() <= 60
    app.model.responses["followup"].append(
        {
            "reply": "I think a little imperfection makes a live performance more memorable",
            "dialogue_act": "opinion",
            "should_send": True,
        }
    )
    await due(app)
    await app.turn_worker.run_once()
    next_due = await scalar(
        app,
        "SELECT created_at FROM outbox_messages WHERE json_extract(proposal_json,'$.origin')='followup'",
    )
    app.clock.advance(max(0, next_due - app.clock.now()))
    await app.drain()
    assert len(app.sender.sent) == 3
    assert await scalar(app, "SELECT COUNT(*) FROM followups") == 1
