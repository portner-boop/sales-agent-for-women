import asyncio
from typing import Any

import pytest

from app.application import Application
from app.modules.agent.entities import SchemaT
from app.shared.storage import fetchall
from tests.helpers import conversation, interpretation_operation, scalar, update
from tests.test_followups import due, enable
from tests.test_message_timing import paced_app

__all__ = ["paced_app"]


async def test_no_questions_keeps_substantive_second_part(app: Application) -> None:
    await app.telegram.ingest(
        [update(1, "No questions please. Share your thoughts about music")]
    )
    current = await conversation(app)
    message = (await app.conversations.messages(current.scope))[-1]
    interpretation = interpretation_operation(
        message, "question_permission", False, "boundary"
    )
    interpretation["continue_conversation"] = True
    app.model.responses["interpret"].append(interpretation)
    app.model.responses["generate"].append(
        {
            "reply": "I like acoustic guitar",
            "continuation": "A simple melody can beat a big orchestra for me",
            "dialogue_act": "opinion",
        }
    )
    await app.drain()
    app.clock.advance(4)
    await app.drain()
    context = next(data for task, data in app.model.calls if task == "generate")
    assert context["DIALOGUE_PLAN"]["requires_continuation"] is False
    assert context["DIALOGUE_PLAN"]["questions_allowed"] is False
    assert [text for _, text in app.sender.sent] == [
        "I like acoustic guitar",
        "A simple melody can beat a big orchestra for me",
    ]
    assert await scalar(app, "SELECT COUNT(*) FROM dialogue_threads") == 0
    assert not any(task in {"repair", "recover"} for task, _ in app.model.calls)


async def test_third_question_repairs_to_declarative_continuation(
    app: Application,
) -> None:
    turns = [
        ("I enjoy music", "I like acoustic music", "What music do you like?", "music"),
        (
            "Mostly jazz",
            "Jazz has a lovely rhythm",
            "Do you enjoy concerts?",
            "concerts",
        ),
        ("Small ones", "Small venues appeal to me too", "Which venue?", "venue"),
    ]
    app.model.responses["repair"].append(
        {
            "reply": "Small venues appeal to me too",
            "continuation": "I'd choose a simple acoustic set over a huge production",
            "dialogue_act": "opinion",
        }
    )
    for index, (user, reply, question, goal) in enumerate(turns, start=1):
        app.model.responses["interpret"].append(
            {
                "intent": "chat",
                "dialogue_mode": "continuation",
                "language": "en",
                "continue_conversation": True,
            }
        )
        app.model.responses["generate"].append(
            {
                "reply": reply,
                "continuation": question,
                "dialogue_act": "reaction_and_question",
                "question_goal": goal,
                "question_part": "continuation",
            }
        )
        await app.telegram.ingest([update(index, user)])
        await app.drain()
        app.clock.advance(4)
        await app.drain()
    contexts = [data for task, data in app.model.calls if task == "generate"]
    assert contexts[-1]["DIALOGUE_PLAN"]["questions_allowed"] is False
    assert contexts[-1]["DIALOGUE_PLAN"]["requires_continuation"] is False
    assert len(app.sender.sent) == 6
    assert not any("?" in text for _, text in app.sender.sent[-2:])
    assert app.sender.sent[-1][1].endswith(
        "choose a simple acoustic set over a huge production"
    )
    assert [task for task, _ in app.model.calls].count("repair") == 1


@pytest.mark.parametrize(
    "mode,continue_conversation",
    [("closing", True), ("repair", True), ("service", False)],
)
async def test_initiative_does_not_force_extra_part_on_resolved_turn(
    app: Application, mode: str, continue_conversation: bool
) -> None:
    app.model.responses["interpret"].append(
        {
            "intent": "chat",
            "dialogue_mode": mode,
            "language": "en",
            "continue_conversation": continue_conversation,
        }
    )
    app.model.responses["generate"].append(
        {"reply": "Understood", "dialogue_act": "acknowledge"}
    )
    await app.telegram.ingest([update(1, "Thanks, that answers it")])
    await app.drain()
    context = next(data for task, data in app.model.calls if task == "generate")
    assert context["DIALOGUE_PLAN"]["requires_continuation"] is False
    assert app.sender.sent == [(1, "Understood")]


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
    assert not app.sender.sent
    delivery_at = await scalar(
        app, "SELECT MIN(created_at) FROM outbox_messages WHERE status='ready'"
    )
    app.clock.advance(delivery_at - app.clock.now())
    await app.drain()
    assert len(app.sender.sent) == 1
    assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"


@pytest.mark.parametrize("dialogue_mode", ["continuation", "repair"])
async def test_conservative_recovery_keeps_opinion_and_later_followup(
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
            "dialogue_act": "opinion_and_question",
            "followup_allowed": False,
        }
    )
    await app.telegram.ingest([update(1, "I went to a lovely park")])
    await app.drain()
    app.clock.advance(4)
    await app.drain()
    assert len(app.sender.sent) == 2
    assert "quiet corner" in app.sender.sent[0][1]
    assert (
        await scalar(app, "SELECT COUNT(*) FROM followups WHERE status='pending'") == 1
    )
    app.model.responses["followup"].append(
        {
            "reply": "Do you have a favorite spot in that park?",
            "dialogue_act": "question",
            "question_goal": "park_spot",
            "should_send": True,
        }
    )
    await due(app)
    await app.drain()
    assert len(app.sender.sent) == 3, await scalar(
        app, "SELECT metrics_json FROM agent_runs ORDER BY rowid DESC LIMIT 1"
    )
    assert app.sender.sent[-1][1] == "Do you have a favorite spot in that park?"
    assert [task for task, _ in app.model.calls][-2:] == ["followup", "validate"]


async def test_all_sentence_parts_arrive_before_followup_timer_starts(
    paced_app: Application,
) -> None:
    app = paced_app
    enable(app)
    app.model.responses["generate"].append(
        {
            "reply": "Jazz with forgotten lyrics sounds like a very human kind of concert. I like it when musicians can laugh at themselves",
            "continuation": "A relaxed crowd can make that moment even better",
            "dialogue_act": "reaction_and_opinion",
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
    assert len(rows) == 3
    assert rows[0]["created_at"] - received_at == pytest.approx(
        min(60, 3 + len(rows[0]["text"]) / 3)
    )
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
    assert await scalar(app, "SELECT COUNT(*) FROM followups") == 0
    third_due = await scalar(
        app, "SELECT created_at FROM outbox_messages WHERE id=?", (rows[2]["id"],)
    )
    assert third_due - app.clock.now() == pytest.approx(3 + len(rows[2]["text"]) / 3)
    app.clock.advance(third_due - app.clock.now())
    await app.drain()
    assert len(app.sender.sent) == 3
    timer = await scalar(app, "SELECT run_at FROM followups WHERE status='pending'")
    assert timer - app.clock.now() == 60
    app.model.responses["followup"].append(
        {
            "reply": "Did the crowd laugh along with her?",
            "dialogue_act": "question",
            "question_goal": "crowd_reaction",
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
    assert len(app.sender.sent) == 4
    assert await scalar(app, "SELECT COUNT(*) FROM followups") == 1


async def test_no_questions_prevents_followup_evaluation(
    app: Application,
) -> None:
    enable(app)
    await app.telegram.ingest([update(1, "No questions. Share a thought about music")])
    current = await conversation(app)
    message = (await app.conversations.messages(current.scope))[-1]
    interpretation = interpretation_operation(
        message, "question_permission", False, "boundary"
    )
    app.model.responses["interpret"].append(interpretation)
    app.model.responses["generate"].append(
        {
            "reply": "I enjoy acoustic guitar",
            "dialogue_act": "opinion",
            "followup_allowed": True,
        }
    )
    await app.drain()
    assert app.sender.sent == [(1, "I enjoy acoustic guitar")]
    assert await scalar(app, "SELECT COUNT(*) FROM followups") == 0
    app.clock.advance(3600)
    await app.drain()
    assert not any(task == "followup" for task, _ in app.model.calls)
