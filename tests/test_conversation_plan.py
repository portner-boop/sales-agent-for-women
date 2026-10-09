import json

import pytest
from pydantic import ValidationError

from app.application import Application
from app.evaluation import receive
from app.modules.agent.entities import BuiltContext
from app.modules.agent.schemas import ReplyProposal
from app.modules.agent.validation import validate_reply
from app.shared.storage import fetchall
from tests.helpers import conversation, interpretation_operation, scalar, update


@pytest.mark.parametrize(
    "text",
    [
        "Да так, посидела немного и выдохнула. А ты как?",
        "I sat in the park for a while. How about you?",
        "Estuve en el parque. Y tú?",
        "Fiquei no parque um pouco. E você?",
    ],
)
async def test_reciprocal_answer_delivers_its_hook_without_next_input(
    app: Application, text: str
) -> None:
    app.model.responses["generate"].append(
        {
            "reply": "Нормально.",
            "continuation": "В парке тихо было или народу много?",
            "dialogue_act": "answer_and_question",
            "question_goal": "park_crowd",
            "question_part": "continuation",
            "question_time_scope": "today",
        }
    )
    await receive(app, app.sender, 1, text, 1, app.clock.now())
    assert [text for _, text in app.sender.sent] == [
        "Нормально",
        "В парке тихо было или народу много?",
    ]
    data = next(data for task, data in app.model.calls if task == "generate")
    assert data["DIALOGUE_PLAN"]["requires_continuation"] is True
    assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"
    assert await scalar(app, "SELECT COUNT(*) FROM said_ledger") == 2
    assert await scalar(app, "SELECT COUNT(*) FROM dialogue_threads") == 1
    assert [task for task, _ in app.model.calls] == [
        "interpret",
        "generate",
        "validate",
    ]


async def test_required_continuation_is_part_of_the_llm_response_contract(
    app: Application, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = app.model.complete
    checked = False

    async def check_schema(task, rules, data, schema, deadline_seconds):
        nonlocal checked
        if task == "generate":
            with pytest.raises(ValidationError):
                schema.model_validate({"reply": "Хорошо.", "dialogue_act": "answer"})
            checked = True
        return await original(task, rules, data, schema, deadline_seconds)

    monkeypatch.setattr(app.model, "complete", check_schema)
    app.model.responses["generate"].append(
        {
            "reply": "Хорошо.",
            "continuation": "Мне нравятся такие короткие передышки.",
            "dialogue_act": "answer_and_opinion",
        }
    )
    await receive(app, app.sender, 1, "А ты как?", 1, app.clock.now())
    assert checked
    assert len(app.sender.sent) == 2
    assert await scalar(app, "SELECT COUNT(*) FROM dialogue_threads") == 0


async def test_question_in_first_part_is_recorded_at_its_delivery(
    app: Application,
) -> None:
    app.model.responses["generate"].append(
        {
            "reply": "Какую музыку ты включаешь вечером?",
            "continuation": "Мне нравится, когда музыка не забивает разговор.",
            "dialogue_act": "question_and_opinion",
            "question_goal": "evening_music",
            "question_part": "reply",
        }
    )
    await app.telegram.ingest([update(1, "Давай про музыку.")])
    await app.turn_worker.run_once()
    async with app.database.read() as tx:
        rows = await fetchall(tx, "SELECT id FROM outbox_messages ORDER BY rowid")
    await app.delivery.send(rows[0]["id"])
    assert await scalar(app, "SELECT COUNT(*) FROM dialogue_threads") == 1
    assert await scalar(
        app, "SELECT last_question_message_id FROM dialogue_threads"
    ) == (await scalar(app, "SELECT id FROM messages WHERE external_id=100001"))
    second = json.loads(
        await scalar(
            app,
            "SELECT proposal_json FROM outbox_messages WHERE id=?",
            (rows[1]["id"],),
        )
    )
    assert second["question_goal"] is None
    assert "continuation" not in second
    app.clock.advance(4)
    await app.delivery.send(rows[1]["id"])
    assert await scalar(app, "SELECT COUNT(*) FROM said_ledger") == 2
    assert await scalar(app, "SELECT COUNT(*) FROM dialogue_threads") == 1


async def test_no_questions_boundary_allows_declarative_parts(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "Сегодня без вопросов.")])
    current = await conversation(app)
    message = (await app.conversations.messages(current.scope))[-1]
    app.model.responses["interpret"].append(
        interpretation_operation(
            message, predicate="question_permission", value=False, kind="boundary"
        )
    )
    await app.drain()
    app.model.responses["generate"].append(
        {
            "reply": "Хорошо.",
            "continuation": "Мне по душе спокойный разговор без спешки.",
            "dialogue_act": "answer_and_opinion",
        }
    )
    await receive(app, app.sender, 2, "Посидела в парке. А ты как?", 1, app.clock.now())
    data = [data for task, data in app.model.calls if task == "generate"][-1]
    assert data["DIALOGUE_PLAN"]["questions_allowed"] is False
    assert data["DIALOGUE_PLAN"]["requires_continuation"] is False
    assert [text for _, text in app.sender.sent][-2:] == [
        "Хорошо",
        "Мне по душе спокойный разговор без спешки",
    ]
    assert await scalar(app, "SELECT COUNT(*) FROM dialogue_threads") == 0


@pytest.mark.parametrize("mode", ["closing", "repair", "service"])
async def test_control_modes_do_not_require_a_reciprocal_hook(
    app: Application, mode: str
) -> None:
    app.model.responses["interpret"].append(
        {"intent": "chat", "dialogue_mode": mode, "language": "ru"}
    )
    app.model.responses["generate"].append(
        {"reply": "Хорошо.", "dialogue_act": "answer"}
    )
    await receive(app, app.sender, 1, "А ты как?", 1, app.clock.now())
    data = next(data for task, data in app.model.calls if task == "generate")
    assert data["DIALOGUE_PLAN"]["requires_continuation"] is False
    assert len(app.sender.sent) == 1
    assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"


async def test_forwarded_question_is_not_a_direct_reciprocal_request(
    app: Application,
) -> None:
    raw = update(1, "А ты как?")
    raw["message"]["forward_origin"] = {
        "type": "hidden_user",
        "sender_user_name": "Other",
        "date": 1900000000,
    }
    await app.telegram.ingest([raw])
    await app.drain()
    data = next(data for task, data in app.model.calls if task == "generate")
    assert data["DIALOGUE_PLAN"]["requires_continuation"] is False


@pytest.mark.parametrize(
    "goal,part,continuation",
    [
        (None, "reply", None),
        ("park_crowd", "continuation", None),
    ],
)
async def test_impossible_question_placement_is_rejected(
    app: Application, goal: str | None, part: str, continuation: str | None
) -> None:
    await app.telegram.ingest([update(1, "Привет")])
    await app.turn_worker.run_once()
    context = next(data for task, data in app.model.calls if task == "generate")
    manifest = json.loads(await scalar(app, "SELECT manifest_json FROM agent_runs"))
    from app.modules.agent.schemas import ContextManifest

    proposal = ReplyProposal.model_validate(
        {
            "reply": "Хорошо.",
            "dialogue_act": "answer",
            "continuation": continuation,
            "question_goal": goal,
            "question_part": part,
        }
    )
    assert "QUESTION_PART_MISMATCH" in validate_reply(
        proposal, BuiltContext(context, ContextManifest.model_validate(manifest))
    )


async def test_rejected_agenda_only_patch_does_not_block_the_reply(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "Посидела немного. А ты как?")])
    current = await conversation(app)
    message = (await app.conversations.messages(current.scope))[-1]
    app.model.responses["interpret"].append(
        {
            "intent": "answer_and_reciprocate",
            "dialogue_mode": "continuation",
            "language": "ru",
            "agenda_operations": [
                {
                    "topic_kind": "unrelated_question",
                    "status": "answered",
                    "question_goal": "unrelated_question",
                    "answer_message_ids": [message.id],
                }
            ],
        }
    )
    app.model.responses["validate_memory"].append(
        {"decision": "reject", "reason_codes": ["AGENDA_WRONG_LINK"]}
    )
    app.model.responses["generate"].append(
        {
            "reply": "Нормально.",
            "continuation": "В парке тихо было?",
            "dialogue_act": "answer_and_question",
            "question_goal": "park_crowd",
        }
    )
    await app.drain()
    app.clock.advance(4)
    await app.drain()
    assert [text for _, text in app.sender.sent] == ["Нормально", "В парке тихо было?"]
    assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"
    assert await scalar(app, "SELECT COUNT(*) FROM dialogue_threads") == 1
    assert (
        await scalar(app, "SELECT question_goal FROM dialogue_threads") == "park_crowd"
    )
    metrics = json.loads(await scalar(app, "SELECT metrics_json FROM agent_runs"))
    assert {"stage": "agenda_patch_discarded"} in metrics["validation_events"]
    assert len(app.model.calls) == 4


async def test_request_to_share_does_not_turn_into_another_question(
    app: Application,
) -> None:
    app.model.responses["generate"].append(
        {
            "reply": "Мне больше нравятся тихие места.",
            "continuation": "А тебе?",
            "dialogue_act": "opinion_and_question",
            "question_goal": "quiet_places_preference",
        }
    )
    app.model.responses["repair"].append(
        {
            "reply": "Мне больше нравятся тихие места.",
            "continuation": "Я бы выбрал лавку подальше от дорожки.",
            "dialogue_act": "opinion",
        }
    )
    await receive(
        app,
        app.sender,
        1,
        "Давай ты что-нибудь расскажи, а не только спрашивай.",
        1,
        app.clock.now(),
    )
    assert [text for _, text in app.sender.sent] == [
        "Мне больше нравятся тихие места",
        "Я бы выбрал лавку подальше от дорожки",
    ]
    repair = next(data for task, data in app.model.calls if task == "repair")
    assert "QUESTION_NOT_ALLOWED" in repair["REPAIR_REASON_CODES"]
    assert len(app.model.calls) == 4
    assert await scalar(app, "SELECT COUNT(*) FROM dialogue_threads") == 0
    assert not await app.memory.repository.items((await conversation(app)).scope)
    await receive(app, app.sender, 2, "Давай про музыку.", 1, app.clock.now())
    data = [data for task, data in app.model.calls if task == "generate"][-1]
    assert data["DIALOGUE_PLAN"]["questions_allowed"] is True
