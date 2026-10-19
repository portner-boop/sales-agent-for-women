import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import pytest

from app.application import Application
from app.modules.agent.persona import persona_projection
from app.modules.agent.questions import question_errors
from app.modules.agent.schemas import ReplyProposal, TurnInterpretation
from app.modules.agent.service import ActiveRun
from app.modules.agent.style import unsolicited_day_part
from app.modules.agent.time import local_clock
from app.modules.agent.validation import validate_reply
from app.modules.delivery.timing import response_delays
from app.shared.config import RuntimeSettings
from app.shared.storage import Transaction
from tests.helpers import conversation, interpretation_operation, scalar, update


@pytest.mark.parametrize("permission", ["family", "relationship"])
def test_children_question_projects_same_canonical_fact(permission: str) -> None:
    assert persona_projection([permission], 1792054800)["children"] == "none"
    assert "children" not in persona_projection([], 1792054800)


async def test_expired_response_does_not_need_forget(app: Application) -> None:
    await app.telegram.ingest([update(1, "An old question")])
    app.clock.advance(181)
    await app.drain()
    assert not app.sender.sent
    assert await scalar(app, "SELECT pending FROM messages WHERE role='user'") == 0
    await app.telegram.ingest([update(2, "A new question")])
    await app.drain()
    assert len(app.sender.sent) == 1
    assert await scalar(app, "SELECT COUNT(*) FROM conversations") == 1


async def test_queue_deadline_is_not_reset_by_retry(app: Application) -> None:
    app.model.responses["interpret"].append(RuntimeError("model unavailable"))
    await app.telegram.ingest([update(1, "Tell me about music")])
    await app.drain()
    app.clock.advance(181)
    await app.drain()
    assert not app.sender.sent
    assert await scalar(app, "SELECT COUNT(*) FROM agent_runs") == 2
    assert (
        await scalar(app, "SELECT status FROM turns ORDER BY rowid DESC LIMIT 1")
        == "unanswered"
    )
    assert [task for task, _ in app.model.calls] == ["interpret"]


async def test_watchdog_restores_missing_job_without_duplicate_delivery(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "Hello")])

    async def lose(tx: Transaction) -> None:
        await tx.execute("DELETE FROM scheduled_jobs WHERE kind='turn'")

    await app.database.write(lose)
    await app.jobs.reconcile_turns(app.clock.now())
    await app.jobs.reconcile_turns(app.clock.now())
    await app.drain()
    assert len(app.sender.sent) == 1


async def test_parallel_chat_survives_a_blocked_scope_lock(app: Application) -> None:
    await app.telegram.ingest([update(1, "Hello", user=1), update(2, "Hola", user=2)])
    first = await conversation(app, 1)
    async with app.agent.locks[first.id]:
        blocked = asyncio.create_task(app.turn_worker.run_once())
        await asyncio.sleep(0)
        await app.turn_worker.run_once()
        await app.delivery.run_once()
        assert app.sender.sent[0][0] == 2
    await blocked
    await app.drain()
    assert {chat for chat, _ in app.sender.sent} == {1, 2}


def test_first_long_response_keeps_length_pacing() -> None:
    settings = RuntimeSettings(
        reply_pause_min_seconds=3, reply_pause_max_seconds=3, reply_delay_jitter_ratio=0
    )
    assert response_delays(
        ("a" * 140, "b" * 60), 100, 110, True, settings
    ) == pytest.approx((3 + 140 / 3 - 10, 23))


async def test_first_response_ignores_legacy_short_preparation_deadline(
    app: Application, monkeypatch: pytest.MonkeyPatch
) -> None:
    app.agent.settings = app.agent.settings.model_copy(
        update={"first_deadline_seconds": 0.001, "turn_deadline_seconds": 5}
    )
    original = app.model.complete

    async def complete(*args, **kwargs):
        await asyncio.sleep(0.02)
        return await original(*args, **kwargs)

    monkeypatch.setattr(app.model, "complete", complete)
    await app.telegram.ingest([update(1, "Hello")])
    await app.drain()
    assert len(app.sender.sent) == 1
    assert await scalar(app, "SELECT status FROM outbox_messages") == "delivered"


async def test_clock_evidence_does_not_change_during_context_rebuild(
    app: Application, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = app.agent._check_proposal
    checked: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    async def check(run: ActiveRun, stage: str, task: str) -> list[str]:
        assert run.context is not None
        before = run.context.data["CURRENT_TIME_FACTS"]
        app.clock.advance(61)
        await app.agent._build_context(run, request_reserve_tokens=1)
        checked.append((before, run.context.data["CURRENT_TIME_FACTS"]))
        return await original(run, stage, task)

    monkeypatch.setattr(app.agent, "_check_proposal", check)
    app.model.responses["generate"].append(
        {"reply": "I prefer a relaxed pace", "dialogue_act": "opinion"}
    )
    await app.telegram.ingest([update(1, "What time is it in Split?")])
    await app.drain()
    assert len(checked) == 1
    assert checked[0][0]
    assert checked[0][0] == checked[0][1]


async def test_selected_memory_excerpt_is_valid_source_without_full_original(
    app: Application, monkeypatch: pytest.MonkeyPatch
) -> None:
    await app.telegram.ingest([update(1, "I have two children")])
    current = await conversation(app)
    source = (await app.conversations.messages(current.scope))[0]
    app.model.responses["interpret"].append(
        interpretation_operation(source, "children_count", 2)
    )
    await app.drain()
    await app.telegram.ingest([update(2, "What do you remember about me?")])
    current = await conversation(app)
    turn = await app.conversations.claim_turn(current.scope, app.clock.now())
    assert turn is not None
    view = await app.memory.retrieve(
        current.scope, ["children_count"], ["user_self"], [], False, []
    )
    original = app.conversations.messages

    async def selected(*args: Any, **kwargs: Any) -> list[Any]:
        return [
            message
            for message in await original(*args, **kwargs)
            if message.id != source.id
        ]

    monkeypatch.setattr(app.conversations, "messages", selected)
    context = await app.agent.builder.build(
        turn,
        TurnInterpretation(
            intent="recall",
            dialogue_mode="continuation",
            language="en",
            recall_scope="general",
            retrieval_predicates=["children_count"],
        ),
        current,
        replace(view, sources=()),
    )
    assert source.id not in {
        message["id"] for message in context.data["RECENT_MESSAGES"]
    }
    reply = ReplyProposal(
        reply="You have two children",
        dialogue_act="recall",
        used_source_message_ids=[source.id],
    )
    assert "UNSUPPORTED_SOURCE" not in validate_reply(reply, context)
    assert "UNSUPPORTED_SOURCE" in validate_reply(
        reply.model_copy(update={"used_source_message_ids": ["invented-id"]}), context
    )


def test_rounding_crosses_midnight_and_exact_is_explicit() -> None:
    timestamp = datetime(2026, 10, 14, 21, 58, tzinfo=UTC).timestamp()
    assert (
        local_clock(timestamp, "Europe/Zagreb", False).isoformat()
        == "2026-10-15T00:00:00+02:00"
    )
    assert local_clock(timestamp, "Europe/Zagreb", True).strftime("%H:%M") == "23:58"


def test_future_evening_is_not_current_evening_even_with_memory_citation() -> None:
    assert unsolicited_day_part(
        "How's your evening going so far?", "What are you doing this evening?", True
    )


@pytest.mark.parametrize("status", ["awaiting_user", "answered", "declined"])
def test_question_subject_goal_and_scope_block_synonymous_repetition(
    status: str,
) -> None:
    data = {
        "AGENDA": [
            {
                "id": "dog",
                "subject_id": "pet_max",
                "question_goal": "dog_breed_and_age",
                "question_time_scope": "stable",
                "status": status,
            }
        ]
    }
    proposal = ReplyProposal(
        reply="How old is he and what breed?",
        dialogue_act="question",
        question_goal="age_and_breed",
        question_subject_id="pet_max",
        question_part="reply",
    )
    assert "ALREADY_ANSWERED" in question_errors(proposal, data)
    assert not question_errors(
        proposal.model_copy(update={"question_subject_id": "pet_luna"}), data
    )


async def test_general_recall_includes_children_after_topic_changes(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "I have two children")])
    current = await conversation(app)
    original = (await app.conversations.messages(current.scope))[0]
    app.model.responses["interpret"].append(
        interpretation_operation(original, "children_count", 2)
    )
    await app.drain()
    for number in range(2, 14):
        await app.telegram.ingest([update(number, "Another conversation topic")])
        await app.drain()
    app.model.responses["interpret"].append(
        {
            "intent": "recall",
            "language": "fr",
            "dialogue_mode": "continuation",
            "recall_scope": "general",
            "policy_topics": ["recall"],
        }
    )
    app.model.responses["generate"].append(
        {"reply": "Tu as deux enfants", "dialogue_act": "recall"}
    )
    await app.telegram.ingest([update(14, "Que sais-tu de moi ?")])
    await app.drain()
    context = [data for task, data in app.model.calls if task == "generate"][-1]
    assert any(
        fact["predicate"] == "children_count" and fact["value"] == 2
        for fact in context["RECALL_REQUIRED"]
    )
    trace = json.loads(
        await scalar(
            app, "SELECT manifest_json FROM agent_runs ORDER BY rowid DESC LIMIT 1"
        )
    )
    assert context["RECALL_REQUIRED"][0]["id"] in trace["memory_ids"]
    assert trace["policy"]["version"] == app.agent.catalog.version


async def test_new_input_is_not_discarded_with_expired_pending_message(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "An old request")])
    app.clock.advance(181)
    await app.telegram.ingest([update(2, "A new request")])
    await app.drain()
    assert len(app.sender.sent) == 1
    data = next(data for task, data in app.model.calls if task == "interpret")
    assert [item["text"] for item in data["CURRENT_INPUT"]] == ["A new request"]


def test_preference_for_mornings_is_not_current_time_assumption() -> None:
    assert not unsolicited_day_part(
        "I like quiet mornings", "What sort of weekends appeal to you?"
    )


async def test_benchmark_waits_for_retry_before_next_user_message(
    app: Application,
) -> None:
    from scripts.compare_policies import settle_response

    app.model.responses["interpret"].append(RuntimeError("temporary failure"))
    await app.telegram.ingest([update(1, "Tell me your opinion")])
    await app.drain()
    assert not app.sender.sent
    await settle_response(app, app.clock)
    assert len(app.sender.sent) == 1
    assert await scalar(app, "SELECT COUNT(*) FROM agent_runs") == 2
    assert await scalar(app, "SELECT COUNT(*) FROM messages WHERE pending=1") == 0


async def test_benchmark_retry_waits_for_first_part_before_second_part(
    app: Application,
) -> None:
    from scripts.compare_policies import settle_response

    app.agent.settings = app.agent.settings.model_copy(
        update={
            "reply_pacing_enabled": True,
            "reply_pause_min_seconds": 3,
            "reply_pause_max_seconds": 3,
            "reply_delay_jitter_ratio": 0,
        }
    )
    app.model.responses["interpret"].append(RuntimeError("temporary failure"))
    app.model.responses["generate"].append(
        {
            "reply": "A quiet walk sounds good",
            "continuation": "I like that kind of calm",
            "dialogue_act": "comment",
        }
    )
    await app.telegram.ingest([update(1, "I like quiet walks")])
    await app.drain()
    assert not app.sender.sent
    await settle_response(app, app.clock)
    assert [message for _, message in app.sender.sent] == [
        "A quiet walk sounds good",
        "I like that kind of calm",
    ]
