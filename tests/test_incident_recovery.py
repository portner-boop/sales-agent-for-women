import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from app.application import Application
from app.modules.dialogue.daily_life import DAILY_LIFE, DailyLife
from app.modules.dialogue.persona import persona_projection
from app.shared.clock import MutableClock
from app.shared.config import Settings
from app.shared.storage import fetchall
from tests.helpers import FakeSender, ScriptedModel, conversation, scalar, update


async def wait_for_calls(model: ScriptedModel, count: int) -> None:
    async with asyncio.timeout(2):
        while len(model.calls) < count:
            model.entered.clear()
            await model.entered.wait()


async def test_new_input_cancels_old_generation_and_delayed_cancel_keeps_current(
    app: Application,
) -> None:
    model = app.model
    model.release = asyncio.Event()
    await app.telegram.ingest([update(1, "first thought")])
    first = asyncio.create_task(app.turn_worker.run_once())
    await wait_for_calls(model, 1)
    await app.telegram.ingest([update(2, "and another thought")])
    second = asyncio.create_task(app.turn_worker.run_once())
    await wait_for_calls(model, 2)
    current = await conversation(app)
    await app.agent.cancel_obsolete(current.scope, current.input_revision)
    assert len(app.agent.active) == 1
    model.release.set()
    await asyncio.gather(first, second)
    await app.drain()
    assert len(app.sender.sent) == 1
    assert [m["content"] for m in model.calls[-1][1]["CHAT_MESSAGES"]] == [
        "first thought",
        "and another thought",
    ]


async def test_three_inputs_replace_only_obsolete_generations(app: Application) -> None:
    model = app.model
    model.release = asyncio.Event()
    workers = []
    for index in range(1, 4):
        await app.telegram.ingest([update(index, f"thought {index}")])
        workers.append(asyncio.create_task(app.turn_worker.run_once()))
        await wait_for_calls(model, index)
    model.release.set()
    await asyncio.gather(*workers)
    await app.drain()
    assert len(app.sender.sent) == 1
    assert (await conversation(app)).input_revision == 3


async def test_first_generation_budget_and_outbox_include_queue_time(
    app: Application,
) -> None:
    budgets = []
    original = app.model.complete

    async def complete(*args: Any, **kwargs: Any) -> Any:
        budgets.append(args[-1])
        return await original(*args, **kwargs)

    app.model.complete = complete
    received = app.clock.now()
    await app.telegram.ingest([update(1, "hi")])
    app.clock.advance(8)
    await app.turn_worker.run_once()
    assert 0 < budgets[0] <= app.settings.runtime.first_deadline_seconds - 8
    deadline = await scalar(app, "SELECT deadline FROM outbox_messages LIMIT 1")
    assert deadline > received + app.settings.runtime.first_deadline_seconds
    app.clock.advance(app.settings.runtime.first_deadline_seconds - 8 + 1)
    await app.delivery.run_once()
    assert len(app.sender.sent) == 1
    assert await scalar(app, "SELECT deadline FROM outbox_messages LIMIT 1") == deadline
    await app.telegram.ingest([update(2, "hello again")])
    await app.drain()
    assert len(app.sender.sent) == 2


async def test_exhausted_generation_does_not_silence_next_input(
    app: Application,
) -> None:
    app.model.responses["dialogue"].append(TimeoutError())
    app.model.responses["dialogue_repair"].append(TimeoutError())
    await app.telegram.ingest([update(1, "hello")])
    await app.drain()
    assert len(app.sender.sent) == 1
    await app.telegram.ingest([update(2, "hello? are you there?")])
    await app.drain()
    assert len(app.sender.sent) == 2
    assert (await conversation(app)).preview_reply_count == 1


@pytest.fixture
async def proactive_app(tmp_path: Path) -> Any:
    settings = Settings.model_validate(
        {
            "environment": "test",
            "storage": {"directory": tmp_path},
            "runtime": {
                "coalesce_seconds": 0,
                "reply_timing": {"enabled": False},
                "followup_enabled": True,
            },
        }
    )
    instance = await Application.open(
        settings, FakeSender(), ScriptedModel(), MutableClock(1900000000)
    )
    try:
        yield instance
    finally:
        await instance.close()


async def test_two_opportunities_survive_restart_and_do_not_add_turns(
    proactive_app: Application,
) -> None:
    app = proactive_app
    await app.telegram.ingest([update(1, "I like music")])
    await app.drain()
    app.clock.advance(60)
    await app.drain()
    current = await conversation(app)
    assert current.preview_reply_count == 1
    assert app.model.calls[-1][1]["state"]["followup_opportunity"]["ordinal"] == 1
    await app.close()
    reopened = await Application.open(app.settings, app.sender, app.model, app.clock)
    try:
        app.clock.advance(120)
        await reopened.drain()
        assert len(app.sender.sent) == 3
        assert app.model.calls[-1][1]["state"]["followup_opportunity"]["ordinal"] == 2
        assert (await conversation(reopened)).preview_reply_count == 1
        app.clock.advance(300)
        await reopened.drain()
        assert len(app.sender.sent) == 3
    finally:
        await reopened.close()


async def test_new_input_cancels_second_opportunity(proactive_app: Application) -> None:
    app = proactive_app
    await app.telegram.ingest([update(1, "a story")])
    await app.drain()
    app.clock.advance(60)
    await app.drain()
    await app.telegram.ingest([update(2, "sorry was busy, what do you remember?")])
    await app.drain()
    assert (
        await scalar(
            app, "SELECT status FROM followups WHERE input_revision=1 AND ordinal=2"
        )
        == "cancelled"
    )
    assert len(app.sender.sent) == 3


async def test_unknown_first_followup_delivery_does_not_schedule_second(
    proactive_app: Application,
) -> None:
    from app.modules.delivery.entities import SendResult

    app = proactive_app
    await app.telegram.ingest([update(1, "a story")])
    await app.drain()
    app.sender.results.append(SendResult("delivery_unknown"))
    app.clock.advance(60)
    await app.drain()
    assert await scalar(app, "SELECT COUNT(*) FROM followups WHERE ordinal=2") == 0


async def test_second_followup_can_skip_after_first_pending_question(
    proactive_app: Application,
) -> None:
    app = proactive_app
    app.model.responses["dialogue"].append(
        {"parts": ["what happened next?"], "language": "en", "awaiting_user": True}
    )
    await app.telegram.ingest([update(1, "a story")])
    await app.drain()
    app.model.responses["dialogue"].append(
        {"parts": ["that customer picked quite a day to visit"], "language": "en"}
    )
    app.clock.advance(60)
    await app.drain()
    assert app.model.calls[-1][1]["state"]["followup_opportunity"]["required"] is True
    app.model.responses["dialogue"].append(
        {"parts": [], "should_send": False, "language": "en"}
    )
    app.clock.advance(180)
    await app.drain()
    assert len(app.sender.sent) == 2
    assert app.model.calls[-1][1]["state"]["followup_opportunity"]["required"] is False
    assert await scalar(app, "SELECT COUNT(*) FROM followups") == 2
    assert (
        await scalar(app, "SELECT status FROM followups WHERE ordinal=2") == "skipped"
    )


async def test_migration_preserves_pending_followup_and_original_messages(
    proactive_app: Application,
) -> None:
    app = proactive_app
    await app.telegram.ingest([update(1, "a story")])
    await app.drain()
    pending_id = await scalar(app, "SELECT id FROM followups")
    await app.close()
    with sqlite3.connect(app.settings.storage.directory / "app.db") as db:
        db.executescript(
            "ALTER TABLE followups RENAME TO followups_new; DROP INDEX followups_scope_status;"
        )
        db.executescript(
            (Path(__file__).parents[1] / "app/migrations/007_followups.sql").read_text()
        )
        db.executescript(
            "INSERT INTO followups SELECT id,scope_id,epoch,input_revision,control_revision,anchor_id,turn_id,run_at,expires_at,status,reason FROM followups_new; DROP TABLE followups_new; DELETE FROM schema_migrations WHERE name='013_followup_opportunities.sql';"
        )
    reopened = await Application.open(app.settings, app.sender, app.model, app.clock)
    try:
        assert await scalar(reopened, "SELECT id FROM followups") == pending_id
        assert await scalar(reopened, "SELECT ordinal FROM followups") == 1
        async with reopened.database.read() as tx:
            assert not await fetchall(tx, "PRAGMA foreign_key_check")
        app.clock.advance(60)
        await reopened.drain()
        assert len(app.sender.sent) == 2
    finally:
        await reopened.close()


def test_local_character_clock_uses_split_and_dst() -> None:
    from datetime import UTC, datetime

    winter = datetime(2026, 1, 20, 12, tzinfo=UTC).timestamp()
    summer = datetime(2026, 10, 20, 12, tzinfo=UTC).timestamp()
    assert persona_projection([], winter)["local_context"]["local_time"] == "13:00"
    assert persona_projection([], summer)["local_context"]["local_time"] == "14:00"
    assert (
        persona_projection([], summer)["local_context"]["timezone"] == "Europe/Zagreb"
    )
    assert (
        DAILY_LIFE.context(summer, "Europe/Zagreb")["daily_scene"]["authorship"]
        == "approved_fictional_character_history"
    )


def test_daily_life_rejects_gaps_or_overlaps() -> None:
    from pydantic import ValidationError

    data = DAILY_LIFE.model_dump()
    data["scenes"].pop()
    with pytest.raises(ValidationError):
        DailyLife.model_validate(data)


async def test_job_worker_survives_a_transient_claim_error(
    app: Application, monkeypatch: pytest.MonkeyPatch
) -> None:
    recovered = asyncio.Event()
    attempts = 0

    async def claim(*args: Any) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise sqlite3.OperationalError("temporary storage failure")
        recovered.set()
        return None

    monkeypatch.setattr(app.jobs, "claim", claim)
    task = asyncio.create_task(app.turn_worker.run())
    try:
        await asyncio.wait_for(recovered.wait(), timeout=3)
        assert not task.done()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_later_bubbles_have_independent_deadlines_after_first_reply(
    app: Application,
) -> None:
    received = app.clock.now()
    app.model.responses["dialogue"].append(
        {
            "parts": ["hello", "what brought you here?"],
            "language": "en",
        }
    )
    await app.telegram.ingest([update(1, "hi")])
    await app.drain()
    assert (
        await scalar(
            app,
            "SELECT json_extract(proposal_json,'$.first_reply_target_at') FROM outbox_messages WHERE json_extract(proposal_json,'$.first_reply_target_at') IS NOT NULL",
        )
        <= received + app.settings.runtime.first_deadline_seconds
    )

    assert (
        await scalar(app, "SELECT MAX(deadline) FROM outbox_messages") > received + 10
    )


async def test_delivery_worker_survives_a_transient_storage_error(
    app: Application, monkeypatch: pytest.MonkeyPatch
) -> None:
    recovered = asyncio.Event()
    attempts = 0

    async def run_once() -> bool:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise sqlite3.OperationalError("temporary storage failure")
        recovered.set()
        return False

    monkeypatch.setattr(app.delivery, "run_once", run_once)
    task = asyncio.create_task(app.delivery.run())
    try:
        await asyncio.wait_for(recovered.wait(), timeout=3)
        assert not task.done()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_terminal_infrastructure_error_notifies_once_and_next_input_recovers(
    app: Application, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = app.agent.process

    async def fail(*args: Any) -> None:
        raise RuntimeError("infrastructure failure before generation")

    monkeypatch.setattr(app.agent, "process", fail)
    await app.telegram.ingest([update(1, "hello")])
    for _ in range(3):
        await app.turn_worker.run_once()
        app.clock.advance(10)
    await app.delivery.run_once()
    assert len(app.sender.sent) == 1
    assert (
        await scalar(
            app, "SELECT COUNT(*) FROM scheduled_jobs WHERE status='dead_letter'"
        )
        == 1
    )
    monkeypatch.setattr(app.agent, "process", original)
    await app.telegram.ingest([update(2, "hello again")])
    await app.drain()
    assert len(app.sender.sent) == 2
    assert (await conversation(app)).preview_reply_count == 1


async def test_terminal_notice_survives_callback_failure_and_restart(
    app: Application, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fail(*args: Any) -> None:
        raise RuntimeError("infrastructure failure before generation")

    async def notice_fails(*args: Any) -> None:
        raise sqlite3.OperationalError("temporary notice write failure")

    monkeypatch.setattr(app.agent, "process", fail)
    app.turn_worker.on_terminal = notice_fails
    await app.telegram.ingest([update(1, "hello")])
    for _ in range(3):
        await app.turn_worker.run_once()
        app.clock.advance(10)
    assert (
        await scalar(app, "SELECT COUNT(*) FROM scheduled_jobs WHERE status='pending'")
        == 1
    )
    assert (
        await scalar(
            app,
            "SELECT json_extract(payload_json,'$.terminal_error') FROM scheduled_jobs",
        )
        == "RuntimeError"
    )
    assert not app.sender.sent
    await app.close()
    reopened = await Application.open(app.settings, app.sender, app.model, app.clock)
    try:
        await reopened.drain()
        assert len(app.sender.sent) == 1
        assert not app.model.calls
        assert (
            await scalar(reopened, "SELECT COUNT(*) FROM messages WHERE pending=1") == 0
        )
        await reopened.jobs.reconcile_turns(app.clock.now())
        await reopened.drain()
        assert len(app.sender.sent) == 1
        await reopened.telegram.ingest([update(2, "hello again")])
        await reopened.drain()
        assert len(app.sender.sent) == 2
        assert (await conversation(reopened)).preview_reply_count == 1
    finally:
        await reopened.close()


async def test_question_and_proactive_permissions_can_be_regranted_independently(
    proactive_app: Application,
) -> None:
    app = proactive_app
    steps = [
        (
            "no questions and do not write first",
            {"no_questions": True, "no_proactive": True},
            False,
            False,
        ),
        (
            "questions are fine now but do not write first",
            {"allow_questions": True},
            True,
            False,
        ),
        ("you can write first too", {"allow_proactive": True}, True, True),
        (
            "no questions again, and no follow-ups",
            {"no_questions": True, "no_proactive": True},
            False,
            False,
        ),
    ]
    for index, (text, flags, questions, proactive) in enumerate(steps, 1):
        app.model.responses["dialogue"].append(
            {
                "parts": ["understood"],
                "language": "en",
                **flags,
            }
        )
        await app.telegram.ingest([update(index, text)])
        await app.drain()
        current = await conversation(app)
        items = await app.memory.repository.items(current.scope)
        permissions = {
            item["predicate"]: item for item in items if item["kind"] == "boundary"
        }
        assert permissions["question_permission"]["value"] is questions
        assert permissions["followup_permission"]["value"] is proactive
        changed = "question_permission" if index == 2 else "followup_permission"
        assert permissions[changed]["sources"][0]["evidence_text"] == text
        assert await scalar(
            app, "SELECT COUNT(*) FROM followups WHERE status='pending'"
        ) == int(proactive)
    history = await app.memory.repository.items(current.scope, history=True)
    assert sum(item["validity_status"] == "superseded" for item in history) == 4


@pytest.mark.parametrize(
    "flags",
    [
        {"no_questions": True, "allow_questions": True},
        {"no_proactive": True, "allow_proactive": True},
    ],
)
def test_contradictory_permission_flags_are_rejected(flags: dict[str, bool]) -> None:
    from pydantic import ValidationError

    from app.modules.dialogue.schemas import DialogueDraft

    with pytest.raises(ValidationError):
        DialogueDraft.model_validate({"parts": ["hello"], "language": "en", **flags})


async def test_terminal_notice_survives_interruption_before_callback_completion(
    app: Application, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fail(*args: Any) -> None:
        raise RuntimeError("infrastructure failure")

    async def interrupted(*args: Any) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(app.agent, "process", fail)
    app.turn_worker.on_terminal = interrupted
    await app.telegram.ingest([update(1, "hello")])
    for _ in range(2):
        await app.turn_worker.run_once()
        app.clock.advance(10)
    with pytest.raises(asyncio.CancelledError):
        await app.turn_worker.run_once()
    assert await scalar(app, "SELECT status FROM scheduled_jobs") == "running"
    await app.close()
    reopened = await Application.open(app.settings, app.sender, app.model, app.clock)
    try:
        await reopened.drain()
        assert len(app.sender.sent) == 1
        assert not app.model.calls
        assert (
            await scalar(reopened, "SELECT COUNT(*) FROM messages WHERE pending=1") == 0
        )
    finally:
        await reopened.close()


async def test_followup_context_distinguishes_ordinary_and_previous_timer_messages(
    proactive_app: Application,
) -> None:
    app = proactive_app
    await app.telegram.ingest([update(1, "I like music")])
    await app.drain()
    app.clock.advance(60)
    await app.drain()
    first = app.model.calls[-1][1]
    assert first["state"]["followup_opportunity"]["previous_messages"] == []
    assert first["original_messages"][-1]["delivery_kind"] == "ordinary_reply"
    app.clock.advance(120)
    await app.drain()
    second = app.model.calls[-1][1]
    previous = second["state"]["followup_opportunity"]["previous_messages"]
    assert len(previous) == 1
    assert previous[0]["message_id"] == second["original_messages"][-1]["id"]
    assert second["original_messages"][-1]["delivery_kind"] == "followup"
    await app.telegram.ingest([update(2, "I am back")])
    await app.drain()
    app.clock.advance(60)
    await app.drain()
    assert (
        app.model.calls[-1][1]["state"]["followup_opportunity"]["previous_messages"]
        == []
    )


def test_timer_request_has_no_new_input_and_keeps_actual_history() -> None:
    from app.modules.dialogue.context import build_request
    from app.modules.dialogue.prompts import rules_for_mode
    from tests.test_v2_dialogue import message, session

    incoming = message(1, text="I like music")
    outgoing = message(2, "assistant", "which music?")
    request = build_request(session(), [incoming, outgoing], [incoming], followup=True)
    assert request.pending_ids == []
    assert [item.content for item in request.CHAT_MESSAGES] == [
        "I like music",
        "which music?",
    ]
    assert "TIMER EVENT" not in rules_for_mode()
    first = rules_for_mode(followup_ordinal=1)
    second = rules_for_mode(followup_ordinal=2)
    assert "FIRST FOLLOW-UP OPPORTUNITY" in first
    assert "SECOND FOLLOW-UP OPPORTUNITY" not in first
    assert "SECOND FOLLOW-UP OPPORTUNITY" in second
    assert "FIRST FOLLOW-UP OPPORTUNITY" not in second


@pytest.mark.parametrize("awaiting_user", [False, True])
async def test_active_turn_requires_first_followup_and_repairs_empty_skip(
    proactive_app: Application,
    awaiting_user: bool,
) -> None:
    app = proactive_app
    app.model.responses["dialogue"].append(
        {
            "parts": ["a little gentleness suits us"],
            "language": "en",
            "awaiting_user": awaiting_user,
        }
    )
    await app.telegram.ingest([update(1, "tease me gently")])
    await app.drain()
    app.model.responses["dialogue"].append(
        {"parts": [], "should_send": False, "language": "en"}
    )
    app.model.responses["dialogue_repair"].append(
        {
            "parts": ["what usually makes you laugh?"],
            "language": "en",
            "awaiting_user": True,
        }
    )
    app.clock.advance(60)
    await app.drain()
    assert [text for _, text in app.sender.sent] == [
        "a little gentleness suits us",
        "what usually makes you laugh?",
    ]
    assert [task for task, _ in app.model.calls] == [
        "dialogue",
        "dialogue",
        "dialogue_repair",
    ]
    assert app.model.calls[-1][1]["state"]["followup_opportunity"]["required"] is True
    assert (await conversation(app)).preview_reply_count == 1
    metrics = await scalar(
        app, "SELECT metrics_json FROM agent_runs ORDER BY rowid DESC LIMIT 1"
    )
    events = json.loads(metrics)["llm_events"]
    assert events[0]["validation_errors"]
    assert not events[1]["validation_errors"]
    app.model.responses["dialogue"].append(
        {"parts": [], "should_send": False, "language": "en"}
    )
    app.clock.advance(120)
    await app.drain()
    assert app.model.calls[-1][1]["state"]["followup_opportunity"]["required"] is False
    assert len(app.sender.sent) == 2


async def test_required_first_followup_survives_restart(
    proactive_app: Application,
) -> None:
    app = proactive_app
    app.model.responses["dialogue"].append(
        {"parts": ["a quiet evening for us"], "language": "en", "awaiting_user": False}
    )
    await app.telegram.ingest([update(1, "a quiet evening")])
    await app.drain()
    await app.close()
    reopened = await Application.open(app.settings, app.sender, app.model, app.clock)
    try:
        app.clock.advance(60)
        await reopened.drain()
        assert (
            app.model.calls[-1][1]["state"]["followup_opportunity"]["required"] is True
        )
        assert len(app.sender.sent) == 2
        assert (await conversation(reopened)).preview_reply_count == 1
    finally:
        await reopened.close()


async def test_unclassified_old_anchor_preserves_optional_followup(
    proactive_app: Application,
) -> None:
    app = proactive_app
    await app.telegram.ingest([update(1, "a story")])
    await app.drain()

    async def remove_new_metadata(tx: Any) -> None:
        await tx.execute(
            "UPDATE outbox_messages SET proposal_json=json_remove(proposal_json,'$.awaiting_user')"
        )

    await app.database.write(remove_new_metadata)
    app.model.responses["dialogue"].append(
        {"parts": [], "should_send": False, "language": "en"}
    )
    app.clock.advance(60)
    await app.drain()
    assert app.model.calls[-1][1]["state"]["followup_opportunity"]["required"] is False
    assert len(app.sender.sent) == 1


async def test_new_boundary_cancels_required_first_followup(
    proactive_app: Application,
) -> None:
    app = proactive_app
    await app.telegram.ingest([update(1, "tease me gently")])
    await app.drain()
    app.model.responses["dialogue"].append(
        {
            "parts": ["rest well"],
            "language": "en",
            "closing": True,
            "no_proactive": True,
        }
    )
    await app.telegram.ingest([update(2, "im off to sleep, dont message first")])
    await app.drain()
    app.clock.advance(240)
    await app.drain()
    assert len(app.model.calls) == 2
    assert len(app.sender.sent) == 2
    assert (
        await scalar(app, "SELECT status FROM followups ORDER BY rowid LIMIT 1")
        == "cancelled"
    )


async def test_required_followup_failure_is_explicit_and_next_input_still_answers(
    proactive_app: Application,
) -> None:
    app = proactive_app
    await app.telegram.ingest([update(1, "tease me gently")])
    await app.drain()
    skipped = {"parts": [], "should_send": False, "language": "en"}
    app.model.responses["dialogue"].append(skipped)
    app.model.responses["dialogue_repair"].append(skipped)
    app.clock.advance(60)
    await app.drain()
    assert (
        await scalar(app, "SELECT reason FROM followups ORDER BY rowid LIMIT 1")
        == "REQUIRED_RESPONSE_FAILED"
    )
    assert (
        await scalar(
            app, "SELECT error_code FROM agent_runs ORDER BY rowid DESC LIMIT 1"
        )
        is not None
    )
    assert len(app.sender.sent) == 1
    await app.telegram.ingest([update(2, "hello, still here")])
    await app.drain()
    assert len(app.sender.sent) == 2
    assert (await conversation(app)).preview_reply_count == 2


async def test_requested_pause_survives_restart_and_next_story_resumes_contact(
    proactive_app: Application,
) -> None:
    app = proactive_app
    app.model.responses["dialogue"].append(
        {
            "parts": ["go on"],
            "language": "en",
            "pause_requested": True,
            "awaiting_user": True,
        }
    )
    await app.telegram.ingest([update(1, "wait, let me finish before you ask")])
    await app.drain()
    proposal = json.loads(
        await scalar(app, "SELECT proposal_json FROM outbox_messages")
    )
    assert proposal["pause_requested"] is True
    assert proposal["followup_allowed"] is False
    assert await scalar(app, "SELECT COUNT(*) FROM followups") == 0
    await app.close()
    reopened = await Application.open(app.settings, app.sender, app.model, app.clock)
    try:
        app.clock.advance(240)
        await reopened.drain()
        assert len(app.model.calls) == 1
        app.model.responses["dialogue"].append(
            {
                "parts": ["oh no, the manager was right there", "what did he say?"],
                "language": "en",
                "awaiting_user": True,
            }
        )
        await reopened.telegram.ingest([update(2, "then the manager walked in")])
        await reopened.drain()
        assert [text for _, text in app.sender.sent] == [
            "go on",
            "oh no, the manager was right there",
            "what did he say?",
        ]
        assert await scalar(reopened, "SELECT COUNT(*) FROM followups") == 1
        app.model.responses["dialogue"].append(
            {"parts": ["id have hidden behind the counter"], "language": "en"}
        )
        app.clock.advance(60)
        await reopened.drain()
        assert (
            app.model.calls[-1][1]["state"]["followup_opportunity"]["required"] is True
        )
        assert app.sender.sent[-1][1] == "id have hidden behind the counter"
        assert (await conversation(reopened)).preview_reply_count == 2
    finally:
        await reopened.close()


async def test_first_timeout_retry_can_finish_after_first_response_target(
    app: Application,
) -> None:
    app.agent.dialogue.settings = app.settings.runtime.model_copy(
        update={
            "first_deadline_seconds": 0.15,
            "llm_attempt_seconds": 0.2,
            "turn_deadline_seconds": 2,
        }
    )
    original = app.model.complete
    budgets: list[float] = []

    async def delayed(*args: Any, **kwargs: Any) -> Any:
        budgets.append(args[-1])
        if len(budgets) == 1:
            await asyncio.sleep(0.01)
            raise TimeoutError()
        await asyncio.sleep(0.05)
        return await original(*args, **kwargs)

    app.model.complete = delayed
    app.model.responses["dialogue_repair"].append(
        {"parts": ["hey, glad you came by"], "language": "en"}
    )
    await app.telegram.ingest([update(1, "hey, an evening to myself at last")])
    await app.drain()
    assert [text for _, text in app.sender.sent] == ["hey, glad you came by"]
    assert 0 < budgets[0] < 0.15
    assert 0.2 < budgets[1] < 2
    assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"
    assert (await conversation(app)).preview_reply_count == 1
    metrics = json.loads(await scalar(app, "SELECT metrics_json FROM agent_runs"))
    assert metrics["elapsed_seconds"] > 0.15
    assert metrics["llm_calls"] == 2
    assert metrics["llm_events"][0]["validation_errors"] == ["DEADLINE_EXCEEDED"]
    assert not metrics["llm_events"][1]["validation_errors"]


@pytest.mark.parametrize(
    ("commands", "reopened"), [(["/stop", "/start"], True), (["/privacy"], False)]
)
async def test_control_reentry_closes_old_question_without_losing_history(
    app: Application,
    commands: list[str],
    reopened: bool,
) -> None:
    app.model.responses["dialogue"].append(
        {"parts": ["what music do you like?"], "language": "en", "awaiting_user": True}
    )
    await app.telegram.ingest([update(1, "im Anya, in Riga")])
    await app.drain()
    for number, command in enumerate(commands, 2):
        await app.telegram.ingest([update(number, command)])
        await app.drain()
    app.model.responses["dialogue"].append(
        {"parts": ["how has your day been?"], "language": "en", "awaiting_user": True}
    )
    await app.telegram.ingest([update(10, "hi")])
    await app.drain()
    data = app.model.calls[-1][1]
    assert data["state"]["exchange_reopened"] is reopened
    assert data["state"]["previous_assistant_turn"] == (
        [] if reopened else ["what music do you like?"]
    )
    assert [m["content"] for m in data["CHAT_MESSAGES"]] == [
        "im Anya, in Riga",
        "what music do you like?",
        "hi",
    ]
    current = await conversation(app)
    assert current.preview_reply_count == 2
    assert current.epoch == 1
    await app.telegram.ingest([update(11, "mostly errands, now im home")])
    await app.drain()
    state = app.model.calls[-1][1]["state"]
    assert state["exchange_reopened"] is False
    assert state["previous_assistant_turn"] == ["how has your day been?"]


async def test_timeout_recovery_uses_remaining_budget_and_preserves_bubbles(
    app: Application,
) -> None:
    app.agent.dialogue.settings = app.settings.runtime.model_copy(
        update={"llm_attempt_seconds": 0.05, "turn_deadline_seconds": 2}
    )
    original = app.model.complete
    budgets: list[float] = []

    async def delayed(*args: Any, **kwargs: Any) -> Any:
        budgets.append(args[-1])
        async with asyncio.timeout(args[-1]):
            if len(budgets) == 1:
                raise TimeoutError()
            await asyncio.sleep(0.1)
            return await original(*args, **kwargs)

    app.model.complete = delayed
    app.model.responses["dialogue_repair"].append(
        {
            "parts": ["glad you came by"],
            "continuation": "how was your day?",
            "language": "en",
        }
    )
    await app.telegram.ingest([update(1, "hello")])
    await app.drain()
    assert [text for _, text in app.sender.sent] == [
        "glad you came by",
        "how was your day?",
    ]
    assert budgets[0] == pytest.approx(0.05)
    assert 0.1 < budgets[1] < 2
    metrics = json.loads(await scalar(app, "SELECT metrics_json FROM agent_runs"))
    assert metrics["llm_calls"] == 2
    assert metrics["llm_events"][1]["budget_seconds"] == budgets[1]
    assert not metrics["llm_events"][1]["validation_errors"]
    assert (await conversation(app)).preview_reply_count == 1


async def test_retry_timeout_does_not_keep_the_previous_validation_error(
    app: Application,
) -> None:
    app.model.responses["dialogue"].append(
        {"parts": [], "should_send": False, "language": "en"}
    )
    app.model.responses["dialogue_repair"].append(TimeoutError())
    await app.telegram.ingest([update(1, "hello")])
    await app.drain()
    metrics = json.loads(await scalar(app, "SELECT metrics_json FROM agent_runs"))
    assert metrics["llm_events"][0]["validation_errors"] == [
        "should_send:literal_error"
    ]
    assert metrics["llm_events"][1]["validation_errors"] == ["DEADLINE_EXCEEDED"]
    assert app.model.calls[1][1]["REPAIR_ERRORS"] == ["should_send:literal_error"]
    assert await scalar(app, "SELECT error_code FROM agent_runs") == "DEADLINE_EXCEEDED"
    assert len(app.sender.sent) == 1


@pytest.mark.parametrize("failure", ["timeout", "empty"])
async def test_failure_notice_survives_transient_send_and_restart_without_regeneration(
    app: Application,
    failure: str,
) -> None:
    from app.modules.delivery.entities import SendResult

    for task in ("dialogue", "dialogue_repair"):
        app.model.responses[task].append(
            TimeoutError()
            if failure == "timeout"
            else {"parts": [], "should_send": False, "language": "en"}
        )
    await app.telegram.ingest([update(1, "hello")])
    await app.turn_worker.run_once()
    assert await scalar(app, "SELECT kind FROM outbox_messages") == "service"
    assert await scalar(app, "SELECT status FROM outbox_messages") == "ready"
    app.sender.results.append(
        SendResult("retryable_failure", error_code="TEMPORARY", retry_after=2)
    )
    await app.delivery.run_once()
    assert await scalar(app, "SELECT status FROM outbox_messages") == "ready"
    assert len(app.model.calls) == 2
    await app.close()
    reopened = await Application.open(app.settings, app.sender, app.model, app.clock)
    try:
        app.clock.advance(3)
        await reopened.drain()
        assert (
            await scalar(reopened, "SELECT status FROM outbox_messages") == "delivered"
        )
        assert (
            await scalar(reopened, "SELECT COUNT(*) FROM messages WHERE role='service'")
            == 1
        )
        assert len(app.model.calls) == 2
        assert len(app.sender.sent) == 2
        await reopened.drain()
        assert len(app.sender.sent) == 2
        await reopened.telegram.ingest([update(2, "hello again")])
        await reopened.drain()
        assert (await conversation(reopened)).preview_reply_count == 1
        assert len(app.sender.sent) == 3
    finally:
        await reopened.close()


async def test_overall_deadline_produces_notice_and_next_input_is_not_stuck(
    app: Application,
) -> None:
    app.agent.dialogue.settings = app.settings.runtime.model_copy(
        update={"turn_deadline_seconds": 0.1}
    )
    app.model.release = asyncio.Event()
    await app.telegram.ingest([update(1, "hello")])
    await app.drain()
    assert await scalar(app, "SELECT status FROM agent_runs") == "fallback"
    assert await scalar(app, "SELECT error_code FROM agent_runs") == "DEADLINE_EXCEEDED"
    assert await scalar(app, "SELECT kind FROM outbox_messages") == "service"
    assert len(app.sender.sent) == 1
    assert not app.agent.active
    app.model.release = None
    app.agent.dialogue.settings = app.settings.runtime
    await app.telegram.ingest([update(2, "still here")])
    await app.drain()
    assert len(app.sender.sent) == 2
    assert (await conversation(app)).preview_reply_count == 1


@pytest.mark.parametrize(
    ("text", "paused"),
    [
        ("im tired, just keep me company gently", False),
        ("побудь со мной спокойно, без шуток", False),
        ("please wait, let me finish before you say anything else", True),
    ],
)
async def test_contact_permission_is_independent_of_a_quiet_reply(
    proactive_app: Application, text: str, paused: bool
) -> None:
    app = proactive_app
    app.model.responses["dialogue"].append(
        {
            "conversation_action": "yield",
            "parts": ["we can slow down a little"],
            "language": "en",
            "pause_requested": paused,
        }
    )
    await app.telegram.ingest([update(1, text)])
    await app.drain()
    proposal = json.loads(
        await scalar(app, "SELECT proposal_json FROM outbox_messages LIMIT 1")
    )
    assert proposal["followup_allowed"] is not paused
    app.clock.advance(60)
    await app.drain()
    assert len(app.model.calls) == (1 if paused else 2)
    assert len(app.sender.sent) == (1 if paused else 2)
    assert (await conversation(app)).preview_reply_count == 1


async def test_later_question_uses_full_history_and_keeps_a_followup(
    proactive_app: Application,
) -> None:
    app = proactive_app
    for number in range(1, 11):
        await app.telegram.ingest([update(number, f"a new detail {number}")])
        await app.drain()
    app.model.responses["dialogue"].append(
        {
            "conversation_action": "ask",
            "parts": ["i took the long way back"],
            "continuation": "where do you go when you want a little quiet?",
            "language": "en",
            "awaiting_user": True,
        }
    )
    await app.telegram.ingest([update(11, "how was your walk?")])
    await app.drain()
    assert [text for _, text in app.sender.sent[-2:]] == [
        "i took the long way back",
        "where do you go when you want a little quiet?",
    ]
    history = app.model.calls[-1][1]["CHAT_MESSAGES"]
    assert history[0]["content"] == "a new detail 1"
    assert history[-1]["content"] == "how was your walk?"
    assert len(history) == 21
    assert len(app.model.calls) == 11
    app.model.responses["dialogue"].append(
        {"parts": ["i like taking the smaller streets"], "language": "en"}
    )
    app.clock.advance(60)
    await app.drain()
    assert app.model.calls[-1][1]["state"]["followup_opportunity"]["required"] is True
    assert app.sender.sent[-1][1] == "i like taking the smaller streets"
    assert (await conversation(app)).preview_reply_count == 11


async def test_ordinary_silence_is_repaired_and_future_contact_stays_blocked(
    proactive_app: Application,
) -> None:
    app = proactive_app
    app.model.responses["dialogue"].append(
        {
            "conversation_action": "yield",
            "parts": [],
            "language": "en",
            "should_send": False,
            "pause_requested": True,
            "no_proactive": True,
        }
    )
    app.model.responses["dialogue_repair"].append(
        {
            "parts": ["so Kyoto is still just a plan"],
            "language": "en",
            "no_questions": True,
            "no_proactive": True,
        }
    )
    incoming = (
        "Kyoto is only a plan, not booked. no questions or messages after a pause"
    )
    await app.telegram.ingest([update(1, incoming)])
    await app.drain()
    assert len(app.model.calls) == 2
    assert app.sender.sent == [(1, "so Kyoto is still just a plan")]
    assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"
    assert await scalar(app, "SELECT status FROM turns") == "delivered"
    assert (
        await scalar(app, "SELECT COUNT(*) FROM outbox_messages WHERE kind='service'")
        == 0
    )
    assert (
        await scalar(
            app,
            "SELECT COUNT(*) FROM memory_items WHERE kind='boundary' AND predicate='followup_permission' AND value_json='false' AND validity_status='current'",
        )
        == 1
    )
    app.clock.advance(200)
    await app.drain()
    assert len(app.model.calls) == 2
    app.model.responses["dialogue"].append(
        {
            "parts": ["good to have you back"],
            "language": "en",
            "allow_questions": True,
            "allow_proactive": True,
        }
    )
    await app.telegram.ingest(
        [update(2, "im back, questions and messages are welcome again")]
    )
    await app.drain()
    assert app.model.calls[-1][1]["CHAT_MESSAGES"][0]["content"] == incoming
    app.clock.advance(60)
    await app.drain()
    assert len(app.sender.sent) == 3
    assert (await conversation(app)).preview_reply_count == 2
