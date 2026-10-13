import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.application import Application
from app.modules.agent.entities import SchemaT
from app.modules.delivery.repository import DeliveryRepository
from app.modules.delivery.timing import message_delay
from app.shared.config import RuntimeSettings, load_settings
from app.shared.storage import Database, Transaction, fetchall, fetchone
from tests.helpers import (
    PLATFORM_INVITATION_EN,
    FakeSender,
    conversation,
    scalar,
    update,
)


@pytest.fixture
def paced_app(app: Application) -> Application:
    runtime = app.settings.runtime.model_copy(
        update={
            "reply_pacing_enabled": True,
            "reply_pause_min_seconds": 3,
            "reply_pause_max_seconds": 3,
            "reply_delay_jitter_ratio": 0,
            "first_deadline_seconds": 30,
        }
    )
    app.settings = app.settings.model_copy(update={"runtime": runtime})
    app.agent.settings = runtime
    app.telegram.settings = runtime
    app.followups.settings = runtime
    return app


async def prepare(
    app: Application, lengths: tuple[int, ...], first: bool = False
) -> list[dict[str, Any]]:
    await app.telegram.ingest([update(1, "Let’s chat")])
    current = await conversation(app)
    if not first:

        async def mark_existing(tx: Transaction) -> None:
            await tx.execute(
                "UPDATE conversations SET first_reply_at=? WHERE id=?",
                (app.clock.now() - 1, current.id),
            )

        await app.database.write(mark_existing)
    proposal = {"reply": "a" * lengths[0], "dialogue_act": "answer"}
    if len(lengths) > 1:
        proposal["continuation"] = "b" * lengths[1]
    app.model.responses["generate"].append(proposal)
    await app.turn_worker.run_once()
    async with app.database.read() as tx:
        return [
            dict(row)
            for row in await fetchall(
                tx, "SELECT * FROM outbox_messages ORDER BY rowid"
            )
        ]


@pytest.mark.parametrize(
    "length,expected",
    [(0, 5), (13, 3 + 13 / 3), (60, 23), (140, 3 + 140 / 3), (4000, 60)],
)
def test_message_length_controls_delay(length: int, expected: float) -> None:
    settings = RuntimeSettings(
        reply_pause_min_seconds=3,
        reply_pause_max_seconds=3,
        reply_delay_jitter_ratio=0,
    )
    assert message_delay("я" * length, settings) == pytest.approx(expected)


@pytest.mark.parametrize("upper", [False, True])
def test_jitter_is_applied_before_clamping(
    monkeypatch: pytest.MonkeyPatch, upper: bool
) -> None:
    monkeypatch.setattr(
        "app.modules.delivery.timing.random.uniform",
        lambda low, high: high if upper else low,
    )
    expected = min(60, (4 + 140 / 3) * 1.2) if upper else (2 + 140 / 3) * 0.8
    assert message_delay("a" * 140, RuntimeSettings()) == pytest.approx(expected)


@pytest.mark.parametrize(
    "values",
    [
        {"reply_typing_chars_per_second": 0},
        {"reply_typing_chars_per_second": float("inf")},
        {"reply_delay_jitter_ratio": 1.1},
        {"reply_delay_jitter_ratio": -0.1},
        {"reply_delay_min_seconds": 61, "reply_delay_max_seconds": 60},
        {"reply_delay_max_seconds": float("nan")},
        {"first_reply_max_seconds": 0},
    ],
)
def test_invalid_timing_settings_are_rejected(values: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        RuntimeSettings.model_validate(values)


def test_all_timing_settings_are_loaded_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        "REPLY_PACING_ENABLED": "true",
        "REPLY_PAUSE_MIN_SECONDS": "3.5",
        "REPLY_PAUSE_MAX_SECONDS": "4.5",
        "REPLY_TYPING_CHARS_PER_SECOND": "6",
        "REPLY_DELAY_JITTER_RATIO": "0.1",
        "REPLY_DELAY_MIN_SECONDS": "1",
        "REPLY_DELAY_MAX_SECONDS": "90",
        "FIRST_REPLY_MAX_SECONDS": "7",
    }
    for key, value in values.items():
        monkeypatch.setenv("BEHAPPY_RUNTIME__" + key, value)
    runtime = load_settings().runtime
    for key, value in values.items():
        expected = True if value == "true" else float(value)
        assert getattr(runtime, key.lower()) == expected


@pytest.mark.parametrize("length", [13, 60, 140, 4000])
async def test_single_message_is_not_sent_before_its_length_based_due_time(
    paced_app: Application, length: int
) -> None:
    app = paced_app
    started = app.clock.now()
    rows = await prepare(app, (length,))
    due = rows[0]["created_at"]
    assert due - started == pytest.approx(min(60, 3 + length / 3))
    await app.delivery.send(rows[0]["id"])
    assert not app.sender.sent
    app.clock.advance(due - app.clock.now() - 0.01)
    await app.delivery.run_once()
    assert not app.sender.sent
    app.clock.advance(0.01)
    await app.delivery.run_once()
    assert app.sender.sent == [(1, "a" * length)]


@pytest.mark.parametrize("first", [False, True])
@pytest.mark.parametrize(
    "length,elapsed", [(13, 0), (140, 0), (140, 6), (13, 9), (13, 18), (140, 70)]
)
async def test_generation_counts_toward_pacing_without_delaying_overdue_answers(
    paced_app: Application,
    monkeypatch: pytest.MonkeyPatch,
    length: int,
    elapsed: float,
    first: bool,
) -> None:
    app = paced_app
    started = app.clock.now()
    complete = app.model.complete

    async def slow(
        task: str,
        rules: str,
        data: dict[str, Any],
        schema: type[SchemaT],
        deadline_seconds: float,
    ) -> SchemaT:
        if task == "generate":
            app.clock.advance(elapsed)
        return await complete(task, rules, data, schema, deadline_seconds)

    monkeypatch.setattr(app.model, "complete", slow)
    rows = await prepare(app, (length,), first=first)
    target = min(10 if first else 60, max(5, 3 + length / 3))
    expected = max(elapsed, target)
    assert rows[0]["created_at"] - started == pytest.approx(expected)
    assert rows[0]["message_delay_seconds"] == pytest.approx(max(0, target - elapsed))


async def test_ten_second_cap_only_applies_until_first_reply_is_delivered(
    paced_app: Application,
) -> None:
    app = paced_app
    started = app.clock.now()
    first = (await prepare(app, (140,), first=True))[0]
    assert first["created_at"] - started == 10
    app.clock.advance(10)
    await app.delivery.run_once()
    assert (await conversation(app)).first_reply_at == app.clock.now()
    app.model.responses["generate"].append(
        {"reply": "c" * 140, "dialogue_act": "answer"}
    )
    await app.telegram.ingest([update(2, "Tell me more")])
    received_at = app.clock.now()
    await app.turn_worker.run_once()
    due = await scalar(
        app, "SELECT created_at FROM outbox_messages ORDER BY rowid DESC LIMIT 1"
    )
    assert due - received_at == pytest.approx(3 + 140 / 3)


async def test_queue_time_does_not_turn_pacing_target_into_generation_deadline(
    paced_app: Application, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = paced_app
    await app.telegram.ingest([update(1, "Hello")])
    app.clock.advance(4)
    timeouts = []
    complete = app.model.complete

    async def record(
        task: str,
        rules: str,
        data: dict[str, Any],
        schema: type[SchemaT],
        deadline_seconds: float,
    ) -> SchemaT:
        timeouts.append(deadline_seconds)
        return await complete(task, rules, data, schema, deadline_seconds)

    monkeypatch.setattr(app.model, "complete", record)
    await app.turn_worker.run_once()
    assert timeouts and all(
        6 < timeout <= app.agent.settings.llm_attempt_seconds for timeout in timeouts
    )


async def test_each_part_waits_from_actual_previous_delivery_and_restart_keeps_delay(
    paced_app: Application, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = paced_app
    first, second = await prepare(app, (13, 140), first=True)
    app.clock.advance(9)
    await app.delivery.send(first["id"])
    due = await scalar(
        app, "SELECT created_at FROM outbox_messages WHERE id=?", (second["id"],)
    )
    assert due - app.clock.now() == pytest.approx(3 + 140 / 3)
    monkeypatch.setattr(
        "app.modules.delivery.timing.message_delay",
        lambda *args: pytest.fail("Persisted delay must not be redrawn"),
    )
    await app.close()
    restarted = await Application.open(app.settings, app.sender, app.model, app.clock)
    try:
        await restarted.delivery.run_once()
        assert len(app.sender.sent) == 1
        assert (
            await scalar(
                restarted,
                "SELECT created_at FROM outbox_messages WHERE id=?",
                (second["id"],),
            )
            == due
        )
        app.clock.advance(due - app.clock.now())
        await restarted.delivery.run_once()
        assert app.sender.sent == [(1, "a" * 13), (1, "b" * 140)]
    finally:
        await restarted.close()


async def test_transport_retry_does_not_repeat_typing_interval(
    paced_app: Application,
) -> None:
    from app.modules.delivery.entities import SendResult

    app = paced_app
    first, second = await prepare(app, (60, 140))
    app.clock.advance(first["created_at"] - app.clock.now())
    app.sender.results.append(SendResult("retryable_failure", retry_after=1))
    await app.delivery.run_once()
    retry_due = await scalar(
        app, "SELECT created_at FROM outbox_messages WHERE id=?", (first["id"],)
    )
    assert retry_due - app.clock.now() == 1
    app.clock.advance(1)
    await app.delivery.run_once()
    assert len(app.sender.sent) == 2
    due = await scalar(
        app, "SELECT created_at FROM outbox_messages WHERE id=?", (second["id"],)
    )
    assert due - app.clock.now() == pytest.approx(3 + 140 / 3)


@pytest.mark.parametrize("action", ["input", "edit", "stop", "kill"])
async def test_pending_first_and_second_parts_are_cancelled(
    paced_app: Application, action: str
) -> None:
    app = paced_app
    rows = await prepare(app, (60, 140))
    if action == "kill":
        await app._control({"command": "kill-switch", "enabled": True})
    else:
        await app.telegram.ingest(
            [
                update(
                    2,
                    "/stop" if action == "stop" else "Another topic",
                    edited=action == "edit",
                    message_id=1 if action == "edit" else 2,
                )
            ]
        )
    app.clock.advance(120)
    for row in rows:
        await app.delivery.send(row["id"])
        assert (
            await scalar(
                app, "SELECT status FROM outbox_messages WHERE id=?", (row["id"],)
            )
            == "cancelled"
        )
    assert not app.sender.sent


async def test_forget_removes_waiting_parts(paced_app: Application) -> None:
    app = paced_app
    await prepare(app, (60, 140))
    await app.telegram.ingest([update(2, "/forget")])
    await app.drain()
    app.clock.advance(120)
    await app.delivery.run_once()
    assert all(text not in {"a" * 60, "b" * 140} for _, text in app.sender.sent)
    assert (
        await scalar(app, "SELECT COUNT(*) FROM outbox_messages WHERE kind='reply'")
        == 0
    )


async def test_typing_runs_during_first_part_wait_and_stops_on_send(
    paced_app: Application,
) -> None:
    app = paced_app
    events = []

    class TypingSender(FakeSender):
        @asynccontextmanager
        async def typing(self, chat_id: int):
            events.append("start")
            try:
                yield
            finally:
                events.append("stop")

    sender = TypingSender()
    app.sender = sender
    app.delivery.sender = sender
    rows = await prepare(app, (60,))
    await app.delivery.run_once()
    await asyncio.sleep(0)
    assert events == ["start"]
    app.clock.advance(rows[0]["created_at"] - app.clock.now())
    await app.delivery.run_once()
    assert events == ["start", "stop"]
    assert sender.sent == [(1, "a" * 60)]


async def test_followup_is_skipped_when_length_delay_exceeds_remaining_ttl(
    paced_app: Application,
) -> None:
    from tests.test_followups import enable

    app = paced_app
    enable(app)
    await app.telegram.ingest([update(1, "Quiet music is nice")])
    app.model.responses["generate"].append(
        {
            "reply": "Quiet music fits",
            "dialogue_act": "opinion",
            "followup_allowed": True,
        }
    )
    await app.turn_worker.run_once()
    due = await scalar(app, "SELECT created_at FROM outbox_messages")
    app.clock.advance(due - app.clock.now())
    await app.delivery.run_once()
    expires = await scalar(app, "SELECT expires_at FROM followups")
    app.clock.advance(expires - app.clock.now() - 5)
    app.model.responses["followup"].append(
        {
            "reply": "A quiet melody feels better than noise",
            "dialogue_act": "opinion",
            "should_send": True,
        }
    )
    await app.turn_worker.run_once()
    assert len(app.sender.sent) == 1
    assert (
        await scalar(
            app,
            "SELECT COUNT(*) FROM outbox_messages WHERE json_extract(proposal_json,'$.origin')='followup'",
        )
        == 0
    )


async def test_delay_migration_preserves_old_parts_and_followup_foreign_keys(
    tmp_path: Path,
) -> None:
    old = tmp_path / "old-migrations"

    def copy_migrations() -> None:
        old.mkdir()
        for path in Path("app/migrations").glob("*.sql"):
            if path.name != "010_message_delay.sql":
                (old / path.name).write_bytes(path.read_bytes())

    await asyncio.to_thread(copy_migrations)
    database = Database(tmp_path / "old.db")
    await database.open(old)

    async def seed(tx: Transaction) -> None:
        await tx.execute(
            "INSERT INTO conversations(id,tenant_id,channel_account_id,external_user_id,chat_id,persona_id,epoch,input_revision,last_activity_at,created_at) VALUES('scope','demo','telegram-demo',1,1,'marko',1,1,100,100)"
        )
        await tx.execute(
            "INSERT INTO turns VALUES('turn','scope',1,'prepared',1,1,1,100)"
        )
        for identifier, predecessor, delay in [
            ("first", None, 0),
            ("second", "first", 16),
        ]:
            await tx.execute(
                "INSERT INTO outbox_messages(id,action_key,scope_id,epoch,input_revision,control_revision,memory_revision,turn_id,kind,text,proposal_json,deadline,created_at,predecessor_id,delay_seconds) VALUES(?,?,'scope',1,1,0,0,'turn','reply','text','{}',200,100,?,?)",
                (identifier, identifier, predecessor, delay),
            )
        await tx.execute(
            "INSERT INTO followups(id,scope_id,epoch,input_revision,control_revision,anchor_id,turn_id,run_at,expires_at) VALUES('followup','scope',1,1,0,'second','turn',130,220)"
        )

    try:
        await database.write(seed)
    finally:
        await database.close()
    await database.open(Path("app/migrations"))
    try:
        async with database.read() as tx:
            assert await fetchall(tx, "PRAGMA foreign_key_check") == []
            assert (await fetchone(tx, "PRAGMA integrity_check"))[0] == "ok"
            rows = await fetchall(
                tx, "SELECT message_delay_seconds FROM outbox_messages"
            )
            assert len(rows) == 2 and all(row[0] is None for row in rows)
            assert (await fetchone(tx, "SELECT anchor_id FROM followups"))[
                0
            ] == "second"
        repository = DeliveryRepository(database)

        async def deliver(tx: Transaction) -> None:
            message = await repository.gate(tx, "first", 100, True, False)
            assert message is not None
            await repository.outcome(tx, message, "delivered", 100, 123, None, 0)

        await database.write(deliver)
        async with database.read() as tx:
            row = await fetchone(
                tx, "SELECT created_at,deadline FROM outbox_messages WHERE id='second'"
            )
            assert tuple(row) == (116, 200)
    finally:
        await database.close()


async def test_platform_transition_uses_length_of_final_text_with_link(
    paced_app: Application,
) -> None:
    app = paced_app
    first = (await prepare(app, (13,), first=True))[0]
    app.clock.advance(first["created_at"] - app.clock.now())
    await app.delivery.run_once()
    settings = app.commerce.settings.model_copy(update={"preview_enabled": True})
    app.commerce.settings = settings

    async def eligible(tx: Transaction) -> None:
        await tx.execute("UPDATE conversations SET preview_reply_count=12")

    await app.database.write(eligible)
    app.model.responses["generate"].append(
        {
            "reply": "a" * 60,
            "dialogue_act": "answer",
            "platform_transition": True,
            "platform_notice": PLATFORM_INVITATION_EN,
        }
    )
    await app.telegram.ingest([update(2, "What’s new with you?")])
    received_at = app.clock.now()
    await app.turn_worker.run_once()
    async with app.database.read() as tx:
        row = await fetchone(
            tx,
            "SELECT text,created_at FROM outbox_messages WHERE json_extract(proposal_json,'$.origin')='platform_redirect'",
        )
    assert row is not None
    assert row["text"].endswith("https://www.behappy2day.com/")
    assert row["created_at"] - received_at == pytest.approx(
        min(60, 3 + len(row["text"]) / 3)
    )
