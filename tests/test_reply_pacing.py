import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from app.application import Application
from app.shared.storage import Database, Transaction, fetchall
from tests.helpers import FakeSender, ScriptedModel, scalar, update
from tests.test_reply_parts import prepare_pair


class TypingFakeSender(FakeSender):
    def __init__(self, fails: bool = False) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.finished = asyncio.Event()
        self.active = False
        self.fails = fails
        self.typing_calls = 0

    @asynccontextmanager
    async def typing(self, chat_id: int) -> AsyncIterator[None]:
        self.typing_calls += 1
        self.started.set()
        if self.fails:
            raise RuntimeError("typing unavailable")
        self.active = True
        try:
            yield
        finally:
            self.active = False
            self.finished.set()


async def start_waiting(
    app: Application, fails: bool = False
) -> tuple[str, TypingFakeSender]:
    first, second = await prepare_pair(app)
    await app.delivery.send(first)
    sender = TypingFakeSender(fails)
    sender.next_id = app.sender.next_id
    app.delivery.sender = sender
    await app.delivery.run_once()
    await asyncio.wait_for(sender.started.wait(), 1)
    return second, sender


async def test_typing_is_visible_between_parts_and_stops_before_delivery(
    app: Application,
) -> None:
    second, sender = await start_waiting(app)
    assert sender.active and not sender.sent
    await app.delivery.run_once()
    assert sender.typing_calls == 1
    assert (
        await scalar(app, "SELECT status FROM outbox_messages WHERE id=?", (second,))
        == "ready"
    )
    app.clock.advance(4)
    await app.delivery.run_once()
    assert sender.finished.is_set() and not sender.active
    assert sender.sent == [(1, "Что теперь на ужин?")]


@pytest.mark.parametrize(
    "event", ["message", "edit", "stop", "forget", "kill", "deadline"]
)
async def test_waiting_typing_stops_when_the_reply_is_invalidated(
    app: Application, event: str
) -> None:
    second, sender = await start_waiting(app)
    if event in {"message", "edit"}:
        await app.telegram.ingest(
            [
                update(
                    2,
                    "Ужин спасла.",
                    edited=event == "edit",
                    message_id=1 if event == "edit" else 2,
                )
            ]
        )
    elif event in {"stop", "forget"}:
        await app.telegram.ingest([update(2, "/" + event)])
    elif event == "kill":
        await app.delivery_repository.kill_switch(True)
    else:

        async def expire(tx: Transaction) -> None:
            await tx.execute(
                "UPDATE outbox_messages SET deadline=? WHERE id=?",
                (app.clock.now(), second),
            )

        await app.database.write(expire)
    await app.delivery.run_once()
    await asyncio.wait_for(sender.finished.wait(), 1)
    assert not sender.active
    assert "Что теперь на ужин?" not in [text for _, text in sender.sent]


async def test_typing_error_does_not_block_or_retry_text(app: Application) -> None:
    _, sender = await start_waiting(app, fails=True)
    await app.delivery.run_once()
    assert sender.typing_calls == 1
    app.clock.advance(4)
    await app.delivery.run_once()
    assert sender.sent == [(1, "Что теперь на ужин?")]


async def test_waiting_typing_does_not_hold_another_users_delivery(
    app: Application,
) -> None:
    _, sender = await start_waiting(app)
    await app.telegram.ingest([update(2, "Hello", user=2)])
    await app.turn_worker.run_once()
    await asyncio.wait_for(app.delivery.run_once(), 1)
    assert sender.active
    assert len(sender.sent) == 1 and sender.sent[0][0] == 2


async def test_restart_resumes_typing_for_durable_wait_without_duplicate_first_part(
    app: Application,
) -> None:
    second, sender = await start_waiting(app)
    due = await scalar(
        app, "SELECT created_at FROM outbox_messages WHERE id=?", (second,)
    )
    await app.close()
    assert sender.finished.is_set()
    restarted_sender = TypingFakeSender()
    restarted_sender.next_id = sender.next_id
    restarted = await Application.open(
        app.settings, restarted_sender, app.model, app.clock
    )
    try:
        await restarted.delivery.run_once()
        await asyncio.wait_for(restarted_sender.started.wait(), 1)
        assert restarted_sender.active and not restarted_sender.sent
        app.clock.advance(due - app.clock.now())
        await restarted.delivery.run_once()
        assert restarted_sender.sent == [(1, "Что теперь на ужин?")]
        assert restarted_sender.finished.is_set()
    finally:
        await restarted.close()


async def test_migration_preserves_parent_child_delivery_and_extends_pause(
    app: Application, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_migrations = tmp_path / "old-migrations"
    old_migrations.mkdir()
    paths = await asyncio.to_thread(lambda: list(Path("app/migrations").glob("*.sql")))
    for path in paths:
        if path.name not in {"006_reply_pacing.sql", "008_reply_pacing.sql"}:
            (old_migrations / path.name).write_text(path.read_text())
    original = Database.open

    async def open_old(database: Database, migrations: Path) -> None:
        await original(database, old_migrations)

    settings = app.settings.model_copy(
        update={
            "storage": app.settings.storage.model_copy(
                update={"directory": tmp_path / "old-data"}
            ),
            "runtime": app.settings.runtime.model_copy(
                update={"reply_pause_min_seconds": 1, "reply_pause_max_seconds": 1}
            ),
        }
    )
    with monkeypatch.context() as context:
        context.setattr(Database, "open", open_old)
        prior = await Application.open(
            settings, FakeSender(), ScriptedModel(), app.clock
        )
    first, second = await prepare_pair(prior)
    await prior.delivery.send(first)
    await prior.database.write(restore_legacy_delay_column)
    async with prior.database.read() as tx:
        before = [
            dict(row)
            for row in await fetchall(
                tx, "SELECT * FROM outbox_messages ORDER BY rowid"
            )
        ]
    await prior.close()
    upgraded_sender = FakeSender()
    upgraded_sender.next_id = prior.sender.next_id
    upgraded = await Application.open(
        settings, upgraded_sender, ScriptedModel(), app.clock
    )
    try:
        async with upgraded.database.read() as tx:
            after = [
                dict(row)
                for row in await fetchall(
                    tx, "SELECT * FROM outbox_messages ORDER BY rowid"
                )
            ]
            assert before == [
                {
                    key: value
                    for key, value in row.items()
                    if key != "message_delay_seconds"
                }
                for row in after
            ]
            assert all(row["message_delay_seconds"] is None for row in after)
            assert not await fetchall(tx, "PRAGMA foreign_key_check")

        async def extend(tx: Transaction) -> None:
            await tx.execute(
                "UPDATE outbox_messages SET delay_seconds=4 WHERE id=?", (second,)
            )

        await upgraded.database.write(extend)
        app.clock.advance(1)
        await upgraded.delivery.run_once()
        assert upgraded.sender.sent == [(1, "Что теперь на ужин?")]
    finally:
        await upgraded.close()


@pytest.mark.parametrize(
    "text",
    [
        "Нормально. С тобой спокойно разговаривать.",
        "Хорошо. Приятно с тобой болтать.",
        "Нормально. Рад, что ты тут.",
    ],
)
async def test_formulaic_reciprocal_answer_is_repaired_before_delivery(
    app: Application, text: str
) -> None:
    app.model.responses["generate"].append(
        {
            "reply": text,
            "continuation": "В парке много народу было?",
            "dialogue_act": "answer_and_question",
            "question_goal": "park_crowd",
        }
    )
    app.model.responses["repair"].append(
        {
            "reply": "Да неплохо.",
            "continuation": "В парке много народу было?",
            "dialogue_act": "answer_and_question",
            "question_goal": "park_crowd",
        }
    )
    await app.telegram.ingest([update(1, "Посидела в парке. А ты как?")])
    await app.drain()
    app.clock.advance(4)
    await app.drain()
    assert app.sender.sent == [(1, "Да неплохо"), (1, "В парке много народу было?")]
    repair = next(data for task, data in app.model.calls if task == "repair")
    assert "FORMULAIC_SOCIAL_ANSWER" in repair["REPAIR_REASON_CODES"]
    assert len(app.model.calls) == 4


async def test_longer_pause_keeps_typing_and_cancels_on_new_input(
    app: Application,
) -> None:
    app.agent.settings = app.agent.settings.model_copy(
        update={"reply_pause_min_seconds": 8, "reply_pause_max_seconds": 16}
    )
    second, sender = await start_waiting(app)
    due = await scalar(
        app, "SELECT created_at FROM outbox_messages WHERE id=?", (second,)
    )
    assert 8 <= due - app.clock.now() <= 16
    app.clock.advance(7)
    await app.delivery.run_once()
    assert sender.active and not sender.sent
    await app.telegram.ingest([update(2, "Давай новую тему.")])
    await app.delivery.run_once()
    assert sender.finished.is_set()
    app.clock.advance(16)
    await app.delivery.send(second)
    assert not sender.sent


async def test_pause_upgrade_preserves_followup_timer(
    app: Application, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_followups import proposal, seed

    old_migrations = tmp_path / "previous-pacing"
    old_migrations.mkdir()
    paths = await asyncio.to_thread(lambda: list(Path("app/migrations").glob("*.sql")))
    for path in paths:
        if path.name != "008_reply_pacing.sql":
            (old_migrations / path.name).write_text(path.read_text())
    original = Database.open

    async def open_old(database: Database, migrations: Path) -> None:
        await original(database, old_migrations)

    settings = app.settings.model_copy(
        update={
            "storage": app.settings.storage.model_copy(
                update={"directory": tmp_path / "previous-data"}
            ),
            "runtime": app.settings.runtime.model_copy(
                update={"followup_enabled": True}
            ),
        }
    )
    with monkeypatch.context() as context:
        context.setattr(Database, "open", open_old)
        prior = await Application.open(
            settings, FakeSender(), ScriptedModel(), app.clock
        )
    await seed(prior)
    await prior.database.write(restore_legacy_delay_column)
    async with prior.database.read() as tx:
        before = [dict(row) for row in await fetchall(tx, "SELECT * FROM followups")]
        anchors = [
            dict(row)
            for row in await fetchall(
                tx, "SELECT * FROM outbox_messages ORDER BY rowid"
            )
        ]
    assert len(before) == 1
    await prior.close()
    sender = FakeSender()
    sender.next_id = prior.sender.next_id
    upgraded = await Application.open(settings, sender, ScriptedModel(), app.clock)
    try:
        async with upgraded.database.read() as tx:
            assert before == [
                dict(row) for row in await fetchall(tx, "SELECT * FROM followups")
            ]
            upgraded_anchors = [
                dict(row)
                for row in await fetchall(
                    tx, "SELECT * FROM outbox_messages ORDER BY rowid"
                )
            ]
            assert anchors == [
                {
                    key: value
                    for key, value in row.items()
                    if key != "message_delay_seconds"
                }
                for row in upgraded_anchors
            ]
            assert all(row["message_delay_seconds"] is None for row in upgraded_anchors)
            assert not await fetchall(tx, "PRAGMA foreign_key_check")

        async def widen(tx: Transaction) -> None:
            await tx.execute(
                "UPDATE outbox_messages SET delay_seconds=16 WHERE id=?",
                (anchors[0]["id"],),
            )

        await upgraded.database.write(widen)
        proposal(upgraded)
        app.clock.advance(before[0]["run_at"] - app.clock.now())
        await upgraded.drain()
        assert len(sender.sent) == 1
    finally:
        await upgraded.close()


async def restore_legacy_delay_column(tx: Transaction) -> None:
    await tx.execute(
        "UPDATE outbox_messages SET delay_seconds=COALESCE(message_delay_seconds,delay_seconds)"
    )
    await tx.execute("ALTER TABLE outbox_messages DROP COLUMN message_delay_seconds")
    await tx.execute("DELETE FROM schema_migrations WHERE name='010_message_delay.sql'")
