import json

import pytest

from app.application import Application
from app.evaluation import EvaluationSender, receive
from app.modules.delivery.entities import SendResult
from app.modules.delivery.schemas import OutboxDraft
from app.shared.entities import StaleStateError
from app.shared.storage import Transaction, fetchall
from tests.helpers import conversation, scalar, update


async def prepare_pair(app: Application) -> tuple[str, str]:
    app.model.responses["generate"].append(
        {
            "reply": "Кухня победила.",
            "continuation": "Что теперь на ужин?",
            "dialogue_act": "reaction_and_question",
            "question_goal": "replacement_dinner",
            "question_time_scope": "today",
        }
    )
    await app.telegram.ingest([update(1, "Ужин сгорел, зато кухня цела.")])
    await app.turn_worker.run_once()
    async with app.database.read() as tx:
        rows = await fetchall(tx, "SELECT id FROM outbox_messages ORDER BY rowid")
    assert len(rows) == 2
    return rows[0]["id"], rows[1]["id"]


async def test_second_part_waits_and_question_is_recorded_only_after_delivery(
    app: Application,
) -> None:
    first, second = await prepare_pair(app)
    await app.delivery.send(second)
    assert not app.sender.sent
    await app.delivery.send(first)
    assert app.sender.sent == [(1, "Кухня победила")]
    assert await scalar(app, "SELECT COUNT(*) FROM said_ledger") == 1
    metrics = await app.metrics()
    assert metrics["reply_latency_samples"] == 1
    assert metrics["reply_completion_samples"] == 0
    assert await scalar(app, "SELECT COUNT(*) FROM dialogue_threads") == 0
    assert (
        await scalar(app, "SELECT COUNT(*) FROM scheduled_jobs WHERE kind='compact'")
        == 0
    )
    stored = json.loads(
        await scalar(
            app, "SELECT proposal_json FROM outbox_messages WHERE id=?", (first,)
        )
    )
    assert stored["reply"] == "Кухня победила"
    assert stored["question_goal"] is None
    assert "continuation" not in stored
    due = await scalar(
        app, "SELECT created_at FROM outbox_messages WHERE id=?", (second,)
    )
    assert 2 <= due - app.clock.now() <= 4
    await app.delivery.send(second)
    assert len(app.sender.sent) == 1
    app.clock.advance(due - app.clock.now())
    await app.delivery.send(second)
    assert app.sender.sent[-1] == (1, "Что теперь на ужин?")
    assert await scalar(app, "SELECT COUNT(*) FROM said_ledger") == 2
    metrics = await app.metrics()
    assert metrics["reply_latency_samples"] == 1
    assert metrics["reply_completion_samples"] == 1
    assert metrics["reply_completion_p95_seconds"] >= 2
    question_id = await scalar(
        app, "SELECT last_question_message_id FROM dialogue_threads"
    )
    assert (
        await scalar(app, "SELECT external_id FROM messages WHERE id=?", (question_id,))
        == 100002
    )
    assert [name for name, _ in app.model.calls] == [
        "interpret",
        "generate",
        "validate",
    ]


@pytest.mark.parametrize("delay", [5, 9])
async def test_slow_first_delivery_still_waits_between_parts(
    app: Application, delay: float
) -> None:
    first, second = await prepare_pair(app)
    app.clock.advance(delay)
    await app.delivery.send(first)
    due = await scalar(
        app, "SELECT created_at FROM outbox_messages WHERE id=?", (second,)
    )
    assert 2 <= due - app.clock.now() <= 4
    await app.delivery.send(second)
    assert len(app.sender.sent) == 1
    app.clock.advance(due - app.clock.now())
    await app.delivery.send(second)
    assert len(app.sender.sent) == 2
    assert (
        await scalar(app, "SELECT status FROM outbox_messages WHERE id=?", (second,))
        == "delivered"
    )


@pytest.mark.parametrize("outcome", ["delivery_unknown", "permanent_failure"])
async def test_unconfirmed_first_part_cancels_continuation(
    app: Application, outcome: str
) -> None:
    first, second = await prepare_pair(app)
    app.sender.results.append(SendResult(outcome, error_code="TEST_FAILURE"))
    await app.delivery.send(first)
    app.clock.advance(3)
    await app.delivery.run_once()
    assert len(app.sender.sent) == 1
    assert (
        await scalar(app, "SELECT status FROM outbox_messages WHERE id=?", (second,))
        == "cancelled"
    )
    assert await scalar(app, "SELECT COUNT(*) FROM said_ledger") == 0
    assert await scalar(app, "SELECT COUNT(*) FROM dialogue_threads") == 0


async def test_rate_limit_retries_first_before_continuation(app: Application) -> None:
    first, second = await prepare_pair(app)
    app.sender.results.append(SendResult("retryable_failure", retry_after=1))
    await app.delivery.send(first)
    await app.delivery.send(second)
    assert len(app.sender.sent) == 1
    app.clock.advance(1)
    await app.delivery.run_once()
    assert [text for _, text in app.sender.sent] == [
        "Кухня победила",
        "Кухня победила",
    ]
    app.clock.advance(4)
    await app.delivery.run_once()
    assert app.sender.sent[-1][1] == "Что теперь на ужин?"
    assert await scalar(app, "SELECT COUNT(*) FROM said_ledger") == 2


async def test_stop_between_parts_cancels_remaining_question(app: Application) -> None:
    first, second = await prepare_pair(app)
    await app.delivery.send(first)
    await app.telegram.ingest([update(2, "/stop")])
    assert (
        await scalar(app, "SELECT status FROM outbox_messages WHERE id=?", (second,))
        == "cancelled"
    )
    app.clock.advance(3)
    await app.drain()
    assert "Что теперь на ужин?" not in [text for _, text in app.sender.sent]
    assert await scalar(app, "SELECT COUNT(*) FROM dialogue_threads") == 0


@pytest.mark.parametrize("edited", [False, True])
async def test_new_input_or_edit_between_parts_cancels_continuation(
    app: Application, edited: bool
) -> None:
    first, second = await prepare_pair(app)
    await app.delivery.send(first)
    await app.telegram.ingest(
        [
            update(
                2,
                "На самом деле ужин спасла.",
                edited=edited,
                message_id=1 if edited else 2,
            )
        ]
    )
    assert (
        await scalar(app, "SELECT status FROM outbox_messages WHERE id=?", (second,))
        == "cancelled"
    )
    app.clock.advance(3)
    await app.delivery.send(second)
    assert app.sender.sent == [(1, "Кухня победила")]
    assert await scalar(app, "SELECT COUNT(*) FROM dialogue_threads") == 0


async def test_forget_between_parts_removes_dependency_and_history(
    app: Application,
) -> None:
    first, _ = await prepare_pair(app)
    await app.delivery.send(first)
    old = await conversation(app)
    await app.telegram.ingest([update(2, "/forget")])
    await app.drain()
    assert (await conversation(app)).epoch == old.epoch + 1
    assert (
        await scalar(app, "SELECT COUNT(*) FROM messages WHERE role='assistant'") == 0
    )
    assert (
        await scalar(
            app, "SELECT COUNT(*) FROM outbox_messages WHERE predecessor_id IS NOT NULL"
        )
        == 0
    )
    async with app.database.read() as tx:
        assert await fetchall(tx, "PRAGMA foreign_key_check") == []


async def test_restart_after_first_delivery_preserves_due_time_without_duplication(
    app: Application,
) -> None:
    first, second = await prepare_pair(app)
    await app.delivery.send(first)
    due = await scalar(
        app, "SELECT created_at FROM outbox_messages WHERE id=?", (second,)
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
                (second,),
            )
            == due
        )
        app.clock.advance(due - app.clock.now())
        await restarted.delivery.run_once()
        assert [text for _, text in app.sender.sent] == [
            "Кухня победила",
            "Что теперь на ужин?",
        ]
        assert await scalar(restarted, "SELECT COUNT(*) FROM said_ledger") == 2
    finally:
        await restarted.close()


async def test_restart_during_first_delivery_cancels_second_without_resend(
    app: Application,
) -> None:
    first, second = await prepare_pair(app)

    async def sending(tx: Transaction) -> None:
        await tx.execute(
            "UPDATE outbox_messages SET status='sending' WHERE id=?", (first,)
        )

    await app.database.write(sending)
    await app.close()
    restarted = await Application.open(app.settings, app.sender, app.model, app.clock)
    try:
        app.clock.advance(3)
        await restarted.delivery.run_once()
        assert not app.sender.sent
        assert (
            await scalar(
                restarted, "SELECT status FROM outbox_messages WHERE id=?", (first,)
            )
            == "delivery_unknown"
        )
        assert (
            await scalar(
                restarted, "SELECT status FROM outbox_messages WHERE id=?", (second,)
            )
            == "cancelled"
        )
    finally:
        await restarted.close()


async def test_expired_continuation_never_creates_a_question(app: Application) -> None:
    first, second = await prepare_pair(app)
    await app.delivery.send(first)

    async def expire(tx: Transaction) -> None:
        await tx.execute(
            "UPDATE outbox_messages SET deadline=? WHERE id=?",
            (app.clock.now(), second),
        )

    await app.database.write(expire)
    await app.delivery.run_once()
    assert len(app.sender.sent) == 1
    assert (
        await scalar(app, "SELECT status FROM outbox_messages WHERE id=?", (second,))
        == "cancelled"
    )
    assert await scalar(app, "SELECT COUNT(*) FROM dialogue_threads") == 0


async def test_continuation_cannot_depend_on_another_scope(app: Application) -> None:
    first, _ = await prepare_pair(app)
    await app.telegram.ingest([update(2, "Hello", user=2)])
    other = await conversation(app, 2)

    async def foreign(tx: Transaction) -> None:
        await app.delivery_repository.prepare(
            tx,
            other.snapshot,
            OutboxDraft(
                action_key="foreign",
                kind="reply",
                text="Foreign",
                deadline=app.clock.now() + 10,
                predecessor_id=first,
            ),
            app.clock.now(),
            turn_id="foreign-turn",
        )

    with pytest.raises(StaleStateError, match="scoped reply"):
        await app.database.write(foreign)


async def test_style_violation_in_second_part_is_repaired_before_any_send(
    app: Application,
) -> None:
    app.model.responses["generate"].append(
        {
            "reply": "Кухня победила.",
            "continuation": "Что теперь на ужин? 😊",
            "dialogue_act": "chat",
            "question_goal": "replacement_dinner",
        }
    )
    app.model.responses["repair"].append(
        {
            "reply": "Кухня победила.",
            "continuation": "Что теперь на ужин?",
            "dialogue_act": "chat",
            "question_goal": "replacement_dinner",
        }
    )
    await app.telegram.ingest([update(1, "Ужин сгорел.")])
    await app.drain()
    app.clock.advance(4)
    await app.drain()
    assert [text for _, text in app.sender.sent] == [
        "Кухня победила",
        "Что теперь на ужин?",
    ]
    repair = next(data for name, data in app.model.calls if name == "repair")
    assert "STYLE_VIOLATION" in repair["REPAIR_REASON_CODES"]
    assert len(app.model.calls) == 4


async def test_second_part_semantic_failure_prevents_sending_first(
    app: Application,
) -> None:
    app.model.responses["generate"].append(
        {
            "reply": "Кухня победила.",
            "continuation": "Ты же была со мной в Риме?",
            "dialogue_act": "chat",
            "question_goal": "rome_trip",
        }
    )
    app.model.responses["validate"].append(
        {"decision": "reject", "reason_codes": ["UNSUPPORTED_LITERAL_RECALL"]}
    )
    await app.telegram.ingest([update(1, "Ужин сгорел.")])
    await app.drain()
    assert "Кухня победила." not in [text for _, text in app.sender.sent]
    assert all("Риме" not in text for _, text in app.sender.sent)
    assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"
    critic = next(data for name, data in app.model.calls if name == "validate")
    assert critic["reply"]["continuation"] == "Ты же была со мной в Риме?"


async def test_evaluation_waits_for_both_parts_with_virtual_clock(
    app: Application,
) -> None:
    sender = EvaluationSender()
    app.sender = sender
    app.delivery.sender = sender
    app.model.responses["generate"].append(
        {
            "reply": "Кухня победила.",
            "continuation": "Что теперь на ужин?",
            "dialogue_act": "chat",
            "question_goal": "replacement_dinner",
        }
    )
    before = app.clock.now()
    await receive(app, sender, 1, "Ужин сгорел.", 1, before)
    assert len(sender.messages) == 2
    assert 2 <= app.clock.now() - before <= 4
