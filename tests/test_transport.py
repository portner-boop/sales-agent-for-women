import asyncio
import json

import pytest

from app.application import Application
from app.modules.delivery.entities import SendResult
from app.modules.delivery.schemas import OutboxDraft
from app.modules.telegram.service import normalize_update
from app.shared.entities import ScopeContext, StaleStateError
from app.shared.storage import Transaction
from tests.helpers import (
    confirm_test_purchase,
    conversation,
    interpretation_operation,
    scalar,
    update,
)


async def test_durable_offset_duplicate_and_coalesced_inputs(app: Application) -> None:
    first = update(1, "hello")
    await app.telegram.ingest([first, first, update(2, "one more detail")])
    assert await app.telegram.repository.offset() == 3
    assert await scalar(app, "SELECT COUNT(*) FROM inbox_events") == 2
    await app.drain()
    assert len(app.sender.sent) == 1
    assert await scalar(app, "SELECT COUNT(*) FROM turns WHERE status='prepared'") == 1
    assert len(app.model.calls[0][1]["CURRENT_INPUT"]) == 2
    assert await scalar(app, "SELECT COUNT(*) FROM said_ledger") == 1


async def test_stop_during_generation_discards_reply(app: Application) -> None:
    app.model.release = asyncio.Event()
    await app.telegram.ingest([update(1, "hello")])
    task = asyncio.create_task(app.turn_worker.run_once())
    await app.model.entered.wait()
    await app.telegram.ingest([update(2, "/stop")])
    await task
    await app.drain()
    assert [text for _, text in app.sender.sent] == [
        "Chat stopped. Memory is retained. Use /start to resume."
    ]
    assert (await conversation(app)).control_state == "stopped"


async def test_purchase_during_generation_handoff_once(app: Application) -> None:
    app.model.release = asyncio.Event()
    await app.telegram.ingest([update(1, "hello")])
    task = asyncio.create_task(app.turn_worker.run_once())
    await app.model.entered.wait()
    await confirm_test_purchase(app, "first-purchase")
    await task
    await app.drain()
    await confirm_test_purchase(app, "first-purchase")
    await app.telegram.ingest([update(4, "continue please")])
    await app.drain()
    assert len(app.sender.sent) == 1
    assert "simulated" in app.sender.sent[0][1]
    assert await scalar(app, "SELECT COUNT(*) FROM purchases") == 1
    assert (await conversation(app)).handoff_state == "delivered"


async def test_unknown_delivery_never_retries_or_records_first_reply(
    app: Application,
) -> None:
    app.sender.results.append(SendResult("delivery_unknown", error_code="TIMEOUT"))
    await app.telegram.ingest([update(1, "hello", user=3)])
    await app.drain()
    await app.delivery_repository.recover()
    await app.drain()
    assert len(app.sender.sent) == 1
    assert (await conversation(app, 3)).first_reply_at is None
    assert await scalar(app, "SELECT COUNT(*) FROM said_ledger") == 0
    assert await scalar(app, "SELECT status FROM outbox_messages") == "delivery_unknown"


async def test_retry_only_explicit_rate_limit(app: Application) -> None:
    app.sender.results.append(
        SendResult("retryable_failure", error_code="RATE_LIMIT", retry_after=1)
    )
    await app.telegram.ingest([update(1, "hello")])
    await app.drain()
    assert len(app.sender.sent) == 1
    app.clock.advance(2)
    await app.drain()
    assert len(app.sender.sent) == 2
    assert await scalar(app, "SELECT COUNT(*) FROM said_ledger") == 1


async def test_new_input_cancels_ready_draft(app: Application) -> None:
    await app.telegram.ingest([update(1, "hello")])
    await app.turn_worker.run_once()
    await app.telegram.ingest([update(2, "correction")])
    await app.drain()
    assert len(app.sender.sent) == 1
    assert (
        await scalar(
            app, "SELECT COUNT(*) FROM outbox_messages WHERE status='cancelled'"
        )
        == 1
    )


async def test_message_edit_does_not_replay_later_history(app: Application) -> None:
    for number in range(1, 4):
        await app.telegram.ingest([update(number, f"message {number}")])
        await app.drain()
    await app.telegram.ingest([update(4, "corrected first", message_id=1, edited=True)])
    await app.drain()
    interpreted = [data for task, data in app.model.calls if task == "interpret"][-1]
    assert len(interpreted["CURRENT_INPUT"]) == 1
    assert interpreted["CURRENT_INPUT"][0]["version"] == 2
    assert (
        await scalar(
            app,
            "SELECT COUNT(*) FROM message_versions WHERE message_id=(SELECT id FROM messages WHERE external_id=1)",
        )
        == 2
    )


async def test_kill_switch_rechecks_ready_actions(app: Application) -> None:
    await app.telegram.ingest([update(1, "hello")])
    await app.turn_worker.run_once()
    await app.delivery_repository.kill_switch(True)
    await app.drain()
    assert not app.sender.sent


async def test_scope_gate_and_foreign_user_are_server_owned(app: Application) -> None:
    await app.telegram.ingest([update(1, "first", user=1), update(2, "second", user=2)])
    first = await conversation(app, 1)
    second = await conversation(app, 2)
    assert first.id != second.id
    assert len(await app.conversations.messages(first.scope)) == 1
    with pytest.raises(StaleStateError):
        await app.conversations.messages(ScopeContext(first.id, 0))


async def test_commands_are_not_parsed_from_model_or_quoted_text(
    app: Application,
) -> None:
    raw = update(1, "/purchase_test")
    raw["message"]["entities"] = []
    await app.telegram.ingest([raw])
    assert (await conversation(app)).purchase_state == "none"
    assert normalize_update(update(2, "hello"), 16000).kind == "message"
    iso = update(3, "hello")
    iso["message"]["date"] = "2030-03-17T17:46:40+00:00"
    assert normalize_update(iso, 16000).kind == "message"


async def test_messages_continue_after_three_minutes_for_any_user(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "/start", user=3)])
    await app.drain()
    assert (await conversation(app, 3)).first_reply_at is None
    await app.telegram.ingest([update(2, "hello", user=3)])
    await app.drain()
    first = (await conversation(app, 3)).first_reply_at
    app.clock.advance(181)
    await app.drain()
    assert (
        await scalar(
            app, "SELECT COUNT(*) FROM scheduled_jobs WHERE kind='expire_trial'"
        )
        == 0
    )
    await app.telegram.ingest([update(3, "still there?", user=3)])
    await app.drain()
    assert app.commerce.can_reply(await conversation(app, 3))
    assert (await conversation(app, 3)).first_reply_at == first
    assert (
        await scalar(app, "SELECT COUNT(*) FROM said_ledger WHERE action_kind='reply'")
        == 2
    )


async def test_stop_blocks_offer_even_if_prepared_after_stop(app: Application) -> None:
    await app.telegram.ingest([update(1, "/stop")])
    await app.drain()
    current = await conversation(app)

    async def prepare(tx: Transaction) -> None:
        await app.delivery_repository.prepare(
            tx,
            current.snapshot,
            OutboxDraft(
                action_key="offer-test",
                kind="offer",
                text="Catalog",
                deadline=app.clock.now() + 10,
            ),
            app.clock.now(),
        )

    await app.database.write(prepare)
    await app.drain()
    assert len(app.sender.sent) == 1


async def test_internal_mock_purchase_preserves_minimal_handoff_context(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "/start")])
    current = await conversation(app)
    await app.drain()
    assert (await conversation(app)).first_reply_at is None
    await app.telegram.ingest([update(2, "Call me Ana.")])
    message = (await app.conversations.messages(current.scope))[-1]
    app.model.responses["interpret"].append(
        interpretation_operation(message, "preferred_name", "Ana", "fact")
    )
    await app.drain()
    await app.telegram.ingest(
        [update(3, "Do not call me darling. Let's discuss gardening.")]
    )
    message = (await app.conversations.messages(current.scope))[-1]
    patch = interpretation_operation(
        message, "forbidden_address", "darling", "boundary"
    )
    patch["agenda_operations"] = [
        {
            "topic_kind": "gardening",
            "status": "open",
            "question_goal": "gardening_preferences",
        }
    ]
    app.model.responses["interpret"].append(patch)
    await app.drain()
    app.clock.advance(181)
    await app.drain()
    await confirm_test_purchase(app, "test-purchase")
    await app.drain()
    current = await conversation(app)
    assert current.control_state == "handoff_only"
    assert current.handoff_state == "delivered"
    payload = json.loads(await scalar(app, "SELECT payload_json FROM handoffs"))
    assert payload["simulation"] is True
    assert payload["context"]["preferred_name"] == "Ana"
    assert payload["context"]["current_topics"] == ["gardening"]
    assert payload["context"]["boundaries"][0]["value"] == "darling"
    assert "sources" not in payload["context"]["boundaries"][0]
    calls = len(app.model.calls)
    await app.telegram.ingest([update(5, "Please continue.")])
    await app.drain()
    assert len(app.model.calls) == calls
