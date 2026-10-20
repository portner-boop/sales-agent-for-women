import asyncio

import pytest

from app.application import Application
from app.modules.delivery.entities import SendResult
from app.modules.telegram.service import normalize_update
from app.shared.entities import ScopeContext, StaleStateError
from tests.helpers import conversation, scalar, update


async def test_durable_offset_duplicate_and_coalesced_inputs(app: Application) -> None:
    first = update(1, "hello")
    await app.telegram.ingest([first, first, update(2, "one more detail")])
    assert await app.telegram.repository.offset() == 3
    assert await scalar(app, "SELECT COUNT(*) FROM inbox_events") == 2
    await app.drain()
    assert len(app.sender.sent) == 1
    assert await scalar(app, "SELECT COUNT(*) FROM turns WHERE status='delivered'") == 1
    assert len(app.model.calls[0][1]["pending_ids"]) == 2
    assert await scalar(app, "SELECT COUNT(*) FROM said_ledger") == 1


async def test_stop_during_generation_discards_reply(app: Application) -> None:
    app.model.release = asyncio.Event()
    await app.telegram.ingest([update(1, "hello")])
    task = asyncio.create_task(app.turn_worker.run_once())
    await asyncio.wait_for(app.model.entered.wait(), 2)
    await app.telegram.ingest([update(2, "/stop")])
    await task
    await app.drain()
    assert [text for _, text in app.sender.sent] == [
        "Chat stopped. Memory is retained. Use /start to resume."
    ]
    assert (await conversation(app)).control_state == "stopped"


async def test_unknown_delivery_never_retries_or_records_first_reply(
    app: Application,
) -> None:
    app.sender.results.append(SendResult("delivery_unknown", error_code="TIMEOUT"))
    await app.telegram.ingest([update(1, "hello", user=2)])
    await app.drain()
    await app.delivery_repository.recover()
    await app.drain()
    assert len(app.sender.sent) == 1
    assert (await conversation(app, 2)).first_reply_at is None
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
    request = app.model.calls[-1][1]
    assert len(request["pending_ids"]) == 1
    assert (
        next(
            item
            for item in request["original_messages"]
            if item["id"] == request["pending_ids"][0]
        )["version"]
        == 2
    )
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
    raw = update(1, "/stop")
    raw["message"]["entities"] = []
    await app.telegram.ingest([raw])
    assert (await conversation(app)).control_state == "enabled"
    assert normalize_update(update(2, "hello"), 16000).kind == "message"
    iso = update(3, "hello")
    iso["message"]["date"] = "2030-03-17T17:46:40+00:00"
    assert normalize_update(iso, 16000).kind == "message"


async def test_messages_continue_after_three_minutes_for_any_user(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "/start", user=2)])
    await app.drain()
    assert (await conversation(app, 2)).first_reply_at is None
    await app.telegram.ingest([update(2, "hello", user=2)])
    await app.drain()
    first = (await conversation(app, 2)).first_reply_at
    app.clock.advance(181)
    await app.drain()
    assert (
        await scalar(
            app, "SELECT COUNT(*) FROM scheduled_jobs WHERE kind='expire_trial'"
        )
        == 0
    )
    await app.telegram.ingest([update(3, "still there?", user=2)])
    await app.drain()
    assert app.commerce.can_reply(await conversation(app, 2))
    assert (await conversation(app, 2)).first_reply_at == first
    assert (
        await scalar(app, "SELECT COUNT(*) FROM said_ledger WHERE action_kind='reply'")
        == 2
    )


async def test_distinct_message_ids_with_equal_text_are_not_webhook_duplicates(
    app: Application,
) -> None:
    first = update(1, "what did i tell you?")
    await app.telegram.ingest([first, first])
    await app.drain()
    assert len(app.model.calls) == 1
    await app.telegram.ingest([update(2, "what did i tell you?")])
    await app.drain()
    assert len(app.model.calls) == 2
    assert len(app.sender.sent) == 2
