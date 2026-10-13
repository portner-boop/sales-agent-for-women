import asyncio
import json

import pytest

from app.application import Application
from app.modules.agent.errors import ModelOutputError
from app.modules.delivery.entities import SendResult
from app.shared.control import request_control
from app.shared.operations import ProcessLock, restore_backup
from app.shared.storage import Transaction
from tests.helpers import FakeSender, conversation, scalar, update


@pytest.mark.parametrize(
    "error", [ModelOutputError("INVALID_JSON"), TimeoutError("timeout")]
)
async def test_invalid_json_and_transient_timeouts_recover_before_sending(
    app: Application, error: Exception
) -> None:
    app.model.responses["interpret"].append(error)
    await app.telegram.ingest([update(1, "hello", user=3)])
    await app.drain()
    assert (await conversation(app, 3)).first_reply_at is not None
    assert await scalar(app, "SELECT kind FROM outbox_messages") == "reply"
    assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"
    assert [task for task, _ in app.model.calls][:2] == ["interpret", "interpret"]


@pytest.mark.parametrize(
    "text", ["Кофе с мёдом — необычный выбор.", "That coffee — quite a surprise."]
)
async def test_em_dash_triggers_repair_before_sending(
    app: Application, text: str
) -> None:
    app.model.responses["generate"].append({"reply": text, "dialogue_act": "comment"})
    app.model.responses["repair"].append(
        {"reply": "Неожиданно. Интересный выбор.", "dialogue_act": "comment"}
    )
    await app.telegram.ingest([update(1, "Кофе с мёдом.")])
    await app.drain()
    assert app.sender.sent == [(1, "Неожиданно. Интересный выбор")]
    repair = next(data for task, data in app.model.calls if task == "repair")
    assert "EM_DASH_NOT_ALLOWED" in repair["REPAIR_REASON_CODES"]
    assert len(app.model.calls) <= 4


async def test_style_repair_is_bounded_and_accepted(app: Application) -> None:
    app.model.responses["generate"].append(
        {"reply": "Hello!!! 😊", "dialogue_act": "greet"}
    )
    app.model.responses["repair"].append(
        {"reply": "Glad to hear that.", "dialogue_act": "acknowledge"}
    )
    await app.telegram.ingest([update(1, "hello")])
    await app.drain()
    assert app.sender.sent[0][1] == "Glad to hear that"
    assert len(app.model.calls) == 3


@pytest.mark.parametrize(
    "text",
    [
        "С тобой – спокойно.",
        "С тобой -- спокойно.",
        "Договорились :)",
        "Договорились)",
        "Договорились!?",
        "**Договорились.**",
        "Договорились 1️⃣",
    ],
)
async def test_dialogue_style_form_is_repaired_before_delivery(
    app: Application, text: str
) -> None:
    app.model.responses["generate"].append({"reply": text, "dialogue_act": "comment"})
    app.model.responses["repair"].append(
        {"reply": "Договорились.", "dialogue_act": "acknowledge"}
    )
    await app.telegram.ingest([update(1, "Давай поболтаем.")])
    await app.drain()
    assert app.sender.sent == [(1, "Договорились")]
    assert "repair" in [task for task, _ in app.model.calls]
    assert len(app.model.calls) <= 4


@pytest.mark.parametrize(
    "text",
    [
        "I have a cat.",
        "I live in Madrid.",
        "We visited Rome together last summer.",
        "У меня спокойно: без суеты, просто тихий вечер.",
    ],
)
async def test_ungrounded_claims_trigger_critic_even_without_declared_sources(
    app: Application, text: str
) -> None:
    app.model.responses["generate"].append({"reply": text, "dialogue_act": "chat"})
    app.model.responses["validate"].append(
        {"decision": "reject", "reason_codes": ["CANON_CONFLICT"]}
    )
    app.model.responses["repair"].append(
        {
            "reply": (
                "Рад это слышать."
                if text.startswith("У меня")
                else "Glad to hear that."
            ),
            "dialogue_act": "acknowledge",
        }
    )
    await app.telegram.ingest(
        [
            update(
                1,
                (
                    "Привет, как проходит вечер?"
                    if text.startswith("У меня")
                    else "hello, how is your evening?"
                ),
            )
        ]
    )
    await app.drain()
    assert [name for name, _ in app.model.calls] == [
        "interpret",
        "generate",
        "validate",
        "repair",
        "validate_repair",
    ]
    assert app.sender.sent[0][1] != text
    assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"


async def test_stop_attempts_cancel_inflight_sender_and_records_unknown(
    app: Application,
) -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    class HangingSender:
        async def send_text(self, chat_id: int, text: str) -> SendResult:
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            raise AssertionError

    await app.telegram.ingest([update(1, "hello")])
    await app.turn_worker.run_once()
    app.delivery.sender = HangingSender()
    task = asyncio.create_task(app.delivery.run_once())
    await entered.wait()
    await app.telegram.ingest([update(2, "/stop")])
    await task
    assert cancelled.is_set()
    assert (
        await scalar(
            app, "SELECT COUNT(*) FROM outbox_messages WHERE status='delivery_unknown'"
        )
        == 1
    )
    assert await scalar(app, "SELECT COUNT(*) FROM said_ledger") == 0


async def test_sender_uses_remaining_deadline(app: Application) -> None:
    class SlowSender:
        async def send_text(self, chat_id: int, text: str) -> SendResult:
            await asyncio.sleep(1)
            return SendResult("delivered", 999)

    await app.telegram.ingest([update(1, "hello")])
    await app.turn_worker.run_once()

    async def shorten(tx: Transaction) -> None:
        await tx.execute(
            "UPDATE outbox_messages SET deadline=?", (app.clock.now() + 0.01,)
        )

    await app.database.write(shorten)
    app.delivery.sender = SlowSender()
    await app.delivery.run_once()
    assert await scalar(app, "SELECT status FROM outbox_messages") == "delivery_unknown"


async def test_live_control_metrics_backup_and_kill_switch(app: Application) -> None:
    await app.start_controls()
    await app.telegram.ingest([update(1, "hello")])
    await app.turn_worker.run_once()
    result = await request_control(
        app.settings.storage.directory, {"command": "kill-switch", "enabled": True}
    )
    assert result == {"kill_switch": True}
    await app.drain()
    assert not app.sender.sent
    assert (
        await request_control(app.settings.storage.directory, {"command": "metrics"})
    )["outbox_cancelled"] == 1
    assert "path" in await request_control(
        app.settings.storage.directory, {"command": "backup"}
    )


async def test_restore_never_resends_snapshot_ready_action(app: Application) -> None:
    await app.telegram.ingest([update(1, "hello")])
    await app.turn_worker.run_once()
    backup = await app.memory.backup()
    await app.delivery.run_once()
    assert len(app.sender.sent) == 1
    await app.close()
    with ProcessLock(app.settings.storage.directory):
        restore_backup(backup, app.settings.storage.directory)
    restarted = await Application.open(app.settings, FakeSender(), app.model, app.clock)
    try:
        await restarted.drain()
        assert not restarted.sender.sent
        assert (await conversation(restarted)).control_state == "paused"
        assert (
            await scalar(restarted, "SELECT status FROM outbox_messages")
            == "delivery_unknown"
        )
    finally:
        await restarted.close()


async def test_common_subject_does_not_outrank_exact_old_topic(
    app: Application,
) -> None:
    from app.modules.memory.entities import CompactionBlock
    from app.modules.memory.service import source_hash

    for number in range(1, 3):
        await app.telegram.ingest([update(number, f"story {number}")])
        await app.drain()
        current = await conversation(app)
        messages = tuple((await app.conversations.messages(current.scope))[-2:])
        block = CompactionBlock(messages, source_hash(messages))
        summary = await app.model.summarize(block, 5, 256)
        summary.topic_kinds = ["chair_purchase" if number == 1 else "gardening"]
        await app.memory.publish_summary(current.scope, block, summary)
    summaries = await app.memory.repository.summaries(
        current.scope, ["chair_purchase"], ["user_self"]
    )
    assert summaries[0]["payload"]["topic_kinds"] == ["chair_purchase"]


async def test_reply_to_original_is_present_in_interpreter_and_generator(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "First or second?")])
    await app.drain()
    raw = update(2, "First.")
    raw["message"]["reply_to_message"] = {"message_id": 1}
    await app.telegram.ingest([raw])
    await app.drain()
    interpreted = [data for name, data in app.model.calls if name == "interpret"][-1]
    assert interpreted["REFERENCE_MESSAGES"][0]["text"] == "First or second?"
    generated = [data for name, data in app.model.calls if name == "generate"][-1]
    assert any(
        source["text"] == "First or second?"
        for source in generated["MEMORY_DATA"]["sources"]
    )


async def test_interpreter_memory_projection_is_bounded(app: Application) -> None:
    from app.modules.memory.schemas import MemoryOperation

    await app.telegram.ingest([update(1, "I enjoy many interests.")])
    current = await conversation(app)
    message = (await app.conversations.messages(current.scope))[0]

    async def seed(tx: Transaction) -> None:
        for number in range(500):
            operation = MemoryOperation.model_validate(
                {
                    "operation": "add",
                    "predicate": f"interest_{number}",
                    "value": "A long description of an interest that must not make the interpreter context grow indefinitely.",
                    "sources": [
                        {
                            "message_id": message.id,
                            "message_version": 1,
                            "evidence_text": message.text,
                        }
                    ],
                }
            )
            await app.memory.repository.apply(
                tx, current.scope, operation, app.clock.now()
            )

    await app.database.write(seed)
    await app.drain()
    interpreted = app.model.calls[0][1]
    assert len(interpreted["MEMORY"]) < 500
    assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"


async def test_runs_store_compact_checkpoint_references_without_prompts(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "confidential-dialogue-marker")])
    await app.drain()
    import aiosqlite

    async with aiosqlite.connect(
        app.settings.storage.directory / "checkpoints.db"
    ) as connection:
        cursor = await connection.execute("SELECT checkpoint FROM checkpoints")
        data = await cursor.fetchall()
        assert all(b"confidential-dialogue-marker" not in row[0] for row in data)
    metrics = json.loads(await scalar(app, "SELECT metrics_json FROM agent_runs"))
    assert metrics["llm_calls"] == 2


async def test_rhetorical_reaction_has_no_agenda_goal_and_is_critically_checked(
    app: Application,
) -> None:
    app.model.responses["generate"].append(
        {"reply": "Мёд в кофе? Неожиданно.", "dialogue_act": "react"}
    )
    await app.telegram.ingest([update(1, "Пью кофе с мёдом.")])
    await app.drain()
    assert app.sender.sent == [(1, "Мёд в кофе? Неожиданно")]
    assert [task for task, _ in app.model.calls] == [
        "interpret",
        "generate",
        "validate",
    ]
    assert await scalar(app, "SELECT COUNT(*) FROM dialogue_threads") == 0
    metrics = json.loads(await scalar(app, "SELECT metrics_json FROM agent_runs"))
    assert metrics["validation_events"][-1]["decision"] == "accept"


async def test_information_question_without_goal_cannot_bypass_critic(
    app: Application,
) -> None:
    app.model.responses["generate"].append(
        {"reply": "Какой кофе ты предпочитаешь?", "dialogue_act": "ask"}
    )
    app.model.responses["validate"].append(
        {"decision": "reject", "reason_codes": ["MISSING_QUESTION_GOAL"]}
    )
    await app.telegram.ingest([update(1, "Привет.")])
    await app.drain()
    assert all("Какой кофе" not in text for _, text in app.sender.sent)
    assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"
    metrics = json.loads(await scalar(app, "SELECT metrics_json FROM agent_runs"))
    verdict = next(
        event
        for event in metrics["validation_events"]
        if event["stage"] == "draft_semantic"
    )
    assert verdict["reason_codes"] == ["MISSING_QUESTION_GOAL"]
    assert metrics["llm_calls"] <= 4
