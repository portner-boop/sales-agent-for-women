import asyncio

import pytest

from app.application import Application
from app.modules.delivery.entities import SendResult
from app.shared.storage import fetchall
from tests.helpers import FakeSender, ScriptedModel, scalar, update


def enable(app: Application) -> None:
    runtime = app.settings.runtime.model_copy(update={"followup_enabled": True})
    app.settings = app.settings.model_copy(update={"runtime": runtime})
    app.agent.settings = runtime
    app.followups.settings = runtime
    app.telegram.settings = runtime


async def seed(
    app: Application,
    text: str = "Сегодня гуляла в парке.",
    user: int = 1,
    number: int = 1,
) -> None:
    app.model.responses["generate"].append(
        {
            "reply": "Люблю парки без толпы.",
            "dialogue_act": "opinion",
            "followup_allowed": True,
        }
    )
    await app.telegram.ingest([update(number, text, user=user)])
    await app.drain()


async def due(app: Application) -> None:
    timer = await scalar(
        app, "SELECT MIN(run_at) FROM followups WHERE status='pending'"
    )
    app.clock.advance(timer - app.clock.now())


def proposal(app: Application, should_send: bool = True) -> None:
    app.model.responses["followup"].append(
        {
            "reply": "А у тебя там есть любимое место?" if should_send else "",
            "dialogue_act": "question",
            "question_goal": "park_place" if should_send else None,
            "should_send": should_send,
        }
    )


async def test_followup_runs_once_without_synthetic_input(app: Application) -> None:
    enable(app)
    await seed(app)
    assert (
        30 <= await scalar(app, "SELECT run_at FROM followups") - app.clock.now() <= 60
    )
    await app.drain()
    assert len(app.sender.sent) == 1
    proposal(app)
    await due(app)
    await app.drain()
    assert len(app.sender.sent) == 2
    data = next(data for task, data in app.model.calls if task == "followup")
    assert data["CURRENT_INPUT"] == []
    assert data["TRIGGER"]["kind"] == "silence"
    assert await scalar(app, "SELECT COUNT(*) FROM messages WHERE role='user'") == 1
    assert await scalar(app, "SELECT COUNT(*) FROM said_ledger") == 2
    assert await scalar(app, "SELECT status FROM followups") == "prepared"
    assert (await app.metrics())["followup_delivery_delivered"] == 1
    app.clock.advance(3600)
    await app.drain()
    assert len(app.sender.sent) == 2


@pytest.mark.parametrize(
    "text", ["Новая тема", "/stop", "/start", "/privacy", "/forget"]
)
async def test_new_input_cancels_followup(app: Application, text: str) -> None:
    enable(app)
    await seed(app)
    await app.telegram.ingest([update(2, text)])
    app.clock.advance(150)
    await app.drain()
    assert all(task != "followup" for task, _ in app.model.calls)
    assert (
        await scalar(
            app,
            "SELECT COUNT(*) FROM followups WHERE status IN ('pending','generating')",
        )
        == 0
    )
    if text == "/forget":
        assert await scalar(app, "SELECT COUNT(*) FROM followups") == 0


async def test_duplicate_update_keeps_timer_but_edit_cancels_it(
    app: Application,
) -> None:
    enable(app)
    await seed(app)
    await app.telegram.ingest([update(1, "Сегодня гуляла в парке.")])
    assert await scalar(app, "SELECT status FROM followups") == "pending"
    await app.telegram.ingest(
        [update(2, "Сегодня гуляла у моря.", message_id=1, edited=True)]
    )
    assert await scalar(app, "SELECT status FROM followups") == "cancelled"


@pytest.mark.parametrize(
    "text",
    [
        "Я пойду спать",
        "Напишу позже",
        "Don't text me first",
        "Hablamos mañana",
        "Falamos depois",
    ],
)
async def test_closing_does_not_schedule_even_if_model_allows(
    app: Application, text: str
) -> None:
    enable(app)
    await seed(app, text)
    assert await scalar(app, "SELECT COUNT(*) FROM followups") == 0


async def test_timer_starts_after_last_confirmed_part(app: Application) -> None:
    enable(app)
    app.model.responses["generate"].append(
        {
            "reply": "Люблю парки.",
            "continuation": "Особенно когда там тихо.",
            "dialogue_act": "opinion",
            "followup_allowed": True,
        }
    )
    await app.telegram.ingest([update(1, "Сегодня была в парке.")])
    await app.turn_worker.run_once()
    async with app.database.read() as tx:
        parts = await fetchall(tx, "SELECT id FROM outbox_messages ORDER BY rowid")
    await app.delivery.send(parts[0]["id"])
    assert await scalar(app, "SELECT COUNT(*) FROM followups") == 0
    app.clock.advance(4)
    await app.delivery.send(parts[1]["id"])
    assert await scalar(app, "SELECT anchor_id FROM followups") == parts[1]["id"]
    assert (
        30 <= await scalar(app, "SELECT run_at FROM followups") - app.clock.now() <= 60
    )


async def test_unknown_delivery_does_not_start_timer(app: Application) -> None:
    enable(app)
    app.sender.results.append(SendResult("delivery_unknown"))
    await seed(app)
    assert await scalar(app, "SELECT COUNT(*) FROM followups") == 0


@pytest.mark.parametrize("mode", ["skip", "error", "expired", "disabled", "kill"])
async def test_optional_failures_are_silent(app: Application, mode: str) -> None:
    enable(app)
    await seed(app)
    if mode == "skip":
        proposal(app, False)
    elif mode == "error":
        app.model.responses["followup"].extend(
            [RuntimeError("gateway down"), RuntimeError("gateway down")]
        )
    elif mode == "disabled":
        app.followups.settings = app.followups.settings.model_copy(
            update={"followup_enabled": False}
        )
    elif mode == "kill":
        await app._control({"command": "kill-switch", "enabled": True})
        await app._control({"command": "kill-switch", "enabled": False})
    if mode == "kill":
        app.clock.advance(150)
    else:
        await due(app)
        if mode == "expired":
            app.clock.advance(91)
    await app.drain()
    assert len(app.sender.sent) == 1
    assert (
        await scalar(
            app,
            "SELECT COUNT(*) FROM followups WHERE status IN ('pending','generating')",
        )
        == 0
    )
    assert (
        await scalar(
            app, "SELECT COUNT(*) FROM scheduled_jobs WHERE status='dead_letter'"
        )
        == 0
    )


async def test_restart_preserves_pending_timer(app: Application) -> None:
    enable(app)
    await seed(app)
    timer = await scalar(app, "SELECT run_at FROM followups")
    await app.close()
    sender, model = FakeSender(), ScriptedModel()
    sender.next_id = app.sender.next_id
    restarted = await Application.open(app.settings, sender, model, app.clock)
    try:
        assert await scalar(restarted, "SELECT run_at FROM followups") == timer
        proposal(restarted)
        await due(restarted)
        await restarted.drain()
        assert len(sender.sent) == 1
        assert await scalar(restarted, "SELECT status FROM followups") == "prepared"
    finally:
        await restarted.close()


async def test_new_input_interrupts_running_followup(
    app: Application, monkeypatch: pytest.MonkeyPatch
) -> None:
    enable(app)
    await seed(app)
    entered, release = asyncio.Event(), asyncio.Event()
    original = app.model.complete

    async def block(task, rules, data, schema, deadline_seconds):
        if task == "followup":
            entered.set()
            await release.wait()
        return await original(task, rules, data, schema, deadline_seconds)

    monkeypatch.setattr(app.model, "complete", block)
    proposal(app)
    await due(app)
    worker = asyncio.create_task(app.turn_worker.run_once())
    await asyncio.wait_for(entered.wait(), 2)
    await app.telegram.ingest([update(2, "/stop")])
    await asyncio.wait_for(worker, 2)
    await app.drain()
    assert await scalar(app, "SELECT status FROM followups") == "cancelled"
    assert all("любимое место" not in text for _, text in app.sender.sent)


async def test_five_scopes_keep_followups_separate(app: Application) -> None:
    enable(app)
    for user in range(1, 6):
        await seed(app, user=user, number=user)
    app.clock.advance(60)
    for _ in range(5):
        proposal(app)
    await app.drain()
    assert sorted(chat for chat, _ in app.sender.sent) == [
        user for user in range(1, 6) for _ in range(2)
    ]
    for task, data in app.model.calls:
        if task == "followup":
            assert len(data["CURRENT_INPUT"]) == 0
    assert await scalar(app, "SELECT COUNT(*) FROM followups") == 5


async def test_prepared_followup_cancelled_before_sender(app: Application) -> None:
    enable(app)
    await seed(app)
    proposal(app)
    await due(app)
    await app.turn_worker.run_once()
    await app.telegram.ingest([update(2, "/stop")])
    await app.drain()
    assert (
        await scalar(
            app,
            "SELECT status FROM outbox_messages WHERE json_extract(proposal_json,'$.origin')='followup'",
        )
        == "cancelled"
    )
    assert len(app.sender.sent) == 2
    assert "остановлена" in app.sender.sent[-1][1]


async def test_unknown_followup_is_not_retried(app: Application) -> None:
    enable(app)
    await seed(app)
    proposal(app)
    await due(app)
    app.sender.results.append(SendResult("delivery_unknown"))
    await app.drain()
    app.clock.advance(400)
    await app.drain()
    assert len(app.sender.sent) == 2
    assert await scalar(app, "SELECT COUNT(*) FROM said_ledger") == 1


async def test_proactive_boundary_overrides_eligibility(app: Application) -> None:
    from tests.helpers import conversation, interpretation_operation

    enable(app)
    await app.telegram.ingest([update(1, "Не пиши первым.")])
    current = await conversation(app)
    message = (await app.conversations.messages(current.scope))[-1]
    app.model.responses["interpret"].append(
        interpretation_operation(
            message, predicate="proactive_permission", value=False, kind="boundary"
        )
    )
    app.model.responses["generate"].append(
        {
            "reply": "Хорошо, буду ждать тебя.",
            "dialogue_act": "acknowledge",
            "followup_allowed": True,
        }
    )
    await app.drain()
    assert await scalar(app, "SELECT COUNT(*) FROM followups") == 0
    await seed(app, "Поговорим о музыке.", number=2)
    assert await scalar(app, "SELECT COUNT(*) FROM followups") == 0


async def test_followup_reads_current_memory_revision(app: Application) -> None:
    enable(app)
    await seed(app)

    async def change(tx):
        await tx.execute("UPDATE conversations SET memory_revision=memory_revision+1")

    await app.database.write(change)
    proposal(app)
    await due(app)
    await app.drain()
    assert len(app.sender.sent) == 2
    assert await scalar(
        app,
        "SELECT memory_revision FROM outbox_messages WHERE json_extract(proposal_json,'$.origin')='followup'",
    ) == await scalar(app, "SELECT memory_revision FROM conversations")


async def test_inconsistent_skip_is_retried_as_structural_error(
    app: Application,
) -> None:
    enable(app)
    await seed(app)
    app.model.responses["followup"].append(
        {
            "reply": "Мне нравятся парки.",
            "dialogue_act": "opinion",
            "should_send": False,
        }
    )
    proposal(app)
    await due(app)
    await app.drain()
    assert len(app.sender.sent) == 2
    assert len([task for task, _ in app.model.calls if task == "followup"]) == 2
    metrics = await app.metrics()
    assert metrics["reply_latency_samples"] == 1
    assert metrics["reply_completion_samples"] == 1


async def test_restore_cancels_timer_without_resurrecting_forgotten_data(
    app: Application,
) -> None:
    from app.shared.operations import ProcessLock, create_backup, restore_backup

    enable(app)
    await seed(app)
    backup = await create_backup(
        app.database, app.memory.deletion_path, app.clock.now()
    )
    await app.telegram.ingest([update(2, "/forget")])
    await app.drain()
    await app.close()
    with ProcessLock(app.settings.storage.directory):
        restore_backup(backup, app.settings.storage.directory)
    restarted = await Application.open(
        app.settings, FakeSender(), ScriptedModel(), app.clock
    )
    try:
        restarted.clock.advance(100)
        await restarted.drain()
        assert await scalar(restarted, "SELECT COUNT(*) FROM followups") == 0
        assert await scalar(restarted, "SELECT COUNT(*) FROM messages") == 0
        assert restarted.sender.sent == []
    finally:
        await restarted.close()


async def test_semantic_rejection_has_one_independently_checked_repair(
    app: Application,
) -> None:
    enable(app)
    await seed(app)
    proposal(app)
    app.model.responses["validate"].append(
        {"decision": "reject", "reason_codes": ["UNWANTED_PRESSURE"]}
    )
    app.model.responses["repair"].append(
        {
            "reply": "Мне нравятся парки без аттракционов.",
            "dialogue_act": "opinion",
            "should_send": True,
        }
    )
    await due(app)
    await app.drain()
    assert app.sender.sent[-1][1] == "Мне нравятся парки без аттракционов"
    assert [task for task, _ in app.model.calls][-4:] == [
        "followup",
        "validate",
        "repair",
        "validate_repair",
    ]


async def test_restore_cancels_pending_followup(app: Application) -> None:
    from app.shared.operations import ProcessLock, create_backup, restore_backup

    enable(app)
    await seed(app)
    backup = await create_backup(
        app.database, app.memory.deletion_path, app.clock.now()
    )
    await app.close()
    with ProcessLock(app.settings.storage.directory):
        restore_backup(backup, app.settings.storage.directory)
    restarted = await Application.open(
        app.settings, FakeSender(), ScriptedModel(), app.clock
    )
    try:
        assert await scalar(restarted, "SELECT status FROM followups") == "cancelled"
        restarted.clock.advance(100)
        await restarted.drain()
        assert restarted.sender.sent == []
    finally:
        await restarted.close()
