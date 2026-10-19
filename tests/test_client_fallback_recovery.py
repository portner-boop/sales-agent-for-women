import json

import pytest

from app.application import Application
from app.modules.agent.recovery import retry_delay
from tests.helpers import conversation, scalar, update
from tests.test_agent_recovery import gateway_error


async def advance_retry(app: Application) -> None:
    ready = await scalar(
        app,
        "SELECT MIN(run_at) FROM scheduled_jobs WHERE status='pending' AND kind='turn'",
    )
    assert ready is not None
    app.clock.advance(max(0, ready - app.clock.now()))
    await app.drain()


async def test_failed_answer_retries_original_without_client_error(
    app: Application,
) -> None:
    app.model.responses["interpret"].append(RuntimeError("unavailable"))
    answer = "Yes, I’m an AI chatbot"
    app.model.responses["generate"].append({"reply": answer, "dialogue_act": "answer"})
    await app.telegram.ingest([update(1, "Hi are you bot")])
    before = await conversation(app)
    original = (await app.conversations.messages(before.scope))[0]
    await app.drain()
    assert not app.sender.sent
    assert await scalar(app, "SELECT COUNT(*) FROM outbox_messages") == 0
    assert await scalar(app, "SELECT status FROM agent_runs") == "deferred"
    assert await scalar(app, "SELECT pending FROM messages") == 1
    assert await scalar(app, "SELECT error_code FROM scheduled_jobs") == "RuntimeError"
    await advance_retry(app)
    assert app.sender.sent == [(1, answer)]
    current = (await app.conversations.messages(before.scope))[0]
    assert (current.id, current.version, current.received_at, current.text) == (
        original.id,
        original.version,
        original.received_at,
        original.text,
    )
    assert await scalar(app, "SELECT COUNT(*) FROM messages WHERE role='user'") == 1
    assert await scalar(app, "SELECT COUNT(*) FROM said_ledger") == 1
    assert await scalar(app, "SELECT status FROM scheduled_jobs") == "done"


async def test_retry_survives_restart_without_duplicate_user_input(
    app: Application,
) -> None:
    app.model.responses["interpret"].append(RuntimeError("unavailable"))
    answer = "Quiet music sounds good"
    app.model.responses["generate"].append({"reply": answer, "dialogue_act": "opinion"})
    await app.telegram.ingest([update(1, "What music do you prefer?")])
    await app.drain()
    await app.close()
    restarted = await Application.open(app.settings, app.sender, app.model, app.clock)
    try:
        restarted.clock.advance(60)
        await restarted.drain()
        assert restarted.sender.sent == [(1, answer)]
        assert (
            await scalar(restarted, "SELECT COUNT(*) FROM messages WHERE role='user'")
            == 1
        )
        assert await scalar(restarted, "SELECT COUNT(*) FROM agent_runs") == 2
        assert await scalar(restarted, "SELECT COUNT(*) FROM said_ledger") == 1
    finally:
        await restarted.close()


async def test_terminal_failure_does_not_poison_the_next_message(
    app: Application,
) -> None:
    app.model.responses["interpret"].extend([RuntimeError("unavailable")] * 3)
    await app.telegram.ingest([update(1, "old question")])
    await app.drain()
    await advance_retry(app)
    await advance_retry(app)
    assert not app.sender.sent
    assert await scalar(app, "SELECT COUNT(*) FROM agent_runs") == 3
    assert await scalar(app, "SELECT status FROM scheduled_jobs") == "dead_letter"
    assert (
        await scalar(app, "SELECT status FROM agent_runs ORDER BY rowid DESC LIMIT 1")
        == "unanswered"
    )
    assert await scalar(app, "SELECT pending FROM messages") == 0
    app.clock.advance(120)
    app.model.responses["generate"].append(
        {"reply": "I’m here", "dialogue_act": "answer"}
    )
    await app.telegram.ingest([update(2, "new question")])
    await app.drain()
    assert app.sender.sent[-1] == (1, "I’m here")
    assert len(app.sender.sent) == 1
    data = next(data for task, data in app.model.calls if task == "generate")
    assert [item["text"] for item in data["CURRENT_INPUT"]] == ["new question"]
    assert await scalar(app, "SELECT COUNT(*) FROM messages WHERE role='user'") == 2


async def test_retry_gets_fresh_first_reply_budget_after_queue_delay(
    app: Application,
) -> None:
    runtime = app.settings.runtime.model_copy(update={"reply_pacing_enabled": True})
    app.agent.settings = runtime
    app.model.responses["interpret"].extend([TimeoutError(), TimeoutError()])
    answer = "Yes, I’m an AI chatbot"
    app.model.responses["generate"].append({"reply": answer, "dialogue_act": "answer"})
    await app.telegram.ingest([update(1, "Are you a bot?")])
    await app.drain()
    assert not app.sender.sent
    app.clock.advance(120)
    await app.drain()
    app.clock.advance(10)
    await app.drain()
    assert app.sender.sent == [(1, answer)]
    assert (
        await scalar(
            app, "SELECT received_at FROM message_versions WHERE text='Are you a bot?'"
        )
        == 1900000000
    )
    assert (
        await scalar(app, "SELECT status FROM agent_runs ORDER BY rowid DESC LIMIT 1")
        == "prepared"
    )


async def test_new_input_joins_pending_retry_without_expired_budget(
    app: Application,
) -> None:
    runtime = app.settings.runtime.model_copy(update={"reply_pacing_enabled": True})
    app.agent.settings = runtime
    app.model.responses["interpret"].append(RuntimeError("unavailable"))
    answer = "Yes, I’m a chatbot. Quiet music is my preference"
    app.model.responses["generate"].append({"reply": answer, "dialogue_act": "answer"})
    await app.telegram.ingest([update(1, "Are you a bot?")])
    await app.drain()
    app.clock.advance(30)
    await app.telegram.ingest([update(2, "What music do you like?")])
    await app.drain()
    app.clock.advance(10)
    await app.drain()
    assert app.sender.sent == [(1, answer)]
    data = next(data for task, data in app.model.calls if task == "generate")
    assert [item["text"] for item in data["CURRENT_INPUT"]] == [
        "Are you a bot?",
        "What music do you like?",
    ]
    assert await scalar(app, "SELECT COUNT(*) FROM said_ledger") == 1


@pytest.mark.parametrize("command", ["/stop", "/forget"])
async def test_command_cancels_pending_reply_retry(
    app: Application, command: str
) -> None:
    app.model.responses["interpret"].append(RuntimeError("unavailable"))
    await app.telegram.ingest([update(1, "Are you a bot?")])
    await app.drain()
    await app.telegram.ingest([update(2, command)])
    await app.drain()
    sent = list(app.sender.sent)
    app.clock.advance(120)
    await app.drain()
    assert app.sender.sent == sent
    assert len(app.model.calls) == 1
    assert (await conversation(app)).control_state != "enabled" or command == "/forget"


async def test_kill_switch_blocks_pending_retry(app: Application) -> None:
    app.model.responses["interpret"].append(RuntimeError("unavailable"))
    await app.telegram.ingest([update(1, "Are you a bot?")])
    await app.drain()
    await app.delivery_repository.kill_switch(True)
    app.clock.advance(120)
    await app.drain()
    assert not app.sender.sent
    assert len(app.model.calls) == 1


@pytest.mark.parametrize("header", ["30", "nan", "inf", "invalid"])
async def test_delayed_retry_respects_gateway_retry_after(
    app: Application, header: str
) -> None:
    error = gateway_error(429, header)
    app.model.responses["interpret"].append(error)
    await app.telegram.ingest([update(1, "hello")])
    await app.drain()
    if header == "30":
        assert not app.sender.sent
        assert (
            await scalar(app, "SELECT run_at FROM scheduled_jobs")
            == app.clock.now() + 30
        )
        app.clock.advance(29)
        await app.drain()
        assert len(app.model.calls) == 1
        app.clock.advance(1)
        await app.drain()
        assert len(app.sender.sent) == 1
    else:
        assert retry_delay(error) is None
        assert not app.sender.sent
        assert await scalar(app, "SELECT status FROM scheduled_jobs") == "dead_letter"
        assert await scalar(app, "SELECT pending FROM messages") == 0


@pytest.mark.parametrize(
    "language,notice,answer",
    [
        (
            "en",
            "I’m still here. I’m having trouble answering that right now",
            "Quiet music is my preference",
        ),
        (
            "ru",
            "Я здесь. Сейчас не получается ответить на твоё сообщение по существу",
            "Мне нравится спокойная музыка",
        ),
        (
            "en",
            "I couldn’t give you a proper reply. Let’s try again",
            "Quiet music is my preference",
        ),
        (
            "en",
            "I couldn't give you a proper reply. Let's try again",
            "Quiet music is my preference",
        ),
        (
            "ru",
            "Не смог ответить как следует. Давай попробуем ещё раз",
            "Мне нравится спокойная музыка",
        ),
        (
            "ru",
            "Сейчас не удалось подготовить ответ. Попробуйте чуть позже",
            "Мне нравится спокойная музыка",
        ),
        (
            "es",
            "No he podido responderte bien. Intentémoslo de nuevo",
            "Me gusta la música tranquila",
        ),
        (
            "pt",
            "Não consegui responder direito. Vamos tentar de novo",
            "Gosto de música tranquila",
        ),
    ],
)
async def test_model_cannot_repeat_legacy_technical_failure_template(
    app: Application, language: str, notice: str, answer: str
) -> None:
    app.model.responses["interpret"].append(
        {"language": language, "dialogue_mode": "continuation"}
    )
    app.model.responses["generate"].append({"reply": notice, "dialogue_act": "answer"})
    app.model.responses["repair"].append({"reply": answer, "dialogue_act": "answer"})
    question = {
        "en": "Tell me about music",
        "ru": "Расскажи про музыку",
        "es": "Háblame de música",
        "pt": "Fale sobre música",
    }[language]
    await app.telegram.ingest([update(1, question, language=language)])
    await app.drain()
    assert app.sender.sent == [(1, answer)]
    repair = next(data for task, data in app.model.calls if task == "repair")
    assert "SERVICE_FAILURE_REPLY" in repair["REPAIR_REASON_CODES"]
    assert await scalar(app, "SELECT COUNT(*) FROM outbox_messages") == 1
    metadata = json.loads(
        await scalar(app, "SELECT proposal_json FROM outbox_messages")
    )
    assert metadata["reply"] == answer
