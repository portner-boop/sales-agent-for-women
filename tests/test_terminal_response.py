import pytest

from app.application import Application
from app.modules.agent.recovery import service_failure_reply
from tests.helpers import conversation, interpretation_operation, scalar, update
from tests.test_client_fallback_recovery import advance_retry


@pytest.mark.parametrize(
    "text",
    [
        "I’m still here. I’m having trouble answering that right now",
        "Я здесь. Сейчас не получается ответить на твоё сообщение по существу",
        "Sigo aquí. Ahora mismo me está costando responder a tu mensaje",
        "Continuo aqui. Neste momento estou com dificuldade para responder à sua mensagem",
        "Je suis toujours là. J’ai du mal à répondre à ton message pour le moment",
        "Ich bin noch da. Gerade fällt es mir schwer, auf deine Nachricht zu antworten",
        "Sono ancora qui. Al momento ho difficoltà a rispondere al tuo messaggio",
        "ما زلت هنا. أواجه صعوبة في الرد على رسالتك الآن",
        "ここにいるよ。今はメッセージにうまく答えられないんだ",
    ],
)
def test_retired_failure_notices_are_rejected_in_every_existing_language(
    text: str,
) -> None:
    assert service_failure_reply(text)


def test_presence_alone_is_not_a_failure_notice() -> None:
    assert not service_failure_reply("I’m still here. What music do you enjoy?")


@pytest.mark.parametrize("questions_allowed", [True, False])
async def test_exhausted_validation_keeps_failures_out_of_the_conversation(
    app: Application, questions_allowed: bool
) -> None:
    text = (
        "Tell me about yourself"
        if questions_allowed
        else "No questions. Tell me about yourself"
    )
    await app.telegram.ingest([update(1, text)])
    current = await conversation(app)
    original = (await app.conversations.messages(current.scope))[-1]
    if not questions_allowed:
        app.model.responses["interpret"].append(
            interpretation_operation(original, "question_permission", False, "boundary")
        )
    for task in ("generate", "repair", "recover"):
        app.model.responses[task].extend(
            [{"reply": "I went sailing today", "dialogue_act": "answer"}] * 3
        )
    for task in ("validate", "validate_repair", "validate_recovery"):
        app.model.responses[task].extend(
            [{"decision": "reject", "reason_codes": ["FABRICATED_EVENT"]}] * 3
        )
    await app.drain()
    await advance_retry(app)
    await advance_retry(app)
    assert not app.sender.sent
    assert await scalar(app, "SELECT pending FROM messages WHERE role='user'") == 0
    assert await scalar(app, "SELECT COUNT(*) FROM outbox_messages") == 0
    assert (
        await scalar(app, "SELECT status FROM agent_runs ORDER BY rowid DESC LIMIT 1")
        == "unanswered"
    )
    assert (
        await scalar(app, "SELECT status FROM scheduled_jobs WHERE kind='turn'")
        == "dead_letter"
    )
    assert (await conversation(app)).first_reply_at is None
    app.clock.advance(3600)
    await app.drain()
    assert not app.sender.sent


@pytest.mark.parametrize("reconcile", [False, True])
async def test_expired_pending_input_logs_failure_without_client_notice(
    app: Application, reconcile: bool
) -> None:
    await app.telegram.ingest([update(1, "What do you remember about me?")])
    app.clock.advance(181)
    if reconcile:
        await app.jobs.reconcile_turns(app.clock.now())
    await app.drain()
    assert not app.sender.sent
    assert not app.model.calls
    assert await scalar(app, "SELECT status FROM agent_runs") == "unanswered"
    assert await scalar(app, "SELECT error_code FROM agent_runs") == "RESPONSE_DEADLINE"
    assert (
        await scalar(app, "SELECT status FROM scheduled_jobs WHERE kind='turn'")
        == "dead_letter"
    )
    await app.jobs.reconcile_turns(app.clock.now())
    await app.drain()
    assert not app.sender.sent


async def test_recovered_exhausted_lease_never_sends_a_technical_notice(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "hello")])
    for _ in range(3):
        assert await app.jobs.claim(app.clock.now(), ("turn",)) is not None
        app.clock.advance(121)
    await app.jobs.reconcile_turns(app.clock.now())
    await app.drain()
    assert not app.sender.sent
    assert not app.model.calls
    assert await scalar(app, "SELECT pending FROM messages WHERE role='user'") == 0


@pytest.mark.parametrize("command", ["/stop", "/forget"])
async def test_command_does_not_revive_an_expired_failure(
    app: Application, command: str
) -> None:
    await app.telegram.ingest([update(1, "hello")])
    app.clock.advance(181)
    await app.turn_worker.run_once()
    assert (
        await scalar(app, "SELECT COUNT(*) FROM outbox_messages WHERE status='ready'")
        == 0
    )
    await app.telegram.ingest([update(2, command)])
    await app.drain()
    assert not any("I’m still here" in text for _, text in app.sender.sent)
