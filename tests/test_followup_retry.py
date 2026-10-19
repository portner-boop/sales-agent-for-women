import json
from collections.abc import Awaitable, Callable

import pytest

from app.application import Application
from app.modules.followups.entities import FollowupTicket
from app.modules.jobs.entities import JobRetryError
from app.shared.storage import Transaction
from tests.helpers import scalar, update
from tests.test_followups import due, enable, seed


async def retry_due(app: Application) -> None:
    timer = await scalar(
        app,
        "SELECT MIN(run_at) FROM scheduled_jobs WHERE kind='followup' AND status='pending'",
    )
    app.clock.advance(max(0, timer - app.clock.now()))


async def test_rejected_followup_retries_same_ticket_and_delivers_once(
    app: Application,
) -> None:
    enable(app)
    await seed(app, "I enjoy baking")
    for task in ("followup", "repair", "recover"):
        app.model.responses[task].append(
            {
                "reply": "What would help you trust me?",
                "question_goal": "trust_requirements",
                "dialogue_act": "question",
                "should_send": True,
            }
        )
    for task in ("validate", "validate_repair", "validate_recovery"):
        app.model.responses[task].append(
            {"decision": "reject", "reason_codes": ["VERIFICATION_TOPIC_LOOP"]}
        )
    await due(app)
    await app.drain()
    assert len(app.sender.sent) == 1
    assert await scalar(app, "SELECT status FROM followups") == "pending"
    assert await scalar(app, "SELECT reason FROM followups") == "ReplyRejectedError"
    assert (
        await scalar(app, "SELECT status FROM scheduled_jobs WHERE kind='followup'")
        == "pending"
    )
    app.model.responses["followup"].append(
        {
            "reply": "What do you most enjoy baking?",
            "question_goal": "baking_preference",
            "dialogue_act": "question",
            "should_send": True,
        }
    )
    await retry_due(app)
    await app.drain()
    assert app.sender.sent[-1] == (1, "What do you most enjoy baking?")
    assert await scalar(app, "SELECT COUNT(*) FROM followups") == 1
    assert await scalar(app, "SELECT status FROM followups") == "prepared"
    assert await scalar(app, "SELECT reason FROM followups") is None
    app.clock.advance(3600)
    await app.drain()
    assert len(app.sender.sent) == 2


@pytest.mark.parametrize("terminal", ["attempts", "deadline", "permanent"])
async def test_failed_followup_has_bounded_retries_and_keeps_failure_reason(
    app: Application, terminal: str
) -> None:
    enable(app)
    await seed(app)
    calls = 0

    async def fail(
        ticket: FollowupTicket,
        prepared: Callable[[Transaction], Awaitable[None]],
    ) -> bool:
        nonlocal calls
        calls += 1
        if terminal == "deadline":
            app.clock.advance(ticket.expires_at - app.clock.now())
        raise JobRetryError("GATEWAY_CONNECTION", permanent=terminal == "permanent")

    app.followups.processor = fail
    await due(app)
    await app.drain()
    if terminal == "attempts":
        for _ in range(2):
            await retry_due(app)
            await app.drain()
    assert calls == (3 if terminal == "attempts" else 1)
    assert await scalar(app, "SELECT status FROM followups") == "skipped"
    assert await scalar(app, "SELECT reason FROM followups") == "GATEWAY_CONNECTION"
    assert (
        await scalar(app, "SELECT status FROM scheduled_jobs WHERE kind='followup'")
        == "dead_letter"
    )
    app.clock.advance(3600)
    await app.drain()
    assert len(app.sender.sent) == 1


@pytest.mark.parametrize("text", ["I prefer quiet gardens", "/stop", "/forget"])
async def test_new_input_cancels_failed_followup_before_retry(
    app: Application, text: str
) -> None:
    enable(app)
    await seed(app)
    calls = 0

    async def fail(
        ticket: FollowupTicket,
        prepared: Callable[[Transaction], Awaitable[None]],
    ) -> bool:
        nonlocal calls
        calls += 1
        raise JobRetryError("FOLLOWUP_DEADLINE_EXCEEDED")

    app.followups.processor = fail
    await due(app)
    await app.drain()
    await app.telegram.ingest([update(2, text)])
    assert (
        await scalar(
            app,
            "SELECT COUNT(*) FROM scheduled_jobs WHERE kind='followup' AND status='pending'",
        )
        == 0
    )
    assert calls == 1


@pytest.mark.parametrize("mode", ["repair", "service", "continuation"])
async def test_verification_answer_has_immediate_personal_initiative(
    app: Application, mode: str
) -> None:
    enable(app)
    app.model.responses["interpret"].append(
        {
            "intent": "identity_question",
            "dialogue_mode": mode,
            "policy_topics": ["verification"],
            "language": "en",
        }
    )
    app.model.responses["generate"].append(
        {
            "reply": "I'm Marko",
            "continuation": "What do you enjoy baking?",
            "dialogue_act": "answer_and_question",
            "question_goal": "baking_preference",
            "question_part": "continuation",
        }
    )
    await app.telegram.ingest([update(1, "I like baking. Are you a bot?")])
    await app.drain()
    app.clock.advance(4)
    await app.drain()
    context = next(data for task, data in app.model.calls if task == "generate")
    assert context["DIALOGUE_PLAN"]["initiative_due"] is True
    assert context["DIALOGUE_PLAN"]["strategy"] == "develop_conversation_naturally"
    assert [text for _, text in app.sender.sent] == [
        "I'm Marko",
        "What do you enjoy baking?",
    ]
    assert await scalar(app, "SELECT COUNT(*) FROM followups") == 0
    metrics = json.loads(await scalar(app, "SELECT metrics_json FROM agent_runs"))
    assert not any(
        event["stage"] == "followup_failure" for event in metrics["validation_events"]
    )


async def test_repeated_recovery_question_keeps_checked_answer_and_silence_timer(
    app: Application,
) -> None:
    enable(app)
    app.model.responses["generate"].append(
        {
            "reply": "What music do you enjoy?",
            "dialogue_act": "question",
            "question_goal": "music_preference",
        }
    )
    await app.telegram.ingest([update(1, "Hello")])
    await app.drain()
    for task in ("generate", "repair", "recover"):
        app.model.responses[task].append(
            {
                "reply": "Jazz has a lovely rhythm",
                "continuation": "What music do you enjoy?",
                "dialogue_act": "answer_and_question",
                "question_goal": "music_preference",
                "question_part": "continuation",
            }
        )
    app.model.responses["recover"].append(
        {
            "reply": "Jazz has a lovely rhythm",
            "continuation": "What draws you to jazz?",
            "dialogue_act": "answer_and_question",
            "question_goal": "jazz_appeal",
            "question_part": "continuation",
        }
    )
    await app.telegram.ingest([update(2, "I like jazz")])
    await app.drain()
    assert [text for _, text in app.sender.sent] == [
        "What music do you enjoy?",
        "Jazz has a lovely rhythm",
    ]
    checked = next(
        data for task, data in app.model.calls if task == "validate_recovery"
    )
    assert checked["reply"]["reply"] == "Jazz has a lovely rhythm"
    assert checked["reply"]["continuation"] is None
    assert checked["reply"]["question_goal"] is None
    assert checked["context"]["DIALOGUE_PLAN"]["initiative_due"] is False
    assert checked["context"]["DIALOGUE_PLAN"]["requires_continuation"] is False
    assert await scalar(app, "SELECT COUNT(*) FROM followups") == 1
    app.clock.advance(4)
    await app.drain()
    assert app.sender.sent[-1] == (1, "Jazz has a lovely rhythm")
    assert not any("APPROVED_ANSWER" in data for _, data in app.model.calls)
    assert not any(task == "followup" for task, _ in app.model.calls)


async def test_repeated_question_in_main_reply_is_not_trimmed_to_bypass_validation(
    app: Application,
) -> None:
    enable(app)
    app.model.responses["generate"].append(
        {
            "reply": "What music do you enjoy?",
            "dialogue_act": "question",
            "question_goal": "music_preference",
        }
    )
    await app.telegram.ingest([update(1, "Hello")])
    await app.drain()
    for task in ("generate", "repair", "recover"):
        app.model.responses[task].append(
            {
                "reply": "What music do you enjoy?",
                "continuation": "Jazz has a lovely rhythm",
                "dialogue_act": "question_and_opinion",
                "question_goal": "music_preference",
                "question_part": "reply",
            }
        )
    await app.telegram.ingest([update(2, "I like jazz")])
    await app.turn_worker.run_once()
    assert app.sender.sent == [(1, "What music do you enjoy?")]
    metrics = json.loads(
        await scalar(
            app, "SELECT metrics_json FROM agent_runs ORDER BY rowid DESC LIMIT 1"
        )
    )
    assert not any(
        event["stage"] == "recovery_continuation_removed"
        for event in metrics["validation_events"]
    )
    assert not any(task == "validate_recovery" for task, _ in app.model.calls)
