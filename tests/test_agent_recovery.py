import asyncio
import json

import httpx
import pytest
from openai import APIConnectionError, APIStatusError

from app.application import Application
from app.modules.agent.errors import ModelOutputError
from app.modules.agent.recovery import retry_delay
from tests.helpers import conversation, interpretation_operation, scalar, update


def gateway_error(status: int, retry_after: str | None = None) -> APIStatusError:
    response = httpx.Response(
        status,
        request=httpx.Request("POST", "https://gateway.example/v1/chat/completions"),
        headers={"retry-after": retry_after} if retry_after is not None else {},
    )
    return APIStatusError("test failure", response=response, body=None)


async def metrics(app: Application) -> dict:
    return json.loads(await scalar(app, "SELECT metrics_json FROM agent_runs"))


async def test_semantic_repair_is_independently_checked_on_fifth_call(
    app: Application,
) -> None:
    app.model.responses["generate"].append(
        {"reply": "Я сегодня гулял в парке.", "dialogue_act": "chat"}
    )
    app.model.responses["validate"].append(
        {
            "decision": "reject",
            "reason_codes": ["INVENTED_PERSONA_CIRCUMSTANCE"],
            "violating_text": "Я сегодня гулял в парке",
            "explanation": "An actual off-screen walk is not supported.",
        }
    )
    app.model.responses["repair"].append(
        {"reply": "Я бы выбрал тихое место.", "dialogue_act": "opinion"}
    )
    await app.telegram.ingest([update(1, "Отдохнула в парке.")])
    await app.drain()
    assert [name for name, _ in app.model.calls] == [
        "interpret",
        "generate",
        "validate",
        "repair",
        "validate_repair",
    ]
    assert app.sender.sent == [(1, "Я бы выбрал тихое место")]
    assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"
    events = (await metrics(app))["validation_events"]
    assert any(
        event.get("proposal", {}).get("reply") == "Я сегодня гулял в парке."
        for event in events
    )
    assert any(
        event.get("violating_text") == "Я сегодня гулял в парке" for event in events
    )


async def test_memory_critic_does_not_prevent_sixth_call_repair_check(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "My usual coffee has honey.")])
    current = await conversation(app)
    message = (await app.conversations.messages(current.scope))[0]
    app.model.responses["interpret"].append(interpretation_operation(message))
    app.model.responses["generate"].append(
        {"reply": "I drank honey coffee today.", "dialogue_act": "chat"}
    )
    app.model.responses["validate"].append(
        {"decision": "reject", "reason_codes": ["INVENTED_PERSONA_CIRCUMSTANCE"]}
    )
    app.model.responses["repair"].append(
        {"reply": "I like that idea.", "dialogue_act": "opinion"}
    )
    await app.drain()
    assert [name for name, _ in app.model.calls] == [
        "interpret",
        "validate_memory",
        "generate",
        "validate",
        "repair",
        "validate_repair",
    ]
    assert app.sender.sent == [(1, "I like that idea")]
    assert len(await app.memory.repository.items(current.scope)) == 1


async def test_rejected_repair_gets_checked_conservative_recovery(
    app: Application,
) -> None:
    for task in ["generate", "repair"]:
        app.model.responses[task].append(
            {"reply": "I walked there today.", "dialogue_act": "chat"}
        )
    for task in ["validate", "validate_repair"]:
        app.model.responses[task].append(
            {"decision": "reject", "reason_codes": ["INVENTED_PERSONA_CIRCUMSTANCE"]}
        )
    app.model.responses["recover"].append(
        {"reply": "I prefer quieter parks.", "dialogue_act": "opinion"}
    )
    await app.telegram.ingest([update(1, "I sat in the park.")])
    await app.drain()
    assert app.sender.sent == [(1, "I prefer quieter parks")]
    assert [task for task, _ in app.model.calls][-2:] == [
        "recover",
        "validate_recovery",
    ]
    assert (await metrics(app))["llm_calls"] == 7


async def test_rejected_recovery_never_leaks_unvalidated_draft(
    app: Application,
) -> None:
    for task in ["generate", "repair", "recover"]:
        app.model.responses[task].append(
            {"reply": "I walked there today.", "dialogue_act": "chat"}
        )
    for task in ["validate", "validate_repair", "validate_recovery"]:
        app.model.responses[task].append(
            {"decision": "reject", "reason_codes": ["INVENTED_PERSONA_CIRCUMSTANCE"]}
        )
    await app.telegram.ingest([update(1, "I sat in the park.")])
    await app.drain()
    assert all("walked" not in text for _, text in app.sender.sent)
    assert (await conversation(app)).first_reply_at is None
    assert await scalar(app, "SELECT COUNT(*) FROM outbox_messages") == 0
    assert await scalar(app, "SELECT status FROM agent_runs") == "deferred"
    assert len(app.model.calls) == 7


async def test_missing_continuation_is_retried_with_structure_feedback(
    app: Application,
) -> None:
    app.model.responses["generate"].extend(
        [
            {"reply": "Нормально.", "dialogue_act": "answer"},
            {
                "reply": "Нормально.",
                "continuation": "Тихие места мне нравятся.",
                "dialogue_act": "answer_and_opinion",
            },
        ]
    )
    await app.telegram.ingest([update(1, "Посидела в парке. А ты как?")])
    await app.drain()
    app.clock.advance(4)
    await app.drain()
    assert app.sender.sent == [(1, "Нормально"), (1, "Тихие места мне нравятся")]
    requests = [data for name, data in app.model.calls if name == "generate"]
    assert requests[1]["STRUCTURE_ERRORS"] == ["continuation:missing"]
    assert (await metrics(app))["llm_retries"] == 1


@pytest.mark.parametrize(
    "error",
    [
        ModelOutputError("TRUNCATED_OUTPUT"),
        gateway_error(503),
        gateway_error(429, "0"),
        APIConnectionError(request=httpx.Request("POST", "https://gateway.example")),
    ],
)
async def test_transient_gateway_failure_is_retried_once(
    app: Application, error: Exception
) -> None:
    app.model.responses["generate"].append(error)
    await app.telegram.ingest([update(1, "hello")])
    await app.drain()
    assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"
    assert (await metrics(app))["llm_retries"] == 1
    assert [task for task, _ in app.model.calls] == [
        "interpret",
        "generate",
        "generate",
    ]


@pytest.mark.parametrize("status", [400, 401, 403, 404])
async def test_permanent_gateway_failure_is_not_retried(
    app: Application, status: int
) -> None:
    app.model.responses["interpret"].append(gateway_error(status))
    await app.telegram.ingest([update(1, "hello")])
    await app.drain()
    assert len(app.model.calls) == 1
    assert (await conversation(app)).first_reply_at is None
    assert (
        await scalar(app, "SELECT error_code FROM agent_runs")
        == f"GATEWAY_HTTP_{status}"
    )
    assert not app.sender.sent
    assert await scalar(app, "SELECT status FROM scheduled_jobs") == "dead_letter"


async def test_rate_limit_longer_than_deadline_is_not_retried_early(
    app: Application,
) -> None:
    app.model.responses["interpret"].append(gateway_error(429, "60"))
    await app.telegram.ingest([update(1, "hello")])
    await app.drain()
    assert len(app.model.calls) == 1
    assert await scalar(app, "SELECT error_code FROM agent_runs") == "GATEWAY_HTTP_429"


async def test_repeated_timeout_is_bounded(app: Application) -> None:
    app.model.responses["interpret"].extend([TimeoutError(), TimeoutError()])
    await app.telegram.ingest([update(1, "hello")])
    await app.drain()
    assert len(app.model.calls) == 2
    assert (await metrics(app))["llm_retries"] == 1
    assert await scalar(app, "SELECT error_code FROM agent_runs") == "DEADLINE_EXCEEDED"


async def test_new_input_cancels_retry_of_old_input(app: Application) -> None:
    app.model.responses["generate"].append(gateway_error(503))
    await app.telegram.ingest([update(1, "old input")])
    task = asyncio.create_task(app.turn_worker.run_once())
    await app.model.entered.wait()
    await app.telegram.ingest([update(2, "new input")])
    await task
    assert not app.sender.sent
    assert await scalar(app, "SELECT status FROM agent_runs") == "cancelled"
    assert len(app.model.calls) == 2
    await app.drain()
    assert len(app.sender.sent) == 1


async def test_forget_removes_rejected_proposal_diagnostics(app: Application) -> None:
    app.model.responses["generate"].append(
        {"reply": "Private marker — rejected", "dialogue_act": "chat"}
    )
    await app.telegram.ingest([update(1, "hello")])
    await app.drain()
    assert "Private marker" in await scalar(app, "SELECT metrics_json FROM agent_runs")
    await app.telegram.ingest([update(2, "/forget")])
    await app.drain()
    assert await scalar(app, "SELECT COUNT(*) FROM agent_runs") == 0


def test_retry_after_invalid_value_disables_retry() -> None:
    assert retry_delay(gateway_error(429, "invalid")) is None


async def test_queue_wait_does_not_consume_execution_deadline(app: Application) -> None:
    await app.telegram.ingest([update(1, "hello")])
    app.clock.advance(120)
    await app.drain()
    assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"
    assert (await metrics(app))["queued_seconds"] == 120
    original = (await app.conversations.messages((await conversation(app)).scope))[0]
    assert original.received_at == 1900000000
    assert (await app.agent.repository.metrics())["reply_latency_p95_seconds"] >= 120


async def test_call_budget_is_distinct_from_timeout(app: Application) -> None:
    await app.telegram.ingest([update(1, "My usual coffee has honey.")])
    current = await conversation(app)
    message = (await app.conversations.messages(current.scope))[0]
    app.model.responses["interpret"].extend(
        [ModelOutputError("INVALID_JSON"), interpretation_operation(message)]
    )
    for task in ["generate", "repair"]:
        app.model.responses[task].append(
            {"reply": "I walked there today.", "dialogue_act": "chat"}
        )
    for task in ["validate", "validate_repair"]:
        app.model.responses[task].append(
            {"decision": "reject", "reason_codes": ["INVENTED_PERSONA_CIRCUMSTANCE"]}
        )
    await app.drain()
    assert (
        await scalar(app, "SELECT error_code FROM agent_runs")
        == "CALL_BUDGET_EXHAUSTED"
    )
    assert len(app.model.calls) == 7
    assert all("walked" not in text for _, text in app.sender.sent)


async def test_total_transport_retries_are_bounded(app: Application) -> None:
    for task in ["interpret", "generate", "validate"]:
        app.model.responses[task].append(ModelOutputError("INVALID_JSON"))
    app.model.responses["generate"].append(
        {"reply": "I prefer quiet parks.", "dialogue_act": "opinion"}
    )
    await app.telegram.ingest([update(1, "hello")])
    await app.drain()
    trace = await metrics(app)
    assert trace["llm_retries"] == 2
    assert trace["llm_calls"] <= 8
    assert [task for task, _ in app.model.calls].count("validate") == 1


async def test_repair_request_is_fitted_with_its_actual_payload(
    app: Application,
) -> None:
    for number in range(1, 9):
        await app.telegram.ingest(
            [update(number, f"Story {number}. " + "flower " * 1000)]
        )
        await app.turn_worker.run_once()
        await app.delivery.run_once()
    app.model.responses["generate"].append(
        {
            "reply": "I have a thoughtful opinion about this topic. " * 80,
            "dialogue_act": "chat",
        }
    )
    app.model.responses["validate"].append(
        {
            "decision": "reject",
            "reason_codes": ["POLISHED_SMALL_TALK"],
            "violating_text": "I have a thoughtful opinion about this topic. " * 21,
            "explanation": "The reply repeats an unrelated supportive opinion. " * 9,
        }
    )
    app.model.responses["repair"].append(
        {"reply": "That was unexpected.", "dialogue_act": "reaction"}
    )
    await app.telegram.ingest([update(9, "That happened yesterday.")])
    await app.turn_worker.run_once()
    await app.delivery.run_once()
    assert app.sender.sent[-1] == (1, "That was unexpected")
    async with app.database.read() as tx:
        row = await tx.execute(
            "SELECT manifest_json FROM agent_runs ORDER BY rowid DESC LIMIT 1"
        )
        manifest = json.loads((await row.fetchone())[0])
    assert manifest["token_count_by_section"]["request_reserve_tokens"] > 0
    repair = next(data for task, data in app.model.calls if task == "repair")
    assert repair["HISTORY_STATUS"]["omitted_uncompressed_messages"] > 0
