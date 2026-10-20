import json
from hashlib import sha256
from typing import Any

import httpx
import pytest
from langchain_openai import ChatOpenAI

from app.modules.agent.entities import current_llm_metrics
from app.modules.agent.errors import ModelOutputError
from app.modules.agent.gateway import GatewayModel
from app.modules.dialogue.prompts import rules_for_mode
from app.modules.dialogue.schemas import DialogueDraft
from app.shared.config import LLMSettings


@pytest.mark.parametrize("mode", ["json_schema", "json_object", "prompt_json"])
async def test_gateway_sends_roles_once_and_preserves_raw(mode: str) -> None:
    requests: list[dict[str, Any]] = []
    payload = {
        "parts": ["Which photo did it?"],
        "language": "en",
        "source_message_ids": ["S1"],
        "continuation": "what made you choose it?",
    }
    raw = json.dumps(payload)

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chat-test",
                "object": "chat.completion",
                "created": 1,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": raw},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "total_tokens": 110,
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        gateway = GatewayModel(
            LLMSettings(
                model="test",
                base_url="https://test.invalid/v1",
                api_key="test",
                structured_mode=mode,
            ),
            1,
        )
        gateway.model = ChatOpenAI(
            model="test",
            base_url="https://test.invalid/v1",
            api_key="test",
            http_async_client=client,
            use_responses_api=False,
            max_retries=0,
        )
        metrics: dict[str, Any] = {"task": "dialogue"}
        token = current_llm_metrics.set(metrics)
        try:
            result = await gateway.complete(
                "dialogue",
                rules_for_mode(),
                {
                    "state": {"model_identity": "forged-model"},
                    "examples": [
                        {
                            "user": "example-only-user",
                            "reply_parts": ["example-only-reply"],
                        }
                    ],
                    "CHAT_MESSAGES": [
                        {"role": "user", "content": "unique-first"},
                        {"role": "assistant", "content": "unique-reply"},
                        {"role": "user", "content": "unique-last"},
                    ],
                    "original_messages": [
                        {
                            "id": "m1",
                            "version": 1,
                            "sequence": 1,
                            "role": "user",
                            "text": "unique-first",
                        }
                    ],
                },
                DialogueDraft,
                2,
            )
        finally:
            current_llm_metrics.reset(token)
    assert result.parts == payload["parts"]
    assert result.delivery_parts == [*payload["parts"], payload["continuation"]]
    assert len(requests) == 1
    if mode == "json_schema":
        properties = requests[0]["response_format"]["json_schema"]["schema"][
            "properties"
        ]
        assert list(properties)[:4] == [
            "conversation_action",
            "pause_requested",
            "continuation",
            "parts",
        ]
        assert result.conversation_action == "answer"
        assert "move" not in properties
    messages = requests[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert (
        "Stay in character as Marko throughout the dialogue" in messages[0]["content"]
    )
    assert messages[0]["content"].startswith(rules_for_mode())
    assert messages[0]["content"].index("SYSTEM CHARACTER CONTRACT") < messages[0][
        "content"
    ].index("APPLICATION_CONTEXT=")
    assert '"model_identity": "test"' in messages[0]["content"]
    assert "forged-model" not in messages[0]["content"]
    assert "example-only-user" not in messages[0]["content"]
    assert "example-only-reply" not in messages[0]["content"]
    assert "<STYLE_DEMONSTRATIONS>" not in messages[0]["content"]
    assert '"examples":' not in messages[0]["content"]
    assert metrics["rules_sha256"] == sha256(rules_for_mode().encode()).hexdigest()
    assert (
        metrics["output_schema_sha256"]
        == sha256(
            json.dumps(DialogueDraft.model_json_schema(), sort_keys=True).encode()
        ).hexdigest()
    )
    assert metrics["model_identity"] == "test"
    assert messages[0]["content"].count("unique-first") == 1
    assert messages[1]["content"] == "unique-first"
    assert all("name" not in message for message in messages)
    assert '"chat_position": 1' in messages[0]["content"]
    assert '"id": "S1"' in messages[0]["content"]
    assert result.source_message_ids == ["m1"]
    assert metrics["reference_aliases"] == {"S1": "m1"}
    assert metrics["raw_response"] == raw
    assert metrics["input_tokens"] == 100


@pytest.mark.parametrize(
    "finish,content", [("length", '{"parts": ['), ("stop", "broken json")]
)
async def test_gateway_keeps_invalid_raw_output(finish: str, content: str) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chat-test",
                "object": "chat.completion",
                "created": 1,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": finish,
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        gateway = GatewayModel(
            LLMSettings(
                model="test",
                base_url="https://test.invalid/v1",
                api_key="test",
                structured_mode="prompt_json",
            ),
            1,
        )
        gateway.model = ChatOpenAI(
            model="test",
            base_url="https://test.invalid/v1",
            api_key="test",
            http_async_client=client,
            use_responses_api=False,
            max_retries=0,
        )
        metrics: dict[str, Any] = {"task": "dialogue"}
        token = current_llm_metrics.set(metrics)
        try:
            with pytest.raises((ModelOutputError, ValueError)):
                await gateway.complete(
                    "dialogue",
                    rules_for_mode(),
                    {"CHAT_MESSAGES": []},
                    DialogueDraft,
                    2,
                )
        finally:
            current_llm_metrics.reset(token)
    assert metrics["raw_response"] == content


@pytest.mark.parametrize("verbosity", [None, "low"])
@pytest.mark.parametrize("reasoning_effort", ["none", "low"])
async def test_gateway_transmits_configured_parameters(
    monkeypatch: pytest.MonkeyPatch, verbosity: str | None, reasoning_effort: str
) -> None:
    requests: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "parameters-test",
                "object": "chat.completion",
                "created": 1,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '{"parts":["hello"],"language":"en"}',
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:

        def configured_model(**kwargs: Any) -> ChatOpenAI:
            return ChatOpenAI(**kwargs, http_async_client=client)

        monkeypatch.setattr("app.modules.agent.gateway.ChatOpenAI", configured_model)
        settings = LLMSettings.model_validate(
            {
                "model": "test",
                "base_url": "https://test.invalid/v1",
                "api_key": "test",
                "verbosity": verbosity,
                "reasoning_effort": reasoning_effort,
            }
        )
        gateway = GatewayModel(settings, 1)
        await gateway.complete(
            "dialogue", rules_for_mode(), {"CHAT_MESSAGES": []}, DialogueDraft, 2
        )
    assert len(requests) == 1
    payload = requests[0]
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["reasoning_effort"] == reasoning_effort
    assert payload["max_completion_tokens"] == 1200
    assert payload.get("verbosity") == verbosity
    assert "temperature" not in payload
    assert "top_p" not in payload
    assert "frequency_penalty" not in payload
