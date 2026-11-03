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


def response_body(
    content: str,
    text_format: dict[str, Any] | None = None,
    status: str = "completed",
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": "resp-test",
        "object": "response",
        "created_at": 1,
        "model": "test",
        "status": status,
        "error": None,
        "incomplete_details": {"reason": reason} if reason else None,
        "output": [
            {
                "id": "msg-test",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": content, "annotations": []}
                ],
            }
        ],
        "text": {"format": text_format or {"type": "text"}},
        "usage": {
            "input_tokens": 100,
            "output_tokens": 10,
            "total_tokens": 110,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 4},
        },
    }


@pytest.mark.parametrize("mode", ["json_schema", "json_object", "prompt_json"])
async def test_gateway_sends_roles_once_and_preserves_raw(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    requests: list[dict[str, Any]] = []
    payload = {
        "parts": ["Which photo did it?"],
        "language": "en",
        "source_message_ids": ["S1"],
        "continuation": "what made you choose it?",
    }
    raw = json.dumps(payload)

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/responses"
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(
            200,
            json=response_body(raw, body.get("text", {}).get("format")),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:

        def configured_model(**kwargs: Any) -> ChatOpenAI:
            return ChatOpenAI(**kwargs, http_async_client=client)

        monkeypatch.setattr("app.modules.agent.gateway.ChatOpenAI", configured_model)
        gateway = GatewayModel(
            LLMSettings(
                model="test",
                base_url="https://test.invalid/v1",
                api_key="test",
                structured_mode=mode,
            ),
            1,
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
    assert requests[0]["store"] is False
    assert "previous_response_id" not in requests[0]
    assert "messages" not in requests[0]
    assert "verbosity" not in requests[0]
    assert requests[0]["text"]["verbosity"] == "low"
    if mode == "json_schema":
        assert requests[0]["text"]["format"]["type"] == "json_schema"
        properties = requests[0]["text"]["format"]["schema"]["properties"]
        assert list(properties)[:4] == [
            "conversation_action",
            "pause_requested",
            "continuation",
            "parts",
        ]
        assert result.conversation_action == "answer"
        assert "move" not in properties
    elif mode == "json_object":
        assert requests[0]["text"]["format"] == {"type": "json_object"}
    else:
        assert "format" not in requests[0]["text"]
    messages = requests[0]["input"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert "You are Marko. A real man" in messages[0]["content"]
    assert messages[0]["content"].startswith(rules_for_mode())
    assert messages[0]["content"].index("WHO YOU ARE") < messages[0]["content"].index(
        "APPLICATION_CONTEXT="
    )
    assert '"model_identity"' not in messages[0]["content"]
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
    assert json.dumps(messages[2]).count("unique-reply") == 1
    assert messages[3]["content"] == "unique-last"
    assert all("name" not in message for message in messages)
    assert '"chat_position": 1' in messages[0]["content"]
    assert '"id": "S1"' in messages[0]["content"]
    assert result.source_message_ids == ["m1"]
    assert metrics["reference_aliases"] == {"S1": "m1"}
    assert metrics["raw_response"] == raw
    assert metrics["raw_response_blocks"][0]["text"] == raw
    assert metrics["response_status"] == "completed"
    assert metrics["input_tokens"] == 100
    assert metrics["output_tokens"] == 10
    assert metrics["reasoning_tokens"] == 4


@pytest.mark.parametrize(
    "status,reason,content,error",
    [
        ("incomplete", "max_output_tokens", '{"parts": [', "TRUNCATED_OUTPUT"),
        ("incomplete", "content_filter", "blocked", "INCOMPLETE_OUTPUT"),
        ("incomplete", None, "partial", "INCOMPLETE_OUTPUT"),
        ("failed", None, "failed", "INCOMPLETE_OUTPUT"),
        ("cancelled", None, "cancelled", "INCOMPLETE_OUTPUT"),
        ("in_progress", None, "waiting", "INCOMPLETE_OUTPUT"),
        (None, None, "missing status", "INCOMPLETE_OUTPUT"),
        ("completed", None, "broken json", "json_invalid"),
    ],
)
async def test_gateway_keeps_invalid_raw_output(
    monkeypatch: pytest.MonkeyPatch,
    status: str | None,
    reason: str | None,
    content: str,
    error: str,
) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/responses"
        body = response_body(content, status=status or "completed", reason=reason)
        if status is None:
            body.pop("status")
        return httpx.Response(
            200,
            json=body,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:

        def configured_model(**kwargs: Any) -> ChatOpenAI:
            return ChatOpenAI(**kwargs, http_async_client=client)

        monkeypatch.setattr("app.modules.agent.gateway.ChatOpenAI", configured_model)
        gateway = GatewayModel(
            LLMSettings(
                model="test",
                base_url="https://test.invalid/v1",
                api_key="test",
                structured_mode="prompt_json",
            ),
            1,
        )
        metrics: dict[str, Any] = {"task": "dialogue"}
        token = current_llm_metrics.set(metrics)
        try:
            with pytest.raises((ModelOutputError, ValueError), match=error):
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
    assert metrics["response_status"] == status


@pytest.mark.parametrize("refusal", [False, True])
async def test_gateway_extracts_output_text_and_preserves_response_blocks(
    monkeypatch: pytest.MonkeyPatch, refusal: bool
) -> None:
    raw = '{"parts":["hello"],"language":"en"}'

    def handle(request: httpx.Request) -> httpx.Response:
        body = response_body(raw)
        body["output"].insert(
            0,
            {
                "id": "rs-test",
                "type": "reasoning",
                "summary": [
                    {"type": "summary_text", "text": "Separate reasoning summary"}
                ],
            },
        )
        body["output"][1]["content"] = (
            [{"type": "refusal", "refusal": "Cannot answer this request"}]
            if refusal
            else [
                {"type": "output_text", "text": part, "annotations": []}
                for part in [raw[:15], raw[15:]]
            ]
        )
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:

        def configured_model(**kwargs: Any) -> ChatOpenAI:
            return ChatOpenAI(**kwargs, http_async_client=client)

        monkeypatch.setattr("app.modules.agent.gateway.ChatOpenAI", configured_model)
        gateway = GatewayModel(
            LLMSettings(
                model="test", base_url="https://test.invalid/v1", api_key="test"
            ),
            1,
        )
        metrics: dict[str, Any] = {"task": "dialogue"}
        token = current_llm_metrics.set(metrics)
        try:
            if refusal:
                with pytest.raises(ModelOutputError, match="NONTEXT_OUTPUT"):
                    await gateway.complete(
                        "dialogue",
                        rules_for_mode(),
                        {"CHAT_MESSAGES": []},
                        DialogueDraft,
                        2,
                    )
            else:
                result = await gateway.complete(
                    "dialogue",
                    rules_for_mode(),
                    {"CHAT_MESSAGES": []},
                    DialogueDraft,
                    2,
                )
                assert result.parts == ["hello"]
        finally:
            current_llm_metrics.reset(token)
    assert metrics["raw_response"] == ("" if refusal else raw)
    blocks = metrics["raw_response_blocks"]
    assert blocks[0]["type"] == "reasoning"
    if refusal:
        assert blocks[1]["refusal"] == "Cannot answer this request"
    else:
        assert "".join(block["text"] for block in blocks[1:]) == raw


@pytest.mark.parametrize("verbosity", [None, "low"])
@pytest.mark.parametrize("reasoning_effort", ["none", "low"])
async def test_gateway_transmits_configured_parameters(
    monkeypatch: pytest.MonkeyPatch, verbosity: str | None, reasoning_effort: str
) -> None:
    requests: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/responses"
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=response_body('{"parts":["hello"],"language":"en"}'),
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
    assert payload["text"]["format"] == {"type": "json_object"}
    assert payload["reasoning"] == {"effort": reasoning_effort}
    assert payload["max_output_tokens"] == 1200
    assert payload["text"].get("verbosity") == verbosity
    assert "verbosity" not in payload
    assert "response_format" not in payload
    assert "reasoning_effort" not in payload
    assert "max_completion_tokens" not in payload
    assert payload["store"] is False
    assert "temperature" not in payload
    assert "top_p" not in payload
    assert "frequency_penalty" not in payload
