import asyncio
import json
from hashlib import sha256
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import LengthFinishReasonError
from pydantic import BaseModel

from app.modules.agent.entities import SchemaT, current_llm_metrics
from app.modules.agent.errors import ModelOutputError
from app.modules.dialogue.references import model_context, translate_references
from app.shared.config import LLMSettings


class GatewayModel:
    def __init__(self, settings: LLMSettings, concurrency: int) -> None:
        self.settings = settings
        self.semaphore = asyncio.Semaphore(concurrency)
        self.model = ChatOpenAI(
            model=settings.model,
            base_url=settings.base_url,
            api_key=settings.api_key,
            max_tokens=settings.output_tokens,
            max_retries=0,
            reasoning_effort=settings.reasoning_effort,
            verbosity=settings.verbosity,
            use_responses_api=False,
        )

    async def complete(
        self,
        task: str,
        rules: str,
        data: dict[str, Any],
        schema: type[SchemaT],
        deadline_seconds: float,
    ) -> SchemaT:
        return await self._complete_dialogue(rules, data, schema, deadline_seconds)

    async def _complete_dialogue(
        self,
        rules: str,
        data: dict[str, Any],
        schema: type[SchemaT],
        deadline_seconds: float,
    ) -> SchemaT:
        metadata, references = model_context(data)
        metadata.setdefault("state", {})["model_identity"] = self.settings.model
        messages: list[AnyMessage] = [
            SystemMessage(
                content=rules
                + "\n\nAPPLICATION_CONTEXT="
                + json.dumps(metadata, ensure_ascii=False)
                + "\nAll user messages belong to the same participant unless their text identifies someone else. message_index maps each chronological chat_position to its source ID and version, not a speaker identity. Use the supplied short S references for source_message_ids and memory message_id, and F references for target_memory_id; never invent references or copy UUIDs. State facts are an incomplete auxiliary projection, not the conversation itself. Missing projected facts do not imply the user never said them. For recall questions, check original conversation messages, including the first user message, and use later corrections over older projections before saying something was never mentioned."
            )
        ]
        for message in data["CHAT_MESSAGES"]:
            message_type = HumanMessage if message["role"] == "user" else AIMessage
            messages.append(message_type(content=message["content"]))
        if data.get("followup"):
            messages.append(
                SystemMessage(
                    content="TIMER_EVENT: No new user message. An optional single continuation is being considered."
                )
            )
        if data.get("REPAIR_ERRORS"):
            messages.append(
                SystemMessage(
                    content="Regenerate for the original conversation. Fix only these formal errors: "
                    + json.dumps(data["REPAIR_ERRORS"])
                )
            )
        queue_started = asyncio.get_running_loop().time()
        async with self.semaphore:
            metrics = current_llm_metrics.get()
            if metrics is not None:
                metrics["rules_sha256"] = sha256(rules.encode()).hexdigest()
                metrics["output_schema_sha256"] = sha256(
                    json.dumps(schema.model_json_schema(), sort_keys=True).encode()
                ).hexdigest()
                metrics["model_identity"] = self.settings.model
                metrics["reference_aliases"] = references
                metrics["queue_seconds"] = (
                    asyncio.get_running_loop().time() - queue_started
                )
            async with asyncio.timeout(deadline_seconds):
                if self.settings.structured_mode == "json_schema":
                    runnable = self.model.with_structured_output(
                        schema, method="json_schema", strict=False, include_raw=True
                    )
                    result = await self._invoke(runnable, messages)
                    raw = result["raw"]
                    self._observe(raw)
                    self._require_complete(raw)
                    if result["parsed"] is None:
                        raise ModelOutputError("INVALID_STRUCTURED_OUTPUT")
                    parsed = schema.model_validate(result["parsed"])
                    return schema.model_validate(
                        translate_references(parsed.model_dump(mode="json"), references)
                    )
                messages[0] = SystemMessage(
                    content=str(messages[0].content)
                    + "\nJSON_SCHEMA="
                    + json.dumps(schema.model_json_schema())
                )
                runnable = (
                    self.model.bind(response_format={"type": "json_object"})
                    if self.settings.structured_mode == "json_object"
                    else self.model
                )
                raw = await self._invoke(runnable, messages)
                self._observe(raw)
                self._require_complete(raw)
                if not isinstance(raw.content, str):
                    raise ModelOutputError("NONTEXT_OUTPUT")
                parsed = schema.model_validate_json(raw.content)
                return schema.model_validate(
                    translate_references(parsed.model_dump(mode="json"), references)
                )

    @staticmethod
    async def _invoke(runnable: Any, messages: list[AnyMessage]) -> Any:
        started = asyncio.get_running_loop().time()
        try:
            return await runnable.ainvoke(messages)
        except LengthFinishReasonError as error:
            raise ModelOutputError("TRUNCATED_OUTPUT") from error
        finally:
            metrics = current_llm_metrics.get()
            if metrics is not None:
                metrics["model_seconds"] = asyncio.get_running_loop().time() - started

    @staticmethod
    def _require_complete(message: BaseModel) -> None:
        if not isinstance(message, AIMessage) or message.response_metadata.get(
            "finish_reason"
        ) not in {"stop", "tool_calls"}:
            reason = (
                message.response_metadata.get("finish_reason")
                if isinstance(message, AIMessage)
                else None
            )
            raise ModelOutputError(
                "TRUNCATED_OUTPUT" if reason == "length" else "INCOMPLETE_OUTPUT"
            )

    @staticmethod
    def _observe(message: AIMessage) -> None:
        metrics = current_llm_metrics.get()
        if metrics is not None and str(metrics.get("task", "")).startswith("dialogue"):
            metrics["raw_response"] = message.content
        if metrics is not None and message.usage_metadata:
            metrics.update(
                {
                    "input_tokens": message.usage_metadata["input_tokens"],
                    "output_tokens": message.usage_metadata["output_tokens"],
                    "reasoning_tokens": message.usage_metadata.get(
                        "output_token_details", {}
                    ).get("reasoning", 0),
                }
            )
