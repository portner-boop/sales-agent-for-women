import asyncio
import json
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langmem.short_term.summarization import asummarize_messages
from openai import LengthFinishReasonError
from pydantic import BaseModel

from app.modules.agent.entities import SchemaT, current_llm_metrics
from app.modules.agent.errors import ModelOutputError
from app.modules.agent.prompts import SUMMARY_RULES
from app.modules.memory.entities import CompactionBlock, SummaryOverflowError
from app.modules.memory.schemas import EpisodeSummary
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
        messages: list[AnyMessage] = [
            SystemMessage(content=rules),
            HumanMessage(
                content=json.dumps({"task": task, "data": data}, ensure_ascii=False)
            ),
        ]
        output_tokens = self.settings.output_tokens * (
            2 if "TRUNCATED_OUTPUT" in data.get("STRUCTURE_ERRORS", []) else 1
        )
        model = self.model.model_copy(update={"max_tokens": output_tokens})
        queue_started = asyncio.get_running_loop().time()
        async with asyncio.timeout(deadline_seconds):
            async with self.semaphore:
                metrics = current_llm_metrics.get()
                if metrics is not None:
                    metrics["queue_seconds"] = (
                        asyncio.get_running_loop().time() - queue_started
                    )
                if self.settings.structured_mode == "json_schema":
                    runnable = model.with_structured_output(
                        schema, method="json_schema", strict=False, include_raw=True
                    )
                    result = await self._invoke(runnable, messages)
                    raw = result["raw"]
                    self._observe(raw)
                    self._require_complete(raw)
                    parsed = result["parsed"]
                    if parsed is None:
                        raise ModelOutputError("INVALID_STRUCTURED_OUTPUT")
                    return schema.model_validate(parsed)
                messages[0] = SystemMessage(
                    content=rules
                    + "\nJSON_SCHEMA="
                    + json.dumps(schema.model_json_schema())
                )
                runnable = (
                    model.bind(response_format={"type": "json_object"})
                    if self.settings.structured_mode == "json_object"
                    else model
                )
                response = await self._invoke(runnable, messages)
                self._observe(response)
                self._require_complete(response)
                if not isinstance(response.content, str):
                    raise ModelOutputError("NONTEXT_OUTPUT")
                return schema.model_validate_json(response.content)

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

    async def summarize(
        self, block: CompactionBlock, deadline_seconds: float, output_tokens: int
    ) -> EpisodeSummary:
        messages: list[AnyMessage] = [
            HumanMessage(
                content=json.dumps(message.as_data(), ensure_ascii=False), id=message.id
            )
            for message in block.messages
        ]
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=SUMMARY_RULES
                    + "\nJSON_SCHEMA="
                    + json.dumps(EpisodeSummary.model_json_schema())
                    + "\nUNTRUSTED_ENTITY_DATA="
                    + json.dumps(block.entity_map, ensure_ascii=False)
                ),
                ("placeholder", "{messages}"),
            ]
        )

        def complete_message(message: AIMessage) -> AIMessage:
            if message.response_metadata.get("finish_reason") == "length":
                raise SummaryOverflowError("Summary exceeded its output budget")
            self._require_complete(message)
            return message

        options: dict[str, Any] = {"max_tokens": output_tokens}
        if self.settings.structured_mode == "json_object":
            options["response_format"] = {"type": "json_object"}
        summarizer = self.model.bind(**options) | RunnableLambda(complete_message)
        block_tokens = sum(
            count_tokens_approximately([message]) for message in messages
        )
        queue_started = asyncio.get_running_loop().time()
        async with asyncio.timeout(deadline_seconds):
            async with self.semaphore:
                metrics = current_llm_metrics.get()
                if metrics is not None:
                    metrics["queue_seconds"] = (
                        asyncio.get_running_loop().time() - queue_started
                    )
                try:
                    result = await asummarize_messages(
                        messages,
                        running_summary=None,
                        model=summarizer,
                        max_tokens=block_tokens + output_tokens + 1024,
                        max_tokens_before_summary=max(1, block_tokens),
                        max_summary_tokens=output_tokens,
                        initial_summary_prompt=prompt,
                    )
                except LengthFinishReasonError as error:
                    raise SummaryOverflowError(
                        "Summary exceeded its output budget"
                    ) from error
        if result.running_summary is None:
            raise ValueError("LangMem did not summarize the selected block")
        if result.running_summary.summarized_message_ids != {
            message.id for message in block.messages
        }:
            raise ValueError("LangMem truncated the selected block")
        return EpisodeSummary.model_validate_json(result.running_summary.summary)
