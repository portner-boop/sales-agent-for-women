from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from app.modules.agent.schemas import ContextManifest
from app.modules.memory.entities import CompactionBlock
from app.modules.memory.schemas import EpisodeSummary

SchemaT = TypeVar("SchemaT", bound=BaseModel)
current_llm_metrics: ContextVar[dict[str, Any] | None] = ContextVar(
    "llm_metrics", default=None
)


class LanguageModel(Protocol):
    async def complete(
        self,
        task: str,
        rules: str,
        data: dict[str, Any],
        schema: type[SchemaT],
        deadline_seconds: float,
    ) -> SchemaT: ...

    async def summarize(
        self, block: CompactionBlock, deadline_seconds: float, output_tokens: int
    ) -> EpisodeSummary: ...


@dataclass(frozen=True)
class BuiltContext:
    data: dict[str, Any]
    manifest: ContextManifest
