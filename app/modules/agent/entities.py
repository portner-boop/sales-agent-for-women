from contextvars import ContextVar
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

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
