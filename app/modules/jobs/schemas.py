from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class JobPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    handler_version: str = "v1"
    data: dict[str, Any] = Field(default_factory=dict)
