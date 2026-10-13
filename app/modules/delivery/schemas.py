from pydantic import BaseModel, ConfigDict, Field

from app.shared.entities import ActionKind


class OutboxDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action_key: str
    kind: ActionKind
    text: str = Field(min_length=1, max_length=4000)
    deadline: float
    predecessor_id: str | None = None
    delay_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
