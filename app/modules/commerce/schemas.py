from pydantic import BaseModel, ConfigDict, Field, JsonValue


class MockPurchase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    actor_user_id: int
    event_key: str


class HandoffBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    predicate: str
    value: JsonValue
    qualifiers: dict[str, JsonValue] = Field(default_factory=dict)


class HandoffContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preferred_name: str | None = None
    current_topics: list[str] = Field(default_factory=list)
    boundaries: list[HandoffBoundary] = Field(default_factory=list)
