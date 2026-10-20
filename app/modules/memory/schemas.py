from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message_id: str
    message_version: int = Field(ge=1)
    evidence_text: str = Field(min_length=1)


class MemoryOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal[
        "add",
        "confirm",
        "supersede",
        "qualify",
        "retract",
        "needs_clarification",
        "no_change",
    ]
    target_memory_id: str | None = None
    subject_id: str = "user_self"
    kind: Literal[
        "fact", "preference", "boundary", "conversation_glossary", "plan", "outcome"
    ] = "fact"
    predicate: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    value: Any = None
    qualifiers: dict[str, str] = Field(default_factory=dict)
    modality: Literal[
        "asserted",
        "planned",
        "hypothetical",
        "figurative",
        "quoted",
        "question",
        "negated",
    ] = "asserted"
    event_time: str | None = None
    sources: list[SourceReference] = Field(default_factory=list)
