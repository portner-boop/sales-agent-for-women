from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message_id: str
    message_version: int = Field(ge=1)
    evidence_text: str = Field(min_length=1)


class EntityCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    kind: Literal["person", "pet"]
    name: str | None = None
    sources: list[SourceReference] = Field(min_length=1)


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


class SummaryClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    subject_id: str
    validity_status: Literal["current", "historical", "disputed"] = "current"
    modality: Literal[
        "asserted",
        "planned",
        "hypothetical",
        "figurative",
        "quoted",
        "question",
        "negated",
    ]
    sources: list[SourceReference] = Field(min_length=1)


class EpisodeSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic_kinds: list[str]
    subject_ids: list[str]
    claims: list[SummaryClaim]
    covered_message_ids: list[str]
    open_questions: list[str] = Field(default_factory=list)
    outcomes: list[str] = Field(default_factory=list)
