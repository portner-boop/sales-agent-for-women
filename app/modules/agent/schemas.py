from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.modules.conversations.schemas import AgendaOperation, QuestionGoal
from app.modules.memory.schemas import EntityCandidate, MemoryOperation


class CurrentTimeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    location: str = Field(min_length=1, max_length=120)
    source_message_id: str = Field(min_length=1, max_length=120)
    timezone: str | None = Field(default=None, min_length=1, max_length=100)
    precision: Literal["approximate", "minute"] = "approximate"


class TurnInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: str
    dialogue_mode: Literal[
        "opening",
        "continuation",
        "clarification",
        "repair",
        "supportive",
        "playful",
        "closing",
        "service",
    ]
    language: str
    continue_conversation: bool = False
    question_interrupted: bool = False
    policy_topics: list[
        Literal[
            "social",
            "recall",
            "time",
            "contact",
            "verification",
            "photo",
            "support",
            "commerce",
            "unknown",
        ]
    ] = Field(default_factory=list)
    recall_scope: Literal["none", "general", "specific", "all"] = "none"
    user_requested_questions: bool = False
    detail_requested: bool = False
    current_day_part_confirmed: bool = False
    current_time_requests: list[CurrentTimeRequest] = Field(
        default_factory=list, max_length=5
    )
    referenced_message_ids: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    entities: list[EntityCandidate] = Field(default_factory=list)
    memory_operations: list[MemoryOperation] = Field(default_factory=list)
    agenda_operations: list[AgendaOperation] = Field(default_factory=list)
    requested_business_action: Literal["none", "show_offer"] = "none"
    requested_persona_fields: list[
        Literal[
            "location",
            "family",
            "languages",
            "appearance",
            "occupation",
            "relationship",
            "religion",
        ]
    ] = Field(default_factory=list)
    retrieval_predicates: list[str] = Field(default_factory=list)
    retrieval_subject_ids: list[str] = Field(default_factory=list)
    retrieval_topic_kinds: list[str] = Field(default_factory=list)
    history_query: bool = False


class ChatImperfection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    written: str = Field(min_length=1, max_length=80)
    standard: str = Field(min_length=1, max_length=80)


class ReplyProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reply: str = Field(min_length=1, max_length=4000)
    continuation: str | None = Field(default=None, min_length=1, max_length=4000)
    dialogue_act: str
    requested_action: Literal["none", "show_offer"] = "none"
    used_memory_ids: list[str] = Field(default_factory=list)
    used_source_message_ids: list[str] = Field(default_factory=list)
    question_goal: QuestionGoal | None = None
    question_part: Literal["reply", "continuation"] | None = None
    question_time_scope: str = "stable"
    question_subject_id: str = "user_self"
    question_thread_id: str | None = None
    casual_imperfection_used: bool = False
    casual_imperfection: ChatImperfection | None = None
    new_commitment_candidate: str | None = None
    followup_allowed: bool = False
    platform_transition: bool = False
    platform_notice: str | None = Field(default=None, min_length=1, max_length=400)

    @property
    def parts(self) -> tuple[str, ...]:
        return (
            (self.reply, self.continuation)
            if self.continuation is not None
            else (self.reply,)
        )

    @property
    def full_reply(self) -> str:
        return "\n\n".join(self.parts)

    @property
    def question_part_index(self) -> int:
        if self.question_part == "reply":
            return 0
        return len(self.parts) - 1


class ContinuingReplyProposal(ReplyProposal):
    continuation: str = Field(min_length=1, max_length=4000)


class RecoveryReplyProposal(ReplyProposal):
    requested_action: Literal["none"] = "none"
    platform_transition: Literal[False] = False
    platform_notice: None = None


class FollowupProposal(ReplyProposal):
    platform_transition: Literal[False] = False
    should_send: bool = Field(
        description="Send one connected question. False is only for a user boundary, departure or unavailability supported by skip_evidence."
    )
    skip_evidence: str | None = Field(default=None, min_length=1, max_length=500)
    reply: str = Field(default="", max_length=280)
    continuation: None = None
    requested_action: Literal["none"] = "none"
    followup_allowed: Literal[False] = False

    @model_validator(mode="after")
    def require_sent_text(self) -> "FollowupProposal":
        if self.should_send and not self.reply.strip():
            raise ValueError("A sent follow-up requires text")
        if not self.should_send and self.reply.strip():
            raise ValueError("A skipped follow-up requires an empty reply")
        return self


class SemanticVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["accept", "reject", "needs_repair"]
    reason_codes: list[str]
    violating_text: str | None = Field(default=None, max_length=1000)
    explanation: str | None = Field(default=None, max_length=500)


class ContextManifest(BaseModel):
    policy: dict[str, Any] = Field(default_factory=dict)
    scope_id: str
    epoch: int
    input_revision: int
    control_revision: int
    memory_revision: int
    policy_version: str = "mvp-2026-10-14-marko-initiative"
    persona_version: str = "marko-v1"
    model_id: str
    messages: list[dict[str, Any]]
    memory_ids: list[str]
    summary_ids: list[str]
    token_count_by_section: dict[str, int]
    retrieval_mode: Literal["structured_sql"] = "structured_sql"
