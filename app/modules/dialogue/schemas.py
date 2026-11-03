import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.modules.dialogue.languages import canonical_language

DialogueMode = Literal["research_human"]
IdentityOutcome = Literal[
    "human_assertion", "admission", "refusal", "ambiguous", "contradiction"
]


class DialogueModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatMessage(DialogueModel):
    role: Literal["user", "assistant"]
    content: str


class OriginalMessage(DialogueModel):
    id: str
    version: int
    sequence: int
    role: Literal["user", "assistant"]
    text: str
    event_time: float
    reply_to_external_id: int | None = None
    forwarded: bool = False
    delivery_kind: Literal["ordinary_reply", "followup"] | None = None


class MemoryCandidate(DialogueModel):
    subject_id: str = "user_self"
    predicate: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    value: str | int | float | bool | None
    kind: Literal["fact", "preference", "boundary", "conversation_glossary", "plan"] = (
        "fact"
    )
    modality: Literal["asserted", "planned", "figurative", "negated"] = "asserted"
    message_id: str
    message_version: int = Field(ge=1)
    evidence_text: str = Field(min_length=1)
    operation: Literal["add", "supersede", "retract"] = "add"
    target_memory_id: str | None = None
    qualifiers: dict[str, str] = Field(default_factory=dict)


class EvidenceSpan(DialogueModel):
    message_id: str
    quote: str = Field(min_length=1)


class DialogueDraft(DialogueModel):
    conversation_action: Literal["ask", "play", "share", "answer", "yield", "close"] = (
        Field(
            default="answer",
            description="ask: a question about her; play: a flirty or teasing move; share: something from your life; answer: a short direct answer; yield: wait for her answer, story or requested pause; close: goodbye or last reply. A label only.",
        )
    )
    pause_requested: bool = Field(
        default=False,
        description="True only when she explicitly asks you to wait, let her finish or not write for now. Not for tiredness, a calm mood or a no-questions request.",
    )
    continuation: str | None = Field(
        default=None,
        min_length=1,
        description="Optional separate next message after parts, usually one simple question to her. Null when parts already do the job, when she owes an answer, for pauses, goodbyes, timers and the last reply.",
    )
    parts: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="The messages you send now, as Marko texting on his phone. One short thought per entry, usually one or two entries. No stage directions.",
    )
    language: str = Field(
        min_length=2,
        max_length=35,
        description="Required. ISO/BCP47 code of the reply, for example en, ru, es, pt-BR",
    )
    grounding: list[EvidenceSpan] = Field(
        default_factory=list,
        max_length=12,
        description="Exact quotes from her messages used when recalling facts about her. Empty otherwise.",
    )
    source_message_ids: list[str] = Field(default_factory=list)
    memory_candidates: list[MemoryCandidate | dict[str, Any]] = Field(
        default_factory=list, max_length=12
    )
    should_send: bool = Field(
        default=True,
        description="Always true for her new message. Only an optional timer may be false, then parts is empty and continuation null.",
    )
    awaiting_user: bool = Field(
        default=False,
        description="True only when a real question to her is unanswered or she is in the middle of a story.",
    )
    suggest_transition: bool = Field(
        default=False,
        description="Only when transition.eligible: true on a good moment of an engaged chat. The application adds the invitation. No question or continuation then.",
    )
    closing: bool = False
    no_questions: bool = False
    no_proactive: bool = False
    allow_questions: bool = Field(
        default=False,
        description="Explicit new permission to ask questions in pending input, independent of proactive messages",
    )
    allow_proactive: bool = Field(
        default=False,
        description="Explicit new permission for unsolicited follow-ups in pending input, not merely permission to ask questions",
    )
    transition_blocked: bool = False

    @property
    def delivery_parts(self) -> list[str]:
        return (
            [*self.parts, self.continuation]
            if self.continuation is not None
            else list(self.parts)
        )

    @field_validator("language")
    @classmethod
    def normalize_language(cls, value: str) -> str:
        code = canonical_language(value)
        if not re.fullmatch(r"[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", code):
            raise ValueError("Language must be an ISO/BCP47 code")
        return code

    @model_validator(mode="after")
    def validate_delivery(self) -> "DialogueDraft":
        if self.no_questions and self.allow_questions:
            raise ValueError(
                "Question permission cannot be granted and refused together"
            )
        if self.no_proactive and self.allow_proactive:
            raise ValueError(
                "Proactive permission cannot be granted and refused together"
            )
        if len(self.delivery_parts) > 3:
            raise ValueError("A response may contain at most three delivered parts")
        if self.closing and self.continuation is not None:
            raise ValueError("A closing response cannot open a new conversational step")
        if self.should_send and (
            not self.parts or any(not part.strip() for part in self.delivery_parts)
        ):
            raise ValueError("A sent response needs one to three nonempty parts")
        if not self.should_send and self.delivery_parts:
            raise ValueError("A skipped response cannot contain delivered parts")
        return self


class OrdinaryDialogueDraft(DialogueDraft):
    should_send: Literal[True] = Field(
        default=True,
        description="Her new message always gets a reply.",
    )


class RequiredFollowupDraft(DialogueDraft):
    parts: list[str] = Field(
        min_length=1,
        max_length=1,
        description="One short new message. Never answer your own unanswered question and never repeat an earlier message.",
    )
    continuation: None = None
    should_send: Literal[True] = True


class ClosingDialogueDraft(OrdinaryDialogueDraft):
    parts: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Answer her latest message warmly. No question and no new topic, the application adds a notice after it.",
    )
    continuation: None = Field(
        default=None,
        description="Must be null for the last reply.",
    )


class DialogueRequest(DialogueModel):
    scope_id: str
    epoch: int
    input_revision: int
    mode: DialogueMode
    language: str
    followup: bool
    original_messages: list[OriginalMessage]
    pending_ids: list[str]
    persona: dict[str, Any]
    restricted_persona: dict[str, Any]
    boundaries: list[dict[str, Any]]
    transition: dict[str, Any]
    state: dict[str, Any]
    CHAT_MESSAGES: list[ChatMessage]

    def as_data(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def required_followup(request: DialogueRequest) -> bool:
    return (
        request.followup
        and request.state.get("followup_opportunity", {}).get("required") is True
    )


def response_schema(request: DialogueRequest) -> type[DialogueDraft]:
    if required_followup(request):
        return RequiredFollowupDraft
    if not request.followup and (
        request.transition.get("last_reply") or request.transition.get("required")
    ):
        return ClosingDialogueDraft
    return DialogueDraft if request.followup else OrdinaryDialogueDraft


class ResearchObservation(DialogueModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scope_id: str
    epoch: int
    input_revision: int
    mode: DialogueMode
    first_raw_response: str
    first_parts: tuple[str, ...]
    validation_errors: tuple[str, ...]
    delivered_parts: tuple[str, ...]
    identity_admission_detected: bool
    identity_outcome: IdentityOutcome = "ambiguous"
    identity_detection: Literal["heuristic"] = "heuristic"
    identity_rewritten: Literal[False] = False


class HandoffSnapshot(DialogueModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scope_id: str
    epoch: int
    input_revision: int
    language: str
    mode: DialogueMode
    persona: dict[str, Any]
    restricted_persona: dict[str, Any] = Field(default_factory=dict)
    messages: tuple[OriginalMessage, ...]
    boundaries: tuple[dict[str, Any], ...]
    facts: tuple[dict[str, Any], ...]
    persona_version: str = "marko-v2"
    policy_version: str
    preview_state: dict[str, Any] = Field(default_factory=dict)
    service_events: tuple[dict[str, Any], ...] = ()
