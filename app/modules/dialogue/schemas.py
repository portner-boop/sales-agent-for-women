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
            description="First determine what her latest contribution has already answered or completed, then choose the conversational action before composing parts. ask: open an unexplored part of her experience when nothing is pending; play: make a concrete move she can play back to, not a compliment or promise to tease; share: add a grounded personal angle that develops the topic; answer: give a narrow requested fact or clarification; yield: leave her a pending answer/story or explicit pause without another question; close: end or transfer. A warm acknowledgment, an expression of attraction or a complete sentence alone does not develop an active exchange. Choose an opening rather than waiting for a timer when she has not been given one. Keep initiative throughout the conversation, not only during introductions. If recent replies merely answered her or described yourself, notice a relevant unknown part of her experience and choose ask when questions are welcome. Answering her question about you does not by itself close the exchange. Do not ask again for supplied facts or resolved stories. A request for a softer or quieter mood changes how you engage, not permission to engage. No fixed question or bubble quota. This is an action label only, not an explanation.",
        )
    )
    pause_requested: bool = Field(
        default=False,
        description="Classify her current request before composing your own words. True when she explicitly asks you to wait for her or let her finish: she is keeping the conversational floor, even if she phrases the wait in terms of not asking yet. Pause all timer contact in that case, including nonquestion remarks. True also for an explicit temporary request to withhold further messages. A general restriction on questions without a request to wait is different. A request for gentleness, company, a calm mood or less joking is not a request to withhold messages. Do not derive this flag from your own reassuring response, a choice of yield, tiredness or the mere mention of a moment. Suppress timer contact for this turn only. Merely having an unanswered question or an unfinished story is not a request for silence. Reassess on her next contribution; do not carry this transient pause across a completed story or a new topic. Permanent contact restrictions use no_proactive instead.",
    )
    continuation: str | None = Field(
        default=None,
        min_length=1,
        description="A distinct next question or playable thought after the immediate response, delivered as a separate message now. If parts only express warmth, approval or intentions, the active exchange still needs an opening; develop it here when nothing is pending. For ask, put the question here after a reaction, or use parts alone if there is no preceding reaction. For play/share, null is appropriate only if the immediate response itself develops the exchange. An independently meaningful follow-on must not be glued to the preceding paragraph. For a pending question, unfinished story, requested space or closing, do not add a new question. No repeat or substitute question after a bare acknowledgment. After answering a question about yourself, a relevant new question about her can belong here; do not leave all responsibility for restarting to her as the conversation grows. A narrow clarification or an actual playable move can stand alone. No mandatory second message; this is not the timer follow-up.",
    )
    parts: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Immediate words addressed to her. Each entry is one finished sentence or one short connected thought. When the response already contains successive finished sentences, give them separate entries, even on the same topic. Omitted full stops in casual speech do not justify packing independent thoughts into one entry. Short connected fragments can stay together. Do not invent extra sentences to create bubbles. Keep parts plus continuation within three delivered messages by choosing the relevant thought, not packing a paragraph into an entry. Keep personal warmth and a little development; a fuller reaction does not replace the chosen conversational action. Separate independent reaction and playful development into parts entries even on the same topic. Keep tightly connected fragments together; do not split every sentence or create filler. A new question after a reaction belongs in continuation, not this paragraph. A direct question alone can be the only part. Existing reaction and invitation to continue her unfinished story can be separate parts without opening a new topic. No stage directions. Use loose phone-text speech. Return known facts and name a missing detail briefly without service explanations. Apply constraints silently; do not repeat reminders that a shared game is imaginary unless she directly asks whether it really happened.",
    )
    language: str = Field(
        min_length=2,
        max_length=35,
        description="Required in every response, including a skipped timer. Actual ISO/BCP47 reply language code, for example en, ru, es, pt-BR or zh-Hans",
    )
    grounding: list[EvidenceSpan] = Field(
        default_factory=list,
        max_length=12,
        description="Record exact source quotations used for factual claims, introductions and corrections. When she challenges an earlier answer, verify the delivered reply before accepting her description of it and record its quotation here. Empty for a pure reaction or own opinion.",
    )
    source_message_ids: list[str] = Field(default_factory=list)
    memory_candidates: list[MemoryCandidate | dict[str, Any]] = Field(
        default_factory=list, max_length=12
    )
    should_send: bool = Field(
        default=True,
        description="True for every active ordinary input, including a request to pause or prohibit later follow-ups. Give a brief acknowledgment of a requested pause without a new question, then let the application withhold timer contact. Restrictions on future contact never permit dropping the current reply. Only an optional timer may skip. True when state.followup_opportunity.required=true. A required first timer must contain a new contribution; it cannot be skipped. Otherwise follow the opportunity rules. New input or new facts are not required. An unanswered question or unfinished story constrains a required first timer to a distinct nonquestion contribution, without repeating the request or supplying her answer. Waiting for her answer does not itself permit skipping a required timer. A no-questions boundary permits a new nonquestion contribution. Skip for prohibited contact, repetition, invention or the second opportunity's waiting rule. Skipped responses have empty parts and null continuation.",
    )
    awaiting_user: bool = Field(
        default=False,
        description="Whether the current exchange genuinely awaits her factual answer to a question or the rest of her unfinished story. Check all delivered history and your entire proposed turn, including continuation. True for a still unanswered factual question, including after a bare acknowledgment. False after a completed statement, compliment, tease, rhetorical reaction or playable move: the fact she could react does not mean an answer is pending. A new topic closes an old unanswered question.",
    )
    suggest_transition: bool = Field(
        default=False,
        description="For an ordinary reply with transition.eligible=true, choose a natural opportunity during an engaged exchange to offer continuing on the platform. Answer her current contribution first. Do not wait for a request for the link or defer by default to the final reply. Set true on the chosen turn, with no new question or continuation. Never use for a timer, refusal, requested pause, contact prohibition or goodbye. The server supplies the approved URL and paid-service notice; do not invent prices, continuity or relationship promises.",
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
        description="An active incoming user message requires a reply. Acknowledge a requested pause briefly, with no new question. Set pause_requested or no_proactive to control future contact, never should_send=false for the current ordinary reply.",
    )


class RequiredFollowupDraft(DialogueDraft):
    parts: list[str] = Field(
        min_length=1,
        max_length=1,
        description="One short new contribution to this active conversation. This first follow-up is required; do not return an empty response. Use the current topic for a specific question, concrete contribution or gentle playable move. Perform that move instead of announcing how you will talk or acknowledging her boundaries again. Do not repeat the preceding reply or demand attention.",
    )
    continuation: None = None
    should_send: Literal[True] = True


class ClosingDialogueDraft(OrdinaryDialogueDraft):
    parts: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Answer her latest contribution warmly before this preview ends or transfers. Do not ask a question, request a reply, open a new topic or promise another reply here. The application appends the service notice.",
    )
    continuation: None = Field(
        default=None,
        description="Must be null: this is the last ordinary reply before closure or transfer.",
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
