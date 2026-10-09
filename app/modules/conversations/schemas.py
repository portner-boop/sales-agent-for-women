from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

QuestionGoal = Annotated[str, Field(min_length=1, max_length=80)]


class AgendaOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic_kind: str = Field(min_length=1, max_length=80)
    subject_id: str = "user_self"
    status: Literal[
        "open", "awaiting_user", "answered", "declined", "ambiguous", "abandoned"
    ]
    question_goal: QuestionGoal | None = None
    question_time_scope: str = "stable"
    answer_message_ids: list[str] = Field(default_factory=list)
    pending_user_question: str | None = None
    next_allowed_step: str | None = None


class IncomingMessage(BaseModel):
    external_id: int
    text: str
    event_time: float
    reply_to_external_id: int | None = None
    forwarded: bool = False
    is_edit: bool = False
