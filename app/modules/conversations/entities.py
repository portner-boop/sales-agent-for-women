from dataclasses import dataclass
from typing import Any

from app.shared.entities import RevisionSnapshot, ScopeContext


@dataclass(frozen=True)
class Conversation:
    id: str
    epoch: int
    external_user_id: int
    chat_id: int
    input_revision: int
    control_revision: int
    memory_revision: int
    control_state: str
    dialogue_mode: str
    language: str
    first_reply_at: float | None
    processed_sequence: int
    last_activity_at: float
    preview_reply_count: int = 0
    preview_closed_at: float | None = None
    engine_version: str = "v2"
    persona_mode: str = "research_human"
    completed_pair_count: int = 0
    pending_outgoing_turn_id: str | None = None
    invitation_state: str = "none"

    @property
    def scope(self) -> ScopeContext:
        return ScopeContext(self.id, self.epoch)

    @property
    def snapshot(self) -> RevisionSnapshot:
        return RevisionSnapshot(
            self.scope, self.input_revision, self.control_revision, self.memory_revision
        )


@dataclass(frozen=True)
class Message:
    id: str
    version: int
    sequence: int
    role: str
    text: str
    event_time: float
    received_at: float
    external_id: int | None
    reply_to_external_id: int | None
    turn_id: str | None
    metadata: dict[str, Any]

    def as_data(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "role": self.role,
            "text": self.text,
            "event_time": self.event_time,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Turn:
    id: str
    scope: ScopeContext
    input_revision: int
    messages: tuple[Message, ...]
