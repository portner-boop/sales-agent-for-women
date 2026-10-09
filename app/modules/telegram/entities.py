from dataclasses import dataclass
from typing import Any, Literal

from app.modules.conversations.schemas import IncomingMessage


@dataclass(frozen=True)
class IncomingEvent:
    update_id: int
    raw: dict[str, Any]
    kind: Literal["message", "command", "blocked", "unsupported", "quarantine"]
    user_id: int | None = None
    chat_id: int | None = None
    message: IncomingMessage | None = None
    command: str | None = None
    reason: str | None = None
    language_hint: str = "en"
