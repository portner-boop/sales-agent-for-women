from dataclasses import dataclass

from app.modules.conversations.entities import Turn


@dataclass(frozen=True)
class FollowupTicket:
    id: str
    turn: Turn
    control_revision: int
    anchor_id: str
    run_at: float
    expires_at: float
    ordinal: int = 1
    silence_started_at: float = 0
    requires_response: bool = False
