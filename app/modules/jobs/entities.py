from dataclasses import dataclass
from typing import Any

from app.shared.entities import ScopeContext


@dataclass(frozen=True)
class Job:
    id: str
    kind: str
    scope: ScopeContext | None
    payload: dict[str, Any]
    attempts: int
    lease_token: str
