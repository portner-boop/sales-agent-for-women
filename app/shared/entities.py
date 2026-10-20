from dataclasses import dataclass
from typing import Literal, Protocol


class Clock(Protocol):
    def now(self) -> float: ...


@dataclass(frozen=True)
class ScopeContext:
    scope_id: str
    epoch: int


@dataclass(frozen=True)
class RevisionSnapshot:
    scope: ScopeContext
    input_revision: int
    control_revision: int
    memory_revision: int


ActionKind = Literal["reply", "service"]


class StaleStateError(Exception):
    pass


class EvidenceError(Exception):
    pass


class ContextOverflowError(Exception):
    pass
