from dataclasses import dataclass
from typing import Any

from app.shared.entities import ScopeContext

MAX_JOB_ATTEMPTS = 3


class JobRetryError(Exception):
    def __init__(
        self, code: str, retry_after: float = 0, permanent: bool = False
    ) -> None:
        super().__init__(code)
        self.code = code
        self.retry_after = retry_after
        self.permanent = permanent


@dataclass(frozen=True)
class Job:
    id: str
    kind: str
    scope: ScopeContext | None
    payload: dict[str, Any]
    attempts: int
    lease_token: str

    @property
    def is_final_attempt(self) -> bool:
        return self.attempts >= MAX_JOB_ATTEMPTS
