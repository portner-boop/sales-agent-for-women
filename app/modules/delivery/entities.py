from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from app.shared.entities import ScopeContext


@dataclass(frozen=True)
class OutboxMessage:
    id: str
    scope: ScopeContext
    chat_id: int
    turn_id: str | None
    kind: str
    text: str
    proposal: dict[str, Any]
    attempts: int
    deadline: float
    predecessor_id: str | None = None


@dataclass(frozen=True)
class SendResult:
    status: Literal[
        "delivered", "retryable_failure", "permanent_failure", "delivery_unknown"
    ]
    external_message_id: int | None = None
    error_code: str | None = None
    retry_after: float = 1


class ChannelSender(Protocol):
    async def send_text(self, chat_id: int, text: str) -> SendResult: ...


@runtime_checkable
class TypingSender(Protocol):
    def typing(self, chat_id: int) -> AbstractAsyncContextManager[Any]: ...
