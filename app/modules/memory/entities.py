from dataclasses import dataclass
from typing import Any

from app.modules.conversations.entities import Message


class SummaryOverflowError(ValueError):
    pass


@dataclass(frozen=True)
class MemoryView:
    facts: tuple[dict[str, Any], ...]
    boundaries: tuple[dict[str, Any], ...]
    summaries: tuple[dict[str, Any], ...]
    sources: tuple[Message, ...]


@dataclass(frozen=True)
class CompactionBlock:
    messages: tuple[Message, ...]
    source_hash: str
    entity_map: tuple[dict[str, Any], ...] = ()
