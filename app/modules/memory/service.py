import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from app.modules.conversations.entities import Message
from app.modules.conversations.repository import ConversationRepository
from app.modules.memory.repository import MemoryRepository
from app.modules.memory.schemas import (
    MemoryOperation,
    SourceReference,
)
from app.shared.entities import (
    Clock,
    EvidenceError,
    RevisionSnapshot,
    ScopeContext,
    StaleStateError,
)
from app.shared.storage import Transaction, fetchall, require_scope

SENSITIVE_PREDICATES = {
    "diagnosis",
    "medication",
    "mental_health",
    "financial_vulnerability",
    "phone",
    "email",
    "contact",
    "sexual_history",
    "dependency_score",
    "payment_probability",
}


def validate_sources(
    sources: list[SourceReference], messages: dict[str, Message], user_only: bool = True
) -> None:
    if not sources:
        raise EvidenceError("Missing evidence")
    for source in sources:
        message = messages.get(source.message_id)
        if message is None or message.version != source.message_version:
            raise EvidenceError("Invalid source version")
        if source.evidence_text not in message.text:
            raise EvidenceError("Evidence is not a source span")
        if user_only and (message.role != "user" or message.metadata.get("forwarded")):
            raise EvidenceError("A quoted or assistant statement is not a client fact")


class MemoryService:
    def __init__(
        self,
        repository: MemoryRepository,
        conversations: ConversationRepository,
        clock: Clock,
        deletion_path: Path,
        purge_artifacts: Callable[[Transaction, str, int], Awaitable[None]],
    ) -> None:
        self.repository = repository
        self.conversations = conversations
        self.clock = clock
        self.deletion_path = deletion_path
        self._deletion_lock = asyncio.Lock()
        self.purge_artifacts = purge_artifacts

    async def resolve(
        self,
        snapshot: RevisionSnapshot,
        messages: tuple[Message, ...],
        operations: list[MemoryOperation],
    ) -> list[str]:
        current_sources = {message.id: message for message in messages}
        known_entities = {
            item["id"] for item in await self.repository.entities(snapshot.scope)
        }
        for operation in operations:
            if operation.operation in {"no_change", "needs_clarification"}:
                continue
            validate_sources(operation.sources, current_sources)
            if (
                operation.subject_id not in known_entities
                or operation.subject_id == "persona"
            ):
                raise EvidenceError("Invalid memory subject")
            if operation.predicate in SENSITIVE_PREDICATES:
                raise EvidenceError("Predicate is not allowed in persistent memory")
            if operation.modality in {"question", "hypothetical", "quoted"}:
                raise EvidenceError("Nonassertions cannot become persistent facts")
            if (
                operation.modality == "figurative"
                and operation.kind != "conversation_glossary"
            ):
                raise EvidenceError("Figurative speech requires glossary memory")

        async def apply(tx: Transaction) -> list[str]:
            await require_scope(tx, snapshot.scope)
            conversation = await self.conversations.get(snapshot.scope.scope_id, tx)
            if (
                conversation.input_revision != snapshot.input_revision
                or conversation.memory_revision != snapshot.memory_revision
            ):
                raise StaleStateError("Memory resolution snapshot changed")
            for message in messages:
                actual = await fetchall(
                    tx,
                    "SELECT current_version FROM messages WHERE id=? AND scope_id=? AND epoch=?",
                    (message.id, snapshot.scope.scope_id, snapshot.scope.epoch),
                )
                if not actual or actual[0]["current_version"] != message.version:
                    raise StaleStateError("Source changed during interpretation")
            accepted = []
            for operation in operations:
                identifier = await self.repository.apply(
                    tx, snapshot.scope, operation, self.clock.now()
                )
                if identifier:
                    accepted.append(identifier)
            if accepted:
                await tx.execute(
                    "UPDATE conversations SET memory_revision=memory_revision+1 WHERE id=?",
                    (snapshot.scope.scope_id,),
                )
            return accepted

        return await self.repository.database.write(apply)

    async def backup(self) -> Path:
        from app.shared.operations import create_backup

        async with self._deletion_lock:
            return await create_backup(
                self.repository.database, self.deletion_path, self.clock.now()
            )

    async def forget(
        self,
        scope: ScopeContext,
        after_purge: (
            Callable[[Transaction, ScopeContext], Awaitable[None]] | None
        ) = None,
    ) -> ScopeContext:
        async with self._deletion_lock:
            current = await self.conversations.get(scope.scope_id)
            if current.scope != scope:
                raise StaleStateError("Deletion generation changed")
            new_scope = ScopeContext(scope.scope_id, scope.epoch + 1)
            await asyncio.to_thread(self._append_tombstone, new_scope)

            async def purge(tx: Transaction) -> None:
                await require_scope(tx, scope)
                await self.repository.purge(
                    tx, scope.scope_id, new_scope.epoch, self.clock.now()
                )
                await self.purge_artifacts(tx, scope.scope_id, new_scope.epoch)
                await self.repository.finish_purge(
                    tx, scope.scope_id, new_scope.epoch, self.clock.now()
                )
                await self.repository.ensure_entities(tx, new_scope)
                if after_purge is not None:
                    await after_purge(tx, new_scope)

            await self.repository.database.write(purge)
            return new_scope

    def _append_tombstone(self, scope: ScopeContext) -> None:
        self.deletion_path.parent.mkdir(parents=True, exist_ok=True)
        with self.deletion_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "scope_id": scope.scope_id,
                        "epoch": scope.epoch,
                        "deleted_at": self.clock.now(),
                    }
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())

    async def apply_tombstones(self) -> None:
        if not self.deletion_path.exists():
            return
        tombstones = [
            json.loads(line)
            for line in self.deletion_path.read_text().splitlines()
            if line.strip()
        ]
        for item in tombstones:
            try:
                conversation = await self.conversations.get(item["scope_id"])
            except LookupError:
                continue
            if conversation.epoch < item["epoch"]:

                async def purge(
                    tx: Transaction, tombstone: dict[str, Any] = item
                ) -> None:
                    await self.repository.purge(
                        tx,
                        tombstone["scope_id"],
                        tombstone["epoch"],
                        tombstone["deleted_at"],
                    )
                    await self.purge_artifacts(
                        tx, tombstone["scope_id"], tombstone["epoch"]
                    )
                    await self.repository.finish_purge(
                        tx,
                        tombstone["scope_id"],
                        tombstone["epoch"],
                        tombstone["deleted_at"],
                    )
                    await self.repository.ensure_entities(
                        tx, ScopeContext(tombstone["scope_id"], tombstone["epoch"])
                    )

                await self.repository.database.write(purge)
