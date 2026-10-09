import asyncio
import hashlib
import json
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from app.modules.conversations.entities import Message
from app.modules.conversations.repository import ConversationRepository
from app.modules.conversations.schemas import AgendaOperation
from app.modules.memory.entities import CompactionBlock, MemoryView
from app.modules.memory.repository import MemoryRepository
from app.modules.memory.schemas import (
    EntityCandidate,
    EpisodeSummary,
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
        entities: list[EntityCandidate],
        operations: list[MemoryOperation],
        agenda: list[AgendaOperation],
        language: str,
        dialogue_mode: str,
    ) -> list[str]:
        current_sources = {message.id: message for message in messages}
        known_entities = {
            item["id"] for item in await self.repository.entities(snapshot.scope)
        }
        for entity in entities:
            if entity.id in {"persona", "user_self"}:
                raise EvidenceError("Canonical entities cannot be changed")
            validate_sources(entity.sources, current_sources)
            known_entities.add(entity.id)
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
        for item in agenda:
            if item.subject_id not in known_entities or not set(
                item.answer_message_ids
            ).issubset(current_sources):
                raise EvidenceError("Invalid agenda evidence")

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
            for entity in entities:
                await self.repository.add_entity(tx, snapshot.scope, entity)
            accepted = []
            for operation in operations:
                identifier = await self.repository.apply(
                    tx, snapshot.scope, operation, self.clock.now()
                )
                if identifier:
                    accepted.append(identifier)
            for item in agenda:
                await self.conversations.update_agenda(
                    tx, snapshot.scope, item, self.clock.now()
                )
            changed = bool(accepted or agenda or entities)
            await tx.execute(
                "UPDATE conversations SET language=?,dialogue_mode=?,memory_revision=memory_revision+? WHERE id=?",
                (language, dialogue_mode, int(changed), snapshot.scope.scope_id),
            )
            return accepted

        return await self.repository.database.write(apply)

    async def retrieve(
        self,
        scope: ScopeContext,
        predicates: list[str],
        subjects: list[str],
        topics: list[str],
        history: bool,
        direct_ids: list[str],
    ) -> MemoryView:
        items = await self.repository.items(scope, history)
        boundaries = [
            item
            for item in items
            if item["kind"] == "boundary" and item["validity_status"] == "current"
        ]
        facts = [item for item in items if item["kind"] != "boundary"]
        facts.sort(
            key=lambda item: (
                item["predicate"] not in predicates,
                item["subject_id"] not in subjects,
                item["predicate"] not in {"preferred_name", "preferred_language"},
            )
        )
        summaries = await self.repository.summaries(scope, topics, subjects)
        ids = list(direct_ids)
        references: list[tuple[str, int]] = []
        for summary in summaries[:3]:
            for claim in summary["payload"]["claims"]:
                ids.extend(source["message_id"] for source in claim["sources"])
        for item in facts[:20] + boundaries:
            ids.extend(source["message_id"] for source in item["sources"])
            references.extend(
                (source["message_id"], source["message_version"])
                for source in item["sources"]
            )
        sources = await self.conversations.sources(scope, ids)
        if history:
            versions = await self.conversations.source_versions(scope, references)
            sources = list(
                {
                    (message.id, message.version): message
                    for message in sources + versions
                }.values()
            )
        return MemoryView(
            tuple(facts), tuple(boundaries), tuple(summaries), tuple(sources)
        )

    async def handoff_data(
        self, tx: Transaction, scope: ScopeContext
    ) -> dict[str, Any]:
        items = await self.repository.items(scope, tx=tx)
        name = next(
            (
                item["value"]
                for item in items
                if item["subject_id"] == "user_self"
                and item["predicate"] == "preferred_name"
                and isinstance(item["value"], str)
            ),
            None,
        )
        agenda = await self.conversations.agenda(scope, tx=tx)
        return {
            "preferred_name": name,
            "current_topics": list(
                dict.fromkeys(
                    item["topic_kind"] for item in agenda if item["status"] == "open"
                )
            )[:3],
            "boundaries": [
                {
                    "predicate": item["predicate"],
                    "value": item["value"],
                    "qualifiers": item["qualifiers"],
                }
                for item in items
                if item["kind"] == "boundary"
            ],
        }

    async def publish_summary(
        self, scope: ScopeContext, block: CompactionBlock, summary: EpisodeSummary
    ) -> str:
        originals = {message.id: message for message in block.messages}
        entities = {item["id"] for item in await self.repository.entities(scope)}
        for claim in summary.claims:
            validate_sources(claim.sources, originals, user_only=False)
            if claim.subject_id not in entities:
                raise EvidenceError("Summary has an unknown subject")
            if claim.subject_id != "persona" and claim.modality in {
                "asserted",
                "planned",
                "negated",
            }:
                validate_sources(claim.sources, originals)
        return await self.repository.publish_summary(
            scope, block, summary, self.clock.now()
        )

    async def backup(self) -> Path:
        from app.shared.operations import create_backup

        async with self._deletion_lock:
            return await create_backup(
                self.repository.database, self.deletion_path, self.clock.now()
            )

    async def forget(
        self,
        scope: ScopeContext,
        delete_checkpoint: Callable[[str], Awaitable[None]],
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
            await delete_checkpoint(f"{scope.scope_id}:{scope.epoch}")
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

    async def apply_tombstones(
        self, delete_checkpoint: Callable[[str], Awaitable[None]]
    ) -> None:
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
            for epoch in range(1, item["epoch"]):
                await delete_checkpoint(f"{item['scope_id']}:{epoch}")


def source_hash(messages: tuple[Message, ...]) -> str:
    payload = [(message.id, message.version, message.text) for message in messages]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()
