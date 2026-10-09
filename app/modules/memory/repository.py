import json
from typing import Any
from uuid import uuid4

from app.modules.memory.entities import CompactionBlock
from app.modules.memory.schemas import (
    EntityCandidate,
    EpisodeSummary,
    MemoryOperation,
)
from app.shared.entities import EvidenceError, ScopeContext
from app.shared.storage import (
    Database,
    Transaction,
    fetchall,
    fetchone,
    require_scope,
)


class MemoryRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def metrics(self) -> dict[str, int]:
        async with self.database.read() as tx:
            summaries = await fetchone(
                tx,
                "SELECT SUM(status='current') current_count,SUM(status='stale') stale_count FROM memory_summaries",
            )
            messages = await fetchone(tx, "SELECT COUNT(*) count FROM messages")
            covered = await fetchone(
                tx,
                "SELECT COUNT(DISTINCT ss.message_id) count FROM summary_sources ss JOIN memory_summaries s ON ss.summary_id=s.id JOIN messages m ON ss.message_id=m.id AND ss.message_version=m.current_version WHERE s.status='current'",
            )
            assert (
                summaries is not None and messages is not None and covered is not None
            )
            return {
                "current_summaries": int(summaries["current_count"] or 0),
                "stale_summaries": int(summaries["stale_count"] or 0),
                "source_messages": messages["count"],
                "summarized_messages": covered["count"],
                "uncovered_messages": messages["count"] - covered["count"],
            }

    async def ensure_entities(self, tx: Transaction, scope: ScopeContext) -> None:
        await require_scope(tx, scope)
        for identifier, kind in (("user_self", "person"), ("persona", "persona")):
            await tx.execute(
                "INSERT OR IGNORE INTO entities(id,scope_id,epoch,kind) VALUES(?,?,?,?)",
                (identifier, scope.scope_id, scope.epoch, kind),
            )

    async def entities(self, scope: ScopeContext) -> list[dict[str, Any]]:
        async with self.database.read() as tx:
            await require_scope(tx, scope)
            return [
                dict(row)
                for row in await fetchall(
                    tx,
                    "SELECT id,kind,name,aliases_json FROM entities WHERE scope_id=? AND epoch=?",
                    (scope.scope_id, scope.epoch),
                )
            ]

    async def add_entity(
        self, tx: Transaction, scope: ScopeContext, candidate: EntityCandidate
    ) -> None:
        await tx.execute(
            "INSERT OR IGNORE INTO entities(id,scope_id,epoch,kind,name) VALUES(?,?,?,?,?)",
            (candidate.id, scope.scope_id, scope.epoch, candidate.kind, candidate.name),
        )

    async def items(
        self, scope: ScopeContext, history: bool = False, tx: Transaction | None = None
    ) -> list[dict[str, Any]]:
        if tx is None:
            async with self.database.read() as connection:
                return await self.items(scope, history, connection)
        await require_scope(tx, scope)
        status_clause = (
            "validity_status IN ('current','superseded','historical','disputed','retracted')"
            if history
            else "validity_status='current'"
        )
        rows = await fetchall(
            tx,
            f"SELECT * FROM memory_items WHERE scope_id=? AND epoch=? AND {status_clause} ORDER BY recorded_at DESC,id",
            (scope.scope_id, scope.epoch),
        )
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["value"] = json.loads(item.pop("value_json"))
            item["qualifiers"] = json.loads(item.pop("qualifiers_json"))
            sources = await fetchall(
                tx,
                "SELECT s.message_id,s.message_version,s.evidence_text FROM memory_sources s JOIN messages m ON m.id=s.message_id WHERE s.memory_id=? AND (m.current_version=s.message_version OR ?)",
                (row["id"], history),
            )
            item["sources"] = [dict(source) for source in sources]
            if sources:
                result.append(item)
        return result

    async def apply(
        self,
        tx: Transaction,
        scope: ScopeContext,
        operation: MemoryOperation,
        now: float,
    ) -> str | None:
        await require_scope(tx, scope)
        if operation.operation in {"no_change", "needs_clarification"}:
            return None
        target = None
        if not operation.target_memory_id and operation.operation in {
            "supersede",
            "qualify",
            "retract",
            "confirm",
        }:
            candidates = await fetchall(
                tx,
                "SELECT id FROM memory_items WHERE scope_id=? AND epoch=? AND subject_id=? AND predicate=? AND qualifiers_json=? AND validity_status='current'",
                (
                    scope.scope_id,
                    scope.epoch,
                    operation.subject_id,
                    operation.predicate,
                    json.dumps(operation.qualifiers, sort_keys=True),
                ),
            )
            if len(candidates) == 1:
                operation = operation.model_copy(
                    update={"target_memory_id": candidates[0]["id"]}
                )
            elif not candidates and operation.operation == "supersede":
                current = await fetchone(
                    tx,
                    "SELECT id FROM memory_items WHERE scope_id=? AND epoch=? AND subject_id=? AND predicate=? AND validity_status='current' LIMIT 1",
                    (
                        scope.scope_id,
                        scope.epoch,
                        operation.subject_id,
                        operation.predicate,
                    ),
                )
                if current is None:
                    operation = operation.model_copy(update={"operation": "add"})
        if operation.target_memory_id:
            target = await fetchone(
                tx,
                "SELECT * FROM memory_items WHERE id=? AND scope_id=? AND epoch=?",
                (operation.target_memory_id, scope.scope_id, scope.epoch),
            )
            if (
                target is None
                or target["subject_id"] != operation.subject_id
                or target["predicate"] != operation.predicate
            ):
                raise EvidenceError("Memory target does not match subject or predicate")
            if (
                operation.operation != "qualify"
                and json.loads(target["qualifiers_json"]) != operation.qualifiers
            ):
                raise EvidenceError("Memory target has different applicability")
        if (
            operation.operation in {"supersede", "qualify", "retract", "confirm"}
            and target is None
        ):
            raise EvidenceError("Operation requires an existing target")
        if operation.operation == "retract":
            assert target is not None
            await tx.execute(
                "UPDATE memory_items SET validity_status='retracted' WHERE id=?",
                (target["id"],),
            )
            await self.invalidate_fact_summaries(tx, scope, target["id"])
            return str(target["id"])
        value_json = json.dumps(operation.value, ensure_ascii=False, sort_keys=True)
        qualifiers_json = json.dumps(operation.qualifiers, sort_keys=True)
        if (
            operation.operation == "qualify"
            and target is not None
            and (
                target["value_json"] != value_json
                or target["modality"] != operation.modality
            )
        ):
            raise EvidenceError("Qualification must preserve the value and modality")
        identical = await fetchone(
            tx,
            "SELECT id FROM memory_items WHERE scope_id=? AND epoch=? AND subject_id=? AND predicate=? AND value_json=? AND qualifiers_json=? AND modality=? AND validity_status='current'",
            (
                scope.scope_id,
                scope.epoch,
                operation.subject_id,
                operation.predicate,
                value_json,
                qualifiers_json,
                operation.modality,
            ),
        )
        if operation.operation == "confirm":
            assert target is not None
            if (
                target["value_json"] != value_json
                or target["modality"] != operation.modality
                or target["validity_status"] != "current"
            ):
                raise EvidenceError("Confirmation must preserve a current fact")
            identifier = str(target["id"])
        elif identical:
            identifier = str(identical["id"])
        else:
            identifier = uuid4().hex
            if target:
                await tx.execute(
                    "UPDATE memory_items SET validity_status='superseded' WHERE id=?",
                    (target["id"],),
                )
                await self.invalidate_fact_summaries(tx, scope, target["id"])
            await tx.execute(
                "INSERT INTO memory_items VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    scope.scope_id,
                    scope.epoch,
                    operation.kind,
                    operation.subject_id,
                    operation.predicate,
                    value_json,
                    qualifiers_json,
                    operation.modality,
                    "self_reported",
                    "current",
                    target["id"] if target else None,
                    operation.event_time,
                    now,
                    now,
                ),
            )
        for source in operation.sources:
            await tx.execute(
                "INSERT OR IGNORE INTO memory_sources VALUES(?,?,?,?,?,?)",
                (
                    identifier,
                    scope.scope_id,
                    scope.epoch,
                    source.message_id,
                    source.message_version,
                    source.evidence_text,
                ),
            )
        return identifier

    async def invalidate_fact_summaries(
        self, tx: Transaction, scope: ScopeContext, memory_id: str
    ) -> None:
        await tx.execute(
            "UPDATE memory_summaries SET status='stale' WHERE scope_id=? AND epoch=? AND id IN (SELECT ss.summary_id FROM summary_sources ss JOIN memory_sources ms ON ss.message_id=ms.message_id AND ss.message_version=ms.message_version WHERE ms.memory_id=?)",
            (scope.scope_id, scope.epoch, memory_id),
        )

    async def invalidate_source(
        self, tx: Transaction, scope: ScopeContext, message_id: str
    ) -> None:
        await tx.execute(
            "UPDATE memory_items SET validity_status='disputed' WHERE scope_id=? AND epoch=? AND id IN (SELECT memory_id FROM memory_sources WHERE message_id=?)",
            (scope.scope_id, scope.epoch, message_id),
        )
        await tx.execute(
            "UPDATE memory_summaries SET status='stale' WHERE scope_id=? AND epoch=? AND id IN (SELECT summary_id FROM summary_sources WHERE message_id=?)",
            (scope.scope_id, scope.epoch, message_id),
        )
        await tx.execute(
            "UPDATE conversations SET memory_revision=memory_revision+1 WHERE id=?",
            (scope.scope_id,),
        )

    async def summaries(
        self, scope: ScopeContext, topics: list[str], subjects: list[str]
    ) -> list[dict[str, Any]]:
        async with self.database.read() as tx:
            await require_scope(tx, scope)
            rows = await fetchall(
                tx,
                "SELECT * FROM memory_summaries WHERE scope_id=? AND epoch=? AND status='current' ORDER BY last_sequence DESC",
                (scope.scope_id, scope.epoch),
            )
            result = []
            for row in rows:
                payload = json.loads(row["payload_json"])
                topic_match = bool(set(topics).intersection(payload["topic_kinds"]))
                subject_match = bool(set(subjects).intersection(payload["subject_ids"]))
                rank = (
                    0
                    if topic_match and subject_match
                    else 1 if topic_match else 2 if subject_match else 3
                )
                result.append(
                    {
                        "id": row["id"],
                        "first_sequence": row["first_sequence"],
                        "last_sequence": row["last_sequence"],
                        "payload": payload,
                        "relevant": rank < 3,
                        "rank": rank,
                    }
                )
            return sorted(
                result, key=lambda item: (item["rank"], -item["last_sequence"])
            )

    async def covered_ids(self, scope: ScopeContext) -> set[str]:
        async with self.database.read() as tx:
            await require_scope(tx, scope)
            rows = await fetchall(
                tx,
                "SELECT ss.message_id FROM summary_sources ss JOIN memory_summaries s ON s.id=ss.summary_id WHERE s.scope_id=? AND s.epoch=? AND s.status='current'",
                (scope.scope_id, scope.epoch),
            )
            return {str(row["message_id"]) for row in rows}

    async def publish_summary(
        self,
        scope: ScopeContext,
        block: CompactionBlock,
        summary: EpisodeSummary,
        now: float,
    ) -> str:
        async def publish(tx: Transaction) -> str:
            await require_scope(tx, scope)
            expected = {message.id for message in block.messages}
            if set(summary.covered_message_ids) != expected or len(
                summary.covered_message_ids
            ) != len(expected):
                raise EvidenceError("Summary coverage does not match the block")
            for message in block.messages:
                row = await fetchone(
                    tx,
                    "SELECT current_version FROM messages WHERE id=? AND scope_id=? AND epoch=?",
                    (message.id, scope.scope_id, scope.epoch),
                )
                if row is None or row["current_version"] != message.version:
                    raise EvidenceError("Summary source was edited")
            claims = []
            for claim in summary.claims:
                statuses: list[str] = []
                for source in claim.sources:
                    rows = await fetchall(
                        tx,
                        "SELECT validity_status FROM memory_items m JOIN memory_sources s ON s.memory_id=m.id WHERE m.scope_id=? AND m.epoch=? AND s.message_id=? AND s.message_version=?",
                        (
                            scope.scope_id,
                            scope.epoch,
                            source.message_id,
                            source.message_version,
                        ),
                    )
                    statuses.extend(row["validity_status"] for row in rows)
                status = (
                    "disputed"
                    if "disputed" in statuses
                    else (
                        "historical"
                        if any(value != "current" for value in statuses)
                        else "current"
                    )
                )
                claims.append(claim.model_copy(update={"validity_status": status}))
            published = summary.model_copy(update={"claims": claims})
            identifier = uuid4().hex
            existing = await fetchone(
                tx,
                "SELECT id FROM memory_summaries WHERE scope_id=? AND epoch=? AND source_hash=? AND status='current'",
                (scope.scope_id, scope.epoch, block.source_hash),
            )
            if existing:
                return str(existing["id"])
            await tx.execute(
                "INSERT INTO memory_summaries VALUES(?,?,?,?,?,?,?,'current',?)",
                (
                    identifier,
                    scope.scope_id,
                    scope.epoch,
                    published.model_dump_json(),
                    block.messages[0].sequence,
                    block.messages[-1].sequence,
                    block.source_hash,
                    now,
                ),
            )
            for message in block.messages:
                await tx.execute(
                    "INSERT INTO summary_sources VALUES(?,?,?,?,?)",
                    (
                        identifier,
                        scope.scope_id,
                        scope.epoch,
                        message.id,
                        message.version,
                    ),
                )
            return identifier

        return await self.database.write(publish)

    async def purge(
        self, tx: Transaction, scope_id: str, new_epoch: int, now: float
    ) -> None:
        await tx.execute("DELETE FROM memory_summaries WHERE scope_id=?", (scope_id,))
        await tx.execute(
            "UPDATE memory_items SET supersedes_id=NULL WHERE scope_id=?", (scope_id,)
        )
        await tx.execute("DELETE FROM memory_items WHERE scope_id=?", (scope_id,))

    async def finish_purge(
        self, tx: Transaction, scope_id: str, new_epoch: int, now: float
    ) -> None:
        await tx.execute("DELETE FROM entities WHERE scope_id=?", (scope_id,))
        await tx.execute(
            "INSERT INTO deletion_ledger VALUES(?,?,?) ON CONFLICT(scope_id) DO UPDATE SET epoch=MAX(epoch,excluded.epoch),deleted_at=excluded.deleted_at",
            (scope_id, new_epoch, now),
        )
