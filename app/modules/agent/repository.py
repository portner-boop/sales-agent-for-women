import json
from math import ceil
from typing import Any
from uuid import uuid4

from app.modules.agent.schemas import ContextManifest
from app.modules.conversations.entities import Turn
from app.shared.storage import Database, Transaction, fetchall, require_scope


class AgentRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def metrics(self) -> dict[str, Any]:
        async with self.database.read() as tx:
            rows = await fetchall(
                tx, "SELECT status,COUNT(*) count FROM agent_runs GROUP BY status"
            )
            runs = await fetchall(
                tx,
                "SELECT metrics_json FROM agent_runs ORDER BY started_at DESC LIMIT 1000",
            )
            latencies = await fetchall(
                tx,
                "SELECT o.created_at-r.started_at+COALESCE(json_extract(r.metrics_json,'$.queued_seconds'),0) seconds FROM outbox_messages o JOIN agent_runs r ON o.turn_id=r.turn_id WHERE o.status='delivered' AND o.kind='reply' AND o.predecessor_id IS NULL AND COALESCE(json_extract(o.proposal_json,'$.origin'),'')!='followup' ORDER BY o.created_at DESC LIMIT 1000",
            )
            completions = await fetchall(
                tx,
                "SELECT MAX(o.created_at)-r.started_at+COALESCE(json_extract(r.metrics_json,'$.queued_seconds'),0) seconds FROM outbox_messages o JOIN agent_runs r ON o.turn_id=r.turn_id WHERE o.kind='reply' AND COALESCE(json_extract(o.proposal_json,'$.origin'),'')!='followup' GROUP BY r.id HAVING COUNT(*)=SUM(CASE WHEN o.status='delivered' THEN 1 ELSE 0 END) ORDER BY MAX(o.created_at) DESC LIMIT 1000",
            )
        values = [json.loads(row["metrics_json"]) for row in runs]
        events = [event for value in values for event in value.get("llm_events", [])]
        seconds = sorted(float(row["seconds"]) for row in latencies)
        completion_seconds = sorted(float(row["seconds"]) for row in completions)
        return {
            **{f"runs_{row['status']}": row["count"] for row in rows},
            "llm_calls_last_1000_runs": sum(
                value.get("llm_calls", 0) for value in values
            ),
            "input_tokens_last_1000_runs": sum(
                event.get("input_tokens", 0) for event in events
            ),
            "output_tokens_last_1000_runs": sum(
                event.get("output_tokens", 0) for event in events
            ),
            "reply_latency_samples": len(seconds),
            "reply_latency_p95_seconds": (
                seconds[max(0, ceil(len(seconds) * 0.95) - 1)] if seconds else None
            ),
            "reply_completion_samples": len(completion_seconds),
            "reply_completion_p95_seconds": (
                completion_seconds[max(0, ceil(len(completion_seconds) * 0.95) - 1)]
                if completion_seconds
                else None
            ),
        }

    async def purge(self, tx: Transaction, scope_id: str) -> None:
        await tx.execute("DELETE FROM agent_runs WHERE scope_id=?", (scope_id,))

    async def start(self, turn: Turn, now: float) -> str:
        identifier = uuid4().hex

        async def start(tx: Transaction) -> None:
            await require_scope(tx, turn.scope)
            await tx.execute(
                "INSERT INTO agent_runs(id,scope_id,epoch,turn_id,status,started_at) VALUES(?,?,?,?,'running',?)",
                (identifier, turn.scope.scope_id, turn.scope.epoch, turn.id, now),
            )

        await self.database.write(start)
        return identifier

    async def finish(
        self,
        identifier: str,
        status: str,
        now: float,
        manifest: ContextManifest | None,
        metrics: dict[str, Any],
        error_code: str | None = None,
    ) -> None:
        async def finish(tx: Transaction) -> None:
            await tx.execute(
                "UPDATE agent_runs SET status=?,finished_at=?,manifest_json=?,metrics_json=?,error_code=? WHERE id=?",
                (
                    status,
                    now,
                    manifest.model_dump_json() if manifest else "{}",
                    json.dumps(metrics),
                    error_code,
                    identifier,
                ),
            )

        await self.database.write(finish)
