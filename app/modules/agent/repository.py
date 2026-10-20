import json
from math import ceil
from typing import Any
from uuid import uuid4

from app.modules.conversations.entities import Turn
from app.modules.delivery.observations import project_research_delivery
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
                "SELECT CAST(json_extract(o.proposal_json,'$.delivery_finished_at') AS REAL)-r.started_at+COALESCE(json_extract(r.metrics_json,'$.queued_seconds'),0) seconds FROM outbox_messages o JOIN agent_runs r ON o.turn_id=r.turn_id WHERE o.status='delivered' AND json_extract(o.proposal_json,'$.delivery_finished_at') IS NOT NULL AND o.kind='reply' AND o.predecessor_id IS NULL AND COALESCE(json_extract(o.proposal_json,'$.origin'),'')!='followup' ORDER BY o.created_at DESC LIMIT 1000",
            )
            completions = await fetchall(
                tx,
                "SELECT MAX(CAST(json_extract(o.proposal_json,'$.delivery_finished_at') AS REAL))-r.started_at+COALESCE(json_extract(r.metrics_json,'$.queued_seconds'),0) seconds FROM outbox_messages o JOIN agent_runs r ON o.turn_id=r.turn_id WHERE o.kind='reply' AND COALESCE(json_extract(o.proposal_json,'$.origin'),'')!='followup' GROUP BY r.id HAVING MIN(json_extract(o.proposal_json,'$.delivery_finished_at')) IS NOT NULL AND COUNT(*)=SUM(CASE WHEN o.status='delivered' THEN 1 ELSE 0 END) ORDER BY MAX(o.created_at) DESC LIMIT 1000",
            )
            transport = await fetchall(
                tx,
                "SELECT json_extract(proposal_json,'$.transport_seconds') seconds FROM outbox_messages WHERE json_extract(proposal_json,'$.transport_seconds') IS NOT NULL ORDER BY created_at DESC LIMIT 1000",
            )
        values = [json.loads(row["metrics_json"]) for row in runs]
        events = [event for value in values for event in value.get("llm_events", [])]
        seconds = sorted(float(row["seconds"]) for row in latencies)
        completion_seconds = sorted(float(row["seconds"]) for row in completions)
        phases = {}
        for phase in (
            "queued_seconds",
            "context_seconds",
            "generation_seconds",
            "repair_seconds",
        ):
            samples = sorted(float(value[phase]) for value in values if phase in value)
            phases[phase] = {
                "samples": len(samples),
                "p50": samples[len(samples) // 2] if samples else None,
                "p95": (
                    samples[max(0, ceil(len(samples) * 0.95) - 1)] if samples else None
                ),
            }
        phase_samples = {
            "transport_seconds": [float(row["seconds"]) for row in transport],
            "llm_queue_seconds": [
                float(event["queue_seconds"])
                for event in events
                if "queue_seconds" in event
            ],
            "llm_request_seconds": [
                float(event["model_seconds"])
                for event in events
                if "model_seconds" in event
            ],
        }
        for phase, samples in phase_samples.items():
            samples.sort()
            phases[phase] = {
                "samples": len(samples),
                "p50": samples[len(samples) // 2] if samples else None,
                "p95": (
                    samples[max(0, ceil(len(samples) * 0.95) - 1)] if samples else None
                ),
            }
        return {
            "phase_latency": phases,
            "policy_versions": sorted(
                {
                    value["policy_version"]
                    for value in values
                    if "policy_version" in value
                }
            ),
            "repair_runs_last_1000": sum(
                any(
                    event["task"] == "dialogue_repair"
                    for event in value.get("llm_events", [])
                )
                for value in values
            ),
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
        manifest: dict[str, Any] | None,
        metrics: dict[str, Any],
        error_code: str | None = None,
    ) -> None:
        async def finish(tx: Transaction) -> None:
            await tx.execute(
                "UPDATE agent_runs SET status=?,finished_at=?,manifest_json=?,metrics_json=?,error_code=? WHERE id=?",
                (
                    status,
                    now,
                    json.dumps(manifest or {}),
                    json.dumps(metrics),
                    error_code,
                    identifier,
                ),
            )

            await project_research_delivery(tx, identifier)

        await self.database.write(finish)
