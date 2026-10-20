import json

from app.shared.storage import Transaction, fetchall, fetchone


async def project_research_delivery(tx: Transaction, run_id: str) -> None:
    run = await fetchone(
        tx, "SELECT scope_id,epoch,metrics_json FROM agent_runs WHERE id=?", (run_id,)
    )
    if run is None:
        return
    rows = await fetchall(
        tx,
        "SELECT text,status FROM outbox_messages WHERE scope_id=? AND epoch=? AND kind='reply' AND json_extract(proposal_json,'$.research_run_id')=? ORDER BY rowid",
        (run["scope_id"], run["epoch"], run_id),
    )
    if not rows:
        return
    delivered = [row["text"] for row in rows if row["status"] == "delivered"]
    metrics = json.loads(run["metrics_json"])
    metrics["delivered_parts"] = delivered
    metrics["delivery_outcomes"] = [row["status"] for row in rows]
    observations = metrics.get("research_observations", [])
    if observations:
        observations[-1]["delivered_parts"] = delivered
    await tx.execute(
        "UPDATE agent_runs SET metrics_json=? WHERE id=? AND scope_id=? AND epoch=?",
        (
            json.dumps(metrics, ensure_ascii=False),
            run_id,
            run["scope_id"],
            run["epoch"],
        ),
    )
