from uuid import uuid4

from app.modules.jobs.entities import Job
from app.modules.jobs.schemas import JobPayload
from app.shared.entities import ScopeContext
from app.shared.storage import Database, Transaction, fetchall, fetchone


class JobRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def schedule(
        self,
        tx: Transaction,
        kind: str,
        dedup_key: str,
        now: float,
        run_at: float,
        scope: ScopeContext | None = None,
        payload: JobPayload | None = None,
    ) -> None:
        if kind in {"backup", "retention"} and await fetchone(
            tx, "SELECT 1 FROM scheduled_jobs WHERE dedup_key=? LIMIT 1", (dedup_key,)
        ):
            return
        await tx.execute(
            "INSERT OR IGNORE INTO scheduled_jobs(id,scope_id,epoch,kind,payload_json,dedup_key,run_at,created_at) VALUES(?,?,?,?,?,?,?,?)",
            (
                uuid4().hex,
                scope.scope_id if scope else None,
                scope.epoch if scope else None,
                kind,
                (payload or JobPayload()).model_dump_json(),
                dedup_key,
                run_at,
                now,
            ),
        )

    async def claim(self, now: float, kinds: tuple[str, ...]) -> Job | None:
        async def claim(tx: Transaction) -> Job | None:
            placeholders = ",".join("?" for _ in kinds)
            await tx.execute(
                "UPDATE scheduled_jobs SET status='pending',lease_token=NULL WHERE status='running' AND lease_until<?",
                (now,),
            )
            row = await fetchone(
                tx,
                f"SELECT * FROM scheduled_jobs WHERE status='pending' AND run_at<=? AND kind IN ({placeholders}) ORDER BY run_at,created_at LIMIT 1",
                (now, *kinds),
            )
            if row is None:
                return None
            token = uuid4().hex
            await tx.execute(
                "UPDATE scheduled_jobs SET status='running',attempts=attempts+1,lease_until=?,lease_token=? WHERE id=?",
                (now + 120, token, row["id"]),
            )
            scope = (
                ScopeContext(row["scope_id"], row["epoch"]) if row["scope_id"] else None
            )
            payload = JobPayload.model_validate_json(row["payload_json"])
            return Job(
                row["id"],
                row["kind"],
                scope,
                payload.data,
                row["attempts"] + 1,
                token,
                payload.terminal_error,
            )

        return await self.database.write(claim)

    async def finish(
        self,
        job: Job,
        now: float,
        error_code: str | None = None,
        permanent: bool = False,
        retry_after: float = 0,
    ) -> None:
        async def finish(tx: Transaction) -> None:
            status = (
                "done"
                if error_code is None
                else "dead_letter" if permanent or job.is_final_attempt else "pending"
            )
            await tx.execute(
                "UPDATE scheduled_jobs SET status=CASE WHEN ?='done' AND rerun_requested=1 THEN 'pending' ELSE ? END,attempts=CASE WHEN ?='done' THEN 0 ELSE attempts END,run_at=?,rerun_requested=0,lease_until=NULL,lease_token=NULL,error_code=? WHERE id=? AND status='running' AND lease_token=?",
                (
                    status,
                    status,
                    status,
                    (
                        now
                        if error_code is None
                        else now + max(min(60, 2 ** min(job.attempts, 6)), retry_after)
                    ),
                    error_code,
                    job.id,
                    job.lease_token,
                ),
            )

        await self.database.write(finish)

    async def mark_terminal(self, job: Job, error_code: str) -> bool:
        async def mark(tx: Transaction) -> bool:
            cursor = await tx.execute(
                "UPDATE scheduled_jobs SET payload_json=json_set(payload_json,'$.terminal_error',?) WHERE id=? AND status='running' AND lease_token=?",
                (error_code, job.id, job.lease_token),
            )
            return cursor.rowcount == 1

        return await self.database.write(mark)

    async def defer_terminal(self, job: Job, now: float) -> None:
        async def defer(tx: Transaction) -> None:
            await tx.execute(
                "UPDATE scheduled_jobs SET status='pending',run_at=?,lease_until=NULL,lease_token=NULL WHERE id=? AND status='running' AND lease_token=?",
                (now + min(30, 2 ** min(job.attempts, 5)), job.id, job.lease_token),
            )

        await self.database.write(defer)

    async def renew(self, job: Job, now: float) -> bool:
        async def renew(tx: Transaction) -> bool:
            cursor = await tx.execute(
                "UPDATE scheduled_jobs SET lease_until=? WHERE id=? AND status='running' AND lease_token=? AND lease_until>=?",
                (now + 120, job.id, job.lease_token, now),
            )
            return cursor.rowcount == 1

        return await self.database.write(renew)

    async def recover(self) -> None:
        async def recover(tx: Transaction) -> None:
            await tx.execute(
                "UPDATE scheduled_jobs SET status='pending',lease_token=NULL,lease_until=NULL WHERE status='running'"
            )

        await self.database.write(recover)

    async def reconcile_turns(self, now: float) -> None:
        async def reconcile(tx: Transaction) -> None:
            await tx.execute(
                "UPDATE scheduled_jobs SET status='pending',lease_token=NULL,lease_until=NULL WHERE status='running' AND lease_until<?",
                (now,),
            )
            rows = await fetchall(
                tx,
                "SELECT c.id,c.epoch,c.input_revision,MIN(v.received_at) received_at FROM conversations c JOIN messages m ON m.scope_id=c.id AND m.epoch=c.epoch JOIN message_versions v ON v.message_id=m.id AND v.version=m.current_version WHERE m.pending=1 AND c.control_state='enabled' AND NOT EXISTS(SELECT 1 FROM runtime_controls WHERE kill_switch=1) GROUP BY c.id,c.epoch,c.input_revision",
            )
            for row in rows:
                scope = ScopeContext(row["id"], row["epoch"])
                active = await fetchone(
                    tx,
                    "SELECT 1 FROM scheduled_jobs WHERE scope_id=? AND epoch=? AND kind='turn' AND status IN ('pending','running')",
                    (scope.scope_id, scope.epoch),
                )
                if active is None:
                    key = f"turn:{scope.scope_id}:{scope.epoch}:{row['input_revision']}"
                    existing = await fetchone(
                        tx,
                        "SELECT status FROM scheduled_jobs WHERE dedup_key=? ORDER BY rowid DESC LIMIT 1",
                        (key,),
                    )
                    if existing is None or existing["status"] in {"done", "cancelled"}:
                        await self.schedule(
                            tx,
                            "turn",
                            key,
                            row["received_at"],
                            now,
                            scope,
                            JobPayload(data={"input_revision": row["input_revision"]}),
                        )

        await self.database.write(reconcile)

    async def purge(self, tx: Transaction, scope_id: str) -> None:
        await tx.execute(
            "UPDATE scheduled_jobs SET status='cancelled',payload_json='{}' WHERE scope_id=?",
            (scope_id,),
        )

    async def metrics(self, now: float) -> dict[str, int | float]:
        async with self.database.read() as tx:
            row = await fetchone(
                tx,
                "SELECT SUM(status='pending') pending,SUM(status='dead_letter') dead_letter,MIN(CASE WHEN status='pending' THEN created_at END) oldest FROM scheduled_jobs",
            )
            assert row is not None
            return {
                "pending_jobs": int(row["pending"] or 0),
                "dead_letter_jobs": int(row["dead_letter"] or 0),
                "oldest_pending_seconds": (
                    max(0, now - row["oldest"]) if row["oldest"] else 0
                ),
            }
