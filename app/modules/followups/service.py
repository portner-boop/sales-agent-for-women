import asyncio
import random
from collections import defaultdict
from collections.abc import Awaitable, Callable

from app.modules.conversations.entities import Conversation
from app.modules.delivery.entities import OutboxMessage
from app.modules.followups.entities import FollowupTicket
from app.modules.followups.repository import FollowupRepository
from app.modules.followups.schemas import FollowupJob
from app.modules.jobs.entities import Job, JobRetryError
from app.modules.jobs.repository import JobRepository
from app.modules.jobs.schemas import JobPayload
from app.shared.config import RuntimeSettings
from app.shared.entities import Clock, ScopeContext
from app.shared.storage import Transaction


class FollowupService:
    def __init__(
        self,
        repository: FollowupRepository,
        jobs: JobRepository,
        settings: RuntimeSettings,
        clock: Clock,
        processor: Callable[
            [FollowupTicket, Callable[[Transaction], Awaitable[None]]], Awaitable[bool]
        ],
    ) -> None:
        self.repository, self.jobs = repository, jobs
        self.settings, self.clock, self.processor = settings, clock, processor
        self.tasks: dict[str, set[asyncio.Task[None]]] = defaultdict(set)

    async def after_reply(
        self, tx: Transaction, conversation: Conversation, message: OutboxMessage
    ) -> None:
        if not self.settings.followup_enabled or not message.proposal.get(
            "followup_allowed"
        ):
            return
        ordinal = int(message.proposal.get("followup_ordinal", 0)) + 1
        if ordinal > self.settings.followup_max_count:
            return
        now = self.clock.now()
        silence_started_at = (
            float(message.proposal["silence_started_at"]) if ordinal > 1 else now
        )
        due = (
            silence_started_at + self.settings.followup_second_seconds
            if ordinal > 1
            else now
            + random.uniform(
                self.settings.followup_min_seconds, self.settings.followup_max_seconds
            )
        )
        if due + self.settings.followup_ttl_seconds <= now:
            return
        identifier = await self.repository.schedule(
            tx,
            message.scope,
            message.id,
            now,
            due,
            due + self.settings.followup_ttl_seconds,
            ordinal,
            silence_started_at,
        )
        if identifier is not None:
            await self.jobs.schedule(
                tx,
                "followup",
                f"followup:{identifier}",
                now,
                due,
                message.scope,
                JobPayload(data=FollowupJob(followup_id=identifier).model_dump()),
            )

    async def process(self, job: Job) -> None:
        if job.scope is None:
            raise ValueError("Follow-up requires a scope")
        task = asyncio.create_task(self._process(job))
        self.tasks[job.scope.scope_id].add(task)
        try:
            await task
        except asyncio.CancelledError:
            parent = asyncio.current_task()
            if parent is not None and parent.cancelling():
                raise
        finally:
            self.tasks[job.scope.scope_id].discard(task)

    async def _process(self, job: Job) -> None:
        assert job.scope is not None
        data = FollowupJob.model_validate(job.payload)
        if not self.settings.followup_enabled:

            async def disable(tx: Transaction) -> None:
                await self.repository.finish(
                    tx, data.followup_id, "cancelled", "DISABLED"
                )

            await self.repository.database.write(disable)
            return
        ticket = await self.repository.start(
            job.scope, data.followup_id, self.clock.now()
        )
        if ticket is None:
            return
        if ticket.ordinal > self.settings.followup_max_count:

            async def limit(tx: Transaction) -> None:
                await self.repository.finish(
                    tx, ticket.id, "cancelled", "OPPORTUNITY_LIMIT"
                )

            await self.repository.database.write(limit)
            return

        async def prepared(tx: Transaction) -> None:
            await self.repository.prepared(tx, ticket)

        try:
            success = await self.processor(ticket, prepared)
        except JobRetryError as error:
            reason = error.code
            retry_at = self.clock.now() + max(2**job.attempts, error.retry_after)
            terminal = (
                error.permanent or job.is_final_attempt or retry_at >= ticket.expires_at
            )

            async def failed(tx: Transaction) -> None:
                if terminal:
                    await self.repository.finish(tx, ticket.id, "skipped", reason)
                else:
                    await self.repository.retry(tx, ticket, reason)

            await self.repository.database.write(failed)
            raise JobRetryError(
                error.code, retry_after=error.retry_after, permanent=terminal
            ) from error
        if not success:

            async def skip(tx: Transaction) -> None:
                await self.repository.finish(
                    tx,
                    ticket.id,
                    "skipped",
                    (
                        "REQUIRED_RESPONSE_FAILED"
                        if ticket.requires_response
                        else "NO_CANDIDATE"
                    ),
                )

            await self.repository.database.write(skip)

    async def cancel_pending(self, tx: Transaction, scope: ScopeContext) -> None:
        await self.repository.cancel(tx, scope)
        await tx.execute(
            "UPDATE scheduled_jobs SET status='cancelled',error_code='NEW_INPUT',lease_token=NULL,lease_until=NULL WHERE scope_id=? AND epoch=? AND kind='followup' AND status IN ('pending','running')",
            (scope.scope_id, scope.epoch),
        )

    async def cancel_scope(self, scope_id: str) -> None:
        tasks = list(self.tasks.get(scope_id, set()))
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def cancel_all(self) -> None:
        async def cancel(tx: Transaction) -> None:
            await tx.execute(
                "UPDATE turns SET status='cancelled' WHERE id IN (SELECT turn_id FROM followups WHERE status IN ('pending','generating'))"
            )
            await tx.execute(
                "UPDATE followups SET status='cancelled',reason='KILL_SWITCH' WHERE status IN ('pending','generating')"
            )
            await tx.execute(
                "UPDATE scheduled_jobs SET status='cancelled',error_code='KILL_SWITCH',lease_token=NULL,lease_until=NULL WHERE kind='followup' AND status IN ('pending','running')"
            )

        await self.repository.database.write(cancel)
        for scope_id in list(self.tasks):
            await self.cancel_scope(scope_id)
