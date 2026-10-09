import asyncio
import logging
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.modules.agent.context import ContextBuilder, TokenCounter
from app.modules.agent.entities import LanguageModel
from app.modules.agent.gateway import GatewayModel
from app.modules.agent.repository import AgentRepository
from app.modules.agent.service import AgentService
from app.modules.commerce.repository import CommerceRepository
from app.modules.commerce.service import CommerceService
from app.modules.conversations.repository import ConversationRepository
from app.modules.delivery.entities import ChannelSender
from app.modules.delivery.repository import DeliveryRepository
from app.modules.delivery.schemas import OutboxDraft
from app.modules.delivery.service import DeliveryService
from app.modules.followups.repository import FollowupRepository
from app.modules.followups.service import FollowupService
from app.modules.jobs.entities import Job
from app.modules.jobs.repository import JobRepository
from app.modules.jobs.service import JobService
from app.modules.memory.compaction import CompactionService
from app.modules.memory.repository import MemoryRepository
from app.modules.memory.service import MemoryService
from app.modules.telegram.adapter import TelegramSender
from app.modules.telegram.repository import TelegramRepository
from app.modules.telegram.service import TelegramService, service_text
from app.shared.clock import SystemClock
from app.shared.config import Settings
from app.shared.control import start_control
from app.shared.entities import Clock, ScopeContext
from app.shared.operations import ProcessLock, expire_backups
from app.shared.storage import Database, Transaction, fetchall

logger = logging.getLogger(__name__)


class Application:
    def __init__(
        self,
        settings: Settings,
        sender: ChannelSender,
        clock: Clock,
        model: LanguageModel,
        channel: str,
    ) -> None:
        self.settings = settings
        self.clock = clock
        self.sender = sender
        self.model = model
        self.channel = channel
        self.stack = AsyncExitStack()
        self.database = Database(
            settings.storage.directory / "app.db",
            settings.storage.queue_size,
            settings.storage.busy_timeout_ms,
        )

    @classmethod
    async def open(
        cls,
        settings: Settings,
        sender: ChannelSender,
        model: LanguageModel | None = None,
        clock: Clock | None = None,
        channel: str = "telegram-demo",
    ) -> "Application":
        instance = cls(
            settings,
            sender,
            clock or SystemClock(),
            model or GatewayModel(settings.llm, settings.runtime.llm_concurrency),
            channel,
        )
        try:
            await instance._open()
        except BaseException:
            await instance.stack.aclose()
            raise
        return instance

    async def _open(self) -> None:
        directory = self.settings.storage.directory
        self.stack.enter_context(ProcessLock(directory))
        migrations = Path(__file__).parent / "migrations"
        self.stack.push_async_callback(self.database.close)
        await self.database.open(migrations)
        self.checkpointer = await self.stack.enter_async_context(
            AsyncSqliteSaver.from_conn_string(str(directory / "checkpoints.db"))
        )
        await self.checkpointer.setup()
        self.conversations = ConversationRepository(self.database)
        self.jobs = JobRepository(self.database)
        self.delivery_repository = DeliveryRepository(self.database)
        self.followup_repository = FollowupRepository(self.database)
        self.commerce = CommerceService(
            CommerceRepository(self.database),
            self.conversations,
            self.clock,
            self.settings.commerce,
            self.settings.runtime.tester_ids,
        )
        self.memory = MemoryService(
            MemoryRepository(self.database),
            self.conversations,
            self.clock,
            directory / "deletions.jsonl",
            self._purge_artifacts,
        )
        await self.memory.apply_tombstones(self.checkpointer.adelete_thread)
        await self.commerce.repository.initialize(
            self.settings.commerce.package_minutes
        )
        await self.jobs.recover()
        await self.delivery_repository.recover()
        if self.settings.runtime.kill_switch:
            await self.delivery_repository.kill_switch(True)
        counter = TokenCounter(self.settings.llm.model)
        self.agent = AgentService(
            AgentRepository(self.database),
            self.conversations,
            self.memory,
            ContextBuilder(
                self.conversations,
                self.memory.repository,
                self.settings.memory,
                counter,
                self.clock,
                self.settings.llm.model,
            ),
            self.delivery_repository,
            self.commerce,
            self.model,
            self.clock,
            self.settings.runtime,
            self.checkpointer,
            self._preview_closed,
        )
        self.compaction = CompactionService(
            self.memory, self.conversations, self.model, counter, self.settings.memory
        )
        self.followups = FollowupService(
            self.followup_repository,
            self.jobs,
            self.settings.runtime,
            self.clock,
            self.agent.process_followup,
        )
        self.delivery = DeliveryService(
            self.delivery_repository,
            self.conversations,
            self.commerce,
            self.jobs,
            self.sender,
            self.clock,
            self.followups.after_reply,
        )
        self.telegram = TelegramService(
            TelegramRepository(self.database, self.channel),
            self.conversations,
            self.memory,
            self.delivery_repository,
            self.commerce,
            self.jobs,
            self.clock,
            self.settings.runtime,
            self.settings.storage.retention_days,
            self._cancel_scope,
            self.followups,
        )
        self.turn_worker = JobService(
            self.jobs, self.clock, {"turn": self._turn, "followup": self._followup}
        )
        self.control_worker = JobService(
            self.jobs,
            self.clock,
            {
                "forget": self._forget,
            },
        )
        self.background_worker = JobService(
            self.jobs,
            self.clock,
            {
                "compact": self.compaction.process,
                "backup": self._backup,
                "retention": self._retention,
            },
        )
        for path in directory.glob("*.db*"):
            path.chmod(0o600)

    async def _preview_closed(self, tx: Transaction, scope: ScopeContext) -> None:
        await self.followups.cancel_pending(tx, scope)

    async def _turn(self, job: Job) -> None:
        if job.scope is not None and isinstance(self.sender, TelegramSender):
            conversation = await self.conversations.get(job.scope.scope_id)
            async with self.database.read() as tx:
                killed = await self.delivery_repository.is_killed(tx)
            if (
                conversation.scope == job.scope
                and not killed
                and self.commerce.can_reply(conversation)
            ):
                async with self.sender.typing(conversation.chat_id):
                    await self.agent.process(job)
                return
        await self.agent.process(job)

    async def _cancel_scope(self, scope_id: str) -> None:
        await asyncio.gather(
            self.agent.cancel_scope(scope_id),
            self.delivery.cancel_scope(scope_id),
            self.compaction.cancel_scope(scope_id),
            self.followups.cancel_scope(scope_id),
        )

    async def _followup(self, job: Job) -> None:
        if job.scope is not None and isinstance(self.sender, TelegramSender):
            conversation = await self.conversations.get(job.scope.scope_id)
            if conversation.scope == job.scope and self.commerce.can_reply(
                conversation
            ):
                async with self.sender.typing(conversation.chat_id):
                    await self.followups.process(job)
                return
        await self.followups.process(job)

    async def _purge_artifacts(
        self, tx: Transaction, scope_id: str, new_epoch: int
    ) -> None:
        await self.followup_repository.purge(tx, scope_id)
        await self.jobs.purge(tx, scope_id)
        await self.delivery_repository.purge(tx, scope_id)
        await AgentRepository(self.database).purge(tx, scope_id)
        await self.commerce.repository.purge(tx, scope_id)
        await TelegramRepository(self.database, self.channel).purge(tx, scope_id)
        await self.conversations.purge(tx, scope_id, new_epoch)

    async def _forget(self, job: Job) -> None:
        if job.scope is None:
            raise ValueError("Deletion requires a scope")
        await self._cancel_scope(job.scope.scope_id)

        async def confirm(tx: Transaction, scope: ScopeContext) -> None:
            conversation = await self.conversations.get(scope.scope_id, tx)
            text = service_text("forgotten", conversation.language)
            await self.delivery_repository.prepare(
                tx,
                conversation.snapshot,
                OutboxDraft(
                    action_key=f"deleted:{scope.scope_id}:{scope.epoch}",
                    kind="service",
                    text=text,
                    deadline=self.clock.now() + 60,
                ),
                self.clock.now(),
            )

        await self.memory.forget(job.scope, self.checkpointer.adelete_thread, confirm)

    async def _backup(self, job: Job) -> None:
        await self.memory.backup()

    async def _retention(self, job: Job) -> None:
        cutoff = self.clock.now() - self.settings.storage.retention_days * 86400
        async with self.database.read() as tx:
            rows = await fetchall(
                tx,
                "SELECT id,epoch FROM conversations WHERE last_activity_at<? AND EXISTS(SELECT 1 FROM inbox_events i WHERE i.scope_id=conversations.id AND i.status!='deleted')",
                (cutoff,),
            )
        for row in rows:
            await self._cancel_scope(row["id"])
            await self.memory.forget(
                ScopeContext(row["id"], row["epoch"]), self.checkpointer.adelete_thread
            )

        async def quarantine(tx: Transaction) -> None:
            await tx.execute(
                "UPDATE inbox_events SET raw_json='{}',status='deleted' WHERE scope_id IS NULL AND created_at<?",
                (cutoff,),
            )

        await self.database.write(quarantine)
        await asyncio.to_thread(expire_backups, self.settings.storage.directory, cutoff)

    async def maintenance(self) -> None:
        while True:
            now = self.clock.now()

            async def schedule(tx: Transaction, scheduled_at: float = now) -> None:
                for kind, period in (
                    ("backup", self.settings.storage.backup_interval_seconds),
                    ("retention", 3600),
                ):
                    await self.jobs.schedule(
                        tx,
                        kind,
                        f"{kind}:{int(scheduled_at // period)}",
                        scheduled_at,
                        scheduled_at,
                    )

            await self.database.write(schedule)
            logger.info("health %s", await self.metrics())
            await asyncio.sleep(30)

    async def start_controls(self) -> None:
        server = await start_control(self.settings.storage.directory, self._control)
        await self.stack.enter_async_context(server)

    async def _control(self, request: dict[str, Any]) -> dict[str, Any]:
        command = request.get("command")
        if command == "metrics":
            return await self.metrics()
        if command == "backup":
            return {"path": str((await self.memory.backup()).resolve())}
        if command == "kill-switch" and isinstance(request.get("enabled"), bool):
            enabled = request["enabled"]
            await self.delivery_repository.kill_switch(enabled)
            if enabled:
                await self.followups.cancel_all()
                for scope_id in set(self.agent.tasks) | set(self.delivery.tasks):
                    await self._cancel_scope(scope_id)
            return {"kill_switch": enabled}
        raise ValueError("Unknown control command")

    async def metrics(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "writer_queue": self.database.queue_size,
            "active_generations": len(self.agent.active),
            "active_compactions": sum(
                len(tasks) for tasks in self.compaction.tasks.values()
            ),
            "active_sends": sum(len(tasks) for tasks in self.delivery.tasks.values()),
            "configured_llm_concurrency": self.settings.runtime.llm_concurrency,
        }
        metrics.update(await self.jobs.metrics(self.clock.now()))
        metrics.update(await self.delivery_repository.metrics())
        metrics.update(await self.memory.repository.metrics())
        metrics.update(await self.agent.repository.metrics())
        metrics.update(await self.followups.repository.metrics())
        return metrics

    async def drain(self, limit: int = 100) -> None:
        for _ in range(limit):
            worked = await self.control_worker.run_once()
            worked = await self.turn_worker.run_once() or worked
            worked = await self.delivery.run_once() or worked
            worked = await self.background_worker.run_once() or worked
            if not worked:
                return
        raise RuntimeError("Drain limit exceeded")

    async def close(self) -> None:
        for scope_id in list(self.followups.tasks):
            await self.followups.cancel_scope(scope_id)
        for scope_id in list(self.agent.tasks):
            await self.agent.cancel_scope(scope_id)
        for scope_id in list(self.delivery.tasks):
            await self.delivery.cancel_scope(scope_id)
        for scope_id in list(self.compaction.tasks):
            await self.compaction.cancel_scope(scope_id)
        await self.stack.aclose()
