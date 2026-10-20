import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import TypeVar

from app.modules.agent.dialogue_engine import DialogueEngine
from app.modules.agent.entities import LanguageModel
from app.modules.agent.repository import AgentRepository
from app.modules.commerce.service import CommerceService
from app.modules.conversations.repository import ConversationRepository
from app.modules.delivery.repository import DeliveryRepository
from app.modules.dialogue.tokens import ContextBudget
from app.modules.followups.entities import FollowupTicket
from app.modules.jobs.entities import Job
from app.modules.memory.service import MemoryService
from app.shared.config import RuntimeSettings
from app.shared.entities import Clock, ScopeContext
from app.shared.storage import Transaction

ResultT = TypeVar("ResultT")


class AgentService:
    def __init__(
        self,
        repository: AgentRepository,
        conversations: ConversationRepository,
        memory: MemoryService,
        context_builder: ContextBudget,
        delivery: DeliveryRepository,
        commerce: CommerceService,
        model: LanguageModel,
        clock: Clock,
        settings: RuntimeSettings,
    ) -> None:
        self.repository = repository
        self.dialogue = DialogueEngine(
            repository,
            conversations,
            memory,
            context_builder,
            delivery,
            commerce,
            model,
            clock,
            settings,
        )
        self.active = self.dialogue.active
        self.tasks: defaultdict[str, set[asyncio.Task[object]]] = defaultdict(set)

    async def _tracked(
        self, scope_id: str, action: Awaitable[ResultT]
    ) -> ResultT | None:
        task = asyncio.ensure_future(action)
        self.tasks[scope_id].add(task)
        try:
            return await task
        except asyncio.CancelledError:
            parent = asyncio.current_task()
            if parent is not None and parent.cancelling():
                raise
            return None
        finally:
            self.tasks[scope_id].discard(task)

    async def process(self, job: Job) -> None:
        if job.scope is None:
            raise ValueError("Turn requires scope")
        await self._tracked(job.scope.scope_id, self.dialogue.process(job))

    async def process_followup(
        self,
        ticket: FollowupTicket,
        on_prepared: Callable[[Transaction], Awaitable[None]],
    ) -> bool:
        return bool(
            await self._tracked(
                ticket.turn.scope.scope_id, self.dialogue.followup(ticket, on_prepared)
            )
        )

    async def cancel_obsolete(self, scope: ScopeContext, revision: int) -> None:
        await self.dialogue.cancel_obsolete(scope, revision)

    async def cancel_scope(self, scope_id: str) -> None:
        tasks = list(self.tasks.get(scope_id, set()))
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
