import asyncio
from collections import defaultdict

from app.modules.agent.context import TokenCounter, recent_messages
from app.modules.agent.entities import LanguageModel
from app.modules.agent.prompts import SUMMARY_VALIDATOR_RULES
from app.modules.agent.schemas import SemanticVerdict
from app.modules.conversations.entities import Message
from app.modules.conversations.repository import ConversationRepository
from app.modules.jobs.entities import Job
from app.modules.memory.entities import CompactionBlock, SummaryOverflowError
from app.modules.memory.service import MemoryService, source_hash
from app.shared.config import MemorySettings
from app.shared.entities import EvidenceError


class CompactionService:
    def __init__(
        self,
        memory: MemoryService,
        conversations: ConversationRepository,
        model: LanguageModel,
        counter: TokenCounter,
        settings: MemorySettings,
    ) -> None:
        self.memory = memory
        self.conversations = conversations
        self.model = model
        self.counter = counter
        self.settings = settings
        self.tasks: dict[str, set[asyncio.Task[None]]] = defaultdict(set)

    async def process(self, job: Job) -> None:
        if job.scope is None:
            raise ValueError("Compaction requires a scope")
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

    async def cancel_scope(self, scope_id: str) -> None:
        tasks = list(self.tasks.get(scope_id, set()))
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _process(self, job: Job) -> None:
        if job.scope is None:
            raise ValueError("Compaction requires a scope")
        messages = await self.conversations.completed_messages(job.scope)
        covered = await self.memory.repository.covered_ids(job.scope)
        recent = {
            message.id
            for message in recent_messages(messages, self.settings.recent_turns)
        }
        pending = [message for message in messages if message.id not in covered]
        if (
            self.counter.count([message.as_data() for message in pending])
            < self.settings.history_threshold_tokens
        ):
            return
        eligible = [message for message in pending if message.id not in recent]
        groups: list[list[Message]] = []
        for message in eligible:
            if not groups or groups[-1][-1].turn_id != message.turn_id:
                groups.append([])
            groups[-1].append(message)
        bounded_groups = []
        for group in groups:
            if (
                self.counter.count([message.as_data() for message in group])
                <= self.settings.block_tokens
            ):
                bounded_groups.append(group)
            else:
                bounded_groups.extend([message] for message in group)
        block: list[Message] = []
        for group in bounded_groups:
            if (
                self.counter.count([message.as_data() for message in group])
                > self.settings.block_tokens
            ):
                raise ValueError(
                    "An original message exceeds the compaction block budget"
                )
            if (
                block
                and self.counter.count([message.as_data() for message in block + group])
                > self.settings.block_tokens
            ):
                await self._publish(job, block)
                block = []
            block.extend(group)
        if block:
            await self._publish(job, block)

    async def _publish(self, job: Job, messages: list[Message]) -> None:
        assert job.scope is not None
        originals = tuple(messages)
        text = " ".join(message.text for message in originals).casefold()
        identifiers = {message.id for message in originals}
        items = await self.memory.repository.items(job.scope, history=True)
        owners = {
            item["subject_id"]
            for item in items
            if any(source["message_id"] in identifiers for source in item["sources"])
        }
        entities = tuple(
            entity
            for entity in await self.memory.repository.entities(job.scope)
            if entity["id"] in {"user_self", "persona"} | owners
            or entity["name"]
            and entity["name"].casefold() in text
        )
        block = CompactionBlock(originals, source_hash(originals), entities)
        try:
            summary = await self.model.summarize(
                block, 45, self.settings.summary_tokens
            )
        except SummaryOverflowError:
            if len(messages) == 1:
                raise
            midpoint = len(messages) // 2
            await self._publish(job, messages[:midpoint])
            await self._publish(job, messages[midpoint:])
            return
        verdict = await self.model.complete(
            "validate_summary",
            SUMMARY_VALIDATOR_RULES,
            {
                "summary": summary.model_dump(),
                "originals": [message.as_data() for message in block.messages],
            },
            SemanticVerdict,
            45,
        )
        if verdict.decision != "accept":
            raise EvidenceError("Episode summary failed semantic validation")
        await self.memory.publish_summary(job.scope, block, summary)
