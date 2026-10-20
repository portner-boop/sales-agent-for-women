import asyncio
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable

from app.modules.commerce.service import CommerceService
from app.modules.conversations.entities import Conversation
from app.modules.conversations.repository import ConversationRepository
from app.modules.delivery.entities import (
    ChannelSender,
    OutboxMessage,
    SendResult,
    TypingSender,
)
from app.modules.delivery.repository import DeliveryRepository
from app.shared.entities import Clock
from app.shared.storage import Transaction

logger = logging.getLogger(__name__)


class DeliveryService:
    def __init__(
        self,
        repository: DeliveryRepository,
        conversations: ConversationRepository,
        commerce: CommerceService,
        sender: ChannelSender,
        clock: Clock,
        after_reply: (
            Callable[[Transaction, Conversation, OutboxMessage], Awaitable[None]] | None
        ) = None,
    ) -> None:
        self.repository = repository
        self.conversations = conversations
        self.commerce = commerce
        self.sender = sender
        self.clock = clock
        self.after_reply = after_reply
        self.tasks: dict[str, set[asyncio.Task[None]]] = defaultdict(set)
        self.locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.semaphore = asyncio.Semaphore(4)
        self.typing_tasks: dict[str, asyncio.Task[None]] = {}

    async def send(self, identifier: str) -> None:
        scope_id = await self.repository.action_scope(identifier)
        if scope_id is None:
            return
        task = asyncio.create_task(self._serialized_send(scope_id, identifier))
        self.tasks[scope_id].add(task)
        try:
            await task
        except asyncio.CancelledError:
            await asyncio.shield(
                self.repository.interrupt_send(identifier, self.clock.now())
            )
            parent = asyncio.current_task()
            if parent is not None and parent.cancelling():
                raise
        finally:
            self.tasks[scope_id].discard(task)

    async def cancel_scope(self, scope_id: str) -> None:
        tasks = list(self.tasks.get(scope_id, set()))
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _serialized_send(self, scope_id: str, identifier: str) -> None:
        async with self.locks[scope_id], self.semaphore:
            await self._send(identifier)

    async def _send(self, identifier: str) -> None:
        async def permit(tx: Transaction) -> OutboxMessage | None:
            from app.shared.storage import fetchone

            row = await fetchone(
                tx, "SELECT scope_id FROM outbox_messages WHERE id=?", (identifier,)
            )
            if row is None:
                return None
            conversation = await self.conversations.get(row["scope_id"], tx)
            message = await self.repository.gate(
                tx,
                identifier,
                self.clock.now(),
                self.commerce.can_reply(conversation),
                await self.repository.is_killed(tx),
            )
            if (
                message is not None
                and message.proposal.get("origin") == "platform_redirect"
            ):
                if (
                    conversation.invitation_state != "planned"
                    or conversation.control_state != "enabled"
                    or not await self.commerce.repository.transition_allowed(
                        tx, conversation.scope
                    )
                ):
                    await tx.execute(
                        "UPDATE outbox_messages SET status='cancelled',error_code='PLATFORM_GATE' WHERE id=?",
                        (identifier,),
                    )
                    return None
            return message

        message = await self.repository.database.write(permit)
        if message is None:
            return
        transport_started = asyncio.get_running_loop().time()
        try:
            async with asyncio.timeout(
                min(10, max(0.01, message.deadline - self.clock.now()))
            ):
                result = await self.sender.send_text(message.chat_id, message.text)
        except asyncio.CancelledError:
            await asyncio.shield(self._unknown(message))
            raise
        except Exception:
            result = SendResult(
                "delivery_unknown", error_code="UNCLASSIFIED_TRANSPORT_ERROR"
            )

        async def record(tx: Transaction) -> None:
            conversation = await self.conversations.get(message.scope.scope_id, tx)
            if conversation.scope != message.scope:
                return
            recorded = await self.repository.outcome(
                tx,
                message,
                result.status,
                self.clock.now(),
                result.external_message_id,
                result.error_code,
                result.retry_after,
            )
            if not recorded:
                return
            await tx.execute(
                "UPDATE outbox_messages SET proposal_json=json_set(proposal_json,'$.transport_seconds',?) WHERE id=?",
                (asyncio.get_running_loop().time() - transport_started, message.id),
            )
            if result.status == "delivered":
                if result.external_message_id is None:
                    raise ValueError("Confirmed delivery must include a message ID")
                await self.conversations.record_sent(
                    tx,
                    message.scope,
                    message.turn_id,
                    message.text,
                    result.external_message_id,
                    message.kind,
                    message.proposal,
                    self.clock.now(),
                )
                if message.kind == "reply":
                    await tx.execute(
                        "UPDATE conversations SET first_reply_at=COALESCE(first_reply_at,?) WHERE id=?",
                        (self.clock.now(), conversation.id),
                    )
                if message.kind == "reply" and not await self.repository.has_successor(
                    tx, message.id
                ):
                    current = await self.conversations.get(conversation.id, tx)
                    if self.after_reply is not None and self.commerce.can_reply(
                        current
                    ):
                        await self.after_reply(tx, current, message)

        await self.repository.database.write(record)
        logger.info("delivery action=%s status=%s", message.id, result.status)

    async def _unknown(self, message: OutboxMessage) -> None:
        async def record(tx: Transaction) -> None:
            await self.repository.outcome(
                tx,
                message,
                "delivery_unknown",
                self.clock.now(),
                None,
                "SEND_CANCELLED",
                0,
            )

        await self.repository.database.write(record)

    async def run_once(self) -> bool:
        await self._sync_typing()
        identifiers = await self.repository.ready_ids(self.clock.now())
        await asyncio.gather(*(self.send(identifier) for identifier in identifiers))
        return bool(identifiers)

    async def _sync_typing(self) -> None:
        waiting = (
            await self.repository.waiting_parts(self.clock.now())
            if isinstance(self.sender, TypingSender)
            else []
        )
        active = {identifier for identifier, _, _ in waiting}
        removed = [
            identifier for identifier in self.typing_tasks if identifier not in active
        ]
        for identifier in removed:
            self.typing_tasks[identifier].cancel()
        await asyncio.gather(
            *(self.typing_tasks.pop(identifier) for identifier in removed),
            return_exceptions=True,
        )
        for identifier, scope_id, chat_id in waiting:
            if identifier not in self.typing_tasks:
                task = asyncio.create_task(self._type_waiting(chat_id))
                self.typing_tasks[identifier] = task
                self.tasks[scope_id].add(task)
                task.add_done_callback(self.tasks[scope_id].discard)

    async def _type_waiting(self, chat_id: int) -> None:
        try:
            if isinstance(self.sender, TypingSender):
                async with self.sender.typing(chat_id):
                    await asyncio.Event().wait()
        except Exception:
            logger.warning("continuation_typing_failed")

    async def run(self) -> None:
        while True:
            try:
                if not await self.run_once():
                    await asyncio.sleep(0.1)
            except Exception:
                logger.exception("delivery_worker_iteration_failed")
                await asyncio.sleep(1)
