import json
from dataclasses import fields
from typing import Any
from uuid import uuid4

import aiosqlite

from app.modules.conversations.entities import Conversation, Message, Turn
from app.modules.conversations.schemas import IncomingMessage
from app.shared.entities import ScopeContext
from app.shared.storage import (
    Database,
    Transaction,
    fetchall,
    fetchone,
    require_scope,
)


def conversation_from_row(row: aiosqlite.Row) -> Conversation:
    return Conversation(
        **{
            field.name: row[field.name]
            for field in fields(Conversation)
            if field.name in row.keys()
        }
    )


def message_from_row(row: aiosqlite.Row) -> Message:
    metadata = json.loads(row["metadata_json"])
    if (
        "sent_control_revision" in row.keys()
        and row["sent_control_revision"] is not None
    ):
        metadata["delivery_control_revision"] = row["sent_control_revision"]
    return Message(
        row["id"],
        row["version"],
        row["sequence"],
        row["role"],
        row["text"],
        row["event_time"],
        row["received_at"],
        row["external_id"],
        row["reply_to_external_id"],
        row["turn_id"],
        metadata,
    )


MESSAGE_SELECT = (
    "SELECT m.*,v.version,v.text,v.event_time,v.received_at,v.metadata_json,"
    "o.control_revision AS sent_control_revision "
    "FROM messages m JOIN message_versions v ON v.message_id=m.id AND v.version=m.current_version "
    "LEFT JOIN outbox_messages o ON o.scope_id=m.scope_id AND o.epoch=m.epoch "
    "AND o.external_message_id=m.external_id AND o.status='delivered' "
    "AND m.role IN ('assistant','service')"
)


class ConversationRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def ensure(
        self,
        tx: Transaction,
        channel: str,
        user_id: int,
        chat_id: int,
        now: float,
        engine_version: str = "v2",
        persona_mode: str = "research_human",
        language: str = "en",
    ) -> Conversation:
        await tx.execute(
            "INSERT OR IGNORE INTO conversations(id,tenant_id,channel_account_id,external_user_id,chat_id,persona_id,last_activity_at,created_at,engine_version,persona_mode,language) VALUES(?,'demo',?,?,?,'marko',?,?,?,?,?)",
            (
                uuid4().hex,
                channel,
                user_id,
                chat_id,
                now,
                now,
                engine_version,
                persona_mode,
                language,
            ),
        )
        row = await fetchone(
            tx,
            "SELECT * FROM conversations WHERE tenant_id='demo' AND channel_account_id=? AND external_user_id=? AND persona_id='marko'",
            (channel, user_id),
        )
        assert row is not None
        return conversation_from_row(row)

    async def get(self, scope_id: str, tx: Transaction | None = None) -> Conversation:
        if tx is None:
            async with self.database.read() as connection:
                return await self.get(scope_id, connection)
        row = await fetchone(tx, "SELECT * FROM conversations WHERE id=?", (scope_id,))
        if row is None:
            raise LookupError("Conversation does not exist")
        return conversation_from_row(row)

    async def append(
        self,
        tx: Transaction,
        scope: ScopeContext,
        incoming: IncomingMessage,
        now: float,
        activity_gap: int,
    ) -> tuple[Message, bool]:
        await require_scope(tx, scope)
        existing = await fetchone(
            tx,
            MESSAGE_SELECT + " WHERE m.scope_id=? AND m.epoch=? AND m.external_id=?",
            (scope.scope_id, scope.epoch, incoming.external_id),
        )
        if existing and not incoming.is_edit:
            return message_from_row(existing), False
        if existing and existing["text"] == incoming.text:
            return message_from_row(existing), False
        if incoming.is_edit and existing is None:
            raise LookupError("Unknown edited message")
        if existing:
            message_id, sequence, version = (
                existing["id"],
                existing["sequence"],
                existing["version"] + 1,
            )
            await tx.execute(
                "UPDATE messages SET current_version=?,pending=1 WHERE id=?",
                (version, message_id),
            )
        else:
            message_id, version = uuid4().hex, 1
            sequence = await self.next_sequence(tx, scope)
            await tx.execute(
                "INSERT INTO messages(id,scope_id,epoch,sequence,external_id,role,current_version,reply_to_external_id,pending) VALUES(?,?,?,?,?,'user',?,?,1)",
                (
                    message_id,
                    scope.scope_id,
                    scope.epoch,
                    sequence,
                    incoming.external_id,
                    version,
                    incoming.reply_to_external_id,
                ),
            )
        metadata = {"forwarded": incoming.forwarded}
        await tx.execute(
            "INSERT INTO message_versions VALUES(?,?,?,?,?,?,?,?)",
            (
                message_id,
                scope.scope_id,
                scope.epoch,
                version,
                incoming.text,
                incoming.event_time,
                now,
                json.dumps(metadata),
            ),
        )
        await tx.execute(
            "UPDATE conversations SET input_revision=input_revision+1,last_activity_at=? WHERE id=?",
            (now, scope.scope_id),
        )
        if existing is None:
            await self._complete_pair(tx, scope, message_id)
        session = await fetchone(
            tx,
            "SELECT id,last_activity_at FROM activity_sessions WHERE scope_id=? AND epoch=? ORDER BY started_at DESC LIMIT 1",
            (scope.scope_id, scope.epoch),
        )
        if session is None or now - session["last_activity_at"] > activity_gap:
            await tx.execute(
                "INSERT INTO activity_sessions VALUES(?,?,?,?,?)",
                (uuid4().hex, scope.scope_id, scope.epoch, now, now),
            )
        else:
            await tx.execute(
                "UPDATE activity_sessions SET last_activity_at=? WHERE id=?",
                (now, session["id"]),
            )
        message = Message(
            message_id,
            version,
            sequence,
            "user",
            incoming.text,
            incoming.event_time,
            now,
            incoming.external_id,
            incoming.reply_to_external_id,
            None,
            metadata,
        )
        return message, True

    async def _complete_pair(
        self, tx: Transaction, scope: ScopeContext, message_id: str
    ) -> None:
        row = await fetchone(
            tx,
            "SELECT pending_outgoing_turn_id FROM conversations WHERE id=? AND epoch=?",
            (scope.scope_id, scope.epoch),
        )
        if row is None or row["pending_outgoing_turn_id"] is None:
            return
        cursor = await tx.execute(
            "UPDATE outgoing_dialogue_turns SET paired_message_id=? WHERE turn_id=? AND scope_id=? AND epoch=? AND outcome='delivered' AND paired_message_id IS NULL",
            (message_id, row["pending_outgoing_turn_id"], scope.scope_id, scope.epoch),
        )
        if cursor.rowcount:
            await tx.execute(
                "UPDATE conversations SET completed_pair_count=completed_pair_count+1,pending_outgoing_turn_id=NULL WHERE id=? AND epoch=?",
                (scope.scope_id, scope.epoch),
            )

    async def next_sequence(self, tx: Transaction, scope: ScopeContext) -> int:
        row = await fetchone(
            tx,
            "SELECT COALESCE(MAX(sequence),0)+1 AS value FROM messages WHERE scope_id=? AND epoch=?",
            (scope.scope_id, scope.epoch),
        )
        assert row is not None
        return int(row["value"])

    async def messages(
        self, scope: ScopeContext, after: int = 0, tx: Transaction | None = None
    ) -> list[Message]:
        if tx is None:
            async with self.database.read() as connection:
                return await self.messages(scope, after, connection)
        await require_scope(tx, scope)
        rows = await fetchall(
            tx,
            MESSAGE_SELECT
            + " WHERE m.scope_id=? AND m.epoch=? AND m.sequence>? ORDER BY m.sequence",
            (scope.scope_id, scope.epoch, after),
        )
        return [message_from_row(row) for row in rows]

    async def _claim_pending(
        self, tx: Transaction, scope: ScopeContext, now: float
    ) -> Turn | None:
        conversation = await self.get(scope.scope_id, tx)
        await require_scope(tx, scope)
        rows = await fetchall(
            tx,
            MESSAGE_SELECT
            + " WHERE m.scope_id=? AND m.epoch=? AND m.pending=1 ORDER BY m.sequence",
            (scope.scope_id, scope.epoch),
        )
        messages = [message_from_row(row) for row in rows]
        if not messages:
            return None
        await tx.execute(
            "UPDATE turns SET status='cancelled' WHERE scope_id=? AND epoch=? AND status='running'",
            (scope.scope_id, scope.epoch),
        )
        turn_id = uuid4().hex
        await tx.execute(
            "INSERT INTO turns VALUES(?,?,?,'running',?,?,?,?)",
            (
                turn_id,
                scope.scope_id,
                scope.epoch,
                conversation.input_revision,
                messages[0].sequence,
                messages[-1].sequence,
                now,
            ),
        )
        for message in messages:
            await tx.execute(
                "UPDATE messages SET turn_id=? WHERE id=?", (turn_id, message.id)
            )
        return Turn(turn_id, scope, conversation.input_revision, tuple(messages))

    async def claim_turn(self, scope: ScopeContext, now: float) -> Turn | None:
        async def claim(tx: Transaction) -> Turn | None:
            return await self._claim_pending(tx, scope, now)

        return await self.database.write(claim)

    async def fail_pending(
        self, tx: Transaction, scope: ScopeContext, now: float
    ) -> Turn | None:
        turn = await self._claim_pending(tx, scope, now)
        if turn is not None:
            await self.finish_turn(tx, turn, "failed")
        return turn

    async def finish_turn(self, tx: Transaction, turn: Turn, status: str) -> None:
        await require_scope(tx, turn.scope)
        await tx.execute("UPDATE turns SET status=? WHERE id=?", (status, turn.id))
        if status in {"prepared", "skipped", "failed"} and turn.messages:
            for message in turn.messages:
                await tx.execute(
                    "UPDATE messages SET pending=0 WHERE id=? AND current_version=?",
                    (message.id, message.version),
                )
            await tx.execute(
                "UPDATE conversations SET processed_sequence=MAX(processed_sequence,?) WHERE id=?",
                (turn.messages[-1].sequence, turn.scope.scope_id),
            )

    async def set_control(
        self, tx: Transaction, scope: ScopeContext, state: str
    ) -> None:
        await require_scope(tx, scope)
        await tx.execute(
            "UPDATE conversations SET control_state=?,control_revision=control_revision+1 WHERE id=?",
            (state, scope.scope_id),
        )

    async def record_sent(
        self,
        tx: Transaction,
        scope: ScopeContext,
        turn_id: str | None,
        text: str,
        external_id: int,
        kind: str,
        proposal: dict[str, Any],
        now: float,
    ) -> str:
        await require_scope(tx, scope)
        message_id = uuid4().hex
        sequence = await self.next_sequence(tx, scope)
        role = "assistant" if kind == "reply" else "service"
        if kind == "reply" and proposal.get("response_language"):
            await tx.execute(
                "UPDATE conversations SET language=? WHERE id=? AND epoch=?",
                (proposal["response_language"], scope.scope_id, scope.epoch),
            )
        await tx.execute(
            "INSERT INTO messages(id,scope_id,epoch,sequence,external_id,role,current_version,turn_id) VALUES(?,?,?,?,?,?,1,?)",
            (
                message_id,
                scope.scope_id,
                scope.epoch,
                sequence,
                external_id,
                role,
                turn_id,
            ),
        )
        await tx.execute(
            "INSERT INTO message_versions VALUES(?,?,?,1,?,?,?,?)",
            (
                message_id,
                scope.scope_id,
                scope.epoch,
                text,
                now,
                now,
                json.dumps(
                    {
                        "confirmed_delivery": True,
                        **{
                            key: proposal[key]
                            for key in (
                                "origin",
                                "response_language",
                                "research_run_id",
                            )
                            if key in proposal
                        },
                    }
                ),
            ),
        )
        await tx.execute(
            "INSERT INTO said_ledger VALUES(?,?,?,?,?,?,?,?)",
            (
                message_id,
                scope.scope_id,
                scope.epoch,
                proposal.get("dialogue_act", "service"),
                None,
                "stable",
                kind,
                now,
            ),
        )
        return message_id

    async def purge(self, tx: Transaction, scope_id: str, new_epoch: int) -> None:
        await tx.execute(
            "DELETE FROM outgoing_dialogue_turns WHERE scope_id=?", (scope_id,)
        )
        await tx.execute("DELETE FROM messages WHERE scope_id=?", (scope_id,))
        await tx.execute("DELETE FROM turns WHERE scope_id=?", (scope_id,))
        await tx.execute("DELETE FROM activity_sessions WHERE scope_id=?", (scope_id,))
        await tx.execute(
            "UPDATE conversations SET epoch=?,input_revision=input_revision+1,control_revision=control_revision+1,memory_revision=memory_revision+1,processed_sequence=0,first_reply_at=NULL,language='en',dialogue_mode='opening',preview_reply_count=0,preview_closed_at=NULL,completed_pair_count=0,pending_outgoing_turn_id=NULL,invitation_state='none' WHERE id=?",
            (new_epoch, scope_id),
        )
