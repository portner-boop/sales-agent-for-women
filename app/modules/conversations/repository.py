import json
from dataclasses import fields
from typing import Any
from uuid import uuid4

import aiosqlite

from app.modules.conversations.entities import Conversation, Message, Turn
from app.modules.conversations.schemas import AgendaOperation, IncomingMessage
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
        json.loads(row["metadata_json"]),
    )


MESSAGE_SELECT = "SELECT m.*,v.version,v.text,v.event_time,v.received_at,v.metadata_json FROM messages m JOIN message_versions v ON v.message_id=m.id AND v.version=m.current_version"


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
    ) -> Conversation:
        await tx.execute(
            "INSERT OR IGNORE INTO conversations(id,tenant_id,channel_account_id,external_user_id,chat_id,persona_id,last_activity_at,created_at) VALUES(?,'demo',?,?,?,'marko',?,?)",
            (uuid4().hex, channel, user_id, chat_id, now, now),
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

    async def sources(self, scope: ScopeContext, ids: list[str]) -> list[Message]:
        async with self.database.read() as tx:
            await require_scope(tx, scope)
            result = []
            for message_id in dict.fromkeys(ids):
                row = await fetchone(
                    tx,
                    MESSAGE_SELECT + " WHERE m.id=? AND m.scope_id=? AND m.epoch=?",
                    (message_id, scope.scope_id, scope.epoch),
                )
                if row:
                    result.append(message_from_row(row))
            return result

    async def source_versions(
        self, scope: ScopeContext, references: list[tuple[str, int]]
    ) -> list[Message]:
        async with self.database.read() as tx:
            await require_scope(tx, scope)
            result = []
            for identifier, version in dict.fromkeys(references):
                row = await fetchone(
                    tx,
                    "SELECT m.*,v.version,v.text,v.event_time,v.received_at,v.metadata_json FROM messages m JOIN message_versions v ON v.message_id=m.id WHERE m.scope_id=? AND m.epoch=? AND m.id=? AND v.version=?",
                    (scope.scope_id, scope.epoch, identifier, version),
                )
                if row:
                    result.append(message_from_row(row))
            return result

    async def claim_turn(self, scope: ScopeContext, now: float) -> Turn | None:
        async def claim(tx: Transaction) -> Turn | None:
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

        return await self.database.write(claim)

    async def expire_pending(self, scope: ScopeContext, cutoff: float) -> int:
        async def expire(tx: Transaction) -> int:
            await require_scope(tx, scope)
            await tx.execute(
                "UPDATE turns SET status='unanswered' WHERE scope_id=? AND epoch=? AND status IN ('running','deferred') AND id IN (SELECT m.turn_id FROM messages m JOIN message_versions v ON v.message_id=m.id AND v.version=m.current_version WHERE m.pending=1 AND v.received_at<=?)",
                (scope.scope_id, scope.epoch, cutoff),
            )
            cursor = await tx.execute(
                "UPDATE messages SET pending=0 WHERE scope_id=? AND epoch=? AND pending=1 AND id IN (SELECT message_id FROM message_versions WHERE received_at<=? AND version=messages.current_version)",
                (scope.scope_id, scope.epoch, cutoff),
            )

            return cursor.rowcount

        return await self.database.write(expire)

    async def finish_turn(self, tx: Transaction, turn: Turn, status: str) -> None:
        await require_scope(tx, turn.scope)
        await tx.execute("UPDATE turns SET status=? WHERE id=?", (status, turn.id))
        if status in {"prepared", "skipped", "unanswered"} and turn.messages:
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

    async def agenda(
        self, scope: ScopeContext, tx: Transaction | None = None
    ) -> list[dict[str, Any]]:
        if tx is None:
            async with self.database.read() as connection:
                return await self.agenda(scope, connection)
        await require_scope(tx, scope)
        rows = await fetchall(
            tx,
            "SELECT * FROM dialogue_threads WHERE scope_id=? AND epoch=? ORDER BY updated_at DESC",
            (scope.scope_id, scope.epoch),
        )
        return [dict(row) for row in rows]

    async def context_agenda(
        self, scope: ScopeContext, topics: list[str]
    ) -> list[dict[str, Any]]:
        items = await self.agenda(scope)
        active = [
            item
            for item in items
            if item["status"] not in {"answered", "declined", "closed"}
        ]
        matched = [
            item
            for item in items
            if item not in active
            and (item["topic_kind"] in topics or item["question_goal"] in topics)
        ]
        recent = [item for item in items if item not in active and item not in matched][
            :12
        ]
        return active + matched + recent

    async def question_state(
        self, scope: ScopeContext, goal: str, time_scope: str
    ) -> list[dict[str, Any]]:
        async with self.database.read() as tx:
            await require_scope(tx, scope)
            return [
                dict(row)
                for row in await fetchall(
                    tx,
                    "SELECT * FROM dialogue_threads WHERE scope_id=? AND epoch=? AND question_goal=? AND question_time_scope=?",
                    (scope.scope_id, scope.epoch, goal, time_scope),
                )
            ]

    async def update_agenda(
        self,
        tx: Transaction,
        scope: ScopeContext,
        operation: AgendaOperation,
        now: float,
    ) -> None:
        await require_scope(tx, scope)
        existing = await fetchone(
            tx,
            "SELECT id FROM dialogue_threads WHERE scope_id=? AND epoch=? AND subject_id=? AND topic_kind=? AND question_time_scope=?",
            (
                scope.scope_id,
                scope.epoch,
                operation.subject_id,
                operation.topic_kind,
                operation.question_time_scope,
            ),
        )
        identifier = existing["id"] if existing else uuid4().hex
        await tx.execute(
            "INSERT INTO dialogue_threads(id,scope_id,epoch,subject_id,topic_kind,status,question_goal,question_time_scope,answer_message_ids_json,pending_user_question,next_allowed_step,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status,question_goal=COALESCE(excluded.question_goal,dialogue_threads.question_goal),answer_message_ids_json=excluded.answer_message_ids_json,pending_user_question=excluded.pending_user_question,next_allowed_step=excluded.next_allowed_step,updated_at=excluded.updated_at",
            (
                identifier,
                scope.scope_id,
                scope.epoch,
                operation.subject_id,
                operation.topic_kind,
                operation.status,
                operation.question_goal,
                operation.question_time_scope,
                json.dumps(operation.answer_message_ids),
                operation.pending_user_question,
                operation.next_allowed_step,
                now,
            ),
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
                        "question_goal": proposal.get("question_goal"),
                        "question_subject_id": proposal.get(
                            "question_subject_id", "user_self"
                        ),
                        **(
                            {
                                "question_time_scope": proposal.get(
                                    "question_time_scope", "stable"
                                )
                            }
                            if proposal.get("question_goal")
                            else {}
                        ),
                        **(
                            {
                                "resume_question_thread_id": proposal[
                                    "resume_question_thread_id"
                                ]
                            }
                            if proposal.get("resume_question_thread_id")
                            else {}
                        ),
                        "casual_imperfection_used": proposal.get(
                            "casual_imperfection_used", False
                        ),
                        "casual_imperfection": proposal.get("casual_imperfection"),
                        "style_plan": proposal.get("style_plan", {}),
                        **(
                            {"origin": "followup"}
                            if proposal.get("origin") == "followup"
                            else {}
                        ),
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
                proposal.get("question_goal"),
                proposal.get("question_time_scope", "stable"),
                kind,
                now,
            ),
        )
        question_goal = proposal.get("question_goal")
        if kind == "reply" and question_goal:
            thread = await fetchone(
                tx,
                "SELECT topic_kind FROM dialogue_threads WHERE id=? AND scope_id=? AND epoch=? AND subject_id=?",
                (
                    proposal.get("question_thread_id"),
                    scope.scope_id,
                    scope.epoch,
                    proposal.get("question_subject_id", "user_self"),
                ),
            )
            operation = AgendaOperation(
                topic_kind=thread["topic_kind"] if thread else question_goal,
                subject_id=proposal.get("question_subject_id", "user_self"),
                status="awaiting_user",
                question_goal=question_goal,
                question_time_scope=proposal.get("question_time_scope", "stable"),
            )
            await self.update_agenda(tx, scope, operation, now)
            await tx.execute(
                "UPDATE dialogue_threads SET last_question_message_id=? WHERE scope_id=? AND epoch=? AND question_goal=? AND question_time_scope=? AND subject_id=?",
                (
                    message_id,
                    scope.scope_id,
                    scope.epoch,
                    question_goal,
                    operation.question_time_scope,
                    operation.subject_id,
                ),
            )
        commitment = proposal.get("new_commitment_candidate")
        if commitment:
            await tx.execute(
                "INSERT INTO commitments VALUES(?,?,?,?,?,'open')",
                (uuid4().hex, scope.scope_id, scope.epoch, message_id, commitment),
            )
        return message_id

    async def commitments(self, scope: ScopeContext) -> list[dict[str, Any]]:
        async with self.database.read() as tx:
            await require_scope(tx, scope)
            return [
                dict(row)
                for row in await fetchall(
                    tx,
                    "SELECT * FROM commitments WHERE scope_id=? AND epoch=? AND status='open'",
                    (scope.scope_id, scope.epoch),
                )
            ]

    async def completed_messages(self, scope: ScopeContext) -> list[Message]:
        async with self.database.read() as tx:
            await require_scope(tx, scope)
            rows = await fetchall(
                tx,
                MESSAGE_SELECT
                + " WHERE m.scope_id=? AND m.epoch=? AND m.pending=0 AND (m.turn_id IS NULL OR EXISTS(SELECT 1 FROM turns t WHERE t.id=m.turn_id AND t.status IN ('prepared','skipped'))) ORDER BY m.sequence",
                (scope.scope_id, scope.epoch),
            )
            return [message_from_row(row) for row in rows]

    async def purge(self, tx: Transaction, scope_id: str, new_epoch: int) -> None:
        await tx.execute("DELETE FROM dialogue_threads WHERE scope_id=?", (scope_id,))
        await tx.execute("DELETE FROM messages WHERE scope_id=?", (scope_id,))
        await tx.execute("DELETE FROM turns WHERE scope_id=?", (scope_id,))
        await tx.execute("DELETE FROM activity_sessions WHERE scope_id=?", (scope_id,))
        await tx.execute(
            "UPDATE conversations SET epoch=?,input_revision=input_revision+1,control_revision=control_revision+1,memory_revision=memory_revision+1,processed_sequence=0,dialogue_mode='continuation',preview_reply_count=0,preview_closed_at=NULL WHERE id=?",
            (new_epoch, scope_id),
        )
