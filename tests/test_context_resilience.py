import json
from typing import Any

import pytest

from app.application import Application
from app.modules.agent.entities import SchemaT
from app.modules.agent.schemas import TurnInterpretation
from app.modules.conversations.entities import Turn
from app.modules.conversations.schemas import AgendaOperation
from app.shared.entities import ContextOverflowError
from app.shared.storage import Transaction
from tests.helpers import conversation, interpretation_operation, scalar, update


async def test_long_chat_continues_when_compaction_is_unavailable(
    app: Application,
) -> None:
    first_message = None
    for number in range(1, 91):
        text = (
            "Do not call me darling."
            if number == 1
            else f"Day {number}. "
            + "I walked in the park and noticed different flowers along the path. " * 15
        )
        incoming = update(number, text)
        if number == 90:
            incoming["message"]["reply_to_message"] = {"message_id": 1}
        await app.telegram.ingest([incoming])
        if number == 1:
            current = await conversation(app)
            first_message = (await app.conversations.messages(current.scope))[0]
            app.model.responses["interpret"].append(
                interpretation_operation(
                    first_message, "forbidden_address", "darling", "boundary"
                )
            )
        await app.turn_worker.run_once()
        await app.delivery.run_once()
    assert first_message is not None
    current = await conversation(app)
    assert len(app.sender.sent) == 90
    assert await scalar(app, "SELECT COUNT(*) FROM memory_summaries") == 0
    assert not await app.memory.repository.covered_ids(current.scope)
    assert await scalar(app, "SELECT COUNT(*) FROM messages") == 180
    assert (
        await scalar(app, "SELECT COUNT(*) FROM agent_runs WHERE status!='prepared'")
        == 0
    )
    data = [data for task, data in app.model.calls if task == "generate"][-1]
    assert data["HISTORY_STATUS"]["omitted_uncompressed_messages"] > 0
    assert data["HISTORY_STATUS"]["originals_preserved"] is True
    assert data["ACTIVE_BOUNDARIES"][0]["value"] == "darling"
    assert any(
        source["id"] == first_message.id for source in data["MEMORY_DATA"]["sources"]
    )
    manifest = json.loads(
        await scalar(
            app, "SELECT manifest_json FROM agent_runs ORDER BY rowid DESC LIMIT 1"
        )
    )
    assert (
        sum(manifest["token_count_by_section"].values())
        <= app.settings.memory.context_tokens
    )
    supplied_ids = {
        message["id"]
        for message in data["CURRENT_INPUT"]
        + data["RECENT_MESSAGES"]
        + data["MEMORY_DATA"]["sources"]
    }
    assert {message["id"] for message in manifest["messages"]} == supplied_ids
    messages = await app.conversations.messages(current.scope)
    last_input = next(
        message for message in reversed(messages) if message.role == "user"
    )
    interpretation = TurnInterpretation(
        intent="chat",
        dialogue_mode="continuation",
        language="en",
        referenced_message_ids=[first_message.id],
    )
    view = await app.memory.retrieve(
        current.scope, [], [], [], False, [first_message.id]
    )
    assert last_input.turn_id is not None
    turn = Turn(
        last_input.turn_id, current.scope, current.input_revision, (last_input,)
    )
    reserved = await app.agent.builder.build(
        turn, interpretation, current, view, request_reserve_tokens=2000
    )
    assert reserved.manifest.token_count_by_section["request_reserve_tokens"] == 2000
    assert sum(reserved.manifest.token_count_by_section.values()) <= 12000
    assert (
        reserved.data["HISTORY_STATUS"]["omitted_uncompressed_messages"]
        > data["HISTORY_STATUS"]["omitted_uncompressed_messages"]
    )
    assert reserved.data["ACTIVE_BOUNDARIES"][0]["value"] == "darling"
    assert any(
        source["id"] == first_message.id
        for source in reserved.data["MEMORY_DATA"]["sources"]
    )
    assert not await app.memory.repository.covered_ids(current.scope)
    with pytest.raises(ContextOverflowError):
        await app.agent.builder.build(
            turn, interpretation, current, view, request_reserve_tokens=12000
        )
    with pytest.raises(ValueError):
        await app.agent.builder.build(
            turn, interpretation, current, view, request_reserve_tokens=-1
        )


async def test_large_open_agenda_does_not_block_a_new_turn(app: Application) -> None:
    await app.telegram.ingest([update(1, "Let us talk about the park.")])
    current = await conversation(app)

    async def seed(tx: Transaction) -> None:
        for index in range(250):
            await app.conversations.update_agenda(
                tx,
                current.scope,
                AgendaOperation(
                    topic_kind=f"old_topic_{index}",
                    status="awaiting_user",
                    question_goal=f"old_question_{index}",
                ),
                app.clock.now() + index,
            )

    await app.database.write(seed)
    await app.turn_worker.run_once()
    await app.delivery.run_once()
    assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"
    assert len(app.sender.sent) == 1
    interpreted = next(data for task, data in app.model.calls if task == "interpret")
    generated = next(data for task, data in app.model.calls if task == "generate")
    assert interpreted["HISTORY_STATUS"]["omitted_agenda_items"] > 0
    assert generated["HISTORY_STATUS"]["omitted_agenda_items"] > 0
    assert interpreted["AGENDA"][0]["question_goal"] == "old_question_249"
    assert generated["AGENDA"][0]["question_goal"] == "old_question_249"
    assert len(await app.conversations.agenda(current.scope)) == 250

    async def answer_old_question(tx: Transaction) -> None:
        await tx.execute(
            "UPDATE dialogue_threads SET status='answered' WHERE question_goal='old_question_0'"
        )

    await app.database.write(answer_old_question)
    app.model.responses["generate"].append(
        {
            "reply": "What about the old topic?",
            "dialogue_act": "ask",
            "question_goal": "old_question_0",
        }
    )
    app.model.responses["repair"].append(
        {"reply": "Flowers make an interesting topic.", "dialogue_act": "comment"}
    )
    await app.telegram.ingest([update(2, "Let us talk about flowers.")])
    await app.turn_worker.run_once()
    await app.delivery.run_once()
    repair = next(data for task, data in app.model.calls if task == "repair")
    assert "ALREADY_ANSWERED" in repair["REPAIR_REASON_CODES"]
    assert app.sender.sent[-1][1] == "Flowers make an interesting topic"
    assert len(await app.conversations.agenda(current.scope)) == 250


async def test_large_recent_turns_are_bounded_for_the_interpreter(
    app: Application,
) -> None:
    for number in range(1, 9):
        await app.telegram.ingest(
            [update(number, f"Day {number}. " + "flower " * 1800)]
        )
        await app.turn_worker.run_once()
        await app.delivery.run_once()
    assert len(app.sender.sent) == 8
    assert (
        await scalar(app, "SELECT COUNT(*) FROM agent_runs WHERE status!='prepared'")
        == 0
    )
    interpreted = [data for task, data in app.model.calls if task == "interpret"][-1]
    assert interpreted["HISTORY_STATUS"]["omitted_uncompressed_messages"] > 0
    assert app.agent.builder.interpreter_tokens(interpreted) <= 12000
    assert any(
        message["text"].startswith("Day 7.")
        for message in interpreted["RECENT_MESSAGES"]
    )


async def test_two_parts_keep_delivery_time_after_slow_generation(
    app: Application, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_reply_parts import prepare_pair

    original = app.model.complete

    async def slow_complete(
        task: str,
        rules: str,
        data: dict[str, Any],
        schema: type[SchemaT],
        deadline_seconds: float,
    ) -> SchemaT:
        result = await original(task, rules, data, schema, deadline_seconds)
        if task == "generate":
            for run in app.agent.active.values():
                run.deadline = run.started + 0.5
        return result

    monkeypatch.setattr(app.model, "complete", slow_complete)
    first, second = await prepare_pair(app)
    await app.delivery.send(first)
    app.clock.advance(5)
    await app.delivery.send(second)
    assert [text for _, text in app.sender.sent] == [
        "Кухня победила",
        "Что теперь на ужин?",
    ]
    assert (
        await scalar(
            app, "SELECT COUNT(*) FROM outbox_messages WHERE status='delivered'"
        )
        == 2
    )
