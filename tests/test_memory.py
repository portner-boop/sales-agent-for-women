import asyncio

import pytest

from app.application import Application
from app.modules.conversations.schemas import AgendaOperation
from app.modules.jobs.entities import Job
from app.modules.memory.entities import CompactionBlock, SummaryOverflowError
from app.modules.memory.schemas import EpisodeSummary, MemoryOperation
from app.modules.memory.service import source_hash
from app.shared.entities import EvidenceError, StaleStateError
from tests.helpers import (
    ScriptedModel,
    conversation,
    interpretation_operation,
    scalar,
    update,
)


async def store(app: Application, number: int, text: str, **operation: object) -> str:
    await app.telegram.ingest([update(number, text)])
    current = await conversation(app)
    message = (await app.conversations.messages(current.scope))[-1]
    app.model.responses["interpret"].append(
        interpretation_operation(message, **operation)
    )
    await app.drain()
    return (await app.memory.repository.items((await conversation(app)).scope))[0]["id"]


async def test_correction_without_a_stored_target_keeps_the_explicit_new_fact(
    app: Application,
) -> None:
    await store(
        app,
        1,
        "Correction: I am fifty-five, not fifty-four.",
        predicate="age_years",
        value=55,
        kind="fact",
        operation="supersede",
    )
    current = await conversation(app)
    items = await app.memory.repository.items(current.scope)
    assert len(items) == 1
    assert items[0]["value"] == 55
    assert items[0]["supersedes_id"] is None
    assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"


async def test_missing_correction_target_does_not_override_other_applicability(
    app: Application,
) -> None:
    await store(
        app, 1, "I drink black coffee in the morning.", qualifiers={"time": "morning"}
    )
    app.model.responses["recover"].append(
        {"reply": "I see the difference.", "dialogue_act": "acknowledge"}
    )
    await store(
        app,
        2,
        "Correction: my coffee has milk.",
        value="with milk",
        operation="supersede",
    )
    items = await app.memory.repository.items((await conversation(app)).scope)
    assert len(items) == 1
    assert items[0]["value"] == "black with honey"
    assert items[0]["qualifiers"] == {"time": "morning"}
    assert (
        await scalar(
            app,
            "SELECT status FROM agent_runs ORDER BY rowid DESC LIMIT 1",
        )
        == "prepared"
    )


async def test_explicit_missing_target_is_never_normalized_to_add(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "My current age is fifty-five.")])
    current = await conversation(app)
    message = (await app.conversations.messages(current.scope))[-1]
    app.model.responses["interpret"].append(
        interpretation_operation(
            message,
            predicate="age_years",
            value=55,
            operation="supersede",
            target_memory_id="nonexistent",
        )
    )
    await app.drain()
    assert not await app.memory.repository.items(current.scope)
    assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"


async def test_compaction_splits_overflow_without_claiming_unpublished_coverage(
    app: Application, monkeypatch: pytest.MonkeyPatch
) -> None:
    for index in range(1, 3):
        await app.telegram.ingest([update(index, f"Detail {index}.")])
        await app.drain()
    current = await conversation(app)
    messages = await app.conversations.messages(current.scope)
    original = app.model.summarize
    attempted_sizes = []

    async def bounded_summary(
        block: CompactionBlock, deadline: float, output_tokens: int
    ) -> EpisodeSummary:
        attempted_sizes.append(len(block.messages))
        if len(block.messages) > 1:
            assert not await app.memory.repository.covered_ids(current.scope)
            raise SummaryOverflowError("Summary exceeded its output budget")
        return await original(block, deadline, output_tokens)

    monkeypatch.setattr(app.model, "summarize", bounded_summary)
    job = Job("diagnostic", "compact", current.scope, {}, 1, "lease")
    await app.compaction._publish(job, messages[:2])
    assert attempted_sizes == [2, 1, 1]
    assert await app.memory.repository.covered_ids(current.scope) == {
        message.id for message in messages[:2]
    }
    assert await scalar(app, "SELECT COUNT(*) FROM memory_summaries") == 2


async def test_single_original_overflow_never_gets_coverage(
    app: Application, monkeypatch: pytest.MonkeyPatch
) -> None:
    await app.telegram.ingest([update(1, "Detail.")])
    await app.drain()
    current = await conversation(app)
    messages = await app.conversations.messages(current.scope)

    async def overflow(
        block: CompactionBlock, deadline: float, output_tokens: int
    ) -> EpisodeSummary:
        raise SummaryOverflowError("Summary exceeded its output budget")

    monkeypatch.setattr(app.model, "summarize", overflow)
    job = Job("diagnostic", "compact", current.scope, {}, 1, "lease")
    with pytest.raises(SummaryOverflowError):
        await app.compaction._publish(job, messages[:1])
    assert not await app.memory.repository.covered_ids(current.scope)


async def test_exact_fact_and_correction_before_generator(app: Application) -> None:
    identifier = await store(app, 1, "I drink black coffee with honey.")
    await store(
        app,
        2,
        "Now I prefer coffee with milk.",
        value="with milk",
        operation="supersede",
        target_memory_id=identifier,
    )
    current = await conversation(app)
    facts = await app.memory.repository.items(current.scope)
    assert [item["value"] for item in facts] == ["with milk"]
    history = await app.memory.repository.items(current.scope, history=True)
    assert {item["value"] for item in history} == {"with milk", "black with honey"}
    generated = [data for task, data in app.model.calls if task == "generate"][-1]
    assert generated["MEMORY_DATA"]["facts"][0]["value"] == "with milk"


async def test_subjects_do_not_mix_client_and_pet(app: Application) -> None:
    await app.telegram.ingest([update(1, "My cat Luna likes fish.")])
    current = await conversation(app)
    message = (await app.conversations.messages(current.scope))[0]
    source = {
        "message_id": message.id,
        "message_version": 1,
        "evidence_text": "My cat Luna likes fish.",
    }
    response = interpretation_operation(
        message, "favorite_food", "fish", "preference", subject_id="cat_luna"
    )
    response["entities"] = [
        {"id": "cat_luna", "kind": "pet", "name": "Luna", "sources": [source]}
    ]
    app.model.responses["interpret"].append(response)
    await app.drain()
    facts = await app.memory.repository.items(current.scope)
    assert facts[0]["subject_id"] == "cat_luna"
    assert not [item for item in facts if item["subject_id"] == "user_self"]


@pytest.mark.parametrize("modality", ["question", "hypothetical", "quoted"])
async def test_nonassertion_is_not_a_persistent_fact(
    app: Application, modality: str
) -> None:
    await app.telegram.ingest([update(1, "If I moved to Spain, would it work?")])
    current = await conversation(app)
    message = (await app.conversations.messages(current.scope))[0]
    app.model.responses["interpret"].append(
        interpretation_operation(message, "residence", "Spain", modality=modality)
    )
    await app.drain()
    assert not await app.memory.repository.items(current.scope)
    assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"


async def test_sensitive_memory_rejected(app: Application) -> None:
    await app.telegram.ingest([update(1, "I take a prescription medication.")])
    current = await conversation(app)
    message = (await app.conversations.messages(current.scope))[0]
    app.model.responses["interpret"].append(
        interpretation_operation(message, "medication", "prescription")
    )
    await app.drain()
    assert not await app.memory.repository.items(current.scope)


async def test_source_must_be_exact_current_span(app: Application) -> None:
    await app.telegram.ingest([update(1, "My name is Ana.")])
    current = await conversation(app)
    messages = tuple(await app.conversations.messages(current.scope))
    operation = MemoryOperation.model_validate(
        {
            "operation": "add",
            "predicate": "preferred_name",
            "value": "Anna",
            "sources": [
                {
                    "message_id": messages[0].id,
                    "message_version": 1,
                    "evidence_text": "My name is Anna.",
                }
            ],
        }
    )
    with pytest.raises(EvidenceError):
        await app.memory.resolve(
            current.snapshot, messages, [], [operation], [], "en", "continuation"
        )
    assert not await app.memory.repository.items(current.scope)


async def test_cross_scope_target_cannot_be_superseded(app: Application) -> None:
    identifier = await store(app, 1, "I drink black coffee with honey.")
    await app.telegram.ingest([update(2, "I like milk.", user=2)])
    other = await conversation(app, 2)
    messages = tuple(await app.conversations.messages(other.scope))
    operation = MemoryOperation.model_validate(
        interpretation_operation(
            messages[0], operation="supersede", target_memory_id=identifier
        )["memory_operations"][0]
    )
    with pytest.raises(EvidenceError):
        await app.memory.resolve(
            other.snapshot, messages, [], [operation], [], "en", "continuation"
        )


async def test_boundary_is_mandatory_even_when_not_retrieved(app: Application) -> None:
    await store(
        app,
        1,
        "Please do not ask about my former partner.",
        predicate="avoid_ex_partner",
        value=True,
        kind="boundary",
    )
    await app.telegram.ingest([update(2, "Let's talk about nature.")])
    await app.drain()
    data = [data for task, data in app.model.calls if task == "generate"][-1]
    assert data["ACTIVE_BOUNDARIES"][0]["predicate"] == "avoid_ex_partner"
    assert app.model.calls[-1][0] == "validate"


async def test_forwarded_message_cannot_establish_client_fact(app: Application) -> None:
    raw = update(1, "I drink black coffee with honey.")
    raw["message"]["forward_origin"] = {
        "type": "hidden_user",
        "sender_user_name": "Other",
        "date": 1900000000,
    }
    await app.telegram.ingest([raw])
    current = await conversation(app)
    message = (await app.conversations.messages(current.scope))[0]
    app.model.responses["interpret"].append(interpretation_operation(message))
    await app.drain()
    assert not await app.memory.repository.items(current.scope)


async def test_summary_requires_complete_coverage_and_current_versions(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "hello")])
    await app.drain()
    current = await conversation(app)
    messages = tuple(await app.conversations.messages(current.scope))
    block = CompactionBlock(messages, source_hash(messages))
    summary = EpisodeSummary(
        topic_kinds=["chat"],
        subject_ids=["user_self"],
        claims=[],
        covered_message_ids=[messages[0].id],
    )
    with pytest.raises(EvidenceError):
        await app.memory.publish_summary(current.scope, block, summary)
    summary.covered_message_ids = [message.id for message in messages]
    await app.memory.publish_summary(current.scope, block, summary)
    await app.telegram.ingest([update(2, "corrected", message_id=1, edited=True)])
    assert not await app.memory.repository.covered_ids(current.scope)
    with pytest.raises(EvidenceError):
        await app.memory.publish_summary(current.scope, block, summary)


async def test_edit_disputes_fact_and_invalidates_summary(app: Application) -> None:
    await store(app, 1, "I drink black coffee with honey.")
    current = await conversation(app)
    messages = tuple(await app.conversations.messages(current.scope))
    block = CompactionBlock(messages, source_hash(messages))
    summary = await app.model.summarize(block, 10, 256)
    await app.memory.publish_summary(current.scope, block, summary)
    await app.telegram.ingest(
        [update(2, "I do not drink coffee.", message_id=1, edited=True)]
    )
    assert not await app.memory.repository.items(current.scope)
    assert not await app.memory.repository.covered_ids(current.scope)


async def test_forget_during_generation_fences_epoch_and_cleans_every_artifact(
    app: Application,
) -> None:
    await store(app, 1, "I drink black coffee with honey.")
    before = await conversation(app)
    app.model.entered.clear()
    app.model.release = asyncio.Event()
    await app.telegram.ingest([update(2, "second message")])
    task = asyncio.create_task(app.turn_worker.run_once())
    await app.model.entered.wait()
    await app.telegram.ingest([update(3, "/forget")])
    await task
    await app.drain()
    current = await conversation(app)
    assert current.epoch == before.epoch + 1
    for table in (
        "memory_items",
        "memory_summaries",
        "agent_runs",
        "turns",
        "commitments",
    ):
        assert await scalar(app, f"SELECT COUNT(*) FROM {table}") == 0
    assert (
        await scalar(app, "SELECT COUNT(*) FROM inbox_events WHERE raw_json!='{}'") == 0
    )
    assert await scalar(app, "SELECT COUNT(*) FROM messages WHERE role='user'") == 0
    assert await scalar(app, "SELECT COUNT(*) FROM said_ledger") == 1
    assert app.memory.deletion_path.exists()
    with pytest.raises(StaleStateError):
        await app.conversations.messages(before.scope)


async def test_forget_is_immediate_and_preserves_first_reply(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "hello", user=3)])
    await app.drain()
    before = await conversation(app, 3)
    await app.telegram.ingest([update(2, "/forget_confirm", user=3)])
    await app.drain()
    assert (await conversation(app, 3)).epoch == before.epoch
    await app.telegram.ingest([update(3, "/forget", user=3)])
    await app.drain()
    assert (await conversation(app, 3)).epoch == before.epoch + 1
    assert (await conversation(app, 3)).first_reply_at == before.first_reply_at


async def test_answered_agenda_is_kept(app: Application) -> None:
    await app.telegram.ingest([update(1, "I already bought the chair.")])
    current = await conversation(app)
    messages = tuple(await app.conversations.messages(current.scope))
    operation = AgendaOperation(
        topic_kind="chair_purchase",
        status="answered",
        question_goal="chair_purchase",
        answer_message_ids=[messages[0].id],
    )
    await app.memory.resolve(
        current.snapshot, messages, [], [], [operation], "en", "continuation"
    )
    assert (await app.conversations.agenda(current.scope))[0]["status"] == "answered"


async def test_semantic_patch_rejection_precedes_persistent_write(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "My name is Ana.")])
    current = await conversation(app)
    message = (await app.conversations.messages(current.scope))[0]
    app.model.responses["interpret"].append(
        interpretation_operation(message, "preferred_name", "Anna", "fact")
    )
    app.model.responses["validate_memory"].append(
        {"decision": "reject", "reason_codes": ["EVIDENCE_VALUE_MISMATCH"]}
    )
    await app.drain()
    assert not await app.memory.repository.items(current.scope)
    assert [task for task, _ in app.model.calls] == [
        "interpret",
        "validate_memory",
        "generate",
        "validate",
    ]
    generated = next(data for task, data in app.model.calls if task == "generate")
    assert generated["CURRENT_INPUT"][0]["text"] == "My name is Ana."
    assert not generated["MEMORY_DATA"]["facts"]
    assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"


async def test_model_memory_projection_keeps_evidence_without_storage_fields(
    app: Application,
) -> None:
    await store(
        app, 1, "My name is Ana", predicate="preferred_name", value="Ana", kind="fact"
    )
    current = await conversation(app)
    stored = (await app.memory.repository.items(current.scope))[0]
    data = next(data for task, data in app.model.calls if task == "generate")
    supplied = data["MEMORY_DATA"]["facts"][0]
    omitted = {"scope_id", "epoch", "recorded_at"}
    assert omitted.issubset(stored)
    assert omitted.isdisjoint(supplied)
    assert supplied == {
        key: value for key, value in stored.items() if key not in omitted
    }
    assert supplied["sources"][0]["evidence_text"] == "My name is Ana"
    assert stored["scope_id"] == current.id


async def test_old_epoch_summary_cannot_publish_after_deletion(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "A private story.")])
    await app.drain()
    old = await conversation(app)
    messages = tuple(await app.conversations.messages(old.scope))
    block = CompactionBlock(messages, source_hash(messages))
    summary = await app.model.summarize(block, 5, 256)
    await app.telegram.ingest([update(2, "/forget")])
    await app.drain()
    with pytest.raises(StaleStateError):
        await app.memory.publish_summary(old.scope, block, summary)
    assert await scalar(app, "SELECT COUNT(*) FROM memory_summaries") == 0


async def test_forget_cancels_active_compaction_call(app: Application) -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    class HangingSummarizer(ScriptedModel):
        async def summarize(
            self, block: CompactionBlock, deadline_seconds: float, output_tokens: int
        ) -> EpisodeSummary:
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            raise AssertionError

    for number in range(1, 17):
        await app.telegram.ingest(
            [update(number, f"A private story about a walk in the park, day {number}.")]
        )
        await app.turn_worker.run_once()
        await app.delivery.run_once()
    app.compaction.model = HangingSummarizer()
    task = asyncio.create_task(app.background_worker.run_once())
    await asyncio.wait_for(entered.wait(), 2)
    await app.telegram.ingest([update(17, "/forget")])
    await task
    await app.drain()
    assert cancelled.is_set()
    assert await scalar(app, "SELECT COUNT(*) FROM memory_summaries") == 0
    assert await scalar(app, "SELECT COUNT(*) FROM messages WHERE role='user'") == 0
