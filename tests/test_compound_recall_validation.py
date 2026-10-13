from typing import Any

import pytest

from app.application import Application
from app.modules.agent.entities import SchemaT
from app.modules.agent.language import reply_validation_instruction
from app.modules.agent.prompts import platform_turn_instruction
from tests.helpers import conversation, scalar, update

COMPOUND_INPUT = (
    "I'm Anna, 42, a nurse. My cat is Pixel, he's five\n"
    "What do you remember about me?\n"
    "What is my cat's name?\n"
    "Actually, Pixel is six. I got his age wrong\n"
    "How old is Pixel now?\n"
    "Please list everything I told you about myself"
)


async def test_compound_recall_keeps_requested_facts_above_optional_fact_budget(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, COMPOUND_INPUT)])
    original_id = await scalar(app, "SELECT id FROM messages WHERE external_id=1")

    def source(text: str) -> dict[str, Any]:
        return {
            "message_id": original_id,
            "message_version": 1,
            "evidence_text": text,
        }

    facts = [
        ("user_self", "preferred_name", "Anna", "I'm Anna"),
        ("user_self", "age_years", 42, "42"),
        ("user_self", "occupation", "nurse", "a nurse"),
        ("pet_pixel", "name", "Pixel", "My cat is Pixel"),
        ("pet_pixel", "species", "cat", "My cat"),
        ("pet_pixel", "age_years", 6, "Pixel is six"),
    ]
    app.model.responses["interpret"].append(
        {
            "intent": "recall_and_update",
            "dialogue_mode": "continuation",
            "language": "en",
            "entities": [
                {
                    "id": "pet_pixel",
                    "kind": "pet",
                    "name": "Pixel",
                    "sources": [source("My cat is Pixel")],
                }
            ],
            "memory_operations": [
                {
                    "operation": "add",
                    "subject_id": subject,
                    "predicate": predicate,
                    "value": value,
                    "sources": [source(evidence)],
                }
                for subject, predicate, value, evidence in facts
            ],
            "retrieval_predicates": list(dict.fromkeys(row[1] for row in facts)),
            "retrieval_subject_ids": ["user_self", "pet_pixel"],
        }
    )
    reply = "You’re Anna, 42, a nurse. Your cat is Pixel, and he is six now"
    app.model.responses["generate"].append(
        {
            "reply": reply,
            "dialogue_act": "memory_recall_and_answer",
            "used_source_message_ids": [original_id],
        }
    )
    await app.drain()
    assert app.sender.sent == [(1, reply)]
    data = next(data for task, data in app.model.calls if task == "generate")
    selected = data["MEMORY_DATA"]["facts"]
    assert len(selected) == 6
    assert app.agent.builder.counter.count(selected) > 800
    assert app.agent.builder.counter.count(data) < app.settings.memory.context_tokens
    assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"


@pytest.mark.parametrize("stage", ["validate", "validate_repair", "validate_recovery"])
async def test_compound_recall_uses_current_language_through_every_validation_stage(
    app: Application, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    await app.telegram.ingest([update(1, "Привет")])
    await app.drain()
    await app.telegram.ingest([update(2, COMPOUND_INPUT)])
    original_id = await scalar(
        app,
        "SELECT id FROM messages WHERE external_id=2 AND role='user'",
    )
    reply = "You’re Anna, 42, and a nurse. Your cat is Pixel"
    continuation = "Pixel is six now"
    complete = app.model.complete
    checked: list[str] = []

    async def capture(
        task: str,
        rules: str,
        data: dict[str, Any],
        schema: type[SchemaT],
        deadline_seconds: float,
    ) -> SchemaT:
        if task in {"validate", "validate_repair", "validate_recovery"}:
            assert data["context"]["APPLICATION_STATE"]["language"] == "en"
            assert rules.endswith(
                reply_validation_instruction("en")
                + platform_turn_instruction(data["context"]["PLATFORM_TRANSITION"])
            )
            assert data["reply"]["used_source_message_ids"] == [original_id]
            checked.append(task)
        return await complete(task, rules, data, schema, deadline_seconds)

    monkeypatch.setattr(app.model, "complete", capture)
    proposal = {
        "reply": reply,
        "continuation": continuation,
        "dialogue_act": "memory_recall_and_answer",
        "used_source_message_ids": [original_id],
    }
    app.model.responses["generate"].append(proposal)
    if stage != "validate":
        app.model.responses["validate"].append(
            {"decision": "reject", "reason_codes": ["UNSUPPORTED_FACT"]}
        )
        app.model.responses["repair"].append(proposal)
    if stage == "validate_recovery":
        app.model.responses["validate_repair"].append(
            {"decision": "reject", "reason_codes": ["IGNORED_USER_QUESTION"]}
        )
        app.model.responses["recover"].append(
            {**proposal, "reply": reply + ". " + continuation, "continuation": None}
        )
    await app.drain()
    if stage != "validate_recovery":
        current = await conversation(app)
        due = await app.delivery_repository.next_ready_at(current.scope)
        assert due is not None
        app.clock.advance(max(0, due - app.clock.now()))
        await app.drain()
    assert stage in checked
    assert await scalar(app, "SELECT language FROM conversations") == "en"
    assert (
        await scalar(app, "SELECT status FROM agent_runs ORDER BY rowid DESC LIMIT 1")
        == "prepared"
    )
    assert not any("proper reply" in text for _, text in app.sender.sent)
    assert app.sender.sent[-1][1] == (
        reply + ". " + continuation if stage == "validate_recovery" else continuation
    )
