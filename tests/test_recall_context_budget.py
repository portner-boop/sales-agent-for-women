import asyncio
from dataclasses import replace
from typing import Any

import pytest

from app.application import Application
from app.modules.agent.prompts import (
    FOLLOWUP_VALIDATOR_RULES,
    GENERATOR_RULES,
    RECOVERY_RULES,
    VALIDATOR_RULES,
)
from app.modules.agent.schemas import ReplyProposal, SemanticVerdict, TurnInterpretation
from app.modules.agent.service import ActiveRun
from app.modules.memory.entities import MemoryView
from app.shared.storage import Transaction
from tests.helpers import conversation, interpretation_operation, update


@pytest.mark.parametrize("stage", ["validate", "repair", "recover", "followup"])
async def test_recall_fits_later_stages_without_discarding_facts_or_latest_reply(
    app: Application, stage: str
) -> None:
    await app.telegram.ingest(
        [
            update(
                1,
                "I'm Anna, 42, a nurse from Paris. I have two children. "
                "Fish is my favourite too. No questions. What do you remember about me?",
            )
        ]
    )
    current = await conversation(app)
    original_message = (await app.conversations.messages(current.scope))[0]
    operations = [
        TurnInterpretation.model_validate(
            interpretation_operation(original_message, predicate, value, kind)
        ).memory_operations[0]
        for predicate, value, kind in [
            ("preferred_name", "Anna", "fact"),
            ("age_years", 42, "fact"),
            ("occupation", "nurse", "fact"),
            ("home_city", "Paris", "fact"),
            ("children_count", 2, "fact"),
            ("favorite_food", "fish", "preference"),
            ("question_permission", False, "boundary"),
        ]
    ]
    memory_ids = await app.memory.resolve(
        current.snapshot, (original_message,), [], operations, [], "en", "continuation"
    )
    current = await conversation(app)

    async def seed(tx: Transaction) -> None:
        for number in range(10):
            await app.conversations.record_sent(
                tx,
                current.scope,
                None,
                f"Thought {number}. " + "I like quiet music and clever comedies. " * 8,
                100 + number,
                "reply",
                {"dialogue_act": "opinion"},
                app.clock.now(),
            )

    await app.database.write(seed)
    turn = await app.conversations.claim_turn(current.scope, app.clock.now())
    assert turn is not None
    started = asyncio.get_running_loop().time()
    proposal = ReplyProposal(
        reply="You're Anna, 42, a nurse from Paris with two children. You like fish",
        dialogue_act="recall",
        used_memory_ids=memory_ids[:-1],
        used_source_message_ids=[turn.messages[0].id],
    )
    run = ActiveRun(
        turn,
        current,
        started + 45,
        started,
        interpretation=TurnInterpretation(
            intent="recall",
            dialogue_mode="continuation",
            language="en",
            recall_scope="general",
            retrieval_predicates=["favorite_food"],
            policy_topics=["recall"],
        ),
        proposal=proposal,
        recovering=stage == "recover",
    )
    await app.agent._build_context(run)
    assert run.context is not None
    original = run.context.data
    rules = {
        "validate": VALIDATOR_RULES,
        "repair": GENERATOR_RULES,
        "recover": GENERATOR_RULES + RECOVERY_RULES,
        "followup": FOLLOWUP_VALIDATOR_RULES,
    }[stage]
    validating = stage in {"validate", "followup"}
    schema = SemanticVerdict if validating else ReplyProposal
    payload = (
        {"context": original, "reply": proposal.model_dump()}
        if validating
        else {**original, "REJECTED_PROPOSAL": proposal.model_dump()}
    )

    def tokens(data: dict[str, Any]) -> int:
        return (
            app.agent.builder.counter.count(
                {"rules": rules, "data": data, "schema": schema.model_json_schema()}
            )
            + 500
        )

    baseline_rules = app.agent.builder.counter.count(GENERATOR_RULES)
    extra = max(0, baseline_rules - app.agent.builder.counter.count(rules))
    rules += "\n" + "validation " * extra
    limit = (
        tokens(payload)
        - app.agent.builder.counter.count(original["RECENT_MESSAGES"])
        - 900
    )
    app.agent.builder.settings = app.agent.builder.settings.model_copy(
        update={"context_tokens": limit}
    )
    fitted = await app.agent._fit_request(run, rules, payload, schema)
    context = fitted["context"] if validating else fitted
    assert tokens(fitted) <= limit
    assert context["CURRENT_INPUT"] == original["CURRENT_INPUT"]
    assert context["RECENT_DELIVERED"][-1] == original["RECENT_DELIVERED"][-1]
    assert context["HISTORY_STATUS"]["omitted_delivered_messages"] > 0
    assert context["ACTIVE_BOUNDARIES"] == original["ACTIVE_BOUNDARIES"]
    assert context["RECALL_REQUIRED"] == original["RECALL_REQUIRED"]
    assert len(context["RECALL_REQUIRED"]) == 6
    assert context["DIALOGUE_PLAN"]["questions_allowed"] is False
    assert set(memory_ids).issubset(run.context.manifest.memory_ids)
    assert proposal.used_source_message_ids[0] in {
        message["id"] for message in run.context.manifest.messages
    }
    assert len(await app.conversations.messages(current.scope)) == 11


@pytest.mark.parametrize("source_kind", ["exact", "other_version", "other_text"])
async def test_current_original_is_not_duplicated_as_a_memory_source(
    app: Application, source_kind: str
) -> None:
    await app.telegram.ingest([update(1, "Fish is my favourite too")])
    current = await conversation(app)
    turn = await app.conversations.claim_turn(current.scope, app.clock.now())
    assert turn is not None
    source = turn.messages[0]
    if source_kind == "other_version":
        source = replace(source, version=source.version + 1)
    elif source_kind == "other_text":
        source = replace(source, text="Fish")
    context = await app.agent.builder.build(
        turn,
        TurnInterpretation(
            intent="recall",
            dialogue_mode="continuation",
            language="en",
            referenced_message_ids=[source.id],
        ),
        current,
        MemoryView((), (), (), (source,)),
    )
    assert context.data["CURRENT_INPUT"][0]["text"] == turn.messages[0].text
    assert bool(context.data["MEMORY_DATA"]["sources"]) == (source_kind != "exact")
    assert {"id": turn.messages[0].id, "version": turn.messages[0].version} in (
        context.manifest.messages
    )


async def test_delivered_trimming_preserves_complete_latest_and_cited_turns(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "Hello")])
    current = await conversation(app)
    original = (await app.conversations.messages(current.scope))[0]
    messages = [
        replace(
            original,
            id=f"reply_{number}",
            turn_id=f"turn_{number // 2}",
            role="assistant",
            text="A delivered answer",
        )
        for number in range(8)
    ]
    data: dict[str, Any] = {
        "RECENT_DELIVERED": [message.as_data() for message in messages],
        "HISTORY_STATUS": {"omitted_delivered_messages": 0},
    }
    app.agent.builder._fit_delivered(
        data,
        messages,
        {"reply_2"},
        lambda: app.settings.memory.context_tokens + 1,
    )
    assert [message["id"] for message in data["RECENT_DELIVERED"]] == [
        "reply_2",
        "reply_3",
        "reply_6",
        "reply_7",
    ]
    assert data["HISTORY_STATUS"]["omitted_delivered_messages"] == 4
