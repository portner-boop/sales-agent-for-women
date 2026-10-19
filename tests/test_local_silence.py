import asyncio
import json

import pytest

from app.application import Application
from app.modules.agent.errors import ModelOutputError
from tests.helpers import conversation, interpretation_operation, scalar, update
from tests.test_followups import due, enable, seed
from tests.test_memory import store


@pytest.mark.parametrize("invalid", ["memory", "source", "unsupported_claim"])
async def test_valid_original_or_memory_can_support_reply_after_bad_reference_metadata(
    app: Application, invalid: str
) -> None:
    await store(
        app, 1, "My name is Ana", predicate="preferred_name", value="Ana", kind="fact"
    )
    current = await conversation(app)
    fact = (await app.memory.repository.items(current.scope))[0]
    app.model.responses["generate"].append(
        {
            "reply": (
                "Your name is Bea"
                if invalid == "unsupported_claim"
                else "Your name is Ana"
            ),
            "dialogue_act": "recall",
            "used_memory_ids": [
                fact["id"] if invalid == "source" else "invalid_memory_id"
            ],
            "used_source_message_ids": [
                (
                    "invalid_source_id"
                    if invalid == "source"
                    else fact["sources"][0]["message_id"]
                )
            ],
        }
    )
    if invalid == "unsupported_claim":
        app.model.responses["validate"].append(
            {"decision": "reject", "reason_codes": ["WRONG_NAME"]}
        )
        app.model.responses["repair"].append(
            {
                "reply": "Your name is Ana",
                "dialogue_act": "recall",
                "used_memory_ids": [fact["id"]],
            }
        )
    await app.telegram.ingest([update(2, "What is my name?")])
    await app.drain()
    assert app.sender.sent[-1] == (1, "Your name is Ana")
    checked = [data for task, data in app.model.calls if task == "validate"][-1][
        "reply"
    ]
    assert "invalid_memory_id" not in checked["used_memory_ids"]
    assert "invalid_source_id" not in checked["used_source_message_ids"]
    assert bool(checked["used_memory_ids"] or checked["used_source_message_ids"])
    assert sum(task == "repair" for task, _ in app.model.calls) == (
        1 if invalid == "unsupported_claim" else 0
    )


async def test_milo_metaphor_preserves_facts_and_normal_response(
    app: Application,
) -> None:
    enable(app)
    app.model.responses["generate"].append(
        {
            "reply": "What do you enjoy most about nursing?",
            "dialogue_act": "question",
            "question_goal": "nursing_enjoyment",
        }
    )
    await app.telegram.ingest([update(1, "I'm Anna, a nurse. My dog is Milo")])
    await app.drain()
    text = "Milo is six, by the way. A rescue dog, very lazy, absolutely convinced he owns the sofa."
    await app.telegram.ingest([update(2, text)])
    current = await conversation(app)
    message = (await app.conversations.messages(current.scope))[-1]
    interpretation = interpretation_operation(
        message, "age_years", 6, kind="fact", subject_id="pet_milo"
    )
    source = {"message_id": message.id, "message_version": 1, "evidence_text": text}
    interpretation["entities"] = [
        {"id": "pet_milo", "kind": "pet", "name": "Milo", "sources": [source]}
    ]
    interpretation["memory_operations"].append(
        {
            "operation": "add",
            "subject_id": "pet_milo",
            "kind": "fact",
            "modality": "figurative",
            "predicate": "sofa_owner",
            "value": "owns the sofa",
            "sources": [source],
        }
    )
    interpretation["continue_conversation"] = True
    app.model.responses["interpret"].append(interpretation)
    app.model.responses["generate"].append(
        {
            "reply": "Sounds like you're renting a corner of his sofa",
            "continuation": "I'd negotiate with treats",
            "dialogue_act": "playful_reaction",
            "followup_allowed": True,
        }
    )
    await app.drain()
    app.clock.advance(5)
    await app.drain()
    items = await app.memory.repository.items(current.scope)
    assert {(item["predicate"], item["kind"], item["modality"]) for item in items} == {
        ("age_years", "fact", "asserted"),
        ("sofa_owner", "conversation_glossary", "figurative"),
    }
    assert (
        next(item for item in items if item["predicate"] == "age_years")["value"] == 6
    )
    assert len(app.sender.sent) == 3
    assert not any(task == "recover" for task, _ in app.model.calls)
    app.model.responses["followup"].append(
        {
            "reply": "Does he let you share the cushions?",
            "dialogue_act": "question",
            "question_goal": "pet_cushion_sharing",
            "should_send": True,
        }
    )
    await due(app)
    await app.drain()
    context = next(data for task, data in app.model.calls if task == "followup")
    assert context["DIALOGUE_PLAN"]["waiting_question_message_ids"] == []
    assert context["AGENDA"][0]["status"] == "awaiting_user"
    assert len(app.sender.sent) == 4
    assert (
        await scalar(app, "SELECT status FROM followups ORDER BY rowid DESC LIMIT 1")
        == "prepared"
    )


@pytest.mark.parametrize(
    "failure",
    [
        "semantic",
        "source",
        "subject",
        "hypothetical",
        "sensitive",
        "invalid_json",
        "timeout",
    ],
)
async def test_invalid_memory_cannot_remove_continuation_or_block_next_input(
    app: Application, failure: str
) -> None:
    await app.telegram.ingest([update(1, "Milo likes the sofa")])
    current = await conversation(app)
    message = (await app.conversations.messages(current.scope))[0]
    interpretation = interpretation_operation(message, "pet_preference", "sofa")
    interpretation["continue_conversation"] = True
    operation = interpretation["memory_operations"][0]
    if failure == "semantic":
        app.model.responses["validate_memory"].append(
            {"decision": "reject", "reason_codes": ["TEST_REJECTION"]}
        )
    elif failure in {"invalid_json", "timeout"}:
        error = (
            ModelOutputError("INVALID_JSON")
            if failure == "invalid_json"
            else TimeoutError()
        )
        app.model.responses["validate_memory"].extend([error, error])
    elif failure == "source":
        operation["sources"][0]["evidence_text"] = "not in the original"
    elif failure == "subject":
        operation["subject_id"] = "unknown_pet"
    elif failure == "hypothetical":
        operation["modality"] = "hypothetical"
    else:
        operation["predicate"] = "medication"
    app.model.responses["interpret"].append(interpretation)
    app.model.responses["generate"].append(
        {
            "reply": "He has good taste",
            "continuation": "I'd pick the comfy spot too",
            "dialogue_act": "opinion",
        }
    )
    await app.drain()
    app.clock.advance(5)
    await app.drain()
    assert len(app.sender.sent) == 2
    assert not await app.memory.repository.items(current.scope)
    assert not any(task == "recover" for task, _ in app.model.calls)
    metrics = json.loads(await scalar(app, "SELECT metrics_json FROM agent_runs"))
    assert any(
        event["stage"] == "memory_patch_discarded"
        for event in metrics["validation_events"]
    )
    generated = next(data for task, data in app.model.calls if task == "generate")
    assert generated["CURRENT_INPUT"][0]["text"] == message.text
    await app.telegram.ingest([update(2, "He also loves biscuits")])
    await app.drain()
    assert len(app.sender.sent) == 3
    assert not app.agent.active


@pytest.mark.parametrize("question_part", ["reply", "continuation"])
async def test_latest_question_blocks_followup_even_when_model_schedules_it(
    app: Application, question_part: str
) -> None:
    enable(app)
    parts = {
        "reply": "Does Milo like long walks?",
        "continuation": "I like a quiet path",
    }
    if question_part == "continuation":
        parts["reply"], parts["continuation"] = parts["continuation"], parts["reply"]
    app.model.responses["generate"].append(
        {
            **parts,
            "dialogue_act": "question",
            "question_goal": "pet_walks",
            "question_part": question_part,
            "followup_allowed": True,
        }
    )
    await app.telegram.ingest([update(1, "Milo is my dog")])
    await app.drain()
    app.clock.advance(5)
    await app.drain()
    assert await scalar(app, "SELECT COUNT(*) FROM followups") == 0
    app.clock.advance(90)
    await app.drain()
    assert len(app.sender.sent) == 2
    assert not any(task == "followup" for task, _ in app.model.calls)
    metrics = json.loads(
        await scalar(
            app, "SELECT metrics_json FROM agent_runs ORDER BY rowid DESC LIMIT 1"
        )
    )
    assert {
        "stage": "followup_eligibility",
        "model_allowed": True,
        "allowed": False,
        "reason": "QUESTION_PENDING",
        "dialogue_mode": "continuation",
    } in metrics["validation_events"]


@pytest.mark.parametrize(
    "language,user_text,reply,followup",
    [
        (
            "en",
            "Milo charges sofa rent",
            "Sofa rent? I hope he accepts biscuits",
            "Does he share the sofa with you?",
        ),
        (
            "ar",
            "ميلو سيطالبني بإيجار الأريكة",
            "إيجار الأريكة؟ أكيد بيطلب الدفع على شكل مكافآت كلاب",
            "هل يشاركك الأريكة؟",
        ),
    ],
)
async def test_rhetorical_punctuation_is_not_a_pending_information_request(
    app: Application, language: str, user_text: str, reply: str, followup: str
) -> None:
    enable(app)
    app.model.responses["interpret"].append(
        {"intent": "playful", "dialogue_mode": "playful", "language": language}
    )
    app.model.responses["generate"].append(
        {"reply": reply, "dialogue_act": "playful_reaction", "followup_allowed": True}
    )
    await app.telegram.ingest([update(1, user_text)])
    await app.drain()
    app.model.responses["followup"].append(
        {
            "reply": followup,
            "dialogue_act": "question",
            "question_goal": "pet_sofa_sharing",
            "should_send": True,
        }
    )
    await due(app)
    await app.drain()
    assert app.sender.sent == [(1, reply), (1, followup)]
    data = next(data for task, data in app.model.calls if task == "followup")
    assert data["DIALOGUE_PLAN"]["waiting_question_message_ids"] == []


async def test_followup_lock_wait_expires_and_new_input_still_works(
    app: Application,
) -> None:
    enable(app)
    await seed(app)
    current = await conversation(app)
    app.agent.settings = app.agent.settings.model_copy(
        update={"turn_deadline_seconds": 0.03}
    )
    await due(app)
    async with app.agent.locks[current.id]:
        async with asyncio.timeout(1):
            await app.turn_worker.run_once()
    assert await scalar(app, "SELECT status FROM followups") == "pending"
    assert (
        await scalar(app, "SELECT reason FROM followups")
        == "FOLLOWUP_DEADLINE_EXCEEDED"
    )
    assert not app.agent.active
    assert not any(task == "followup" for task, _ in app.model.calls)
    app.agent.settings = app.agent.settings.model_copy(
        update={"turn_deadline_seconds": 45}
    )
    await app.telegram.ingest([update(2, "Let's talk about Milo")])
    await app.drain()
    assert len(app.sender.sent) == 2


async def test_typography_cannot_silence_a_checked_two_part_reply(
    app: Application,
) -> None:
    app.model.responses["generate"].append(
        {
            "reply": "Майло — хорошее имя для собаки",
            "continuation": "8–12 минут — короткая прогулка",
            "dialogue_act": "reaction_and_opinion",
        }
    )
    await app.telegram.ingest(
        [update(1, "Мою собаку зовут Майло. Гулять с ним 8–12 минут - мало?")]
    )
    await app.drain()
    app.clock.advance(5)
    await app.drain()
    assert app.sender.sent == [
        (1, "Майло - хорошее имя для собаки"),
        (1, "8-12 минут - короткая прогулка"),
    ]
    assert [task for task, _ in app.model.calls] == [
        "interpret",
        "generate",
        "validate",
    ]
    checked = next(data for task, data in app.model.calls if task == "validate")
    assert checked["reply"]["reply"] == app.sender.sent[0][1]
    assert checked["reply"]["continuation"] == app.sender.sent[1][1]


async def test_typography_normalization_does_not_bypass_content_rejection(
    app: Application,
) -> None:
    app.model.responses["generate"].append(
        {"reply": "Я был в Париже — вчера", "dialogue_act": "answer"}
    )
    app.model.responses["validate"].append(
        {"decision": "reject", "reason_codes": ["INVENTED_PERSONA_CIRCUMSTANCE"]}
    )
    app.model.responses["repair"].append(
        {"reply": "Мне нравятся тихие места", "dialogue_act": "opinion"}
    )
    await app.telegram.ingest([update(1, "Какие места тебе нравятся?")])
    await app.drain()
    assert app.sender.sent == [(1, "Мне нравятся тихие места")]
    assert [task for task, _ in app.model.calls][-3:] == [
        "validate",
        "repair",
        "validate_repair",
    ]


@pytest.mark.parametrize(
    "outcome",
    [
        "accept",
        "reject",
        "decline",
        "budget",
        "repeated_sentiment",
        "identical",
    ],
)
async def test_followup_gets_one_checked_bounded_recovery(
    app: Application, outcome: str
) -> None:
    enable(app)
    await seed(app, text="Мне нравятся тихие парки")
    for task in ("followup", "repair"):
        app.model.responses[task].append(
            {
                "reply": "Тебе нравятся тихие парки?",
                "dialogue_act": "question",
                "question_goal": "park_preference",
                "should_send": True,
            }
        )
    for task in ("validate", "validate_repair"):
        app.model.responses[task].append(
            {"decision": "reject", "reason_codes": ["CONVERSATION.REPETITION"]}
        )
    app.model.responses["recover"].append(
        {
            "reply": (
                ""
                if outcome == "decline"
                else (
                    "Тебе нравятся тихие парки?"
                    if outcome == "identical"
                    else "Ты обычно гуляешь одна?"
                )
            ),
            "dialogue_act": "question",
            "question_goal": None if outcome == "decline" else "park_company",
            "should_send": outcome != "decline",
        }
    )
    if outcome == "reject":
        app.model.responses["validate_recovery"].append(
            {"decision": "reject", "reason_codes": ["TEST_REJECTION"]}
        )
    if outcome in {"repeated_sentiment", "identical"}:
        app.model.responses["validate_recovery"].append(
            {
                "decision": "reject",
                "reason_codes": ["CONVERSATION.REPETITION:REPEATED_SENTIMENT"],
            }
        )
    if outcome == "budget":
        app.agent.settings = app.agent.settings.model_copy(update={"llm_max_calls": 4})
    await due(app)
    await app.drain()
    assert len(app.sender.sent) == (2 if outcome == "accept" else 1)
    assert await scalar(app, "SELECT status FROM followups") == (
        "prepared" if outcome == "accept" else "pending"
    )
    calls = [task for task, _ in app.model.calls]
    assert calls.count("recover") == (0 if outcome == "budget" else 1)
    assert calls.count("validate_recovery") == (
        0 if outcome in {"decline", "budget"} else 1
    )
    assert not app.agent.active
    app.clock.advance(3600)
    await app.drain()
    assert len(app.sender.sent) == (2 if outcome == "accept" else 1)


@pytest.mark.parametrize("questions_allowed", [True, False])
async def test_followup_recovery_keeps_existing_question_permissions(
    app: Application, questions_allowed: bool
) -> None:
    enable(app)
    text = "I like quiet parks" if questions_allowed else "No questions. I like parks"
    await app.telegram.ingest([update(1, text)])
    current = await conversation(app)
    message = (await app.conversations.messages(current.scope))[-1]
    interpretation = (
        {"intent": "chat", "language": "en", "dialogue_mode": "continuation"}
        if questions_allowed
        else interpretation_operation(message, "question_permission", False, "boundary")
    )
    interpretation["language"] = "en"
    app.model.responses["interpret"].append(interpretation)
    app.model.responses["generate"].append(
        {
            "reply": "I prefer parks without crowds",
            "dialogue_act": "opinion",
            "followup_allowed": True,
        }
    )
    await app.drain()
    if not questions_allowed:
        assert await scalar(app, "SELECT COUNT(*) FROM followups") == 0
        app.clock.advance(90)
        await app.drain()
        assert not any(task in {"followup", "recover"} for task, _ in app.model.calls)
        return

    for task in ("followup", "repair"):
        app.model.responses[task].append(
            {
                "reply": "Parks are better without crowds",
                "dialogue_act": "opinion",
                "should_send": True,
            }
        )
    for task in ("validate", "validate_repair"):
        app.model.responses[task].append(
            {"decision": "reject", "reason_codes": ["CONVERSATION.REPETITION"]}
        )
    app.model.responses["recover"].append(
        {
            "reply": "Do you usually walk there alone?",
            "dialogue_act": "question",
            "question_goal": "park_walking_company",
            "should_send": True,
        }
    )
    await due(app)
    await app.drain()
    context = next(data for task, data in app.model.calls if task == "recover")
    assert context["DIALOGUE_PLAN"]["questions_allowed"] is questions_allowed
    assert context["DIALOGUE_PLAN"]["strategy"] == (
        "ask_connected_question"
        if questions_allowed
        else "share_opinion_without_question"
    )
    assert len(app.sender.sent) == (2 if questions_allowed else 1)
    assert await scalar(app, "SELECT status FROM followups") == (
        "prepared" if questions_allowed else "skipped"
    )
    if questions_allowed:
        assert app.sender.sent[-1][1] == "Do you usually walk there alone?"
    assert [task for task, _ in app.model.calls].count("recover") == 1
    app.clock.advance(3600)
    await app.drain()
    assert len(app.sender.sent) == (2 if questions_allowed else 1)


@pytest.mark.parametrize(
    "outcome", ["accepted", "rejected", "primary", "unlocalized", "other"]
)
async def test_recovery_checks_answer_without_repeated_continuation(
    app: Application, outcome: str
) -> None:
    enable(app)
    answer = "The sky looks blue because air scatters blue sunlight more than red"
    repeated = "That turkey-on-his-head bit is hard to beat"
    for task in ("generate", "repair", "recover"):
        app.model.responses[task].append(
            {"reply": answer, "continuation": repeated, "dialogue_act": "answer"}
        )
    for task in ("validate", "validate_repair", "validate_recovery"):
        app.model.responses[task].append(
            {
                "decision": "reject",
                "reason_codes": [
                    (
                        "UNSUPPORTED_CLAIM"
                        if outcome == "other"
                        else "CONVERSATION.REPETITION"
                    )
                ],
                "violating_text": (
                    answer
                    if outcome == "primary"
                    else None if outcome == "unlocalized" else repeated
                ),
            }
        )
    if outcome in {"accepted", "rejected"}:
        app.model.responses["validate_recovery"].append(
            {
                "decision": "accept" if outcome == "accepted" else "reject",
                "reason_codes": [] if outcome == "accepted" else ["INCOMPLETE_ANSWER"],
            }
        )
    await app.telegram.ingest(
        [
            update(
                1,
                "if you get a weekend? haha you work so much? btw random question - why is the sky blue?",
            )
        ]
    )
    await app.drain()
    checks = [data for task, data in app.model.calls if task == "validate_recovery"]
    assert len(checks) == (2 if outcome in {"accepted", "rejected"} else 1)
    if len(checks) == 2:
        assert checks[-1]["reply"]["reply"] == answer
        assert checks[-1]["reply"]["continuation"] is None
    assert app.sender.sent == ([(1, answer)] if outcome == "accepted" else [])
    assert await scalar(
        app, "SELECT COUNT(*) FROM followups WHERE status='pending'"
    ) == (1 if outcome == "accepted" else 0)
    app.model.responses["generate"].append(
        {"reply": "Sunsets have warmer colors", "dialogue_act": "answer"}
    )
    await app.telegram.ingest([update(2, "And sunsets?")])
    await app.drain()
    assert app.sender.sent[-1] == (1, "Sunsets have warmer colors")
