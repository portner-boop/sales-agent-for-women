import json
from typing import Any

import pytest

from app.application import Application
from app.modules.agent.initiative import followup_eligibility
from app.modules.agent.schemas import FollowupProposal, ReplyProposal
from tests.helpers import scalar, update
from tests.test_followups import due, enable
from tests.test_platform_transition import finish_delivery


@pytest.mark.parametrize("model_allowed", [None, False, True])
async def test_open_reply_schedules_question_evaluation_independently_of_model_flag(
    app: Application, model_allowed: bool | None
) -> None:
    enable(app)
    proposal: dict[str, Any] = {
        "reply": "I prefer getting to know each other here",
        "dialogue_act": "contact_boundary",
    }
    if model_allowed is not None:
        proposal["followup_allowed"] = model_allowed
    app.model.responses["generate"].append(proposal)
    await app.telegram.ingest(
        [update(1, "Can we talk on WhatsApp? I also enjoy baking")]
    )
    await app.drain()
    assert app.sender.sent == [(1, proposal["reply"])]
    assert (
        await scalar(app, "SELECT COUNT(*) FROM followups WHERE status='pending'") == 1
    )
    metrics = json.loads(await scalar(app, "SELECT metrics_json FROM agent_runs"))
    assert {
        "stage": "followup_eligibility",
        "model_allowed": bool(model_allowed),
        "allowed": True,
        "reason": "OPEN_DECLARATIVE_REPLY",
        "dialogue_mode": "continuation",
    } in metrics["validation_events"]
    app.model.responses["followup"].append(
        {
            "reply": "What do you most enjoy baking?",
            "dialogue_act": "question",
            "question_goal": "baking_preference",
            "should_send": True,
        }
    )
    await due(app)
    await app.drain()
    assert app.sender.sent[-1] == (1, "What do you most enjoy baking?")
    checked = next(
        data
        for task, data in app.model.calls
        if task == "validate" and data["reply"].get("should_send")
    )
    assert "followup_allowed" not in checked["reply"]
    assert all(
        "followup_allowed" not in message["metadata"]
        for message in checked["context"]["RECENT_MESSAGES"]
    )
    app.clock.advance(3600)
    await app.drain()
    assert len(app.sender.sent) == 2
    assert await scalar(app, "SELECT COUNT(*) FROM followups") == 1


@pytest.mark.parametrize(
    "draft,goal",
    [
        ("A quiet park is relaxing", None),
        ("Do you usually walk alone?", None),
    ],
)
async def test_followup_without_question_contract_repairs_before_delivery(
    app: Application, draft: str, goal: str | None
) -> None:
    enable(app)
    app.model.responses["generate"].append(
        {"reply": "I like quiet parks too", "dialogue_act": "opinion"}
    )
    await app.telegram.ingest([update(1, "I like quiet parks")])
    await app.drain()
    app.model.responses["followup"].append(
        {
            "reply": draft,
            "dialogue_act": "followup",
            "question_goal": goal,
            "should_send": True,
        }
    )
    app.model.responses["repair"].append(
        {
            "reply": "Do you usually walk alone?",
            "dialogue_act": "question",
            "question_goal": "walking_company",
            "should_send": True,
        }
    )
    await due(app)
    await app.drain()
    assert app.sender.sent == [
        (1, "I like quiet parks too"),
        (1, "Do you usually walk alone?"),
    ]
    repair = next(data for task, data in app.model.calls if task == "repair")
    assert "FOLLOWUP_QUESTION_REQUIRED" in repair["REPAIR_REASON_CODES"]
    assert [task for task, _ in app.model.calls].count("validate_repair") == 1


@pytest.mark.parametrize(
    "boundary",
    [
        "question_permission",
        "proactive_permission",
        "followup_permission",
        "contact_permission",
    ],
)
def test_boundary_overrides_open_reply_eligibility(boundary: str) -> None:
    proposal = ReplyProposal(
        reply="I enjoy music", dialogue_act="opinion", followup_allowed=True
    )
    context = {
        "APPLICATION_STATE": {
            "followup_enabled": True,
            "dialogue_mode": "continuation",
        },
        "DIALOGUE_PLAN": {"strategy": "answer_then_connected_hook"},
        "ACTIVE_BOUNDARIES": [{"predicate": boundary, "value": False}],
    }
    assert followup_eligibility(proposal, context) == (False, "USER_BOUNDARY")


def test_followup_never_schedules_another_followup() -> None:
    proposal = FollowupProposal(
        reply="Do you enjoy music?",
        dialogue_act="question",
        question_goal="music_interest",
        should_send=True,
    )
    context = {
        "APPLICATION_STATE": {
            "followup_enabled": True,
            "dialogue_mode": "continuation",
        },
        "DIALOGUE_PLAN": {"strategy": "ask_connected_question"},
    }
    assert followup_eligibility(proposal, context) == (False, "NO_CHAIN")


async def test_recovery_keeps_useful_second_part_and_question(app: Application) -> None:
    enable(app)
    app.model.responses["interpret"].append(
        {
            "intent": "chat",
            "dialogue_mode": "continuation",
            "language": "en",
            "continue_conversation": True,
        }
    )
    for task in ("generate", "repair"):
        app.model.responses[task].append(
            {
                "reply": "I baked a cake today",
                "continuation": "Do you bake often?",
                "dialogue_act": "answer_and_question",
                "question_goal": "baking_frequency",
            }
        )
    for task in ("validate", "validate_repair"):
        app.model.responses[task].append(
            {"decision": "reject", "reason_codes": ["CORE.GROUNDING"]}
        )
    app.model.responses["recover"].append(
        {
            "reply": "I like honey cake",
            "continuation": "Do you bake it often, or save it for special occasions?",
            "dialogue_act": "answer_and_question",
            "question_goal": "baking_frequency",
            "question_part": "continuation",
        }
    )
    await app.telegram.ingest([update(1, "I bake honey cake. Do you like sweets?")])
    await app.drain()
    app.clock.advance(4)
    await app.drain()
    context = next(data for task, data in app.model.calls if task == "recover")
    assert context["DIALOGUE_PLAN"]["questions_allowed"] is True
    assert context["DIALOGUE_PLAN"]["strategy"] == "develop_conversation_naturally"
    assert context["DIALOGUE_PLAN"]["requires_continuation"] is False
    assert app.sender.sent == [
        (1, "I like honey cake"),
        (1, "Do you bake it often, or save it for special occasions?"),
    ]
    assert await scalar(app, "SELECT COUNT(*) FROM followups") == 0


@pytest.mark.parametrize(
    "evidence", [None, "I'm going to sleep", "here its almost 9:30 morning"]
)
async def test_old_morning_question_does_not_silently_cancel_new_followup(
    app: Application, evidence: str | None
) -> None:
    from tests.test_platform_transition import finish_delivery

    enable(app)
    app.model.responses["generate"].append(
        {
            "reply": "How’s your morning going?",
            "dialogue_act": "question",
            "question_goal": "morning_status",
        }
    )
    await app.telegram.ingest([update(1, "hi Marko, good morning")])
    await app.drain()
    app.model.responses["generate"].append(
        {
            "reply": "It’s about two in Split",
            "continuation": "Hope your morning’s off to a good start",
            "dialogue_act": "answer",
        }
    )
    await app.telegram.ingest(
        [update(2, "what time is it there in Split now? here its almost 9:30 morning")]
    )
    await finish_delivery(app)
    app.model.responses["followup"].append(
        {
            "reply": "",
            "dialogue_act": "wait_for_old_pending_question",
            "should_send": False,
            "skip_evidence": evidence,
        }
    )
    if evidence == "here its almost 9:30 morning":
        app.model.responses["validate"].append(
            {"decision": "reject", "reason_codes": ["FOLLOWUP_SKIP_NOT_JUSTIFIED"]}
        )
    app.model.responses["repair"].append(
        {
            "reply": "Any plans for today?",
            "dialogue_act": "question",
            "question_goal": "plans_today",
            "should_send": True,
        }
    )
    await due(app)
    await finish_delivery(app)
    assert app.sender.sent[-1] == (1, "Any plans for today?")
    request = next(data for task, data in app.model.calls if task == "followup")
    assert request["DIALOGUE_PLAN"]["waiting_question_message_ids"] == []
    assert any(item["question_goal"] == "morning_status" for item in request["AGENDA"])
    assert (
        await scalar(
            app,
            "SELECT status FROM outbox_messages WHERE json_extract(proposal_json,'$.origin')='followup'",
        )
        == "delivered"
    )
    assert any(task == "validate_repair" for task, _ in app.model.calls)


async def test_real_unavailability_can_skip_after_independent_check(
    app: Application,
) -> None:
    enable(app)
    app.model.responses["generate"].append(
        {"reply": "Hope it goes smoothly", "dialogue_act": "reaction"}
    )
    await app.telegram.ingest([update(1, "I'm heading into a meeting now")])
    await app.drain()
    app.model.responses["followup"].append(
        {
            "reply": "",
            "dialogue_act": "respect_unavailability",
            "should_send": False,
            "skip_evidence": "I'm heading into a meeting now",
        }
    )
    await due(app)
    await app.drain()
    assert app.sender.sent == [(1, "Hope it goes smoothly")]
    checks = [data for task, data in app.model.calls if task == "validate"]
    assert checks[-1]["reply"]["should_send"] is False
    assert await scalar(app, "SELECT status FROM followups") == "skipped"
    assert not any(task == "repair" for task, _ in app.model.calls)


async def test_repair_cannot_silently_drop_a_rejected_followup(
    app: Application,
) -> None:
    enable(app)
    app.model.responses["generate"].append(
        {"reply": "I like clever mysteries", "dialogue_act": "opinion"}
    )
    await app.telegram.ingest([update(1, "I like comedy films")])
    await app.drain()
    app.model.responses["followup"].append(
        {
            "reply": "Hope you have a good evening",
            "should_send": True,
            "dialogue_act": "wish",
        }
    )
    app.model.responses["repair"].append(
        {"reply": "", "should_send": False, "dialogue_act": "no_novel_reaction"}
    )
    app.model.responses["recover"].append(
        {
            "reply": "Which comedy always makes you laugh?",
            "should_send": True,
            "dialogue_act": "question",
            "question_goal": "favorite_comedy_film",
        }
    )
    await due(app)
    await app.drain()
    assert app.sender.sent[-1] == (1, "Which comedy always makes you laugh?")
    assert any(task == "validate_recovery" for task, _ in app.model.calls)


@pytest.mark.parametrize(
    "language,user,answer,question",
    [
        (
            "en",
            "its a secret ;) ok then lets talk on whatsapp, its more comfortable for me. whats your number?",
            "I’d rather keep getting to know each other here, Maria. It feels more comfortable to me for now",
            "Do you enjoy baking more than cooking?",
        ),
        (
            "ru",
            "Давай перейдём в WhatsApp, какой у тебя номер?",
            "Мне пока комфортнее узнавать друг друга здесь",
            "Тебе больше нравится готовить или печь?",
        ),
        (
            "es",
            "Hablemos por WhatsApp, ¿cuál es tu número?",
            "Prefiero que nos sigamos conociendo aquí por ahora",
            "¿Te gusta más cocinar o hacer pasteles?",
        ),
        (
            "fr",
            "On peut parler sur WhatsApp ? Quel est ton numéro ?",
            "Je préfère continuer à faire connaissance ici pour le moment",
            "Tu préfères cuisiner ou faire des gâteaux ?",
        ),
        (
            "ar",
            "لنتحدث على واتساب، ما رقمك؟",
            "أفضل أن نواصل التعارف هنا في الوقت الحالي",
            "هل تفضلين الطبخ أم صنع الكعك؟",
        ),
    ],
)
async def test_service_contact_answer_schedules_and_delivers_followup(
    app: Application, language: str, user: str, answer: str, question: str
) -> None:
    enable(app)
    app.model.responses["interpret"].append(
        {"intent": "contact_request", "dialogue_mode": "service", "language": language}
    )
    app.model.responses["generate"].append(
        {"reply": answer, "dialogue_act": "contact_boundary", "followup_allowed": True}
    )
    await app.telegram.ingest([update(1, user)])
    await finish_delivery(app)
    assert ". ".join(text for _, text in app.sender.sent) == answer
    assert await scalar(app, "SELECT status FROM followups") == "pending"
    app.model.responses["followup"].append(
        {
            "reply": question,
            "dialogue_act": "question",
            "question_goal": "cooking_or_baking_preference",
            "should_send": True,
        }
    )
    await due(app)
    await app.drain()
    assert app.sender.sent[-1] == (1, question)
    metrics = json.loads(
        await scalar(
            app, "SELECT metrics_json FROM agent_runs ORDER BY started_at LIMIT 1"
        )
    )
    event = next(
        e for e in metrics["validation_events"] if e["stage"] == "followup_eligibility"
    )
    assert event["dialogue_mode"] == "service"
    assert event["allowed"] is True


async def test_service_label_allows_connected_second_question(app: Application) -> None:
    from tests.test_platform_transition import finish_delivery

    enable(app)
    app.model.responses["interpret"].append(
        {
            "intent": "contact_request",
            "dialogue_mode": "service",
            "language": "en",
            "continue_conversation": True,
        }
    )
    app.model.responses["generate"].append(
        {
            "reply": "I prefer getting to know each other here",
            "continuation": "Do you bake often?",
            "dialogue_act": "answer_and_question",
            "question_goal": "baking_frequency",
            "question_part": "continuation",
        }
    )
    await app.telegram.ingest([update(1, "I enjoy baking. Can we talk on WhatsApp?")])
    await finish_delivery(app)
    context = next(data for task, data in app.model.calls if task == "generate")
    assert context["DIALOGUE_PLAN"]["requires_continuation"] is False
    assert app.sender.sent == [
        (1, "I prefer getting to know each other here"),
        (1, "Do you bake often?"),
    ]
    assert await scalar(app, "SELECT COUNT(*) FROM followups") == 0


@pytest.mark.parametrize(
    "mode,action,transition,reason",
    [
        ("closing", "none", False, "CLOSED"),
        ("service", "show_offer", False, "BUSINESS_ACTION"),
        ("service", "none", True, "BUSINESS_ACTION"),
    ],
)
def test_service_change_keeps_actual_closure_and_business_gates(
    mode: str, action: str, transition: bool, reason: str
) -> None:
    proposal = ReplyProposal.model_validate(
        {
            "reply": "A response",
            "dialogue_act": "answer",
            "requested_action": action,
            "platform_transition": transition,
        }
    )
    context = {
        "APPLICATION_STATE": {"followup_enabled": True, "dialogue_mode": mode},
        "DIALOGUE_PLAN": {"strategy": "respond_in_context"},
    }
    assert followup_eligibility(proposal, context) == (False, reason)


@pytest.mark.parametrize("mark", ["?", "؟"])
async def test_rhetorical_reaction_does_not_count_as_pending_initiative(
    app: Application, mark: str
) -> None:
    enable(app)
    app.model.responses["generate"].append(
        {
            "reply": "How big is your team?",
            "dialogue_act": "question",
            "question_goal": "team_size",
        }
    )
    await app.telegram.ingest([update(1, "I work at a bank")])
    await app.drain()
    app.model.responses["generate"].append(
        {
            "reply": f"A thousand presidents{mark} That’s one very crowded branch",
            "dialogue_act": "playful_reaction",
            "followup_allowed": True,
        }
    )
    await app.telegram.ingest([update(2, "1000 presidents")])
    await app.drain()
    app.model.responses["followup"].append(
        {
            "reply": "Do you enjoy working with your team?",
            "dialogue_act": "question",
            "question_goal": "team_enjoyment",
            "should_send": True,
        }
    )
    await due(app)
    await app.drain()
    context = next(data for task, data in app.model.calls if task == "followup")
    assert context["DIALOGUE_PLAN"]["consecutive_question_turns"] == 0
    assert context["DIALOGUE_PLAN"]["questions_allowed"] is True
    assert context["DIALOGUE_PLAN"]["waiting_question_message_ids"] == []
    assert app.sender.sent[-1] == (1, "Do you enjoy working with your team?")
