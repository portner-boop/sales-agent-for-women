from dataclasses import replace
from pathlib import Path

import pytest

from app.application import Application
from app.modules.agent.schemas import ChatImperfection, ReplyProposal
from app.modules.agent.style_plan import (
    StylePolicy,
    build_style_plan,
    relax_internal_punctuation,
    supported_imperfection,
)
from app.modules.conversations.entities import Message
from tests.helpers import conversation, update

pytestmark = pytest.mark.usefixtures("full_letter_typo_rate", "fixed_style_interval")


@pytest.fixture
def casual_text_edits() -> bool:
    return True


def delivered(index: int, eligible: bool, used: bool = False) -> Message:
    return Message(
        id=str(index),
        version=1,
        sequence=index,
        role="assistant",
        text="A fresh thought",
        event_time=float(index),
        received_at=float(index),
        external_id=index,
        reply_to_external_id=None,
        turn_id=f"turn-{index}",
        metadata={
            "confirmed_delivery": True,
            "style_plan": {"eligible": eligible},
            "casual_imperfection_used": used,
        },
    )


def test_message_interval_varies_and_counts_both_parts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.modules.agent.style_plan.style_policy",
        lambda: StylePolicy(imperfection_interval=5, interval_jitter=1),
    )
    history: list[Message] = []
    requested = []
    for index in range(1, 501):
        plan = build_style_plan(history, True)
        assert build_style_plan(history, True) == plan
        used = plan["request_imperfection"]
        if used:
            requested.append(index)
        first = delivered(index, True, used)
        if index % 2 == 0:
            first = replace(first, turn_id=f"turn-{index - 1}")
        history.append(first)
    intervals = [
        right - left for left, right in zip(requested, requested[1:], strict=False)
    ]
    assert set(intervals) == {4, 5, 6}
    assert 0.18 <= len(requested) / len(history) <= 0.22
    assert build_style_plan(history, True)["delivered_messages"] == 500
    assert build_style_plan(history, True)["delivered_eligible_turns"] == 250


def test_default_policy_requests_frequent_visible_imperfections() -> None:
    policy = StylePolicy.model_validate_json(
        (Path("app/policies/style.json")).read_text(encoding="utf-8")
    )
    assert policy.imperfection_interval == 1
    assert policy.interval_jitter == 0
    assert not build_style_plan([], True, max_parts=2)["request_imperfection"]
    history = [delivered(1, True)]
    plan = build_style_plan(history, True, max_parts=1)
    assert plan["request_imperfection"]
    assert plan["target_message_interval"] == 1


@pytest.mark.parametrize(
    "mode,topics,eligible",
    [
        ("continuation", ["contact"], True),
        ("repair", ["verification"], True),
        ("supportive", ["support"], True),
        ("closing", ["social"], False),
        ("continuation", ["photo"], False),
        ("continuation", ["commerce"], False),
    ],
)
async def test_style_eligibility_covers_ordinary_sensitive_conversation(
    app: Application, mode: str, topics: list[str], eligible: bool
) -> None:
    app.model.responses["interpret"].append(
        {
            "intent": "chat",
            "dialogue_mode": mode,
            "language": "en",
            "policy_topics": topics,
        }
    )
    app.model.responses["generate"].append(
        {"reply": "I prefer a calm chat", "dialogue_act": "answer"}
    )
    await app.telegram.ingest([update(1, "Tell me what you think")])
    await app.drain()
    context = next(data for task, data in app.model.calls if task == "generate")
    assert context["STYLE_PLAN"]["eligible"] is eligible


def test_ineligible_messages_advance_schedule_without_creating_style_debt() -> None:
    history = [delivered(index, False) for index in range(1, 21)]
    assert not build_style_plan(history, False)["request_imperfection"]
    assert build_style_plan(history, True)["request_imperfection"]
    history.append(delivered(21, True, True))
    assert build_style_plan(history, True)["request_imperfection"]
    history.append(delivered(22, True))
    assert build_style_plan(history, True)["request_imperfection"]


def test_two_part_reply_can_reach_interval_before_a_single_followup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.modules.agent.style_plan.style_policy",
        lambda: StylePolicy(imperfection_interval=3, interval_jitter=0),
    )
    history = [delivered(1, True)]
    assert not build_style_plan(history, True, max_parts=1)["request_imperfection"]
    assert build_style_plan(history, True, max_parts=2)["request_imperfection"]
    assert not build_style_plan(history, False, max_parts=2)["request_imperfection"]
    history.extend(
        [delivered(2, True), replace(delivered(3, True, True), turn_id="turn-2")]
    )
    plan = build_style_plan(history, True, max_parts=2)
    assert plan["messages_since_imperfection"] == 0
    assert not plan["request_imperfection"]


def test_unknown_or_cancelled_delivery_does_not_advance_style_schedule() -> None:
    history = [delivered(index, True) for index in range(1, 7)]
    draft = delivered(5, True, True)
    history.append(
        replace(draft, metadata={**draft.metadata, "confirmed_delivery": False})
    )
    plan = build_style_plan(history, True)
    assert plan["request_imperfection"]
    assert plan["delivered_imperfect_turns"] == 0
    assert plan["delivered_messages"] == 6


async def test_style_schedule_uses_delivered_metadata_and_survives_missing_edit(
    app: Application,
) -> None:
    for index in range(1, 7):
        await app.telegram.ingest([update(index, "I like a quiet cafe")])
        await app.drain()
    contexts = [data for task, data in app.model.calls if task == "generate"]
    assert contexts[-1]["STYLE_PLAN"]["request_imperfection"]
    assert not contexts[0]["STYLE_PLAN"]["request_imperfection"]
    assert contexts[-1]["STYLE_PLAN"]["visible_forms"] == {}
    app.model.responses["generate"].append(
        {
            "reply": "I m more of a small cafe person",
            "continuation": "Good coffee beats a fancy menu",
            "dialogue_act": "opinion",
            "casual_imperfection_used": True,
            "casual_imperfection": {"written": "I m", "standard": "I'm"},
        }
    )
    await app.telegram.ingest([update(7, "A small cafe sounds better")])
    await app.drain()
    app.clock.advance(5)
    await app.drain()
    assert app.sender.sent[-2:] == [
        (1, "Im more of a small cafe person"),
        (1, "Good cofee beats a fancy menu"),
    ]
    current = await conversation(app)
    history = await app.conversations.messages(current.scope)
    plan = build_style_plan(history, True)
    assert plan["delivered_eligible_turns"] == 7
    assert plan["delivered_messages"] == 8
    assert plan["messages_since_imperfection"] == 0
    assert plan["delivered_imperfect_turns"] == 6
    assert plan["allow_imperfection"]


@pytest.mark.parametrize(
    "evidence",
    [
        None,
        {"written": "missing", "standard": "absent"},
        {"written": "quiet", "standard": "quiet"},
    ],
)
async def test_false_imperfection_label_does_not_suppress_good_answer(
    app: Application, evidence: dict[str, str] | None
) -> None:
    for index in range(1, 5):
        await app.telegram.ingest([update(index, "I like quiet places")])
        await app.drain()
    app.model.responses["generate"].append(
        {
            "reply": "A small cafe wins for me",
            "dialogue_act": "opinion",
            "casual_imperfection_used": True,
            "casual_imperfection": evidence,
        }
    )
    app.model.responses["repair"].append(
        {"reply": "Small cafes have more charm", "dialogue_act": "opinion"}
    )
    await app.telegram.ingest([update(5, "I agree about small cafes")])
    await app.drain()
    assert not any(task == "repair" for task, _ in app.model.calls)
    assert app.sender.sent[-1] == (1, "A small cafe wins for me")
    current = await conversation(app)
    history = await app.conversations.messages(current.scope)
    assert not history[-1].metadata["casual_imperfection_used"]
    assert build_style_plan(history, True)["request_imperfection"]


@pytest.mark.parametrize(
    "written,standard,expected",
    [
        ("I m", "I'm", True),
        ("Мне кажется это уютно", "Мне кажется, это уютно", True),
        ("Si quieres vamos", "Si quieres, vamos", True),
        ("Franchement je préfère", "Franchement, je préfère", True),
        ("بصراحة أحب ذلك", "بصراحة، أحب ذلك", True),
        ("тоже", "также", False),
        ("1000", "1,000", False),
        ("1 000", "1, 000", False),
        ("big views", "expansive views", False),
        ("I have 3 children", "I have 2 children", False),
        ("I like it", "I do not like it", False),
        ("Maro", "Marko", False),
    ],
)
def test_imperfection_is_a_small_punctuation_change_not_a_rewritten_fact(
    written: str, standard: str, expected: bool
) -> None:
    assert (
        supported_imperfection(
            ChatImperfection(written=written, standard=standard), written
        )
        is expected
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Yes, I prefer small cafes", "Yes I prefer small cafes"),
        ("Мне кажется, это уютно", "Мне кажется это уютно"),
        ("Si quieres, vamos", "Si quieres vamos"),
        (
            "Franchement, je préfère les petits cafés",
            "Franchement je préfère les petits cafés",
        ),
        ("بصراحة، أحب المقاهي الصغيرة", "بصراحة أحب المقاهي الصغيرة"),
        ("1,000 is a number", "1,000 is a number"),
        ("Anna has 2 children", "Anna has 2 children"),
        ("I'd choose eggs", "Id choose eggs"),
        ("I’d choose eggs", "Id choose eggs"),
        ("I'm listening", "Im listening"),
        ("C’est un beau choix", "Cest un beau choix"),
        ("O’Connor likes eggs", "O’Connor likes eggs"),
        ("I don't want tea", "I dont want tea"),
        ("Je n'aime pas le thé", "Je n'aime pas le thé"),
    ],
)
def test_punctuation_relaxation_preserves_letters_numbers_and_other_punctuation(
    text: str, expected: str
) -> None:
    proposal = relax_internal_punctuation(
        ReplyProposal(reply=text, dialogue_act="opinion")
    )
    assert proposal.reply == expected
    assert proposal.casual_imperfection_used == (text != expected)
    if text != expected:
        assert supported_imperfection(proposal.casual_imperfection, proposal.full_reply)
        assert relax_internal_punctuation(proposal) is proposal


def test_imperfection_evidence_keeps_complete_words() -> None:
    proposal = relax_internal_punctuation(
        ReplyProposal(
            reply="La montagne, c’est un beau choix pour s’évader, surtout avec les grands paysages",
            dialogue_act="opinion",
        )
    )
    assert proposal.casual_imperfection == ChatImperfection(
        written="montagne c’est", standard="montagne, c’est"
    )


async def test_scheduled_punctuation_edit_is_validated_before_delivery(
    app: Application,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.modules.agent.style_plan.random.choice", lambda items: items[0]
    )
    for index in range(1, 5):
        await app.telegram.ingest([update(index, "I like quiet places")])
        await app.drain()
    app.model.responses["generate"].append(
        {"reply": "Yes, I prefer quiet cafes", "dialogue_act": "opinion"}
    )
    app.model.responses["validate"].append(
        {"decision": "reject", "reason_codes": ["INJECTED_MEANING_CHANGE"]}
    )
    app.model.responses["repair"].append(
        {"reply": "For me, a small cafe wins", "dialogue_act": "opinion"}
    )
    await app.telegram.ingest([update(5, "Small cafes sound nice")])
    await app.drain()
    checks = [
        data
        for task, data in app.model.calls
        if task in {"validate", "validate_repair"}
    ]
    assert checks[-2]["reply"]["reply"] == "Yes, I prefer quiet cafes"
    assert checks[-1]["reply"]["reply"] == "For me, a small cafe wins"
    assert app.sender.sent[-1] == (1, "For me,a small cafe wins")
    assert (1, "Yes, I prefer qiuet cafes") not in app.sender.sent


async def test_style_diagnostics_stay_in_storage_outside_model_history(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "I like quiet cafes")])
    await app.drain()
    current = await conversation(app)
    history = await app.conversations.messages(current.scope)
    stored = next(message for message in history if message.role == "assistant")
    assert stored.metadata["style_plan"]["eligible"]
    await app.telegram.ingest([update(2, "Especially small ones")])
    await app.drain()
    for task, data in app.model.calls:
        if task not in {"interpret", "generate", "validate"}:
            continue
        context = data.get("context", data)
        messages = [
            message
            for section in (
                "CURRENT_INPUT",
                "RECENT_MESSAGES",
                "RECENT_DELIVERED",
                "REFERENCE_MESSAGES",
            )
            for message in context.get(section, [])
        ] + context.get("MEMORY_DATA", {}).get("sources", [])
        for message in messages:
            assert not {
                "style_plan",
                "casual_imperfection",
                "casual_imperfection_used",
            }.intersection(message["metadata"])
    latest = [data for task, data in app.model.calls if task == "generate"][-1]
    assert latest["STYLE_PLAN"]["delivered_eligible_turns"] == 1
    assert any(
        message["metadata"].get("confirmed_delivery")
        for message in latest["RECENT_DELIVERED"]
    )


async def test_scheduled_contraction_is_changed_before_validation_and_delivery(
    app: Application,
) -> None:
    for index in range(1, 5):
        await app.telegram.ingest([update(index, "I like quiet places")])
        await app.drain()
    app.model.responses["generate"].append(
        {"reply": "I’d choose eggs for breakfast", "dialogue_act": "opinion"}
    )
    await app.telegram.ingest([update(5, "Eggs or porridge for breakfast?")])
    await app.drain()
    checked = [data for task, data in app.model.calls if task == "validate"][-1]
    assert checked["reply"]["reply"] == "I’d choose eggs for breakfast"
    assert checked["reply"]["casual_imperfection"] is None
    assert checked["reply"]["casual_imperfection_used"] is False
    assert app.sender.sent[-1] == (1, "Id choose eggs for breakfast")
    current = await conversation(app)
    history = await app.conversations.messages(current.scope)
    plan = build_style_plan(history, True)
    assert plan["delivered_imperfect_turns"] == 4
    assert plan["request_imperfection"]
