import tomllib
from pathlib import Path

import pytest

from app.application import Application
from app.evaluation import receive
from app.modules.agent.schemas import ChatImperfection, ReplyProposal
from app.modules.agent.style_plan import (
    apply_casual_imperfection,
    semantic_reply,
    supported_imperfection,
    visible_edit_count,
)
from app.shared.config import RuntimeSettings
from tests.helpers import scalar, update
from tests.test_followups import due, enable


@pytest.mark.parametrize("text", ["How about you?", "Do you have any pets?"])
async def test_open_personal_answer_can_stand_alone(
    app: Application, text: str
) -> None:
    app.model.responses["interpret"].append(
        {
            "intent": "personal_question",
            "dialogue_mode": "continuation",
            "language": "en",
            "continue_conversation": True,
        }
    )
    app.model.responses["generate"].append(
        {"reply": "I prefer a quiet place", "dialogue_act": "opinion"}
    )
    await receive(app, app.sender, 1, text, 1, app.clock.now())
    assert app.sender.sent == [(1, "I prefer a quiet place")]
    assert [task for task, _ in app.model.calls] == [
        "interpret",
        "generate",
        "validate",
    ]


@pytest.mark.parametrize("followup", [False, True])
async def test_question_without_mark_is_delivered_and_tracks_pending_answer(
    app: Application, followup: bool
) -> None:
    enable(app)
    if followup:
        app.model.responses["generate"].append(
            {"reply": "I like quiet cafes", "dialogue_act": "opinion"}
        )
        await receive(app, app.sender, 1, "I enjoy cafes", 1, app.clock.now())
    question = {
        "reply": "Do you prefer a table by the window",
        "dialogue_act": "question",
        "question_goal": "cafe_table_preference",
        "question_part": "reply",
    }
    if followup:
        app.model.responses["followup"].append({**question, "should_send": True})
        await due(app)
        await app.drain()
    else:
        app.model.responses["generate"].append(question)
        await receive(app, app.sender, 1, "I enjoy cafes", 1, app.clock.now())
    assert app.sender.sent[-1][1] == question["reply"]
    assert await scalar(app, "SELECT COUNT(*) FROM dialogue_threads") == 1
    assert await scalar(app, "SELECT status FROM dialogue_threads") == "awaiting_user"
    assert (
        await scalar(app, "SELECT COUNT(*) FROM followups WHERE status='pending'") == 0
    )
    assert not any(task in {"repair", "recover"} for task, _ in app.model.calls)


async def test_declarative_followup_with_goal_still_needs_semantic_question(
    app: Application,
) -> None:
    enable(app)
    app.model.responses["generate"].append(
        {"reply": "I like quiet cafes", "dialogue_act": "opinion"}
    )
    await receive(app, app.sender, 1, "I enjoy cafes", 1, app.clock.now())
    app.model.responses["followup"].append(
        {
            "reply": "Cafes have windows",
            "dialogue_act": "opinion",
            "question_goal": "cafe_table_preference",
            "should_send": True,
        }
    )
    app.model.responses["validate"].append(
        {"decision": "reject", "reason_codes": ["FOLLOWUP_QUESTION_REQUIRED"]}
    )
    app.model.responses["repair"].append(
        {
            "reply": "Do you prefer a table by the window",
            "dialogue_act": "question",
            "question_goal": "cafe_table_preference",
            "should_send": True,
        }
    )
    await due(app)
    await app.drain()
    assert (1, "Cafes have windows") not in app.sender.sent
    assert app.sender.sent[-1][1] == "Do you prefer a table by the window"


async def test_followup_waits_a_full_minute_and_new_input_cancels_it(
    app: Application,
) -> None:
    enable(app)
    app.model.responses["generate"].append(
        {"reply": "I like quiet cafes", "dialogue_act": "opinion"}
    )
    await receive(app, app.sender, 1, "I enjoy cafes", 1, app.clock.now())
    assert await scalar(app, "SELECT run_at FROM followups") - app.clock.now() == 60
    app.clock.advance(59)
    await app.drain()
    assert not any(task == "followup" for task, _ in app.model.calls)
    await app.telegram.ingest([update(2, "Especially the little ones")])
    assert await scalar(app, "SELECT status FROM followups") == "cancelled"


def test_toml_and_runtime_followup_defaults_are_one_minute() -> None:
    defaults = RuntimeSettings()
    configured = tomllib.loads(Path("app/config/settings.toml").read_text())["default"][
        "runtime"
    ]
    assert defaults.followup_min_seconds == configured["followup_min_seconds"] == 60
    assert defaults.followup_max_seconds == configured["followup_max_seconds"] == 60


@pytest.mark.parametrize("probability,count", [(0.0, 2), (0.99, 1)])
def test_long_reply_has_one_or_two_reversible_letter_errors(
    monkeypatch: pytest.MonkeyPatch, probability: float, count: int
) -> None:
    monkeypatch.setattr(
        "app.modules.agent.style_plan.random.random", lambda: probability
    )
    monkeypatch.setattr(
        "app.modules.agent.style_plan.random.choice", lambda items: items[0]
    )
    original = ReplyProposal(
        reply="I really enjoy quiet music, especially when a song has an unexpected ending",
        dialogue_act="opinion",
    )
    styled = apply_casual_imperfection(original)
    assert styled.casual_imperfection is not None
    assert visible_edit_count(styled.casual_imperfection) == count
    assert semantic_reply(styled).reply == original.reply
    assert apply_casual_imperfection(styled) is styled


def test_short_reply_has_only_one_letter_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.modules.agent.style_plan.random.random", lambda: 0.0)
    original = ReplyProposal(reply="I really enjoy music", dialogue_act="opinion")
    styled = apply_casual_imperfection(original)
    assert styled.casual_imperfection is not None
    assert visible_edit_count(styled.casual_imperfection) == 1
    assert semantic_reply(styled).reply == original.reply


@pytest.mark.parametrize(
    "written,standard",
    [
        ("realy enojy msuic", "really enjoy music"),
        ("realy do enojy", "really not enjoy"),
        ("realy 3 enojy", "really 2 enjoy"),
        ("realy adore music", "really enjoy music"),
    ],
)
def test_combined_evidence_cannot_hide_extra_edits(written: str, standard: str) -> None:
    edit = ChatImperfection(written=written, standard=standard)
    assert not supported_imperfection(edit, written)


def test_two_edits_do_not_span_protected_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.modules.agent.style_plan.random.random", lambda: 0.0)
    monkeypatch.setattr(
        "app.modules.agent.style_plan.random.choice", lambda items: items[0]
    )
    original = ReplyProposal(
        reply='I really believe "quiet music" has an effect on the entire room, especially this song',
        dialogue_act="opinion",
    )
    styled = apply_casual_imperfection(original)
    assert styled.reply == original.reply.replace("really", "realy")
    assert semantic_reply(styled).reply == original.reply


async def test_long_delivery_preserves_validation_and_auditable_two_edits(
    app: Application, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "app.modules.agent.service.apply_casual_imperfection", apply_casual_imperfection
    )
    monkeypatch.setattr("app.modules.agent.style_plan.random.random", lambda: 0.0)
    monkeypatch.setattr(
        "app.modules.agent.style_plan.random.choice", lambda items: items[0]
    )
    await receive(app, app.sender, 1, "Hello", 1, app.clock.now())
    original = (
        "I really enjoy quiet music, especially when a song has an unexpected ending"
    )
    app.model.responses["generate"].append(
        {"reply": original, "dialogue_act": "opinion"}
    )
    await receive(app, app.sender, 2, "Music has surprises", 1, app.clock.now())
    checked = [data for task, data in app.model.calls if task == "validate"][-1]
    assert checked["reply"]["reply"] == original
    assert app.sender.sent[-1][1] == original.replace("really", "realy").replace(
        "quiet", "qiuet"
    )


def test_combined_typos_respect_evidence_length_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.modules.agent.style_plan.random.random", lambda: 0.0)
    monkeypatch.setattr(
        "app.modules.agent.style_plan.random.choice", lambda items: items[0]
    )
    original = ReplyProposal(
        reply="melody " + "x" * 66 + " melody", dialogue_act="opinion"
    )
    assert len(original.reply) == 80
    styled = apply_casual_imperfection(original)
    assert styled.casual_imperfection is not None
    assert visible_edit_count(styled.casual_imperfection) == 1
    assert semantic_reply(styled).reply == original.reply
