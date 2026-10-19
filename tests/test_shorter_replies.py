import pytest

from app.application import Application
from app.evaluation import receive

FIRST = "I like exploring old streets on foot and finding a small cafe along the way, especially when there is time to sit by a window and watch the neighborhood settle into its own rhythm"
SECOND = "Good food and a little music can make a simple evening memorable, and I enjoy the small details of a place as much as its famous landmarks, especially when there is no need to rush"


async def test_casual_answer_between_300_and_350_is_kept(app: Application) -> None:
    text = (
        FIRST
        + ", with good food and a little music to make the evening memorable, and enough time to enjoy the small details of a place as much as its famous landmarks"
    )
    assert 300 < len(text) <= 350
    app.model.responses["interpret"].append(
        {
            "intent": "chat",
            "dialogue_mode": "continuation",
            "language": "en",
            "policy_topics": ["social"],
        }
    )
    app.model.responses["generate"].append({"reply": text, "dialogue_act": "opinion"})
    await receive(
        app,
        app.sender,
        1,
        "What do you enjoy about exploring new places?",
        1,
        app.clock.now(),
    )
    assert app.sender.sent == [(1, text)]
    assert not any(task == "repair" for task, _ in app.model.calls)


@pytest.mark.parametrize("split", [False, True])
async def test_combined_casual_answer_is_rewritten_at_reduced_limit(
    app: Application, split: bool
) -> None:
    assert 350 < len(FIRST + ". " + SECOND) < 400
    app.model.responses["interpret"].append(
        {
            "intent": "chat",
            "dialogue_mode": "continuation",
            "language": "en",
            "policy_topics": ["social"],
        }
    )
    app.model.responses["generate"].append(
        {
            "reply": FIRST if split else FIRST + ". " + SECOND,
            "continuation": SECOND if split else None,
            "dialogue_act": "opinion",
        }
    )
    app.model.responses["repair"].append(
        {"reply": "I like old streets and small cafes", "dialogue_act": "opinion"}
    )
    await receive(
        app,
        app.sender,
        1,
        "What do you enjoy about exploring new places?",
        1,
        app.clock.now(),
    )
    assert app.sender.sent == [(1, "I like old streets and small cafes")]
    repair = next(data for task, data in app.model.calls if task == "repair")
    assert repair["DIALOGUE_PLAN"]["casual_character_limit"] == 350
    assert "OVERLOADED_CASUAL_REPLY" in repair["REPAIR_REASON_CODES"]
    assert any(task == "validate_repair" for task, _ in app.model.calls)


@pytest.mark.parametrize("detail,recall", [(True, "none"), (False, "all")])
async def test_requested_detail_and_full_recall_keep_long_answer(
    app: Application, detail: bool, recall: str
) -> None:
    text = FIRST + ". " + SECOND
    app.model.responses["interpret"].append(
        {
            "intent": "chat",
            "dialogue_mode": "continuation",
            "language": "en",
            "policy_topics": ["social"],
            "detail_requested": detail,
            "recall_scope": recall,
        }
    )
    app.model.responses["generate"].append({"reply": text, "dialogue_act": "opinion"})
    await receive(app, app.sender, 1, "Tell me in detail", 1, app.clock.now())
    assert app.sender.sent == [(1, text)]
    context = next(data for task, data in app.model.calls if task == "generate")
    assert context["DIALOGUE_PLAN"]["casual_character_limit"] is None
    assert not any(task == "repair" for task, _ in app.model.calls)
