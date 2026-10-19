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
from tests.helpers import conversation


@pytest.fixture
def casual_text_edits() -> bool:
    return True


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I'd go", "Id go"),
        ("I’d go", "Id go"),
        ("I'm here", "Im here"),
        ("I’m here", "Im here"),
        ("I don't agree", "I dont agree"),
        ("I don’t agree", "I dont agree"),
        ("I CAN'T go", "I CANT go"),
    ],
)
def test_missing_apostrophe_preserves_exact_original(text: str, expected: str) -> None:
    original = ReplyProposal(reply=text, dialogue_act="answer")
    edited = apply_casual_imperfection(original)
    assert edited.reply == expected
    assert supported_imperfection(edited.casual_imperfection, edited.reply)
    assert semantic_reply(edited).reply == text
    assert apply_casual_imperfection(edited) is edited


@pytest.mark.parametrize(
    "probability,expected", [(0.0, "Id like tea"), (0.9, "I'd liek tea")]
)
def test_contraction_can_win_over_dictionary_typo(
    monkeypatch: pytest.MonkeyPatch, probability: float, expected: str
) -> None:
    monkeypatch.setattr(
        "app.modules.agent.style_plan.random.random", lambda: probability
    )
    monkeypatch.setattr(
        "app.modules.agent.style_plan.random.choice", lambda items: items[0]
    )
    edited = apply_casual_imperfection(
        ReplyProposal(reply="I'd like tea", dialogue_act="opinion")
    )
    assert edited.reply == expected


def test_contraction_and_letter_typo_can_share_short_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.modules.agent.style_plan.random.random", lambda: 0.0)
    monkeypatch.setattr(
        "app.modules.agent.style_plan.random.choice", lambda items: items[0]
    )
    original = ReplyProposal(
        reply="I’d enjoy a walk by the sea", dialogue_act="opinion"
    )
    edited = apply_casual_imperfection(original)
    assert edited.reply == "Id enojy a walk by the sea"
    assert edited.casual_imperfection is not None
    assert visible_edit_count(edited.casual_imperfection) == 2
    assert semantic_reply(edited).reply == original.reply


@pytest.mark.parametrize(
    "text,protected",
    [
        ("O’Connor is here", []),
        ('She said "I’d go"', []),
        ("https://example.test/I'd", []),
        ("I'd go", ["I'd"]),
    ],
)
def test_apostrophe_omission_preserves_names_quotes_and_protected_text(
    text: str, protected: list[str]
) -> None:
    original = ReplyProposal(reply=text, dialogue_act="answer")
    assert apply_casual_imperfection(original, protected) is original
    assert not supported_imperfection(
        ChatImperfection(standard="O’Connor", written="OConnor"), "OConnor"
    )


async def test_both_delivered_parts_can_omit_apostrophes(app: Application) -> None:
    await receive(app, app.sender, 1, "Hello", 1, app.clock.now())
    app.model.responses["generate"].append(
        {"reply": "I’d go", "continuation": "I’m here", "dialogue_act": "opinion"}
    )
    await receive(app, app.sender, 2, "Tell me more", 1, app.clock.now())
    assert [text for _, text in app.sender.sent[-2:]] == ["Id go", "Im here"]
    current = await conversation(app)
    history = await app.conversations.messages(current.scope)
    for message, original in zip(history[-2:], ["I’d go", "I’m here"], strict=True):
        edit = ChatImperfection.model_validate(message.metadata["casual_imperfection"])
        proposal = ReplyProposal(
            reply=message.text,
            dialogue_act="opinion",
            casual_imperfection_used=True,
            casual_imperfection=edit,
        )
        assert semantic_reply(proposal).reply == original
