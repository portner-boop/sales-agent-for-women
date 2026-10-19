import pytest

from app.application import Application
from app.evaluation import receive
from app.modules.agent.schemas import ChatImperfection, ReplyProposal
from app.modules.agent.style_plan import (
    reduce_letter_imperfections,
    semantic_reply,
    style_policy,
    supported_imperfection,
)
from tests.helpers import conversation


def test_half_of_selected_letter_typos_are_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert style_policy().letter_typo_keep_probability == 0.5
    proposal = ReplyProposal(
        reply="I liek music",
        dialogue_act="opinion",
        casual_imperfection_used=True,
        casual_imperfection=ChatImperfection(standard="like", written="liek"),
    )
    draws = iter(number / 100 for number in range(100))
    monkeypatch.setattr(
        "app.modules.agent.style_plan.random.random", lambda: next(draws)
    )
    results = [reduce_letter_imperfections(proposal) for _ in range(100)]
    assert sum(result.reply == "I liek music" for result in results) == 50
    assert sum(result.reply == "I like music" for result in results) == 50
    assert all(semantic_reply(result).reply == "I like music" for result in results)
    assert all(result.casual_imperfection is None for result in results[50:])


@pytest.mark.parametrize(
    "standard,written,expected",
    [("I'm like", "Im liek", "Im like"), ("like music", "liek msuic", "like music")],
)
def test_reducing_letter_errors_preserves_apostrophe_omissions(
    monkeypatch: pytest.MonkeyPatch, standard: str, written: str, expected: str
) -> None:
    monkeypatch.setattr("app.modules.agent.style_plan.random.random", lambda: 0.5)
    original = ReplyProposal(
        reply=written,
        dialogue_act="chat",
        casual_imperfection_used=True,
        casual_imperfection=ChatImperfection(standard=standard, written=written),
    )
    reduced = reduce_letter_imperfections(original)
    assert reduced.reply == expected
    assert semantic_reply(reduced).reply == standard
    if expected != standard:
        assert supported_imperfection(reduced.casual_imperfection, reduced.reply)


def test_two_letter_errors_are_thinned_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draws = iter([0.49, 0.5])
    monkeypatch.setattr(
        "app.modules.agent.style_plan.random.random", lambda: next(draws)
    )
    original = ReplyProposal(
        reply="I liek msuic",
        dialogue_act="opinion",
        casual_imperfection_used=True,
        casual_imperfection=ChatImperfection(
            standard="like music", written="liek msuic"
        ),
    )
    reduced = reduce_letter_imperfections(original)
    assert reduced.reply == "I liek music"
    assert supported_imperfection(reduced.casual_imperfection, reduced.reply)
    assert semantic_reply(reduced).reply == "I like music"


@pytest.mark.parametrize(
    "standard,written",
    [("I'd", "Id"), ("don't", "dont"), ("a little", "alittle"), ("Yes, I", "Yes,I")],
)
def test_other_chat_reductions_do_not_enter_letter_lottery(
    monkeypatch: pytest.MonkeyPatch, standard: str, written: str
) -> None:
    def unexpected_draw() -> float:
        raise AssertionError("Only letter errors are reduced")

    monkeypatch.setattr("app.modules.agent.style_plan.random.random", unexpected_draw)
    proposal = ReplyProposal(
        reply=written,
        dialogue_act="chat",
        casual_imperfection_used=True,
        casual_imperfection=ChatImperfection(standard=standard, written=written),
    )
    assert reduce_letter_imperfections(proposal) is proposal


@pytest.mark.parametrize("casual_text_edits", [True])
async def test_delivery_does_not_reinsert_removed_typo_and_keeps_apostrophe_style(
    app: Application, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.modules.agent.style_plan.random.random", lambda: 0.99)
    await receive(app, app.sender, 1, "Hello", 1, app.clock.now())
    app.model.responses["generate"].append(
        {"reply": "I like music", "continuation": "I'm here", "dialogue_act": "chat"}
    )
    await receive(app, app.sender, 2, "Tell me more", 1, app.clock.now())
    assert [text for _, text in app.sender.sent[-2:]] == ["I like music", "Im here"]
    current = await conversation(app)
    history = await app.conversations.messages(current.scope)
    assert history[-2].metadata["casual_imperfection"] is None
    assert history[-2].metadata["casual_imperfection_used"] is False
    assert history[-1].metadata["casual_imperfection_used"] is True
