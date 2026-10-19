import pytest

from app.application import Application
from app.evaluation import receive
from app.modules.agent.schemas import ChatImperfection, ReplyProposal
from app.modules.agent.style_plan import (
    apply_casual_imperfection,
    semantic_reply,
    style_policy,
)
from tests.helpers import conversation

pytestmark = pytest.mark.usefixtures("full_letter_typo_rate")


@pytest.fixture
def casual_text_edits() -> bool:
    return True


@pytest.mark.parametrize("interval", [1, 2])
@pytest.mark.parametrize("part_count", [1, 2])
async def test_new_frequency_doubles_edits_in_single_and_two_part_replies(
    app: Application,
    monkeypatch: pytest.MonkeyPatch,
    interval: int,
    part_count: int,
) -> None:
    policy = style_policy().model_copy(
        update={
            "imperfection_interval": interval,
            "interval_jitter": 0,
            "letter_typo_keep_probability": 1,
        }
    )
    monkeypatch.setattr("app.modules.agent.style_plan.style_policy", lambda: policy)
    monkeypatch.setattr(
        "app.modules.agent.style_plan.random.choice", lambda items: items[0]
    )
    original = ReplyProposal(
        reply="I like music",
        continuation="I like coffee" if part_count == 2 else None,
        dialogue_act="opinion",
    )
    for number in range(1, 6):
        app.model.responses["generate"].append(original.model_dump())
        await receive(app, app.sender, number, "Tell me more", 1, app.clock.now())
    current = await conversation(app)
    history = await app.conversations.messages(current.scope)
    delivered = [message for message in history if message.role == "assistant"]
    assert len(delivered) == 5 * part_count
    assert not any(
        message.metadata["casual_imperfection_used"]
        for message in delivered[:part_count]
    )
    edited = sum(
        message.metadata["casual_imperfection_used"]
        for message in delivered[part_count:]
    )
    assert edited == 4 * part_count // interval
    for index, message in enumerate(delivered):
        part = ReplyProposal.model_validate(
            {
                **{
                    key: value
                    for key, value in message.metadata.items()
                    if key in ReplyProposal.model_fields
                },
                "reply": message.text,
                "dialogue_act": "opinion",
            }
        )
        assert semantic_reply(part).reply == original.parts[index % part_count]


@pytest.mark.parametrize("probability", [0.0, 0.99])
async def test_extra_part_styling_cannot_change_validated_meaning(
    app: Application, monkeypatch: pytest.MonkeyPatch, probability: float
) -> None:
    monkeypatch.setattr(
        "app.modules.agent.style_plan.random.random", lambda: probability
    )
    monkeypatch.setattr(
        "app.modules.agent.style_plan.random.choice", lambda items: items[0]
    )
    await receive(app, app.sender, 1, "Hello", 1, app.clock.now())

    def unsafe_edit(proposal, protected):
        if proposal.continuation is not None:
            return apply_casual_imperfection(proposal, protected)
        return proposal.model_copy(
            update={
                "reply": "I realy have a dog",
                "casual_imperfection_used": True,
                "casual_imperfection": ChatImperfection(
                    standard="really", written="realy"
                ),
            }
        )

    monkeypatch.setattr(
        "app.modules.agent.service.apply_casual_imperfection", unsafe_edit
    )
    app.model.responses["generate"].append(
        {
            "reply": "I like music",
            "continuation": "No, I don’t have any pets",
            "dialogue_act": "opinion_and_answer",
        }
    )
    await receive(app, app.sender, 2, "Do you have any pets?", 1, app.clock.now())
    current = await conversation(app)
    delivered = (await app.conversations.messages(current.scope))[-1]
    checked = semantic_reply(
        ReplyProposal(
            reply=delivered.text,
            dialogue_act="answer",
            casual_imperfection_used=delivered.metadata["casual_imperfection_used"],
            casual_imperfection=delivered.metadata["casual_imperfection"],
        )
    )
    assert checked.reply == "No, I don’t have any pets"
    assert all("have a dog" not in text for _, text in app.sender.sent)
