import json

import pytest
from pydantic import ValidationError

from app.application import Application
from app.modules.dialogue.context import build_request
from app.modules.dialogue.schemas import DialogueDraft
from app.modules.dialogue.validation import validate_draft
from tests.helpers import conversation, scalar, update
from tests.test_v2_dialogue import message, session


@pytest.mark.parametrize(
    "fields",
    [
        {"parts": ["hello"], "continuation": " "},
        {"parts": [], "continuation": "hello"},
        {"parts": [], "should_send": False, "continuation": "hello"},
        {"parts": ["goodnight"], "closing": True, "continuation": "stay?"},
        {"parts": ["one", "two", "three"], "continuation": "four"},
    ],
)
def test_invalid_continuation_cannot_be_delivered(fields: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        DialogueDraft.model_validate({**fields, "language": "en"})


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("what do you think 🙂", "EMOJI_NOT_ALLOWED"),
        ("visit https://example.com", "UNAPPROVED_URL"),
    ],
)
def test_continuation_receives_all_text_checks(text: str, error: str) -> None:
    request = build_request(session(), [], [message(1)])
    draft = DialogueDraft(parts=["hello"], continuation=text, language="en")
    assert error in validate_draft(draft, request)


def test_timer_cannot_add_an_ordinary_continuation() -> None:
    request = build_request(session(), [], [message(1)], followup=True)
    draft = DialogueDraft(parts=["one thought"], continuation="another", language="en")
    assert "FOLLOWUP_CONTINUATION_NOT_ALLOWED" in validate_draft(draft, request)


async def test_continuation_is_delivered_now_as_one_turn_and_observed(
    app: Application,
) -> None:
    app.model.responses["dialogue"].append(
        {
            "parts": ["a free evening at last"],
            "continuation": "what are you doing with it?",
            "language": "en",
        }
    )
    await app.telegram.ingest([update(1, "first evening off this week")])
    await app.drain()
    assert [text for _, text in app.sender.sent] == [
        "a free evening at last",
        "what are you doing with it?",
    ]
    assert len(app.model.calls) == 1
    current = await conversation(app)
    assert current.preview_reply_count == 1
    history = await app.conversations.messages(current.scope)
    assert [item.text for item in history if item.role == "assistant"] == [
        text for _, text in app.sender.sent
    ]
    assert await scalar(app, "SELECT COUNT(DISTINCT turn_id) FROM outbox_messages") == 1
    metrics = json.loads(await scalar(app, "SELECT metrics_json FROM agent_runs"))
    observation = metrics["research_observations"][0]
    assert observation["first_parts"] == metrics["delivered_parts"]
    assert json.loads(observation["first_raw_response"])["continuation"] == (
        "what are you doing with it?"
    )


async def test_new_input_cancels_the_unsent_continuation(app: Application) -> None:
    app.model.responses["dialogue"].append(
        {"parts": ["hello"], "continuation": "old question?", "language": "en"}
    )
    await app.telegram.ingest([update(1, "hello")])
    await app.turn_worker.run_once()
    await app.delivery.run_once()
    assert [text for _, text in app.sender.sent] == ["hello"]
    await app.telegram.ingest([update(2, "actually a different topic")])
    await app.drain()
    assert "old question?" not in [text for _, text in app.sender.sent]
    assert (
        await scalar(
            app, "SELECT status FROM outbox_messages WHERE text='old question?'"
        )
        == "cancelled"
    )


@pytest.mark.parametrize(
    ("action", "parts", "continuation"),
    [
        ("ask", ["glad you came by"], "what made today different?"),
        ("play", ["your turn to deal"], None),
        ("share", ["the walk was nice", "i took the long way home"], None),
    ],
)
async def test_conversation_action_preserves_chosen_bubbles_without_a_quota(
    app: Application,
    action: str,
    parts: list[str],
    continuation: str | None,
) -> None:
    app.model.responses["dialogue"].append(
        {
            "conversation_action": action,
            "parts": parts,
            "continuation": continuation,
            "language": "en",
        }
    )
    await app.telegram.ingest([update(1, "keep me company a bit")])
    await app.drain()
    expected = parts + ([continuation] if continuation is not None else [])
    assert [text for _, text in app.sender.sent] == expected
    assert len(app.model.calls) == 1
    assert (await conversation(app)).preview_reply_count == 1
    proposal = json.loads(
        await scalar(app, "SELECT proposal_json FROM outbox_messages LIMIT 1")
    )
    assert proposal["conversation_action"] == action
