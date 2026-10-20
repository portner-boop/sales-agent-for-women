import pytest

from app.application import Application
from app.modules.dialogue.presentation import ChatRendering


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        ("Nora, Max is 9.", "Nora, Max is 9"),
        ("I'd choose 3.5.", "I'd choose 3.5"),
        ("так бывает...", "так бывает..."),
        ("ты серьёзно?", "ты серьёзно?"),
        ("我在这里。", "我在这里"),
        ("میں یہاں ہوں۔", "میں یہاں ہوں"),
    ],
)
def test_chat_presentation_changes_only_single_terminal_period(
    original: str, expected: str
) -> None:
    assert ChatRendering().render(original) == expected
    assert ChatRendering(omit_terminal_period=False).render(original) == original


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        ("Eva is 9; Leon is 12", "Eva is 9, Leon is 12"),
        ("她九岁；他十二岁", "她九岁，他十二岁"),
        ("مساء هادئ؛ موسيقى", "مساء هادئ، موسيقى"),
        ("quiet — not lonely", "quiet - not lonely"),
        ("between 3.5–7.5, not 8", "between 3.5-7.5, not 8"),
    ],
)
def test_cosmetic_punctuation_is_rendered_without_rejecting_the_draft(
    original: str,
    expected: str,
) -> None:
    from app.modules.dialogue.context import build_request
    from app.modules.dialogue.presentation import CHAT_RENDERING
    from app.modules.dialogue.schemas import DialogueDraft
    from app.modules.dialogue.validation import validate_draft
    from tests.test_v2_dialogue import message, session

    request = build_request(session(), [], [message(1, text="hello")])
    draft = DialogueDraft(parts=[original], language="en")
    assert "FORBIDDEN_PUNCTUATION" not in validate_draft(draft, request)
    assert CHAT_RENDERING.render(original) == expected
    assert CHAT_RENDERING.render(expected) == expected
    assert draft.parts == [original]


@pytest.mark.parametrize("replacements", [{"a": ","}, {";": "9"}, {";": "word"}])
def test_rendering_cannot_replace_words_or_numbers(
    replacements: dict[str, str],
) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ChatRendering(punctuation_replacements=replacements)


def test_rendering_replacements_are_editable() -> None:
    renderer = ChatRendering(punctuation_replacements={";": ","})
    assert renderer.render("hello; still here.") == "hello, still here"
    assert ChatRendering().render("hello; still here.") == "hello; still here"


def test_semicolon_emoticon_is_still_rejected() -> None:
    from app.modules.dialogue.context import build_request
    from app.modules.dialogue.schemas import DialogueDraft
    from app.modules.dialogue.validation import validate_draft
    from tests.test_v2_dialogue import message, session

    request = build_request(session(), [], [message(1)])
    assert "EMOJI_NOT_ALLOWED" in validate_draft(
        DialogueDraft(parts=["hello ;)"], language="en"), request
    )


@pytest.mark.parametrize(
    ("punctuation", "replacement"),
    [(";", ","), ("；", "，"), ("؛", "،"), ("—", "-"), ("–", "-")],
)
async def test_punctuation_keeps_the_reply_and_question_in_one_model_call(
    app: Application,
    punctuation: str,
    replacement: str,
) -> None:
    import json

    from tests.helpers import conversation, scalar, update

    original = f"Eva is 9{punctuation} Leon is 12"
    expected = f"Eva is 9{replacement} Leon is 12"
    question = "what are you cooking tonight?"
    app.model.responses["dialogue"].append(
        {
            "parts": [original],
            "continuation": question,
            "conversation_action": "ask",
            "language": "en",
        }
    )
    await app.telegram.ingest([update(1, "what do you remember?")])
    await app.drain()
    assert [text for _, text in app.sender.sent] == [expected, question]
    assert len(app.model.calls) == 1
    assert (await conversation(app)).preview_reply_count == 1
    history = await app.conversations.messages((await conversation(app)).scope)
    assert [m.text for m in history if m.role == "assistant"] == [expected, question]
    metrics = json.loads(await scalar(app, "SELECT metrics_json FROM agent_runs"))
    assert not metrics["llm_events"][0]["validation_errors"]
    raw = json.loads(metrics["research_observations"][0]["first_raw_response"])
    assert raw["parts"] == [original]
    assert raw["continuation"] == question
