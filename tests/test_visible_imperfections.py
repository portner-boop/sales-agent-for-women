import pytest

from app.application import Application
from app.modules.agent.schemas import ChatImperfection, ReplyProposal
from app.modules.agent.style_plan import (
    apply_casual_imperfection,
    semantic_reply,
    supported_imperfection,
)
from tests.helpers import conversation, scalar, update
from tests.test_followups import due, enable

pytestmark = pytest.mark.usefixtures("fixed_style_interval")


def test_semantic_view_restores_only_verified_edit_without_mutating_delivery() -> None:
    original = ReplyProposal(
        reply="I like music",
        continuation="Do you like cofee?",
        dialogue_act="question",
        question_goal="coffee_preference",
        question_part="continuation",
        casual_imperfection_used=True,
        casual_imperfection=ChatImperfection(written="cofee", standard="coffee"),
    )
    checked = semantic_reply(original)
    assert checked.reply == original.reply
    assert checked.continuation == "Do you like coffee?"
    assert checked.question_goal == original.question_goal
    assert checked.casual_imperfection is None
    assert original.continuation == "Do you like cofee?"
    assert original.casual_imperfection_used


@pytest.mark.parametrize(
    "text,written,standard,protected",
    [
        ("You have three children", "three", "two", ()),
        ('You wrote "cofee"', "cofee", "coffee", ()),
        ("I know O Neil", "O Neil", "O'Neil", ("O'Neil",)),
        ("cofee and cofee", "cofee", "coffee", ()),
    ],
)
def test_semantic_view_does_not_hide_unsupported_or_protected_changes(
    text: str, written: str, standard: str, protected: tuple[str, ...]
) -> None:
    original = ReplyProposal(
        reply=text,
        dialogue_act="answer",
        casual_imperfection_used=True,
        casual_imperfection=ChatImperfection(written=written, standard=standard),
    )
    assert semantic_reply(original, protected) is original


pytestmark = pytest.mark.usefixtures("full_letter_typo_rate")


@pytest.fixture
def casual_text_edits() -> bool:
    return True


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I really like that", "I realy like that"),
        ("It works because it is simple", "It works becuase it is simple"),
        ("I like it a little", "I like it alittle"),
        ("Мне очень нравится эта идея", "Мне очнь нравится эта идея"),
        ("Это кстати неплохая идея", "Это ксттаи неплохая идея"),
        ("Me gusta porque es sencillo", "Me gusta porqe es sencillo"),
        ("Cela me plaît vraiment", "Cela me plaît vraimnt"),
        ("هذا جميل", "هذا جمييل"),
    ],
)
def test_visible_edits_have_exact_auditable_evidence(
    text: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.modules.agent.style_plan.random.random", lambda: 0.99)
    monkeypatch.setattr(
        "app.modules.agent.style_plan.random.choice", lambda items: items[0]
    )
    original = ReplyProposal(reply=text, dialogue_act="opinion")
    edited = apply_casual_imperfection(original)
    assert edited.reply == expected
    assert edited.casual_imperfection_used
    assert supported_imperfection(edited.casual_imperfection, edited.full_reply)
    assert apply_casual_imperfection(edited) is edited


@pytest.mark.parametrize(
    "text,protected",
    [
        ('She wrote "really"', []),
        ("She wrote 'really'", []),
        ("Она сказала «очень»", []),
        ("`because`", []),
        ("https://example.test/really", []),
        ("really@example.test", []),
        ("My pet is little", ["little"]),
        ("My pet is coffee", ["coffee"]),
        ("Really likes dogs", []),
        ("Olga has 2 children and Max is 6", []),
        ("I do not agree", []),
        ("1,000 or 2,500", []),
    ],
)
def test_visible_edits_preserve_protected_content(
    text: str, protected: list[str]
) -> None:
    original = ReplyProposal(reply=text, dialogue_act="opinion")
    assert apply_casual_imperfection(original, protected) is original


def test_visible_edits_preserve_negation_and_only_change_one_part(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.modules.agent.style_plan.random.random", lambda: 0.99)
    monkeypatch.setattr(
        "app.modules.agent.style_plan.random.choice", lambda items: items[0]
    )
    original = ReplyProposal(
        reply="I do not really like that",
        continuation="I prefer something simple",
        dialogue_act="opinion",
    )
    result = apply_casual_imperfection(original)
    assert result.reply == "I do not realy like that"
    assert result.continuation == original.continuation
    assert not supported_imperfection(
        ChatImperfection(standard="not", written="now"), "now"
    )


async def test_visible_typos_are_checked_and_allowed_in_consecutive_messages(
    app: Application,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.modules.agent.style_plan.random.random", lambda: 0.99)
    monkeypatch.setattr(
        "app.modules.agent.style_plan.random.choice", lambda items: items[0]
    )
    for index in range(1, 5):
        await app.telegram.ingest([update(index, "I like a small cafe")])
        await app.drain()
    app.model.responses["generate"].append(
        {"reply": "I really like that idea", "dialogue_act": "opinion"}
    )
    await app.telegram.ingest([update(5, "A cafe near the park sounds nice")])
    await app.drain()
    assert app.sender.sent[-1] == (1, "I realy like that idea")
    checked = [data for task, data in app.model.calls if task == "validate"][-1]
    assert checked["reply"]["reply"] == "I really like that idea"
    assert checked["reply"]["casual_imperfection"] is None
    assert checked["reply"]["casual_imperfection_used"] is False
    current = await conversation(app)
    stored = (await app.conversations.messages(current.scope))[-1]
    assert stored.metadata["casual_imperfection_used"]
    assert "visible_forms" not in stored.metadata["style_plan"]
    generated = [data for task, data in app.model.calls if task == "generate"][-1]
    assert generated["STYLE_PLAN"]["request_imperfection"]
    assert generated["STYLE_PLAN"]["visible_forms"] == {}
    app.model.responses["generate"].append(
        {"reply": "I prefer a small table", "dialogue_act": "opinion"}
    )
    await app.telegram.ingest([update(6, "I agree with you")])
    await app.drain()
    assert app.sender.sent[-1] == (1, "I prefr a small table")
    app.model.responses["generate"].append(
        {"reply": "I really like round tables", "dialogue_act": "opinion"}
    )
    await app.telegram.ingest([update(7, "A round table is nice")])
    await app.drain()
    assert app.sender.sent[-1] == (1, "I realy like round tables")


def test_visible_edit_replaces_punctuation_instead_of_stacking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.modules.agent.style_plan.random.choice", lambda items: items[0]
    )
    original = ReplyProposal(
        reply="Yes I like music",
        dialogue_act="opinion",
        casual_imperfection_used=True,
        casual_imperfection=ChatImperfection(standard="Yes, I", written="Yes I"),
    )
    edited = apply_casual_imperfection(original)
    assert edited.reply == "Yes, I like msuic"
    assert edited.casual_imperfection == ChatImperfection(
        standard="music", written="msuic"
    )
    assert apply_casual_imperfection(edited) is edited


def test_edit_can_change_unprotected_phrase_around_protected_word() -> None:
    original = ReplyProposal(
        reply="Yes I like coffee",
        dialogue_act="opinion",
        casual_imperfection_used=True,
        casual_imperfection=ChatImperfection(standard="Yes, I", written="Yes I"),
    )
    edited = apply_casual_imperfection(original, ["coffee"])
    assert edited.reply == "Yes, I liek coffee"
    assert edited.casual_imperfection == ChatImperfection(
        standard="like", written="liek"
    )


def test_punctuation_upgrade_does_not_rewrite_a_quote() -> None:
    original = ReplyProposal(
        reply='She wrote "Yes I" about music',
        dialogue_act="opinion",
        casual_imperfection_used=True,
        casual_imperfection=ChatImperfection(standard="Yes, I", written="Yes I"),
    )
    assert apply_casual_imperfection(original) is original


@pytest.mark.parametrize("pick,expected_part", [(0, "reply"), (-1, "continuation")])
def test_random_selection_can_edit_either_part(
    monkeypatch: pytest.MonkeyPatch, pick: int, expected_part: str
) -> None:
    monkeypatch.setattr(
        "app.modules.agent.style_plan.random.choice", lambda items: items[pick]
    )
    original = ReplyProposal(
        reply="I like music", continuation="An acoustic set", dialogue_act="opinion"
    )
    edited = apply_casual_imperfection(original)
    for field in ("reply", "continuation"):
        assert (getattr(edited, field) != getattr(original, field)) == (
            field == expected_part
        )
    assert supported_imperfection(edited.casual_imperfection, edited.full_reply)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Yes, it fits", "Yes,it fits"),
        ("По-моему, это да", "По-моему,это да"),
        ("Si quieres, vamos", "Si quieres,vamos"),
        ("Franchement, cela convient", "Franchement,cela convient"),
        ("نعم، هذا مناسب", "نعم،هذا مناسب"),
    ],
)
def test_one_missing_comma_space_has_exact_evidence(
    text: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "app.modules.agent.style_plan.random.choice", lambda items: items[-1]
    )
    edited = apply_casual_imperfection(
        ReplyProposal(reply=text, dialogue_act="opinion")
    )
    assert edited.reply == expected
    assert supported_imperfection(edited.casual_imperfection, edited.reply)


@pytest.mark.parametrize(
    "text,protected",
    [
        ('She said "Yes, please"', []),
        ("Anna, welcome", ["Anna"]),
        ("1, 000", []),
        ("June, 12", []),
        ("https://example.test/yes,no", []),
    ],
)
def test_comma_space_preserves_quotes_names_and_numbers(
    text: str, protected: list[str]
) -> None:
    original = ReplyProposal(reply=text, dialogue_act="answer")
    assert apply_casual_imperfection(original, protected) is original


async def test_question_followup_can_deliver_the_scheduled_imperfection(
    app: Application,
) -> None:
    enable(app)
    for index in range(1, 5):
        app.model.responses["generate"].append(
            {
                "reply": "That fits",
                "dialogue_act": "reaction",
                "followup_allowed": index == 4,
            }
        )
        await app.telegram.ingest([update(index, "I enjoy cafes")])
        await app.drain()
    app.model.responses["followup"].append(
        {
            "reply": "Do you like coffee?",
            "dialogue_act": "question",
            "question_goal": "coffee_preference",
            "question_part": "reply",
            "should_send": True,
        }
    )
    await due(app)
    await app.drain()
    assert app.sender.sent[-1][1] != "Do you like coffee?"
    current = await conversation(app)
    message = (await app.conversations.messages(current.scope))[-1]
    edit = ChatImperfection.model_validate(message.metadata["casual_imperfection"])
    assert supported_imperfection(edit, message.text)
    assert message.text.replace(edit.written, edit.standard, 1) == "Do you like coffee?"
    assert message.metadata["origin"] == "followup"
    assert message.metadata["casual_imperfection_used"]
    assert (
        await scalar(app, "SELECT COUNT(*) FROM followups WHERE status='pending'") == 0
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I like fish best", "I liek fish best"),
        ("Макс явно выбрал диван", "Макс явнно выбрал диван"),
        ("Eso me parece genial", "Eso me parcee genial"),
        ("Ce plat je trouve excellent", "Ce plat je trouev excellent"),
        ("أحب السمك", "أححب السمك"),
    ],
)
def test_short_casual_reply_has_a_supported_visible_form(
    text: str,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.modules.agent.style_plan.random.choice", lambda items: items[0]
    )
    edited = apply_casual_imperfection(
        ReplyProposal(reply=text, dialogue_act="opinion")
    )
    assert edited.reply == expected
    assert supported_imperfection(edited.casual_imperfection, edited.reply)
