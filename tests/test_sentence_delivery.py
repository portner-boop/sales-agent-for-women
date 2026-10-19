import pytest

from app.application import Application
from app.evaluation import receive
from app.modules.agent.schemas import ChatImperfection, ReplyProposal
from app.modules.agent.segmentation import delivery_parts, sentence_parts
from app.modules.agent.style_plan import semantic_reply
from app.modules.delivery.entities import SendResult
from app.shared.storage import fetchall
from tests.helpers import scalar, update
from tests.test_followups import enable

INTERESTS = "Im into discovering new places and good food too. What kind of food do you find interesting?"
GREETING = "Nice to meet you, Olga. History must give you plenty of fascinating stories to share, and Max is a gerat name for a dog"
THREE_SENTENCES = "I enjoy discovering new places on foot. Small streets often have the best cafes. A quiet table by the window suits me"


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            INTERESTS,
            (
                "Im into discovering new places and good food too.",
                "What kind of food do you find interesting?",
            ),
        ),
        (
            GREETING,
            (
                "Nice to meet you, Olga.",
                "History must give you plenty of fascinating stories to share, and Max is a gerat name for a dog",
            ),
        ),
        (
            THREE_SENTENCES,
            (
                "I enjoy discovering new places on foot.",
                "Small streets often have the best cafes.",
                "A quiet table by the window suits me",
            ),
        ),
        ("Yes. I agree", ("Yes. I agree",)),
        (
            "A long walk through the old streets with a quiet stop for coffee by the water sounds like a lovely afternoon",
            (
                "A long walk through the old streets with a quiet stop for coffee by the water sounds like a lovely afternoon",
            ),
        ),
    ],
)
def test_sentence_boundaries_match_user_examples(
    text: str, expected: tuple[str, ...]
) -> None:
    assert sentence_parts(text) == expected


@pytest.mark.parametrize(
    "fragment",
    [
        "Dr. Smith",
        "J. Smith",
        "U.S. history",
        "19. October",
        '"First sentence. Second sentence"',
        "https://example.test/story. Next",
    ],
)
def test_sentences_do_not_split_abbreviations_dates_quotes_or_links(
    fragment: str,
) -> None:
    text = f"I was thinking about {fragment} and how many different details there are to talk about"
    assert sentence_parts(text) == (text,)


def test_sentence_split_preserves_protected_fact() -> None:
    text = "Her description was First sentence. Second sentence and it included many other interesting details"
    assert sentence_parts(text, ["First sentence. Second sentence"]) == (text,)


@pytest.mark.parametrize("first_question", [False, True])
def test_question_metadata_follows_actual_sentence(first_question: bool) -> None:
    question = "What kind of food do you find interesting?"
    statement = "Im into discovering new places and good food too"
    text = f"{question} {statement}" if first_question else f"{statement}. {question}"
    parts = delivery_parts(
        ReplyProposal(
            reply=text,
            dialogue_act="chat",
            question_goal="food_preference",
            question_part="reply",
        ),
        True,
    )
    assert len(parts) == 2
    assert [part.question_goal for part in parts] == (
        ["food_preference", None] if first_question else [None, "food_preference"]
    )


def test_unmarked_question_keeps_original_part() -> None:
    proposal = ReplyProposal(
        reply=INTERESTS.rstrip("?"),
        dialogue_act="chat",
        question_goal="food_preference",
        question_part="reply",
    )
    assert delivery_parts(proposal, True)[0].reply == proposal.reply
    assert len(delivery_parts(proposal, True)) == 1


def test_style_evidence_across_boundary_is_restored_before_splitting() -> None:
    proposal = ReplyProposal(
        reply="I enjoy discovering new places and good food too. What kind of food do you find intresting?",
        dialogue_act="chat",
        question_goal="food_preference",
        casual_imperfection_used=True,
        casual_imperfection=ChatImperfection(
            standard="interesting", written="intresting"
        ),
    )
    parts = delivery_parts(proposal, True)
    assert len(parts) == 2
    assert all(part.casual_imperfection is None for part in parts)
    assert (
        semantic_reply(parts[1]).reply == "What kind of food do you find interesting?"
    )


async def prepare_sentences(app: Application, question: bool = False) -> list[str]:
    app.model.responses["generate"].append(
        {
            "reply": THREE_SENTENCES
            + (". What is your favorite place to walk?" if question else ""),
            "dialogue_act": "opinion",
            "question_goal": "walking_place" if question else None,
            "question_part": "reply" if question else None,
        }
    )
    await app.telegram.ingest([update(1, "Tell me what you enjoy")])
    await app.turn_worker.run_once()
    async with app.database.read() as tx:
        rows = await fetchall(tx, "SELECT id FROM outbox_messages ORDER BY rowid")
    assert len(rows) == (4 if question else 3)
    return [row["id"] for row in rows]


async def deliver_part(app: Application, identifier: str) -> None:
    due = await scalar(
        app, "SELECT created_at FROM outbox_messages WHERE id=?", (identifier,)
    )
    app.clock.advance(max(0, due - app.clock.now()))
    await app.delivery.send(identifier)


@pytest.mark.parametrize("pacing", [False, True])
async def test_sentence_chain_waits_and_followup_starts_only_after_last_part(
    app: Application,
    pacing: bool,
) -> None:
    enable(app)
    runtime = app.settings.runtime.model_copy(update={"reply_pacing_enabled": pacing})
    app.settings = app.settings.model_copy(update={"runtime": runtime})
    app.agent.settings = runtime
    app.telegram.settings = runtime
    app.followups.settings = runtime
    identifiers = await prepare_sentences(app)
    await app.delivery.send(identifiers[-1])
    assert not app.sender.sent
    await deliver_part(app, identifiers[0])
    assert await scalar(app, "SELECT COUNT(*) FROM followups") == 0
    await app.close()
    restarted = await Application.open(app.settings, app.sender, app.model, app.clock)
    try:
        await deliver_part(restarted, identifiers[1])
        assert await scalar(restarted, "SELECT COUNT(*) FROM followups") == 0
        await deliver_part(restarted, identifiers[2])
        assert len(app.sender.sent) == 3
        assert (
            await scalar(restarted, "SELECT run_at FROM followups")
            == app.clock.now() + 60
        )
        metrics = await restarted.metrics()
        assert metrics["reply_latency_samples"] == 1
        assert metrics["reply_completion_samples"] == 1
    finally:
        await restarted.close()


async def test_question_is_recorded_only_after_its_sentence_is_delivered(
    app: Application,
) -> None:
    identifiers = await prepare_sentences(app, question=True)
    for identifier in identifiers[:-1]:
        await deliver_part(app, identifier)
        assert await scalar(app, "SELECT COUNT(*) FROM dialogue_threads") == 0
    await deliver_part(app, identifiers[-1])
    assert await scalar(app, "SELECT COUNT(*) FROM dialogue_threads") == 1
    assert (
        await scalar(
            app,
            "SELECT external_id FROM messages WHERE id=(SELECT last_question_message_id FROM dialogue_threads)",
        )
        == app.sender.next_id
    )


@pytest.mark.parametrize("outcome", ["delivery_unknown", "permanent_failure"])
@pytest.mark.parametrize("failed_index", [0, 1])
async def test_failed_first_sentence_cancels_all_descendants(
    app: Application, outcome: str, failed_index: int
) -> None:
    identifiers = await prepare_sentences(app)
    for identifier in identifiers[:failed_index]:
        await deliver_part(app, identifier)
    app.sender.results.append(SendResult(outcome, error_code="TEST_FAILURE"))
    await deliver_part(app, identifiers[failed_index])
    assert (
        await scalar(
            app, "SELECT COUNT(*) FROM outbox_messages WHERE status='cancelled'"
        )
        == len(identifiers) - failed_index - 1
    )
    assert (
        await scalar(app, "SELECT COUNT(*) FROM outbox_messages WHERE status='ready'")
        == 0
    )


@pytest.mark.parametrize("text", ["/stop", "Actually, let me tell you something else"])
async def test_new_input_cancels_every_unsent_sentence(
    app: Application, text: str
) -> None:
    identifiers = await prepare_sentences(app, question=True)
    await deliver_part(app, identifiers[0])
    await app.telegram.ingest([update(2, text)])
    assert (
        await scalar(
            app, "SELECT COUNT(*) FROM outbox_messages WHERE status='cancelled'"
        )
        == 3
    )
    assert await scalar(app, "SELECT COUNT(*) FROM dialogue_threads") == 0


async def test_recovery_cancels_whole_chain_after_unknown_first_send(
    app: Application,
) -> None:
    identifiers = await prepare_sentences(app)

    async def interrupted(tx):
        await tx.execute(
            "UPDATE outbox_messages SET status='sending' WHERE id=?", (identifiers[0],)
        )

    await app.database.write(interrupted)
    await app.delivery_repository.recover()
    assert (
        await scalar(
            app, "SELECT COUNT(*) FROM outbox_messages WHERE status='cancelled'"
        )
        == 2
    )


@pytest.mark.parametrize("detail,recall", [(True, "none"), (False, "all")])
async def test_detailed_and_recall_replies_remain_whole(
    app: Application, detail: bool, recall: str
) -> None:
    app.model.responses["interpret"].append(
        {
            "intent": "chat",
            "dialogue_mode": "continuation",
            "language": "en",
            "detail_requested": detail,
            "recall_scope": recall,
        }
    )
    app.model.responses["generate"].append(
        {"reply": THREE_SENTENCES, "dialogue_act": "opinion"}
    )
    await receive(app, app.sender, 1, "Tell me in detail", 1, app.clock.now())
    assert [text for _, text in app.sender.sent] == [THREE_SENTENCES]


@pytest.mark.parametrize(
    "text,question", [(INTERESTS, "food_preference"), (GREETING, None)]
)
async def test_user_examples_are_split_even_when_generator_uses_one_field(
    app: Application, text: str, question: str | None
) -> None:
    app.model.responses["generate"].append(
        {"reply": text, "dialogue_act": "chat", "question_goal": question}
    )
    await receive(app, app.sender, 1, "Tell me more", 1, app.clock.now())
    assert len(app.sender.sent) == 2
    assert [message for _, message in app.sender.sent] == [
        sentence.rstrip(".") for sentence in sentence_parts(text)
    ]
    assert [task for task, _ in app.model.calls] == [
        "interpret",
        "generate",
        "validate",
    ]


async def test_sentences_count_as_one_commerce_reply(app: Application) -> None:
    commerce = app.settings.commerce.model_copy(update={"preview_enabled": True})
    app.settings = app.settings.model_copy(update={"commerce": commerce})
    app.commerce.settings = commerce
    identifiers = await prepare_sentences(app)
    for identifier in identifiers:
        await deliver_part(app, identifier)
    assert await scalar(app, "SELECT preview_reply_count FROM conversations") == 1


@pytest.mark.usefixtures("full_letter_typo_rate")
@pytest.mark.parametrize("casual_text_edits", [True])
async def test_each_sentence_keeps_reversible_style(app: Application) -> None:
    await receive(app, app.sender, 1, "Hello", 1, app.clock.now())
    text = "I enjoy discovering new places and good food too. What kind of food do you find interesting?"
    app.model.responses["generate"].append(
        {"reply": text, "dialogue_act": "chat", "question_goal": "food_preference"}
    )
    await receive(app, app.sender, 2, "Tell me about food", 1, app.clock.now())
    from tests.helpers import conversation

    current = await conversation(app)
    history = await app.conversations.messages(current.scope)
    for message, expected in zip(history[-2:], sentence_parts(text), strict=True):
        assert message.metadata["casual_imperfection_used"]
        proposal = ReplyProposal(
            reply=message.text,
            dialogue_act="chat",
            casual_imperfection_used=True,
            casual_imperfection=message.metadata["casual_imperfection"],
        )
        assert semantic_reply(proposal).reply == expected.rstrip(".")
