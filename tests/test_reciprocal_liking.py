import pytest

from app.application import Application
from app.evaluation import receive
from app.modules.conversations.schemas import AgendaOperation
from app.shared.storage import Transaction
from tests.helpers import conversation, scalar, update


@pytest.mark.parametrize(
    "language,compliment,first_reply,question,answer,continuation",
    [
        (
            "en",
            "I really like our talk Marko, you are nice",
            "I like talking with you too",
            "And you? Do you like talking with me too?",
            "Yes, I do",
            "What kind of music do you enjoy?",
        ),
        (
            "ru",
            "Мне нравится с тобой общаться, Марко",
            "Мне тоже приятно с тобой разговаривать",
            "А тебе тоже нравится со мной общаться?",
            "Да, нравится",
            "Какую музыку ты любишь?",
        ),
    ],
)
async def test_repeated_liking_prepares_both_parts_in_the_same_turn(
    app: Application,
    monkeypatch: pytest.MonkeyPatch,
    language: str,
    compliment: str,
    first_reply: str,
    question: str,
    answer: str,
    continuation: str,
) -> None:
    app.model.responses["generate"].append(
        {"reply": first_reply, "dialogue_act": "reciprocate_liking"}
    )
    await receive(app, app.sender, 1, compliment, 1, app.clock.now())
    app.model.responses["interpret"].append(
        {
            "intent": "confirm_reciprocal_liking",
            "dialogue_mode": "continuation",
            "continue_conversation": True,
            "language": language,
            "policy_topics": ["social"],
        }
    )
    app.model.responses["generate"].append(
        {
            "reply": answer,
            "continuation": continuation,
            "dialogue_act": "affirm_and_ask",
            "question_goal": "music_preference",
            "question_part": "continuation",
        }
    )
    original = app.model.complete
    checked = False

    async def check_schema(task, rules, data, schema, deadline_seconds):
        nonlocal checked
        if task == "generate":
            assert data["DIALOGUE_PLAN"]["requires_continuation"] is False
            assert data["DIALOGUE_PLAN"]["strategy"] == "develop_conversation_naturally"
            candidate = schema.model_validate(
                {"reply": answer, "dialogue_act": "affirm"}
            )
            assert candidate.continuation is None
            checked = True
        return await original(task, rules, data, schema, deadline_seconds)

    monkeypatch.setattr(app.model, "complete", check_schema)
    await receive(app, app.sender, 2, question, 1, app.clock.now())
    assert checked
    assert [text for _, text in app.sender.sent][-2:] == [answer, continuation]
    assert (
        await scalar(
            app,
            "SELECT COUNT(*) FROM outbox_messages WHERE turn_id=(SELECT turn_id FROM messages WHERE role='user' ORDER BY sequence DESC LIMIT 1) AND status='delivered'",
        )
        == 2
    )
    assert (
        await scalar(
            app,
            "SELECT COUNT(*) FROM followups WHERE status IN ('pending','generating')",
        )
        == 0
    )
    assert not any(task == "followup" for task, _ in app.model.calls)


async def test_old_optional_history_does_not_hide_previous_questions(
    app: Application,
) -> None:
    app.agent.builder.settings = app.settings.memory.model_copy(
        update={"context_tokens": 12000}
    )
    await app.telegram.ingest([update(1, "An earlier story. " + "hi " * 5000)])
    current = await conversation(app)
    goals = ["music_preference", "outdoor_interests", "favorite_food", "travel_places"]

    async def seed(tx: Transaction) -> None:
        await tx.execute("UPDATE messages SET pending=0")
        for goal in goals:
            await app.conversations.update_agenda(
                tx,
                current.scope,
                AgendaOperation(
                    topic_kind=goal, status="awaiting_user", question_goal=goal
                ),
                app.clock.now(),
            )

    await app.database.write(seed)
    app.model.responses["interpret"].append(
        {
            "intent": "reciprocal_liking",
            "dialogue_mode": "continuation",
            "continue_conversation": True,
            "language": "en",
            "policy_topics": ["social"],
        }
    )
    app.model.responses["generate"].append(
        {
            "reply": "Yes, I do",
            "continuation": "Do you enjoy cooking?",
            "dialogue_act": "affirm_and_ask",
            "question_goal": "enjoys_cooking",
            "question_part": "continuation",
        }
    )
    await receive(
        app, app.sender, 2, "Do you like talking with me too?", 1, app.clock.now()
    )
    context = next(data for task, data in app.model.calls if task == "generate")
    assert {item["question_goal"] for item in context["AGENDA"]} == set(goals)
    assert {item["goal"] for item in context["QUESTION_EXCLUSIONS"]} == set(goals)
    assert context["HISTORY_STATUS"]["omitted_uncompressed_messages"] > 0
    assert context["HISTORY_STATUS"]["omitted_agenda_items"] == 0
    assert [text for _, text in app.sender.sent] == [
        "Yes, I do",
        "Do you enjoy cooking?",
    ]


async def test_repeated_question_recovery_keeps_the_answer_without_another_question(
    app: Application,
) -> None:
    app.model.responses["generate"].append(
        {
            "reply": "What music do you enjoy?",
            "dialogue_act": "ask",
            "question_goal": "music_preference",
            "question_part": "reply",
        }
    )
    await receive(app, app.sender, 1, "I like our conversation", 1, app.clock.now())
    app.model.responses["interpret"].append(
        {
            "intent": "reciprocal_liking",
            "dialogue_mode": "continuation",
            "continue_conversation": True,
            "language": "en",
            "policy_topics": ["social"],
        }
    )
    repeated = {
        "reply": "Yes, I do",
        "continuation": "What music do you enjoy?",
        "dialogue_act": "affirm_and_ask",
        "question_goal": "music_preference",
        "question_part": "continuation",
    }
    for task in ("generate", "repair", "recover"):
        app.model.responses[task].append(repeated)
    await receive(
        app,
        app.sender,
        2,
        "I like jazz. Do you like talking with me too?",
        1,
        app.clock.now(),
    )
    assert [text for _, text in app.sender.sent] == [
        "What music do you enjoy?",
        "Yes, I do",
    ]
    checks = [data for task, data in app.model.calls if task == "validate_recovery"]
    assert checks[-1]["reply"]["continuation"] is None
    assert not any("APPROVED_ANSWER" in data for _, data in app.model.calls)
    assert (
        await scalar(app, "SELECT status FROM agent_runs ORDER BY rowid DESC LIMIT 1")
        == "prepared"
    )
    assert not any(task == "followup" for task, _ in app.model.calls)
