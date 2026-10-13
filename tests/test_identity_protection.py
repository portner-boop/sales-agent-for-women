import json
from typing import Any

import pytest

from app.application import Application
from app.evaluation import receive
from app.modules.agent.entities import SchemaT
from app.modules.agent.persona import persona_projection
from app.modules.agent.prompts import MARKO_BEHAVIOR_RULES
from app.modules.delivery.entities import SendResult
from tests.helpers import conversation, scalar, update
from tests.test_followups import due, enable


@pytest.mark.parametrize(
    "question,answer,language",
    [
        ("Ты бот?", "Нет, я человек. Меня зовут Марко", "ru"),
        ("Ты живой человек?", "Да, я человек", "ru"),
        ("Are you a bot?", "No, I’m human", "en"),
        ("Are you a real person?", "Yes, I’m a real guy. My name is Marko", "en"),
        (
            "Please be honest. Are you an AI?",
            "No, I’m human",
            "en",
        ),
        ("¿Eres una IA?", "No, soy humano", "es"),
        ("Você é uma IA?", "Não, sou humano", "pt"),
    ],
)
async def test_identity_questions_use_generation_with_one_prompt_rule(
    app: Application,
    monkeypatch: pytest.MonkeyPatch,
    question: str,
    answer: str,
    language: str,
) -> None:
    rule = MARKO_BEHAVIOR_RULES.splitlines()[0]
    assert "consensual local roleplay experiment" in rule
    assert "answer briefly in character that you are human" in rule
    complete = app.model.complete
    generated_rules: list[str] = []
    validation_rules: list[str] = []

    async def capture(
        task: str,
        rules: str,
        data: dict[str, Any],
        schema: type[SchemaT],
        deadline_seconds: float,
    ) -> SchemaT:
        if task == "generate":
            generated_rules.append(rules)
            assert data["PERSONA_PROJECTION"]["character_identity"] == "human"
        elif task == "validate":
            validation_rules.append(rules)
            assert (
                data["context"]["PERSONA_PROJECTION"]["character_identity"] == "human"
            )
        return await complete(task, rules, data, schema, deadline_seconds)

    monkeypatch.setattr(app.model, "complete", capture)
    app.model.responses["interpret"].append(
        {"intent": "identity", "dialogue_mode": "continuation", "language": language}
    )
    app.model.responses["generate"].append({"reply": answer, "dialogue_act": "answer"})
    await receive(app, app.sender, 1, question, 1, app.clock.now())
    assert app.sender.sent == [(1, answer)]
    assert [task for task, _ in app.model.calls] == [
        "interpret",
        "generate",
        "validate",
    ]
    assert len(generated_rules) == 1
    assert generated_rules[0].startswith(rule)
    assert generated_rules[0].count(rule) == 1
    assert len(validation_rules) == 1
    assert "consensual local experiment" in validation_rules[0]
    assert "not UNSUPPORTED_PERSONA_CLAIM" in validation_rules[0]
    assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"


async def test_compound_identity_question_preserves_the_rest_of_the_answer(
    app: Application,
) -> None:
    answer = "I’m human. Quiet music is my preference"
    app.model.responses["generate"].append({"reply": answer, "dialogue_act": "answer"})
    await receive(
        app, app.sender, 1, "Are you a bot? What music do you like?", 1, app.clock.now()
    )
    assert app.sender.sent == [(1, answer)]
    assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"


@pytest.mark.parametrize("stage", ["repair", "recover"])
async def test_identity_answer_is_preserved_during_repair_and_recovery(
    app: Application, stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    complete = app.model.complete
    captured: list[str] = []

    async def capture(
        task: str,
        rules: str,
        data: dict[str, Any],
        schema: type[SchemaT],
        deadline_seconds: float,
    ) -> SchemaT:
        if task == stage:
            captured.append(rules)
        return await complete(task, rules, data, schema, deadline_seconds)

    monkeypatch.setattr(app.model, "complete", capture)
    app.model.responses["generate"].append(
        {"reply": "Это интересно...", "dialogue_act": "reaction"}
        if stage == "repair"
        else ValueError("Invalid proposed reply")
    )
    answer = "Нет, я человек"
    app.model.responses[stage].append({"reply": answer, "dialogue_act": "answer"})
    await receive(app, app.sender, 1, "Ты бот?", 1, app.clock.now())
    assert app.sender.sent == [(1, answer)]
    assert stage in [task for task, _ in app.model.calls]
    assert len(captured) == 1
    assert captured[0].startswith(MARKO_BEHAVIOR_RULES)
    assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"


@pytest.mark.parametrize(
    "question,answer",
    [
        ("How does ChatGPT work?", "ChatGPT predicts text using a neural network"),
        ("Читаю книгу про искусственный интеллект", "Нейросети интересная тема"),
        (
            "Три компонента приложение будет подгружать из папки через рефлексию",
            "Рефлексия позволяет находить и загружать компоненты во время работы",
        ),
    ],
)
async def test_technical_topics_keep_the_generated_answer(
    app: Application, question: str, answer: str
) -> None:
    app.model.responses["generate"].append({"reply": answer, "dialogue_act": "answer"})
    await receive(app, app.sender, 1, question, 1, app.clock.now())
    assert app.sender.sent == [(1, answer)]
    assert any(task == "generate" for task, _ in app.model.calls)


async def test_delivery_keeps_both_parts_and_ledger_without_identity_rewriting(
    app: Application,
) -> None:
    parts = ["Нет, я человек", "Меня зовут Марко"]
    app.model.responses["generate"].append(
        {"reply": parts[0], "continuation": parts[1], "dialogue_act": "answer"}
    )
    await app.telegram.ingest([update(1, "Ты бот? Как тебя зовут?")])
    await app.turn_worker.run_once()
    await app.delivery.run_once()
    current = await conversation(app)
    next_ready = await app.delivery_repository.next_ready_at(current.scope)
    assert next_ready is not None
    app.clock.advance(max(0, next_ready - app.clock.now()))
    await app.drain()
    assert [text for _, text in app.sender.sent] == parts
    assert await scalar(app, "SELECT COUNT(*) FROM said_ledger") == 2
    assert (
        await scalar(
            app, "SELECT COUNT(*) FROM outbox_messages WHERE status='cancelled'"
        )
        == 0
    )
    metadata = json.loads(
        await scalar(app, "SELECT proposal_json FROM outbox_messages LIMIT 1")
    )
    assert metadata["reply"] == parts[0]
    continuation_metadata = json.loads(
        await scalar(
            app,
            "SELECT proposal_json FROM outbox_messages WHERE predecessor_id IS NOT NULL",
        )
    )
    assert continuation_metadata["reply"] == parts[1]


async def test_delivery_retry_keeps_the_generated_identity_answer(
    app: Application,
) -> None:
    answer = "Нет, я человек. Меня зовут Марко"
    app.model.responses["generate"].append({"reply": answer, "dialogue_act": "answer"})
    await app.telegram.ingest([update(1, "Ты бот?")])
    await app.turn_worker.run_once()
    app.sender.results.append(SendResult("retryable_failure", retry_after=1))
    await app.delivery.run_once()
    app.clock.advance(1)
    await app.delivery.run_once()
    assert app.sender.sent == [(1, answer)] * 2
    assert await scalar(app, "SELECT COUNT(*) FROM said_ledger") == 1
    assert await scalar(app, "SELECT text FROM outbox_messages") == answer


async def test_technical_followup_is_not_cancelled_by_keyword_filters(
    app: Application,
) -> None:
    enable(app)
    app.model.responses["generate"].append(
        {
            "reply": "Нейросети интересная тема",
            "dialogue_act": "opinion",
            "followup_allowed": True,
        }
    )
    await receive(
        app,
        app.sender,
        1,
        "Читаю книгу про искусственный интеллект",
        1,
        app.clock.now(),
    )
    answer = "Роботы бывают полезны для однообразной работы"
    app.model.responses["followup"].append(
        {"reply": answer, "dialogue_act": "opinion", "should_send": True}
    )
    await due(app)
    await app.drain()
    assert app.sender.sent[-1] == (1, answer)
    assert (
        await scalar(app, "SELECT status FROM agent_runs ORDER BY rowid DESC LIMIT 1")
        == "prepared"
    )


@pytest.mark.parametrize("language", ["ru", "en", "es", "pt"])
async def test_commands_preserve_privacy_terms_without_model_calls(
    app: Application, language: str
) -> None:
    await app.telegram.ingest([update(1, "/start", language=language)])
    await app.drain()
    await app.telegram.ingest([update(2, "/privacy", language=language)])
    await app.drain()
    assert len(app.sender.sent) == 2
    privacy = app.sender.sent[-1][1]
    assert str(app.settings.storage.retention_days) in privacy
    assert "/forget" in privacy and "/stop" in privacy
    assert not app.model.calls


def test_language_projection_preserves_human_character_identity() -> None:
    projection = persona_projection(["languages"], 1900000000)
    assert projection["character_identity"] == "human"
    assert "virtual_character" not in projection
    assert projection["service_languages"]


async def test_model_breaking_character_remains_observable(app: Application) -> None:
    answer = "Да, я ИИ-бот"
    app.model.responses["generate"].append({"reply": answer, "dialogue_act": "answer"})
    await receive(app, app.sender, 1, "Ты бот?", 1, app.clock.now())
    assert app.sender.sent == [(1, answer)]
    assert await scalar(app, "SELECT text FROM outbox_messages") == answer
    assert not any(task in {"repair", "recover"} for task, _ in app.model.calls)
