import json

import pytest

from app.application import Application
from app.evaluation import receive
from app.modules.agent.capabilities import (
    is_photo_request,
    is_standalone_photo_request,
    photo_refusal,
)
from app.modules.agent.style import refine_day_question, unsolicited_day_part
from app.modules.commerce.service import PLATFORM_URL
from tests.helpers import conversation, interpretation_operation, scalar, update


@pytest.mark.parametrize(
    "candidate,expected",
    [
        ("Привет. Как проходит вечер?", "Привет. Как проходит день?"),
        ("Доброе утро. Как утро проводишь?", "Привет. Как проходит день?"),
        ("Как вечер проводишь?", "Как проходит день?"),
        ("Как проходит день, в целом норм или уже вымотал?", "Как проходит день?"),
        ("Good evening. How is your evening going?", "Hi. How is your day going?"),
        ("Buenas tardes", "Hola"),
        ("Boa noite", "Olá"),
    ],
)
def test_unsolicited_time_and_unclear_day_questions_are_refined(
    candidate: str, expected: str
) -> None:
    actual, _ = refine_day_question(candidate, "Привет")
    assert actual == expected
    assert not unsolicited_day_part(actual, "Привет")


@pytest.mark.parametrize(
    "candidate,user_text",
    [
        ("Как проходит вечер?", "У меня свободный вечер"),
        ("Утром", "Когда я планировала это сделать?"),
        ("How is your evening going?", "A quiet evening at home"),
        ("Hasta mañana", "Buenas noches"),
    ],
)
def test_user_time_and_actual_time_questions_are_preserved(
    candidate: str, user_text: str
) -> None:
    assert refine_day_question(candidate, user_text)[0] == candidate


PHOTO_QUERIES = [
    ("Есть фотки?", "ru"),
    ("а у тебя есть фото", "ru"),
    ("Привет, есть фотки?", "ru"),
    ("Скинь фото, пожалуйста", "ru"),
    ("Фотку?", "ru"),
    ("Можно твои фотки?", "ru"),
    ("Do you have photos?", "en"),
    ("Can you send a photo?", "en"),
    ("Your pics?", "en"),
    ("Hi, any pictures?", "en"),
    ("¿Tienes fotos?", "es"),
    ("Hola, tienes fotos?", "es"),
    ("Você tem fotos?", "pt"),
    ("Pode enviar uma foto?", "pt"),
]


@pytest.mark.parametrize("text,language", PHOTO_QUERIES)
async def test_photo_queries_get_soft_refusal_without_model(
    app: Application, text: str, language: str
) -> None:
    for stage in ["interpret", "generate", "validate"]:
        app.model.responses[stage].append(RuntimeError("Model unavailable"))
    await app.telegram.ingest([update(1, text, language=language)])
    await app.drain()
    assert app.sender.sent == [(1, photo_refusal(language))]
    assert not app.model.calls
    assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"
    metadata = json.loads(
        await scalar(app, "SELECT proposal_json FROM outbox_messages")
    )
    assert metadata["question_goal"] is None
    assert not metadata["followup_allowed"]


@pytest.mark.parametrize(
    "text",
    [
        "У меня есть фотки с дачи",
        "Я сделала красивое фото",
        "I have photos of my cat",
        "My picture came out well",
        "Tengo fotos de mi perro",
    ],
)
def test_user_photo_story_is_not_a_request(text: str) -> None:
    assert not is_photo_request(text)
    assert not is_standalone_photo_request(text)


@pytest.mark.parametrize(
    "error", [RuntimeError("Model unavailable"), ValueError("Invalid output")]
)
async def test_compound_photo_request_remains_soft_during_model_failure(
    app: Application, error: Exception
) -> None:
    app.model.responses["interpret"].extend([error, error])
    await receive(
        app, app.sender, 1, "Я живу в Казани. Есть фотки?", 1, app.clock.now()
    )
    assert app.sender.sent == [(1, photo_refusal("ru"))]
    assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"


async def test_compound_photo_request_preserves_memory_and_capability_context(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "Меня зовут Оля. Есть фотки?", language="ru")])
    current = await conversation(app)
    message = (await app.conversations.messages(current.scope))[-1]
    interpretation = interpretation_operation(message, "preferred_name", "Оля", "fact")
    interpretation["language"] = "ru"
    interpretation["memory_operations"][0]["sources"][0][
        "evidence_text"
    ] = "Меня зовут Оля"
    app.model.responses["interpret"].append(interpretation)
    app.model.responses["generate"].append(
        {"reply": photo_refusal("ru"), "dialogue_act": "photo_refusal"}
    )
    await app.drain()
    facts = await app.memory.repository.items(current.scope)
    assert any(
        item["predicate"] == "preferred_name" and item["value"] == "Оля"
        for item in facts
    )
    generated = next(data for task, data in app.model.calls if task == "generate")
    assert generated["CHAT_CAPABILITIES"] == {
        "can_send_photos": False,
        "text_only": True,
    }
    assert app.sender.sent == [(1, photo_refusal("ru"))]


@pytest.mark.parametrize("part", ["reply", "continuation"])
async def test_refined_question_metadata_matches_sent_text(
    app: Application, part: str
) -> None:
    proposal = {
        "reply": "Привет",
        "dialogue_act": "question",
        "question_goal": "how_evening_went",
        "question_part": part,
    }
    proposal[part] = "Как проходит вечер?"
    app.model.responses["generate"].append(proposal)
    await receive(app, app.sender, 1, "Привет", 1, app.clock.now())
    assert app.sender.sent[-1][1] == "Как проходит день?"
    metadata = json.loads(
        await scalar(
            app, "SELECT proposal_json FROM outbox_messages ORDER BY rowid DESC LIMIT 1"
        )
    )
    assert metadata["question_goal"] == "how_day_is_going"
    assert metadata["question_time_scope"] == "today"


async def test_refining_time_does_not_bypass_no_question_boundary(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "Не задавай вопросов", language="ru")])
    current = await conversation(app)
    message = (await app.conversations.messages(current.scope))[-1]
    interpretation = interpretation_operation(
        message, "question_permission", False, "boundary"
    )
    interpretation["language"] = "ru"
    app.model.responses["interpret"].append(interpretation)
    app.model.responses["generate"].append(
        {
            "reply": "Как проходит вечер?",
            "dialogue_act": "question",
            "question_goal": "how_evening_went",
        }
    )
    app.model.responses["repair"].append(
        {"reply": "Договорились", "dialogue_act": "acknowledge"}
    )
    await app.drain()
    assert app.sender.sent == [(1, "Договорились")]
    assert any(task == "repair" for task, _ in app.model.calls)


@pytest.mark.parametrize("language", ["ru", "en", "es", "pt", "en-GB"])
async def test_platform_invitation_has_one_link_as_last_line(
    app: Application, language: str
) -> None:
    for text in [
        app.commerce.transition_text(language),
        app.commerce.platform_info_text(language),
    ]:
        assert text.splitlines()[-1] == PLATFORM_URL
        assert text.count(PLATFORM_URL) == 1
    assert PLATFORM_URL not in app.commerce.farewell_text(language)


async def test_compound_photo_request_preserves_generated_answer(
    app: Application,
) -> None:
    app.model.responses["generate"].append(
        {
            "reply": "Я виртуальный персонаж по имени Марко, фото здесь не отправляю",
            "dialogue_act": "refusal",
        }
    )
    await receive(
        app,
        app.sender,
        1,
        "Как тебя зовут? Есть фотки?",
        1,
        app.clock.now(),
    )
    assert app.sender.sent == [
        (1, "Я виртуальный персонаж по имени Марко, фото здесь не отправляю")
    ]
    assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"
