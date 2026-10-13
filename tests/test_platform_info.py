import json

import pytest

from app.application import Application
from app.modules.commerce.requests import requests_platform_info
from app.modules.commerce.service import PLATFORM_URL
from tests.helpers import conversation, scalar, update
from tests.test_platform_transition import enable_preview

PLATFORM_QUESTIONS = [
    ("Как мне пообщаться с тобой на платформе", "ru"),
    ("Как продолжить общение на платформе?", "ru"),
    ("Где можно продолжить переписку на сайте?", "ru"),
    ("Дай ссылку на платформу", "ru"),
    ("Пришли ссылку на BeHappy2Day", "ru"),
    ("Привет, как перейти на платформу?", "ru"),
    ("Как перейти из Telegram на BeHappy2Day?", "ru"),
    ("Как тебя найти на сайте?", "ru"),
    ("Как перейти на сайт?", "ru"),
    ("На какой платформе можно с тобой разговаривать?", "ru"),
    ("How can I chat with you on the platform?", "en"),
    ("Send me the BeHappy2Day link", "en"),
    ("¿Cómo puedo hablar contigo en la plataforma?", "es"),
    ("Como posso conversar com você na plataforma?", "pt"),
]


@pytest.mark.parametrize("text,language", PLATFORM_QUESTIONS)
async def test_first_platform_question_gets_server_link_without_model(
    app: Application, text: str, language: str
) -> None:
    enable_preview(app)
    app.model.responses["interpret"].append(RuntimeError("Model unavailable"))
    await app.telegram.ingest([update(1, text, language=language)])
    await app.drain()
    assert app.sender.sent == [(1, app.commerce.platform_info_text(language))]
    assert not app.model.calls
    current = await conversation(app)
    assert current.language == language
    assert current.preview_reply_count == 0
    assert current.preview_closed_at is None
    assert app.commerce.can_reply(current)
    assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"
    metadata = json.loads(
        await scalar(app, "SELECT proposal_json FROM outbox_messages")
    )
    assert metadata["origin"] == "platform_info"
    assert metadata["question_goal"] is None
    assert not metadata["followup_allowed"]
    assert app.sender.sent[0][1].splitlines()[-1] == PLATFORM_URL


@pytest.mark.parametrize(
    "text",
    [
        "Я сегодня была на платформе BeHappy2Day",
        "Как общаться в Telegram?",
        "How can I chat on the Zoom platform?",
        "Как сделать платформу для моего проекта?",
        "How can I build a website?",
        "Не присылай ссылку на платформу",
        "Дай ссылку на рецепт",
        "Ты откуда?",
    ],
)
def test_unrelated_stories_and_requests_do_not_trigger_platform_link(text: str) -> None:
    assert not requests_platform_info(text)


async def test_direct_info_at_last_reply_does_not_close_preview_or_count(
    app: Application,
) -> None:
    enable_preview(app)
    await app.telegram.ingest([update(1, "/start")])
    await app.drain()

    async def seed(tx):
        await tx.execute("UPDATE conversations SET preview_reply_count=19")

    await app.database.write(seed)
    await app.telegram.ingest([update(2, "Как перейти на платформу?", language="ru")])
    await app.drain()
    current = await conversation(app)
    assert app.sender.sent[-1][1] == app.commerce.platform_info_text("ru")
    assert current.preview_reply_count == 19
    assert current.preview_closed_at is None
    assert not app.model.calls


async def test_closed_preview_can_answer_explicit_platform_info_request(
    app: Application,
) -> None:
    enable_preview(app)
    await app.telegram.ingest([update(1, "/start")])
    await app.drain()

    async def seed(tx):
        await tx.execute(
            "UPDATE conversations SET preview_closed_at=?,preview_reply_count=20",
            (app.clock.now(),),
        )

    await app.database.write(seed)
    await app.telegram.ingest(
        [update(2, "Как мне пообщаться с тобой на платформе", language="ru")]
    )
    await app.drain()
    current = await conversation(app)
    assert app.sender.sent[-1][1] == app.commerce.platform_info_text("ru")
    assert current.preview_closed_at is not None
    assert not app.commerce.can_reply(current)
    assert not app.model.calls


@pytest.mark.parametrize("control", ["stopped", "paused", "handoff_only"])
async def test_platform_question_does_not_override_stopped_or_paused_chat(
    app: Application, control: str
) -> None:
    await app.telegram.ingest([update(1, "/start")])
    await app.drain()

    async def stop(tx):
        await app.conversations.set_control(
            tx, (await conversation(app)).scope, control
        )

    await app.database.write(stop)
    offset = len(app.sender.sent)
    await app.telegram.ingest([update(2, "Как перейти на платформу?", language="ru")])
    await app.drain()
    assert len(app.sender.sent) == offset
    assert not app.model.calls


async def test_platform_request_respects_kill_switch(app: Application) -> None:
    await app.delivery_repository.kill_switch(True)
    await app.telegram.ingest([update(1, "Как перейти на платформу?", language="ru")])
    await app.drain()
    assert not app.sender.sent
    assert not app.model.calls


@pytest.mark.parametrize("followup", ["Дай ссылку", "А где ссылка?", "Ссылку?"])
async def test_link_again_requires_a_delivered_platform_reply_and_duplicates_are_ignored(
    app: Application,
    followup: str,
) -> None:
    first = update(1, "Как перейти на платформу?", language="ru")
    await app.telegram.ingest([first, first])
    await app.drain()
    assert len(app.sender.sent) == 1
    await app.telegram.ingest([update(2, followup, language="ru")])
    await app.drain()
    assert len(app.sender.sent) == 2
    assert all(
        text == app.commerce.platform_info_text("ru") for _, text in app.sender.sent
    )
    assert not app.model.calls
    assert await scalar(app, "SELECT COUNT(*) FROM said_ledger") == 2


async def test_link_request_without_platform_context_keeps_normal_generation(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "Дай ссылку", language="ru")])
    await app.drain()
    assert any(task == "generate" for task, _ in app.model.calls)
    assert all(PLATFORM_URL not in text for _, text in app.sender.sent)


async def test_forwarded_platform_question_does_not_authorize_server_reply(
    app: Application,
) -> None:
    incoming = update(1, "Как перейти на платформу?", language="ru")
    incoming["message"]["forward_origin"] = {
        "type": "hidden_user",
        "date": 1900000000,
        "sender_user_name": "someone",
    }
    await app.telegram.ingest([incoming])
    await app.drain()
    assert any(task == "generate" for task, _ in app.model.calls)
    assert all(PLATFORM_URL not in text for _, text in app.sender.sent)


async def test_explicit_request_does_not_enable_ordinary_early_advertising(
    app: Application,
) -> None:
    enable_preview(app)
    await app.telegram.ingest([update(1, "Как перейти на платформу?", language="ru")])
    await app.drain()
    app.model.responses["generate"].append(
        {"reply": "Перейди на BeHappy2Day", "dialogue_act": "chat"}
    )
    app.model.responses["repair"].append(
        {
            "reply": "Как проходит день?",
            "dialogue_act": "question",
            "question_goal": "how_day_is_going",
        }
    )
    await app.telegram.ingest([update(2, "Давай пока поболтаем тут", language="ru")])
    await app.drain()
    assert app.sender.sent[-1][1] == "Как проходит день?"
    assert any(task == "repair" for task, _ in app.model.calls)


async def test_coalesced_link_cancellation_overrides_earlier_request(
    app: Application,
) -> None:
    await app.telegram.ingest(
        [
            update(1, "Как перейти на платформу?", language="ru"),
            update(2, "Не присылай ссылку", language="ru"),
        ]
    )
    await app.drain()
    assert all(PLATFORM_URL not in text for _, text in app.sender.sent)
    assert any(task == "generate" for task, _ in app.model.calls)
