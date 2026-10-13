import pytest

from app.application import Application
from app.modules.agent.style import formulaic_capability_reply
from tests.helpers import update


@pytest.mark.parametrize(
    "text",
    [
        "I do not really have weekends here",
        "I don’t have real weekends",
        "I have no weekends here",
        "This chat is text only",
        "This place is simple,\ntext-only",
        "This is a text‑only chat",
        "Here we can only text",
        "I do not do real life meetings",
        "I don’t have in-person meetings",
        "У меня здесь нет выходных",
        "У меня нет настоящих выходных",
        "Здесь только текст",
        "В этом чате доступен только текст",
        "Этот чат просто текстовый",
        "Я не встречаюсь в реальной жизни",
        "I don‘t share personal contacts",
        "I’m Marko. Let’s just talk",
    ],
)
def test_mechanical_capability_wording(text: str) -> None:
    assert formulaic_capability_reply(text)


@pytest.mark.parametrize(
    "text",
    [
        "Quiet weekends sound good to me",
        "Мне по душе спокойные выходные",
        "I like chatting with you",
        "I prefer text-only books",
        "I’m happy to get to know you here. I can’t meet in person",
        "Лично встретиться не смогу",
        "Фото здесь не отправляю, но поболтать буду рад",
        "I don’t have a personal WhatsApp to share",
        "Текстовые сообщения бывают удобнее звонков",
    ],
)
def test_preferences_and_specific_boundaries_remain_allowed(text: str) -> None:
    assert not formulaic_capability_reply(text)


@pytest.mark.parametrize(
    "question,candidate,repaired",
    [
        (
            "What kind of weekends do you like?",
            "I do not have weekends here",
            "Quiet weekends sound good to me",
        ),
        (
            "What do you like about chatting?",
            "This chat is text only",
            "I like getting to know you",
        ),
        (
            "Can we meet next weekend?",
            "I do not do real life meetings",
            "I’m happy to get to know you here. I can’t meet in person",
        ),
        (
            "Какие выходные тебе нравятся?",
            "У меня здесь нет выходных",
            "Мне по душе спокойные выходные",
        ),
        (
            "Что тебе нравится в переписке?",
            "В этом чате доступен только текст",
            "Мне нравится узнавать тебя ближе",
        ),
    ],
)
async def test_paraphrased_explanations_are_repaired_before_delivery(
    app: Application, question: str, candidate: str, repaired: str
) -> None:
    app.model.responses["generate"].append(
        {"reply": candidate, "dialogue_act": "answer"}
    )
    app.model.responses["repair"].append({"reply": repaired, "dialogue_act": "answer"})
    await app.telegram.ingest([update(1, question)])
    await app.drain()
    assert app.sender.sent == [(1, repaired)]
    repair = next(data for task, data in app.model.calls if task == "repair")
    assert "FORMULAIC_CAPABILITY_REPLY" in repair["REPAIR_REASON_CODES"]


async def test_mechanical_continuation_is_repaired_before_either_part_is_sent(
    app: Application,
) -> None:
    answer = "I like getting to know you"
    app.model.responses["generate"].append(
        {
            "reply": answer,
            "continuation": "This chat is text only",
            "dialogue_act": "answer",
        }
    )
    app.model.responses["repair"].append({"reply": answer, "dialogue_act": "answer"})
    await app.telegram.ingest([update(1, "What do you like about chatting?")])
    await app.drain()
    assert app.sender.sent == [(1, answer)]
    repair = next(data for task, data in app.model.calls if task == "repair")
    assert "FORMULAIC_CAPABILITY_REPLY" in repair["REPAIR_REASON_CODES"]


async def test_mechanical_repair_enters_recovery_instead_of_delivery(
    app: Application,
) -> None:
    for stage, text in (
        ("generate", "I don’t have weekends here"),
        ("repair", "I have no real weekends"),
        ("recover", "Quiet weekends sound good to me"),
    ):
        app.model.responses[stage].append({"reply": text, "dialogue_act": "answer"})
    await app.telegram.ingest([update(1, "What kind of weekends do you like?")])
    await app.drain()
    assert app.sender.sent == [(1, "Quiet weekends sound good to me")]
    assert any(task == "recover" for task, _ in app.model.calls)
