import json
from datetime import datetime

import pytest

from app.application import Application
from app.modules.agent.language import reply_language, wrong_reply_language
from app.modules.agent.time import current_time_facts, standalone_time_reply
from tests.helpers import conversation, scalar, update
from tests.test_followups import due, enable


@pytest.mark.parametrize(
    "utc,expected,offset",
    [
        ("2026-01-15T12:00:00+00:00", "13:00", "+0100"),
        ("2026-07-15T12:00:00+00:00", "14:00", "+0200"),
        ("2026-03-29T00:59:00+00:00", "01:59", "+0100"),
        ("2026-03-29T01:00:00+00:00", "03:00", "+0200"),
        ("2026-10-25T00:59:00+00:00", "02:59", "+0200"),
        ("2026-10-25T01:00:00+00:00", "02:00", "+0100"),
    ],
)
def test_split_time_uses_iana_daylight_saving(
    utc: str, expected: str, offset: str
) -> None:
    facts = current_time_facts(
        "What time is it now in Split?", datetime.fromisoformat(utc).timestamp()
    )
    assert facts[0]["local_time"] == expected
    assert facts[0]["utc_offset"] == offset


@pytest.mark.parametrize(
    "text",
    [
        "what time is it in Split?",
        "What time is it now in Split?",
        "Который час в Сплите?",
        "Сколько сейчас времени в Сплите?",
        "Время в Сплите?",
    ],
)
async def test_seven_repeated_time_requests_never_need_model(
    app: Application, text: str
) -> None:
    language = "ru" if "Сплите" in text else "en"
    app.model.responses["interpret"].append(RuntimeError("gateway unavailable"))
    for number in range(1, 8):
        await app.telegram.ingest([update(number, text)])
        await app.drain()
        assert app.sender.sent[-1] == (
            1,
            standalone_time_reply(text, app.clock.now(), language),
        )
        app.clock.advance(60)
    assert len(app.sender.sent) == 7
    assert all(task == "validate_summary" for task, _ in app.model.calls)
    assert (
        await scalar(
            app, "SELECT sum(json_extract(metrics_json, '$.llm_calls')) FROM agent_runs"
        )
        == 0
    )
    assert (
        await scalar(app, "SELECT count(*) FROM agent_runs WHERE status='fallback'")
        == 0
    )


async def test_compound_time_question_keeps_both_parts_and_public_city(
    app: Application,
) -> None:
    text = "haha ok professor. what time is it now in Split and what are you doing?"
    local_time = current_time_facts(text, app.clock.now())[0]["local_time"]
    answer = f"It’s {local_time} in Split. I’m reading your messages and replying"
    app.model.responses["generate"].append({"reply": answer, "dialogue_act": "answer"})
    await app.telegram.ingest([update(1, text)])
    await app.drain()
    assert app.sender.sent == [(1, answer)]
    data = next(data for task, data in app.model.calls if task == "generate")
    assert data["CURRENT_INPUT"][0]["text"] == text
    assert data["CURRENT_TIME_FACTS"][0]["local_time"] == local_time
    assert "residence" not in data["PERSONA_PROJECTION"]
    assert not any(task == "repair" for task, _ in app.model.calls)


async def test_unsolicited_persona_location_still_requires_repair(
    app: Application,
) -> None:
    app.model.responses["generate"].append(
        {"reply": "I live in Split", "dialogue_act": "answer"}
    )
    app.model.responses["repair"].append(
        {"reply": "Hi, good to hear from you", "dialogue_act": "greeting"}
    )
    await app.telegram.ingest([update(1, "Hi Marko")])
    await app.drain()
    data = next(data for task, data in app.model.calls if task == "repair")
    assert "LOCATION_DISCLOSURE" in data["REPAIR_REASON_CODES"]


@pytest.mark.parametrize(
    "text",
    [
        "What time was it in Split yesterday?",
        "What time is it in London?",
        "What time is it in Split? And where is my link?",
        "She asked what time is it in Split",
    ],
)
def test_clock_shortcut_does_not_swallow_other_questions(text: str) -> None:
    assert standalone_time_reply(text, 1900000000, "en") is None


@pytest.mark.parametrize(
    "text,prior,interpretation,expected",
    [
        ("im back. today was a hard shift at the hospital", "ru", "ru", "en"),
        ("Why did you reply in Russian?", "en", "ru", "en"),
        ("English is fine", "ru", "ru", "en"),
        ("no, english is ok. random question, why is the sky blue?", "ru", "ru", "en"),
        ("Please reply in Russian", "en", "en", "ru"),
        ("ok", "ru", "en", "ru"),
        ("🙂", "en", "ru", "en"),
        ("Привет, мне так легче", "en", "en", "ru"),
        ("Hola, ¿cómo estás?", "en", "es", "es"),
    ],
)
def test_language_follows_current_user_not_old_service_messages(
    text: str, prior: str, interpretation: str, expected: str
) -> None:
    assert reply_language(text, prior, interpretation) == expected


@pytest.mark.parametrize(
    "text,language,wrong",
    [
        ("После смены хочется просто тишины", "en", True),
        ("Quiet music sounds good", "ru", True),
        ("You wrote “Мне нравится эта песня”", "en", False),
        ("I like Чайковский too", "en", False),
        ("Мне нравится Radiohead", "ru", False),
    ],
)
def test_language_guard_allows_quotes_and_names(
    text: str, language: str, wrong: bool
) -> None:
    assert wrong_reply_language(text, language) == wrong


@pytest.mark.parametrize("repair_language", ["en", "ru"])
async def test_english_followup_is_repaired_or_silently_skipped(
    app: Application, repair_language: str
) -> None:
    enable(app)
    app.model.responses["generate"].append(
        {
            "reply": "Quiet music sounds good",
            "dialogue_act": "opinion",
            "followup_allowed": True,
        }
    )
    await app.telegram.ingest([update(1, "I like listening to music")])
    await app.drain()
    app.model.responses["followup"].append(
        {
            "reply": "Мне нравится тихая музыка без слов",
            "dialogue_act": "opinion",
            "should_send": True,
        }
    )
    repaired = (
        "Acoustic guitar has a lovely sound"
        if repair_language == "en"
        else "Мне больше нравится музыка без слов"
    )
    app.model.responses["repair"].append(
        {"reply": repaired, "dialogue_act": "opinion", "should_send": True}
    )
    await due(app)
    await app.drain()
    data = next(data for task, data in app.model.calls if task == "followup")
    assert data["APPLICATION_STATE"]["language"] == "en"
    repair = next(data for task, data in app.model.calls if task == "repair")
    assert "WRONG_LANGUAGE" in repair["REPAIR_REASON_CODES"]
    assert len(app.sender.sent) == (2 if repair_language == "en" else 1)
    assert not any(wrong_reply_language(text, "en") for _, text in app.sender.sent)
    assert (await conversation(app)).language == "en"


async def test_wrong_interpretation_and_repair_cannot_change_english(
    app: Application,
) -> None:
    app.model.responses["interpret"].append(
        {"intent": "chat", "dialogue_mode": "continuation", "language": "ru"}
    )
    app.model.responses["generate"].append(
        {"reply": "Теперь буду отвечать на русском", "dialogue_act": "answer"}
    )
    app.model.responses["repair"].append(
        {"reply": "Ты раньше писала на русском языке", "dialogue_act": "answer"}
    )
    app.model.responses["recover"].append(
        {"reply": "Sorry, I switched languages by mistake", "dialogue_act": "repair"}
    )
    await app.telegram.ingest([update(1, "Why did you reply in Russian?")])
    await app.drain()
    assert app.sender.sent == [(1, "Sorry, I switched languages by mistake")]
    assert (await conversation(app)).language == "en"
    trace = json.loads(await scalar(app, "SELECT metrics_json FROM agent_runs"))
    assert any(
        "WRONG_LANGUAGE" in event.get("errors", [])
        for event in trace["validation_events"]
    )


@pytest.mark.parametrize(
    "candidate,repaired,question",
    [
        (
            "I don't really have weekends here",
            "Quiet weekends sound good to me",
            "What kind of weekends do you like?",
        ),
        (
            "This place is simple, text-only",
            "I’m happy to chat",
            "What do you like about chatting?",
        ),
        (
            "I don't do real-life meetings",
            "I can chat with you here, but I can’t meet in person",
            "Can we meet next weekend?",
        ),
    ],
)
async def test_awkward_capability_wording_is_repaired_without_false_promises(
    app: Application, candidate: str, repaired: str, question: str
) -> None:
    app.model.responses["generate"].append(
        {"reply": candidate, "dialogue_act": "answer"}
    )
    app.model.responses["repair"].append({"reply": repaired, "dialogue_act": "answer"})
    await app.telegram.ingest([update(1, question)])
    await app.drain()
    assert app.sender.sent == [(1, repaired)]
    data = next(data for task, data in app.model.calls if task == "repair")
    assert "FORMULAIC_CAPABILITY_REPLY" in data["REPAIR_REASON_CODES"]
