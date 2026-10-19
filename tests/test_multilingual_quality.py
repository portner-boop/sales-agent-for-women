from datetime import datetime

import pytest

from app.application import Application
from app.modules.agent.language import reply_language
from app.modules.agent.schemas import CurrentTimeRequest
from app.modules.agent.time import resolve_current_times
from app.modules.commerce.service import PLATFORM_URL
from tests.helpers import conversation, scalar, update
from tests.test_followups import due, enable
from tests.test_platform_transition import finish_delivery, seed_count


@pytest.mark.parametrize(
    "language,question,candidate,answer,reason",
    [
        (
            "fr",
            "Tu me donnes ton WhatsApp ?",
            "Aucun contact autorisé",
            "Je préfère discuter ici avec toi",
            "FORMULAIC_CAPABILITY_REPLY",
        ),
        (
            "de",
            "Wie kann ich dir vertrauen?",
            "Beweise gibt es nicht",
            "Ich verstehe deine Zweifel",
            "HOSTILE_VERIFICATION_BOUNDARY",
        ),
        (
            "ja",
            "空はどうして青いの？",
            "選択的散乱の波長依存性です",
            "太陽の光のうち、青い光が空気で広がりやすいからだよ",
            "OVEREXPLAINED_CASUAL_KNOWLEDGE",
        ),
        (
            "ar",
            "ماذا تتذكر عني؟",
            "ملفك الشخصي: تحبين الموسيقى الهادئة",
            "أتذكر أنك تحبين الموسيقى الهادئة",
            "MEMORY_PROFILE_READOUT",
        ),
    ],
)
async def test_semantic_quality_is_checked_in_every_language(
    app: Application,
    language: str,
    question: str,
    candidate: str,
    answer: str,
    reason: str,
) -> None:
    app.model.responses["interpret"].append(
        {"intent": "chat", "dialogue_mode": "continuation", "language": language}
    )
    app.model.responses["generate"].append(
        {"reply": candidate, "dialogue_act": "answer"}
    )
    app.model.responses["validate"].append(
        {"decision": "reject", "reason_codes": [reason]}
    )
    app.model.responses["repair"].append({"reply": answer, "dialogue_act": "answer"})
    await app.telegram.ingest([update(1, question)])
    await app.drain()
    assert app.sender.sent == [(1, answer)]
    assert [task for task, _ in app.model.calls] == [
        "interpret",
        "generate",
        "validate",
        "repair",
        "validate_repair",
    ]
    data = next(data for task, data in app.model.calls if task == "repair")
    assert data["APPLICATION_STATE"]["language"] == language
    assert data["REPAIR_REASON_CODES"] == [reason]


@pytest.mark.parametrize(
    "language,greeting",
    [("fr", "Bonjour"), ("de", "Hallo"), ("ja", "こんにちは"), ("ar", "مرحبا")],
)
async def test_even_one_word_requires_semantic_validation(
    app: Application, language: str, greeting: str
) -> None:
    app.model.responses["interpret"].append(
        {"intent": "chat", "dialogue_mode": "opening", "language": language}
    )
    app.model.responses["generate"].append(
        {"reply": greeting, "dialogue_act": "greeting"}
    )
    await app.telegram.ingest([update(1, greeting)])
    await app.drain()
    assert app.sender.sent == [(1, greeting)]
    assert [task for task, _ in app.model.calls] == [
        "interpret",
        "generate",
        "validate",
    ]


async def test_short_foreign_followup_is_repaired_in_conversation_language(
    app: Application,
) -> None:
    enable(app)
    app.model.responses["interpret"].append(
        {"intent": "chat", "dialogue_mode": "continuation", "language": "ja"}
    )
    app.model.responses["generate"].append(
        {
            "reply": "静かな音楽がいいね",
            "dialogue_act": "opinion",
            "followup_allowed": True,
        }
    )
    await app.telegram.ingest([update(1, "静かな音楽が好き")])
    await app.drain()
    app.model.responses["followup"].append(
        {
            "reply": "Ti piace la chitarra?",
            "dialogue_act": "question",
            "question_goal": "guitar_interest",
            "should_send": True,
        }
    )
    app.model.responses["validate"].append(
        {"decision": "reject", "reason_codes": ["WRONG_LANGUAGE"]}
    )
    app.model.responses["repair"].append(
        {
            "reply": "ギターの音も好きですか？",
            "dialogue_act": "question",
            "question_goal": "guitar_interest",
            "should_send": True,
        }
    )
    await due(app)
    await app.drain()
    assert app.sender.sent[-1] == (1, "ギターの音も好きですか？")
    assert (await conversation(app)).language == "ja"
    assert [task for task, _ in app.model.calls][-4:] == [
        "followup",
        "validate",
        "repair",
        "validate_repair",
    ]


@pytest.mark.parametrize(
    "language,question,location,timezone,local_time",
    [
        ("fr", "Quelle heure est-il à Paris ?", "Paris", "Europe/Paris", "13:00"),
        ("de", "Wie spät ist es in Berlin?", "Berlin", "Europe/Berlin", "13:00"),
        ("ja", "東京は今何時？", "東京", "Asia/Tokyo", "21:00"),
        ("ar", "كم الساعة الآن في دبي؟", "دبي", "Asia/Dubai", "16:00"),
    ],
)
async def test_structured_time_uses_server_clock_in_any_language(
    app: Application,
    language: str,
    question: str,
    location: str,
    timezone: str,
    local_time: str,
) -> None:
    timestamp = datetime.fromisoformat("2026-01-15T12:00:00+00:00").timestamp()
    app.clock.advance(timestamp - app.clock.now())
    await app.telegram.ingest([update(1, question)])
    current = await conversation(app)
    message = (await app.conversations.messages(current.scope))[-1]
    app.model.responses["interpret"].append(
        {
            "intent": "chat",
            "dialogue_mode": "continuation",
            "language": language,
            "current_time_requests": [
                {
                    "location": location,
                    "timezone": timezone,
                    "source_message_id": message.id,
                }
            ],
        }
    )
    app.model.responses["generate"].append(
        {"reply": local_time, "dialogue_act": "answer"}
    )
    await app.drain()
    assert app.sender.sent == [(1, local_time)]
    data = next(data for task, data in app.model.calls if task == "generate")
    assert data["CURRENT_TIME_FACTS"] == [
        {
            "precision": "approximate",
            "city": location,
            "timezone": timezone,
            "local_time": local_time,
            "local_date": "2026-01-15",
            "utc_offset": {"fr": "+0100", "de": "+0100", "ja": "+0900", "ar": "+0400"}[
                language
            ],
        }
    ]
    assert data["UNRESOLVED_TIME_REQUESTS"] == []
    assert await scalar(app, "SELECT count(*) FROM memory_items") == 0


@pytest.mark.parametrize(
    "location,timezone,source_valid,reason",
    [
        ("Paris", None, True, "ambiguous_timezone"),
        ("Paris", "Mars/City", True, "unknown_timezone"),
        ("Paris", "Europe/Paris", False, "unverified_location"),
        ("London", "Europe/London", True, "unverified_location"),
    ],
)
async def test_unverified_time_requests_never_supply_clock_facts(
    app: Application,
    location: str,
    timezone: str | None,
    source_valid: bool,
    reason: str,
) -> None:
    await app.telegram.ingest([update(1, "Quelle heure est-il à Paris ?")])
    current = await conversation(app)
    messages = await app.conversations.messages(current.scope)
    request = CurrentTimeRequest(
        location=location,
        timezone=timezone,
        source_message_id=messages[-1].id if source_valid else "unknown",
    )
    facts, unresolved = resolve_current_times([request], messages, app.clock.now())
    assert facts == []
    assert unresolved == [{"location": location, "reason": reason}]


@pytest.mark.parametrize(
    "language,text",
    [
        ("uk", "Я люблю тиху музику"),
        ("bg", "Харесвам тиха музика"),
        ("sr", "Волим тиху музику"),
        ("de", "Das ist mir wichtig"),
        ("fr", "Please answer in French"),
        ("ja", "日本語で話して"),
    ],
)
def test_interpreted_language_has_priority_over_alphabet_heuristics(
    language: str, text: str
) -> None:
    assert reply_language(text, "en", language) == language


@pytest.mark.parametrize(
    "language,question,answer,notice",
    [
        (
            "fr",
            "Quoi de neuf ?",
            "Je suis content de discuter avec toi",
            "Si tu veux continuer, tu peux acheter du temps de discussion sur BeHappy2Day",
        ),
        (
            "de",
            "Was gibt es Neues?",
            "Ich freue mich über unser Gespräch",
            "Wenn du möchtest, kannst du auf BeHappy2Day Chatzeit kaufen",
        ),
        (
            "ja",
            "最近どう？",
            "君と話せてうれしいよ",
            "続けたければ、BeHappy2Dayでチャット時間を購入できるよ",
        ),
        (
            "ar",
            "ما الجديد لديك؟",
            "يسعدني الحديث معك",
            "إذا أردت متابعة الحديث، يمكنك شراء وقت للدردشة على BeHappy2Day",
        ),
    ],
)
async def test_transition_keeps_answer_and_invitation_in_any_language(
    app: Application, language: str, question: str, answer: str, notice: str
) -> None:
    await seed_count(app, 19)
    app.model.responses["interpret"].append(
        {"intent": "chat", "dialogue_mode": "continuation", "language": language}
    )
    app.model.responses["generate"].append(
        {"reply": answer, "dialogue_act": "answer", "platform_notice": notice}
    )
    await app.telegram.ingest([update(2, question)])
    await app.drain()
    await finish_delivery(app)
    assert app.sender.sent[-2:] == [(1, answer), (1, f"{notice}\n{PLATFORM_URL}")]
    checked = next(data for task, data in app.model.calls if task == "validate")
    assert checked["reply"]["platform_notice"] == notice


@pytest.mark.parametrize(
    "candidate,reason",
    [
        (None, "MISSING_PLATFORM_NOTICE"),
        ("https://unapproved.example", "PLATFORM_NOTICE_URL"),
        ("Continue chatting", "WRONG_LANGUAGE"),
    ],
)
async def test_missing_or_invalid_invitation_requires_repair(
    app: Application, candidate: str | None, reason: str
) -> None:
    await seed_count(app, 19)
    answer = "Je suis content de discuter avec toi"
    notice = (
        "Si tu veux continuer, tu peux acheter du temps de discussion sur BeHappy2Day"
    )
    app.model.responses["interpret"].append(
        {"intent": "chat", "dialogue_mode": "continuation", "language": "fr"}
    )
    app.model.responses["generate"].append(
        {"reply": answer, "dialogue_act": "answer", "platform_notice": candidate}
    )
    if reason == "WRONG_LANGUAGE":
        app.model.responses["validate"].append(
            {"decision": "reject", "reason_codes": [reason]}
        )
    app.model.responses["repair"].append(
        {"reply": answer, "dialogue_act": "answer", "platform_notice": notice}
    )
    await app.telegram.ingest([update(2, "Quoi de neuf ?")])
    await app.drain()
    await finish_delivery(app)
    assert app.sender.sent[-2:] == [(1, answer), (1, f"{notice}\n{PLATFORM_URL}")]
    repaired = next(data for task, data in app.model.calls if task == "repair")
    assert reason in repaired["REPAIR_REASON_CODES"]


async def test_seven_repeated_time_requests_allow_same_accurate_answer(
    app: Application,
) -> None:
    timestamp = datetime.fromisoformat("2026-01-15T12:00:00+00:00").timestamp()
    app.clock.advance(timestamp - app.clock.now())
    answer = "In Berlin ist es jetzt 13:00 Uhr"
    for number in range(1, 8):
        await app.telegram.ingest([update(number, "Wie spät ist es jetzt in Berlin?")])
        current = await conversation(app)
        message = (await app.conversations.messages(current.scope))[-1]
        app.model.responses["interpret"].append(
            {
                "intent": "current_time",
                "dialogue_mode": "service",
                "language": "de",
                "current_time_requests": [
                    {
                        "location": "Berlin",
                        "timezone": "Europe/Berlin",
                        "source_message_id": message.id,
                    }
                ],
            }
        )
        app.model.responses["generate"].append(
            {"reply": answer, "dialogue_act": "current_time"}
        )
        await app.drain()
        assert len(app.sender.sent) == number
        assert app.sender.sent[-1] == (1, answer)
    assert not any(task in {"repair", "recover"} for task, _ in app.model.calls)
    assert sum(task == "validate" for task, _ in app.model.calls) == 7


async def test_transition_preserves_question_metadata_until_validation(
    app: Application,
) -> None:
    await seed_count(app, 19)
    notice = "続けたければ、BeHappy2Dayでチャット時間を購入できるよ"
    app.model.responses["interpret"].append(
        {"intent": "chat", "dialogue_mode": "continuation", "language": "ja"}
    )
    app.model.responses["generate"].append(
        {
            "reply": "君は最近どう？",
            "dialogue_act": "question",
            "question_goal": "recent_news",
            "question_part": "reply",
            "platform_notice": notice,
        }
    )
    answer = "君と話せてうれしいよ"
    app.model.responses["repair"].append(
        {"reply": answer, "dialogue_act": "answer", "platform_notice": notice}
    )
    await app.telegram.ingest([update(2, "最近どう？")])
    await app.drain()
    repaired = next(data for task, data in app.model.calls if task == "repair")
    assert "TRANSITION_ANSWER_ONLY" in repaired["REPAIR_REASON_CODES"]
    assert repaired["REJECTED_PROPOSAL"]["question_goal"] == "recent_news"
    await finish_delivery(app)
    assert app.sender.sent[-2:] == [(1, answer), (1, f"{notice}\n{PLATFORM_URL}")]
