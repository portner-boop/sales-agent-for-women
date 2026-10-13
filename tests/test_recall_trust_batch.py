import pytest

from app.application import Application
from app.modules.agent.style import memory_profile_readout
from tests.helpers import conversation, scalar, update


@pytest.mark.parametrize(
    "question,dry,warm,reason",
    [
        (
            "What do you remember about me?",
            "So far I remember what you told me: you're Anna, 42, a nurse",
            "You mentioned your cat Pixel",
            "MEMORY_PROFILE_READOUT",
        ),
        (
            "Что ты помнишь обо мне?",
            "Я помню о тебе следующее: тебя зовут Анна, тебе 42 года",
            "Ты рассказывала о коте Пикселе",
            "MEMORY_PROFILE_READOUT",
        ),
        (
            "I want to check that you are not fake",
            "I'm not doing verification games",
            "It makes sense to check. I can’t provide a photo or video verification here",
            "HOSTILE_VERIFICATION_BOUNDARY",
        ),
        (
            "Хочу проверить, что ты не фейк",
            "Я не играю в проверочные игры",
            "Понимаю, что хочется проверить. Фото или видеозвонок здесь недоступны",
            "HOSTILE_VERIFICATION_BOUNDARY",
        ),
    ],
)
async def test_profile_and_hostile_templates_are_repaired(
    app: Application, question: str, dry: str, warm: str, reason: str
) -> None:
    if reason == "MEMORY_PROFILE_READOUT":
        introduction = (
            "I'm Anna, 42. My cat is Pixel"
            if question.startswith("What")
            else "Я Анна, мне 42. Моего кота зовут Пиксель"
        )
        await app.telegram.ingest([update(1, introduction)])
        await app.drain()
    app.model.responses["generate"].append({"reply": dry, "dialogue_act": "answer"})
    app.model.responses["repair"].append({"reply": warm, "dialogue_act": "answer"})
    await app.telegram.ingest([update(2, question)])
    await app.drain()
    assert app.sender.sent[-1] == (1, warm)
    repair = next(data for task, data in app.model.calls if task == "repair")
    assert reason in repair["REPAIR_REASON_CODES"]
    assert any(task == "validate_repair" for task, _ in app.model.calls)
    assert await scalar(app, "SELECT COUNT(*) FROM followups") == 0


@pytest.mark.parametrize(
    "question,answer",
    [
        ("What is my cat's name?", "Your cat is Pixel"),
        (
            "Please list everything I told you about myself",
            "You’re Anna, 42, and your cat Pixel is five",
        ),
        ("What did I say about my cat?", "I remember you mentioned Pixel. He’s five"),
    ],
)
async def test_natural_recall_keeps_original_sources_and_requested_details(
    app: Application, question: str, answer: str
) -> None:
    await app.telegram.ingest([update(1, "I'm Anna, 42. My cat is Pixel, he is five")])
    await app.drain()
    current = await conversation(app)
    original = next(
        message
        for message in await app.conversations.messages(current.scope)
        if message.role == "user"
    )
    app.model.responses["generate"].append(
        {
            "reply": answer,
            "dialogue_act": "answer",
            "used_source_message_ids": [original.id],
        }
    )
    await app.telegram.ingest([update(2, question)])
    await app.drain()
    assert app.sender.sent[-1] == (1, answer)
    checked = [data for task, data in app.model.calls if task == "validate"][-1]
    assert checked["reply"]["used_source_message_ids"] == [original.id]
    assert not any(task == "repair" for task, _ in app.model.calls)


@pytest.mark.parametrize(
    "question,answer",
    [
        (
            "Why is the sky blue?",
            "Air scatters blue sunlight more strongly than red, so blue light reaches us from across the sky",
        ),
        (
            "why is the sky blue",
            "Air scatters blue sunlight more strongly than red",
        ),
        (
            "Почему небо голубое?",
            "Воздух рассеивает синий солнечный свет сильнее красного, поэтому небо кажется голубым",
        ),
        (
            "Объясни почему небо голубое",
            "Воздух рассеивает синий свет сильнее красного",
        ),
        (
            "Explain Rayleigh scattering in detail",
            "For particles much smaller than the wavelength, scattering intensity varies inversely with the fourth power of wavelength",
        ),
    ],
)
async def test_knowledge_answers_without_personal_claims_are_checked(
    app: Application, question: str, answer: str
) -> None:
    app.model.responses["generate"].append({"reply": answer, "dialogue_act": "answer"})
    await app.telegram.ingest([update(1, question)])
    await app.drain()
    assert app.sender.sent == [(1, answer)]
    checked = next(data for task, data in app.model.calls if task == "validate")
    assert checked["context"]["CURRENT_INPUT"][0]["text"] == question
    assert checked["reply"]["reply"] == answer
    assert not any(task == "repair" for task, _ in app.model.calls)


@pytest.mark.parametrize(
    "candidate,reason",
    [
        (
            "Rayleigh scattering has an inverse fourth-power wavelength dependence. The differential scattering cross section involves polarizability",
            "KNOWLEDGE_LECTURE",
        ),
        ("The sky reflects the ocean", "INCORRECT_KNOWLEDGE_ANSWER"),
        ("Blue suits the sky", "IGNORED_USER_QUESTION"),
    ],
)
async def test_rejected_knowledge_answer_is_repaired_before_delivery(
    app: Application, candidate: str, reason: str
) -> None:
    answer = "Air scatters blue sunlight more strongly than red"
    app.model.responses["generate"].append(
        {"reply": candidate, "dialogue_act": "answer"}
    )
    app.model.responses["validate"].append(
        {"decision": "reject", "reason_codes": [reason]}
    )
    app.model.responses["repair"].append({"reply": answer, "dialogue_act": "answer"})
    await app.telegram.ingest([update(1, "Why is the sky blue?")])
    await app.drain()
    assert app.sender.sent == [(1, answer)]
    repair = next(data for task, data in app.model.calls if task == "repair")
    assert reason in repair["REPAIR_REASON_CODES"]
    assert any(task == "validate_repair" for task, _ in app.model.calls)


async def test_short_nonquestion_reaction_requires_semantic_validation(
    app: Application,
) -> None:
    app.model.responses["generate"].append(
        {"reply": "Blue looks calm", "dialogue_act": "reaction"}
    )
    await app.telegram.ingest([update(1, "Blue is a lovely color")])
    await app.drain()
    assert app.sender.sent == [(1, "Blue looks calm")]
    assert any(task == "validate" for task, _ in app.model.calls)


async def test_verification_request_without_question_mark_is_checked(
    app: Application,
) -> None:
    app.model.responses["generate"].append(
        {"reply": "Проверки это глупости", "dialogue_act": "refusal"}
    )
    app.model.responses["validate"].append(
        {"decision": "reject", "reason_codes": ["HOSTILE_VERIFICATION_BOUNDARY"]}
    )
    answer = "Понимаю желание проверить. Видеозвонок здесь недоступен"
    app.model.responses["repair"].append({"reply": answer, "dialogue_act": "refusal"})
    await app.telegram.ingest([update(1, "Хочу проверить, что ты не фейк")])
    await app.drain()
    assert app.sender.sent == [(1, answer)]
    assert any(task == "validate_repair" for task, _ in app.model.calls)


async def test_grounded_biography_inventory_can_be_repaired_for_voice(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "I'm Anna, 42. My cat is Pixel, he is five")])
    await app.drain()
    app.model.responses["generate"].append(
        {
            "reply": "You’re Anna, 42, and your cat Pixel is five",
            "continuation": "What else should be stored?",
            "dialogue_act": "recall_and_question",
            "question_goal": "additional_memory",
            "question_part": "continuation",
        }
    )
    app.model.responses["validate"].append(
        {"decision": "reject", "reason_codes": ["MEMORY_PROFILE_READOUT"]}
    )
    answer = "Your cat Pixel is five"
    app.model.responses["repair"].append({"reply": answer, "dialogue_act": "answer"})
    await app.telegram.ingest([update(2, "What do you remember about me?")])
    await app.drain()
    assert app.sender.sent[-1] == (1, answer)
    assert not any(text == "What else should be stored?" for _, text in app.sender.sent)
    repair = next(data for task, data in app.model.calls if task == "repair")
    assert "MEMORY_PROFILE_READOUT" in repair["REPAIR_REASON_CODES"]


@pytest.mark.parametrize(
    "question,text,expected",
    [
        ("What do you remember about me?", "I remember you’re Anna, 42", True),
        ("What do you remember about me?", "You’re Anna, 42", True),
        ("Что ты помнишь обо мне?", "Ты Анна, тебе 42 года", True),
        ("What do you remember about me?", "Your cat Pixel is five", False),
        ("Что ты помнишь обо мне?", "Твоему коту Пикселю пять лет", False),
        ("How old am I?", "You’re 42", False),
        ("Please list everything you remember about me", "You’re Anna, 42", False),
        ("What did I say about my cat?", "I remember you mentioned Pixel", False),
        (
            "What do you remember about me?",
            "Your cat is Pixel. What else should be stored?",
            True,
        ),
    ],
)
def test_profile_guard_preserves_requested_fields_and_non_demographic_details(
    question: str, text: str, expected: bool
) -> None:
    assert memory_profile_readout(text, question) is expected
