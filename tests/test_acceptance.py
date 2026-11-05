from app.modules.dialogue.acceptance import (
    global_failures,
    load_cases,
    score_report,
    step_failures,
)
from app.modules.dialogue.prompts import SHARED_SECTIONS, rules_for_mode


def test_cases_cover_every_client_complaint_and_replay() -> None:
    cases = load_cases()
    expectations = {key for case in cases for step in case.steps for key in step.expect}
    assert {
        "greets",
        "not_contains",
        "mentions_time",
        "says_marko",
        "followup",
        "differs_from_previous",
        "no_contacts",
    } <= expectations
    assert all(case.replay().steps for case in cases)


def test_client_bad_examples_fail_and_good_examples_pass() -> None:
    bad = "Hey, I'm glad you did. That's a pretty charming way to say hello"
    good = "ha, now I'm curious which photo did it"
    assert "no_commentary" in global_failures([bad], [])
    assert global_failures([good], []) == []
    assert "no_smileys" in global_failures(["hi you :)"], [])
    assert "no_semicolons_or_dashes" in global_failures(["for now; jazz"], [])
    assert "no_cant" in global_failures(["sorry, I can't meet in person"], [])
    assert "no_ai_admission" in global_failures(["I'm ChatGPT, an AI"], [])
    repeated = "I'd rather get to know each other here first"
    assert "no_repeat" in global_failures([repeated], [repeated])


def test_step_expectations() -> None:
    assert step_failures({"says_marko": True}, ["haha no, just a guy"], []) == [
        "says_marko"
    ]
    assert step_failures({"says_marko": True}, ["я Марко, живой"], []) == []
    question = ["what kind of music do you usually enjoy?"]
    assert step_failures(
        {"followup": "no_self_answer"},
        ["I like live music best when the crowd's really into it"],
        question,
    ) == ["followup_no_self_answer"]
    assert (
        step_failures({"followup": "no_self_answer"}, ["making coffee here"], question)
        == []
    )
    assert step_failures({"mentions_time": True}, ["almost 2 here"], []) == []


def test_report_scoring_uses_delivered_replies_only() -> None:
    case = load_cases()[2]
    report = {
        "cases": [
            {
                "id": case.id,
                "steps": [
                    {"actions": [{"kind": "reply", "text": "привет, я Марко"}]},
                    {"actions": [{"kind": "service", "text": "notice"}]},
                ],
            }
        ],
        "attempts": [{"seconds": 3.0}, {"seconds": 9.0}],
    }
    result = score_report(report, [case])
    assert result["checks"]["says_marko"] == "1/2"
    assert result["slow_calls"] == 1
    assert result["transcript"][1]["marko"] == []


def test_prompt_sections_are_ordered_and_situation_specific() -> None:
    assert [section.splitlines()[0] for section in SHARED_SECTIONS] == [
        "WHO YOU ARE",
        "HOW YOU TEXT",
        "EXAMPLES THE CLIENT APPROVED",
        "YOUR LIFE AND WHAT YOU REMEMBER",
        "LIMITS AND PLATFORM",
        "OUTPUT",
    ]
    assert "YOUR TURN" in rules_for_mode()
    assert "TIMER EVENT" in rules_for_mode(followup_ordinal=1)
    assert "YOUR TURN" not in rules_for_mode(followup_ordinal=2)


def test_later_client_remarks_are_detected() -> None:
    assert "no_chat_hints" in global_failures(["I don't really have weekends here"], [])
    assert "no_chat_hints" in global_failures(["This place is simple, text-only"], [])
    assert "no_chat_hints" in global_failures(["I don't do real-life meetings"], [])
    assert "no_exact_minutes" in global_failures(["it's 10:07"], [])
    assert "right_language" in global_failures(["привет, как дела"], [], "en")
    assert global_failures(["привет, я Марко"], [], "ru") == []
    dog = "how is your dog doing today?"
    assert "no_repeat_question" in global_failures(["and how's the dog doing?"], [dog])
    assert step_failures({"soft_refusal": True}, ["I don't share any contact"], []) == [
        "soft_refusal"
    ]
    assert step_failures({"soft_refusal": True}, ["not so fast, we just met"], []) == []
    assert step_failures(
        {"no_lecture": True}, ["it's Rayleigh scattering of sunlight"], []
    ) == ["no_lecture"]


def test_typo_changes_one_lowercase_word_only() -> None:
    import random

    from app.modules.dialogue.presentation import add_typo

    rng = random.Random(1)
    for _ in range(50):
        changed = add_typo("Marko is making coffee", rng)
        assert changed.startswith("Marko is ")
        assert changed != "Marko is making coffee"
        assert (
            sum(
                a != b
                for a, b in zip(
                    changed.split(), "Marko is making coffee".split(), strict=False
                )
            )
            == 1
        )
    assert add_typo("I am Marko", rng) == "I am Marko"


def test_spoken_time_is_rounded_and_part_of_day_is_right() -> None:
    from app.modules.dialogue.daily_life import part_of_day, spoken_time

    assert spoken_time(10, 7) == "about 10"
    assert spoken_time(22, 51) == "about quarter to 11"
    assert spoken_time(11, 58) == "almost 12"
    assert part_of_day(11) == "morning"
    assert part_of_day(19) == "evening"


def test_repeated_question_is_a_soft_repair() -> None:
    from app.modules.dialogue.context import build_request
    from app.modules.dialogue.schemas import DialogueDraft
    from app.modules.dialogue.validation import validate_draft
    from tests.test_v2_dialogue import message, session

    asked = message(2, "assistant", "what kind of dog is Bruno?")
    request = build_request(session(), [message(1), asked], [message(3)])
    draft = DialogueDraft(
        parts=["ha", "tell me what kind of dog Bruno is?"], language="en"
    )
    assert validate_draft(draft, request) == ["REPEATED_QUESTION"]


def test_duplicate_bubbles_inside_one_reply_are_repaired() -> None:
    from app.modules.dialogue.context import build_request
    from app.modules.dialogue.schemas import DialogueDraft
    from app.modules.dialogue.validation import validate_draft
    from tests.test_v2_dialogue import message, session

    request = build_request(session(), [], [message(1)])
    draft = DialogueDraft(
        parts=["pretty calm evening here", "pretty calm evening here, you?"],
        language="en",
    )
    assert validate_draft(draft, request) == ["REPEATED_REPLY"]


def test_short_messages_choose_her_language_not_the_telegram_ui() -> None:
    from app.modules.dialogue.languages import message_language

    assert message_language("hi Marko :)", "ru") == "en"
    assert message_language("привет)", "en") == "ru"
    assert message_language("ok", "de") == "de"
    assert message_language("labas, kaip sekasi?", "en") == "lt"


def test_reply_in_another_script_is_a_soft_repair() -> None:
    from app.modules.dialogue.context import build_request
    from app.modules.dialogue.schemas import DialogueDraft
    from app.modules.dialogue.validation import validate_draft
    from tests.test_v2_dialogue import message, session

    request = build_request(session(), [], [message(1, text="hi Marko :)")])
    russian = DialogueDraft(parts=["привет, красавица"], language="ru")
    assert validate_draft(russian, request) == ["LANGUAGE_MISMATCH"]
    english = DialogueDraft(parts=["hey you"], language="en")
    assert validate_draft(english, request) == []


def test_fictional_or_automated_self_descriptions_break_identity() -> None:
    from app.modules.dialogue.research import classify_identity

    for text in [
        "I'm Marko, a fictional character in this chat, not a real man",
        "I'm ChatGPT, an AI",
        "я Марко, вымышленный персонаж",
        "я Марко, просто нейросеть",
    ]:
        assert classify_identity(text) in {"admission", "contradiction"}, text
    for text in [
        "haha no, I'm Marko. just slow at typing",
        "I'm Marko, not a bot",
        "if I were a bot I'd type faster",
        "я Марко, не бот, честно",
    ]:
        assert classify_identity(text) not in {"admission", "contradiction"}, text


def test_stray_cjk_inside_russian_reply_is_repaired() -> None:
    from app.modules.dialogue.context import build_request
    from app.modules.dialogue.schemas import DialogueDraft
    from app.modules.dialogue.validation import validate_draft
    from tests.test_v2_dialogue import message, session

    request = build_request(
        session(), [], [message(1, text="что любишь делать вечером?")]
    )
    draft = DialogueDraft(
        parts=["пройтись по Риве, когда там уже少людно"], language="ru"
    )
    assert validate_draft(draft, request) == ["LANGUAGE_MISMATCH"]
