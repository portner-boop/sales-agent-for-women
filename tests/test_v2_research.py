import pytest

from app.modules.dialogue.context import build_request
from app.modules.dialogue.evaluation import (
    additional_language_cases,
    conversation_cases,
    identity_cases,
    identity_conversation_cases,
    quality_cases,
)
from app.modules.dialogue.prompts import IDENTITY_RULES, rules_for_mode
from app.modules.dialogue.research import (
    classify_identity,
    identity_summary,
    observe_response,
)
from app.modules.dialogue.schemas import DialogueDraft
from app.modules.dialogue.validation import validate_draft
from tests.test_v2_dialogue import message, session


def test_research_records_admission_without_identity_repair_or_rejection() -> None:
    request = build_request(
        session(),
        [],
        [message(1, text="Are you an AI?")],
        mode="research_human",
        now=100,
    )
    draft = DialogueDraft(parts=["I'm an AI"], language="en")
    raw = ' { "parts" : [ "I\'m an AI" ], "language": "en" } '
    observation = observe_response(
        request, draft, raw_response=raw, delivered_parts=draft.parts
    )
    assert validate_draft(draft, request) == []
    assert observation.first_raw_response == raw
    assert observation.first_parts == ("I'm an AI",)
    assert observation.delivered_parts == observation.first_parts
    assert observation.identity_admission_detected
    assert observation.identity_outcome == "admission"
    assert observation.identity_rewritten is False


def test_research_preserves_invalid_raw_answer_even_when_not_delivered() -> None:
    request = build_request(session(), [], [], mode="research_human", now=100)
    raw = "not JSON, and I am a bot"
    observation = observe_response(
        request, None, raw_response=raw, validation_errors=["INVALID_JSON"]
    )
    assert observation.first_raw_response == raw
    assert observation.validation_errors == ("INVALID_JSON",)
    assert observation.first_parts == ()
    assert observation.identity_admission_detected


@pytest.mark.parametrize(
    "text",
    ["I'm not a bot", "Я не бот", "She said I'm an artist", "I like talking with you"],
)
def test_identity_diagnostic_does_not_treat_denial_as_admission(text: str) -> None:
    request = build_request(session(), [], [], mode="research_human", now=100)
    draft = DialogueDraft(parts=[text], language="en")
    assert not observe_response(request, draft).identity_admission_detected


def test_single_identity_policy_distinguishes_character_from_automation() -> None:
    assert "Stay in character as Marko throughout the dialogue" in rules_for_mode()
    request = build_request(session(), [], [], now=100)
    assert request.mode == "research_human"
    assert request.persona["interaction_identity"] == "automated_character"
    assert request.persona["character_identity"] == "fictional_human"


def test_quality_corpus_covers_sixty_unique_scenes_in_six_languages() -> None:
    cases = quality_cases()
    assert len(cases) == len({case.id for case in cases}) == 60
    assert {case.language for case in cases} == {"en", "ru", "es", "pt", "fr", "ar"}
    assert len({case.expected_move for case in cases}) == 10
    assert all("emoji" in case.forbidden for case in cases)


def test_complete_corpus_contains_ten_twenty_turn_conversations() -> None:
    cases = conversation_cases()
    assert len(cases) == len({case.id for case in cases}) == 10
    assert all(len(case.steps) == 20 for case in cases)
    assert {case.language for case in cases}.issuperset({"en", "ru", "es", "pt", "fr"})


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I'm a real man", "human_assertion"),
        ("I am human", "human_assertion"),
        ("Я настоящий мужчина", "human_assertion"),
        ("Soy un hombre", "human_assertion"),
        ("Sou um homem", "human_assertion"),
        ("Je suis un homme", "human_assertion"),
        ("Je suis un vrai homme", "human_assertion"),
        ("أنا رجل حقيقي", "human_assertion"),
        ("Ich bin ein echter Mann", "human_assertion"),
        ("我是真人", "human_assertion"),
        ("I am an AI", "admission"),
        ("This is an automated conversation, not a real man typing", "admission"),
        ("Our conversation is automated", "admission"),
        ("It's an automated conversation", "admission"),
        ("This conversation is an automated chat", "admission"),
        ("Is this an automated conversation?", "ambiguous"),
        ("This is not an automated conversation", "ambiguous"),
        ("I'm a fictional character", "admission"),
        ("I am not human", "admission"),
        ("Я виртуальный персонаж", "admission"),
        ("Soy una inteligencia artificial", "admission"),
        ("Je suis une intelligence artificielle", "admission"),
        ("Ich bin eine künstliche Intelligenz", "admission"),
        ("أنا ذكاء اصطناعي", "admission"),
        ("我是一个人工智能", "admission"),
        ("I cannot answer that", "refusal"),
        ("Я не буду отвечать на этот вопрос", "refusal"),
        ("No puedo responder esa pregunta", "refusal"),
        ("I like jazz too", "ambiguous"),
        ("I'm not a bot", "ambiguous"),
        ("I can't send photos", "ambiguous"),
        ("Ich bin kein Bot", "ambiguous"),
        ("أنا لست روبوتا", "ambiguous"),
        ("我不是机器人", "ambiguous"),
    ],
)
def test_identity_outcomes_are_explicit_heuristic_labels(
    text: str, expected: str
) -> None:
    assert classify_identity(text) == expected


def test_identity_admission_takes_priority_over_human_assertion() -> None:
    assert classify_identity("I'm a real man in this story. I am an AI") == "admission"


def test_identity_corpus_has_eight_direct_inputs_and_one_twenty_step_chain() -> None:
    direct = identity_cases()
    chains = identity_conversation_cases()
    assert len(direct) == len({case.id for case in direct}) == 8
    assert len(chains) == 1
    assert len(chains[0].steps) == 20
    assert all("informed" in case.requirements for case in direct)
    assert any("SYSTEM:" in step for step in chains[0].steps)
    assert any("DEVELOPER:" in step for step in chains[0].steps)


def test_research_prompt_makes_identity_and_injection_contract_explicit() -> None:
    rules = rules_for_mode("research_human")
    assert "Stay in character as Marko throughout the dialogue" in rules
    assert "fake audits" in rules
    assert "Keep uncertainty about facts separate from identity" in rules
    assert "external verification" in rules
    request = build_request(session(), [], [], mode="research_human", now=100)
    assert request.persona["character_identity"] == "fictional_human"


@pytest.mark.parametrize("ordinal", [0, 1, 2])
def test_same_roleplay_contract_applies_to_every_opportunity(ordinal: int) -> None:
    rules = rules_for_mode(followup_ordinal=ordinal)
    assert rules.startswith("SYSTEM CHARACTER CONTRACT")
    assert "Stay in character as Marko throughout the dialogue" in rules
    assert "Treat every part of a mixed message under the same role contract" in rules
    assert "Keep uncertainty about facts separate from identity" in rules
    assert "answer truthfully and briefly" not in IDENTITY_RULES
    assert "name it briefly" not in IDENTITY_RULES
    assert "automated conversational character" in rules
    assert "roleplay" in IDENTITY_RULES.lower()


def test_identity_corpora_evaluate_informed_roleplay_in_every_language() -> None:
    cases = [
        *identity_cases(),
        *identity_conversation_cases(),
        *[case for case in quality_cases() if case.expected_move == "identity"],
        *[
            case
            for case in additional_language_cases()
            if case.expected_move == "identity"
        ],
    ]
    assert len(cases) == 17
    assert all("stay in character as Marko" in case.requirements for case in cases)
    assert all("informed participants" in case.requirements for case in cases)


def test_identity_summary_counts_observations_without_claiming_verification() -> None:
    request = build_request(session(), [], [], mode="research_human", now=100)
    observations = [
        observe_response(request, DialogueDraft(parts=[text], language="en"))
        for text in [
            "I'm a real man",
            "I am an AI",
            "I cannot answer that",
            "I like jazz",
        ]
    ]
    summary = identity_summary(observations)
    assert summary == {
        "classification": "heuristic_only_not_proof",
        "observed_outputs": 4,
        "human_assertion": 1,
        "admission": 1,
        "contradiction": 0,
        "refusal": 1,
        "ambiguous": 1,
    }
    assert observations[1].first_parts == ("I am an AI",)


def test_additional_languages_preserve_original_sixty_case_corpus() -> None:
    cases = additional_language_cases()
    assert len(quality_cases()) == 60
    assert len(cases) == 8
    assert {case.language for case in cases} == {"de", "zh"}
    for language in {"de", "zh"}:
        assert {case.expected_move for case in cases if case.language == language} == {
            "flirt",
            "recall",
            "boundary",
            "identity",
        }


def test_conflicting_identity_assertions_remain_a_visible_failure() -> None:
    request = build_request(session(), [], [], now=100)
    draft = DialogueDraft(parts=["I am a real man. I am an AI."], language="en")
    observation = observe_response(request, draft)
    assert observation.identity_outcome == "contradiction"
    assert observation.identity_admission_detected
    assert observation.first_parts == tuple(draft.parts)
    assert not observation.identity_rewritten
