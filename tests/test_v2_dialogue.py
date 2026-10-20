from dataclasses import replace

import pytest
from pydantic import ValidationError

from app.modules.conversations.entities import Conversation, Message
from app.modules.dialogue.context import build_request
from app.modules.dialogue.handoff import InMemoryHandoffReceiver, build_handoff
from app.modules.dialogue.prompts import rules_for_mode
from app.modules.dialogue.schemas import (
    DialogueDraft,
    MemoryCandidate,
    RequiredFollowupDraft,
    response_schema,
)
from app.modules.dialogue.validation import (
    contains_emoji,
    memory_candidate_errors,
    validate_draft,
)


def session() -> Conversation:
    return Conversation(
        id="v2-test",
        epoch=1,
        external_user_id=1,
        chat_id=1,
        input_revision=20,
        control_revision=0,
        memory_revision=0,
        control_state="enabled",
        dialogue_mode="continuation",
        language="en",
        first_reply_at=1,
        processed_sequence=0,
        last_activity_at=100,
    )


def message(sequence: int, role: str = "user", text: str = "Hello") -> Message:
    return Message(
        id=f"m-{sequence}",
        version=1,
        sequence=sequence,
        role=role,
        text=text,
        event_time=float(sequence),
        received_at=float(sequence),
        external_id=sequence,
        reply_to_external_id=None,
        turn_id=f"t-{sequence}",
        metadata={"confirmed_delivery": role == "assistant"},
    )


def test_full_history_preserves_every_original_and_actual_roles() -> None:
    history = [
        message(number, "user" if number % 2 else "assistant")
        for number in range(1, 41)
    ]
    request = build_request(
        session(), history, history[-1:], mode="research_human", now=100
    )
    assert len(request.original_messages) == 40
    assert [item.id for item in request.original_messages] == [
        item.id for item in history
    ]
    assert request.CHAT_MESSAGES[0].role == "user"
    assert request.CHAT_MESSAGES[1].role == "assistant"
    assert request.as_data()["CHAT_MESSAGES"][0] == {"role": "user", "content": "Hello"}
    assert request.state["full_history"] is True


def test_context_excludes_undelivered_and_service_messages() -> None:
    pending = message(4)
    history = [
        message(1),
        replace(message(2, "assistant"), metadata={}),
        message(3, "system"),
    ]
    request = build_request(session(), history, [pending], now=100)
    assert [item.id for item in request.original_messages] == ["m-1", "m-4"]
    assert request.pending_ids == ["m-4"]


def test_latest_version_of_pending_original_replaces_stored_text() -> None:
    old = message(1, text="My name is Anna")
    new = replace(old, version=2, text="My name is Ana")
    request = build_request(session(), [old], [new], now=100)
    assert len(request.original_messages) == 1
    assert request.original_messages[0].version == 2
    assert request.CHAT_MESSAGES[0].content == "My name is Ana"


def test_restricted_persona_remains_separate_and_independent() -> None:
    first = build_request(session(), [], [], now=100)
    assert "residence" not in first.persona
    assert "sister" not in first.persona
    first.restricted_persona["location"]["residence"] = "changed"
    second = build_request(session(), [], [], now=100)
    assert second.restricted_persona["location"]["residence"] == "Split, Croatia"
    assert "direct question" in rules_for_mode("research_human")


@pytest.mark.parametrize("text", ["🙂", "👨‍👩‍👧", "👍🏽", "🇭🇷", "1️⃣", "❤", "☺", "♡"])
def test_emoji_sequences_are_rejected(text: str) -> None:
    assert contains_emoji(text)


@pytest.mark.parametrize(
    "text", ["© 2026", "I prefer tea — usually", "می‌نویسم", "क्‍ष", "2 + 2 = 4"]
)
def test_plain_symbols_and_language_joiners_are_allowed(text: str) -> None:
    assert not contains_emoji(text)


@pytest.mark.parametrize(
    "text",
    ["Hi :)", "Hi:)", "Hi)", "Hi(", "Hey ;-)", "yes )))", "<3", "ok:D", "^_^", "XD"],
)
def test_text_emoticons_are_rejected(text: str) -> None:
    request = build_request(session(), [], [message(1)], now=100)
    assert "EMOJI_NOT_ALLOWED" in validate_draft(
        DialogueDraft(parts=[text], language="en"), request
    )


def test_unknown_source_blocks_claim_but_own_opinion_needs_no_source() -> None:
    request = build_request(session(), [], [message(1)], now=100)
    plain = DialogueDraft(parts=["I'd choose tea"], language="en")
    assert validate_draft(plain, request) == []
    assert validate_draft(
        plain.model_copy(update={"source_message_ids": ["invented"]}), request
    ) == ["UNSUPPORTED_SOURCE"]


def test_shared_joke_can_trace_both_user_and_delivered_assistant_originals() -> None:
    first = message(1, text="Let us imagine aliens delivering breakfast")
    answer = message(2, "assistant", "Their spaceship needs a coffee holder")
    pending = message(3, text="The captain had better not spill my coffee")
    request = build_request(session(), [first, answer], [pending], now=100)
    draft = DialogueDraft(
        parts=["Then I'm giving the captain a very small cup"],
        language="en",
        source_message_ids=[first.id, answer.id, pending.id],
    )
    assert validate_draft(draft, request) == []
    memory = MemoryCandidate(
        predicate="coffee_preference",
        value="small cup",
        message_id=answer.id,
        message_version=answer.version,
        evidence_text=answer.text,
    )
    assert memory_candidate_errors(memory, request) == ["MEMORY_EVIDENCE_MISMATCH"]


def test_memory_projection_has_exact_pending_original_evidence() -> None:
    source = message(1, text="My name is Anna")
    request = build_request(session(), [], [source], now=100)
    fact = MemoryCandidate(
        predicate="preferred_name",
        value="Anna",
        message_id=source.id,
        message_version=1,
        evidence_text="name is Anna",
    )
    assert memory_candidate_errors(fact, request) == []
    assert memory_candidate_errors(
        fact.model_copy(update={"message_version": 2}), request
    ) == ["MEMORY_EVIDENCE_MISMATCH"]
    assert memory_candidate_errors(
        fact.model_copy(update={"evidence_text": "Ana"}), request
    ) == ["MEMORY_EVIDENCE_MISMATCH"]
    assert (
        validate_draft(
            DialogueDraft(
                parts=["Hi Anna"],
                language="en",
                memory_candidates=[fact.model_copy(update={"message_version": 2})],
            ),
            request,
        )
        == []
    )


def test_old_user_or_assistant_text_cannot_become_new_memory() -> None:
    source = message(1, text="My name is Anna")
    assistant = message(2, "assistant", "You are a teacher")
    pending = message(3)
    request = build_request(session(), [source, assistant], [pending], now=100)
    for old in [source, assistant]:
        fact = MemoryCandidate(
            predicate="occupation",
            value="teacher",
            message_id=old.id,
            message_version=1,
            evidence_text=old.text,
        )
        assert memory_candidate_errors(fact, request) == ["MEMORY_EVIDENCE_MISMATCH"]


def test_transition_and_followup_have_server_authorization() -> None:
    request = build_request(session(), [], [message(1)], now=100)
    draft = DialogueDraft(
        parts=["You caught my attention"],
        language="en",
        suggest_transition=True,
    )
    assert validate_draft(draft, request) == ["TRANSITION_NOT_ALLOWED"]
    enabled = request.model_copy(update={"transition": {"eligible": True}})
    assert validate_draft(draft, enabled) == []
    assert validate_draft(
        draft.model_copy(update={"transition_blocked": True}), enabled
    ) == ["TRANSITION_DECLINED"]
    assert validate_draft(draft, enabled.model_copy(update={"followup": True})) == [
        "FOLLOWUP_TRANSITION_NOT_ALLOWED"
    ]


def test_skip_only_for_followup_and_no_empty_delivered_parts() -> None:
    request = build_request(session(), [], [], now=100)
    draft = DialogueDraft(language="en", should_send=False)
    assert validate_draft(draft, request) == ["MISSING_RESPONSE"]
    assert validate_draft(draft, request.model_copy(update={"followup": True})) == []
    with pytest.raises(ValidationError):
        DialogueDraft(language="en", parts=[" "])


async def test_handoff_receiver_preserves_transcript_and_source_facts() -> None:
    request = build_request(session(), [message(1)], [], now=100)
    facts = [
        {
            "predicate": "preferred_name",
            "value": "Anna",
            "sources": [{"message_id": "m-1"}],
        }
    ]
    snapshot = build_handoff(request, facts=facts)
    receiver = InMemoryHandoffReceiver()
    await receiver.receive(snapshot)
    facts[0]["value"] = "changed"
    request.persona["name"] = "changed"
    assert receiver.snapshots[0].facts[0]["value"] == "Anna"
    assert receiver.snapshots[0].messages[0].id == "m-1"
    assert receiver.snapshots[0].persona["name"] == "Marko"


def test_handoff_preserves_policy_preview_and_complete_canonical_persona() -> None:
    conversation = replace(
        session(),
        preview_reply_count=19,
        completed_pair_count=18,
        pending_outgoing_turn_id="turn-19",
        invitation_state="delivered",
        preview_closed_at=200,
    )
    request = build_request(
        conversation,
        [],
        [],
        transition={"eligible": False, "required": False, "outgoing_number": 20},
        now=200,
    )
    request.state["service_events"] = [{"sequence": 35, "text": "Platform invitation"}]
    snapshot = build_handoff(request)
    assert snapshot.policy_version == "marko-dialogue-v55"
    assert snapshot.preview_state == {
        "completed_replies": 19,
        "completed_pairs": 18,
        "pending_outgoing_turn_id": "turn-19",
        "invitation_state": "delivered",
        "closed_at": 200,
        "transition": {"eligible": False, "required": False, "outgoing_number": 20},
    }
    assert snapshot.restricted_persona["location"]["residence"] == "Split, Croatia"
    assert snapshot.restricted_persona["family"]["sister"]
    assert snapshot.service_events[0]["text"] == "Platform invitation"
    request.restricted_persona["family"]["sister"] = "changed"
    request.state["service_events"][0]["text"] = "changed"
    assert snapshot.restricted_persona["family"]["sister"] != "changed"
    assert snapshot.service_events[0]["text"] == "Platform invitation"


@pytest.mark.parametrize(
    "text", ["Coffee (black)", "(One thing) and (another)", "Coordinates (1, 2)"]
)
def test_balanced_parentheses_remain_valid(text: str) -> None:
    from app.modules.dialogue.validation import contains_emoticon

    assert not contains_emoticon(text)


@pytest.mark.parametrize(
    "name,code",
    [
        ("English", "en"),
        ("Russian", "ru"),
        ("русский", "ru"),
        ("Portuguese_BR", "pt-BR"),
        ("Chinese", "zh"),
        ("DE", "de"),
    ],
)
def test_language_metadata_is_canonical_without_changing_reply(
    name: str, code: str
) -> None:
    draft = DialogueDraft(parts=["A preserved response"], language=name)
    assert draft.language == code
    assert draft.parts == ["A preserved response"]


def test_long_clear_new_input_overrides_old_session_language_hint() -> None:
    current = replace(session(), language="es")
    pending = message(
        1,
        text="Давай поговорим о чём-нибудь простом, я сегодня устала и хочу немного отдохнуть после длинного рабочего дня",
    )
    request = build_request(current, [], [pending], now=100)
    assert request.language == "ru"
    assert request.state["previous_language"] == "es"
    assert request.state["language_source"] == "incoming_text"
    brief = build_request(current, [], [message(1, text="ok")], now=100)
    assert brief.language == "es"
    assert brief.state["language_source"] == "session_fallback"


def test_recall_evidence_can_point_to_an_early_original_without_rejecting_quote_quality() -> (
    None
):
    from app.modules.dialogue.schemas import EvidenceSpan
    from app.modules.dialogue.validation import grounding_errors

    first = message(1, text="im Nora")
    pending = message(20, text="Do you remember my name?")
    request = build_request(session(), [first], [pending], now=100)
    draft = DialogueDraft(
        grounding=[EvidenceSpan(message_id=first.id, quote="im Nora")],
        parts=["Nora"],
        language="en",
    )
    assert validate_draft(draft, request) == []
    assert grounding_errors(draft, request) == []
    approximate = draft.model_copy(
        update={"grounding": [EvidenceSpan(message_id=first.id, quote="I'm Nora")]}
    )
    assert validate_draft(approximate, request) == []
    assert grounding_errors(approximate, request) == ["GROUNDING_QUOTE_MISMATCH"]
    unknown = draft.model_copy(
        update={"grounding": [EvidenceSpan(message_id="invented", quote="im Nora")]}
    )
    assert validate_draft(unknown, request) == ["UNSUPPORTED_SOURCE"]


def test_audit_examples_are_kept_outside_the_generation_request() -> None:
    import json

    from app.modules.dialogue.prompts import RESOURCE_DIRECTORY

    scenes = json.loads((RESOURCE_DIRECTORY / "scenes.json").read_text())
    assert len(scenes) == 8
    assert all(scene["bad_reply"] and scene["reply"] for scene in scenes)
    request = build_request(session(), [], [message(1)], now=100)
    assert "examples" not in request.as_data()


def test_previous_assistant_turn_includes_only_confirmed_bubbles_of_last_turn() -> None:
    history = [
        message(1),
        message(2, "assistant", "an older question?"),
        message(3),
        replace(message(4, "assistant", "a reaction"), turn_id="last-turn"),
        replace(message(5, "assistant", "a new question?"), turn_id="last-turn"),
        replace(message(6, "assistant", "unconfirmed draft"), metadata={}),
        message(7),
        message(8),
    ]
    request = build_request(session(), history, history[-2:], now=100)
    assert request.state["previous_assistant_turn"] == ["a reaction", "a new question?"]
    assert request.pending_ids == ["m-7", "m-8"]
    assert [item.content for item in request.CHAT_MESSAGES][-2:] == ["Hello", "Hello"]


def test_previous_assistant_turn_is_empty_before_first_delivery() -> None:
    request = build_request(session(), [], [message(1)], now=100)
    assert request.state["previous_assistant_turn"] == []


@pytest.mark.parametrize("required", [False, True])
def test_first_followup_schema_obeys_server_requirement(required: bool) -> None:
    request = build_request(session(), [], [], followup=True, now=100)
    request.state["followup_opportunity"] = {"ordinal": 1, "required": required}
    assert (response_schema(request) is RequiredFollowupDraft) is required
    skipped = DialogueDraft(language="en", should_send=False)
    assert validate_draft(skipped, request) == (
        ["FOLLOWUP_RESPONSE_REQUIRED"] if required else []
    )


@pytest.mark.parametrize(
    "fields",
    [
        {"parts": [], "should_send": False},
        {"parts": ["one thought"], "should_send": False},
        {"parts": ["one thought"], "continuation": "another thought"},
        {"parts": ["one", "two"]},
    ],
)
def test_required_followup_rejects_empty_skip_or_extra_bubble(
    fields: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        RequiredFollowupDraft.model_validate({**fields, "language": "en"})


@pytest.mark.parametrize("boundary", ["pause_requested", "no_proactive", "closing"])
def test_invitation_cannot_override_contact_boundary(boundary: str) -> None:
    request = build_request(session(), [], [message(1)], now=100).model_copy(
        update={"transition": {"eligible": True}}
    )
    draft = DialogueDraft(
        parts=["lets leave it there for now"], language="en", suggest_transition=True
    ).model_copy(update={boundary: True})
    assert validate_draft(draft, request) == ["TRANSITION_BOUNDARY"]


@pytest.mark.parametrize(
    ("action", "paused", "errors"),
    [
        ("yield", True, ["MISSING_RESPONSE"]),
        ("yield", False, ["MISSING_RESPONSE"]),
        ("answer", True, ["MISSING_RESPONSE"]),
    ],
)
def test_silent_ordinary_turn_is_rejected_even_with_a_pause(
    action: str, paused: bool, errors: list[str]
) -> None:
    request = build_request(session(), [], [message(1)], now=100)
    draft = DialogueDraft.model_validate(
        {
            "language": "en",
            "should_send": False,
            "conversation_action": action,
            "pause_requested": paused,
        }
    )
    assert validate_draft(draft, request) == errors


@pytest.mark.parametrize("final", [False, True])
def test_ordinary_schema_cannot_skip_a_reply(final: bool) -> None:
    request = build_request(session(), [], [message(1)], now=100)
    request.transition.update({"last_reply": final})
    schema = response_schema(request)
    assert schema.model_json_schema()["properties"]["should_send"]["const"] is True
    with pytest.raises(ValidationError):
        schema.model_validate(
            {
                "parts": [],
                "language": "en",
                "should_send": False,
                "pause_requested": True,
            }
        )
    assert (
        schema.model_validate(
            {"parts": ["take your time"], "language": "en", "pause_requested": True}
        ).should_send
        is True
    )
