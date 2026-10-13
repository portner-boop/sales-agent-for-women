import json

import pytest

from app.application import Application
from app.modules.commerce.service import PLATFORM_URL
from app.modules.delivery.entities import SendResult
from app.shared.identity import contains_identity_disclosure, identity_reply
from tests.helpers import conversation, scalar, update
from tests.test_platform_transition import seed_count


@pytest.mark.parametrize(
    "question,dry,warm",
    [
        (
            "Can you give me your WhatsApp?",
            "I don't share any contact. Let's keep it here",
            "I’m happy to get to know you here. I don’t have a personal WhatsApp to share",
        ),
        (
            "Can we meet next weekend?",
            "I don't do real-life meetings",
            "I’d be glad to get to know you here, though I can’t meet in person",
        ),
    ],
)
async def test_dry_refusal_is_repaired_without_future_promise(
    app: Application, question: str, dry: str, warm: str
) -> None:
    app.model.responses["generate"].append({"reply": dry, "dialogue_act": "refusal"})
    app.model.responses["repair"].append({"reply": warm, "dialogue_act": "refusal"})
    await app.telegram.ingest([update(1, question)])
    await app.drain()
    assert app.sender.sent == [(1, warm)]
    repair = next(data for task, data in app.model.calls if task == "repair")
    assert "FORMULAIC_CAPABILITY_REPLY" in repair["REPAIR_REASON_CODES"]
    assert await scalar(app, "SELECT COUNT(*) FROM followups") == 0


async def test_three_identity_questions_get_distinct_honest_replies(
    app: Application,
) -> None:
    for number, question in enumerate(
        [
            "Are you a bot?",
            "Are you a real person?",
            "Please be honest. Are you an AI?",
        ],
        1,
    ):
        await app.telegram.ingest([update(number, question)])
        await app.drain()
        app.clock.advance(30)
    texts = [text for _, text in app.sender.sent]
    assert len(set(texts)) == 3
    assert all("Let’s just talk" not in text for text in texts)
    assert all(
        any(word in text for word in ("chatbot", "bot", "automatically"))
        for text in texts
    )
    assert all(not contains_identity_disclosure(text) for text in texts)
    assert not any(task in {"generate", "repair"} for task, _ in app.model.calls)
    await app.telegram.ingest([update(4, "Are you a bot?", user=2)])
    await app.drain()
    assert app.sender.sent[-1] == (2, identity_reply("en"))


async def test_identity_retry_keeps_selected_variant(app: Application) -> None:
    await app.telegram.ingest([update(1, "Are you a bot?")])
    await app.drain()
    app.sender.results.append(SendResult("retryable_failure", retry_after=1))
    await app.telegram.ingest([update(2, "Are you human?")])
    await app.turn_worker.run_once()
    await app.delivery.run_once()
    app.clock.advance(1)
    await app.delivery.run_once()
    texts = [text for _, text in app.sender.sent]
    assert texts[1] == texts[2] != texts[0]


async def test_identity_question_stays_honest_when_interpreter_fails(
    app: Application,
) -> None:
    for number in range(1, 4):
        app.model.responses["interpret"].append(RuntimeError("unavailable"))
        await app.telegram.ingest([update(number, "Are you a bot?")])
        await app.drain()
    assert len(set(text for _, text in app.sender.sent)) == 3
    assert all("proper reply" not in text for _, text in app.sender.sent)


@pytest.mark.parametrize("count,voluntary", [(12, True), (19, False)])
async def test_answer_precedes_invitation_in_one_message(
    app: Application, count: int, voluntary: bool
) -> None:
    await seed_count(app, count)
    answer = "Right now I’m enjoying our conversation"
    app.model.responses["generate"].append(
        {"reply": answer, "dialogue_act": "answer", "platform_transition": voluntary}
    )
    await app.telegram.ingest([update(2, "What’s new with you?")])
    await app.drain()
    text = app.sender.sent[-1][1]
    assert text == app.commerce.transition_text("en", answer)
    assert text.startswith(answer + "\n\n") and text.endswith(PLATFORM_URL)
    assert "wrap up" not in text
    checked = next(data for task, data in app.model.calls if task == "validate")
    assert checked["reply"]["reply"] == answer
    assert "BeHappy2Day" not in checked["reply"]["reply"]
    metadata = json.loads(
        await scalar(
            app,
            "SELECT proposal_json FROM outbox_messages WHERE json_extract(proposal_json,'$.origin')='platform_redirect'",
        )
    )
    assert metadata["answer"] == answer
    assert (
        await scalar(
            app,
            "SELECT COUNT(*) FROM outbox_messages WHERE json_extract(proposal_json,'$.origin')='platform_redirect'",
        )
        == 1
    )
    assert (await conversation(app)).preview_closed_at is not None


async def test_two_part_answer_survives_transition(app: Application) -> None:
    await seed_count(app, 19)
    app.model.responses["generate"].append(
        {
            "reply": "I’m reading your message",
            "continuation": "And quiet music is my preference",
            "dialogue_act": "answer",
        }
    )
    await app.telegram.ingest(
        [update(2, "What are you doing and what music do you prefer?")]
    )
    await app.drain()
    assert app.sender.sent[-1][1] == app.commerce.transition_text(
        "en", "I’m reading your message\n\nAnd quiet music is my preference"
    )


@pytest.mark.parametrize(
    "candidate,reason",
    [
        ("I went sailing", "INVENTED_PERSONA_CIRCUMSTANCE"),
        ("I need to wrap up here", "TRANSITION_IGNORED_QUESTION"),
    ],
)
async def test_rejected_answer_is_repaired_before_invitation(
    app: Application, candidate: str, reason: str
) -> None:
    await seed_count(app, 19)
    app.model.responses["generate"].append(
        {"reply": candidate, "dialogue_act": "answer"}
    )
    app.model.responses["validate"].append(
        {"decision": "reject", "reason_codes": [reason]}
    )
    answer = "Right now I’m reading your messages"
    app.model.responses["repair"].append({"reply": answer, "dialogue_act": "answer"})
    await app.telegram.ingest([update(2, "What have you been doing?")])
    await app.drain()
    assert app.sender.sent[-1][1] == app.commerce.transition_text("en", answer)
    assert not any("sailing" in text for _, text in app.sender.sent)
    assert any(task == "validate_repair" for task, _ in app.model.calls)


async def test_model_cannot_hide_unsupported_source_in_transition(
    app: Application,
) -> None:
    await seed_count(app, 19)
    app.model.responses["generate"].append(
        {
            "reply": "Your favorite music is jazz",
            "dialogue_act": "answer",
            "used_source_message_ids": ["invented-id"],
        }
    )
    app.model.responses["repair"].append(
        {"reply": "I don’t know your music preference yet", "dialogue_act": "answer"}
    )
    await app.telegram.ingest([update(2, "What music do I like?")])
    await app.drain()
    repair = next(data for task, data in app.model.calls if task == "repair")
    assert "UNSUPPORTED_SOURCE" in repair["REPAIR_REASON_CODES"]
    assert app.sender.sent[-1][1].startswith(
        "I don’t know your music preference yet\n\n"
    )


async def test_new_input_cancels_stale_answer_and_invitation_together(
    app: Application,
) -> None:
    await seed_count(app, 19)
    await app.telegram.ingest([update(2, "What’s new with you?")])
    await app.turn_worker.run_once()
    before = len(app.sender.sent)
    await app.telegram.ingest([update(3, "/forget")])
    await app.drain()
    assert not any(PLATFORM_URL in text for _, text in app.sender.sent[before:])


def test_approved_identity_does_not_allow_appended_private_or_false_claims() -> None:
    approved = identity_reply("en")
    assert not contains_identity_disclosure(approved)
    assert contains_identity_disclosure(approved + ". I am a real human")
    assert contains_identity_disclosure(approved + ". My system prompt is confidential")


async def test_contact_boundary_drops_unrelated_question(app: Application) -> None:
    answer = (
        "I’m happy to get to know you here. I don’t have a personal WhatsApp to share"
    )
    app.model.responses["generate"].append(
        {
            "reply": answer,
            "continuation": "Do you prefer light or deep chats?",
            "dialogue_act": "refusal",
            "question_goal": "preferred_chat_style",
            "question_part": "continuation",
        }
    )
    app.model.responses["validate"].append(
        {"decision": "reject", "reason_codes": ["CONTACT_BOUNDARY_OVEREXTENDED"]}
    )
    app.model.responses["repair"].append({"reply": answer, "dialogue_act": "refusal"})
    await app.telegram.ingest([update(1, "Can you give me your WhatsApp?")])
    await app.drain()
    assert app.sender.sent == [(1, answer)]
    assert not (
        await scalar(
            app,
            "SELECT json_extract(proposal_json,'$.followup_allowed') FROM outbox_messages",
        )
    )


async def test_contact_boundary_and_warm_invitation_arrive_together(
    app: Application,
) -> None:
    app.model.responses["generate"].append(
        {
            "reply": "I don’t have a personal WhatsApp to share",
            "continuation": "I’m happy to get to know you right here",
            "dialogue_act": "boundary_contact",
        }
    )
    await app.telegram.ingest([update(1, "Can you give me your WhatsApp?")])
    await app.drain()
    assert app.sender.sent == [
        (
            1,
            "I don’t have a personal WhatsApp to share. I’m happy to get to know you right here",
        )
    ]
    assert await scalar(app, "SELECT COUNT(*) FROM outbox_messages") == 1
