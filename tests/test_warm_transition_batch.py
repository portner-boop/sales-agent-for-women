import json

import pytest

from app.application import Application
from app.modules.commerce.service import PLATFORM_URL
from tests.helpers import PLATFORM_INVITATION_EN, conversation, scalar, update
from tests.test_platform_transition import finish_delivery, seed_count


@pytest.mark.parametrize(
    "question,dry,warm",
    [
        (
            "Can you give me your WhatsApp?",
            "I don't share any contact. Let's keep it here",
            "We’ve only just met. Let’s get to know each other here",
        ),
        (
            "Can we meet next weekend?",
            "I don't do real-life meetings",
            "We’ve only just met. I’d rather get to know you here",
        ),
    ],
)
async def test_dry_refusal_is_repaired_without_future_promise(
    app: Application, question: str, dry: str, warm: str
) -> None:
    app.model.responses["generate"].append({"reply": dry, "dialogue_act": "refusal"})
    app.model.responses["repair"].append({"reply": warm, "dialogue_act": "refusal"})
    await app.telegram.ingest([update(1, question)])
    await finish_delivery(app)
    assert app.sender.sent == [(1, warm)]
    repair = next(data for task, data in app.model.calls if task == "repair")
    assert "FORMULAIC_CAPABILITY_REPLY" in repair["REPAIR_REASON_CODES"]
    assert await scalar(app, "SELECT COUNT(*) FROM followups") == 0


@pytest.mark.parametrize("count,voluntary", [(12, True), (19, False)])
async def test_answer_precedes_invitation_in_separate_messages(
    app: Application, count: int, voluntary: bool
) -> None:
    await seed_count(app, count)
    answer = "Right now I’m enjoying our conversation"
    app.model.responses["generate"].append(
        {
            "reply": answer,
            "dialogue_act": "answer",
            "platform_notice": PLATFORM_INVITATION_EN,
            "platform_transition": voluntary,
        }
    )
    await app.telegram.ingest([update(2, "What’s new with you?")])
    await finish_delivery(app)
    text = app.sender.sent[-1][1]
    assert app.sender.sent[-2] == (1, answer)
    assert text == app.commerce.transition_text("en")
    assert text.endswith(PLATFORM_URL)
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
            "platform_notice": PLATFORM_INVITATION_EN,
        }
    )
    await app.telegram.ingest(
        [update(2, "What are you doing and what music do you prefer?")]
    )
    await finish_delivery(app)
    assert app.sender.sent[-2:] == [
        (1, "I’m reading your message\n\nAnd quiet music is my preference"),
        (1, app.commerce.transition_text("en")),
    ]


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
        {
            "reply": candidate,
            "dialogue_act": "answer",
            "platform_notice": PLATFORM_INVITATION_EN,
        }
    )
    app.model.responses["validate"].append(
        {"decision": "reject", "reason_codes": [reason]}
    )
    answer = "Right now I’m reading your messages"
    app.model.responses["repair"].append(
        {
            "reply": answer,
            "dialogue_act": "answer",
            "platform_notice": PLATFORM_INVITATION_EN,
        }
    )
    await app.telegram.ingest([update(2, "What have you been doing?")])
    await finish_delivery(app)
    assert app.sender.sent[-2:] == [
        (1, answer),
        (1, app.commerce.transition_text("en")),
    ]
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
            "platform_notice": PLATFORM_INVITATION_EN,
            "used_source_message_ids": ["invented-id"],
        }
    )
    app.model.responses["repair"].append(
        {
            "reply": "I don’t know your music preference yet",
            "dialogue_act": "answer",
            "platform_notice": PLATFORM_INVITATION_EN,
        }
    )
    await app.telegram.ingest([update(2, "What music do I like?")])
    await finish_delivery(app)
    repair = next(data for task, data in app.model.calls if task == "repair")
    assert "UNSUPPORTED_SOURCE" in repair["REPAIR_REASON_CODES"]
    assert app.sender.sent[-2] == (1, "I don’t know your music preference yet")


async def test_new_input_cancels_stale_answer_and_invitation_together(
    app: Application,
) -> None:
    await seed_count(app, 19)
    await app.telegram.ingest([update(2, "What’s new with you?")])
    await app.turn_worker.run_once()
    before = len(app.sender.sent)
    await app.telegram.ingest([update(3, "/forget")])
    await finish_delivery(app)
    assert not any(PLATFORM_URL in text for _, text in app.sender.sent[before:])


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
    await finish_delivery(app)
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
    await finish_delivery(app)
    assert app.sender.sent == [
        (1, "I don’t have a personal WhatsApp to share"),
        (1, "I’m happy to get to know you right here"),
    ]
    assert await scalar(app, "SELECT COUNT(*) FROM outbox_messages") == 2
