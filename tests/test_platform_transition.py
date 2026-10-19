import asyncio

import pytest

from app.application import Application
from app.modules.delivery.entities import SendResult
from app.shared.storage import Transaction
from tests.helpers import (
    PLATFORM_INVITATION_EN,
    conversation,
    scalar,
    update,
)


def enable_preview(app: Application) -> None:
    app.commerce.settings = app.commerce.settings.model_copy(
        update={"preview_enabled": True, "preview_min_replies": 12}
    )


async def send_turn(app: Application, number: int, user: int = 1) -> None:
    await app.telegram.ingest([update(number, f"A new topic {number}", user=user)])
    await app.turn_worker.run_once()
    for _ in range(3):
        await app.delivery.run_once()
        due = await app.delivery_repository.next_ready_at(
            (await conversation(app, user)).scope
        )
        if due is None:
            break
        app.clock.advance(max(0, due - app.clock.now()))


async def finish_delivery(app: Application) -> None:
    await app.drain()
    for _ in range(3):
        due = await app.delivery_repository.next_ready_at(
            (await conversation(app)).scope
        )
        if due is None:
            return
        app.clock.advance(max(0, due - app.clock.now()))
        await app.delivery.run_once()
    raise AssertionError("Delivery did not finish")


async def seed_count(app: Application, count: int = 12) -> None:
    enable_preview(app)
    await app.telegram.ingest([update(1, "/start")])
    await app.delivery.run_once()

    async def seed(tx: Transaction) -> None:
        await tx.execute("UPDATE conversations SET preview_reply_count=?", (count,))

    await app.database.write(seed)


async def test_rejected_invitation_does_not_block_answer_at_reply_limit(
    app: Application,
) -> None:
    await seed_count(app, 19)
    for task in ("generate", "repair"):
        app.model.responses[task].append(
            {
                "reply": "I like clever comedies",
                "dialogue_act": "answer",
                "platform_transition": True,
                "platform_notice": PLATFORM_INVITATION_EN,
            }
        )
    for task in ("validate", "validate_repair"):
        app.model.responses[task].append(
            {"decision": "reject", "reason_codes": ["SITUATION.COMMERCE"]}
        )
    app.model.responses["recover"].append(
        {"reply": "I like clever comedies", "dialogue_act": "answer"}
    )
    await app.telegram.ingest([update(2, "What movies do you like?")])
    await finish_delivery(app)
    assert app.sender.sent[-2:] == [
        (1, "I like clever comedies"),
        (1, app.commerce.transition_text("en")),
    ]
    current = await conversation(app)
    assert current.preview_closed_at is not None
    assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"
    recovered = next(data for task, data in app.model.calls if task == "recover")
    assert recovered["PLATFORM_TRANSITION"] == {
        "eligible": False,
        "required": False,
        "server_invitation_pending": True,
    }
    assert any(task == "validate_recovery" for task, _ in app.model.calls)
    await send_turn(app, 3)
    assert "BeHappy2Day" not in app.sender.sent[-1][1]


async def test_first_twelve_replies_do_not_transition_and_duplicates_do_not_count(
    app: Application,
) -> None:
    enable_preview(app)
    for number in range(1, 13):
        await send_turn(app, number)
        await app.telegram.ingest([update(number, f"A new topic {number}")])
    current = await conversation(app)
    assert current.preview_reply_count == 12
    assert current.preview_closed_at is None
    assert len(app.sender.sent) == 12
    states = [
        data["PLATFORM_TRANSITION"]
        for task, data in app.model.calls
        if task == "generate"
    ]
    assert not any(state["eligible"] for state in states)
    app.clock.advance(86400)
    assert app.commerce.can_reply(await conversation(app))


async def test_default_invitation_window_starts_after_fifteen_answers(
    app: Application,
) -> None:
    from app.shared.config import CommerceSettings

    app.commerce.settings = CommerceSettings(preview_enabled=True)
    await app.telegram.ingest([update(1, "/start")])
    for count, eligible, required in [
        (14, False, False),
        (15, True, False),
        (19, True, True),
    ]:

        async def seed(tx: Transaction, count: int = count) -> None:
            await tx.execute("UPDATE conversations SET preview_reply_count=?", (count,))

        await app.database.write(seed)
        state = await app.commerce.transition_state(await conversation(app))
        assert state["eligible"] is eligible
        assert state["required"] is required


async def test_sunset_answer_with_optional_invitation_needs_no_notice_or_repair(
    app: Application,
) -> None:
    await seed_count(app, 15)
    answer = "At sunset, light travels through more air. More blue gets scattered away"
    app.model.responses["generate"].append(
        {"reply": answer, "dialogue_act": "explanation", "followup_allowed": True}
    )
    await app.telegram.ingest(
        [
            update(
                2, "And why does it turn red at sunset? A little more detail this time."
            )
        ]
    )
    await finish_delivery(app)
    assert app.sender.sent[-1] == (1, answer)
    assert not any(task in {"repair", "recover"} for task, _ in app.model.calls)
    assert (await conversation(app)).preview_closed_at is None


@pytest.mark.parametrize("count", [15, 19])
async def test_invitation_answers_first_and_later_inputs_continue_without_forget(
    app: Application, count: int
) -> None:
    await seed_count(app, count)
    answer = "I like the idea of a quiet evening with good music"
    app.model.responses["generate"].append(
        {
            "reply": answer,
            "dialogue_act": "opinion",
            "platform_transition": True,
            "platform_notice": PLATFORM_INVITATION_EN,
            "followup_allowed": True,
        }
    )
    await app.telegram.ingest([update(2, "What is new with you?")])
    await finish_delivery(app)
    assert app.sender.sent[-2:] == [
        (1, answer),
        (1, app.commerce.transition_text("en")),
    ]
    checked = next(data for task, data in app.model.calls if task == "validate")
    assert not checked["reply"]["followup_allowed"]
    assert not any(task in {"repair", "recover"} for task, _ in app.model.calls)
    scope = (await conversation(app)).scope
    for index in range(3, 6):
        before = len(app.sender.sent)
        await send_turn(app, index)
        assert len(app.sender.sent) == before + 1
        assert "BeHappy2Day" not in app.sender.sent[-1][1]
        assert (await conversation(app)).scope == scope
    assert (await app.commerce.transition_state(await conversation(app)))[
        "eligible"
    ] is False


async def test_two_parts_count_once_and_failed_delivery_does_not_count(
    app: Application,
) -> None:
    enable_preview(app)
    app.model.responses["generate"].append(
        {
            "reply": "The kitchen survived.",
            "continuation": "Dinner was less lucky.",
            "dialogue_act": "reaction",
        }
    )
    await send_turn(app, 1)
    assert len(app.sender.sent) == 2
    assert (await conversation(app)).preview_reply_count == 1
    app.sender.results.append(SendResult("delivery_unknown"))
    await send_turn(app, 2)
    assert (await conversation(app)).preview_reply_count == 1


async def test_agent_selects_invitation_and_next_input_still_gets_an_answer(
    app: Application,
) -> None:
    await seed_count(app)
    app.model.responses["generate"].append(
        {
            "reply": "I’m enjoying our conversation",
            "dialogue_act": "reaction",
            "platform_transition": True,
            "platform_notice": PLATFORM_INVITATION_EN,
        }
    )
    await send_turn(app, 2)
    current = await conversation(app)
    assert current.preview_closed_at is not None
    assert app.sender.sent[-1][1] == app.commerce.transition_text("en")
    assert "real men" not in app.sender.sent[-1][1]
    calls = len(app.model.calls)
    await send_turn(app, 3)
    assert len(app.model.calls) > calls
    assert "BeHappy2Day" not in app.sender.sent[-1][1]
    assert (
        await scalar(
            app,
            "SELECT COUNT(*) FROM outbox_messages WHERE json_extract(proposal_json,'$.origin')='platform_redirect'",
        )
        == 1
    )
    await app.telegram.ingest([update(4, "/start")])
    await app.delivery.run_once()
    assert "BeHappy2Day" not in app.sender.sent[-1][1]
    assert app.commerce.can_reply(await conversation(app))


async def test_optional_choice_can_defer_but_twentieth_response_is_transition(
    app: Application,
) -> None:
    await seed_count(app)
    for number in range(2, 9):
        await send_turn(app, number)
    assert (await conversation(app)).preview_reply_count == 19
    assert (await conversation(app)).preview_closed_at is None
    await send_turn(app, 9)
    assert (await conversation(app)).preview_closed_at is not None
    assert app.sender.sent[-1][1] == app.commerce.transition_text("en")


async def test_premature_model_decision_is_repaired_without_closing(
    app: Application,
) -> None:
    enable_preview(app)
    app.model.responses["generate"].append(
        {
            "reply": "Go to the platform",
            "dialogue_act": "reaction",
            "platform_transition": True,
            "platform_notice": PLATFORM_INVITATION_EN,
        }
    )
    await send_turn(app, 1)
    assert (await conversation(app)).preview_closed_at is None
    assert "BeHappy2Day" not in app.sender.sent[-1][1]
    assert any(task == "repair" for task, _ in app.model.calls)


async def test_stop_during_transition_generation_prevents_invitation(
    app: Application,
) -> None:
    await seed_count(app)
    app.model.release = asyncio.Event()
    app.model.responses["generate"].append(
        {
            "reply": "bye",
            "dialogue_act": "reaction",
            "platform_transition": True,
            "platform_notice": PLATFORM_INVITATION_EN,
        }
    )
    await app.telegram.ingest([update(2, "one more topic")])
    task = asyncio.create_task(app.turn_worker.run_once())
    await app.model.entered.wait()
    await app.telegram.ingest([update(3, "/stop")])
    await task
    await app.delivery.run_once()
    assert (await conversation(app)).preview_closed_at is None
    assert "BeHappy2Day" not in app.sender.sent[-1][1]


async def test_unknown_invitation_is_not_retried_but_new_input_gets_an_answer(
    app: Application,
) -> None:
    await seed_count(app, 19)
    app.sender.results.extend(
        [SendResult("delivered", 999999), SendResult("delivery_unknown")]
    )
    await send_turn(app, 2)
    await app.delivery.run_once()
    await send_turn(app, 3)
    assert (await conversation(app)).preview_closed_at is not None
    assert app.commerce.can_reply(await conversation(app))
    assert "BeHappy2Day" not in app.sender.sent[-1][1]
    assert (
        await scalar(
            app,
            "SELECT attempts FROM outbox_messages WHERE json_extract(proposal_json,'$.origin')='platform_redirect'",
        )
        == 1
    )


async def test_forget_resets_window_and_start_does_not(app: Application) -> None:
    await seed_count(app, 19)
    await app.telegram.ingest([update(2, "/start")])
    await app.delivery.run_once()
    assert (await conversation(app)).preview_reply_count == 19
    await send_turn(app, 3)
    epoch = (await conversation(app)).epoch
    await app.telegram.ingest([update(4, "/forget")])
    await app.control_worker.run_once()
    current = await conversation(app)
    assert current.epoch == epoch + 1
    assert current.preview_reply_count == 0
    assert current.preview_closed_at is None
    await app.telegram.ingest([update(5, "/start")])
    await send_turn(app, 6)
    assert app.commerce.can_reply(await conversation(app))


async def test_five_users_have_separate_counts(app: Application) -> None:
    enable_preview(app)
    await asyncio.gather(
        *(
            app.telegram.ingest([update(user, "hello", user=user)])
            for user in range(1, 6)
        )
    )
    for _ in range(5):
        await app.turn_worker.run_once()
    await app.delivery.run_once()
    assert [
        (await conversation(app, user)).preview_reply_count for user in range(1, 6)
    ] == [1] * 5


@pytest.mark.parametrize("language", ["ru", "en", "es", "pt"])
async def test_transition_text_does_not_disclose_conditions_or_claim_same_person(
    app: Application, language: str
) -> None:
    text = app.commerce.transition_text(language)
    assert "BeHappy2Day" in text
    body, link = text.rsplit("\n", 1)
    assert link == "https://www.behappy2day.com/"
    assert not any(
        word in body.lower()
        for word in (
            "12",
            "20",
            "minute",
            "минут",
            "real men",
            "реальн",
            "http",
            "marko",
        )
    )


async def test_current_refusal_is_applied_before_transition_decision(
    app: Application,
) -> None:
    from tests.helpers import interpretation_operation

    await seed_count(app, 19)
    await app.telegram.ingest([update(2, "No offers please")])
    current = await conversation(app)
    message = (await app.conversations.messages(current.scope))[-1]
    app.model.responses["interpret"].append(
        interpretation_operation(message, "offer_refusal", True, "boundary")
    )
    app.model.responses["generate"].append(
        {
            "reply": "Understood. Take care.",
            "dialogue_act": "closing",
        }
    )
    await app.turn_worker.run_once()
    await app.delivery.run_once()
    assert (await conversation(app)).preview_closed_at is None
    assert app.commerce.can_reply(await conversation(app))
    assert app.sender.sent[-1][1] == "Understood. Take care"
    assert "BeHappy2Day" not in app.sender.sent[-1][1]
    assert (
        await scalar(
            app,
            "SELECT COUNT(*) FROM outbox_messages WHERE json_extract(proposal_json,'$.origin')='platform_redirect'",
        )
        == 0
    )


async def test_restart_preserves_count_closed_state_and_pending_notice(
    app: Application,
) -> None:
    from tests.helpers import FakeSender, ScriptedModel

    await seed_count(app, 19)
    settings = app.settings.model_copy(update={"commerce": app.commerce.settings})
    clock = app.clock
    await app.telegram.ingest([update(2, "one more thing")])
    await app.turn_worker.run_once()
    scope = (await conversation(app)).scope
    await app.close()
    sender = FakeSender()
    sender.next_id = app.sender.next_id
    reopened = await Application.open(settings, sender, ScriptedModel(), clock)
    try:
        assert (await conversation(reopened)).scope == scope
        assert (await conversation(reopened)).preview_reply_count == 19
        assert reopened.commerce.can_reply(await conversation(reopened))
        await finish_delivery(reopened)
        assert len(reopened.sender.sent) == 2
        await send_turn(reopened, 3)
        assert len(reopened.sender.sent) == 3
        assert "BeHappy2Day" not in reopened.sender.sent[-1][1]
        assert reopened.sender.sent[1][1] == reopened.commerce.transition_text("en")
        await reopened.delivery.run_once()
        assert len(reopened.sender.sent) == 3
    finally:
        await reopened.close()


async def test_stop_after_transition_preparation_blocks_delivery(
    app: Application,
) -> None:
    await seed_count(app, 19)
    await app.telegram.ingest([update(2, "one more topic")])
    await app.turn_worker.run_once()
    await app.telegram.ingest([update(3, "/stop")])
    await app.delivery.run_once()
    assert "BeHappy2Day" not in app.sender.sent[-1][1]
    assert (
        await scalar(
            app,
            "SELECT status FROM outbox_messages WHERE json_extract(proposal_json,'$.origin')='platform_redirect'",
        )
        == "cancelled"
    )


def test_preview_configuration_rejects_invalid_range() -> None:
    from pydantic import ValidationError

    from app.shared.config import CommerceSettings

    with pytest.raises(ValidationError):
        CommerceSettings(preview_min_replies=20, preview_max_replies=12)


async def test_delivery_count_changed_during_generation_still_enforces_maximum(
    app: Application,
) -> None:
    await seed_count(app, 18)
    app.model.release = asyncio.Event()
    await app.telegram.ingest([update(2, "one more thing")])
    task = asyncio.create_task(app.turn_worker.run_once())
    await app.model.entered.wait()

    async def delivery_finished(tx: Transaction) -> None:
        await tx.execute("UPDATE conversations SET preview_reply_count=19")

    await app.database.write(delivery_finished)
    app.model.release.set()
    await task
    await finish_delivery(app)
    assert (await conversation(app)).preview_closed_at is not None
    assert app.sender.sent[-1][1] == app.commerce.transition_text("en")


async def test_transition_answer_is_checked_by_semantic_critic(
    app: Application,
) -> None:
    await seed_count(app)
    app.model.responses["generate"].append(
        {
            "reply": "done",
            "dialogue_act": "closing",
            "platform_transition": True,
            "platform_notice": PLATFORM_INVITATION_EN,
        }
    )
    await send_turn(app, 2)
    assert [task for task, _ in app.model.calls] == [
        "interpret",
        "generate",
        "validate",
    ]
    assert app.sender.sent[-1][1] == app.commerce.transition_text("en")


async def test_unconfirmed_inflight_previous_reply_cannot_push_past_maximum(
    app: Application,
) -> None:
    await seed_count(app, 18)
    await app.telegram.ingest([update(2, "previous thought")])
    await app.turn_worker.run_once()

    async def sending(tx: Transaction) -> None:
        await tx.execute(
            "UPDATE outbox_messages SET status='sending' WHERE kind='reply' AND status='ready'"
        )

    await app.database.write(sending)
    await app.telegram.ingest([update(3, "next thought")])
    await app.turn_worker.run_once()
    assert (await conversation(app)).preview_closed_at is not None
    assert (
        await scalar(
            app,
            "SELECT COUNT(*) FROM outbox_messages WHERE json_extract(proposal_json,'$.origin')='platform_redirect'",
        )
        == 1
    )


async def test_public_preview_does_not_expose_internal_mock_package(
    app: Application,
) -> None:
    enable_preview(app)
    app.model.responses["interpret"].append(
        {
            "intent": "purchase_question",
            "dialogue_mode": "service",
            "language": "en",
            "requested_business_action": "show_offer",
        }
    )
    app.model.responses["generate"].append(
        {
            "reply": "test package",
            "dialogue_act": "show_offer",
            "requested_action": "show_offer",
        }
    )
    await send_turn(app, 1)
    assert (
        await scalar(app, "SELECT COUNT(*) FROM outbox_messages WHERE kind='offer'")
        == 0
    )
    assert not any("Test package" in text for _, text in app.sender.sent)
    assert (await conversation(app)).preview_closed_at is None


@pytest.mark.parametrize("failure", ["delivery_unknown", "permanent_failure"])
async def test_invitation_never_sends_without_confirmed_answer(
    app: Application, failure: str
) -> None:
    await seed_count(app, 19)
    app.sender.results.append(SendResult(failure))
    before = len(app.sender.sent)
    await send_turn(app, 2)
    await finish_delivery(app)
    assert len(app.sender.sent) == before + 1
    assert "BeHappy2Day" not in app.sender.sent[-1][1]
    assert (
        await scalar(
            app,
            "SELECT status FROM outbox_messages WHERE json_extract(proposal_json,'$.origin')='platform_redirect'",
        )
        == "cancelled"
    )
    assert (
        await scalar(
            app,
            "SELECT attempts FROM outbox_messages WHERE json_extract(proposal_json,'$.origin')='platform_redirect'",
        )
        == 0
    )
    await send_turn(app, 3)
    assert len(app.sender.sent) == before + 2


async def test_new_input_between_answer_and_invitation_cancels_only_old_invitation(
    app: Application,
) -> None:
    await seed_count(app, 19)
    await app.telegram.ingest([update(2, "What movies do you like?")])
    await app.turn_worker.run_once()
    await app.delivery.run_once()
    answer = app.sender.sent[-1]
    assert "BeHappy2Day" not in answer[1]
    await app.telegram.ingest([update(3, "No offers please")])
    await finish_delivery(app)
    assert app.sender.sent.count(answer) == 1
    assert not any("BeHappy2Day" in text for _, text in app.sender.sent)
    assert (
        await scalar(
            app,
            "SELECT status FROM outbox_messages WHERE json_extract(proposal_json,'$.origin')='platform_redirect'",
        )
        == "cancelled"
    )


@pytest.mark.parametrize("language", ["en", "ru", "es", "fr", "ar"])
async def test_recovery_invitation_uses_localized_catalog_template(
    app: Application, language: str
) -> None:
    from app.modules.agent.resources import response_templates

    assert language in response_templates()["platform_invitation"]
    await seed_count(app, 19)
    texts = {
        "en": ("What movies do you like?", "I like clever comedies"),
        "ru": ("Какие фильмы тебе нравятся?", "Люблю умные комедии"),
        "es": ("¿Qué películas te gustan?", "Me gustan las comedias inteligentes"),
        "fr": ("Quels films aimes-tu ?", "J’aime les comédies intelligentes"),
        "ar": ("ما الأفلام التي تحبها؟", "أحب الأفلام الكوميدية الذكية"),
    }
    question, answer = texts[language]
    app.model.responses["interpret"].append(
        {"intent": "chat", "dialogue_mode": "continuation", "language": language}
    )
    for task in ("generate", "repair"):
        app.model.responses[task].append(
            {
                "reply": answer,
                "dialogue_act": "answer",
                "platform_notice": "https://invalid.example",
            }
        )
    app.model.responses["recover"].append({"reply": answer, "dialogue_act": "answer"})
    await app.telegram.ingest([update(2, question)])
    await finish_delivery(app)
    assert app.sender.sent[-2:] == [
        (1, answer),
        (1, app.commerce.transition_text(language)),
    ]
    assert not any("invalid.example" in text for _, text in app.sender.sent)
