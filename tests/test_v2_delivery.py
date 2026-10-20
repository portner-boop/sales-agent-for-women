import asyncio
import json
from typing import Any

import pytest

from app.application import Application
from app.modules.conversations.entities import Conversation, Turn
from app.modules.delivery.entities import SendResult
from app.modules.delivery.schemas import OutboxDraft
from app.shared.storage import Transaction
from tests.helpers import conversation, scalar, update


async def v2_conversation(app: Application, user: int = 1) -> Conversation:
    async def ensure(tx: Transaction) -> Conversation:
        return await app.conversations.ensure(
            tx,
            app.channel,
            user,
            user,
            app.clock.now(),
            engine_version="v2",
            persona_mode="research_human",
        )

    return await app.database.write(ensure)


async def prepare_reply(
    app: Application,
    number: int = 1,
    texts: tuple[str, ...] = ("That was a good surprise",),
    proposal: dict[str, Any] | None = None,
) -> tuple[Turn, list[str]]:
    current = await v2_conversation(app)
    await app.telegram.ingest([update(number, f"A new detail {number}")])
    turn = await app.conversations.claim_turn(current.scope, app.clock.now())
    assert turn is not None

    async def prepare(tx: Transaction) -> list[str]:
        latest = await app.conversations.get(current.id, tx)
        identifiers: list[str] = []
        for index, text in enumerate(texts):
            identifier = await app.delivery_repository.prepare(
                tx,
                latest.snapshot,
                OutboxDraft(
                    action_key=f"v2:{turn.id}:{index}",
                    kind="reply",
                    text=text,
                    deadline=app.clock.now() + 30,
                    predecessor_id=identifiers[-1] if identifiers else None,
                ),
                app.clock.now(),
                proposal or {"dialogue_act": "reaction"},
                turn.id,
            )
            identifiers.append(identifier)
        await app.conversations.finish_turn(tx, turn, "prepared")
        return identifiers

    return turn, await app.database.write(prepare)


async def test_v2_is_the_only_engine_and_mode_stays_bound(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "hello", user=2)])
    existing = await conversation(app, 2)
    assert existing.engine_version == "v2"
    assert (await v2_conversation(app, 2)).engine_version == "v2"
    current = await v2_conversation(app)
    assert current.engine_version == "v2"
    assert current.persona_mode == "research_human"


async def test_v2_telegram_binds_configured_engine_and_research_mode(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "/start")])
    current = await conversation(app)
    assert current.engine_version == "v2"
    assert current.persona_mode == "research_human"


async def test_v2_any_user_can_chat_without_an_id_allowlist(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "hello", user=3)])
    await app.drain()
    current = await conversation(app, 3)
    assert current.persona_mode == "research_human"
    assert current.control_state == "enabled"
    assert app.commerce.can_reply(current)
    assert len(app.model.calls) == 1
    assert await scalar(app, "SELECT COUNT(*) FROM messages WHERE role='user'") == 1
    assert app.sender.sent[-1][0] == 3
    await app.telegram.ingest([update(2, "/privacy", user=3)])
    await app.drain()
    assert len(app.sender.sent) == 2


async def test_v2_unrestricted_user_can_stop_resume_and_forget(
    app: Application,
) -> None:
    user = 300031
    await app.telegram.ingest([update(1, "hello", user=user)])
    await app.drain()
    await app.telegram.ingest([update(2, "/stop", user=user)])
    await app.drain()
    assert (await conversation(app, user)).control_state == "stopped"
    await app.telegram.ingest([update(3, "not yet", user=user)])
    await app.drain()
    assert len(app.model.calls) == 1
    await app.telegram.ingest(
        [update(4, "/start", user=user), update(5, "back", user=user)]
    )
    await app.drain()
    assert (await conversation(app, user)).control_state == "enabled"
    assert len(app.model.calls) == 2
    await app.telegram.ingest([update(6, "/forget", user=user)])
    await app.drain()
    current = await conversation(app, user)
    assert current.epoch == 2
    assert current.preview_reply_count == 0
    assert not [
        message
        for message in await app.conversations.messages(current.scope)
        if message.role in {"user", "assistant"}
    ]


async def test_v2_parts_count_once_and_next_fresh_input_pairs_once(
    app: Application,
) -> None:
    turn, _ = await prepare_reply(app, texts=("You caught me", "I was smiling"))
    assert (await conversation(app)).preview_reply_count == 0
    await app.delivery.run_once()
    await app.delivery.run_once()
    current = await conversation(app)
    assert current.preview_reply_count == 1
    assert current.completed_pair_count == 0
    assert current.pending_outgoing_turn_id == turn.id
    await app.telegram.ingest(
        [update(2, "I knew it"), update(2, "I knew it"), update(3, "one more thing")]
    )
    current = await conversation(app)
    assert current.completed_pair_count == 1
    assert current.pending_outgoing_turn_id is None
    assert await scalar(app, "SELECT COUNT(*) FROM outgoing_dialogue_turns") == 1
    assert await scalar(app, "SELECT COUNT(*) FROM said_ledger") == 2
    assert await scalar(app, "SELECT status FROM turns") == "delivered"
    assert (
        await scalar(app, "SELECT COUNT(*) FROM scheduled_jobs WHERE kind='compact'")
        == 0
    )


async def test_v2_edit_does_not_complete_an_exchange(app: Application) -> None:
    await prepare_reply(app)
    await app.delivery.run_once()
    await app.telegram.ingest([update(2, "corrected", message_id=1, edited=True)])
    assert (await conversation(app)).completed_pair_count == 0
    await app.telegram.ingest([update(3, "a new reply")])
    assert (await conversation(app)).completed_pair_count == 1


@pytest.mark.parametrize("origin", ["followup", "response_recovery"])
async def test_v2_followup_and_service_do_not_use_a_reply_slot(
    app: Application, origin: str
) -> None:
    turn, identifiers = await prepare_reply(app, proposal={"origin": origin})
    if origin == "response_recovery":

        async def service(tx: Transaction) -> None:
            await tx.execute(
                "UPDATE outbox_messages SET kind='service' WHERE id=?",
                (identifiers[0],),
            )

        await app.database.write(service)
    await app.delivery.run_once()
    current = await conversation(app)
    assert current.preview_reply_count == 0
    assert current.pending_outgoing_turn_id is None
    assert turn.id is not None


async def test_v2_unknown_uses_slot_without_confirmed_history_or_pair(
    app: Application,
) -> None:
    app.sender.results.append(SendResult("delivery_unknown", error_code="NETWORK"))
    await prepare_reply(app)
    await app.delivery.run_once()
    await app.delivery.run_once()
    await app.delivery_repository.recover()
    current = await conversation(app)
    assert current.preview_reply_count == 1
    assert current.first_reply_at is None
    assert current.pending_outgoing_turn_id is None
    assert (
        await scalar(app, "SELECT COUNT(*) FROM messages WHERE role='assistant'") == 0
    )
    assert await scalar(app, "SELECT status FROM turns") == "delivery_unknown"
    assert len(app.sender.sent) == 1
    await app.telegram.ingest([update(2, "are you there?")])
    assert (await conversation(app)).completed_pair_count == 0


async def test_v2_interrupted_send_recovers_unknown_once(app: Application) -> None:
    _, identifiers = await prepare_reply(app)

    async def sending(tx: Transaction) -> None:
        await tx.execute(
            "UPDATE outbox_messages SET status='sending' WHERE id=?", (identifiers[0],)
        )

    await app.database.write(sending)
    await app.delivery_repository.recover()
    await app.delivery_repository.recover()
    await app.delivery.run_once()
    assert (await conversation(app)).preview_reply_count == 1
    assert not app.sender.sent
    assert await scalar(app, "SELECT outcome FROM outgoing_dialogue_turns") == "unknown"


async def test_v2_definite_failure_does_not_count_and_has_an_outcome(
    app: Application,
) -> None:
    app.sender.results.append(SendResult("permanent_failure", error_code="REJECTED"))
    await prepare_reply(app)
    await app.delivery.run_once()
    assert (await conversation(app)).preview_reply_count == 0
    assert await scalar(app, "SELECT status FROM turns") == "delivery_failed"
    assert (
        await scalar(app, "SELECT status FROM outbox_messages") == "permanent_failure"
    )


async def test_v2_known_rate_limit_retries_even_past_original_deadline(
    app: Application,
) -> None:
    app.sender.results.append(
        SendResult("retryable_failure", error_code="RATE_LIMIT", retry_after=40)
    )
    await prepare_reply(app)
    await app.delivery.run_once()
    app.clock.advance(41)
    await app.delivery.run_once()
    assert (await conversation(app)).preview_reply_count == 1
    assert len(app.sender.sent) == 2
    assert await scalar(app, "SELECT status FROM outbox_messages") == "delivered"
    assert (
        await scalar(
            app,
            "SELECT json_extract(proposal_json,'$.delivery_finished_at') FROM outbox_messages",
        )
        == app.clock.now()
    )


async def test_v2_expired_unsent_reply_is_sent_if_still_current(
    app: Application,
) -> None:
    await prepare_reply(app)
    app.clock.advance(31)
    await app.delivery.run_once()
    assert len(app.sender.sent) == 1
    assert await scalar(app, "SELECT status FROM outbox_messages") == "delivered"


async def test_v2_invitation_is_committed_only_when_delivered(app: Application) -> None:
    await prepare_reply(app, proposal={"platform_transition": True})
    current = await conversation(app)
    assert current.invitation_state == "planned"
    assert current.preview_closed_at is None
    await app.delivery.run_once()
    current = await conversation(app)
    assert current.invitation_state == "delivered"
    assert current.preview_closed_at == app.clock.now()


async def test_v2_cancelled_invitation_can_be_planned_again(app: Application) -> None:
    await prepare_reply(app, proposal={"platform_transition": True})
    await app.telegram.ingest([update(2, "wait, one more detail")])
    current = await conversation(app)
    assert current.invitation_state == "none"
    assert current.preview_closed_at is None
    assert await scalar(app, "SELECT status FROM outbox_messages") == "cancelled"


@pytest.mark.parametrize("origin", ["platform_redirect", "preview_closing"])
async def test_v2_dependent_notice_is_sent_after_final_reply(
    app: Application, origin: str
) -> None:
    current = await v2_conversation(app)

    async def seed(tx: Transaction) -> None:
        await tx.execute(
            "UPDATE conversations SET preview_reply_count=19 WHERE id=?", (current.id,)
        )

    await app.database.write(seed)
    turn, identifiers = await prepare_reply(app)

    async def notice(tx: Transaction) -> None:
        latest = await app.conversations.get(current.id, tx)
        await app.delivery_repository.prepare(
            tx,
            latest.snapshot,
            OutboxDraft(
                action_key=f"notice:{turn.id}",
                kind="service",
                text="The trial chat has ended",
                deadline=app.clock.now() + 30,
                predecessor_id=identifiers[0],
            ),
            app.clock.now(),
            {"origin": origin},
            turn.id,
        )

    await app.database.write(notice)
    assert (await conversation(app)).preview_closed_at is None
    await app.delivery.run_once()
    assert (await conversation(app)).preview_reply_count == 20
    await app.delivery.run_once()
    assert len(app.sender.sent) == 2
    assert (await conversation(app)).invitation_state == (
        "delivered" if origin == "platform_redirect" else "none"
    )


async def test_v2_kill_switch_releases_undelivered_invitation(app: Application) -> None:
    await prepare_reply(app, proposal={"platform_transition": True})
    await app.delivery_repository.kill_switch(True)
    assert (await conversation(app)).invitation_state == "none"
    assert (await conversation(app)).preview_closed_at is None


async def test_v2_unknown_invitation_is_not_automatically_repeated(
    app: Application,
) -> None:
    app.sender.results.append(SendResult("delivery_unknown"))
    await prepare_reply(app, proposal={"platform_transition": True})
    await app.delivery.run_once()
    current = await conversation(app)
    assert current.invitation_state == "unknown"
    assert current.preview_closed_at is None
    assert not (await app.commerce.transition_state(current))["eligible"]


@pytest.mark.parametrize(
    ("count", "eligible", "required"),
    [
        (13, False, False),
        (14, True, False),
        (18, True, False),
        (19, True, True),
        (20, False, False),
    ],
)
async def test_v2_invitation_window_uses_next_outgoing_number(
    app: Application, count: int, eligible: bool, required: bool
) -> None:
    current = await v2_conversation(app)

    async def seed(tx: Transaction) -> None:
        await tx.execute(
            "UPDATE conversations SET preview_reply_count=? WHERE id=?",
            (count, current.id),
        )

    await app.database.write(seed)
    state = await app.commerce.transition_state(await conversation(app))
    assert state["eligible"] is eligible
    assert state["required"] is required
    assert state["outgoing_number"] == count + 1


async def test_v2_cap_allows_last_parts_but_blocks_next_ordinary_reply(
    app: Application,
) -> None:
    current = await v2_conversation(app)

    async def seed(tx: Transaction) -> None:
        await tx.execute(
            "UPDATE conversations SET preview_reply_count=19 WHERE id=?", (current.id,)
        )

    await app.database.write(seed)
    await prepare_reply(app, texts=("One final thought", "I enjoyed that"))
    await app.delivery.run_once()
    await app.delivery.run_once()
    assert len(app.sender.sent) == 2
    assert (await conversation(app)).preview_reply_count == 20
    assert not app.commerce.can_reply(await conversation(app))
    await prepare_reply(app, number=2)
    await app.delivery.run_once()
    assert len(app.sender.sent) == 2
    assert (await conversation(app)).completed_pair_count == 1
    await app.telegram.ingest([update(3, "/privacy")])
    await app.delivery.run_once()
    assert len(app.sender.sent) == 3
    assert "ai roleplay simulation" in app.sender.sent[-1][1].lower()


async def test_v2_forget_resets_counters_and_outgoing_ledger(app: Application) -> None:
    await prepare_reply(app)
    await app.delivery.run_once()
    await app.telegram.ingest([update(2, "/forget")])
    await app.control_worker.run_once()
    current = await conversation(app)
    assert current.engine_version == "v2"
    assert current.preview_reply_count == 0
    assert current.completed_pair_count == 0
    assert current.pending_outgoing_turn_id is None
    assert current.first_reply_at is None
    assert current.dialogue_mode == "opening"
    assert await scalar(app, "SELECT COUNT(*) FROM outgoing_dialogue_turns") == 0


async def test_v2_timeout_during_outbox_commit_never_sends_success_and_failure(
    app: Application, monkeypatch: pytest.MonkeyPatch
) -> None:
    await v2_conversation(app)
    app.model.responses["dialogue"].append(
        {
            "parts": ["You caught me off guard"],
            "language": "en",
        }
    )
    app.agent.dialogue.settings = app.agent.dialogue.settings.model_copy(
        update={"turn_deadline_seconds": 0.1, "first_deadline_seconds": 0.1}
    )
    original = app.delivery_repository.prepare

    async def delayed(*args: Any, **kwargs: Any) -> str:
        draft = args[2]
        if draft.kind == "reply":
            await asyncio.sleep(0.2)
        return await original(*args, **kwargs)

    monkeypatch.setattr(app.delivery_repository, "prepare", delayed)
    await app.telegram.ingest([update(1, "hello")])
    await app.turn_worker.run_once()
    await app.delivery.run_once()
    assert len(app.sender.sent) == 1
    assert await scalar(app, "SELECT COUNT(*) FROM outbox_messages") == 1


@pytest.mark.parametrize("unknown", [False, True])
async def test_v2_raw_observation_is_preserved_and_delivery_is_confirmed_only(
    app: Application, unknown: bool
) -> None:
    if unknown:
        app.sender.results.append(SendResult("delivery_unknown"))
    turn, _ = await prepare_reply(app, texts=("I'm an AI", "That is the honest answer"))
    identifier = await app.agent.repository.start(turn, app.clock.now())
    metrics = {
        "engine_version": "v2",
        "research_observations": [
            {
                "first_raw_response": '{"parts":["I am a bot :)"]}',
                "first_parts": ["I am a bot :)"],
                "delivered_parts": [],
                "validation_errors": ["EMOJI_NOT_ALLOWED"],
            },
            {
                "first_raw_response": '{"parts":["I\'m an AI","That is the honest answer"]}',
                "first_parts": ["I'm an AI", "That is the honest answer"],
                "delivered_parts": [],
                "validation_errors": [],
            },
        ],
    }
    await app.agent.repository.finish(
        identifier, "prepared", app.clock.now(), None, metrics
    )

    async def attach(tx: Transaction) -> None:
        await tx.execute(
            "UPDATE outbox_messages SET proposal_json=json_set(proposal_json,'$.research_run_id',?) WHERE turn_id=?",
            (identifier, turn.id),
        )

    await app.database.write(attach)
    await app.delivery.run_once()
    first = json.loads(await scalar(app, "SELECT metrics_json FROM agent_runs"))
    assert first["research_observations"][0] == metrics["research_observations"][0]
    assert first["delivered_parts"] == ([] if unknown else ["I'm an AI"])
    await app.delivery.run_once()
    final = json.loads(await scalar(app, "SELECT metrics_json FROM agent_runs"))
    delivered = [] if unknown else ["I'm an AI", "That is the honest answer"]
    assert final["delivered_parts"] == delivered
    assert final["research_observations"][-1]["delivered_parts"] == delivered
    assert final["research_observations"][-1]["first_raw_response"] == (
        metrics["research_observations"][-1]["first_raw_response"]
    )


async def test_v2_run_finish_reprojects_already_delivered_parts(
    app: Application,
) -> None:
    turn, _ = await prepare_reply(app)
    identifier = await app.agent.repository.start(turn, app.clock.now())

    async def attach(tx: Transaction) -> None:
        await tx.execute(
            "UPDATE outbox_messages SET proposal_json=json_set(proposal_json,'$.research_run_id',?) WHERE turn_id=?",
            (identifier, turn.id),
        )

    await app.database.write(attach)
    await app.delivery.run_once()
    await app.agent.repository.finish(
        identifier,
        "prepared",
        app.clock.now(),
        None,
        {
            "research_observations": [
                {"first_parts": ["That was a good surprise"], "delivered_parts": []}
            ]
        },
    )
    metrics = json.loads(await scalar(app, "SELECT metrics_json FROM agent_runs"))
    assert metrics["delivered_parts"] == ["That was a good surprise"]
    assert metrics["research_observations"][0]["delivered_parts"] == [
        "That was a good surprise"
    ]


async def test_start_after_limit_explains_completion_without_resetting_it(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "Hello")])
    await app.drain()

    async def close_preview(tx: Transaction) -> None:
        await tx.execute("UPDATE conversations SET preview_reply_count=20")

    await app.database.write(close_preview)
    await app.telegram.ingest([update(2, "/start")])
    await app.drain()
    assert "trial conversation has ended" in app.sender.sent[-1][1]
    assert "/forget" in app.sender.sent[-1][1]
    assert (await conversation(app)).preview_reply_count == 20
    assert len(app.model.calls) == 1
