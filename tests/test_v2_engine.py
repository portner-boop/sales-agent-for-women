from collections import deque
from pathlib import Path
from typing import Any

import pytest

from app.application import Application
from app.modules.agent.entities import SchemaT
from app.modules.dialogue.tokens import ContextBudget
from app.shared.clock import MutableClock
from app.shared.config import Settings
from tests.helpers import FakeSender, conversation, scalar, update


class DialogueModel:
    def __init__(self) -> None:
        self.responses: deque[dict[str, Any] | Exception] = deque()
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.remember = False

    async def complete(
        self,
        task: str,
        rules: str,
        data: dict[str, Any],
        schema: type[SchemaT],
        deadline_seconds: float,
    ) -> SchemaT:
        self.calls.append((task, rules, data))
        result = (
            self.responses.popleft()
            if self.responses
            else {
                "parts": ["You caught me off guard"],
                "language": "en",
            }
        )
        if isinstance(result, Exception):
            raise result
        if self.remember:
            source = data["original_messages"][-1]
            result["memory_candidates"] = [
                {
                    "predicate": "preferred_name",
                    "value": "Ann",
                    "message_id": source["id"],
                    "message_version": source["version"],
                    "evidence_text": source["text"],
                }
            ]
        return schema.model_validate(result)

    async def summarize(self, *args: Any) -> Any:
        raise AssertionError("Short sessions must not compact")


@pytest.fixture
async def v2_app(tmp_path: Path) -> Any:
    settings = Settings.model_validate(
        {
            "environment": "test",
            "storage": {"directory": tmp_path},
            "runtime": {
                "coalesce_seconds": 0,
                "reply_timing": {"enabled": False},
                "followup_enabled": True,
            },
        }
    )
    app = await Application.open(
        settings, FakeSender(), DialogueModel(), MutableClock(1900000000)
    )
    try:
        yield app
    finally:
        await app.close()


async def test_v2_single_call_and_original_full_history(v2_app: Application) -> None:
    await v2_app.telegram.ingest(
        [update(1, "My name is Ann"), update(2, "I like jazz")]
    )
    await v2_app.drain()
    await v2_app.telegram.ingest([update(3, "What did I tell you?")])
    await v2_app.drain()
    calls = v2_app.model.calls
    assert len(calls) == 2
    assert [item["content"] for item in calls[-1][2]["CHAT_MESSAGES"]] == [
        "My name is Ann",
        "I like jazz",
        "You caught me off guard",
        "What did I tell you?",
    ]
    assert calls[-1][2]["mode"] == "research_human"
    assert (
        await scalar(
            v2_app, "SELECT COUNT(*) FROM sqlite_master WHERE name='memory_summaries'"
        )
        == 0
    )


@pytest.mark.parametrize(
    "reply",
    ["I'm an AI", "this is an automated conversation, not a real man typing"],
)
async def test_v2_identity_admission_is_observed_and_not_rewritten(
    v2_app: Application,
    reply: str,
) -> None:
    v2_app.model.responses.append({"parts": [reply], "language": "en"})
    await v2_app.telegram.ingest([update(1, "Are you a human?")])
    await v2_app.drain()
    assert v2_app.sender.sent == [(1, reply)]
    assert len(v2_app.model.calls) == 1
    assert (
        await scalar(
            v2_app,
            "SELECT json_extract(metrics_json,'$.research_observations[0].identity_admission_detected') FROM agent_runs",
        )
        == 1
    )
    assert (
        await scalar(
            v2_app,
            "SELECT json_extract(metrics_json,'$.research_observations[0].identity_rewritten') FROM agent_runs",
        )
        == 0
    )


async def test_v2_formal_repair_is_bounded(v2_app: Application) -> None:
    v2_app.model.responses.append({"parts": ["Hello :)"], "language": "en"})
    await v2_app.telegram.ingest([update(1, "Hello")])
    await v2_app.drain()
    assert len(v2_app.model.calls) == 2
    assert v2_app.model.calls[1][2]["REPAIR_ERRORS"] == ["EMOJI_NOT_ALLOWED"]
    assert v2_app.sender.sent == [(1, "You caught me off guard")]


async def test_v2_exhaustion_delivers_service_outcome_and_next_input_works(
    v2_app: Application,
) -> None:
    v2_app.model.responses.extend([ValueError("invalid"), ValueError("invalid again")])
    await v2_app.telegram.ingest([update(1, "Hello")])
    await v2_app.drain()
    assert len(v2_app.model.calls) == 2
    assert len(v2_app.sender.sent) == 1
    assert "try again" in v2_app.sender.sent[0][1]
    assert (await conversation(v2_app)).preview_reply_count == 0
    await v2_app.telegram.ingest([update(2, "Are you there?")])
    await v2_app.drain()
    assert len(v2_app.model.calls) == 3
    assert v2_app.sender.sent[-1][1] == "You caught me off guard"


async def test_v2_memory_projection_failure_does_not_block_response(
    v2_app: Application, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fail(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("projection unavailable")

    monkeypatch.setattr(v2_app.memory, "resolve", fail)
    v2_app.model.remember = True
    await v2_app.telegram.ingest([update(1, "My name is Ann")])
    await v2_app.drain()
    assert v2_app.sender.sent[-1][1] == "You caught me off guard"
    assert len(v2_app.model.calls) == 1


async def test_v2_fifteen_to_twenty_and_no_twenty_first(v2_app: Application) -> None:
    for number in range(1, 22):
        await v2_app.telegram.ingest([update(number, f"A new detail {number}")])
        await v2_app.drain()
    current = await conversation(v2_app)
    assert len(v2_app.model.calls) == 20
    assert current.preview_reply_count == 20
    assert current.completed_pair_count == 20
    assert v2_app.model.calls[13][2]["transition"]["eligible"] is False
    assert v2_app.model.calls[14][2]["transition"]["eligible"] is True
    assert current.invitation_state == "delivered"
    assert (
        sum("https://www.behappy2day.com/" in text for _, text in v2_app.sender.sent)
        == 1
    )


async def test_v2_two_followups_do_not_consume_a_slot(v2_app: Application) -> None:
    await v2_app.telegram.ingest([update(1, "Tell me something")])
    await v2_app.drain()
    v2_app.clock.advance(60)
    await v2_app.drain()
    assert len(v2_app.model.calls) == 2
    assert v2_app.model.calls[-1][2]["followup"] is True
    assert (await conversation(v2_app)).preview_reply_count == 1
    v2_app.clock.advance(120)
    await v2_app.drain()
    assert len(v2_app.model.calls) == 3
    assert v2_app.model.calls[-1][2]["state"]["followup_opportunity"]["ordinal"] == 2
    assert (await conversation(v2_app)).preview_reply_count == 1
    v2_app.clock.advance(300)
    await v2_app.drain()
    assert len(v2_app.model.calls) == 3


async def test_v2_old_accepted_input_gets_notice(v2_app: Application) -> None:
    await v2_app.telegram.ingest([update(1, "Hello")])
    v2_app.clock.advance(181)
    await v2_app.drain()
    assert not v2_app.model.calls
    assert len(v2_app.sender.sent) == 1


async def test_v2_overflow_is_visible_and_history_is_not_summarized(
    v2_app: Application,
) -> None:
    v2_app.agent.dialogue.builder = ContextBudget(
        v2_app.settings.memory.model_copy(update={"context_tokens": 2000}),
        v2_app.agent.dialogue.builder.counter,
    )
    await v2_app.telegram.ingest([update(1, "Hello")])
    await v2_app.drain()
    assert not v2_app.model.calls
    assert "context" in v2_app.sender.sent[-1][1]


async def test_followup_at_nineteen_does_not_close_preview(v2_app: Application) -> None:
    await v2_app.telegram.ingest([update(1, "A detail")])
    await v2_app.drain()

    async def set_count(tx: Any) -> None:
        await tx.execute("UPDATE conversations SET preview_reply_count=19")

    await v2_app.database.write(set_count)
    v2_app.clock.advance(60)
    await v2_app.drain()
    current = await conversation(v2_app)
    assert len(v2_app.model.calls) == 2
    assert current.preview_reply_count == 19
    assert current.dialogue_mode != "closing"
    assert all("trial conversation" not in text for _, text in v2_app.sender.sent)


async def test_boundary_flags_persist_without_optional_memory_candidate(
    v2_app: Application,
) -> None:
    v2_app.model.responses.append(
        {
            "parts": ["Of course"],
            "language": "en",
            "no_proactive": True,
            "no_questions": True,
            "transition_blocked": True,
        }
    )
    await v2_app.telegram.ingest(
        [update(1, "No questions or followups, and no platform link")]
    )
    await v2_app.drain()
    current = await conversation(v2_app)
    facts = await v2_app.memory.repository.items(current.scope)
    assert {item["predicate"] for item in facts} == {
        "followup_permission",
        "question_permission",
        "offer_refusal",
    }
    assert all(
        item["sources"][0]["evidence_text"]
        == "No questions or followups, and no platform link"
        for item in facts
    )
    await v2_app.telegram.ingest([update(2, "Something else")])
    await v2_app.drain()
    assert len(v2_app.model.calls[-1][2]["boundaries"]) == 3
    v2_app.clock.advance(60)
    await v2_app.drain()
    assert len(v2_app.model.calls) == 2
    assert not (await v2_app.commerce.transition_state(await conversation(v2_app)))[
        "eligible"
    ]


async def test_handoff_uses_latest_confirmed_history(v2_app: Application) -> None:
    from app.modules.dialogue.handoff import InMemoryHandoffReceiver

    await v2_app.telegram.ingest([update(1, "First detail")])
    await v2_app.turn_worker.run_once()
    current = await conversation(v2_app)
    before = await v2_app.handoff_snapshot(current.scope)
    assert [message.role for message in before.messages] == ["user"]
    await v2_app.drain()
    after = await v2_app.handoff_snapshot(current.scope)
    assert [message.role for message in after.messages] == ["user", "assistant"]
    assert after.preview_state["completed_replies"] == 1
    assert after.policy_version == "marko-dialogue-v55"
    assert after.restricted_persona["location"]["residence"] == "Split, Croatia"
    receiver = InMemoryHandoffReceiver()
    await receiver.receive(after)
    assert receiver.snapshots[0].model_dump() == after.model_dump()


async def test_descriptive_action_metadata_does_not_discard_usable_reply(
    v2_app: Application,
) -> None:
    text = "Eight and twelve, sounds like there’s plenty of life in the house"
    v2_app.model.responses.append(
        {
            "parts": [text],
            "language": "en",
        }
    )
    await v2_app.telegram.ingest([update(1, "My children are eight and twelve")])
    await v2_app.drain()
    assert v2_app.sender.sent == [(1, text)]
    assert len(v2_app.model.calls) == 1


async def test_invalid_optional_memory_does_not_discard_good_reply(
    v2_app: Application,
) -> None:
    v2_app.model.responses.append(
        {
            "parts": ["Got it, Luna is four"],
            "language": "English",
            "memory_candidates": [
                {
                    "predicate": "cat_age",
                    "value": 4,
                    "message_id": "missing-version",
                    "evidence_text": "four",
                }
            ],
        }
    )
    await v2_app.telegram.ingest([update(1, "Luna is four")])
    await v2_app.drain()
    assert v2_app.sender.sent == [(1, "Got it, Luna is four")]
    assert len(v2_app.model.calls) == 1
    assert (await conversation(v2_app)).language == "en"
    assert (
        await scalar(
            v2_app,
            "SELECT json_array_length(metrics_json,'$.memory_errors') FROM agent_runs",
        )
        > 0
    )


async def test_three_natural_bubbles_are_delivered_unchanged_as_one_turn(
    v2_app: Application,
) -> None:
    parts = [
        "ну всё",
        "теперь кресло принадлежит коту",
        "тебе оставили хоть подлокотник?",
    ]
    v2_app.model.responses.append({"parts": parts, "language": "ru"})
    await v2_app.telegram.ingest([update(1, "Кот занял кресло", language="ru")])
    await v2_app.drain()
    current = await conversation(v2_app)
    assert [text for _, text in v2_app.sender.sent] == parts
    assert current.preview_reply_count == 1
    delivered = await v2_app.conversations.messages(current.scope)
    assert [item.text for item in delivered if item.role == "assistant"] == parts
    assert len(v2_app.model.calls) == 1
    assert (
        await scalar(v2_app, "SELECT COUNT(*) FROM followups WHERE status='pending'")
        == 1
    )


async def test_first_failure_notice_uses_language_hint_without_overwriting_session(
    v2_app: Application,
) -> None:
    v2_app.model.responses.extend([TimeoutError(), TimeoutError()])
    await v2_app.telegram.ingest([update(1, "Привет", language="ru")])
    await v2_app.drain()
    assert (
        v2_app.sender.sent[-1][1]
        == "Не удалось отправить ответ. Попробуй написать ещё раз"
    )
    assert (await conversation(v2_app)).language == "ru"
    v2_app.model.responses.extend([TimeoutError(), TimeoutError()])
    await v2_app.telegram.ingest([update(2, "Ещё раз", language="es")])
    await v2_app.drain()
    assert (await conversation(v2_app)).language == "ru"
    assert v2_app.sender.sent[-1][1].startswith("Не удалось")


async def test_failure_after_clear_language_switch_uses_current_request_language(
    v2_app: Application,
) -> None:
    v2_app.model.responses.append(
        {"parts": ["me alegra que escribieras"], "language": "es"}
    )
    await v2_app.telegram.ingest([update(1, "hola", language="es")])
    await v2_app.drain()
    v2_app.model.responses.extend([TimeoutError(), TimeoutError()])
    await v2_app.telegram.ingest(
        [
            update(
                2,
                "Теперь давай по-русски, расскажи мне что-нибудь интересное о себе и о том, чем ты любишь заниматься",
                language="ru",
            )
        ]
    )
    await v2_app.drain()
    assert (await conversation(v2_app)).language == "es"
    assert v2_app.sender.sent[-1][1].startswith("Не удалось")


async def test_rendered_delivery_is_the_history_while_raw_response_stays_observable(
    v2_app: Application,
) -> None:
    v2_app.model.responses.append(
        {"parts": ["Nora, you caught me off guard."], "language": "en"}
    )
    await v2_app.telegram.ingest([update(1, "I'm Nora")])
    await v2_app.drain()
    current = await conversation(v2_app)
    assert v2_app.sender.sent[-1][1] == "Nora, you caught me off guard"
    history = await v2_app.conversations.messages(current.scope)
    assert history[-1].text == v2_app.sender.sent[-1][1]
    assert (
        await scalar(
            v2_app,
            "SELECT json_extract(metrics_json,'$.research_observations[0].first_parts[0]') FROM agent_runs",
        )
        == "Nora, you caught me off guard."
    )


async def test_no_questions_allows_related_followup(v2_app: Application) -> None:
    v2_app.model.responses.extend(
        [
            {"parts": ["just having coffee"], "language": "en", "no_questions": True},
            {"parts": ["a quiet start suits me too"], "language": "en"},
        ]
    )
    await v2_app.telegram.ingest([update(1, "No questions for now, just talk to me")])
    await v2_app.drain()
    v2_app.clock.advance(60)
    await v2_app.drain()
    assert v2_app.sender.sent[-1][1] == "a quiet start suits me too"
    assert v2_app.model.calls[-1][2]["followup"] is True
    assert any(
        item["predicate"] == "question_permission" and item["value"] is False
        for item in v2_app.model.calls[-1][2]["boundaries"]
    )
    assert (await conversation(v2_app)).preview_reply_count == 1


async def test_question_permission_and_proactive_permission_are_independent(
    v2_app: Application,
) -> None:
    v2_app.model.responses.extend(
        [
            {
                "parts": ["jazz suits a slow evening"],
                "continuation": "which song do you put on first?",
                "language": "en",
                "allow_questions": True,
                "no_proactive": True,
            },
            {
                "parts": ["i like the quiet around it"],
                "language": "en",
                "allow_proactive": True,
            },
            {"parts": ["do you listen while you cook too?"], "language": "en"},
        ]
    )
    await v2_app.telegram.ingest(
        [update(1, "Questions are fine, I cook with jazz, but don't message me first")]
    )
    await v2_app.drain()
    assert v2_app.sender.sent[-1][1] == "which song do you put on first?"
    v2_app.clock.advance(180)
    await v2_app.drain()
    assert len(v2_app.model.calls) == 1
    await v2_app.telegram.ingest([update(2, "You can message me first again")])
    await v2_app.drain()
    v2_app.clock.advance(60)
    await v2_app.drain()
    assert len(v2_app.model.calls) == 3
    assert v2_app.sender.sent[-1][1] == "do you listen while you cook too?"
    assert (await conversation(v2_app)).preview_reply_count == 2


@pytest.mark.parametrize(("count", "suggest"), [(14, True), (18, True), (19, False)])
async def test_invitation_is_chosen_in_window_or_added_at_final_reply(
    v2_app: Application, count: int, suggest: bool
) -> None:
    app = v2_app
    await app.telegram.ingest([update(1, "hello")])
    await app.drain()

    async def seed(tx: Any) -> None:
        await tx.execute("UPDATE conversations SET preview_reply_count=?", (count,))

    await app.database.write(seed)
    app.model.responses.append(
        {
            "parts": ["id like to keep talking with you"],
            "language": "en",
            "suggest_transition": suggest,
        }
    )
    await app.telegram.ingest([update(2, "id like to hear more")])
    await app.drain()
    current = await conversation(app)
    assert current.invitation_state == "delivered"
    assert current.preview_reply_count == count + 1
    assert (
        sum("https://www.behappy2day.com/" in text for _, text in app.sender.sent) == 1
    )
    assert "paid" in app.sender.sent[-1][1].lower()
    assert len(app.model.calls) == 2
    await app.telegram.ingest([update(3, "another message")])
    await app.drain()
    assert len(app.model.calls) == 2


async def test_explicit_pause_suppresses_final_invitation(v2_app: Application) -> None:
    app = v2_app
    await app.telegram.ingest([update(1, "hello")])
    await app.drain()

    async def seed(tx: Any) -> None:
        await tx.execute("UPDATE conversations SET preview_reply_count=19")

    await app.database.write(seed)
    app.model.responses.append(
        {"parts": ["go on"], "language": "en", "pause_requested": True}
    )
    await app.telegram.ingest([update(2, "wait, let me finish")])
    await app.drain()
    assert (await conversation(app)).invitation_state == "none"
    assert all("https://" not in text for _, text in app.sender.sent)
    app.clock.advance(200)
    await app.drain()
    assert len(app.model.calls) == 2


async def test_inherited_contact_boundary_blocks_invitation_eligibility(
    v2_app: Application,
) -> None:
    app = v2_app
    app.model.responses.append(
        {"parts": ["understood"], "language": "en", "no_proactive": True}
    )
    await app.telegram.ingest([update(1, "dont send unsolicited messages")])
    await app.drain()

    async def seed(tx: Any) -> None:
        await tx.execute("UPDATE conversations SET preview_reply_count=14")

    await app.database.write(seed)
    await app.telegram.ingest([update(2, "what did you cook?")])
    await app.drain()
    assert app.model.calls[-1][2]["transition"]["eligible"] is False
    assert (await conversation(app)).invitation_state == "none"
