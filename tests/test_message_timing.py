import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.application import Application
from app.modules.delivery.timing import DeliveryTiming
from app.modules.dialogue.context import build_request
from app.modules.dialogue.schemas import DialogueDraft
from app.modules.dialogue.validation import validate_draft
from app.shared.clock import MutableClock
from app.shared.config import ReplyTimingSettings, Settings
from app.shared.storage import fetchall
from tests.helpers import FakeSender, ScriptedModel, conversation, scalar, update
from tests.test_v2_dialogue import message, session


@pytest.mark.parametrize(
    ("length", "seconds"), [(0, 5), (12, 7), (60, 23), (140, 149 / 3), (500, 60)]
)
def test_phone_typing_examples(length: int, seconds: float) -> None:
    timing = DeliveryTiming(ReplyTimingSettings(), lambda low, high: (low + high) / 2)
    assert timing.duration("a" * length) == pytest.approx(seconds)


@pytest.mark.parametrize(("edge", "seconds"), [(0, 17.6), (1, 28.8)])
def test_pause_and_twenty_percent_jitter(edge: int, seconds: float) -> None:
    timing = DeliveryTiming(ReplyTimingSettings(), lambda low, high: (low, high)[edge])
    assert timing.duration("a" * 60) == pytest.approx(seconds)


@pytest.mark.parametrize("character", ["a", "я", "水"])
def test_timing_counts_characters_not_bytes(character: str) -> None:
    timing = DeliveryTiming(ReplyTimingSettings(), lambda low, high: (low + high) / 2)
    assert timing.duration(character * 60) == 23


@pytest.mark.parametrize(
    "values",
    [
        {"characters_per_second": 0},
        {"characters_per_second": float("inf")},
        {"jitter_fraction": 1},
        {"pause_min_seconds": 5, "pause_max_seconds": 2},
        {"minimum_seconds": 61},
    ],
)
def test_invalid_timing_configuration(values: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        ReplyTimingSettings.model_validate(values)


@pytest.fixture
async def timed_app(tmp_path: Path) -> AsyncIterator[Application]:
    app = await Application.open(
        Settings.model_validate(
            {
                "environment": "test",
                "storage": {"directory": tmp_path},
                "runtime": {"coalesce_seconds": 0, "followup_enabled": True},
            }
        ),
        FakeSender(),
        ScriptedModel(),
        MutableClock(1900000000),
    )
    assert app.settings.runtime.reply_timing.enabled
    try:
        yield app
    finally:
        await app.close()


async def queued(app: Application) -> list[Any]:
    async with app.database.read() as tx:
        return await fetchall(
            tx, "SELECT * FROM outbox_messages WHERE status='ready' ORDER BY rowid"
        )


async def send_next(app: Application) -> None:
    current = await conversation(app)
    due = await app.delivery_repository.next_ready_at(current.scope)
    assert due is not None
    app.clock.advance(max(0, due - app.clock.now()))
    await app.delivery.run_once()


async def first_reply(app: Application) -> None:
    await app.telegram.ingest([update(1, "hello")])
    await app.drain()
    await send_next(app)


async def test_first_bubble_meets_cap_and_later_bubble_types_separately(
    timed_app: Application,
) -> None:
    app = timed_app
    app.model.responses["dialogue"].append(
        {"parts": ["a" * 140], "continuation": "b" * 140, "language": "en"}
    )
    received = app.clock.now()
    await app.telegram.ingest([update(1, "hello")])
    await app.drain()
    rows = await queued(app)
    assert not app.sender.sent
    first, second = rows
    assert app.settings.runtime.first_deadline_seconds == 15
    assert first["created_at"] <= received + 14
    assert json.loads(first["proposal_json"])["first_reply_target_at"] == received + 15
    assert json.loads(first["proposal_json"])["hard_deadline"] is False
    assert json.loads(second["proposal_json"])["hard_deadline"] is False
    assert second["deadline"] > first["deadline"]
    assert len(await app.delivery_repository.waiting_parts(app.clock.now())) == 1
    await send_next(app)
    assert app.clock.now() <= received + 15
    assert [text for _, text in app.sender.sent] == ["a" * 140]
    assert await scalar(app, "SELECT COUNT(*) FROM followups") == 0
    second = (await queued(app))[0]
    delay = second["message_delay_seconds"]
    assert 38 <= delay <= 60
    assert second["created_at"] == pytest.approx(app.clock.now() + delay)
    assert len(await app.delivery_repository.waiting_parts(app.clock.now())) == 1
    await app.delivery.run_once()
    assert len(app.sender.sent) == 1
    await send_next(app)
    assert [text for _, text in app.sender.sent] == ["a" * 140, "b" * 140]
    assert (await conversation(app)).preview_reply_count == 1
    assert await scalar(app, "SELECT run_at FROM followups") == app.clock.now() + 60


@pytest.mark.parametrize(("length", "generation"), [(60, 8), (12, 18)])
async def test_generation_and_queue_time_are_not_added_twice(
    timed_app: Application, length: int, generation: float
) -> None:
    app = timed_app
    await first_reply(app)
    original = app.model.complete

    async def delayed(*args: Any, **kwargs: Any) -> Any:
        app.clock.advance(generation)
        return await original(*args, **kwargs)

    app.model.complete = delayed
    app.model.responses["dialogue"].append({"parts": ["a" * length], "language": "en"})
    await app.telegram.ingest([update(2, "one more thing")])
    app.clock.advance(2)
    await app.turn_worker.run_once()
    row = (await queued(app))[0]
    timing = json.loads(row["proposal_json"])["timing"]
    assert timing["elapsed_seconds"] >= generation + 2
    assert timing["delay_seconds"] == pytest.approx(
        max(0, timing["target_seconds"] - timing["elapsed_seconds"])
    )
    if generation == 18:
        assert timing["delay_seconds"] == 0
    await send_next(app)
    assert app.sender.sent[-1][1] == "a" * length


@pytest.mark.parametrize("command", ["wait, another detail", "/stop", "/forget"])
async def test_new_input_and_controls_cancel_typing(
    timed_app: Application, command: str
) -> None:
    app = timed_app
    app.model.responses["dialogue"].append({"parts": ["old draft"], "language": "en"})
    await app.telegram.ingest([update(1, "hello")])
    await app.drain()
    assert not app.sender.sent
    await app.telegram.ingest([update(2, command)])
    await app.drain()
    app.clock.advance(60)
    await app.drain()
    assert "old draft" not in [text for _, text in app.sender.sent]


async def test_timing_survives_restart_without_regeneration(
    timed_app: Application,
) -> None:
    app = timed_app
    await app.telegram.ingest([update(1, "hello")])
    await app.drain()
    before = (await queued(app))[0]
    await app.close()
    reopened = await Application.open(app.settings, app.sender, app.model, app.clock)
    try:
        after = (await queued(reopened))[0]
        assert after["created_at"] == before["created_at"]
        assert after["message_delay_seconds"] == before["message_delay_seconds"]
        await send_next(reopened)
        assert len(app.sender.sent) == 1
        assert len(app.model.calls) == 1
    finally:
        await reopened.close()


async def test_final_parts_finish_before_service_and_do_not_schedule_followup(
    timed_app: Application,
) -> None:
    app = timed_app
    await first_reply(app)

    async def seed(tx: Any) -> None:
        await tx.execute("UPDATE conversations SET preview_reply_count=19")

    await app.database.write(seed)
    app.model.responses["dialogue"].append(
        {"parts": ["a" * 140, "b" * 140], "language": "en", "transition_blocked": True}
    )
    await app.telegram.ingest([update(2, "I won't use another site")])
    await app.drain()
    assert app.model.calls[-1][1]["transition"]["last_reply"] is True
    await send_next(app)
    assert (await conversation(app)).preview_reply_count == 20
    await send_next(app)
    await app.delivery.run_once()
    assert [text for _, text in app.sender.sent][-3:] == [
        "a" * 140,
        "b" * 140,
        "The trial conversation here has ended",
    ]
    assert (
        await scalar(app, "SELECT COUNT(*) FROM followups WHERE status='pending'") == 0
    )


def test_last_reply_cannot_open_a_continuation_when_platform_declined() -> None:
    request = build_request(
        session(), [], [message(1)], transition={"last_reply": True, "eligible": False}
    )
    draft = DialogueDraft(
        parts=["lovely evening"], continuation="what music?", language="en"
    )
    assert "CLOSING_CONTINUATION_NOT_ALLOWED" in validate_draft(draft, request)


def test_last_reply_schema_requires_null_continuation_and_timer_stays_separate() -> (
    None
):
    from app.modules.dialogue.schemas import ClosingDialogueDraft, response_schema

    request = build_request(
        session(), [], [message(1)], transition={"last_reply": True, "eligible": False}
    )
    assert response_schema(request) is ClosingDialogueDraft
    with pytest.raises(ValidationError):
        response_schema(request)(
            parts=["good evening"], continuation="what music?", language="en"
        )
    request.followup = True
    assert response_schema(request) is DialogueDraft


async def test_first_reply_can_finish_after_the_previous_ten_second_window(
    timed_app: Application,
) -> None:
    app = timed_app
    original = app.model.complete

    async def delayed(*args: Any, **kwargs: Any) -> Any:
        app.clock.advance(11)
        return await original(*args, **kwargs)

    app.model.complete = delayed
    app.model.responses["dialogue"].append({"parts": ["hello again"], "language": "en"})
    received = app.clock.now()
    await app.telegram.ingest([update(1, "hello")])
    await app.drain()
    if not app.sender.sent:
        await send_next(app)
    assert [text for _, text in app.sender.sent] == ["hello again"]
    assert received + 11 <= app.clock.now() <= received + 15
    assert len(app.model.calls) == 1
