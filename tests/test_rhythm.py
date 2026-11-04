from pathlib import Path
from typing import Any

from app.application import Application
from app.modules.agent.dialogue_engine import reply_shape
from app.modules.dialogue.prompts import rules_for_mode
from app.shared.clock import MutableClock
from app.shared.config import Settings
from tests.helpers import FakeSender, update
from tests.test_v2_engine import DialogueModel


async def open_app(tmp_path: Path, **runtime: Any) -> Application:
    settings = Settings.model_validate(
        {
            "environment": "test",
            "storage": {"directory": tmp_path},
            "runtime": {
                "coalesce_seconds": 0,
                "reply_timing": {"enabled": False},
                "followup_enabled": True,
                **runtime,
            },
        }
    )
    return await Application.open(
        settings, FakeSender(), DialogueModel(), MutableClock(1900000000)
    )


def test_reply_shape_varies_and_first_reply_may_ask() -> None:
    shapes = [reply_shape(first=False) for _ in range(300)]
    assert {shape["messages"] for shape in shapes} == {1, 2, 3}
    assert {shape["end_with_question"] for shape in shapes} == {True, False}
    assert all(reply_shape(first=True)["end_with_question"] for _ in range(20))


async def test_ordinary_turn_carries_reply_shape_but_timer_does_not(
    tmp_path: Path,
) -> None:
    app = await open_app(tmp_path, reply_shape_enabled=True)
    try:
        await app.telegram.ingest([update(1, "Tell me something")])
        await app.drain()
        assert app.model.calls[0][2]["state"]["reply_shape"]["messages"] in {1, 2, 3}
        app.clock.advance(60)
        await app.drain()
        assert app.model.calls[-1][2]["followup"] is True
        assert "reply_shape" not in app.model.calls[-1][2]["state"]
    finally:
        await app.close()


async def test_third_followup_waits_longer_and_is_configurable(tmp_path: Path) -> None:
    app = await open_app(
        tmp_path,
        followup_max_count=3,
        followup_second_seconds=300,
        followup_third_seconds=1800,
    )
    try:
        await app.telegram.ingest([update(1, "Tell me something")])
        await app.drain()
        app.clock.advance(60)
        await app.drain()
        app.clock.advance(240)
        await app.drain()
        assert len(app.model.calls) == 3
        app.clock.advance(1000)
        await app.drain()
        assert len(app.model.calls) == 3
        app.clock.advance(500)
        await app.drain()
        assert len(app.model.calls) == 4
        state = app.model.calls[-1][2]["state"]["followup_opportunity"]
        assert state["ordinal"] == 3
        assert len(state["previous_messages"]) == 2
        assert "THIRD TIMER" in rules_for_mode(followup_ordinal=3)
    finally:
        await app.close()
