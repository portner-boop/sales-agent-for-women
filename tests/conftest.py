from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from app.application import Application
from app.shared.clock import MutableClock
from app.shared.config import Settings
from tests.helpers import FakeSender, ScriptedModel


@pytest.fixture
async def app(tmp_path: Path) -> AsyncIterator[Application]:
    settings = Settings.model_validate(
        {
            "environment": "test",
            "storage": {"directory": tmp_path},
            "runtime": {
                "tester_ids": [1, 2],
                "coalesce_seconds": 0,
                "reply_pause_min_seconds": 2,
                "reply_pause_max_seconds": 4,
                "reply_pacing_enabled": False,
            },
            "memory": {
                "history_threshold_tokens": 700,
                "block_tokens": 650,
                "summary_tokens": 256,
            },
        }
    )
    instance = await Application.open(
        settings, FakeSender(), ScriptedModel(), MutableClock(1900000000)
    )
    try:
        yield instance
    finally:
        await instance.close()
