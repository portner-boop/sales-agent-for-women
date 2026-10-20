from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from app.application import Application
from app.shared.clock import MutableClock
from app.shared.config import MemorySettings, Settings
from tests.helpers import FakeSender, ScriptedModel


@pytest.fixture
def context_token_budget() -> int:
    return MemorySettings().context_tokens


@pytest.fixture
async def app(tmp_path: Path, context_token_budget: int) -> AsyncIterator[Application]:
    settings = Settings.model_validate(
        {
            "environment": "test",
            "storage": {"directory": tmp_path},
            "runtime": {
                "coalesce_seconds": 0,
                "reply_timing": {"enabled": False},
            },
            "memory": {"context_tokens": context_token_budget},
        }
    )
    instance = await Application.open(
        settings, FakeSender(), ScriptedModel(), MutableClock(1900000000)
    )
    try:
        yield instance
    finally:
        await instance.close()
