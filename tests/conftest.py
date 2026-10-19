from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from app.application import Application
from app.modules.agent.style_plan import StylePolicy, style_policy
from app.shared.clock import MutableClock
from app.shared.config import MemorySettings, Settings
from tests.helpers import FakeSender, ScriptedModel


@pytest.fixture
def casual_text_edits() -> bool:
    return False


@pytest.fixture
def full_letter_typo_rate(monkeypatch: pytest.MonkeyPatch) -> StylePolicy:
    policy = style_policy().model_copy(update={"letter_typo_keep_probability": 1})
    monkeypatch.setattr("app.modules.agent.style_plan.style_policy", lambda: policy)
    return policy


@pytest.fixture
def fixed_style_interval(
    monkeypatch: pytest.MonkeyPatch, full_letter_typo_rate: StylePolicy
) -> None:
    policy = full_letter_typo_rate.model_copy(update={"interval_jitter": 0})
    monkeypatch.setattr("app.modules.agent.style_plan.style_policy", lambda: policy)


@pytest.fixture
def context_token_budget() -> int:
    return MemorySettings().context_tokens


@pytest.fixture
async def app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    casual_text_edits: bool,
    context_token_budget: int,
) -> AsyncIterator[Application]:
    if not casual_text_edits:
        monkeypatch.setattr(
            "app.modules.agent.service.apply_casual_imperfection",
            lambda proposal, protected_texts: proposal,
        )
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
                "context_tokens": context_token_budget,
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
