import asyncio
from pathlib import Path

import pytest

from app.application import Application
from app.modules.dialogue.tokens import TokenCounter
from app.shared.clock import MutableClock
from app.shared.config import LLMSettings, Settings
from tests.helpers import FakeSender, ScriptedModel, conversation, update


async def test_five_users_keep_separate_role_histories(tmp_path: Path) -> None:
    model = ScriptedModel()
    model.release = asyncio.Event()
    settings = Settings.model_validate(
        {
            "environment": "test",
            "storage": {"directory": tmp_path},
            "runtime": {"coalesce_seconds": 0, "reply_timing": {"enabled": False}},
        }
    )
    app = await Application.open(
        settings, FakeSender(), model, MutableClock(1900000000)
    )
    try:
        await app.telegram.ingest(
            [update(i, f"private detail user-{i}", user=i) for i in range(1, 6)]
        )
        workers = [asyncio.create_task(app.turn_worker.run_once()) for _ in range(5)]
        for _ in range(100):
            if len(model.calls) == 5:
                break
            await asyncio.sleep(0.01)
        assert len(model.calls) == 5
        assert len(app.agent.active) == 5
        model.release.set()
        await asyncio.gather(*workers)
        await app.drain()
        assert len(app.sender.sent) == 5
        assert {data["CHAT_MESSAGES"][0]["content"] for _, data in model.calls} == {
            f"private detail user-{i}" for i in range(1, 6)
        }
        assert all(len(data["CHAT_MESSAGES"]) == 1 for _, data in model.calls)
        assert len({(await conversation(app, i)).id for i in range(1, 6)}) == 5
    finally:
        model.release.set()
        await app.close()


async def test_application_accepts_users_without_a_participant_list(
    tmp_path: Path,
) -> None:
    settings = Settings.model_validate(
        {
            "storage": {"directory": tmp_path},
            "runtime": {"coalesce_seconds": 0, "reply_timing": {"enabled": False}},
        }
    )
    app = await Application.open(settings, FakeSender(), ScriptedModel())
    try:
        await app.telegram.ingest([update(1, "Hello", user=987654321)])
        await app.drain()
        assert len(app.model.calls) == 1
        assert app.sender.sent[0][0] == 987654321
    finally:
        await app.close()


def test_engine_and_identity_have_no_configuration_switches() -> None:
    for runtime in ({"dialogue_engine": "legacy"}, {"persona_mode": "client_virtual"}):
        with pytest.raises(ValueError):
            Settings.model_validate({"runtime": runtime})
    assert "persona_mode" not in type(Settings().runtime).model_fields
    assert "dialogue_engine" not in type(Settings().runtime).model_fields
    assert TokenCounter("gateway/custom-model").count("hello") > 0
    assert not LLMSettings().model
    assert not LLMSettings().base_url


async def test_public_user_prepared_reply_survives_restart(app: Application) -> None:
    await app.telegram.ingest([update(1, "Hello", user=987654321)])
    await app.turn_worker.run_once()
    await app.close()
    reopened = await Application.open(app.settings, app.sender, app.model, app.clock)
    try:
        await reopened.drain()
        assert app.sender.sent[0][0] == 987654321
        assert len(app.model.calls) == 1
    finally:
        await reopened.close()
