import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.modules.dialogue.replay import ReplayCase, ReplayStep, run_replay
from app.shared.config import RuntimeSettings, Settings, StorageSettings
from tests.helpers import ScriptedModel


async def test_replay_uses_delivery_timer_and_one_followup(tmp_path: Path) -> None:
    model = ScriptedModel()
    model.responses["dialogue"].extend(
        [
            {
                "parts": ["ну вот", "кот опять победил?"],
                "language": "ru",
            },
            {"parts": ["кресло официально его"], "language": "ru"},
            {
                "parts": ["до встречи"],
                "language": "ru",
                "closing": True,
            },
        ]
    )
    settings = Settings(
        storage=StorageSettings(directory=tmp_path / "unused"),
        runtime=RuntimeSettings(followup_enabled=True, followup_max_count=1),
    )
    case = ReplayCase(
        id="timer",
        steps=[
            ReplayStep(message="Кот занял кресло", language="ru"),
            ReplayStep(wait_seconds=61),
            ReplayStep(wait_seconds=120),
            ReplayStep(message="Мне пора, не пиши первым", language="ru"),
            ReplayStep(wait_seconds=90),
        ],
    )
    output = tmp_path / "replay.json"
    report = await run_replay([case], settings, model, output, max_calls=5)
    steps = report["cases"][0]["steps"]
    assert [a["text"] for a in steps[0]["actions"]] == ["ну вот", "кот опять победил?"]
    assert [a["text"] for a in steps[1]["actions"]] == ["кресло официально его"]
    assert steps[1]["actions"][0]["proposal"]["origin"] == "followup"
    assert not steps[2]["actions"]
    assert not steps[4]["actions"]
    assert steps[1]["state"]["preview_reply_count"] == 1
    assert report["llm_parameters"] == {
        "structured_mode": "json_object",
        "reasoning_effort": "none",
        "verbosity": "low",
        "output_tokens": 1200,
    }
    assert report["calls"] == 3
    assert model.calls[1][1]["followup"] is True
    assert json.loads(output.read_text())["status"] == "complete"
    with pytest.raises(FileExistsError):
        await run_replay([case], settings, model, output)
    assert len(model.calls) == 3


@pytest.mark.parametrize("fields", [{}, {"message": "hi", "wait_seconds": 60}])
def test_replay_rejects_ambiguous_steps(fields: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ReplayStep.model_validate(fields)


async def test_replay_preserves_partial_result_at_call_budget(tmp_path: Path) -> None:
    model = ScriptedModel()
    case = ReplayCase(
        id="budget",
        steps=[ReplayStep(message="hello"), ReplayStep(message="one more thing")],
    )
    report = await run_replay(
        [case], Settings(), model, tmp_path / "budget.json", max_calls=1
    )
    assert report["status"] == "budget_exhausted"
    assert report["calls"] == len(model.calls) == 1
    assert len(report["cases"][0]["steps"]) == 1
    assert report["cases"][0]["steps"][0]["actions"]
    assert report["attempts"][0]["draft"]["parts"]


async def test_replay_keeps_formal_rejection_and_repair_diagnostics(
    tmp_path: Path,
) -> None:
    model = ScriptedModel()
    model.responses["dialogue"].append({"parts": ["🙂"], "language": "en"})
    case = ReplayCase(id="repair", steps=[ReplayStep(message="hello")])
    report = await run_replay(
        [case], Settings(), model, tmp_path / "repair.json", max_calls=3
    )
    assert report["calls"] == 2
    attempts = report["cases"][0]["steps"][0]["attempts"]
    assert attempts[0]["metrics"]["validation_errors"] == ["EMOJI_NOT_ALLOWED"]
    assert attempts[1]["metrics"]["validation_errors"] == []
    assert report["cases"][0]["steps"][0]["actions"][0]["text"] != "🙂"
