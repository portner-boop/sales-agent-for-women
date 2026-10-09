from pathlib import Path

from app.evaluation import EvaluationCase, evaluate_cases
from app.shared.config import load_settings
from tests.helpers import ScriptedModel


def test_all_regression_cases_are_valid_and_present() -> None:
    cases = [
        EvaluationCase.model_validate_json(line)
        for line in Path("docs/regression_cases.jsonl").read_text().splitlines()
    ]
    assert {case.id for case in cases} == {f"M{number:02}" for number in range(1, 31)}


async def test_evaluation_runner_reports_unknown_delivery_and_requires_human_review(
    tmp_path: Path,
) -> None:
    settings = load_settings()
    model = ScriptedModel()
    model.responses["evaluate"].append({"decision": "accept", "reason_codes": []})
    report = tmp_path / "evaluation.json"
    result = await evaluate_cases(
        settings, model, Path("docs/regression_cases.jsonl"), report, "M30"
    )
    assert result["grading"] == "automated_same_model_requires_human_review"
    assert result["total"] == 1
    payload = [data for task, data in model.calls if task == "evaluate"][0]
    assert payload["send_attempts"][0]["status"] == "delivery_unknown"
    assert len(payload["messages"]) == 1
    assert report.exists()


def test_dynaconf_nested_environment_and_secret_redaction(monkeypatch: object) -> None:
    monkeypatch.setenv("BEHAPPY_LLM__API_KEY", "configuration-test-key")
    monkeypatch.setenv("BEHAPPY_LLM__BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("BEHAPPY_LLM__MODEL", "openai/gpt-5.2")
    monkeypatch.setenv("BEHAPPY_RUNTIME__TESTER_IDS", "[123,456]")
    settings = load_settings()
    assert settings.llm.api_key.get_secret_value() == "configuration-test-key"
    assert "configuration-test-key" not in repr(settings)
    assert settings.runtime.tester_ids == frozenset({123, 456})
    settings.require_credentials(telegram=False)
    assert settings.llm.base_url == "https://gateway.example/v1"
    assert settings.llm.model == "openai/gpt-5.2"
