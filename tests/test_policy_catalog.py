import json
from pathlib import Path

import pytest

from app.modules.agent.context import TokenCounter
from app.modules.agent.policy import (
    POLICY_DIRECTORY,
    PolicyCatalog,
    RuleSelector,
    select_examples,
)
from app.modules.agent.schemas import TurnInterpretation
from scripts.qmd_policy import QmdPolicySelector


def test_catalog_manifest_and_dependencies() -> None:
    catalog = PolicyCatalog.load()
    assert json.loads((POLICY_DIRECTORY / "manifest.json").read_text()) == json.loads(
        json.dumps(catalog.manifest())
    )
    for rule in catalog.rules:
        assert all(Path(path).exists() for path in rule.tests)
    selector = RuleSelector(catalog, TokenCounter().count)
    bundle = selector.select(
        TurnInterpretation(
            intent="phone",
            language="ar",
            dialogue_mode="continuation",
            policy_topics=["contact"],
        ),
        {},
    )
    assert "SITUATION.CONTACT" in bundle.ids
    assert "CORE.BOUNDARIES" in bundle.ids
    assert "SITUATION.RECALL" not in bundle.ids
    assert not bundle.fallback
    assert bundle.tokens < selector.full().tokens


@pytest.mark.parametrize("language", ["en", "ru", "es", "fr", "ar"])
def test_unknown_situation_keeps_complete_rules(language: str) -> None:
    selector = RuleSelector(PolicyCatalog.load(), TokenCounter().count)
    bundle = selector.select(
        TurnInterpretation(
            intent="unrecognized", language=language, dialogue_mode="continuation"
        ),
        {},
    )
    assert bundle.fallback
    assert len(bundle.ids) == len(selector.catalog.rules)


def test_examples_have_provenance_and_negative_cards_never_enter_context() -> None:
    counter = TokenCounter().count
    cards = json.loads((POLICY_DIRECTORY / "examples.json").read_text())
    selected = select_examples(
        ["social", "support", "time", "recall"], "established", counter
    )
    assert 0 < len(selected) <= 2
    assert counter(selected) <= 300
    assert all(card["source"]["paragraphs"] for card in cards)
    negative_ids = {card["id"] for card in cards if card["polarity"] == "negative"}
    assert not negative_ids.intersection(card["id"] for card in selected)


async def test_qmd_version_mismatch_falls_back_without_search(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text('{"version":"old","sha256":"old"}')
    selector = RuleSelector(PolicyCatalog.load(), TokenCounter().count)
    qmd = QmdPolicySelector(selector, tmp_path / "does-not-exist", tmp_path)
    result = await qmd.select(
        TurnInterpretation(intent="chat", language="en", dialogue_mode="continuation"),
        {},
    )
    assert result.fallback
    assert result.ids == selector.full().ids
    assert qmd.last_result["reason"] == "index_version_mismatch"


async def test_qmd_unknown_situation_never_runs_search(tmp_path: Path) -> None:
    catalog = PolicyCatalog.load()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"version": catalog.version, "sha256": catalog.checksum})
    )
    selector = RuleSelector(catalog, TokenCounter().count)
    qmd = QmdPolicySelector(selector, tmp_path / "must-not-run", tmp_path)
    bundle = await qmd.select(
        TurnInterpretation(
            intent="unknown", language="ar", dialogue_mode="continuation"
        ),
        {},
    )
    assert bundle.ids == selector.full().ids
    assert qmd.last_result["reason"] == "unknown_situation_or_budget"


async def test_qmd_timeout_reaps_child_and_keeps_full_policy(tmp_path: Path) -> None:
    import time

    catalog = PolicyCatalog.load()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"version": catalog.version, "sha256": catalog.checksum})
    )
    executable = tmp_path / "slow-search"
    executable.write_text("#!/bin/sh\nexec sleep 10\n")
    executable.chmod(0o700)
    selector = RuleSelector(catalog, TokenCounter().count)
    qmd = QmdPolicySelector(selector, executable, tmp_path)
    started = time.monotonic()
    bundle = await qmd.select(
        TurnInterpretation(
            intent="contact",
            policy_topics=["contact"],
            language="en",
            dialogue_mode="continuation",
        ),
        {},
    )
    assert time.monotonic() - started < 2
    assert bundle.ids == selector.full().ids
    assert qmd.last_result["reason"] == "TimeoutError"


async def test_qmd_lexical_query_does_not_require_every_intent_word(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    catalog = PolicyCatalog.load()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"version": catalog.version, "sha256": catalog.checksum})
    )
    payload = json.dumps(
        [
            {
                "file": "qmd://marko-policy/SITUATION.CONTACT.md",
                "body": f"Version: {catalog.version}\nCatalog: {catalog.checksum}",
            }
        ]
    ).encode()
    process = SimpleNamespace(
        returncode=0, communicate=AsyncMock(return_value=(payload, b""))
    )
    start = AsyncMock(return_value=process)
    monkeypatch.setattr("scripts.qmd_policy.asyncio.create_subprocess_exec", start)
    qmd = QmdPolicySelector(
        RuleSelector(catalog, TokenCounter().count), tmp_path / "qmd", tmp_path
    )
    bundle = await qmd.select(
        TurnInterpretation(
            intent="request_private_whatsapp_number",
            policy_topics=["social", "contact"],
            language="en",
            dialogue_mode="continuation",
        ),
        {},
    )
    assert start.call_args.args[2] == "contact"
    assert "SITUATION.CONTACT" in bundle.ids
    assert not bundle.fallback


def test_enabled_followup_policy_is_available_before_silence_trigger() -> None:
    selector = RuleSelector(PolicyCatalog.load(), TokenCounter().count)
    interpretation = TurnInterpretation(
        intent="chat",
        language="es",
        dialogue_mode="continuation",
        policy_topics=["social"],
    )
    assert "SITUATION.TIME" in selector.select(interpretation, {}).ids
    assert (
        "SITUATION.FOLLOWUP"
        in selector.select(interpretation, {"followup_enabled": True}).ids
    )
    assert "SITUATION.FOLLOWUP" not in selector.select(interpretation, {}).ids
    unknown = interpretation.model_copy(update={"policy_topics": []})
    assert selector.select(
        unknown, {"followup_enabled": True, "commerce": True}
    ).fallback


async def test_benchmark_stops_remote_calls_after_payment_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock

    import httpx
    from openai import APIStatusError

    from app.modules.agent.gateway import GatewayModel
    from app.modules.agent.schemas import ReplyProposal
    from scripts.compare_policies import BenchmarkGateway

    response = httpx.Response(
        402, request=httpx.Request("POST", "https://example.invalid")
    )
    complete = AsyncMock(
        side_effect=APIStatusError("quota", response=response, body={})
    )
    monkeypatch.setattr(GatewayModel, "complete", complete)
    model = object.__new__(BenchmarkGateway)
    with pytest.raises(APIStatusError):
        await model.complete("evaluate", "rules", {}, ReplyProposal, 1)
    assert model.unavailable_reason == "GATEWAY_HTTP_402"
    with pytest.raises(RuntimeError, match="GATEWAY_HTTP_402"):
        await model.complete("evaluate", "rules", {}, ReplyProposal, 1)
    assert complete.await_count == 1


@pytest.mark.parametrize("language", ["en", "ru", "es", "fr", "ar"])
def test_language_contract_cannot_reintroduce_single_detail_recall(
    language: str,
) -> None:
    from app.modules.agent.language import reply_validation_instruction
    from app.modules.agent.prompts import VALIDATOR_RULES

    contract = reply_validation_instruction(language)
    assert "RECALL" not in contract
    assert "distinctive detail" not in contract
    assert "SITUATION.RECALL" in VALIDATOR_RULES
    assert "coverage categories ONLY WHEN KNOWN" in VALIDATOR_RULES
