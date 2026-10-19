import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.modules.agent.schemas import TurnInterpretation

POLICY_DIRECTORY = Path(__file__).parents[2] / "policies"


@dataclass(frozen=True)
class PolicyRule:
    id: str
    version: str
    source: str
    conditions: tuple[str, ...]
    dependencies: tuple[str, ...]
    stages: tuple[str, ...]
    tests: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class RuleBundle:
    version: str
    checksum: str
    ids: tuple[str, ...]
    reasons: tuple[tuple[str, str], ...]
    tokens: int
    budget: int
    fallback: bool
    text: str

    def as_data(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "checksum": self.checksum,
            "ids": list(self.ids),
            "reasons": dict(self.reasons),
            "tokens": self.tokens,
            "budget": self.budget,
            "fallback": self.fallback,
        }


@dataclass(frozen=True)
class PolicyCatalog:
    version: str
    checksum: str
    rules: tuple[PolicyRule, ...]

    @classmethod
    def load(cls, path: Path = POLICY_DIRECTORY / "catalog.md") -> "PolicyCatalog":
        raw = path.read_text(encoding="utf-8")
        version = re.search(r"^Version: (\S+)$", raw, re.M)
        if version is None:
            raise ValueError("Policy version missing")
        rules = []
        for section in raw.split("\n## ")[1:]:
            identifier, content = section.split("\n", 1)
            match = re.fullmatch(r"\s*```json\n(.*?)\n```\s*(.+?)\s*", content, re.S)
            if match is None or not re.fullmatch(r"[A-Z]+\.[A-Z_]+", identifier):
                raise ValueError("Invalid policy card")
            meta = json.loads(match[1])
            if meta["version"] != version[1] or not meta["source"] or not meta["tests"]:
                raise ValueError("Incomplete or mixed-version policy")
            rules.append(
                PolicyRule(
                    identifier,
                    meta["version"],
                    meta["source"],
                    tuple(meta["conditions"]),
                    tuple(meta["dependencies"]),
                    tuple(meta["stages"]),
                    tuple(meta["tests"]),
                    match[2],
                )
            )
        ids = {rule.id for rule in rules}
        if not rules or len(ids) != len(rules):
            raise ValueError("Empty or duplicate policy IDs")
        by_id = {rule.id: rule for rule in rules}

        def visit(identifier: str, ancestors: frozenset[str]) -> None:
            if identifier not in by_id or identifier in ancestors:
                raise ValueError("Missing or cyclic policy dependency")
            for dependency in by_id[identifier].dependencies:
                visit(dependency, ancestors | {identifier})

        for identifier in ids:
            visit(identifier, frozenset())
        return cls(version[1], hashlib.sha256(raw.encode()).hexdigest(), tuple(rules))

    def render(self, ids: tuple[str, ...] | None = None) -> str:
        selected = set(ids) if ids is not None else {rule.id for rule in self.rules}
        return f"POLICY VERSION {self.version}\n" + "\n\n".join(
            f"[{rule.id}] {rule.text}" for rule in self.rules if rule.id in selected
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "sha256": self.checksum,
            "rules": [
                {
                    "id": rule.id,
                    "version": rule.version,
                    "source": rule.source,
                    "conditions": rule.conditions,
                    "dependencies": rule.dependencies,
                    "stages": rule.stages,
                    "tests": rule.tests,
                    "sha256": hashlib.sha256(rule.text.encode()).hexdigest(),
                }
                for rule in self.rules
            ],
        }


class RuleSelector:
    def __init__(
        self, catalog: PolicyCatalog, counter: Callable[[Any], int], budget: int = 5000
    ) -> None:
        self.catalog = catalog
        self.counter = counter
        self.budget = budget

    def bundle(self, selected: Mapping[str, str], fallback: bool = False) -> RuleBundle:
        reasons = dict(selected)
        by_id = {rule.id: rule for rule in self.catalog.rules}
        if not set(reasons).issubset(by_id):
            return self.full("unknown_rule")
        for identifier in list(reasons):
            pending = list(by_id[identifier].dependencies)
            while pending:
                dependency = pending.pop()
                if dependency not in reasons:
                    reasons[dependency] = "dependency:" + identifier
                    pending.extend(by_id[dependency].dependencies)
        ids = tuple(rule.id for rule in self.catalog.rules if rule.id in reasons)
        text = self.catalog.render(ids)
        tokens = self.counter(text)
        if tokens > self.budget and len(ids) < len(self.catalog.rules):
            return self.full("selection_budget")
        return RuleBundle(
            self.catalog.version,
            self.catalog.checksum,
            ids,
            tuple(reasons.items()),
            tokens,
            self.budget,
            fallback,
            text,
        )

    def full(self, reason: str = "full_baseline") -> RuleBundle:
        return self.bundle(
            {rule.id: reason for rule in self.catalog.rules}, fallback=True
        )

    def select(
        self, interpretation: TurnInterpretation, state: Mapping[str, Any]
    ) -> RuleBundle:
        topics: set[str] = set(interpretation.policy_topics)
        if interpretation.recall_scope != "none" or interpretation.retrieval_predicates:
            topics.add("recall")
        if interpretation.current_time_requests:
            topics.add("time")
        if interpretation.dialogue_mode in {"supportive", "repair", "playful"}:
            topics.add("support")
        if state.get("followup"):
            topics.add("followup")
        if interpretation.requested_business_action != "none":
            topics.add("commerce")
        if not topics or "unknown" in topics:
            return self.full("unknown_situation")
        if state.get("followup_enabled"):
            topics.add("followup")
        if state.get("commerce"):
            topics.add("commerce")
        topics.add("always")
        return self.bundle(
            {
                rule.id: "condition:"
                + ",".join(sorted(topics.intersection(rule.conditions)))
                for rule in self.catalog.rules
                if topics.intersection(rule.conditions)
            }
        )


def select_examples(
    topics: list[str], stage: str, counter: Callable[[Any], int]
) -> list[dict[str, Any]]:
    cards = json.loads((POLICY_DIRECTORY / "examples.json").read_text(encoding="utf-8"))
    selected: list[dict[str, Any]] = []
    for card in cards:
        if card["polarity"] != "positive" or not set(card["topics"]).intersection(
            topics
        ):
            continue
        if card["stage"] not in {"any", stage}:
            continue
        example = {
            key: card[key] for key in ("id", "move", "context", "reply", "limits")
        }
        if counter(selected + [example]) <= 300:
            selected.append(example)
        if len(selected) == 2:
            break
    return selected
