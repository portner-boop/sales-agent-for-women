import asyncio
import json
import os
from pathlib import Path
from typing import Any

from app.modules.agent.policy import PolicyCatalog, RuleBundle, RuleSelector
from app.modules.agent.schemas import TurnInterpretation


class QmdPolicySelector:
    def __init__(
        self, selector: RuleSelector, executable: Path, directory: Path
    ) -> None:
        self.selector = selector
        self.executable = executable
        self.directory = directory
        self.environment = {
            **os.environ,
            "INDEX_PATH": str(directory / "index.sqlite"),
            "QMD_CONFIG_DIR": str(directory / "config"),
            "XDG_CACHE_HOME": str(directory / "cache"),
        }
        self.last_result: dict[str, Any] = {}

    async def select(
        self, interpretation: TurnInterpretation, state: dict[str, Any]
    ) -> RuleBundle:
        started = asyncio.get_running_loop().time()
        process = None
        query = None
        try:
            index_manifest = json.loads((self.directory / "manifest.json").read_text())
            catalog = self.selector.catalog
            if index_manifest != {
                "version": catalog.version,
                "sha256": catalog.checksum,
            }:
                raise ValueError("index_version_mismatch")
            mandatory = self.selector.select(interpretation, state)
            if mandatory.fallback:
                raise ValueError("unknown_situation_or_budget")
            conditions = {
                condition
                for rule in catalog.rules
                for condition in rule.conditions
                if condition != "always"
            }
            query = (
                "followup"
                if state.get("followup")
                else next(
                    (
                        topic
                        for topic in interpretation.policy_topics
                        if topic in conditions
                    ),
                    "conversation",
                )
            )
            async with asyncio.timeout(1):
                process = await asyncio.create_subprocess_exec(
                    str(self.executable),
                    "search",
                    query,
                    "--json",
                    "--full",
                    "-n",
                    "4",
                    "-c",
                    "marko-policy",
                    env=self.environment,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await process.communicate()
            if process.returncode:
                raise ValueError("qmd_process_error")
            results = json.loads(stdout)
            if not isinstance(results, list) or not results:
                raise ValueError("qmd_no_results")
            by_id = {rule.id: rule for rule in catalog.rules}
            selected = {
                rule.id: "mandatory_core"
                for rule in catalog.rules
                if "always" in rule.conditions
            }
            for identifier in mandatory.ids:
                if identifier in {
                    "SITUATION.TIME",
                    "SITUATION.RECALL",
                    "SITUATION.FOLLOWUP",
                    "SITUATION.COMMERCE",
                }:
                    selected[identifier] = "mandatory_server_state"
            for result in results:
                filename = str(result.get("file", ""))
                body = str(result.get("body", result.get("content", "")))
                identifier = Path(filename).stem.upper()
                if (
                    identifier not in by_id
                    or f"Version: {catalog.version}" not in body
                    or f"Catalog: {catalog.checksum}" not in body
                ):
                    raise ValueError("qmd_unverified_card")
                selected[identifier] = "qmd_bm25"
            bundle = self.selector.bundle(selected)
            self.last_result = {
                "fallback": False,
                "reason": None,
                "retrieved": [result.get("file") for result in results],
            }
        except (OSError, ValueError, TimeoutError, KeyError, TypeError) as error:
            bundle = self.selector.full("qmd:" + type(error).__name__)
            self.last_result = {
                "fallback": True,
                "reason": str(error) or type(error).__name__,
            }
        finally:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
        self.last_result["selection_seconds"] = (
            asyncio.get_running_loop().time() - started
        )
        self.last_result["query"] = query
        return bundle


def export_catalog(catalog: PolicyCatalog, directory: Path) -> None:
    cards = directory / "cards"
    cards.mkdir(parents=True, exist_ok=True)
    if list(cards.iterdir()):
        raise ValueError("Use an empty isolated index directory")
    for rule in catalog.rules:
        (cards / f"{rule.id}.md").write_text(
            f"# {rule.id}\n\nVersion: {catalog.version}\nCatalog: {catalog.checksum}\nConditions: {', '.join(rule.conditions)}\n\n{rule.text}\n"
        )
    (directory / "manifest.json").write_text(
        json.dumps({"version": catalog.version, "sha256": catalog.checksum})
    )
