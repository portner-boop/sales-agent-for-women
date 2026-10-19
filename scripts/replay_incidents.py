import argparse
import asyncio
import hashlib
import json
import traceback
from pathlib import Path
from typing import Any

from app.modules.agent.policy import PolicyCatalog
from app.shared.config import load_settings
from scripts.compare_policies import BenchmarkGateway, dialogue, implementation_checksum


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument(
        "--modes", nargs="+", choices=["python", "qmd"], default=["python", "qmd"]
    )
    parser.add_argument(
        "--qmd", type=Path, default=Path("/tmp/marko-qmd-2.8.3/node_modules/.bin/qmd")
    )
    parser.add_argument("--index", type=Path, default=Path("/tmp/marko-qmd-index-v4"))
    args = parser.parse_args()
    raw = args.cases.read_bytes()
    cases = json.loads(raw)
    settings = load_settings()
    settings.require_credentials(telegram=False)
    model = BenchmarkGateway(settings.llm, args.parallel)
    judge = (Path(__file__).parent / "resources" / "incident_judge.md").read_text()
    report: dict[str, Any] = {
        "catalog": PolicyCatalog.load().manifest(),
        "implementation_sha256": implementation_checksum(),
        "cases_sha256": hashlib.sha256(raw).hexdigest(),
        "source_sha256": cases["source_sha256"],
        "model": settings.llm.model,
        "structured_mode": settings.llm.structured_mode,
        "qmd_mode": "BM25 CLI, one query per turn, timeout 1 second, full fallback",
        "method": "New answers to fixed recorded user inputs, fresh temporary database per dialogue, no Telegram sends, accelerated pacing; same-model final judge",
        "dialogues": [],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        args.report.chmod(0o600)

    save()
    semaphore = asyncio.Semaphore(args.parallel)

    async def run(case: dict[str, Any], mode: str, repeat: int) -> None:
        async with semaphore:
            if model.unavailable_reason:
                return
            try:
                result = await dialogue(
                    args,
                    model,
                    case["language"],
                    mode,
                    repeat,
                    [step["text"] for step in case["steps"]],
                    judge_rules=judge + "\nCASE_REQUIREMENTS:\n" + case["requirements"],
                    user_timestamps=[step["received_at"] for step in case["steps"]],
                    silence_after=(
                        set(case["silence_after"]) if "silence_after" in case else None
                    ),
                )
            except Exception as error:
                result = {
                    "mode": mode,
                    "repeat": repeat,
                    "error": model.unavailable_reason or type(error).__name__,
                    "error_site": [
                        f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"
                        for frame in traceback.extract_tb(error.__traceback__)[-4:]
                    ],
                }
            result["case_id"] = case["id"]
            report["dialogues"].append(result)
            if model.unavailable_reason:
                report["interruption"] = model.unavailable_reason
            save()
            print(
                json.dumps(
                    {
                        key: result.get(key)
                        for key in (
                            "case_id",
                            "mode",
                            "repeat",
                            "missing_answers",
                            "error",
                        )
                    }
                ),
                flush=True,
            )

    await asyncio.gather(
        *(
            run(case, mode, repeat)
            for repeat in range(args.repetitions)
            for case in cases["cases"]
            for mode in args.modes
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
