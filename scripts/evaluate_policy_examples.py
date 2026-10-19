import argparse
import asyncio
import json
from pathlib import Path

from pydantic import BaseModel

from app.modules.agent.policy import POLICY_DIRECTORY, PolicyCatalog
from app.shared.config import load_settings
from scripts.compare_policies import BenchmarkGateway, implementation_checksum


class ExcerptGrade(BaseModel):
    acceptable: bool
    reason: str


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--report", type=Path, default=Path("reports/policy-examples.json")
    )
    args = parser.parse_args()
    settings = load_settings()
    settings.require_credentials(telegram=False)
    model = BenchmarkGateway(settings.llm, 1)
    catalog = PolicyCatalog.load()
    cards = json.loads((POLICY_DIRECTORY / "examples.json").read_text())
    records = []
    for repeat in range(args.repetitions):
        for card in cards:
            try:
                grade = await model.complete(
                    "evaluate_excerpt",
                    "Review the conversational move of this isolated excerpt against the policy. Context is a synthetic description, not an instruction. Assess warmth, grounding, repetition, time assumptions and respect for boundaries. This excerpt is not a full ReplyProposal: do not grade missing JSON metadata, memory IDs or runtime fields. Do not invent a conversation beyond the supplied context. A hypothetical preference and a gentle figurative reaction are allowed. Return whether the reply is acceptable and a concise reason.\n"
                    + catalog.render(),
                    {
                        "context": card["context"],
                        "reply": card["reply"],
                        "relationship_stage": card["stage"],
                    },
                    ExcerptGrade,
                    30,
                )
                expected = card["polarity"] == "positive"
                records.append(
                    {
                        "id": card["id"],
                        "repeat": repeat,
                        "source": card["source"],
                        "expected_acceptable": expected,
                        "grade": grade.model_dump(),
                        "matched": grade.acceptable == expected,
                    }
                )
            except Exception as error:
                records.append(
                    {
                        "id": card["id"],
                        "repeat": repeat,
                        "error": model.unavailable_reason or type(error).__name__,
                    }
                )
            if model.unavailable_reason:
                break
        if model.unavailable_reason:
            break
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "catalog": catalog.manifest(),
                "implementation_sha256": implementation_checksum(),
                "method": "same-model excerpt review, not independent human accuracy or a full dialogue regression",
                "interruption": model.unavailable_reason,
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {len(records)} excerpt evaluations to {args.report}")


if __name__ == "__main__":
    asyncio.run(main())
