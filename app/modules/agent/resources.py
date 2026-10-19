import json
from functools import cache

from app.modules.agent.policy import POLICY_DIRECTORY


@cache
def prompt_contract(name: str) -> str:
    text = (
        (POLICY_DIRECTORY / "contracts" / f"{name}.md")
        .read_text(encoding="utf-8")
        .rstrip("\n")
    )
    if not text.strip():
        raise ValueError(f"Empty prompt contract: {name}")
    return text


@cache
def response_templates() -> dict[str, dict[str, str]]:
    data = json.loads((POLICY_DIRECTORY / "responses.json").read_text(encoding="utf-8"))
    if not isinstance(data, dict) or any(
        not isinstance(variants, dict)
        or not variants.get("en")
        or any(not isinstance(value, str) or not value for value in variants.values())
        for variants in data.values()
    ):
        raise ValueError("Invalid response templates")
    return data


def response_text(name: str, language: str, **values: str) -> str:
    variants = response_templates()[name]
    return variants.get(language.split("-")[0], variants["en"]).format_map(values)
