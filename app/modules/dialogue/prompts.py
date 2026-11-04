from pathlib import Path

from app.modules.dialogue.schemas import DialogueMode

POLICY_VERSION = "marko-dialogue-v58"

RESOURCE_DIRECTORY = Path(__file__).with_name("resources")
PROMPT_DIRECTORY = RESOURCE_DIRECTORY / "prompt"


def _section(name: str) -> str:
    return (PROMPT_DIRECTORY / name).read_text(encoding="utf-8").strip()


# Shared sections, in prompt order; see resources/prompt/README.md.
SHARED_SECTIONS = tuple(
    _section(path.name) for path in sorted(PROMPT_DIRECTORY.glob("0*.md"))
)
IDENTITY_RULES = SHARED_SECTIONS[0]
TURN_RULES = _section("turn.md")
TIMER_RULES = _section("timer.md")
TIMER_OPPORTUNITY_RULES = {
    1: _section("timer_first.md"),
    2: _section("timer_second.md"),
}


def rules_for_mode(
    mode: DialogueMode = "research_human", *, followup_ordinal: int = 0
) -> str:
    if mode != "research_human":
        raise ValueError("Only the fixed character contract is supported")
    if followup_ordinal not in {0, 1, 2}:
        raise ValueError("Unsupported follow-up opportunity")
    situation = (
        [TIMER_RULES, TIMER_OPPORTUNITY_RULES[followup_ordinal]]
        if followup_ordinal
        else [TURN_RULES]
    )
    return "\n\n".join([*SHARED_SECTIONS, *situation])
