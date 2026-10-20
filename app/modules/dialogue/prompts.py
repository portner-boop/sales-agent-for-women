from pathlib import Path

from app.modules.dialogue.schemas import DialogueMode

POLICY_VERSION = "marko-dialogue-v55"

RESOURCE_DIRECTORY = Path(__file__).with_name("resources")
COMMON_RULES = (RESOURCE_DIRECTORY / "conversation.md").read_text(encoding="utf-8")
VOICE_RULES = (RESOURCE_DIRECTORY / "voice.md").read_text(encoding="utf-8")
INITIATIVE_RULES = (RESOURCE_DIRECTORY / "initiative.md").read_text(encoding="utf-8")
FOLLOWUP_RULES = (RESOURCE_DIRECTORY / "followup.md").read_text(encoding="utf-8")
FIRST_FOLLOWUP_RULES = (RESOURCE_DIRECTORY / "followup_first.md").read_text(
    encoding="utf-8"
)
SECOND_FOLLOWUP_RULES = (RESOURCE_DIRECTORY / "followup_second.md").read_text(
    encoding="utf-8"
)


IDENTITY_RULES = (RESOURCE_DIRECTORY / "identity.md").read_text(encoding="utf-8")


def rules_for_mode(
    mode: DialogueMode = "research_human", *, followup_ordinal: int = 0
) -> str:
    if mode != "research_human":
        raise ValueError("Only the fixed character contract is supported")
    if followup_ordinal not in {0, 1, 2}:
        raise ValueError("Unsupported follow-up opportunity")
    rules = [IDENTITY_RULES, COMMON_RULES, VOICE_RULES]
    if followup_ordinal:
        rules.extend(
            [
                FOLLOWUP_RULES,
                (
                    FIRST_FOLLOWUP_RULES
                    if followup_ordinal == 1
                    else SECOND_FOLLOWUP_RULES
                ),
            ]
        )
    else:
        rules.append(INITIATIVE_RULES)
    return "\n\n".join(rules)
