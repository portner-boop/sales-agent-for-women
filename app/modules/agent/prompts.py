from collections.abc import Mapping

from app.modules.agent.policy import PolicyCatalog
from app.modules.agent.resources import prompt_contract

CATALOG = PolicyCatalog.load()
MARKO_BEHAVIOR_RULES = CATALOG.render()
INTERPRETER_RULES = prompt_contract("interpreter_rules")
GENERATOR_CONTRACT = prompt_contract("generator_contract")
VALIDATOR_CONTRACT = prompt_contract("validator_contract")
MEMORY_VALIDATOR_RULES = prompt_contract("memory_validator_rules")
SUMMARY_RULES = prompt_contract("summary_rules")
SUMMARY_VALIDATOR_RULES = prompt_contract("summary_validator_rules")
FOLLOWUP_RULES = prompt_contract("followup_rules")
FOLLOWUP_VALIDATOR_RULES = (
    prompt_contract("followup_validator_rules")
    + "\n"
    + CATALOG.render(
        tuple(
            rule.id
            for rule in CATALOG.rules
            if rule.id.startswith("CORE.")
            or rule.id in {"CONVERSATION.REPETITION", "SITUATION.FOLLOWUP"}
        )
    )
)
PLATFORM_TRANSITION_RULES = prompt_contract("platform_transition_rules")
RECOVERY_RULES = prompt_contract("recovery_rules")
STRUCTURE_REPAIR_RULES = prompt_contract("structure_repair_rules")
GENERATOR_RULES = GENERATOR_CONTRACT + "\n" + MARKO_BEHAVIOR_RULES
VALIDATOR_RULES = MARKO_BEHAVIOR_RULES + "\n" + VALIDATOR_CONTRACT


def platform_turn_instruction(state: Mapping[str, object]) -> str:
    if state.get("eligible"):
        return prompt_contract(
            "platform_required" if state.get("required") else "platform_optional"
        )
    return prompt_contract(
        "platform_closing" if state.get("required") else "platform_none"
    )
