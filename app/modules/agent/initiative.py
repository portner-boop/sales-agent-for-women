from typing import Any

from app.modules.agent.schemas import FollowupProposal, ReplyProposal


def followup_eligibility(
    proposal: ReplyProposal, context: dict[str, Any]
) -> tuple[bool, str]:
    state = context["APPLICATION_STATE"]
    plan = context["DIALOGUE_PLAN"]
    if not state.get("followup_enabled"):
        return False, "DISABLED"
    if isinstance(proposal, FollowupProposal):
        return False, "NO_CHAIN"
    if proposal.question_goal:
        return False, "QUESTION_PENDING"
    if state["dialogue_mode"] == "closing":
        return False, "CLOSED"
    if (
        proposal.requested_action != "none"
        or proposal.platform_transition
        or proposal.platform_notice
        or context.get("PLATFORM_TRANSITION", {}).get("required")
    ):
        return False, "BUSINESS_ACTION"
    if any(
        item["predicate"]
        in {
            "question_permission",
            "proactive_permission",
            "followup_permission",
            "contact_permission",
        }
        and item["value"] is False
        for item in context.get("ACTIVE_BOUNDARIES", [])
    ):
        return False, "USER_BOUNDARY"
    if plan["strategy"] == "share_opinion_without_question":
        return False, "SHARING_ONLY"
    return True, "OPEN_DECLARATIVE_REPLY"
