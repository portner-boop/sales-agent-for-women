from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Protocol

from app.modules.dialogue.prompts import POLICY_VERSION
from app.modules.dialogue.schemas import DialogueRequest, HandoffSnapshot


class HandoffReceiver(Protocol):
    async def receive(self, snapshot: HandoffSnapshot) -> None: ...


class InMemoryHandoffReceiver:
    def __init__(self) -> None:
        self.snapshots: list[HandoffSnapshot] = []

    async def receive(self, snapshot: HandoffSnapshot) -> None:
        self.snapshots.append(snapshot.model_copy(deep=True))


def build_handoff(
    request: DialogueRequest,
    *,
    facts: Sequence[Mapping[str, Any]] = (),
) -> HandoffSnapshot:
    return HandoffSnapshot(
        scope_id=request.scope_id,
        epoch=request.epoch,
        input_revision=request.input_revision,
        language=request.language,
        mode=request.mode,
        persona=deepcopy(request.persona),
        restricted_persona=deepcopy(request.restricted_persona),
        messages=tuple(
            message.model_copy(deep=True) for message in request.original_messages
        ),
        boundaries=tuple(deepcopy(request.boundaries)),
        facts=tuple(deepcopy(dict(fact)) for fact in facts),
        policy_version=str(request.state.get("policy_version", POLICY_VERSION)),
        preview_state={
            **deepcopy(request.state.get("preview_state", {})),
            "transition": deepcopy(request.transition),
        },
        service_events=tuple(deepcopy(request.state.get("service_events", []))),
    )
