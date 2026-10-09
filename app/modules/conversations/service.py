from app.modules.conversations.entities import Conversation, Turn
from app.modules.conversations.repository import ConversationRepository
from app.shared.entities import Clock, StaleStateError


class ConversationService:
    def __init__(self, repository: ConversationRepository, clock: Clock) -> None:
        self.repository = repository
        self.clock = clock

    async def current(self, turn: Turn) -> Conversation:
        conversation = await self.repository.get(turn.scope.scope_id)
        if (
            conversation.scope != turn.scope
            or conversation.input_revision != turn.input_revision
        ):
            raise StaleStateError("Turn input changed")
        return conversation
