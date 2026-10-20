from app.modules.commerce.repository import CommerceRepository
from app.modules.conversations.entities import Conversation
from app.shared.config import CommerceSettings
from app.shared.storage import Transaction

PLATFORM_URL = "https://www.behappy2day.com/"


class CommerceService:
    def __init__(
        self, repository: CommerceRepository, settings: CommerceSettings
    ) -> None:
        self.repository = repository
        self.settings = settings

    def can_reply(self, conversation: Conversation) -> bool:
        return (
            conversation.control_state == "enabled"
            and conversation.preview_closed_at is None
            and conversation.preview_reply_count < self.settings.preview_max_replies
        )

    async def transition_state(
        self, conversation: Conversation, tx: Transaction | None = None
    ) -> dict[str, bool | int]:
        if tx is None:
            async with self.repository.database.read() as connection:
                return await self.transition_state(conversation, connection)
        outgoing_number = conversation.preview_reply_count + 1
        eligible = (
            self.can_reply(conversation)
            and conversation.invitation_state == "none"
            and outgoing_number >= self.settings.preview_min_replies
        )
        allowed = eligible and await self.repository.transition_allowed(
            tx, conversation.scope
        )
        return {
            "completed_replies": conversation.preview_reply_count,
            "completed_pairs": conversation.completed_pair_count,
            "outgoing_number": outgoing_number,
            "last_reply": outgoing_number >= self.settings.preview_max_replies,
            "eligible": allowed,
            "required": allowed
            and outgoing_number >= self.settings.preview_max_replies,
        }

    def transition_text(self, language: str) -> str:
        invitations = {
            "ru": "На BeHappy2Day можно продолжить платную переписку",
            "en": "You can continue paid chatting on BeHappy2Day",
            "es": "Puedes continuar el chat de pago en BeHappy2Day",
            "pt": "Você pode continuar a conversa paga no BeHappy2Day",
        }
        invitation = invitations.get(language.split("-")[0], invitations["en"])
        return invitation + "\n" + PLATFORM_URL
