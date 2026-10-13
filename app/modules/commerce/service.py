from app.modules.commerce.repository import CommerceRepository
from app.modules.commerce.schemas import HandoffContext, MockPurchase
from app.modules.conversations.entities import Conversation
from app.modules.conversations.repository import ConversationRepository
from app.shared.config import CommerceSettings
from app.shared.entities import Clock, ScopeContext
from app.shared.storage import Transaction

PLATFORM_URL = "https://www.behappy2day.com/"


class CommerceService:
    def __init__(
        self,
        repository: CommerceRepository,
        conversations: ConversationRepository,
        clock: Clock,
        settings: CommerceSettings,
        tester_ids: frozenset[int],
    ) -> None:
        self.repository = repository
        self.conversations = conversations
        self.clock = clock
        self.settings = settings
        self.tester_ids = tester_ids

    def can_reply(self, conversation: Conversation) -> bool:
        return (
            conversation.control_state == "enabled"
            and conversation.purchase_state != "confirmed"
            and not self.preview_closed(conversation)
        )

    def preview_closed(self, conversation: Conversation) -> bool:
        return (
            self.settings.preview_enabled and conversation.preview_closed_at is not None
        )

    async def transition_state(
        self, conversation: Conversation
    ) -> dict[str, bool | int]:
        eligible = (
            self.settings.preview_enabled
            and self.can_reply(conversation)
            and conversation.preview_reply_count >= self.settings.preview_min_replies
        )
        async with self.repository.database.read() as tx:
            allowed = eligible and await self.repository.transition_allowed(
                tx, conversation.scope
            )
        return {
            "completed_replies": conversation.preview_reply_count,
            "eligible": allowed,
            "required": self.settings.preview_enabled
            and conversation.preview_reply_count
            >= self.settings.preview_max_replies - 1,
        }

    def transition_text(self, language: str, answer: str | None = None) -> str:
        texts = {
            "ru": "Если захочешь продолжить общение, на BeHappy2Day можно купить время для переписки",
            "en": "If you’d like to keep chatting, you can buy chat time on BeHappy2Day",
            "es": "Si te apetece seguir charlando, puedes comprar tiempo de chat en BeHappy2Day",
            "pt": "Se quiser continuar conversando, pode comprar tempo de conversa no BeHappy2Day",
        }
        invitation = (
            texts.get(language.split("-")[0], texts["en"]) + "\n" + PLATFORM_URL
        )
        return f"{answer.strip()}\n\n{invitation}" if answer else invitation

    def platform_info_text(self, language: str) -> str:
        texts = {
            "ru": "Продолжить переписку можно на BeHappy2Day",
            "en": "You can continue chatting on BeHappy2Day",
            "es": "Puedes continuar charlando en BeHappy2Day",
            "pt": "Você pode continuar conversando no BeHappy2Day",
        }
        return texts.get(language.split("-")[0], texts["en"]) + "\n" + PLATFORM_URL

    def farewell_text(self, language: str, answer: str | None = None) -> str:
        texts = {
            "ru": "Понял, тогда закончим здесь",
            "en": "Understood, let’s leave it here",
            "es": "Entendido, lo dejamos aquí",
            "pt": "Entendido, vamos encerrar por aqui",
        }
        closing = texts.get(language.split("-")[0], texts["en"])
        return f"{answer.strip()}\n\n{closing}" if answer else closing

    def closed_text(self, language: str) -> str:
        texts = {
            "ru": "Тут переписка уже закончилась\nПродолжить можно на BeHappy2Day",
            "en": "This chat has ended here\nYou can continue on BeHappy2Day",
            "es": "Este chat terminó aquí\nPuedes continuar en BeHappy2Day",
            "pt": "Esta conversa terminou aqui\nVocê pode continuar no BeHappy2Day",
        }
        return texts.get(language.split("-")[0], texts["en"]) + "\n" + PLATFORM_URL

    async def confirm(
        self,
        tx: Transaction,
        scope: ScopeContext,
        event: MockPurchase,
        context: HandoffContext,
    ) -> bool:
        conversation = await self.conversations.get(scope.scope_id, tx)
        if (
            event.actor_user_id != conversation.external_user_id
            or event.actor_user_id not in self.tester_ids
        ):
            raise PermissionError("Test purchase is restricted to testers")
        if conversation.purchase_state == "confirmed":
            return False
        if await self.repository.confirm_mock(
            tx, scope, event.event_key, context, self.clock.now()
        ):
            await self.conversations.set_control(tx, scope, "handoff_only")
            await tx.execute(
                "UPDATE conversations SET purchase_state='confirmed',handoff_state='requested' WHERE id=?",
                (scope.scope_id,),
            )
            return True
        return False

    async def offer_text(self, language: str) -> str:
        offer = await self.repository.offer()
        texts = {
            "ru": f"Тестовый пакет: {offer.duration_minutes} минут. Списания денег нет. Подтверждение выполняется в тестовом стенде.",
            "en": f"Test package: {offer.duration_minutes} minutes. No charge. Confirmation is handled in the test harness.",
            "es": f"Paquete de prueba: {offer.duration_minutes} minutos. Sin cobro. La confirmación se realiza en el entorno de pruebas.",
            "pt": f"Pacote de teste: {offer.duration_minutes} minutos. Sem cobrança. A confirmação é feita no ambiente de testes.",
        }
        return texts.get(language, texts["en"])
