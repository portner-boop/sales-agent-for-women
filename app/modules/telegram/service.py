import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from app.modules.commerce.service import CommerceService
from app.modules.conversations.repository import ConversationRepository
from app.modules.conversations.schemas import IncomingMessage
from app.modules.delivery.repository import DeliveryRepository
from app.modules.delivery.schemas import OutboxDraft
from app.modules.dialogue.languages import canonical_language
from app.modules.followups.service import FollowupService
from app.modules.jobs.repository import JobRepository
from app.modules.jobs.schemas import JobPayload
from app.modules.memory.service import MemoryService
from app.modules.telegram.entities import IncomingEvent
from app.modules.telegram.repository import TelegramRepository
from app.modules.telegram.schemas import RawUpdate
from app.shared.config import RuntimeSettings
from app.shared.entities import ActionKind, Clock, ScopeContext
from app.shared.storage import Transaction

logger = logging.getLogger(__name__)

SERVICE_TEXTS = {
    "commands": {
        "ru": "Доступны /start — начать, /stop — остановить, /privacy — условия, /forget — удалить память.",
        "en": "Available commands: /start — start, /stop — stop, /privacy — terms, /forget — delete memory.",
        "es": "Comandos disponibles: /start — iniciar, /stop — detener, /privacy — condiciones, /forget — borrar recuerdos.",
        "pt": "Comandos disponíveis: /start — iniciar, /stop — parar, /privacy — condições, /forget — apagar memórias.",
    },
    "start": {
        "ru": "Тестовая ролевая переписка с ИИ: Марко — вымышленный персонаж, который остаётся в образе внутри сцены. /start — начать или возобновить разговор, /stop — остановить, /privacy — условия, /forget — удалить память.",
        "en": "AI roleplay test: Marko is a fictional character who stays in character inside the scene. /start starts or resumes a conversation, /stop stops messages, /privacy shows data terms, /forget deletes memory.",
        "es": "Prueba de rol con IA: Marko es un personaje ficticio que mantiene su papel dentro de la escena. /start inicia o reanuda la conversación, /stop detiene mensajes, /privacy muestra condiciones, /forget elimina recuerdos.",
        "pt": "Teste de interpretação com IA: Marko é uma personagem fictícia que mantém o papel dentro da cena. /start inicia ou retoma a conversa, /stop interrompe mensagens, /privacy mostra condições, /forget apaga memórias.",
    },
    "preview_closed": {
        "ru": "Пробная беседа завершена. Для нового теста можно удалить историю через /forget.",
        "en": "The trial conversation has ended. To begin a new test, you can delete its history with /forget.",
        "es": "La conversación de prueba ha terminado. Para empezar otra prueba, puedes borrar el historial con /forget.",
        "pt": "A conversa de teste terminou. Para iniciar outro teste, podes apagar o histórico com /forget.",
        "fr": "La conversation d’essai est terminée. Pour recommencer un test, tu peux effacer l’historique avec /forget.",
        "de": "Das Probegespräch ist beendet. Für einen neuen Test kannst du den Verlauf mit /forget löschen.",
        "ar": "انتهت المحادثة التجريبية. لبدء اختبار جديد يمكنك حذف سجلها باستخدام /forget.",
        "zh": "试聊已经结束。如需重新测试，可以通过 /forget 删除聊天记录。",
    },
    "stop": {
        "ru": "Переписка остановлена. Память сохранена. Возобновить можно через /start.",
        "en": "Chat stopped. Memory is retained. Use /start to resume.",
        "es": "Chat detenido. Los recuerdos se conservan. /start permite volver.",
        "pt": "Conversa interrompida. As memórias foram mantidas. /start permite voltar.",
    },
    "forgotten": {
        "ru": "Память разговора удалена. Чтобы начать новый разговор, отправьте /start. Старые резервные копии очищаются по сроку хранения.",
        "en": "Conversation memory deleted. Use /start to begin a new conversation. Older backups expire under retention policy.",
        "es": "Recuerdos eliminados. Usa /start para iniciar una nueva conversación.",
        "pt": "Memórias apagadas. Usa /start para iniciar uma nova conversa.",
    },
    "unsupported": {
        "ru": "В этом прототипе доступны только текстовые сообщения. Содержание фото и голосовых я не вижу.",
        "en": "This prototype supports text only. I cannot see photos or hear voice messages.",
        "es": "Este prototipo solo admite texto. No puedo ver fotos ni escuchar mensajes de voz.",
        "pt": "Este protótipo aceita apenas texto. Não vejo fotos nem ouço mensagens de voz.",
    },
}


def service_text(kind: str, language: str) -> str:
    values = SERVICE_TEXTS[kind]
    return values.get(language.split("-")[0], values["en"])


def normalize_update(raw: dict[str, Any], max_chars: int) -> IncomingEvent:
    try:
        update = RawUpdate.model_validate(raw)
        payload = update.edited_message or update.message
        if payload is None:
            member = update.my_chat_member
            if member and member["chat"]["type"] == "private":
                return IncomingEvent(
                    update.update_id,
                    raw,
                    "blocked",
                    int(member["from"]["id"]),
                    int(member["chat"]["id"]),
                    command=(
                        "stop"
                        if member["new_chat_member"]["status"] in {"kicked", "left"}
                        else None
                    ),
                )
            return IncomingEvent(
                update.update_id, raw, "quarantine", reason="UNSUPPORTED_UPDATE"
            )
        if payload["chat"]["type"] != "private" or payload["from"].get("is_bot"):
            return IncomingEvent(
                update.update_id, raw, "quarantine", reason="NONPRIVATE_OR_BOT"
            )
        user_id, chat_id = int(payload["from"]["id"]), int(payload["chat"]["id"])
        language = str(payload["from"].get("language_code", "en")).split("-")[0]
        text = payload.get("text")
        if not isinstance(text, str):
            return IncomingEvent(
                update.update_id,
                raw,
                "unsupported",
                user_id,
                chat_id,
                language_hint=language,
            )
        if len(text) > max_chars:
            return IncomingEvent(
                update.update_id,
                raw,
                "quarantine",
                user_id,
                chat_id,
                reason="INPUT_TOO_LARGE",
                language_hint=language,
            )
        entities = payload.get("entities", [])
        command_entity = next(
            (
                entity
                for entity in entities
                if entity.get("type") == "bot_command" and entity.get("offset") == 0
            ),
            None,
        )
        command = text.split()[0].split("@")[0][1:] if command_entity else None
        timestamp = payload["date"]
        event_time = (
            datetime.fromisoformat(timestamp).timestamp()
            if isinstance(timestamp, str)
            else float(timestamp)
        )
        incoming = IncomingMessage(
            external_id=payload["message_id"],
            text=text,
            event_time=event_time,
            reply_to_external_id=payload.get("reply_to_message", {}).get("message_id"),
            forwarded=bool(payload.get("forward_origin")),
            is_edit=update.edited_message is not None,
        )
        return IncomingEvent(
            update.update_id,
            raw,
            "command" if command else "message",
            user_id,
            chat_id,
            incoming,
            command,
            language_hint=language,
        )
    except (KeyError, TypeError, ValueError, ValidationError):
        update_id = raw.get("update_id")
        if not isinstance(update_id, int):
            raise ValueError("Update lacks a usable update_id") from None
        return IncomingEvent(update_id, raw, "quarantine", reason="INVALID_UPDATE")


class TelegramService:
    def __init__(
        self,
        repository: TelegramRepository,
        conversations: ConversationRepository,
        memory: MemoryService,
        delivery: DeliveryRepository,
        commerce: CommerceService,
        jobs: JobRepository,
        clock: Clock,
        settings: RuntimeSettings,
        retention_days: int,
        on_control: Callable[[str], Awaitable[None]],
        followups: FollowupService | None = None,
        on_input: Callable[[ScopeContext, int], Awaitable[None]] | None = None,
    ) -> None:
        self.repository = repository
        self.conversations = conversations
        self.memory = memory
        self.delivery = delivery
        self.commerce = commerce
        self.jobs = jobs
        self.clock = clock
        self.settings = settings
        self.retention_days = retention_days
        self.on_control = on_control
        self.followups = followups
        self.on_input = on_input

    async def ingest(self, raws: list[dict[str, Any]]) -> None:
        received_at = self.clock.now()
        events = [
            normalize_update(raw, self.settings.max_message_chars) for raw in raws
        ]
        revisions: dict[ScopeContext, int] = {}
        controls: set[str] = set()
        followup_controls: set[str] = set()

        async def receive(tx: Transaction) -> None:
            for event in sorted(events, key=lambda item: item.update_id):
                if await self.repository.exists(tx, event.update_id):
                    continue
                scope = None
                if event.user_id is not None and event.chat_id is not None:
                    conversation = await self.conversations.ensure(
                        tx,
                        self.repository.channel,
                        event.user_id,
                        event.chat_id,
                        self.clock.now(),
                        engine_version="v2",
                        persona_mode="research_human",
                        language=canonical_language(event.language_hint or "en"),
                    )
                    scope = conversation.scope
                    await self.memory.repository.ensure_entities(tx, scope)
                await self.repository.record(tx, event, scope, received_at)
                if scope is None:
                    continue
                if self.followups is not None:
                    await self.followups.cancel_pending(tx, scope)
                    followup_controls.add(scope.scope_id)
                    await self.delivery.cancel_followups(tx, scope)
                if event.kind == "message" and event.message:
                    try:
                        message, changed = await self.conversations.append(
                            tx,
                            scope,
                            event.message,
                            received_at,
                            self.settings.activity_gap_seconds,
                        )
                    except LookupError:
                        continue
                    if changed:
                        await tx.execute(
                            "UPDATE scheduled_jobs SET status='cancelled',error_code='NEW_INPUT' WHERE scope_id=? AND epoch=? AND kind='turn' AND status='pending'",
                            (scope.scope_id, scope.epoch),
                        )
                        await self.delivery.cancel_pending(tx, scope, "NEW_INPUT")
                        if event.message.is_edit:
                            await self.memory.repository.invalidate_source(
                                tx, scope, message.id
                            )
                        conversation = await self.conversations.get(scope.scope_id, tx)
                        revisions[scope] = conversation.input_revision
                        await self.jobs.schedule(
                            tx,
                            "turn",
                            f"turn:{scope.scope_id}:{scope.epoch}:{conversation.input_revision}",
                            self.clock.now(),
                            self.clock.now()
                            + (
                                0
                                if conversation.first_reply_at is None
                                else min(
                                    self.settings.coalesce_seconds,
                                    self.settings.max_coalesce_seconds,
                                )
                            ),
                            scope,
                            JobPayload(
                                data={"input_revision": conversation.input_revision}
                            ),
                        )
                elif event.kind == "command" or event.command == "stop":
                    await self._command(tx, event, scope)
                    if event.command in {"stop", "forget"}:
                        controls.add(scope.scope_id)
                elif event.kind == "unsupported":
                    await self._service(tx, scope, event, "unsupported")
                elif event.reason == "INPUT_TOO_LARGE":
                    await self._text(
                        tx,
                        scope,
                        event.update_id,
                        "Message is too large. Please split it into shorter messages.",
                    )
            if events:
                await self.repository.advance_offset(
                    tx, max(event.update_id for event in events) + 1
                )

        await self.repository.database.write(receive)
        for event in events:
            if event.kind == "quarantine":
                logger.warning(
                    "update_quarantined update_id=%s reason=%s",
                    event.update_id,
                    event.reason,
                )
        if self.on_input is not None:
            for scope, revision in revisions.items():
                await self.on_input(scope, revision)
        for scope_id in controls:
            await self.on_control(scope_id)
        if self.followups is not None:
            for scope_id in followup_controls:
                await self.followups.cancel_scope(scope_id)

    async def _command(
        self, tx: Transaction, event: IncomingEvent, scope: ScopeContext
    ) -> None:
        command = event.command
        conversation = await self.conversations.get(scope.scope_id, tx)
        if command == "stop":
            await self.conversations.set_control(tx, scope, "stopped")
            await self.delivery.cancel_pending(tx, scope, "STOP")
            if event.kind != "blocked":
                await self._service(tx, scope, event, "stop")
        elif command == "start":
            await self.conversations.set_control(tx, scope, "enabled")
            key = (
                "preview_closed"
                if conversation.preview_reply_count
                >= self.commerce.settings.preview_max_replies
                else "start"
            )
            await self._service(tx, scope, event, key)
        elif command == "forget":
            await self.conversations.set_control(tx, scope, "paused")
            await self.delivery.cancel_pending(tx, scope, "FORGET")
            await self.jobs.schedule(
                tx,
                "forget",
                f"forget:{scope.scope_id}:{scope.epoch}",
                self.clock.now(),
                self.clock.now(),
                scope,
                JobPayload(data={"update_id": event.update_id}),
            )
        elif command == "privacy":
            texts = {
                "ru": f"Это ролевая симуляция с ИИ. Марко — вымышленный персонаж; его ответы внутри сцены не подтверждают существование реального человека. Сообщения передаются внешнему сервису обработки и хранятся локально до {self.retention_days} дней с последней активности. /stop останавливает переписку; /forget удаляет сообщения и память. Резервные копии удаляются по тому же сроку. Минимальные записи удаления сохраняются. Не отправляйте чувствительные персональные данные.",
                "en": f"This is an AI roleplay simulation. Marko is fictional; his replies inside the scene do not verify a real human. Messages are sent to an external processing service and stored locally for up to {self.retention_days} days after last activity. /stop pauses messages; /forget deletes conversation memory. Backups expire under the same retention policy. Minimal deletion records remain. Do not send sensitive personal data.",
                "es": f"Simulación de rol con IA. Marko es ficticio; sus respuestas dentro de la escena no verifican a una persona real. Los mensajes se envían a un servicio externo de procesamiento y se guardan localmente hasta {self.retention_days} días tras la última actividad. /stop pausa; /forget elimina mensajes y recuerdos. Las copias caducan con el mismo plazo. Los registros mínimos de eliminación se conservan. No envíes datos sensibles.",
                "pt": f"Simulação de interpretação com IA. Marko é fictício; as respostas na cena não comprovam uma pessoa real. Mensagens são enviadas a um serviço externo de processamento e guardadas localmente até {self.retention_days} dias após a última atividade. /stop pausa; /forget apaga mensagens e memórias. Cópias expiram no mesmo prazo. Registros mínimos de exclusão são mantidos. Não envie dados sensíveis.",
            }
            text = texts.get(conversation.language, texts["en"])
            if self.settings.followup_enabled:
                text += " " + self._followup_notice(conversation.language)
            await self._text(tx, scope, event.update_id, text)
        else:
            await self._service(tx, scope, event, "commands")

    async def _service(
        self,
        tx: Transaction,
        scope: ScopeContext,
        event: IncomingEvent,
        key: str,
        kind: ActionKind = "service",
    ) -> None:
        conversation = await self.conversations.get(scope.scope_id, tx)
        language = event.language_hint if key == "start" else conversation.language
        text = service_text(key, language)
        if key == "start" and self.settings.followup_enabled:
            text += " " + self._followup_notice(language)
        await self._text(tx, scope, event.update_id, text, kind)

    def _followup_notice(self, language: str) -> str:
        texts = {
            "ru": "После паузы могу один раз продолжить разговор сам. /stop отключает сообщения.",
            "en": "After a pause I may send one follow-up. /stop stops messages.",
            "es": "Tras una pausa puedo continuar una vez. /stop detiene los mensajes.",
            "pt": "Após uma pausa posso continuar uma vez. /stop interrompe as mensagens.",
        }
        return texts.get(language.split("-")[0], texts["en"])

    async def _text(
        self,
        tx: Transaction,
        scope: ScopeContext,
        update_id: int,
        text: str,
        kind: ActionKind = "service",
    ) -> None:
        conversation = await self.conversations.get(scope.scope_id, tx)
        draft = OutboxDraft(
            action_key=f"command:{self.repository.channel}:{update_id}:{kind}",
            kind=kind,
            text=text,
            deadline=self.clock.now() + 60,
        )
        await self.delivery.prepare(tx, conversation.snapshot, draft, self.clock.now())
