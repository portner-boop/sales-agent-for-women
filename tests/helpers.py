import asyncio
from collections import defaultdict, deque
from typing import Any

from app.application import Application
from app.modules.agent.entities import SchemaT
from app.modules.commerce.schemas import HandoffContext, MockPurchase
from app.modules.delivery.entities import SendResult
from app.modules.delivery.schemas import OutboxDraft
from app.modules.memory.entities import CompactionBlock
from app.modules.memory.schemas import EpisodeSummary, SummaryClaim
from app.modules.telegram.service import service_text
from app.shared.storage import Transaction, fetchone


class ScriptedModel:
    def __init__(self) -> None:
        self.responses: dict[str, deque[dict[str, Any] | Exception]] = defaultdict(
            deque
        )
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.blocks: list[CompactionBlock] = []
        self.entered = asyncio.Event()
        self.release: asyncio.Event | None = None

    async def complete(
        self,
        task: str,
        rules: str,
        data: dict[str, Any],
        schema: type[SchemaT],
        deadline_seconds: float,
    ) -> SchemaT:
        self.calls.append((task, data))
        if task == "generate":
            self.entered.set()
            if self.release is not None:
                await self.release.wait()
        if self.responses[task]:
            result = self.responses[task].popleft()
            if isinstance(result, Exception):
                raise result
        elif task == "interpret":
            text = data["CURRENT_INPUT"][-1]["text"]
            language = (
                "ru"
                if any("а" <= char.lower() <= "я" for char in text)
                else (
                    "pt"
                    if text.startswith("Olá")
                    else "es" if text.startswith("Hola") else "en"
                )
            )
            result = {
                "intent": "chat",
                "dialogue_mode": "continuation",
                "language": language,
            }
        elif task.startswith("validate"):
            result = {"decision": "accept", "reason_codes": []}
        else:
            number = sum(name == "generate" for name, _ in self.calls)
            language = data["APPLICATION_STATE"]["language"]
            texts = {
                "ru": f"Спасибо, что рассказала мне об этом. Реплика {number}.",
                "en": f"Thank you for sharing this with me. Turn {number}.",
                "es": f"Gracias por compartir esto conmigo. Turno {number}.",
                "pt": f"Obrigado por compartilhar isso comigo. Turno {number}.",
            }
            result = {"reply": texts[language], "dialogue_act": "acknowledge"}
        return schema.model_validate(result)

    async def summarize(
        self, block: CompactionBlock, deadline_seconds: float, output_tokens: int
    ) -> EpisodeSummary:
        self.blocks.append(block)
        user = next(
            (message for message in block.messages if message.role == "user"), None
        )
        claims = (
            []
            if user is None
            else [
                SummaryClaim(
                    text=user.text,
                    subject_id="user_self",
                    modality="asserted",
                    sources=[
                        {
                            "message_id": user.id,
                            "message_version": user.version,
                            "evidence_text": user.text,
                        }
                    ],
                )
            ]
        )
        return EpisodeSummary(
            topic_kinds=["chat"],
            subject_ids=["user_self"],
            claims=claims,
            covered_message_ids=[message.id for message in block.messages],
        )


class FakeSender:
    def __init__(self) -> None:
        self.results: deque[SendResult] = deque()
        self.sent: list[tuple[int, str]] = []
        self.next_id = 100000

    async def send_text(self, chat_id: int, text: str) -> SendResult:
        self.sent.append((chat_id, text))
        self.next_id += 1
        return (
            self.results.popleft()
            if self.results
            else SendResult("delivered", self.next_id)
        )


def update(
    number: int,
    text: str,
    user: int = 1,
    message_id: int | None = None,
    edited: bool = False,
    language: str = "en",
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "message_id": message_id or number,
        "date": 1900000000,
        "chat": {"id": user, "type": "private"},
        "from": {"id": user, "is_bot": False, "language_code": language},
        "text": text,
    }
    if text.startswith("/"):
        message["entities"] = [
            {"type": "bot_command", "offset": 0, "length": len(text.split()[0])}
        ]
    return {"update_id": number, "edited_message" if edited else "message": message}


async def conversation(app: Application, user: int = 1) -> Any:
    async with app.database.read() as tx:
        row = await fetchone(
            tx, "SELECT id FROM conversations WHERE external_user_id=?", (user,)
        )
        assert row is not None
    return await app.conversations.get(row["id"])


async def scalar(app: Application, sql: str, parameters: tuple[Any, ...] = ()) -> Any:
    async with app.database.read() as tx:
        row = await fetchone(tx, sql, parameters)
        assert row is not None
        return row[0]


def interpretation_operation(
    message: Any,
    predicate: str = "coffee",
    value: Any = "black with honey",
    kind: str = "preference",
    **overrides: Any,
) -> dict[str, Any]:
    operation = {
        "operation": "add",
        "kind": kind,
        "predicate": predicate,
        "value": value,
        "sources": [
            {
                "message_id": message.id,
                "message_version": message.version,
                "evidence_text": message.text,
            }
        ],
        **overrides,
    }
    return {
        "intent": "chat",
        "dialogue_mode": "continuation",
        "language": "en",
        "memory_operations": [operation],
    }


async def confirm_test_purchase(app: Application, event_key: str) -> None:
    current = await conversation(app)

    async def confirm(tx: Transaction) -> None:
        changed = await app.commerce.confirm(
            tx,
            current.scope,
            MockPurchase(actor_user_id=current.external_user_id, event_key=event_key),
            HandoffContext.model_validate(
                await app.memory.handoff_data(tx, current.scope)
            ),
        )
        if changed:
            confirmed = await app.conversations.get(current.id, tx)
            await app.delivery_repository.prepare(
                tx,
                confirmed.snapshot,
                OutboxDraft(
                    action_key=f"test-handoff:{event_key}",
                    kind="handoff",
                    text=service_text("handoff", confirmed.language),
                    deadline=app.clock.now() + 60,
                ),
                app.clock.now(),
            )

    await app.database.write(confirm)
    await app._cancel_scope(current.id)
