import asyncio
from collections import defaultdict, deque
from typing import Any

from app.application import Application
from app.modules.agent.entities import SchemaT
from app.modules.conversations.entities import Conversation
from app.modules.delivery.entities import SendResult
from app.shared.storage import fetchone


class ScriptedModel:
    def __init__(self) -> None:
        self.responses: dict[str, deque[dict[str, Any] | Exception]] = defaultdict(
            deque
        )
        self.calls: list[tuple[str, dict[str, Any]]] = []
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
        self.entered.set()
        if self.release is not None:
            await self.release.wait()
        if self.responses[task]:
            result = self.responses[task].popleft()
            if isinstance(result, Exception):
                raise result
        else:
            number = len(self.calls)
            latest = next(
                (
                    item["text"]
                    for item in reversed(data["original_messages"])
                    if item["role"] == "user"
                ),
                "",
            )
            language = (
                "ru"
                if any("а" <= char.casefold() <= "я" for char in latest)
                else data["language"]
            )
            text = (
                f"Ты застала меня врасплох {number}"
                if language == "ru"
                else f"You caught me off guard {number}"
            )
            result = {"parts": [text], "language": language}
        return schema.model_validate(result)


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


async def conversation(app: Application, user: int = 1) -> Conversation:
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
