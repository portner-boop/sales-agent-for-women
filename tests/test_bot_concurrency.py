import asyncio
from typing import Any

from app.application import Application
from app.modules.agent.context import TokenCounter
from app.modules.agent.entities import SchemaT
from app.modules.agent.schemas import ReplyProposal
from app.shared.config import Settings
from tests.helpers import ScriptedModel, conversation, scalar, update


async def test_five_simultaneous_users_have_isolated_memory_and_replies(
    app: Application,
) -> None:
    ready = asyncio.Event()

    class ParallelModel(ScriptedModel):
        def __init__(self) -> None:
            super().__init__()
            self.active_generators = 0
            self.maximum_generators = 0

        async def complete(
            self,
            task: str,
            rules: str,
            data: dict[str, Any],
            schema: type[SchemaT],
            deadline_seconds: float,
        ) -> SchemaT:
            if task == "interpret":
                self.calls.append((task, data))
                message = data["CURRENT_INPUT"][-1]
                name = message["text"].removeprefix("Call me ").removesuffix(".")
                return schema.model_validate(
                    {
                        "language": "en",
                        "intent": "introduce",
                        "dialogue_mode": "continuation",
                        "memory_operations": [
                            {
                                "operation": "add",
                                "kind": "fact",
                                "predicate": "preferred_name",
                                "value": name,
                                "sources": [
                                    {
                                        "message_id": message["id"],
                                        "message_version": message["version"],
                                        "evidence_text": message["text"],
                                    }
                                ],
                            }
                        ],
                    }
                )
            if task == "generate":
                self.calls.append((task, data))
                self.active_generators += 1
                self.maximum_generators = max(
                    self.maximum_generators, self.active_generators
                )
                if self.active_generators == 5:
                    ready.set()
                try:
                    await asyncio.wait_for(ready.wait(), 3)
                    facts = data["MEMORY_DATA"]["facts"]
                    name = next(
                        item["value"]
                        for item in facts
                        if item["predicate"] == "preferred_name"
                    )
                    return schema.model_validate(
                        ReplyProposal(
                            reply=f"Good to meet you, {name}.", dialogue_act="greet"
                        ).model_dump()
                    )
                finally:
                    self.active_generators -= 1
            return await super().complete(task, rules, data, schema, deadline_seconds)

    model = ParallelModel()
    app.agent.model = model
    names = {1: "Ana", 2: "Bea", 3: "Clara", 4: "Diana", 5: "Eva"}
    await app.telegram.ingest(
        [update(user, f"Call me {name}.", user=user) for user, name in names.items()]
    )
    await asyncio.wait_for(
        asyncio.gather(*(app.turn_worker.run_once() for _ in names)), 5
    )
    await app.drain()
    assert model.maximum_generators == 5
    assert dict(app.sender.sent) == {
        user: f"Good to meet you, {name}" for user, name in names.items()
    }
    scopes = []
    for user, name in names.items():
        current = await conversation(app, user)
        scopes.append(current.id)
        facts = await app.memory.repository.items(current.scope)
        assert [item["value"] for item in facts] == [name]
        assert all(
            message.text == f"Call me {name}."
            or message.text == f"Good to meet you, {name}"
            for message in await app.conversations.messages(current.scope)
        )
    assert len(set(scopes)) == 5
    assert await scalar(app, "SELECT COUNT(*) FROM said_ledger") == 5


async def test_start_uses_server_identity_without_llm(app: Application) -> None:
    await app.telegram.ingest([update(1, "/start", user=17)])
    await app.drain()
    assert app.sender.sent == [
        (
            17,
            "I'm Marko, a virtual companion. This is a test chat. /start starts or resumes a conversation, /stop stops messages, /privacy shows data terms, /forget deletes memory.",
        )
    ]
    assert not app.model.calls


async def test_users_outside_tester_list_can_chat_after_stop_start(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "hello", user=17)])
    await app.drain()
    before = await conversation(app, 17)
    app.clock.advance(86400)
    await app.telegram.ingest([update(2, "/stop", user=17)])
    await app.drain()
    assert not app.commerce.can_reply(await conversation(app, 17))
    await app.telegram.ingest(
        [update(3, "/start", user=17), update(4, "back", user=17)]
    )
    await app.drain()
    current = await conversation(app, 17)
    assert current.id == before.id and current.epoch == before.epoch
    assert app.commerce.can_reply(current)
    assert (
        await scalar(app, "SELECT COUNT(*) FROM said_ledger WHERE action_kind='reply'")
        == 2
    )


def test_token_budget_counter_supports_non_openai_gateway_model() -> None:
    assert TokenCounter("anthropic/claude-sonnet-4-6").count("Привет, Марко") > 0


def test_gateway_endpoint_and_model_have_no_embedded_defaults() -> None:
    import pytest

    settings = Settings()
    assert settings.llm.base_url == ""
    assert settings.llm.model == ""
    with pytest.raises(ValueError, match="BASE_URL"):
        settings.require_credentials()
