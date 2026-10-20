from typing import Any

from app.application import Application
from app.modules.conversations.entities import Message
from tests.helpers import conversation, update


def candidate(
    message: Message, predicate: str, value: Any, **changes: Any
) -> dict[str, Any]:
    return {
        "predicate": predicate,
        "value": value,
        "message_id": message.id,
        "message_version": message.version,
        "evidence_text": message.text,
        **changes,
    }


async def reply_with_projection(app: Application, values: list[dict[str, Any]]) -> None:
    app.model.responses["dialogue"].append(
        {
            "parts": ["Got it"],
            "language": "en",
            "memory_candidates": values,
        }
    )
    await app.drain()


async def last_input(app: Application) -> Message:
    current = await conversation(app)
    messages = await app.conversations.messages(current.scope)
    return next(message for message in reversed(messages) if message.role == "user")


async def test_correction_preserves_sources_and_supersedes_fact(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "Otto is five")])
    first = await last_input(app)
    await reply_with_projection(app, [candidate(first, "dog_age", 5)])
    current = await conversation(app)
    original = (await app.memory.repository.items(current.scope))[0]
    await app.telegram.ingest([update(2, "Actually Otto is six")])
    correction = await last_input(app)
    await reply_with_projection(
        app,
        [
            candidate(
                correction,
                "dog_age",
                6,
                operation="supersede",
                target_memory_id=original["id"],
            )
        ],
    )
    facts = await app.memory.repository.items(current.scope, history=True)
    assert {(item["value"], item["validity_status"]) for item in facts} == {
        (5, "superseded"),
        (6, "current"),
    }
    latest = next(item for item in facts if item["value"] == 6)
    assert latest["sources"][0]["message_id"] == correction.id
    await app.telegram.ingest([update(3, "What do you remember?")])
    await app.drain()
    request = app.model.calls[-1][1]
    assert [item["value"] for item in request["state"]["facts"]] == [6]
    assert [
        item["content"] for item in request["CHAT_MESSAGES"] if item["role"] == "user"
    ] == ["Otto is five", "Actually Otto is six", "What do you remember?"]


async def test_edited_source_invalidates_old_projection(app: Application) -> None:
    await app.telegram.ingest([update(1, "I live in Rome")])
    first = await last_input(app)
    await reply_with_projection(app, [candidate(first, "residence", "Rome")])
    await app.telegram.ingest([update(2, "I live in Paris", message_id=1, edited=True)])
    edited = await last_input(app)
    assert edited.version == 2
    current = await conversation(app)
    assert not await app.memory.repository.items(current.scope)
    await reply_with_projection(app, [candidate(edited, "residence", "Paris")])
    facts = await app.memory.repository.items(current.scope)
    assert len(facts) == 1 and facts[0]["value"] == "Paris"
    assert facts[0]["sources"][0]["message_version"] == 2


async def test_unknown_subject_does_not_merge_another_person_into_user(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "My sister is a teacher")])
    source = await last_input(app)
    await reply_with_projection(
        app, [candidate(source, "occupation", "teacher", subject_id="sister")]
    )
    current = await conversation(app)
    assert not await app.memory.repository.items(current.scope)
    assert app.sender.sent == [(1, "Got it")]
