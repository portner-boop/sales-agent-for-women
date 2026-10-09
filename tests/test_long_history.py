import json

import pytest

from app.application import Application
from tests.helpers import conversation, interpretation_operation, scalar, update


@pytest.mark.parametrize("turns", [200, 1000])
async def test_long_history_keeps_exact_facts_corrections_boundaries_and_coverage(
    app: Application, turns: int, record_property: object
) -> None:
    coffee_id = ""
    name_id = ""
    for number in range(1, turns + 1):
        language = ("en", "ru", "es", "pt")[(number // 25) % 4]
        texts = {
            "en": f"Today I enjoyed walking through the park and seeing the flowers, day {number}.",
            "ru": f"Сегодня я гуляла в парке и смотрела на цветы, день {number}.",
            "es": f"Hola, hoy disfruté de un paseo por el parque y vi las flores, día {number}.",
            "pt": f"Olá, hoje gostei de caminhar pelo parque e ver as flores, dia {number}.",
        }
        text = (
            "I drink black coffee without milk, with honey."
            if number == 1
            else (
                "My name is Maria."
                if number == 2
                else (
                    "Do not call me darling."
                    if number == 3
                    else (
                        "Not Maria, call me Mariana."
                        if number == turns // 2
                        else texts[language]
                    )
                )
            )
        )
        await app.telegram.ingest([update(number, text)])
        current = await conversation(app)
        message = (await app.conversations.messages(current.scope))[-1]
        if number in {1, 2, 3, turns // 2}:
            if number == 1:
                response = interpretation_operation(
                    message, "coffee", "black without milk, with honey"
                )
            elif number == 2:
                response = interpretation_operation(
                    message, "preferred_name", "Maria", "fact"
                )
            elif number == 3:
                response = interpretation_operation(
                    message, "forbidden_address", "darling", "boundary"
                )
            else:
                response = interpretation_operation(
                    message,
                    "preferred_name",
                    "Mariana",
                    "fact",
                    operation="supersede",
                    target_memory_id=name_id,
                )
            app.model.responses["interpret"].append(response)
        await app.drain()
        facts = await app.memory.repository.items(current.scope)
        if number == 1:
            coffee_id = facts[0]["id"]
        if number == 2:
            name_id = next(
                item["id"] for item in facts if item["predicate"] == "preferred_name"
            )
        if number > 3:
            generated = [data for task, data in app.model.calls if task == "generate"][
                -1
            ]
            assert generated["ACTIVE_BOUNDARIES"][0]["value"] == "darling"
        if number % 100 == 0:
            app.clock.advance(1801)
    current = await conversation(app)
    facts = await app.memory.repository.items(current.scope)
    assert (
        next(item for item in facts if item["id"] == coffee_id)["value"]
        == "black without milk, with honey"
    )
    assert (
        next(item for item in facts if item["predicate"] == "preferred_name")["value"]
        == "Mariana"
    )
    assert len(app.sender.sent) == turns
    assert (
        await scalar(app, "SELECT COUNT(*) FROM agent_runs WHERE status!='prepared'")
        == 0
    )
    summaries = await scalar(
        app, "SELECT COUNT(*) FROM memory_summaries WHERE status='current'"
    )
    assert summaries >= 20
    covered = await app.memory.repository.covered_ids(current.scope)
    total = await scalar(app, "SELECT COUNT(*) FROM messages")
    assert len(covered) >= total - 24
    assert all(
        app.agent.builder.counter.count(
            [message.as_data() for message in block.messages]
        )
        <= app.settings.memory.block_tokens
        for block in app.model.blocks
    )
    assert all(
        sum(json.loads(manifest)["token_count_by_section"].values())
        <= app.settings.memory.context_tokens
        for manifest in await manifests(app)
    )
    record_property("turns", turns)
    record_property("summary_cycles", len(app.model.blocks))
    record_property("covered_messages", len(covered))
    record_property("total_messages", total)


async def manifests(app: Application) -> list[str]:
    from app.shared.storage import fetchall

    async with app.database.read() as tx:
        rows = await fetchall(tx, "SELECT manifest_json FROM agent_runs")
        return [row[0] for row in rows]
