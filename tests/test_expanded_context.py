from typing import Any

import pytest

from app.application import Application
from app.modules.agent.entities import SchemaT
from tests.helpers import scalar, update


async def test_large_coalesced_input_reaches_validation_and_delivery_intact(
    app: Application, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = [
        f"Entry {number}: " + "tea " * 3900 + f"End of entry {number}."
        for number in range(1, 13)
    ]
    inputs[0] = "My cat Pixel is five. " + inputs[0]
    inputs[6] = "Correction: Pixel is six, not five. " + inputs[6]
    inputs[-1] += " No questions. What is my cat's age now?"
    complete = app.model.complete
    sizes: dict[str, int] = {}

    async def capture(
        task: str,
        rules: str,
        data: dict[str, Any],
        schema: type[SchemaT],
        deadline_seconds: float,
    ) -> SchemaT:
        sizes[task] = (
            app.agent.builder.counter.count(
                {"rules": rules, "data": data, "schema": schema.model_json_schema()}
            )
            + 500
        )
        context = data.get("context", data)
        assert [item["text"] for item in context["CURRENT_INPUT"]] == inputs
        if task == "generate":
            app.model.responses[task].append(
                {
                    "reply": "Pixel is six now",
                    "dialogue_act": "answer",
                    "used_source_message_ids": [context["CURRENT_INPUT"][6]["id"]],
                }
            )
        return await complete(task, rules, data, schema, deadline_seconds)

    monkeypatch.setattr(app.model, "complete", capture)
    await app.telegram.ingest(
        [update(number, text) for number, text in enumerate(inputs, 1)]
    )
    await app.turn_worker.run_once()
    await app.delivery.run_once()
    assert set(sizes) == {"interpret", "generate", "validate"}
    assert all(48000 < size <= 64000 for size in sizes.values())
    assert app.sender.sent == [(1, "Pixel is six now")]
    assert await scalar(app, "SELECT status FROM agent_runs") == "prepared"
    assert await scalar(app, "SELECT COUNT(*) FROM messages WHERE pending=1") == 0
