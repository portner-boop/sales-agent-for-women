from scripts.server_dialogue import local_timestamp, render_export, replay_steps


def test_replay_preserves_corrections_and_omits_service_commands() -> None:
    messages = [
        {"sequence": 1, "role": "service", "text": "Ready", "received_at": 1},
        {"sequence": 2, "role": "user", "text": "Destroyed", "received_at": 2},
        {"sequence": 3, "role": "user", "text": "Built", "received_at": 3},
        {"sequence": 4, "role": "assistant", "text": "A response", "received_at": 4},
        {"sequence": 5, "role": "user", "text": "/forget", "received_at": 5},
        {"sequence": 6, "role": "user", "text": "Hello?", "received_at": 6},
    ]
    assert replay_steps(messages) == [
        {"text": "Destroyed\nBuilt", "sequences": [2, 3], "received_at": 2},
        {"text": "Hello?", "sequences": [6], "received_at": 6},
    ]


def test_export_keeps_literal_text_and_first_delivery_latency() -> None:
    messages = [
        {
            "sequence": 1,
            "epoch": 1,
            "role": "user",
            "text": "a typo?\nline two",
            "received_at": 100,
        },
        {
            "sequence": 2,
            "epoch": 1,
            "role": "assistant",
            "text": "I’m here",
            "received_at": 131,
        },
        {
            "sequence": 3,
            "epoch": 1,
            "role": "assistant",
            "text": "Another part",
            "received_at": 150,
        },
    ]
    result = render_export(
        {
            "exported_at": "now",
            "source": "snapshot",
            "conversations": [
                {"label": "Case", "scope_id": "test", "messages": messages, "runs": []}
            ],
        },
        "checksum",
    )
    assert "> a typo?\n> line two" in result
    assert "> I’m here" in result
    assert result.count("первый ответ через") == 1
    assert "31.00 с" in result
    assert local_timestamp(0) == "1970-01-01T03:00:00.000+03:00"
