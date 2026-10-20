from app.modules.dialogue.references import model_context, translate_references


def test_reference_codec_preserves_words_roles_and_unknown_references() -> None:
    data = {
        "original_messages": [
            {
                "id": "long-message-id",
                "version": 2,
                "sequence": 4,
                "role": "user",
                "text": "I said long-message-id, not S1",
            }
        ],
        "CHAT_MESSAGES": [
            {"role": "user", "content": "I said long-message-id, not S1"}
        ],
        "pending_ids": ["long-message-id"],
        "boundaries": [
            {
                "id": "long-memory-id",
                "value": "long-message-id",
                "sources": [{"message_id": "long-message-id"}],
            }
        ],
    }
    metadata, aliases = model_context(data)
    assert metadata["pending_ids"] == ["S1"]
    assert metadata["boundaries"][0]["id"] == "F1"
    assert metadata["boundaries"][0]["value"] == "long-message-id"
    assert metadata["boundaries"][0]["sources"][0]["message_id"] == "S1"
    draft = {
        "parts": ["S1 is just something you typed"],
        "source_message_ids": ["S1", "S999"],
        "memory_candidates": [
            {"message_id": "S1", "target_memory_id": "F1", "evidence_text": "S1"}
        ],
    }
    restored = translate_references(draft, aliases)
    assert restored["parts"] == draft["parts"]
    assert restored["source_message_ids"] == ["long-message-id", "S999"]
    assert restored["memory_candidates"][0] == {
        "message_id": "long-message-id",
        "target_memory_id": "long-memory-id",
        "evidence_text": "S1",
    }
    assert data["pending_ids"] == ["long-message-id"]
    assert data["CHAT_MESSAGES"][0]["role"] == "user"
