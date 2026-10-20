from copy import deepcopy
from typing import Any

REFERENCE_FIELDS = {
    "id",
    "message_id",
    "target_memory_id",
    "pending_ids",
    "source_message_ids",
}


def translate_references(value: Any, mapping: dict[str, str], field: str = "") -> Any:
    if isinstance(value, dict):
        return {
            key: translate_references(item, mapping, key) for key, item in value.items()
        }
    if isinstance(value, list):
        return [translate_references(item, mapping, field) for item in value]
    if isinstance(value, str) and field in REFERENCE_FIELDS:
        return mapping.get(value, value)
    return value


def model_context(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    originals = data.get("original_messages", [])
    aliases = {
        str(item["id"]): f"S{position}" for position, item in enumerate(originals, 1)
    }
    memory_items = data.get("boundaries", []) + data.get("state", {}).get("facts", [])
    memory_ids = sorted({str(item["id"]) for item in memory_items if "id" in item})
    aliases.update(
        {
            identifier: f"F{position}"
            for position, identifier in enumerate(memory_ids, 1)
        }
    )
    metadata = deepcopy(
        {
            key: value
            for key, value in data.items()
            if key not in {"CHAT_MESSAGES", "original_messages", "examples"}
        }
    )
    metadata["message_index"] = [
        {
            "chat_position": position,
            **{key: item[key] for key in ("id", "version", "sequence", "role")},
            **({"user_text": item["text"]} if item["role"] == "user" else {}),
            **(
                {"delivery_kind": item["delivery_kind"]}
                if item.get("delivery_kind")
                else {}
            ),
        }
        for position, item in enumerate(originals, 1)
    ]
    return translate_references(metadata, aliases), {
        alias: identifier for identifier, alias in aliases.items()
    }
