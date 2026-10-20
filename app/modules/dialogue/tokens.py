import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tiktoken

from app.shared.config import MemorySettings


class TokenCounter:
    def __init__(self, model: str) -> None:
        os.environ.setdefault(
            "TIKTOKEN_CACHE_DIR",
            str(Path(__file__).parents[2] / "resources" / "tokenizer"),
        )
        try:
            self.encoding = tiktoken.encoding_for_model(model.removeprefix("openai/"))
        except KeyError:
            self.encoding = tiktoken.get_encoding("o200k_base")

    def count(self, value: Any) -> int:
        text = (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, default=str)
        )
        return len(self.encoding.encode(text, disallowed_special=()))


@dataclass(frozen=True)
class ContextBudget:
    settings: MemorySettings
    counter: TokenCounter
