import tomllib
from pathlib import Path
from typing import Annotated
from unicodedata import category

from pydantic import BaseModel, ConfigDict, Field, field_validator

PunctuationCharacter = Annotated[str, Field(min_length=1, max_length=1)]


class ChatRendering(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    omit_terminal_period: bool = True
    punctuation_replacements: dict[PunctuationCharacter, PunctuationCharacter] = Field(
        default_factory=dict
    )

    @field_validator("punctuation_replacements")
    @classmethod
    def validate_punctuation(cls, replacements: dict[str, str]) -> dict[str, str]:
        if any(
            not category(symbol).startswith("P")
            for pair in replacements.items()
            for symbol in pair
        ):
            raise ValueError("Rendering replacements must contain punctuation only")
        return replacements

    def render(self, text: str) -> str:
        rendered = text.translate(
            {
                ord(source): replacement
                for source, replacement in self.punctuation_replacements.items()
            }
        )
        stripped = rendered.rstrip()
        if (
            self.omit_terminal_period
            and len(stripped) > 1
            and stripped[-1] in ".。۔"
            and stripped[-2] != stripped[-1]
        ):
            return stripped[:-1]
        return rendered


CHAT_RENDERING = ChatRendering.model_validate(
    tomllib.loads(
        (Path(__file__).with_name("resources") / "rendering.toml").read_text(
            encoding="utf-8"
        )
    )
)
