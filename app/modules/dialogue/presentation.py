import random
import re
import tomllib
from pathlib import Path
from typing import Annotated
from unicodedata import category

import emoji
from pydantic import BaseModel, ConfigDict, Field, field_validator

EMOTICONS = re.compile(
    r"[:;=][\-^']?[()]|[:;=][\-^']?[DPp/\\|*](?!\w)|(?<!\w)(?:8[\-^']?[)DPp]|[xX][dD]|\^_\^|[Tt]_[Tt])(?!\w)|<3|[()]{2,}"
)
SMILEY_SYMBOLS = re.compile("[♡♥❤❣❥❦❧☺☻]\ufe0f?")
WINK_TAIL = r"(?![-^']?[()DPp*])"
PunctuationCharacter = Annotated[str, Field(min_length=1, max_length=1)]


def strip_smileys(text: str) -> str:
    stripped = SMILEY_SYMBOLS.sub("", emoji.replace_emoji(text, ""))
    stripped = EMOTICONS.sub("", stripped)
    kept: list[str] = []
    opened: list[int] = []
    for character in stripped:
        if character == "(":
            opened.append(len(kept))
        elif character == ")":
            if not opened:
                continue
            opened.pop()
        kept.append(character)
    for index in reversed(opened):
        del kept[index]
    lines = [" ".join(line.split()) for line in "".join(kept).splitlines()]
    return re.sub(r" ([,.!?])", r"\1", "\n".join(line for line in lines if line))


TYPO_WORD = re.compile(r"\b[^\W\d_]{5,}\b")


def add_typo(text: str, rng: random.Random) -> str:
    words = [match for match in TYPO_WORD.finditer(text) if match.group().islower()]
    if not words:
        return text
    match = rng.choice(words)
    word = match.group()
    index = rng.randrange(1, len(word) - 2)
    kind = rng.choice(("swap", "drop", "double"))
    if kind == "swap":
        changed = word[:index] + word[index + 1] + word[index] + word[index + 2 :]
    elif kind == "drop":
        changed = word[:index] + word[index + 1 :]
    else:
        changed = word[: index + 1] + word[index:]
    return text[: match.start()] + changed + text[match.end() :]


class ChatRendering(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    omit_terminal_period: bool = True
    strip_smileys: bool = False
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
        rendered = strip_smileys(text) if self.strip_smileys else text
        for source, replacement in self.punctuation_replacements.items():
            rendered = re.sub(re.escape(source) + WINK_TAIL, replacement, rendered)
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
