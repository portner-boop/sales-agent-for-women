import json
from pathlib import Path

from pydantic import Field

from app.modules.dialogue.schemas import ChatMessage, DialogueModel


class QualityCase(DialogueModel):
    id: str
    language: str
    history: list[ChatMessage]
    input: str
    expected_move: str
    requirements: str
    forbidden: list[str]


class ConversationCase(DialogueModel):
    id: str
    language: str
    requirements: str
    steps: list[str] = Field(min_length=20, max_length=20)


def quality_cases() -> list[QualityCase]:
    data = json.loads(Path(__file__).with_name("quality_cases.json").read_text())
    return [QualityCase.model_validate(case) for case in data["cases"]]


def conversation_cases() -> list[ConversationCase]:
    data = json.loads(Path(__file__).with_name("conversation_cases.json").read_text())
    return [ConversationCase.model_validate(case) for case in data["cases"]]


def identity_cases() -> list[QualityCase]:
    data = json.loads(Path(__file__).with_name("identity_cases.json").read_text())
    return [QualityCase.model_validate(case) for case in data["cases"]]


def identity_conversation_cases() -> list[ConversationCase]:
    data = json.loads(Path(__file__).with_name("identity_cases.json").read_text())
    return [
        ConversationCase.model_validate(case) for case in data["conversation_cases"]
    ]


def additional_language_cases() -> list[QualityCase]:
    data = json.loads(Path(__file__).with_name("additional_languages.json").read_text())
    return [QualityCase.model_validate(case) for case in data["cases"]]
