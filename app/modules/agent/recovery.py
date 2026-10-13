import re
import unicodedata
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from math import isfinite

from openai import APIConnectionError, APIStatusError
from pydantic import ValidationError

from app.modules.agent.errors import CallBudgetError, ModelOutputError

ACTIVITY_QUESTIONS = re.compile(
    r"(?:(?:а|ну|так|и|скажи|расскажи)\s+)*"
    r"(?:(?:что|чем)\s+(?:ты\s+)?(?:сегодня\s+|сейчас\s+|вчера\s+)?"
    r"(?:делал|делаешь|занимался|занимаешься|занят)"
    r"(?:\s+ты)?(?:\s+(?:сегодня|сейчас|вчера))?|"
    r"как\s+(?:у тебя\s+)?(?:прош[её]л|проходит)\s+(?:твой\s+)?день|"
    r"what\s+(?:did you do|have you been doing|are you doing)"
    r"(?:\s+(?:today|yesterday|now))?|"
    r"how\s+(?:was|is)\s+your\s+day|"
    r"qu[eé]\s+(?:hiciste|has hecho|haces)(?:\s+(?:hoy|ayer|ahora))?|"
    r"(?:o que|que)\s+(?:voc[eê]\s+)?(?:fez|faz|est[aá] fazendo)"
    r"(?:\s+(?:hoje|ontem|agora))?)",
    re.I,
)

SERVICE_FAILURE_TEXT = re.compile(
    r"(?:couldn['’]?t|could not)\s+give\s+you\s+(?:a\s+)?proper\s+reply|"
    r"не смог\s+ответить\s+как следует|"
    r"(?:сейчас\s+)?не удалось\s+подготовить\s+ответ|"
    r"не получилось\s+над[её]жно\s+обработать\s+(?:это\s+)?сообщение|"
    r"no he podido responderte bien|não consegui responder direito",
    re.I,
)


def activity_reply(text: str, language: str) -> str | None:
    normalized = unicodedata.normalize("NFKC", text).strip(" \n\t.!?¿¡")
    if not ACTIVITY_QUESTIONS.fullmatch(normalized):
        return None
    if re.search(r"[а-яё]", normalized, re.I):
        language = "ru"
    elif re.search(r"\b(?:voc[eê]|fez|faz|fazendo)\b", normalized, re.I):
        language = "pt"
    elif re.search(r"\b(?:hiciste|has hecho|haces)\b", normalized, re.I):
        language = "es"
    else:
        language = "en"
    values = {
        "ru": "Сейчас читаю твои сообщения и отвечаю тебе",
        "en": "Right now I’m reading your messages and replying to you",
        "es": "Ahora estoy leyendo tus mensajes y respondiéndote",
        "pt": "Agora estou lendo suas mensagens e respondendo a você",
    }
    return values.get(language.split("-")[0], values["en"])


def service_failure_reply(text: str) -> bool:
    return bool(SERVICE_FAILURE_TEXT.search(unicodedata.normalize("NFKC", text)))


def can_retry_response(error: Exception) -> bool:
    return not isinstance(error, APIStatusError) or retry_delay(error) is not None


def failure_code(error: Exception) -> str:
    if isinstance(error, CallBudgetError):
        return "CALL_BUDGET_EXHAUSTED"
    if isinstance(error, TimeoutError):
        return "DEADLINE_EXCEEDED"
    if isinstance(error, ValidationError):
        return "INVALID_STRUCTURED_OUTPUT"
    if isinstance(error, ModelOutputError):
        return str(error)
    if isinstance(error, APIStatusError):
        return f"GATEWAY_HTTP_{error.status_code}"
    if isinstance(error, APIConnectionError):
        return "GATEWAY_CONNECTION"
    return type(error).__name__


def retry_delay(error: Exception) -> float | None:
    if isinstance(error, (TimeoutError, APIConnectionError)):
        return 0.25
    if isinstance(error, (ValidationError, ModelOutputError)):
        return 0
    if not isinstance(error, APIStatusError) or error.status_code not in {
        408,
        409,
        429,
        500,
        502,
        503,
        504,
    }:
        return None
    header = error.response.headers.get("retry-after")
    if header is None:
        return 0.25
    try:
        seconds = float(header)
        return max(0, seconds) if isfinite(seconds) else None
    except ValueError:
        try:
            return max(
                0,
                (parsedate_to_datetime(header) - datetime.now(UTC)).total_seconds(),
            )
        except (ValueError, TypeError, OverflowError):
            return None


def structure_errors(error: Exception) -> list[str]:
    if isinstance(error, ValidationError):
        return [
            f"{'.'.join(map(str, item['loc']))}:{item['type']}"
            for item in error.errors(include_input=False, include_url=False)[:8]
        ]
    return [failure_code(error)]
