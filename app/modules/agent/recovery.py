from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from math import isfinite

from openai import APIConnectionError, APIStatusError
from pydantic import ValidationError

from app.modules.agent.errors import CallBudgetError, ModelOutputError


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
