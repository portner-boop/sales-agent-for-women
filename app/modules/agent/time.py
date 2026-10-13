import re
import unicodedata
from collections.abc import Sequence
from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.modules.agent.schemas import CurrentTimeRequest
from app.modules.conversations.entities import Message

PLACE_PATTERNS = {
    "Split": r"\b(?:split|сплит(?:е|а)?)\b",
    "Zagreb": r"\b(?:zagreb|загреб(?:е|а)?)\b",
}
TIME_QUESTION = re.compile(
    r"\b(?:what(?:'s|’s| is) the (?:local |current )?time|what time is it|"
    r"time in|время в|сколько (?:сейчас )?времени|который час|какое (?:сейчас )?время)\b",
    re.I,
)
STANDALONE_TIME_QUESTION = re.compile(
    r"(?:what time is it|what(?:'s|’s| is) the (?:local |current )?time|"
    r"сколько(?: сейчас)? времени|который час|какое(?: сейчас)? время)\s+(?:now\s+|сейчас\s+)?"
    r"(?:in|в)\s+(?:split|zagreb|сплите|загребе)(?:\s+(?:now|сейчас))?"
    r"|(?:time|время)\s+(?:in|в)\s+(?:split|zagreb|сплите|загребе)",
    re.I,
)


def current_time_facts(text: str, timestamp: float) -> list[dict[str, str]]:
    if not TIME_QUESTION.search(text):
        return []
    local = datetime.fromtimestamp(timestamp, UTC).astimezone(ZoneInfo("Europe/Zagreb"))
    return [
        {
            "city": city,
            "timezone": "Europe/Zagreb",
            "local_time": local.strftime("%H:%M"),
            "local_date": local.date().isoformat(),
            "utc_offset": local.strftime("%z"),
        }
        for city, pattern in PLACE_PATTERNS.items()
        if re.search(pattern, text, re.I)
    ]


def standalone_time_reply(text: str, timestamp: float, language: str) -> str | None:
    if not STANDALONE_TIME_QUESTION.fullmatch(text.strip(" \n\t?!.,")):
        return None
    facts = current_time_facts(text, timestamp)
    if len(facts) != 1:
        return None
    fact = facts[0]
    if language.split("-")[0] == "ru":
        city = {"Split": "Сплите", "Zagreb": "Загребе"}[fact["city"]]
        return f"В {city} сейчас {fact['local_time']}"
    return f"It’s {fact['local_time']} in {fact['city']} right now"


def resolve_current_times(
    requests: Sequence[CurrentTimeRequest],
    messages: Sequence[Message],
    timestamp: float,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    originals = {
        message.id: unicodedata.normalize("NFKC", message.text).casefold()
        for message in messages
        if message.role == "user" and not message.metadata.get("forwarded")
    }
    facts: list[dict[str, str]] = []
    unresolved: list[dict[str, str]] = []
    for request in requests:
        source = originals.get(request.source_message_id)
        location = unicodedata.normalize("NFKC", request.location).casefold()
        if source is None or location not in source:
            unresolved.append(
                {"location": request.location, "reason": "unverified_location"}
            )
            continue
        if request.timezone is None:
            unresolved.append(
                {"location": request.location, "reason": "ambiguous_timezone"}
            )
            continue
        try:
            zone = ZoneInfo(request.timezone)
        except (ZoneInfoNotFoundError, ValueError):
            unresolved.append(
                {"location": request.location, "reason": "unknown_timezone"}
            )
            continue
        local = datetime.fromtimestamp(timestamp, UTC).astimezone(zone)
        fact = {
            "city": request.location,
            "timezone": request.timezone,
            "local_time": local.strftime("%H:%M"),
            "local_date": local.date().isoformat(),
            "utc_offset": local.strftime("%z"),
        }
        if fact not in facts:
            facts.append(fact)
    return facts, unresolved
