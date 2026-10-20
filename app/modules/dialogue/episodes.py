from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def calendar_context(timestamp: float) -> dict[str, Any]:
    today = datetime.fromtimestamp(timestamp, UTC).date()
    sunday = today - timedelta(days=(today.weekday() + 1) % 7 or 7)
    return {
        "today": today.isoformat(),
        "weekday": today.strftime("%A"),
        "last_completed_weekend": {
            "starts_on": (sunday - timedelta(days=1)).isoformat(),
            "ends_on": sunday.isoformat(),
        },
    }


class CharacterEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    date: date
    time_of_day: str
    action: str
    details: list[str] = Field(default_factory=list)
    companions: list[str] | Literal["alone", "unknown"]


class CharacterEpisode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    starts_on: date
    ends_on: date
    authorship: Literal["approved_fictional_character_history"]
    events: list[CharacterEvent] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dates(self) -> "CharacterEpisode":
        if self.starts_on > self.ends_on or any(
            not self.starts_on <= event.date <= self.ends_on for event in self.events
        ):
            raise ValueError("Character episode events must be within its dates")
        return self


class CharacterHistory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    recent_days: int = Field(ge=1, le=90)
    episodes: list[CharacterEpisode]

    def before(self, timestamp: float) -> list[dict[str, Any]]:
        today = datetime.fromtimestamp(timestamp, UTC).date()
        weekend = calendar_context(timestamp)["last_completed_weekend"]
        projected = []
        for episode in self.episodes:
            if 0 < (today - episode.ends_on).days <= self.recent_days:
                item = episode.model_dump(mode="json")
                for event, source in zip(item["events"], episode.events, strict=True):
                    event["days_ago"] = (today - source.date).days
                item["relative_period"] = (
                    "last_completed_weekend"
                    if item["starts_on"] == weekend["starts_on"]
                    and item["ends_on"] == weekend["ends_on"]
                    else "earlier_dated_episode"
                )
                projected.append(item)
        return projected


CHARACTER_HISTORY = CharacterHistory.model_validate_json(
    (Path(__file__).with_name("resources") / "episodes.json").read_text(
        encoding="utf-8"
    )
)
