from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DailyScene(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    starts_hour: int = Field(ge=0, le=23)
    ends_hour: int = Field(ge=1, le=24)
    setting: str = Field(min_length=1)
    activity: str = Field(min_length=1)


class DailyLife(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    authorship: Literal["approved_fictional_character_history"]
    scenes: list[DailyScene] = Field(min_length=1)

    @model_validator(mode="after")
    def cover_each_hour(self) -> "DailyLife":
        if any(
            sum(scene.starts_hour <= hour < scene.ends_hour for scene in self.scenes)
            != 1
            for hour in range(24)
        ):
            raise ValueError("Daily scenes must cover every hour once")
        return self

    def context(self, timestamp: float, timezone: str) -> dict[str, Any]:
        local = datetime.fromtimestamp(timestamp, ZoneInfo(timezone))
        scene = next(
            scene
            for scene in self.scenes
            if scene.starts_hour <= local.hour < scene.ends_hour
        )
        return {
            "timezone": timezone,
            "local_time": local.strftime("%H:%M"),
            "local_date": local.date().isoformat(),
            "utc_offset": local.strftime("%z"),
            "daily_scene": {"authorship": self.authorship, **scene.model_dump()},
        }


DAILY_LIFE = DailyLife.model_validate_json(
    (Path(__file__).with_name("resources") / "daily_life.json").read_text(
        encoding="utf-8"
    )
)
