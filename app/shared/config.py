import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from dynaconf import Dynaconf
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TelegramSettings(ConfigModel):
    token: SecretStr = SecretStr("")
    poll_timeout: int = Field(default=20, ge=1, le=50)
    proxy_url: str | None = None


class LLMSettings(ConfigModel):
    base_url: str = ""
    model: str = ""
    api_key: SecretStr = SecretStr("")
    output_tokens: int = Field(default=1200, ge=256)
    structured_mode: Literal["json_schema", "json_object", "prompt_json"] = (
        "json_object"
    )
    reasoning_effort: Literal["none", "low"] | None = "none"
    verbosity: Literal["low", "medium", "high"] | None = "low"


class StorageSettings(ConfigModel):
    directory: Path = Path("data")
    queue_size: int = Field(default=256, ge=1)
    busy_timeout_ms: int = Field(default=5000, ge=1)
    backup_interval_seconds: int = Field(default=3600, ge=1)
    retention_days: int = Field(default=30, ge=1)


class ReplyTimingSettings(ConfigModel):
    enabled: bool = True
    pause_min_seconds: float = Field(default=2, ge=0, allow_inf_nan=False)
    pause_max_seconds: float = Field(default=4, ge=0, allow_inf_nan=False)
    characters_per_second: float = Field(default=3, gt=0, allow_inf_nan=False)
    jitter_fraction: float = Field(default=0.2, ge=0, lt=1, allow_inf_nan=False)
    minimum_seconds: float = Field(default=5, ge=0, allow_inf_nan=False)
    maximum_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered_ranges(self) -> "ReplyTimingSettings":
        if self.pause_min_seconds > self.pause_max_seconds:
            raise ValueError("Reply pause range is reversed")
        if self.minimum_seconds > self.maximum_seconds:
            raise ValueError("Reply duration range is reversed")
        return self


class RuntimeSettings(ConfigModel):
    reply_timing: ReplyTimingSettings = ReplyTimingSettings()
    response_max_age_seconds: float = Field(default=180, gt=0)
    dialogue_max_calls: int = Field(default=2, ge=1, le=2)
    research_api_call_limit: int = Field(default=2500, ge=1, le=2500)
    llm_concurrency: int = Field(default=5, ge=1)
    llm_attempt_seconds: float = Field(default=12, gt=0)
    turn_deadline_seconds: float = Field(default=45, gt=0)
    first_deadline_seconds: float = Field(default=15, gt=0)
    coalesce_seconds: float = Field(default=0.5, ge=0)
    max_coalesce_seconds: float = Field(default=2, gt=0)
    activity_gap_seconds: int = Field(default=1800, ge=1)
    max_message_chars: int = Field(default=16000, ge=1)
    kill_switch: bool = False
    followup_enabled: bool = False
    followup_min_seconds: float = Field(default=60, ge=1)
    followup_max_seconds: float = Field(default=60, ge=1)
    followup_ttl_seconds: float = Field(default=90, ge=10)
    followup_max_count: int = Field(default=2, ge=1, le=2)
    followup_second_seconds: float = Field(default=180, ge=1)

    @model_validator(mode="after")
    def validate_followup_range(self) -> "RuntimeSettings":
        if self.followup_min_seconds > self.followup_max_seconds:
            raise ValueError("Minimum follow-up delay exceeds maximum")
        if (
            self.followup_max_count > 1
            and self.followup_second_seconds <= self.followup_max_seconds
        ):
            raise ValueError("Second follow-up must follow the first opportunity")
        return self


class MemorySettings(ConfigModel):
    context_tokens: int = Field(default=64000, ge=2000)


class CommerceSettings(ConfigModel):
    preview_min_replies: int = Field(default=15, ge=15, le=19)
    preview_max_replies: int = Field(default=20, ge=2, le=20)

    @model_validator(mode="after")
    def validate_preview_range(self) -> "CommerceSettings":
        if self.preview_max_replies <= self.preview_min_replies:
            raise ValueError("Preview maximum must exceed minimum")
        return self


class Settings(ConfigModel):
    environment: Literal["demo", "test"] = "demo"
    telegram: TelegramSettings = TelegramSettings()
    llm: LLMSettings = LLMSettings()
    storage: StorageSettings = StorageSettings()
    runtime: RuntimeSettings = RuntimeSettings()
    memory: MemorySettings = MemorySettings()
    commerce: CommerceSettings = CommerceSettings()

    def require_credentials(self, telegram: bool = True) -> None:
        if not self.llm.base_url.strip():
            raise ValueError("BEHAPPY_LLM__BASE_URL is required")
        if not self.llm.model.strip():
            raise ValueError("BEHAPPY_LLM__MODEL is required")
        if not self.llm.api_key.get_secret_value():
            raise ValueError("BEHAPPY_LLM__API_KEY is required")
        if telegram and not self.telegram.token.get_secret_value():
            raise ValueError("BEHAPPY_TELEGRAM__TOKEN is required")


def normalize_settings_keys(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key).lower(): normalize_settings_keys(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [normalize_settings_keys(item) for item in value]
    return value


def load_settings(path: Path | None = None) -> Settings:
    root = Path(__file__).parents[1]
    settings_path = path or Path(
        os.environ.get("BEHAPPY_SETTINGS", str(root / "config" / "settings.toml"))
    )
    if not settings_path.exists():
        raise ValueError(f"Settings file does not exist: {settings_path}")
    source = Dynaconf(
        settings_files=[str(settings_path)],
        environments=True,
        envvar_prefix="BEHAPPY",
        load_dotenv=True,
        dotenv_path=str(root.parent / ".env"),
    )
    values = {
        key: normalize_settings_keys(source.get(key)) for key in Settings.model_fields
    }
    return Settings.model_validate(
        {key: value for key, value in values.items() if value is not None}
    )
