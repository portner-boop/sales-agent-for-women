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
        "json_schema"
    )
    reasoning_effort: Literal["none", "low"] | None = "none"


class StorageSettings(ConfigModel):
    directory: Path = Path("data")
    queue_size: int = Field(default=256, ge=1)
    busy_timeout_ms: int = Field(default=5000, ge=1)
    backup_interval_seconds: int = Field(default=3600, ge=1)
    retention_days: int = Field(default=30, ge=1)


class RuntimeSettings(ConfigModel):
    tester_ids: frozenset[int] = frozenset()
    llm_concurrency: int = Field(default=5, ge=1)
    llm_max_calls: int = Field(default=8, ge=6, le=8)
    llm_attempt_seconds: float = Field(default=12, gt=0)
    turn_deadline_seconds: float = Field(default=15, gt=0)
    first_deadline_seconds: float = Field(default=10, gt=0)
    coalesce_seconds: float = Field(default=0.5, ge=0)
    max_coalesce_seconds: float = Field(default=2, gt=0)
    activity_gap_seconds: int = Field(default=1800, ge=1)
    max_message_chars: int = Field(default=16000, ge=1)
    kill_switch: bool = False
    reply_pacing_enabled: bool = True
    reply_pause_min_seconds: float = Field(default=2, ge=0, le=16)
    reply_pause_max_seconds: float = Field(default=4, ge=0, le=16)
    reply_typing_chars_per_second: float = Field(default=3, gt=0, allow_inf_nan=False)
    reply_delay_jitter_ratio: float = Field(default=0.2, ge=0, le=1)
    reply_delay_min_seconds: float = Field(default=5, ge=0, allow_inf_nan=False)
    reply_delay_max_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    first_reply_max_seconds: float = Field(default=10, gt=0, allow_inf_nan=False)
    followup_enabled: bool = False
    followup_min_seconds: float = Field(default=30, ge=1)
    followup_max_seconds: float = Field(default=60, ge=1)
    followup_ttl_seconds: float = Field(default=90, ge=10)

    @model_validator(mode="after")
    def validate_reply_pause(self) -> "RuntimeSettings":
        if self.reply_pause_min_seconds > self.reply_pause_max_seconds:
            raise ValueError("Minimum reply pause exceeds maximum")
        if self.reply_delay_min_seconds > self.reply_delay_max_seconds:
            raise ValueError("Minimum reply delay exceeds maximum")
        if self.followup_min_seconds > self.followup_max_seconds:
            raise ValueError("Minimum follow-up delay exceeds maximum")
        return self


class MemorySettings(ConfigModel):
    context_tokens: int = Field(default=12000, ge=2000)
    recent_turns: int = Field(default=6, ge=1)
    history_threshold_tokens: int = Field(default=6000, ge=1)
    block_tokens: int = Field(default=2500, ge=100)
    summary_tokens: int = Field(default=700, ge=64)
    summary_context_tokens: int = Field(default=900, ge=64)


class CommerceSettings(ConfigModel):
    package_minutes: int = Field(default=20, ge=1)
    preview_enabled: bool = False
    preview_min_replies: int = Field(default=12, ge=1)
    preview_max_replies: int = Field(default=20, ge=2)

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
