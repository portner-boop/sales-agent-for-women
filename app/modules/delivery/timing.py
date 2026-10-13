import random

from app.shared.config import RuntimeSettings


def message_delay(text: str, settings: RuntimeSettings) -> float:
    pause = random.uniform(
        settings.reply_pause_min_seconds, settings.reply_pause_max_seconds
    )
    jitter = random.uniform(
        1 - settings.reply_delay_jitter_ratio, 1 + settings.reply_delay_jitter_ratio
    )
    seconds = (pause + len(text) / settings.reply_typing_chars_per_second) * jitter
    return min(
        settings.reply_delay_max_seconds, max(settings.reply_delay_min_seconds, seconds)
    )


def response_delays(
    texts: tuple[str, ...],
    received_at: float,
    now: float,
    first_reply: bool,
    settings: RuntimeSettings,
) -> tuple[float, ...]:
    if not settings.reply_pacing_enabled:
        return (0,) + tuple(
            random.uniform(
                settings.reply_pause_min_seconds, settings.reply_pause_max_seconds
            )
            for _ in texts[1:]
        )
    intervals = tuple(message_delay(text, settings) for text in texts)
    first_interval = (
        min(intervals[0], settings.first_reply_max_seconds)
        if first_reply
        else intervals[0]
    )
    return (max(0, received_at + first_interval - now),) + intervals[1:]
