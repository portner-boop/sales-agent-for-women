import random
from collections.abc import Callable

from app.shared.config import ReplyTimingSettings


class DeliveryTiming:
    def __init__(
        self,
        settings: ReplyTimingSettings,
        uniform: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self.settings, self.uniform = settings, uniform

    def duration(self, text: str) -> float:
        if not self.settings.enabled:
            return 0
        pause = self.uniform(
            self.settings.pause_min_seconds, self.settings.pause_max_seconds
        )
        jitter = self.uniform(
            1 - self.settings.jitter_fraction, 1 + self.settings.jitter_fraction
        )
        seconds = (pause + len(text) / self.settings.characters_per_second) * jitter
        return min(
            self.settings.maximum_seconds, max(self.settings.minimum_seconds, seconds)
        )
