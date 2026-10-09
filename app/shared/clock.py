import time
from dataclasses import dataclass


class SystemClock:
    def now(self) -> float:
        return time.time()


@dataclass
class MutableClock:
    timestamp: float

    def now(self) -> float:
        return self.timestamp

    def advance(self, seconds: float) -> None:
        self.timestamp += seconds
