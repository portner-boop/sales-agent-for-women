from dataclasses import dataclass


@dataclass(frozen=True)
class Offer:
    id: str
    title: str
    duration_minutes: int
    is_test: bool = True
