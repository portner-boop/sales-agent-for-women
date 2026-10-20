from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.modules.dialogue.episodes import CHARACTER_HISTORY, CharacterEpisode
from app.modules.dialogue.persona import persona_projection


def timestamp(day: str) -> float:
    return datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp()


def test_approved_character_history_is_dated_and_excludes_future_and_expired_events() -> (
    None
):
    assert not CHARACTER_HISTORY.before(timestamp("2026-10-17"))
    episodes = persona_projection([], timestamp("2026-10-20"))["authored_episodes"]
    assert len(episodes) == 1
    assert episodes[0]["authorship"] == "approved_fictional_character_history"
    assert episodes[0]["events"][0]["companions"] == ["Ante"]
    assert episodes[0]["events"][1]["companions"] == "alone"
    assert all(event["companions"] == "alone" for event in episodes[0]["events"][2:])
    assert [event["days_ago"] for event in episodes[0]["events"]] == [3, 2, 2, 2]
    assert not CHARACTER_HISTORY.before(timestamp("2026-11-15"))
    episodes[0]["events"][0]["companions"].append("invented")
    assert CHARACTER_HISTORY.before(timestamp("2026-10-20"))[0]["events"][0][
        "companions"
    ] == ["Ante"]


def test_character_episode_rejects_inconsistent_event_dates() -> None:
    episode = CHARACTER_HISTORY.episodes[0].model_dump(mode="json")
    episode["events"][0]["date"] = "2026-10-20"
    with pytest.raises(ValidationError):
        CharacterEpisode.model_validate(episode)


def test_relative_weekend_is_explicit_without_model_epoch_arithmetic() -> None:
    from app.modules.dialogue.episodes import calendar_context

    assert calendar_context(timestamp("2026-10-20"))["last_completed_weekend"] == {
        "starts_on": "2026-10-17",
        "ends_on": "2026-10-18",
    }
    assert (
        CHARACTER_HISTORY.before(timestamp("2026-10-20"))[0]["relative_period"]
        == "last_completed_weekend"
    )
    assert (
        calendar_context(timestamp("2026-10-18"))["last_completed_weekend"]["ends_on"]
        == "2026-10-11"
    )
    assert (
        CHARACTER_HISTORY.before(timestamp("2026-10-27"))[0]["relative_period"]
        == "earlier_dated_episode"
    )
