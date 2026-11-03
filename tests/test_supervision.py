import asyncio
import socket
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application import Application
from app.shared.supervision import notify_service


def test_service_notification_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    factory = MagicMock()
    monkeypatch.setattr("app.shared.supervision.socket.socket", factory)

    notify_service("READY=1")

    factory.assert_not_called()


def test_service_notification_reaches_unix_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as directory:
        address = str(Path(directory) / "notify.sock")
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as listener:
            listener.bind(address)
            listener.settimeout(1)
            monkeypatch.setenv("NOTIFY_SOCKET", address)

            notify_service("READY=1\nWATCHDOG=1")

            assert listener.recv(1024) == b"READY=1\nWATCHDOG=1"


def test_service_notification_supports_abstract_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOTIFY_SOCKET", "@behappy-notify")
    factory = MagicMock()
    monkeypatch.setattr("app.shared.supervision.socket.socket", factory)

    notify_service("WATCHDOG=1")

    factory.return_value.__enter__.return_value.sendto.assert_called_once_with(
        b"WATCHDOG=1", "\0behappy-notify"
    )


def test_service_notification_failure_is_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as directory:
        monkeypatch.setenv("NOTIFY_SOCKET", str(Path(directory) / "missing.sock"))

        with pytest.raises(OSError):
            notify_service("READY=1")


@pytest.mark.parametrize("failure", [None, "schedule", "metrics"])
async def test_watchdog_requires_successful_database_maintenance(
    app: Application, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    schedule = AsyncMock(
        side_effect=RuntimeError("unavailable") if failure == "schedule" else None
    )
    metrics = AsyncMock(
        return_value={},
        side_effect=RuntimeError("unavailable") if failure == "metrics" else None,
    )
    notify = MagicMock()
    with monkeypatch.context() as patch:
        patch.setattr(app, "schedule_maintenance", schedule)
        patch.setattr(app, "metrics", metrics)
        patch.setattr("app.application.notify_service", notify)
        patch.setattr(
            "app.application.asyncio.sleep",
            AsyncMock(side_effect=asyncio.CancelledError),
        )

        with pytest.raises(asyncio.CancelledError):
            await app.maintenance()

    if failure is None:
        notify.assert_called_once_with("WATCHDOG=1")
    else:
        notify.assert_not_called()
