import asyncio
import errno
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.application import Application
from app.shared import operations
from app.shared.operations import create_backup, verify_backup
from app.shared.storage import Database
from tests.helpers import scalar, update


async def test_hourly_maintenance_does_not_repeat_after_completion_or_restart(
    app: Application,
) -> None:
    root = app.settings.storage.directory / "backups"
    await app.schedule_maintenance()
    await app.drain()
    assert len(list(root.glob("snapshot-*"))) == 1
    for _ in range(5):
        await app.schedule_maintenance()
        await app.drain()
    assert await scalar(app, "SELECT COUNT(*) FROM scheduled_jobs") == 2
    assert len(list(root.glob("snapshot-*"))) == 1
    await app.close()
    restarted = await Application.open(app.settings, app.sender, app.model, app.clock)
    try:
        await restarted.schedule_maintenance()
        await restarted.drain()
        assert len(list(root.glob("snapshot-*"))) == 1
        app.clock.advance(3600)
        await restarted.schedule_maintenance()
        await restarted.drain()
        assert len(list(root.glob("snapshot-*"))) == 2
        assert await scalar(restarted, "SELECT COUNT(*) FROM scheduled_jobs") == 4
    finally:
        await restarted.close()


async def test_failed_backup_is_bounded_and_next_interval_can_retry(
    app: Application, monkeypatch: pytest.MonkeyPatch
) -> None:
    failing = AsyncMock(side_effect=OSError(errno.ENOSPC, "disk full"))
    monkeypatch.setattr(app.memory, "backup", failing)
    for _ in range(10):
        await app.schedule_maintenance()
        await app.drain()
        app.clock.advance(10)
    assert failing.await_count == 3
    assert (
        await scalar(app, "SELECT COUNT(*) FROM scheduled_jobs WHERE kind='backup'")
        == 1
    )
    assert (
        await scalar(app, "SELECT status FROM scheduled_jobs WHERE kind='backup'")
        == "dead_letter"
    )
    app.clock.advance(3600)
    await app.schedule_maintenance()
    await app.drain()
    assert failing.await_count == 4


@pytest.mark.parametrize("cancelled", [False, True])
async def test_failed_or_cancelled_backup_removes_partial_directory(
    app: Application, monkeypatch: pytest.MonkeyPatch, cancelled: bool
) -> None:
    original = await app.memory.backup()

    async def fail(destination: Path) -> None:
        await asyncio.to_thread(destination.write_bytes, b"partial")
        if cancelled:
            raise asyncio.CancelledError()
        raise OSError(errno.ENOSPC, "disk full")

    monkeypatch.setattr(app.database, "backup", fail)
    with pytest.raises(asyncio.CancelledError if cancelled else OSError):
        await app.memory.backup()
    assert not list(original.parent.glob(".pending-*"))
    verify_backup(original)


async def test_backup_rotation_respects_budget_and_keeps_latest_valid_copy(
    app: Application, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = await app.memory.backup()
    size = operations.backup_size(first)
    monkeypatch.setattr(operations, "BACKUP_MAX_BYTES", size * 2 + 8192)
    for _ in range(8):
        app.clock.advance(3600)
        latest = await app.memory.backup()
    copies = list(first.parent.glob("snapshot-*"))
    assert latest in copies
    assert not first.exists()
    assert (
        sum(operations.backup_size(path) for path in copies)
        <= operations.BACKUP_MAX_BYTES
    )
    verify_backup(latest)
    assert not list(first.parent.glob(".pending-*"))


async def test_low_disk_space_does_not_remove_last_backup_or_stop_chat(
    app: Application, monkeypatch: pytest.MonkeyPatch
) -> None:
    last = await app.memory.backup()
    monkeypatch.setattr(
        operations.shutil, "disk_usage", lambda _: SimpleNamespace(free=1)
    )
    with pytest.raises(OSError) as error:
        await app.memory.backup()
    assert error.value.errno == errno.ENOSPC
    verify_backup(last)
    assert list(last.parent.iterdir()) == [last]
    await app.telegram.ingest([update(1, "hello")])
    await app.drain()
    assert len(app.sender.sent) == 1


async def test_oversize_backup_is_rejected_before_copy(
    app: Application, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(operations, "BACKUP_MAX_BYTES", 1)
    backup = AsyncMock()
    monkeypatch.setattr(app.database, "backup", backup)
    with pytest.raises(OSError) as error:
        await create_backup(app.database, app.memory.deletion_path, app.clock.now())
    assert error.value.errno == errno.ENOSPC
    backup.assert_not_awaited()


async def test_connection_is_closed_when_initialization_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connection = SimpleNamespace(
        execute=AsyncMock(side_effect=sqlite3.OperationalError("disk full")),
        close=AsyncMock(),
    )
    monkeypatch.setattr(
        "app.shared.storage.aiosqlite.connect", AsyncMock(return_value=connection)
    )
    with pytest.raises(sqlite3.OperationalError):
        await Database(tmp_path / "app.db")._connect()
    connection.close.assert_awaited_once()
