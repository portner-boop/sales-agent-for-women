import asyncio
import errno
import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
from pathlib import Path
from types import TracebackType
from typing import IO, Any
from uuid import uuid4

from app.shared.storage import Database, fetchone

BACKUP_MAX_BYTES = 1024**3
BACKUP_MIN_FREE_BYTES = 256 * 1024**2


class ProcessLock:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.stream: IO[str] | None = None

    def __enter__(self) -> "ProcessLock":
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory.chmod(0o700)
        self.stream = (self.directory / "runtime.lock").open("a")
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close()
            raise RuntimeError("Another process owns this data directory") from None
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.stream is not None:
            fcntl.flock(self.stream, fcntl.LOCK_UN)
            self.stream.close()


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_backup(directory: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if (
            name not in {"app.db", "deletions.jsonl"}
            or digest(directory / name) != expected
        ):
            raise ValueError("Backup checksum mismatch")
    connection = sqlite3.connect(f"file:{directory / 'app.db'}?mode=ro", uri=True)
    try:
        if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise ValueError("Backup integrity check failed")
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("Backup foreign key check failed")
    finally:
        connection.close()
    return manifest


async def create_backup(database: Database, deletion_path: Path, now: float) -> Path:
    root = database.path.parent / "backups"
    root.mkdir(exist_ok=True, mode=0o700)
    async with database.read() as tx:
        pages = await fetchone(tx, "PRAGMA page_count")
        page_size = await fetchone(tx, "PRAGMA page_size")
    assert pages is not None and page_size is not None
    journal_size = (
        (await asyncio.to_thread(deletion_path.stat)).st_size
        if await asyncio.to_thread(deletion_path.exists)
        else 0
    )
    expected_size = pages[0] * page_size[0] + journal_size + 4096
    if expected_size > BACKUP_MAX_BYTES:
        raise OSError(errno.ENOSPC, "Backup exceeds storage budget")
    await asyncio.to_thread(prune_backups, root, BACKUP_MAX_BYTES - expected_size)
    if shutil.disk_usage(root).free < expected_size + BACKUP_MIN_FREE_BYTES:
        raise OSError(errno.ENOSPC, "Insufficient free space for backup")
    staging = root / f".pending-{uuid4().hex}"
    staging.mkdir(mode=0o700)
    try:
        await database.backup(staging / "app.db")
        if await asyncio.to_thread(deletion_path.exists):
            shutil.copyfile(deletion_path, staging / "deletions.jsonl")
        else:
            (staging / "deletions.jsonl").write_text("")
        manifest = {
            "created_at": now,
            "files": {
                name: await asyncio.to_thread(digest, staging / name)
                for name in ("app.db", "deletions.jsonl")
            },
            "checkpoint_policy": "rebuild_from_app_db",
        }
        (staging / "manifest.json").write_text(json.dumps(manifest))
        if backup_size(staging) > BACKUP_MAX_BYTES:
            raise OSError(errno.ENOSPC, "Backup exceeds storage budget")
        await asyncio.to_thread(verify_backup, staging)
        target = root / f"snapshot-{int(now)}-{uuid4().hex[:8]}"
        staging.rename(target)
        await asyncio.to_thread(prune_backups, root, BACKUP_MAX_BYTES)
        return target
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def backup_size(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.iterdir() if path.is_file())


def prune_backups(root: Path, max_bytes: int) -> None:
    snapshots = sorted(
        (path for path in root.glob("snapshot-*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime_ns,
    )
    sizes = {path: backup_size(path) for path in snapshots}
    total = sum(sizes.values())
    for path in snapshots[:-1]:
        if total <= max_bytes:
            break
        shutil.rmtree(path)
        total -= sizes[path]


def restore_backup(backup: Path, directory: Path) -> None:
    verify_backup(backup)
    journal = directory / "deletions.jsonl"
    latest: dict[str, dict[str, Any]] = {}
    for path in (journal, backup / "deletions.jsonl"):
        if path.exists():
            for line in path.read_text().splitlines():
                record = json.loads(line)
                previous = latest.get(record["scope_id"])
                if previous is None or previous["epoch"] < record["epoch"]:
                    latest[record["scope_id"]] = record
    with journal.open("w") as stream:
        for record in latest.values():
            stream.write(json.dumps(record) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    staging = directory / "restore.pending.db"
    shutil.copyfile(backup / "app.db", staging)
    restored = sqlite3.connect(staging)
    try:
        restored.execute(
            "UPDATE outbox_messages SET status='delivery_unknown',error_code='RESTORED_SNAPSHOT' WHERE status IN ('ready','sending')"
        )
        restored.execute(
            "UPDATE scheduled_jobs SET status='cancelled',lease_token=NULL,lease_until=NULL WHERE status IN ('pending','running')"
        )
        restored.execute("UPDATE turns SET status='cancelled' WHERE status='running'")
        if restored.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='followups'"
        ).fetchone():
            restored.execute(
                "UPDATE turns SET status='cancelled' WHERE id IN (SELECT turn_id FROM followups WHERE status IN ('pending','generating'))"
            )
            restored.execute(
                "UPDATE followups SET status='cancelled',reason='RESTORED_SNAPSHOT' WHERE status IN ('pending','generating')"
            )
        restored.execute("UPDATE messages SET pending=0")
        restored.execute(
            "UPDATE conversations SET control_state=CASE WHEN purchase_state='confirmed' THEN 'handoff_only' ELSE 'paused' END,control_revision=control_revision+1"
        )
        restored.commit()
    finally:
        restored.close()
    for name in (
        "app.db-wal",
        "app.db-shm",
        "checkpoints.db",
        "checkpoints.db-wal",
        "checkpoints.db-shm",
    ):
        (directory / name).unlink(missing_ok=True)
    staging.replace(directory / "app.db")
    for path in directory.glob("*.db"):
        path.chmod(0o600)


def expire_backups(directory: Path, cutoff: float) -> None:
    for path in (directory / "backups").glob("snapshot-*"):
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["created_at"] < cutoff:
            shutil.rmtree(path)
