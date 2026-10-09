import asyncio
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

from app.shared.storage import Database


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
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    staging = root / f".pending-{uuid4().hex}"
    staging.mkdir(mode=0o700)
    await database.backup(staging / "app.db")
    if await asyncio.to_thread(deletion_path.exists):
        shutil.copyfile(deletion_path, staging / "deletions.jsonl")
    else:
        (staging / "deletions.jsonl").write_text("")
    manifest = {
        "created_at": now,
        "files": {
            name: digest(staging / name) for name in ("app.db", "deletions.jsonl")
        },
        "checkpoint_policy": "rebuild_from_app_db",
    }
    (staging / "manifest.json").write_text(json.dumps(manifest))
    verify_backup(staging)
    target = root / f"snapshot-{int(now)}-{uuid4().hex[:8]}"
    staging.rename(target)
    return target


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
