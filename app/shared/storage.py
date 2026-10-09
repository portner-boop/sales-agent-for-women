import asyncio
import hashlib
import sqlite3
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, TypeVar

import aiosqlite

from app.shared.entities import ScopeContext, StaleStateError

T = TypeVar("T")
Transaction = aiosqlite.Connection


async def fetchone(
    connection: Transaction, sql: str, parameters: tuple[Any, ...] = ()
) -> aiosqlite.Row | None:
    async with connection.execute(sql, parameters) as cursor:
        return await cursor.fetchone()


async def fetchall(
    connection: Transaction, sql: str, parameters: tuple[Any, ...] = ()
) -> list[aiosqlite.Row]:
    async with connection.execute(sql, parameters) as cursor:
        return list(await cursor.fetchall())


async def require_scope(connection: Transaction, scope: ScopeContext) -> None:
    row = await fetchone(
        connection, "SELECT epoch FROM conversations WHERE id=?", (scope.scope_id,)
    )
    if row is None or row["epoch"] != scope.epoch:
        raise StaleStateError("Scope generation changed")


class Database:
    def __init__(
        self, path: Path, queue_size: int = 256, busy_timeout_ms: int = 5000
    ) -> None:
        self.path = path
        self.busy_timeout_ms = busy_timeout_ms
        self._queue: asyncio.Queue[
            tuple[Callable[[Transaction], Awaitable[Any]], asyncio.Future[Any]] | None
        ] = asyncio.Queue(queue_size)
        self._writer: asyncio.Task[None] | None = None
        self._connection: Transaction | None = None

    async def open(self, migrations: Path) -> None:
        version = tuple(map(int, sqlite3.sqlite_version.split(".")))
        if version < (3, 51, 3) and version not in {(3, 44, 6), (3, 50, 7)}:
            raise RuntimeError("SQLite runtime must include the WAL-reset fix")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = await self._connect()
        await self._connection.execute("PRAGMA journal_mode=WAL")
        await self._connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (name TEXT PRIMARY KEY, checksum TEXT NOT NULL)"
        )
        paths = await asyncio.to_thread(lambda: sorted(migrations.glob("*.sql")))
        for path in paths:
            source = path.read_text()
            checksum = hashlib.sha256(source.encode()).hexdigest()
            existing = await fetchone(
                self._connection,
                "SELECT checksum FROM schema_migrations WHERE name=?",
                (path.name,),
            )
            if existing and existing["checksum"] != checksum:
                raise RuntimeError(f"Applied migration changed: {path.name}")
            if not existing:
                quoted_name = path.name.replace("'", "''")
                await self._connection.executescript(
                    f"BEGIN IMMEDIATE;\n{source}\nINSERT INTO schema_migrations VALUES ('{quoted_name}', '{checksum}');\nCOMMIT;"
                )
        await self._connection.commit()
        self._writer = asyncio.create_task(self._write_loop())

    async def _connect(self) -> Transaction:
        connection = await aiosqlite.connect(self.path)
        connection.row_factory = aiosqlite.Row
        await connection.execute("PRAGMA foreign_keys=ON")
        await connection.execute("PRAGMA synchronous=FULL")
        await connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        return connection

    @asynccontextmanager
    async def read(self) -> AsyncIterator[Transaction]:
        connection = await self._connect()
        try:
            await connection.execute("BEGIN")
            yield connection
        finally:
            await connection.rollback()
            await connection.close()

    async def write(self, operation: Callable[[Transaction], Awaitable[T]]) -> T:
        if self._writer is None or self._writer.done():
            raise RuntimeError("Writer is unavailable")
        future: asyncio.Future[T] = asyncio.get_running_loop().create_future()
        await self._queue.put((operation, future))
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            future.add_done_callback(consume_exception)
            raise

    async def _write_loop(self) -> None:
        assert self._connection is not None
        while True:
            entry = await self._queue.get()
            try:
                if entry is None:
                    return
                operation, future = entry
                try:
                    await self._connection.execute("BEGIN IMMEDIATE")
                    result = await operation(self._connection)
                    await self._connection.commit()
                except Exception as error:
                    await self._connection.rollback()
                    if not future.done():
                        future.set_exception(error)
                else:
                    if not future.done():
                        future.set_result(result)
            finally:
                self._queue.task_done()

    async def backup(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        async with self.read() as source:
            target = await aiosqlite.connect(destination)
            try:
                await source.backup(target)
            finally:
                await target.close()

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    async def close(self) -> None:
        if self._writer is not None:
            await self._queue.put(None)
            await self._writer
            self._writer = None
        if self._connection is not None:
            await self._connection.close()
            self._connection = None


def consume_exception(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()
