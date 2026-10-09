import asyncio
import hashlib
import json
import os
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

ControlHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def control_path(directory: Path) -> Path:
    root = Path(tempfile.gettempdir()) / f"behappy-control-{os.getuid()}"
    root.mkdir(exist_ok=True, mode=0o700)
    identifier = hashlib.sha256(str(directory.resolve()).encode()).hexdigest()[:24]
    return root / f"{identifier}.sock"


async def start_control(directory: Path, handler: ControlHandler) -> asyncio.Server:
    path = control_path(directory)
    path.unlink(missing_ok=True)

    async def connection(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            async with asyncio.timeout(15):
                request = json.loads(await reader.readline())
                if not isinstance(request, dict):
                    raise ValueError("Invalid control request")
                response = await handler(request)
                writer.write(json.dumps({"result": response}).encode() + b"\n")
                await writer.drain()
        except Exception as error:
            writer.write(json.dumps({"error": type(error).__name__}).encode() + b"\n")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_unix_server(connection, str(path), limit=4096)
    path.chmod(0o600)
    return server


async def request_control(
    directory: Path, request: dict[str, Any]
) -> dict[str, Any] | None:
    try:
        reader, writer = await asyncio.open_unix_connection(
            str(control_path(directory))
        )
    except (FileNotFoundError, ConnectionRefusedError):
        return None
    try:
        async with asyncio.timeout(15):
            writer.write(json.dumps(request).encode() + b"\n")
            await writer.drain()
            response = json.loads(await reader.readline())
            if "error" in response:
                raise RuntimeError(f"Control request failed: {response['error']}")
            result: dict[str, Any] = response["result"]
            return result
    finally:
        writer.close()
        await writer.wait_closed()
