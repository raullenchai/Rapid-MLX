# SPDX-License-Identifier: Apache-2.0
"""Killable document parser; no model imports or parser threads."""

import asyncio
import json
import math
import sys
from pathlib import Path

from .extraction import (
    MAX_BYTES,
    OFFICE_SUFFIXES,
    SKIP_REASONS,
    SkippedError,
    Unit,
    extract,
)

PARSER_TIMEOUT_SECONDS = 10.0
PARSER_MEMORY_BYTES = 512 * 1024 * 1024


def parser_timeout_seconds(size_bytes: int) -> float:
    return min(120.0, PARSER_TIMEOUT_SECONDS + 4 * size_bytes / (1024 * 1024))


async def parse_document(path, data):
    if path.suffix.lower() not in {".pdf", *OFFICE_SUFFIXES}:
        return await asyncio.to_thread(extract, path, data)
    return await run_parser(
        [sys.executable, "-m", "rapid_mlx.local_search.parsing", path.suffix.lower()],
        data,
    )


async def run_parser(command, data):
    try:
        return await asyncio.wait_for(
            _run_parser(command, data), timeout=parser_timeout_seconds(len(data))
        )
    except asyncio.TimeoutError as exc:
        raise SkippedError(SKIP_REASONS["read"]) from exc


async def _run_parser(command, data):
    """Deadline includes input/output; always kill and reap on cancel/failure.

    RSS monitoring enforces the bound on Darwin, where RLIMIT_AS alone does
    not reliably cap resident memory. Children also set allocation limits.
    """
    import psutil

    spawn = asyncio.create_task(
        asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    )
    try:
        process = await asyncio.shield(spawn)
    except asyncio.CancelledError:
        process = await spawn
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise

    async def communicate_bounded():
        async def send():
            try:
                process.stdin.write(data)
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except asyncio.CancelledError:
                process.stdin.transport.abort()
                raise
            finally:
                process.stdin.close()
                try:
                    await process.stdin.wait_closed()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        sender = asyncio.create_task(send())
        output = bytearray()
        try:
            while True:
                block = await process.stdout.read(
                    min(65_536, MAX_BYTES * 8 + 1 - len(output))
                )
                if not block:
                    break
                output.extend(block)
                if len(output) > MAX_BYTES * 8:
                    raise SkippedError(SKIP_REASONS["read"])
            await sender
            await process.wait()
            return output
        finally:
            if not sender.done():
                sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)

    communication = asyncio.create_task(communicate_bounded())

    async def bounded():
        probe = None
        while not communication.done():
            try:
                # Short-lived children can exit before the RSS probe is created.
                if probe is None:
                    probe = psutil.Process(process.pid)
                if probe.memory_info().rss > PARSER_MEMORY_BYTES:
                    raise SkippedError(SKIP_REASONS["read"])
            except psutil.NoSuchProcess:
                break
            await asyncio.sleep(0.02)
        output = await communication
        if process.returncode or len(output) > MAX_BYTES * 8:
            raise SkippedError(SKIP_REASONS["read"])
        message = json.loads(output)
        if "error" in message:
            raise SkippedError(message["error"])
        return [Unit(**unit) for unit in message["units"]]

    try:
        return await bounded()
    except SkippedError:
        raise
    except (asyncio.TimeoutError, ValueError, OSError) as exc:
        raise SkippedError(SKIP_REASONS["read"]) from exc
    finally:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await process.wait()
        if not communication.done():
            communication.cancel()
        await asyncio.gather(communication, return_exceptions=True)


def main():
    import resource
    from dataclasses import asdict

    if sys.platform != "darwin":
        resource.setrlimit(
            resource.RLIMIT_AS, (PARSER_MEMORY_BYTES, PARSER_MEMORY_BYTES)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (PARSER_MEMORY_BYTES, PARSER_MEMORY_BYTES)
        )
    try:
        data = sys.stdin.buffer.read(MAX_BYTES + 1)
        # RLIMIT_CPU accepts whole seconds; round the shared deadline up.
        cpu_seconds = math.ceil(parser_timeout_seconds(len(data)))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        if len(data) > MAX_BYTES:
            raise SkippedError(SKIP_REASONS["size"])
        units = extract(Path("source" + sys.argv[1]), data)
        message = {"units": [asdict(unit) for unit in units]}
    except SkippedError as exc:
        message = {"error": str(exc)}
    except Exception:
        message = {"error": SKIP_REASONS["read"]}
    sys.stdout.write(json.dumps(message))


if __name__ == "__main__":
    main()
