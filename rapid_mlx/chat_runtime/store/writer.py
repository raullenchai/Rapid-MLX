"""Single SQLite owner thread; acknowledgements follow durable group commit."""

from __future__ import annotations

import asyncio
import concurrent.futures
import queue
import sqlite3
import threading
import time
from collections.abc import Callable


class Writer:
    def __init__(self, connect: Callable[[], sqlite3.Connection]):
        self.queue: queue.Queue = queue.Queue(maxsize=4096)
        self.ready: concurrent.futures.Future = concurrent.futures.Future()
        self.closed = False
        self.thread = threading.Thread(
            target=self._run, args=(connect,), name="chat-store", daemon=True
        )
        self.thread.start()

    async def start(self):
        await asyncio.wrap_future(self.ready)

    async def call(self, operation: Callable, *, immediate=True):
        if self.closed:
            raise RuntimeError("store is closed")
        future: concurrent.futures.Future = concurrent.futures.Future()
        try:
            self.queue.put_nowait((operation, future, immediate))
        except queue.Full:
            raise RuntimeError("chat store is busy; retry") from None
        # Cancelling a waiter cannot undo an admitted durable operation.
        return await asyncio.shield(asyncio.wrap_future(future))

    async def close(self):
        if not self.closed:
            await self.call(lambda db: None)
            self.closed = True
            await asyncio.to_thread(self.queue.put, None)
            await asyncio.to_thread(self.thread.join)

    def _run(self, connect):
        try:
            db = connect()
        except BaseException as exc:
            self.ready.set_exception(exc)
            return
        self.ready.set_result(None)
        try:
            while True:
                first = self.queue.get()
                if first is None:
                    break
                batch = [first]
                deadline = time.monotonic() + 0.002
                while not batch[-1][2] and len(batch) < 128:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        item = self.queue.get(timeout=remaining)
                    except queue.Empty:
                        break
                    if item is None:
                        raise RuntimeError("writer closed while batching")
                    batch.append(item)
                results = []
                try:
                    db.execute("BEGIN IMMEDIATE")
                    for operation, future, _ in batch:
                        db.execute("SAVEPOINT job")
                        try:
                            value = operation(db)
                        except Exception as exc:
                            db.execute("ROLLBACK TO job")
                            results.append((future, None, exc))
                        else:
                            results.append((future, value, None))
                        db.execute("RELEASE job")
                    db.execute("COMMIT")
                except BaseException as exc:
                    if db.in_transaction:
                        db.execute("ROLLBACK")
                    results = [(future, None, exc) for _, future, _ in batch]
                for future, value, error in results:
                    if error is None:
                        future.set_result(value)
                    else:
                        future.set_exception(error)
        finally:
            db.close()
