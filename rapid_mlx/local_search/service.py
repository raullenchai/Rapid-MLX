# SPDX-License-Identifier: Apache-2.0
"""Async coordinator; CPU/SQLite and MLX each use their owning workers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np

from . import MODEL_ID
from .chunking import Chunk, chunk_units
from .eligibility import semantic_chunk
from .extraction import (
    AUDIO_SUFFIXES,
    LEGACY_OFFICE_SUFFIXES,
    MAX_BYTES,
    MAX_EMBEDDED_CHUNKS_PER_FILE,
    OFFICE_SUFFIXES,
    SKIP_REASONS,
    TEXT_SUFFIXES,
    VIDEO_MAX_BYTES,
    SkippedError,
    excluded,
    file_kind,
    media_frames,
    next_media_batch,
    open_root,
    open_safe,
    read_safe,
    scan_batches,
    skip_reason,
    stat_safe,
    validate_folder,
)
from .parsing import parse_document
from .search import (
    KINDS,
    fuse,
    grouped_rankings,
    query_statistics,
    query_terms,
    semantic_ranking,
    snippet,
    top_files,
)
from .store import StoreWorker, bounded_terms

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Model loading failed or no sanctioned worker is available."""

    def __init__(self, message, *, load_failed=True):
        super().__init__(message)
        self.load_failed = load_failed


class EmbeddingInputError(RuntimeError):
    """Embedding or validation failed for the current inputs."""


class Embedder(Protocol):
    @property
    def is_loaded(self) -> bool: ...

    def embed(self, texts: str | list[str]) -> list[list[float]]: ...

    def embed_images(self, images: list) -> list[list[float]]: ...


def local_model_path(model_id: str) -> str | None:
    """Same fail-closed runnability probe as model_downloads.require_local.

    require_local itself is conditional on the global download flag, so it
    cannot alone enforce this feature's unconditional no-download policy.
    Resolve a runnable snapshot locally and hand the loader a filesystem path.
    """
    from ..cli import _cache_runnability
    from ..model_downloads import ModelDownloadsDisabledError, snapshot_download
    from ..model_metadata import (
        _cached_file,
        _complete_cached_checkpoint,
        resolve_offline_cached_snapshot,
        resolve_unreferenced_cached_snapshot,
    )

    try:
        if _cache_runnability(model_id) is not True:
            return None
        resolved = resolve_unreferenced_cached_snapshot(
            model_id
        ) or resolve_offline_cached_snapshot(model_id)
        if resolved is not None:
            return _runnable_search_snapshot(resolved)
        config = _cached_file(model_id, "config.json")
        if config is not None:
            from huggingface_hub.constants import HF_HUB_CACHE

            repo = Path(HF_HUB_CACHE) / ("models--" + model_id.replace("/", "--"))
            resolved = _complete_cached_checkpoint(repo, config.parent, model_id)
            return _runnable_search_snapshot(resolved) if resolved is not None else None
        return _runnable_search_snapshot(
            Path(snapshot_download(model_id, local_files_only=True))
        )
    except (OSError, ValueError, ModelDownloadsDisabledError):
        return None


def _runnable_search_snapshot(path: Path) -> str | None:
    """Check loader assets without importing a tokenizer or loading the model.

    Vision checkpoints are all-or-nothing: the factory enables vision, so a
    text-capable snapshot missing its image processor is still incomplete.
    Optional repository files do not affect local availability.
    """
    config = json.loads((path / "config.json").read_text())
    required = ["tokenizer_config.json", "tokenizer.json"]
    if config.get("vision_config"):
        required.append("processor_config.json")
    if any(
        not (path / name).is_file() or (path / name).stat().st_size == 0
        for name in required
    ):
        return None
    return str(path)


def make_embedder(path: str) -> Embedder:
    from ..embedding import EmbeddingEngine

    return EmbeddingEngine(
        path, max_length="auto", overflow_policy="error", vision=True
    )


def _checkpoint_identity(path: str | None) -> tuple | None:
    """Cheap metadata probe, including in-place updates and HF blob targets."""
    if path is None:
        return None
    root = Path(path)
    try:
        entries = [root, *sorted(root.iterdir())]
        metadata = []
        for entry in entries:
            info = entry.stat()
            metadata.append(
                (entry.name, info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
            )
        return path, tuple(metadata)
    except OSError:
        # A disappearing/unreadable checkpoint still has a stable identity.
        return path, ()


class LocalSearch:
    def __init__(
        self,
        directory: Path,
        *,
        model_id=MODEL_ID,
        resolve_model=local_model_path,
        embedder_factory: Callable[[str], Embedder] = make_embedder,
        model_worker=None,
        batch_size=4,
        yield_seconds=0.01,
    ):
        self.model_id = model_id
        self.store = StoreWorker(directory / "index.sqlite3", model_id)
        self.resolve_model = resolve_model
        self.embedder_factory = embedder_factory
        self.engine: Embedder | None = None
        self.model_worker = model_worker
        self._model_gate = asyncio.Lock()
        self._coordination = asyncio.Lock()
        self._owner_worker = None
        self.worker_error = None
        self._attempted_checkpoints: dict[str, tuple | None] = {}
        self.batch_size = max(1, min(4, batch_size))
        self.yield_seconds = yield_seconds
        self.pending: list[str] = []
        self.task: asyncio.Task | None = None
        self.progress = self._idle()
        self.closed = False

    @staticmethod
    def _idle():
        return {
            "active": False,
            "folder_id": None,
            "current_path": None,
            "files_done": 0,
            "files_total": 0,
        }

    def start(self):
        self.task = asyncio.create_task(self._resume())

    async def _resume(self):
        for folder in await self.store.call("folders"):
            await self.queue(folder["id"])
        await self._run()

    async def folder(self, folder_id: str) -> dict | None:
        return await self.store.call("folder", folder_id)

    async def folders(self) -> list[dict]:
        return await self.store.call("folders")

    async def queue(self, folder_id: str):
        if self.closed or not await self.store.call("folder", folder_id):
            return
        if folder_id not in self.pending:
            self.pending.append(folder_id)
        if self.progress["folder_id"] != folder_id:
            await self.store.call("set_folder", folder_id, "queued")
        self._start_worker()

    def _start_worker(self):
        if self.pending and (self.task is None or self.task.done()):
            self.task = asyncio.create_task(self._run())

    def status(self) -> dict:
        return self._status(
            self.resolve_model(self.model_id), self.store.folders(), self.store.totals()
        )

    async def status_async(self) -> dict:
        model_path = await asyncio.to_thread(self.resolve_model, self.model_id)
        checkpoint = await asyncio.to_thread(_checkpoint_identity, model_path)
        if model_path is None:
            self._attempted_checkpoints.clear()
        worker = await self._refresh_worker()
        if not callable(worker):
            self._attempted_checkpoints.clear()
        folders = await self.store.call("folders")
        totals = await self.store.call("totals")
        vision = await asyncio.to_thread(self._vision_available, model_path)
        for folder in folders:
            if not await asyncio.to_thread(Path(folder["path"]).is_dir):
                folder["state"] = "missing"
            elif (
                model_path is not None
                and callable(worker)
                and folder["state"] == "waiting_for_model"
                and self._attempted_checkpoints.get(folder["id"])
                != (checkpoint, worker)
                and self.progress["folder_id"] != folder["id"]
            ):
                await self.queue(folder["id"])
                folder["state"] = "queued"
        return self._status(model_path, folders, totals, vision)

    def _status(self, model_path, folders, totals, vision=None) -> dict:
        downloaded = model_path is not None
        return {
            "enabled": True,
            "model": {
                "id": self.model_id,
                "downloaded": downloaded,
                "loaded": bool(self.engine and self.engine.is_loaded),
                "vision": self._vision_available(model_path)
                if vision is None
                else vision,
            },
            "folders": folders,
            "indexing": dict(self.progress),
            "totals": totals,
            "semantic_unavailable_reason": self.worker_error
            or (
                None
                if self._worker() is not None
                else "Engine has no sanctioned model worker; keyword-only"
            ),
        }

    def _vision_available(self, model_path):
        if self.engine is not None and self.engine.is_loaded:
            return bool(getattr(self.engine, "vision", False))
        if model_path is None:
            return False
        try:
            config = json.loads((Path(model_path) / "config.json").read_text())
            return config.get("model_type") == "embedding_gemma2" and bool(
                config.get("vision_config")
            )
        except (OSError, ValueError):
            return False

    async def add_folder(self, value: str) -> tuple[dict, bool]:
        path = await asyncio.to_thread(validate_folder, value)
        async with self._coordination:
            # Drain the current write before absorption. Cancellation cannot
            # race a commit against a root being removed on the DB worker.
            active_id = self.progress["folder_id"]
            if active_id:
                # Re-queue before draining: the worker cannot claim another
                # root while this coordination lock is held.
                if active_id not in self.pending:
                    self.pending.append(active_id)
                await self._cancel_worker()
            folder_id, created = await self.store.call("add_folder", path)
            live = {f["id"] for f in await self.store.call("folders")}
            self.pending = [i for i in self.pending if i in live]
            await self.queue(folder_id)
            return await self.store.call("folder", folder_id), created

    async def delete_folder(self, folder_id: str):
        async with self._coordination:
            if self.progress["folder_id"] == folder_id:
                await self._cancel_worker()
            self.pending = [i for i in self.pending if i != folder_id]
            await self.store.call("delete_folder", folder_id)
            self._attempted_checkpoints.pop(folder_id, None)
            self._start_worker()

    async def _cancel_worker(self):
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        self.task = None

    async def close(self):
        self.closed = True
        await self._cancel_worker()
        try:
            if self.engine is not None:
                await self._model_call(self._release_embedder, _teardown=True)
        finally:
            await asyncio.to_thread(self.store.close)

    def _worker(self):
        if self.model_worker is not None:
            return self.model_worker
        from ..config import get_config

        return getattr(get_config().engine, "execute_on_model_worker", None)

    async def _refresh_worker(self):
        """Release a resident embedder on its owner before adopting a new worker."""
        worker = self._worker()
        if self.engine is None or self._owner_worker == worker:
            return worker
        async with self._model_gate:
            worker = self._worker()
            if self.engine is None or self._owner_worker == worker:
                return worker
            if not callable(self._owner_worker):
                self.worker_error = "Model worker changed before teardown"
                return None
            try:
                await self._model_call_owned(self._release_embedder, _teardown=True)
            except ModelLoadError as exc:
                self.worker_error = str(exc)
                return None
        self.worker_error = None
        return worker

    async def _model_call(self, func, *args, _teardown=False, **kwargs):
        # FIFO admission before submitting to the engine: the sole indexer can
        # have only its current <=4-input batch ahead of a waiting query.
        async with self._model_gate:
            return await self._model_call_owned(
                func, *args, _teardown=_teardown, **kwargs
            )

    async def _model_call_owned(self, func, *args, _teardown=False, **kwargs):
        worker = self._owner_worker if _teardown else self._worker()
        if not callable(worker):
            message = (
                self.worker_error
                or "Engine has no sanctioned model worker; keyword-only"
            )
            if self.engine is not None and callable(self._owner_worker):
                await self._model_call_owned(self._release_embedder, _teardown=True)
            self.worker_error = message
            raise ModelLoadError(message, load_failed=False)
        # Retain the exact boundary owning this embedder even if cfg.engine changes.
        self._owner_worker = worker

        def work():
            try:
                return "ok", func(*args, **kwargs)
            except EmbeddingInputError as exc:
                return "input", str(exc)
            except Exception as exc:
                # Never transfer a traceback retaining MLX/model locals to the loop.
                self._release_embedder()
                return "model", str(exc)

        try:
            task = asyncio.create_task(worker(work))
            kind, result = await asyncio.shield(task)
            if kind == "input":
                self.worker_error = None
                raise EmbeddingInputError(result)
            if kind == "model":
                self.worker_error = result
                raise ModelLoadError(result)
            self.worker_error = None
            return result
        except asyncio.CancelledError:
            # A running GPU call cannot be interrupted; drain before teardown.
            await task
            raise
        except (EmbeddingInputError, ModelLoadError):
            raise
        except Exception as exc:
            self.worker_error = str(exc)
            raise ModelLoadError(str(exc), load_failed=False) from None

    def _release_embedder(self):
        self.engine = None
        self._owner_worker = None

    async def _release_after_model_error(self):
        if self.engine is None:
            return
        try:
            await self._model_call(self._release_embedder, _teardown=True)
        except ModelLoadError:
            # Keep the owner boundary for a later status-driven recovery attempt.
            pass

    def _generation_busy(self):
        worker = self._worker()
        owner = getattr(worker, "__self__", None)
        if self.model_worker is None:
            from ..config import get_config

            owner = get_config().engine
        status = getattr(owner, "lifecycle_status", None)
        if callable(status):
            # Includes route admission, before requests reach the scheduler.
            return status()["active_requests"] > 0
        stats = getattr(owner, "get_stats", None)
        if callable(stats):
            snapshot = stats()
            return bool(
                snapshot.get("num_running", 0) or snapshot.get("num_waiting", 0)
            )
        return False

    async def _wait_for_generation(self):
        delay = 0.01
        try:
            while self._generation_busy():
                self.progress["paused_reason"] = "chat"
                await asyncio.sleep(delay)
                delay = min(0.1, delay * 2)
        finally:
            self.progress.pop("paused_reason", None)

    async def _embed_async(self, inputs, *, background=False, **kwargs):
        await self._refresh_worker()
        if not background:
            return await self._model_call(self._embed, inputs, **kwargs)
        while True:
            # Do not hold the query's admission lock while waiting for chat.
            await self._wait_for_generation()
            async with self._model_gate:
                # Chat may have arrived while an interactive query held the lock.
                if self._generation_busy():
                    continue
                return await self._model_call_owned(self._embed, inputs, **kwargs)

    async def wait_idle(self):
        if self.task:
            await self.task

    def _embed(self, inputs: list, *, images=False):
        if self.engine is None:
            try:
                path = self.resolve_model(self.model_id)
                if path is None:
                    return None
                self.engine = self.embedder_factory(path)
            except Exception as exc:
                raise ModelLoadError(str(exc)) from None
        engine = self.engine
        method = engine.embed_images if images else engine.embed
        try:
            vectors = np.asarray(method(inputs), dtype=np.float32)
        except EmbeddingInputError:
            raise
        except Exception as exc:
            from ..embedding import (
                EmbeddingInputTooLongError,
                EmbeddingUnsupportedMediaError,
            )

            if (
                isinstance(
                    exc, (EmbeddingInputTooLongError, EmbeddingUnsupportedMediaError)
                )
                or str(exc) == "Tokenizer failed to expand image placeholders"
            ):
                raise EmbeddingInputError(str(exc)) from None
            raise
        if (
            vectors.ndim != 2
            or vectors.shape[0] != len(inputs)
            or not np.isfinite(vectors).all()
        ):
            raise EmbeddingInputError("invalid embedding output")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise EmbeddingInputError("zero embedding output")
        return vectors / norms

    async def _index_media(self, path, root, available, root_fd):
        frames = media_frames(path, root, root_fd)
        chunks, vectors = [], []
        error = None
        input_failed = False
        try:
            while True:
                # Finish the CPU read before closing its pinned descriptor on cancel.
                read = asyncio.create_task(
                    asyncio.to_thread(next_media_batch, frames, 4)
                )
                try:
                    batch = await asyncio.shield(read)
                except asyncio.CancelledError:
                    try:
                        await read
                    except Exception:
                        pass
                    raise
                if not batch:
                    break
                chunks.extend(
                    Chunk("", None, None, file_kind(path), frame.timestamp)
                    for frame in batch
                )
                if available and not input_failed:
                    try:
                        embedded = await self._embed_async(
                            [frame.image for frame in batch],
                            images=True,
                            background=True,
                        )
                        if embedded is None:
                            available = False
                        else:
                            vectors.extend(embedded)
                    except ModelLoadError as exc:
                        logger.warning("Local-search image model unavailable: %s", exc)
                        await self._release_after_model_error()
                        available = False
                        error = exc
                    except Exception as exc:
                        logger.debug("Local-search image embedding skipped: %s", exc)
                        input_failed = True
                del batch
                await asyncio.sleep(self.yield_seconds)
        finally:
            await asyncio.to_thread(frames.close)
        return (
            chunks,
            vectors if available and not input_failed else None,
            available,
            error,
        )

    async def _run(self):
        try:
            while self.pending:
                # Claim the root atomically with registration/absorption. A DB
                # await must not hide a dequeued worker from add_folder().
                async with self._coordination:
                    if not self.pending:
                        break
                    folder_id = self.pending.pop(0)
                    folder = await self.store.call("folder", folder_id)
                    if folder is None:
                        continue
                    self.progress = {
                        **self._idle(),
                        "active": True,
                        "folder_id": folder_id,
                    }
                    await self.store.call("set_folder", folder_id, "indexing")
                try:
                    await self._index(folder)
                except asyncio.CancelledError:
                    if (
                        not self.closed
                        and await self.store.call("folder", folder_id) is None
                    ):
                        # An absorbed root resumes through the ordinary queue;
                        # committed files now belong to the containing root.
                        continue
                    await self.store.call("set_folder", folder_id, "queued")
                    raise
                except (FileNotFoundError, NotADirectoryError):
                    await self.store.call("set_folder", folder_id, "missing")
                except Exception as exc:
                    logger.exception("Local-search indexing failed")
                    await self.store.call(
                        "set_folder", folder_id, "error", error=str(exc)
                    )
                finally:
                    self.progress = self._idle()
        finally:
            self.progress = self._idle()

    async def _index(self, folder: dict):
        root = Path(folder["path"])
        pinned = open_root(root)
        opening = asyncio.create_task(asyncio.to_thread(pinned.__enter__))
        try:
            root_fd = await asyncio.shield(opening)
        except asyncio.CancelledError:
            await opening
            await asyncio.to_thread(pinned.__exit__, None, None, None)
            raise
        try:
            await self._index_pinned(folder, root, root_fd)
        finally:
            await asyncio.to_thread(pinned.__exit__, None, None, None)

    async def _index_pinned(self, folder, root, root_fd):
        batches = scan_batches(root, root_fd)
        try:
            old = await self.store.call("files", folder["id"])
            existing = set()
            skipped = 0
            index_directory = await asyncio.to_thread(self.store.path.parent.resolve)
            # Only the index's own files are skipped; user files beside it are indexed.
            index_names = {
                self.store.path.name + suffix
                for suffix in ("", "-wal", "-shm", "-journal")
            }

            async def discovered_paths():
                nonlocal skipped
                while True:
                    # next(..., None) avoids propagating StopIteration through a Future.
                    read = asyncio.create_task(asyncio.to_thread(next, batches, None))
                    try:
                        batch = await asyncio.shield(read)
                    except asyncio.CancelledError:
                        await read
                        raise
                    if batch is None:
                        break
                    paths, count = batch
                    skipped += count
                    await self.store.call(
                        "set_folder", folder["id"], "indexing", skipped=skipped
                    )
                    for path in paths:
                        if path.parent == index_directory and path.name in index_names:
                            continue
                        existing.add(str(path))
                        yield path
                    await asyncio.sleep(self.yield_seconds)

            # Zero means discovery is ongoing; committed files are already searchable.
            self.progress["files_total"] = 0
            model_path = await asyncio.to_thread(self.resolve_model, self.model_id)
            checkpoint = await asyncio.to_thread(_checkpoint_identity, model_path)
            worker = await self._refresh_worker()
            available = model_path is not None
            model_error = None
            load_failed = False
            if not callable(worker):
                available = False
                model_error = "Engine has no sanctioned model worker; keyword-only"
                self.worker_error = model_error
            async for path in discovered_paths():
                self.progress["current_path"] = str(path)
                previous = old.get(str(path))
                try:
                    info = await asyncio.to_thread(path.lstat)
                    if excluded(path):
                        raise SkippedError(SKIP_REASONS["private"])
                    if (
                        previous
                        and previous["size"] == info.st_size
                        and previous["mtime_ns"] == info.st_mtime_ns
                        and previous["status"]
                        not in {"pending", "skipped: " + SKIP_REASONS["read"]}
                        and not (
                            available
                            and previous["status"] in {"keyword", "budget"}
                            and await self.store.call("embedding_room", path)
                            > previous.get("embedded", 0)
                        )
                    ):
                        if previous["status"].startswith("skipped"):
                            skipped += 1
                        continue
                    if file_kind(path) != "text":
                        media_cap = 1 if file_kind(path) == "image" else 120
                        media_available = (
                            available
                            and await self.store.call("embedding_room", path)
                            >= media_cap
                        )
                        chunks, vectors, media_ok, error = await self._index_media(
                            path, root, media_available, root_fd
                        )
                        model_error = str(error) if error else model_error
                        load_failed |= (
                            isinstance(error, ModelLoadError) and error.load_failed
                        )
                        if media_available:
                            available = media_ok
                        after = await asyncio.to_thread(path.lstat)
                        if (info.st_size, info.st_mtime_ns, info.st_ino) != (
                            after.st_size,
                            after.st_mtime_ns,
                            after.st_ino,
                        ):
                            raise OSError(
                                "file changed during extraction; reindex to retry"
                            )
                        await self.store.call(
                            "replace",
                            folder["id"],
                            path,
                            info,
                            None,
                            chunks,
                            vectors,
                            "indexed" if vectors is not None else "keyword",
                        )
                        continue
                    if path.suffix.lower() in LEGACY_OFFICE_SUFFIXES:
                        raise SkippedError(SKIP_REASONS["unsupported"])
                    if path.suffix.lower() in AUDIO_SUFFIXES:
                        raise SkippedError(SKIP_REASONS["audio"])
                    if path.suffix.lower() not in TEXT_SUFFIXES | OFFICE_SUFFIXES | {
                        ".pdf"
                    } and path.name.lower() not in {
                        "makefile",
                        "dockerfile",
                        "license",
                        "readme",
                    }:
                        raise SkippedError(SKIP_REASONS["unsupported"])
                    if info.st_size > MAX_BYTES:
                        raise SkippedError(SKIP_REASONS["size"])
                    data, info = await cpu_call(read_safe, path, root, root_fd=root_fd)
                    digest = await asyncio.to_thread(
                        lambda data=data: hashlib.sha256(data).hexdigest()
                    )
                    if (
                        previous
                        and previous["content_hash"] == digest
                        and previous["status"] == "indexed"
                    ):
                        await self.store.call("touch", previous["id"], info)
                        continue
                    try:
                        units = await parse_document(path, data)
                    except SkippedError:
                        raise
                    except Exception as exc:
                        raise SkippedError("extraction failed") from exc
                    chunks = await asyncio.to_thread(chunk_units, units)
                    if not chunks:
                        raise SkippedError(SKIP_REASONS["empty"])
                    vectors = None
                    eligible = await asyncio.to_thread(eligible_chunks, path, chunks)
                    remaining = await self.store.call("embedding_room", path)
                    if available and eligible and remaining > 0:
                        order = sorted(
                            eligible[: min(MAX_EMBEDDED_CHUNKS_PER_FILE, remaining)],
                            key=lambda i: len(chunks[i].text),
                        )
                        ordered_vectors = []
                        try:
                            for start in range(0, len(order), self.batch_size):
                                batch = order[start : start + self.batch_size]
                                embedded = await self._embed_async(
                                    [
                                        f"title: {path.stem} | text: {chunks[i].text}"
                                        for i in batch
                                    ],
                                    background=True,
                                )
                                if embedded is None:
                                    available = False
                                    break
                                ordered_vectors.extend(embedded)
                                await asyncio.sleep(self.yield_seconds)
                            if available:
                                vectors = [None] * len(chunks)
                                for i, vector in zip(order, ordered_vectors):
                                    vectors[i] = vector
                        except ModelLoadError as exc:
                            logger.warning("Local-search model unavailable: %s", exc)
                            await self._release_after_model_error()
                            available = False
                            model_error = str(exc)
                            load_failed |= exc.load_failed
                        except Exception as exc:
                            logger.debug("Local-search text embedding skipped: %s", exc)
                    await self.store.call(
                        "replace",
                        folder["id"],
                        path,
                        info,
                        digest,
                        chunks,
                        vectors,
                        (
                            "budget"
                            if available
                            and eligible
                            and remaining
                            < min(len(eligible), MAX_EMBEDDED_CHUNKS_PER_FILE)
                            else "indexed"
                            if vectors is not None or not eligible
                            else "keyword"
                        ),
                    )
                except SkippedError as exc:
                    skipped += 1
                    try:
                        skipped_info = await asyncio.to_thread(path.lstat)
                    except FileNotFoundError:
                        if previous:
                            await self.store.call("remove_file", previous["id"])
                        continue
                    await self.store.call(
                        "replace",
                        folder["id"],
                        path,
                        skipped_info,
                        None,
                        [],
                        None,
                        "skipped: " + skip_reason(str(exc)),
                    )
                except FileNotFoundError:
                    if previous:
                        await self.store.call("remove_file", previous["id"])
                except (OSError, ValueError) as exc:
                    # Unreadable documents do not abort the folder.
                    skipped += 1
                    try:
                        skipped_info = await asyncio.to_thread(path.lstat)
                    except OSError:
                        if previous:
                            await self.store.call("remove_file", previous["id"])
                    else:
                        await self.store.call(
                            "replace",
                            folder["id"],
                            path,
                            skipped_info,
                            None,
                            [],
                            None,
                            "skipped: " + SKIP_REASONS["read"],
                        )
                    logger.debug("Local-search skipped %s: %s", path, exc)
                finally:
                    self.progress["files_done"] += 1
                    await self.store.call(
                        "set_folder", folder["id"], "indexing", skipped=skipped
                    )
                    await asyncio.sleep(self.yield_seconds)
            self.progress["files_total"] = self.progress["files_done"]
            for path, row in old.items():
                if path not in existing:
                    await self.store.call("remove_file", row["id"])
            # Only a real model failure on this checkpoint/worker suppresses
            # polls. Missing models/workers and replacement workers can recover.
            # Retain the callable itself: bound methods have stable equality.
            if (
                checkpoint is not None
                and callable(worker)
                and load_failed
                and self._worker() == worker
            ):
                self._attempted_checkpoints[folder["id"]] = (checkpoint, worker)
            else:
                self._attempted_checkpoints.pop(folder["id"], None)
            await self.store.call(
                "set_folder",
                folder["id"],
                "idle" if available else "waiting_for_model",
                error=model_error,
                skipped=skipped,
                completed=True,
            )
        finally:
            await asyncio.to_thread(batches.close)

    async def search(
        self,
        query: str,
        limit: int,
        folder_ids: list[str] | None,
        kinds: list[str] | None = None,
        group_limit: int | None = None,
    ) -> dict:
        start = time.perf_counter()
        terms = query_terms(query)
        rows = await self.store.call("search_rows", folder_ids, kinds)
        distinctive = await self.store.call("distinctive_terms", terms)
        terms = bounded_terms(terms, distinctive)
        distinctive = [term for term in distinctive if term in terms]
        highlight_terms = distinctive or [query.strip()]
        evidence: dict[int, int] = {}
        keyword = top_files(
            await self.store.call(
                "keyword",
                terms,
                rows,
                query=query,
                distinctive=distinctive,
                evidence=evidence,
            ),
            rows,
            max(100, limit * 5),
        )
        semantic_legs = []
        semantic_scores = None
        semantic_standardized = {}
        mode = "keyword"
        try:
            vector = await self._embed_async([f"task: search result | query: {query}"])
            if vector is not None:
                pooled_scores = await self.store.call(
                    "semantic_scores", vector[0], rows
                )
                semantic_scores = dict(pooled_scores)
                background = await self.store.call("semantic_background", rows)
                by_kind: dict[str, list[tuple[int, float]]] = {
                    kind: [] for kind in KINDS
                }
                for i, score in pooled_scores:
                    by_kind[rows[i]["kind"]].append((i, score))
                for kind in KINDS:
                    scores = by_kind[kind]
                    statistics = await asyncio.to_thread(query_statistics, scores, rows)
                    ids, mean, deviation = statistics
                    semantic_standardized.update(
                        {
                            i: (semantic_scores[i] - mean) / deviation
                            if deviation > 0
                            else 0.0
                            for i in ids
                        }
                    )
                    semantic_legs.append(
                        semantic_ranking(
                            scores, rows, background[kind], statistics=statistics
                        )
                    )
                mode = "hybrid"
        except Exception as exc:
            logger.warning("Local-search keyword fallback: %s", exc)
        include_groups = (
            group_limit is not None and len(set(KINDS if kinds is None else kinds)) > 1
        )
        folders = {
            f["id"]: Path(f["path"])
            for f in await self.store.call(
                "folders",
            )
        }
        valid_rows = {}

        def admitted(ranking):
            seen = set()
            for chunk_id, score, matched in ranking:
                row = rows[chunk_id]
                path = Path(row["path"])
                root = folders.get(row["folder_id"])
                if root is None:
                    continue
                if row["path"] in seen or excluded(path):
                    continue
                # Do not expose stale results after a source/root has been replaced
                # by a link pointing outside the registered folder.
                try:
                    if row["content_indexed"]:
                        with open_safe(path, root, VIDEO_MAX_BYTES) as (_, info):
                            current = info
                    else:
                        current = stat_safe(path, root)
                    if (
                        current.st_size != row["size"]
                        or current.st_mtime != row["mtime"]
                    ):
                        continue
                except (OSError, ValueError):
                    continue
                seen.add(row["path"])
                valid_rows[chunk_id] = row
                yield chunk_id, score, matched

        result_cache: dict[int, dict[str, object]] = {}

        async def result(hit):
            chunk_id, score, matched = hit
            if chunk_id in result_cache:
                return {**result_cache[chunk_id], "score": score, "matched": matched}
            row = rows[chunk_id]
            text, highlights = (
                await asyncio.to_thread(
                    snippet,
                    await self.store.call("chunk_text", chunk_id),
                    highlight_terms,
                )
                if row["kind"] == "text" and row["content_indexed"]
                else ("", [])
            )
            locator = json.loads(row["locator"])
            result_cache[chunk_id] = {
                "content_indexed": row["content_indexed"],
                "skip_reason": row["skip_reason"],
                "kind": row["kind"],
                "timestamp": locator.get("timestamp")
                if row["kind"] == "video"
                else None,
                "path": row["path"],
                "folder_id": row["folder_id"],
                "title": row["title"],
                "snippet": text,
                "highlights": highlights,
                "score": score,
                "matched": matched,
                "page": locator.get("page"),
                "line": locator.get("line"),
                "modified_at": row["mtime"],
                "size": row["size"],
            }

            return result_cache[chunk_id]

        support = {}
        lexical_paths = {rows[i]["path"] for i in keyword}
        if terms and any(
            rows[i]["kind"] == "text" and rows[i]["path"] not in lexical_paths
            for leg in semantic_legs
            for i in leg
        ):
            support = await self.store.call(
                "query_evidence",
                set(keyword).union(*(set(leg) for leg in semantic_legs)),
                rows,
                terms,
            )
        ranking = await asyncio.to_thread(
            fuse,
            [],
            keyword,
            rows=rows,
            semantic_legs=semantic_legs,
            semantic_scores=semantic_scores,
            keyword_evidence=evidence,
            semantic_standardized=semantic_standardized,
            limit=limit,
            query_evidence=support,
        )
        hits = []
        admitted_hits = await asyncio.to_thread(lambda: list(admitted(ranking)))
        for hit in admitted_hits:
            hits.append(hit)
            if not include_groups and len(hits) == limit:
                break
        response: dict[str, object] = {
            "results": [await result(hit) for hit in hits[:limit]],
            "mode": mode,
        }
        if include_groups:
            # Preserve all safe chunks of admitted files for lexical/semantic
            # representative selection, including the best video frame.
            paths = {row["path"] for row in valid_rows.values()}
            candidates = set(keyword).union(*(set(leg) for leg in semantic_legs))
            group_rows = {i: rows[i] for i in candidates if rows[i]["path"] in paths}
            groups = await asyncio.to_thread(
                grouped_rankings,
                [i for i in keyword if i in group_rows],
                rows=group_rows,
                semantic_legs=[
                    [i for i in leg if i in group_rows] for leg in semantic_legs
                ],
                semantic_scores=semantic_scores,
                semantic_standardized={
                    i: z for i, z in semantic_standardized.items() if i in group_rows
                },
                keyword_evidence=evidence,
                query_evidence=support,
            )
            result_groups: list[dict[str, object]] = []
            response["groups"] = result_groups
            for kind, group in groups:
                result_groups.append(
                    {
                        "kind": kind,
                        "results": [await result(hit) for hit in group[:group_limit]],
                        "total": len(group),
                    }
                )
        response["took_ms"] = (time.perf_counter() - start) * 1000
        return response


def eligible_chunks(path, chunks):
    return [i for i, chunk in enumerate(chunks) if semantic_chunk(path, chunk.text)]


async def cpu_call(func, *args, **kwargs):
    task = asyncio.create_task(asyncio.to_thread(func, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise
