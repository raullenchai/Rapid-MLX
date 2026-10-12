# SPDX-License-Identifier: Apache-2.0
"""Opt-in, disposable system-prefix snapshots for the pinned serial adapter.

Use the runtime's tensor writer, but restore only the live model's cache types.
A disk miss (including corruption) always falls back to normal prefill.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import types
from typing import Any

from .. import disk_caches
from .._env import env_truthy
from .tensorfold_runtime import SUPPORTED_MLX_VERSION, SUPPORTED_REVISION

logger = logging.getLogger(__name__)
ENABLE_ENV = "RAPID_MLX_TENSORFOLD_PREFIX_DISK"
BUDGET_ENV = "RAPID_MLX_TENSORFOLD_PREFIX_DISK_BYTES"
DEFAULT_BUDGET = 2 * 1024**3


def _budget() -> int:
    try:
        return max(0, int(os.environ.get(BUDGET_ENV, DEFAULT_BUDGET)))
    except ValueError:
        logger.warning("Invalid %s; disk prefixes disabled", BUDGET_ENV)
        return 0


def _identity(target: Path, drafter: Path | None, app: Any) -> str:
    return json.dumps({
        "format": 1, "target": str(target.resolve()),
        "drafter": str(drafter.resolve()) if drafter else None,
        "runtime": SUPPORTED_REVISION, "mlx": SUPPORTED_MLX_VERSION,
        "prefill": str(getattr(app._model, "prefill_key", "")),
        "step": app.engine.prefill_plan.step,
    }, sort_keys=True)


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class PrefixDisk:
    def __init__(self, directory: Path, model_id: str, templates: list[Any], budget: int):
        self.directory, self.model_id, self.budget = directory, model_id, budget
        self.templates = [c for c in templates if getattr(c, "stored", True)]

    def _files(self) -> list[Path]:
        return sorted((p for p in self.directory.glob("*.safetensors")
                       if not p.is_symlink() and not p.name.endswith(".partial.safetensors")),
                      key=lambda p: p.stat().st_mtime, reverse=True)

    def load(self, path: Path) -> tuple[list[int], list[Any]] | None:
        try:
            if disk_caches.disabled() or path.is_symlink() or path.stat().st_size > self.budget:
                return None
            if path.with_suffix(".sha256").read_text() != _digest(path):
                return None
            import mlx.core as mx
            import numpy as np
            arrays, meta = mx.load(str(path), return_metadata=True)
            if meta.get("model") != self.model_id or meta.get("format") != "1":
                return None
            tokens, layers = json.loads(meta["tokens"]), json.loads(meta["layers"])
            if (not isinstance(tokens, list) or not tokens
                    or any(type(t) is not int or t < 0 for t in tokens)
                    or not isinstance(layers, list) or len(layers) != len(self.templates)):
                return None
            cache = []
            used = set()
            for index, (entry, template) in enumerate(zip(layers, self.templates, strict=True)):
                cls = type(template)
                if entry["class"] != f"{cls.__module__}:{cls.__qualname__}":
                    return None
                groups = [entry["plain"], entry["arrays"], entry.get("numpy", []), entry["lists"]]
                names = [name for group in groups for name in group]
                allowed = set(vars(template)) - set(getattr(cls, "transient", ()))
                if len(names) != len(set(names)) or set(names) != allowed:
                    return None
                item = copy.copy(template)
                for name, value in entry["plain"].items():
                    if (value is not None and type(value) not in (bool, int, float, str)) or (
                            isinstance(value, float) and not math.isfinite(value)):
                        return None
                    if name == "offset" and (type(value) is not int or value != len(tokens)):
                        return None
                    setattr(item, name, value)
                for name in entry["arrays"] + entry.get("numpy", []):
                    key = f"{index}.{name}"
                    value = arrays[key]
                    used.add(key)
                    setattr(item, name, np.array(value) if name in entry.get("numpy", []) else value)
                for name, spec in entry["lists"].items():
                    length, slots = spec["length"], spec["slots"]
                    if (type(length) is not int or not isinstance(getattr(template, name), list)
                            or length != len(getattr(template, name)) or not isinstance(slots, list)
                            or len(slots) != len(set(slots))
                            or any(type(s) is not int or not 0 <= s < length for s in slots)):
                        return None
                    values = [None] * length
                    for slot in slots:
                        key = f"{index}.{name}.{slot}"
                        values[slot] = arrays[key]
                        used.add(key)
                    setattr(item, name, values)
                cache.append(item)
            if used != set(arrays):
                return None
            mx.eval(list(arrays.values()))
            return tokens, cache
        except Exception:
            logger.debug("Unusable disk prefix; using normal prefill", exc_info=True)
            return None

    def save(self, tokens: list[int], cache: list[Any]) -> None:
        if disk_caches.disabled():
            return
        try:
            from tensorfold.engine.prefix_snapshots import save_snapshot, snapshot_key
            from tensorfold.server.memory_budget import cache_nbytes
            from ..memory_cache import _free_disk_bytes, _resolve_persist_min_free_disk_bytes

            estimate = cache_nbytes(cache) + 4 * len(tokens) + 1024**2
            if estimate > self.budget:
                return
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            reserve = _resolve_persist_min_free_disk_bytes()
            free = _free_disk_bytes(str(self.directory))
            if reserve and (free is None or free - estimate < reserve):
                return
            path = self.directory / f"{snapshot_key(self.model_id, tokens)}.safetensors"
            if path.exists():
                if self.load(path) is not None:
                    os.utime(path)
                    return
                path.unlink()
                path.with_suffix(".sha256").unlink(missing_ok=True)
            # Evict before writing so the peak remains inside the byte budget.
            files = self._files()
            total = sum(p.stat().st_size + 64 for p in files)
            for old in reversed(files):
                if total + estimate <= self.budget:
                    break
                total -= old.stat().st_size + 64
                old.unlink()
                old.with_suffix(".sha256").unlink(missing_ok=True)
            save_snapshot(self.directory, self.model_id, tokens, cache, keep=8)
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            checksum = path.with_suffix(".sha256")
            partial = checksum.with_suffix(f".{os.getpid()}.partial")
            try:
                with partial.open("w") as stream:
                    stream.write(_digest(path))
                    stream.flush()
                    os.fsync(stream.fileno())
                partial.replace(checksum)
            finally:
                partial.unlink(missing_ok=True)
            fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except Exception:
            logger.debug("Disk prefix write failed; keeping memory prefix", exc_info=True)

    def restore(self, scheduler: Any, prompt: list[int], usable: Any = None) -> None:
        if disk_caches.disabled() or scheduler.checkpoints is None:
            return
        try:
            from tensorfold.engine.prefix_snapshots import read_metadata
            have = scheduler.checkpoints.longest(prompt, usable)
            candidates = []
            for path in self._files():
                try:
                    meta = read_metadata(path)
                    if meta.get("model") != self.model_id:
                        continue
                    tokens = json.loads(meta["tokens"])
                    n = len(tokens)
                    if (have < n < len(prompt) and prompt[:n] == tokens
                            and (usable is None or usable(n))):
                        candidates.append((n, path))
                except Exception:
                    continue
            for _, path in sorted(candidates, key=lambda pair: pair[0], reverse=True):
                memory = scheduler.prompt_memory
                if memory is not None and not memory.allow_load(path.stat().st_size):
                    continue
                loaded = self.load(path)
                if loaded is None:
                    continue
                tokens, cache = loaded
                scheduler.checkpoints.insert(tokens, cache, last_prompt=tokens, pinned=True)
                os.utime(path)
                return
        except Exception:
            logger.debug("Disk prefix restore failed; using normal prefill", exc_info=True)


def install_prefix_disk(app: Any, target: Path, drafter: Path | None = None) -> None:
    """Attach instance-local persistence; never enable the runtime's arbitrary imports."""
    if not env_truthy(ENABLE_ENV) or disk_caches.disabled() or not _budget():
        return
    if app.checkpoints is None:
        return
    model_id = _identity(target, drafter, app)
    directory = (Path.home() / ".cache" / "rapid-mlx" / "prefix_cache" /
                 "tensorfold" / hashlib.sha256(model_id.encode()).hexdigest())
    disk = PrefixDisk(directory, model_id, app._model.make_cache(), _budget())
    scheduler = app.scheduler
    scheduler._persist = types.MethodType(lambda _self, tokens, cache: disk.save(tokens, cache), scheduler)
    scheduler._read_disk_block = types.MethodType(lambda self, prompt, usable=None: disk.restore(self, prompt, usable), scheduler)
    scheduler.snapshot_dir = directory
    app._rapid_prefix_disk = disk
