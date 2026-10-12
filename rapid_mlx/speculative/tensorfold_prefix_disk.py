# SPDX-License-Identifier: Apache-2.0
"""Disposable system blocks for one pinned TensorFold profile, off by default.

Never invoke the runtime's import-by-name loader. Read bounded, verified bytes
once and rebuild only the live cache schema. All failures are ordinary misses.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import inspect
import json
import logging
import math
import os
import shutil
import stat
import struct
import tempfile
import types
from pathlib import Path
from typing import Any

from .. import disk_caches
from .._env import env_truthy
from .tensorfold_runtime import (
    SUPPORTED_MLX_VERSION,
    SUPPORTED_REVISION,
    SUPPORTED_VERSION,
)

logger = logging.getLogger(__name__)
ENABLE_ENV = "RAPID_MLX_TENSORFOLD_PREFIX_DISK"
BUDGET_ENV = "RAPID_MLX_TENSORFOLD_PREFIX_DISK_BYTES"
DIRECTORY_ENV = "RAPID_MLX_TENSORFOLD_PREFIX_DISK_DIR"
DEFAULT_BUDGET = 2 * 1024**3
HEADER_LIMIT = 1024**2
# Deliberately independent of the adapter's mutable pin: moving it requires a
# new seam audit and parity qualification, not merely changing an import.
QUALIFIED = ("0.6.6", "cb2ebf0540f42604e2759b2ddef497861e928248", "0.32.3")
SEAMS = {
    "server/scheduler.py": "07ae6d1b214789a450163b54c8e734d64b20be324147938f9c9a69eee453ad2d",
    "server/prompt_fill.py": "f9e5fd8fef71885505ded377405e9d6e428ae063db4882024d856ff89ba5b079",
    "server/app.py": "3f47ab553c35f72001127ea89801f567885a58ca6bfa65d2ae40498041d7b5e3",
    "server/checkpoints.py": "f1868970b7c93f365c212bf6c6591774486df5e6a1adeb0e9b373a5afdbd2bbb",
    "engine/prefix_snapshots.py": "d16bcddf58f884782dc5088b2f6307029a63b9164a4f41166f5491d6a5e0e62d",
    "engine/family_prefill.py": "48f8bed50b8528fd4f55da5593b9ff340a8642e804939314a027c311993ed6dc",
    "families/nemotron_h/model.py": "d4931226bb41221d7abb791c76330c3b8fe8dd40161655aa14db600370c8f19c",
}
CACHE_CLASSES = {
    "tensorfold.engine.alternating_kv:AlternatingKVCache",
    "tensorfold.families.nemotron_h.mtp:MTPCache",
    "tensorfold.families.nemotron_h.state_cache:RowStateCache",
}
DTYPES = {
    "float32": ("F32", "<f4"),
    "float16": ("F16", "<f2"),
    "bfloat16": ("BF16", "<u2"),
}


def _budget() -> int:
    try:
        return max(0, int(os.environ.get(BUDGET_ENV, DEFAULT_BUDGET)))
    except ValueError:
        logger.warning("Invalid %s; disk prefixes disabled", BUDGET_ENV)
        return 0


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _artifact(path: Path | None) -> Any:
    """Hash contents, including quantized weights and tokenizer/template assets.

    No stat-only memoization: replacing a checkpoint in place must invalidate it.
    This intentionally adds startup I/O, outside request TTFT.
    """
    if path is None:
        return None
    files = (
        [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    )
    if not files:
        raise ValueError("empty model artifact")
    return {
        "path": str(path.resolve()),
        "contents": [
            (str(p.relative_to(path)) if p != path else p.name, _digest(p))
            for p in files
        ],
    }


def _identity(target: Path, drafter: Path | None, app: Any) -> str:
    model = app._model
    mtp = None
    if getattr(model, "mtp", None) is not None:
        from tensorfold.families.nemotron_h.model import find_mtp_head

        mtp = _artifact(find_mtp_head(target))
        if mtp is None:
            raise ValueError("unidentified MTP head")
    return json.dumps(
        {
            "format": 2,
            "target": _artifact(target),
            "drafter": _artifact(drafter),
            "mtp": mtp,
            "runtime": SUPPORTED_REVISION,
            "mlx": SUPPORTED_MLX_VERSION,
            "dependencies": {
                k: importlib.metadata.version(k)
                for k in ("mlx-lm", "transformers", "tokenizers")
            },
            "template": getattr(app.tokenizer, "chat_template", None),
            "prefill": str(getattr(model, "prefill_key", "")),
            "plan": vars(app.engine.prefill_plan),
            "engine": {
                k: getattr(app.engine, k, None) for k in ("prefill_pass", "pass_cache")
            },
            "model": {
                k: getattr(model, k, None)
                for k in ("fused_rows", "window_rows", "exact_width", "drafts")
            },
            "environment": {
                k: v for k, v in os.environ.items() if k.startswith(("TF_", "MLX_"))
            },
        },
        sort_keys=True,
    )


def _qualified(scheduler: Any) -> bool:
    import tensorfold

    from .tensorfold_qwen27 import require_environment

    if (SUPPORTED_VERSION, SUPPORTED_REVISION, SUPPORTED_MLX_VERSION) != QUALIFIED:
        return False
    require_environment()
    root = Path(tensorfold.__file__).parent
    if any(_digest(root / name) != digest for name, digest in SEAMS.items()):
        return False
    return (
        tuple(inspect.signature(scheduler._persist).parameters) == ("tokens", "cache")
        and tuple(inspect.signature(scheduler._read_disk_block).parameters)
        == ("prompt", "usable")
        and scheduler.snapshot_dir is None
        and scheduler.session_dir is None
        and scheduler.disk_blocks is None
        and scheduler.session_blocks is None
    )


def _json(value: bytes | str) -> Any:
    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = item
        return result

    return json.loads(
        value,
        object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")),
    )


def _read_file(path: Path, limit: int) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_size > limit
            or info.st_size <= 0
        ):
            raise ValueError("unsafe snapshot file")
        data = stream.read(limit + 1)
        if len(data) != info.st_size or len(data) > limit:
            raise ValueError("snapshot changed or oversized")
        return data


def _header(stream: Any, size: int) -> tuple[dict, int]:
    length = struct.unpack("<Q", stream.read(8))[0]
    if not 0 < length <= min(HEADER_LIMIT, size - 8):
        raise ValueError("unbounded tensor header")
    header = _json(stream.read(length))
    if not isinstance(header, dict):
        raise ValueError("invalid tensor header")
    return header, 8 + length


class PrefixDisk:
    def __init__(
        self,
        directory: Path,
        model_id: str,
        templates: list[Any],
        budget: int,
        max_tokens: int = 32768,
    ):
        self.directory, self.model_id, self.budget = directory, model_id, budget
        self.max_tokens = max_tokens
        self.templates = [c for c in templates if getattr(c, "stored", True)]
        self.schema = []
        for template in self.templates:
            cls = type(template)
            name = f"{cls.__module__}:{cls.__qualname__}"
            if name not in CACHE_CLASSES:
                raise ValueError("unqualified live cache class")
            getattr(template, "materialize", lambda: None)()
            fields = {
                k: v
                for k, v in vars(template).items()
                if k not in getattr(cls, "transient", ())
            }
            self.schema.append((name, fields))

    def _files(self) -> list[Path]:
        return sorted(
            (
                p
                for p in self.directory.glob("*.safetensors")
                if not p.is_symlink() and p.is_file()
            ),
            key=lambda p: p.stat().st_mtime_ns,
            reverse=True,
        )

    def _metadata(self, path: Path) -> dict:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_size > self.budget
            ):
                raise ValueError("unsafe snapshot")
            header, _ = _header(stream, info.st_size)
        return header["__metadata__"]

    def _decode(self, data: bytes) -> tuple[list[int], list[Any]]:
        """Validate all tensor geometry before constructing any MLX array.

        Converting from verified bytes avoids a pathname/checksum race and keeps
        damaged headers out of MLX's native file loader entirely.
        """
        import io

        import mlx.core as mx
        import numpy as np

        header, start = _header(io.BytesIO(data), len(data))
        meta = header.pop("__metadata__")
        if meta.get("model") != self.model_id or meta.get("format") != "1":
            raise ValueError("foreign snapshot")
        tokens, layers = _json(meta["tokens"]), _json(meta["layers"])
        if (
            not isinstance(tokens, list)
            or not 0 < len(tokens) <= self.max_tokens
            or any(type(t) is not int or t < 0 for t in tokens)
            or not isinstance(layers, list)
            or len(layers) != len(self.schema)
        ):
            raise ValueError("invalid tokens/layers")
        specs, restored = {}, []
        for i, (entry, (clsname, fields), template) in enumerate(
            zip(layers, self.schema, self.templates, strict=True)
        ):
            if (
                set(entry) != {"class", "plain", "arrays", "lists", "numpy"}
                or entry["class"] != clsname
                or entry["numpy"] != []
            ):
                raise ValueError("invalid layer schema")
            plain, names, lists = entry["plain"], entry["arrays"], entry["lists"]
            if (
                not isinstance(plain, dict)
                or not isinstance(names, list)
                or not isinstance(lists, dict)
            ):
                raise ValueError("invalid fields")
            all_names = [*plain, *names, *lists]
            if len(all_names) != len(set(all_names)) or set(all_names) != set(fields):
                raise ValueError("unknown or missing attribute")
            item = copy.copy(template)
            for name, live in fields.items():
                if isinstance(live, mx.array):
                    if name not in names:
                        raise ValueError("array field changed kind")
                    specs[f"{i}.{name}"] = (
                        item,
                        name,
                        None,
                        live,
                        name in ("keys", "values"),
                    )
                elif isinstance(live, list):
                    spec = lists[name]
                    if spec != {"length": len(live), "slots": list(range(len(live)))}:
                        raise ValueError("invalid list slots")
                    setattr(item, name, [None] * len(live))
                    for slot, value in enumerate(live):
                        if not isinstance(value, mx.array):
                            raise ValueError("uninitialized live array schema")
                        specs[f"{i}.{name}.{slot}"] = (item, name, slot, value, False)
                else:
                    value = plain[name]
                    if name == "offset":
                        if type(value) is not int or value != len(tokens):
                            raise ValueError("invalid offset")
                    elif type(value) is not type(live) or value != live:
                        raise ValueError("scalar configuration changed")
                    setattr(item, name, value)
            restored.append(item)
        if set(specs) != set(header):
            raise ValueError("extra or missing tensor")
        spans = []
        for key, (_, _, _, live, sequence) in specs.items():
            spec = header[key]
            dtype, numpy_dtype = DTYPES[str(live.dtype).removeprefix("mlx.core.")]
            shape = spec["shape"]
            if (
                not isinstance(shape, list)
                or any(type(n) is not int or n <= 0 for n in shape)
                or len(shape) != live.ndim
                or spec["dtype"] != dtype
            ):
                raise ValueError("wrong tensor shape/dtype")
            expected = list(live.shape)
            if sequence:
                if live.ndim != 4:
                    raise ValueError("unqualified KV geometry")
                # Writers only retain valid rows; spare capacity could contain
                # user/reply data and must never be persisted or restored.
                expected[2] = len(tokens)
            if shape != expected:
                raise ValueError("wrong live tensor geometry")
            offsets = spec["data_offsets"]
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(n) is not int or n < 0 for n in offsets)
                or offsets[1] - offsets[0]
                != math.prod(shape) * np.dtype(numpy_dtype).itemsize
                or start + offsets[1] > len(data)
            ):
                raise ValueError("invalid tensor span")
            spans.append(tuple(offsets))
        cursor = 0
        for begin, end in sorted(spans):
            if begin != cursor:
                raise ValueError("overlapping or unaccounted tensor bytes")
            cursor = end
        if start + cursor != len(data):
            raise ValueError("trailing tensor bytes")
        for key, (item, name, slot, live, _) in specs.items():
            spec = header[key]
            _, numpy_dtype = DTYPES[str(live.dtype).removeprefix("mlx.core.")]
            value = mx.array(
                np.frombuffer(
                    data,
                    dtype=numpy_dtype,
                    count=math.prod(spec["shape"]),
                    offset=start + spec["data_offsets"][0],
                ).reshape(spec["shape"])
            )
            if str(live.dtype).removeprefix("mlx.core.") == "bfloat16":
                value = value.view(mx.bfloat16)
            if not bool(mx.all(mx.isfinite(value)).item()):
                raise ValueError("nonfinite tensor")
            if slot is None:
                setattr(item, name, value)
            else:
                getattr(item, name)[slot] = value
        mx.eval(
            [v for c in restored for v in vars(c).values() if isinstance(v, mx.array)]
        )
        return tokens, restored

    def load(self, path: Path) -> tuple[list[int], list[Any]] | None:
        if disk_caches.disabled():
            return None
        try:
            if path.parent != self.directory or self.directory.is_symlink():
                return None
            data = _read_file(path, self.budget)
            checksum = _read_file(path.with_suffix(".sha256"), 64).decode("ascii")
            if checksum != hashlib.sha256(data).hexdigest():
                return None
            return self._decode(data)
        except Exception:
            logger.debug("Unusable disk prefix; using normal prefill", exc_info=True)
            return None

    def _private_directory(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        info = self.directory.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
        ):
            raise ValueError(
                "snapshot directory must be private, owned, and not a symlink"
            )
        lock = Path(str(self.directory) + ".txlock")
        if lock.is_symlink():
            raise ValueError("unsafe lock path")

    def save(self, tokens: list[int], cache: list[Any]) -> None:
        if disk_caches.disabled() or not self.budget:
            return
        try:
            from tensorfold.engine.prefix_snapshots import save_snapshot, snapshot_key
            from tensorfold.server.memory_budget import cache_nbytes

            from ..memory_cache import (
                _free_disk_bytes,
                _resolve_persist_min_free_disk_bytes,
            )
            from ..runtime.cache import _exclusive_cache_lock

            estimate = cache_nbytes(cache) + 12 * len(tokens) + HEADER_LIMIT
            if estimate > self.budget or not 0 < len(tokens) <= self.max_tokens:
                return
            self._private_directory()
            with _exclusive_cache_lock(
                str(self.directory), operation="TensorFold save"
            ) as locked:
                if not locked or disk_caches.disabled():
                    return
                # Holding the whole transaction lock means any leftover stage
                # belongs to an interrupted writer, never a live peer.
                for stage in self.directory.glob(".stage-*"):
                    if stage.is_dir() and not stage.is_symlink():
                        shutil.rmtree(stage)
                for checksum in self.directory.glob("*.sha256"):
                    if not checksum.with_suffix(".safetensors").exists():
                        checksum.unlink()
                path = (
                    self.directory
                    / f"{snapshot_key(self.model_id, tokens)}.safetensors"
                )
                if self.load(path) is not None:
                    os.utime(path, follow_symlinks=False)
                    return
                path.unlink(missing_ok=True)
                path.with_suffix(".sha256").unlink(missing_ok=True)
                files = self._files()
                total = sum(p.stat().st_size + 64 for p in files)
                for old in reversed(files):
                    if total + estimate <= self.budget and len(files) < 8:
                        break
                    total -= old.stat().st_size + 64
                    old.unlink()
                    old.with_suffix(".sha256").unlink(missing_ok=True)
                    files.remove(old)
                reserve = _resolve_persist_min_free_disk_bytes()
                free = _free_disk_bytes(str(self.directory))
                if reserve and (free is None or free - estimate < reserve):
                    return
                with tempfile.TemporaryDirectory(
                    prefix=".stage-", dir=self.directory
                ) as staging:
                    staged = save_snapshot(Path(staging), self.model_id, tokens, cache)
                    if staged is None or staged.stat().st_size + 64 > estimate:
                        return
                    # Also validate our writer's actual output against the live
                    # schema. Unqualified runtime cache changes cannot publish.
                    data = _read_file(staged, self.budget)
                    self._decode(data)
                    os.chmod(staged, 0o600)
                    with staged.open("rb") as stream:
                        os.fsync(stream.fileno())
                    checksum = Path(staging) / "digest"
                    fd = os.open(checksum, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(fd, "w") as stream:
                        stream.write(hashlib.sha256(data).hexdigest())
                        stream.flush()
                        os.fsync(stream.fileno())
                    staged.replace(path)
                    checksum.replace(path.with_suffix(".sha256"))
                fd = os.open(
                    self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                )
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        except Exception:
            logger.debug(
                "Disk prefix write failed; keeping memory prefix", exc_info=True
            )

    def restore(self, scheduler: Any, prompt: list[int], usable: Any = None) -> None:
        if (
            disk_caches.disabled()
            or scheduler.checkpoints is None
            or not self.directory.exists()
        ):
            return
        try:
            from ..runtime.cache import _exclusive_cache_lock

            self._private_directory()
            with _exclusive_cache_lock(
                str(self.directory), operation="TensorFold load"
            ) as locked:
                if not locked or disk_caches.disabled():
                    return
                have = scheduler.checkpoints.longest(prompt, usable)
                candidates = []
                for path in self._files():
                    try:
                        meta = self._metadata(path)
                        if meta.get("model") != self.model_id:
                            continue
                        tokens = _json(meta["tokens"])
                        n = len(tokens)
                        if (
                            have < n < len(prompt)
                            and prompt[:n] == tokens
                            and (usable is None or usable(n))
                        ):
                            candidates.append((n, path))
                    except Exception:
                        continue
                for _, path in sorted(
                    candidates, key=lambda pair: pair[0], reverse=True
                ):
                    memory = scheduler.prompt_memory
                    # Verified bytes + host conversion + MLX arrays are admitted
                    # conservatively, rather than charging just the final cache.
                    if memory is not None and not memory.allow_load(
                        3 * path.stat().st_size
                    ):
                        continue
                    loaded = self.load(path)
                    if loaded is None:
                        continue
                    tokens, cache = loaded
                    n = len(tokens)
                    if not (
                        have < n < len(prompt)
                        and prompt[:n] == tokens
                        and (usable is None or usable(n))
                    ):
                        continue
                    scheduler.checkpoints.insert(
                        tokens, cache, last_prompt=tokens, pinned=True
                    )
                    os.utime(path, follow_symlinks=False)
                    return
        except Exception:
            logger.debug(
                "Disk prefix restore failed; using normal prefill", exc_info=True
            )


def install_prefix_disk(app: Any, target: Path, drafter: Path | None = None) -> None:
    """Install only after construction with snapshot/session I/O disabled."""
    if not env_truthy(ENABLE_ENV) or disk_caches.disabled() or not _budget():
        return
    try:
        scheduler = app.scheduler
        if app.checkpoints is None or not _qualified(scheduler):
            return
        model_id = _identity(target, drafter, app)
        directory = Path(
            os.environ.get(
                DIRECTORY_ENV,
                str(
                    Path.home() / ".cache" / "rapid-mlx" / "prefix_cache" / "tensorfold"
                ),
            )
        ).expanduser()

        # Populate a tiny live template once during initialization, not from
        # disk metadata. No generated tokens or checkpoint entries are created.
        def prepare(engine):
            templates = engine.prefill_prefix(
                [int(app.tokenizer.encode("\n", add_special_tokens=False)[-1])] * 16
            )
            try:
                return PrefixDisk(
                    directory, model_id, templates, _budget(), app.context_window
                )
            finally:
                getattr(app._model, "release_rounds", lambda: None)()

        disk = scheduler.on_engine(prepare)
        scheduler._persist = types.MethodType(
            lambda _self, tokens, cache: disk.save(tokens, cache), scheduler
        )
        scheduler._read_disk_block = types.MethodType(
            lambda self, prompt, usable=None: disk.restore(self, prompt, usable),
            scheduler,
        )
        scheduler.snapshot_dir = directory
        app._rapid_prefix_disk = disk
    except Exception:
        logger.warning(
            "TensorFold disk prefixes unavailable; feature off", exc_info=True
        )
