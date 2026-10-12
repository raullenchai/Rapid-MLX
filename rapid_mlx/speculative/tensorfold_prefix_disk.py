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
import re
import shutil
import stat
import struct
import tempfile
import threading
import time
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
READ_TIMEOUT = 2.0  # Healthy full restores measured <= 0.309 s, including MLX.
# Deliberately independent of the adapter's mutable pin: moving it requires a
# new seam audit and parity qualification, not merely changing an import.
QUALIFIED = ("0.6.6", "cb2ebf0540f42604e2759b2ddef497861e928248", "0.32.3")
SEAMS = {
    "server/text.py": "c07b60d17e966e5d568f87dcc7d85e532d553089a23e6580e5982cce8c4c58ad",
    "server/prompt_blocks.py": "e8b5e11f3b7eb5159360df5ec8446f8d544561384ecaa231ed969402cba401e5",
    "server/scheduler.py": "07ae6d1b214789a450163b54c8e734d64b20be324147938f9c9a69eee453ad2d",
    "server/prompt_memory.py": "c63a3ed69c75cac641171f6242d37b8804870e3e7e348ddf61b4ce92f58b4248",
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


def _generation(path: Path, *, follow: bool = True) -> tuple[int, ...]:
    info = path.stat() if follow else path.lstat()
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _hf_lfs_identity(path: Path) -> Any:
    """HF LFS content address plus mutable generation; other layouts hash fully."""
    if path.suffix != ".safetensors" or not path.is_symlink():
        return None
    for snapshot in path.parents:
        if (
            snapshot.parent.name != "snapshots"
            or not snapshot.parent.parent.name.startswith("models--")
            or re.fullmatch(r"[0-9a-f]{40}", snapshot.name) is None
        ):
            continue
        blobs = snapshot.parent.parent / "blobs"
        direct = path.parent / path.readlink()
        resolved = path.resolve(strict=True)
        if (
            blobs.is_symlink()
            or direct.is_symlink()
            or resolved.parent != blobs.resolve(strict=True)
            or re.fullmatch(r"[0-9a-f]{64}", resolved.name) is None
            or not stat.S_ISREG(resolved.lstat().st_mode)
        ):
            return None
        return {
            "source": "hf-lfs-sha256-generation",
            "revision": snapshot.name,
            "blob": str(resolved),
            "sha256": resolved.name,
            "generation": _generation(resolved),
            "link_generation": _generation(path, follow=False),
        }
    return None


def _artifact(path: Path | None) -> Any:
    """Identify immutable HF LFS generations; fully hash every other file."""
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
            (
                str(p.relative_to(path)) if p != path else p.name,
                _hf_lfs_identity(p) or _digest(p),
            )
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
            # Older namespaces trusted a shared prefix that could include replies.
            "format": 7,
            "asset_policy": "hf-lfs-generation-or-content-unchanged-before-start",
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


def _assets_unchanged(
    target: Path, drafter: Path | None, app: Any, started_at: float
) -> bool:
    """Reject local ctime changes across loading; process birth precedes loading."""
    roots = [target, *([drafter] if drafter is not None else [])]
    if getattr(app._model, "mtp", None) is not None:
        from tensorfold.families.nemotron_h.model import find_mtp_head

        head = find_mtp_head(target)
        if head is None:
            return False
        roots.append(head)
    for root in roots:
        entries = [root.parent, root]
        for ancestor in (root, *root.parents):
            if ancestor.is_symlink():
                entries.append(ancestor)
            if ancestor.parent.name == "snapshots" and (
                ancestor.parent.parent.name.startswith("models--")
            ):
                # A whole repo/blob-directory swap can preserve every child
                # status time. Check the mutable HF containers as well.
                repo = ancestor.parent.parent
                entries.extend((repo, repo / "blobs"))
        if root.is_dir():
            entries.extend(root.rglob("*"))
        for entry in entries:
            if entry.is_symlink():
                direct = entry.parent / entry.readlink()
                # A blob replaced by another symlink must not hide its new
                # status time behind an older resolved file. Chained links are
                # conservatively unqualified rather than chasing mutable hops.
                if direct.is_symlink() or direct.lstat().st_ctime >= started_at:
                    return False
            # lstat sees replacement of a link; stat sees changes to its blob.
            if (
                entry.lstat().st_ctime >= started_at
                or entry.stat().st_ctime >= started_at
            ):
                return False
    return True


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
        and tuple(inspect.signature(scheduler._keep_checkpoints).parameters)
        == ("job", "shared_at")
        and tuple(inspect.signature(scheduler._open_job).parameters) == ("job",)
        and tuple(inspect.signature(scheduler.submit).parameters) == ("job",)
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
        data = stream.read(info.st_size + 1)
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
        self._busy = threading.Lock()
        self._closed = threading.Event()
        self._worker = None
        self.reserved_bytes = 0
        self.read_reserved_bytes = 0
        self._reading = threading.Lock()
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

    def _decode(
        self, data: bytes, *, validate_only: bool = False, prompt=None
    ) -> tuple[list[int], list[Any]]:
        """Validate verified bytes against live geometry before any MLX allocation."""
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
        if any(name.endswith(":MTPCache") for name, _ in self.schema):
            lookahead = _json(meta["lookahead"])
            if type(lookahead) is not int or lookahead < 0:
                raise ValueError("missing MTP lookahead")
        if prompt is not None and (
            prompt[: len(tokens)] != tokens
            or (
                "lookahead" in meta
                and (
                    len(tokens) >= len(prompt)
                    or prompt[len(tokens)] != _json(meta["lookahead"])
                )
            )
        ):
            raise ValueError("verified snapshot lookahead does not match prompt")
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
        if validate_only:
            return tokens, []
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

    def _verified_bytes(self, path):
        if path.parent != self.directory or self.directory.is_symlink():
            raise ValueError("unsafe snapshot path")
        data = _read_file(path, self.budget)
        if (
            _read_file(path.with_suffix(".sha256"), 64).decode("ascii")
            != hashlib.sha256(data).hexdigest()
        ):
            raise ValueError("bad checksum")
        return data

    def load(self, path: Path, *, prompt=None) -> tuple[list[int], list[Any]] | None:
        if disk_caches.disabled():
            return None
        try:
            return self._decode(self._verified_bytes(path), prompt=prompt)
        except Exception:
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

    def _lookahead(self, tokens, prompt, shared_prefix_len):
        if not any(name.endswith(":MTPCache") for name, _ in self.schema):
            return None
        n = len(tokens)
        if (
            prompt is None
            or not n < shared_prefix_len <= len(prompt)
            or prompt[:n] != tokens
        ):
            raise ValueError("MTP lookahead is not inside the shared system prefix")
        return int(prompt[n])

    def submit(self, tokens, cache, *, prompt=None, shared_prefix_len=0, memory=None):
        """Admit/copy owned host bytes; one in-flight write, drop when busy."""
        if disk_caches.disabled() or self._closed.is_set() or not self.budget:
            return False
        if not self._busy.acquire(blocking=False):
            return False
        transferred = False
        try:
            from tensorfold.server.memory_budget import cache_nbytes

            tokens = list(tokens)
            size = cache_nbytes(cache)
            if size + HEADER_LIMIT > self.budget or (
                memory is not None and not memory.allow_load(3 * size + HEADER_LIMIT)
            ):
                return False
            lookahead = self._lookahead(tokens, prompt, shared_prefix_len)
            payload = self._capture(tokens, cache, lookahead)
            self.reserved_bytes = 3 * size + HEADER_LIMIT

            def publish():
                try:
                    self.save(tokens, [], _payload=payload)
                finally:
                    self.reserved_bytes = 0
                    self._busy.release()

            self._worker = threading.Thread(
                target=publish, name="tensorfold-prefix-writer", daemon=True
            )
            self._worker.start()
            transferred = True
            return True
        except Exception:
            logger.debug("Disk prefix copy refused", exc_info=True)
            return False
        finally:
            # The worker owns the reservation only after a successful start.
            if not transferred:
                self.reserved_bytes = 0
                self._busy.release()

    def wait(self, timeout=0.05):
        """Bounded diagnostic drain; production close never waits past 50 ms."""
        worker = self._worker
        if worker is not None:
            worker.join(timeout)
        return worker is None or not worker.is_alive()

    def close(self):
        self._closed.set()
        self.wait(0.05)

    def _capture(self, tokens, cache, lookahead):
        import mlx.core as mx
        import numpy as np

        arrays, layers = {}, []
        for index, live in enumerate(c for c in cache if getattr(c, "stored", True)):
            getattr(live, "materialize", lambda: None)()
            entry = {
                "class": f"{type(live).__module__}:{type(live).__qualname__}",
                "plain": {},
                "arrays": [],
                "lists": {},
                "numpy": [],
            }
            for name, value in vars(live).items():
                if name in getattr(type(live), "transient", ()):
                    continue

                def capture(key, value, name=name, live=live):
                    if name in ("keys", "values"):
                        if live.offset != len(tokens):
                            raise ValueError("cache not at boundary")
                        value = value[..., : len(tokens), :]
                    mx.eval(value)
                    dtype, _ = DTYPES[str(value.dtype).removeprefix("mlx.core.")]
                    host = np.array(value.view(mx.uint16) if dtype == "BF16" else value)
                    arrays[key] = (dtype, list(value.shape), host.tobytes())

                if isinstance(value, mx.array):
                    capture(f"{index}.{name}", value)
                    entry["arrays"].append(name)
                elif isinstance(value, list):
                    for slot, element in enumerate(value):
                        capture(f"{index}.{name}.{slot}", element)
                    entry["lists"][name] = {
                        "length": len(value),
                        "slots": list(range(len(value))),
                    }
                elif value is None or type(value) in (bool, int, float, str):
                    entry["plain"][name] = value
                else:
                    raise ValueError("unqualified host field")
            layers.append(entry)
        meta = {
            "format": "1",
            "model": self.model_id,
            "tokens": json.dumps(tokens),
            "layers": json.dumps(layers),
            "saved": str(time.time()),
        }
        if lookahead is not None:
            meta["lookahead"] = json.dumps(lookahead)
        return arrays, meta

    def _write_host(self, path, payload):
        # Standard safetensors wire format; no MLX work on the I/O worker.
        arrays, meta = payload
        header, offset = {"__metadata__": meta}, 0
        for key, (dtype, shape, data) in arrays.items():
            header[key] = {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + len(data)],
            }
            offset += len(data)
        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        with path.open("xb") as stream:
            stream.write(struct.pack("<Q", len(encoded)))
            stream.write(encoded)
            for _, _, data in arrays.values():
                stream.write(data)

    def save(
        self,
        tokens: list[int],
        cache: list[Any],
        *,
        prompt=None,
        shared_prefix_len=0,
        _payload=None,
    ) -> None:
        if disk_caches.disabled() or not self.budget:
            return
        try:
            from tensorfold.engine.prefix_snapshots import snapshot_key

            from ..memory_cache import (
                _free_disk_bytes,
                _resolve_persist_min_free_disk_bytes,
            )
            from ..runtime.cache import _exclusive_cache_lock

            if _payload is None:
                _payload = self._capture(
                    tokens, cache, self._lookahead(tokens, prompt, shared_prefix_len)
                )
            estimate = (
                sum(len(data) for _, _, data in _payload[0].values())
                + 12 * len(tokens)
                + HEADER_LIMIT
            )
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
                if path.exists():
                    try:
                        data = self._verified_bytes(path)
                        self._decode(data, validate_only=True)
                        os.utime(path, follow_symlinks=False)
                        return
                    except Exception:
                        pass
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
                    staged = Path(staging) / "snapshot.safetensors"
                    self._write_host(staged, _payload)
                    if staged.stat().st_size + 64 > estimate:
                        return
                    data = _read_file(staged, self.budget)
                    # Geometry check only: no MLX allocation on the I/O worker.
                    self._decode(data, validate_only=True)
                    os.chmod(staged, 0o600)
                    with staged.open("rb") as stream:
                        os.fsync(stream.fileno())
                    checksum = Path(staging) / "digest"
                    fd = os.open(checksum, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(fd, "w") as stream:
                        stream.write(hashlib.sha256(data).hexdigest())
                        stream.flush()
                        os.fsync(stream.fileno())
                    if self._closed.is_set() or disk_caches.disabled():
                        return
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

    def _read_prefix(self, prompt, have, usable, system_len):
        from ..runtime.cache import _exclusive_cache_lock

        if not self.directory.exists():
            return None
        self._private_directory()
        with _exclusive_cache_lock(
            str(self.directory), operation="TensorFold load"
        ) as locked:
            if not locked or disk_caches.disabled() or self._closed.is_set():
                return None
            candidates = []
            for path in self._files():
                try:
                    meta = self._metadata(path)
                    tokens = _json(meta["tokens"])
                    n = len(tokens)
                    if (
                        meta.get("model") == self.model_id
                        and have < n < system_len
                        and prompt[:n] == tokens
                        and (usable is None or usable(n))
                    ):
                        candidates.append((n, path))
                except Exception:
                    continue
            for _, path in sorted(candidates, reverse=True):
                try:
                    data = self._verified_bytes(path)
                    os.utime(path, follow_symlinks=False)
                    return data
                except Exception:
                    continue
        return None

    def restore(
        self, scheduler: Any, prompt: list[int], usable: Any = None, *, system_len=None
    ) -> None:
        acquired = transferred = False
        try:
            if (
                disk_caches.disabled()
                or self._closed.is_set()
                or scheduler.checkpoints is None
            ):
                return
            acquired = self._reading.acquire(blocking=False)
            if not acquired:
                return
            memory = scheduler.prompt_memory
            # No stat/read on the owner: admit the maximum bounded read first.
            if memory is not None and not memory.allow_load(3 * self.budget):
                return
            self.read_reserved_bytes = 3 * self.budget
            have = scheduler.checkpoints.longest(prompt, usable)
            done, abandoned, result = threading.Event(), threading.Event(), []

            def read():
                try:
                    data = self._read_prefix(
                        prompt, have, usable, system_len or len(prompt)
                    )
                    if (
                        data is not None
                        and not abandoned.is_set()
                        and not self._closed.is_set()
                    ):
                        result.append(data)
                except Exception:
                    logger.debug("Disk prefix read missed", exc_info=True)
                finally:
                    self.read_reserved_bytes = 0
                    done.set()
                    self._reading.release()

            deadline = time.monotonic() + READ_TIMEOUT
            threading.Thread(
                target=read, name="tensorfold-prefix-reader", daemon=True
            ).start()
            transferred = True
            while not done.wait(0.01):
                if self._closed.is_set() or time.monotonic() >= deadline:
                    abandoned.set()
                    return
            if (
                not result
                or self._closed.is_set()
                or disk_caches.disabled()
                or time.monotonic() >= deadline
            ):
                return
            tokens, cache = self._decode(result[0], prompt=prompt)
            n = len(tokens)
            if (
                not self._closed.is_set()
                and have < n < (system_len or len(prompt))
                and (usable is None or usable(n))
            ):
                scheduler.checkpoints.insert(
                    tokens, cache, last_prompt=tokens, pinned=True
                )
        except Exception:
            logger.debug("Disk prefix restore missed", exc_info=True)
        finally:
            if acquired and not transferred:
                self.read_reserved_bytes = 0
                self._reading.release()


def install_prefix_disk(app: Any, target: Path, drafter: Path | None = None) -> None:
    """Install only after construction with snapshot/session I/O disabled."""
    if not env_truthy(ENABLE_ENV) or disk_caches.disabled() or not _budget():
        return
    try:
        scheduler = app.scheduler
        if app.checkpoints is None or not _qualified(scheduler):
            return
        import psutil

        started_at = psutil.Process().create_time()
        if not _assets_unchanged(target, drafter, app, started_at):
            logger.warning(
                "TensorFold model assets changed since process start; disk prefixes off"
            )
            return
        model_id = _identity(target, drafter, app)
        if not _assets_unchanged(target, drafter, app, started_at):
            logger.warning(
                "TensorFold model assets changed during hashing; disk prefixes off"
            )
            return
        directory = Path(
            os.environ.get(
                DIRECTORY_ENV,
                str(
                    Path.home() / ".cache" / "rapid-mlx" / "prefix_cache" / "tensorfold"
                ),
            )
        ).expanduser()

        # Live owner-thread schema, never a class/schema from disk.
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
        memory = scheduler.prompt_memory
        if memory is not None:
            projected = memory.projected

            def with_pending_copy(*args, **kwargs):
                return (
                    projected(*args, **kwargs)
                    + disk.reserved_bytes
                    + disk.read_reserved_bytes
                )

            memory.projected = with_pending_copy
        # Carry request provenance by object identity, once, on the submitting thread.
        from tensorfold.server.checkpoints import longest_common_prefix
        from tensorfold.server.text import render_prompt_ids

        original_boundary, original_submit = app.system_prefix_len, scheduler.submit
        request = threading.local()

        def boundary(self, messages, tools, prompt_ids, thinking=None):
            request.fact = None
            shared = original_boundary(messages, tools, prompt_ids, thinking=thinking)
            try:
                first = next(
                    i for i, m in enumerate(messages) if m.get("role") == "user"
                )
                if first and all(m.get("role") == "system" for m in messages[:first]):
                    with self.tokenizer_lock:
                        probes = [
                            render_prompt_ids(
                                self.tokenizer,
                                [*messages[:first], {"role": "user", "content": word}],
                                tools=tools,
                                enable_thinking=self.enable_thinking
                                if thinking is None
                                else thinking,
                                late_system=self.late_system,
                            )
                            for word in ("Alpha", "Beta")
                        ]
                    request.fact = (
                        prompt_ids,
                        min(
                            shared,
                            longest_common_prefix(prompt_ids, probes[0]),
                            longest_common_prefix(prompt_ids, probes[1]),
                        ),
                    )
            except Exception:
                pass  # Unknown message/template shape must not authorize disk I/O.
            return shared

        def submit_job(self, job):
            fact, request.fact = getattr(request, "fact", None), None
            job._rapid_disk_system_len = (
                fact[1] if fact is not None and fact[0] is job.prompt_ids else 0
            )
            return original_submit(job)

        app.system_prefix_len = types.MethodType(boundary, app)
        scheduler.submit = types.MethodType(submit_job, scheduler)
        original_open = scheduler._open_job
        reading_boundary = 0

        def open_job(self, job):
            nonlocal reading_boundary
            reading_boundary = getattr(job, "_rapid_disk_system_len", 0)
            try:
                return original_open(job)
            finally:
                reading_boundary = 0

        scheduler._open_job = types.MethodType(open_job, scheduler)
        original_keep = scheduler._keep_checkpoints
        context = None

        def keep(self, job, shared_at):
            nonlocal context
            shared = min(
                max(job.shared_prefix_lens, default=0),
                getattr(job, "_rapid_disk_system_len", 0),
            )
            eligible = [
                len(t)
                for t, _ in getattr(job.stream, "history_checkpoints", ())
                if len(t) in shared_at and len(t) < shared
            ]
            context = (job.prompt_ids, shared, max(eligible, default=0))
            try:
                return original_keep(job, shared_at)
            finally:
                context = None

        def persist(self, tokens, cache):
            if context is not None and len(tokens) == context[2]:
                disk.submit(
                    tokens,
                    cache,
                    prompt=context[0],
                    shared_prefix_len=context[1],
                    memory=self.prompt_memory,
                )

        scheduler._keep_checkpoints = types.MethodType(keep, scheduler)
        scheduler._persist = types.MethodType(persist, scheduler)
        original_close = app.close

        def close(self):
            disk.close()
            original_close()

        app.close = types.MethodType(close, app)
        scheduler._read_disk_block = types.MethodType(
            lambda self, prompt, usable=None: (
                disk.restore(self, prompt, usable, system_len=reading_boundary)
                if reading_boundary
                else None
            ),
            scheduler,
        )
        scheduler.snapshot_dir = directory
        app._rapid_prefix_disk = disk
    except Exception:
        logger.warning(
            "TensorFold disk prefixes unavailable; feature off", exc_info=True
        )
