"""Bounded source inventory, native shard sink and restart state for GPTQ."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from scripts.deepseek_v41_gptq_fullrun import sync_parent
from scripts.deepseek_v41_rd_build import header
from scripts.deepseek_v41_rd_common import checkpoint_hashes, file_hash
from scripts.deepseek_v41_streaming_teacher import SourceWeights
from scripts.repack_deepseek_v41_native import (
    METADATA_FILES,
    CheckpointIndex,
    SourceTensor,
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)
    sync_parent(path)


MAX_HEADER_BYTES = 100_000_000


def read_header(path):
    """Read a safetensors header only after bounding its advertised allocation."""
    with Path(path).open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors header length")
        size = struct.unpack("<Q", prefix)[0]
        if size > MAX_HEADER_BYTES or size > os.fstat(stream.fileno()).st_size - 8:
            raise ValueError("invalid safetensors header length")
        raw = stream.read(size)
        if len(raw) != size:
            raise ValueError("truncated safetensors header")
        return prefix + raw


def write_all(stream, data):
    remaining = memoryview(data)
    while remaining:
        count = stream.write(remaining)
        if count is None or count <= 0:
            raise OSError("short shard write")
        remaining = remaining[count:]


def copy_verified(source, destination, expected_hash):
    """Publishable shards have independent inodes and verified, durable bytes."""
    shutil.copyfile(source, destination)
    if file_hash(destination) != expected_hash:
        raise ValueError("copied shard payload mismatch")
    with Path(destination).open("rb") as stream:
        os.fsync(stream.fileno())
    sync_parent(destination)


def metadata_hashes(source):
    return {
        name: file_hash(Path(source) / name)
        for name in sorted(METADATA_FILES)
        if (Path(source) / name).is_file()
    }


def copy_bound_metadata(source, destination):
    expected = source.manifest["metadata_sha256"]
    if metadata_hashes(source.source) != expected:
        raise ValueError("source metadata identity mismatch")
    for name, checksum in expected.items():
        copy_verified(source.source / name, destination / name, checksum)


def sync_files(paths):
    for path in paths:
        with Path(path).open("rb") as stream:
            os.fsync(stream.fileno())
        sync_parent(path)


def inventory(source, source_revision=None):
    """Run on the source host once; subsequent commands need only this JSON."""
    source = Path(source)
    before = checkpoint_hashes(source)
    metadata = metadata_hashes(source)
    index_json = (source / "model.safetensors.index.json").read_bytes().decode("utf-8")
    index = CheckpointIndex(source, header_reader=read_header)
    shards = {}
    for t in index.tensors.values():
        name = str(t.path.relative_to(source))
        if name not in shards:
            shards[name] = dict(
                sha256=before[name],
                size=t.path.stat().st_size,
                header_sha256=hashlib.sha256(read_header(t.path)).hexdigest(),
            )
    tensors = {
        n: {**asdict(t), "path": str(t.path.relative_to(source))}
        for n, t in index.tensors.items()
    }
    if checkpoint_hashes(source) != before or metadata_hashes(source) != metadata:
        raise ValueError("source changed during inventory")
    return dict(
        checkpoint_sha256=before,
        metadata_sha256=metadata,
        checkpoint_index_json=index_json,
        config=json.loads((source / "config.json").read_text()),
        tensors=tensors,
        shards=shards,
        source_revision=source_revision or source.resolve().name,
    )


def required_shards(manifest, component):
    """Exact filenames from the source index, including weight scale tensors."""

    def wanted(name):
        if isinstance(component, int):
            return name.startswith(f"layers.{component}.")
        if component == "top":
            return not name.startswith(
                ("layers.", "mtp.", "vision.", "aligner.", "image_")
            )
        if component == "mtp":
            return name.startswith("mtp.")
        raise ValueError("component must be a layer ID, top or mtp")

    paths = (
        {n: t["path"] for n, t in manifest["tensors"].items()}
        if "tensors" in manifest
        else manifest["weight_map"]
    )
    selected = {path for name, path in paths.items() if wanted(name)}
    if any(
        not isinstance(path, str)
        or not path
        or Path(path).is_absolute()
        or ".." in Path(path).parts
        for path in selected
    ):
        raise ValueError("unsafe source shard name")
    return sorted(selected)


class PartialSource(SourceWeights):
    """Keep absent-shard descriptors; open only the current stage's source files."""

    def __init__(self, source, manifest):
        from rapid_mlx.models.deepseek_v41_native.config import ModelArgs
        from scripts.evaluate_deepseek_v41_rd import validate_checkpoint_identity

        validate_checkpoint_identity(
            manifest.get("checkpoint_sha256"), manifest.get("checkpoint_index_json")
        )
        self.source = Path(source)
        self.manifest = manifest
        self.config = manifest["config"]
        self.args = ModelArgs.from_dict(self.config)
        if self.config.get("quantization") or self.args.experts_per_layer:
            raise ValueError("GPTQ requires an unpruned FP4/FP8 source")
        if self.config.get("quantization_config", {}).get(
            "expert_dtype", "fp4"
        ) not in ("fp4", "fp8"):
            raise ValueError("unsupported source expert type")
        tensors = {}
        for name, value in manifest["tensors"].items():
            path = Path(value["path"])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("unsafe source shard name")
            tensors[name] = SourceTensor(
                **{**value, "shape": tuple(value["shape"]), "path": self.source / path}
            )
        self.index = SimpleNamespace(tensors=tensors)

    def tensor(self, name):
        if ".ffn.experts." in name:
            from rapid_mlx.models.deepseek_v41_native.dequant import (
                dequant_fp4,
                dequant_fp8,
            )
            from scripts.deepseek_v41_expert_quant_error import read_tensor

            w = read_tensor(self.index.tensors[name])
            scale = self.index.tensors.get(name.removesuffix(".weight") + ".scale")
            if scale is not None:
                dtype = self.config.get("quantization_config", {}).get(
                    "expert_dtype", "fp4"
                )
                dequant = dequant_fp4 if dtype == "fp4" else dequant_fp8
                w = dequant(w, read_tensor(scale), mx.bfloat16)
            return w.astype(mx.bfloat16)
        return super().tensor(name)

    def verify(self, component):
        for name in required_shards(self.manifest, component):
            path = self.source / name
            meta = self.manifest["shards"][name]
            if path.stat().st_size != meta["size"]:
                raise ValueError(f"source shard size mismatch: {name}")
            sha = hashlib.sha256(read_header(path)).hexdigest()
            if sha != meta["header_sha256"]:
                raise ValueError(f"source shard header mismatch: {name}")
            expected = self.manifest["checkpoint_sha256"].get(name)
            if meta.get("sha256") != expected or file_hash(path) != expected:
                raise ValueError(f"source shard payload mismatch: {name}")


class ShardSink:
    """Random range writes into a preplanned native shard, one expert at a time."""

    def __init__(self, path, entries):
        self.path = Path(path)
        self.temporary = self.path.with_suffix(".partial")
        self.entries = {e.name: e for e in entries}
        self.locations = {}
        raw = header(entries)
        self.file = self.temporary.open("w+b")
        write_all(self.file, struct.pack("<Q", len(raw)))
        write_all(self.file, raw)
        offset = 8 + len(raw)
        for e in entries:
            self.locations[e.name] = offset
            offset += e.nbytes
        self.file.truncate(offset)
        self.written = {}

    def put(self, name, tensor, expert=None):
        entry = self.entries[name]
        if expert is not None and (
            type(expert) is not int or not 0 <= expert < entry.shape[0]
        ):
            raise ValueError("invalid expert output range")
        key = (name, expert)
        if (
            key in self.written
            or (expert is not None and (name, None) in self.written)
            or (expert is None and any(n == name for n, _ in self.written))
        ):
            raise ValueError("duplicate output range")
        expected = entry.nbytes if expert is None else entry.nbytes // entry.shape[0]
        mx.eval(tensor)
        raw = np.asarray(
            tensor.view(mx.uint16) if tensor.dtype == mx.bfloat16 else tensor
        )
        if raw.nbytes != expected:
            raise ValueError(f"wrong output range size: {name}")
        expected_shape = entry.shape if expert is None else entry.shape[1:]
        if tuple(tensor.shape) != tuple(expected_shape):
            raise ValueError(f"wrong output range shape: {name}")
        dtype = {"BF16": mx.bfloat16, "F32": mx.float32, "U32": mx.uint32}[entry.dtype]
        if tensor.dtype != dtype:
            raise ValueError(f"wrong native output dtype: {name}")
        offset = self.locations[name] + (0 if expert is None else expert * expected)
        self.file.seek(offset)
        write_all(self.file, raw.tobytes())
        self.written[key] = True

    def finish(self):
        for name, e in self.entries.items():
            if (name, None) in self.written:
                continue
            if not all((name, i) in self.written for i in range(e.shape[0])):
                raise ValueError(f"incomplete tensor: {name}")
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()
        self.temporary.replace(self.path)
        sync_parent(self.path)

    def close(self):
        if not self.file.closed:
            self.file.close()


def save_state(path, state):
    """Safe typed graph serialization, preserving shared cache aliases and BF16.

    Only one window is materialized at once. Safetensors is mmap-readable;
    a JSON tree describes scalars, array references and whitelisted native state.
    """
    arrays, seen, array_keys = {}, {}, {}

    def encode(obj):
        if isinstance(obj, mx.array):
            if id(obj) not in array_keys:
                array_keys[id(obj)] = f"a{len(arrays)}"
                arrays[array_keys[id(obj)]] = obj
            return {"array": array_keys[id(obj)]}
        if obj is None or isinstance(obj, (str, bool, int, float)):
            return obj
        if isinstance(obj, mx.Dtype):
            return {"dtype": str(obj).split(".")[-1]}
        if isinstance(obj, dict):
            return {"dict": {k: encode(v) for k, v in obj.items()}}
        if isinstance(obj, (list, tuple)):
            return {"list": [encode(v) for v in obj]}
        if id(obj) in seen:
            return {"ref": seen[id(obj)]}
        key = str(len(seen))
        seen[id(obj)] = key
        return {
            "object": type(obj).__name__,
            "id": key,
            "fields": {k: encode(v) for k, v in vars(obj).items()},
        }

    tree = encode(state)
    mx.eval(list(arrays.values()))
    path = Path(path)
    temporary = path.with_suffix(".partial.safetensors")
    mx.save_safetensors(str(temporary), arrays)
    with temporary.open("rb") as f:
        os.fsync(f.fileno())
    sha = file_hash(temporary)
    payload = path.with_name(f"{path.name}.{sha}.safetensors")
    temporary.replace(payload)
    sync_parent(payload)
    atomic_json(path.with_suffix(".json"), dict(tree=tree, tensor_sha256=sha))
    prune_state_payloads(path, payload)


def prune_state_payloads(path, committed):
    """Reclaim unreferenced generations under the conversion's single-writer lock."""
    path = Path(path)
    removed = False
    for candidate in path.parent.glob(f"{path.name}.*.safetensors"):
        generation = candidate.name[len(path.name) + 1 : -len(".safetensors")]
        if (
            candidate != committed
            and len(generation) == 64
            and all(c in "0123456789abcdef" for c in generation)
        ):
            candidate.unlink()
            removed = True
    if removed:
        sync_parent(committed)


def state_snapshot(path):
    """Resolve one committed graph/payload generation, rejecting corruption."""
    path = Path(path)
    record = json.loads(path.with_suffix(".json").read_text())
    sha = record.get("tensor_sha256")
    if (
        not isinstance(sha, str)
        or len(sha) != 64
        or any(c not in "0123456789abcdef" for c in sha)
    ):
        raise ValueError("invalid state payload identity; regenerate legacy state")
    payload = path.with_name(f"{path.name}.{sha}.safetensors")
    if file_hash(payload) != sha:
        raise ValueError("state payload identity mismatch")
    prune_state_payloads(path, payload)
    return record["tree"], payload


def load_state(path):
    from rapid_mlx.models.deepseek_v41_native.cache import LayerCache
    from rapid_mlx.models.deepseek_v41_native.compressor import CompressorState
    from rapid_mlx.models.deepseek_v41_native.model import SharedState

    path = Path(path)
    tree, payload = state_snapshot(path)
    arrays = mx.load(str(payload))
    objects = {}
    classes = {
        c.__name__: c
        for c in (LayerCache, CompressorState, SharedState, SimpleNamespace)
    }

    def decode(obj):
        if not isinstance(obj, dict):
            return obj
        if "array" in obj:
            return arrays[obj["array"]]
        if "dtype" in obj:
            return getattr(mx, obj["dtype"])
        if "ref" in obj:
            return objects[obj["ref"]]
        if "dict" in obj:
            return {k: decode(v) for k, v in obj["dict"].items()}
        if "list" in obj:
            return [decode(v) for v in obj["list"]]
        cls = classes[obj["object"]]
        value = cls.__new__(cls)
        objects[obj["id"]] = value
        value.__dict__.update({k: decode(v) for k, v in obj["fields"].items()})
        return value

    return decode(tree)


def entries_json(entries):
    return [asdict(e) for e in entries]


def entries_from_json(values):
    from scripts.deepseek_v41_rd_build import Entry

    return [
        Entry(
            **{
                **v,
                "shape": tuple(v["shape"]),
                "sources": tuple(v["sources"]),
                "rows": None if v["rows"] is None else tuple(v["rows"]),
            }
        )
        for v in values
    ]


def verify_artifact(directory, record):
    root = Path(directory).resolve()
    for name, sha in record["files"].items():
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in (".", "..")
            or "\\" in name
        ):
            raise ValueError("unsafe checkpoint artifact name")
        path = root / name
        if not path.resolve().is_relative_to(root):
            raise ValueError("checkpoint artifact escapes parts directory")
        if file_hash(path) != sha:
            raise ValueError(f"checkpoint artifact changed: {name}")


def conversion_record(source, destination):
    """Bind an assembled candidate to the inventoried source checkpoint."""
    from scripts.evaluate_deepseek_v41_rd import validate_checkpoint_identity

    manifest = source.manifest
    original = validate_checkpoint_identity(
        manifest["checkpoint_sha256"], manifest["checkpoint_index_json"]
    )
    candidate = checkpoint_hashes(destination)
    index = (destination / "model.safetensors.index.json").read_bytes().decode("utf-8")
    validate_checkpoint_identity(candidate, index)
    return dict(
        source_checkpoint_sha256=original,
        source_index_json=manifest["checkpoint_index_json"],
        checkpoint_sha256=candidate,
        checkpoint_index_json=index,
    )


def rename_directory_exclusive(source, destination):
    """Atomically publish without replacing even an empty competing destination."""
    libc = ctypes.CDLL(None, use_errno=True)
    paths = (
        os.fsencode(Path(source).absolute()),
        os.fsencode(Path(destination).absolute()),
    )
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        args = (*paths, 4)  # RENAME_EXCL, from sys/stdio.h.
    elif sys.platform == "linux" and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        args = (
            0,
            paths[0],
            0,
            paths[1],
            1,
        )  # RENAME_NOREPLACE; absolute paths ignore dirfds.
    else:
        raise RuntimeError("atomic exclusive directory publication is unavailable")
    rename.restype = ctypes.c_int
    if rename(*args) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


@contextmanager
def cleanup_after_exit(root):
    """A separate process reclaims only this private root, even after _exit.

    EOF is the lifetime signal; no polling or PID reuse is involved. The helper
    has its own session so terminating a conversion process group still cleans
    its unpublished assembly. Committed/restart artifacts are outside this root.
    """
    code = (
        "import shutil,sys; print('ready',flush=True); "
        "sys.stdin.buffer.read(); shutil.rmtree(sys.argv[1],ignore_errors=True)"
    )
    with subprocess.Popen(
        [sys.executable, "-I", "-S", "-c", code, str(root)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    ) as cleanup:
        if cleanup.stdout.readline() != b"ready\n":
            raise RuntimeError("assembly cleanup helper failed to start")
        try:
            yield
        finally:
            cleanup.stdin.close()


@contextmanager
def atomic_destination(destination):
    """Keep incomplete assemblies private and remove them after a failed attempt."""
    destination = Path(destination)
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        tempfile.TemporaryDirectory(
            dir=destination.parent, prefix=f".{destination.name}.partial-"
        ) as root,
        cleanup_after_exit(root),
    ):
        stage = Path(root) / "output"
        yield stage
        # Include reports and other metadata copied outside the shard writer.
        # File data and each containing directory must precede publication.
        sync_files(path for path in stage.rglob("*") if path.is_file())
        rename_directory_exclusive(stage, destination)
        sync_parent(destination)
