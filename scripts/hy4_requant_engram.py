#!/usr/bin/env python3
"""Derive Engram8 from a finished v1 build without recalibrating GPTQ.

Inputs must remain immutable. Work resumes from the sibling <out>.tmp directory;
only a complete build is renamed to --out. Dry runs read headers/metadata only.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import struct
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.deepseek_v41_rd_common import file_hash
from scripts.hy4_gptq_adapter import Hy4Adapter, Recipe
from scripts.repack_deepseek_v41_native import _DTYPE_BYTES
from scripts.streamed_gptq_io import (
    TensorReader,
    artifact_lock,
    atomic_json,
    digest,
    mx_module,
)

RECIPE = "hy4-gptq-engram8"
ENGRAM = ".multi_head_embedding.embedding"
COPY_CHUNK = 8 * 1024 * 1024
# Engram tables are 96 wide, so group 64 cannot tile them.
GROUP = 32


def engram_module(name):
    module, _, suffix = name.rpartition(".")
    return (
        module
        if module.endswith(ENGRAM) and suffix in ("weight", "scales", "biases")
        else None
    )


def layout(entries):
    """Exact safetensors header and offsets, shared by planning and writing."""
    header, offset = {}, 0
    for name, (dtype, shape) in entries.items():
        size = math.prod(shape) * _DTYPE_BYTES[dtype]
        header[name] = dict(
            dtype=dtype, shape=shape, data_offsets=[offset, offset + size]
        )
        offset += size
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw, header, offset


def shard_plan(base, adapter):
    """Plan from headers only; retain base shard names and all indexed tensors."""
    base = Path(base)
    reader = TensorReader(base)
    index = json.loads((base / "model.safetensors.index.json").read_text())
    config = json.loads((base / "config.json").read_text())
    report = json.loads((base / "size-report.json").read_text())
    if report.get("recipe", {}).get("name") != "hy4-gptq-v1" or not report.get(
        "identity"
    ):
        raise ValueError("base must be a finished hy4-gptq-v1 build with an identity")
    modules = sorted({m for n in reader.index.tensors if (m := engram_module(n))})
    if not modules:
        raise ValueError("base has no Engram tables")
    replacements = {}
    for module in modules:
        name = module + ".weight"
        shape = adapter.shape(name)
        if len(shape) != 2 or shape[1] % GROUP:
            raise ValueError(f"invalid Engram source shape: {name}")
        if config["quantization"].get(module) != dict(
            bits=4, group_size=32, mode="affine"
        ):
            raise ValueError(f"base Engram quantization is not v1: {module}")
        for suffix, dtype, old_cols, new_cols in (
            ("weight", "U32", shape[1] // 8, shape[1] // 4),
            ("scales", "BF16", shape[1] // 32, shape[1] // GROUP),
            ("biases", "BF16", shape[1] // 32, shape[1] // GROUP),
        ):
            key = module + "." + suffix
            tensor = reader.index.tensors.get(key)
            if (
                tensor is None
                or tensor.dtype != dtype
                or tensor.shape != (shape[0], old_cols)
            ):
                raise ValueError(f"base/source Engram shape mismatch: {key}")
            replacements[key] = (dtype, (shape[0], new_cols))
    shards = []
    for filename in sorted(set(index["weight_map"].values())):
        names = sorted(n for n, f in index["weight_map"].items() if f == filename)
        rewrite = any(n in replacements for n in names)
        entries = {
            n: replacements.get(
                n, (reader.index.tensors[n].dtype, reader.index.tensors[n].shape)
            )
            for n in names
        }
        raw, _, payload = layout(entries)
        shards.append(
            dict(
                shard=filename,
                action="rewrite" if rewrite else "hardlink",
                base_bytes=(base / filename).stat().st_size,
                estimated_bytes=len(raw) + payload
                if rewrite
                else (base / filename).stat().st_size,
                engram_modules=sorted({m for n in names if (m := engram_module(n))}),
                entries=entries,
            )
        )
    plan = dict(
        recipe=RECIPE,
        base_identity=report["identity"],
        shards=shards,
        hardlink_bytes=sum(
            s["estimated_bytes"] for s in shards if s["action"] == "hardlink"
        ),
        rewrite_bytes=sum(
            s["estimated_bytes"] for s in shards if s["action"] == "rewrite"
        ),
        base_bytes=sum(s["base_bytes"] for s in shards),
        estimated_bytes=sum(s["estimated_bytes"] for s in shards),
    )
    return plan, reader, index, config, report


def rewrite_shard(path, shard, reader, adapter, chunk_rows=4096):
    """Copy arbitrary unchanged payloads verbatim; quantize bounded source rows."""
    mx = mx_module()
    raw, header, payload = layout(shard["entries"])
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w+b") as output:
        output.write(raw)
        output.truncate(len(raw) + payload)
        for name in shard["entries"]:
            if engram_module(name):
                continue
            tensor = reader.index.tensors[name]
            output.seek(len(raw) + header[name]["data_offsets"][0])
            with tensor.path.open("rb") as source:
                source.seek(tensor.offset)
                remaining = tensor.nbytes
                while remaining:
                    data = source.read(min(COPY_CHUNK, remaining))
                    if not data:
                        raise ValueError("truncated base tensor")
                    output.write(data)
                    remaining -= len(data)
        for module in shard["engram_modules"]:
            name = module + ".weight"
            rows, width = adapter.shape(name)
            # Bound both the number of rows and the dequantized input bytes.
            chunk = max(1, min(chunk_rows, (16 * 1024 * 1024) // (width * 2)))
            for start in range(0, rows, chunk):
                stop = min(rows, start + chunk)
                value = adapter.tensor(name, rows=(start, stop)).astype(mx.bfloat16)
                values = mx.quantize(value, bits=8, group_size=GROUP)
                mx.eval(*values)
                for suffix, value in zip(
                    ("weight", "scales", "biases"), values, strict=True
                ):
                    key = module + "." + suffix
                    # Affine components may live in different base shards.
                    if key not in header:
                        continue
                    dtype, shape = shard["entries"][key]
                    array = np.array(
                        value.view(mx.uint16) if dtype == "BF16" else value
                    )
                    expected = np.dtype("uint16" if dtype == "BF16" else "uint32")
                    if array.dtype != expected or array.shape != (
                        stop - start,
                        shape[1],
                    ):
                        raise ValueError(
                            f"quantized output shape/dtype mismatch: {key}"
                        )
                    output.seek(
                        len(raw)
                        + header[key]["data_offsets"][0]
                        + start * shape[1] * expected.itemsize
                    )
                    output.write(array.tobytes())
                    del array
                del value, values
                mx.clear_cache()
            gc.collect()
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def input_identity(root):
    """Bind metadata hashes and immutable shard stat identities without scanning weights."""
    root = Path(root).resolve()
    reader = TensorReader(root)
    return dict(
        path=str(root),
        metadata={
            n: file_hash(root / n)
            for n in ("config.json", "model.safetensors.index.json")
        },
        shards={
            str(p.relative_to(root)): [
                p.stat().st_dev,
                p.stat().st_ino,
                p.stat().st_size,
                p.stat().st_mtime_ns,
            ]
            for p in sorted({t.path for t in reader.index.tensors.values()})
        },
    )


def derive(base, source, out, recipe=RECIPE, dry_run=False):
    if recipe != RECIPE:
        raise ValueError("only hy4-gptq-engram8 can be derived")
    base, source, out = (Path(p).resolve() for p in (base, source, out))
    stage = out.with_name(out.name + ".tmp")
    for target in (out, stage):
        if any(
            target.is_relative_to(p) or p.is_relative_to(target) for p in (base, source)
        ):
            raise ValueError("output/staging and inputs must not overlap")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("refusing to overwrite non-empty output")
    adapter = Hy4Adapter(source, Recipe(recipe))
    plan, reader, index, config, base_report = shard_plan(base, adapter)
    if dry_run:
        print(
            json.dumps(
                {
                    **plan,
                    "shards": [
                        {k: v for k, v in s.items() if k != "entries"}
                        for s in plan["shards"]
                    ],
                },
                indent=2,
            )
        )
        return plan
    identity = digest(
        dict(
            base=input_identity(base),
            source=input_identity(source),
            report=base_report,
            recipe=recipe,
            tool={
                n: file_hash(Path(__file__).with_name(n))
                for n in (
                    "hy4_requant_engram.py",
                    "hy4_gptq_adapter.py",
                    "streamed_gptq_io.py",
                )
            },
        )
    )
    # Only matching staging work can be resumed; final non-empty outputs never are.
    if stage.exists() and any(stage.iterdir()) and not (stage / "resume.json").exists():
        raise ValueError("unrecognized staging directory")
    with artifact_lock(stage):
        marker = stage / "resume.json"
        if marker.exists():
            receipt = json.loads(marker.read_text())
            if receipt["identity"] != identity:
                raise ValueError("resume identity mismatch")
        else:
            receipt = dict(identity=identity, completed={})
            atomic_json(marker, receipt)
        for shard in plan["shards"]:
            filename = shard["shard"]
            target = stage / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if filename in receipt["completed"]:
                saved = receipt["completed"][filename]
                valid = (
                    target.exists() and os.path.samefile(base / filename, target)
                    if shard["action"] == "hardlink"
                    else target.exists() and file_hash(target) == saved
                )
                if not valid:
                    raise ValueError(f"resume artifact mismatch: {filename}")
                continue
            if shard["action"] == "hardlink":
                # An interruption can leave a link before its receipt is saved.
                if target.exists():
                    if not os.path.samefile(base / filename, target):
                        raise ValueError("unexpected staging shard")
                else:
                    os.link(base / filename, target)  # EXDEV is deliberately fatal.
            else:
                rewrite_shard(target, shard, reader, adapter)
            receipt["completed"][filename] = (
                "hardlink" if shard["action"] == "hardlink" else file_hash(target)
            )
            atomic_json(marker, receipt)
        for shard in plan["shards"]:
            for module in shard["engram_modules"]:
                config["quantization"][module] = dict(
                    bits=8, group_size=GROUP, mode="affine"
                )
        index.setdefault("metadata", {})["total_size"] = sum(
            math.prod(shape) * _DTYPE_BYTES[dtype]
            for s in plan["shards"]
            for dtype, shape in s["entries"].values()
        )
        atomic_json(stage / "model.safetensors.index.json", index)
        atomic_json(stage / "config.json", config)
        for pattern in (
            "tokenizer*",
            "chat_template*",
            "special_tokens_map.json",
            "added_tokens.json",
            "generation_config.json",
            "vocab.*",
            "merges.txt",
        ):
            for path in base.glob(pattern):
                if path.is_file():
                    temporary = stage / (path.name + ".tmp")
                    shutil.copyfile(path, temporary)
                    temporary.replace(stage / path.name)
        report = dict(
            recipe=asdict(adapter.recipe),
            base_identity=base_report["identity"],
            base_build=base_report,
            identity=identity,
            safetensors_bytes=plan["estimated_bytes"],
            hardlink_bytes=plan["hardlink_bytes"],
            rewrite_bytes=plan["rewrite_bytes"],
            tensor_count=len(index["weight_map"]),
        )
        atomic_json(stage / "size-report.json", report)
        if out.exists():
            out.rmdir()  # Refuses a directory populated concurrently.
        stage.rename(out)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "source", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--recipe", choices=(RECIPE,), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    derive(**vars(args))


if __name__ == "__main__":
    main()
