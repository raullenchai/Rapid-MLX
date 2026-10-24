#!/usr/bin/env python3
"""Header-only allocation planner and bounded FP4/FP8 -> native affine packer.

No whole-shard load. A projection/expert or Engram row chunk is the largest
source read. The dry run imports no MLX and never creates a destination.

--reuse-from copies unchanged shards after source identity, semantic plan and
payload digest checks. Inputs must remain immutable during a build.
Reuse provenance lives in reuse-provenance.json to preserve fresh config/index bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.deepseek_v41_rd_common import (
    checkpoint_hashes,
    file_hash,
    require_all_moe_layers,
)
from scripts.repack_deepseek_v41_native import CheckpointIndex, _copy_metadata_files

EXPERT = re.compile(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.weight$")
ROUTER = re.compile(r"layers\.(\d+)\.ffn\.gate\.(weight|bias|bias_vl)$")
PROJECTIONS = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
DROP = ("mtp.", "vision.", "aligner.", "image_")


def fixed_precision(name, shape):
    """Owner policy; quantize only runtime Linear/MultiLinear modules."""
    leaf = name.removesuffix(".weight").rsplit(".", 1)[-1]
    eligible = leaf in {
        "wq_a",
        "wq_b",
        "wkv",
        "wo_a",
        "wo_b",
        "w1",
        "w2",
        "w3",
        "wk",
        "wgate",
        "weights_proj",
        "key_proj",
        "value_proj",
        "main_proj",
        "proj",
    }
    return (
        8
        if name.endswith(".weight")
        and len(shape) == 2
        and eligible
        and shape[-1] % 64 == 0
        else None
    )


@dataclass(frozen=True)
class Entry:
    name: str
    shape: tuple
    dtype: str
    sources: tuple[str, ...]
    component: str
    bits: int | None = None
    group_size: int = 64
    suffix: str = "weight"
    rows: tuple[int, ...] | None = None

    @property
    def nbytes(self):
        return math.prod(self.shape) * {"BF16": 2, "F32": 4, "U32": 4}[self.dtype]


def output_entries(
    base, shape, sources, component, bits=None, group_size=64, rows=None, fp32=False
):
    if bits is None:
        return [
            Entry(
                base,
                tuple(shape),
                "F32" if fp32 else "BF16",
                sources,
                component,
                rows=rows,
            )
        ]
    if shape[-1] % group_size or shape[-1] * bits % 32:
        raise ValueError(f"nonintegral affine packing: {base} {shape}")
    base = base.removesuffix(".weight")
    return [
        Entry(
            base + "." + suffix,
            (*shape[:-1], cols),
            dtype,
            sources,
            component,
            bits,
            group_size,
            suffix,
            rows,
        )
        for suffix, cols, dtype in [
            ("weight", shape[-1] * bits // 32, "U32"),
            ("scales", shape[-1] // group_size, "BF16"),
            ("biases", shape[-1] // group_size, "BF16"),
        ]
    ]


def validate_allocation(allocation, config):
    mtp_bits = allocation.get("mtp_expert_bits", 4)
    if type(mtp_bits) is not int or mtp_bits not in (4, 8):
        raise ValueError("mtp_expert_bits must be 4 or 8")
    c = config.get("text_config", config)
    count = c.get("num_hidden_layers", c.get("n_layers"))
    experts = c["n_routed_experts"]
    topk = c.get("num_experts_per_tok", c.get("n_activated_experts", 6))
    if set(allocation["layers"]) != {str(i) for i in range(count)}:
        raise ValueError("allocation must cover every layer")
    for layer, spec in allocation["layers"].items():
        ids = spec["kept_expert_ids"]
        if (
            len(ids) < topk
            or ids != sorted(set(ids))
            or any(type(e) is not int or not 0 <= e < experts for e in ids)
        ):
            raise ValueError(f"invalid kept expert IDs in layer {layer}")
        bits = spec.get(
            "projection_bits", {p: spec.get("bits") for p in PROJECTIONS.values()}
        )
        if set(bits) != set(PROJECTIONS.values()) or any(
            type(b) is not int or b not in (2, 3, 4, 16) for b in bits.values()
        ):
            raise ValueError(
                "layer/projection bits must be scalar 2/3/4 (16 is reference-only); per-expert bits unsupported"
            )
        if allocation.get("reference") and (
            ids != list(range(experts)) or any(b != 16 for b in bits.values())
        ):
            raise ValueError("reference must keep all experts at BF16")
        spec["projection_bits"] = bits
    return c


def mtp_entries(index, config, expert_bits=4):
    """Draft-only affine policy; dense Markov I/O and norms remain BF16."""
    if type(expert_bits) is not int or expert_bits not in (4, 8):
        raise ValueError("mtp_expert_bits must be 4 or 8")
    entries = []
    for name, t in sorted(index.tensors.items()):
        if not name.startswith("mtp.") or name.endswith(".scale"):
            continue
        shape = list(t.shape)
        expert = ".ffn.experts." in name and name.endswith(".weight")
        scale = name.removesuffix(".weight") + ".scale" in index.tensors
        if (
            expert
            and scale
            and config.get("quantization_config", {}).get("expert_dtype", "fp4")
            == "fp4"
        ):
            shape[-1] *= 2
        bits = expert_bits if expert else fixed_precision(name, shape)
        entries.extend(
            output_entries(
                name,
                shape,
                (name,),
                "mtp",
                bits,
                fp32=bool(re.search(r"gate\.bias(_vl)?$", name)),
            )
        )
    return entries


def mtp_byte_costs(index, config):
    """Exact resident bytes, including unchanged draft non-expert policy."""
    return {
        str(bits): sum(e.nbytes for e in mtp_entries(index, config, bits))
        for bits in (4, 8)
    }


def plan(index, config, allocation):
    validate_allocation(allocation, config)
    entries, experts = [], {}
    layers_with_experts = set()
    for name, t in sorted(index.tensors.items()):
        if name.startswith("mtp."):
            continue
        if name.startswith(DROP) or name.endswith(".scale"):
            continue
        shape = list(t.shape)
        match = EXPERT.fullmatch(name)
        if match:
            layer, expert, proj = match.groups()
            layers_with_experts.add(int(layer))
            if int(expert) not in allocation["layers"][layer]["kept_expert_ids"]:
                continue
            experts.setdefault((layer, proj), {})[int(expert)] = name
            continue
        if ".engram.embed." in name:
            entries.extend(output_entries(name, shape, (name,), "engram_ssd", 4, 32))
            continue
        router = ROUTER.fullmatch(name)
        rows = None
        component = "fixed_bf16"
        fp32 = False
        bits = None if allocation.get("reference") else fixed_precision(name, shape)
        # Non-reference builds store source-F32 non-router tensors (hc_*, attn_sink)
        # as BF16 to match the released artifact format; reference builds keep F32.
        # Router biases stay F32 below.
        fp32 = bool(allocation.get("reference") and t.dtype == "F32")
        if router:
            rows = tuple(allocation["layers"][router[1]]["kept_expert_ids"])
            shape[0] = len(rows)
            component = "router"
            fp32 = router[2] != "weight"
            bits = None
        elif bits:
            component = (
                "shared_experts_8bit"
                if ".shared_experts." in name
                else "indexer_compressor_8bit"
                if ".indexer." in name or ".compressor." in name
                else "attention_other_8bit"
            )
        entries.extend(
            output_entries(name, shape, (name,), component, bits, rows=rows, fp32=fp32)
        )
    require_all_moe_layers(config, layers_with_experts)
    if allocation.get("keep_mtp", False):
        entries.extend(mtp_entries(index, config, allocation.get("mtp_expert_bits", 4)))
    for layer, spec in allocation["layers"].items():
        for proj, final in PROJECTIONS.items():
            parts = experts.get((layer, proj), {})
            ids = spec["kept_expert_ids"]
            if set(parts) != set(ids):
                raise ValueError(f"missing kept experts: layer {layer} {proj}")
            names = tuple(parts[e] for e in ids)
            shapes = [index.tensors[n].shape for n in names]
            if len(set(shapes)) != 1:
                raise ValueError("inconsistent expert shapes")
            shape = list(shapes[0])
            if (
                config.get("quantization_config", {}).get("expert_dtype", "fp4")
                == "fp4"
                and names[0].removesuffix(".weight") + ".scale" in index.tensors
            ):
                shape[-1] *= 2
            bits = spec["projection_bits"][final]
            entries.extend(
                output_entries(
                    f"layers.{layer}.ffn.experts.{final}.weight",
                    [len(ids), *shape],
                    names,
                    "routed_experts",
                    None if bits == 16 else bits,
                )
            )
    if not entries:
        raise ValueError("empty checkpoint")
    modules = {
        e.name.rsplit(".", 1)[0]: {"bits": e.bits, "group_size": e.group_size}
        for e in entries
        if e.bits and e.suffix == "weight" and e.component != "engram_ssd"
    }
    totals = {}
    for e in entries:
        totals[e.component] = totals.get(e.component, 0) + e.nbytes
    resident = sum(v for k, v in totals.items() if k != "engram_ssd")
    return (
        entries,
        modules,
        {
            "components": totals,
            "resident_tensor_bytes": resident,
            "shared_engram_bytes": totals.get("engram_ssd", 0),
            "total_payload_bytes": sum(totals.values()),
            "dropped_mtp_source_bytes": 0
            if allocation.get("keep_mtp")
            else sum(
                t.nbytes for n, t in index.tensors.items() if n.startswith("mtp.")
            ),
            "source_payload_bytes": sum(t.nbytes for t in index.tensors.values()),
            "container_policy": "exact tensor payload; safetensors headers reported separately by shard layout",
        },
    )


def header(entries):
    off, meta = 0, {"__metadata__": {"format": "mlx"}}
    for e in entries:
        meta[e.name] = {
            "dtype": e.dtype,
            "shape": list(e.shape),
            "data_offsets": [off, off + e.nbytes],
        }
        off += e.nbytes
    raw = json.dumps(meta, separators=(",", ":")).encode()
    return raw + b" " * (-len(raw) % 8)


def groups_for(entries):
    groups = {}
    for e in entries:
        layer = re.match(r"layers\.(\d+)\.", e.name)
        group = (
            "mtp"
            if e.component == "mtp"
            # Weight and affine tensors split so each file fits hub size limits.
            else f"engram-{layer[1]}-{'weight' if e.name.endswith('.weight') else 'affine'}"
            if e.component == "engram_ssd"
            else "layer-" + layer[1]
            if layer
            else "top"
        )
        groups.setdefault(group, []).append(e)
    return groups


def layout_report(entries, report):
    groups = groups_for(entries)
    report = dict(report)
    report["safetensors_container_bytes"] = sum(
        8 + len(header(es)) for es in groups.values()
    )
    report["total_safetensors_file_bytes"] = (
        report["total_payload_bytes"] + report["safetensors_container_bytes"]
    )
    report["largest_resident_shard_payload"] = max(
        sum(e.nbytes for e in es)
        for g, es in groups.items()
        if not g.startswith("engram-")
    )
    return report


def read_rows(t, start, stop):
    from scripts.deepseek_v41_expert_quant_error import read_tensor

    return read_tensor(
        replace(
            t,
            offset=t.offset + start * t.row_bytes,
            nbytes=(stop - start) * t.row_bytes,
            shape=(stop - start, *t.shape[1:]),
        )
    )


def transformed(index, config, e, source_name, start=None, stop=None):
    import mlx.core as mx

    from rapid_mlx.models.deepseek_v41_native.dequant import (
        dequant_fp4,
        dequant_fp8,
        dequant_fp8_rows,
    )
    from scripts.deepseek_v41_expert_quant_error import read_tensor

    t = index.tensors[source_name]
    read = (
        (lambda t: read_tensor(t))
        if start is None
        else (lambda t: read_rows(t, start, stop))
    )
    w = read(t)
    scale = (
        index.tensors.get(source_name.removesuffix(".weight") + ".scale")
        if source_name.endswith(".weight")
        else None
    )
    if scale is not None:
        if e.component == "engram_ssd":
            w = dequant_fp8_rows(w, read(scale)).astype(mx.bfloat16)
        elif (
            EXPERT.fullmatch(source_name) or ".ffn.experts." in source_name
        ) and config.get("quantization_config", {}).get("expert_dtype", "fp4") == "fp4":
            w = dequant_fp4(w, read_tensor(scale), mx.bfloat16)
        else:
            w = dequant_fp8(w, read_tensor(scale), mx.bfloat16)
    w = w.astype(mx.float32 if e.dtype == "F32" else mx.bfloat16)
    if e.rows is not None:
        w = w[mx.array(e.rows)]
    if e.bits:
        triple = mx.quantize(w, group_size=e.group_size, bits=e.bits)
        w = triple[("weight", "scales", "biases").index(e.suffix)]
    mx.eval(w)
    import numpy as np

    return np.array(w.view(mx.uint16) if w.dtype == mx.bfloat16 else w).tobytes()


def write_group(
    path, entries, index, config, row_chunk=65536, *, publish_exclusive=False
):
    """Append output directly; repeated suffix quantization trades CPU time for bounded RAM."""
    import mlx.core as mx

    raw = header(entries)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".partial-")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(struct.pack("<Q", len(raw)))
            f.write(raw)
            for e in entries:
                before = f.tell()
                for source in e.sources:
                    if e.component == "engram_ssd":
                        rows = index.tensors[source].shape[0]
                        for start in range(0, rows, row_chunk):
                            f.write(
                                transformed(
                                    index,
                                    config,
                                    e,
                                    source,
                                    start,
                                    min(rows, start + row_chunk),
                                )
                            )
                            mx.clear_cache()
                    else:
                        f.write(transformed(index, config, e, source))
                        mx.clear_cache()
                if f.tell() - before != e.nbytes:
                    raise ValueError(f"output size mismatch: {e.name}")
            f.flush()
            os.fsync(f.fileno())
        if publish_exclusive:
            try:
                os.link(temporary, path)
            except FileExistsError:
                validate_shared_group(path, entries)
                if file_hash(path) != file_hash(temporary):
                    raise ValueError("shared Engram payload mismatch")
            temporary.unlink()
        else:
            temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def validate_shared_group(path, entries):
    """Validate a previously published immutable shared Engram group."""
    raw = header(entries)
    with path.open("rb") as f:
        if os.fstat(f.fileno()).st_size != 8 + len(raw) + sum(
            e.nbytes for e in entries
        ):
            raise ValueError("shared Engram file size mismatch")
        if f.read(8) != struct.pack("<Q", len(raw)) or f.read(len(raw)) != raw:
            raise ValueError("shared Engram header mismatch")


def matching_file(path, entries):
    """Compare exact serialized header and size without reading tensor payloads."""
    raw = header(entries)
    if not path.is_file() or path.stat().st_size != 8 + len(raw) + sum(
        e.nbytes for e in entries
    ):
        return False
    with path.open("rb") as f:
        return f.read(8) == struct.pack("<Q", len(raw)) and f.read(len(raw)) == raw


def group_path(group):
    filename = f"model-{group}.safetensors"
    return Path("dspark") / filename if group == "mtp" else Path(filename)


def existing_parent(path):
    while not path.exists():
        path = path.parent
    return path


def reuse_plan(reuse_from, destination, index, config, identity, groups):
    """Check semantic plans as well as headers (same-shaped expert swaps differ)."""
    config_bytes = (reuse_from / "config.json").read_bytes()
    previous = json.loads(config_bytes)["rapid_quantization"]
    if previous["shared_engram_identity"] != identity:
        raise ValueError("reuse build source identity mismatch")
    old_allocation = previous["allocation"]
    result = {
        "from": str(reuse_from),
        "allocation_sha256": hashlib.sha256(
            json.dumps(old_allocation, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "reused": [],
        "rewritten": [],
    }
    old_entries, _, _ = plan(index, config, old_allocation)
    old_groups = groups_for(old_entries)
    for group, es in groups.items():
        path = reuse_from / group_path(group)
        # Entry equality covers sources, kept router rows, precision and policy.
        match = old_groups.get(group) == es and matching_file(path, es)
        if match and file_hash(path) != previous.get("payload_sha256", {}).get(
            str(group_path(group))
        ):
            raise ValueError(
                "reuse payload digest mismatch; rebuild legacy or changed inputs"
            )
        result["reused" if match else "rewritten"].append(str(group_path(group)))
    return result


def build(source, destination, shared, allocation, dry_run=False, reuse_from=None):
    if allocation.get("keep_mtp"):
        allocation = {"mtp_expert_bits": 4, **allocation}
    config = json.loads((source / "config.json").read_text())
    index = CheckpointIndex(source)
    entries, modules, report = plan(index, config, allocation)
    report = layout_report(entries, report)
    if (
        allocation.get("budget_bytes") is not None
        and report["resident_tensor_bytes"] > allocation["budget_bytes"]
    ):
        raise ValueError("allocation exceeds resident byte target")
    # Content binding distinguishes equal-layout checkpoints with different weights.

    identity = hashlib.sha256(
        json.dumps(
            {
                "config": config,
                "checkpoint_sha256": checkpoint_hashes(source),
                "tensors": [
                    (n, t.shape, t.dtype, t.nbytes, t.path.name, t.offset)
                    for n, t in sorted(index.tensors.items())
                ],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    groups = groups_for(entries)
    reuse = (
        reuse_plan(reuse_from, destination, index, config, identity, groups)
        if reuse_from is not None
        else None
    )
    if dry_run:
        return {**report, **({"reuse": reuse} if reuse is not None else {})}
    if destination.exists():
        raise FileExistsError(destination)
    shared_path = shared / identity
    shared_path.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(dir=destination.parent, prefix=destination.name + ".partial-")
    )
    try:
        mapping = {}
        mtp_mapping = {}
        for group, es in groups.items():
            filename = f"model-{group}.safetensors"
            target = staging / filename
            if group == "mtp":
                (staging / "dspark").mkdir()
                target = staging / "dspark" / filename
            if reuse is not None and str(group_path(group)) in reuse["reused"]:
                # Verify the independent copy as well as the reuse-plan input.
                original = reuse_from / group_path(group)
                if not matching_file(original, es):
                    raise ValueError(f"reuse file changed during build: {original}")
                shutil.copyfile(original, target)
                expected = json.loads((reuse_from / "config.json").read_text())[
                    "rapid_quantization"
                ]["payload_sha256"][str(group_path(group))]
                if file_hash(target) != expected:
                    raise ValueError("reuse payload changed during copy")
            elif group.startswith("engram-"):
                common = shared_path / filename
                write_group(common, es, index, config, publish_exclusive=True)
                expected = file_hash(common)
                shutil.copyfile(common, target)
                if file_hash(target) != expected:
                    raise ValueError("shared Engram payload changed during copy")
            else:
                write_group(target, es, index, config)
            (mtp_mapping if group == "mtp" else mapping).update(
                {e.name: filename for e in es}
            )
        config["model_type"] = "deepseek_v41"
        counts = [
            len(allocation["layers"][str(i)]["kept_expert_ids"])
            for i in range(len(allocation["layers"]))
        ]
        config["experts_per_layer"] = counts
        if "text_config" in config:
            config["text_config"]["experts_per_layer"] = counts
        mtp_modules = {n: v for n, v in modules.items() if n.startswith("mtp.")}
        modules = {n: v for n, v in modules.items() if not n.startswith("mtp.")}
        config["quantization"] = {
            "bits": 8,
            "group_size": 64,
            "modules": modules,
            "engram_bits": 4,
            "engram_group_size": 32,
        }
        if mtp_mapping:
            sidecar_config = {**config, **config.get("text_config", {})}
            sidecar_config["quantization"] = {
                "bits": 8,
                "group_size": 64,
                "modules": mtp_modules,
            }
            (staging / "dspark" / "config.json").write_text(
                json.dumps(sidecar_config, indent=2) + "\n"
            )
            (staging / "dspark" / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": mtp_mapping}, indent=2) + "\n"
            )
        config["rapid_quantization"] = {
            "format": "native-packed-affine",
            "allocation": allocation,
            "shared_engram_identity": identity,
            "payload_sha256": {
                str(group_path(g)): file_hash(staging / group_path(g)) for g in groups
            },
            "mtp": bool(mtp_mapping),
            "vision": False,
            "bytes": report,
        }
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (staging / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "total_size": report["total_payload_bytes"]
                        - report["components"].get("mtp", 0)
                    },
                    "weight_map": mapping,
                },
                indent=2,
            )
            + "\n"
        )
        _copy_metadata_files(source, staging)
        (staging / "README.md").write_text(
            "# Experimental pruned native build\n\nUnqualified. Vision omitted; see config for optional DSpark sidecar. Engram files are independent copies of verified shared payloads.\n"
        )
        if reuse is not None:
            (staging / "reuse-provenance.json").write_text(
                json.dumps(reuse, indent=2) + "\n"
            )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {**report, **({"reuse": reuse} if reuse is not None else {})}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--allocation", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--shared-engram", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--reuse-from", type=Path)
    a = p.parse_args()
    print(
        json.dumps(
            build(
                a.source.resolve(),
                a.destination.resolve(),
                a.shared_engram.resolve(),
                json.loads(a.allocation.read_text()),
                a.dry_run,
                a.reuse_from.resolve() if a.reuse_from else None,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
