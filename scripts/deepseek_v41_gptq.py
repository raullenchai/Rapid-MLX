#!/usr/bin/env python3
"""Resumable, bounded, layer-streamed calibration-aware native affine builder."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import shutil
import sys
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.deepseek_v41_gptq_io import (
    PartialSource,
    ShardSink,
    atomic_destination,
    atomic_json,
    conversion_record,
    copy_bound_metadata,
    copy_verified,
    digest,
    entries_from_json,
    entries_json,
    inventory,
    load_state,
    required_shards,
    save_state,
    state_snapshot,
    sync_files,
    verify_artifact,
)
from scripts.deepseek_v41_gptq_quant import QuantConfig, gptq, gram
from scripts.deepseek_v41_rd_build import (
    fixed_precision,
    groups_for,
    layout_report,
    output_entries,
    plan,
    write_group,
)
from scripts.deepseek_v41_rd_common import file_hash, read_slices

DEFAULTS = dict(
    group_size=64,
    block_size=128,
    damping=0.1,
    min_tokens=2048,
    low_token_damping=1.0,
    low_token_bump=False,
    gate_up_bits=2,
    down_bits=3,
    gate_up_3bit_layers=[],
    dense_bits=4,
    dense_8bit_layers=[],
    pilot=True,
    pilot_compare_gate_up=True,
    engram_bits=6,
    engram_group_size=32,
    keep_mtp=True,
    memory_limit_gb=26.0,
    cache_limit_gb=0.25,
    prefill_chunk=128,
    ffn_chunk=512,
    seed=1017,
    resident_budget_gb=210.0,
)
DENSE = (
    "attn.wq_a",
    "attn.wq_b",
    "attn.wkv",
    "attn.wo_b",
    "ffn.shared_experts.w1",
    "ffn.shared_experts.w3",
    "ffn.shared_experts.w2",
)
SUFFIXES = ("weight", "scales", "biases")


def recipe(overrides=None):
    c = copy.deepcopy({**DEFAULTS, **(overrides or {})})
    if set(c) != set(DEFAULTS):
        raise ValueError(f"unknown recipe keys: {set(c) - set(DEFAULTS)}")
    QuantConfig(**{k: c[k] for k in asdict(QuantConfig())})
    for key in ("pilot", "pilot_compare_gate_up", "low_token_bump", "keep_mtp"):
        if type(c[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key in ("dense_8bit_layers", "gate_up_3bit_layers"):
        values = c[key]
        if (
            not isinstance(values, list)
            or any(type(i) is not int or i < 0 for i in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"invalid {key}")
    for k in ("prefill_chunk", "ffn_chunk"):
        if type(c[k]) is not int or c[k] < 1:
            raise ValueError(f"invalid {k}")
    for k in ("memory_limit_gb", "cache_limit_gb", "resident_budget_gb"):
        if type(c[k]) not in (int, float) or not math.isfinite(c[k]) or c[k] <= 0:
            raise ValueError(f"invalid {k}")
    for key in (
        "gate_up_bits",
        "down_bits",
        "dense_bits",
        "engram_bits",
        "engram_group_size",
    ):
        if type(c[key]) is not int:
            raise ValueError(f"invalid {key}")
    if c["gate_up_bits"] not in (2, 3, 4) or c["down_bits"] not in (2, 3, 4):
        raise ValueError("routed expert bits must be 2, 3 or 4")
    if c["dense_bits"] not in (4, 8) or c["engram_bits"] not in (6, 8):
        raise ValueError("invalid dense or Engram bits")
    if c["engram_group_size"] not in (32, 64, 128):
        raise ValueError("invalid Engram group")
    return c


def quant_config(c):
    return QuantConfig(**{k: c[k] for k in asdict(QuantConfig())})


def allocation(source, c, layer=None, counts=None):
    a = source.args
    specs = {}
    for i in range(a.n_layers):
        gu = 3 if i in c["gate_up_3bit_layers"] else c["gate_up_bits"]
        down = c["down_bits"]
        if c["low_token_bump"] and i == layer and np.any(counts < c["min_tokens"]):
            # The runtime's stacked expert bank has a scalar width. A low-count
            # expert therefore promotes its complete projection bank, never an
            # incompatible per-expert layout.
            gu, down = min(4, gu + 1), min(4, down + 1)
        specs[str(i)] = dict(
            kept_expert_ids=list(range(a.n_routed_experts)),
            projection_bits=dict(gate_proj=gu, up_proj=gu, down_proj=down),
        )
    return dict(layers=specs, keep_mtp=c["keep_mtp"], mtp_expert_bits=4)


def output_plan(source, c, layer=None, counts=None):
    for key in ("dense_8bit_layers", "gate_up_3bit_layers"):
        if any(i >= source.args.n_layers for i in c[key]):
            raise ValueError(f"out-of-range {key}")
    if not c["pilot"] and len(c["dense_8bit_layers"]) != min(8, source.args.n_layers):
        raise ValueError(
            "final recipe requires eight pilot-selected dense_8bit_layers (all layers in tiny fixtures)"
        )
    alloc = allocation(source, c, layer, counts)
    entries, _, _ = plan(source.index, source.config, alloc)
    changed = []
    for e in entries:
        if e.suffix != "weight":
            continue
        base = e.name.removesuffix(".weight")
        local = ".".join(base.split(".")[2:]) if base.startswith("layers.") else ""
        if e.component == "engram_ssd":
            shape = (*e.shape[:-1], e.shape[-1] * 32 // e.bits)
            changed.extend(
                output_entries(
                    e.name,
                    shape,
                    e.sources,
                    e.component,
                    c["engram_bits"],
                    c["engram_group_size"],
                )
            )
        elif local in DENSE:
            i = int(base.split(".")[1])
            shape = (
                e.shape
                if e.bits is None
                else (*e.shape[:-1], e.shape[-1] * 32 // e.bits)
            )
            if shape[-1] % c["group_size"]:
                changed.extend(x for x in entries if x.name.rsplit(".", 1)[0] == base)
                continue
            bits = 8 if i in c["dense_8bit_layers"] else c["dense_bits"]
            changed.extend(
                output_entries(
                    e.name, shape, e.sources, "gptq_dense", bits, c["group_size"]
                )
            )
        elif e.component == "routed_experts" and c["group_size"] != e.group_size:
            shape = (*e.shape[:-1], e.shape[-1] * 32 // e.bits)
            changed.extend(
                output_entries(
                    e.name, shape, e.sources, e.component, e.bits, c["group_size"]
                )
            )
        else:
            changed.extend(x for x in entries if x.name.rsplit(".", 1)[0] == base)
    # All non-weight auxiliary tensors (norms, HC, biases) are standalone entries.
    covered = {e.name for e in changed}
    changed.extend(e for e in entries if e.bits is None and e.name not in covered)
    if len({e.name for e in changed}) != len(changed):
        raise ValueError("duplicate output entries")
    totals = {}
    for e in changed:
        totals[e.component] = totals.get(e.component, 0) + e.nbytes
    report = dict(
        components=totals,
        resident_tensor_bytes=sum(v for k, v in totals.items() if k != "engram_ssd"),
        shared_engram_bytes=totals.get("engram_ssd", 0),
        total_payload_bytes=sum(totals.values()),
    )
    return changed, layout_report(changed, report)


def boundary_estimate(a, lengths, layer, c):
    """Shape accounting for serialized states, including chunk-specific masks."""
    tokens = sum(lengths)
    hashes = (
        len(a.engram_layer_ids)
        * a.engram_n_heads
        * max(0, a.engram_max_ngram_size - 1)
        * 8
    )
    size = tokens * (2 * a.hc_mult * a.dim + 4 * a.hc_mult + 4 + hashes)
    owners = [i for i in a.kv_source_layers if i <= layer and a.compress_ratio(i)]
    index_owners = [i for i in owners if i in a.index_source_layers]
    caches = set(owners[-1:] + index_owners[-1:])
    for owner in caches:
        ratio = a.compress_ratio(owner)
        for n in lengths:
            size += 4 * a.window_size * a.head_dim
            size += 4 * (n // ratio) * a.head_dim
            if owner in a.index_source_layers:
                size += 4 * (n // ratio) * a.index_head_dim
            size += 8 * ratio * a.head_dim if ratio > 1 else 0
    sources = [i for i in a.index_source_layers if i <= layer and a.compress_ratio(i)]
    if sources:
        ratio = a.compress_ratio(sources[-1])
        for n in lengths:
            for start in range(0, n, c["prefill_chunk"]):
                end = min(n, start + c["prefill_chunk"])
                size += (end - start) * min(a.index_topk, end // ratio) * 4
    if 0 <= a.candidate_source_layer <= layer:
        ratio = a.compress_ratio(a.candidate_source_layer)
        if ratio:
            for n in lengths:
                size += sum(
                    (min(n, start + c["prefill_chunk"]) - start)
                    * (min(n, start + c["prefill_chunk"]) // ratio)
                    * np.dtype(
                        np.bool_
                    ).itemsize  # Native candidates are Boolean masks.
                    for start in range(0, n, c["prefill_chunk"])
                )
    return size


def estimates(source, c, lengths):
    if not lengths:
        raise ValueError("calibration manifest contains no windows")
    a = source.args
    tokens, window = sum(lengths), max(lengths)
    dim = max(a.dim, a.moe_inter_dim, a.q_lora_rank, a.o_groups * a.o_lora_rank)
    matrix_workspace = 16 * dim * dim * 4
    chunk_workspace = min(c["prefill_chunk"], window) * (
        64 * a.n_heads * a.head_dim
        + 16 * a.index_n_heads * window
        + 32 * a.hc_mult * a.hc_mult * a.dim
    )
    ffn_workspace = min(c["ffn_chunk"], tokens) * 32 * (a.dim + a.moe_inter_dim)
    entries, costs = output_plan(source, c)
    boundaries = [boundary_estimate(a, lengths, i, c) for i in range(-1, a.n_layers)]
    source_bytes = [
        sum(
            source.manifest["shards"][s]["size"]
            for s in required_shards(source.manifest, i)
        )
        for i in range(a.n_layers)
    ]
    ffn_inputs = tokens * (2 * a.dim + 8 * a.n_activated_experts)
    ffn_output = tokens * 4 * a.dim
    per_layer = []
    output_groups = groups_for(entries)
    for i in range(a.n_layers):
        before, after = boundaries[i : i + 2]
        one_window = after * window // tokens + 2**20
        if c["pilot"]:
            scratch = before + 2 * after + ffn_inputs + ffn_output + one_window
        else:
            # Final runs commit the attention/FFN-input phase before deleting
            # obsolete input boundaries, then replace output windows in place.
            scratch = (
                max(before + after + ffn_inputs, after + ffn_inputs + ffn_output)
                + one_window
            )
        output = sum(e.nbytes for e in output_groups[f"layer-{i}"])
        per_layer.append(
            dict(
                layer=i,
                scratch_bytes=scratch,
                source_shard_bytes=source_bytes[i],
                local_disk_with_export_bytes=scratch + source_bytes[i] + output,
            )
        )
    peak = int(
        4e9
        + matrix_workspace
        + chunk_workspace
        + ffn_workspace
        + 3 * max(boundaries) * window // tokens
    )
    scratch = max(r["scratch_bytes"] for r in per_layer)
    return dict(
        tokens=tokens,
        windows=len(lengths),
        max_window=window,
        estimated_peak_bytes=peak,
        scratch_bytes=scratch,
        checkpoint_bytes=max(boundaries),
        incoming_layer_shard_bytes=max(source_bytes),
        final_payload_bytes=costs["total_payload_bytes"],
        resident_tensor_bytes=costs["resident_tensor_bytes"],
        local_disk_with_export_bytes=max(
            r["local_disk_with_export_bytes"] for r in per_layer
        ),
        local_disk_with_all_outputs_bytes=max(
            r["scratch_bytes"] + r["source_shard_bytes"] for r in per_layer
        )
        + costs["total_safetensors_file_bytes"],
        per_layer=per_layer,
        note="Shape estimates, plus conservative matrix/attention RAM allowances. Export completed layer payloads; assemble on a larger disk. No M5 throughput is implied.",
    )


class MemoryGuard:
    """Fail-closed RSS watchdog plus MLX allocator/cache limits.

    The watchdog exits during long native operations too, leaving the last
    committed layer boundary untouched. RSS and MLX memory overlap: cap each,
    do not add the counters and double-count unified memory.
    """

    def __init__(self, c):
        import psutil

        self.process = psutil.Process()
        self.limit = int(c["memory_limit_gb"] * 1e9)
        self.peak = self.mlx_peak = 0
        self.event = threading.Event()
        self.thread = None
        self.previous_memory_limit = mx.set_memory_limit(self.limit)
        try:
            self.previous_cache_limit = mx.set_cache_limit(
                int(c["cache_limit_gb"] * 1e9)
            )
        except BaseException:
            mx.set_memory_limit(self.previous_memory_limit)
            raise

    def restore_limits(self):
        mx.set_cache_limit(self.previous_cache_limit)
        mx.set_memory_limit(self.previous_memory_limit)

    def check(self):
        rss = self.process.memory_info().rss
        mlx_bytes = mx.get_active_memory() + mx.get_cache_memory()
        self.peak, self.mlx_peak = max(self.peak, rss), max(self.mlx_peak, mlx_bytes)
        if max(rss, mlx_bytes) > self.limit:
            raise MemoryError(
                f"GPTQ memory cap exceeded: RSS={rss}, MLX={mlx_bytes}, cap={self.limit}"
            )

    def __enter__(self):
        try:
            self.check()
        except BaseException:
            self.restore_limits()
            raise

        def watch():
            while not self.event.wait(0.02):
                try:
                    self.check()
                except MemoryError as exc:
                    print(str(exc), file=sys.stderr, flush=True)
                    os._exit(86)

        self.thread = threading.Thread(target=watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.event.set()
        self.thread.join()
        self.restore_limits()

    def report(self):
        self.check()
        return dict(
            peak_rss_bytes=self.peak,
            peak_mlx_active_cache_bytes=self.mlx_peak,
            memory_cap_bytes=self.limit,
            rss_poll_seconds=0.02,
        )


class EmptyExperts(nn.Module):
    def __call__(self, *_):
        raise RuntimeError("expert banks must be streamed individually")


class AffineSourceEngram:
    def __init__(self, index, layer, bits, group):
        from scripts.deepseek_v41_streaming_teacher import SourceEngram

        self.source = SourceEngram(index, layer)
        self.bits, self.group = bits, group

    def __call__(self, ids):
        # Row quantization is independent of the table, so these bytes have the
        # same grids as final assembly, without converting a giant table first.
        w = self.source(ids).astype(mx.bfloat16)
        shape = w.shape
        w = w.reshape(-1, shape[-1])
        return mx.dequantize(
            *mx.quantize(w, bits=self.bits, group_size=self.group),
            bits=self.bits,
            group_size=self.group,
        ).reshape(shape)


def module_at(block, name):
    value = block
    for key in name.split("."):
        value = getattr(value, key)
    return value


def set_module(block, name, module):
    parent, _, leaf = name.rpartition(".")
    setattr(module_at(block, parent), leaf, module)


def cpu_affine_call(self, x, transpose=True):
    """Float32 BLAS adapter for CPU checks; Metal retains native packed kernels.

    The CPU scalar affine kernel rounds a BF16 accumulator on every term. Use
    the same float32 arithmetic in both bounded calibration and its comparator,
    preserving the public module's output dtype and serialized native bytes.
    """
    w = mx.dequantize(
        self.weight,
        self.scales.astype(mx.float32),
        self.biases.astype(mx.float32),
        bits=self.bits,
        group_size=self.group_size,
    )
    if w.ndim == 3:
        # Flatten token/row dimensions per group. Broadcasting a grouped weight
        # across a long window can make the CPU matmul copy a huge RHS batch.
        outputs = []
        for group in range(w.shape[0]):
            value = x[..., group, :, :].astype(mx.float32)
            matrix = w[group].T if transpose else w[group]
            out = value.reshape(-1, value.shape[-1]) @ matrix
            outputs.append(out.reshape(*value.shape[:-1], matrix.shape[-1]))
        y = mx.stack(outputs, axis=-3)
    else:
        y = x.astype(mx.float32) @ (w.T if transpose else w)
    if "bias" in self:
        y += self.bias
    return y.astype(mx.result_type(x, self.scales))


def cpu_dense_call(self, x):
    y = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
    if "bias" in self:
        y += self.bias
    return y.astype(mx.result_type(x, self.weight))


def cpu_grouped_dense_call(self, x, transpose=True):
    w = self.weight.astype(mx.float32)
    outputs = []
    for group in range(w.shape[0]):
        value = x[..., group, :, :].astype(mx.float32)
        matrix = w[group].T if transpose else w[group]
        out = value.reshape(-1, value.shape[-1]) @ matrix
        outputs.append(out.reshape(*value.shape[:-1], matrix.shape[-1]))
    return mx.stack(outputs, axis=-3).astype(mx.result_type(x, self.weight))


def cpu_switch_call(self, x, indices, sorted_indices=False):
    w = mx.dequantize(
        self.weight,
        self.scales.astype(mx.float32),
        self.biases.astype(mx.float32),
        bits=self.bits,
        group_size=self.group_size,
    )
    y = mx.gather_mm(
        x.astype(mx.float32),
        w.swapaxes(-1, -2),
        rhs_indices=indices,
        sorted_indices=sorted_indices,
    )
    return y.astype(mx.result_type(x, self.scales))


def install_cpu_affine(module):
    from mlx.utils import tree_flatten

    if mx.default_device() != mx.cpu:
        return
    children = [
        module,
        *(
            child
            for _, child in tree_flatten(
                module.leaf_modules(), is_leaf=nn.Module.is_module
            )
        ),
    ]
    for child in children:
        if (
            hasattr(child, "scales")
            and hasattr(child, "bits")
            and child.weight.dtype == mx.uint32
        ):
            switch = any(
                cls.__name__ == "QuantizedSwitchLinear" for cls in type(child).__mro__
            )
            child.__class__ = type(
                "CPUAffine" + type(child).__name__,
                (type(child),),
                {"__call__": cpu_switch_call if switch else cpu_affine_call},
            )

        elif isinstance(child, nn.Linear) and child.weight.ndim == 2:
            child.__class__ = type(
                "CPUDense" + type(child).__name__,
                (type(child),),
                {"__call__": cpu_dense_call},
            )
        elif (
            hasattr(child, "weight")
            and child.weight.ndim == 3
            and any(cls.__name__ == "MultiLinear" for cls in type(child).__mro__)
        ):
            child.__class__ = type(
                "CPUGroupedDense" + type(child).__name__,
                (type(child),),
                {"__call__": cpu_grouped_dense_call},
            )


def packed_module(weight, packed, bits, group):
    m = nn.QuantizedLinear(
        weight.shape[-1], weight.shape[0], bias=False, group_size=group, bits=bits
    )
    m.weight, m.scales, m.biases = packed
    install_cpu_affine(m)
    return m


def bounded_block(source, layer, c, *, original=False):
    from rapid_mlx.models.deepseek_v41_native.load import reshape_grouped_wo_a
    from rapid_mlx.models.deepseek_v41_native.model import Block

    args = replace(
        source.args,
        n_routed_experts=1,
        engram_num_embeddings=tuple(1 for _ in source.args.engram_layer_ids),
    )
    # Construct only a one-expert placeholder, then remove it before any eval.
    block = Block(layer, args)
    block.ffn.experts = EmptyExperts()
    from rapid_mlx.models.deepseek_v41_native.moe import Gate

    block.ffn.gate = Gate(source.args)
    if block.engram is not None:
        if original:
            from scripts.deepseek_v41_streaming_teacher import SourceEngram

            block.engram.embed = SourceEngram(source.index, layer)
        else:
            block.engram.embed = AffineSourceEngram(
                source.index, layer, c["engram_bits"], c["engram_group_size"]
            )
    prefix = f"layers.{layer}."
    values = [
        (
            n,
            source.tensor(n)
            if original
            else source.tensor(n).astype(
                mx.float32 if n.endswith(("gate.bias", "gate.bias_vl")) else mx.bfloat16
            ),
        )
        for n in source.index.tensors
        if n.startswith(prefix)
        and not n.endswith(".scale")
        and ".ffn.experts." not in n
        and ".engram.embed." not in n
    ]
    mx.eval([v for _, v in values])
    values = reshape_grouped_wo_a(values, source.args)
    block.load_weights([(n[len(prefix) :], v) for n, v in values], strict=True)

    # Fixed policy remains the existing eight-bit policy, including grouped wo_a.
    def pred(name, module):
        if not hasattr(module, "to_quantized"):
            return False
        if name in DENSE and module.weight.shape[-1] % c["group_size"] == 0:
            return False
        w = module.weight
        shape = w.shape if w.ndim == 2 else (math.prod(w.shape[:-1]), w.shape[-1])
        bits = fixed_precision(prefix + name + ".weight", shape)
        return dict(bits=bits, group_size=64) if bits else False

    if not original:
        nn.quantize(block, class_predicate=pred)
    install_cpu_affine(block)
    block.eval()
    mx.eval(block.parameters())
    return block


class Probe(nn.Module):
    def __init__(self, module, guard):
        super().__init__()
        self.module = module
        self.h = mx.zeros((module.weight.shape[-1],) * 2, mx.float32)
        self.count = 0
        self.guard = guard

    def __call__(self, x):
        xf = x.reshape(-1, x.shape[-1]).astype(mx.float32)
        self.h += xf.T @ xf
        self.count += len(xf)
        mx.eval(self.h)
        self.guard.check()
        return self.module(x)


def attention_window(block, state, args, c, guard):
    """Replay native attention with fresh current-layer caches on every pass.

    A selection belongs to its exact prefill chunk: native top-k indices include
    the local window offset. Preserve every chunk's selections and masks, along
    with the full source KV/index caches; a last-chunk-only snapshot is invalid.
    """
    from rapid_mlx.models.deepseek_v41_native.cache import LayerCache
    from rapid_mlx.models.deepseek_v41_native.hyper_connections import (
        hc_mixes,
        hc_post,
        hc_pre_norm,
    )

    length = state["h"].shape[1]
    cache = SimpleNamespace(layers=[None] * args.n_layers)
    cache.layers[block.layer_id] = LayerCache(1, args, block.layer_id, length)
    shared = state["shared"]
    xs, pres, selections = [], [], []
    for j, start in enumerate(range(0, length, c["prefill_chunk"])):
        end = min(length, start + c["prefill_chunk"])
        incoming = state["selections"][j] if state["selections"] else {}
        shared.topk_idxs = incoming.get("topk")
        shared.candidates = incoming.get("candidates")
        x = state["h"][:, start:end]
        if block.engram is not None:
            x = block.engram(
                x, state["hashes"][:, start:end, block.engram.layer_hash_index]
            )
        ap, post, comb = hc_mixes(
            x,
            block.hc_attn_fn,
            block.hc_attn_scale,
            block.hc_attn_base,
            block.hc_mult,
            block.hc_iters,
            block.norm_eps,
            block.hc_eps,
        )
        h = hc_pre_norm(
            x, state["pre"][:, start:end], block.attn_norm.weight, block.attn_norm.eps
        )
        h = block.attn(h, start, cache, shared)
        x = hc_post(h, x, post, comb)
        mx.eval(x, ap, shared.topk_idxs, shared.candidates)
        xs.append(x)
        pres.append(ap)
        selections.append(dict(topk=shared.topk_idxs, candidates=shared.candidates))
        guard.check()
    # Selection arrays are stored only once, in their chunk list.
    shared.topk_idxs = shared.candidates = None
    return dict(
        h=mx.concatenate(xs, axis=1),
        attn_pre=mx.concatenate(pres, axis=1),
        shared=shared,
        selections=selections,
        ids=state["ids"],
        hashes=state["hashes"],
    )


def projection_errors(weight, packed, bits, chunks, group):
    """Measure sum(g * squared projection error), the weighted-Gram objective.

    Routing weights are sample weights here, as in gram(): H = sum(g*x*x.T).
    This projection ranking proxy is not end-to-end routed FFN squared error.
    """
    q = mx.dequantize(*packed, group_size=group, bits=bits, dtype=mx.float32)
    rtn = mx.dequantize(
        *mx.quantize(weight, bits=bits, group_size=group),
        bits=bits,
        group_size=group,
        dtype=mx.float32,
    )
    w = weight.astype(mx.float32)
    errors, energy = np.zeros(2, np.float64), 0.0
    tokens = 0
    for x, g in chunks():
        x = mx.array(x).astype(mx.float32)
        tokens += len(x)
        ref = x @ w.T
        scale = 1 if g is None else mx.array(g).astype(mx.float32)[:, None]
        energy += float(mx.sum(ref * ref * scale).item())
        for i, delta in enumerate((w - rtn, w - q)):
            e = x @ delta.T
            errors[i] += float(mx.sum(e * e * scale).item())
    return dict(
        rtn_relative_output_error=float(errors[0] / max(energy, 1e-30)),
        gptq_relative_output_error=float(errors[1] / max(energy, 1e-30)),
        calibration_output_energy=energy,
        error_token_count=tokens,
    )


def fit(
    weight, h, count, bits, c, guard, chunks=None, factor=None, heldout_chunks=None
):
    if count == 0:
        packed = mx.quantize(weight, group_size=c["group_size"], bits=bits)
        stats = dict(
            tokens=0,
            low_token=True,
            fallback="unobserved_rtn",
            damping_coefficient=c["low_token_damping"],
            damping_absolute=0,
        )
    else:
        packed, stats = gptq(
            weight,
            h,
            bits,
            quant_config(c),
            count=count,
            check=guard.check,
            factor=factor,
        )
    if chunks is not None:
        errors = projection_errors(weight, packed, bits, chunks, c["group_size"])
        stats.update(errors)
        stats["calibration"] = dict(error_scope="calibration", **errors)
    stats["heldout"] = dict(status="not_configured")
    if heldout_chunks is not None:
        errors = projection_errors(
            weight, packed, bits, heldout_chunks, c["group_size"]
        )
        energy = errors.pop("calibration_output_energy")
        tokens = errors.pop("error_token_count")
        stats["heldout"] = dict(
            status="measured" if energy else "unobserved",
            output_energy=energy,
            error_scope="heldout_quantized_prefix_inputs",
            tokens=tokens,
            **{k: v if energy else None for k, v in errors.items()},
        )
    stats["bits"] = bits
    return packed, stats


def spill_ffn(block, stage, paths, total, args, c, guard):
    from rapid_mlx.models.deepseek_v41_native.hyper_connections import hc_pre_norm

    with paths["x"].open("wb") as f:
        f.truncate(total * args.dim * 2)
    ids = np.memmap(
        paths["ids"], mode="w+", dtype=np.int32, shape=(total, args.n_activated_experts)
    )
    gates = np.memmap(paths["gates"], mode="w+", dtype=np.float32, shape=ids.shape)
    offset = 0
    counts = np.zeros(args.n_routed_experts, np.int64)
    for path in stage:
        state = load_state(path)
        n = state["h"].shape[1]
        for start in range(0, n, c["ffn_chunk"]):
            end = min(n, start + c["ffn_chunk"])
            h = hc_pre_norm(
                state["h"][:, start:end],
                state["attn_pre"][:, start:end],
                block.ffn_norm.weight,
                block.ffn_norm.eps,
            ).reshape(-1, args.dim)
            g, ix = block.ffn.gate(h)
            mx.eval(h, g, ix)
            lo, hi = offset + start, offset + end
            x = np.memmap(
                paths["x"],
                mode="r+",
                dtype=np.uint16,
                offset=lo * args.dim * 2,
                shape=(hi - lo, args.dim),
            )
            x[:] = np.asarray(h.astype(mx.bfloat16).view(mx.uint16))
            x.flush()
            del x
            ids[lo:hi], gates[lo:hi] = np.asarray(ix), np.asarray(g)
            counts += np.bincount(ids[lo:hi].ravel(), minlength=args.n_routed_experts)
            guard.check()
        offset += n
        del state
    ids.flush()
    gates.flush()
    del ids, gates
    return counts


def ffn_chunks(paths, total, args, c, expert=None):
    """Batch routed rows, rather than emitting a few rows per global token batch.

    Reopen each bounded input mapping after gathering so a million-token shared
    pass cannot accumulate its entire activation file into process RSS.
    """
    if expert is None:
        positions, weights = None, None
        length = total
    else:
        ids = np.memmap(
            paths["ids"],
            mode="r",
            dtype=np.int32,
            shape=(total, args.n_activated_experts),
        )
        gates = np.memmap(paths["gates"], mode="r", dtype=np.float32, shape=ids.shape)
        positions, slots = np.where(ids == expert)
        weights = np.array(gates[positions, slots])
        length = len(positions)
        del ids, gates, slots
    for start in range(0, length, c["ffn_chunk"]):
        end = min(length, start + c["ffn_chunk"])
        if positions is None:
            rows = np.arange(start, end)
            mapped = np.memmap(
                paths["x"],
                mode="r",
                dtype=np.uint16,
                offset=start * args.dim * 2,
                shape=(end - start, args.dim),
            )
            raw = np.array(mapped)
        else:
            rows = positions[start:end]
            mapped = np.memmap(
                paths["x"], mode="r", dtype=np.uint16, shape=(total, args.dim)
            )
            raw = np.array(mapped[rows])
        del mapped
        x = mx.array(raw).view(mx.bfloat16)
        yield rows, x, None if weights is None else weights[start:end]


def intermediate(x, gate, up, limit, *, routed=False):
    g, u = gate(x).astype(mx.float32), up(x).astype(mx.float32)
    if limit > 0:
        g, u = mx.minimum(g, limit), mx.clip(u, -limit, limit)
    h = nn.silu(g) * u
    return h if routed else h.astype(x.dtype)


def write_packed(sink, base, triple, expert=None):
    for suffix, tensor in zip(SUFFIXES, triple):
        sink.put(base + "." + suffix, tensor, expert)


def quantize_ffn(
    source, layer, block, sink, entries, paths, counts, total, c, guard, heldout=None
):
    args = source.args
    stats = {}
    # Truncating a new extent supplies logical zeros without faulting the whole
    # output into RSS (30.7 GB at 1.5M tokens). Expert updates touch bounded rows.
    with paths["y"].open("wb") as f:
        f.truncate(total * args.dim * 4)
    if heldout is not None:
        with heldout["paths"]["y"].open("wb") as f:
            f.truncate(heldout["total"] * args.dim * 4)
    specs = {e.name.removesuffix(".weight"): e for e in entries if e.suffix == "weight"}
    prefix = f"layers.{layer}.ffn."
    from scripts.deepseek_v41_gptq_batched import quantize_routed

    stats.update(
        quantize_routed(source, layer, sink, specs, paths, total, c, guard, heldout)
    )
    for expert in range(args.n_routed_experts, args.n_routed_experts + 1):
        expert_started = time.monotonic()
        shared = expert == args.n_routed_experts
        eid = None if shared else expert

        def chunks(eid=eid):
            for _, x, g in ffn_chunks(paths, total, args, c, eid):
                yield x, g

        def heldout_chunks(eid=eid):
            for _, x, g in ffn_chunks(heldout["paths"], heldout["total"], args, c, eid):
                yield x, g

        observed = heldout_chunks if heldout is not None else None

        h, n = gram(chunks(), args.dim, check=guard.check)
        from scripts.deepseek_v41_gptq_quant import inverse_factor

        factor = inverse_factor(h, quant_config(c), n) if n else None
        modules = []
        for source_proj, final in (("w1", "gate_proj"), ("w3", "up_proj")):
            base = prefix + (
                "shared_experts." + source_proj if shared else "experts." + final
            )
            source_name = prefix + (
                "shared_experts." + source_proj
                if shared
                else f"experts.{expert}." + source_proj
            )
            w = source.tensor(source_name + ".weight")
            spec = specs[base]
            packed, report = fit(
                w,
                h,
                n,
                spec.bits,
                c,
                guard,
                chunks,
                factor=factor,
                heldout_chunks=observed,
            )
            stats[
                (
                    f"shared.{source_proj}"
                    if shared
                    else f"expert.{expert}.{source_proj}"
                )
            ] = report
            write_packed(sink, base, packed, None if shared else expert)
            module = packed_module(w, packed, spec.bits, spec.group_size)
            modules.append(module)
            if shared:
                set_module(block, "ffn.shared_experts." + source_proj, module)
            del packed, w
        del h, factor

        def down_chunks(modules=modules, chunks=chunks, shared=shared):
            for x, g in chunks():
                yield intermediate(x, *modules, args.swiglu_limit, routed=not shared), g

        def heldout_down(modules=modules, observed=observed, shared=shared):
            for x, g in observed():
                yield intermediate(x, *modules, args.swiglu_limit, routed=not shared), g

        h, _ = gram(down_chunks(), args.moe_inter_dim, check=guard.check)
        base = prefix + ("shared_experts.w2" if shared else "experts.down_proj")
        source_name = prefix + (
            "shared_experts.w2" if shared else f"experts.{expert}.w2"
        )
        w = source.tensor(source_name + ".weight")
        spec = specs[base]
        packed, report = fit(
            w,
            h,
            n,
            spec.bits,
            c,
            guard,
            down_chunks,
            heldout_chunks=heldout_down if heldout is not None else None,
        )
        stats["shared.w2" if shared else f"expert.{expert}.w2"] = report
        write_packed(sink, base, packed, None if shared else expert)
        down = packed_module(w, packed, spec.bits, spec.group_size)
        if shared:
            block.ffn.shared_experts.w2 = down
        # Routed banks are complete; add the unweighted shared-expert output.
        for rows, x, _g in ffn_chunks(paths, total, args, c, eid):
            y = down(
                intermediate(x, *modules, args.swiglu_limit, routed=not shared)
            ).astype(mx.float32)
            mx.eval(y)
            output = np.memmap(
                paths["y"], mode="r+", dtype=np.float32, shape=(total, args.dim)
            )
            output[rows] += np.asarray(y)
            output.flush()
            del output
            guard.check()
        if heldout is not None:
            from scripts.deepseek_v41_gptq_quality import add_outputs

            add_outputs(down, modules, heldout, args, c, guard, eid)
        del h, w, packed, modules, down
        gc.collect()
        mx.clear_cache()
        print(
            json.dumps(
                dict(
                    layer=layer,
                    expert="shared" if shared else expert,
                    tokens=n,
                    seconds=time.monotonic() - expert_started,
                    **guard.report(),
                )
            ),
            flush=True,
        )
    return stats


def finalize_windows(
    block, stage, paths, target, total, args, c, guard, *, run_identity, layer
):
    from rapid_mlx.models.deepseek_v41_native.hyper_connections import hc_mixes, hc_post

    target.mkdir(exist_ok=True)
    offset = 0
    for window, path in enumerate(stage):
        completed = target / f"window-{window}"
        binding = dict(run_identity=run_identity, layer=layer + 1, window=window)
        if (
            not path.with_suffix(".json").exists()
            and completed.with_suffix(".json").exists()
        ):
            # A previous process committed this window before its interruption.
            saved = load_state(completed)
            if saved.get("gptq_boundary") != binding:
                raise ValueError("completed boundary window identity mismatch")
            offset += saved["h"].shape[1]
            del saved
            continue
        state = load_state(path)
        n = state["h"].shape[1]
        y = np.memmap(paths["y"], mode="r", dtype=np.float32, shape=(total, args.dim))
        xs, pres = [], []
        for start in range(0, n, c["prefill_chunk"]):
            end = min(n, start + c["prefill_chunk"])
            x = state["h"][:, start:end]
            pre, post, comb = hc_mixes(
                x,
                block.hc_ffn_fn,
                block.hc_ffn_scale,
                block.hc_ffn_base,
                block.hc_mult,
                block.hc_iters,
                block.norm_eps,
                block.hc_eps,
            )
            out = mx.array(np.array(y[offset + start : offset + end])).astype(x.dtype)[
                None
            ]
            out = hc_post(out, x, post, comb)
            mx.eval(out, pre)
            xs.append(out)
            pres.append(pre)
            guard.check()
        del y
        state.pop("attn_pre")
        state["h"], state["pre"] = (
            mx.concatenate(xs, axis=1),
            mx.concatenate(pres, axis=1),
        )
        state["gptq_boundary"] = binding
        save_state(target / f"window-{window}", state)
        if not c["pilot"]:
            _, payload = state_snapshot(path)
            path.with_suffix(".json").unlink()
            payload.unlink()
        offset += n
        del state
        gc.collect()
        mx.clear_cache()


class DiskExperts:
    """Bounded native-bank adapter used only for the pilot's block comparator."""

    def __init__(self, path, entries, layer, dim, limit, guard):
        import struct

        self.path, self.dim, self.limit, self.guard = path, dim, limit, guard
        with path.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.start = 8 + n
        self.specs = {
            e.name.removesuffix(".weight"): e for e in entries if e.suffix == "weight"
        }
        self.prefix = f"layers.{layer}.ffn.experts."

    def module(self, projection, expert):
        base = self.prefix + projection
        spec = self.specs[base]
        triple = []
        for suffix in SUFFIXES:
            meta = self.header[base + "." + suffix]
            shape = meta["shape"]
            rowbytes = (meta["data_offsets"][1] - meta["data_offsets"][0]) // shape[0]
            raw = np.memmap(
                self.path,
                mode="r",
                dtype=np.uint32 if suffix == "weight" else np.uint16,
                offset=self.start + meta["data_offsets"][0] + expert * rowbytes,
                shape=tuple(shape[1:]),
            )
            value = mx.array(np.array(raw))
            del raw
            triple.append(value if suffix == "weight" else value.view(mx.bfloat16))
        cols = spec.shape[-1] * 32 // spec.bits
        return packed_module(
            SimpleNamespace(shape=(spec.shape[-2], cols)),
            triple,
            spec.bits,
            spec.group_size,
        )

    def __call__(self, x, indices):
        ids = np.asarray(indices)
        result = mx.zeros((*indices.shape, self.dim), mx.float32)
        for expert in np.unique(ids):
            rows, slots = np.where(ids == expert)
            gate, up, down = [
                self.module(p, int(expert))
                for p in ("gate_proj", "up_proj", "down_proj")
            ]
            out = down(
                intermediate(x[mx.array(rows)], gate, up, self.limit, routed=True)
            )
            result[mx.array(rows), mx.array(slots)] = out
            mx.eval(result)
            self.guard.check()
        return result


def pilot_block_movement(source, layer, c, windows, target, parts, entries, guard):
    """Isolate dense four-bit perturbation at a complete block's output.

    Both branches use the same quantized routed experts and incoming boundary.
    The comparator restores source attention/shared projections, while retaining
    the fixed eight-bit auxiliaries and six-bit Engram of the propagated branch.
    Recompute routing independently after its changed attention output.
    """
    from rapid_mlx.models.deepseek_v41_native.hyper_connections import (
        hc_mixes,
        hc_post,
        hc_pre_norm,
    )

    block = bounded_block(source, layer, c)
    block.ffn.experts = DiskExperts(
        parts / f"model-layer-{layer}.safetensors",
        entries,
        layer,
        source.args.dim,
        source.args.swiglu_limit,
        guard,
    )
    numerator = denominator = 0.0
    for window, path in enumerate(windows):
        print(
            json.dumps(dict(layer=layer, stage="pilot_block_reference", window=window)),
            flush=True,
        )
        state = attention_window(block, load_state(path), source.args, c, guard)
        actual = load_state(target / f"window-{window}")
        n = state["h"].shape[1]
        for start in range(0, n, c["ffn_chunk"]):
            end = min(n, start + c["ffn_chunk"])
            x = state["h"][:, start:end]
            _, post, comb = hc_mixes(
                x,
                block.hc_ffn_fn,
                block.hc_ffn_scale,
                block.hc_ffn_base,
                block.hc_mult,
                block.hc_iters,
                block.norm_eps,
                block.hc_eps,
            )
            inp = hc_pre_norm(
                x,
                state["attn_pre"][:, start:end],
                block.ffn_norm.weight,
                block.ffn_norm.eps,
            )
            expected = hc_post(block.ffn(inp), x, post, comb).astype(mx.float32)
            diff = actual["h"][:, start:end].astype(mx.float32) - expected
            numerator += float(mx.sum(diff * diff).item())
            denominator += float(mx.sum(expected * expected).item())
            guard.check()
        del state, actual
        gc.collect()
        mx.clear_cache()
    return numerator / max(denominator, 1e-30)


def select_recipe(parts, source, c, *, scan=False):
    """Choose eight dense layers from pilot block movement, then spend spare bytes."""
    from scripts.deepseek_v41_gptq_amend import layer_epoch

    if scan:
        from scripts.deepseek_v41_gptq_scan import validate_records

        records = validate_records(parts, source, c)
    else:
        records = [
            json.loads((parts / f"layer-{i}-stats.json").read_text())
            for i in range(source.args.n_layers)
        ]
        if source.args.n_layers < 8 or any(
            r.get("block_output_4bit_relative_error") is None for r in records
        ):
            raise ValueError(
                "selection requires a complete pilot and at least eight layers"
            )
        run = json.loads((parts / "run.json").read_text())
        if (
            any(r["run_identity"] != layer_epoch(run, r["layer"])[1] for r in records)
            or c != run["recipe"]
        ):
            raise ValueError("mixed pilot runs or recipes")
    error_key = "rtn_relative_output_error" if scan else "gptq_relative_output_error"
    three_key = "three_bit_" + error_key
    layers = sorted(
        records, key=lambda r: (-r["block_output_4bit_relative_error"], r["layer"])
    )
    result = {
        **c,
        "pilot": False,
        "dense_8bit_layers": sorted(r["layer"] for r in layers[:8]),
        "gate_up_3bit_layers": [],
    }
    benefit = []
    for r in records:
        value = sum(
            max(
                0,
                p[error_key]
                - p.get(
                    three_key,
                    p[error_key],
                ),
            )
            * p.get("calibration_output_energy", 0)
            for key, p in r["projections"].items()
            if key.startswith("expert.") and key.endswith((".w1", ".w3"))
        )
        benefit.append((value, r["layer"]))
    for value, i in sorted(benefit, key=lambda x: (-x[0], x[1])):
        if value <= 0:
            continue
        candidate = {
            **result,
            "gate_up_3bit_layers": sorted([*result["gate_up_3bit_layers"], i]),
        }
        _, costs = output_plan(source, candidate)
        if costs["resident_tensor_bytes"] <= result["resident_budget_gb"] * 1e9:
            result = candidate
    _, costs = output_plan(source, result)
    if costs["resident_tensor_bytes"] > result["resident_budget_gb"] * 1e9:
        raise ValueError("eight selected dense layers exceed budget")
    return result


def initialize(
    source,
    c,
    windows,
    scratch,
    parts,
    calibration_identity,
    guard,
    token_map=None,
    calibration_manifest_sha256=None,
    heldout_reference=None,
    separation=None,
):
    from rapid_mlx.models.deepseek_v41_native.engram import EngramHasher
    from rapid_mlx.models.deepseek_v41_native.hyper_connections import (
        make_identity_pre_mix,
    )
    from rapid_mlx.models.deepseek_v41_native.load import load_token_map
    from rapid_mlx.models.deepseek_v41_native.model import SharedState

    if separation is not None:
        from scripts.deepseek_v41_gptq_quality import (
            validate_reference,
            validate_separation,
        )

        validate_reference(source, heldout_reference)
        validate_separation(
            separation["reference"],
            heldout_reference,
            windows,
            separation["corpus_hashes"],
        )
    lengths = [len(w) for w in windows]
    if not lengths or any(n < 1 or n > source.args.max_seq_len for n in lengths):
        raise ValueError("invalid calibration windows")
    estimate = estimates(source, c, lengths)
    if heldout_reference is not None:
        from scripts.deepseek_v41_gptq_quality import planning_with_quality

        estimate = planning_with_quality(source, estimate, heldout_reference)
    print(json.dumps(dict(estimate=estimate)), flush=True)
    if estimate["estimated_peak_bytes"] > c["memory_limit_gb"] * 1e9:
        raise MemoryError(
            "planning estimate exceeds memory cap; reduce prefill_chunk/window"
        )
    if estimate["resident_tensor_bytes"] > c["resident_budget_gb"] * 1e9:
        raise ValueError("planned resident payload exceeds budget")
    if parts.exists() or scratch.exists():
        raise FileExistsError("new run requires fresh parts and scratch directories")
    scratch.mkdir(parents=True)
    parts.mkdir(parents=True)
    run = dict(
        schema_version=1,
        recipe=c,
        source_identity=digest(source.manifest),
        source_revision=source.manifest["source_revision"],
        calibration_identity=calibration_identity,
        calibration_manifest_sha256=calibration_manifest_sha256 or calibration_identity,
        token_hashes=[digest(np.asarray(w, dtype=np.int32).tolist()) for w in windows],
        lengths=lengths,
        estimate=estimate,
    )
    if separation is not None:
        run["calibration_corpus_hashes"] = separation["corpus_hashes"]
        run["calibration_token_windows"] = [
            np.asarray(w, np.int32).tolist() for w in windows
        ]
    if heldout_reference is not None:
        run["heldout_reference_identity"] = heldout_reference["identity"]
        run["heldout_token_hashes"] = heldout_reference["token_hashes"]
    run["identity"] = digest(run)
    atomic_json(parts / "run.json", run)
    atomic_json(parts / "source-manifest.json", source.manifest)
    source.verify("top")
    hasher = None
    if source.args.engram_layer_ids:
        if token_map is None:
            token_map = load_token_map(
                str(source.source), source.args, cache_dir=str(parts)
            )
        else:
            atomic_json(parts / "engram_token_map.json", token_map)
        hasher = EngramHasher(source.args, token_map)
    directory = scratch / "boundary-0"
    directory.mkdir()
    # No full Model/decoder layer exists while the embedding is resident.
    embed = source.tensor("embed.weight").astype(mx.bfloat16)
    mx.eval(embed)
    for i, ids in enumerate(windows):
        ids = np.asarray(ids, np.int32)
        h = embed[mx.array(ids)[None]]
        h = mx.broadcast_to(
            h[:, :, None], (1, len(ids), source.args.hc_mult, source.args.dim)
        )
        hashes = (
            None
            if hasher is None
            else mx.array(hasher(ids[None], 0, np.zeros((1, len(ids)), np.int64)))
        )
        state = dict(
            ids=mx.array(ids),
            h=h,
            pre=make_identity_pre_mix(1, len(ids), source.args.hc_mult),
            hashes=hashes,
            shared=SharedState(),
            selections=[],
        )
        save_state(directory / f"window-{i}", state)
        del state, h, hashes
        guard.check()
    source.verify("top")
    atomic_json(
        scratch / "checkpoint.json", dict(next_layer=0, identity=run["identity"])
    )
    return run


def prepare_attention(source, layer, c, block, windows, work, guard):
    from scripts.deepseek_v41_gptq_quant import relative_error

    probes = {}
    for name in DENSE[:4]:
        module = module_at(block, name)
        # Shapes that cannot use native affine groups retain the fixed policy.
        if hasattr(module, "scales") or module.weight.shape[-1] % c["group_size"]:
            continue
        probe = Probe(module, guard)
        probes[name] = probe
        set_module(block, name, probe)
    for path in windows:
        state = load_state(path)
        observed = attention_window(block, state, source.args, c, guard)
        del observed, state
        gc.collect()
        mx.clear_cache()
    attention_stats, attention_packed = {}, {}
    for name, probe in probes.items():
        print(
            json.dumps(
                dict(
                    layer=layer,
                    stage="attention_gptq",
                    projection=name,
                    **guard.report(),
                )
            ),
            flush=True,
        )
        bits = 8 if layer in c["dense_8bit_layers"] else c["dense_bits"]
        w = probe.module.weight
        packed, stats = fit(w, probe.h, probe.count, bits, c, guard)
        stats.update(
            rtn_relative_output_error=relative_error(
                w,
                mx.quantize(w, bits=bits, group_size=c["group_size"]),
                probe.h,
                bits,
                c["group_size"],
            ),
            gptq_relative_output_error=relative_error(
                w, packed, probe.h, bits, c["group_size"]
            ),
        )
        stats["calibration"] = dict(
            error_scope="calibration",
            rtn_relative_output_error=stats["rtn_relative_output_error"],
            gptq_relative_output_error=stats["gptq_relative_output_error"],
        )
        attention_stats[name] = stats
        attention_packed[name] = packed
        set_module(block, name, packed_module(w, packed, bits, c["group_size"]))
    probes.clear()
    probe = w = None
    mx.clear_cache()
    stage = []
    for i, path in enumerate(windows):
        state = load_state(path)
        out = attention_window(block, state, source.args, c, guard)
        path = work / f"attention-{i}"
        save_state(path, out)
        stage.append(path)
        del state, out
        gc.collect()
        mx.clear_cache()
    tensors = {
        name + "." + suffix: value
        for name, triple in attention_packed.items()
        for suffix, value in zip(SUFFIXES, triple)
    }
    mx.save_safetensors(str(work / "attention-quantized.safetensors"), tensors)
    atomic_json(work / "attention-stats.json", attention_stats)
    return stage, attention_packed, attention_stats


def run_layer(source, c, parts, scratch, layer, guard):
    from mlx.utils import tree_flatten

    from scripts.deepseek_v41_gptq_amend import layer_epoch, validate_record

    started = time.monotonic()
    run = json.loads((parts / "run.json").read_text())
    checkpoint = json.loads((scratch / "checkpoint.json").read_text())
    if (
        c != layer_epoch(run, layer)[2]
        or digest(source.manifest) != run["source_identity"]
    ):
        raise ValueError("recipe/source identity changed; start a new run")
    if checkpoint["identity"] != run["identity"] or layer != checkpoint["next_layer"]:
        raise ValueError("requested layer is not the committed restart boundary")
    if not 0 <= layer < source.args.n_layers:
        raise ValueError("invalid layer")
    print(
        json.dumps(
            dict(
                layer=layer,
                stage=checkpoint.get("phase", "attention"),
                **guard.report(),
            )
        ),
        flush=True,
    )
    source.verify(layer)
    work = scratch / f"working-{layer}"
    if work.exists() and checkpoint.get("phase") != "ffn":
        shutil.rmtree(work)
    work.mkdir(exist_ok=True)
    target = scratch / f"boundary-{layer + 1}"
    if target.exists() and checkpoint.get("phase") != "ffn":
        shutil.rmtree(target)
    boundary = scratch / f"boundary-{layer}"
    windows = [boundary / f"window-{i}" for i in range(len(run["lengths"]))]
    total = sum(run["lengths"])
    block = bounded_block(source, layer, c)
    if checkpoint.get("phase") == "ffn":
        stage = [work / f"attention-{i}" for i in range(len(windows))]
        attention_stats = json.loads((work / "attention-stats.json").read_text())
        packed_values = mx.load(str(work / "attention-quantized.safetensors"))
        attention_packed = {
            name: tuple(packed_values[name + "." + suffix] for suffix in SUFFIXES)
            for name in attention_stats
        }
        for name, triple in attention_packed.items():
            original = module_at(block, name)
            set_module(
                block,
                name,
                packed_module(
                    original.weight,
                    triple,
                    attention_stats[name]["bits"],
                    c["group_size"],
                ),
            )
        counts = np.asarray(
            json.loads((work / "ffn-inputs-ready.json").read_text())["counts"], np.int64
        )
    else:
        stage, attention_packed, attention_stats = prepare_attention(
            source, layer, c, block, windows, work, guard
        )
    paths = {key: work / f"{key}.bin" for key in ("x", "ids", "gates", "y")}
    quality = None
    if (parts / "quality-config.json").exists():
        from scripts.deepseek_v41_gptq_quality import prepare_for_build

        quality = prepare_for_build(source, parts, layer, block, c, guard)
        for name, report in attention_stats.items():
            report["heldout"] = quality["inputs"]["projections"][name]
    else:
        for report in attention_stats.values():
            report["heldout"] = dict(status="not_configured")
    if checkpoint.get("phase") != "ffn":
        counts = spill_ffn(block, stage, paths, total, source.args, c, guard)
        sync_files(
            [
                work / "attention-quantized.safetensors",
                *(paths[key] for key in ("x", "ids", "gates")),
            ]
        )
        atomic_json(work / "ffn-inputs-ready.json", dict(counts=counts.tolist()))
    entries, costs = output_plan(source, c, layer, counts)
    candidate = groups_for(entries)
    for previous in range(layer):
        record = json.loads((parts / f"layer-{previous}-complete.json").read_text())
        validate_record(run, previous, record)
        candidate[f"layer-{previous}"] = entries_from_json(record["entries"])
    actual_resident = sum(
        e.nbytes
        for group in candidate.values()
        for e in group
        if e.component != "engram_ssd"
    )
    if actual_resident > c["resident_budget_gb"] * 1e9:
        if checkpoint.get("phase") != "ffn":
            # Leave the committed input boundary amendable after a rejected plan.
            shutil.rmtree(work)
        raise ValueError("low-token bank promotion exceeds cumulative resident budget")
    if checkpoint.get("phase") != "ffn":
        atomic_json(
            scratch / "checkpoint.json",
            dict(next_layer=layer, identity=run["identity"], phase="ffn"),
        )
    if not c["pilot"] and boundary.exists():
        # Commit the durable FFN restart point only after accepting its budget.
        shutil.rmtree(boundary)
    es = groups_for(entries)[f"layer-{layer}"]
    sink = ShardSink(parts / f"model-layer-{layer}.safetensors", es)
    try:
        for name, packed in attention_packed.items():
            write_packed(sink, f"layers.{layer}." + name, packed)
        params = dict(tree_flatten(block.parameters()))
        for e in es:
            local = e.name[len(f"layers.{layer}.") :]
            if e.component in ("routed_experts", "gptq_dense"):
                continue
            value = params[local]
            value = value.reshape(e.shape).astype(
                {"BF16": mx.bfloat16, "F32": mx.float32, "U32": mx.uint32}[e.dtype]
            )
            sink.put(e.name, value)
        ffn_stats = quantize_ffn(
            source,
            layer,
            block,
            sink,
            es,
            paths,
            counts,
            total,
            c,
            guard,
            heldout=quality["inputs"] if quality else None,
        )
        sink.finish()
    finally:
        sink.close()
    finalize_windows(
        block,
        stage,
        paths,
        target,
        total,
        source.args,
        c,
        guard,
        run_identity=run["identity"],
        layer=layer,
    )
    if quality is not None:
        from scripts.deepseek_v41_gptq_quality import finish_for_build

        finish_for_build(source, parts, layer, block, quality, ffn_stats, guard)
    del block, params, attention_packed
    gc.collect()
    mx.clear_cache()
    movement = (
        pilot_block_movement(source, layer, c, windows, target, parts, es, guard)
        if c["pilot"]
        else None
    )
    stats = dict(
        layer=layer,
        run_identity=run["identity"],
        execution_backend="cpu_float32_blas"
        if mx.default_device() == mx.cpu
        else "native_metal",
        mlx_version=mx.__version__,
        tokens=total,
        routed_tokens=counts.tolist(),
        low_token_experts=np.flatnonzero(counts < c["min_tokens"]).tolist(),
        routed_token_histogram=dict(
            zip(
                ("0", "1-127", "128-511", "512-2047", "2048+"),
                np.histogram(counts, bins=[0, 1, 128, 512, 2048, np.inf])[0].tolist(),
            )
        ),
        block_output_4bit_relative_error=movement,
        projections={**attention_stats, **ffn_stats},
    )
    if quality is not None:
        stats["heldout_reference_identity"] = quality["reference"]["identity"]
    profile = dict(layer=layer, seconds=time.monotonic() - started, **guard.report())
    atomic_json(parts / f"layer-{layer}-profile.json", profile)
    atomic_json(parts / f"layer-{layer}-stats.json", stats)
    record = dict(
        run_identity=run["identity"],
        entries=entries_json(es),
        files={
            f"model-layer-{layer}.safetensors": file_hash(
                parts / f"model-layer-{layer}.safetensors"
            ),
            f"layer-{layer}-stats.json": file_hash(parts / f"layer-{layer}-stats.json"),
        },
    )
    source.verify(layer)
    atomic_json(parts / f"layer-{layer}-complete.json", record)
    # Last operation commits the new boundary. Crashes before this point replay
    # the same layer from its immutable input; orphaned output is overwritten.
    atomic_json(
        scratch / "checkpoint.json",
        dict(next_layer=layer + 1, identity=run["identity"]),
    )
    if boundary.exists():
        shutil.rmtree(boundary)
    if quality is not None:
        from scripts.deepseek_v41_gptq_quality import release_previous_quality

        release_previous_quality(quality, layer)
    shutil.rmtree(work)
    print(json.dumps(dict(completed_layer=layer, **profile)), flush=True)
    return stats


def validated_layer_entries(source, run, layer, record, parts):
    from scripts.deepseek_v41_gptq_amend import layer_epoch

    stats = json.loads((parts / f"layer-{layer}-stats.json").read_text())
    counts = np.asarray(stats["routed_tokens"])
    if (
        counts.shape != (source.args.n_routed_experts,)
        or not np.issubdtype(counts.dtype, np.integer)
        or np.any(counts < 0)
    ):
        raise ValueError("invalid completed layer routed-token counts")
    expected, _ = output_plan(source, layer_epoch(run, layer)[2], layer, counts)
    expected = groups_for(expected)[f"layer-{layer}"]
    if digest(record["entries"]) != digest(entries_json(expected)):
        raise ValueError("completed layer entries differ from source/recipe plan")
    return expected


def assemble(source, c, parts, destination, guard):
    with atomic_destination(destination) as stage:
        if Path(parts).stat().st_dev != stage.parent.stat().st_dev:
            raise ValueError(
                "direct assembly requires parts and destination on the same filesystem; "
                "choose a destination on the parts volume"
            )
        return _assemble(source, c, parts, stage, guard)


def _assemble(source, c, parts, destination, guard):
    from scripts.deepseek_v41_gptq_amend import epochs, validate_record
    from scripts.deepseek_v41_rd_build import group_path

    run = json.loads((parts / "run.json").read_text())
    epochs(run)
    if run["recipe"] != c or run["source_identity"] != digest(source.manifest):
        raise ValueError("assembly recipe/source identity mismatch")
    entries, _ = output_plan(source, c)
    layer_groups, completion = {}, {}
    for i in range(source.args.n_layers):
        record = json.loads((parts / f"layer-{i}-complete.json").read_text())
        validate_record(run, i, record)
        verify_artifact(parts, record)
        completion[str(i)] = record
        layer_groups[f"layer-{i}"] = validated_layer_entries(
            source, run, i, record, parts
        )
    other = {
        g: es for g, es in groups_for(entries).items() if not g.startswith("layer-")
    }
    all_entries = [e for es in [*layer_groups.values(), *other.values()] for e in es]
    totals = {}
    for e in all_entries:
        totals[e.component] = totals.get(e.component, 0) + e.nbytes
    costs = layout_report(
        all_entries,
        dict(
            components=totals,
            resident_tensor_bytes=sum(
                v for k, v in totals.items() if k != "engram_ssd"
            ),
            shared_engram_bytes=totals.get("engram_ssd", 0),
            total_payload_bytes=sum(totals.values()),
        ),
    )
    if costs["resident_tensor_bytes"] > c["resident_budget_gb"] * 1e9:
        raise ValueError("assembled resident payload exceeds budget")
    print(json.dumps(dict(assembly_costs=costs)), flush=True)
    destination.mkdir(parents=True)
    mapping, mtp_mapping = {}, {}
    for group, es in {**layer_groups, **other}.items():
        target = destination / group_path(group)
        target.parent.mkdir(parents=True, exist_ok=True)
        if group.startswith("layer-"):
            layer = group.split("-")[1]
            copy_verified(
                parts / target.name, target, completion[layer]["files"][target.name]
            )
        else:
            # Source stages may be supplied independently to the assembly host.
            # Engram conversion is row-bounded and retains the SSD-native layout.
            component = (
                int(group.split("-")[1])
                if group.startswith("engram-")
                else ("mtp" if group == "mtp" else "top")
            )
            source.verify(component)
            write_group(target, es, source.index, source.config, row_chunk=65536)
            source.verify(component)
            guard.check()
        for e in es:
            (mtp_mapping if group == "mtp" else mapping)[e.name] = target.name
    config = json.loads(json.dumps(source.config))
    config.pop("quantization_config", None)
    config["model_type"] = "deepseek_v41"
    modules = {
        e.name.rsplit(".", 1)[0]: dict(bits=e.bits, group_size=e.group_size)
        for e in all_entries
        if e.bits and e.suffix == "weight" and e.component != "engram_ssd"
    }
    config["quantization"] = dict(
        bits=8,
        group_size=64,
        modules={n: v for n, v in modules.items() if not n.startswith("mtp.")},
        engram_bits=c["engram_bits"],
        engram_group_size=c["engram_group_size"],
    )
    actual_allocation = allocation(source, c)
    for e in all_entries:
        if e.component == "routed_experts" and e.suffix == "weight":
            layer_id, projection = e.name.split(".")[1], e.name.split(".")[-2]
            actual_allocation["layers"][layer_id]["projection_bits"][projection] = (
                e.bits
            )
    config["rapid_quantization"] = dict(
        format="native-packed-affine",
        allocation=actual_allocation,
        shared_engram_identity=digest(
            dict(
                source=digest(source.manifest),
                bits=c["engram_bits"],
                group_size=c["engram_group_size"],
            )
        ),
        mtp=bool(mtp_mapping),
        vision=False,
        bytes=costs,
        gptq=dict(
            recipe=c,
            recipe_amendments=run.get("recipe_amendments", []),
            calibration_manifest_sha256=run.get(
                "calibration_manifest_sha256", run["calibration_identity"]
            ),
            calibration_content_sha256=run["calibration_identity"],
            source_revision=run["source_revision"],
            run_identity=run["identity"],
            heldout_reference_identity=run.get("heldout_reference_identity"),
        ),
    )
    if mtp_mapping:
        draft = {**config, **config.get("text_config", {})}
        draft["quantization"] = dict(
            bits=8,
            group_size=64,
            modules={n: v for n, v in modules.items() if n.startswith("mtp.")},
        )
        atomic_json(destination / "dspark" / "config.json", draft)
        atomic_json(
            destination / "dspark" / "model.safetensors.index.json",
            dict(weight_map=mtp_mapping),
        )
    copy_bound_metadata(source, destination)
    if (parts / "engram_token_map.json").exists():
        shutil.copyfile(
            parts / "engram_token_map.json", destination / "engram_token_map.json"
        )
    atomic_json(
        destination / "model.safetensors.index.json",
        dict(
            metadata=dict(
                total_size=costs["total_payload_bytes"] - totals.get("mtp", 0)
            ),
            weight_map=mapping,
        ),
    )
    atomic_json(destination / "config.json", config)
    atomic_json(
        destination / "conversion-provenance.json",
        conversion_record(source, destination),
    )
    atomic_json(destination / "gptq-provenance.json", run)
    statistics = destination / "gptq-statistics"
    statistics.mkdir()
    for i in range(source.args.n_layers):
        shutil.copyfile(parts / f"layer-{i}-stats.json", statistics / f"layer-{i}.json")
    for name in ("heldout-drift.jsonl", "heldout-reference.json"):
        if (parts / name).exists():
            shutil.copyfile(parts / name, destination / ("gptq-" + name))
    atomic_json(destination / "gptq-layer-provenance.json", completion)
    (destination / "README.md").write_text(
        "# Experimental native GPTQ build\n\nSee gptq-provenance.json and per-layer calibration reports. Quality qualification pending.\n"
    )
    known = {
        name: sha
        for record in completion.values()
        for name, sha in record["files"].items()
    }
    pinned = []
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            relative = str(path.relative_to(destination))
            pinned.append(
                dict(
                    name=relative,
                    size=path.stat().st_size,
                    sha256=known.get(relative) or file_hash(path),
                )
            )
    atomic_json(destination / "gptq-artifacts.json", dict(files=pinned))
    return costs


def calibration_windows(source, slices, tokens, total_tokens=None):
    from tokenizers import Tokenizer

    if not 1 <= tokens <= source.args.max_seq_len or (
        total_tokens is not None and total_tokens < 1
    ):
        raise ValueError("invalid calibration token limit")
    tokenizer = Tokenizer.from_file(str(source.source / "tokenizer.json"))
    paths = read_slices(slices, [])
    windows, evidence = [], {}
    remaining = total_tokens
    for name, files in paths.items():
        if remaining == 0:
            break
        requested = tokens if remaining is None else min(tokens, remaining)
        ids = tokenizer.encode("\n\n".join(p.read_text() for p in files)).ids
        if len(ids) < requested:
            raise ValueError(
                f"{name}: need {requested} tokens, only {len(ids)} present"
            )
        windows.append(np.asarray(ids[:requested], np.int32))
        if remaining is not None:
            remaining -= requested
        evidence[name] = [
            dict(sha256=file_hash(p), bytes=p.stat().st_size) for p in files
        ]
    if remaining not in (None, 0):
        raise ValueError(
            "calibration manifest has insufficient windows for total token limit"
        )
    return windows, digest(
        dict(
            manifest_sha256=file_hash(slices),
            files=evidence,
            tokens=tokens,
            total_tokens=total_tokens,
        )
    )


def main():
    from scripts.deepseek_v41_gptq_quality import add_parsers

    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    add_parsers(sub)
    from scripts.deepseek_v41_gptq_scan import add_parser as add_scan_parser

    add_scan_parser(sub)
    inv = sub.add_parser("inventory")
    inv.add_argument("--source", type=Path, required=True)
    inv.add_argument(
        "--source-revision",
        help="immutable source revision; default snapshot directory name",
    )
    inv.add_argument("--output", type=Path, required=True)
    shards = sub.add_parser("shards")
    shards.add_argument("--manifest", "--index", type=Path, required=True)
    shards.add_argument(
        "--component", required=True, help="integer layer ID, top or mtp"
    )
    sub.add_parser("default-config")
    select = sub.add_parser("select-recipe")
    selection_input = select.add_mutually_exclusive_group(required=True)
    selection_input.add_argument("--parts", type=Path)
    selection_input.add_argument("--scan", type=Path)
    select.add_argument("--output", type=Path, required=True)
    amend = sub.add_parser("amend-recipe")
    for name in ("parts", "scratch", "recipe"):
        amend.add_argument("--" + name, type=Path, required=True)
    amend.add_argument("--from-layer", type=int, required=True)
    amend.add_argument("--reason", required=True)
    for cmd in ("estimate", "init", "run", "assemble"):
        q = sub.add_parser(cmd)
        q.add_argument("--manifest", type=Path, required=True)
        q.add_argument("--source", type=Path, required=True)
        q.add_argument("--recipe", type=Path)
        q.add_argument("--cpu", action="store_true")
        if cmd in ("estimate", "init"):
            q.add_argument("--slices", type=Path, required=True)
            q.add_argument(
                "--tokens",
                type=int,
                required=True,
                help="tokens per independent calibration window",
            )
        if cmd in ("estimate", "init"):
            q.add_argument(
                "--total-tokens",
                type=int,
                help="optional total cap; last independent window may be shorter",
            )
        if cmd in ("init", "run", "assemble"):
            q.add_argument("--parts", type=Path, required=True)
            q.add_argument("--scratch", type=Path, required=cmd != "assemble")
        if cmd in ("estimate", "init", "run"):
            q.add_argument(
                "--reference",
                type=Path,
                required=cmd == "init",
                help="held-out boundary cache; content bound to this run",
            )
            q.add_argument(
                "--quality-scratch",
                type=Path,
                help="portable held-out prefix checkpoints; default SCRATCH/heldout",
            )
            q.add_argument(
                "--rtn-build",
                type=Path,
                help="optional accessible native baseline; may be evaluated on Studio after export",
            )
        if cmd == "run":
            q.add_argument(
                "--layer", type=int, help="default: resume committed next layer"
            )
            q.add_argument(
                "--through-layer", type=int, help="inclusive; default run one layer"
            )
        if cmd == "assemble":
            q.add_argument("--destination", type=Path, required=True)
    a = p.parse_args()
    from scripts.deepseek_v41_gptq_lock import command_locks

    with command_locks(a):
        execute_command(a)


def execute_command(a):
    from scripts.deepseek_v41_gptq_quality import execute

    if a.command == "amend-recipe":
        from scripts.deepseek_v41_gptq_amend import amend_recipe

        result = amend_recipe(
            a.parts,
            a.scratch,
            recipe(json.loads(a.recipe.read_text())),
            a.from_layer,
            a.reason,
        )
        print(json.dumps(dict(identity=result["identity"], from_layer=a.from_layer)))
        return
    if a.command == "scan":
        from scripts.deepseek_v41_gptq_scan import execute as execute_scan

        execute_scan(a)
        return
    if a.command in ("reference", "drift", "milestone"):
        execute(a)
        return
    if a.command == "inventory":
        atomic_json(a.output, inventory(a.source, a.source_revision))
        return
    if a.command == "shards":
        component = int(a.component) if a.component.isdecimal() else a.component
        print("\n".join(required_shards(json.loads(a.manifest.read_text()), component)))
        return
    if a.command == "default-config":
        print(json.dumps(DEFAULTS, indent=2))
        return
    if a.command == "select-recipe":
        directory = a.scan or a.parts
        run = json.loads((directory / "run.json").read_text())
        source = PartialSource(
            directory, json.loads((directory / "source-manifest.json").read_text())
        )
        atomic_json(
            a.output, select_recipe(directory, source, run["recipe"], scan=bool(a.scan))
        )
        return
    c = recipe(json.loads(a.recipe.read_text()) if a.recipe else None)
    if a.cpu:
        mx.set_default_device(mx.cpu)
    mx.random.seed(c["seed"])
    source = PartialSource(a.source, json.loads(a.manifest.read_text()))
    if a.command == "estimate":
        windows, _ = calibration_windows(source, a.slices, a.tokens, a.total_tokens)
        estimate = estimates(source, c, [len(w) for w in windows])
        if a.reference:
            from scripts.deepseek_v41_gptq_quality import planning_with_quality

            estimate = planning_with_quality(
                source,
                estimate,
                json.loads((a.reference / "reference.json").read_text()),
                bool(a.rtn_build),
            )
        print(json.dumps(estimate, indent=2))
        return
    with MemoryGuard(c) as guard:
        if a.command == "init":
            windows, identity = calibration_windows(
                source, a.slices, a.tokens, a.total_tokens
            )
            reference = json.loads((a.reference / "reference.json").read_text())
            initialize(
                source,
                c,
                windows,
                a.scratch,
                a.parts,
                identity,
                guard,
                calibration_manifest_sha256=file_hash(a.slices),
                heldout_reference=reference,
                separation=dict(
                    reference=str(a.reference),
                    corpus_hashes=sorted(
                        {
                            file_hash(p)
                            for files in read_slices(a.slices, []).values()
                            for p in files
                        }
                    ),
                ),
            )
            from scripts.deepseek_v41_gptq_quality import bind_quality

            bind_quality(
                source,
                a.parts,
                a.reference,
                a.quality_scratch or a.scratch / "heldout",
                a.rtn_build,
                windows=windows,
                slices=a.slices,
            )
        elif a.command == "run":
            from scripts.deepseek_v41_gptq_quality import bind_quality

            if a.reference:
                bind_quality(
                    source,
                    a.parts,
                    a.reference,
                    a.quality_scratch or a.scratch / "heldout",
                    a.rtn_build,
                )
            if not (a.parts / "quality-config.json").exists():
                raise ValueError(
                    "run needs --reference held-out cache; calibration-only errors are insufficient"
                )
            from scripts.deepseek_v41_gptq_quality import (
                validate_reference,
                validate_separation,
            )

            run = json.loads((a.parts / "run.json").read_text())
            config = json.loads((a.parts / "quality-config.json").read_text())
            if (
                "calibration_corpus_hashes" not in run
                or "calibration_token_windows" not in run
            ):
                raise ValueError(
                    "missing calibration separation evidence; start a new run"
                )
            reference_root = Path(config["reference"])
            reference = json.loads((reference_root / "reference.json").read_text())
            validate_reference(source, reference)
            validate_separation(
                reference_root,
                reference,
                run["calibration_token_windows"],
                run["calibration_corpus_hashes"],
            )
            layer = a.layer
            if layer is None:
                layer = json.loads((a.scratch / "checkpoint.json").read_text())[
                    "next_layer"
                ]
            last = a.through_layer if a.through_layer is not None else layer
            if not 0 <= layer <= last < source.args.n_layers:
                raise ValueError("invalid layer range")
            for i in range(layer, last + 1):
                run_layer(source, c, a.parts, a.scratch, i, guard)
        elif a.command == "assemble":
            assemble(source, c, a.parts, a.destination, guard)


if __name__ == "__main__":
    main()
