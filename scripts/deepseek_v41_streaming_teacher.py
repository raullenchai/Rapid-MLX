#!/usr/bin/env python3
"""Source-dequantized, layer-major teacher. Never loads a complete source shard."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.deepseek_v41_merge_calibration import save_scores
from scripts.deepseek_v41_public_ruler import (
    compact_hashes,
    compact_logits,
    create_compact,
    read_slice_manifest,
    retain_ruler_manifest,
    slice_tokens,
    validate_ruler_manifest,
    verify_input_hashes,
)
from scripts.deepseek_v41_rd_common import checkpoint_hashes, file_hash
from scripts.repack_deepseek_v41_native import CheckpointIndex


class SourceWeights:
    def __init__(self, source, args=None):
        from rapid_mlx.models.deepseek_v41_native.config import ModelArgs

        self.source = Path(source)
        self.config = json.loads((self.source / "config.json").read_text())
        self.args = args or ModelArgs.from_dict(self.config)
        self.index = CheckpointIndex(self.source)
        if (
            self.config.get("quantization_config", {}).get("expert_dtype", "fp4")
            != "fp4"
        ):
            raise ValueError("streaming teacher requires the FP4/FP8 source")
        if self.config.get("quantization") or self.args.experts_per_layer:
            raise ValueError("teacher requires unpruned source weights")

    def tensor(self, name):
        """Use converter sanitization, preserving source F32 auxiliary tensors."""
        from rapid_mlx.models.deepseek_v41_native.convert import sanitize_group
        from scripts.deepseek_v41_expert_quant_error import read_tensor

        names = [name]
        scale = name.removesuffix(".weight") + ".scale"
        if name.endswith(".weight") and scale in self.index.tensors:
            names.append(scale)
        return sanitize_group(
            names, {n: read_tensor(self.index.tensors[n]) for n in names}
        )[name]

    def layer(self, layer_id):
        import mlx.core as mx
        from mlx.utils import tree_flatten

        from rapid_mlx.models.deepseek_v41_native.load import reshape_grouped_wo_a
        from rapid_mlx.models.deepseek_v41_native.model import Block
        from scripts.deepseek_v41_expert_quant_error import load_expert

        # Do not even construct a giant table placeholder. Disk lookup has no parameters.
        small = replace(
            self.args,
            engram_num_embeddings=tuple(1 for _ in self.args.engram_layer_ids),
        )
        block = Block(layer_id, small)
        if block.engram is not None:
            block.engram.embed = SourceEngram(self.index, layer_id)
        prefix = f"layers.{layer_id}."
        values = {}
        # Three unstacked projection banks become three stacked banks. Each
        # collection is released before stacking the next bank: the peak is
        # all three resident banks plus one transient stack, as budgeted below.
        parts = [[], [], []]
        for expert in range(self.args.n_routed_experts):
            ws = load_expert(self.index, layer_id, expert)
            for collection, w in zip(parts, ws):
                collection.append(w)
            del ws, w
        for p, collection in zip(("gate_proj", "up_proj", "down_proj"), parts):
            value = mx.stack(collection)
            mx.eval(value)
            collection.clear()
            values[f"ffn.experts.{p}.weight"] = value
        del parts, collection, value
        for name in self.index.tensors:
            if (
                name.startswith(prefix)
                and not name.endswith(".scale")
                and ".ffn.experts." not in name
                and ".engram.embed." not in name
            ):
                w = self.tensor(name)
                mx.eval(w)
                values[name[len(prefix) :]] = w
        items = reshape_grouped_wo_a(
            [(prefix + n, w) for n, w in values.items()], self.args
        )
        values = {n[len(prefix) :]: w for n, w in items}
        expected = {n for n, _ in tree_flatten(block.parameters())}
        if set(values) != expected:
            raise ValueError(
                f"layer {layer_id} key mismatch: missing={expected - set(values)}, extra={set(values) - expected}"
            )
        block.load_weights(list(values.items()), strict=True)
        block.eval()
        if mx.default_device() == mx.cpu:
            install_cpu_gather(block)
        return block


# Defined without inheriting nn.Module: has no resident tensors or hidden table cache.
class SourceEngram:
    def __init__(self, index, layer_id):
        self.weight = index.tensors[f"layers.{layer_id}.engram.embed.weight"]
        self.scale = index.tensors[f"layers.{layer_id}.engram.embed.scale"]

    def __call__(self, indices):
        import mlx.core as mx

        from rapid_mlx.models.deepseek_v41_native.dequant import dequant_fp8_rows

        host = np.asarray(indices, dtype=np.int64)
        unique, inverse = np.unique(host, return_inverse=True)
        if np.any(unique < 0) or np.any(unique >= self.weight.shape[0]):
            raise ValueError("Engram row out of range")

        def gather(t):
            mapped = np.memmap(
                t.path, mode="r", dtype=np.uint8, offset=t.offset, shape=t.shape
            )
            rows = np.array(mapped[unique])
            del mapped
            return mx.array(rows)

        # Preserve FP32 row dequantization; Engram casts the concatenated rows itself.
        rows = dequant_fp8_rows(gather(self.weight), gather(self.scale))
        return rows[mx.array(inverse)].reshape(*host.shape, self.weight.shape[-1])


@dataclass
class Sequence:
    ids: np.ndarray
    h: object
    pre: object
    cache: object
    shared: object
    hashes: object = None


def materialize_state(seq):
    """Evaluate every surviving cache array before releasing its producer weights."""
    import mlx.core as mx

    arrays, seen = [], set()

    def visit(obj):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, mx.array):
            arrays.append(obj)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                visit(v)
        elif hasattr(obj, "__dict__"):
            for v in vars(obj).values():
                visit(v)

    visit(seq)
    mx.eval(arrays)


def stream_forward(
    args,
    tokens,
    embedding,
    layer_loader,
    *,
    hasher=None,
    observer=None,
    hidden_observer=None,
    progress=None,
    prefill_chunk=None,
):
    """Layer-major full causal prefill; sequences never share attention/hash state.

    Returns final streams, not logits. Caller releases embed before loading blocks,
    and loads norm/head only after the last block has been freed.
    """
    import mlx.core as mx

    from rapid_mlx.models.deepseek_v41_native.cache import LayerCache
    from rapid_mlx.models.deepseek_v41_native.hyper_connections import (
        make_identity_pre_mix,
    )
    from rapid_mlx.models.deepseek_v41_native.model import SharedState

    if prefill_chunk is not None and (prefill_chunk < 1 or observer is not None):
        raise ValueError("chunking requires a positive chunk and reference mode")
    sequences = []
    embed = embedding()
    del embedding
    for ids in tokens:
        ids = np.asarray(ids, np.int32)
        if ids.ndim != 1 or not 0 < len(ids) <= args.max_seq_len:
            raise ValueError("tokens must be a nonempty sequence within context length")
        n = len(ids)
        hashes = None
        if args.engram_layer_ids:
            if hasher is None:
                raise ValueError("Engram requires source token map")
            hashes = mx.array(hasher(ids[None], 0, np.zeros((1, n), np.int64)))
        h = embed(mx.array(ids[None]))
        h = mx.broadcast_to(h[:, :, None, :], (1, n, args.hc_mult, args.dim))
        seq = Sequence(
            ids,
            h,
            make_identity_pre_mix(1, n, args.hc_mult),
            SimpleNamespace(layers=[None] * args.n_layers),
            [SharedState() for _ in range(0, n, prefill_chunk)]
            if prefill_chunk
            else SharedState(),
            hashes,
        )
        materialize_state(seq)
        sequences.append(seq)
    # No closure in this function retains embedding during block loading.
    del embed
    for layer_id in range(args.n_layers):
        started = time.monotonic()
        block = layer_loader(layer_id)
        if observer is not None:
            install_prefill_probe(block, observer)
        for domain, seq in enumerate(sequences):
            seq.cache.layers[layer_id] = LayerCache(1, args, layer_id, len(seq.ids))
            if observer is not None:
                block.ffn._stream_domain = domain
            chunk = prefill_chunk or len(seq.ids)
            hs, pres = [], []
            for i, start in enumerate(range(0, len(seq.ids), chunk)):
                end = min(start + chunk, len(seq.ids))
                h, pre = seq.h[:, start:end], seq.pre[:, start:end]
                if block.engram is not None:
                    h = block.engram(
                        h, seq.hashes[:, start:end, block.engram.layer_hash_index]
                    )
                shared = seq.shared[i] if prefill_chunk else seq.shared
                h, pre = block(h, pre, start, seq.cache, shared)
                mx.eval(h, pre)
                hs.append(h)
                pres.append(pre)
                # Flush mutated caches while producer weights are still resident.
                materialize_state(seq)
                mx.clear_cache()
            seq.h, seq.pre = mx.concatenate(hs, axis=1), mx.concatenate(pres, axis=1)
            materialize_state(seq)
            # Only the current shared source caches survive; past windows are dead.
            seq.cache.layers[layer_id] = None
            if hidden_observer:
                hidden_observer(domain, layer_id, seq.h, seq.pre)
        del block
        gc.collect()
        mx.clear_cache()
        if progress:
            progress(layer_id, time.monotonic() - started)
    return sequences


def install_cpu_gather(block):
    """CPU GatherMM supports only F32. Preserve each projection's output dtype.

    Used for bounded CPU checks only; Metal retains the native SwitchLinear.
    The in-memory comparator must use the same CPU adapter.
    """
    import mlx.core as mx
    from mlx_lm.models.switch_layers import SwitchLinear

    class CPUSwitchLinear(SwitchLinear):
        def __call__(self, x, indices, sorted_indices=False):
            dtype = mx.result_type(x, self.weight)
            y = mx.gather_mm(
                x.astype(mx.float32),
                self.weight.astype(mx.float32).swapaxes(-1, -2),
                rhs_indices=indices,
                sorted_indices=sorted_indices,
            )
            if "bias" in self:
                y = y + mx.expand_dims(self.bias[indices], -2)
            return y.astype(dtype)

    for projection in (
        block.ffn.experts.gate_proj,
        block.ffn.experts.up_proj,
        block.ffn.experts.down_proj,
    ):
        projection.__class__ = CPUSwitchLinear


def install_prefill_probe(block, observer):
    import mlx.core as mx

    from rapid_mlx.models.deepseek_v41_native.moe import MoE

    class ObservedMoE(MoE):
        def __call__(self, x):
            shape = x.shape
            xf = x.reshape(-1, self.dim)
            weights, indices = self.gate(xf)
            outputs = self.experts(xf, indices)
            norms = mx.sqrt(mx.sum(mx.square(outputs.astype(mx.float32)), axis=-1))
            normalized = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
            mx.eval(norms, normalized, indices)
            observer(
                self._stream_domain,
                self._stream_layer,
                np.array(indices),
                np.array(normalized),
                np.array(norms),
            )
            y = mx.sum(outputs.astype(mx.float32) * weights[..., None], axis=-2)
            y = y + self.shared_experts(xf).astype(mx.float32)
            return y.reshape(shape).astype(x.dtype)

    block.ffn.__class__ = ObservedMoE
    block.ffn._stream_layer = block.layer_id


def memory_estimate(weights, slices, tokens, prefill_chunk=None):
    """Conservative planning estimate, not a measured device peak."""
    a = weights.args
    layer_bytes = [0] * a.n_layers
    largest_stack = 2 * a.n_routed_experts * a.dim * a.moe_inter_dim
    read_bytes = 0
    for n, t in weights.index.tensors.items():
        m = re.match(r"layers\.(\d+)\.", n)
        if m and int(m[1]) < a.n_layers and ".engram.embed." not in n:
            read_bytes += t.nbytes
            if not n.endswith(".scale"):
                size = math.prod(t.shape)
                if ".ffn.experts." in n and n.endswith(".weight"):
                    size *= 2
                layer_bytes[int(m[1])] += size * (4 if t.dtype == "F32" else 2)
        elif n in ("embed.weight", "head.weight", "norm.weight"):
            read_bytes += t.nbytes
    # Hidden + outgoing pre + current shared KV/index + topk + candidate masks.
    per_token = (
        2 * a.hc_mult * a.dim
        + 4 * a.hc_mult
        + 8 * a.head_dim
        + 8 * a.index_head_dim
        + 4 * a.index_topk
        + tokens
    )
    states = slices * tokens * per_token
    # Full-slice prefill temporaries, including gather-based sparse attention.
    chunk = min(prefill_chunk or tokens, tokens)
    scratch = (
        chunk
        * (
            32 * a.n_heads * a.head_dim
            + 32 * a.n_activated_experts * (a.dim + a.moe_inter_dim)
            + 32 * a.hc_mult * a.hc_mult * a.dim
        )
        + 16 * a.index_n_heads * chunk * tokens
    )
    # Input + accumulated chunk outputs + concatenation coexist at a boundary.
    if prefill_chunk:
        states += 2 * slices * tokens * (2 * a.hc_mult * a.dim + 4 * a.hc_mult)
    # layer_bytes already includes ALL THREE dequantized expert projections.
    # Stacking duplicates only the current projection before collection.clear().
    peak = max(layer_bytes) + largest_stack + states + scratch + 8 * 2**30
    return dict(
        layer_bf16_bytes=max(layer_bytes),
        stack_transient_bytes=largest_stack,
        persistent_state_bytes=states,
        prefill_scratch_allowance_bytes=scratch,
        estimated_peak_bytes=peak,
        source_pass_bytes=read_bytes,
        source_pass_seconds_at_0_9_GBps=read_bytes / 0.9e9,
        conservative_475GB_seconds=475e9 / 0.9e9,
        tokens_per_slice=tokens,
        slices=slices,
        prefill_chunk=chunk,
        note="analytical estimate plus 8 GiB reserve, not a peak bound; Engram random I/O extra",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("calibrate", "reference"))
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--slices", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--tokens",
        type=int,
        required=True,
        help="per slice, all slices run in one source pass",
    )
    p.add_argument("--keep-experts", type=int, default=288)
    p.add_argument("--model-revision")
    p.add_argument("--memory-limit-gib", type=float, default=180)
    p.add_argument("--logit-chunk", type=int, default=16)
    p.add_argument("--prefill-chunk", type=int)
    p.add_argument(
        "--compact-k", type=int, help="store top-k plus tail instead of full logits"
    )
    p.add_argument("--estimate-only", action="store_true")
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()
    import mlx.core as mx
    from tokenizers import Tokenizer

    from rapid_mlx.models.deepseek_v41_native.engram import EngramHasher
    from rapid_mlx.models.deepseek_v41_native.hyper_connections import hc_pre_norm
    from rapid_mlx.models.deepseek_v41_native.load import load_token_map

    if a.cpu:
        mx.set_default_device(mx.cpu)
    source_identity = None if a.estimate_only else checkpoint_hashes(a.source)
    weights = SourceWeights(a.source)
    if source_identity is not None and checkpoint_hashes(a.source) != source_identity:
        raise ValueError("checkpoint changed during initialization")
    args = weights.args
    ruler_digest = validate_ruler_manifest(a.slices)
    manifest_raw, manifest, slices = read_slice_manifest(a.slices)
    if (
        not slices
        or a.tokens < (2 if a.mode == "calibrate" else 1)
        or a.tokens > args.max_seq_len
        or a.logit_chunk < 1
        or (
            a.prefill_chunk is not None
            and (a.prefill_chunk < 1 or a.mode != "reference")
        )
        or (
            a.compact_k is not None
            and (not 1 <= a.compact_k <= args.vocab_size or a.mode != "reference")
        )
        or not args.n_activated_experts <= a.keep_experts <= args.n_routed_experts
        or not math.isfinite(a.memory_limit_gib)
        or a.memory_limit_gib <= 0
    ):
        raise ValueError("invalid tokens, keep count, memory limit or chunk")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in slices):
        raise ValueError("unsafe slice name")
    estimate = memory_estimate(weights, len(slices), a.tokens, a.prefill_chunk)
    print(json.dumps({"memory_estimate": estimate}), flush=True)
    if a.estimate_only:
        return
    if estimate["estimated_peak_bytes"] > a.memory_limit_gib * 2**30:
        raise ValueError("estimated memory exceeds limit; reduce --tokens")
    if a.output.exists():
        raise FileExistsError(a.output)
    input_hashes = {
        a.slices: hashlib.sha256(manifest_raw).hexdigest(),
        a.source / "tokenizer.json": file_hash(a.source / "tokenizer.json"),
    }
    tokenizer = Tokenizer.from_file(str(a.source / "tokenizer.json"))
    expected_tokenizer = manifest.get("tokenizer_sha256")
    if (
        expected_tokenizer
        and expected_tokenizer != input_hashes[a.source / "tokenizer.json"]
    ):
        raise ValueError("slice manifest tokenizer mismatch")
    input_hashes.update({p: file_hash(p) for paths in slices.values() for p in paths})
    tokens = []
    for paths in slices.values():
        tokens.append(slice_tokens(paths, tokenizer, a.tokens))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    hasher = None
    if args.engram_layer_ids:
        token_map = load_token_map(str(a.source), args, cache_dir=str(a.output.parent))
        hasher = EngramHasher(args, token_map)
    shape = (len(slices), 2, args.n_layers, args.n_routed_experts)
    reap, norms, squared = [np.zeros(shape, np.float64) for _ in range(3)]
    counts = np.zeros(shape, np.int64)

    def observe(domain, layer, indices, gates, magnitudes):
        halves = np.minimum(1, np.arange(len(indices)) * 2 // a.tokens)
        ix = (domain, halves[:, None], layer, indices)
        np.add.at(reap, ix, gates.astype(np.float64) * magnitudes)
        np.add.at(norms, ix, magnitudes)
        np.add.at(squared, ix, magnitudes.astype(np.float64) ** 2)
        np.add.at(counts, ix, 1)

    # Read/evaluate embed on demand; function drops its sole matrix reference
    # before the first layer loads. No resident full Model is constructed.
    def embedding():
        matrix = weights.tensor("embed.weight")
        mx.eval(matrix)
        return lambda ids: matrix[ids]

    sequences = stream_forward(
        args,
        tokens,
        embedding,
        weights.layer,
        hasher=hasher,
        prefill_chunk=a.prefill_chunk,
        observer=observe if a.mode == "calibrate" else None,
        progress=lambda layer, s: print(
            json.dumps({"layer": layer, "seconds": s}), flush=True
        ),
    )
    metadata = dict(
        schema_version=1,
        num_layers=args.n_layers,
        slices_manifest_sha256=input_hashes[a.slices],
        model_revision=a.model_revision or a.source.resolve().name,
        checkpoint_sha256=source_identity,
        checkpoint_index_json=(a.source / "model.safetensors.index.json")
        .read_bytes()
        .decode("utf-8"),
        source_index_json=(a.source / "model.safetensors.index.json")
        .read_bytes()
        .decode("utf-8"),
        source_checkpoint_sha256=source_identity,
        config_sha256=source_identity["config.json"],
        checkpoint_config_sha256=source_identity["config.json"],
        tokenizer_sha256=input_hashes[a.source / "tokenizer.json"],
        teacher="source_fp4_fp8_dequantized_bf16_native",
        engram="source_fp8_rowwise",
        execution="layer_major_full_prefill",
        prefill_chunk=a.prefill_chunk,
        ruler=manifest.get("ruler"),
        ruler_manifest_sha256=ruler_digest,
        compact_k=a.compact_k,
        vocab_size=args.vocab_size,
        memory_estimate=estimate,
        slice_names=list(slices),
        tokens_per_slice=[a.tokens] * len(slices),
        routing_topk=args.n_activated_experts,
        score_definitions={
            "reap": "sum normalized_gate * output_L2",
            "man": "routed-token mean output_L2",
            "msan": "routed-token mean output_L2_squared",
        },
        normalization="raw score max over domains; equal tokens per slice; halves checked separately",
        corpus={
            n: [dict(file=str(p), sha256=input_hashes[p]) for p in ps]
            for n, ps in slices.items()
        },
    )
    if a.mode == "calibrate":
        verify_input_hashes(input_hashes)
        if checkpoint_hashes(a.source) != source_identity:
            raise ValueError("checkpoint changed during calibration")
        save_scores(
            a.output, reap, norms, squared, counts, metadata, a.keep_experts, a.tokens
        )
        return
    a.output.mkdir()
    (a.output / "slices-manifest.json").write_bytes(manifest_raw)
    retain_ruler_manifest(a.slices, a.output, ruler_digest)
    metadata.update(mode="unpruned_reference", slices={})
    norm, head = weights.tensor("norm.weight"), weights.tensor("head.weight")
    mx.eval(norm, head)
    for (name, paths), seq in zip(slices.items(), sequences):
        np.save(a.output / f"{name}-tokens.npy", seq.ids)
        compact = (
            create_compact(a.output, name, a.tokens, args.vocab_size, a.compact_k)
            if a.compact_k
            else None
        )
        logits = (
            None
            if compact
            else np.lib.format.open_memmap(
                a.output / f"{name}-logits.npy",
                mode="w+",
                dtype=np.float32,
                shape=(a.tokens, args.vocab_size),
            )
        )
        for start in range(0, a.tokens, a.logit_chunk):
            end = start + a.logit_chunk
            h = hc_pre_norm(
                seq.h[:, start:end], seq.pre[:, start:end], norm, args.norm_eps
            )
            value = h.astype(mx.float32) @ head.T
            mx.eval(value)
            host = np.asarray(value).reshape(-1, args.vocab_size)
            if compact:
                for field, values in compact_logits(host, a.compact_k).items():
                    compact[field][start:end] = values
            else:
                logits[start:end] = host
        if compact:
            for values in compact.values():
                values.flush()
        else:
            logits.flush()
        del logits
        np.save(
            a.output / f"{name}-mass.npy",
            np.ones((a.tokens, args.n_layers), np.float32),
        )
        metadata["slices"][name] = dict(
            tokens=a.tokens,
            mass_sha256=file_hash(a.output / f"{name}-mass.npy"),
            tokens_sha256=file_hash(a.output / f"{name}-tokens.npy"),
            corpus_sha256=[input_hashes[p] for p in paths],
            **(
                dict(compact_sha256=compact_hashes(a.output, name))
                if compact
                else dict(logits_sha256=file_hash(a.output / f"{name}-logits.npy"))
            ),
        )
    verify_input_hashes(input_hashes)
    if checkpoint_hashes(a.source) != source_identity:
        raise ValueError("checkpoint changed during reference inference")
    (a.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
