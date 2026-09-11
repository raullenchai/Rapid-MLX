#!/usr/bin/env python3
"""Measure V4.1 target-only decode against Rapid-owned DSpark drafting.

This first gate deliberately uses serial target verification.  It establishes
draft latency, greedy acceptance, output equivalence, and memory headroom before
we invest in exact batched verification and rollback.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import json
import os
import sys
import time
import weakref
from pathlib import Path
from types import MethodType, SimpleNamespace

import mlx.core as mx
from transformers import PreTrainedTokenizerFast

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from deepseek_v41_affine_route_qmv import affine2_route_down_qmv  # noqa: E402
from mlx_lm.models.switch_layers import SwitchGLU  # noqa: E402

from vllm_mlx.models.deepseek_v41_native import dspark as rapid_dspark  # noqa: E402
from vllm_mlx.models.deepseek_v41_native.load import load  # noqa: E402

BOS = "<｜begin▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"
CHAT_START = "</think>"


def _load_module_file(name: str, path: Path):
    path = path.absolute()
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load checkpoint runtime module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).absolute() != path:
        raise ImportError(f"checkpoint runtime resolved outside requested file: {path}")
    return module


@contextlib.contextmanager
def _checkpoint_runtime(root: Path):
    """Bind the two trusted files exactly for the benchmark lifetime."""
    root = root.absolute()
    if not root.is_dir():
        raise NotADirectoryError(root)
    previous_runtime = sys.modules.get("runtime")
    previous_dspark = sys.modules.get("_rapid_checkpoint_dspark")
    try:
        runtime = _load_module_file("runtime", root / "runtime.py")
        dspark = _load_module_file("_rapid_checkpoint_dspark", root / "dspark.py")
        yield runtime, dspark
    finally:
        if previous_runtime is None:
            sys.modules.pop("runtime", None)
        else:
            sys.modules["runtime"] = previous_runtime
        if previous_dspark is None:
            sys.modules.pop("_rapid_checkpoint_dspark", None)
        else:
            sys.modules["_rapid_checkpoint_dspark"] = previous_dspark


def _prompt(text: str) -> str:
    return f"{BOS}{USER}{text}{ASSISTANT}{CHAT_START}"


def _run_ar(model, input_ids, output_tokens, eos_id):
    cache = model.make_cache(max_seq_len=len(input_ids) + output_tokens + 8)
    logits = model(mx.array([input_ids]), cache, last_logit_only=True)
    mx.eval(logits)
    output = []
    started = time.perf_counter()
    for index in range(output_tokens):
        token = int(mx.argmax(logits[:, -1]))
        output.append(token)
        if token == eos_id or index == output_tokens - 1:
            break
        logits = model(mx.array([[token]]), cache, last_logit_only=True)
        mx.eval(logits)
    seconds = time.perf_counter() - started
    transitions = max(len(output) - 1, 0)
    return output, {
        "decode_seconds": seconds,
        "decode_transitions": transitions,
        "decode_tok_s": transitions / seconds if seconds else None,
    }


def _run_oracle_verify(model, input_ids, output, verify_k):
    """Measure target chunk execution using the known-correct greedy sequence."""
    cache = model.make_cache(max_seq_len=len(input_ids) + len(output) + 8)
    logits = model(mx.array([input_ids]), cache, last_logit_only=True)
    mx.eval(logits)
    checked = 0
    rows = 0
    rounds = 0
    seconds = 0.0
    # Feeding output[i:i+k] yields target predictions for output[i+1:i+k+1].
    for start in range(0, max(len(output) - 1, 0), verify_k):
        chunk = output[start : min(start + verify_k, len(output) - 1)]
        started = time.perf_counter()
        chunk_logits = model(mx.array([chunk]), cache, last_logit_only=False)
        mx.eval(chunk_logits)
        seconds += time.perf_counter() - started
        expected = output[start + 1 : start + 1 + len(chunk)]
        actual = mx.argmax(chunk_logits, axis=-1).tolist()[0]
        checked += sum(left == right for left, right in zip(actual, expected))
        rows += len(chunk)
        rounds += 1
    return {
        "event": "target_oracle_verify",
        "verify_k": verify_k,
        "target_seconds": seconds,
        "target_rows": rows,
        "target_rows_per_second": rows / seconds if seconds else None,
        "rounds": rounds,
        "greedy_rows_checked": checked,
        "greedy_rows_match": checked == rows,
    }


def _install_native_single_stream_moe(model, source_root: Path) -> int:
    """Switch loaded MoE modules to the native implementation without copies."""
    os.environ["OMLX_DEEPSEEK_SORT_MIN_ROUTES"] = "1"
    os.environ["OMLX_DEEPSEEK_AFFINE_BLOCK_MIN_ROUTES"] = "1"
    sys.path.insert(0, str(source_root))
    switch = importlib.import_module("omlx.patches.deepseek_v4.switch_layers")
    replaced = 0
    for layer in model.layers:
        experts = layer.ffn.experts
        experts.__class__ = switch.SwitchGLU
        for name in ("gate_proj", "up_proj", "down_proj"):
            projection = getattr(experts, name)
            projection.__class__ = switch.QuantizedSwitchLinear
        replaced += 1
    return replaced


class ExactDirectDownSwitchGLU(SwitchGLU):
    """Preserve stock gate/up numerics while specializing the down projection."""

    def __call__(self, x, indices):
        if int(x.shape[0]) * int(indices.shape[-1]) > 36:
            return super().__call__(x, indices)
        expanded = mx.expand_dims(x, (-2, -3))
        up = self.up_proj(expanded, indices).squeeze(-2)
        gate = self.gate_proj(expanded, indices).squeeze(-2)
        activated = self.activation(up, gate)
        tokens, topk, width = map(int, activated.shape)
        down = affine2_route_down_qmv(
            self.down_proj,
            activated.reshape(tokens * topk, width),
            indices.reshape(tokens * topk, 1),
        )
        return down.reshape(tokens, topk, -1)


def _install_direct_down_qmv(model) -> int:
    """Specialize affine-2bit expert down projections without copying weights."""

    experts_by_layer = [layer.ffn.experts for layer in model.layers]
    unsupported = [
        type(experts).__name__
        for experts in experts_by_layer
        if not isinstance(experts, SwitchGLU)
    ]
    if unsupported:
        raise TypeError(f"unsupported expert module: {unsupported[0]}")

    replaced = 0
    for experts in experts_by_layer:
        if type(experts) is ExactDirectDownSwitchGLU:
            continue
        experts.__class__ = ExactDirectDownSwitchGLU
        replaced += 1
    return replaced


def _share_target_head(weights, model):
    if hasattr(weights, "attach_target"):
        weights.attach_target(model)
        return
    for suffix in ("weight", "scales", "biases"):
        value = getattr(model.head, suffix, None)
        key = f"head.{suffix}"
        if value is not None and key in weights.entries:
            weights.resident[key] = value


def _pin_mtp_with_headroom(weights, reserve_gb: float = 8.0) -> int:
    if hasattr(weights, "pin_mtp"):
        return weights.pin_mtp(reserve_gb)
    import psutil

    keys = [
        key
        for key in weights.entries
        if key.startswith("mtp.") and key not in weights.resident
    ]
    total = sum(
        weights.entries[key]["data_offsets"][1]
        - weights.entries[key]["data_offsets"][0]
        for key in keys
    )
    reserve = int(reserve_gb * 1e9)
    metal_available = (
        int(mx.device_info()["max_recommended_working_set_size"])
        - mx.get_active_memory()
    )
    safe = min(psutil.virtual_memory().available, metal_available) - reserve
    if total > safe:
        raise MemoryError(
            f"MTP needs {total / 1e9:.2f} GB but only {safe / 1e9:.2f} GB "
            f"remains after the {reserve_gb:.1f} GB safety reserve"
        )
    for key in keys:
        value = weights.read(key, _pin=True)
        weights.resident[key] = value
        weights.resident_bytes += value.nbytes
    mx.clear_cache()
    return total


def _dspark_topk(config: dict) -> int:
    return int(config["dspark_num_experts_per_tok"])


def _install_packed_mtp_moe(adapter) -> int:
    """Pack per-expert DSpark tensors and replace its serial Python MoE loop."""
    weights = adapter.w
    expert_count = adapter.c["dspark_n_routed_experts"]
    packed = {}
    packed_bytes = 0
    bases = sorted(
        {
            key.split(".experts.", 1)[0]
            for key in weights.entries
            if key.startswith("mtp.") and ".ffn.experts." in key
        }
    )
    for base in bases:
        projections = {}
        for projection in ("w1", "w3", "w2"):
            values = {}
            for suffix in ("weight", "scales", "biases"):
                keys = [
                    f"{base}.experts.{expert}.{projection}.{suffix}"
                    for expert in range(expert_count)
                ]
                value = mx.stack([weights.read(key) for key in keys])
                mx.eval(value)
                values[suffix] = value
                packed_bytes += value.nbytes
                if hasattr(weights, "release_tensor"):
                    for key in keys:
                        weights.release_tensor(key)
                else:
                    for key in keys:
                        old = weights.resident.pop(key, None)
                        if old is not None:
                            weights.resident_bytes -= old.nbytes
            values["config"] = weights.quant_config(f"{base}.experts.0.{projection}")
            projections[projection] = values
        packed[base] = projections
        mx.clear_cache()

    def packed_moe(self, base, x):
        config = self.c
        logits = (
            x.astype(mx.float32)
            @ self.w.read(base + ".gate.weight").astype(mx.float32).T
        )
        scores = mx.sqrt(mx.logaddexp(logits, mx.zeros_like(logits)))
        topk = _dspark_topk(config)
        picks = mx.argsort(scores + self.w.read(base + ".gate.bias"), axis=-1)[
            ..., -topk:
        ]
        selected = mx.take_along_axis(scores, picks, axis=-1)
        if config["norm_topk_prob"] and topk > 1:
            selected = selected / (mx.sum(selected, axis=-1, keepdims=True) + 1e-20)
        selected = selected * config["routed_scaling_factor"]
        expanded = mx.expand_dims(x, (-2, -3))

        def project(name, values):
            projection = packed[base][name]
            return mx.gather_qmm(
                values,
                projection["weight"],
                projection["scales"],
                projection["biases"],
                rhs_indices=picks,
                transpose=True,
                sorted_indices=False,
                **projection["config"],
            )

        gate = project("w1", expanded).astype(mx.float32)
        up = project("w3", expanded).astype(mx.float32)
        limit = config["swiglu_limit"]
        if limit > 0:
            gate = mx.minimum(gate, limit)
            up = mx.clip(up, -limit, limit)
        # Match the checkpoint runtime exactly: route weights are applied to
        # the activation before its bfloat16 cast and down projection.
        hidden = (gate * mx.sigmoid(gate) * up * selected[..., None, None]).astype(
            x.dtype
        )
        routed = project("w2", hidden).squeeze(-2).astype(mx.float32)
        routed = mx.sum(routed, axis=-2)
        shared = self.expert(base + ".shared_experts", x).astype(mx.float32)
        return (routed + shared).astype(x.dtype)

    # Bind through a weak proxy so adapter -> bound method does not retain
    # adapter (and its 4+ GB packed tensors) in a reference cycle between runs.
    adapter.moe = MethodType(packed_moe, weakref.proxy(adapter))
    adapter._packed_mtp_moe = packed
    return packed_bytes


def _install_vectorized_mtp_attention(draft) -> None:
    """Batch DSpark's five positions through grouped output projections."""
    runtime_globals = draft.attention.__func__.__globals__
    draft_attention = runtime_globals["draft_attention"]
    quantize_cache = runtime_globals["quantize_cache"]

    def rotate_positions(x, positions, config, inverse=False):
        rope_dim = config["qk_rope_head_dim"]
        frequencies = 1 / (
            config["rope_theta"]
            ** (mx.arange(0, rope_dim, 2, dtype=mx.float32) / rope_dim)
        )
        angles = positions[:, None] * frequencies[None, :]
        if inverse:
            angles = -angles
        tail = x[..., -rope_dim:].astype(mx.float32)
        tail = tail.reshape(*tail.shape[:-1], rope_dim // 2, 2)
        real, imag = tail[..., 0], tail[..., 1]
        expand = (slice(None),) + (None,) * (real.ndim - 2) + (slice(None),)
        cosine = mx.cos(angles)[expand]
        sine = mx.sin(angles)[expand]
        rotated = mx.stack(
            (real * cosine - imag * sine, real * sine + imag * cosine),
            axis=-1,
        ).reshape(*x.shape[:-1], rope_dim)
        return mx.concatenate((x[..., :-rope_dim], rotated.astype(x.dtype)), axis=-1)

    def vectorized_attention(self, base, x, stage):
        adapter = self.adapter
        config = self.c
        count = x.shape[0]
        positions = mx.arange(count, dtype=mx.float32) + self.position + 1
        query_a = adapter.norm(base + ".q_norm", adapter.w.linear(base + ".wq_a", x))
        query = adapter.w.linear(base + ".wq_b", query_a).reshape(
            count, config["num_attention_heads"], config["head_dim"]
        )
        key_value = adapter.norm(base + ".kv_norm", adapter.w.linear(base + ".wkv", x))
        query = rotate_positions(query, positions, config)
        key_value = quantize_cache(
            rotate_positions(key_value, positions, config), 8, 32
        )
        keys = mx.concatenate([*self.windows[stage], key_value])
        output = draft_attention(query, keys, adapter.w.read(base + ".attn_sink"))
        output = rotate_positions(output, positions, config, inverse=True)
        output = output.reshape(count, config["o_groups"], -1)

        shape = adapter.w.entries[base + ".wo_a.weight"]["shape"]
        rows = shape[0] // config["o_groups"]
        weight = adapter.w.read(base + ".wo_a.weight").reshape(
            config["o_groups"], rows, -1
        )
        scales = adapter.w.read(base + ".wo_a.scales").reshape(
            config["o_groups"], rows, -1
        )
        biases = adapter.w.read(base + ".wo_a.biases").reshape(
            config["o_groups"], rows, -1
        )
        grouped = mx.quantized_matmul(
            output.swapaxes(0, 1),
            weight,
            scales,
            biases,
            transpose=True,
            **adapter.w.quant_config(base + ".wo_a"),
        )
        projected = grouped.swapaxes(0, 1).reshape(count, -1)
        return adapter.w.linear(base + ".wo_b", projected)

    draft.attention = MethodType(vectorized_attention, draft)


def _run_serial_dspark(
    model, input_ids, output_tokens, eos_id, weights_cls, dspark_cls, verify_greedy
):
    target = SimpleNamespace()
    target.w = weights_cls(model._dspark_overlay_path)
    target.c = target.w.config["text_config"]
    target.execution_mode = "compiled"
    _share_target_head(target.w, model)
    mtp_bytes = _pin_mtp_with_headroom(target.w)
    with contextlib.redirect_stdout(sys.stderr):
        draft = dspark_cls(target, pin_weights=False)

    cache = model.make_cache(max_seq_len=len(input_ids) + output_tokens + 8)
    logits, hidden = model(
        mx.array([input_ids]),
        cache,
        last_logit_only=False,
        return_dspark_hidden=True,
    )
    mx.eval(logits, hidden)
    logits = logits[:, -1]
    for position in range(hidden.shape[1]):
        draft.observe(hidden[:, position], position)

    output = []
    blocks = 0
    accepted = 0
    draft_seconds = 0.0
    target_seconds = 0.0

    def advance(token):
        nonlocal target_seconds
        started = time.perf_counter()
        next_logits, next_hidden = model(
            mx.array([[token]]),
            cache,
            last_logit_only=True,
            return_dspark_hidden=True,
        )
        mx.eval(next_logits, next_hidden)
        target_seconds += time.perf_counter() - started
        return next_logits[:, -1], next_hidden[:, -1]

    started_all = time.perf_counter()
    while len(output) < output_tokens:
        seed = int(mx.argmax(logits))
        started = time.perf_counter()
        proposals, _confidence = draft.propose(seed)
        draft_seconds += time.perf_counter() - started
        blocks += 1
        verified, logits, stats = verify_greedy(
            proposals,
            logits,
            advance,
            lambda value: draft.observe(value, cache.offset - 1),
            eos_id,
            output_tokens - len(output),
        )
        output.extend(verified)
        accepted += stats["accepted_draft_tokens"]
        if output[-1] == eos_id:
            break
    seconds = time.perf_counter() - started_all
    transitions = max(len(output) - 1, 0)
    return output, {
        "decode_seconds": seconds,
        "decode_transitions": transitions,
        "decode_tok_s": transitions / seconds if seconds else None,
        "draft_seconds": draft_seconds,
        "target_seconds": target_seconds,
        "draft_blocks": blocks,
        "accepted_draft_tokens": accepted,
        "accepted_per_block": accepted / blocks if blocks else 0.0,
        "mtp_bytes": mtp_bytes,
    }


def _run_batched_dspark(
    model,
    input_ids,
    output_tokens,
    eos_id,
    weights_cls,
    dspark_cls,
    verify_k,
    confidence_threshold=0.0,
    packed_mtp=False,
    collect_target_margins=False,
):
    target = SimpleNamespace()
    target.w = weights_cls(model._dspark_overlay_path)
    target.c = target.w.config["text_config"]
    target.execution_mode = "compiled"
    _share_target_head(target.w, model)
    mtp_bytes = _pin_mtp_with_headroom(target.w)
    with contextlib.redirect_stdout(sys.stderr):
        draft = dspark_cls(target, pin_weights=False)
    if packed_mtp:
        _install_vectorized_mtp_attention(draft)
        packed_mtp_bytes = _install_packed_mtp_moe(draft.adapter)
    else:
        packed_mtp_bytes = 0

    cache = model.make_cache(max_seq_len=len(input_ids) + output_tokens + 8)
    logits, hidden = model(
        mx.array([input_ids]),
        cache,
        last_logit_only=False,
        return_dspark_hidden=True,
    )
    mx.eval(logits, hidden)
    logits = logits[:, -1]
    for position in range(hidden.shape[1]):
        draft.observe(hidden[:, position], position)

    output = []
    blocks = 0
    accepted = 0
    deferred_corrections = 0
    seed_already_emitted = False
    draft_seconds = 0.0
    target_seconds = 0.0
    rollback_seconds = 0.0
    proposed_depths = []
    confidence_values = []
    target_margins = []
    started_all = time.perf_counter()
    while len(output) < output_tokens:
        seed = int(mx.argmax(logits))
        if not seed_already_emitted and seed == eos_id:
            output.append(seed)
            break
        started = time.perf_counter()
        proposals, confidence = draft.propose(seed)
        draft_seconds += time.perf_counter() - started
        confidence_probs = mx.sigmoid(confidence.reshape(-1)).tolist()
        confidence_values.extend(float(value) for value in confidence_probs)
        confident_depth = len(confidence_probs)
        if confidence_threshold > 0:
            confident_depth = next(
                (
                    index
                    for index, value in enumerate(confidence_probs)
                    if value < confidence_threshold
                ),
                len(confidence_probs),
            )
        proposed_depth = min(verify_k, confident_depth)
        proposed_depths.append(proposed_depth)
        candidate = proposals[
            : min(
                proposed_depth + 1,
                output_tokens - len(output) + int(seed_already_emitted),
            )
        ]
        blocks += 1

        base_offset = cache.offset
        started = time.perf_counter()
        target_logits, target_hidden = model(
            mx.array([candidate]),
            cache,
            last_logit_only=False,
            return_dspark_hidden=True,
            enable_rollback=True,
        )
        mx.eval(target_logits, target_hidden)
        target_seconds += time.perf_counter() - started
        if collect_target_margins:
            top_two = mx.topk(target_logits[0], k=2, axis=-1)
            target_margins.extend(
                (top_two[:, -1] - top_two[:, -2]).astype(mx.float32).tolist()
            )

        committed, mismatch, hit_eos, accepted_now = _match_greedy_prefix(
            candidate, target_logits, eos_id, seed_already_emitted
        )
        accepted += accepted_now

        if hit_eos:
            output.extend(committed)
            break

        if mismatch is None:
            for index in range(len(candidate)):
                draft.observe(target_hidden[:, index], base_offset + index)
            logits = target_logits[:, -1]
            seed_already_emitted = False
        else:
            started = time.perf_counter()
            cache.rollback(base_offset + mismatch)
            rollback_seconds += time.perf_counter() - started
            for index in range(mismatch):
                draft.observe(target_hidden[:, index], base_offset + index)
            # The mismatch token is target-authoritative and may be emitted
            # immediately, but advancing it with a singleton target call would
            # erase most batching gains. Keep the cache/logits one token behind
            # and fold that correction into the next verify block's first row.
            logits = target_logits[:, mismatch - 1]
            seed_already_emitted = True
            deferred_corrections += 1

        output.extend(committed)
        if output and output[-1] == eos_id:
            break

    seconds = time.perf_counter() - started_all
    transitions = max(len(output) - 1, 0)
    return output, {
        "decode_seconds": seconds,
        "decode_transitions": transitions,
        "decode_tok_s": transitions / seconds if seconds else None,
        "verify_k": verify_k,
        "confidence_threshold": confidence_threshold,
        "draft_seconds": draft_seconds,
        "target_seconds": target_seconds,
        "rollback_seconds": rollback_seconds,
        "draft_blocks": blocks,
        "accepted_draft_tokens": accepted,
        "accepted_per_block": accepted / blocks if blocks else 0.0,
        "mean_proposed_depth": (
            sum(proposed_depths) / len(proposed_depths) if proposed_depths else 0.0
        ),
        "confidence_min": min(confidence_values) if confidence_values else None,
        "confidence_mean": (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else None
        ),
        "target_margin_min": min(target_margins) if target_margins else None,
        "target_margin_below_0_01": sum(value < 0.01 for value in target_margins),
        "target_margin_below_0_05": sum(value < 0.05 for value in target_margins),
        "target_margin_below_0_1": sum(value < 0.1 for value in target_margins),
        "confidence_max": max(confidence_values) if confidence_values else None,
        "deferred_corrections": deferred_corrections,
        "mtp_bytes": mtp_bytes,
        "packed_mtp_bytes": packed_mtp_bytes,
    }


def _match_greedy_prefix(candidate, target_logits, eos_id, seed_already_emitted):
    """Return only target-authoritative tokens through the first EOS/mismatch."""
    committed = [] if seed_already_emitted else [candidate[0]]
    accepted = 0
    for index in range(1, len(candidate)):
        expected = int(mx.argmax(target_logits[:, index - 1]))
        committed.append(expected)
        if candidate[index] != expected:
            return committed, index, expected == eos_id, accepted
        accepted += 1
        if expected == eos_id:
            return committed, None, True, accepted
    return committed, None, False, accepted


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--checkpoint-runtime", type=Path)
    parser.add_argument("--trust-checkpoint-runtime", action="store_true")
    parser.add_argument("--omlx-source", type=Path)
    parser.add_argument(
        "--native-moe-only",
        action="store_true",
        help="Run baseline and native single-stream MoE A/B, skipping DSpark.",
    )
    parser.add_argument(
        "--packed-mtp-only",
        action="store_true",
        help="Run baseline and packed-MTP K4/K5, skipping other DSpark variants.",
    )
    parser.add_argument(
        "--fuse-target-moe",
        action="store_true",
        help="Fuse target gate/up expert projections after loading.",
    )
    parser.add_argument(
        "--direct-down-qmv",
        action="store_true",
        help="Use the experimental exact affine-2bit route QMV for target down projections.",
    )
    parser.add_argument(
        "--target-only",
        action="store_true",
        help="Run target AR/oracle checks without loading checkpoint draft code.",
    )
    parser.add_argument("--tokens", type=_positive_int, default=32)
    parser.add_argument("--eval-interval", type=int, default=40)
    parser.add_argument(
        "--prompt",
        default=(
            "Write a Python function that merges overlapping integer intervals. "
            "Return only the function and keep it concise."
        ),
    )
    return parser.parse_args()


def _validate_mode(args) -> None:
    if args.target_only and (args.native_moe_only or args.packed_mtp_only):
        raise SystemExit(
            "--target-only cannot be combined with --native-moe-only or "
            "--packed-mtp-only"
        )


def _run_benchmark(args, checkpoint_runtime, dspark_module) -> None:
    started = time.perf_counter()
    model, _ = load(
        str(args.target.resolve()),
        lazy=False,
        fuse_moe_gate_up=args.fuse_target_moe,
    )
    model.eval_interval = args.eval_interval
    if args.overlay is not None:
        model._dspark_overlay_path = str(args.overlay.resolve())
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(args.target.resolve() / "tokenizer.json")
    )
    input_ids = tokenizer.encode(_prompt(args.prompt), add_special_tokens=False)
    print(
        json.dumps(
            {
                "event": "loaded",
                "seconds": time.perf_counter() - started,
                "active_gb": mx.get_active_memory() / 1e9,
                "peak_gb": mx.get_peak_memory() / 1e9,
                "prompt_tokens": len(input_ids),
                "eval_interval": args.eval_interval,
                "fuse_target_moe": args.fuse_target_moe,
            }
        ),
        flush=True,
    )

    ar_tokens, ar = _run_ar(model, input_ids, args.tokens, eos_id=1)
    ar.update(
        event="ar",
        output_tokens=ar_tokens,
        text=tokenizer.decode(ar_tokens),
        active_gb=mx.get_active_memory() / 1e9,
        peak_gb=mx.get_peak_memory() / 1e9,
    )
    print(json.dumps(ar), flush=True)

    if args.direct_down_qmv:
        replaced = _install_direct_down_qmv(model)
        direct_tokens, direct = _run_ar(model, input_ids, args.tokens, eos_id=1)
        direct.update(
            event="ar_direct_down_qmv",
            replaced_layers=replaced,
            output_tokens=direct_tokens,
            text=tokenizer.decode(direct_tokens),
            greedy_matches_ar=direct_tokens == ar_tokens,
            active_gb=mx.get_active_memory() / 1e9,
            peak_gb=mx.get_peak_memory() / 1e9,
        )
        print(json.dumps(direct), flush=True)

    if args.native_moe_only:
        if args.omlx_source is None:
            raise SystemExit("--native-moe-only requires --omlx-source")
        replaced = _install_native_single_stream_moe(model, args.omlx_source.resolve())
        native_tokens, native = _run_ar(model, input_ids, args.tokens, eos_id=1)
        native.update(
            event="ar_native_small_route_moe",
            replaced_layers=replaced,
            output_tokens=native_tokens,
            text=tokenizer.decode(native_tokens),
            greedy_matches_ar=native_tokens == ar_tokens,
            active_gb=mx.get_active_memory() / 1e9,
            peak_gb=mx.get_peak_memory() / 1e9,
        )
        print(json.dumps(native), flush=True)
        return

    if args.packed_mtp_only:
        for verify_k in (4, 5):
            batched_tokens, batched = _run_batched_dspark(
                model,
                input_ids,
                args.tokens,
                1,
                checkpoint_runtime.Weights,
                dspark_module.DSpark,
                verify_k,
                packed_mtp=True,
            )
            batched.update(
                event="dspark_batched_packed_mtp",
                output_tokens=batched_tokens,
                text=tokenizer.decode(batched_tokens),
                greedy_matches_ar=batched_tokens == ar_tokens,
                active_gb=mx.get_active_memory() / 1e9,
                peak_gb=mx.get_peak_memory() / 1e9,
            )
            print(json.dumps(batched), flush=True)
        return

    for verify_k in (2, 3, 4, 5):
        oracle = _run_oracle_verify(model, input_ids, ar_tokens, verify_k)
        oracle.update(
            active_gb=mx.get_active_memory() / 1e9,
            peak_gb=mx.get_peak_memory() / 1e9,
        )
        print(json.dumps(oracle), flush=True)

    if args.target_only:
        return

    dspark_tokens, dspark = _run_serial_dspark(
        model,
        input_ids,
        args.tokens,
        1,
        checkpoint_runtime.Weights,
        dspark_module.DSpark,
        dspark_module.verify_greedy,
    )
    dspark.update(
        event="dspark_serial",
        output_tokens=dspark_tokens,
        text=tokenizer.decode(dspark_tokens),
        greedy_matches_ar=dspark_tokens == ar_tokens,
        active_gb=mx.get_active_memory() / 1e9,
        peak_gb=mx.get_peak_memory() / 1e9,
    )
    print(json.dumps(dspark), flush=True)

    for verify_k, confidence_threshold in (
        (3, 0.0),
        (4, 0.0),
        (5, 0.0),
        (5, 0.3),
        (5, 0.5),
        (5, 0.7),
    ):
        batched_tokens, batched = _run_batched_dspark(
            model,
            input_ids,
            args.tokens,
            1,
            checkpoint_runtime.Weights,
            dspark_module.DSpark,
            verify_k,
            confidence_threshold,
        )
        batched.update(
            event="dspark_batched",
            output_tokens=batched_tokens,
            text=tokenizer.decode(batched_tokens),
            greedy_matches_ar=batched_tokens == ar_tokens,
            active_gb=mx.get_active_memory() / 1e9,
            peak_gb=mx.get_peak_memory() / 1e9,
        )
        print(json.dumps(batched), flush=True)

    if args.omlx_source is not None:
        replaced = _install_native_single_stream_moe(model, args.omlx_source.resolve())
        native_tokens, native = _run_ar(model, input_ids, args.tokens, eos_id=1)
        native.update(
            event="ar_native_small_route_moe",
            replaced_layers=replaced,
            output_tokens=native_tokens,
            text=tokenizer.decode(native_tokens),
            greedy_matches_ar=native_tokens == ar_tokens,
            active_gb=mx.get_active_memory() / 1e9,
            peak_gb=mx.get_peak_memory() / 1e9,
        )
        print(json.dumps(native), flush=True)


def main() -> None:
    args = parse_args()
    _validate_mode(args)
    if args.target_only:
        _run_benchmark(args, None, None)
        return
    if args.overlay is None:
        raise SystemExit("--overlay is required for DSpark")
    if args.checkpoint_runtime is None:
        _run_benchmark(args, rapid_dspark, rapid_dspark)
        return
    if not args.trust_checkpoint_runtime:
        raise SystemExit(
            "refusing to execute checkpoint-bundled Python without "
            "--trust-checkpoint-runtime"
        )
    with _checkpoint_runtime(args.checkpoint_runtime) as (runtime, dspark_module):
        _run_benchmark(args, runtime, dspark_module)


if __name__ == "__main__":
    main()
