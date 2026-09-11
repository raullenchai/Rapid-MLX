#!/usr/bin/env python3
"""Locate the first V4.1 target operator that changes with verify batch width."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
from transformers import PreTrainedTokenizerFast

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from benchmark_deepseek_v41_dspark import _prompt, _run_ar  # noqa: E402

from vllm_mlx.models.deepseek_v41_native.attention import (  # noqa: E402
    Attention,
    window_idx_matrix,
)
from vllm_mlx.models.deepseek_v41_native.fakequant import (
    fake_quant_fp8_ue8m0,  # noqa: E402
)
from vllm_mlx.models.deepseek_v41_native.hyper_connections import (  # noqa: E402
    hc_mixes,
    hc_post,
    hc_pre,
)
from vllm_mlx.models.deepseek_v41_native.layers import rope_tail  # noqa: E402
from vllm_mlx.models.deepseek_v41_native.load import load  # noqa: E402
from vllm_mlx.models.deepseek_v41_native.model import Block  # noqa: E402
from vllm_mlx.models.deepseek_v41_native.sparse_attention import (
    sparse_attn,  # noqa: E402
)

DEFAULT_PROMPT = (
    "A train travels 180 km at 60 km/h, waits 35 minutes, then travels "
    "120 km at 80 km/h. Explain the total elapsed time step by step."
)


class Trace:
    def __init__(self):
        self.mode = "off"
        self.values = {"batch": defaultdict(list), "serial": defaultdict(list)}

    def record(self, layer: int, stage: str, value) -> None:
        if self.mode == "off":
            return
        mx.eval(value)
        self.values[self.mode][layer, stage].append(value)


def _install_trace(trace: Trace):
    original = Block.__call__

    def traced(self, x, pre_mix, start_pos, cache, shared):
        trace.record(self.layer_id, "input", x)
        residual = x
        attn_pre, attn_post, attn_comb = hc_mixes(
            x,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.hc_mult,
            self.hc_iters,
            self.norm_eps,
            self.hc_eps,
        )
        trace.record(self.layer_id, "attn_pre", attn_pre)
        h = self.attn_norm(hc_pre(x, pre_mix))
        trace.record(self.layer_id, "attn_input", h)
        h = self.attn(h, start_pos, cache, shared)
        trace.record(self.layer_id, "attn_output", h)
        x = hc_post(h, residual, attn_post, attn_comb)
        trace.record(self.layer_id, "attn_post", x)

        residual = x
        ffn_pre, ffn_post, ffn_comb = hc_mixes(
            x,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.hc_mult,
            self.hc_iters,
            self.norm_eps,
            self.hc_eps,
        )
        trace.record(self.layer_id, "ffn_pre", ffn_pre)
        h = self.ffn_norm(hc_pre(x, attn_pre))
        trace.record(self.layer_id, "ffn_input", h)
        h = self.ffn(h)
        trace.record(self.layer_id, "ffn_output", h)
        x = hc_post(h, residual, ffn_post, ffn_comb)
        trace.record(self.layer_id, "ffn_post", x)
        return x, ffn_pre

    Block.__call__ = traced
    return original


def _install_attention_trace(trace: Trace):
    original = Attention.__call__

    def traced(self, x, start_pos, cache, shared):
        if self.layer_id != 0:
            return original(self, x, start_pos, cache, shared)
        if self.ratio:
            raise RuntimeError("layer-zero diagnostic expects sliding-window attention")
        bsz, count, _ = x.shape
        end_pos = start_pos + count
        cos, sin = self._freqs(end_pos)
        cosine, sine = cos[start_pos:end_pos], sin[start_pos:end_pos]

        query_a = self.wq_a(x)
        trace.record(self.layer_id, "inside_wq_a", query_a)
        query_a = self.q_norm(query_a)
        trace.record(self.layer_id, "inside_q_norm", query_a)
        query = self.wq_b(query_a).reshape(bsz, count, self.n_heads, self.head_dim)
        trace.record(self.layer_id, "inside_wq_b", query)
        query = rope_tail(query, self.rope_head_dim, cosine, sine)
        trace.record(self.layer_id, "inside_q_rope", query)

        key_value = self.wkv(x)
        trace.record(self.layer_id, "inside_wkv", key_value)
        key_value = self.kv_norm(key_value)
        trace.record(self.layer_id, "inside_kv_norm", key_value)
        key_value = rope_tail(key_value, self.rope_head_dim, cosine, sine)
        trace.record(self.layer_id, "inside_kv_rope", key_value)
        key_value = fake_quant_fp8_ue8m0(key_value, 32)
        trace.record(self.layer_id, "inside_kv_quant", key_value)

        layer_cache = cache.layers[self.layer_id]
        previous = layer_cache.window_chrono(start_pos)
        previous_count = previous.shape[1]
        all_kv = (
            mx.concatenate([previous.astype(key_value.dtype), key_value], axis=1)
            if previous_count
            else key_value
        )
        indices = mx.broadcast_to(
            window_idx_matrix(previous_count, count, self.window_size)[None],
            (bsz, count, min(self.window_size, previous_count + count)),
        )
        layer_cache.write_window(start_pos, key_value)
        output = sparse_attn(query, all_kv, self.attn_sink, indices, self.softmax_scale)
        trace.record(self.layer_id, "inside_sparse_attn", output)
        output = rope_tail(output, self.rope_head_dim, cosine, sine, inverse=True)
        trace.record(self.layer_id, "inside_inverse_rope", output)
        output = output.reshape(bsz, count, self.n_groups, -1)
        output = self.wo_a(output.astype(mx.float32)[..., None, :]).squeeze(-2)
        trace.record(self.layer_id, "inside_wo_a", output)
        output = self.wo_b(output.reshape(bsz, count, -1).astype(x.dtype))
        trace.record(self.layer_id, "inside_wo_b", output)
        return output

    Attention.__call__ = traced
    return original


def _comparison(trace: Trace) -> list[dict]:
    rows = []
    for key, batch_values in trace.values["batch"].items():
        serial_values = trace.values["serial"].get(key)
        if len(batch_values) != 1 or not serial_values:
            continue
        batch = batch_values[0].astype(mx.float32)
        serial = mx.concatenate(serial_values, axis=1).astype(mx.float32)
        mx.eval(batch, serial)
        difference = mx.abs(batch - serial)
        mx.eval(difference)
        rows.append(
            {
                "layer": key[0],
                "stage": key[1],
                "equal": bool(mx.array_equal(batch, serial)),
                "max_abs": float(mx.max(difference)),
                "mean_abs": float(mx.mean(difference)),
                "changed": int(mx.sum(difference != 0)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--width", type=int, default=5)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()
    if args.width < 2:
        raise SystemExit("--width must be at least 2")

    model, _ = load(str(args.target.resolve()), lazy=False)
    model.eval_interval = 1
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(args.target.resolve() / "tokenizer.json")
    )
    prompt_ids = tokenizer.encode(_prompt(args.prompt), add_special_tokens=False)
    tokens, _ = _run_ar(model, prompt_ids, args.width + 1, eos_id=1)
    candidate = tokens[: args.width]

    batch_cache = model.make_cache(max_seq_len=len(prompt_ids) + args.width + 8)
    serial_cache = model.make_cache(max_seq_len=len(prompt_ids) + args.width + 8)
    prompt_array = mx.array([prompt_ids])
    batch_seed = model(prompt_array, batch_cache, last_logit_only=True)
    serial_seed = model(prompt_array, serial_cache, last_logit_only=True)
    mx.eval(batch_seed, serial_seed)
    if not bool(mx.array_equal(batch_seed, serial_seed)):
        raise RuntimeError("independent prompt prefills are not deterministic")

    trace = Trace()
    original_block = _install_trace(trace)
    original_attention = _install_attention_trace(trace)
    try:
        trace.mode = "batch"
        batch_logits = model(mx.array([candidate]), batch_cache, last_logit_only=False)
        mx.eval(batch_logits)
        trace.mode = "serial"
        serial_logits = []
        for token in candidate:
            value = model(mx.array([[token]]), serial_cache, last_logit_only=True)
            mx.eval(value)
            serial_logits.append(value)
    finally:
        trace.mode = "off"
        Block.__call__ = original_block
        Attention.__call__ = original_attention

    serial_logits = mx.concatenate(serial_logits, axis=1)
    mx.eval(serial_logits)
    rows = _comparison(trace)
    first_changed = next((row for row in rows if not row["equal"]), None)
    print(
        json.dumps(
            {
                "event": "batch_parity",
                "width": args.width,
                "candidate": candidate,
                "batch_predictions": mx.argmax(batch_logits, axis=-1).tolist()[0],
                "serial_predictions": mx.argmax(serial_logits, axis=-1).tolist()[0],
                "logits_equal": bool(mx.array_equal(batch_logits, serial_logits)),
                "logits_max_abs": float(mx.max(mx.abs(batch_logits - serial_logits))),
                "first_changed": first_changed,
            }
        )
    )
    for row in rows:
        if not row["equal"]:
            print(json.dumps({"event": "operator_difference", **row}))


if __name__ == "__main__":
    main()
