#!/usr/bin/env python3
"""Emit fixed real-weight Qwen4-Exp probes for sequential parity comparison.

The two backends are intentionally run in separate processes so the 98 GiB
checkpoint is never resident twice.  ``--backend upstream`` expects the pinned
mlx-vlm source tree to be first on ``PYTHONPATH``.  Its small sanitizer adapter
only translates the converter's fused-expert quantized parameter layout; it
does not alter model math.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def _input(shape: tuple[int, ...], *, scale: float) -> mx.array:
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return mx.array(np.sin(values * 0.013) * scale)


def _patch_upstream_fused_quantized_sanitizer() -> None:
    from mlx_vlm.models.qwen4_exp import Model

    original = Model.sanitize

    def sanitize(self, weights):
        expanded = {}
        for key, value in weights.items():
            if ".mlp.experts.gate_up_proj" in key:
                prefix, suffix = key.split("experts.gate_up_proj", 1)
                leaf = {
                    "": "weight",
                    ".scales": "scales",
                    ".biases": "biases",
                }.get(suffix)
                if leaf is not None:
                    midpoint = value.shape[-2] // 2
                    expanded[f"{prefix}switch_mlp.gate_proj.{leaf}"] = value[
                        ..., :midpoint, :
                    ]
                    expanded[f"{prefix}switch_mlp.up_proj.{leaf}"] = value[
                        ..., midpoint:, :
                    ]
                    continue
            if ".mlp.experts.down_proj" in key:
                prefix, suffix = key.split("experts.down_proj", 1)
                leaf = {
                    "": "weight",
                    ".scales": "scales",
                    ".biases": "biases",
                }.get(suffix)
                if leaf is not None:
                    expanded[f"{prefix}switch_mlp.down_proj.{leaf}"] = value
                    continue
            expanded[key] = value
        return original(self, expanded)

    Model.sanitize = sanitize


def _load(checkpoint: Path, backend: str):
    if backend == "rapid":
        from mlx_lm.utils import load_model

        from vllm_mlx.utils.tokenizer import _register_vendored_archs

        _register_vendored_archs()
        model, _ = load_model(checkpoint, strict=True)
        return model, model.model.layers, model

    _patch_upstream_fused_quantized_sanitizer()
    from mlx_vlm.utils import load_model

    model = load_model(checkpoint, lazy=True, strict=True)
    language = model.language_model
    return model, language.model.layers, language


def _logits(result, backend: str) -> mx.array:
    return result if backend == "rapid" else result.logits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("rapid", "upstream"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    holder, layers, language = _load(args.checkpoint.resolve(), args.backend)
    del holder
    token_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    hidden = _input((1, 4, 2560), scale=0.02)
    hyper = _input((1, 4, 4 * 2560), scale=0.02)

    gdn = layers[0].linear_attn(hidden, mask=None, cache=None)
    gdn_cached = layers[0].linear_attn(
        hidden, mask=None, cache=language.make_cache()[0]
    )
    moe = layers[0].mlp(hidden)
    ple = layers[1].ple(hyper, token_ids, None, None)
    qsa_cache = language.make_cache()[3]
    full_cache = language.make_cache()
    if args.backend == "rapid":
        qsa = layers[3].self_attn(hidden, cache=qsa_cache)
    else:
        qsa = layers[3].self_attn(
            hidden, mask="causal", cache=qsa_cache, position_ids=None
        )

    captured_layers = []
    layer_class = type(layers[0])
    original_layer_call = layer_class.__call__

    def capture_layer(self, *call_args, **call_kwargs):
        output = original_layer_call(self, *call_args, **call_kwargs)
        captured_layers.append(output)
        return output

    layer_class.__call__ = capture_layer
    try:
        result = language(token_ids, cache=full_cache)
    finally:
        layer_class.__call__ = original_layer_call
    logits = _logits(result, args.backend)
    layer_last = mx.stack([value[:, -1, :] for value in captured_layers])

    # Cross the checkpoint-declared 2,048-token QSA admission budget by the
    # smallest complete compressed block. This keeps the pinned reference's
    # dense selector bounded while forcing both implementations through sparse
    # block selection. The following one-token call then proves that the
    # prefill-owned GDN/QSA/PLE caches remain numerically aligned at decode.
    sparse_length = 2052
    sparse_ids = (mx.arange(sparse_length, dtype=mx.int32) % 32000)[None, :]
    sparse_hidden = _input((1, sparse_length, 2560), scale=0.02)
    sparse_qsa_cache = language.make_cache()[3]
    if args.backend == "rapid":
        sparse_qsa = layers[3].self_attn(sparse_hidden, cache=sparse_qsa_cache)
    else:
        sparse_qsa = layers[3].self_attn(
            sparse_hidden,
            mask="causal",
            cache=sparse_qsa_cache,
            position_ids=None,
        )
    mx.eval(sparse_qsa)

    sparse_full_cache = language.make_cache()
    sparse_result = language(sparse_ids, cache=sparse_full_cache)
    sparse_logits = _logits(sparse_result, args.backend)
    sparse_last = sparse_logits[:, -1, :]
    mx.eval(sparse_last)
    next_token = mx.argmax(sparse_last, axis=-1).astype(mx.int32)[:, None]
    decode_result = language(next_token, cache=sparse_full_cache)
    decode_logits = _logits(decode_result, args.backend)[:, -1, :]

    probes = {
        "gdn": gdn,
        "gdn_cached": gdn_cached,
        "qsa": qsa,
        "ple": ple,
        "moe": moe,
        "layer_last": layer_last,
        "logits_last": logits[:, -1, :],
        "sparse_qsa_last": sparse_qsa[:, -1, :],
        "sparse_logits_last": sparse_last,
        "cached_decode_logits_last": decode_logits,
    }
    mx.eval(list(probes.values()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        **{
            name: np.asarray(value.astype(mx.float32)) for name, value in probes.items()
        },
    )
    print(
        json.dumps(
            {
                "backend": args.backend,
                "output": str(args.output),
                "probes": {name: list(value.shape) for name, value in probes.items()},
                "greedy_tokens": {
                    "short": int(mx.argmax(logits[:, -1, :], axis=-1).item()),
                    "sparse_prefill": int(mx.argmax(sparse_last, axis=-1).item()),
                    "cached_decode": int(mx.argmax(decode_logits, axis=-1).item()),
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
