#!/usr/bin/env python3
"""Collect per-domain REAP, MAN and MSAN with an explicitly trusted disk runtime.

MAN/MSAN are gate-free means over routed tokens (not all-expert evaluation).
The legacy positional corpus and trust flag remain supported.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path
from types import MethodType

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.deepseek_v41_rd_common import (
    aggregate_scores,
    checkpoint_hashes,
    file_hash,
    masked_route,
    read_slices,
)


def _load_runtime(model: Path):
    path = model / "runtime" / "runtime.py"
    spec = importlib.util.spec_from_file_location(
        "rapid_deepseek_v41_calibration_runtime", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import checkpoint runtime: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _corpus_tokens(runtime, paths: list[Path], count: int) -> list[int]:
    """Return exactly ``count`` token ids or raise ValueError for a shorter corpus."""
    text = "\n\n".join(path.read_text(errors="replace") for path in paths)
    ids = runtime.tokenizer.encode(text).ids
    if len(ids) < count:
        raise ValueError(f"corpus has {len(ids)} tokens, need {count}")
    return ids[:count]


def _overlap(a: np.ndarray, b: np.ndarray, keep: int) -> float:
    return float(
        np.mean(
            [
                len(
                    set(np.argsort(-left, kind="stable")[:keep])
                    & set(np.argsort(-right, kind="stable")[:keep])
                )
                / keep
                for left, right in zip(a, b)
            ]
        )
    )


def install_moe_probe(runtime, observer=None, allocation=None, mass_observer=None):
    """Instrument the existing split-expert runtime; preserve shared expert path.

    Candidate masks change selection before top-k and renormalize unbiased
    scores. They measure pruning only; bit-widths are deliberately not simulated.
    """
    import mlx.core as mx

    def instrumented_moe(self, base, x):
        match = re.fullmatch(r"layers\.(\d+)\.ffn", base)
        if match is None or x.shape[0] != 1:
            raise ValueError(
                f"probe requires single-token text MoE, got {base}, {x.shape}"
            )
        layer = int(match.group(1))
        config = self.c
        logits = (
            x.astype(mx.float32)
            @ self.w.read(base + ".gate.weight").astype(mx.float32).T
        )
        logits = logits / config.get("gate_temp", 1.0)
        scores = mx.sqrt(mx.logaddexp(logits, mx.zeros_like(logits)))
        bias = self.w.read(base + ".gate.bias")
        topk = config["num_experts_per_tok"]
        if allocation is not None:
            mx.eval(scores, bias)
            kept = allocation["layers"][str(layer)]["kept_expert_ids"]
            chosen, _, mass = masked_route(np.array(scores), np.array(bias), kept, topk)
            picks = mx.array(chosen, mx.int32)
            if mass_observer:
                mass_observer(layer, float(mass[0]))
        else:
            picks = mx.argsort(scores + bias, axis=-1)[:, -topk:]
        selected = mx.take_along_axis(scores, picks, axis=-1)
        normalized = selected / (mx.sum(selected, axis=-1, keepdims=True) + 1e-20)
        weights = normalized if config["norm_topk_prob"] and topk > 1 else selected
        weights = weights * config["routed_scaling_factor"]
        mx.eval(picks, normalized, weights)
        result = mx.zeros_like(x).astype(mx.float32)
        for expert, gate, weight in zip(
            picks.tolist()[0], normalized.tolist()[0], weights.tolist()[0]
        ):
            # Unit weight gives the gate-free output; the routing weight is
            # applied exactly once below.
            output = self.expert(base + f".experts.{expert}", x, 1.0).astype(mx.float32)
            if observer is not None:
                norm = mx.sqrt(mx.sum(mx.square(output)))
                mx.eval(norm)
                observer(layer, expert, gate, float(norm.item()))
            result = result + output * weight
        return (
            result + self.expert(base + ".shared_experts", x).astype(mx.float32)
        ).astype(x.dtype)

    runtime.moe = MethodType(instrumented_moe, runtime)


def install_native_score_probe(model, observer):
    """Observe unweighted SwitchGLU outputs without another expert evaluation."""
    import mlx.core as mx

    from rapid_mlx.models.deepseek_v41_native.moe import MoE

    class CalibrationMoE(MoE):
        def __call__(self, x):
            shape = x.shape
            xf = x.reshape(-1, self.dim)
            if xf.shape[0] != 1:
                raise ValueError("calibration probe requires single-token forwards")
            weights, indices = self.gate(xf)
            outputs = self.experts(xf, indices).astype(mx.float32)
            norms = mx.sqrt(mx.sum(mx.square(outputs), axis=-1))
            normalized = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
            mx.eval(norms, normalized, indices)
            for expert, gate, norm in zip(
                indices.tolist()[0], normalized.tolist()[0], norms.tolist()[0]
            ):
                observer(self._rd_layer, expert, gate, norm)
            y = mx.sum(outputs * weights[..., None], axis=-2)
            y = y + self.shared_experts(xf).astype(mx.float32)
            return y.reshape(shape).astype(x.dtype)

    for layer_id, layer in enumerate(model.layers):
        layer.ffn._rd_layer = layer_id
        layer.ffn.__class__ = CalibrationMoE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=1024, help="tokens per slice")
    p.add_argument(
        "--keep-experts",
        type=int,
        default=288,
        help=(
            "experts kept when measuring split-half top-k overlap "
            "(diagnostic only; default matches the 1017 recipe)"
        ),
    )
    p.add_argument(
        "--slices", type=Path, help="JSON: slices maps domain names to local text paths"
    )
    p.add_argument(
        "--model-revision",
        help="source revision; default resolved snapshot directory name",
    )
    p.add_argument(
        "--resident-backbone",
        action="store_true",
        help="opt in to resident backbone; default disk streamed",
    )
    p.add_argument(
        "--native",
        action="store_true",
        help="use native mmap loader for a converted checkpoint",
    )
    p.add_argument("--trust-checkpoint-runtime", action="store_true")
    p.add_argument("corpus", type=Path, nargs="*")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if not a.native and not a.trust_checkpoint_runtime:
        raise SystemExit(
            "refusing to execute checkpoint-bundled Python without --trust-checkpoint-runtime"
        )
    import mlx.core as mx

    if a.tokens < 2 or not (a.slices or a.corpus):
        raise ValueError("provide a corpus/slices and at least two tokens per slice")
    model = a.model.resolve()
    slices = read_slices(a.slices, a.corpus)
    module = None if a.native else _load_runtime(model)
    config = json.loads((model / "config.json").read_text())
    c = config.get("text_config", config)
    layers, experts = int(c["num_hidden_layers"]), int(c["n_routed_experts"])
    if not c["num_experts_per_tok"] <= a.keep_experts <= experts:
        raise ValueError("keep-experts must preserve top-k and fit the expert pool")
    shape = (len(slices), 2, layers, experts)
    reap, norms, squared = [np.zeros(shape, np.float64) for _ in range(3)]
    counts = np.zeros(shape, np.int64)
    started = time.monotonic()
    corpus_metadata = {}
    if a.native:
        from tokenizers import Tokenizer

        from rapid_mlx.models.deepseek_v41_native.load import load

        native_model, _ = load(str(model), lazy=True, engram_ssd_offload=True)
        tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    for domain, (name, paths) in enumerate(slices.items()):
        corpus_metadata[name] = [
            {"file": str(p), "sha256": file_hash(p)} for p in paths
        ]
        if a.native:
            from types import SimpleNamespace

            runtime = SimpleNamespace(tokenizer=tokenizer)
            cache = native_model.make_cache(max_seq_len=a.tokens + 1)
        else:
            runtime = module.TextRuntime(
                model,
                max_tokens=a.tokens,
                resident_backbone=a.resident_backbone,
                execution_mode="compiled",
            )
        # The half boundary and recorded token count rely on len(ids) == a.tokens.
        ids = _corpus_tokens(runtime, paths, a.tokens)
        cursor = {"half": 0}

        def observe(layer, expert, gate, norm, domain=domain, cursor=cursor):
            idx = (domain, cursor["half"], layer, expert)
            reap[idx] += gate * norm
            norms[idx] += norm
            squared[idx] += norm * norm
            counts[idx] += 1

        if a.native:
            install_native_score_probe(native_model, observe)
        else:
            install_moe_probe(runtime, observe)
        for token_index, token in enumerate(ids):
            cursor["half"] = min(1, token_index * 2 // a.tokens)
            if a.native:
                logits = native_model(mx.array([[token]]), cache, last_logit_only=True)
                mx.eval(logits)
            else:
                runtime.step(token)
            if (token_index + 1) % 32 == 0:
                print(
                    json.dumps(
                        {
                            "slice": name,
                            "tokens": token_index + 1,
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
        del runtime
        if a.native:
            del cache
        mx.clear_cache()
    scores = aggregate_scores(reap, norms, squared, counts)
    overlap = np.array(
        [
            [
                _overlap(
                    scores[metric + "_halves"][d, 0],
                    scores[metric + "_halves"][d, 1],
                    a.keep_experts,
                )
                for metric in ("reap", "man", "msan")
            ]
            for d in range(len(slices))
        ]
    )
    split_keep_overlap = float(overlap[:, 0].min())
    metadata = {
        "schema_version": 1,
        "model_revision": a.model_revision or model.name,
        "config_sha256": file_hash(model / "config.json"),
        "checkpoint_sha256": checkpoint_hashes(model),
        "corpus": corpus_metadata,
        "slice_names": list(slices),
        "tokens_per_slice": [a.tokens] * len(slices),
        "routing_topk": c["num_experts_per_tok"],
        "score_definitions": {
            "reap": "sum normalized_gate * output_L2",
            "man": "routed-token mean output_L2",
            "msan": "routed-token mean output_L2_squared",
        },
        "normalization": "raw score max over domains; equal tokens per slice; halves checked separately",
        "split_overlap_columns": ["reap", "man", "msan"],
        "split_keep_overlap": split_keep_overlap,
        "split_keep_overlap_by_domain": overlap.tolist(),
        "keep_experts": a.keep_experts,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        a.output,
        **scores,
        metadata=np.array(json.dumps(metadata)),
        saliency=reap.sum((0, 1)),
        robust_saliency=scores["robust_reap"],
        saliency_halves=reap.sum(0),
        counts=counts.sum((0, 1)),
        counts_halves=counts.sum(0),
        tokens=np.array(a.tokens * len(slices)),
        keep_experts=np.array(a.keep_experts),
        split_keep_overlap=split_keep_overlap,
        split_keep_overlap_by_domain=overlap,
        method=np.array("reap_output_norm"),
        selection_policy=np.array("max_domain_with_half_stability"),
    )
    print(
        json.dumps(
            {
                "output": str(a.output),
                "split_keep_overlap": split_keep_overlap,
                "split_keep_overlap_by_domain": overlap.tolist(),
                "never_routed": int((counts.sum((0, 1)) == 0).sum()),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
