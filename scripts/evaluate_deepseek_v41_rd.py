#!/usr/bin/env python3
"""Cache teacher-forced logits, screen pruning masks, and compare full-vocabulary KL.

Collection uses the existing native mmap/Engram SSD path. Mask screening leaves
expert weights unchanged and does NOT qualify mixed-bit quantization quality.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.deepseek_v41_rd_common import bank_plan, file_hash, read_slices


def token_metrics(reference, candidate):
    """Stable KL(reference || candidate), in nats, and top-1 disagreements."""
    r, c = np.asarray(reference, np.float64), np.asarray(candidate, np.float64)
    if (
        r.ndim != 2
        or r.shape != c.shape
        or not np.isfinite(r).all()
        or not np.isfinite(c).all()
    ):
        raise ValueError("logits must be finite matching [tokens, vocabulary] arrays")
    r = r - r.max(-1, keepdims=True)
    c = c - c.max(-1, keepdims=True)
    r = r - np.log(np.exp(r).sum(-1, keepdims=True))
    c = c - np.log(np.exp(c).sum(-1, keepdims=True))
    kl = np.maximum(0.0, (np.exp(r) * (r - c)).sum(-1))
    return kl, np.argmax(reference, -1) != np.argmax(candidate, -1)


def validate_allocation(allocation, layers, experts, topk):
    if set(allocation["layers"]) != {str(layer_id) for layer_id in range(layers)}:
        raise ValueError("allocation layer coverage mismatch")
    for entry in allocation["layers"].values():
        ids = entry["kept_expert_ids"]
        if (
            len(ids) < topk
            or ids != sorted(set(ids))
            or any(type(e) is not int or not 0 <= e < experts for e in ids)
        ):
            raise ValueError("invalid kept expert IDs")
        if "bit_widths" in entry:
            bits = entry["bit_widths"]
            if len(bits) != experts or ids != bank_plan(bits, topk)["kept_expert_ids"]:
                raise ValueError("allocation bit widths and keep IDs disagree")
        elif set(entry.get("projection_bits", {})) != {
            "gate_proj",
            "up_proj",
            "down_proj",
        }:
            raise ValueError("allocation needs layer projection bits")


def install_native_masks(model, allocation, masses):
    import mlx.core as mx

    from rapid_mlx.models.deepseek_v41_native.moe import Gate

    class EvaluationGate(Gate):
        def __call__(self, x):
            weights, indices = super().__call__(x)
            retained = mx.take(self._rd_mask, indices)
            mass = mx.sum(weights * retained, axis=-1) / mx.maximum(
                mx.sum(weights, axis=-1), 1e-30
            )
            mx.eval(mass)
            masses[self._rd_layer].extend(np.array(mass).reshape(-1).tolist())
            original = self.bias
            try:
                self.bias = mx.where(self._rd_mask, original, -mx.inf)
                return super().__call__(x)
            finally:
                self.bias = original

    validate_allocation(
        allocation,
        len(model.layers),
        model.args.n_routed_experts,
        model.args.n_activated_experts,
    )
    for layer_id, layer in enumerate(model.layers):
        gate = layer.ffn.gate
        mask = np.zeros(model.args.n_routed_experts, bool)
        mask[allocation["layers"][str(layer_id)]["kept_expert_ids"]] = True
        gate._rd_mask = mx.array(mask)
        gate._rd_layer = layer_id
        gate.__class__ = EvaluationGate


def collect(a):
    import mlx.core as mx
    from tokenizers import Tokenizer

    from rapid_mlx.models.deepseek_v41_native.load import load

    if a.output.exists():
        raise FileExistsError("cache destination must not exist")
    if a.tokens < 1:
        raise ValueError("tokens must be positive")
    slices = read_slices(a.slices, [])
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in slices):
        raise ValueError("slice names must be safe filenames")
    tokenizer = Tokenizer.from_file(str(a.model / "tokenizer.json"))
    # Explicit lazy avoids full materialization and automatic page-cache prewarm.
    model, config = load(str(a.model), lazy=True, engram_ssd_offload=True)
    allocation = json.loads(a.allocation.read_text()) if a.allocation else None
    masses = {layer_id: [] for layer_id in range(config.n_layers)}
    if getattr(a, "serving_qmv", False):
        from rapid_mlx.models.deepseek_v41_native.serving import install_target_qmv

        install_target_qmv(model)
    if allocation:
        install_native_masks(model, allocation, masses)
    a.output.mkdir(parents=True)
    checkpoint_config = json.loads((a.model / "config.json").read_text())
    packed = checkpoint_config.get("rapid_quantization", {}).get("allocation")
    metadata = {
        "schema_version": 1,
        "serving_qmv": getattr(a, "serving_qmv", False),
        "model_revision": a.model_revision,
        "checkpoint_config_sha256": file_hash(a.model / "config.json"),
        "tokenizer_sha256": file_hash(a.model / "tokenizer.json"),
        "mode": "pruning_mask_only"
        if allocation
        else "packed_candidate"
        if packed and not packed.get("reference")
        else "unpruned_reference",
        "allocation_sha256": file_hash(a.allocation) if allocation else None,
        "slices": {},
    }
    for name, paths in slices.items():
        ids = tokenizer.encode("\n\n".join(p.read_text() for p in paths)).ids
        if len(ids) < a.tokens:
            raise ValueError(f"slice {name} has {len(ids)} tokens; need {a.tokens}")
        ids = np.asarray(ids[: a.tokens], np.int32)
        np.save(a.output / f"{name}-tokens.npy", ids)
        cache = model.make_cache(max_seq_len=a.tokens + 1)
        logits = np.lib.format.open_memmap(
            a.output / f"{name}-logits.npy",
            mode="w+",
            dtype=np.float32,
            shape=(a.tokens, config.vocab_size),
        )
        for layer_id in masses:
            masses[layer_id].clear()
        for pos, token in enumerate(ids):
            value = model(mx.array([[int(token)]]), cache, last_logit_only=True)
            mx.eval(value)
            logits[pos] = np.array(value).reshape(-1)
            if (pos + 1) % 16 == 0:
                logits.flush()
                print(
                    json.dumps(
                        {"slice": name, "tokens": pos + 1, "mode": metadata["mode"]}
                    ),
                    flush=True,
                )
        logits.flush()
        del logits, cache
        if allocation:
            mass = np.array([masses[layer_id] for layer_id in range(config.n_layers)]).T
        else:
            mass = np.ones((a.tokens, config.n_layers), np.float32)
        np.save(a.output / f"{name}-mass.npy", mass)
        metadata["slices"][name] = {
            "tokens": a.tokens,
            "corpus_sha256": [file_hash(p) for p in paths],
            "logits_sha256": file_hash(a.output / f"{name}-logits.npy"),
        }
    (a.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def compare(reference: Path, candidate: Path):
    rm = json.loads((reference / "metadata.json").read_text())
    cm = json.loads((candidate / "metadata.json").read_text())
    if rm["mode"] != "unpruned_reference":
        raise ValueError("reference must be an unpruned cache")
    if (
        rm["model_revision"] != cm["model_revision"]
        or rm["tokenizer_sha256"] != cm["tokenizer_sha256"]
        or set(rm["slices"]) != set(cm["slices"])
        or not rm["slices"]
    ):
        raise ValueError("cache provenance/slice mismatch")
    results = {}
    for name in rm["slices"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("unsafe slice name")
        if rm["slices"][name]["corpus_sha256"] != cm["slices"][name]["corpus_sha256"]:
            raise ValueError("cache corpus mismatch")
        for root, meta in ((reference, rm), (candidate, cm)):
            if (
                file_hash(root / f"{name}-logits.npy")
                != meta["slices"][name]["logits_sha256"]
            ):
                raise ValueError("cache logits hash mismatch")
        if not np.array_equal(
            np.load(reference / f"{name}-tokens.npy", allow_pickle=False),
            np.load(candidate / f"{name}-tokens.npy", allow_pickle=False),
        ):
            raise ValueError("teacher-forced token IDs differ")
        r = np.load(reference / f"{name}-logits.npy", mmap_mode="r", allow_pickle=False)
        c = np.load(candidate / f"{name}-logits.npy", mmap_mode="r", allow_pickle=False)
        if r.shape != c.shape or len(r) == 0:
            raise ValueError("cache logits shapes differ or are empty")
        total_kl, flips = 0.0, 0
        for start in range(0, len(r), 8):
            kl, flip = token_metrics(r[start : start + 8], c[start : start + 8])
            total_kl += float(kl.sum())
            flips += int(flip.sum())
        mass = np.load(candidate / f"{name}-mass.npy", allow_pickle=False)
        if (
            mass.ndim != 2
            or mass.shape[0] != len(r)
            or not np.isfinite(mass).all()
            or np.any((mass < 0) | (mass > 1 + 1e-6))
        ):
            raise ValueError("invalid routing mass cache")
        results[name] = {
            "tokens": len(r),
            "kl_nats": total_kl / len(r),
            "top1_flip_rate": flips / len(r),
            "mean_retained_routing_mass": float(mass.mean()),
        }
    return {
        "mode": cm["mode"],
        "slices": results,
        "worst_domain_kl_nats": max(v["kl_nats"] for v in results.values()),
        "mean_domain_kl_nats": float(np.mean([v["kl_nats"] for v in results.values()])),
    }


def select_knee(points, threshold=0.02):
    """Largest rise in worst-domain KL/GB; select one larger tested size."""
    ordered = sorted(points, key=lambda p: p["budget_gb"], reverse=True)
    if (
        len(ordered) < 3
        or threshold < 0
        or len({p["budget_gb"] for p in ordered}) != len(ordered)
    ):
        raise ValueError("need >=3 distinct budgets and a nonnegative fixed threshold")
    slopes = [
        (b["worst_domain_kl_nats"] - a["worst_domain_kl_nats"])
        / (a["budget_gb"] - b["budget_gb"])
        for a, b in zip(ordered, ordered[1:])
    ]
    if max(slopes) <= 0:
        raise ValueError("no KL knee detected; extend sweep")
    edge = int(np.argmax(slopes))
    # The knee is the smaller endpoint of this edge; safe side is its predecessor.
    eligible = [
        p for p in ordered[: edge + 1] if p["worst_domain_kl_nats"] <= threshold
    ]
    if not eligible:
        raise ValueError(
            "no safe-side candidate passes the fixed worst-domain KL threshold"
        )
    return eligible[-1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("collect")
    cache.add_argument(
        "--model",
        type=Path,
        required=True,
        help="converted native checkpoint, never raw FP4 release",
    )
    cache.add_argument("--model-revision", required=True)
    cache.add_argument("--slices", type=Path, required=True)
    cache.add_argument("--tokens", type=int, default=128)
    cache.add_argument("--allocation", type=Path)
    cache.add_argument(
        "--serving-qmv",
        action="store_true",
        help="exercise guarded serving down-QMV where eligible",
    )
    cache.add_argument("--output", type=Path, required=True)
    metrics = sub.add_parser("compare")
    metrics.add_argument("--reference", type=Path, required=True)
    metrics.add_argument("--candidate", type=Path, required=True)
    metrics.add_argument("--output", type=Path, required=True)
    select = sub.add_parser("select")
    select.add_argument(
        "--points",
        type=Path,
        required=True,
        help="JSON list: budget_gb, worst_domain_kl_nats",
    )
    select.add_argument("--worst-domain-kl-threshold", type=float, default=0.02)
    a = p.parse_args()
    if a.command == "collect":
        collect(a)
    elif a.command == "compare":
        result = compare(a.reference, a.candidate)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    else:
        print(
            json.dumps(
                select_knee(
                    json.loads(a.points.read_text()), a.worst_domain_kl_threshold
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
