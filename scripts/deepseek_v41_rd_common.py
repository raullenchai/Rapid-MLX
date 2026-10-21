"""Pure CPU contracts shared by the experimental V4.1 RD tools."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

BITS = np.array([0, 2, 3, 4])
GROUP_SIZE = 64


def require_all_moe_layers(config, layers_with_experts):
    """Reject dense FFN layers, whether declared or absent from the index."""
    c = config.get("text_config", config)
    missing = sorted(
        set(range(c.get("num_hidden_layers", c.get("n_layers")))) - layers_with_experts
    )
    if (
        c.get("first_k_dense_replace", 0) > 0
        or c.get("first_k_dense", 0) > 0
        or missing
    ):
        raise ValueError(
            "dense FFN layers are not supported: this tooling targets all-MoE "
            "DeepSeek V4.1 checkpoints "
            f"(layers without routed experts: {missing})"
        )


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_slices(manifest: Path | None, corpus: list[Path]) -> dict[str, list[Path]]:
    if manifest is None:
        return {"general": corpus}
    raw = json.loads(manifest.read_text())
    result = {
        name: [(manifest.parent / p).resolve() for p in paths]
        for name, paths in raw["slices"].items()
    }
    if not result or any(not paths for paths in result.values()):
        raise ValueError("each slice needs at least one local text file")
    return result


def aggregate_scores(reap, norms, squared_norms, counts):
    """Inputs [slice, half, layer, expert]; no gate factors in MAN/MSAN."""
    arrays = [
        np.asarray(a, dtype=np.float64) for a in (reap, norms, squared_norms, counts)
    ]
    if len({a.shape for a in arrays}) != 1 or arrays[0].ndim != 4:
        raise ValueError("score arrays must agree: [slice, half, layer, expert]")
    if any(np.any(~np.isfinite(a)) or np.any(a < 0) for a in arrays):
        raise ValueError("scores must be finite and nonnegative")
    r, n, n2, c = arrays
    sums = {
        "reap": r.sum(1),
        "man": n.sum(1) / np.maximum(c.sum(1), 1),
        "msan": n2.sum(1) / np.maximum(c.sum(1), 1),
    }
    halves = {"reap": r, "man": n / np.maximum(c, 1), "msan": n2 / np.maximum(c, 1)}
    out = {
        "counts_by_slice": c.sum(1).astype(np.int64),
        "counts_halves_by_slice": c.astype(np.int64),
    }
    for name, values in sums.items():
        out[name + "_by_slice"] = values
        out[name + "_halves"] = halves[name]
        # Keep layer magnitudes: normalizing each layer separately would erase
        # the cross-layer sensitivity needed by a global byte allocator.
        out["robust_" + name] = values.max(axis=0)
    return out


def affine_bytes(shape, bits: int, scale_bytes: int = 2) -> int:
    """Exact tensor payload: uint32 packed words plus scales and affine biases."""
    if bits not in (2, 3, 4) or shape[-1] % GROUP_SIZE:
        raise ValueError("requires bits 2/3/4 and group-64 input dimension")
    rows = int(np.prod(shape[:-1]))
    cols = int(shape[-1])
    if cols * bits % 32:
        raise ValueError("packed row must contain whole uint32 words")
    return rows * (cols * bits // 8 + cols // GROUP_SIZE * 2 * scale_bytes)


def bank_plan(bits, topk: int = 6):
    """Packing design contract; original IDs -> compact router -> bank/local IDs."""
    values = np.asarray(bits)
    if values.ndim != 1 or not np.isin(values, BITS).all():
        raise ValueError("invalid expert bit widths")
    kept = np.flatnonzero(values)
    if len(kept) < topk:
        raise ValueError("allocation leaves fewer experts than routing top-k")
    banks = {
        str(b): np.flatnonzero(values == b).tolist()
        for b in (2, 3, 4)
        if np.any(values == b)
    }
    original_to_compact = np.full(len(values), -1, dtype=int)
    original_to_compact[kept] = np.arange(len(kept))
    return {
        "kept_expert_ids": kept.tolist(),
        "banks": banks,
        "original_to_compact": original_to_compact.tolist(),
        "compact_to_bank": [int(values[e]) for e in kept],
        "compact_to_local": [banks[str(values[e])].index(int(e)) for e in kept],
    }


def masked_route(scores, bias, kept, topk=6):
    scores, bias = np.asarray(scores), np.asarray(bias)
    ids = np.asarray(kept, dtype=int)
    if ids.ndim != 1 or len(set(ids.tolist())) != len(ids) or len(ids) < topk:
        raise ValueError("keep IDs must be unique and preserve top-k")
    if np.any(ids < 0) or np.any(ids >= scores.shape[-1]):
        raise ValueError("expert ID outside router")
    rank = scores + bias
    ref = np.argsort(-rank, axis=-1, kind="stable")[..., :topk]
    weights = np.take_along_axis(scores, ref, axis=-1)
    mass = (weights * np.isin(ref, ids)).sum(-1) / np.maximum(weights.sum(-1), 1e-30)
    mask = np.zeros(scores.shape[-1], bool)
    mask[ids] = True
    chosen = np.argsort(-np.where(mask, rank, -np.inf), axis=-1, kind="stable")[
        ..., :topk
    ]
    w = np.take_along_axis(scores, chosen, axis=-1)
    return chosen, w / np.maximum(w.sum(-1, keepdims=True), 1e-30), mass
