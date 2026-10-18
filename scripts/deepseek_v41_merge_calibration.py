#!/usr/bin/env python3
"""Merge document-disjoint streaming calibration passes without loading a model.

Usage: python scripts/deepseek_v41_merge_calibration.py a.npz b.npz --output merged.npz

Passes must have identical per-slice token budgets. Half 0 pools input indices
0, 2, 4, ...; half 1 pools indices 1, 3, 5, ... (zero-based). Thus two inputs
become [pass A total, pass B total], discarding their original token-position
halves. Odd input counts give half 0 one extra pass; use an even number for
balanced stability evidence. Inputs must already be document-disjoint: score
files alone cannot establish document identity.

REAP and route counts are additive. MAN/MSAN are means, so recover their raw
norm sums from the saved means and route counts, then let the teacher recompute
all means, robust scores, aliases and stability checks. Floating-point recovery
of norm sums may introduce roundoff. Never sum MAN/MSAN means directly.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.deepseek_v41_rd_common import file_hash
from scripts.deepseek_v41_streaming_teacher import save_scores

COMPATIBILITY_FIELDS = (
    "schema_version",
    "model_revision",
    "config_sha256",
    "checkpoint_config_sha256",
    "tokenizer_sha256",
    "slice_names",
    "routing_topk",
    "keep_experts",
    "tokens_per_slice",
    "score_definitions",
    "normalization",
    "teacher",
    "engram",
    "execution",
    "split_overlap_columns",
)
MERGE_RULE = (
    "half_0=sum(input[0::2]); half_1=sum(input[1::2]); original halves collapsed"
)


def _read_pass(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        for key in COMPATIBILITY_FIELDS:
            if key not in metadata:
                raise ValueError(f"{path}: missing metadata {key}")
        names = metadata["slice_names"]
        budgets = metadata["tokens_per_slice"]
        if (
            not names
            or len(set(names)) != len(names)
            or len(budgets) != len(names)
            or any(type(t) is not int or t <= 0 for t in budgets)
            or len(set(budgets)) != 1
        ):
            raise ValueError(f"{path}: invalid slice_names or tokens_per_slice")
        counts = data["counts_halves_by_slice"]
        if (
            counts.ndim != 4
            or counts.shape[:2] != (len(names), 2)
            or not all(counts.shape)
            or counts.dtype.kind not in "iu"
            or np.any(counts < 0)
        ):
            raise ValueError(f"{path}: invalid counts_halves_by_slice")
        counts = counts.astype(np.int64)
        keep, topk = metadata["keep_experts"], metadata["routing_topk"]
        if (
            type(keep) is not int
            or type(topk) is not int
            or not 1 <= topk <= keep <= counts.shape[-1]
        ):
            raise ValueError(f"{path}: invalid keep_experts or routing_topk")
        for key, expected in (
            ("keep_experts", keep),
            ("tokens", sum(budgets)),
            ("method", "reap_output_norm"),
            ("selection_policy", "max_domain_with_half_stability"),
        ):
            if data[key].shape != () or data[key].item() != expected:
                raise ValueError(f"{path}: inconsistent {key}")
        if not np.array_equal(data["counts_by_slice"], counts.sum(1)):
            raise ValueError(f"{path}: inconsistent counts_by_slice")
        raw = []
        for metric in ("reap", "man", "msan"):
            halves = np.asarray(data[metric + "_halves"], dtype=np.float64)
            if (
                halves.shape != counts.shape
                or not np.isfinite(halves).all()
                or np.any(halves < 0)
                or np.any(halves[counts == 0] != 0)
            ):
                raise ValueError(f"{path}: invalid {metric}_halves")
            raw.append((halves if metric == "reap" else halves * counts).sum(1))
        return metadata, (*raw, counts.sum(1))


def merge_calibration(pass_files, output):
    """Validate all inputs, then create a new NPZ; never overwrite any file."""
    paths = [Path(p).resolve() for p in pass_files]
    output = Path(output)
    if len(paths) < 2:
        raise ValueError("need at least two pass files")
    if output.resolve() in paths:
        raise ValueError("output must differ from input pass files")
    sources, seen, pooled, metadata = [], set(), None, None
    for index, path in enumerate(paths):
        digest = file_hash(path)
        if digest in seen:
            raise ValueError(f"duplicate source pass: {path}")
        seen.add(digest)
        current, totals = _read_pass(path)
        if metadata is None:
            metadata = copy.deepcopy(current)
            pooled = [
                np.zeros((a.shape[0], 2, *a.shape[1:]), dtype=a.dtype) for a in totals
            ]
        for key in COMPATIBILITY_FIELDS:
            if current[key] != metadata[key]:
                raise ValueError(f"{path}: metadata mismatch: {key}")
        if any(a.shape != b[:, 0].shape for a, b in zip(totals, pooled)):
            raise ValueError(f"{path}: score array shape mismatch")
        if file_hash(path) != digest:
            raise ValueError(f"source changed while reading: {path}")
        for accumulator, total in zip(pooled, totals):
            accumulator[:, index % 2] += total
        sources.append(
            dict(
                file=str(path),
                sha256=digest,
                half=index % 2,
                corpus=current.get("corpus", {}),
                memory_estimate=current.get("memory_estimate"),
            )
        )
    budgets = metadata["tokens_per_slice"]
    metadata["tokens_per_slice"] = [t * len(paths) for t in budgets]
    # A pass's memory estimate and corpus must not masquerade as merged evidence.
    metadata.pop("memory_estimate", None)
    metadata["corpus"] = {
        name: [
            record for source in sources for record in source["corpus"].get(name, [])
        ]
        for name in metadata["slice_names"]
    }
    metadata["merge"] = dict(
        rule=MERGE_RULE,
        sources=sources,
        tokens_per_slice_halves=[
            [t * len(paths[h::2]) for t in budgets] for h in range(2)
        ],
        document_disjointness="caller responsibility; not provable from score arrays",
        mean_reconstruction="saved MAN/MSAN half means multiplied by route counts",
    )
    with output.open("xb") as handle:
        save_scores(
            handle,
            *pooled,
            metadata,
            metadata["keep_experts"],
            metadata["tokens_per_slice"][0],
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("passes", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    merge_calibration(args.passes, args.output)


if __name__ == "__main__":
    main()
