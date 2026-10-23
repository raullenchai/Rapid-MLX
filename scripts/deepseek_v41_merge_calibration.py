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
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.deepseek_v41_rd_common import aggregate_scores, file_hash

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


class CalibrationRollbackError(OSError):
    def __init__(self, failures):
        self.failures = tuple(failures)
        super().__init__(
            "calibration publication failed; rollback incomplete: "
            + "; ".join(str(error) for error in failures)
        )


def fsync_directory(path):
    """Persist publication names in an existing output directory."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_scores(path, reap, norms, squared, counts, metadata, keep, tokens):
    from scripts.calibrate_deepseek_v41_reap import _overlap

    counts = np.asarray(counts)
    if (
        counts.dtype.kind not in "iu"
        or np.any(counts < 0)
        or counts.sum(dtype=object) > np.iinfo(np.int64).max
    ):
        raise ValueError(
            "routing counts must be nonnegative integers with an int64-safe total"
        )
    counts = counts.astype(np.int64)
    with np.errstate(over="ignore", invalid="ignore"):
        scores = aggregate_scores(reap, norms, squared, counts)
        saliency = reap.sum((0, 1))
        saliency_halves = reap.sum(0)
    # Preserve exact integral routing evidence across the scorer's float arithmetic.
    scores["counts_halves_by_slice"] = counts
    scores["counts_by_slice"] = counts.sum(1)
    if any(
        not np.isfinite(value).all()
        for value in [*scores.values(), saliency, saliency_halves]
    ):
        raise ValueError("aggregate scores are not finite")
    overlap = np.array(
        [
            [
                _overlap(scores[m + "_halves"][d, 0], scores[m + "_halves"][d, 1], keep)
                for m in ("reap", "man", "msan")
            ]
            for d in range(len(reap))
        ]
    )
    metadata.update(keep_experts=keep, split_overlap_columns=["reap", "man", "msan"])
    np.savez(
        path,
        **scores,
        metadata=np.array(json.dumps(metadata)),
        saliency=saliency,
        robust_saliency=scores["robust_reap"],
        saliency_halves=saliency_halves,
        counts=counts.sum((0, 1)),
        counts_halves=counts.sum(0),
        tokens=np.array(tokens * len(reap)),
        keep_experts=np.array(keep),
        split_keep_overlap=overlap,
        method=np.array("reap_output_norm"),
        selection_policy=np.array("max_domain_with_half_stability"),
    )


def _read_pass(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if not isinstance(metadata, dict):
            raise ValueError(f"{path}: metadata must be an object")
        corpus = metadata.get("corpus")
        if corpus is None:
            metadata["corpus"] = {}
        elif not isinstance(corpus, dict) or any(
            not isinstance(records, list) for records in corpus.values()
        ):
            raise ValueError(f"{path}: corpus must map slice names to lists")
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
            or np.any(counts > np.iinfo(np.int64).max)
        ):
            raise ValueError(f"{path}: invalid counts_halves_by_slice")
        counts = counts.astype(np.int64)
        if np.any(counts < 0):
            raise ValueError(f"{path}: invalid counts_halves_by_slice")
        if np.any(counts[:, 0] > np.iinfo(np.int64).max - counts[:, 1]):
            raise ValueError(f"{path}: route-count sum overflows int64")
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
        if current.get("checkpoint_sha256") != metadata.get("checkpoint_sha256"):
            raise ValueError(f"{path}: metadata mismatch: checkpoint_sha256")
        if any(a.shape != b[:, 0].shape for a, b in zip(totals, pooled)):
            raise ValueError(f"{path}: score array shape mismatch")
        if file_hash(path) != digest:
            raise ValueError(f"source changed while reading: {path}")
        for accumulator, total in zip(pooled, totals):
            half = accumulator[:, index % 2]
            if accumulator.dtype.kind == "i" and np.any(
                total > np.iinfo(np.int64).max - half
            ):
                raise ValueError("merged route-count sum overflows int64")
            with np.errstate(over="ignore", invalid="ignore"):
                updated = half + total
            if not np.isfinite(updated).all():
                raise ValueError("merged scores are not finite")
            half[:] = updated
        sources.append(
            dict(
                file=str(path),
                sha256=digest,
                half=index % 2,
                corpus=current.get("corpus", {}),
                memory_estimate=current.get("memory_estimate"),
            )
        )
    if pooled[-1].sum(dtype=object) > np.iinfo(np.int64).max:
        raise ValueError("aggregate route-count sum overflows int64")
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
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent) as handle:
        save_scores(
            handle,
            *pooled,
            metadata,
            metadata["keep_experts"],
            metadata["tokens_per_slice"][0],
        )
        handle.flush()
        os.fsync(handle.fileno())
        # Link publication is atomic and refuses an existing destination.
        # Closing the temporary removes its name on success and on failure.
        os.link(handle.name, output)
        try:
            fsync_directory(output.parent)
        except BaseException as publication_error:
            failures = []
            try:
                if output.samefile(handle.name):
                    output.unlink()
            except FileNotFoundError:
                pass
            except OSError as error:
                failures.append(error)
            try:
                fsync_directory(output.parent)
            except OSError as error:
                failures.append(error)
            if failures:
                raise CalibrationRollbackError(failures) from publication_error
            raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("passes", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    merge_calibration(args.passes, args.output)


if __name__ == "__main__":
    main()
