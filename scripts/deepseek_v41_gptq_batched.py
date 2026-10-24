# Copyright © 2025 Apple Inc.
# SPDX-License-Identifier: MIT
# Adapted from the installed mlx_lm/quant/gptq.py.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Memory-bounded routed-expert GPTQ; shared experts retain the scalar path.

RAPID_MLX_GPTQ_EXPERT_BATCH sets a maximum expert chunk (1 for a scalar
reference). RAPID_MLX_GPTQ_WORKING_GB sets the working allowance, default 48 GB,
additionally capped at 25% of the run memory cap and available physical memory.
RAPID_MLX_GPTQ_FACTOR_WORKERS controls the persistent NumPy-only pool (default 4).
These execution controls do not alter the recipe or artifact/statistics schema.
"""

from __future__ import annotations

import gc
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np

from scripts.deepseek_v41_gptq_quant import gptq_batched, gram
from scripts.deepseek_v41_gptq_quant import inverse_factor as inverse_factor


def inverse_factor_numpy(h, config, count):
    """NumPy-only counterpart of inverse_factor; never touches MLX.

    Preserve the float32 input rounding, float64 arithmetic/call order, float32
    output, and statistics of the scalar implementation. Own the damped copy so
    callers may safely retain or reuse the input Gram.
    """
    h = np.array(h.astype(np.float32), dtype=np.float64)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("Hessian must be square")
    diag = np.diag(h).copy()
    if not np.all(np.isfinite(h)) or np.any(diag < 0):
        raise ValueError("nonfinite or negative Hessian")
    dead = diag <= 0
    coefficient = (
        config.low_token_damping if count < config.min_tokens else config.damping
    )
    damp = max(coefficient * float(np.mean(diag)), 1e-6)
    h[np.diag_indices_from(h)] += damp
    lower_inv = np.linalg.inv(np.linalg.cholesky(h))
    upper = np.linalg.cholesky(lower_inv.T @ lower_inv).T
    return (
        upper.astype(np.float32),
        dead,
        dict(
            tokens=count,
            low_token=count < config.min_tokens,
            damping_coefficient=coefficient,
            damping_absolute=damp,
            dead_columns=int(np.sum(dead)),
        ),
    )


def factor_workers():
    workers = int(os.environ.get("RAPID_MLX_GPTQ_FACTOR_WORKERS", "4"))
    if workers < 1:
        raise ValueError("invalid GPTQ factor worker count")
    return workers


def expert_batch_size(args, c):
    import psutil

    maximum = int(os.environ.get("RAPID_MLX_GPTQ_EXPERT_BATCH", "64"))
    working = float(os.environ.get("RAPID_MLX_GPTQ_WORKING_GB", "48")) * 1e9
    if maximum < 1 or not math.isfinite(working) or working <= 0:
        raise ValueError("invalid GPTQ expert batch or working memory budget")
    budget = min(
        working, c["memory_limit_gb"] * 1e9 / 4, psutil.virtual_memory().available / 4
    )
    # Includes stacked/source/updated weights, codes, factors and block temporaries.
    # Reserve one extra batch of host inputs/results plus concurrent float64
    # LAPACK scratch (four dense matrices per worker, beyond host input).
    rows, cols = max(args.dim, args.moe_inter_dim), max(args.dim, args.moe_inter_dim)
    per_expert = 4 * (3 * cols * cols + 6 * rows * min(args.dim, args.moe_inter_dim))
    per_expert += 8 * cols * cols
    scratch = (
        32 * cols * cols * min(factor_workers(), 2 * maximum, args.n_routed_experts)
    )
    capacity = int((budget - scratch) // per_expert)
    if capacity < 1:
        raise MemoryError("GPTQ working memory budget cannot hold one expert")
    return min(maximum, args.n_routed_experts, capacity)


class RoutedSpill:
    """One RAM activation copy with stable expert-sorted routing metadata.

    Sorting activations would replicate each token K times (not one extra X),
    so keep X unsorted and gather bounded chunks from RAM. Stable sorting the
    flattened row-major IDs preserves np.where's row/slot order, even duplicates.
    At N=1,474,560, D=5120: X=15.10 GB, Y=30.20 GB. At K=6 routing
    positions/weights cost 0.106 GB resident; sort/load scratch is <0.5 GB.
    With the measured 48 GB baseline, heldout, pool scratch and headroom this
    targets <120 GB, below the recipe's 180 GB guard. No sorted X copy exists.
    """

    @staticmethod
    def allowance(total, args):
        # Includes output and conservative peak sort/read metadata temporaries.
        return total * (6 * args.dim + 48 * args.n_activated_experts)

    @staticmethod
    def read(path, dtype, shape, check):
        # Stream into an owning ndarray so mapping pages never add another X to
        # RSS. Bounded readinto checks the existing guard during large reads.
        data = np.empty(shape, dtype=dtype)
        view = memoryview(data).cast("B")
        with path.open("rb") as f:
            if os.fstat(f.fileno()).st_size != len(view):
                raise ValueError(f"spill size mismatch: {path}")
            for start in range(0, len(view), 64 * 1024**2):
                part = view[start : start + 64 * 1024**2]
                while part:
                    n = f.readinto(part)
                    if not n:
                        raise EOFError(f"truncated spill: {path}")
                    part = part[n:]
                check()
        return data

    def __init__(self, paths, total, args, c, guard):
        self.paths, self.total = paths, total
        self.chunk = c["ffn_chunk"]
        shape = (total, args.n_activated_experts)
        ids = self.read(paths["ids"], np.int32, shape, guard.check)
        gates = self.read(paths["gates"], np.float32, shape, guard.check)
        if ids.size and (ids.min() < 0 or ids.max() >= args.n_routed_experts):
            raise ValueError("invalid routed expert ID")
        counts = np.bincount(ids.ravel(), minlength=args.n_routed_experts)
        order = np.argsort(ids.ravel(), kind="stable")
        guard.check()
        self.offsets = np.concatenate(([0], np.cumsum(counts)))
        self.rows = order // args.n_activated_experts
        self.weights = gates.ravel()[order]
        del ids, gates, order
        self.x = self.read(paths["x"], np.uint16, (total, args.dim), guard.check)
        # Preserve preexisting output bytes as well as the normal all-zero file.
        self.output = self.read(paths["y"], np.float32, (total, args.dim), guard.check)
        guard.check()

    def chunks(self, expert=None):
        lo, hi = (
            (0, self.total) if expert is None else self.offsets[expert : expert + 2]
        )
        for start in range(lo, hi, self.chunk):
            end = min(hi, start + self.chunk)
            if expert is None:
                rows = np.arange(start, end)
                raw, gates = self.x[start:end], None
            else:
                rows = self.rows[start:end]
                raw, gates = self.x[rows], self.weights[start:end]
            yield rows, mx.array(raw).view(mx.bfloat16), gates

    def add_output(self, rows, values):
        """Accumulate every route, including repeated token rows in one chunk."""
        np.add.at(self.output, rows, values)

    def write_output(self, guard):
        # Shared experts and finalize_windows read Y only after routed returns.
        # Keep each exact output[rows] += update operation on float32 RAM above.
        with self.paths["y"].open("r+b") as f:
            view = memoryview(self.output).cast("B")
            for start in range(0, len(view), 64 * 1024**2):
                part = view[start : start + 64 * 1024**2]
                while part:
                    n = f.write(part)
                    if not n:
                        raise OSError("short routed output write")
                    part = part[n:]
                guard.check()
            f.flush()
        guard.check()


def check_spill_budget(total, args, c, heldout):
    import psutil

    reserve = RoutedSpill.allowance(total, args)
    if heldout is not None:
        reserve += RoutedSpill.allowance(heldout["total"], args)
    # Baseline measured at 48 GB; retain 12 GB for factor overlap/temporaries.
    # Also account for an unusually large current resident working set.
    limit = min(120e9, c["memory_limit_gb"] * 1e9)
    baseline = max(min(60e9, limit / 2), psutil.Process().memory_info().rss)
    if reserve + baseline > limit:
        raise MemoryError(
            f"RAM routed spill estimate exceeds budget: {reserve + baseline:.0f} > {limit:.0f}"
        )


def quantize_routed(source, layer, sink, specs, paths, total, c, guard, heldout):
    # Import lazily to avoid a driver/module cycle (including direct CLI execution).
    from scripts.deepseek_v41_gptq import (
        fit,
        intermediate,
        packed_module,
        projection_errors,
        quant_config,
        write_packed,
    )

    args = source.args
    config = quant_config(c)
    prefix = f"layers.{layer}.ffn."
    stats = {}
    check_spill_budget(total, args, c, heldout)
    calibration = RoutedSpill(paths, total, args, c, guard)
    observed_spill = (
        RoutedSpill(heldout["paths"], heldout["total"], args, c, guard)
        if heldout is not None
        else None
    )
    size = expert_batch_size(args, c)

    def inputs(expert, held=False):
        spill = observed_spill if held else calibration
        for _, x, g in spill.chunks(expert):
            yield x, g

    def submit_factors(chunks, dim):
        futures, counts = [], []
        for rows in chunks:
            h, n = gram(rows(), dim, check=guard.check)
            # Copy/evaluate on the main thread; even the input's destructor in a
            # worker must not release an MLX array. Queued Grams stay float32.
            futures.append(
                pool.submit(
                    inverse_factor_numpy, np.array(h.astype(mx.float32)), config, n
                )
                if n
                else None
            )
            counts.append(n)
            del h
            guard.check()
        return futures, counts

    def collect_factors(pending):
        futures, counts = pending
        factors = []
        for i, future in enumerate(futures):
            if future is None:
                factors.append(None)
            else:
                upper, dead, report = future.result()  # Fail closed on worker error.
                factors.append((mx.array(upper), mx.array(dead), report))
                futures[i] = None  # Release the Future's host result promptly.
            guard.check()
        return factors, counts

    def fits(weights, factors, counts, bits, chunks, observed):
        active = [i for i, n in enumerate(counts) if n]
        packed = {}
        if active and size > 1:
            batch = gptq_batched(
                mx.stack([weights[i] for i in active]),
                mx.stack([factors[i][0] for i in active]),
                bits,
                config,
                dead=mx.stack([factors[i][1] for i in active]),
                check=guard.check,
            )
            for j, i in enumerate(active):
                packed[i] = tuple(t[j] for t in batch)
        result = []
        for i, (w, n, factor) in enumerate(zip(weights, counts, factors)):
            if i not in packed:
                h = mx.broadcast_to(mx.array(0, mx.float32), (w.shape[-1], w.shape[-1]))
                result.append(
                    fit(
                        w,
                        h,
                        n,
                        bits,
                        c,
                        guard,
                        chunks[i],
                        factor=factor,
                        heldout_chunks=observed[i],
                    )
                )
                continue
            report = dict(factor[2])
            errors = projection_errors(w, packed[i], bits, chunks[i], c["group_size"])
            report.update(errors)
            report["calibration"] = dict(error_scope="calibration", **errors)
            report["heldout"] = dict(status="not_configured")
            if observed[i] is not None:
                errors = projection_errors(
                    w, packed[i], bits, observed[i], c["group_size"]
                )
                energy = errors.pop("calibration_output_energy")
                tokens = errors.pop("error_token_count")
                report["heldout"] = dict(
                    status="measured" if energy else "unobserved",
                    output_energy=energy,
                    error_scope="heldout_quantized_prefix_inputs",
                    tokens=tokens,
                    **{k: v if energy else None for k, v in errors.items()},
                )
            report["bits"] = bits
            result.append((packed[i], report))
        return result

    # Exactly one lookahead batch: current gate/up + next gate/up, or current
    # down + next gate/up. Never submit b+2 until b's down/output is finished.
    # Thus at most two batches of Hessians are in flight: at E=64, D=5120,
    # even storing both in fp64 would be 2 * 13.4 GB (decimal). Queued inputs
    # are actually fp32; only active workers allocate fp64 Hessian/scratch.
    # All MLX work, conversion and output accumulation stays on this thread.
    pool = ThreadPoolExecutor(
        max_workers=factor_workers(), thread_name_prefix="gptq-factor"
    )
    try:
        pending = submit_factors(
            [lambda e=e: inputs(e) for e in range(min(size, args.n_routed_experts))],
            args.dim,
        )
        for start in range(0, args.n_routed_experts, size):
            started = time.monotonic()
            experts = list(range(start, min(start + size, args.n_routed_experts)))
            chunks = [lambda e=e: inputs(e) for e in experts]
            observed = (
                [lambda e=e: inputs(e, True) for e in experts]
                if heldout
                else [None] * len(experts)
            )
            current, pending = pending, None
            if start + size < args.n_routed_experts:
                pending = submit_factors(
                    [
                        lambda e=e: inputs(e)
                        for e in range(
                            start + size, min(start + 2 * size, args.n_routed_experts)
                        )
                    ],
                    args.dim,
                )
            factors, counts = collect_factors(current)
            del current
            modules = [[] for _ in experts]
            for proj, final in (("w1", "gate_proj"), ("w3", "up_proj")):
                base = prefix + "experts." + final
                spec = specs[base]
                weights = [
                    source.tensor(prefix + f"experts.{e}.{proj}.weight")
                    for e in experts
                ]
                fitted = fits(weights, factors, counts, spec.bits, chunks, observed)
                if c["pilot"] and c["pilot_compare_gate_up"] and spec.bits == 2:
                    trials = fits(weights, factors, counts, 3, chunks, observed)
                    for (_, report), (_, trial) in zip(fitted, trials):
                        report["three_bit_gptq_relative_output_error"] = trial[
                            "gptq_relative_output_error"
                        ]
                        report["three_bit_heldout"] = trial["heldout"]
                    del trials
                for i, (e, w, (packed, report)) in enumerate(
                    zip(experts, weights, fitted)
                ):
                    stats[f"expert.{e}.{proj}"] = report
                    write_packed(sink, base, packed, e)
                    modules[i].append(
                        packed_module(w, packed, spec.bits, spec.group_size)
                    )
                del weights, fitted, w, packed
            del factors

            def down_rows(rows, mods):
                for x, g in rows():
                    yield intermediate(x, *mods, args.swiglu_limit, routed=True), g

            down_chunks = [
                lambda rows=rows, mods=mods: down_rows(rows, mods)
                for rows, mods in zip(chunks, modules)
            ]
            down_held = (
                [
                    lambda rows=rows, mods=mods: down_rows(rows, mods)
                    for rows, mods in zip(observed, modules)
                ]
                if heldout
                else [None] * len(experts)
            )
            factors, _ = collect_factors(
                submit_factors(down_chunks, args.moe_inter_dim)
            )
            base = prefix + "experts.down_proj"
            spec = specs[base]
            weights = [
                source.tensor(prefix + f"experts.{e}.w2.weight") for e in experts
            ]
            fitted = fits(weights, factors, counts, spec.bits, down_chunks, down_held)
            for e, w, mods, (packed, report), n in zip(
                experts, weights, modules, fitted, counts
            ):
                stats[f"expert.{e}.w2"] = report
                write_packed(sink, base, packed, e)
                down = packed_module(w, packed, spec.bits, spec.group_size)
                # Preserve ascending expert accumulation, including held-out output.
                for rows, x, g in calibration.chunks(e):
                    y = down(
                        intermediate(x, *mods, args.swiglu_limit, routed=True)
                    ).astype(mx.float32)
                    y *= mx.array(g)[:, None]
                    mx.eval(y)
                    calibration.add_output(rows, np.asarray(y))
                    guard.check()
                if observed_spill is not None:
                    for rows, x, g in observed_spill.chunks(e):
                        y = down(
                            intermediate(x, *mods, args.swiglu_limit, routed=True)
                        ).astype(mx.float32)
                        y *= mx.array(g)[:, None]
                        mx.eval(y)
                        observed_spill.add_output(rows, np.asarray(y))
                        guard.check()
                print(
                    json.dumps(
                        dict(
                            layer=layer,
                            expert=e,
                            tokens=n,
                            seconds=(time.monotonic() - started) / len(experts),
                            **guard.report(),
                        )
                    ),
                    flush=True,
                )
            del (
                factors,
                weights,
                fitted,
                modules,
                down_chunks,
                down_held,
                w,
                mods,
                packed,
                down,
            )
            gc.collect()
            mx.clear_cache()
    finally:
        # Cancel queued work and join running LAPACK calls on every exit path.
        pool.shutdown(wait=True, cancel_futures=True)
    calibration.write_output(guard)
    if observed_spill is not None:
        observed_spill.write_output(guard)
    return stats
