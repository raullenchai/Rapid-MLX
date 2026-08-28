#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Parser microbenchmark — catch >10× regressions in tool-call extraction.

Tool-call parsers run on every streamed chunk + on the final non-
streaming pass. A 10x regression in the parser cuts effective TPS
proportionally on tool-calling workloads. Unit tests cover correctness
but won't catch "still correct, 10x slower" — which has shipped twice
historically (regex rebuilding in a hot path, AST walk where a string
search worked).

The bench uses a same-run machine factor and a generous relative budget
because ubuntu-latest is shared hardware and performance varies between
runs. The point is to catch *order-of-magnitude* regressions, not
fine-grained perf tracking. For real perf measurement use a M3 + a
stable baseline; see `docs/development/releasing.md` §"Pre-release
validation gauntlet".

Per parser:

* hermes — `<tool_call>{json}</tool_call>` wrap (Qwen family)
* minimax — `<tool_call><function>...args</tool_call>` (MiniMax M2)
* glm47 — `<tool_call>{json}</tool_call>` (GLM 4.7)
* harmony — `<|channel|>commentary to=name<|message|>{json}<|call|>` (gpt-oss)

Usage:
    python3 scripts/microbench_parsers.py            # bench + threshold gate
    python3 scripts/microbench_parsers.py --report   # bench + print only
    python3 scripts/microbench_parsers.py --iters 100  # smoke run

Exit 0 = all parsers under threshold (or --report mode), exit 1 = any
parser over threshold.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Relative threshold budget (issue #2344).
#
# Historical design: an ABSOLUTE μs/call ceiling per parser, hand-tuned to
# M3 with slack for shared runners. That flakes on hosted runners — the whole
# runner slows down and an unchanged parser trips an absolute wall-clock gate
# (glm47 90.15 μs vs an 80 μs gate on a busy ubuntu-latest job, 1.13×, no
# parser change). Mature perf gates avoid absolute wall-clock on shared
# hardware by comparing against a SAME-RUN baseline / relative budget.
#
# New design — a relative budget that normalizes runner speed:
#
#   parser_us_per_call  ≈  BASE_US[name] × runner_factor × ε
#   effective_threshold  =  BASE_US[name] × REGRESSION_LIMIT × runner_factor
#
# where ``runner_factor`` is how much slower THIS runner is than the reference M3,
# measured from the calibration workload in the same run. When calibration and
# parser costs scale together, ``runner_factor`` scales both sides and cancels,
# leaving ``ε ≤ REGRESSION_LIMIT``. Interleaved rounds plus a median handle
# isolated scheduling pauses. Regex calibration cannot model every parser
# operation across architectures, so hosted CI uses ``--report``; the enforced
# verdict runs serially on the stable M3 release host. (Note
# ``_CAL_REF_US_PER_ITER`` cancels when both workloads scale together, so its
# precise value only affects the printed μs scale, not that verdict.)
# ---------------------------------------------------------------------------

# Intrinsic per-parser cost on the reference M3, μs/call (from the historical
# notes: hermes ~5.5 μs, glm47 similar shape, minimax/harmony heavier).
BASE_US: dict[str, float] = {
    "hermes": 5.5,
    "minimax": 8.0,
    "glm47": 5.5,
    "harmony": 11.0,
}

# The relative gate: a parser may be at most this many × its own calibrated
# baseline before the bench goes red. 12× keeps the documented "order of
# magnitude" bar with a little headroom so a borderline run doesn't flake.
REGRESSION_LIMIT: float = 12.0

# Calibration workload: cheap, pure-string (no mlx), representative of a
# parser's hot path (regex locating tool-call markers in a parser-shaped
# string). Every parser shares ONE calibration so the per-parser base only
# carries intrinsic-cost differences, not runner-variance.
_CAL_STRING = (
    "<tool_call>get_weather\n"
    "<arg_key>city</arg_key>\n<arg_value>San Francisco</arg_value>\n"
    "</tool_call>"
)
_CAL_PATTERN = re.compile(r"<tool_call>.*?</tool_call>", re.S)
_CAL_ITERS: int = 20_000
_CAL_REPS: int = 5
# Reference cost of one calibration op, MEASURED on the same reference M3
# used for ``BASE_US`` (warmed, 100k iters, 9 reps → 0.476 μs/op; set to
# 0.48). Because both ``BASE_US`` and this reference are measured on the same
# M3, their ratio is the parser's intrinsic cost expressed in calibration-ops
# — so on any runner the gate reduces to ε ≤ REGRESSION_LIMIT regardless of
# that runner's speed (#2344). This is a real measurement, not a knob to
# relax the gate.
_CAL_REF_US_PER_ITER: float = 0.48

# Floor so a suspiciously-fast calibration can't compress the effective
# threshold below the reference machine's own budget (guards the signal).
_RUNNER_FACTOR_FLOOR: float = 0.5

# Realistic sample inputs for each parser. Each represents a single
# tool call the parser should successfully extract — not edge cases,
# not malformed input. The hot-path test is "happy path, fast"; edge
# cases are covered by unit tests in `tests/test_tool_parsers.py`.
SAMPLES: dict[str, str] = {
    "hermes": (
        "<tool_call>\n"
        '{"name": "get_weather", "arguments": {"city": "San Francisco"}}\n'
        "</tool_call>"
    ),
    "minimax": (
        "<minimax:tool_call>\n"
        '<minimax:invoke name="get_weather">\n'
        '<minimax:parameter name="city">San Francisco</minimax:parameter>\n'
        "</minimax:invoke>\n"
        "</minimax:tool_call>"
    ),
    "glm47": (
        "<tool_call>get_weather\n"
        "<arg_key>city</arg_key>\n"
        "<arg_value>San Francisco</arg_value>\n"
        "</tool_call>"
    ),
    "harmony": (
        "<|channel|>commentary to=functions.get_weather"
        '<|message|>{"city": "San Francisco"}<|call|>'
    ),
}


@dataclass
class BenchResult:
    name: str
    total_ms: float
    us_per_call: float
    iters: int
    threshold_us: float
    # Runner slowdown factor (vs the M3 reference) on the VERDICT round — the
    # pair whose numbers are reported, so the printed threshold always matches
    # the printed factor (Codex #2409 NIT: the old display-only global
    # calibrate-then-print could disagree with the actual verdict).
    runner_factor: float
    passed: bool


def _build_parsers() -> dict[str, Callable[[str], object]]:
    """Build (name → callable) map. Defer imports until needed because
    the harmony parser pulls openai-harmony which may not be installed
    on every CI run (it's a soft dep)."""
    parsers: dict[str, Callable[[str], object]] = {}

    from vllm_mlx.tool_parsers.hermes_tool_parser import HermesToolParser

    hermes = HermesToolParser()
    parsers["hermes"] = lambda text: hermes.extract_tool_calls(text, None)

    from vllm_mlx.tool_parsers.minimax_tool_parser import MiniMaxToolParser

    minimax = MiniMaxToolParser()
    parsers["minimax"] = lambda text: minimax.extract_tool_calls(text, None)

    from vllm_mlx.tool_parsers.glm47_tool_parser import Glm47ToolParser

    glm47 = Glm47ToolParser()
    parsers["glm47"] = lambda text: glm47.extract_tool_calls(text, None)

    try:
        from vllm_mlx.tool_parsers.harmony_tool_parser import HarmonyToolParser

        harmony = HarmonyToolParser()
        parsers["harmony"] = lambda text: harmony.extract_tool_calls(text, None)
    except (ImportError, RuntimeError) as e:
        # Soft dep — if openai-harmony isn't importable, skip this
        # parser rather than fail the gate. The real check is the
        # OTHER parsers passing their threshold.
        print(f"  [skip] harmony parser unavailable: {e}", file=sys.stderr)

    return parsers


def _run_calibration() -> float:
    """Time one pass of the calibration workload, return μs per op."""
    t0 = time.perf_counter()
    for _ in range(_CAL_ITERS):
        _CAL_PATTERN.search(_CAL_STRING)
    dt = time.perf_counter() - t0
    return (dt / _CAL_ITERS) * 1_000_000


def _measure_runner_factor() -> float:
    """How much slower this runner is than the reference M3, right now.

    Times the calibration workload a few times and takes the MIN so a
    transient slow stretch during calibration can't over-relax the gate
    (issue #2344: the whole point is to normalize the runner's *typical*
    speed this run). Returns the ratio (≈ 1 on an M3, larger on slower
    shared runners), floored at ``_RUNNER_FACTOR_FLOOR`` (0.5) so a
    suspiciously fast measurement can't compress the effective threshold
    below the reference budget. ``bench_one`` calls this fresh for EACH
    round (right before each parser segment) so a runner that becomes busy
    mid-run still normalizes the parser it is about to time (issue #2344 #2).
    """
    per_op = min(_run_calibration() for _ in range(_CAL_REPS))
    runner_factor = per_op / _CAL_REF_US_PER_ITER
    return max(runner_factor, _RUNNER_FACTOR_FLOOR)


def _median_verdict(
    pairs: list[tuple[float, float]], base_us: float
) -> tuple[float, float, int]:
    """Robust (MEDIAN) ε across interleaved ``(parser_us, runner_factor)`` rounds.

    Returns ``(eps, verdict_factor, verdict_index)``. Median, not max
    (Codex #2409): ``max`` lets ONE round where the runner descheduled during
    the parser segment (but not the adjacent calibration — the back-to-back
    interleave) dominate an already-noisy ratio and false-fail the now-
    ENFORCED gate, amplifying the original shared-runner flake. A genuine
    regression is slow in EVERY round → every ε is high → the median still
    fails it; a single transient spike lifts only one round, which the median
    ignores. ``verdict_index`` points at the median-ε round so the caller can
    report the pair that PRODUCED the verdict (printed numbers always agree
    with pass/fail).
    """
    per_round_eps = [us / (base_us * runner_factor) for us, runner_factor in pairs]
    idx = sorted(range(len(pairs)), key=lambda i: per_round_eps[i])[len(pairs) // 2]
    return per_round_eps[idx], pairs[idx][1], idx


def bench_one(
    name: str,
    fn: Callable[[str], object],
    sample: str,
    iters: int,
    *,
    runner_factor: float | None = None,
) -> BenchResult:
    """Run ``fn(sample)`` ``iters`` times, return timing + verdict.

    Relative-budget gate (issue #2344): the parser's M3 base scaled by the
    same-run runner factor and the regression limit,
    ``threshold = BASE_US[name] × REGRESSION_LIMIT × runner_factor``. Because both
    the parser measurement and the calibration baseline scale with the same
    runner speed, the verdict reduces to ``ε ≤ REGRESSION_LIMIT`` — a
    runner-speed-normalized order-of-magnitude regression check when the
    control and parser costs scale together. Hosted CI therefore reports this
    value without enforcing it; the stable M3 release lane enforces it.

    To be robust to load changing mid-run, calibration and the parser bench
    are INTERLEAVED: each round times the calibration then the parser
    immediately after, and the verdict uses the MEDIAN ε across rounds (a
    robust aggregate, not the max). A regressed parser is slow in every
    round, so every ε is high and the median still fails it; a single round
    where the runner descheduled during the parser segment (but not the
    adjacent calibration) inflates only that one ε, which the median ignores
    — so the aggregate does not false-fail on one transient shared-runner
    pause (Codex #2409). A transiently-idle parser adjacent to
    a busy calibration reads as FAST relative to that busy baseline (low ε),
    so it does not false-fail either.

    ``runner_factor`` overrides the calibration when provided (unit tests);
    when ``None`` (production), each round's factor is measured live.
    """
    # Every parser the bench knows about must carry an explicit ``BASE_US``
    # entry; silently defaulting an unknown name to a magic 5.0 would mask a
    # wiring mistake (new parser added to the loop but forgotten in BASE_US)
    # behind an arbitrary threshold (Codex #2409 NIT).
    if name not in BASE_US:
        raise KeyError(
            f"no BASE_US baseline for parser {name!r}; every parser in "
            f"_build_parsers() must have an explicit μs/call baseline "
            f"(see BASE_US in scripts/microbench_parsers.py)"
        )
    base_us = BASE_US[name]
    # Interleave calibration and parser timing across rounds; with a tiny
    # ``iters`` (unit tests) a single round suffices.
    rounds = _CAL_REPS if iters >= _CAL_REPS else 1
    # Distribute ``iters`` across rounds so the full count is actually run:
    # the first ``iters % rounds`` rounds get one extra call, and every round
    # runs at least one. Accounting is exact — ``sum(n_per_round) == iters``
    # and the reported ``iters`` equals what was executed (issue #2344 review).
    per_round = [
        max(1, iters // rounds + (1 if i < iters % rounds else 0))
        for i in range(rounds)
    ]
    pairs: list[tuple[float, float]] = []
    t_total0 = time.perf_counter()
    for i in range(rounds):
        factor = _measure_runner_factor() if runner_factor is None else runner_factor
        n = per_round[i]
        t0 = time.perf_counter()
        for _ in range(n):
            fn(sample)
        dt = time.perf_counter() - t0
        us = (dt / n) * 1_000_000
        pairs.append((us, factor))
    total_ms = (time.perf_counter() - t_total0) * 1000
    iters_executed = sum(per_round)
    # Verdict = MEDIAN ε across all paired rounds (robust aggregate), not the
    # max — see ``_median_verdict`` (Codex #2409). The reported us_per_call and
    # threshold come from the SAME pair that produced the verdict (the
    # median-ε round), so the printed numbers always agree with pass/fail.
    eps, verdict_factor, median_idx = _median_verdict(pairs, base_us)
    threshold_us = base_us * REGRESSION_LIMIT * verdict_factor
    us_per_call = pairs[median_idx][0]
    return BenchResult(
        name=name,
        total_ms=total_ms,
        us_per_call=us_per_call,
        iters=iters_executed,
        threshold_us=threshold_us,
        runner_factor=verdict_factor,
        passed=eps <= REGRESSION_LIMIT,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--iters",
        type=int,
        default=10_000,
        help="Iterations per parser (default: 10000).",
    )
    p.add_argument(
        "--report",
        action="store_true",
        help="Print timing and exit 0 even if thresholds exceeded.",
    )
    args = p.parse_args(argv)

    parsers = _build_parsers()
    if not parsers:
        print("FAIL: no parsers loaded — import path broken", file=sys.stderr)
        return 1

    # NOTE (Codex #2409 NIT): no display-only upfront calibration. Each
    # verdict carries the factor of its own verdict round (see BenchResult),
    # so the printed factor ALWAYS describes the reported threshold. Pre-
    # measuring a global runner factor solely to print a headline cost
    # ~100k extra regex ops AND could disagree with the actual verdict.
    print(f"Parser microbench × {args.iters} iters/parser")
    print(
        f"{'parser':<12}{'us/call':>12}{'runner':>10}{'threshold':>14}{'verdict':>10}"
    )
    print("-" * 58)

    results: list[BenchResult] = []
    for name, fn in parsers.items():
        sample = SAMPLES.get(name, "")
        if not sample:
            print(f"FAIL: {name}: no sample wired", file=sys.stderr)
            return 1
        # bench_one interleaves calibration with the parser bench itself
        # (issue #2344 #2); no explicit upfront factor here.
        r = bench_one(name, fn, sample, args.iters)
        results.append(r)
        verdict = "OK" if r.passed else "FAIL"
        print(
            f"{r.name:<12}{r.us_per_call:>12.2f}{r.runner_factor:>10.2f}"
            f"{r.threshold_us:>14.2f}{verdict:>10}"
        )

    failed = [r for r in results if not r.passed]
    print()
    if not failed:
        print(f"All {len(results)} parsers under threshold. OK.")
        return 0
    print(
        f"⚠  {len(failed)}/{len(results)} parser(s) exceeded threshold:",
        file=sys.stderr,
    )
    for r in failed:
        ratio = r.us_per_call / r.threshold_us
        print(
            f"  {r.name}: {r.us_per_call:.2f} μs/call "
            f"(threshold {r.threshold_us:.2f} μs, {ratio:.2f}× over)",
            file=sys.stderr,
        )
    if args.report:
        print("(--report mode: exit 0 despite failures)", file=sys.stderr)
        return 0
    print(
        "\nIf this is a legitimate algorithm change (e.g. moving from "
        "regex to AST), adjust `BASE_US` / `REGRESSION_LIMIT` in "
        "`scripts/microbench_parsers.py` with a comment citing the PR + the "
        "new baseline measurement.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
