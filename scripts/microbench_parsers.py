#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Parser microbenchmark — catch >10× regressions in tool-call extraction.

Tool-call parsers run on every streamed chunk + on the final non-
streaming pass. A 10x regression in the parser cuts effective TPS
proportionally on tool-calling workloads. Unit tests cover correctness
but won't catch "still correct, 10x slower" — which has shipped twice
historically (regex rebuilding in a hot path, AST walk where a string
search worked).

The bench is intentionally generous on absolute numbers — ubuntu-
latest is shared hardware, perf varies by ±50% run to run. The point
is to catch *order-of-magnitude* regressions, not fine-grained perf
tracking. For real perf measurement use a M3 + a stable baseline; see
`docs/development/releasing.md` §"Pre-release validation gauntlet".

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
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

# Threshold: microseconds-per-call. Set 3-5x what's been measured on
# M3 + small buffer for ubuntu's variance + GIL contention. If a
# parser EXCEEDS its threshold, we want a hard signal; this is much
# looser than a "perf regression" check would be.
THRESHOLDS_US_PER_CALL: dict[str, float] = {
    "hermes": 30.0,  # measured ~5.5 μs on M3
    "minimax": 60.0,  # complex regex, larger budget
    "glm47": 40.0,  # similar shape to hermes
    "harmony": 80.0,  # multi-channel protocol, heavier
}

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


def bench_one(
    name: str, fn: Callable[[str], object], sample: str, iters: int
) -> BenchResult:
    """Run ``fn(sample)`` ``iters`` times, return timing + verdict.

    Uses ``perf_counter`` rather than ``time.time()`` for the
    monotonic+high-resolution guarantees the bench needs.
    """
    threshold_us = THRESHOLDS_US_PER_CALL.get(name, 100.0)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(sample)
    dt = time.perf_counter() - t0
    us_per_call = (dt / iters) * 1_000_000
    return BenchResult(
        name=name,
        total_ms=dt * 1000,
        us_per_call=us_per_call,
        iters=iters,
        threshold_us=threshold_us,
        passed=us_per_call <= threshold_us,
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

    print(f"Parser microbench × {args.iters} iters/parser")
    print(f"{'parser':<12}{'us/call':>12}{'threshold':>14}{'verdict':>10}")
    print("-" * 48)

    results: list[BenchResult] = []
    for name, fn in parsers.items():
        sample = SAMPLES.get(name, "")
        if not sample:
            print(f"  [skip] {name}: no sample wired", file=sys.stderr)
            continue
        r = bench_one(name, fn, sample, args.iters)
        results.append(r)
        verdict = "OK" if r.passed else "FAIL"
        print(f"{r.name:<12}{r.us_per_call:>12.2f}{r.threshold_us:>14.2f}{verdict:>10}")

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
        "regex to AST), bump the threshold in `scripts/microbench_parsers.py` "
        "with a comment citing the PR + the new baseline measurement.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
