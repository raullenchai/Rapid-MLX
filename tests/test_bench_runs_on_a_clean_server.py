# SPDX-License-Identifier: Apache-2.0
"""The perf bench must measure a server that has served nothing else.

`stress_e2e_bench` used to run the bench LAST, after the stress battery
and the whole SDK matrix had hammered the same server process. It
therefore measured residual state rather than the code under review,
while baselines are captured on a fresh server
(`harness/README.md`) — two different protocols compared against each
other.

Measured on Qwen3.5-35B-A3B-8bit / M3 Ultra, each group highly
reproducible within itself::

    after stress + agents : cold 287.6, 288.2 ms
    fresh server          : cold 252.8, 253.1, 252.0 ms

A ~14% cold gap with under 0.5% spread inside each group — far past the
5% threshold, so the gate reported a "regression" for a change that only
edits prompt assembly and a regex, and the identical delta appeared on
main. Warm moved the other way (~4% faster once the engine is hot),
which is what made the symptom look like noise.

Ordering is the fix, so ordering is what this pins.
"""

import ast
import inspect

from scripts.pr_validate.steps import stress_e2e_bench


def _run_body():
    return ast.parse(
        inspect.getsource(stress_e2e_bench.StressE2EBenchStep.run).lstrip()
    )


def _capture_runner_lines():
    """``{label: lineno}`` for each literal ``_capture_runner("<label>", ...)``.

    Ordering comes from ``lineno``, never from traversal order:
    ``ast.walk`` is breadth-first, so it yields shallower nodes before
    deeper ones regardless of where they sit in the source. The bench
    call and the agent-matrix call are at different nesting depths — the
    latter lives inside a ``for`` — so a walk-order test would have
    reported an order the file does not have, and would have kept
    passing with the bench moved back to the end (raised in review).
    """
    lines = {}
    for node in ast.walk(_run_body()):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, "id", None) or getattr(fn, "attr", None)
        if name != "_capture_runner" or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            lines[first.value] = first.lineno
    return lines


def _sdk_matrix_lineno():
    """Line of ``for agent in registry["agents"]:`` — the matrix loop.

    The ``agents`` key holds SDK/framework scripts (Anthropic SDK, LangChain,
    Pydantic-AI), not the coding agents of ``docs/agents/matrix.md``. The key
    name is load-bearing for this AST lookup, so it is matched literally even
    though the prose calls it the SDK matrix.
    """
    for node in ast.walk(_run_body()):
        if not isinstance(node, ast.For):
            continue
        it = node.iter
        if (
            isinstance(it, ast.Subscript)
            and isinstance(it.value, ast.Name)
            and it.value.id == "registry"
            and isinstance(it.slice, ast.Constant)
            and it.slice.value == "agents"
        ):
            return node.lineno
    raise AssertionError("no `for agent in registry['agents']` loop found")


def test_bench_runs_before_the_stress_battery():
    lines = _capture_runner_lines()
    assert "bench" in lines, f"no bench runner found; got {sorted(lines)}"
    assert "stress" in lines, f"no stress runner found; got {sorted(lines)}"

    order = sorted(lines, key=lines.get)
    assert lines["bench"] < lines["stress"], (
        "the bench must run BEFORE the stress battery, on a server that has "
        f"served nothing else. Runner order is {order}. Benching afterwards "
        "measures residual state and cannot be compared against a baseline "
        "captured on a fresh server — a ~14% cold gap on Qwen3.5-35B-A3B-8bit."
    )


def test_bench_runs_before_the_sdk_matrix():
    """The SDK matrix is the heavier half of the contamination."""
    lines = _capture_runner_lines()
    assert "bench" in lines, f"no bench runner found; got {sorted(lines)}"

    assert lines["bench"] < _sdk_matrix_lineno(), (
        "the bench must run before the SDK matrix; benching after it "
        "measures a server that has already served every integration battery"
    )
