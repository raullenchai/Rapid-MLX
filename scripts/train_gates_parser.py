#!/usr/bin/env python3
"""Single source of truth: parse the exact gate definitions out of the hosted
CI workflows so that ``scripts/train_gates.sh`` reproduces, locally, the same
gates the GitHub Actions matrix runs.

This module is imported by the drift test
``tests/test_train_gates_matches_ci.py`` and executed as a subprocess by
``scripts/train_gates.sh`` (via ``python -m scripts.train_gates_parser``). It
must parse the workflows the same way in both cases; if the workflow layout
changes such that the parser can no longer find a gate, the drift test fails.

Gate surface parsed here (see ``scripts/train_gates.sh`` for the full gate
list):
  * Linux no-MLX pytest blocks (ci.yml test-matrix, "Run unit tests (no MLX
    required)" step) — a LIST of invocations, one per distinct ``pytest``
    process in the step, because ci.yml deliberately runs the automatically
    discovered unit directory and the isolated fake-MLX lifecycle directory in
    TWO separate processes.
  * Apple-MLX pytest roster + ``-m``/``-k`` filters (ci.yml test-apple-silicon,
    "Run MLX-dependent tests" step)
  * mypy budget invocation (ci.yml type-check job)
  * diff-cover invocation (ci.yml changed-lines-coverage job)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - yaml is a dev-only dependency
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
MAC_CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "rapid-mac-ci.yml"
MYPY_BUDGET_SCRIPT = "scripts/check_mypy_error_budget.py"

# A pytest target can be a directory, a file, or a selected node.
_TEST_TARGET = re.compile(r"^tests(?:/[A-Za-z0-9_.-]+)*(?:::[A-Za-z0-9_]+)*$")

_DESELECT = re.compile(r"--deselect=([^ \t\\]+)")

_IGNORE = re.compile(r"--ignore=([^ \t\\]+)")

_K_FILTER = re.compile(r'-k\s+"([^"]+)"')

_M_FILTER = re.compile(r'-m\s+"([^"]+)"')

# The exact diff-cover pin the hosted changed-lines-coverage job installs
# (``pip install coverage "diff-cover==8.0.3"``); Gate 3 installs the same pin.
_DIFF_COVER_PIN = re.compile(r"diff-cover==([0-9][0-9A-Za-z.]*)")


def _load_workflow() -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to parse the CI workflows")
    return yaml.safe_load(CI_WORKFLOW.read_text())


def _split_pytest_blocks(run_text: str) -> list[list[str]]:
    """Split a ci.yml step's ``run`` text into ONE item per ``pytest`` process.

    A new block starts at a line whose stripped form is ``pytest`` (optionally
    followed by a trailing backslash). Following non-pytest lines (paths, flags,
    comments) are collected into the block until the next ``pytest`` line. This
    mirrors the fact that ci.yml runs each ``pytest <args>`` block in its own
    separate process (the engine-lifecycle tests MUST be isolated: their
    ``tests/headless_mlx`` conftest installs MagicMock ``mlx`` modules into
    ``sys.modules``, which would leak into ordinary discovery if they shared a
    process).
    """
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in run_text.splitlines():
        stripped = line.strip()
        if stripped == "pytest" or stripped.startswith("pytest \\"):
            current = []
            blocks.append(current)
            continue
        if current is not None:
            current.append(line)
    return blocks


def _parse_block(block: list[str]) -> dict[str, Any]:
    """Parse ONE pytest block (a list of lines) into an invocation dict.

    Returns ``{paths, ignore, deselect, marker, m, cov_declaration}`` where
    ``cov_declaration`` captures the ``--cov-append`` / ``--cov-report=xml``
    flags the block declares (the script mirrors them so the per-process
    coverage union equals the hosted combined coverage).
    """
    paths: list[str] = []
    ignore: list[str] = []
    deselect: list[str] = []
    marker: str | None = None
    m_filter: str | None = None
    cov_append = False
    cov_report_xml = False

    for line in block:
        if line.strip().startswith("#"):
            # A comment line mentioning `tests/x.py` must never become a phantom
            # path/deselect/marker — skip it entirely.
            continue

        # cov flags can appear on any line of the block.
        if "--cov-append" in line:
            cov_append = True
        if "--cov-report=xml" in line:
            cov_report_xml = True

        target = line.strip().rstrip("\\").strip()
        if _TEST_TARGET.fullmatch(target):
            paths.append(target)

        ignore_match = _IGNORE.search(line)
        if ignore_match:
            ignore.append(ignore_match.group(1))

        deselect_match = _DESELECT.search(line)
        if deselect_match:
            deselect.append(deselect_match.group(1))

        k_match = _K_FILTER.search(line)
        if k_match:
            marker = k_match.group(1)

        m_match = _M_FILTER.search(line)
        if m_match:
            m_filter = m_match.group(1)

    return {
        "paths": paths,
        "ignore": ignore,
        "deselect": deselect,
        "marker": marker,
        "m": m_filter,
        "cov_declaration": {
            "cov_append": cov_append,
            "cov_report_xml": cov_report_xml,
        },
    }


def parse_linux_pytest_args() -> list[dict[str, Any]]:
    """Parse the Linux no-MLX pytest blocks from ci.yml.

    Returns a LIST of invocation dicts, one per distinct ``pytest`` process in
    the "Run unit tests (no MLX required)" step (currently 2), each
    ``{paths, ignore, deselect, marker, m, cov_declaration}`` in ci.yml's run
    order.
    """
    workflow = _load_workflow()
    job = workflow["jobs"]["test-matrix"]
    step = _find_step_by_name(job, "Run unit tests (no MLX required)")
    run_text = step["run"]

    invocations: list[dict[str, Any]] = []
    for block in _split_pytest_blocks(run_text):
        invocation = _parse_block(block)
        if not invocation["paths"]:
            raise ValueError(
                "found a pytest block in ci.yml with no test paths; the "
                "test-matrix 'Run unit tests' step layout may have changed"
            )
        invocations.append(invocation)

    if not invocations:
        raise ValueError(
            "could not find any Linux no-MLX pytest block in ci.yml; "
            "the test-matrix 'Run unit tests' step layout may have changed"
        )
    return invocations


def parse_apple_pytest_args() -> dict[str, Any]:
    """Parse the Apple-MLX pytest block from ci.yml.

    Returns ``{paths, m, k}``: the roster plus the ``-m "..."`` marker
    expression and ``-k "..."`` filter verbatim from the workflow (so if ci.yml
    changes the Apple filter, the local Gate 4 follows instead of hardcoding
    them).
    """
    workflow = _load_workflow()
    job = workflow["jobs"]["test-apple-silicon"]
    step = _find_step_by_name(job, "Run MLX-dependent tests")
    run_text = step["run"]

    paths: list[str] = []
    m_filter: str | None = None
    k_filter: str | None = None
    for line in run_text.splitlines():
        if line.strip().startswith("#"):
            continue
        target = line.strip().rstrip("\\").strip()
        if _TEST_TARGET.fullmatch(target) and target.endswith(".py"):
            paths.append(target)

        m_match = _M_FILTER.search(line)
        if m_match:
            m_filter = m_match.group(1)

        k_match = _K_FILTER.search(line)
        if k_match:
            k_filter = k_match.group(1)

    if not paths:
        raise ValueError(
            "could not find the Apple-MLX pytest roster in ci.yml; "
            "the test-apple-silicon 'Run MLX-dependent tests' step may have "
            "changed"
        )
    return {"paths": paths, "m": m_filter, "k": k_filter}


def parse_mypy_invocation() -> dict[str, Any]:
    """Parse the mypy budget invocation from ci.yml (type-check job)."""
    workflow = _load_workflow()
    job = workflow["jobs"]["type-check"]
    step = _find_step_by_name(job, "Enforce shrink-only mypy error budget")
    script = step["run"].strip()
    if script != f"python {MYPY_BUDGET_SCRIPT}":
        raise ValueError(
            f"mypy budget invocation drifted; expected 'python "
            f"{MYPY_BUDGET_SCRIPT}', found {script!r}"
        )
    return {"script": MYPY_BUDGET_SCRIPT}


def parse_diff_cover_invocation() -> dict[str, Any]:
    """Parse the diff-cover invocation from ci.yml (changed-lines job).

    Returns ``{linux, apple, fail_under, diff_cover_pin}`` — the two coverage
    inputs, the threshold, and the exact ``diff-cover==X.Y.Z`` pin the job's
    "Install coverage tools" step installs (Gate 3 installs the same pin into
    its own fresh venv, so a hosted pin bump moves the local gate with it).
    """
    workflow = _load_workflow()
    job = workflow["jobs"]["changed-lines-coverage"]
    step = _find_step_by_name(job, "Combine coverage and enforce changed lines")
    run_text = step["run"]

    linux = "coverage-data/linux/coverage-linux-3.11.data"
    apple = "coverage-data/apple/coverage-apple.data"
    if linux not in run_text or apple not in run_text:
        raise _drift(
            f"diff-cover combine inputs drifted; expected {linux!r} and {apple!r}"
        )
    if "--compare-branch" not in run_text:
        raise _drift("diff-cover --compare-branch missing")
    if "--fail-under 100" not in run_text:
        raise _drift("diff-cover --fail-under 100 missing")

    install_step = _find_step_by_name(job, "Install coverage tools")
    install_text = install_step["run"]
    pin_match = _DIFF_COVER_PIN.search(install_text)
    if pin_match is None or "coverage" not in install_text:
        raise _drift(
            "changed-lines-coverage 'Install coverage tools' drifted; expected "
            f"'pip install coverage \"diff-cover==X.Y.Z\"', found {install_text!r}"
        )
    return {
        "linux": "coverage-linux-3.11.data",
        "apple": "coverage-apple.data",
        "fail_under": 100,
        "diff_cover_pin": pin_match.group(1),
    }


def parse_swift_test_invocation() -> dict[str, Any]:
    """Parse the Desktop ``swift test`` invocation from rapid-mac-ci.yml.

    Since #2488 the hosted Desktop gate wraps ``swift test --no-parallel`` in
    ``scripts/desktop-test-timeout.sh`` (per-run hang deadline + sample
    artifact). The authoritative invocation is now that wrapper; the legacy
    bare ``swift test --no-parallel`` remains accepted so an unpinned checkout
    (or a pre-#2488 branch) doesn't drift-fail. ``train_gates.sh`` Gate 5 runs
    the real ``swift test --no-parallel`` itself; this parser's value only
    feeds the deterministic gates-hash.
    """
    desktop_wrapper = "./scripts/desktop-test-timeout.sh"
    parsed = yaml.safe_load(MAC_CI_WORKFLOW.read_text())
    for job in parsed["jobs"].values():
        for step in job.get("steps", []):
            if isinstance(step, dict) and step.get("name") == "swift test":
                run = step["run"].strip()
                if run not in (desktop_wrapper, "swift test --no-parallel"):
                    raise _drift(
                        f"Desktop swift test drifted: expected "
                        f"{desktop_wrapper!r} or 'swift test --no-parallel', "
                        f"found {run!r}"
                    )
                return {"cmd": run}
    raise _drift("could not find the Desktop 'swift test' step")


def _find_step_by_name(job: dict[str, Any], name: str) -> dict[str, Any]:
    for step in job["steps"]:
        if isinstance(step, dict) and step.get("name") == name:
            return step
    raise _drift(f"step {name!r} not found in job")


def _drift(message: str) -> RuntimeError:
    return RuntimeError(f"[train-gates drift] {message}")


def main() -> int:  # pragma: no cover - exercised via train_gates.sh
    """CLI entry: ``python -m scripts.train_gates_parser PARSE_NAME``."""
    if len(sys.argv) != 2:
        print(
            "usage: python -m scripts.train_gates_parser "
            "<linux|apple|mypy|diff_cover|swift_test|all>",
            file=sys.stderr,
        )
        return 2
    what = sys.argv[1]
    try:
        if what == "linux":
            result = parse_linux_pytest_args()
        elif what == "apple":
            result = parse_apple_pytest_args()
        elif what == "mypy":
            result = parse_mypy_invocation()
        elif what == "diff_cover":
            result = parse_diff_cover_invocation()
        elif what == "swift_test":
            result = parse_swift_test_invocation()
        elif what == "all":
            result = {
                "linux": parse_linux_pytest_args(),
                "apple": parse_apple_pytest_args(),
                "mypy": parse_mypy_invocation(),
                "diff_cover": parse_diff_cover_invocation(),
                "swift_test": parse_swift_test_invocation(),
            }
        else:
            print(f"unknown parse target {what!r}", file=sys.stderr)
            return 2
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - surface drift as non-zero
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
