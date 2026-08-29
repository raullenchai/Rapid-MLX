"""Drift test: keep `scripts/train_gates.sh` (and its parser helper) honest
against the hosted CI workflows.

`scripts/train_gates.sh <base-sha>` reproduces the 5 validation gates the CI
matrix runs. It does NOT hardcode those gates — it parses them at runtime from
`.github/workflows/ci.yml` and `.github/workflows/rapid-mac-ci.yml` via
`scripts/train_gates_parser.py`. This test guards that reproduction from
drifting away from the workflows: if a CI-definition edit changes the Linux
pytest discovery surface, the Apple pytest roster, the mypy invocation, the
diff-cover invocation, or the Desktop swift invocation, one of the assertions
below must fail — exactly the machine-readable tripwire that keeps the local
train-gates reproduction honest.

Pure-pytest, Linux-friendly, no MLX import (the parser is stdlib + PyYAML).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.train_gates_parser import (
    CI_WORKFLOW,
    MAC_CI_WORKFLOW,
    MYPY_BUDGET_SCRIPT,
    parse_apple_pytest_args,
    parse_diff_cover_invocation,
    parse_linux_pytest_args,
    parse_mypy_invocation,
    parse_swift_test_invocation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
yaml = pytest.importorskip("yaml")

CI = CI_WORKFLOW.read_text()
MAC_CI = MAC_CI_WORKFLOW.read_text()
CI_PARSED = yaml.safe_load(CI)
MAC_CI_PARSED = yaml.safe_load(MAC_CI)


def _split_blocks(run_text: str) -> list[list[str]]:
    """Independent split of a step's run text into per-pytest-process blocks.

    This re-derives the split WITHOUT using the shared parser (so the parser
    cannot mask a drift against a bug of its own). A block starts at a line
    whose stripped form is ``pytest``.
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


def _extract_path_tokens(block: list[str]) -> list[str]:
    """Extract literal tests directory/file targets independently."""
    tokens: list[str] = []
    for line in block:
        if line.strip().startswith("#"):
            continue
        if line.strip().startswith("--"):
            continue
        stripped = line.strip().rstrip("\\").strip()
        if stripped == "tests" or stripped.startswith("tests/"):
            tok = stripped.split()[0]
            tokens.append(tok)
    return tokens


def _run_step(job_name: str, step_name: str, ci: dict) -> dict:
    job = ci["jobs"][job_name]
    matches = [s for s in job["steps"] if s.get("name") == step_name]
    assert matches, f"step {step_name!r} not found in job {job_name}"
    return matches[0]


def _run_text(job_name: str, step_name: str, ci: dict) -> str:
    return _run_step(job_name, step_name, ci)["run"]


# ---------------------------------------------------------------------------
# Linux no-MLX automatic discovery (TWO separate pytest processes)
# ---------------------------------------------------------------------------
def test_linux_parser_returns_two_invocations() -> None:
    # ci.yml's "Run unit tests" step runs TWO separate pytest processes: the
    # ordinary automatically discovered unit suite, and a SECOND process for
    # tests/headless_mlx (its conftest installs fake modules in sys.modules).
    parsed = parse_linux_pytest_args()
    assert isinstance(parsed, list)
    run_text = _run_text("test-matrix", "Run unit tests (no MLX required)", CI_PARSED)
    blocks = _split_blocks(run_text)
    assert len(parsed) == len(blocks) == 2, (
        "the local gate must reproduce ci.yml's per-process pytest split; "
        f"parser produced {len(parsed)} invocation(s), ci.yml has {len(blocks)}"
    )


def test_linux_parser_matches_workflow_targets() -> None:
    parsed = parse_linux_pytest_args()
    run_text = _run_text("test-matrix", "Run unit tests (no MLX required)", CI_PARSED)
    blocks = _split_blocks(run_text)
    for invocation, block in zip(parsed, blocks):
        workflow_paths = _extract_path_tokens(block)
        assert workflow_paths, "workflow carries no Linux test paths"
        assert invocation["paths"] == workflow_paths, (
            "Linux pytest targets drifted: the parser extracts\n"
            f"{invocation['paths']}\nbut ci.yml carries\n{workflow_paths}\n"
            "Update the parser (or the workflow) so they agree."
        )


def test_linux_parser_captures_discovery_boundaries_and_filters() -> None:
    parsed = parse_linux_pytest_args()
    first, second = parsed
    assert first["paths"] == ["tests"]
    assert first["ignore"] == ["tests/integrations", "tests/headless_mlx"]
    assert first["m"] == (
        "not requires_mlx and not real_hf_cache and not requires_network "
        "and not slow and not integration and not needle"
    )
    assert first["marker"] is None
    assert first["deselect"] == []
    assert first["cov_declaration"]["cov_append"] is False

    second = parsed[1]
    assert second["paths"] == ["tests/headless_mlx"]
    assert second["ignore"] == []
    assert second["deselect"] == []
    assert second["marker"] is None
    assert second["m"] is None
    assert second["cov_declaration"]["cov_append"] is True


def test_linux_discovery_targets_exist() -> None:
    parsed = parse_linux_pytest_args()
    for invocation in parsed:
        for path in invocation["paths"]:
            target = path.split("::", 1)[0]
            assert (CI_WORKFLOW.parents[2] / target).exists(), f"{target} missing"


# ---------------------------------------------------------------------------
# Apple-MLX roster
# ---------------------------------------------------------------------------
def test_apple_parser_matches_workflow_roster() -> None:
    parsed = parse_apple_pytest_args()
    step = _run_step("test-apple-silicon", "Run MLX-dependent tests", CI_PARSED)
    run_text = step["run"]
    workflow_paths = _extract_path_tokens(_split_blocks(run_text)[0])
    assert workflow_paths, "workflow carries no Apple test paths"
    assert parsed["paths"] == workflow_paths, (
        "Apple pytest roster drifted: the parser extracts\n"
        f"{parsed['paths']}\nbut ci.yml carries\n{workflow_paths}\n"
    )
    for path in parsed["paths"]:
        file_part = path.split("::", 1)[0]
        assert (CI_WORKFLOW.parents[2] / file_part).is_file(), f"{file_part} missing"


def test_apple_parser_extracts_m_and_k_filters() -> None:
    # Gate 4 must follow ci.yml's Apple -m / -k flags, not hardcode them.
    parsed = parse_apple_pytest_args()
    run_text = _run_text("test-apple-silicon", "Run MLX-dependent tests", CI_PARSED)
    assert parsed["m"], "Apple -m marker expression is empty"
    assert parsed["k"], "Apple -k filter is empty"
    assert parsed["m"] in run_text
    assert parsed["k"] in run_text
    assert parsed["m"] == "not slow"  # current hosted value, guarded verbatim
    assert parsed["k"] == "not Integration"  # current hosted value


# ---------------------------------------------------------------------------
# mypy budget
# ---------------------------------------------------------------------------
def test_mypy_invocation_matches_workflow() -> None:
    parsed = parse_mypy_invocation()
    run_text = _run_text(
        "type-check", "Enforce shrink-only mypy error budget", CI_PARSED
    )
    assert parsed == {"script": MYPY_BUDGET_SCRIPT}
    assert f"python {MYPY_BUDGET_SCRIPT}" in run_text
    # The pinned budget file must exist (it feeds the gates-hash).
    assert (CI_WORKFLOW.parents[2] / "config/mypy-requirements.txt").is_file()
    assert (CI_WORKFLOW.parents[2] / "config/mypy-error-baseline.txt").is_file()


# ---------------------------------------------------------------------------
# diff-cover (Gate 3)
# ---------------------------------------------------------------------------
def test_diff_cover_invocation_matches_workflow() -> None:
    parsed = parse_diff_cover_invocation()
    run_text = _run_text(
        "changed-lines-coverage",
        "Combine coverage and enforce changed lines",
        CI_PARSED,
    )
    assert "coverage combine" in run_text
    assert "coverage-data/linux/coverage-linux-3.11.data" in run_text
    assert "coverage-data/apple/coverage-apple.data" in run_text
    assert "--fail-under 100" in run_text
    assert parsed["fail_under"] == 100
    assert parsed["linux"] == "coverage-linux-3.11.data"
    assert parsed["apple"] == "coverage-apple.data"


def test_diff_cover_pin_matches_workflow_install_step() -> None:
    # Gate 3 installs `coverage` + the SAME diff-cover pin the hosted job
    # installs, parsed from its "Install coverage tools" step — never
    # hardcoded in the script. Guard the pin verbatim so a hosted bump that
    # the parser fails to pick up trips here.
    parsed = parse_diff_cover_invocation()
    install_text = _run_text(
        "changed-lines-coverage", "Install coverage tools", CI_PARSED
    )
    assert f"diff-cover=={parsed['diff_cover_pin']}" in install_text
    assert "coverage" in install_text
    assert parsed["diff_cover_pin"] == "8.0.3"  # current hosted pin, verbatim


# ---------------------------------------------------------------------------
# train_gates.sh argument handling — these paths exit BEFORE any interpreter
# resolution, workflow parsing or venv creation, so they are cheap and
# Linux-friendly (bash + git only) and never touch the environment.
# ---------------------------------------------------------------------------
TRAIN_GATES_SH = REPO_ROOT / "scripts" / "train_gates.sh"


def _run_train_gates(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(TRAIN_GATES_SH), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_train_gates_help_exits_zero_with_usage_on_stdout() -> None:
    for flag in ("-h", "--help"):
        proc = _run_train_gates(flag)
        assert proc.returncode == 0, (flag, proc.stderr)
        assert proc.stdout.startswith("usage: scripts/train_gates.sh <base-sha>")
        assert "GATES DIRTY" in proc.stdout  # the receipt contract is documented
        assert proc.stderr == ""


def test_train_gates_missing_or_bad_args_exit_2_with_usage() -> None:
    for args in ((), ("--bogus",), ("a", "b")):
        proc = _run_train_gates(*args)
        assert proc.returncode == 2, (args, proc.stdout, proc.stderr)
        assert proc.stderr.startswith("ERROR: "), (args, proc.stderr)
        assert "usage: scripts/train_gates.sh <base-sha>" in proc.stderr
        assert "GATES OK" not in proc.stdout


def test_train_gates_unresolvable_base_is_one_clear_error_not_git_fatal() -> None:
    proc = _run_train_gates("no-such-rev-zz")
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    first_line = proc.stderr.splitlines()[0]
    assert first_line.startswith("ERROR: cannot resolve base 'no-such-rev-zz'")
    assert "fatal:" not in proc.stderr
    assert "usage: scripts/train_gates.sh <base-sha>" in proc.stderr


def test_train_gates_refuses_base_equal_to_head() -> None:
    # base == HEAD would make diff-cover compare an EMPTY diff and pass 100%
    # by construction; the script must refuse before running any gate.
    proc = _run_train_gates("HEAD")
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    assert "IS the current HEAD" in proc.stderr
    assert "merge-base" in proc.stderr
    assert "GATES OK" not in proc.stdout
    assert "== Gate 1" not in proc.stdout


# ---------------------------------------------------------------------------
# Desktop swift test (Gate 5)
# ---------------------------------------------------------------------------
def test_swift_test_invocation_matches_workflow() -> None:
    # Since #2488 the hosted Desktop gate wraps `swift test --no-parallel` in
    # ``scripts/desktop-test-timeout.sh``. The parser must reflect THAT as the
    # authoritative desktop-test invocation (it feeds the gates-hash).
    parsed = parse_swift_test_invocation()
    assert parsed in (
        {"cmd": "swift test --no-parallel"},
        {"cmd": "./scripts/desktop-test-timeout.sh"},
    )
    hosted = MAC_CI
    assert (
        "swift test --no-parallel" in hosted
        or "./scripts/desktop-test-timeout.sh" in hosted
    )


# ---------------------------------------------------------------------------
# The training gates must actually be wired into the shared parser so the
# hash that freeze relies on stays stable under renames.
# ---------------------------------------------------------------------------
def test_train_gates_script_parses_workflows() -> None:
    # Exercise the subprocess entry path too (what train_gates.sh runs), so a
    # break in `python -m scripts.train_gates_parser` surfaces here, in CI,
    # before it surfaces in a local train run.
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "scripts.train_gates_parser", "all"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    import json

    payload = json.loads(proc.stdout)
    assert set(payload) == {"linux", "apple", "mypy", "diff_cover", "swift_test"}
    assert isinstance(payload["linux"], list) and len(payload["linux"]) == 2
    assert payload["linux"][0]["paths"]
    assert payload["linux"][1]["paths"]
    assert payload["apple"]["paths"]


def test_train_gates_script_cli_targets_are_reachable() -> None:
    # `train_gates.sh` invokes the parser for exactly these single targets via
    # `python -m scripts.train_gates_parser <target>`. A new gate must not add
    # a subprocess target the script uses without this tripwire noticing, and
    # an existing target must never become unreachable.
    import json
    import subprocess
    import sys

    expected_targets = ("linux", "apple", "mypy", "diff_cover", "swift_test")
    for target in expected_targets:
        proc = subprocess.run(
            [sys.executable, "-m", "scripts.train_gates_parser", target],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, (target, proc.stderr)
        payload = json.loads(proc.stdout)
        assert payload, (target, "empty payload")
