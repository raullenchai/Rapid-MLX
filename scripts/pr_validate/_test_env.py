# SPDX-License-Identifier: Apache-2.0
"""Test-environment self-check + canonical test-deps installer.

Closes #185 — root cause of the recurring "pr_validate skipped due to
env issue / missing pytest-asyncio" reports across multiple fix waves.

The problem: pr_validate's ``targeted_tests`` and ``full_unit`` steps
invoke pytest via ``sys.executable -m pytest``. If the host Python lost
``pytest-asyncio`` (e.g. an orchestrated agent ran
``pip install --no-deps --force-reinstall .``, which is the documented
pattern in the project's CLAUDE memory), every ``async def test_*``
fails at collection with::

    async def functions are not natively supported.
    You need to install a suitable plugin for your async framework,
    for example:
      - pytest-asyncio

Across 124 tests in PR #731's full-unit log alone — and the next agent
just reports "skipped due to env issue" and moves on, so the tooling
debt compounds. The systematic fix is two-part:

1. Publish the test-runtime deps as a `test` extras in pyproject.toml
   (the canonical source — pr_validate does NOT maintain a duplicate
   list).
2. Have pr_validate self-check that those plugins are importable in
   the same Python that will run pytest; if not, attempt a one-shot
   ``pip install .[test]`` from the repo root (opt out via
   ``PR_VALIDATE_NO_AUTO_INSTALL=1`` for sandboxed CI).

The self-check runs as a first-class step (TestEnvCheckStep) so when
auto-install is disabled and the env is broken, the operator sees a
clear "pr_validate venv is misconfigured" error in the scorecard
rather than a cryptic 124-failure pytest log.

Why import-time checks and not just shelling to pytest with a "did the
asyncio plugin load" flag? Because pytest itself is configured to
auto-load every installed plugin via setuptools entrypoints, so a
clean ``import pytest_asyncio`` is the most direct evidence the
plugin is wired into THIS interpreter — same path pytest itself takes.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Packages the test suite REQUIRES at collection time. Keep this list
# narrow — anything that's only used by a single test should be
# soft-imported by that test (and tagged ``pytest.importorskip``),
# not added here. The pytest plugin set is the load-bearing one
# because plugin discovery happens before tests are collected, so a
# missing plugin breaks the entire run.
#
# Each entry is (import_name, pip_name, why). ``import_name`` is what
# the self-check tries; ``pip_name`` is what would be installed if we
# fell back to ad-hoc ``pip install`` (we don't — we install the full
# canonical extras instead — but it's surfaced in the error message
# so the operator can manually recover).
REQUIRED_TEST_PACKAGES: tuple[tuple[str, str, str], ...] = (
    (
        "pytest",
        "pytest>=7.0.0",
        "test runner itself; should be present but check anyway",
    ),
    (
        "pytest_asyncio",
        "pytest-asyncio>=0.21.0",
        "pytest.ini sets asyncio_mode=auto; without this every "
        "`async def test_*` fails at collection",
    ),
)

# Canonical extras name from pyproject.toml. If you rename the extras,
# update this constant too (and the unit test that pins it).
TEST_EXTRAS_NAME = "test"

# Files whose modification by an external PR makes the auto-install
# path UNSAFE — installing from the PR's working tree would let the
# attacker's build hook / fake package source run inside the
# validator venv. Detection is conservative: if ANY of these paths
# show up in ``ctx.files_changed`` we refuse to auto-install and
# require the operator to either install manually (after reading the
# diff) or re-run with the dep-file change rolled back. See
# scripts/pr_validate/README.md "Threat model".
#
# Two parts: exact-path matches and a prefix-glob list. The prefix
# list catches every ``requirements*.txt`` variant a contributor
# might invent (``requirements-test.txt``, ``requirements-prod.txt``,
# etc.) without us having to enumerate them — codex r2 BLOCKING was
# that ``requirements-test.txt`` slipped through.
DEP_DECLARATION_FILES_DENYLIST: tuple[str, ...] = (
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
)

# Filename prefixes whose ``.txt`` (or no-extension) variants at the
# repo root all count as dep-declaration files. Kept here so the
# supply-chain step can import the same source of truth via
# ``is_dep_declaration_file()`` — see ``steps/supply_chain.py``
# ``HOOK_PATHS`` for the install-hook matcher that also uses this.
DEP_DECLARATION_FILE_PREFIXES: tuple[str, ...] = (
    "requirements",  # requirements.txt, requirements-dev.txt, requirements-test.txt, …
)


def is_dep_declaration_file(path: str) -> bool:
    """Return True iff ``path`` (a repo-relative file name) is a
    dep-declaration file that an external PR must NOT be allowed to
    influence the validator's install from.

    Exact match against ``DEP_DECLARATION_FILES_DENYLIST`` OR
    starts-with match against ``DEP_DECLARATION_FILE_PREFIXES`` for
    repo-root ``.txt`` files. Subdirectory files (e.g.
    ``vendor/requirements.txt``) are intentionally NOT matched —
    they don't drive pr_validate's ``pip install '.[test]'``.

    Public so the supply-chain step can share the matcher.
    """
    if path in DEP_DECLARATION_FILES_DENYLIST:
        return True
    # Only match the repo root — subdirectory files don't drive the
    # validator's recovery install. Strip path separators to test.
    if "/" in path:
        return False
    for prefix in DEP_DECLARATION_FILE_PREFIXES:
        if path.startswith(prefix) and (path.endswith(".txt") or path == prefix):
            return True
    return False


# Hardcoded, version-pinned set of pytest plugins pr_validate needs
# IN ITS OWN venv to run ``targeted_tests`` / ``full_unit`` reliably.
# Installed from PyPI directly (not from the PR's working tree) so a
# malicious PR that ships a typo-squat or replaces ``pytest-asyncio``
# in pyproject.toml CANNOT subvert the validator's runtime. Keep this
# list tiny and pinned to a narrow range: the goal is "validator
# always boots", not "validator can run every test in every PR".
#
# Versions chosen to track the project's ``[test]`` extras at the
# time of pinning (#275). A bump here is a deliberate operator
# decision; pr_validate refuses to silently follow a PR's lead.
TRUSTED_TEST_PINS: tuple[str, ...] = (
    "pytest>=7.0.0,<9",
    "pytest-asyncio>=0.21.0,<1",
)


@dataclass(frozen=True)
class TestEnvStatus:
    """Result of a test-env check.

    ``missing`` is the list of import names that failed; ``ok`` mirrors
    the bool the caller usually wants. ``message`` is a one-liner
    suitable for a step-result summary.
    """

    ok: bool
    missing: tuple[str, ...]
    message: str
    interpreter: str

    @property
    def install_hint(self) -> str:
        """Manual recovery command the operator can paste, in case
        auto-install is disabled or also fails."""
        # Use the exact interpreter that ran the check — `pip install`
        # with a different python silently installs into the wrong
        # site-packages and the next pr_validate run fails the same
        # way. The operator should see the FULL command.
        return (
            f"{self.interpreter} -m pip install '.[{TEST_EXTRAS_NAME}]'"
            " (from the repo root)"
        )


def check_test_env(python: str | None = None) -> TestEnvStatus:
    """Probe ``python`` for the required test-runtime packages.

    ``python`` defaults to ``sys.executable`` — i.e. the interpreter
    currently running pr_validate, which is also the one
    ``targeted_tests`` and ``full_unit`` will hand pytest to. Using a
    different interpreter for the check than for the actual run would
    defeat the point of the check.

    The probe is a single ``python -c "import pytest, pytest_asyncio"``
    so the import side-effects happen in a fresh process — keeps this
    function safe to call from inside pytest itself (where importing
    pytest_asyncio twice could trip a "plugin already registered" warning).
    """
    interp = python or sys.executable
    import_names = [pkg for pkg, _, _ in REQUIRED_TEST_PACKAGES]
    probe = "; ".join(f"import {name}" for name in import_names)

    proc = subprocess.run(  # noqa: S603
        [interp, "-c", probe],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return TestEnvStatus(
            ok=True,
            missing=(),
            message=f"all {len(import_names)} required test packages importable",
            interpreter=interp,
        )

    # Identify exactly which import failed. Re-probe each one
    # individually — cheap (a handful of process spawns) and gives the
    # operator the precise list instead of just "something broke".
    missing: list[str] = []
    for name in import_names:
        single = subprocess.run(  # noqa: S603
            [interp, "-c", f"import {name}"],
            capture_output=True,
            text=True,
        )
        if single.returncode != 0:
            missing.append(name)

    if not missing:
        # The batch probe failed but every individual import passed.
        # This is a real condition pytest will hit at startup — plugin
        # registration order / "plugin already registered" / a sys.path
        # mutation by one import that breaks the next. Codex r1
        # BLOCKING: returning ok=True here let a broken env masquerade
        # as healthy, exactly the failure mode #185 is about.
        # Surface the batch stderr so the operator can diagnose
        # without re-running by hand.
        batch_err = (proc.stderr or proc.stdout or "").strip() or (
            "(no diagnostic output from the failing import batch — "
            f"exit code: {proc.returncode})"
        )
        return TestEnvStatus(
            ok=False,
            missing=tuple(import_names),
            message=(
                "batch import probe failed (every individual import "
                "passed, but the combined load order pytest takes is "
                f"broken). Diagnostic: {batch_err[:512]}"
            ),
            interpreter=interp,
        )

    return TestEnvStatus(
        ok=False,
        missing=tuple(missing),
        message=f"missing required test packages: {', '.join(missing)}",
        interpreter=interp,
    )


def auto_install_disabled() -> bool:
    """Honor ``PR_VALIDATE_NO_AUTO_INSTALL=1`` — set in CI sandboxes
    where the validator must NOT mutate the host Python environment
    (e.g. GitHub Actions on a hardened runner with a read-only venv).

    In that mode the self-check still runs and still emits a clear
    error; the operator is just expected to install the extras
    themselves before re-invoking pr_validate.
    """
    return os.environ.get("PR_VALIDATE_NO_AUTO_INSTALL", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def pr_touches_dep_files(files_changed: list[str]) -> list[str]:
    """Return the subset of ``files_changed`` that ``is_dep_declaration_file``
    flags.

    Returning the (possibly empty) list rather than a bool lets the
    caller surface the exact filenames in the warning the operator
    sees — "skipped because the PR touches pyproject.toml" is much
    more actionable than just "skipped". An empty list means the
    auto-install path is safe to take.

    Matching delegates to ``is_dep_declaration_file`` so the supply-
    chain step and this guard share a single source of truth. Catches
    every repo-root ``requirements*.txt`` variant — codex r2
    BLOCKING was an under-enumeration here.
    """
    return [f for f in files_changed if is_dep_declaration_file(f)]


def install_trusted_pins(python: str | None = None) -> tuple[bool, str]:
    """Install ``TRUSTED_TEST_PINS`` from PyPI into ``python``.

    Bypasses the PR's pyproject.toml entirely — the pin list is
    hardcoded above and version-bounded so a malicious PR cannot
    influence what gets installed into the validator venv. Used as
    the recovery path when ``pr_touches_dep_files`` reports the PR
    has modified dep-declaration files (in which case
    ``install_test_extras`` is unsafe).

    Returns ``(ok, log)`` mirroring ``install_test_extras``. ``--no-deps``
    is intentionally NOT passed — pytest-asyncio needs its own
    transitive deps and those come from PyPI too, not the PR.

    ``--isolated`` blocks the user's pip.conf from injecting a
    malicious index URL via ``--extra-index-url``; combined with the
    pinned versions this gives the validator a stable install path.
    """
    interp = python or sys.executable
    cmd = [
        interp,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--isolated",
        "--disable-pip-version-check",
        *TRUSTED_TEST_PINS,
    ]
    proc = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
    )
    log = (proc.stdout or "") + (proc.stderr or "")
    if len(log) > 2048:
        log = log[:1024] + "\n…[truncated]…\n" + log[-1024:]
    return proc.returncode == 0, log


def install_test_extras(repo_root: Path, python: str | None = None) -> tuple[bool, str]:
    """Install the project's ``[test]`` extras into ``python`` from
    ``repo_root``. Returns ``(ok, log)`` where ``log`` is the combined
    stdout+stderr of pip (truncated to ~2 KB for scorecard inclusion).

    Uses ``pip install '.[test]' --no-deps`` style? No — we want pip to
    resolve `pytest`, `pytest-asyncio`, etc. The project's own runtime
    deps (mlx, transformers, …) are already present from the user's
    initial install; pip's resolver will treat them as satisfied and
    short-circuit. ``--no-deps`` would prevent pytest-asyncio's own
    deps from being picked up, which is wrong.

    We do pass ``--quiet`` to keep the scorecard log readable. Full
    output is also written to an artifact file via the calling step.
    """
    interp = python or sys.executable
    cmd = [interp, "-m", "pip", "install", "--quiet", f".[{TEST_EXTRAS_NAME}]"]
    proc = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )
    log = (proc.stdout or "") + (proc.stderr or "")
    # Cap the log to ~2 KB — pip's "Successfully installed …" line is
    # the load-bearing bit; pages of dep-resolver output just pad the
    # scorecard. Full unabridged version still gets written to the
    # artifact file by the step that called us.
    if len(log) > 2048:
        log = log[:1024] + "\n…[truncated]…\n" + log[-1024:]
    return proc.returncode == 0, log
