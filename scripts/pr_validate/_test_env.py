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

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

# Packages the test suite REQUIRES at collection time. Keep this list
# narrow — anything that's only used by a single test should be
# soft-imported by that test (and tagged ``pytest.importorskip``),
# not added here. The pytest plugin set is the load-bearing one
# because plugin discovery happens before tests are collected, so a
# missing plugin breaks the entire run.
#
# Each entry is (import_name, distribution_hint, why). ``import_name`` is
# what the self-check tries. ``distribution_hint`` identifies the matching
# entry in the canonical ``[project.optional-dependencies].test`` table;
# its specifier and marker are loaded from pyproject.toml at runtime so this
# probe cannot silently drift from the dependency declaration.
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
    (
        "aiohttp",
        "aiohttp>=3.9.0",
        "async HTTP tests import aiohttp at module load",
    ),
    (
        "PIL",
        "pillow>=10.0.0",
        "image/aspect-ratio and Gemma MTP tests import PIL at collection/run time",
    ),
    (
        "mlx_vlm",
        "mlx-vlm>=0.6.3; platform_system == 'Darwin'",
        "Gemma 4 / DFlash / vision lock-in tests expect mlx_vlm importability",
    ),
    (
        "mlx_audio",
        "mlx-audio>=0.5.3,<0.6; platform_system == 'Darwin'",
        "audio route tests expect mlx-audio importability",
    ),
)

# MLX runtime surfaces are Apple-only. Local Apple-Silicon validation
# should require them, but Linux CI must not treat these imports as a
# readiness signal.
DARWIN_ONLY_TEST_IMPORTS = frozenset({"mlx_vlm", "mlx_audio"})

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
    "aiohttp>=3.9.0,<4",
    "pillow>=10.0.0,<13",
    "mlx-vlm==0.6.17; platform_system == 'Darwin'",
    "mlx-audio>=0.5.3,<0.6; platform_system == 'Darwin'",
)


_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _normalized_requirement_name(requirement: str) -> str:
    """Extract and normalize the distribution name from a PEP 508 string."""
    match = _REQUIREMENT_NAME.match(requirement)
    if match is None:
        raise ValueError(
            f"invalid requirement without a distribution name: {requirement}"
        )
    return re.sub(r"[-_.]+", "-", match.group(1)).lower()


def canonical_test_packages(
    repo_root: Path | None = None,
) -> tuple[tuple[str, str, str], ...]:
    """Join import probes to canonical ``test`` requirements.

    Only the import/distribution mapping lives in this module. Version
    specifiers and environment markers come from pyproject.toml, while the
    target interpreter still performs the actual PEP 508 parsing and marker
    evaluation inside ``_REQUIREMENT_PROBE``.
    """
    root = repo_root or Path(__file__).resolve().parents[2]
    data = tomllib.loads((root / "pyproject.toml").read_text())
    declared = data["project"]["optional-dependencies"][TEST_EXTRAS_NAME]
    requirements_by_name = {
        _normalized_requirement_name(requirement): requirement
        for requirement in declared
    }

    packages: list[tuple[str, str, str]] = []
    for import_name, distribution_hint, why in REQUIRED_TEST_PACKAGES:
        name = _normalized_requirement_name(distribution_hint)
        try:
            canonical_requirement = requirements_by_name[name]
        except KeyError as exc:
            raise ValueError(
                f"required import {import_name!r} has no canonical "
                f"[{TEST_EXTRAS_NAME}] requirement for {name!r}"
            ) from exc
        packages.append((import_name, canonical_requirement, why))
    return tuple(packages)


def required_test_packages_for_platform(
    platform: str | None = None,
) -> tuple[tuple[str, str, str], ...]:
    """Return the test-env import probes required on ``platform``."""
    platform_name = platform or sys.platform
    if platform_name == "darwin":
        return REQUIRED_TEST_PACKAGES
    return tuple(
        pkg for pkg in REQUIRED_TEST_PACKAGES if pkg[0] not in DARWIN_ONLY_TEST_IMPORTS
    )


@dataclass(frozen=True)
class TestEnvStatus:
    """Result of a test-env check.

    ``missing`` is the list of import names whose requirement is not
    satisfied (missing import/distribution metadata or an incompatible
    installed version); ``ok`` mirrors the bool the caller usually wants.
    ``message`` is a one-liner suitable for a step-result summary.
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


_REQUIREMENT_PROBE = r"""
import contextlib
import importlib
import io
import json
import sys
from importlib import metadata

from packaging.requirements import Requirement
from packaging.version import InvalidVersion

packages = json.loads(sys.stdin.read())
results = []
for import_name, requirement_text, _why in packages:
    requirement = Requirement(requirement_text)
    if requirement.marker is not None and not requirement.marker.evaluate():
        results.append(
            {
                "import_name": import_name,
                "distribution": requirement.name,
                "installed": None,
                "required": str(requirement.specifier) or "any version",
                "state": "skipped",
                "import_error": None,
            }
        )
        continue

    try:
        installed = metadata.version(requirement.name)
    except metadata.PackageNotFoundError:
        installed = None

    import_error = None
    try:
        # Some optional runtimes print notices while importing. Keep stdout
        # reserved for the machine-readable result consumed by the parent.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            importlib.import_module(import_name)
    except Exception as exc:
        import_error = f"{type(exc).__name__}: {exc}"

    if import_error is not None:
        state = "missing_import"
    elif installed is None:
        state = "missing_distribution"
    else:
        try:
            compatible = not requirement.specifier or requirement.specifier.contains(
                installed,
                # Pip ignores prereleases unless the requirement itself
                # opts into one. Mirror that behavior for an already
                # installed distribution instead of silently widening a
                # stable-only canonical range.
                prereleases=requirement.specifier.prereleases is True,
            )
        except InvalidVersion:
            compatible = False
        state = "ok" if compatible else "version_mismatch"

    results.append(
        {
            "import_name": import_name,
            "distribution": requirement.name,
            "installed": installed,
            "required": str(requirement.specifier) or "any version",
            "state": state,
            "import_error": import_error,
        }
    )

print(json.dumps(results))
"""


def _problem_message(problem: dict[str, str | None]) -> str:
    """Format one failed requirement probe for operator-facing output."""
    distribution = problem["distribution"]
    import_name = problem["import_name"]
    installed = problem["installed"] or "not installed"
    required = problem["required"]

    if problem["state"] == "version_mismatch":
        return f"{distribution} {installed} does not satisfy {required}"
    if problem["state"] == "missing_distribution":
        return (
            f"{distribution} has no installed distribution metadata "
            f"(import {import_name} succeeded; requires {required})"
        )
    return (
        f"{distribution} is not importable as {import_name} "
        f"(installed: {installed}; requires {required})"
    )


def check_test_env(
    python: str | None = None, repo_root: Path | None = None
) -> TestEnvStatus:
    """Probe ``python`` for required imports and compatible distributions.

    ``python`` defaults to ``sys.executable`` — i.e. the interpreter
    currently running pr_validate, which is also the one
    ``targeted_tests`` and ``full_unit`` will hand pytest to. Using a
    different interpreter for the check than for the actual run would
    defeat the point of the check.

    The probe runs in one fresh child process so imports, PEP 508 marker
    evaluation, distribution metadata lookup, and version comparison all use
    the exact interpreter that will run pytest. This also keeps the function
    safe to call from inside pytest itself (where importing pytest plugins a
    second time could trip a "plugin already registered" warning).
    """
    interp = python or sys.executable
    try:
        packages = canonical_test_packages(repo_root)
    except (KeyError, OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        return TestEnvStatus(
            ok=False,
            missing=tuple(pkg for pkg, _, _ in REQUIRED_TEST_PACKAGES),
            message=f"could not load canonical test requirements: {exc}",
            interpreter=interp,
        )

    proc = subprocess.run(  # noqa: S603
        [interp, "-c", _REQUIREMENT_PROBE],
        input=json.dumps(packages),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        batch_err = (proc.stderr or proc.stdout or "").strip() or (
            "(no diagnostic output from the failing requirement probe — "
            f"exit code: {proc.returncode})"
        )
        return TestEnvStatus(
            ok=False,
            missing=tuple(pkg for pkg, _, _ in packages),
            message=(
                "test requirement probe failed before it could inspect the "
                f"environment. Diagnostic: {batch_err[:512]}"
            ),
            interpreter=interp,
        )

    try:
        results = json.loads(proc.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        return TestEnvStatus(
            ok=False,
            missing=tuple(pkg for pkg, _, _ in packages),
            message=f"test requirement probe returned invalid output: {exc}",
            interpreter=interp,
        )

    expected_imports = [pkg for pkg, _, _ in packages]
    valid_states = {
        "ok",
        "skipped",
        "missing_import",
        "missing_distribution",
        "version_mismatch",
    }
    valid_result = (
        isinstance(results, list)
        and len(results) == len(packages)
        and all(
            isinstance(result, dict)
            and result.get("import_name") == expected_imports[index]
            and isinstance(result.get("distribution"), str)
            and (
                result.get("installed") is None
                or isinstance(result.get("installed"), str)
            )
            and isinstance(result.get("required"), str)
            and result.get("state") in valid_states
            and (
                result.get("import_error") is None
                or isinstance(result.get("import_error"), str)
            )
            for index, result in enumerate(results)
        )
    )
    if not valid_result:
        return TestEnvStatus(
            ok=False,
            missing=tuple(expected_imports),
            message="test requirement probe returned an invalid result schema",
            interpreter=interp,
        )

    applicable = [result for result in results if result["state"] != "skipped"]
    problems = [result for result in applicable if result["state"] != "ok"]
    if not problems:
        return TestEnvStatus(
            ok=True,
            missing=(),
            message=(
                f"all {len(applicable)} applicable required test packages "
                "importable and version-compatible"
            ),
            interpreter=interp,
        )

    return TestEnvStatus(
        ok=False,
        missing=tuple(str(problem["import_name"]) for problem in problems),
        message="unsatisfied test requirements: "
        + "; ".join(_problem_message(problem) for problem in problems),
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
