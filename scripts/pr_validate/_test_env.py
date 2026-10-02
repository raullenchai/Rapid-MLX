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
2. Have pr_validate self-check that those plugins are importable and their
   installed distributions satisfy the canonical version ranges in the same
   Python that will run pytest; if not, attempt a one-shot
   ``pip install .[test]`` from the repo root (opt out via
   ``PR_VALIDATE_NO_AUTO_INSTALL=1`` for sandboxed CI).

The self-check runs as a first-class step (TestEnvCheckStep) so when
auto-install is disabled and the env is broken, the operator sees a
clear "pr_validate venv is misconfigured" error in the scorecard
rather than a cryptic 124-failure pytest log.

Why retain import-time checks in addition to distribution metadata? Because an
in-range distribution can still be corrupted or shadowed. Pytest is configured to
auto-load every installed plugin via setuptools entrypoints, so a
clean ``import pytest_asyncio`` is the most direct evidence the
plugin is wired into THIS interpreter — same path pytest itself takes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from packaging.requirements import Requirement

# Packages the test suite REQUIRES at collection time. Keep this list
# narrow — anything that's only used by a single test should be
# soft-imported by that test (and tagged ``pytest.importorskip``),
# not added here. The pytest plugin set is the load-bearing one
# because plugin discovery happens before tests are collected, so a
# missing plugin breaks the entire run.
#
# Each entry is (import_name, distribution_name, why). Import and
# distribution names are deliberately separate: e.g. ``PIL`` is provided by
# the ``pillow`` distribution. Version ranges and markers are NOT duplicated
# here; they are parsed from the canonical ``[project.optional-dependencies]
# .test`` declarations in pyproject.toml.
REQUIRED_TEST_PACKAGES: tuple[tuple[str, str, str], ...] = (
    (
        "pytest",
        "pytest",
        "test runner itself; should be present but check anyway",
    ),
    (
        "pytest_asyncio",
        "pytest-asyncio",
        "pytest.ini sets asyncio_mode=auto; without this every "
        "`async def test_*` fails at collection",
    ),
    (
        "aiohttp",
        "aiohttp",
        "async HTTP tests import aiohttp at module load",
    ),
    (
        "PIL",
        "pillow",
        "image/aspect-ratio and Gemma MTP tests import PIL at collection/run time",
    ),
    (
        "mlx_vlm",
        "mlx-vlm",
        "Gemma 4 / DFlash / vision lock-in tests expect mlx_vlm importability",
    ),
    (
        "mlx_audio",
        "mlx-audio",
        "audio route tests expect mlx-audio importability",
    ),
)

# MLX runtime surfaces are Apple-only. Local Apple-Silicon validation
# should require them, but Linux CI must not treat these imports as a
# readiness signal. Platform selection deliberately lives here rather than in
# REQUIRED_TEST_PACKAGES' distribution-name field: that field must stay a bare
# name suitable for ``importlib.metadata.version()``. Canonical PEP 508 markers
# are evaluated separately by ``_active_test_packages``.
DARWIN_ONLY_TEST_IMPORTS = frozenset({"mlx_vlm", "mlx_audio"})

# Canonical extras name from pyproject.toml. If you rename the extras,
# update this constant too (and the unit test that pins it).
TEST_EXTRAS_NAME = "test"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_METADATA_TIMEOUT_SECONDS = 15

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
    "packaging>=23,<27",
    'tomli>=2.0.1,<3; python_version < "3.11"',
    "pytest>=7.0.0,<9",
    "pytest-asyncio>=0.21.0,<1",
    "aiohttp>=3.9.0,<4",
    "pillow>=10.0.0,<13",
    "mlx-vlm==0.7.2; platform_system == 'Darwin'",
    "mlx-audio>=0.5.3,<0.6; platform_system == 'Darwin'",
)


@dataclass(frozen=True)
class DependencyProblem:
    """One active test requirement not satisfied by the target interpreter."""

    import_name: str
    distribution_name: str
    installed_version: str | None
    requirement: str

    def render(self) -> str:
        installed = self.installed_version or "not installed"
        return (
            f"{self.distribution_name} installed {installed}, "
            f"requires {self.requirement}"
        )


def canonical_test_requirements(
    repo_root: Path | None = None,
) -> dict[str, tuple[Requirement, ...]]:
    """Parse the canonical PEP 508 requirements for the test extra."""

    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover - exercised by the Python 3.10 CI lane
        import tomli as tomllib

    root = repo_root or PROJECT_ROOT
    data = tomllib.loads((root / "pyproject.toml").read_text())
    raw_requirements = data["project"]["optional-dependencies"][TEST_EXTRAS_NAME]
    requirements = [Requirement(raw) for raw in raw_requirements]
    by_name: dict[str, list[Requirement]] = {}
    for requirement in requirements:
        by_name.setdefault(canonicalize_name(requirement.name), []).append(requirement)
    return {name: tuple(entries) for name, entries in by_name.items()}


_TARGET_METADATA_PROBE = """
import importlib.metadata
import json
import os
import platform
import sys

names = json.loads(sys.argv[1])
versions = {}
for name in names:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None

implementation = sys.implementation.version
implementation_version = f"{implementation.major}.{implementation.minor}.{implementation.micro}"
if implementation.releaselevel != "final":
    implementation_version += implementation.releaselevel[0] + str(implementation.serial)
environment = {
    "implementation_name": sys.implementation.name,
    "implementation_version": implementation_version,
    "os_name": os.name,
    "platform_machine": platform.machine(),
    "platform_release": platform.release(),
    "platform_system": platform.system(),
    "platform_version": platform.version(),
    "platform_python_implementation": platform.python_implementation(),
    "python_full_version": platform.python_version(),
    "python_version": ".".join(platform.python_version_tuple()[:2]),
    "sys_platform": sys.platform,
}
print(json.dumps({"environment": environment, "versions": versions}))
"""


def _target_metadata(
    interpreter: str,
    distribution_names: list[str],
) -> tuple[dict[str, str], dict[str, str | None], str | None]:
    """Read marker environment + installed versions from ``interpreter``."""

    try:
        proc = subprocess.run(  # noqa: S603
            [
                interpreter,
                "-c",
                _TARGET_METADATA_PROBE,
                json.dumps(distribution_names),
            ],
            capture_output=True,
            text=True,
            timeout=TARGET_METADATA_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return (
            {},
            {},
            "target interpreter metadata probe timed out after "
            f"{TARGET_METADATA_TIMEOUT_SECONDS}s",
        )
    except OSError as error:
        return {}, {}, f"could not run target interpreter {interpreter!r}: {error}"
    if proc.returncode != 0:
        diagnostic = (proc.stderr or proc.stdout or "").strip()
        return {}, {}, diagnostic or f"metadata probe exited {proc.returncode}"
    try:
        payload = json.loads(proc.stdout)
        return payload["environment"], payload["versions"], None
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        return {}, {}, f"invalid metadata probe response: {error}"


def _active_test_packages(
    *,
    environment: dict[str, str],
    versions: dict[str, str | None],
    requirements: dict[str, tuple[Requirement, ...]],
) -> tuple[tuple[tuple[str, str, str], ...], tuple[DependencyProblem, ...]]:
    """Apply canonical markers and evaluate installed distribution versions."""

    from packaging.markers import default_environment
    from packaging.utils import canonicalize_name
    from packaging.version import InvalidVersion, Version

    marker_environment = default_environment()
    marker_environment.update(environment)
    active: list[tuple[str, str, str]] = []
    problems: list[DependencyProblem] = []
    for import_name, distribution_name, why in REQUIRED_TEST_PACKAGES:
        canonical_name = canonicalize_name(distribution_name)
        candidates = requirements.get(canonical_name)
        if candidates is None:
            problems.append(
                DependencyProblem(
                    import_name=import_name,
                    distribution_name=distribution_name,
                    installed_version=versions.get(distribution_name),
                    requirement=f"declared in .[{TEST_EXTRAS_NAME}]",
                )
            )
            continue
        applicable = tuple(
            requirement
            for requirement in candidates
            if requirement.marker is None
            or requirement.marker.evaluate(environment=marker_environment)
        )
        if not applicable:
            continue
        active.append((import_name, distribution_name, why))
        installed = versions.get(distribution_name)
        required_range = " and ".join(
            str(requirement.specifier) or "any version" for requirement in applicable
        )
        if installed is None:
            problems.append(
                DependencyProblem(
                    import_name=import_name,
                    distribution_name=distribution_name,
                    installed_version=None,
                    requirement=required_range,
                )
            )
            continue
        try:
            parsed_version = Version(installed)
            satisfies = all(
                not requirement.specifier or parsed_version in requirement.specifier
                for requirement in applicable
            )
        except InvalidVersion:
            satisfies = False
        if not satisfies:
            problems.append(
                DependencyProblem(
                    import_name=import_name,
                    distribution_name=distribution_name,
                    installed_version=installed,
                    requirement=required_range,
                )
            )
    return tuple(active), tuple(problems)


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

    ``missing`` is the list of import names that failed or whose installed
    distributions are unsatisfied; ``ok`` mirrors the bool the caller usually
    wants. ``message`` is a one-liner suitable for a step-result summary.
    """

    ok: bool
    missing: tuple[str, ...]
    message: str
    interpreter: str
    problems: tuple[DependencyProblem, ...] = ()

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
    """Probe imports and canonical distribution versions in ``python``.

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
    try:
        requirements = canonical_test_requirements()
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        ModuleNotFoundError,
    ) as error:
        import_names = tuple(pkg for pkg, _, _ in REQUIRED_TEST_PACKAGES)
        return TestEnvStatus(
            ok=False,
            missing=import_names,
            message=f"canonical .[{TEST_EXTRAS_NAME}] requirements invalid: {error}",
            interpreter=interp,
        )
    distribution_names = [pkg for _, pkg, _ in REQUIRED_TEST_PACKAGES]
    environment, versions, metadata_error = _target_metadata(interp, distribution_names)
    if metadata_error:
        import_names = tuple(pkg for pkg, _, _ in REQUIRED_TEST_PACKAGES)
        return TestEnvStatus(
            ok=False,
            missing=import_names,
            message=f"target interpreter metadata probe failed: {metadata_error[:512]}",
            interpreter=interp,
        )

    active_packages, version_problems = _active_test_packages(
        environment=environment,
        versions=versions,
        requirements=requirements,
    )
    import_names = [pkg for pkg, _, _ in active_packages]
    probe = "; ".join(f"import {name}" for name in import_names)

    proc = subprocess.run(  # noqa: S603
        [interp, "-c", probe],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0 and not version_problems:
        return TestEnvStatus(
            ok=True,
            missing=(),
            message=(
                f"all {len(import_names)} required test packages importable "
                "and version-compatible"
            ),
            interpreter=interp,
        )

    if proc.returncode == 0:
        return TestEnvStatus(
            ok=False,
            missing=tuple(problem.import_name for problem in version_problems),
            message=(
                "unsatisfied test requirements: "
                + "; ".join(problem.render() for problem in version_problems)
            ),
            interpreter=interp,
            problems=version_problems,
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
            problems=version_problems,
        )

    import_problems = tuple(
        DependencyProblem(
            import_name=name,
            distribution_name=next(
                distribution
                for import_name, distribution, _ in active_packages
                if import_name == name
            ),
            installed_version=versions.get(
                next(
                    distribution
                    for import_name, distribution, _ in active_packages
                    if import_name == name
                )
            ),
            requirement="importable",
        )
        for name in missing
    )
    all_problems = (*version_problems, *import_problems)
    return TestEnvStatus(
        ok=False,
        missing=tuple(
            dict.fromkeys((*missing, *(p.import_name for p in version_problems)))
        ),
        message=(
            f"missing required test imports: {', '.join(missing)}"
            + (
                "; unsatisfied versions: "
                + "; ".join(problem.render() for problem in version_problems)
                if version_problems
                else ""
            )
        ),
        interpreter=interp,
        problems=all_problems,
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
