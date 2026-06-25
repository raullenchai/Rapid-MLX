# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts.pr_validate._test_env`` and the
``TestEnvCheckStep`` pipeline gate (issue #185).

Three contracts pinned:

1. A healthy host (where every `REQUIRED_TEST_PACKAGES` import works
   in the running interpreter) passes the check.
2. A broken host (one of the required imports raises in a fresh
   subprocess) gets a `fail` result whose `details` names the missing
   package AND surfaces the exact `pip install '.[test]'` recovery
   command with the interpreter that needs the install.
3. The canonical `test` extras in `pyproject.toml` includes
   `pytest-asyncio` — this is the load-bearing dep that the bug was
   silently dropping. A future refactor that renames the extras or
   drops the plugin should fail this test.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Standard pyproject parser — stdlib on 3.11+, vendored tomli marker
# on 3.10 (already an existing dev dep).
if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover — branch only taken on 3.10 CI
    import tomli as tomllib

from scripts.pr_validate._test_env import (
    REQUIRED_TEST_PACKAGES,
    TEST_EXTRAS_NAME,
    TestEnvStatus,
    auto_install_disabled,
    check_test_env,
)
from scripts.pr_validate.context import Context
from scripts.pr_validate.steps.test_env_check import TestEnvCheckStep

# ---------------------------------------------------------------------------
# pyproject.toml canonical-source contract
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def project_table() -> dict:
    """Load the repo root's pyproject.toml. The test deliberately
    walks up from this file (not ``Path.cwd()``) so a CI runner that
    invokes pytest from a sub-directory still finds the right file."""
    repo_root = Path(__file__).resolve().parent.parent
    pyproject = repo_root / "pyproject.toml"
    return tomllib.loads(pyproject.read_text())


class TestPyprojectTestExtras:
    def test_test_extras_exists(self, project_table):
        """The `test` extras must exist — it's the canonical source the
        venv builder installs from. If a future refactor deletes it,
        pr_validate's auto-recover falls back to ad-hoc lists and the
        same bug returns."""
        extras = project_table["project"]["optional-dependencies"]
        assert TEST_EXTRAS_NAME in extras, (
            f"`{TEST_EXTRAS_NAME}` extras missing from pyproject.toml — "
            "pr_validate's `test_env_check` step cannot auto-recover. "
            "See scripts/pr_validate/_test_env.py."
        )

    def test_test_extras_includes_pytest_asyncio(self, project_table):
        """The load-bearing dep. ``pytest.ini`` sets `asyncio_mode = auto`
        which means every `async def test_*` is collected as an asyncio
        test — without pytest-asyncio, those fail at collection and the
        full_unit step reports a hundreds-deep "regression" that's
        really just a missing plugin (#185)."""
        deps = project_table["project"]["optional-dependencies"][TEST_EXTRAS_NAME]
        assert any("pytest-asyncio" in d for d in deps), (
            f"`pytest-asyncio` missing from [{TEST_EXTRAS_NAME}] extras — "
            "this is the dep whose absence reopens issue #185."
        )

    def test_test_extras_includes_pytest_itself(self, project_table):
        """Defensive: pytest is the runner. A `test` extras that didn't
        include pytest would be useless — and prior to this fix, the
        only reason it ever worked was because pytest happened to be
        installed transitively from `dev`."""
        deps = project_table["project"]["optional-dependencies"][TEST_EXTRAS_NAME]
        assert any(d.startswith("pytest") and "asyncio" not in d for d in deps), (
            "no pytest entry in `test` extras — see _test_env.py"
        )

    def test_dev_extras_superset_of_test(self, project_table):
        """`dev` must contain every dep `test` does so the existing
        `pip install .[dev] && pytest` contributor workflow still
        works after the split. Pure containment check on package-name
        prefixes (versions allowed to differ — only the package name
        is load-bearing)."""
        extras = project_table["project"]["optional-dependencies"]
        # Compare just the package-name token before any version
        # specifier or environment marker.
        names_in = lambda key: {  # noqa: E731
            _pkg_name(d) for d in extras[key]
        }
        missing = names_in(TEST_EXTRAS_NAME) - names_in("dev")
        assert not missing, (
            f"`dev` extras is missing test deps: {sorted(missing)}. "
            "Keep `dev` a strict superset of `test` so `pip install "
            "'.[dev]' && pytest` keeps working — there's a comment in "
            "pyproject.toml asking the next editor to maintain this."
        )


def _pkg_name(dep: str) -> str:
    """Pull the bare package name out of a PEP 508-ish requirement
    string. ``"pytest-asyncio>=0.21.0"`` → ``"pytest-asyncio"``;
    ``'tomli>=2.0.1; python_version < "3.11"'`` → ``"tomli"``."""
    # Strip env markers (everything after the ; ), then version spec.
    head = dep.split(";", 1)[0].strip()
    for sep in ("<=", ">=", "==", "<", ">", "~=", "!="):
        if sep in head:
            head = head.split(sep, 1)[0]
            break
    # Strip extras like "outlines[mlxlm]".
    return head.split("[", 1)[0].strip()


# ---------------------------------------------------------------------------
# check_test_env() — host-state probe
# ---------------------------------------------------------------------------


class TestCheckTestEnv:
    def test_returns_ok_on_healthy_host(self):
        """The running interpreter HAS pytest + pytest-asyncio (we're
        in a pytest invocation right now), so the probe must report
        ok with an empty missing list."""
        status = check_test_env()
        assert status.ok is True
        assert status.missing == ()
        assert status.interpreter == sys.executable

    def test_reports_missing_module_on_broken_host(self, tmp_path):
        """Build a tiny venv with pytest but NOT pytest-asyncio, then
        probe IT — this is the exact failure mode #185 documents.

        Skipped if the host doesn't have `venv` available (very rare
        on macOS Homebrew Python 3.12, the only documented target)."""
        try:
            import venv  # noqa: F401
        except ImportError:  # pragma: no cover
            pytest.skip("venv module unavailable")

        venv_dir = tmp_path / "broken"
        import venv as _venv

        # ``with_pip=True`` so we can install pytest into it. Use
        # ``symlinks=True`` (the POSIX default) — on macOS with a
        # uv-managed CPython the python binary is dynamically linked
        # against ``@rpath/libpython3.12.dylib``; with ``symlinks=False``
        # the copied bin/python3.12 can't resolve the dylib and
        # ensurepip's ``_call_new_python`` aborts with SIGABRT during
        # venv bootstrap. The original comment about "permission
        # surprises on weird filesystems" doesn't apply on tmp_path
        # (always a real fs the test runner created itself), so the
        # default-symlinks shape is both safer and correct here.
        _venv.create(venv_dir, with_pip=True, symlinks=True)
        python = venv_dir / "bin" / "python"
        assert python.exists(), "venv builder didn't produce a python"

        # Install pytest only — pytest-asyncio deliberately omitted.
        # --quiet keeps the test output readable; --disable-pip-version-check
        # suppresses the upgrade nag that adds noise on slow CI.
        import subprocess

        subprocess.run(  # noqa: S603
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--quiet",
                "--disable-pip-version-check",
                "pytest",
            ],
            check=True,
            capture_output=True,
        )

        status = check_test_env(python=str(python))
        assert status.ok is False
        # pytest_asyncio is the import name (underscore), not the pip
        # name (dash). Pin the import name — that's what the probe
        # actually runs.
        assert "pytest_asyncio" in status.missing
        # pytest itself was installed → must NOT appear in missing.
        assert "pytest" not in status.missing
        # The hint must reference the canonical extras name AND the
        # exact interpreter — handing a different python to pip would
        # install into the wrong site-packages and the operator's next
        # pr_validate run would fail identically.
        assert f"'.[{TEST_EXTRAS_NAME}]'" in status.install_hint
        assert str(python) in status.install_hint

    def test_batch_fail_with_individual_passes_is_treated_as_fail(self, tmp_path):
        """Codex r1 BLOCKING: previously a batch-import failure that
        re-probed clean per-module returned ``ok=True``. That hides a
        real failure mode pytest hits at startup (plugin registration
        order, sys.path mutation by one import that breaks the next).
        We simulate it by patching subprocess.run so the batch probe
        exits non-zero with a recognizable stderr while each
        individual probe exits 0 — the helper must report
        ``ok=False`` and surface the batch stderr."""
        import subprocess

        from scripts.pr_validate import _test_env as mod

        # The batch probe is the FIRST call (one combined "import X;
        # import Y" command); individual probes are subsequent calls.
        # We construct a side_effect list that returns a non-zero
        # CompletedProcess for the batch and zero for each individual.
        batch_stderr = (
            "Traceback (most recent call last):\n"
            "  File '<string>', line 1, in <module>\n"
            "RuntimeError: simulated plugin-order collision\n"
        )
        n_packages = len(mod.REQUIRED_TEST_PACKAGES)
        results = [
            subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr=batch_stderr
            ),
            *[
                subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
                for _ in range(n_packages)
            ],
        ]
        with patch("scripts.pr_validate._test_env.subprocess.run", side_effect=results):
            status = mod.check_test_env(python="/fake/python")

        assert status.ok is False, (
            "batch-fail+individual-pass must report broken, not ok — the "
            "combined import is the order pytest will actually take"
        )
        # The diagnostic surfaces the batch stderr so the operator can
        # see WHY the batch failed without re-running by hand.
        assert "simulated plugin-order collision" in status.message
        # `missing` should be populated (the helper marks every package
        # as suspect when it can't pinpoint which one breaks the batch)
        # so downstream auto-install still has something to act on.
        assert len(status.missing) == n_packages

    def test_install_hint_uses_the_probed_interpreter_not_sys_executable(self):
        """Edge: a TestEnvStatus constructed for some OTHER python (a
        CI worker, a docker container) must surface THAT interpreter
        in the recovery hint, not ``sys.executable``. Cheap regression
        guard for a class of bug where someone "simplifies" the
        property by hardcoding sys.executable."""
        fake_python = "/opt/sandbox/python3.13"
        status = TestEnvStatus(
            ok=False,
            missing=("pytest_asyncio",),
            message="…",
            interpreter=fake_python,
        )
        assert status.install_hint.startswith(fake_python)
        assert (
            sys.executable not in status.install_hint or sys.executable == fake_python
        )


# ---------------------------------------------------------------------------
# auto_install_disabled() — env-var feature flag
# ---------------------------------------------------------------------------


class TestAutoInstallDisabled:
    @pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy_values_disable(self, val):
        with patch.dict(os.environ, {"PR_VALIDATE_NO_AUTO_INSTALL": val}):
            assert auto_install_disabled() is True

    @pytest.mark.parametrize("val", ["", "0", "false", "no", "off", "maybe"])
    def test_falsy_values_enable(self, val):
        with patch.dict(os.environ, {"PR_VALIDATE_NO_AUTO_INSTALL": val}):
            assert auto_install_disabled() is False

    def test_unset_enables(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PR_VALIDATE_NO_AUTO_INSTALL", None)
            assert auto_install_disabled() is False


# ---------------------------------------------------------------------------
# TestEnvCheckStep — pipeline integration
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_ctx(tmp_path):
    """A Context pointing at a tmpdir with a stub pyproject so
    ``__post_init__`` doesn't bail on cwd."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fake'\n")
    old_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        ctx = Context(pr_number=999, verbose=False)
        ctx.work_dir = tmp_path / "work"
        ctx.work_dir.mkdir()
        yield ctx
    finally:
        os.chdir(old_cwd)


class TestStepIntegration:
    def test_pass_on_healthy_host(self, fake_ctx):
        """End-to-end on the actual running interpreter — same logic
        as `test_returns_ok_on_healthy_host` but driven through the
        Step API so the scorecard wiring is also exercised. The
        artifact path must be created and the result must include it."""
        step = TestEnvCheckStep()
        result = step.run(fake_ctx)
        assert result.status == "pass"
        # Artifact must exist and name the interpreter so the operator
        # can audit which Python actually got probed.
        assert result.artifacts
        log = Path(result.artifacts[0]).read_text()
        assert sys.executable in log
        assert "status: ok" in log

    def test_fail_when_auto_install_disabled_and_packages_missing(self, fake_ctx):
        """Patch `check_test_env` to fake a broken host AND set the
        opt-out env var. The step must emit fail with a `details`
        block that names the missing package AND surfaces the
        canonical recovery command — those are the two operator-
        actionable bits."""
        bad = TestEnvStatus(
            ok=False,
            missing=("pytest_asyncio",),
            message="missing required test packages: pytest_asyncio",
            interpreter="/opt/fake/python",
        )
        with (
            patch.dict(os.environ, {"PR_VALIDATE_NO_AUTO_INSTALL": "1"}),
            patch(
                "scripts.pr_validate.steps.test_env_check.check_test_env",
                return_value=bad,
            ),
        ):
            step = TestEnvCheckStep()
            result = step.run(fake_ctx)

        assert result.status == "fail"
        assert "pytest_asyncio" in result.details
        # The recovery command must reference the canonical extras AND
        # the probed interpreter so the operator copy-pastes the right
        # python.
        assert f"'.[{TEST_EXTRAS_NAME}]'" in result.details
        assert "/opt/fake/python" in result.details
        # Summary one-liner (shown in the scorecard table) must call
        # out the auto-install state — otherwise the operator has to
        # open the details block to know what to do.
        assert "auto-install disabled" in result.summary

    def test_auto_install_attempts_and_recovers(self, fake_ctx):
        """Initial probe fails; the trusted-pins install path (added
        for #275) runs first and recovers — the step must report
        ``pass`` with a summary mentioning "trusted-pins" so the
        operator knows the PyPI-only path (not the PR's working
        tree) was the source.

        Critically, ``install_test_extras`` must NOT be called when
        trusted pins suffice — that's the whole #275 fix.
        """
        broken = TestEnvStatus(
            ok=False,
            missing=("pytest_asyncio",),
            message="missing required test packages: pytest_asyncio",
            interpreter=sys.executable,
        )
        healthy = TestEnvStatus(
            ok=True,
            missing=(),
            message="all 2 required test packages importable",
            interpreter=sys.executable,
        )
        # Two-shot probe: first call sees the broken state, second
        # (after the trusted-pins install) sees the recovered state.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PR_VALIDATE_NO_AUTO_INSTALL", None)
            with (
                patch(
                    "scripts.pr_validate.steps.test_env_check.check_test_env",
                    side_effect=[broken, healthy],
                ),
                patch(
                    "scripts.pr_validate.steps.test_env_check.install_trusted_pins",
                    return_value=(
                        True,
                        "Successfully installed pytest-asyncio-0.21.0\n",
                    ),
                ) as mock_pins,
                patch(
                    "scripts.pr_validate.steps.test_env_check.install_test_extras",
                ) as mock_install,
            ):
                step = TestEnvCheckStep()
                result = step.run(fake_ctx)

        assert result.status == "pass"
        # Pin the exact trusted-pins wording — codex r2 NIT was that
        # the previous loose ``or "installed"`` matcher would also
        # accept a summary from the old project-extras path, masking
        # a regression that flipped the order.
        assert "trusted-pins" in result.summary, (
            f"expected 'trusted-pins' in summary, got: {result.summary!r}"
        )
        # Trusted-pins path runs exactly once.
        assert mock_pins.call_count == 1
        # Project-extras path MUST NOT run when trusted pins recover —
        # that's the supply-chain integrity guarantee.
        assert mock_install.call_count == 0

    def test_auto_install_runs_but_does_not_fix(self, fake_ctx):
        """Initial probe fails; trusted-pins succeed but don't fix
        everything; project-extras install runs (PR didn't touch dep
        files); post-install probe STILL fails. This is the "pip
        silently fell back to --user and the plugin is in the wrong
        site-packages" case — the step must report fail with both
        the initial AND post-install missing list so the operator
        sees the install didn't take."""
        broken = TestEnvStatus(
            ok=False,
            missing=("pytest_asyncio",),
            message="missing required test packages: pytest_asyncio",
            interpreter=sys.executable,
        )
        # Three probes: initial (broken), after trusted-pins
        # (still broken — so we fall through to project-extras), and
        # after project-extras (still broken — install didn't take).
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PR_VALIDATE_NO_AUTO_INSTALL", None)
            with (
                patch(
                    "scripts.pr_validate.steps.test_env_check.check_test_env",
                    side_effect=[broken, broken, broken],
                ),
                patch(
                    "scripts.pr_validate.steps.test_env_check.install_trusted_pins",
                    return_value=(True, "(fake trusted-pins output)"),
                ),
                patch(
                    "scripts.pr_validate.steps.test_env_check.install_test_extras",
                    return_value=(True, "(fake pip output)"),
                ),
            ):
                step = TestEnvCheckStep()
                result = step.run(fake_ctx)

        assert result.status == "fail"
        # Operator must see both the failure message AND the pip log
        # (truncated or not) so they can diagnose where it went wrong.
        assert "auto-install attempted" in result.details.lower()
        assert "(fake pip output)" in result.details


# ---------------------------------------------------------------------------
# Required-packages constant — pin the shape
# ---------------------------------------------------------------------------


class TestRequiredPackages:
    def test_pytest_and_pytest_asyncio_required(self):
        """The two non-negotiables for this repo. A future refactor
        that drops `pytest_asyncio` from this tuple (e.g. because the
        author tests with `pytest-anyio` locally and forgets the gate
        runs on a different host) silently reopens #185."""
        names = {pkg for pkg, _, _ in REQUIRED_TEST_PACKAGES}
        assert "pytest" in names
        assert "pytest_asyncio" in names

    def test_every_entry_has_a_pip_name_with_version(self):
        """Defensive: each entry must carry a version constraint so
        the auto-install message isn't ambiguous."""
        for pkg, pip_name, _ in REQUIRED_TEST_PACKAGES:
            assert pip_name, f"empty pip name for {pkg!r}"
            assert any(c in pip_name for c in (">=", "==", "~=", ">")), (
                f"{pip_name!r} for {pkg!r} has no version constraint"
            )


# ---------------------------------------------------------------------------
# Supply-chain integrity (#275 / codex review on PR #885)
# ---------------------------------------------------------------------------


class TestSupplyChainIntegrity:
    """Pin the #275 fix: the test_env_check step MUST NOT auto-install
    from the PR's working tree when the PR has touched a dep-declaration
    file, because the install would execute attacker-controlled build
    hooks inside the validator venv.

    The trusted-pins path (hardcoded version-pinned PyPI install) is the
    safe substitute; the project-extras path is the unsafe one.
    """

    def test_pr_touches_dep_files_flags_pyproject_toml(self):
        """The canonical case — a PR that modifies pyproject.toml is
        the exact threat #275 describes."""
        from scripts.pr_validate._test_env import pr_touches_dep_files

        assert pr_touches_dep_files(["pyproject.toml"]) == ["pyproject.toml"]
        assert pr_touches_dep_files(["docs/foo.md", "pyproject.toml"]) == [
            "pyproject.toml"
        ]

    def test_pr_touches_dep_files_returns_empty_for_safe_diffs(self):
        """A diff that only touches docs / source must NOT trip the
        safety gate — otherwise every non-trivial PR would block at
        test_env_check."""
        from scripts.pr_validate._test_env import pr_touches_dep_files

        assert pr_touches_dep_files(["docs/foo.md"]) == []
        assert pr_touches_dep_files(["vllm_mlx/server.py"]) == []
        assert pr_touches_dep_files([]) == []

    def test_pr_touches_dep_files_catches_all_dep_files(self):
        """Every entry in DEP_DECLARATION_FILES_DENYLIST must be
        detectable — a typo there would silently disable the gate."""
        from scripts.pr_validate._test_env import (
            DEP_DECLARATION_FILES_DENYLIST,
            pr_touches_dep_files,
        )

        for dep_file in DEP_DECLARATION_FILES_DENYLIST:
            assert pr_touches_dep_files([dep_file]) == [dep_file], (
                f"{dep_file!r} not detected by pr_touches_dep_files"
            )

    def test_pr_touches_dep_files_catches_arbitrary_requirements_variants(self):
        """Codex r2 BLOCKING: the previous exact-match list let
        ``requirements-test.txt`` / ``requirements-prod.txt`` through
        the gate. The matcher is now prefix-based for repo-root
        ``requirements*.txt`` files, so every variant a contributor
        might invent is caught without us enumerating them."""
        from scripts.pr_validate._test_env import (
            is_dep_declaration_file,
            pr_touches_dep_files,
        )

        # Variants that MUST be flagged (the regression cases).
        for variant in (
            "requirements.txt",
            "requirements-dev.txt",
            "requirements-test.txt",
            "requirements-prod.txt",
            "requirements-pin.txt",
            "requirements-ci.txt",
        ):
            assert is_dep_declaration_file(variant), (
                f"{variant!r} not flagged by is_dep_declaration_file"
            )
            assert pr_touches_dep_files([variant]) == [variant]

        # Negative cases — subdirectory requirements files don't drive
        # pr_validate's recovery install and should NOT trip the gate.
        # (Their contents may still be supply-chain interesting via
        # other means, but they don't enable the #275 attack vector.)
        for safe in (
            "vendor/requirements.txt",
            "tests/fixtures/requirements.txt",
            "docs/requirements.md",  # not .txt
            "requirements.yaml",  # not .txt
        ):
            assert not is_dep_declaration_file(safe), (
                f"{safe!r} incorrectly flagged by is_dep_declaration_file"
            )

    def test_supply_chain_hook_matcher_catches_arbitrary_requirements_variants(self):
        """Codex r2 BLOCKING: supply-chain's hook matcher and the
        test-env-check matcher previously had divergent lists. Now
        they share ``is_dep_declaration_file`` so any new
        ``requirements*.txt`` variant is BLOCKING for external authors
        at the supply-chain step too."""
        from scripts.pr_validate.steps.supply_chain import _is_hook_file

        for variant in (
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "requirements.txt",
            "requirements-dev.txt",
            "requirements-test.txt",
            "requirements-pin.txt",
            "requirements-anything.txt",
            ".github/workflows/ci.yml",
            "conftest.py",
        ):
            assert _is_hook_file(variant), (
                f"{variant!r} not flagged by supply_chain._is_hook_file"
            )

        # Negative: source files and docs MUST NOT be flagged or
        # every PR would trip supply-chain.
        for safe in (
            "vllm_mlx/scheduler.py",
            "docs/foo.md",
            "tests/test_foo.py",
        ):
            assert not _is_hook_file(safe), (
                f"{safe!r} incorrectly flagged by supply_chain._is_hook_file"
            )

    def test_auto_install_refused_when_pr_touches_pyproject(self, fake_ctx):
        """The headline #275 test: when the PR modifies pyproject.toml
        and the trusted-pins install doesn't fully recover, the step
        REFUSES to fall back to ``pip install '.[test]'`` and reports
        ``fail`` with the offending file named in the warning."""
        broken = TestEnvStatus(
            ok=False,
            missing=("pytest_asyncio",),
            message="missing required test packages: pytest_asyncio",
            interpreter=sys.executable,
        )
        fake_ctx.files_changed = ["pyproject.toml", "scripts/foo.py"]

        install_called = {"yes": False}

        def fail_if_install_runs(*_args, **_kwargs):
            install_called["yes"] = True
            return (True, "ATTACK: build hook ran from pyproject.toml")

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PR_VALIDATE_NO_AUTO_INSTALL", None)
            with (
                patch(
                    "scripts.pr_validate.steps.test_env_check.check_test_env",
                    side_effect=[broken, broken],
                ),
                patch(
                    "scripts.pr_validate.steps.test_env_check.install_trusted_pins",
                    return_value=(True, "trusted-pins partial recovery"),
                ),
                patch(
                    "scripts.pr_validate.steps.test_env_check.install_test_extras",
                    side_effect=fail_if_install_runs,
                ),
            ):
                step = TestEnvCheckStep()
                result = step.run(fake_ctx)

        assert result.status == "fail"
        # The offending file MUST be named in the warning so the
        # operator knows what to scrutinize before installing manually.
        assert "pyproject.toml" in result.details
        # Summary must call out the refusal so it's visible without
        # opening the details block.
        assert (
            "refused" in result.summary.lower()
            or "project-extras" in result.summary.lower()
        )
        # The unsafe install path MUST NOT have run.
        assert install_called["yes"] is False, (
            "install_test_extras ran despite the PR touching pyproject.toml — "
            "this is the supply-chain integrity bug #275 is supposed to fix"
        )

    def test_auto_install_proceeds_when_pr_does_not_touch_dep_files(self, fake_ctx):
        """No regression on happy path: a PR that doesn't touch any
        dep file still gets the project-extras fallback when trusted
        pins don't fully recover."""
        broken = TestEnvStatus(
            ok=False,
            missing=("pytest_asyncio",),
            message="missing required test packages: pytest_asyncio",
            interpreter=sys.executable,
        )
        healthy = TestEnvStatus(
            ok=True,
            missing=(),
            message="all 2 required test packages importable",
            interpreter=sys.executable,
        )
        # Pure source-only change — must NOT be flagged.
        fake_ctx.files_changed = ["vllm_mlx/scheduler.py", "tests/test_scheduler.py"]

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PR_VALIDATE_NO_AUTO_INSTALL", None)
            with (
                patch(
                    "scripts.pr_validate.steps.test_env_check.check_test_env",
                    side_effect=[broken, broken, healthy],
                ),
                patch(
                    "scripts.pr_validate.steps.test_env_check.install_trusted_pins",
                    return_value=(True, "trusted-pins partial"),
                ),
                patch(
                    "scripts.pr_validate.steps.test_env_check.install_test_extras",
                    return_value=(True, "Successfully installed"),
                ) as mock_install,
            ):
                step = TestEnvCheckStep()
                result = step.run(fake_ctx)

        assert result.status == "pass"
        # The project-extras path DID run since trusted pins were
        # insufficient and the PR was clean.
        assert mock_install.call_count == 1

    def test_trusted_pins_path_used_first_even_on_clean_pr(self, fake_ctx):
        """Trusted-pins runs FIRST regardless of whether the PR is
        clean. That's the design — never install from the PR's working
        tree unless we have to. A clean PR is just allowed to fall
        through to the project-extras path if trusted pins fall short.

        Hardened (codex r1 NIT, #275 follow-up): the test exercises
        BOTH paths by leaving the post-trusted-pins probe broken so
        the step is forced through the project-extras fallback. A
        future refactor that flipped the order (or dropped the
        fallback) would break ``call_order == ["trusted_pins",
        "project_extras"]``, not just the first element."""
        broken = TestEnvStatus(
            ok=False,
            missing=("pytest_asyncio",),
            message="missing required test packages: pytest_asyncio",
            interpreter=sys.executable,
        )
        healthy = TestEnvStatus(
            ok=True,
            missing=(),
            message="all 2 required test packages importable",
            interpreter=sys.executable,
        )
        fake_ctx.files_changed = ["vllm_mlx/scheduler.py"]

        call_order: list[str] = []

        def trusted_pins(*_args, **_kwargs):
            call_order.append("trusted_pins")
            return (True, "trusted partial")

        def project_extras(*_args, **_kwargs):
            call_order.append("project_extras")
            return (True, "extras recovered")

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PR_VALIDATE_NO_AUTO_INSTALL", None)
            with (
                patch(
                    "scripts.pr_validate.steps.test_env_check.check_test_env",
                    # initial → broken; after trusted-pins → STILL
                    # broken (forces fallback); after project-extras
                    # → healthy.
                    side_effect=[broken, broken, healthy],
                ),
                patch(
                    "scripts.pr_validate.steps.test_env_check.install_trusted_pins",
                    side_effect=trusted_pins,
                ),
                patch(
                    "scripts.pr_validate.steps.test_env_check.install_test_extras",
                    side_effect=project_extras,
                ),
            ):
                step = TestEnvCheckStep()
                result = step.run(fake_ctx)

        assert result.status == "pass"
        assert call_order == ["trusted_pins", "project_extras"], (
            f"expected trusted_pins THEN project_extras, got: {call_order}"
        )

    def test_trusted_pins_uses_isolated_pip_flag(self):
        """``--isolated`` blocks a user-level ``pip.conf`` from injecting
        a malicious ``--extra-index-url`` while the validator's
        recovery install is running. Pin the flag so a "simplification"
        refactor can't quietly drop it."""
        import subprocess as _sub

        from scripts.pr_validate import _test_env as mod

        captured: dict = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return _sub.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch.object(mod.subprocess, "run", side_effect=fake_run):
            ok, _ = mod.install_trusted_pins()

        assert ok is True
        assert "--isolated" in captured["cmd"], (
            "pip must run with --isolated to block user-level config injection"
        )
        # Every TRUSTED_TEST_PINS entry must appear in the install command —
        # if a future edit drops one, the validator silently degrades.
        for pin in mod.TRUSTED_TEST_PINS:
            assert pin in captured["cmd"]

    def test_supply_chain_runs_before_auto_installing_steps(self):
        """ORDERING INVARIANT (#275): ``supply_chain`` MUST appear in
        ``STEPS`` before ``test_env_check`` and before any other step
        that may ``pip install`` from the PR's working tree.

        Otherwise a malicious PR can get its build hook executed inside
        the validator venv before the supply-chain scan flags it. This
        test exists so the invariant is preserved across future
        refactors — a `grep`able guarantee, not a code-review hope."""
        from scripts.pr_validate.runner import STEPS

        names = [s.name for s in STEPS]
        assert "supply_chain" in names
        assert "test_env_check" in names

        sc_idx = names.index("supply_chain")
        tec_idx = names.index("test_env_check")
        assert sc_idx < tec_idx, (
            f"supply_chain (idx {sc_idx}) must run BEFORE test_env_check "
            f"(idx {tec_idx}) — see scripts/pr_validate/README.md "
            "'Threat model' (#275)"
        )

    def test_malicious_pyproject_blocked_at_supply_chain_before_test_env(
        self, tmp_path, monkeypatch, capsys
    ):
        """End-to-end simulation of the #275 attack: an external-author
        PR whose diff adds a build-hook to ``pyproject.toml``. The
        supply-chain step MUST flag it [BLOCKING] BEFORE
        test_env_check has a chance to run ``pip install '.[test]'``."""
        from scripts.pr_validate.runner import run_pipeline
        from scripts.pr_validate.steps.supply_chain import SupplyChainStep
        from scripts.pr_validate.steps.test_env_check import TestEnvCheckStep

        # Build a fake repo root so Context.__post_init__ is happy.
        (tmp_path / "pyproject.toml").write_text("[project]\nname='fake'\n")
        monkeypatch.chdir(tmp_path)

        # Malicious diff: external PR adds a build hook in pyproject.toml.
        # Note the `+++ b/` path, a hunk header, and an added line that
        # would execute on `pip install` (a fake build-backend swap).
        diff_path = tmp_path / "pr.diff"
        diff_path.write_text(
            "diff --git a/pyproject.toml b/pyproject.toml\n"
            "--- a/pyproject.toml\n"
            "+++ b/pyproject.toml\n"
            "@@ -1,3 +1,4 @@\n"
            " [build-system]\n"
            "+requires = ['evil-build-backend @ file:///tmp/attack']\n"
            ' build-backend = "setuptools.build_meta"\n'
        )

        # Stand-in fetch step that wires up the Context fields the
        # supply-chain + test-env steps read.
        from scripts.pr_validate.base import Step, StepResult

        class _MaliciousFetch(Step):
            name = "fetch"
            description = "fake malicious PR fetch"

            def run(self, ctx):  # type: ignore[no-untyped-def]
                ctx.pr_title = "innocent looking title"
                ctx.pr_author = "untrusted-contributor"
                ctx.head_sha = "deadbeef"
                ctx.diff_path = str(diff_path)
                ctx.files_changed = ["pyproject.toml"]
                ctx.pr_is_external = True  # critical: external author
                return StepResult(name=self.name, status="pass", summary="ok")

        # Sentinel: if test_env_check EVER calls install_test_extras
        # in this scenario, the bug is back.
        install_ran = {"yes": False}

        def trap_install(*_args, **_kwargs):
            install_ran["yes"] = True
            return (True, "PWNED — build hook would have run here")

        monkeypatch.setattr(
            "scripts.pr_validate.steps.test_env_check.install_test_extras",
            trap_install,
        )

        # Use a real SupplyChainStep + TestEnvCheckStep so we exercise
        # the actual ordering invariant.
        steps = [_MaliciousFetch(), SupplyChainStep(), TestEnvCheckStep()]
        rc = run_pipeline(pr_number=275, fail_fast=True, steps=steps)
        captured = capsys.readouterr()

        # 1. Pipeline exits non-zero because supply_chain blocked.
        assert rc == 1, (
            f"pipeline should have BLOCKED, got rc={rc}. stdout:\n"
            f"{captured.out}\nstderr:\n{captured.err}"
        )
        # 2. supply_chain ran and flagged the change.
        assert "## [supply_chain]" in captured.err
        # 3. With fail_fast=True, test_env_check should NOT have run
        # AFTER supply_chain failed.
        assert "## [test_env_check]" not in captured.err, (
            "test_env_check ran after supply_chain failed — fail-fast "
            "should have stopped the pipeline before the unsafe install"
        )
        # 4. The unsafe install path was never invoked.
        assert install_ran["yes"] is False, (
            "install_test_extras was called from a malicious-pyproject PR — "
            "this is the exact #275 bug; supply_chain must run first AND "
            "block AND test_env_check's auto-install must be gated"
        )
