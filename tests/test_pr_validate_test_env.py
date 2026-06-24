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

        # `with_pip=True` so we can install pytest into it; turn off
        # symlinks to avoid permission surprises on weird filesystems.
        _venv.create(venv_dir, with_pip=True, symlinks=False)
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
        """Initial probe fails; auto-install path runs; post-install
        probe passes. The step must report `pass` and the summary
        must mention "installed" so the operator knows the run
        recovered (and the host was mutated)."""
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
        # (after install) sees the recovered state.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PR_VALIDATE_NO_AUTO_INSTALL", None)
            with (
                patch(
                    "scripts.pr_validate.steps.test_env_check.check_test_env",
                    side_effect=[broken, healthy],
                ),
                patch(
                    "scripts.pr_validate.steps.test_env_check.install_test_extras",
                    return_value=(
                        True,
                        "Successfully installed pytest-asyncio-0.21.0\n",
                    ),
                ) as mock_install,
            ):
                step = TestEnvCheckStep()
                result = step.run(fake_ctx)

        assert result.status == "pass"
        assert "installed" in result.summary
        # Install was called once with the ctx's repo root.
        assert mock_install.call_count == 1

    def test_auto_install_runs_but_does_not_fix(self, fake_ctx):
        """Initial probe fails; pip succeeds (exit 0); post-install
        probe STILL fails. This is the "pip silently fell back to
        --user and the plugin is in the wrong site-packages" case —
        the step must report fail with both the initial AND
        post-install missing list so the operator sees the install
        didn't take."""
        broken = TestEnvStatus(
            ok=False,
            missing=("pytest_asyncio",),
            message="missing required test packages: pytest_asyncio",
            interpreter=sys.executable,
        )
        # Same broken status both times — install reported success
        # but the import still doesn't work.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PR_VALIDATE_NO_AUTO_INSTALL", None)
            with (
                patch(
                    "scripts.pr_validate.steps.test_env_check.check_test_env",
                    side_effect=[broken, broken],
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
