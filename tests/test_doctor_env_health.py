# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the env-health probes in ``vllm_mlx.doctor.env_health``.

These tests are the safety net for the user-facing ``rapid-mlx doctor``
contract:

* Apple-Silicon detection works on macOS-arm64 and falls back gracefully.
* Required-package matrix flags missing packages as ✗ (fail), missing
  optional packages as ⚠ (warn).
* HF cache writability is correctly probed.
* Network timeout produces a ⚠, never a ✗ — air-gapped CI must not break.
* Exit code is 0 with only ✓/⚠; 1 if any ✗.

Every test is dependency-injected (no real subprocess / network / disk
mutation) so the suite runs identically on every Python and every OS.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest import mock

import pytest

from vllm_mlx.doctor import env_health as eh

# ---------------------------------------------------------------------------
# Section: System
# ---------------------------------------------------------------------------


def test_apple_silicon_detected():
    """On a real arm64 macOS box the System section should report the chip."""
    with (
        mock.patch.object(eh.platform, "system", return_value="Darwin"),
        mock.patch.object(eh.platform, "machine", return_value="arm64"),
        mock.patch.object(
            eh.platform, "mac_ver", return_value=("14.3", ("", "", ""), "arm64")
        ),
        mock.patch.object(eh.platform, "release", return_value="23.3.0"),
        mock.patch.object(
            eh, "_detect_apple_silicon", return_value=("Apple M3 Pro", 36)
        ),
        mock.patch.object(eh, "_disk_free_gb", return_value=162.0),
        mock.patch.object(eh, "_dir_size_gb", return_value=12.0),
    ):
        section = eh.section_system()

    labels = [c.label for c in section.checks]
    assert any("Apple Silicon" in label and "M3 Pro" in label for label in labels), (
        labels
    )
    assert any("36 GB" in label for label in labels), labels
    assert all(
        c.status is eh.CheckStatus.OK
        for c in section.checks
        if "Apple Silicon" in c.label
    )


def test_apple_silicon_warn_on_non_arm64_mac():
    """Intel Mac should produce a WARN row, not a FAIL."""
    with (
        mock.patch.object(eh.platform, "system", return_value="Darwin"),
        mock.patch.object(eh.platform, "machine", return_value="x86_64"),
        mock.patch.object(
            eh.platform, "mac_ver", return_value=("14.3", ("", "", ""), "x86_64")
        ),
        mock.patch.object(eh.platform, "release", return_value="23.3.0"),
        mock.patch.object(eh, "_disk_free_gb", return_value=200.0),
        mock.patch.object(eh, "_dir_size_gb", return_value=None),
    ):
        section = eh.section_system()

    assert any(
        c.status is eh.CheckStatus.WARN and "Non-Apple-Silicon" in c.label
        for c in section.checks
    )


def test_low_disk_marks_fail():
    """< 5 GB free disk is a hard FAIL (next download will fail)."""
    with (
        mock.patch.object(eh.platform, "system", return_value="Linux"),
        mock.patch.object(eh.platform, "machine", return_value="x86_64"),
        mock.patch.object(eh.platform, "mac_ver", return_value=("", ("", "", ""), "")),
        mock.patch.object(eh.platform, "release", return_value="6.5.0"),
        mock.patch.object(eh, "_disk_free_gb", return_value=2.0),
        mock.patch.object(eh, "_dir_size_gb", return_value=None),
    ):
        section = eh.section_system()

    fail_rows = [c for c in section.checks if c.status is eh.CheckStatus.FAIL]
    assert any("Free disk" in c.label for c in fail_rows), [
        c.label for c in section.checks
    ]


def test_huge_hf_cache_marks_warn():
    """> 100 GB HF cache → WARN with cleanup hint."""
    with (
        mock.patch.object(eh.platform, "system", return_value="Darwin"),
        mock.patch.object(eh.platform, "machine", return_value="arm64"),
        mock.patch.object(
            eh.platform, "mac_ver", return_value=("14.3", ("", "", ""), "arm64")
        ),
        mock.patch.object(eh.platform, "release", return_value="23.3.0"),
        mock.patch.object(eh, "_detect_apple_silicon", return_value=("Apple M3", 36)),
        mock.patch.object(eh, "_disk_free_gb", return_value=200.0),
        mock.patch.object(eh, "_dir_size_gb", return_value=246.0),
    ):
        section = eh.section_system()

    warn_rows = [c for c in section.checks if c.status is eh.CheckStatus.WARN]
    assert any("HF cache size: 246 GB" in c.label for c in warn_rows)
    assert any("rapid-mlx rm" in c.label for c in warn_rows)


# ---------------------------------------------------------------------------
# Section: Python
# ---------------------------------------------------------------------------


def test_python_version_reported():
    """The Python section always reports the running interpreter version."""
    section = eh.section_python()
    py_row = section.checks[0]
    assert py_row.label.startswith("Python ")
    # Anything >= 3.10 is OK, < 3.10 is FAIL — the running interpreter is
    # whatever pytest is using, which is always supported by our matrix.
    assert py_row.status is eh.CheckStatus.OK


def test_install_location_reported():
    section = eh.section_python()
    # Second row is always the install-location classifier.
    loc_row = section.checks[1]
    assert "Install location" in loc_row.label


# ---------------------------------------------------------------------------
# Section: Required + optional packages
# ---------------------------------------------------------------------------


def test_required_packages_all_present_marks_ok():
    """When every required dist is installed, every row is OK."""
    fake_ver = lambda dist: "9.9.9"  # noqa: E731
    with mock.patch.object(eh, "_safe_version", side_effect=fake_ver):
        section = eh.section_required_packages()
    assert all(c.status is eh.CheckStatus.OK for c in section.checks)
    # Each row carries the fake version string.
    assert all("9.9.9" in c.label for c in section.checks)


def test_required_package_missing_marks_fail():
    def fake_ver(dist: str) -> str | None:
        return None if dist == "transformers" else "1.2.3"

    with mock.patch.object(eh, "_safe_version", side_effect=fake_ver):
        section = eh.section_required_packages()

    transformers_row = next(c for c in section.checks if "transformers" in c.label)
    assert transformers_row.status is eh.CheckStatus.FAIL
    assert "not installed" in transformers_row.label


def test_missing_optional_package_marks_warning():
    """A missing optional package is ⚠ with an install hint, never ✗."""

    def fake_ver(dist: str) -> str | None:
        # mlx-audio missing; the rest present.
        return None if dist == "mlx-audio" else "1.0.0"

    with mock.patch.object(eh, "_safe_version", side_effect=fake_ver):
        section = eh.section_optional_packages()

    audio_row = next(c for c in section.checks if "mlx-audio" in c.label)
    assert audio_row.status is eh.CheckStatus.WARN
    assert "pip install" in audio_row.label  # hint preserved


# ---------------------------------------------------------------------------
# Section: HuggingFace cache
# ---------------------------------------------------------------------------


def test_hf_cache_writable_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A writable cache dir produces OK; a missing one produces WARN."""
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    (tmp_path / "hub").mkdir()
    section = eh.section_hf_cache()
    writable_row = section.checks[0]
    assert writable_row.status is eh.CheckStatus.OK
    assert "writable" in writable_row.label

    # Now point HF_HOME at a non-existent dir; first row should WARN.
    monkeypatch.setenv("HF_HOME", str(tmp_path / "doesnotexist"))
    section = eh.section_hf_cache()
    missing_row = section.checks[0]
    assert missing_row.status is eh.CheckStatus.WARN
    assert "does not exist" in missing_row.label


def test_hf_cache_readonly_marks_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A readonly cache dir is a hard FAIL — downloads can't proceed."""
    cache_root = tmp_path / "ro"
    cache_root.mkdir()
    (cache_root / "hub").mkdir()
    monkeypatch.setenv("HF_HOME", str(cache_root))
    # mock os.access so the test doesn't depend on chmod semantics across CI.
    with mock.patch.object(eh.os, "access", return_value=False):
        section = eh.section_hf_cache()
    first = section.checks[0]
    assert first.status is eh.CheckStatus.FAIL
    assert "NOT writable" in first.label


# ---------------------------------------------------------------------------
# Section: Network
# ---------------------------------------------------------------------------


def test_network_probe_timeout_is_warning_not_failure():
    """An unreachable huggingface.co must produce ⚠, never ✗.

    This is the contract that lets air-gapped CI runners still get a
    green ``rapid-mlx doctor`` (the spec is explicit about this).
    """

    def fake_probe() -> tuple[eh.CheckStatus, str]:
        return eh.CheckStatus.WARN, "TimeoutError: timed out after 2.0s"

    section = eh.section_network(probe=fake_probe)
    hf_row = next(c for c in section.checks if "huggingface.co" in c.label)
    assert hf_row.status is eh.CheckStatus.WARN
    # Spec: WARN never contributes to exit code, only FAIL does.
    assert hf_row.status is not eh.CheckStatus.FAIL


def test_network_probe_ok():
    def fake_probe() -> tuple[eh.CheckStatus, str]:
        return eh.CheckStatus.OK, "HTTP 200"

    section = eh.section_network(probe=fake_probe)
    hf_row = next(c for c in section.checks if "huggingface.co" in c.label)
    assert hf_row.status is eh.CheckStatus.OK


# ---------------------------------------------------------------------------
# Section: Shell integration
# ---------------------------------------------------------------------------


def test_argcomplete_not_in_rc_marks_warning(tmp_path: Path):
    """No rc file contains the argcomplete hook → WARN with activation hint."""
    fake_rc = tmp_path / ".zshrc"
    fake_rc.write_text("export PATH=$PATH:~/bin\n")  # no argcomplete hook

    section = eh.section_shell_integration(
        which=lambda name: "/usr/local/bin/rapid-mlx" if name == "rapid-mlx" else None,
        rcs=[fake_rc],
    )
    argc_row = next(c for c in section.checks if "argcomplete" in c.label)
    assert argc_row.status is eh.CheckStatus.WARN
    assert "register-python-argcomplete rapid-mlx" in argc_row.label


def test_argcomplete_present_marks_ok(tmp_path: Path):
    fake_rc = tmp_path / ".zshrc"
    fake_rc.write_text('eval "$(register-python-argcomplete rapid-mlx)"\n')

    section = eh.section_shell_integration(
        which=lambda name: "/usr/local/bin/rapid-mlx" if name == "rapid-mlx" else None,
        rcs=[fake_rc],
    )
    argc_row = next(c for c in section.checks if "argcomplete" in c.label)
    assert argc_row.status is eh.CheckStatus.OK


def test_rapid_mlx_not_on_path_marks_fail(tmp_path: Path):
    section = eh.section_shell_integration(
        which=lambda name: None,
        rcs=[tmp_path / "missing.zshrc"],
    )
    path_row = section.checks[0]
    assert path_row.status is eh.CheckStatus.FAIL
    assert "NOT on $PATH" in path_row.label


# ---------------------------------------------------------------------------
# Exit-code aggregation
# ---------------------------------------------------------------------------


def _make_report(*statuses: eh.CheckStatus) -> eh.Report:
    report = eh.Report()
    section = eh.Section("test")
    for i, st in enumerate(statuses):
        section.add(f"check-{i}", st)
    report.sections.append(section)
    return report


def test_overall_exit_code_zero_when_no_issues():
    report = _make_report(eh.CheckStatus.OK, eh.CheckStatus.OK, eh.CheckStatus.WARN)
    assert report.exit_code == 0
    assert report.n_warn == 1
    assert report.n_fail == 0


def test_overall_exit_code_one_when_any_issue():
    report = _make_report(eh.CheckStatus.OK, eh.CheckStatus.WARN, eh.CheckStatus.FAIL)
    assert report.exit_code == 1
    assert report.n_fail == 1


def test_overall_exit_code_zero_with_only_warnings():
    report = _make_report(eh.CheckStatus.WARN, eh.CheckStatus.WARN)
    # Spec rule: warnings never affect exit code.
    assert report.exit_code == 0


# ---------------------------------------------------------------------------
# Top-level run_all + render smoke
# ---------------------------------------------------------------------------


def test_run_all_returns_all_eight_sections():
    """run_all() must emit exactly the eight sections the spec mandates,
    in the spec order. Test pins the order so future drift is loud."""
    report = eh.run_all()
    titles = [s.title for s in report.sections]
    expected = [
        "System",
        "Python",
        "Required Packages",
        "Optional Packages",
        "HuggingFace Cache",
        "Network",
        "Shell Integration",
        "Optional Tools",
    ]
    assert titles == expected, (
        f"sections drifted from spec order. got {titles}, expected {expected}"
    )


def test_run_all_crashing_probe_does_not_abort_report(monkeypatch):
    """If a section builder crashes, run_all() records it as a single ✗
    row but keeps going. A buggy probe must not blank the whole report."""

    def boom() -> eh.Section:
        raise RuntimeError("synthetic")

    # Patch the second builder so we can confirm earlier sections still rendered.
    builders = list(eh._SECTION_BUILDERS)
    builders[1] = boom
    monkeypatch.setattr(eh, "_SECTION_BUILDERS", tuple(builders))

    report = eh.run_all()
    assert len(report.sections) == len(builders)
    crashed = report.sections[1]
    assert any("probe crashed" in c.label for c in crashed.checks)
    assert any(c.status is eh.CheckStatus.FAIL for c in crashed.checks)


def test_render_outputs_section_headers(capsys):
    """Sanity-check the renderer: every section title appears in the output."""
    report = _make_report(eh.CheckStatus.OK)
    report.sections[0].title = "MySection"

    import io

    from vllm_mlx.doctor.cli import render

    buf = io.StringIO()
    render(report, stream=buf)
    out = buf.getvalue()
    assert "MySection" in out
    assert "Summary:" in out
    assert "Rapid-MLX Doctor" in out


def test_render_verbose_includes_detail():
    from vllm_mlx.doctor.cli import render

    report = eh.Report()
    section = eh.Section("X")
    section.add("the-label", eh.CheckStatus.OK, detail="the-detail-string")
    report.sections.append(section)

    import io

    buf = io.StringIO()
    render(report, verbose=True, stream=buf)
    assert "the-detail-string" in buf.getvalue()

    buf2 = io.StringIO()
    render(report, verbose=False, stream=buf2)
    assert "the-detail-string" not in buf2.getvalue()


# ---------------------------------------------------------------------------
# Importable sanity — guarantees the module is wired into the public surface.
# ---------------------------------------------------------------------------


def test_env_health_public_exports():
    pkg = importlib.import_module("vllm_mlx.doctor")
    for name in ("run_all", "Report", "Section", "Check", "CheckStatus"):
        assert hasattr(pkg, name), f"vllm_mlx.doctor missing {name}"
