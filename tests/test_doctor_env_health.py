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
import os
import plistlib
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


def test_huge_hf_cache_marks_warn(tmp_path):
    """> 100 GB HF cache → WARN with cleanup hint.

    The cache dir is pointed at a fresh ``tmp_path`` that EXISTS so the
    ``cache.exists()`` branch is taken deterministically on every platform
    (a fresh Linux CI runner has no ``~/.cache/huggingface`` yet, which would
    otherwise short-circuit to "HF cache: not present" and skip the WARN). The
    reported size comes from the mocked ``_dir_size_gb``, not host state.
    """
    (tmp_path / "dummy").mkdir()
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
        mock.patch.object(eh, "_hf_cache_dir", return_value=tmp_path),
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
# Section: Agent integrations
# ---------------------------------------------------------------------------


class _Connection:
    def close(self):
        pass


def test_agent_integrations_are_quiet_when_no_client_is_configured(tmp_path):
    section = eh.section_agent_integrations(home=tmp_path)
    assert section.checks[0].status is eh.CheckStatus.OK


def test_agent_integrations_report_config_and_server_reachability(tmp_path):
    claude = tmp_path / ".claude/settings.json"
    claude.parent.mkdir(parents=True)
    claude.write_text('{"env":{"ANTHROPIC_BASE_URL":"http://127.0.0.1:8000"}}')
    cont = tmp_path / ".continue/config.json"
    cont.parent.mkdir(parents=True)
    cont.write_text(
        '{"models":[{"title":"rapid-mlx","provider":"openai",'
        '"apiBase":"http://localhost:8001/v1"}]}'
    )

    def connect(address, *, timeout):
        assert timeout == 0.25
        if address[1] == 8001:
            raise ConnectionRefusedError
        return _Connection()

    section = eh.section_agent_integrations(home=tmp_path, connect=connect)

    assert [check.status for check in section.checks] == [
        eh.CheckStatus.OK,
        eh.CheckStatus.WARN,
    ]
    assert [check.label for check in section.checks] == [
        "Claude Code server is reachable",
        "Continue.dev server is not reachable",
    ]


@pytest.mark.parametrize("claude_config", ["not json", "null", "[]"])
def test_agent_integrations_warn_for_malformed_or_inactive_config(
    tmp_path, claude_config
):
    claude = tmp_path / ".claude/settings.json"
    claude.parent.mkdir(parents=True)
    claude.write_text(claude_config)
    cline = (
        tmp_path
        / "Library/Application Support/Code/User/globalStorage"
        / "saoudrizwan.claude-dev/settings/cline_mcp_settings.json"
    )
    cline.parent.mkdir(parents=True)
    cline.write_text(
        '{"apiProvider":"anthropic","openAiBaseUrl":"http://localhost:8000/v1"}'
    )

    section = eh.section_agent_integrations(
        home=tmp_path,
        connect=lambda *_args, **_kwargs: pytest.fail("must not connect"),
    )

    assert len(section.checks) == 2
    assert all(check.status is eh.CheckStatus.WARN for check in section.checks)


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

    with (
        mock.patch.object(eh, "_safe_version", side_effect=fake_ver),
        mock.patch.object(eh, "_module_available", return_value=True),
    ):
        section = eh.section_optional_packages()

    audio_row = next(c for c in section.checks if "mlx-audio" in c.label)
    assert audio_row.status is eh.CheckStatus.WARN
    assert "pip install" in audio_row.label  # hint preserved


def test_unsupported_mlx_audio_version_marks_warning():
    """A transitive mlx-audio outside Rapid-MLX's pin is not healthy."""

    def fake_ver(dist: str) -> str | None:
        return "0.4.6" if dist == "mlx-audio" else None

    with mock.patch.object(eh, "_safe_version", side_effect=fake_ver):
        section = eh.section_optional_packages()

    audio_row = next(c for c in section.checks if "mlx-audio" in c.label)
    assert audio_row.status is eh.CheckStatus.WARN
    assert "0.4.6" in audio_row.label
    assert "requires mlx-audio>=0.2.9,<0.4.4" in audio_row.label


def test_supported_mlx_audio_version_marks_ok():
    """A version inside the declared audio range remains healthy."""

    def fake_ver(dist: str) -> str | None:
        return "0.4.3" if dist == "mlx-audio" else None

    with (
        mock.patch.object(eh, "_safe_version", side_effect=fake_ver),
        mock.patch.object(eh, "_module_available", return_value=True),
    ):
        section = eh.section_optional_packages()

    audio_row = next(c for c in section.checks if "mlx-audio" in c.label)
    assert audio_row.status is eh.CheckStatus.OK


def test_absent_audio_dependency_stack_marks_warning():
    """An absent audio extra remains one concise optional-package warning."""
    with mock.patch.object(eh, "_safe_version", return_value=None):
        section = eh.section_optional_packages()

    audio_rows = [c for c in section.checks if "mlx-audio" in c.label]
    assert len(audio_rows) == 1
    assert audio_rows[0].status is eh.CheckStatus.WARN


def test_healthy_complete_audio_dependency_stack_marks_ok():
    """When every audio dependency imports and mlx-audio is in range, all
    audio rows are OK."""
    with (
        mock.patch.object(eh, "_safe_version", return_value="0.4.3"),
        mock.patch.object(eh, "_module_available", return_value=True),
    ):
        section = eh.section_optional_packages()

    audio_rows = [
        c
        for c in section.checks
        if "audio" in c.label.lower() or "mlx-audio" in c.label
    ]
    assert audio_rows
    assert all(c.status is eh.CheckStatus.OK for c in audio_rows), [
        (c.label, c.status) for c in audio_rows
    ]


def test_incomplete_audio_dependency_import_stack_marks_warning():
    """A present mlx-audio with a missing/broken audio dependency import is
    WARN, not OK — the audio feature set is not actually usable."""

    def fake_ver(dist: str) -> str | None:
        return "0.4.3" if dist == "mlx-audio" else None

    with (
        mock.patch.object(eh, "_safe_version", side_effect=fake_ver),
        mock.patch.object(
            eh, "_module_available", side_effect=lambda module: module != "f5_tts_mlx"
        ),
    ):
        section = eh.section_optional_packages()

    broken = next(c for c in section.checks if "f5-tts-mlx" in c.label)
    assert broken.status is eh.CheckStatus.WARN


def _stage_sidecar_bundle(tmp_path: Path, *, slot: str = "embedded") -> Path:
    """Build the on-disk shape ``build-sidecar.sh`` produces and return the
    interpreter.

    ``slot`` picks which managed location it sits in — ``embedded`` mirrors
    ``Rapid-MLX Desktop.app/Contents/Resources/rapid-mlx/`` and
    ``runtime-override`` mirrors ``~/Library/Application Support/Rapid/
    runtime-override/rapid-mlx/``. Both share the identical bundle layout.
    """
    if slot == "runtime-override":
        parent = (
            tmp_path / "Library" / "Application Support" / "Rapid" / "runtime-override"
        )
    else:
        parent = tmp_path / "Rapid-MLX Desktop.app" / "Contents" / "Resources"
        parent.parent.mkdir(parents=True)
        with (parent.parent / "Info.plist").open("wb") as handle:
            plistlib.dump({"CFBundleIdentifier": "com.rapidmlx.rapid"}, handle)
    root = parent / "rapid-mlx"
    (root / "site-packages").mkdir(parents=True)
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "rapid-mlx").write_text("#!/bin/sh\n")
    exe = root / "python" / "bin" / "python3.12"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    return exe


def test_bundled_sidecar_detected_from_layout(tmp_path: Path):
    """The bundle is fingerprinted off ``sys.executable``'s directory shape."""
    exe = _stage_sidecar_bundle(tmp_path)
    with mock.patch.object(eh.sys, "executable", str(exe)):
        assert eh._bundled_sidecar_root() == exe.parents[2].resolve()


def test_ordinary_install_is_not_treated_as_bundled_sidecar(tmp_path: Path):
    """A venv interpreter has no ``python/bin`` + ``site-packages`` sibling
    shape, so CLI installs keep the full-extra contract."""
    exe = tmp_path / "venv" / "bin" / "python3.12"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    with mock.patch.object(eh.sys, "executable", str(exe)):
        assert eh._bundled_sidecar_root() is None


def test_custom_install_with_sidecar_shape_is_not_treated_as_desktop(
    tmp_path: Path,
):
    """The three internal bundle paths are user-creatable and therefore do
    not prove that Desktop owns the environment without a managed location."""
    root = tmp_path / "custom-python" / "rapid-mlx"
    (root / "site-packages").mkdir(parents=True)
    (root / "bin").mkdir()
    (root / "bin" / "rapid-mlx").write_text("#!/bin/sh\n")
    exe = root / "python" / "bin" / "python3.12"
    exe.parent.mkdir(parents=True)
    exe.write_text("")

    with mock.patch.object(eh.sys, "executable", str(exe)):
        assert eh._bundled_sidecar_root() is None


def test_unrelated_app_with_sidecar_shape_is_not_treated_as_desktop(
    tmp_path: Path,
):
    """A generic app-bundle suffix is not Rapid-MLX provenance."""
    exe = _stage_sidecar_bundle(tmp_path)
    info = exe.parents[4] / "Info.plist"
    with info.open("wb") as handle:
        plistlib.dump({"CFBundleIdentifier": "com.example.other-app"}, handle)

    with mock.patch.object(eh.sys, "executable", str(exe)):
        assert eh._bundled_sidecar_root() is None


def test_non_mapping_info_plist_is_not_treated_as_desktop(tmp_path: Path):
    """A valid plist may have a non-dictionary root and must not crash doctor."""
    exe = _stage_sidecar_bundle(tmp_path)
    info = exe.parents[4] / "Info.plist"
    with info.open("wb") as handle:
        plistlib.dump(["not", "a", "bundle", "dictionary"], handle)

    with mock.patch.object(eh.sys, "executable", str(exe)):
        assert eh._bundled_sidecar_root() is None


def test_runtime_override_must_belong_to_active_home(tmp_path: Path):
    """A matching suffix under another home is not Desktop provenance."""
    foreign_home = tmp_path / "foreign-home"
    exe = _stage_sidecar_bundle(foreign_home, slot="runtime-override")
    with (
        mock.patch.object(eh.sys, "executable", str(exe)),
        mock.patch.dict(eh.os.environ, {"HOME": str(tmp_path / "active-home")}),
    ):
        assert eh._bundled_sidecar_root() is None


def test_bundled_sidecar_grades_audio_against_desktop_extra(tmp_path: Path):
    """RC 0.12.18: the signed bundle installs ``[audio-desktop]`` (mlx-audio +
    soundfile only). Grading it against the full ``[audio]`` closure reported a
    healthy build as incomplete."""

    def fake_ver(dist: str) -> str | None:
        return "0.4.3" if dist == "mlx-audio" else None

    # Everything outside the audio-desktop extra is absent, exactly like the
    # real bundle.
    desktop_modules = {module for _, module in eh._AUDIO_DESKTOP_IMPORTS}
    exe = _stage_sidecar_bundle(tmp_path)

    with (
        mock.patch.object(eh.sys, "executable", str(exe)),
        mock.patch.object(eh, "_safe_version", side_effect=fake_ver),
        mock.patch.object(
            eh, "_module_available", side_effect=lambda m: m in desktop_modules
        ),
    ):
        section = eh.section_optional_packages()

    audio_row = next(c for c in section.checks if c.label.startswith("mlx-audio"))
    assert audio_row.status is eh.CheckStatus.OK
    assert "incomplete" not in audio_row.label


def test_bundled_sidecar_never_recommends_pip_install(tmp_path: Path):
    """Mutating a managed sidecar's Python environment is never the right
    advice, so no row may hand the user a ``pip install`` line."""
    exe = _stage_sidecar_bundle(tmp_path)
    with (
        mock.patch.object(eh.sys, "executable", str(exe)),
        mock.patch.object(eh, "_safe_version", return_value=None),
    ):
        section = eh.section_optional_packages()

    offenders = [
        (c.label, c.detail)
        for c in section.checks
        if "pip install 'rapid-mlx[" in c.label + c.detail
    ]
    assert not offenders, offenders


def test_bundle_does_not_ask_user_to_reinstall_for_an_extra_it_never_ships(
    tmp_path: Path,
):
    """mlx-embeddings is outside the desktop product surface — ``build-
    sidecar.sh`` never installs it. Flagging it ⚠ with a "reinstall" hint
    reported a healthy bundle as broken and the warning would survive every
    reinstall."""
    exe = _stage_sidecar_bundle(tmp_path)
    with (
        mock.patch.object(eh.sys, "executable", str(exe)),
        mock.patch.object(eh, "_safe_version", return_value=None),
    ):
        section = eh.section_optional_packages()

    row = next(c for c in section.checks if c.label.startswith("mlx-embeddings"))
    assert row.status is eh.CheckStatus.OK
    assert "not bundled" in row.label
    assert "reinstall" not in row.label.lower()


def test_cli_install_still_warns_about_missing_embeddings(tmp_path: Path):
    """The exclusion is bundle-only: an ordinary pip install that lacks
    mlx-embeddings must still get the ⚠ + install hint."""
    exe = tmp_path / "venv" / "bin" / "python3.12"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    with (
        mock.patch.object(eh.sys, "executable", str(exe)),
        mock.patch.object(eh, "_safe_version", return_value=None),
    ):
        section = eh.section_optional_packages()

    row = next(c for c in section.checks if c.label.startswith("mlx-embeddings"))
    assert row.status is eh.CheckStatus.WARN
    assert "pip install 'rapid-mlx[embeddings]'" in row.label


def test_embedded_bundle_hint_names_the_shipping_app(tmp_path: Path):
    """``Rapid.app`` is a legacy name that release verification rejects; the
    shipping product is ``Rapid-MLX Desktop.app``."""
    exe = _stage_sidecar_bundle(tmp_path, slot="embedded")
    with mock.patch.object(eh.sys, "executable", str(exe)):
        hint = eh._sidecar_repair_hint(eh._bundled_sidecar_root())

    assert "Rapid-MLX Desktop.app" in hint


def test_runtime_override_is_not_told_to_reinstall_the_app(tmp_path: Path):
    """The runtime override lives outside the app bundle, survives desktop
    upgrades, and the bootstrapper skips its download while the cache exists —
    so "reinstall the .app" alone repairs nothing."""
    exe = _stage_sidecar_bundle(tmp_path, slot="runtime-override")
    with (
        mock.patch.object(eh.sys, "executable", str(exe)),
        mock.patch.dict(eh.os.environ, {"HOME": str(tmp_path)}),
    ):
        root = eh._bundled_sidecar_root()
        hint = eh._sidecar_repair_hint(root)

    assert "outside the app bundle" in hint
    # The step has to be actionable, so it names the exact directory to remove.
    assert str(root) in hint


def test_runtime_override_hint_avoids_the_nonexistent_update_entry_point(
    tmp_path: Path,
):
    """Settings → check for updates hands off to Sparkle, which updates the app
    bundle; ``UpdateChecker`` explicitly does not act on the manifest's
    ``sidecar_*`` fields. Pointing users there for a broken runtime override is
    a dead-end recovery flow."""
    exe = _stage_sidecar_bundle(tmp_path, slot="runtime-override")
    with (
        mock.patch.object(eh.sys, "executable", str(exe)),
        mock.patch.dict(eh.os.environ, {"HOME": str(tmp_path)}),
    ):
        hint = eh._sidecar_repair_hint(eh._bundled_sidecar_root())

    assert "check for updates" not in hint.lower()


def test_runtime_override_hint_does_not_promise_an_automatic_reinstall(
    tmp_path: Path,
):
    """Nothing re-downloads a sidecar today: the bootstrapper module is not in
    the source tree and the release workflow builds only the full DMG. A slim
    install that merely deletes the override lands on the missing-runtime
    overlay, whose only actions are Recheck and an *app* update download. So
    the hint must order "install the app that ships a sidecar" BEFORE the
    removal, and must not claim relaunching reinstalls the runtime."""
    exe = _stage_sidecar_bundle(tmp_path, slot="runtime-override")
    with (
        mock.patch.object(eh.sys, "executable", str(exe)),
        mock.patch.dict(eh.os.environ, {"HOME": str(tmp_path)}),
    ):
        root = eh._bundled_sidecar_root()
        hint = eh._sidecar_repair_hint(root)

    # No promise of self-repair.
    assert "reinstalls it" not in hint
    assert "then reinstalls" not in hint
    # Install-first ordering: the app install must precede the removal step,
    # otherwise a slim user is stranded with no sidecar at all.
    assert hint.index("Rapid-MLX Desktop.app") < hint.index(f"remove {root}")


def test_runtime_override_broken_audio_row_uses_the_runtime_hint(tmp_path: Path):
    """End-to-end: a genuinely broken override bundle is still ⚠, and the
    remediation the user reads is the runtime one, not the app one."""

    def fake_ver(dist: str) -> str | None:
        return "0.4.3" if dist == "mlx-audio" else None

    exe = _stage_sidecar_bundle(tmp_path, slot="runtime-override")
    with (
        mock.patch.object(eh.sys, "executable", str(exe)),
        mock.patch.dict(eh.os.environ, {"HOME": str(tmp_path)}),
        mock.patch.object(eh, "_safe_version", side_effect=fake_ver),
        mock.patch.object(
            eh, "_module_available", side_effect=lambda m: m != "soundfile"
        ),
    ):
        section = eh.section_optional_packages()

    broken = next(c for c in section.checks if "soundfile" in c.label)
    assert broken.status is eh.CheckStatus.WARN
    assert "outside the app bundle" in broken.label


def test_bundled_sidecar_still_flags_a_genuinely_broken_audio_install(
    tmp_path: Path,
):
    """Narrowing the contract must not make the bundle unfalsifiable — a
    missing soundfile is still inside the audio-desktop contract."""

    def fake_ver(dist: str) -> str | None:
        return "0.4.3" if dist == "mlx-audio" else None

    exe = _stage_sidecar_bundle(tmp_path)
    with (
        mock.patch.object(eh.sys, "executable", str(exe)),
        mock.patch.object(eh, "_safe_version", side_effect=fake_ver),
        mock.patch.object(
            eh, "_module_available", side_effect=lambda m: m != "soundfile"
        ),
    ):
        section = eh.section_optional_packages()

    broken = next(c for c in section.checks if "soundfile" in c.label)
    assert broken.status is eh.CheckStatus.WARN
    assert "reinstall Rapid-MLX Desktop.app" in broken.label


# ---------------------------------------------------------------------------
# Section: HuggingFace cache
# ---------------------------------------------------------------------------


def test_hf_cache_writable_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A writable cache dir produces OK; a missing one produces WARN."""
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
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
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
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


def test_hf_cache_resolves_hf_hub_cache_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """``$HF_HUB_CACHE`` wins over ``$HF_HOME`` over the default — matches
    huggingface_hub itself. Codex review round 1 caught the previous
    revision returning ``~/.cache/huggingface`` instead of the actual
    ``~/.cache/huggingface/hub`` subdir where downloads land."""
    hub = tmp_path / "external_ssd_hub"
    hub.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    # HF_HOME is *also* set, to a different (writable) path — HF_HUB_CACHE
    # must take precedence.
    other_home = tmp_path / "wrong_home"
    other_home.mkdir()
    (other_home / "hub").mkdir()
    monkeypatch.setenv("HF_HOME", str(other_home))

    resolved = eh._hf_cache_dir()
    assert resolved == hub, (
        f"HF_HUB_CACHE should win over HF_HOME; resolved {resolved} != {hub}"
    )


def test_hf_cache_default_includes_hub_subdir(monkeypatch: pytest.MonkeyPatch):
    """Default (no env vars) resolution must include the trailing ``hub``
    segment — that's where huggingface_hub actually writes downloads."""
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    resolved = eh._hf_cache_dir()
    assert resolved.name == "hub", (
        f"default cache dir should end in .../huggingface/hub; got {resolved}"
    )


def test_dir_size_walk_aborts_on_budget(tmp_path: Path):
    """A walk that exceeds ``budget_s`` returns None rather than running
    indefinitely — keeps doctor under its 5 s wall-clock contract on
    network-mounted caches. Codex review round 1 flagged the unbounded
    walk as a contract violation."""
    # Populate a tiny tree so there's something to walk.
    for i in range(3):
        (tmp_path / f"f{i}.bin").write_bytes(b"\0" * 1024)
    # budget_s=0 forces an immediate abort on the first deadline check.
    result = eh._dir_size_gb(tmp_path, budget_s=0.0)
    assert result is None, (
        f"zero-budget walk should return None, got {result!r} — "
        "the deadline guard isn't firing"
    )


def test_dir_size_walk_aborts_inside_flat_directory(tmp_path: Path):
    """Flat directory with many files must respect the per-file deadline,
    not just the per-directory one. HF cache's ``blobs/`` subdir is the
    real-world example: thousands of files in a single dir; a per-dir
    deadline check would let a single cold-cache stat() storm blow past
    the budget. Codex review round 2 caught this; the per-file check
    fixes it."""
    # 500 files in one flat dir.
    for i in range(500):
        (tmp_path / f"f{i:04d}.bin").write_bytes(b"\0")
    # budget_s=0 must abort *before* iterating all 500 files.
    import time as _time

    t0 = _time.monotonic()
    result = eh._dir_size_gb(tmp_path, budget_s=0.0)
    elapsed = _time.monotonic() - t0
    assert result is None, "zero-budget flat-dir walk should return None"
    # Tight ceiling: must abort within a small multiple of one file's worth.
    assert elapsed < 0.5, (
        f"flat-dir walk took {elapsed:.3f}s with budget_s=0 — "
        "per-file deadline check isn't firing"
    )


def test_hf_cache_non_directory_marks_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """``HF_HUB_CACHE`` pointing at a writable regular file must FAIL —
    ``os.access`` returns True for writable files too, so the previous
    revision would have shipped ✓ here. Codex review round 2 fixed."""
    file_target = tmp_path / "i_am_a_file_not_a_dir"
    file_target.write_text("oops")
    monkeypatch.setenv("HF_HUB_CACHE", str(file_target))

    section = eh.section_hf_cache()
    first = section.checks[0]
    assert first.status is eh.CheckStatus.FAIL
    assert "NOT a directory" in first.label


def test_hf_cache_missing_with_readonly_parent_marks_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Missing cache + readonly nearest-existing-parent → FAIL, not WARN.
    Codex review round 2: previous unconditional WARN exited 0 even when
    the first download was guaranteed to fail."""
    readonly_root = tmp_path / "readonly"
    readonly_root.mkdir()
    target = readonly_root / "nonexistent_hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(target))

    # Mock os.access so the test doesn't depend on chmod semantics across
    # CI runners (where the actual chmod may not stick in tmpfs).
    real_access = eh.os.access

    def fake_access(p, mode):
        if str(p) == str(readonly_root):
            return False  # parent isn't writable
        return real_access(p, mode)

    with mock.patch.object(eh.os, "access", side_effect=fake_access):
        section = eh.section_hf_cache()
    first = section.checks[0]
    assert first.status is eh.CheckStatus.FAIL
    assert "parent" in first.label and "NOT writable" in first.label


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


def test_running_exe_mismatch_with_path_warns(tmp_path: Path):
    """Issue #2352: PATH resolves a different rapid-mlx than the one actually
    running this doctor (e.g. a venv/bin that precedes a global install) →
    WARN with BOTH paths and an actionable fix, rather than a silently-green
    PATH row that points troubleshooting at the wrong install."""
    section = eh.section_shell_integration(
        which=lambda name: (
            "/Users/x/.local/bin/rapid-mlx" if name == "rapid-mlx" else None
        ),
        rcs=[tmp_path / "missing.zshrc"],
        running_exe="/private/tmp/env/bin/rapid-mlx",
    )
    mismatch = next(c for c in section.checks if "differs from the $PATH" in c.label)
    assert mismatch.status is eh.CheckStatus.WARN
    assert "/private/tmp/env/bin/rapid-mlx" in mismatch.label
    assert "/Users/x/.local/bin/rapid-mlx" in mismatch.label
    assert "activate this environment" in mismatch.label
    # The PATH row itself is still OK (it IS on PATH); only the divergence warns.
    path_row = next(c for c in section.checks if "in $PATH" in c.label)
    assert path_row.status is eh.CheckStatus.OK


def test_running_exe_matching_path_has_no_mismatch_warn(tmp_path: Path):
    """When the running CLI and the PATH-resolved CLI agree, no mismatch warn."""
    section = eh.section_shell_integration(
        which=lambda name: (
            "/private/tmp/env/bin/rapid-mlx" if name == "rapid-mlx" else None
        ),
        rcs=[tmp_path / "missing.zshrc"],
        running_exe="/private/tmp/env/bin/rapid-mlx",
    )
    assert not any("differs from the $PATH" in c.label for c in section.checks)


def test_running_cli_exe_prefers_argv0(monkeypatch):
    """``_running_cli_exe`` uses sys.argv[0] (the console-script path) when it
    resolves to a `rapid-mlx` entry point."""
    monkeypatch.setattr(eh.sys, "argv", ["/opt/venv/bin/rapid-mlx", "doctor"])
    assert eh._running_cli_exe() == "/opt/venv/bin/rapid-mlx"


def test_running_cli_exe_falls_back_to_executable_sibling(monkeypatch):
    """When sys.argv[0] isn't a `rapid-mlx` script (e.g. python -m), fall back
    to the `rapid-mlx` console script beside sys.executable."""
    monkeypatch.setattr(eh.sys, "argv", ["-m", "rapid_mlx.doctor"])
    monkeypatch.setattr(eh.sys, "executable", "/opt/venv/bin/python")
    monkeypatch.setattr(eh.os.path, "exists", lambda p: p == "/opt/venv/bin/rapid-mlx")
    assert eh._running_cli_exe() == "/opt/venv/bin/rapid-mlx"


def test_running_cli_exe_uses_executable_when_named_rapid_mlx(monkeypatch):
    """If sys.executable itself is `rapid-mlx` (rare direct exec), return it."""
    monkeypatch.setattr(eh.sys, "argv", ["-m", "rapid_mlx.doctor"])
    monkeypatch.setattr(eh.sys, "executable", "/opt/venv/bin/rapid-mlx")
    assert eh._running_cli_exe() == "/opt/venv/bin/rapid-mlx"


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


def test_run_all_returns_all_sections():
    """run_all() must emit exactly the sections the spec mandates, in the spec
    order. Test pins the order so future drift is loud."""
    report = eh.run_all()
    titles = [s.title for s in report.sections]
    expected = [
        "System",
        "Python",
        "Required Packages",
        "Updates",
        "Optional Packages",
        "HuggingFace Cache",
        "Network",
        "Shell Integration",
        "Optional Tools",
        "Agent Integrations",
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


# ---------------------------------------------------------------------------
# Section: Updates (version freshness)
# ---------------------------------------------------------------------------


def test_updates_up_to_date_marks_ok():
    section = eh.section_updates(
        installed=lambda: "0.10.15",
        fetch_latest=lambda: "0.10.15",
    )
    row = section.checks[0]
    assert row.status is eh.CheckStatus.OK
    assert "up to date" in row.label


def test_update_available_marks_warn_with_upgrade_command():
    info = mock.Mock(upgrade_command="brew upgrade rapid-mlx", method="brew")
    section = eh.section_updates(
        installed=lambda: "0.10.12",
        fetch_latest=lambda: "0.10.15",
        install_info=info,
    )
    row = section.checks[0]
    assert row.status is eh.CheckStatus.WARN
    assert "update available: 0.10.15" in row.label
    assert "brew upgrade rapid-mlx" in row.label


def test_updates_offline_marks_warn_never_fail():
    section = eh.section_updates(
        installed=lambda: "0.10.15",
        fetch_latest=lambda: None,
    )
    row = section.checks[0]
    assert row.status is eh.CheckStatus.WARN
    # Air-gapped doctor must never escalate the update check to a hard fail.
    assert all(c.status is not eh.CheckStatus.FAIL for c in section.checks)


def test_updates_unknown_installed_version_marks_warn():
    section = eh.section_updates(
        installed=lambda: None,
        fetch_latest=lambda: "0.10.15",
    )
    row = section.checks[0]
    assert row.status is eh.CheckStatus.WARN
    assert "unknown" in row.label


@pytest.mark.parametrize(
    "installed,latest",
    [
        ("0.10", "0.10.15"),  # installed unparseable (too few components)
        ("0.10.15", "0.11.0rc1"),  # latest unparseable (rc suffix)
        ("0.10.15.dev3", "0.10.16.dev1"),  # both dev builds
    ],
)
def test_updates_unparseable_version_marks_warn_not_up_to_date(installed, latest):
    """A version ``_parse_version`` can't order (dev/rc/short) must NOT
    silently green-light "up to date" — that falsely reassures a user who
    may well be behind. It downgrades to ⚠ like every other uncertain
    branch, and never to a hard fail."""
    section = eh.section_updates(
        installed=lambda: installed,
        fetch_latest=lambda: latest,
    )
    row = section.checks[0]
    assert row.status is eh.CheckStatus.WARN
    assert "up to date" not in row.label
    assert all(c.status is not eh.CheckStatus.FAIL for c in section.checks)


# ---------------------------------------------------------------------------
# Section: Shell Integration — shadowed / duplicate installs
# ---------------------------------------------------------------------------


def test_shadowed_install_marks_warn(tmp_path: Path):
    section = eh.section_shell_integration(
        which=lambda name: (
            "/opt/homebrew/bin/rapid-mlx" if name == "rapid-mlx" else None
        ),
        rcs=[tmp_path / "missing.zshrc"],
        find_all=lambda: [
            "/opt/homebrew/bin/rapid-mlx",
            "/Users/x/.local/bin/rapid-mlx",
        ],
    )
    shadow_row = next(c for c in section.checks if "shadowed" in c.label)
    assert shadow_row.status is eh.CheckStatus.WARN
    assert "2 places" in shadow_row.label
    assert ".local/bin/rapid-mlx" in shadow_row.label


def test_single_install_has_no_shadow_warn(tmp_path: Path):
    section = eh.section_shell_integration(
        which=lambda name: (
            "/opt/homebrew/bin/rapid-mlx" if name == "rapid-mlx" else None
        ),
        rcs=[tmp_path / "missing.zshrc"],
        find_all=lambda: ["/opt/homebrew/bin/rapid-mlx"],
    )
    assert not any("shadowed" in c.label for c in section.checks)


def test_rapid_mlx_on_path_dedupes_by_resolved_target(tmp_path: Path):
    """A symlink to the same binary is one install; a distinct binary is two."""
    d1, d2, d3 = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    for d in (d1, d2, d3):
        d.mkdir()
    real = d1 / "rapid-mlx"
    real.write_text("#!/bin/sh\n")
    real.chmod(0o755)
    (d2 / "rapid-mlx").symlink_to(real)  # same target → deduped
    other = d3 / "rapid-mlx"
    other.write_text("#!/bin/sh\n")
    other.chmod(0o755)

    path_env = os.pathsep.join(str(d) for d in (d1, d2, d3))
    found = eh._rapid_mlx_on_path(path_env=path_env)
    assert found == [str(real), str(other)]


def test_rapid_mlx_on_path_empty_component_is_cwd(tmp_path: Path, monkeypatch):
    """An empty PATH component means the current directory (shutil.which
    semantics). A ``./rapid-mlx`` shadow must be surfaced, not skipped."""
    cli = tmp_path / "rapid-mlx"
    cli.write_text("#!/bin/sh\n")
    cli.chmod(0o755)
    monkeypatch.chdir(tmp_path)

    # Leading empty component ("" before the sep) resolves to cwd.
    found = eh._rapid_mlx_on_path(path_env=os.pathsep + "/nonexistent-doctor-dir")
    resolved = [os.path.realpath(p) for p in found]
    assert os.path.realpath(str(cli)) in resolved
