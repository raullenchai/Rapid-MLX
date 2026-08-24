"""The engine version and the desktop app version are one number.

``test_repo_is_in_sync`` is the guard itself — it goes red the moment
``pyproject.toml`` and ``apps/rapid-mac/Resources/Info.plist`` disagree.
Everything else exists so that test cannot pass for the wrong reason: a
checker that returns success on a missing file, an unreadable plist, or a
version it never actually compared would keep CI green through exactly
the drift it was written to stop.

Pure stdlib — no mlx, no macOS — so this runs on the Linux CI runner.
"""

from __future__ import annotations

import json
import plistlib
from pathlib import Path

import pytest

from scripts.check_version_sync import (
    VersionSyncError,
    app_version,
    check,
    engine_version,
    main,
)

PLIST_TEMPLATE = {
    "CFBundleName": "Rapid-MLX",
    "CFBundleShortVersionString": "1.2.3",
    "CFBundleVersion": "150",
}


def write_plist(path: Path, **overrides: object) -> Path:
    data = dict(PLIST_TEMPLATE)
    data.update(overrides)
    data = {k: v for k, v in data.items() if v is not None}
    with path.open("wb") as fh:
        plistlib.dump(data, fh)
    return path


def write_pyproject(path: Path, version: str | None = "1.2.3") -> Path:
    body = '[project]\nname = "rapid-mlx"\n'
    if version is not None:
        # json.dumps, not an f-string interpolation: a value containing a
        # newline or a quote would otherwise produce INVALID TOML, and the
        # test would pass on a parse error instead of on the check it is
        # supposed to exercise. TOML basic strings use JSON's escapes.
        body += f"version = {json.dumps(version)}\n"
    path.write_text(body, encoding="utf-8")
    return path


# --- the guard ------------------------------------------------------


def test_repo_is_in_sync():
    """The real files agree.

    When this fails, set BOTH to the same X.Y.Z in one PR — and only
    ever upward: a ``rapid-mac-vX.Y.Z`` tag already exists for every
    version the app has shipped, and the in-app updater orders these
    values.
    """
    engine, app = check()
    assert engine == app


# --- the guard is not vacuous ---------------------------------------


def test_mismatch_is_rejected(tmp_path):
    """The exact 2026-08-07 drift: engine 0.12.5, app 0.12.6."""
    pyproject = write_pyproject(tmp_path / "pyproject.toml", "0.12.5")
    plist = write_plist(tmp_path / "Info.plist", CFBundleShortVersionString="0.12.6")
    with pytest.raises(VersionSyncError) as exc:
        check(pyproject, plist)
    # Both numbers must appear, or the message can't be acted on.
    assert "0.12.5" in str(exc.value)
    assert "0.12.6" in str(exc.value)


def test_match_is_accepted(tmp_path):
    pyproject = write_pyproject(tmp_path / "pyproject.toml", "0.12.7")
    plist = write_plist(tmp_path / "Info.plist", CFBundleShortVersionString="0.12.7")
    assert check(pyproject, plist) == ("0.12.7", "0.12.7")


def test_matching_release_candidate_is_accepted(tmp_path):
    pyproject = write_pyproject(tmp_path / "pyproject.toml", "0.13.0-rc1")
    plist = write_plist(
        tmp_path / "Info.plist", CFBundleShortVersionString="0.13.0-rc1"
    )
    assert check(pyproject, plist) == ("0.13.0-rc1", "0.13.0-rc1")


@pytest.mark.parametrize("missing", ["pyproject", "plist"])
def test_missing_input_fails_rather_than_passes(tmp_path, missing):
    """A file move must turn the guard RED, not silently disable it."""
    pyproject = write_pyproject(tmp_path / "pyproject.toml")
    plist = write_plist(tmp_path / "Info.plist")
    (pyproject if missing == "pyproject" else plist).unlink()
    with pytest.raises(VersionSyncError, match="not found"):
        check(pyproject, plist)


def test_absent_plist_key_fails(tmp_path):
    plist = write_plist(tmp_path / "Info.plist", CFBundleShortVersionString=None)
    with pytest.raises(VersionSyncError, match="CFBundleShortVersionString"):
        app_version(plist)


def test_absent_pyproject_version_fails(tmp_path):
    pyproject = write_pyproject(tmp_path / "pyproject.toml", version=None)
    with pytest.raises(VersionSyncError, match="no \\[project\\] version"):
        engine_version(pyproject)


def test_unreadable_plist_fails(tmp_path):
    """Not a plist at all — must raise, not return a default."""
    plist = tmp_path / "Info.plist"
    plist.write_bytes(b"this is not a plist")
    with pytest.raises(VersionSyncError, match="not a readable plist"):
        app_version(plist)


def test_unreadable_pyproject_fails(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[project\nname =", encoding="utf-8")
    with pytest.raises(VersionSyncError, match="not readable TOML"):
        engine_version(pyproject)


def test_unreadable_pyproject_oserror_is_not_a_traceback(tmp_path, monkeypatch):
    """A PermissionError must land on the ``::error::`` path like any other
    unreadable input — escaping as a traceback would make ``main`` exit
    non-zero for a reason nobody can act on."""
    pyproject = write_pyproject(tmp_path / "pyproject.toml")

    def denied(*_a, **_k):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(VersionSyncError, match="not readable TOML"):
        engine_version(pyproject)


def test_non_table_project_fails(tmp_path):
    """``project = "x"`` is valid TOML; ``.get`` on the str is not."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('project = "not-a-table"\n', encoding="utf-8")
    with pytest.raises(VersionSyncError, match="no \\[project\\] version"):
        engine_version(pyproject)


def test_non_dict_plist_root_fails(tmp_path):
    """``<plist><array/></plist>`` parses fine and has no ``.get``."""
    plist = tmp_path / "Info.plist"
    with plist.open("wb") as fh:
        plistlib.dump(["not", "a", "dict"], fh)
    with pytest.raises(VersionSyncError, match="CFBundleShortVersionString"):
        app_version(plist)


@pytest.mark.parametrize(
    "bad",
    [
        "0.12",
        "1.0.0-rc0",
        "1.0.0-beta1",
        "v1.2.3",
        "",
        # ``$`` matches BEFORE a trailing newline, so an ``^…$`` check
        # would accept this and then build the tag ``v1.2.3\n``. A
        # hand-edited ``<string>`` element is how it gets in.
        "1.2.3\n",
        " 1.2.3",
        # ``\d`` is Unicode-aware; no release tag can carry these.
        "١.٢.٣",
        # PEP 440 normalises leading zeros, so these would publish to
        # PyPI as 1.2.3 while the plist and tag kept the literal string —
        # two files agreeing on a version that ships as three.
        "01.02.3",
        "1.2.03",
    ],
)
def test_non_semver_is_rejected_on_both_sides(tmp_path, bad):
    """Release tags are built from these strings; the updater orders them."""
    plist = write_plist(tmp_path / "a.plist", CFBundleShortVersionString=bad)
    with pytest.raises(VersionSyncError) as plist_exc:
        app_version(plist)
    # Rejected for its VALUE, not because the file failed to parse — a
    # malformed fixture would make this test green while the SemVer
    # branch it targets never ran.
    assert "not a readable plist" not in str(plist_exc.value)

    pyproject = write_pyproject(tmp_path / "b.toml", version=bad)
    with pytest.raises(VersionSyncError) as toml_exc:
        engine_version(pyproject)
    assert "not readable TOML" not in str(toml_exc.value)


# --- the CLI entry point CI actually calls --------------------------


def test_main_returns_zero_when_repo_is_in_sync(capsys):
    assert main() == 0
    assert "agree" in capsys.readouterr().out


def test_main_reports_failure_as_an_actionable_annotation(monkeypatch, capsys):
    """CI reads ``::error::`` — a bare traceback is not a review signal."""
    import scripts.check_version_sync as mod

    def boom(*_a, **_k):
        raise VersionSyncError("engine 9.9.9 != app 8.8.8")

    monkeypatch.setattr(mod, "check", boom)
    assert mod.main() == 1
    assert "::error::" in capsys.readouterr().err
