# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from scripts.check_release_notes import check_release_notes, main


def _inputs(tmp_path: Path, version: str = "0.13.2") -> tuple[Path, Path]:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        f"# Changelog\n\n## [{version}] — 2026-08-27\n\nDesktop changes.\n",
        encoding="utf-8",
    )
    notes_dir = tmp_path / "release-notes"
    notes_dir.mkdir()
    (notes_dir / f"v{version}.md").write_text("Release highlights.\n", encoding="utf-8")
    return changelog, notes_dir


@pytest.mark.parametrize("version", ["0.13.2", "0.14.0-rc1"])
def test_version_bound_inputs_are_synchronized(tmp_path: Path, version: str) -> None:
    changelog, notes_dir = _inputs(tmp_path, version)
    check_release_notes(version, changelog, notes_dir)


def test_missing_changelog_section_names_exact_fix(tmp_path: Path) -> None:
    changelog, notes_dir = _inputs(tmp_path)
    changelog.write_text("## [0.13.1]\n\nOld notes.\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"add '## \[0.13.2\]'"):
        check_release_notes("0.13.2", changelog, notes_dir)


def test_duplicate_changelog_section_is_rejected(tmp_path: Path) -> None:
    changelog, notes_dir = _inputs(tmp_path)
    changelog.write_text("## [0.13.2]\nOne.\n## [0.13.2]\nTwo.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="keep exactly one"):
        check_release_notes("0.13.2", changelog, notes_dir)


def test_empty_changelog_section_is_rejected(tmp_path: Path) -> None:
    changelog, notes_dir = _inputs(tmp_path)
    changelog.write_text("## [0.13.2]\n\n## [0.13.1]\nOld.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty '##"):
        check_release_notes("0.13.2", changelog, notes_dir)


def test_missing_version_bound_notes_file_is_rejected(tmp_path: Path) -> None:
    changelog, notes_dir = _inputs(tmp_path)
    (notes_dir / "v0.13.2.md").unlink()
    with pytest.raises(ValueError, match="create it for this bump PR"):
        check_release_notes("0.13.2", changelog, notes_dir)


def test_empty_version_bound_notes_file_is_rejected(tmp_path: Path) -> None:
    changelog, notes_dir = _inputs(tmp_path)
    (notes_dir / "v0.13.2.md").write_text(" \n", encoding="utf-8")
    with pytest.raises(ValueError, match="add curated release notes"):
        check_release_notes("0.13.2", changelog, notes_dir)


@pytest.mark.parametrize("version", ["0.13", "0.13.2-rc0", "../0.13.2"])
def test_invalid_version_is_rejected(tmp_path: Path, version: str) -> None:
    changelog, notes_dir = _inputs(tmp_path)
    with pytest.raises(ValueError, match="invalid release version"):
        check_release_notes(version, changelog, notes_dir)


def test_cli_reports_success(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    changelog, notes_dir = _inputs(tmp_path)
    assert (
        main(
            [
                "--version",
                "0.13.2",
                "--changelog",
                str(changelog),
                "--notes-dir",
                str(notes_dir),
            ]
        )
        == 0
    )
    assert "synchronized for 0.13.2" in capsys.readouterr().out
