"""Release metadata preparation must preserve product and published history."""

import datetime
import plistlib
from pathlib import Path

import pytest

from scripts import release_prepare as target


@pytest.fixture
def inputs():
    project = '[project]\nname = "rapid-mlx"\nversion = "0.15.7"\n\n[tool.example]\nversion = "42"\n'
    plist = '<?xml version="1.0"?><plist version="1.0"><dict><!--retain--><key>CFBundleVersion</key><string>182</string><key>CFBundleShortVersionString</key><string>0.15.7</string><key>Other</key><string>unchanged</string></dict></plist>'
    root = "https://github.com/raullenchai/Rapid-MLX/compare/"
    changelog = f"# Changelog\n\n## [Unreleased]\n\n## [0.15.7] — 2026-10-07\n\nOld notes.\n\n[Unreleased]: {root}rapid-mac-v0.15.7...HEAD\n[0.15.7]: {root}rapid-mac-v0.15.6...rapid-mac-v0.15.7\n"
    published = changelog.replace(
        "## [0.15.7]",
        "## [0.16.0] — 2026-10-09\n\nPublished frozen release.\n\n## [0.15.7]",
        1,
    )
    published += f"[0.16.0]: {root}rapid-mac-v0.15.7...rapid-mac-v0.16.0\n"
    return dict(
        project=project,
        plist=plist,
        changelog=changelog,
        published_changelog=published,
        previous_version="0.16.0",
        version="0.16.1",
        previous_builds=[180, 183],
        notes="# Release\nCurated notes.",
        highlights="### Highlights\n- User benefit.",
        date="2026-10-10",
    )


def test_frozen_release_ahead_of_main_restores_published_history(inputs, tmp_path):
    files = target.render(**inputs)
    assert set(files) == {
        "pyproject.toml",
        target.PLIST,
        target.CHANGELOG,
        "docs/release-notes/v0.16.1.md",
    }
    assert '[tool.example]\nversion = "42"' in files["pyproject.toml"]
    assert target.project_version(files["pyproject.toml"]) == "0.16.1"
    info = plistlib.loads(files[target.PLIST].encode())
    assert info == {
        "CFBundleVersion": "184",
        "CFBundleShortVersionString": "0.16.1",
        "Other": "unchanged",
    }
    assert "<!--retain-->" in files[target.PLIST]
    assert "Published frozen release." in files[target.CHANGELOG]
    assert "Old notes." in files[target.CHANGELOG]
    assert files[target.CHANGELOG].index("## [0.16.1]") < files[target.CHANGELOG].index(
        "## [0.16.0]"
    )
    target.write_metadata(tmp_path, files, inputs["version"])
    with pytest.raises(ValueError, match="overwrite"):
        target.write_metadata(tmp_path, files, inputs["version"])


def test_existing_published_sections_are_not_duplicated(inputs):
    inputs["changelog"] = inputs["published_changelog"]
    result = target.render(**inputs)
    assert result[target.CHANGELOG].count("## [0.16.0]") == 1


@pytest.mark.parametrize("version", ["0.15.6", "0.15.7", "0.16.0", "bad", "../foo"])
def test_rejects_old_reserved_or_invalid_version(inputs, version):
    inputs["version"] = version
    with pytest.raises(ValueError):
        target.render(**inputs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("notes", ""),
        ("highlights", "  "),
        ("date", "bad"),
        ("previous_builds", []),
        ("previous_builds", [True]),
        ("previous_builds", [-1]),
        ("changelog", "# No unreleased"),
        ("published_changelog", "# No baseline"),
    ],
)
def test_rejects_missing_inputs_without_touching_files(inputs, field, value):
    inputs[field] = value
    with pytest.raises(ValueError):
        target.render(**inputs)


@pytest.mark.parametrize(
    "project",
    [
        '[tool.a]\nversion="1.0.0"',
        '[project]\nname="x"',
        '[project]\nversion="1.0.0"\nversion="1.1.0"',
    ],
)
def test_project_version_rejects_ambiguous_metadata(project):
    with pytest.raises(ValueError):
        target.project_version(project)


def test_rejects_duplicate_target_and_bad_plist(inputs):
    inputs["changelog"] += "\n## [0.16.1]\nAlready reserved.\n"
    with pytest.raises(ValueError, match="already"):
        target.render(**inputs)
    inputs["changelog"] = inputs["changelog"].split("\n## [0.16.1]")[0]
    inputs["plist"] = inputs["plist"].replace(
        "<string>182</string>", "<string>abc</string>"
    )
    with pytest.raises(ValueError, match="numeric"):
        target.render(**inputs)


def test_plist_replacement_rejects_missing_or_duplicate_keys():
    for content in (
        "<dict/>",
        "<key>x</key><string>a</string><key>x</key><string>b</string>",
    ):
        with pytest.raises(ValueError):
            target.replace_plist(content, "x", "new")


def test_history_and_reference_errors_are_not_silenced(inputs):
    with pytest.raises(ValueError, match="no section"):
        target.changelog_section("## [1.0.0]\nNotes", "1.1.0")
    inputs["published_changelog"] = inputs["published_changelog"].split("[0.16.0]:")[0]
    with pytest.raises(ValueError, match="comparison reference"):
        target.render(**inputs)


def test_missing_unreleased_reference_rejects(inputs):
    inputs["changelog"] = "\n".join(
        x for x in inputs["changelog"].splitlines() if not x.startswith("[Unreleased]:")
    )
    with pytest.raises(ValueError, match="comparison reference"):
        target.render(**inputs)


def test_real_published_0160_metadata_compatibility(tmp_path):
    # Actual checked-in metadata is an older source than the frozen release;
    # synthetic published history above covers that reconciliation separately.
    root = Path(__file__).parents[1]
    project = (root / "pyproject.toml").read_text()
    current = target.project_version(project)
    major, minor, patch = current.split(".")
    version = f"{major}.{minor}.{int(patch) + 1}"
    files = target.render(
        project=project,
        plist=(root / target.PLIST).read_text(),
        changelog=(root / target.CHANGELOG).read_text(),
        published_changelog=(root / target.CHANGELOG).read_text(),
        previous_version=current,
        version=version,
        previous_builds=[999],
        notes="# Candidate\nUser-facing notes.",
        highlights="### Highlights\n- Change.",
        date=datetime.date.today().isoformat(),
    )
    target.write_metadata(tmp_path, files, version)
    assert plistlib.loads(files[target.PLIST].encode())["CFBundleVersion"] == "1000"


def test_unrelated_plist_mutation_is_rejected(inputs, monkeypatch):
    original = target.replace_plist

    def corrupt(text, key, value):
        return original(text, key, value).replace(
            "<string>unchanged</string>", "<string>corrupt</string>"
        )

    monkeypatch.setattr(target, "replace_plist", corrupt)
    with pytest.raises(ValueError, match="unexpected plist"):
        target.render(**inputs)


def test_direct_metadata_import_matches_package_renderer(monkeypatch, inputs):
    """Exercise fresh direct-script imports even after full-suite collection."""
    import runpy
    import sys
    from unittest.mock import patch

    script = Path(__file__).parents[1] / "scripts" / "release_prepare.py"
    monkeypatch.syspath_prepend(str(script.parent))
    with patch.dict(sys.modules):
        for name in ("check_release_notes", "release_version"):
            sys.modules.pop(name, None)
        direct = runpy.run_path(str(script), run_name="__direct_metadata_import__")
        assert direct["render"](**inputs) == target.render(**inputs)
