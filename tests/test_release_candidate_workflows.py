"""RC publishing must not promote prerelease bits to stable users."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DESKTOP_WORKFLOW = ROOT / ".github/workflows/rapid-mac-release.yml"
PREFLIGHT_WORKFLOW = ROOT / ".github/workflows/release-preflight.yml"
PUBLISH_WORKFLOWS = [
    ROOT / ".github/workflows/publish.yml",
    ROOT / ".github/workflows/release-artifact-matrix.yml",
]


def _step(workflow: str, name: str) -> str:
    start = workflow.index(f"      - name: {name}")
    next_step = workflow.find("\n      - name:", start + 1)
    return workflow[start:] if next_step == -1 else workflow[start:next_step]


def test_rc_tag_is_accepted_for_immutable_desktop_artifacts():
    workflow = DESKTOP_WORKFLOW.read_text(encoding="utf-8")
    assert "(-rc[1-9][0-9]*)?" in workflow


def test_bump_detection_checks_out_version_parser_before_invoking_it():
    workflow = PREFLIGHT_WORKFLOW.read_text(encoding="utf-8")
    detect = workflow[workflow.index("  detect-bump-pr:") : workflow.index("\n  pf1-")]
    checkout = detect.index("actions/checkout@")
    invocation = detect.index("scripts/release_version.py")
    assert checkout < invocation


def test_rc_never_replaces_stable_updater_pointer():
    workflow = DESKTOP_WORKFLOW.read_text(encoding="utf-8")
    publish = _step(workflow, "Publish updater fallback monotonically")
    assert "if: ${{ !contains(github.ref_name, '-rc') }}" in publish
    assert 'r2 object put "${R2_BUCKET}/latest.json"' in publish
    assert 'r2 object put "${R2_BUCKET}/appcast.xml"' in publish
    assert '"${R2_BUCKET}/rapid-mac/rapid-mlx-desktop.dmg"' in publish


def test_rc_github_release_is_still_created_as_prerelease():
    workflow = DESKTOP_WORKFLOW.read_text(encoding="utf-8")
    release = _step(
        workflow, "Create the GitHub Release (last — nothing ships before the pointer)"
    )
    assert '[[ "$VERSION" == *-* ]] && PRERELEASE="--prerelease"' in release
    assert "gh release create" in release


def test_privileged_publish_verifiers_use_pep440_rc_filenames():
    for path in PUBLISH_WORKFLOWS:
        workflow = path.read_text(encoding="utf-8")
        assert 'artifact_version = version.replace("-rc", "rc")' in workflow
        assert 'f"rapid_mlx-{artifact_version}-"' in workflow
        assert 'f"rapid_mlx-{artifact_version}.tar.gz"' in workflow
