"""RC publishing must not promote prerelease bits to stable users."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
AUTO_RELEASE = ROOT / ".github/workflows/auto-release.yml"
DESKTOP_WORKFLOW = ROOT / ".github/workflows/rapid-mac-release.yml"
DESKTOP_RELEASABLE = ROOT / ".github/actions/desktop-releasable/action.yml"
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


def test_desktop_raw_bundle_headroom_does_not_weaken_dmg_growth_gate():
    workflow = DESKTOP_WORKFLOW.read_text(encoding="utf-8")
    shared_action = DESKTOP_RELEASABLE.read_text(encoding="utf-8")
    assert 'BUNDLE_SIZE_CAP_MB: "550"' in workflow
    assert 'BUNDLE_SIZE_DELTA_CAP_MB: "50"' in workflow
    assert 'CAP_MB="${BUNDLE_SIZE_CAP_MB:-550}"' in shared_action
    assert 'DELTA_CAP_MB="${BUNDLE_SIZE_DELTA_CAP_MB:-50}"' in shared_action


def test_auto_release_stages_one_sha_labelled_complete_desktop_bundle():
    workflow = AUTO_RELEASE.read_text(encoding="utf-8")
    candidate = _step(workflow, "Stage exact pre-tag Desktop candidate bundle")
    upload = _step(workflow, "Upload SHA-labelled pre-tag Desktop candidate")
    assert "desktop_promotion.py create" in candidate
    assert "rapid-mlx-desktop.dmg" in candidate
    assert "rapid-mlx-desktop.manifest.json" in candidate
    assert "release-notes.md" in candidate
    assert "sparkle/appcast.xml" in candidate
    assert "sparkle/*.zip" in candidate
    assert "rapid-mlx-desktop-pre-tag-candidate-${{ github.sha }}" in upload


def test_tier1_candidate_installs_reduced_vision_runtime_for_multimodal_roster():
    workflow = AUTO_RELEASE.read_text(encoding="utf-8")
    gate = _step(workflow, "Re-verify Tier-1 agents against the release source")
    roster = yaml.safe_load(
        (ROOT / "tests/integrations/top10_sequences.yaml").read_text(encoding="utf-8")
    )

    assert "gemma-4-26b-4bit" in roster["top_10_aliases"]
    candidate_install = gate.index(
        '"$V/bin/pip" install -q --constraint "$CONSTRAINTS" "$CANDIDATE_WHEEL"'
    )
    pip_check = gate.index('"$V/bin/pip" check')
    reduced_install = gate.index(
        '"$V/bin/pip" install -q --no-deps --constraint "$CONSTRAINTS"'
    )
    distribution_check = gate.index("check-sidecar-distributions.py", reduced_install)
    roster_run = gate.index(
        'RAPID_MLX_VENV="$V" bash tests/integrations/agent_smoke.sh'
    )
    assert (
        candidate_install
        < pip_check
        < reduced_install
        < distribution_check
        < roster_run
    )
    assert "'mlx-vlm' 'Pillow>=10.0'" in gate


def test_tagged_promotion_and_standalone_build_are_mutually_exclusive():
    workflow = yaml.load(DESKTOP_WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert set(inputs) == {"promote_run_id", "promote_sha"}
    jobs = workflow["jobs"]
    assert "github.event_name != 'workflow_dispatch'" in jobs["build"]["if"]
    assert "inputs.promote_run_id == ''" in jobs["build"]["if"]
    assert "inputs.promote_sha == ''" in jobs["build"]["if"]
    assert "github.event_name == 'workflow_dispatch'" in jobs["promote-candidate"]["if"]
    assert jobs["promote-candidate"]["if"] == (
        "github.event_name == 'workflow_dispatch' "
        "&& inputs.promote_run_id != '' && inputs.promote_sha != ''"
    )
    assert jobs["desktop-ready"]["needs"] == ["build", "promote-candidate"]
    assert jobs["mirror-dist"]["needs"] == "desktop-ready"
    assert jobs["publish-updater-fallback"]["needs"] == [
        "desktop-ready",
        "mirror-dist",
    ]
    publish_condition = (
        "startsWith(github.ref, 'refs/tags/') "
        "|| (github.event_name == 'workflow_dispatch' "
        "&& inputs.promote_run_id != '' "
        "&& inputs.promote_sha != '')"
    )
    assert " ".join(jobs["mirror-dist"]["if"].split()) == publish_condition
    assert " ".join(jobs["publish-updater-fallback"]["if"].split()) == publish_condition


def test_release_preflight_is_dispatch_only_and_exact_head_bound():
    workflow = PREFLIGHT_WORKFLOW.read_text(encoding="utf-8")
    parsed = yaml.load(workflow, Loader=yaml.BaseLoader)
    assert set(parsed["on"]) == {"workflow_dispatch"}
    inputs = parsed["on"]["workflow_dispatch"]["inputs"]
    assert inputs["pr_number"]["required"] == "true"
    assert inputs["expected_sha"]["required"] == "true"
    bind = parsed["jobs"]["bind-bump-pr"]
    script = bind["steps"][1]["run"]
    assert 'gh api "repos/${REPO}/pulls/${PR_NUMBER}"' in script
    assert 'EXPECTED_SHA" != "$HEAD_SHA' in script
    assert 'DISPATCH_SHA" != "$HEAD_SHA' in script


def test_release_preflight_rechecks_contract_before_privileged_jobs():
    workflow = yaml.load(
        PREFLIGHT_WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader
    )
    pf1 = workflow["jobs"]["pf1-release-contract"]
    scripts = [step.get("run", "") for step in pf1["steps"]]
    assert pf1["needs"] == "bind-bump-pr"
    assert any("validate_release_subject.py" in script for script in scripts)
    assert any("check_release_notes.py" in script for script in scripts)


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
