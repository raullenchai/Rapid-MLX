import os
import shlex
import subprocess
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "rapid-mac-release.yml"
ACTION = ROOT / ".github" / "actions" / "desktop-releasable" / "action.yml"
BUILD = ROOT / "apps" / "rapid-mac" / "scripts" / "build.sh"


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def test_dispatch_candidate_identity_is_sha_bound_and_separate_from_versions() -> None:
    text = WORKFLOW.read_text()
    action = yaml.safe_load(ACTION.read_text())

    assert 'CANDIDATE_IDENTITY="candidate-${GITHUB_SHA:0:8}"' in text
    assert 'CANDIDATE_IDENTITY=""' in text
    assert "candidate_identity" in action["inputs"]
    assert action["inputs"]["candidate_identity"]["default"] == ""
    assert "RAPID_CANDIDATE_IDENTITY" in ACTION.read_text()


def test_candidate_tester_dmg_is_additive_and_dispatch_only() -> None:
    jobs = _workflow()["jobs"]
    build = jobs["build"]
    by_name = {step.get("name"): step for step in build["steps"]}

    canonical = by_name["Upload workflow artifact (DMG + manifest)"]
    stage = by_name["Stage candidate-labelled tester DMG"]
    candidate = by_name["Upload candidate-labelled tester DMG"]

    assert "rapid-mlx-desktop.dmg" in canonical["with"]["path"]
    assert stage["if"] == "steps.appmeta.outputs.is_tag != 'true'"
    assert candidate["if"] == "steps.appmeta.outputs.is_tag != 'true'"
    assert "rapid-mlx-desktop-${CANDIDATE_IDENTITY}.dmg" in stage["run"]
    assert 'cmp -s "$SOURCE" "$TARGET"' in stage["run"]
    assert "candidate_identity" in candidate["with"]["name"]

    # The exact pre-tag promotion path added by #2451 never enters this build
    # job, so it cannot acquire a tester-only identity or renamed artifact.
    assert "inputs.promote_run_id == ''" in build["if"]
    assert "inputs.promote_sha == ''" in build["if"]
    promotion_names = {step.get("name") for step in jobs["promote-candidate"]["steps"]}
    assert "Stage candidate-labelled tester DMG" not in promotion_names
    assert "Upload candidate-labelled tester DMG" not in promotion_names


def test_build_script_validates_and_embeds_separate_candidate_key() -> None:
    text = BUILD.read_text()

    assert "^candidate-[0-9a-f]{8}$" in text
    assert "plutil -insert RapidCandidateIdentity" in text
    for version_key in ("CFBundleVersion", "CFBundleShortVersionString"):
        assert f"plutil -insert {version_key}" not in text
        assert f"plutil -replace {version_key}" not in text


def test_signed_release_overlaps_sidecar_with_swift_and_joins_before_staging() -> None:
    text = BUILD.read_text()
    action = yaml.safe_load(ACTION.read_text())
    build_step = next(
        step
        for step in action["runs"]["steps"]
        if step.get("name") == "Build + sign Rapid-MLX Desktop.app"
    )

    assert build_step["env"]["PARALLEL_SIDECAR_BUILD"] == (
        "${{ inputs.parallel_sidecar_build == 'true' && inputs.signed == 'true' "
        "&& '1' || '0' }}"
    )
    assert action["inputs"]["parallel_sidecar_build"]["default"] == "false"
    start = text.index("starting signed sidecar build beside Swift compilation")
    swift = text.index('echo "==> swift build -c $CONFIG"')
    join = text.index("\njoin_parallel_sidecar\n")
    assemble = text.index('echo "==> assembling Rapid-MLX Desktop.app"')
    stage = text.index(
        'cp -R "$SIDECAR_STAGE/rapid-mlx" "$CONTENTS/Resources/rapid-mlx"'
    )
    assert start < swift < join < assemble < stage
    assert "parallel build timing: Swift" in text


def test_parallel_sidecar_failure_and_early_app_exit_are_fail_closed() -> None:
    text = BUILD.read_text()

    assert "trap cleanup_parallel_sidecar EXIT" in text
    assert 'kill -TERM -- "-$SIDECAR_BUILD_PID"' in text
    assert 'wait "$SIDECAR_BUILD_PID"' in text
    assert 'if [[ "$SIDECAR_BUILD_STATUS" -ne 0 ]]' in text
    assert 'exit "$SIDECAR_BUILD_STATUS"' in text
    assert '"${CODESIGN_IDENTITY:--}" != "-"' in text


def test_swift_failure_terminates_parallel_sidecar_process_group(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    child_pid = tmp_path / "child.pid"
    engine_root_seen = tmp_path / "engine-root-seen"
    sidecar = tmp_path / "fake-sidecar.sh"
    quoted_child_pid = shlex.quote(str(child_pid))
    quoted_engine_root_seen = shlex.quote(str(engine_root_seen))
    sidecar.write_text(
        "#!/bin/bash\n"
        "set -eu\n"
        f"printf '%s\\n' \"$RAPID_MLX_SOURCE\" > {quoted_engine_root_seen}\n"
        "(trap '' TERM; while :; do sleep 1; done) &\n"
        f"echo $! > {quoted_child_pid}\n"
        "wait\n"
    )
    sidecar.chmod(0o755)
    swift = bin_dir / "swift"
    swift.write_text(
        "#!/bin/bash\n"
        "set -eu\n"
        "attempt=0\n"
        f"while (( attempt < 400 )); do [[ -s {quoted_child_pid} ]] && exit 47; "
        "attempt=$((attempt + 1)); sleep 0.01; done\n"
        "exit 48\n"
    )
    swift.chmod(0o755)

    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "PARALLEL_SIDECAR_BUILD": "1",
        "CODESIGN_IDENTITY": "test identity",
        "RAPID_SIDECAR_SCRIPT": str(sidecar),
        "RAPID_SIDECAR_STAGE": str(tmp_path / "sidecar-stage"),
        "RAPID_MLX_ENGINE_ROOT": str(ROOT),
        "RUNNER_TEMP": str(tmp_path),
    }
    result = subprocess.run(
        ["bash", str(BUILD)],
        cwd=BUILD.parent.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 47, result.stdout + result.stderr
    assert engine_root_seen.read_text().strip() == str(ROOT)
    assert "ignored SIGTERM; sending SIGKILL" in result.stderr
    pid = int(child_pid.read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        pytest.fail(f"sidecar grandchild {pid} survived process-group cleanup")


def test_parallel_sidecar_rejects_unsafe_stage_before_recursive_delete() -> None:
    env = os.environ | {
        "PARALLEL_SIDECAR_BUILD": "1",
        "CODESIGN_IDENTITY": "test identity",
        "RAPID_SIDECAR_STAGE": "/",
        "RAPID_MLX_ENGINE_ROOT": str(ROOT),
    }
    result = subprocess.run(
        ["bash", str(BUILD)],
        cwd=BUILD.parent.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 1
    assert "refusing unsafe sidecar staging path: /" in result.stderr
    assert "starting signed sidecar build" not in result.stdout


def test_parallel_sidecar_failure_is_propagated_before_app_assembly(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sidecar = tmp_path / "failing-sidecar.sh"
    sidecar.write_text("#!/bin/bash\nexit 63\n")
    sidecar.chmod(0o755)
    swift = bin_dir / "swift"
    swift.write_text("#!/bin/bash\nexit 0\n")
    swift.chmod(0o755)

    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "PARALLEL_SIDECAR_BUILD": "1",
        "CODESIGN_IDENTITY": "test identity",
        "RAPID_SIDECAR_SCRIPT": str(sidecar),
        "RAPID_SIDECAR_STAGE": str(tmp_path / "sidecar-stage"),
        "RAPID_MLX_ENGINE_ROOT": str(ROOT),
        "RUNNER_TEMP": str(tmp_path),
    }
    result = subprocess.run(
        ["bash", str(BUILD)],
        cwd=BUILD.parent.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 63, result.stdout + result.stderr
    assert "parallel rapid-mlx sidecar build failed (63)" in result.stderr
    assert "assembling Rapid-MLX Desktop.app" not in result.stdout
