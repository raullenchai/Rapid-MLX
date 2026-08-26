# SPDX-License-Identifier: Apache-2.0
"""Offline behavioral tests for scripts/check_desktop_publish.py.

Exercises the exact-tagged-publication gate with a mock ``gh`` and --sleep-sec 0
(via direct ``verify()`` calls with tiny deadlines), so no network or real wait.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import textwrap

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "check_desktop_publish.py"

ACCEPTED = "a" * 40
OTHER = "b" * 40
DEFAULT_DMG = b"exact tagged desktop dmg"
DEFAULT_DIGEST = hashlib.sha256(DEFAULT_DMG).hexdigest()


def _load_module():
    spec = importlib.util.spec_from_file_location("check_desktop_publish", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gate():
    return _load_module()


def _write(tree, rel, content):
    p = tree / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


def _write_bytes(tree, rel, content):
    p = tree / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return p


def _mock_state(tree):
    """Build a mock ``gh`` + state dir; return (gh_path, state)."""
    tree.mkdir(parents=True, exist_ok=True)
    state = tree / "state"
    state.mkdir()
    _write(state, "count", "0")
    _write(state, "ref_count", "0")
    _write(
        state, "ref.json", json.dumps({"object": {"type": "commit", "sha": ACCEPTED}})
    )
    _write(
        state,
        "annotated.json",
        json.dumps({"object": {"type": "commit", "sha": ACCEPTED}}),
    )
    # Default: no runs (nothing yet) and no release.
    _write(state, "runs.sh", "#!/usr/bin/env bash\necho '[]'\n")
    _write(
        state,
        "release.json",
        json.dumps({"tag_name": "rapid-mac-v0.13.0-rc2", "draft": False, "assets": []}),
    )
    _write(state, "fail", "0")
    artifacts = state / "artifacts"
    artifacts.mkdir()
    _write_bytes(artifacts, "rapid-mlx-desktop.dmg", DEFAULT_DMG)
    _write(
        artifacts,
        "rapid-mlx-desktop.manifest.json",
        json.dumps(
            {
                "schema": 1,
                "project": "rapid-mlx",
                "artifact_kind": "desktop-dmg",
                "version": "0.13.0-rc2",
                "app_tag": "rapid-mac-v0.13.0-rc2",
                "source_sha": ACCEPTED,
                "embedded_version": {
                    "CFBundleShortVersionString": "0.13.0-rc2",
                    "CFBundleVersion": "170",
                },
                "signed": True,
                "dmg_size_delta_compared": True,
                "validation_gate": (
                    "signed-build|bundle-size|app-notarize|dmg-build|"
                    "dmg-size-delta|validate-dmg|dmg-notarize|final-validate-dmg"
                ),
                "artifacts": [
                    {
                        "filename": "rapid-mlx-desktop.dmg",
                        "size": len(DEFAULT_DMG),
                        "sha256": DEFAULT_DIGEST,
                    }
                ],
            }
        ),
    )

    gh = tree / "gh"
    # The absolute STATE path is EMBEDDED so the mock needs no global env (the
    # helper only supplies GH_REPO via env).
    gh.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            STATE={str(state)}
            # The helper invokes "gh api <url> ...", so argv[1] is the "api"
            # subcommand and argv[2] is the URL. Validate/shift past it the way a
            # real gh expects, then dispatch on the URL.
            if [[ "${{1:-}}" == "run" && "${{2:-}}" == "download" ]]; then
              dest=""
              while (( $# )); do
                if [[ "$1" == "--dir" ]]; then dest="$2"; shift 2; else shift; fi
              done
              [[ -n "$dest" ]] || {{ echo "mock gh: missing --dir" >&2; exit 2; }}
              mkdir -p "$dest"
              for artifact in "$STATE/artifacts"/*; do cp "$artifact" "$dest/"; done
              exit 0
            fi
            if [[ "${{1:-}}" != "api" ]]; then
              echo "mock gh: expected 'api' or 'run download', got '${{1:-}}'" >&2; exit 2
            fi
            url="${{2:-}}"
            if [[ -f "${{STATE}}/fail" && "$(cat "${{STATE}}/fail")" == "1" ]]; then
              echo "mock gh: simulated API failure" >&2; exit 1
            fi
            case "$url" in
              *"/git/ref/tags/"*)
                if [[ -f "${{STATE}}/ref.sh" ]]; then bash "${{STATE}}/ref.sh"; else cat "${{STATE}}/ref.json"; fi ;;
              *"/git/tags/"*)     cat "${{STATE}}/annotated.json" ;;
              *"/actions/workflows/"*) bash "${{STATE}}/runs.sh" ;;
              *"/releases/tags/"*) cat "${{STATE}}/release.json" ;;
              *) echo "mock gh: unhandled url $url" >&2; exit 2 ;;
            esac
            """
        )
    )
    gh.chmod(0o755)
    return gh, state


def _runs_with(*records):
    """Return a runs.sh body returning the given workflow_runs records."""
    arr = json.dumps(records)
    return f"#!/usr/bin/env bash\necho '{arr}'\n"


def _run_record(
    run_id,
    *,
    status="completed",
    conclusion="success",
    event="push",
    head_sha=ACCEPTED,
    head_branch="rapid-mac-v0.13.0-rc2",
):
    return {
        "id": run_id,
        "event": event,
        "head_sha": head_sha,
        "head_branch": head_branch,
        "status": status,
        "conclusion": conclusion,
    }


def _release_with(*assets, draft=False):
    # The release-by-tag REST endpoint uses "draft" (not "isDraft"). Modeling the
    # real field is what proves the helper doesn't key on the wrong name.
    return json.dumps(
        {
            "tag_name": "rapid-mac-v0.13.0-rc2",
            "draft": draft,
            "assets": list(assets),
        }
    )


def _asset(
    name="rapid-mlx-desktop.dmg",
    state="uploaded",
    size=len(DEFAULT_DMG),
    digest="default",
):
    a = {"name": name, "state": state, "size": size}
    if digest == "default":
        a["digest"] = "sha256:" + DEFAULT_DIGEST
    elif digest is not None:
        a["digest"] = digest
    return a


def _verify(gate, gh, state, *, deadline_sec=0.05, sleep_sec=0.0):
    return gate.verify(
        app_tag="rapid-mac-v0.13.0-rc2",
        accepted_sha=ACCEPTED,
        repo="raullenchai/Rapid-MLX",
        workflow="rapid-mac-release.yml",
        gh=str(gh),
        deadline_sec=deadline_sec,
        sleep_sec=sleep_sec,
    )


def test_happy_path(gate, tmp_path):
    gh, state = _mock_state(tmp_path)
    _write(state, "runs.sh", _runs_with(_run_record(1)))
    _write(state, "release.json", _release_with(_asset()))
    evidence = _verify(gate, gh, state)
    assert any("run" in e and "succeeded" in e for e in evidence)
    assert any("publishes the exact run DMG" in e for e in evidence)


def test_older_failure_plus_active_rerun_waits_then_succeeds(gate, tmp_path):
    # A completed FAILED run plus an in_progress rerun: the gate must NOT fail on
    # the older failure while an exact run is active; it waits for the success.
    gh, state = _mock_state(tmp_path)
    # runs.sh: call 1 returns [failed, in_progress]; later calls return [failed, success].
    runs_script = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        n="$(cat "{state}/count")"
        echo "$((n + 1))" > "{state}/count"
        if [[ "$n" == "0" ]]; then
          echo '['
          echo '  {{"id": 10, "event": "push", "head_sha": "{ACCEPTED}", "head_branch": "rapid-mac-v0.13.0-rc2", "status": "completed", "conclusion": "failure"}},'
          echo '  {{"id": 11, "event": "push", "head_sha": "{ACCEPTED}", "head_branch": "rapid-mac-v0.13.0-rc2", "status": "in_progress", "conclusion": null}}'
          echo ']'
        else
          echo '['
          echo '  {{"id": 10, "event": "push", "head_sha": "{ACCEPTED}", "head_branch": "rapid-mac-v0.13.0-rc2", "status": "completed", "conclusion": "failure"}},'
          echo '  {{"id": 11, "event": "push", "head_sha": "{ACCEPTED}", "head_branch": "rapid-mac-v0.13.0-rc2", "status": "completed", "conclusion": "success"}}'
          echo ']'
        fi
        """
    )
    _write(state, "runs.sh", runs_script)
    _write(state, "release.json", _release_with(_asset()))
    evidence = _verify(gate, gh, state, deadline_sec=10)
    assert any("run 11 succeeded" in e for e in evidence)


def test_old_failure_plus_success_prefers_success(gate, tmp_path):
    gh, state = _mock_state(tmp_path)
    _write(
        state,
        "runs.sh",
        _runs_with(_run_record(10, conclusion="failure"), _run_record(11)),
    )
    _write(state, "release.json", _release_with(_asset()))
    evidence = _verify(gate, gh, state)
    assert any("run 11 succeeded" in e for e in evidence)


def test_newer_failed_rerun_invalidates_older_success(gate, tmp_path):
    gh, state = _mock_state(tmp_path)
    _write(
        state,
        "runs.sh",
        _runs_with(_run_record(10), _run_record(11, conclusion="failure")),
    )
    _write(state, "release.json", _release_with(_asset()))
    with pytest.raises(gate.PublishGateError, match="completed without success"):
        _verify(gate, gh, state)


def test_success_waits_for_active_exact_rerun_then_uses_newest(gate, tmp_path):
    gh, state = _mock_state(tmp_path)
    runs_script = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        n="$(cat "{state}/count")"
        echo "$((n + 1))" > "{state}/count"
        if [[ "$n" == "0" ]]; then
          echo '{json.dumps([_run_record(10), _run_record(11, status="in_progress", conclusion=None)])}'
        else
          echo '{json.dumps([_run_record(10), _run_record(11)])}'
        fi
        """
    )
    _write(state, "runs.sh", runs_script)
    _write(state, "release.json", _release_with(_asset()))
    evidence = _verify(gate, gh, state, deadline_sec=10)
    assert any("run 11 succeeded" in e for e in evidence)
    assert (state / "count").read_text().strip() == "2"


def test_delayed_tag_run_appearance_is_polled_to_success(gate, tmp_path):
    gh, state = _mock_state(tmp_path)
    runs_script = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        n="$(cat "{state}/count")"
        echo "$((n + 1))" > "{state}/count"
        if [[ "$n" -lt 3 ]]; then echo '[]'; else echo '{json.dumps([_run_record(12)])}'; fi
        """
    )
    _write(state, "runs.sh", runs_script)
    _write(state, "release.json", _release_with(_asset()))
    evidence = _verify(gate, gh, state, deadline_sec=10)
    assert any("run 12 succeeded" in e for e in evidence)
    assert (state / "count").read_text().strip() == "4"


def test_only_failed_no_active_fails_closed(gate, tmp_path):
    gh, state = _mock_state(tmp_path)
    _write(state, "runs.sh", _runs_with(_run_record(10, conclusion="failure")))
    with pytest.raises(gate.PublishGateError, match="completed without success"):
        _verify(gate, gh, state, deadline_sec=0.05)


def test_still_in_progress_times_out(gate, tmp_path):
    gh, state = _mock_state(tmp_path)
    # in_progress forever -> never a success, never a settled failure -> timeout.
    _write(
        state,
        "runs.sh",
        _runs_with(_run_record(10, status="in_progress", conclusion=None)),
    )
    with pytest.raises(gate.PublishGateError, match="within the deadline"):
        _verify(gate, gh, state, deadline_sec=0.05, sleep_sec=0.0)


def test_api_failure_fails_immediately(gate, tmp_path):
    gh, state = _mock_state(tmp_path)
    _write(state, "fail", "1")
    with pytest.raises(gate.PublishGateError, match="simulated API failure"):
        _verify(gate, gh, state, deadline_sec=10)


def test_wrong_head_branch_is_not_exact(gate, tmp_path):
    gh, state = _mock_state(tmp_path)
    # A run on a DIFFERENT tag branch is not the exact publish for this tag.
    _write(
        state,
        "runs.sh",
        _runs_with(_run_record(9, head_branch="rapid-mac-v0.13.0-rc1")),
    )
    # Resolution still says ACCEPTED (ref.json) but no exact run -> timeout/fail.
    with pytest.raises(gate.PublishGateError):
        _verify(gate, gh, state, deadline_sec=0.05)


def test_wrong_head_sha_is_not_exact(gate, tmp_path):
    gh, state = _mock_state(tmp_path)
    _write(state, "runs.sh", _runs_with(_run_record(9, head_sha=OTHER)))
    with pytest.raises(gate.PublishGateError):
        _verify(gate, gh, state, deadline_sec=0.05)


def test_exact_tag_workflow_dispatch_recovery_is_accepted(gate, tmp_path):
    gh, state = _mock_state(tmp_path)
    _write(
        state,
        "runs.sh",
        _runs_with(_run_record(12, event="workflow_dispatch")),
    )
    _write(state, "release.json", _release_with(_asset()))
    evidence = _verify(gate, gh, state)
    assert any("event workflow_dispatch" in line for line in evidence)


def test_unrelated_event_is_not_exact(gate, tmp_path):
    gh, state = _mock_state(tmp_path)
    _write(state, "runs.sh", _runs_with(_run_record(13, event="pull_request")))
    with pytest.raises(gate.PublishGateError, match="within the deadline"):
        _verify(gate, gh, state, deadline_sec=0.05)


def test_malformed_workflow_run_fails_immediately(gate, tmp_path):
    gh, state = _mock_state(tmp_path)
    malformed = _run_record(14)
    del malformed["status"]
    _write(state, "runs.sh", _runs_with(malformed))
    with pytest.raises(gate.PublishGateError, match="malformed record"):
        _verify(gate, gh, state, deadline_sec=10)


def test_draft_release_fails(gate, tmp_path):
    gh, state = _mock_state(tmp_path)
    _write(state, "runs.sh", _runs_with(_run_record(1)))
    _write(state, "release.json", _release_with(_asset(), draft=True))
    with pytest.raises(gate.PublishGateError, match="published"):
        _verify(gate, gh, state)


def test_missing_asset_fails(gate, tmp_path):
    gh, state = _mock_state(tmp_path)
    _write(state, "runs.sh", _runs_with(_run_record(1)))
    _write(state, "release.json", _release_with())
    with pytest.raises(gate.PublishGateError, match="published"):
        _verify(gate, gh, state)


def test_zero_size_asset_fails(gate, tmp_path):
    gh, state = _mock_state(tmp_path)
    _write(state, "runs.sh", _runs_with(_run_record(1)))
    _write(state, "release.json", _release_with(_asset(size=0)))
    with pytest.raises(gate.PublishGateError, match="published"):
        _verify(gate, gh, state)


def test_boolean_size_asset_fails(gate, tmp_path):
    gh, state = _mock_state(tmp_path)
    _write(state, "runs.sh", _runs_with(_run_record(1)))
    _write(state, "release.json", _release_with(_asset(size=True)))
    with pytest.raises(gate.PublishGateError, match="published"):
        _verify(gate, gh, state)


def test_wrong_release_tag_binding_fails(gate, tmp_path):
    gh, state = _mock_state(tmp_path)
    _write(state, "runs.sh", _runs_with(_run_record(1)))
    _write(
        state,
        "release.json",
        json.dumps(
            {
                "tag_name": "rapid-mac-v0.13.0-rc1",
                "draft": False,
                "assets": [_asset()],
            }
        ),
    )
    with pytest.raises(gate.PublishGateError, match="not bound"):
        _verify(gate, gh, state)


def test_non_uploaded_asset_fails(gate, tmp_path):
    gh, state = _mock_state(tmp_path)
    _write(state, "runs.sh", _runs_with(_run_record(1)))
    _write(state, "release.json", _release_with(_asset(state="starter")))
    with pytest.raises(gate.PublishGateError, match="published"):
        _verify(gate, gh, state)


def test_bad_digest_fails(gate, tmp_path):
    gh, state = _mock_state(tmp_path)
    _write(state, "runs.sh", _runs_with(_run_record(1)))
    _write(state, "release.json", _release_with(_asset(digest="sha256:zz")))
    with pytest.raises(gate.PublishGateError, match="published"):
        _verify(gate, gh, state)


def test_good_digest_passes(gate, tmp_path):
    gh, state = _mock_state(tmp_path)
    _write(state, "runs.sh", _runs_with(_run_record(1)))
    _write(state, "release.json", _release_with(_asset()))
    evidence = _verify(gate, gh, state)
    assert any("publishes the exact run DMG" in e for e in evidence)


def test_release_digest_must_match_exact_run_manifest(gate, tmp_path):
    gh, state = _mock_state(tmp_path)
    _write(state, "runs.sh", _runs_with(_run_record(1)))
    _write(
        state,
        "release.json",
        _release_with(_asset(digest="sha256:" + "e" * 64)),
    )
    with pytest.raises(gate.PublishGateError, match="does not match exact run"):
        _verify(gate, gh, state)


def test_run_dmg_bytes_must_match_manifest(gate, tmp_path):
    gh, state = _mock_state(tmp_path)
    _write(state, "runs.sh", _runs_with(_run_record(1)))
    _write(state, "release.json", _release_with(_asset()))
    _write_bytes(state / "artifacts", "rapid-mlx-desktop.dmg", b"tampered")
    with pytest.raises(gate.PublishGateError, match="bytes do not match"):
        _verify(gate, gh, state)


def test_run_manifest_must_bind_signed_candidate(gate, tmp_path):
    gh, state = _mock_state(tmp_path)
    _write(state, "runs.sh", _runs_with(_run_record(1)))
    _write(state, "release.json", _release_with(_asset()))
    manifest_path = state / "artifacts" / "rapid-mlx-desktop.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["signed"] = False
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(gate.PublishGateError, match="identity/signing"):
        _verify(gate, gh, state)


def test_run_artifact_requires_manifest(gate, tmp_path):
    gh, state = _mock_state(tmp_path)
    _write(state, "runs.sh", _runs_with(_run_record(1)))
    _write(state, "release.json", _release_with(_asset()))
    (state / "artifacts" / "rapid-mlx-desktop.manifest.json").unlink()
    with pytest.raises(gate.PublishGateError, match="must contain one"):
        _verify(gate, gh, state)


def test_initial_tag_mismatch_fails(gate, tmp_path):
    gh, state = _mock_state(tmp_path)
    _write(state, "runs.sh", _runs_with(_run_record(1)))
    _write(state, "release.json", _release_with(_asset()))
    # Tag no longer points at the accepted SHA -> fail before any run chance.
    _write(state, "ref.json", json.dumps({"object": {"type": "commit", "sha": OTHER}}))
    with pytest.raises(gate.PublishGateError, match="not the validated candidate"):
        _verify(gate, gh, state)


def test_tag_drift_after_desktop_publish_fails_final_recheck(gate, tmp_path):
    gh, state = _mock_state(tmp_path)
    _write(state, "runs.sh", _runs_with(_run_record(1)))
    _write(state, "release.json", _release_with(_asset()))
    accepted_json = json.dumps({"object": {"type": "commit", "sha": ACCEPTED}})
    other_json = json.dumps({"object": {"type": "commit", "sha": OTHER}})
    _write(
        state,
        "ref.sh",
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            n="$(cat '{state / "ref_count"}')"
            echo "$((n + 1))" > '{state / "ref_count"}'
            if [[ "$n" == "0" ]]; then
              echo '{accepted_json}'
            else
              echo '{other_json}'
            fi
            """
        ),
    )
    with pytest.raises(gate.PublishGateError, match="no longer resolves"):
        _verify(gate, gh, state)


def test_invalid_app_tag_fails(gate, tmp_path):
    gh, state = _mock_state(tmp_path)
    with pytest.raises(gate.PublishGateError, match="app-tag"):
        gate.verify(
            app_tag="rapid-mac-v0.13",
            accepted_sha=ACCEPTED,
            repo="raullenchai/Rapid-MLX",
            workflow="rapid-mac-release.yml",
            gh=str(gh),
            deadline_sec=0.05,
            sleep_sec=0.0,
        )


def test_malformed_tag_json_fails(gate, tmp_path):
    gh, state = _mock_state(tmp_path)
    _write(state, "ref.json", "{not json")
    with pytest.raises(gate.PublishGateError, match="malformed"):
        _verify(gate, gh, state)
