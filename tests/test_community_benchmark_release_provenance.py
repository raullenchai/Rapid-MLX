# SPDX-License-Identifier: Apache-2.0
"""Which build published a benchmark, and whether it may publish at all.

Three failures lived here together:

1. ``build-sidecar.sh`` defaulted to ``distribution: source``, and the shared
   release action never set otherwise — so official apps stamped themselves as
   source builds.
2. A dirty build recorded ``dirty: true`` and ``resolve_provenance`` threw it
   away, so benchmarks measured by modified code published under the clean
   commit's identity.
3. Dirtiness was detected with ``git diff --quiet HEAD``, which does not see
   untracked files — and an untracked ``rapid_mlx/*.py`` is copied straight into
   the packaged site-packages.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from rapid_mlx.community_bench import atomic_upload, run_builder
from rapid_mlx.community_bench.publication import (
    PublicationRefused,
    ensure_publishable,
)
from rapid_mlx.community_bench.workspace import LocalRunArchive
from tests.ingestion_contract import (
    IngestionRejected,
    cross_check_against_worker,
    validate_execution,
    validate_submission,
    worker_source,
)

REPO = Path(__file__).resolve().parents[1]
ACTION = REPO / ".github/actions/desktop-releasable/action.yml"
BUILD_SIDECAR = REPO / "apps/rapid-mac/scripts/build-sidecar.sh"


@pytest.fixture(autouse=True)
def _forget_cached_provenance():
    run_builder._reset_provenance_cache()
    yield
    run_builder._reset_provenance_cache()


def _script(name: str):
    path = REPO / "apps/rapid-mac/scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _cached_run() -> dict[str, Any]:
    return json.loads(
        (Path(__file__).parent / "fixtures/community_bench_cached_run.json").read_text()
    )


# ---------------------------------------------------------------------------
# 1. The shipping workflows
# ---------------------------------------------------------------------------


def test_the_action_defaults_to_not_official() -> None:
    """A caller that forgets must produce an honest source build."""

    text = ACTION.read_text()
    block = text[text.index("  official_release:") : text.index("  require_signed:")]
    assert "default: 'false'" in block
    assert "required: false" in block


def test_the_action_passes_official_release_into_the_build() -> None:
    text = ACTION.read_text()
    assert "OFFICIAL_RELEASE: ${{ inputs.official_release }}" in text
    assert "export RAPID_MLX_OFFICIAL_RELEASE=1" in text
    assert "export RAPID_MLX_OFFICIAL_RELEASE=0" in text
    # A typo must fail the build rather than silently pick a branch.
    assert "official_release must be 'true' or 'false'" in text


def test_the_action_verifies_the_stamp_in_the_built_app() -> None:
    text = ACTION.read_text()
    assert "verify-sidecar-stamp.py" in text
    assert "Verify sidecar provenance stamp in the built app" in text


def test_the_tag_workflow_marks_only_real_tags_official() -> None:
    text = (REPO / ".github/workflows/rapid-mac-release.yml").read_text()
    assert "official_release: ${{ steps.appmeta.outputs.is_tag == 'true' }}" in text


def test_auto_release_marks_only_non_dry_runs_official() -> None:
    text = (REPO / ".github/workflows/auto-release.yml").read_text()
    # The workflow first fail-closes unless detect emits the literal string
    # ``true`` or ``false``.  Keep the positive comparison here: only the
    # validated real-release value may stamp distributable bytes as official.
    assert "official_release: ${{ needs.detect.outputs.dry_run == 'false' }}" in text


def test_the_smoke_sidecar_build_states_its_provenance() -> None:
    """The direct build-sidecar.sh call builds a throwaway smoke artifact."""

    text = (REPO / ".github/workflows/auto-release.yml").read_text()
    assert (
        "RAPID_MLX_OFFICIAL_RELEASE=0 \\\n            bash apps/rapid-mac/scripts/build-sidecar.sh"
        in text
    )


def test_every_desktop_releasable_caller_sets_official_release() -> None:
    """A new caller that omits it gets a source build, which is safe — but an
    existing release lane that omits it is the bug this test exists for."""

    for workflow in (REPO / ".github/workflows").glob("*.yml"):
        text = workflow.read_text()
        uses = text.count("uses: ./.github/actions/desktop-releasable")
        if uses:
            assert text.count("official_release:") == uses, (
                f"{workflow.name} invokes desktop-releasable {uses}x but sets "
                f"official_release {text.count('official_release:')}x"
            )


# ---------------------------------------------------------------------------
# The stamp verifier
# ---------------------------------------------------------------------------


def test_the_verifier_accepts_a_correct_release_stamp() -> None:
    verify = _script("verify-sidecar-stamp.py").verify
    assert verify({"distribution": "release"}, official=True) == []


def test_the_verifier_rejects_a_source_stamp_on_a_release_lane() -> None:
    verify = _script("verify-sidecar-stamp.py").verify
    problems = verify({"distribution": "source", "revision": "a" * 40}, official=True)
    assert problems and "expected 'release'" in problems[0]


def test_the_verifier_rejects_a_release_stamp_on_a_dry_run() -> None:
    verify = _script("verify-sidecar-stamp.py").verify
    problems = verify({"distribution": "release"}, official=False)
    assert problems and "must not claim to be a release" in problems[0]


def test_the_verifier_rejects_a_release_carrying_a_revision() -> None:
    """The closed schema states this once, so the message is the schema's."""

    verify = _script("verify-sidecar-stamp.py").verify
    problems = verify({"distribution": "release", "revision": "a" * 40}, official=True)
    assert any("must contain exactly" in p for p in problems)


def test_the_verifier_requires_a_real_sha_on_a_source_build() -> None:
    verify = _script("verify-sidecar-stamp.py").verify
    assert verify({"distribution": "source", "revision": "abc"}, official=False)
    assert (
        verify({"distribution": "source", "revision": "a" * 40}, official=False) == []
    )


def test_the_verifier_tolerates_a_dirty_source_build() -> None:
    verify = _script("verify-sidecar-stamp.py").verify
    stamp = {"distribution": "source", "revision": "a" * 40, "dirty": True}
    # A warning, not a build failure: the artifact is honest about itself.
    assert verify(stamp, official=False) == []


# ---------------------------------------------------------------------------
# 3. Dirty detection sees everything that reaches site-packages
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _dirty_probe(repo: Path) -> bool:
    """The exact command build-sidecar.sh runs."""

    out = _git(repo, "status", "--porcelain", "--untracked-files=normal")
    return bool(out.strip())


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "T")
    package = tmp_path / "rapid_mlx"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (tmp_path / ".gitignore").write_text("build/\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-m", "initial")
    return tmp_path


def test_a_clean_checkout_is_not_dirty(repo: Path) -> None:
    assert _dirty_probe(repo) is False


def test_an_untracked_python_module_makes_the_build_dirty(repo: Path) -> None:
    """The case ``git diff --quiet HEAD`` missed entirely.

    An untracked module under ``rapid_mlx/`` is installed into the packaged
    site-packages and runs in the shipped app, so the build is not the commit
    it names.
    """

    (repo / "rapid_mlx" / "experimental_patch.py").write_text("SPEEDUP = True\n")
    # The old probe saw nothing…
    assert (
        subprocess.run(
            ["git", "-C", str(repo), "diff", "--quiet", "HEAD"], capture_output=True
        ).returncode
        == 0
    )
    # …the new one does.
    assert _dirty_probe(repo) is True


def test_a_tracked_modification_makes_the_build_dirty(repo: Path) -> None:
    (repo / "rapid_mlx" / "__init__.py").write_text("# edited\n")
    assert _dirty_probe(repo) is True


def test_a_staged_change_makes_the_build_dirty(repo: Path) -> None:
    (repo / "rapid_mlx" / "staged.py").write_text("x = 1\n")
    _git(repo, "add", "rapid_mlx/staged.py")
    assert _dirty_probe(repo) is True


def test_a_deletion_makes_the_build_dirty(repo: Path) -> None:
    (repo / "rapid_mlx" / "__init__.py").unlink()
    assert _dirty_probe(repo) is True


def test_ignored_build_output_does_not_make_the_build_dirty(repo: Path) -> None:
    """Otherwise every build after the first would be 'dirty'."""

    (repo / "build").mkdir()
    (repo / "build" / "artifact.bin").write_text("x")
    assert _dirty_probe(repo) is False


def test_the_build_script_uses_the_robust_probe() -> None:
    text = BUILD_SIDECAR.read_text()
    assert "status --porcelain --untracked-files=normal" in text
    # Comments may still name the old probe (they explain why it was wrong);
    # no executable line may still run it.
    code = [
        line
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert not any("diff --quiet HEAD" in line for line in code)


# ---------------------------------------------------------------------------
# 2. A dirty build may measure, but may not publish
# ---------------------------------------------------------------------------


def test_resolve_provenance_carries_the_dirty_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stamp = tmp_path / "_build_stamp.json"
    stamp.write_text(
        json.dumps({"distribution": "source", "revision": "a" * 40, "dirty": True})
    )
    monkeypatch.setattr(run_builder, "_BUILD_STAMP", stamp)
    assert run_builder.resolve_provenance() == {
        "distribution": "source",
        "revision": "a" * 40,
        "dirty": True,
    }


def test_a_clean_source_build_is_publishable() -> None:
    ensure_publishable({"distribution": "source", "revision": "a" * 40})


def test_an_unrecorded_provenance_is_not_an_objection() -> None:
    """Runs archived before provenance was stored carry no claim either way.

    Only a genuinely absent record — ``None`` — means that. An empty or
    partial document is a damaged one, and is refused below.
    """

    ensure_publishable(None)


def test_an_empty_provenance_document_is_refused() -> None:
    """``{}`` is not "no claim"; it is a document that says nothing."""

    with pytest.raises(PublicationRefused, match="cannot be verified"):
        ensure_publishable({})


def test_a_dirty_build_refuses_to_publish() -> None:
    with pytest.raises(PublicationRefused) as error:
        ensure_publishable(
            {"distribution": "source", "revision": "a" * 40, "dirty": True}
        )
    message = str(error.value)
    # It has to say what is wrong and what to do, not just "refused".
    assert "differed from its commit" in message
    assert "saved on this Mac" in message
    assert "commit your changes, rebuild" in message


def test_preview_refuses_a_dirty_run_before_building_a_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(atomic_upload, "peek_install_id", lambda: "0123456789ab")
    with pytest.raises(PublicationRefused):
        atomic_upload.preview_run(
            _cached_run(),
            url="https://rapidmlx.com/api/benchmarks/atomic",
            provenance={
                "distribution": "source",
                "revision": "a" * 40,
                "dirty": True,
            },
        )


def test_a_dirty_run_stays_unpublishable_after_a_clean_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of storing provenance per run.

    The app is rebuilt cleanly tomorrow; the result measured by yesterday's
    modified build is still attributable to nothing, so it still cannot be
    published.
    """

    archive = LocalRunArchive(tmp_path)
    run = _cached_run()
    archive.save(run)
    archive.save_provenance(
        run["run_id"],
        {"distribution": "source", "revision": "a" * 40, "dirty": True},
    )

    # A clean rebuild: the *process* provenance is now clean…
    stamp = tmp_path / "_build_stamp.json"
    stamp.write_text(json.dumps({"distribution": "source", "revision": "b" * 40}))
    monkeypatch.setattr(run_builder, "_BUILD_STAMP", stamp)
    assert "dirty" not in run_builder.resolve_provenance()

    # …but the stored run still knows how it was measured.
    stored = archive.provenance(run["run_id"])
    assert stored["dirty"] is True
    with pytest.raises(PublicationRefused):
        ensure_publishable(stored)


def test_provenance_is_stored_per_run(tmp_path: Path) -> None:
    archive = LocalRunArchive(tmp_path)
    run = _cached_run()
    archive.save(run)
    archive.save_provenance(
        run["run_id"],
        {"distribution": "source", "revision": "a" * 40, "dirty": True},
    )
    assert archive.provenance(run["run_id"])["dirty"] is True
    assert archive.provenance("00000000-0000-4000-8000-000000000000") is None


def test_the_dirty_state_is_not_hidden_by_the_projection() -> None:
    """The projection narrows the model block; it must not be a place where a
    dirty build quietly becomes publishable."""

    from rapid_mlx.community_bench.publication import project_run_for_publication

    public, withheld = project_run_for_publication(_cached_run())
    # The projection does not touch provenance at all — the refusal happens
    # before it, in `preview_run`.
    assert all("execution" not in fact.path for fact in withheld)
    assert public["execution"] == _cached_run()["execution"]


# ---------------------------------------------------------------------------
# 4. The mirrored execution-runtime contract
# ---------------------------------------------------------------------------


def _runtime(**overrides: Any) -> dict[str, Any]:
    runtime = {
        "distribution": "release",
        "rapid_mlx": "0.13.4",
        "mlx": "0.32.2",
        "python": "3.12.13",
    }
    runtime.update(overrides)
    return runtime


def _execution(runtime: dict[str, Any]) -> dict[str, Any]:
    execution = json.loads(json.dumps(_cached_run()["execution"]))
    execution["runtime"] = runtime
    return execution


def test_a_release_runtime_validates() -> None:
    validate_execution(_execution(_runtime()), "text_generation")


def test_a_release_runtime_may_not_carry_a_revision() -> None:
    with pytest.raises(IngestionRejected, match="rapid_mlx_revision"):
        validate_execution(
            _execution(_runtime(rapid_mlx_revision="a" * 40)), "text_generation"
        )


def test_a_source_runtime_requires_a_revision() -> None:
    with pytest.raises(IngestionRejected, match="rapid_mlx_revision"):
        validate_execution(
            _execution(_runtime(distribution="source")), "text_generation"
        )


def test_a_source_runtime_with_a_valid_revision_validates() -> None:
    validate_execution(
        _execution(_runtime(distribution="source", rapid_mlx_revision="a" * 40)),
        "text_generation",
    )


@pytest.mark.parametrize("revision", ["A" * 40, "a" * 39, "a" * 41, "z" * 40, "", "  "])
def test_a_source_runtime_rejects_a_malformed_revision(revision: str) -> None:
    with pytest.raises(IngestionRejected, match="rapid_mlx_revision"):
        validate_execution(
            _execution(_runtime(distribution="source", rapid_mlx_revision=revision)),
            "text_generation",
        )


def test_an_unknown_runtime_key_is_refused() -> None:
    with pytest.raises(IngestionRejected, match="not upload-allowlisted"):
        validate_execution(_execution(_runtime(dirty=True)), "text_generation")


def test_an_unknown_distribution_is_refused() -> None:
    with pytest.raises(IngestionRejected, match="distribution"):
        validate_execution(
            _execution(_runtime(distribution="nightly")), "text_generation"
        )


def test_a_missing_required_runtime_key_is_refused() -> None:
    runtime = _runtime()
    del runtime["mlx"]
    with pytest.raises(IngestionRejected, match="execution.runtime.mlx is required"):
        validate_execution(_execution(runtime), "text_generation")


def test_optional_package_versions_may_be_absent_or_present() -> None:
    validate_execution(_execution(_runtime()), "text_generation")
    validate_execution(
        _execution(_runtime(mlx_lm="0.31.3", mlx_vlm="0.6.17", mflux="0.19.0")),
        "text_generation",
    )


def test_the_complete_projected_submission_passes_the_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model block AND execution runtime, over the exact bytes that would be
    sent for a clean source build."""

    run = _cached_run()
    run["execution"]["runtime"] = _runtime(
        distribution="source", rapid_mlx_revision="a" * 40
    )
    monkeypatch.setattr(atomic_upload, "peek_install_id", lambda: "0123456789ab")
    preview = atomic_upload.preview_run(
        run,
        url="https://rapidmlx.com/api/benchmarks/atomic",
        provenance={"distribution": "source", "revision": "a" * 40},
    )
    validate_submission(json.loads(preview["payload_json"]))


def test_the_mirror_still_agrees_with_a_local_worker_checkout() -> None:
    if worker_source() is None:
        pytest.skip("no rapidmlx.com worker checkout on this machine")
    assert cross_check_against_worker() == []
