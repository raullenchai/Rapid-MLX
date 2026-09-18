# SPDX-License-Identifier: Apache-2.0
"""Provenance must fail closed, at every archived outcome and every read.

Two fail-open paths, one consequence. A run archived *without* its provenance,
and a provenance file that exists but cannot be read, both looked exactly like
a legacy run — and legacy runs may publish. So a dirty build's numbers could
reach the leaderboard by way of a swallowed write error or a corrupt file.

The invariant these tests hold: a *missing* provenance file can only mean
"archived before provenance existed". It can never mean "we tried and failed",
and it can never be produced by damaging a file.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rapid_mlx.community_bench import atomic_upload
from rapid_mlx.community_bench import cli as community_cli
from rapid_mlx.community_bench.publication import (
    PublicationRefused,
    ensure_publishable,
)
from rapid_mlx.community_bench.workspace import (
    LocalRunArchive,
    ProvenanceUnreadable,
    validate_provenance,
)

REPO = Path(__file__).resolve().parents[1]
CLEAN_SOURCE = {"distribution": "source", "revision": "a" * 40}
DIRTY_SOURCE = {"distribution": "source", "revision": "a" * 40, "dirty": True}


def _run() -> dict[str, Any]:
    return json.loads(
        (Path(__file__).parent / "fixtures/community_bench_cached_run.json").read_text()
    )


# ---------------------------------------------------------------------------
# 1. One operation: provenance before the run becomes visible
# ---------------------------------------------------------------------------


def test_a_run_is_never_visible_without_its_provenance(tmp_path: Path) -> None:
    archive = LocalRunArchive(tmp_path)
    run = _run()
    archive.save_with_provenance(run, CLEAN_SOURCE)
    assert archive.get(run["run_id"])["run_id"] == run["run_id"]
    assert archive.provenance(run["run_id"]) == CLEAN_SOURCE


def test_a_provenance_write_failure_leaves_no_archived_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ordering guarantee, stated as a test.

    If the provenance write fails, the run file must not exist — otherwise it
    is indistinguishable from a legacy run and may publish.
    """

    archive = LocalRunArchive(tmp_path)
    run = _run()

    def explode(directory: Path, name: str, value: dict) -> Path:
        if directory == archive.provenance_dir:
            raise OSError("read-only volume")
        raise AssertionError("the run was written before its provenance")

    monkeypatch.setattr(LocalRunArchive, "_atomic_save", staticmethod(explode))
    with pytest.raises(OSError, match="read-only volume"):
        archive.save_with_provenance(run, CLEAN_SOURCE)

    assert not (archive.runs_dir / f"{run['run_id']}.json").exists()
    with pytest.raises(Exception):
        archive.get(run["run_id"])


def test_a_provenance_write_failure_is_not_swallowed_by_the_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runner used to `except Exception: pass` around the provenance save.

    A completed benchmark then appeared in the archive with no build record —
    the exact state that lets a dirty run publish.
    """

    archive = LocalRunArchive(tmp_path)
    original = LocalRunArchive._atomic_save

    def explode(directory: Path, name: str, value: dict) -> Path:
        if directory == archive.provenance_dir:
            raise OSError("disk full")
        return original(directory, name, value)

    monkeypatch.setattr(LocalRunArchive, "_atomic_save", staticmethod(explode))
    with pytest.raises(OSError, match="disk full"):
        archive.save_with_provenance(_run(), DIRTY_SOURCE)
    # Nothing publishable was produced.
    assert list(archive.runs_dir.glob("*.json")) == []


def test_an_orphan_provenance_file_is_harmless(tmp_path: Path) -> None:
    """The acceptable leftover: provenance with no run.

    Nothing reads it without a run, and archiving that run id later overwrites
    it.
    """

    archive = LocalRunArchive(tmp_path)
    archive.save_provenance("00000000-0000-4000-8000-000000000001", DIRTY_SOURCE)
    assert list(archive.runs_dir.glob("*.json")) == []
    run = _run()
    archive.save_with_provenance(run, CLEAN_SOURCE)
    assert archive.provenance(run["run_id"]) == CLEAN_SOURCE


def test_the_runner_archives_every_outcome_with_provenance() -> None:
    """Completed, failed and cancelled all go through the one operation.

    The service accepts failed and cancelled submissions, so an archived
    failure is a publishable result and needs the same build record.
    """

    source = (REPO / "rapid_mlx/community_bench/local_runner.py").read_text()
    code = [
        line
        for line in source.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    saves = [line for line in code if "destination.save" in line]
    assert saves, "no archive writes found"
    assert all("save_with_provenance" in line for line in saves), saves
    # And the swallow is gone.
    assert "provenance must not fail a save" not in source


# ---------------------------------------------------------------------------
# 2. Absent legacy provenance vs unreadable provenance
# ---------------------------------------------------------------------------


def test_a_genuinely_absent_file_reads_as_none(tmp_path: Path) -> None:
    archive = LocalRunArchive(tmp_path)
    assert archive.provenance("00000000-0000-4000-8000-000000000002") is None


def test_malformed_json_fails_closed(tmp_path: Path) -> None:
    archive = LocalRunArchive(tmp_path)
    archive.provenance_dir.mkdir(parents=True)
    (archive.provenance_dir / "run-1.json").write_text("{not json")
    with pytest.raises(ProvenanceUnreadable, match="not valid JSON"):
        archive.provenance("run-1")


def test_non_object_json_fails_closed(tmp_path: Path) -> None:
    archive = LocalRunArchive(tmp_path)
    archive.provenance_dir.mkdir(parents=True)
    (archive.provenance_dir / "run-1.json").write_text("[1, 2, 3]")
    with pytest.raises(ProvenanceUnreadable, match="not a JSON object"):
        archive.provenance("run-1")


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_an_unreadable_file_fails_closed(tmp_path: Path) -> None:
    archive = LocalRunArchive(tmp_path)
    archive.provenance_dir.mkdir(parents=True)
    path = archive.provenance_dir / "run-1.json"
    path.write_text(json.dumps(CLEAN_SOURCE))
    path.chmod(0o000)
    try:
        with pytest.raises(ProvenanceUnreadable, match="could not be read"):
            archive.provenance("run-1")
    finally:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)


@pytest.mark.parametrize(
    "document",
    [
        {},
        {"revision": "a" * 40},
        {"distribution": "nightly"},
        {"distribution": "source"},
        {"distribution": "source", "revision": "abc"},
        {"distribution": "source", "revision": "A" * 40},
        {"distribution": "release", "revision": "zz"},
        {"distribution": "source", "revision": "a" * 40, "dirty": "true"},
        {"distribution": "source", "revision": "a" * 40, "dirty": 1},
    ],
)
def test_an_invalid_provenance_shape_fails_closed(
    tmp_path: Path, document: dict
) -> None:
    """Validate the shape, not just ``dirty is True``.

    ``dirty: "true"`` is the one to notice: a truthiness check would have read
    that string as clean and published a modified build.
    """

    archive = LocalRunArchive(tmp_path)
    archive.provenance_dir.mkdir(parents=True)
    (archive.provenance_dir / "run-1.json").write_text(json.dumps(document))
    with pytest.raises(ProvenanceUnreadable):
        archive.provenance("run-1")


def test_valid_shapes_are_accepted() -> None:
    validate_provenance({"distribution": "release"})
    validate_provenance(CLEAN_SOURCE)
    validate_provenance(DIRTY_SOURCE)
    validate_provenance(
        {"distribution": "source", "revision": "a" * 40, "dirty": False}
    )


def test_ensure_publishable_refuses_a_damaged_document() -> None:
    with pytest.raises(PublicationRefused, match="cannot be verified"):
        ensure_publishable(
            {"distribution": "source", "revision": "a" * 40, "dirty": "true"}
        )


# ---------------------------------------------------------------------------
# The real CLI preview/share path
# ---------------------------------------------------------------------------


def _cli_share(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preview: bool = True):
    archive = LocalRunArchive(tmp_path)
    monkeypatch.setattr(
        community_cli.LocalRunArchive, "default", classmethod(lambda cls: archive)
    )
    return archive


def _share_args(run_id: str, *, preview: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        benchmark_action="share",
        run_id=run_id,
        json=True,
        yes=True,
        preview=preview,
        install_id=None,
        payload_digest=None,
        body_digest=None,
        target=None,
    )


def test_cli_preview_refuses_a_corrupt_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Through the real command, not `ensure_publishable` directly."""

    archive = _cli_share(tmp_path, monkeypatch)
    run = _run()
    archive.save(run)
    archive.provenance_dir.mkdir(parents=True, exist_ok=True)
    (archive.provenance_dir / f"{run['run_id']}.json").write_text("{corrupt")

    assert community_cli.benchmark_command(_share_args(run["run_id"])) != 0
    document = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert document["refused"] is True
    assert "cannot be verified" in document["error"]
    assert document["saved"] is False


def test_cli_preview_refuses_a_dirty_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = _cli_share(tmp_path, monkeypatch)
    run = _run()
    archive.save_with_provenance(run, DIRTY_SOURCE)

    assert community_cli.benchmark_command(_share_args(run["run_id"])) != 0
    document = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert document["refused"] is True
    assert "differed from its commit" in document["error"]


def test_cli_upload_refuses_a_dirty_run_before_any_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = _cli_share(tmp_path, monkeypatch)
    run = _run()
    archive.save_with_provenance(run, DIRTY_SOURCE)

    def no_network(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("a refused publication contacted the service")

    monkeypatch.setattr(atomic_upload, "post_submission", no_network)

    assert (
        community_cli.benchmark_command(_share_args(run["run_id"], preview=False)) != 0
    )
    document = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert document["refused"] is True


def test_cli_preview_allows_a_clean_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = _cli_share(tmp_path, monkeypatch)
    run = _run()
    archive.save_with_provenance(run, CLEAN_SOURCE)
    monkeypatch.setattr(atomic_upload, "peek_install_id", lambda: "0123456789ab")

    assert community_cli.benchmark_command(_share_args(run["run_id"])) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["target"]
    assert "payload_json" in payload


def test_cli_preview_allows_a_legacy_run_with_no_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The explicitly chosen legacy policy: a genuinely absent record is not
    an objection. Runs archived before provenance existed stay publishable."""

    archive = _cli_share(tmp_path, monkeypatch)
    run = _run()
    archive.save(run)  # no provenance, as an older build would have left it
    assert archive.provenance(run["run_id"]) is None
    monkeypatch.setattr(atomic_upload, "peek_install_id", lambda: "0123456789ab")

    assert community_cli.benchmark_command(_share_args(run["run_id"])) == 0
    assert "payload_json" in json.loads(capsys.readouterr().out)


def test_a_failed_outcome_keeps_its_provenance_and_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The server accepts failed and cancelled submissions, so they need the
    same refusal a completed run gets."""

    archive = _cli_share(tmp_path, monkeypatch)
    failed = _run()
    failed["outcome"] = {"status": "failed", "failure_code": "runtime_error"}
    archive.save_with_provenance(failed, DIRTY_SOURCE)

    assert archive.provenance(failed["run_id"])["dirty"] is True
    assert community_cli.benchmark_command(_share_args(failed["run_id"])) != 0
    document = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert document["refused"] is True


def test_a_dirty_run_stays_refused_through_the_cli_after_a_clean_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rapid_mlx.community_bench import run_builder

    archive = _cli_share(tmp_path, monkeypatch)
    run = _run()
    archive.save_with_provenance(run, DIRTY_SOURCE)

    # Rebuild cleanly: the process provenance is now clean…
    run_builder._reset_provenance_cache()
    stamp = tmp_path / "_build_stamp.json"
    stamp.write_text(json.dumps({"distribution": "source", "revision": "b" * 40}))
    monkeypatch.setattr(run_builder, "_BUILD_STAMP", stamp)
    assert "dirty" not in run_builder.resolve_provenance()
    run_builder._reset_provenance_cache()

    # …and the old result is still refused.
    assert community_cli.benchmark_command(_share_args(run["run_id"])) != 0
    assert (
        json.loads(capsys.readouterr().err.strip().splitlines()[-1])["refused"] is True
    )


# ---------------------------------------------------------------------------
# 3. Dirty official release builds
# ---------------------------------------------------------------------------


def _stamp_writer():
    import importlib.util

    path = REPO / "apps/rapid-mac/scripts/write-sidecar-stamp.py"
    spec = importlib.util.spec_from_file_location("write_sidecar_stamp", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _verifier():
    import importlib.util

    path = REPO / "apps/rapid-mac/scripts/verify-sidecar-stamp.py"
    spec = importlib.util.spec_from_file_location("verify_sidecar_stamp", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_a_clean_official_build_is_stamped_release() -> None:
    assert _stamp_writer().build_stamp("1", "a" * 40, "0") == {
        "distribution": "release"
    }


def test_the_stamp_writer_refuses_official_plus_dirty() -> None:
    """Defence in depth: build-sidecar.sh refuses first, but a release stamp
    carries no revision, so a dishonest one leaves no trace at all."""

    with pytest.raises(SystemExit, match="modified working tree"):
        _stamp_writer().build_stamp("1", "a" * 40, "1")


def test_the_build_script_refuses_official_plus_dirty() -> None:
    text = (REPO / "apps/rapid-mac/scripts/build-sidecar.sh").read_text()
    official = text[
        text.index('if [[ "$OFFICIAL_RELEASE" == "1" ]]') : text.index(
            'else\n    echo "    distribution: source'
        )
    ]
    assert '"$SIDECAR_DIRTY" == "1"' in official
    assert "refusing to build an OFFICIAL RELEASE from a modified" in official
    assert "exit 1" in official


def test_being_at_the_candidate_sha_does_not_excuse_a_dirty_tree() -> None:
    """A SHA names a commit, not the tree that was compiled.

    The candidate gate checks HEAD == candidate_sha; that check passes for a
    checkout with uncommitted edits, so it cannot be the thing that authorises
    a release stamp.
    """

    # The writer takes a perfectly valid candidate revision and still refuses.
    with pytest.raises(SystemExit, match="modified working tree"):
        _stamp_writer().build_stamp(
            "1", "60be718538e44dd5497a75be419501080cbfff32", "1"
        )


def test_the_verifier_rejects_a_release_stamp_with_a_dirty_flag() -> None:
    problems = _verifier().verify(
        {"distribution": "release", "dirty": True}, official=True
    )
    # `dirty` is simply not a field a release stamp may carry.
    assert any("must contain exactly" in p for p in problems)


def test_the_verifier_rejects_a_release_stamp_with_dirty_false() -> None:
    """Even ``dirty: false`` is unexpected on a release stamp: it means
    something wrote a field the release path never writes."""

    problems = _verifier().verify(
        {"distribution": "release", "dirty": False}, official=True
    )
    assert problems


def test_the_verifier_rejects_unexpected_release_fields() -> None:
    problems = _verifier().verify(
        {"distribution": "release", "note": "hello"}, official=True
    )
    assert any("must contain exactly" in p for p in problems)


def test_the_verifier_accepts_an_exact_release_stamp() -> None:
    assert _verifier().verify({"distribution": "release"}, official=True) == []


def test_a_dirty_local_source_build_still_runs_and_saves(tmp_path: Path) -> None:
    """Development keeps working. Only publishing is refused."""

    archive = LocalRunArchive(tmp_path)
    run = _run()
    archive.save_with_provenance(run, DIRTY_SOURCE)
    assert archive.get(run["run_id"])["run_id"] == run["run_id"]
    assert [r["run_id"] for r in archive.list()] == [run["run_id"]]
    with pytest.raises(PublicationRefused):
        ensure_publishable(archive.provenance(run["run_id"]))
