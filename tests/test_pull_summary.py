# SPDX-License-Identifier: Apache-2.0
"""Tests for the 0.9.7 ``rapid-mlx pull`` post-download summary line.

A ~6 GB pull that succeeds silently leaves the user wondering "did
that actually finish, and how much disk did I just burn?". The
summary line printed by ``pull_command`` answers both in one line:

    Downloaded <repo_id> — <size with units> in <duration with units>

These tests pin three things and three things only:

1. The summary line is emitted on the HuggingFace-fallback success
   path (the common case once R2 misses).
2. The summary line is emitted on the R2 mirror success path.
3. The summary line is NOT emitted when the pull fails with a 404 —
   we exit before we'd otherwise mislead the user.

The actual HuggingFace download (``snapshot_download``) and the R2
prefetch (``_try_mirror_prefetch``) are mocked; we only exercise the
summary code path in ``pull_command``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from vllm_mlx import cli


def _make_fake_snapshot(root: Path, total_bytes: int) -> Path:
    """Create a snapshot dir on disk with one file of ``total_bytes`` bytes."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.safetensors").write_bytes(b"\x00" * total_bytes)
    return root


def _hf_snapshot_layout(
    repo_id: str,
    revision: str,
    root: Path,
    *,
    blob_name: str = "abcdef",
    already_cached: bool = False,
) -> tuple[Path, Path]:
    """Build a deterministic HF cache entry ``root/models--<id>/`` for the
    HF-fallback transfer-account tests, keyed on the BLOB store.

    Returns ``(cache_root, blob_dir)``. Points ``refs/main`` at ``revision``
    and creates ``blobs/``. When ``already_cached`` a blob is already present
    (a warm, fully-cached pull leaves ``blobs/`` untouched — the transfer seam
    compares the blob inventory before/after); when not, ``blobs/`` starts
    empty and the test's ``snapshot_download`` mock is expected to create a
    blob during the pull (a fetch). ``repo_id`` maps to no catalog subfolder.
    """
    cache_root = root / "hub"
    repo_root = cache_root / f"models--{repo_id.replace('/', '--')}"
    (repo_root / "refs").mkdir(parents=True, exist_ok=True)
    (repo_root / "refs" / "main").write_text(revision)
    blob_dir = repo_root / "blobs"
    blob_dir.mkdir(parents=True, exist_ok=True)
    if already_cached:
        (blob_dir / blob_name).write_bytes(b"\x00" * 2048)
    return cache_root, blob_dir


def _looks_like_size(token: str) -> bool:
    """Loose acceptance of either SI (``GB``) or IEC (``GiB``) suffixes.

    The task spec says ``X.Y GB`` but the project's shared
    ``_format_bytes`` helper renders IEC (``GiB``); we reuse it per
    the "do not invent a new size formatter" rule, so the test
    accepts whichever the helper produces.
    """
    return any(
        unit in token
        for unit in ("B", "KB", "KiB", "MB", "MiB", "GB", "GiB", "TB", "TiB")
    )


def _summary_line(captured: str) -> str:
    for line in captured.splitlines():
        if "Downloaded" in line and "in" in line:
            return line
    raise AssertionError(f"summary line missing from stdout, got:\n{captured!r}")


def test_summary_printed_on_hf_success(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HF-fallback path prints ``Downloaded ... — <size> in <duration>``."""
    revision = "abc123" * 6
    cache_root, blob_dir = _hf_snapshot_layout(
        "mlx-community/Qwen3-0.6B-4bit", revision, tmp_path, already_cached=False
    )
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    def _download(*_args, **_kwargs):
        # Not cached at entry (blobs/ empty); the pull transfers bytes — the
        # before-inventory is empty and the after-side (this new blob) differs,
        # so the summary is a download.
        (blob_dir / "cafe").write_bytes(b"\x00" * 2048)
        return str(blob_dir.parent / "snapshots" / revision)

    args = argparse.Namespace(model="mlx-community/Qwen3-0.6B-4bit")

    with (
        patch.object(cli, "_try_mirror_prefetch", return_value=False),
        patch("huggingface_hub.snapshot_download", side_effect=_download),
    ):
        cli.pull_command(args)

    out = capsys.readouterr().out
    line = _summary_line(out)

    # Model name appears verbatim.
    assert "mlx-community/Qwen3-0.6B-4bit" in line
    # Some size token with a recognized unit.
    parts = line.split()
    assert any(_looks_like_size(p) for p in parts), line
    # Some duration token ending in 's' (e.g. ``4.2s`` or ``1m 23s``).
    assert any(p.endswith("s") and p[0].isdigit() for p in parts), line


def test_hf_cached_fallback_reports_verified(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cached no-op on the HF-fallback path is labelled ``Already cached``.

    Even when the mirror miss forces the HF fallback, a pull that transfers
    zero bytes must NOT print ``Downloaded``. The transfer account is the
    stable BLOB-store inventory BEFORE vs AFTER the pull (Codex #2392, no
    huggingface_hub tqdm-progress internals): a warm, fully-cached pull leaves
    ``blobs/`` untouched, so before == after and it reports verified.
    """
    revision = "abc123" * 6
    cache_root, blob_dir = _hf_snapshot_layout(
        "mlx-community/Qwen3-0.6B-4bit", revision, tmp_path, already_cached=True
    )
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    def _download(*_args, **_kwargs):
        # A warm pull touches NOTHING in blobs/: the blobs were already there.
        # before == after (the blob inventory) -> verified.
        return str(blob_dir.parent / "snapshots" / revision)

    args = argparse.Namespace(model="mlx-community/Qwen3-0.6B-4bit")

    with (
        patch.object(cli, "_try_mirror_prefetch", return_value=False),
        patch("huggingface_hub.snapshot_download", side_effect=_download),
    ):
        cli.pull_command(args)

    out = capsys.readouterr().out
    assert "Already cached" in out, out
    assert "verified (nothing to download)" in out, out
    assert "Downloaded" not in out, out


def test_partial_mirror_fetch_combines_into_fallback_verdict(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mirror that fetched SOME bytes before failing must still report
    ``Downloaded``, even though the snapshot_download that follows is a no-op
    (Codex #2353).

    When the mirror returns False (partial/failed) it still records
    ``out["network_fetch"] = True`` for the blobs it DID fetch. The HF-fallback
    baseline is captured AFTER the mirror wrote those blobs, so its own
    before/after blob comparison is a no-op -> ""Already cached"". The mirror's
    transfer state must therefore be OR'd in: bytes did cross the wire this
    invocation, so the summary says ``Downloaded``.
    """
    revision = "abc123" * 6
    cache_root, blob_dir = _hf_snapshot_layout(
        "mlx-community/Qwen3-0.6B-4bit", revision, tmp_path, already_cached=True
    )
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    def _mirror_partial(model_name: str, *, out=None, allow_patterns=None) -> bool:
        # Mirror fetched some blobs (network_fetch=True) but ultimately
        # returned False (a file missed both R2 and HF).
        if out is not None:
            out["network_fetch"] = True
        return False

    def _download(*_args, **_kwargs):
        # The fallback is a no-op: the mirror already laid down the blobs, so
        # this invocation's before/after blob comparison is unchanged.
        return str(blob_dir.parent / "snapshots" / revision)

    args = argparse.Namespace(model="mlx-community/Qwen3-0.6B-4bit")

    with (
        patch.object(cli, "_try_mirror_prefetch", side_effect=_mirror_partial),
        patch("huggingface_hub.snapshot_download", side_effect=_download),
    ):
        cli.pull_command(args)

    out = capsys.readouterr().out
    assert "Downloaded" in out, out
    assert "Already cached" not in out, out


def test_hf_fetch_zero_byte_file_counts_as_download(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fetched zero-byte file is a network fetch, not a cache hit.

    codex round-4 BLOCKING #3 carried over to the blob-inventory seam: a
    fetched zero-byte file creates a NEW blob (the file did not exist before),
    so the blob inventory changes even though the file carries no bytes, and
    the summary says ``Downloaded``.

    The cache is seeded with ONE pre-existing blob so ``_before != ()`` —
    otherwise an empty pre-pull fingerprint already forces ``_was_cached = False``
    and the test would pass without ever exercising the blob transition.
    """
    revision = "abc123" * 6
    cache_root, blob_dir = _hf_snapshot_layout(
        "mlx-community/Qwen3-0.6B-4bit", revision, tmp_path, already_cached=True
    )
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    def _download(*_args, **_kwargs):
        # The pull fetched a NEW file that is 0 bytes: a distinct empty blob
        # appears (the store starts non-empty via the seeded blob). The
        # transition is what must drive the "Downloaded" verdict.
        (blob_dir / "deadbeef").write_bytes(b"")
        return str(blob_dir.parent / "snapshots" / revision)

    args = argparse.Namespace(model="mlx-community/Qwen3-0.6B-4bit")

    with (
        patch.object(cli, "_try_mirror_prefetch", return_value=False),
        patch("huggingface_hub.snapshot_download", side_effect=_download),
    ):
        cli.pull_command(args)

    out = capsys.readouterr().out
    assert "Downloaded" in out, out
    assert "Already cached" not in out, out


def test_hf_fallback_transfers_bytes_as_download(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real HF-fallback transfer is a download: the blob inventory changes
    across the pull (a new blob appears), so before != after -> Downloaded."""
    revision = "abc123" * 6
    cache_root, blob_dir = _hf_snapshot_layout(
        "mlx-community/Qwen3-0.6B-4bit", revision, tmp_path, already_cached=False
    )
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    def _download(*_args, **_kwargs):
        # A real fetch materializes a new 2048-byte blob.
        (blob_dir / "cafebabe").write_bytes(b"\x00" * 2048)
        return str(blob_dir.parent / "snapshots" / revision)

    args = argparse.Namespace(model="mlx-community/Qwen3-0.6B-4bit")

    with (
        patch.object(cli, "_try_mirror_prefetch", return_value=False),
        patch("huggingface_hub.snapshot_download", side_effect=_download),
    ):
        cli.pull_command(args)

    out = capsys.readouterr().out
    assert "Downloaded" in out, out
    assert "Already cached" not in out, out


def test_summary_printed_on_mirror_success(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R2-mirror success path also prints the summary line.

    We point the HF cache root at ``tmp_path`` via the ``HF_HUB_CACHE``
    constant so ``pull_command`` resolves the snapshot dir under our fixture.
    The mirror mock simulates an actual fetch: it creates ``refs/main`` and
    populates the snapshot ONLY during the pull, reporting ``out[
    transferred_bytes] = 4096`` so the summary reports a real download.
    """
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "abc123" * 6  # 36 hex chars; shape doesn't matter for the test

    cache_root = tmp_path / "hub"
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    def _mirror_fetch(model_name: str, *, out=None, allow_patterns=None) -> bool:
        """Simulate the mirror downloading the snapshot during this pull."""
        repo_root = cache_root / "models--mlx-community--Qwen3-0.6B-4bit"
        refs_dir = repo_root / "refs"
        refs_dir.mkdir(parents=True, exist_ok=True)
        (refs_dir / "main").write_text(revision)
        snapshot_dir = repo_root / "snapshots" / revision
        _make_fake_snapshot(snapshot_dir, total_bytes=4096)
        if out is not None:
            out["network_fetch"] = True
        return True

    args = argparse.Namespace(model=repo_id)

    with patch.object(cli, "_try_mirror_prefetch", side_effect=_mirror_fetch):
        cli.pull_command(args)

    out = capsys.readouterr().out
    line = _summary_line(out)
    assert repo_id in line
    parts = line.split()
    assert any(_looks_like_size(p) for p in parts), line
    assert any(p.endswith("s") and p[0].isdigit() for p in parts), line


def test_cached_pull_reports_verified_not_downloaded(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fully-cached pull must say the cache was verified, not ``Downloaded``.

    Issue #2349: ``rapid-mlx pull <cached-model>`` printed
    ``Downloaded ... in 3.8s`` after "[10/10] ... cached". The final outcome
    now reserves ``Downloaded`` + transfer timing for a pull that actually
    fetched bytes; an already-complete snapshot reports the cache was reused.
    """
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "abc123" * 6
    cache_root = tmp_path / "hub"
    repo_root = cache_root / "models--mlx-community--Qwen3-0.6B-4bit"
    refs_dir = repo_root / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "main").write_text(revision)
    snapshot_dir = repo_root / "snapshots" / revision
    _make_fake_snapshot(snapshot_dir, total_bytes=4096)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    args = argparse.Namespace(model=repo_id)

    # The mirror reports it fetched ZERO bytes this pull (every file was
    # already cached) -> "verified (nothing to download)", not "Downloaded".
    def _mirror_already_cached(
        model_name: str, *, out=None, allow_patterns=None
    ) -> bool:
        if out is not None:
            out["network_fetch"] = False
        return True

    with patch.object(cli, "_try_mirror_prefetch", side_effect=_mirror_already_cached):
        cli.pull_command(args)

    out = capsys.readouterr().out
    assert "Already cached" in out, out
    assert "Downloaded" not in out, out
    assert "verified (nothing to download)" in out, out


def test_moved_main_reported_as_download(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local snapshot may be complete while ``main`` advanced remote-side.

    The codex BLOCKING on #2349: ``is_repo_cached`` (local presence) alone is
    wrong for a mutable ``main`` — the subsequent mirror/HF call can transfer
    new files while the summary falsely says "nothing to download". The
    summary must reflect the ACTUAL transfer. Here a stale rev_A is fully
    cached pre-pull, then the mirror reports it fetched NEWER files (rev_B) as
    ``out["transferred_bytes"]``; the pull is reported as a download.
    """
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    stale_rev = "aaaaaa" * 6
    head_rev = "bbbbbb" * 6
    cache_root = tmp_path / "hub"
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    repo_root = cache_root / "models--mlx-community--Qwen3-0.6B-4bit"
    refs_dir = repo_root / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    # A fully-cached STALE rev exists before the pull.
    (refs_dir / "main").write_text(stale_rev)
    _make_fake_snapshot(repo_root / "snapshots" / stale_rev, total_bytes=4096)

    def _mirror_fetch(model_name: str, *, out=None, allow_patterns=None) -> bool:
        # Remote main advanced mid-pull: refs/main now points at a NEWER rev
        # whose snapshot bytes were actually fetched over the wire.
        (refs_dir / "main").write_text(head_rev)
        _make_fake_snapshot(repo_root / "snapshots" / head_rev, total_bytes=4096)
        if out is not None:
            out["network_fetch"] = True
        return True

    args = argparse.Namespace(model=repo_id)

    with patch.object(cli, "_try_mirror_prefetch", side_effect=_mirror_fetch):
        cli.pull_command(args)

    out = capsys.readouterr().out
    assert "Downloaded" in out, out
    assert "Already cached" not in out, out


def test_summary_not_printed_on_404(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A 404 must bail before the summary — we don't lie about success.

    ``pull_command`` matches 404 via either ``RepositoryNotFoundError``
    isinstance OR a ``"404" / "not found"`` substring on the exception
    string, so a plain ``Exception("404 Client Error")`` is enough to
    drive the error branch without constructing HF's response-bound
    exception class.
    """
    args = argparse.Namespace(model="mlx-community/does-not-exist")

    with (
        patch.object(cli, "_try_mirror_prefetch", return_value=False),
        patch(
            "huggingface_hub.snapshot_download",
            side_effect=Exception("404 Client Error"),
        ),
        pytest.raises(SystemExit) as excinfo,
    ):
        cli.pull_command(args)

    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "Downloaded" not in out, out


def test_format_pull_duration_units() -> None:
    """Sub-minute keeps decimals; ``>=60s`` switches to ``m`` + ``s``."""
    assert cli._format_pull_duration(0.0) == "0.0s"
    assert cli._format_pull_duration(4.2) == "4.2s"
    assert cli._format_pull_duration(59.9) == "59.9s"
    assert cli._format_pull_duration(60.0) == "1m 0s"
    assert cli._format_pull_duration(125.0) == "2m 5s"
    # Rounding rule: 119.9s reads as 2m 0s, not 1m 59s.
    assert cli._format_pull_duration(119.9) == "2m 0s"


def test_blob_identifier_is_a_stable_transfer_seam(tmp_path: Path) -> None:
    """``_blob_identifier`` fingerprints the BLOB store so a before/after
    comparison classifies the pull without huggingface_hub tqdm internals
    (Codex #2392). A new/modified blob is the only thing meaning bytes crossed
    the wire; re-linking snapshot symlinks to existing blobs (unchanged) is a
    cache hit. The cases directly mirror the old byte-bar verdicts.
    """
    # No cache entry / empty blob store -> empty fingerprint (fresh pull).
    assert cli._blob_identifier(None) == ()
    assert cli._blob_identifier(tmp_path / "does-not-exist") == ()

    # A warm blob store left untouched: before == after -> verified (no
    # transfer), even if snapshot symlinks were recreated (we only fingerprint
    # blobs/).
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "aabbcc").write_bytes(b"\x00" * 2048)
    before = cli._blob_identifier(tmp_path)
    after = cli._blob_identifier(tmp_path)
    assert before == after != ()

    # A real fetch materializes a new blob -> the inventory changes -> Download.
    (blobs / "config").write_bytes(b"{}")
    assert cli._blob_identifier(tmp_path) != before

    # A fetched ZERO-byte file (new, empty blob) still changes the inventory
    # -> Download, never a false cache hit (carried from codex round-4 #3).
    # Compare against the fingerprint captured immediately BEFORE this write,
    # not the original ``before`` (the ``config`` addition above already made
    # ``before`` unequal, so that comparison would be tautological).
    pre_empty = cli._blob_identifier(tmp_path)
    (blobs / "empty").write_bytes(b"")
    assert cli._blob_identifier(tmp_path) != pre_empty

    # A REPAIR of an existing blob (mtime/size change) -> Download (the exact
    # case a snapshot-symlink fingerprint would miss). Snapshot the fingerprint
    # immediately BEFORE the rewrite and compare against THAT, not the original
    # ``before`` — the intervening ``config``/``empty`` additions already make
    # ``before`` unequal, so comparing to ``before`` would not prove the
    # modification itself is detected.
    pre_repair = cli._blob_identifier(tmp_path)
    (blobs / "aabbcc").write_bytes(b"\x00" * 4096)
    repaired = cli._blob_identifier(tmp_path)
    assert repaired != pre_repair

    # .incomplete* scratch churn must NOT change the transfer verdict.
    (blobs / ".incomplete-somehash").write_bytes(b"")
    assert cli._blob_identifier(tmp_path) == repaired

    # Authentic fresh-pull transition: absent blobs (()) before, populated
    # after — a genuine download (not tautological).
    fresh = tmp_path / "fresh"
    assert cli._blob_identifier(fresh) == ()
    (fresh / "blobs").mkdir(parents=True)
    (fresh / "blobs" / "cafe").write_bytes(b"\x00" * 512)
    assert cli._blob_identifier(fresh) != ()


def test_hf_cache_root_resolves_owner_and_single_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_hf_cache_root`` resolves the HF cache ``models--<id>`` dir from the
    hub cache with no network, for BOTH ``owner/repo`` and single-component
    repo ids (Codex #2392), and returns None when HF_HUB_CACHE is unavailable.
    """
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    assert cli._hf_cache_root("mlx-community/Qwen3-0.6B-4bit") == Path(
        tmp_path / "models--mlx-community--Qwen3-0.6B-4bit"
    )
    # Single-component (no '/') must map to models--<repo>, not a broken path.
    assert cli._hf_cache_root("Qwen3-0.6B") == Path(tmp_path / "models--Qwen3-0.6B")


def test_hf_cache_root_prefers_repo_name_to_id_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_hf_cache_root`` uses HF's ``repo_name_to_id`` when the installed
    version exposes it (Codex #2392 success branch). The concurrent no-MLX
    matrix's huggingface_hub usually lacks this symbol, so the fallback
    (``repo_id.replace``) is what the other tests exercise; this pins the
    preferred branch so diff-cover sees it and a future HF upgrade that adds
    the symbol can't regress the path silently.
    """
    import huggingface_hub.utils as _hf_utils

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    # The current huggingface_hub lacks ``repo_name_to_id`` (that's exactly
    # why the fallback branch is what other tests exercise). Inject it on the
    # real module so ``_hf_cache_root``'s ``from huggingface_hub.utils import
    # repo_name_to_id`` resolves and runs the success branch.
    monkeypatch.setattr(
        _hf_utils,
        "repo_name_to_id",
        lambda repo_id: repo_id.replace("/", "--").upper(),
        raising=False,
    )
    # ``owner/repo`` is normalized by the monkeypatched repo_name_to_id.
    assert cli._hf_cache_root("mlx-community/Qwen3-0.6B-4bit") == Path(
        tmp_path / "models--MLX-COMMUNITY--QWEN3-0.6B-4BIT"
    )


def test_hf_cache_root_returns_none_when_hf_constants_unimportable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_hf_cache_root`` returns None when ``HF_HUB_CACHE`` cannot be
    imported (Codex #2392 OSError/import-failure branch) — the caller treats
    this as "no cache configured" rather than crashing.
    """
    import builtins

    real_import = builtins.__import__

    def _guard(name, *args, **kwargs):
        if name == "huggingface_hub.constants":
            raise ImportError("simulated missing HF_HUB_CACHE")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guard)
    assert cli._hf_cache_root("mlx-community/Qwen3-0.6B-4bit") is None


def test_blob_identifier_listdir_oserror_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_blob_identifier`` returns ``()`` when listing ``blobs/`` raises
    OSError (EACCES / a permissive mount) instead of propagating — an empty
    inventory is a safe no-transfer baseline (Codex #2392).
    """
    import os

    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "cafe").write_bytes(b"\x00" * 512)

    original_listdir = os.listdir

    def _raising_listdir(path):
        if str(path) == str(blobs):
            raise OSError("simulated listdir failure")
        return original_listdir(path)

    monkeypatch.setattr(os, "listdir", _raising_listdir)
    assert cli._blob_identifier(tmp_path) == ()


def test_blob_identifier_stat_oserror_skips_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_blob_identifier`` skips (does not crash on) a blob whose ``stat()``
    raises OSError (e.g. a vanished/transiently-locked file) and still returns
    the rest of the inventory (Codex #2392).
    """
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    good = blobs / "good"
    good.write_bytes(b"\x00" * 64)
    bad = blobs / "bad"
    bad.write_bytes(b"\x00" * 32)

    original_stat = Path.stat
    bad_s = f"{bad}"

    def _raising_stat(self, *args, **kwargs):
        if str(self) == bad_s:
            raise OSError("simulated stat failure")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", _raising_stat)
    ident = cli._blob_identifier(tmp_path)
    # "bad" is skipped; "good" still fingerprints.
    assert ("good", 64, ident[0][2]) in ident
    assert all(row[0] != "bad" for row in ident)


def test_print_pull_summary_was_cached_branch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``was_cached=True`` prints the "Already cached ... verified" line
    (issue #2349) rather than the "Downloaded" line — a proven no-transfer."""
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "model.safetensors").write_bytes(b"\x00" * 4096)
    cli._print_pull_summary("mlx-community/Qwen3-0.6B-4bit", snap, 1.5, was_cached=True)
    out_str = capsys.readouterr().out
    assert "Already cached" in out_str
    assert "Downloaded" not in out_str
    assert "verified (nothing to download)" in out_str
