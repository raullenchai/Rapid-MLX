# SPDX-License-Identifier: Apache-2.0
"""Tests for `rapid-mlx models` — pins the per-alias profile rendering."""

from __future__ import annotations

import io
import os
import sys
from unittest.mock import patch

import pytest

from vllm_mlx import cli
from vllm_mlx.model_aliases import list_profiles


def _capture_models_output() -> str:
    """Run `models_command` with stdout captured. Patches the version-check
    helper so the test stays hermetic (no PyPI lookup at test time)."""
    buf = io.StringIO()
    with (
        patch.object(sys, "stdout", buf),
        patch("vllm_mlx._version_check.print_staleness_warning_if_any"),
    ):
        cli.models_command(None)
    return buf.getvalue()


def _split_by_modality() -> tuple[dict, dict]:
    """``(text, video, image)`` profiles — the split the listing renders."""
    profiles = list_profiles()
    video = {
        a: p
        for a, p in profiles.items()
        if getattr(p, "modality", "text") == "video-gen"
    }
    image = {
        a: p
        for a, p in profiles.items()
        if getattr(p, "modality", "text") == "image-gen"
    }
    text = {a: p for a, p in profiles.items() if a not in video and a not in image}
    return text, video, image


def test_models_command_lists_all_aliases():
    """Every alias in aliases.json must appear somewhere in the output.

    The listing is split by modality: chat-capable aliases in the main
    table, video-generation aliases in their own tagged section (#1603).
    Every alias is still listed — the split is about telling a GUI
    catalog consumer which ones can answer a chat request, not about
    hiding them from the CLI.
    """
    out = _capture_models_output()
    profiles = list_profiles()
    assert len(profiles) >= 20, "expected 20+ aliases (per project goal)"
    for alias in profiles:
        assert alias in out, f"alias {alias!r} missing from `rapid-mlx models` output"

    text_profiles, video_profiles, image_profiles = _split_by_modality()
    assert f"({len(text_profiles)} aliases)" in out
    assert len(text_profiles) + len(video_profiles) + len(image_profiles) == len(
        profiles
    )


def test_models_command_sections_video_aliases_out_of_the_text_table():
    """Video-gen aliases must not sit in the chat table, and must carry a Kind tag.

    A ``video-gen`` model has no tokenizer and no ``stream_chat``, so it
    can never answer ``/v1/chat/completions``. The desktop catalog reads
    this output and had no way to tell one apart from a chat model, so it
    offered up-to-64 GiB downloads that dead-end at "Couldn't start"
    (#1603). The section header and the ``[video:gen]`` tag are the
    contract it filters on — mirroring ``[audio:tts]`` / ``[audio:stt]``.
    """
    _, video_profiles, _ = _split_by_modality()
    if not video_profiles:
        pytest.skip("no video-gen aliases in the registry")

    out = _capture_models_output()
    head, _, video_section = out.partition("Video models (")
    assert video_section, "expected a 'Video models (N aliases)' section"
    assert f"{len(video_profiles)} aliases)" in video_section.split("\n", 1)[0]

    for alias in video_profiles:
        assert alias not in head, (
            f"video alias {alias!r} leaked into the chat table — a catalog "
            "consumer would offer it as a chat model"
        )
        assert alias in video_section
    # Every video row carries the Kind tag the desktop filters on.
    tagged = [ln for ln in video_section.splitlines() if "[video:gen]" in ln]
    assert len(tagged) == len(video_profiles)


def test_models_command_sections_image_aliases_out_of_the_text_table():
    """Image-gen aliases must not sit in the chat table, and must carry a Kind tag.

    An ``image-gen`` model (mflux FLUX / Qwen-Image) has no ``stream_chat``,
    so it can never answer ``/v1/chat/completions``. Same catalog-integrity
    contract as video (#1603): the ``[image:gen]`` tag + section header are
    what the desktop catalog filters on so it never offers a multi-GB image
    checkpoint as a chat model.
    """
    _, _, image_profiles = _split_by_modality()
    if not image_profiles:
        pytest.skip("no image-gen aliases in the registry")

    out = _capture_models_output()
    head, _, image_section = out.partition("Image models (")
    assert image_section, "expected an 'Image models (N aliases)' section"
    assert f"{len(image_profiles)} aliases)" in image_section.split("\n", 1)[0]

    for alias in image_profiles:
        assert alias not in head, (
            f"image alias {alias!r} leaked into the chat table — a catalog "
            "consumer would offer it as a chat model"
        )
        assert alias in image_section
    tagged = [ln for ln in image_section.splitlines() if "[image:gen]" in ln]
    assert len(tagged) == len(image_profiles)


def test_models_command_shows_capability_columns():
    """Capability columns (Tools, Reasoning, Spec-Decode, Suffix Tier) appear."""
    out = _capture_models_output()
    for header in ("Tools", "Reasoning", "Spec-Decode", "Suffix Tier"):
        assert header in out, f"column header {header!r} missing"


def test_models_command_renders_hybrid_marker_for_qwen35_moe():
    """Hybrid MoE models (e.g. qwen3.5-35b-4bit, the A3B variant) must
    show '✗ hybrid' + tier 'n/a'.

    r6-A R6-C1: the test was previously written against the DENSE
    ``qwen3.5-4b-4bit`` alias, which we've now flipped to non-hybrid (it
    was the metal::malloc wedge surface). The CLI column contract is the
    same — only the surface alias changes — so pivot to the A3B MoE
    Qwen3.5 variant which still legitimately wears the hybrid marker.

    The point of the column is to spare users an `info` round-trip when
    deciding whether spec-decode/suffix-decode will help. Trust the gate.
    """
    out = _capture_models_output()
    profiles = list_profiles()
    qwen35_moe = profiles.get("qwen3.5-35b-4bit")
    assert qwen35_moe is not None, "qwen3.5-35b-4bit alias missing — fixture drift"
    assert qwen35_moe.is_hybrid, (
        "qwen3.5-35b-4bit (A3B MoE) should remain is_hybrid=True"
    )

    matches = [line for line in out.splitlines() if "qwen3.5-35b-4bit " in line]
    assert matches, "no row found for qwen3.5-35b-4bit"
    row = matches[0]
    assert "✗ hybrid" in row, f"expected '✗ hybrid' marker in row: {row!r}"
    assert "n/a" in row, f"expected suffix tier 'n/a' in row: {row!r}"


def test_models_command_renders_parser_for_hermes3_8b():
    """Non-hybrid model with parser + benched tier renders both columns.

    Two contracts: (1) the tool parser cell shows the value from the
    alias registry, (2) the suffix-tier cell shows the tier currently
    recorded in ``aliases.json``. Reading the expected tier from the
    registry (not hardcoding it) means a future bench re-sweep that
    reclassifies hermes3-8b-4bit doesn't break this test, while a *display*
    regression (tier dropped from the row entirely) still does.
    """
    out = _capture_models_output()
    matches = [line for line in out.splitlines() if "hermes3-8b-4bit " in line]
    assert matches, "no row found for hermes3-8b-4bit"
    row = matches[0]
    profile = list_profiles().get("hermes3-8b-4bit")
    assert profile is not None, "hermes3-8b-4bit alias missing — fixture drift"
    assert (profile.tool_call_parser or "") in row, (
        f"expected tool parser {profile.tool_call_parser!r} in row: {row!r}"
    )
    assert profile.suffix_decoding_tier in row, (
        f"expected suffix tier {profile.suffix_decoding_tier!r} in row: {row!r}"
    )


def test_models_command_renders_em_dash_for_unset_parsers():
    """An alias without tool_call_parser / reasoning_parser should show '—'."""
    out = _capture_models_output()
    profiles = list_profiles()
    # Find any alias that has neither parser populated.
    candidates = [
        a
        for a, p in profiles.items()
        if not p.tool_call_parser and not p.reasoning_parser
    ]
    if not candidates:
        # Schema may have tightened — skip cleanly.
        import pytest

        pytest.skip("no aliases without parsers (schema may have tightened)")
    alias = candidates[0]
    matches = [line for line in out.splitlines() if f"{alias} " in line]
    assert matches, f"no row found for {alias}"
    row = matches[0]
    # Both placeholders should appear.
    assert row.count("—") >= 2, f"expected two em-dashes in row: {row!r}"


def test_models_command_mentions_chat_pull_serve_in_tip():
    """The footer should advertise the four canonical actions."""
    out = _capture_models_output()
    for cmd in ("info", "pull", "chat", "serve"):
        assert f"rapid-mlx {cmd}" in out, (
            f"footer tip missing 'rapid-mlx {cmd}' suggestion"
        )


def test_models_command_subparser_smoke():
    """`rapid-mlx models --help` exits 0 (subparser still wired)."""
    import pytest

    with (
        patch.object(sys, "argv", ["rapid-mlx", "models", "--help"]),
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 0


def test_retired_ministral_alias_fails_before_server_start(capsys):
    """The known-broken short alias must stop in CLI preflight, not load a model."""
    import pytest

    with (
        patch.object(sys, "argv", ["rapid-mlx", "serve", "ministral-3b-4bit"]),
        patch.object(cli, "serve_command") as serve,
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()

    assert exc.value.code == 1
    assert not serve.called
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "alias was retired" in captured.err
    assert "--no-mllm" in captured.err


# ----------------------------------------------------------------------
# D2 — --cached / ls view
# ----------------------------------------------------------------------


def test_ls_subcommand_registered():
    """``rapid-mlx ls --help`` exits 0 (top-level alias is wired)."""
    import pytest

    with (
        patch.object(sys, "argv", ["rapid-mlx", "ls", "--help"]),
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 0


def test_ls_routes_to_models_with_cached(monkeypatch):
    """``rapid-mlx ls`` invokes the cached view via ``models_command``
    with ``args.cached = True``. We patch ``models_command`` to capture
    the args namespace before the body runs."""
    captured: list = []
    with (
        patch.object(sys, "argv", ["rapid-mlx", "ls"]),
        patch.object(cli, "models_command", side_effect=captured.append),
    ):
        cli.main()
    assert len(captured) == 1
    assert captured[0].cached is True


def test_models_cached_flag_routes_to_cached_view(monkeypatch, capsys):
    """``models --cached`` must call the cached-view path (not the
    full alias table). We assert via the printed header difference."""
    # No cached models will be found in a fresh tmp HF cache.
    monkeypatch.setenv("HF_HOME", "/nonexistent_path_for_this_test_xyz")
    # huggingface_hub.constants.HF_HUB_CACHE is evaluated at module load,
    # so we instead patch the helper directly.
    monkeypatch.setattr(cli, "_scan_hf_cache_models", lambda: [])

    with (
        patch.object(sys, "argv", ["rapid-mlx", "models", "--cached"]),
        patch("vllm_mlx._version_check.print_staleness_warning_if_any"),
    ):
        cli.main()
    out = capsys.readouterr().out
    assert "No models cached yet" in out
    # And the full alias table header is absent.
    assert "Available models" not in out


def test_models_default_view_unchanged(monkeypatch, capsys):
    """Bare ``rapid-mlx models`` still prints the capability table —
    --cached is opt-in. Backward-compat contract."""
    with (
        patch.object(sys, "argv", ["rapid-mlx", "models"]),
        patch("vllm_mlx._version_check.print_staleness_warning_if_any"),
    ):
        cli.main()
    out = capsys.readouterr().out
    assert "Available models" in out
    assert "Tools" in out and "Reasoning" in out


def test_cached_view_renders_alias_for_known_repo(tmp_path, monkeypatch, capsys):
    """A cached HF repo whose path matches an alias should render under
    the alias name (e.g. ``qwen3.5-4b-4bit``), not the raw HF path."""
    from vllm_mlx.model_aliases import list_profiles

    profiles = list_profiles()
    # Pick any alias for the test; we'll synthesize a fake cache entry
    # at its hf_path.
    alias = next(iter(profiles))
    hf_path = profiles[alias].hf_path

    monkeypatch.setattr(
        cli, "_scan_hf_cache_models", lambda: [(hf_path, 1024 * 1024 * 100, 0.0)]
    )
    monkeypatch.setattr(cli, "_cache_entry_is_runnable", lambda _repo: True)
    cli._print_cached_models()
    out = capsys.readouterr().out
    assert alias in out, f"expected alias {alias!r} in cached view"
    assert hf_path[:40] in out, "expected HF path in cached view"


def test_cached_view_renders_unmapped_for_unknown_repo(monkeypatch, capsys):
    """A cached HF repo with no alias entry must show ``(unmapped)``."""
    monkeypatch.setattr(
        cli,
        "_scan_hf_cache_models",
        lambda: [("some/totally-unmapped-repo", 1024, 0.0)],
    )
    cli._print_cached_models()
    out = capsys.readouterr().out
    assert "(unmapped)" in out
    assert "totally-unmapped-repo" in out


def test_cached_view_marks_known_partial_repo_incomplete(tmp_path, monkeypatch, capsys):
    """Metadata-only cache directories must not advertise an alias as ready."""
    from vllm_mlx.model_aliases import list_profiles

    alias, profile = next(iter(list_profiles().items()))
    cache_root = tmp_path / "hf-cache"
    repo_root = cache_root / f"models--{profile.hf_path.replace('/', '--')}"
    snapshot = repo_root / "snapshots" / "deadbeef"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "tokenizer.json").write_text("{}")
    refs = repo_root / "refs"
    refs.mkdir()
    (refs / "main").write_text("deadbeef")

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))
    monkeypatch.setattr(
        cli,
        "_scan_hf_cache_models",
        lambda: [(profile.hf_path, 61 * 1024 * 1024, 0.0)],
    )

    cli._print_cached_models()
    out = capsys.readouterr().out
    assert "(incomplete)" in out
    assert alias not in out


def test_cached_view_renders_complete_mflux_alias(monkeypatch, capsys):
    """A complete mflux cache must map back to its image alias in ``ls``."""
    repo = "Runpod/FLUX.2-klein-4B-mflux-4bit"
    monkeypatch.setattr(
        cli,
        "_scan_hf_cache_models",
        lambda: [(repo, 4 * 1024**3, 0.0)],
    )
    monkeypatch.setattr(cli, "_cache_entry_is_runnable", lambda _repo: True)

    cli._print_cached_models()

    out = capsys.readouterr().out
    assert "flux2-klein-4b" in out
    assert "(incomplete)" not in out


def test_format_bytes_unit_selection():
    """``_format_bytes`` picks the largest unit where value >= 1.

    Suffixes are IEC base-1024 (KiB/MiB/GiB) — aligned with
    ``_format_size`` in ``vllm_mlx._download_gate`` so the same byte
    count is rendered identically by ``ls --cached`` and the B2 prompt
    (DeepSeek round-3 NIT #4)."""
    assert cli._format_bytes(0) == "0 B"
    assert cli._format_bytes(512) == "512 B"
    assert cli._format_bytes(2048) == "2.0 KiB"
    assert cli._format_bytes(5 * 1024 * 1024) == "5.0 MiB"
    assert cli._format_bytes(int(2.5 * 1024**3)) == "2.5 GiB"


def test_scan_hf_cache_models_filters_to_models_only(tmp_path, monkeypatch):
    """Only ``models--*`` directories should show in the listing — not
    ``datasets--*`` or ``spaces--*``."""
    cache_root = tmp_path / "hub"
    cache_root.mkdir()
    (cache_root / "models--mlx-community--FakeModel").mkdir()
    (cache_root / "models--mlx-community--FakeModel" / "blob1").write_bytes(b"x" * 128)
    (cache_root / "datasets--squad").mkdir()
    (cache_root / "datasets--squad" / "data").write_bytes(b"y" * 999)
    (cache_root / "spaces--gradio--demo").mkdir()

    # Patch the constants lookup inside _scan_hf_cache_models.
    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_CACHE", str(cache_root), raising=False
    )
    rows = cli._scan_hf_cache_models()
    repos = [r[0] for r in rows]
    assert "mlx-community/FakeModel" in repos
    assert all("squad" not in r for r in repos)
    assert all("demo" not in r for r in repos)


# ---------------------------------------------------------------------------
# _dir_size_bytes — HF cache blob/snapshot double counting
#
# The HF cache stores every file once under ``blobs/<sha>`` and links it
# from ``snapshots/<rev>/<file>``. Following those links tallies the same
# bytes twice, which is how a 7.9 GB model came to be advertised as
# "15.9 GiB on disk" in ``ls --cached`` and in the macOS app's
# Settings → Models panel. These tests pin the count at "each distinct
# file exactly once".
# ---------------------------------------------------------------------------


def _make_hf_cache_repo(root, blobs: dict[str, int]):
    """Build ``root/blobs/<sha>`` files of the given byte sizes.

    Returns the created ``blobs`` directory; callers add snapshots on top.
    """
    blob_dir = root / "blobs"
    blob_dir.mkdir(parents=True)
    for sha, size in blobs.items():
        (blob_dir / sha).write_bytes(b"\0" * size)
    return blob_dir


def test_dir_size_counts_snapshot_symlinks_once(tmp_path):
    """A blob plus its snapshot symlink is one file's worth of bytes."""
    repo = tmp_path / "models--acme--Widget-4bit"
    blob_dir = _make_hf_cache_repo(repo, {"sha_weights": 4096, "sha_config": 64})

    snap = repo / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").symlink_to(blob_dir / "sha_weights")
    (snap / "config.json").symlink_to(blob_dir / "sha_config")

    assert cli._dir_size_bytes(str(repo)) == 4096 + 64


def test_dir_size_counts_blob_shared_by_two_revisions_once(tmp_path):
    """Two cached revisions sharing one blob must not triple it.

    ``rev2`` re-downloads only the weights; ``config.json`` is unchanged
    so both snapshots link the same blob. The old follow-links walk
    charged the user for three copies of a file that exists once.
    """
    repo = tmp_path / "models--acme--Widget-4bit"
    blob_dir = _make_hf_cache_repo(
        repo, {"sha_w1": 4096, "sha_w2": 2048, "sha_config": 64}
    )

    rev1 = repo / "snapshots" / "rev1"
    rev1.mkdir(parents=True)
    (rev1 / "model.safetensors").symlink_to(blob_dir / "sha_w1")
    (rev1 / "config.json").symlink_to(blob_dir / "sha_config")

    rev2 = repo / "snapshots" / "rev2"
    rev2.mkdir(parents=True)
    (rev2 / "model.safetensors").symlink_to(blob_dir / "sha_w2")
    (rev2 / "config.json").symlink_to(blob_dir / "sha_config")

    assert cli._dir_size_bytes(str(repo)) == 4096 + 2048 + 64


def test_dir_size_counts_hardlinked_snapshot_once(tmp_path):
    """Hardlinked caches need inode dedupe — skipping symlinks won't do it.

    ``cp -al`` cache clones, restored CI caches and older hub versions
    all produce a snapshot entry hardlinked to its blob. Both names are
    regular files, so only ``(st_dev, st_ino)`` dedupe stops the double
    count.
    """
    repo = tmp_path / "models--acme--Widget-4bit"
    blob_dir = _make_hf_cache_repo(repo, {"sha_weights": 4096})

    snap = repo / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    os.link(blob_dir / "sha_weights", snap / "model.safetensors")

    assert cli._dir_size_bytes(str(repo)) == 4096


def test_dir_size_counts_copied_snapshot_twice(tmp_path):
    """A genuinely *copied* blob really does occupy twice the disk.

    Dedupe is by inode, not by name or content, so the count must not
    collapse two independent copies — reporting 4096 here would
    under-report what deleting the model frees.
    """
    repo = tmp_path / "models--acme--Widget-4bit"
    blob_dir = _make_hf_cache_repo(repo, {"sha_weights": 4096})

    snap = repo / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(
        (blob_dir / "sha_weights").read_bytes()  # real copy, distinct inode
    )

    assert cli._dir_size_bytes(str(repo)) == 8192


def test_dir_size_follows_symlinked_root(tmp_path):
    """A relocated cache entry reports its contents, not 0.

    ``path`` is what the caller asked to measure, so it is resolved
    before the walk. Links found *inside* are still skipped, so the
    blob/snapshot pair is not double counted through the far side.
    """
    real = tmp_path / "elsewhere" / "Widget-4bit"
    blob_dir = _make_hf_cache_repo(real, {"sha_weights": 4096})
    snap = real / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").symlink_to(blob_dir / "sha_weights")

    hub = tmp_path / "hub"
    hub.mkdir()
    link = hub / "models--acme--Widget-4bit"
    link.symlink_to(real, target_is_directory=True)

    assert cli._dir_size_bytes(str(link)) == 4096


def test_dir_size_does_not_traverse_symlinked_subdirectory(tmp_path):
    """A directory symlink is not descended — no escapes, no loops."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "huge.bin").write_bytes(b"\0" * 100_000)

    repo = tmp_path / "models--acme--Widget-4bit"
    _make_hf_cache_repo(repo, {"sha_weights": 4096})
    (repo / "escape").symlink_to(outside, target_is_directory=True)
    (repo / "loop").symlink_to(repo, target_is_directory=True)

    assert cli._dir_size_bytes(str(repo)) == 4096


def test_dir_size_skips_unreadable_subdirectory(tmp_path):
    """A directory we cannot open contributes 0 instead of raising."""
    if os.geteuid() == 0:
        pytest.skip("root can read any directory, so the mode has no effect")

    repo = tmp_path / "models--acme--Widget-4bit"
    _make_hf_cache_repo(repo, {"sha_weights": 4096})
    locked = repo / "locked"
    locked.mkdir()
    (locked / "hidden.bin").write_bytes(b"\0" * 512)
    os.chmod(locked, 0o000)
    try:
        assert cli._dir_size_bytes(str(repo)) == 4096
    finally:
        os.chmod(locked, 0o700)  # let tmp_path cleanup succeed


def test_dir_size_without_usable_inodes_counts_every_file(tmp_path):
    """Some network/FUSE mounts report ``st_ino == 0`` for everything.

    Deduping on that identity would collapse the whole tree into one
    file, so the fallback is to count each entry.
    """
    repo = tmp_path / "models--acme--Widget-4bit"
    _make_hf_cache_repo(repo, {"sha_a": 4096, "sha_b": 2048})

    real_scandir = os.scandir

    class _NoInodeEntry:
        def __init__(self, entry):
            self._entry = entry

        def __getattr__(self, name):
            return getattr(self._entry, name)

        def stat(self, *, follow_symlinks=True):
            st = self._entry.stat(follow_symlinks=follow_symlinks)
            fields = list(st)
            fields[1] = 0  # st_ino
            return os.stat_result(fields)

    class _NoInodeScandir:
        def __init__(self, path):
            self._it = real_scandir(path)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._it.close()
            return False

        def __iter__(self):
            for entry in self._it:
                yield _NoInodeEntry(entry)

    with patch.object(cli.os, "scandir", _NoInodeScandir):
        assert cli._dir_size_bytes(str(repo)) == 4096 + 2048


def test_dir_size_ignores_dangling_symlink(tmp_path):
    """A broken link (interrupted pull, pruned blob) neither crashes nor counts."""
    repo = tmp_path / "models--acme--Widget-4bit"
    blob_dir = _make_hf_cache_repo(repo, {"sha_weights": 4096})

    snap = repo / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").symlink_to(blob_dir / "sha_weights")
    (snap / "gone.safetensors").symlink_to(blob_dir / "sha_missing")

    assert cli._dir_size_bytes(str(repo)) == 4096


def test_dir_size_counts_plain_directory_normally(tmp_path):
    """No HF structure at all — every regular file still counts, recursively."""
    plain = tmp_path / "my-converted-model"
    (plain / "nested").mkdir(parents=True)
    (plain / "model.safetensors").write_bytes(b"\0" * 1000)
    (plain / "config.json").write_bytes(b"\0" * 24)
    (plain / "nested" / "tokenizer.json").write_bytes(b"\0" * 7)

    assert cli._dir_size_bytes(str(plain)) == 1031


def test_dir_size_missing_path_is_zero(tmp_path):
    """A vanished directory reports 0, not an exception."""
    assert cli._dir_size_bytes(str(tmp_path / "nope")) == 0


def test_scan_hf_cache_models_reports_blob_bytes_not_double(tmp_path, monkeypatch):
    """End-to-end: the ``ls --cached`` row carries the honest byte count."""
    cache_root = tmp_path / "hub"
    repo = cache_root / "models--acme--Widget-4bit"
    blob_dir = _make_hf_cache_repo(repo, {"sha_weights": 8192})
    snap = repo / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").symlink_to(blob_dir / "sha_weights")

    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_CACHE", str(cache_root), raising=False
    )
    rows = cli._scan_hf_cache_models()
    sizes = {repo_id: size for repo_id, size, _mtime in rows}
    assert sizes["acme/Widget-4bit"] == 8192
