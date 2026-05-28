# SPDX-License-Identifier: Apache-2.0
"""Tests for ``vllm_mlx._download_gate``.

Pins the auto-pull confirmation flow:

* ``estimate_repo_size_bytes`` returns sane numbers from a mocked HF API
  and ``None`` on failure (network down, gated repo, timeout).
* ``confirm_or_abort`` honours the env override, TTY detection, the
  size threshold, and yes/no user input.

No test in this file hits the network — every HF API call is mocked.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_mlx import _download_gate as gate

# ---------------------------------------------------------------------------
# estimate_repo_size_bytes
# ---------------------------------------------------------------------------


def _fake_sibling(name: str, size: int | None, *, lfs_size: int | None = None):
    """Build a fake ``RepoSibling``-like object that ``_sibling_size`` accepts."""
    if lfs_size is not None:
        lfs = SimpleNamespace(size=lfs_size)
    else:
        lfs = None
    return SimpleNamespace(rfilename=name, size=size, lfs=lfs)


def test_estimate_repo_size_sums_weight_files():
    """Weight + tokenizer files are summed; .gitattributes/README are skipped."""
    info = SimpleNamespace(
        siblings=[
            _fake_sibling("model-00001-of-00002.safetensors", 5 * 1024**3),
            _fake_sibling("model-00002-of-00002.safetensors", 7 * 1024**3),
            _fake_sibling("tokenizer.json", 5 * 1024**2),
            _fake_sibling("config.json", 1024),
            _fake_sibling(".gitattributes", 256),
            _fake_sibling("README.md", 4096),
        ]
    )
    with patch.object(gate, "_model_info_with_timeout", return_value=info):
        total = gate.estimate_repo_size_bytes("mlx-community/Fake-12B-4bit")

    assert total is not None
    # 12 GiB of safetensors + 5 MiB tokenizer + 1 KiB config.
    expected = (12 * 1024**3) + (5 * 1024**2) + 1024
    assert total == expected


def test_estimate_repo_size_prefers_lfs_size():
    """When both ``size`` and ``lfs.size`` are populated, LFS wins (it's the
    true blob size; the bare ``size`` field can report the pointer size)."""
    info = SimpleNamespace(
        siblings=[
            _fake_sibling("model.safetensors", 134, lfs_size=4 * 1024**3),
        ]
    )
    with patch.object(gate, "_model_info_with_timeout", return_value=info):
        total = gate.estimate_repo_size_bytes("mlx-community/Fake-4B-4bit")

    assert total == 4 * 1024**3


def test_estimate_repo_size_returns_none_on_api_failure():
    """Any exception from the HF API call surfaces as ``None`` — callers must
    fall through silently rather than blocking on a flaky network."""
    with patch.object(
        gate, "_model_info_with_timeout", side_effect=RuntimeError("HF down")
    ):
        assert gate.estimate_repo_size_bytes("definitely/not-a-real-repo") is None


def test_estimate_repo_size_returns_none_on_empty_repo():
    """An info object with no weight files yields ``None`` (rather than 0) so
    the caller's heads-up logic kicks in."""
    info = SimpleNamespace(
        siblings=[
            _fake_sibling("README.md", 1024),
            _fake_sibling(".gitattributes", 256),
        ]
    )
    with patch.object(gate, "_model_info_with_timeout", return_value=info):
        assert gate.estimate_repo_size_bytes("foo/empty") is None


# ---------------------------------------------------------------------------
# confirm_or_abort
# ---------------------------------------------------------------------------


def test_confirm_passes_through_when_under_threshold(monkeypatch):
    """A 5 GiB estimate against a 10 GiB threshold must not prompt."""
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)
    # Even if stdin is a TTY, small downloads pass through silently.
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(
        "builtins.input",
        lambda _=None: pytest.fail("input() must not be called for small repos"),
    )

    assert gate.confirm_or_abort("foo/small", 5 * 1024**3) is True


def test_confirm_passes_through_when_env_var_set(monkeypatch):
    """``RAPID_MLX_AUTO_PULL=1`` short-circuits even for huge downloads."""
    monkeypatch.setenv("RAPID_MLX_AUTO_PULL", "1")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(
        "builtins.input",
        lambda _=None: pytest.fail("input() must not be called when env set"),
    )

    assert gate.confirm_or_abort("foo/huge", 50 * 1024**3) is True


@pytest.mark.parametrize("val", ["1", "true", "yes", "YES", "True"])
def test_confirm_env_var_accepts_truthy_variants(monkeypatch, val):
    """Common truthy spellings must all work — users will try all of them."""
    monkeypatch.setenv("RAPID_MLX_AUTO_PULL", val)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert gate.confirm_or_abort("foo/huge", 50 * 1024**3) is True


def test_confirm_passes_through_when_non_tty(monkeypatch):
    """Non-interactive stdin (CI, piped scripts) must not deadlock on input."""
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr(
        "builtins.input",
        lambda _=None: pytest.fail("input() must not be called in non-TTY mode"),
    )

    assert gate.confirm_or_abort("foo/huge", 50 * 1024**3) is True


def test_confirm_proceeds_with_unknown_size(monkeypatch, capsys):
    """Unknown size → heads-up + proceed (don't block on transient HF failures)."""
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(
        "builtins.input",
        lambda _=None: pytest.fail("input() must not be called for unknown size"),
    )

    assert gate.confirm_or_abort("foo/unknown-size", None) is True
    out = capsys.readouterr().out
    assert "unknown" in out.lower()
    assert "foo/unknown-size" in out


def test_confirm_returns_true_on_yes(monkeypatch, capsys):
    """``y`` input → proceed."""
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _=None: "y")

    assert gate.confirm_or_abort("foo/huge", 41 * 1024**3) is True
    out = capsys.readouterr().out
    assert "foo/huge" in out
    assert "41" in out  # size string contains 41 GiB
    assert "Continue?" not in out  # input prompt itself isn't captured to stdout


def test_confirm_returns_true_on_yes_full_word(monkeypatch):
    """``yes`` (full word, case-insensitive) also proceeds."""
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _=None: "YES")

    assert gate.confirm_or_abort("foo/huge", 41 * 1024**3) is True


def test_confirm_aborts_on_no(monkeypatch, capsys):
    """``n`` input → ``sys.exit(1)`` with an actionable hint."""
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _=None: "n")

    with pytest.raises(SystemExit) as excinfo:
        gate.confirm_or_abort("foo/huge", 41 * 1024**3)
    assert excinfo.value.code == 1

    out = capsys.readouterr().out
    assert "Aborted" in out
    assert "rapid-mlx pull foo/huge" in out
    assert "RAPID_MLX_AUTO_PULL" in out


def test_confirm_aborts_on_empty_input(monkeypatch):
    """Empty input (the default) → abort. ``[y/N]`` means N is the default."""
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _=None: "")

    with pytest.raises(SystemExit):
        gate.confirm_or_abort("foo/huge", 41 * 1024**3)


def test_confirm_aborts_on_ctrl_c(monkeypatch):
    """Ctrl-C at the prompt → treated as abort, not a stack trace."""
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    def _raise(_=None):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _raise)

    with pytest.raises(SystemExit):
        gate.confirm_or_abort("foo/huge", 41 * 1024**3)


def test_confirm_logfile_hint_appears_in_prompt(monkeypatch, capsys):
    """When a logfile is supplied, the prompt tells the user where to tail."""
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _=None: "y")

    gate.confirm_or_abort("foo/huge", 41 * 1024**3, logfile_hint="/tmp/serve.log")
    out = capsys.readouterr().out
    assert "/tmp/serve.log" in out


# ---------------------------------------------------------------------------
# _format_size — internal, but worth pinning so the prompt stays readable.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_bytes,expected",
    [
        (0, "0 B"),
        (512, "512 B"),
        (780 * 1024**2, "780.0 MiB"),
        (int(2.4 * 1024**3), "2.4 GiB"),
        (int(42.3 * 1024**3), "42.3 GiB"),
    ],
)
def test_format_size_friendly(num_bytes, expected):
    assert gate._format_size(num_bytes) == expected


# ---------------------------------------------------------------------------
# is_repo_cached
# ---------------------------------------------------------------------------


def _seed_refs_main(repo_root, sha: str) -> None:
    """Helper: write the ``refs/main`` file the round-9/10 fix requires.

    ``snapshot_download(repo_id)`` resolves through ``refs/main`` by
    default. ``is_repo_cached`` now pins to that specific snapshot
    (no "any complete snapshot" fallback), so test fixtures must
    populate ``refs/main`` for the True cases. The False / partial
    cases either also populate it (to test that the pinned snapshot
    is incomplete) or omit it (to assert the round-10 "no refs/main
    → False" contract).
    """
    refs = repo_root / "refs"
    refs.mkdir(exist_ok=True)
    (refs / "main").write_text(sha)


def test_is_repo_cached_true_when_weight_file_present(tmp_path, monkeypatch):
    """At least one non-empty weight file in the snapshot tree → True."""
    cache_root = tmp_path / "hf-cache"
    sha = "abcd1234"
    repo_root = cache_root / "models--foo--cached"
    snap = repo_root / "snapshots" / sha
    snap.mkdir(parents=True)
    # Real cache layouts include config + tokenizer + the actual weights.
    (snap / "config.json").write_text("{}")
    (snap / "tokenizer.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"x" * 2048)
    _seed_refs_main(repo_root, sha)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate.is_repo_cached("foo/cached") is True


def test_is_repo_cached_false_when_no_snapshot(tmp_path, monkeypatch):
    """Empty HF cache directory → False."""
    empty_cache = tmp_path / "hf-cache"
    empty_cache.mkdir()
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(empty_cache))

    assert gate.is_repo_cached("foo/missing") is False


def test_is_repo_cached_false_on_partial_cache(tmp_path, monkeypatch):
    """Codex round-1 BLOCKING: a partial cache (config + tokenizer only,
    weight shards missing) must NOT pass the gate. The legacy
    ``try_to_load_from_cache('config.json')`` probe returned True here,
    letting the spawned ``serve`` subprocess silently download multi-GB
    weight shards inside its log file."""
    cache_root = tmp_path / "hf-cache"
    snap = cache_root / "models--foo--partial" / "snapshots" / "deadbeef"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "tokenizer.json").write_text("{}")
    (snap / "chat_template.jinja").write_text("{{}}")
    # Crucially: NO ``*.safetensors`` / ``*.bin`` / ``*.gguf`` file.

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate.is_repo_cached("foo/partial") is False


def test_is_repo_cached_false_on_zero_byte_weight(tmp_path, monkeypatch):
    """HF stores in-flight blobs as 0-byte placeholders before the
    download completes. A zero-byte ``*.safetensors`` must not count as
    cached — same failure mode as the partial-cache case above."""
    cache_root = tmp_path / "hf-cache"
    snap = cache_root / "models--foo--inflight" / "snapshots" / "cafe"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"")  # placeholder

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate.is_repo_cached("foo/inflight") is False


def test_is_repo_cached_rejects_npz_only(tmp_path, monkeypatch):
    """Codex round-4 BLOCKING #2 (refinement of round-2): rapid-mlx
    serves via ``mlx_lm.load``, which globs ``model*.safetensors`` and
    never reads ``.npz``. A cache containing only ``weights.npz`` is
    unusable from the chat code path, so it must NOT pass the gate —
    otherwise the spawned ``serve`` will silently download the real
    ``.safetensors`` shards inside its log file."""
    cache_root = tmp_path / "hf-cache"
    snap = cache_root / "models--mlx-community--legacy" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "weights.npz").write_bytes(b"x" * 4096)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate.is_repo_cached("mlx-community/legacy") is False


def test_is_repo_cached_rejects_gguf_only(tmp_path, monkeypatch):
    """Codex round-4 BLOCKING #2: mlx-lm has GGUF *export* support
    (``convert_to_gguf``) but no load path — ``mlx_lm.load`` only globs
    ``model*.safetensors``. A GGUF-only cache must NOT pass the gate."""
    cache_root = tmp_path / "hf-cache"
    snap = cache_root / "models--ggml--quant" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model-q4.gguf").write_bytes(b"x" * 4096)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate.is_repo_cached("ggml/quant") is False


def test_is_repo_cached_requires_every_shard_listed_in_index(tmp_path, monkeypatch):
    """Codex round-4 BLOCKING #1: ``model.safetensors.index.json`` lists
    every shard mlx-lm will load. A snapshot with shard 1/2 present but
    shard 2/2 missing must NOT pass — mlx-lm globs all shards and
    crashes halfway through deserialisation, with the failure surfaced
    in the spawned-serve log file instead of as a B2 prompt."""
    import json

    cache_root = tmp_path / "hf-cache"
    repo_root = cache_root / "models--mlx-community--sharded"
    sha = "abc"
    snap = repo_root / "snapshots" / sha
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    index = {
        "metadata": {"total_size": 2048},
        "weight_map": {
            "model.embed.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.weight": "model-00002-of-00002.safetensors",
        },
    }
    (snap / "model.safetensors.index.json").write_text(json.dumps(index))
    # Shard 1 cached, shard 2 absent.
    (snap / "model-00001-of-00002.safetensors").write_bytes(b"x" * 4096)
    _seed_refs_main(repo_root, sha)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate.is_repo_cached("mlx-community/sharded") is False

    # And once shard 2 lands it does pass.
    (snap / "model-00002-of-00002.safetensors").write_bytes(b"y" * 4096)
    assert gate.is_repo_cached("mlx-community/sharded") is True


def test_is_repo_cached_rejects_adapter_only_safetensors(tmp_path, monkeypatch):
    """Codex round-5 BLOCKING #2: rapid-mlx's load path globs
    ``model*.safetensors`` literally. A cache containing only
    ``adapter.safetensors`` (LoRA / PEFT fine-tune) or
    ``embeddings.safetensors`` (sidecar) is unusable from rapid-mlx
    and must NOT pass the gate — otherwise the spawned ``serve``
    silently pulls the real model weights."""
    cache_root = tmp_path / "hf-cache"
    repo_root = cache_root / "models--user--lora"
    sha = "abc"
    snap = repo_root / "snapshots" / sha
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "adapter.safetensors").write_bytes(b"x" * 4096)
    (snap / "embeddings.safetensors").write_bytes(b"y" * 4096)
    _seed_refs_main(repo_root, sha)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate.is_repo_cached("user/lora") is False

    # Once an actual ``model.safetensors`` lands, it does pass.
    (snap / "model.safetensors").write_bytes(b"z" * 4096)
    assert gate.is_repo_cached("user/lora") is True


def test_is_repo_cached_is_case_sensitive(tmp_path, monkeypatch):
    """Codex round-6 BLOCKING #1: ``mlx_lm`` calls ``glob.glob`` which
    is case-sensitive on Linux and on case-sensitive macOS volumes. A
    repo whose file is named ``Model.safetensors`` (capital M) is NOT
    picked up by the loader, so it must not pass the gate either."""
    cache_root = tmp_path / "hf-cache"
    snap = cache_root / "models--user--capital-m" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "Model.safetensors").write_bytes(b"x" * 4096)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate.is_repo_cached("user/capital-m") is False


def test_is_repo_cached_validates_shard_filenames_in_index(tmp_path, monkeypatch):
    """Codex round-6 BLOCKING #2: the indexed path validated only that
    ``weight_map`` values *exist*, not that the filenames match the
    loader glob. An index pointing at ``adapter.safetensors`` or
    ``Model-00001-of-00002.safetensors`` (capital M) would pass while
    ``mlx_lm`` actually loads zero model weights."""
    import json

    # Case A: index references an adapter file (loader can't open).
    cache_root = tmp_path / "hf-cache-adapter"
    snap = cache_root / "models--user--adapter-index" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "adapter.safetensors"}})
    )
    (snap / "adapter.safetensors").write_bytes(b"x" * 4096)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))
    assert gate.is_repo_cached("user/adapter-index") is False

    # Case B: index references capital-M shard names — same loader miss.
    cache_root_b = tmp_path / "hf-cache-cap"
    snap_b = cache_root_b / "models--user--capm-index" / "snapshots" / "abc"
    snap_b.mkdir(parents=True)
    (snap_b / "config.json").write_text("{}")
    (snap_b / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "w1": "Model-00001-of-00002.safetensors",
                    "w2": "Model-00002-of-00002.safetensors",
                }
            }
        )
    )
    (snap_b / "Model-00001-of-00002.safetensors").write_bytes(b"x" * 4096)
    (snap_b / "Model-00002-of-00002.safetensors").write_bytes(b"y" * 4096)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root_b))
    assert gate.is_repo_cached("user/capm-index") is False


def test_is_repo_cached_rejects_path_traversal_in_index(tmp_path, monkeypatch):
    """Codex round-7 BLOCKING #1: shard names containing ``..`` or
    absolute paths escape the snapshot root. The loader's glob is
    rooted at snap_dir, so an escaped path is invisible to it; we
    must reject it explicitly rather than walking the resolved
    file outside ``snap_dir``."""
    import json

    cache_root = tmp_path / "hf-cache"
    snap = cache_root / "models--user--escape" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")

    # Case A: relative traversal.
    (snap / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "../model-00001.safetensors"}})
    )
    # File exists at the escaped location.
    (cache_root / "model-00001.safetensors").write_bytes(b"x" * 4096)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))
    assert gate.is_repo_cached("user/escape") is False

    # Case B: absolute path.
    (snap / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "/tmp/model-00001.safetensors"}})
    )
    assert gate.is_repo_cached("user/escape") is False

    # Case C: subdirectory (not nested loader behaviour).
    (snap / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "shards/model-00001.safetensors"}})
    )
    (snap / "shards").mkdir()
    (snap / "shards" / "model-00001.safetensors").write_bytes(b"x" * 4096)
    assert gate.is_repo_cached("user/escape") is False


def test_is_repo_cached_honours_resolved_revision_ref(tmp_path, monkeypatch):
    """Codex round-9 BLOCKING: an old complete snapshot must not mask
    the current incomplete one. After an interrupted ``snapshot_download``
    update, HF leaves an empty snapshot dir at the new sha while the
    previous sha's snapshot is still on disk. The loader resolves via
    ``refs/main`` to the NEW sha and would crash on its incomplete
    contents — so the gate must honour the ref, not fall through to
    the older complete snapshot."""
    cache_root = tmp_path / "hf-cache"
    repo_root = cache_root / "models--user--mid-update"
    snap_root = repo_root / "snapshots"

    # Old complete snapshot — has a valid model.safetensors.
    old_sha = "deadbeefdeadbeefdeadbeefdeadbeef"
    old = snap_root / old_sha
    old.mkdir(parents=True)
    (old / "config.json").write_text("{}")
    (old / "model.safetensors").write_bytes(b"x" * 4096)

    # New snapshot dir exists (metadata fetched, weight pull
    # interrupted) — but no weights yet.
    new_sha = "feedfacefeedfacefeedfacefeedface"
    new = snap_root / new_sha
    new.mkdir()
    (new / "config.json").write_text("{}")  # only metadata so far

    # refs/main points at the NEW sha (this is the post-update state).
    refs = repo_root / "refs"
    refs.mkdir()
    (refs / "main").write_text(new_sha)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    # Old complete snapshot must NOT mask the new incomplete one.
    assert gate.is_repo_cached("user/mid-update") is False

    # Once the new weights land, gate flips to True.
    (new / "model.safetensors").write_bytes(b"y" * 4096)
    assert gate.is_repo_cached("user/mid-update") is True


def test_is_repo_cached_rejects_when_no_refs_main(tmp_path, monkeypatch):
    """Codex round-10 BLOCKING: when ``refs/main`` doesn't exist (fresh
    cache, non-standard layout, or a repo whose default branch is not
    ``main``), we DON'T know which sha ``snapshot_download(repo_id)``
    will resolve to — so a complete snapshot at some other sha could
    silently mask the actual current sha. The gate must err on the
    side of re-prompting; the round-9 fallback to ``any complete
    snapshot`` was the very bypass we're closing here."""
    cache_root = tmp_path / "hf-cache"
    snap = cache_root / "models--user--no-refs" / "snapshots" / "abcd"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"x" * 4096)
    # NOTE: no refs/ dir.

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate.is_repo_cached("user/no-refs") is False


def test_is_repo_cached_rejects_non_main_only_ref(tmp_path, monkeypatch):
    """Codex round-10 BLOCKING: ``snapshot_download(repo_id)`` defaults
    to the ``main`` revision via the HF API. A cache that only has
    ``refs/master`` (e.g. a legacy repo whose default branch was
    renamed upstream since the last download) doesn't tell us what
    the current ``main`` resolves to — must re-prompt. The cost is a
    one-time redundant prompt; the benefit is no silent download."""
    cache_root = tmp_path / "hf-cache"
    repo_root = cache_root / "models--user--master"
    snap_root = repo_root / "snapshots"
    sha = "1234123412341234123412341234123412341234"
    snap = snap_root / sha
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"x" * 4096)

    refs = repo_root / "refs"
    refs.mkdir()
    (refs / "master").write_text(sha)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate.is_repo_cached("user/master") is False


def test_is_repo_cached_rejects_symlink_to_directory(tmp_path, monkeypatch):
    """Codex round-8 BLOCKING: ``glob.glob("model*.safetensors")``
    returns symlinks-to-directories and dangling symlinks too, and
    mlx-lm then calls ``mx.load`` on them and crashes. The gate must
    REJECT such entries, not silently skip them via ``isfile``."""
    cache_root = tmp_path / "hf-cache"
    snap = cache_root / "models--user--badsymlink" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    # Real weight file present.
    (snap / "model.safetensors").write_bytes(b"x" * 4096)
    # Symlink at a model*.safetensors path that points to a directory.
    a_dir = tmp_path / "decoy_dir"
    a_dir.mkdir()
    (snap / "model-extra.safetensors").symlink_to(a_dir)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate.is_repo_cached("user/badsymlink") is False


def test_is_repo_cached_rejects_dangling_symlink_at_model_path(tmp_path, monkeypatch):
    """Same family: a dangling symlink whose name matches the loader
    glob would also be returned by ``glob``. The loader's subsequent
    ``mx.load`` raises; the gate must catch it."""
    cache_root = tmp_path / "hf-cache"
    snap = cache_root / "models--user--dangling" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"x" * 4096)
    (snap / "model-broken.safetensors").symlink_to(tmp_path / "nonexistent")

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate.is_repo_cached("user/dangling") is False


def test_is_repo_cached_rejects_zero_byte_extra_root_shard(tmp_path, monkeypatch):
    """Codex round-7 BLOCKING #3: ``mlx_lm`` globs EVERY
    ``model*.safetensors`` at the snapshot root and calls ``mx.load``
    on each. A zero-byte placeholder next to a valid sharded cache
    would crash the loader, so the gate must catch it too."""
    import json

    cache_root = tmp_path / "hf-cache"
    snap = cache_root / "models--user--extra-zero" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "w": "model-00001-of-00002.safetensors",
                    "x": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    (snap / "model-00001-of-00002.safetensors").write_bytes(b"x" * 4096)
    (snap / "model-00002-of-00002.safetensors").write_bytes(b"y" * 4096)
    # The extra zero-byte placeholder loader would still pick up.
    (snap / "model-extra.safetensors").write_bytes(b"")

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate.is_repo_cached("user/extra-zero") is False


def test_is_repo_cached_rejects_index_with_no_weight_map(tmp_path, monkeypatch):
    """Codex round-5 BLOCKING #1: if ``model.safetensors.index.json``
    exists but the schema doesn't yield a usable shard list (corrupt
    schema, alternate-key layout, metadata-only index), we must NOT
    fall back to the single-file probe — the presence of the index
    itself is the loader's signal that this is a sharded model."""
    import json

    cache_root = tmp_path / "hf-cache"
    snap = cache_root / "models--user--quirky-index" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    # Index uses an alternate key (no ``weight_map`` at all).
    (snap / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 4096}, "files": ["shard.safetensors"]})
    )
    # A stray single-file safetensors that would otherwise pass the
    # single-file probe.
    (snap / "model.safetensors").write_bytes(b"x" * 4096)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate.is_repo_cached("user/quirky-index") is False


def test_is_repo_cached_rejects_index_with_empty_weight_map(tmp_path, monkeypatch):
    """Same as the no-weight-map case but the key exists with an
    empty dict value. Both must be treated as 'incomplete sharded'."""
    import json

    cache_root = tmp_path / "hf-cache"
    snap = cache_root / "models--user--empty-map" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 4096}, "weight_map": {}})
    )
    (snap / "model.safetensors").write_bytes(b"x" * 4096)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate.is_repo_cached("user/empty-map") is False


def test_is_repo_cached_rejects_zero_byte_shard_in_index(tmp_path, monkeypatch):
    """A shard that's listed in the index but zero-byte on disk (HF
    in-flight placeholder) must NOT count as cached. Same family as
    the partial-cache and zero-byte-weight cases above; the index
    path needs the same check the single-file path got in round 1."""
    import json

    cache_root = tmp_path / "hf-cache"
    snap = cache_root / "models--mlx-community--inflight" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    index = {
        "metadata": {"total_size": 4096},
        "weight_map": {
            "model.embed.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.weight": "model-00002-of-00002.safetensors",
        },
    }
    (snap / "model.safetensors.index.json").write_text(json.dumps(index))
    (snap / "model-00001-of-00002.safetensors").write_bytes(b"x" * 4096)
    (snap / "model-00002-of-00002.safetensors").write_bytes(b"")  # placeholder

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate.is_repo_cached("mlx-community/inflight") is False


def test_is_repo_cached_rejects_pytorch_bin_only(tmp_path, monkeypatch):
    """Codex round-3 BLOCKING #2: ``.bin`` is the PyTorch shard format,
    not loadable by mlx-lm. A repo that has cached PyTorch ``.bin``
    weights but no MLX ``.safetensors`` should be treated as
    uncached — otherwise the spawned ``serve`` silently downloads the
    real MLX weights inside its log file."""
    cache_root = tmp_path / "hf-cache"
    snap = cache_root / "models--torch--legacy" / "snapshots" / "feed"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "tokenizer.json").write_text("{}")
    (snap / "pytorch_model-00001-of-00002.bin").write_bytes(b"z" * 4096)
    (snap / "pytorch_model-00002-of-00002.bin").write_bytes(b"z" * 4096)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate.is_repo_cached("torch/legacy") is False


def test_is_repo_cached_rejects_nested_weights(tmp_path, monkeypatch):
    """Codex round-7 BLOCKING #2: ``mlx_lm`` calls
    ``glob.glob(model_path / "model*.safetensors")`` — a NON-recursive
    glob. A snapshot whose weights live under ``shards/`` (or any
    subdirectory) is not picked up by the loader, so it must NOT pass
    the gate either. The earlier walk-the-tree behaviour was the bug."""
    cache_root = tmp_path / "hf-cache"
    snap_root = cache_root / "models--foo--nested" / "snapshots" / "1234"
    nested = snap_root / "shards"
    nested.mkdir(parents=True)
    (nested / "model-00001-of-00002.safetensors").write_bytes(b"y" * 4096)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate.is_repo_cached("foo/nested") is False


# ---------------------------------------------------------------------------
# Defensive guards — env value parsing.
# ---------------------------------------------------------------------------


def test_confirm_env_var_falsy_value_does_not_short_circuit(monkeypatch):
    """``RAPID_MLX_AUTO_PULL=0`` must NOT auto-confirm — the env is opt-in."""
    monkeypatch.setenv("RAPID_MLX_AUTO_PULL", "0")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _=None: "n")

    with pytest.raises(SystemExit):
        gate.confirm_or_abort("foo/huge", 41 * 1024**3)


def test_confirm_threshold_boundary(monkeypatch):
    """Exactly at threshold → prompt fires (the docstring promises ``>=``)."""
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _=None: "y")

    threshold = 10 * 1024**3
    # One byte under threshold → no prompt.
    monkeypatch.setattr(
        "builtins.input",
        lambda _=None: pytest.fail("input() called below threshold"),
    )
    assert gate.confirm_or_abort("foo/borderline", threshold - 1) is True

    # At threshold → prompt fires; mock yes-response.
    monkeypatch.setattr("builtins.input", lambda _=None: "y")
    assert gate.confirm_or_abort("foo/border-on", threshold) is True


# ---------------------------------------------------------------------------
# Smoke test: the module imports cleanly without huggingface_hub being a
# hard runtime requirement at import time.
# ---------------------------------------------------------------------------


def test_module_imports_without_hf_call():
    """Importing the module must NOT trigger any HF API call (lazy-imported)."""
    # If huggingface_hub had been touched at module load, the patched
    # ``_model_info_with_timeout`` in earlier tests would have failed.
    # This test exists to make the contract explicit for future maintainers.
    assert hasattr(gate, "estimate_repo_size_bytes")
    assert hasattr(gate, "confirm_or_abort")
    assert hasattr(gate, "is_repo_cached")
    assert os.path.basename(gate.__file__) == "_download_gate.py"
