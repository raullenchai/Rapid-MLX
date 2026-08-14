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

import json
import os
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


def test_confirm_proceeds_on_empty_input(monkeypatch):
    """Empty input (just hit Enter) → proceed. ``[Y/n]`` means Y is the
    default — the user already typed the subcommand on a specific alias,
    so Enter should respect their intent, not abort.
    """
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _=None: "")

    assert gate.confirm_or_abort("foo/huge", 41 * 1024**3) is True


def test_confirm_aborts_on_ctrl_c(monkeypatch, capsys):
    """Ctrl-C at the prompt → treated as abort (mapped to ``n`` internally,
    same ``sys.exit(1)`` path as a typed ``n``), not a stack trace.

    Pinning the exit code (not just ``SystemExit``) keeps the contract
    explicit: a future refactor that re-maps Ctrl-C to ``130`` would
    flip the gate's semantics for callers that distinguish "user
    cancelled" from "abort hint printed and exited".
    """
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    def _raise(_=None):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _raise)

    with pytest.raises(SystemExit) as excinfo:
        gate.confirm_or_abort("foo/huge", 41 * 1024**3)
    assert excinfo.value.code == 1

    out = capsys.readouterr().out
    assert "Aborted" in out
    assert "rapid-mlx pull foo/huge" in out


def test_confirm_proceeds_on_eof(monkeypatch):
    """EOFError on ``input()`` (e.g. stdin closed mid-prompt by a script
    that fed an empty pipe but kept the TTY check tricked) → treat as
    Enter and proceed. Previously bundled with ``KeyboardInterrupt``
    (both → abort); now split so Ctrl-C cancels but EOF defers to the
    ``[Y/n]`` default. Pins the new split-handler contract."""
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    def _eof(_=None):
        raise EOFError

    monkeypatch.setattr("builtins.input", _eof)

    assert gate.confirm_or_abort("foo/huge", 41 * 1024**3) is True


@pytest.mark.parametrize("typo", ["ys", "yres", "nope", "q", "abort", "go"])
def test_confirm_proceeds_on_non_no_typos(monkeypatch, typo):
    """``[Y/n]`` semantics: only an explicit ``n``/``no`` aborts. Typos
    (``ys``, ``yres``) and stray words (``nope``, ``q``, ``abort``) all
    proceed because the user already invoked the subcommand and the
    only "danger" is downloading the very alias they typed.

    This generosity is intentional — pin it so a stricter rewrite
    (e.g. "only accept y/yes/empty, re-prompt on anything else") has
    to delete this test on purpose. ``nope`` proceeding is the most
    surprising one; documented here.
    """
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _=None: typo)

    assert gate.confirm_or_abort("foo/huge", 41 * 1024**3) is True


def test_confirm_aborts_on_no_uppercase(monkeypatch):
    """``N`` (single capital) → abort. ``.lower()`` already runs on the
    raw input, but pin the case-insensitivity contract explicitly."""
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _=None: "N")

    with pytest.raises(SystemExit) as excinfo:
        gate.confirm_or_abort("foo/huge", 41 * 1024**3)
    assert excinfo.value.code == 1


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


def test_is_repo_cached_rejects_partial_numbered_shards_without_index(
    tmp_path, monkeypatch
):
    """An interrupted pull may land shard 1 before the index manifest.

    The numbered filename already proves this is a multi-file checkpoint, so
    one non-empty shard cannot fall through to the single-file cache check.
    """
    cache_root = tmp_path / "hf-cache"
    repo_root = cache_root / "models--audio--partial-tts"
    sha = "audio123"
    snap = repo_root / "snapshots" / sha
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model-00001-of-00004.safetensors").write_bytes(b"x" * 2048)
    _seed_refs_main(repo_root, sha)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate.is_repo_cached("audio/partial-tts") is False


def test_is_repo_cached_accepts_complete_numbered_shards_without_index(
    tmp_path, monkeypatch
):
    """A complete inferred shard set stays usable if a repo omits the index."""
    cache_root = tmp_path / "hf-cache"
    repo_root = cache_root / "models--audio--complete-tts"
    sha = "audio456"
    snap = repo_root / "snapshots" / sha
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    for index in range(1, 4):
        name = f"model-{index:05d}-of-00003.safetensors"
        (snap / name).write_bytes(b"x" * 2048)
    _seed_refs_main(repo_root, sha)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate.is_repo_cached("audio/complete-tts") is True


def test_whisper_cache_accepts_its_npz_checkpoint_layout(tmp_path, monkeypatch):
    """mlx-audio Whisper uses config.json + weights.npz, not safetensors."""
    cache_root = tmp_path / "hf-cache"
    repo_root = cache_root / "models--mlx-community--whisper-medium-mlx"
    sha = "whisper123"
    snap = repo_root / "snapshots" / sha
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "weights.npz").write_bytes(b"x" * 2048)
    _seed_refs_main(repo_root, sha)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert (
        gate._snapshot_is_complete_whisper_model("mlx-community/whisper-medium-mlx")
        is True
    )
    # The generic text-model probe stays strict about NPZ-only repositories.
    assert gate.is_repo_cached("mlx-community/whisper-medium-mlx") is False


@pytest.mark.parametrize("missing", ["config.json", "weights.npz"])
def test_whisper_cache_rejects_incomplete_npz_layout(tmp_path, monkeypatch, missing):
    cache_root = tmp_path / "hf-cache"
    repo_root = cache_root / "models--mlx-community--whisper-small-mlx"
    sha = "whisper456"
    snap = repo_root / "snapshots" / sha
    snap.mkdir(parents=True)
    files = {"config.json": b"{}", "weights.npz": b"weights"}
    for name, payload in files.items():
        if name != missing:
            (snap / name).write_bytes(payload)
    _seed_refs_main(repo_root, sha)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert (
        gate._snapshot_is_complete_whisper_model("mlx-community/whisper-small-mlx")
        is False
    )


@pytest.mark.parametrize("escaped_name", ["config.json", "weights.npz"])
def test_whisper_cache_rejects_files_symlinked_outside_repo(
    tmp_path, monkeypatch, escaped_name
):
    """A crafted cache symlink must not borrow proof from an unrelated file."""
    cache_root = tmp_path / "hf-cache"
    repo_root = cache_root / "models--mlx-community--whisper-small-mlx"
    sha = "whisper789"
    snap = repo_root / "snapshots" / sha
    snap.mkdir(parents=True)
    files = {"config.json": b"{}", "weights.npz": b"weights"}
    outside = tmp_path / f"outside-{escaped_name}"
    outside.write_bytes(files[escaped_name])
    for name, payload in files.items():
        path = snap / name
        if name == escaped_name:
            path.symlink_to(outside)
        else:
            path.write_bytes(payload)
    _seed_refs_main(repo_root, sha)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert (
        gate._snapshot_is_complete_whisper_model("mlx-community/whisper-small-mlx")
        is False
    )


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
    picked up by the loader, so it must not pass the gate either.

    DeepSeek pr_validate round-3 raised this as a false-positive
    BLOCKING (claiming macOS APFS case-insensitive default would
    make ``glob`` match ``Model.safetensors``). Empirical verification
    on the exact deployment platform (APFS volume on macOS 15) shows
    Python's ``glob.glob`` filters case-sensitively even when the
    underlying filesystem treats names case-insensitively for lookup::

        >>> Path('Model.safetensors').write_bytes(b'x')
        >>> os.path.exists('model.safetensors')   # FS case-insensitive
        True
        >>> glob.glob('model*.safetensors')       # glob case-sensitive
        []

    The implementation pins to that behaviour via
    ``_is_model_weight_filename``'s case-sensitive ``startswith``.
    Pin kept; DeepSeek finding noted in commit history.
    """
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
    assert hasattr(gate, "is_weightless_stub")
    assert hasattr(gate, "weightless_stub_notice")
    assert os.path.basename(gate.__file__) == "_download_gate.py"


# ---------------------------------------------------------------------------
# is_weightless_stub / weightless_stub_notice (0.10.16 dogfood finding ⑥):
# a config-only "weightless stub" cache (config.json present, model shards
# absent) LOOKS cached but makes ``serve`` eat a surprise multi-GB download.
# ---------------------------------------------------------------------------


def _patch_try_to_load(monkeypatch, return_value):
    """Force ``huggingface_hub.try_to_load_from_cache`` to a fixed result.

    ``is_weightless_stub`` does ``from huggingface_hub import
    try_to_load_from_cache`` at call time, so setting the attribute on the
    package re-binds what the in-function import resolves to.
    """
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", lambda *a, **k: return_value
    )


def test_is_weightless_stub_true_config_cached_weights_missing(monkeypatch):
    """config.json cached (str path) + weights missing (is_repo_cached False)
    → the stub state that finding ⑥ warns about."""
    _patch_try_to_load(monkeypatch, "/hf/cache/.../config.json")
    monkeypatch.setattr(gate, "is_repo_cached", lambda _r: False)

    assert gate.is_weightless_stub("mlx-community/gemma-4-e4b-it-4bit") is True


def test_is_weightless_stub_false_when_fully_cached(monkeypatch):
    """config cached AND weights present → a complete cache, not a stub."""
    _patch_try_to_load(monkeypatch, "/hf/cache/.../config.json")
    monkeypatch.setattr(gate, "is_repo_cached", lambda _r: True)

    assert gate.is_weightless_stub("mlx-community/complete-4bit") is False


def test_is_weightless_stub_false_when_config_absent(monkeypatch):
    """No config in the cache (``None``) → a totally-absent repo, not a
    config-only stub. ``is_repo_cached`` must not even be consulted."""
    _patch_try_to_load(monkeypatch, None)
    monkeypatch.setattr(
        gate,
        "is_repo_cached",
        lambda _r: pytest.fail("is_repo_cached must not run when config absent"),
    )

    assert gate.is_weightless_stub("mlx-community/never-touched") is False


def test_is_weightless_stub_false_on_non_string_cache_sentinel(monkeypatch):
    """``try_to_load_from_cache`` returns a non-str sentinel (huggingface_hub's
    private ``_CACHED_NO_EXIST``) when a file is known-absent. Any non-str
    result must be treated as 'config NOT present'. Exercise it with an
    opaque stand-in object rather than importing the private sentinel, so a
    hub refactor of that name can't break collection here."""
    _known_absent_sentinel = object()  # stands in for HF's _CACHED_NO_EXIST
    _patch_try_to_load(monkeypatch, _known_absent_sentinel)
    monkeypatch.setattr(
        gate,
        "is_repo_cached",
        lambda _r: pytest.fail(
            "is_repo_cached must not run for a non-str cache result"
        ),
    )

    assert gate.is_weightless_stub("mlx-community/known-absent") is False


def test_is_weightless_stub_false_for_local_path_without_touching_hf(
    tmp_path, monkeypatch
):
    """A local directory path short-circuits to False via ``os.path.exists``
    WITHOUT any HF cache lookup — ``serve /path/to/model`` is never a 'stub'.

    Deleting that short-circuit must turn this test RED. The HF loaders are
    wired to fail-and-record (``side_effect=AssertionError``), and we assert
    they were never invoked. The recorded-call assertion is the load-bearing
    check: ``is_weightless_stub`` wraps its body in a broad ``except
    Exception`` that would swallow the raised AssertionError and still return
    False, so a raise alone (or the old ``return None`` mock) wouldn't catch
    the regression — ``assert_not_called`` does."""
    import huggingface_hub

    load_from_cache = MagicMock(
        side_effect=AssertionError("must not touch the HF cache for a local path")
    )
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", load_from_cache)
    repo_cached = MagicMock(
        side_effect=AssertionError("must not probe weights for a local path")
    )
    monkeypatch.setattr(gate, "is_repo_cached", repo_cached)

    # A real directory on disk → the os.path.exists guard returns False first.
    assert gate.is_weightless_stub(str(tmp_path)) is False
    load_from_cache.assert_not_called()
    repo_cached.assert_not_called()


def test_is_weightless_stub_false_on_internal_error(monkeypatch):
    """A best-effort diagnostic must never raise — any internal failure
    yields False so an otherwise-fine serve is not broken."""

    def _boom(*_a, **_k):
        raise RuntimeError("cache probe blew up")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", _boom)

    assert gate.is_weightless_stub("mlx-community/anything") is False


def test_is_weightless_stub_real_tree(tmp_path, monkeypatch):
    """End-to-end against a REAL on-disk cache tree (config.json symlink +
    refs/main, zero safetensors) — the exact shape of the ~20 Gemma-4
    config-only stubs a warm cache holds. Exercises the real
    ``try_to_load_from_cache`` + ``is_repo_cached`` together."""
    cache_root = tmp_path / "hf-cache"
    repo_root = cache_root / "models--mlx-community--gemma-4-e4b-it-4bit"
    sha = "475b9088d29754a3379866cf5aeb6b41acd313c2"
    snap = repo_root / "snapshots" / sha
    snap.mkdir(parents=True)
    blobs = repo_root / "blobs"
    blobs.mkdir()
    blob = blobs / "cfgblob"
    blob.write_text("{}")
    # Cache layout stores config.json as a symlink into blobs/.
    (snap / "config.json").symlink_to(blob)
    _seed_refs_main(repo_root, sha)
    # NOTE: zero model*.safetensors → weightless stub.

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))
    # try_to_load_from_cache resolves its default cache dir from
    # huggingface_hub.file_download.HF_HUB_CACHE — point it at the fake tree
    # too so the real helper reads our on-disk stub.
    import huggingface_hub.file_download as _fd

    monkeypatch.setattr(_fd, "HF_HUB_CACHE", str(cache_root), raising=False)

    # is_repo_cached (real) sees no weights → False; config.json (real) is
    # resolvable → is_weightless_stub is True.
    assert gate.is_repo_cached("mlx-community/gemma-4-e4b-it-4bit") is False
    assert gate.is_weightless_stub("mlx-community/gemma-4-e4b-it-4bit") is True

    # Drop a real weight shard → no longer a stub.
    (snap / "model.safetensors").write_bytes(b"w" * 4096)
    assert gate.is_weightless_stub("mlx-community/gemma-4-e4b-it-4bit") is False


def test_is_weightless_stub_false_for_video_component_weights(tmp_path, monkeypatch):
    """Video-gen / diffusers repos ship their weights as component files
    (``transformer.safetensors`` / ``vae.safetensors`` / ...) that mlx-lm's
    text ``model*.safetensors`` glob never matches — so ``is_repo_cached``
    reads a fully-cached video model as weightless. That must NOT surface the
    "config cached, weights missing — will download ~N GB" notice on every
    serve of an already-downloaded video model (the CogVideoX-Fun / LTX-2.3
    false alarm). Exercises the real ``_snapshot_is_complete_split_model``."""
    cache_root = tmp_path / "hf-cache"
    repo_root = cache_root / "models--dgrauet--CogVideoX-Fun-V1.5-5b-InP-mlx-q4"
    sha = "027bc0493a9dc41fad584568a9453961e18abb55"
    snap = repo_root / "snapshots" / sha
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    # Real CogVideoX-Fun layout: a diffusers pipeline manifest + the mlx-video
    # ``split_model.json`` component manifest + one ``<component>.safetensors``
    # per component at the snapshot root, none named ``model*.safetensors``.
    (snap / "model_index.json").write_text('{"_class_name": "CogVideoXPipeline"}')
    (snap / "split_model.json").write_text(
        '{"components": ["transformer", "text_encoder", "vae"]}'
    )
    (snap / "transformer.safetensors").write_bytes(b"t" * 4096)
    (snap / "vae.safetensors").write_bytes(b"v" * 4096)
    (snap / "text_encoder.safetensors").write_bytes(b"e" * 4096)
    _seed_refs_main(repo_root, sha)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))
    import huggingface_hub.file_download as _fd

    monkeypatch.setattr(_fd, "HF_HUB_CACHE", str(cache_root), raising=False)

    # is_repo_cached (text glob) still reads False — video weights aren't
    # ``model*.safetensors`` — but the stub notice must be suppressed.
    assert gate.is_repo_cached("dgrauet/CogVideoX-Fun-V1.5-5b-InP-mlx-q4") is False
    assert (
        gate._snapshot_is_complete_split_model(
            "dgrauet/CogVideoX-Fun-V1.5-5b-InP-mlx-q4"
        )
        is True
    )
    assert gate.is_weightless_stub("dgrauet/CogVideoX-Fun-V1.5-5b-InP-mlx-q4") is False


def _seed_mflux_snapshot(repo_root, sha: str, *, omit: tuple[str, str] | None = None):
    snap = repo_root / "snapshots" / sha
    (snap / "tokenizer").mkdir(parents=True)
    (snap / "tokenizer" / "tokenizer.json").write_text("{}")
    for component in ("transformer", "text_encoder", "vae"):
        component_dir = snap / component
        component_dir.mkdir()
        (component_dir / "model.safetensors.index.json").write_text(
            '{"weight_map": {"a": "0.safetensors", "b": "1.safetensors"}}'
        )
        for shard in ("0.safetensors", "1.safetensors"):
            if omit != (component, shard):
                (component_dir / shard).write_bytes(b"weights")
    _seed_refs_main(repo_root, sha)


def test_complete_mflux_snapshot_is_runnable(tmp_path, monkeypatch):
    """A complete image-gen component layout is a cached runnable model."""
    cache_root = tmp_path / "hf-cache"
    repo = "Runpod/FLUX.2-klein-4B-mflux-4bit"
    repo_root = cache_root / "models--Runpod--FLUX.2-klein-4B-mflux-4bit"
    _seed_mflux_snapshot(repo_root, "a" * 40)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate.is_repo_cached(repo) is False
    assert gate._snapshot_is_complete_mflux_model(repo) is True
    assert gate.is_weightless_stub(repo) is False


def test_partial_mflux_snapshot_is_not_runnable(tmp_path, monkeypatch):
    """Every shard from every required mflux component must be present."""
    cache_root = tmp_path / "hf-cache"
    repo = "Runpod/FLUX.2-klein-4B-mflux-4bit"
    repo_root = cache_root / "models--Runpod--FLUX.2-klein-4B-mflux-4bit"
    _seed_mflux_snapshot(repo_root, "b" * 40, omit=("transformer", "1.safetensors"))

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate._snapshot_is_complete_mflux_model(repo) is False


def test_mflux_missing_weights_checks_single_partial_snapshot_without_ref(
    tmp_path, monkeypatch
):
    """Interrupted first pulls can leave indexes before refs/main is written."""
    cache_root = tmp_path / "hf-cache"
    repo = "Runpod/FLUX.2-klein-4B-mflux-4bit"
    repo_root = cache_root / "models--Runpod--FLUX.2-klein-4B-mflux-4bit"
    _seed_mflux_snapshot(repo_root, "e" * 40, omit=("text_encoder", "1.safetensors"))
    (repo_root / "refs" / "main").unlink()

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate.mflux_missing_weights(repo) == ["text_encoder/1.safetensors"]


def test_mflux_missing_weights_no_verdict_for_multiple_unpinned_snapshots(
    tmp_path, monkeypatch
):
    """Never let an old complete snapshot mask a newer partial snapshot."""
    cache_root = tmp_path / "hf-cache"
    repo = "Runpod/FLUX.2-klein-4B-mflux-4bit"
    repo_root = cache_root / "models--Runpod--FLUX.2-klein-4B-mflux-4bit"
    _seed_mflux_snapshot(repo_root, "f" * 40)
    _seed_mflux_snapshot(repo_root, "0" * 40, omit=("transformer", "0.safetensors"))
    (repo_root / "refs" / "main").unlink()

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate.mflux_missing_weights(repo) is None


def test_mflux_missing_weights_names_the_absent_shard(tmp_path, monkeypatch):
    """The gate reports WHICH file is missing, so the error can be acted on."""
    cache_root = tmp_path / "hf-cache"
    repo = "Runpod/FLUX.2-klein-4B-mflux-4bit"
    repo_root = cache_root / "models--Runpod--FLUX.2-klein-4B-mflux-4bit"
    _seed_mflux_snapshot(repo_root, "c" * 40, omit=("transformer", "0.safetensors"))

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate.mflux_missing_weights(repo) == ["transformer/0.safetensors"]


def test_mflux_missing_weights_empty_when_complete(tmp_path, monkeypatch):
    """Complete is ``[]``, not ``None`` — the two must stay distinguishable."""
    cache_root = tmp_path / "hf-cache"
    repo = "Runpod/FLUX.2-klein-4B-mflux-4bit"
    repo_root = cache_root / "models--Runpod--FLUX.2-klein-4B-mflux-4bit"
    _seed_mflux_snapshot(repo_root, "d" * 40)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate.mflux_missing_weights(repo) == []


def test_mflux_missing_weights_no_verdict_when_nothing_cached(tmp_path, monkeypatch):
    """An absent snapshot is "no verdict", never "incomplete".

    mflux downloads a snapshot it cannot find, so there is no partial
    checkpoint to guard against — reporting this as missing weights would
    block a first-run pull that was going to succeed.
    """
    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path / "empty-cache")
    )

    assert gate.mflux_missing_weights("Runpod/FLUX.2-klein-4B-mflux-4bit") is None


def test_mflux_missing_weights_no_verdict_for_non_image_alias(tmp_path, monkeypatch):
    """A repo outside the image-gen registry has no mflux index contract."""
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))

    assert gate.mflux_missing_weights("mlx-community/Qwen3-0.6B-4bit") is None


def test_mflux_missing_weights_no_verdict_without_huggingface_hub(monkeypatch):
    """A missing dependency must not masquerade as a corrupt model.

    Regression guard for the shape this function used to have: a blanket
    ``except Exception: return False`` reported an import failure with the
    same value as genuinely absent weights. Harmless while it only tinted a
    column in ``ls``; once it gates loading, it would send a user off to
    re-download weights that were never broken.
    """
    import builtins

    real_import = builtins.__import__

    def _no_hub(name, *args, **kwargs):
        if name.startswith("huggingface_hub"):
            raise ImportError("no huggingface_hub")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_hub)

    assert gate.mflux_missing_weights("Runpod/FLUX.2-klein-4B-mflux-4bit") is None


def test_mflux_missing_weights_reports_non_string_index_without_raising(
    tmp_path, monkeypatch
):
    """A weight_map whose values are not strings fails the component closed.

    ``sorted(set(weight_map.values()))`` raises ``TypeError`` on an unhashable
    (``list``) or unorderable (mixed ``str``/``int``) value. That must not
    escape as a bare stack trace from the load-time and serve-time gates, which
    do not wrap this call — an index that cannot name its shards as plain
    strings is treated as missing, exactly like a corrupt one.
    """
    cache_root = tmp_path / "hf-cache"
    repo = "Runpod/FLUX.2-klein-4B-mflux-4bit"
    repo_root = cache_root / "models--Runpod--FLUX.2-klein-4B-mflux-4bit"
    _seed_mflux_snapshot(repo_root, "9" * 40)
    # Corrupt one component index with a non-string (unhashable) shard value.
    bad_index = (
        repo_root
        / "snapshots"
        / ("9" * 40)
        / "transformer"
        / "model.safetensors.index.json"
    )
    bad_index.write_text('{"weight_map": {"a": ["0.safetensors"]}}')

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate.mflux_missing_weights(repo) == [
        "transformer/model.safetensors.index.json"
    ]


def _seed_blob(blobs_dir, name: str, *, size: int, age_seconds: float = 0.0):
    blobs_dir.mkdir(parents=True, exist_ok=True)
    path = blobs_dir / name
    path.write_bytes(b"x" * size)
    if age_seconds:
        stamp = time.time() - age_seconds
        os.utime(path, (stamp, stamp))
    return path


def test_reap_removes_only_stale_incomplete_scratch_files(tmp_path, monkeypatch):
    """Stale scratch files go; fresh ones and real blobs stay.

    Each interrupted attempt strands one uniquely-named ``.incomplete`` blob
    (huggingface_hub removes it while unwinding, which a killed process never
    does), and nothing else in the cache collects them.
    """
    blobs = tmp_path / "models--org--repo" / "blobs"
    stale = _seed_blob(
        blobs, f"{'a' * 64}.deadbeef.incomplete", size=2048, age_seconds=99999
    )
    fresh = _seed_blob(blobs, f"{'b' * 64}.cafebabe.incomplete", size=99)
    real = _seed_blob(blobs, "c" * 64, size=10)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))

    assert gate.reap_orphan_incomplete_blobs("org/repo") == (1, 2048)
    assert not stale.exists()
    assert fresh.exists()  # a live download's scratch file is untouchable
    assert real.exists()  # and a finished blob is not scratch at all


def test_reap_is_quiet_when_there_is_nothing_to_collect(tmp_path, monkeypatch):
    """An unknown repo, or one with no scratch files, reclaims nothing."""
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))

    assert gate.reap_orphan_incomplete_blobs("org/never-pulled") == (0, 0)


def test_reap_collects_every_attempt_for_one_blob(tmp_path, monkeypatch):
    """The observed failure mode: one blob, several stranded attempts.

    Measured on a developer machine as three files for a single blob written
    49, 75 and 170 hours apart — not concurrency, just one file per interrupted
    attempt with nothing ever reclaiming them.
    """
    blobs = tmp_path / "models--org--repo" / "blobs"
    etag = "d" * 64
    for suffix, age in (
        ("595efb08", 49 * 3600),
        ("5e932968", 75 * 3600),
        ("8594cacd", 170 * 3600),
    ):
        _seed_blob(blobs, f"{etag}.{suffix}.incomplete", size=512, age_seconds=age)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))

    assert gate.reap_orphan_incomplete_blobs("org/repo") == (3, 1536)


def test_reap_does_not_follow_symlinks(tmp_path, monkeypatch):
    """A symlink shaped like scratch must not delete whatever it points at."""
    blobs = tmp_path / "models--org--repo" / "blobs"
    blobs.mkdir(parents=True)
    victim = tmp_path / "precious.bin"
    victim.write_bytes(b"keep me")
    link = blobs / f"{'e' * 64}.12345678.incomplete"
    link.symlink_to(victim)
    stamp = time.time() - 99999
    os.utime(link, (stamp, stamp), follow_symlinks=False)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))

    assert gate.reap_orphan_incomplete_blobs("org/repo") == (0, 0)
    assert victim.exists()
    assert link.is_symlink()


def test_call_with_deadline_gives_up_on_a_hanging_call():
    """A call that never returns must raise, not block its caller forever.

    This is the only bound that holds on a huggingface_hub metadata request:
    the library hands httpx an explicit ``timeout=None``, which disables the
    client's own timeout rather than inheriting it, so configuring the client
    does not help.
    """
    started = threading.Event()

    def _hang():
        started.set()
        threading.Event().wait()  # never set

    with pytest.raises(TimeoutError):
        gate.call_with_deadline(_hang, 0.2)
    assert started.is_set()  # it really did run, rather than failing early


def test_call_with_deadline_passes_through_result_and_error():
    """Within the deadline it is transparent — value out, exception out."""
    assert gate.call_with_deadline(lambda a, b=0: a + b, 5, 1, b=2) == 3

    def _boom():
        raise ValueError("upstream")

    with pytest.raises(ValueError, match="upstream"):
        gate.call_with_deadline(_boom, 5)


def test_pin_main_ref_is_atomic_and_populates_warm_cache_ref(tmp_path, monkeypatch):
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))

    gate.pin_main_ref("org/repo", "a" * 40)

    ref = tmp_path / "models--org--repo" / "refs" / "main"
    assert ref.read_text() == "a" * 40
    assert list(ref.parent.glob("main.*.tmp")) == []


def _seed_split_model_snapshot(repo_root, sha: str, *, omit: str | None = None):
    snap = repo_root / "snapshots" / sha
    snap.mkdir(parents=True)
    components = ["transformer", "text_encoder", "vae"]
    (snap / "split_model.json").write_text(json.dumps({"components": components}))
    for component in components:
        if component != omit:
            (snap / f"{component}.safetensors").write_bytes(b"w" * 4096)
    _seed_refs_main(repo_root, sha)
    return snap


def test_split_model_local_snapshot_resolves_a_complete_checkpoint(
    tmp_path, monkeypatch
):
    """A complete video checkpoint resolves locally, so the load stays offline."""
    cache_root = tmp_path / "hf-cache"
    repo = "dgrauet/CogVideoX-Fun-V1.5-5b-InP-mlx-q4"
    repo_root = cache_root / f"models--{repo.replace('/', '--')}"
    snap = _seed_split_model_snapshot(repo_root, "1" * 40)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate.split_model_local_snapshot(repo) == str(snap)


def test_split_model_local_snapshot_declines_a_partial_checkpoint(
    tmp_path, monkeypatch
):
    """A missing component must keep downloading rather than load half a model."""
    cache_root = tmp_path / "hf-cache"
    repo = "dgrauet/CogVideoX-Fun-V1.5-5b-InP-mlx-q4"
    repo_root = cache_root / f"models--{repo.replace('/', '--')}"
    _seed_split_model_snapshot(repo_root, "2" * 40, omit="vae")

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate.split_model_local_snapshot(repo) is None


def test_mflux_local_snapshot_resolves_a_complete_checkpoint(tmp_path, monkeypatch):
    """A verified-complete mflux cache resolves to its snapshot directory.

    Handing mflux this path instead of the repo id is what keeps a warm start
    off the network — the resolution mflux would otherwise do has no timeout.
    """
    cache_root = tmp_path / "hf-cache"
    repo = "Runpod/FLUX.2-klein-4B-mflux-4bit"
    repo_root = cache_root / "models--Runpod--FLUX.2-klein-4B-mflux-4bit"
    sha = "e" * 40
    _seed_mflux_snapshot(repo_root, sha)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate.mflux_local_snapshot(repo) == str(repo_root / "snapshots" / sha)


def test_mflux_local_snapshot_declines_a_partial_checkpoint(tmp_path, monkeypatch):
    """One missing shard means no local path — the caller must still pull.

    The whole point of resolving locally is to skip the download; doing that
    for a half-pulled checkpoint would boot mflux on randomly initialised
    weights, which renders noise rather than failing.
    """
    cache_root = tmp_path / "hf-cache"
    repo = "Runpod/FLUX.2-klein-4B-mflux-4bit"
    repo_root = cache_root / "models--Runpod--FLUX.2-klein-4B-mflux-4bit"
    _seed_mflux_snapshot(repo_root, "f" * 40, omit=("transformer", "1.safetensors"))

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    assert gate.mflux_local_snapshot(repo) is None


def test_mflux_local_snapshot_declines_when_there_is_no_verdict(tmp_path, monkeypatch):
    """ "No verdict" (nothing cached, or not an image-gen alias) is not a path.

    ``mflux_missing_weights`` returns ``None`` rather than ``[]`` here, and the
    two must not collapse: only an explicit ``[]`` may short-circuit the pull.
    """
    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path / "empty-cache")
    )

    assert gate.mflux_local_snapshot("Runpod/FLUX.2-klein-4B-mflux-4bit") is None
    assert gate.mflux_local_snapshot("mlx-community/Qwen3-0.6B-4bit") is None


def test_is_weightless_stub_true_for_partial_split_model_components(
    tmp_path, monkeypatch
):
    """Codex round-5 BLOCKING: an INTERRUPTED video pull that has landed its
    ``split_model.json`` manifest + only SOME of its components (here ``vae``
    is on disk but ``transformer`` is not) must NOT be read as fully weighted.
    ``_snapshot_is_complete_split_model`` requires EVERY declared component's
    ``<component>.safetensors`` to be present + non-empty; a missing one means
    the download is incomplete, so the stub notice must still fire."""
    cache_root = tmp_path / "hf-cache"
    repo_root = cache_root / "models--dgrauet--CogVideoX-Fun-partial-q4"
    sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    snap = repo_root / "snapshots" / sha
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "split_model.json").write_text(
        '{"components": ["transformer", "text_encoder", "vae"]}'
    )
    # Only ``vae`` arrived; ``transformer`` + ``text_encoder`` still pending.
    (snap / "vae.safetensors").write_bytes(b"v" * 4096)
    _seed_refs_main(repo_root, sha)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))
    import huggingface_hub.file_download as _fd

    monkeypatch.setattr(_fd, "HF_HUB_CACHE", str(cache_root), raising=False)

    assert (
        gate._snapshot_is_complete_split_model("dgrauet/CogVideoX-Fun-partial-q4")
        is False
    )
    assert gate.is_repo_cached("dgrauet/CogVideoX-Fun-partial-q4") is False
    assert gate.is_weightless_stub("dgrauet/CogVideoX-Fun-partial-q4") is True


def test_is_weightless_stub_true_for_zero_byte_split_model_component(
    tmp_path, monkeypatch
):
    """A component whose ``<component>.safetensors`` is a 0-byte in-flight
    placeholder (HF writes these before the blob lands) must count as
    incomplete — same failure family as the text zero-byte-shard case. The
    manifest is present and lists every component, but one file has no bytes,
    so the pull isn't done and the notice must still fire."""
    cache_root = tmp_path / "hf-cache"
    repo_root = cache_root / "models--dgrauet--CogVideoX-Fun-inflight-q4"
    sha = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    snap = repo_root / "snapshots" / sha
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "split_model.json").write_text('{"components": ["transformer", "vae"]}')
    (snap / "transformer.safetensors").write_bytes(b"t" * 4096)
    (snap / "vae.safetensors").write_bytes(b"")  # 0-byte placeholder
    _seed_refs_main(repo_root, sha)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))
    import huggingface_hub.file_download as _fd

    monkeypatch.setattr(_fd, "HF_HUB_CACHE", str(cache_root), raising=False)

    assert (
        gate._snapshot_is_complete_split_model("dgrauet/CogVideoX-Fun-inflight-q4")
        is False
    )
    assert gate.is_weightless_stub("dgrauet/CogVideoX-Fun-inflight-q4") is True


def test_snapshot_split_model_rejects_malformed_or_empty_manifest(
    tmp_path, monkeypatch
):
    """Defensive: a ``split_model.json`` that is malformed JSON, not an object,
    or names no components tells us nothing is complete — the helper must
    return False (fall through to the text-glob path) rather than raise or
    wrongly suppress. All three shapes share one fixture tree; the component
    weights are on disk so only the manifest shape is under test."""
    import json

    cache_root = tmp_path / "hf-cache"
    repo_root = cache_root / "models--vendor--bad-manifest"
    sha = "cccccccccccccccccccccccccccccccccccccccc"
    snap = repo_root / "snapshots" / sha
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "transformer.safetensors").write_bytes(b"t" * 4096)
    (snap / "vae.safetensors").write_bytes(b"v" * 4096)
    _seed_refs_main(repo_root, sha)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))
    import huggingface_hub.file_download as _fd

    monkeypatch.setattr(_fd, "HF_HUB_CACHE", str(cache_root), raising=False)

    manifest = snap / "split_model.json"
    for shape in (
        "{ this is not valid json",  # malformed
        "[]",  # not an object
        "{}",  # object, no components key
        '{"components": []}',  # empty list
        '{"components": "transformer"}',  # not a list
        '{"components": [""]}',  # empty component name
        '{"components": [123]}',  # non-string component
    ):
        manifest.write_text(shape)
        assert gate._snapshot_is_complete_split_model("vendor/bad-manifest") is False, (
            shape
        )

    # And a well-formed manifest over the same on-disk components DOES pass —
    # proving the False results above are the manifest's doing, not the tree's.
    manifest.write_text(json.dumps({"components": ["transformer", "vae"]}))
    assert gate._snapshot_is_complete_split_model("vendor/bad-manifest") is True


def test_snapshot_split_model_rejects_path_traversal_component(tmp_path, monkeypatch):
    """Security: a component name that is absolute, contains a path separator,
    or contains ``..`` could point ``<component>.safetensors`` outside the
    snapshot root. The loader never reads such a path, so the helper must
    reject it rather than stat a file elsewhere on disk."""
    cache_root = tmp_path / "hf-cache"
    repo_root = cache_root / "models--vendor--escape-component"
    sha = "dddddddddddddddddddddddddddddddddddddddd"
    snap = repo_root / "snapshots" / sha
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "transformer.safetensors").write_bytes(b"t" * 4096)
    # A real file at the escaped location the traversal would resolve to.
    (cache_root / "vae.safetensors").write_bytes(b"v" * 4096)
    _seed_refs_main(repo_root, sha)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))
    import huggingface_hub.file_download as _fd

    monkeypatch.setattr(_fd, "HF_HUB_CACHE", str(cache_root), raising=False)

    import json

    manifest = snap / "split_model.json"
    for bad in ("../vae", "/etc/passwd", "sub/dir/vae", ".."):
        manifest.write_text(json.dumps({"components": ["transformer", bad]}))
        assert (
            gate._snapshot_is_complete_split_model("vendor/escape-component") is False
        ), bad


def test_snapshot_split_model_rejects_directory_named_component(tmp_path, monkeypatch):
    """Codex round-5 MAJOR: ``os.path.getsize`` reports a positive size for a
    directory, so a component that is a DIRECTORY named
    ``vae.safetensors/`` (or a symlink to a directory inside the repo root)
    would pass a size-only check even though the real weight never arrived.
    The helper must require a regular file (``os.path.isfile``), so this stays
    incomplete and the notice still fires."""
    cache_root = tmp_path / "hf-cache"
    repo_root = cache_root / "models--vendor--dir-component"
    sha = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee0"
    snap = repo_root / "snapshots" / sha
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "split_model.json").write_text('{"components": ["transformer", "vae"]}')
    (snap / "transformer.safetensors").write_bytes(b"t" * 4096)
    # ``vae`` is a DIRECTORY, not a weight file — getsize() > 0 but not a file.
    (snap / "vae.safetensors").mkdir()
    (snap / "vae.safetensors" / "placeholder").write_bytes(b"x" * 16)
    _seed_refs_main(repo_root, sha)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))
    import huggingface_hub.file_download as _fd

    monkeypatch.setattr(_fd, "HF_HUB_CACHE", str(cache_root), raising=False)

    assert gate._snapshot_is_complete_split_model("vendor/dir-component") is False
    assert gate.is_weightless_stub("vendor/dir-component") is True

    # A symlink-to-directory at the component path is the same failure mode.
    (snap / "vae.safetensors" / "placeholder").unlink()
    (snap / "vae.safetensors").rmdir()
    real_dir = snap / "vae_real_dir"
    real_dir.mkdir()
    (snap / "vae.safetensors").symlink_to(real_dir)
    assert gate._snapshot_is_complete_split_model("vendor/dir-component") is False


def test_is_weightless_stub_true_for_interrupted_multimodal_aux_weight_first(
    tmp_path, monkeypatch
):
    """Regression guard (codex BLOCKING): an interrupted MULTIMODAL text
    download can land an auxiliary weight (``vision_model.safetensors``) BEFORE
    its index/shards — with NO text-layout signal on disk yet. Inferring
    "non-text" from the absence of ``model*.safetensors`` would misread this as
    a fully-weighted non-text model and wrongly suppress the notice. Requiring
    a POSITIVE mlx-video ``split_model.json`` manifest, which a text repo never
    ships, keeps it a stub so the notice fires."""
    cache_root = tmp_path / "hf-cache"
    repo_root = cache_root / "models--mlx-community--interrupted-vlm-4bit"
    sha = "5555555555555555555555555555555555555555"
    snap = repo_root / "snapshots" / sha
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    # Only the vision tower arrived so far — no index, no model*.safetensors,
    # and (crucially) no non-text pipeline manifest.
    (snap / "vision_model.safetensors").write_bytes(b"v" * 4096)
    _seed_refs_main(repo_root, sha)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))
    import huggingface_hub.file_download as _fd

    monkeypatch.setattr(_fd, "HF_HUB_CACHE", str(cache_root), raising=False)

    assert (
        gate._snapshot_is_complete_split_model("mlx-community/interrupted-vlm-4bit")
        is False
    )
    assert gate.is_repo_cached("mlx-community/interrupted-vlm-4bit") is False
    assert gate.is_weightless_stub("mlx-community/interrupted-vlm-4bit") is True


def test_is_weightless_stub_true_for_component_weights_without_manifest(
    tmp_path, monkeypatch
):
    """Documented scope limit: a non-text repo shipping component weights but
    NEITHER positive manifest falls back to the (cosmetic) false alarm rather
    than a wrong suppression — the conservative failure direction. Pins that we
    require positive metadata evidence, never inference-from-absence."""
    cache_root = tmp_path / "hf-cache"
    repo_root = cache_root / "models--vendor--manifestless-video"
    sha = "6666666666666666666666666666666666666666"
    snap = repo_root / "snapshots" / sha
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "transformer.safetensors").write_bytes(b"t" * 4096)
    (snap / "vae.safetensors").write_bytes(b"v" * 4096)
    _seed_refs_main(repo_root, sha)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))
    import huggingface_hub.file_download as _fd

    monkeypatch.setattr(_fd, "HF_HUB_CACHE", str(cache_root), raising=False)

    assert gate._snapshot_is_complete_split_model("vendor/manifestless-video") is False
    assert gate.is_weightless_stub("vendor/manifestless-video") is True


def test_is_weightless_stub_false_for_split_model_manifest(tmp_path, monkeypatch):
    """LTX-2.3's positive marker is ``split_model.json`` (no model_index.json).
    A cached component layout carrying it must be recognized as weight-present
    so the notice is suppressed."""
    cache_root = tmp_path / "hf-cache"
    repo_root = cache_root / "models--notapalindrome--ltx23-mlx-av-q4"
    sha = "88b4b5b2ed7697c25f281e76e3c692f659027ab1"
    snap = repo_root / "snapshots" / sha
    snap.mkdir(parents=True)
    (snap / "config.json").write_text('{"model_type": "AudioVideo"}')
    (snap / "split_model.json").write_text(
        '{"components": ["transformer", "vae_decoder", "vocoder"]}'
    )
    (snap / "transformer.safetensors").write_bytes(b"t" * 4096)
    (snap / "vae_decoder.safetensors").write_bytes(b"v" * 4096)
    (snap / "vocoder.safetensors").write_bytes(b"o" * 4096)
    _seed_refs_main(repo_root, sha)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))
    import huggingface_hub.file_download as _fd

    monkeypatch.setattr(_fd, "HF_HUB_CACHE", str(cache_root), raising=False)

    assert (
        gate._snapshot_is_complete_split_model("notapalindrome/ltx23-mlx-av-q4") is True
    )
    assert gate.is_weightless_stub("notapalindrome/ltx23-mlx-av-q4") is False


def test_is_weightless_stub_true_for_partial_text_download(tmp_path, monkeypatch):
    """A genuinely-incomplete TEXT download (one shard present, a later shard
    still missing) must STAY a weightless stub — finding ⑥'s original intent.
    Its shard is ``model-*.safetensors`` at the root, which the alt-layout
    walk deliberately skips (that's the text loader's own glob), so the stub
    notice still fires and warns the user about the pending download."""
    cache_root = tmp_path / "hf-cache"
    repo_root = cache_root / "models--mlx-community--gemma-4-27b-it-4bit"
    sha = "1111111111111111111111111111111111111111"
    snap = repo_root / "snapshots" / sha
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    # Sharded index expects two shards; only shard 1/2 is on disk.
    (snap / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "model-00001-of-00002.safetensors",'
        ' "b": "model-00002-of-00002.safetensors"}}'
    )
    (snap / "model-00001-of-00002.safetensors").write_bytes(b"s" * 4096)
    _seed_refs_main(repo_root, sha)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))
    import huggingface_hub.file_download as _fd

    monkeypatch.setattr(_fd, "HF_HUB_CACHE", str(cache_root), raising=False)

    # No non-text manifest → not an alt-layout model; the cache is an
    # incomplete text pull, so the stub notice must still fire.
    assert (
        gate._snapshot_is_complete_split_model("mlx-community/gemma-4-27b-it-4bit")
        is False
    )
    assert gate.is_repo_cached("mlx-community/gemma-4-27b-it-4bit") is False
    assert gate.is_weightless_stub("mlx-community/gemma-4-27b-it-4bit") is True


def test_is_weightless_stub_true_for_incomplete_text_with_aux_weight(
    tmp_path, monkeypatch
):
    """Regression guard (codex MAJOR): a multimodal TEXT repo can carry an
    auxiliary ``.safetensors`` (e.g. a vision tower) that isn't
    ``model*.safetensors``. That aux file must NOT mask an incomplete text
    shard set. Because the repo ships NO positive non-text manifest,
    ``_snapshot_is_complete_split_model`` returns False and is_repo_cached is
    the sole authority, so the missing shard still fires the notice."""
    cache_root = tmp_path / "hf-cache"
    repo_root = cache_root / "models--mlx-community--some-vlm-4bit"
    sha = "2222222222222222222222222222222222222222"
    snap = repo_root / "snapshots" / sha
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "model-00001-of-00002.safetensors",'
        ' "b": "model-00002-of-00002.safetensors"}}'
    )
    (snap / "model-00001-of-00002.safetensors").write_bytes(b"s" * 4096)
    # shard 2/2 is MISSING; but an auxiliary weight IS present.
    (snap / "vision_model.safetensors").write_bytes(b"v" * 4096)
    _seed_refs_main(repo_root, sha)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))
    import huggingface_hub.file_download as _fd

    monkeypatch.setattr(_fd, "HF_HUB_CACHE", str(cache_root), raising=False)

    assert (
        gate._snapshot_is_complete_split_model("mlx-community/some-vlm-4bit") is False
    )
    assert gate.is_repo_cached("mlx-community/some-vlm-4bit") is False
    assert gate.is_weightless_stub("mlx-community/some-vlm-4bit") is True


def test_is_weightless_stub_ignores_adapter_only_safetensors(tmp_path, monkeypatch):
    """A config + ``adapter_model.safetensors`` cache (no base weights) is not
    a fully-weighted model — the adapter sidecar must NOT be counted as
    alt-layout weights, so the stub notice still fires (behaviour unchanged
    from before this fix)."""
    cache_root = tmp_path / "hf-cache"
    repo_root = cache_root / "models--some--lora-adapter"
    sha = "3333333333333333333333333333333333333333"
    snap = repo_root / "snapshots" / sha
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "adapter_model.safetensors").write_bytes(b"a" * 4096)
    _seed_refs_main(repo_root, sha)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))
    import huggingface_hub.file_download as _fd

    monkeypatch.setattr(_fd, "HF_HUB_CACHE", str(cache_root), raising=False)

    assert gate._snapshot_is_complete_split_model("some/lora-adapter") is False
    assert gate.is_weightless_stub("some/lora-adapter") is True


def test_is_weightless_stub_true_for_model_index_without_split_model(
    tmp_path, monkeypatch
):
    """Documented scope limit: a bare diffusers ``model_index.json`` (pipeline
    manifest) is NOT accepted on its own — it names components but not their
    on-disk weight filenames (flat vs ``component/`` subdir vs sharded), so it
    can't be completeness-checked. Only the mlx-video ``split_model.json``
    manifest is authoritative. A repo shipping model_index.json + component
    weights but no split_model.json falls back to the (cosmetic) false alarm —
    the conservative direction — rather than risking a wrong suppression."""
    cache_root = tmp_path / "hf-cache"
    repo_root = cache_root / "models--vendor--index-only-diffusers"
    sha = "7777777777777777777777777777777777777777"
    snap = repo_root / "snapshots" / sha
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model_index.json").write_text('{"_class_name": "SomePipeline"}')
    # Full component weights present — but no split_model.json manifest.
    (snap / "transformer.safetensors").write_bytes(b"t" * 4096)
    (snap / "vae.safetensors").write_bytes(b"v" * 4096)
    _seed_refs_main(repo_root, sha)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))
    import huggingface_hub.file_download as _fd

    monkeypatch.setattr(_fd, "HF_HUB_CACHE", str(cache_root), raising=False)

    assert (
        gate._snapshot_is_complete_split_model("vendor/index-only-diffusers") is False
    )
    assert gate.is_weightless_stub("vendor/index-only-diffusers") is True


def test_is_weightless_stub_true_for_dangling_text_shard_with_aux_weight(
    tmp_path, monkeypatch
):
    """A corrupted/interrupted text snapshot whose ``model-*.safetensors``
    shard is a DANGLING symlink (target not yet materialized) — plus an
    auxiliary ``vision_model.safetensors`` — must still be a stub. It ships no
    positive non-text manifest, so ``_snapshot_is_complete_split_model`` returns
    False and is_repo_cached (which treats the dangling shard as incomplete) is
    the sole authority. The aux weight cannot mask the incomplete cache."""
    cache_root = tmp_path / "hf-cache"
    repo_root = cache_root / "models--mlx-community--dangling-vlm-4bit"
    sha = "4444444444444444444444444444444444444444"
    snap = repo_root / "snapshots" / sha
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    # A dangling shard symlink (target never downloaded) — os.path.exists is
    # False for it, but the entry name is present in the directory listing.
    (snap / "model-00001-of-00002.safetensors").symlink_to(
        tmp_path / "never-materialized-blob"
    )
    (snap / "vision_model.safetensors").write_bytes(b"v" * 4096)
    _seed_refs_main(repo_root, sha)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))
    import huggingface_hub.file_download as _fd

    monkeypatch.setattr(_fd, "HF_HUB_CACHE", str(cache_root), raising=False)

    assert (
        gate._snapshot_is_complete_split_model("mlx-community/dangling-vlm-4bit")
        is False
    )
    assert gate.is_repo_cached("mlx-community/dangling-vlm-4bit") is False
    assert gate.is_weightless_stub("mlx-community/dangling-vlm-4bit") is True


def test_weightless_stub_notice_is_size_free_and_no_extra_hf_call(monkeypatch):
    """The notice names the repo and says config cached / weights missing —
    and is deliberately SIZE-FREE. Computing a byte figure here would fire a
    second synchronous HF metadata request on the startup path, redundant
    with the download's own lookup. Pin that ``estimate_repo_size_bytes`` is
    NOT called (codex #1175 NIT)."""
    monkeypatch.setattr(gate, "is_weightless_stub", lambda _r: True)
    monkeypatch.setattr(
        gate,
        "estimate_repo_size_bytes",
        lambda *_a, **_k: pytest.fail(
            "weightless_stub_notice must not make an HF size request"
        ),
    )

    notice = gate.weightless_stub_notice("mlx-community/gemma-4-e4b-it-4bit")
    assert notice is not None
    assert "mlx-community/gemma-4-e4b-it-4bit" in notice
    assert "config cached" in notice
    assert "weights are missing" in notice
    assert "downloading the missing weights first" in notice
    # No byte figure / size unit leaked into the size-free message.
    assert "~" not in notice
    assert "GiB" not in notice and "MiB" not in notice


def test_weightless_stub_notice_none_when_not_stub(monkeypatch):
    """A fully-cached (or absent) repo yields no notice — the warning must
    not fire on the warm-cache happy path."""
    monkeypatch.setattr(gate, "is_weightless_stub", lambda _r: False)
    monkeypatch.setattr(
        gate,
        "estimate_repo_size_bytes",
        lambda *_a, **_k: pytest.fail("size lookup must be skipped for non-stubs"),
    )

    assert gate.weightless_stub_notice("mlx-community/complete-4bit") is None
