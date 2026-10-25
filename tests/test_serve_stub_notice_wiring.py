# SPDX-License-Identifier: Apache-2.0
"""0.10.16 dogfood finding ⑥ — serve_command wiring for the weightless-stub
notice, including alias canonicalization (codex #1175 BLOCKING).

The unit tests in ``test_download_gate.py`` prove ``weightless_stub_notice``
returns the right string; these integration tests prove ``serve_command``
actually wires it up on the REAL execution path:

* a shorthand ALIAS (``qwen3.5-4b-4bit``) whose config-only stub lives on
  disk under its RESOLVED ``org/repo`` id still produces the notice — i.e.
  serve_command canonicalizes the alias before probing the cache (otherwise
  the feature silently no-ops for the common naive-user alias invocation),
* the notice reaches **stderr** and does so **before**
  ``_ensure_model_downloaded`` runs, and
* a fully-weighted alias cache emits nothing.

We stop execution at the ``_ensure_model_downloaded`` call (raise a sentinel)
so the heavy server boot never runs — the notice line sits immediately above
it, so both stream (stderr) and ordering (before download) are exercised
faithfully. The HF cache is a fake on-disk tree so no network/download runs.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from rapid_mlx import cli
from rapid_mlx.model_aliases import resolve_model

_ALIAS = "qwen3.5-4b-4bit"


class _StopServeError(Exception):
    """Sentinel to abort serve_command right at the download step."""


def _serve_ns():
    """Resolve a full serve Namespace via argparse (mirrors the established
    ``_minimal_serve_ns`` pattern), then force ``model`` back to the SHORTHAND
    alias so serve_command is entered with the un-resolved id — the exact
    naive-user shape the canonicalization fix must handle."""
    captured: list = []
    argv = ["rapid-mlx", "serve", _ALIAS]
    with (
        patch.object(sys, "argv", argv),
        patch.object(cli, "serve_command", side_effect=captured.append),
    ):
        cli.main()
    ns = captured[0]
    ns.model = _ALIAS
    ns._original_alias = None
    return ns


def _seed_alias_cache(monkeypatch, tmp_path, *, with_weights: bool):
    """Seed a fake HF cache under the alias's RESOLVED ``org/repo`` id.

    ``with_weights=False`` produces a weightless stub (config.json symlink +
    refs/main, no ``model*.safetensors``); ``True`` adds a real shard. Points
    both HF cache bindings (``huggingface_hub.constants`` for is_repo_cached
    and ``huggingface_hub.file_download`` for try_to_load_from_cache) at the
    fake tree. Returns the resolved repo id.
    """
    resolved = resolve_model(_ALIAS)
    assert "/" in resolved and resolved != _ALIAS, (
        f"test needs a real alias→repo mapping; got {resolved!r}"
    )

    cache_root = tmp_path / "hf-cache"
    repo_root = cache_root / ("models--" + resolved.replace("/", "--"))
    sha = "seed0000seed0000seed0000seed0000seed0000"
    snap = repo_root / "snapshots" / sha
    snap.mkdir(parents=True)
    blobs = repo_root / "blobs"
    blobs.mkdir()
    blob = blobs / "cfgblob"
    blob.write_text("{}")
    (snap / "config.json").symlink_to(blob)  # cache stores config as a symlink
    (repo_root / "refs").mkdir()
    (repo_root / "refs" / "main").write_text(sha)
    if with_weights:
        (snap / "model.safetensors").write_bytes(b"w" * 4096)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))
    import huggingface_hub.file_download as _fd

    monkeypatch.setattr(_fd, "HF_HUB_CACHE", str(cache_root), raising=False)
    return resolved


@pytest.fixture
def _quiet_version_check(monkeypatch):
    """Neutralise serve_command's interactive upgrade prompt deterministically.

    serve_command reaches the upgrade prompt via a FUNCTION-SCOPE import —
    ``from rapid_mlx._version_check import prompt_upgrade_if_available``
    (cli.py) — so the name is looked up on ``rapid_mlx._version_check`` at call
    time, NOT bound into the ``cli`` module namespace. Patching either
    ``cli`` (no such attribute — a no-op) or a specific module is therefore
    fragile / import-target-dependent.

    The robust, import-target-AGNOSTIC neutralisation is the version check's
    own documented opt-out: ``RAPID_MLX_DISABLE_VERSION_CHECK`` makes the real
    ``prompt_upgrade_if_available`` return ``False`` immediately (``_disabled()``
    short-circuit — no network, no prompt, no ``sys.exit``) wherever it is
    looked up. That keeps the prologue deterministic regardless of how the
    symbol is imported. (``_ensure_model_downloaded`` IS a module-level ``cli``
    function called unqualified, so it is correctly patched on ``cli`` in
    :func:`_capture_stderr_at_download`.)
    """
    monkeypatch.setenv("RAPID_MLX_DISABLE_VERSION_CHECK", "1")
    return monkeypatch


def _capture_stderr_at_download(monkeypatch, capsys):
    """Patch ``_ensure_model_downloaded`` to snapshot stderr at the moment the
    download step begins, then abort. Returns the list the test asserts on."""
    order: list = []

    def _fake_download(model):
        order.append(capsys.readouterr().err)
        raise _StopServeError()

    monkeypatch.setattr(cli, "_ensure_model_downloaded", _fake_download)
    return order


def test_serve_resolves_alias_and_emits_stub_notice_before_download(
    tmp_path, _quiet_version_check, capsys
):
    """BLOCKING regression: serve_command must canonicalize the shorthand
    alias → ``org/repo`` before probing, so a stub cached under the resolved
    id produces the notice. The notice reaches stderr BEFORE the download."""
    monkeypatch = _quiet_version_check
    resolved = _seed_alias_cache(monkeypatch, tmp_path, with_weights=False)
    order = _capture_stderr_at_download(monkeypatch, capsys)

    ns = _serve_ns()
    assert ns.model == _ALIAS, "serve_command must be entered with the shorthand alias"

    with pytest.raises(_StopServeError):
        cli.serve_command(ns)

    assert len(order) == 1, "download step must run exactly once"
    err_at_download = order[0]
    assert resolved in err_at_download, (
        "notice must name the RESOLVED repo id (proves alias canonicalization "
        f"before the cache probe); stderr was: {err_at_download!r}"
    )
    assert "config cached but its model weights are missing" in err_at_download


def test_serve_alias_full_cache_emits_nothing(tmp_path, _quiet_version_check, capsys):
    """Negative: a fully-weighted alias cache is NOT a stub, so no notice is
    emitted — but the download step still runs (wiring stays on the normal
    path)."""
    monkeypatch = _quiet_version_check
    _seed_alias_cache(monkeypatch, tmp_path, with_weights=True)
    order = _capture_stderr_at_download(monkeypatch, capsys)

    ns = _serve_ns()
    with pytest.raises(_StopServeError):
        cli.serve_command(ns)

    assert len(order) == 1
    assert "config cached but" not in order[0]
    assert "weights are missing" not in order[0]
