# SPDX-License-Identifier: Apache-2.0
"""``serve`` must refuse an image-gen model whose download ended early.

The prefetch inside ``serve_command`` swallows transport errors by design
("Pre-download skipped; server will retry"), which is the right call for a
text model — its loader retries and raises something legible. mflux has no
such backstop: it loads whichever ``*.safetensors`` reached disk, never reads
the index beside them, and renders noise. A server booted on that snapshot can
only produce garbage, so the failure belongs here, while the operator is still
watching the command they typed.

Drives the real ``cli.main()`` with the heavy boundaries stubbed, following the
pattern in ``test_disk_stream_cli_wiring.py``.
"""

from __future__ import annotations

import sys

import pytest

_ALIAS = "flux2-klein-4b"
_REPO_DIR = "models--Runpod--FLUX.2-klein-4B-mflux-4bit"


def _seed_cache(tmp_path, monkeypatch, *, omit=None):
    """Write an mflux snapshot into a throwaway HF cache.

    ``omit`` drops one ``(component, shard)`` so the layout matches what an
    interrupted pull leaves: every small file present, one big shard absent.
    """
    repo_root = tmp_path / "hf-cache" / _REPO_DIR
    snap = repo_root / "snapshots" / ("f" * 40)
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
    (repo_root / "refs").mkdir(parents=True)
    (repo_root / "refs" / "main").write_text("f" * 40)
    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path / "hf-cache")
    )


def _drive_serve(monkeypatch):
    """Run ``rapid-mlx serve flux2-klein-4b`` with the heavy boundaries stubbed.

    Only ever driven to the point where the gate refuses. Letting a serve run
    past it would configure the module-level FastAPI app and poison every
    later test in the session, so the complete-snapshot half of this contract
    is pinned at unit level (``test_download_gate`` / ``test_image_lane``)
    rather than by booting a second server here.
    """
    from vllm_mlx import cli, server

    monkeypatch.setattr(server, "load_model", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_run_uvicorn", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_ensure_model_downloaded", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_port_preflight_or_die", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_check_memory_capacity", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_check_alias_min_memory", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_resolve_audio_model_for_serve", lambda _n: None)
    monkeypatch.setattr("vllm_mlx.api.utils.is_mllm_model", lambda _n: False)
    monkeypatch.setattr("vllm_mlx.audio.probe.is_audio_model_alias", lambda _n: False)
    monkeypatch.setattr(
        "vllm_mlx._version_check.prompt_upgrade_if_available", lambda: False
    )
    monkeypatch.setattr(
        "vllm_mlx._version_check.print_staleness_warning_if_any",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "serve", _ALIAS, "--port", "0"])
    cli.main()


def test_complete_image_model_starts_without_network(tmp_path, monkeypatch):
    """A fully-downloaded mflux checkpoint must start offline.

    ``is_repo_cached`` only knows the root ``model*.safetensors`` layout, so it
    answers False for every mflux repo however complete it is. That sent each
    image-model start through the disk-space probe and the mirror — a network
    round-trip on the warm path, which hangs rather than merely slows when DNS
    is poisoned (socket stuck in SYN_SENT, UI stuck on "Starting").
    """
    from vllm_mlx import cli

    _seed_cache(tmp_path, monkeypatch)

    def _no_network(*_a, **_kw):
        raise AssertionError("a complete image model reached the network")

    monkeypatch.setattr(cli, "_check_disk_space", _no_network)
    monkeypatch.setattr(cli, "_try_mirror_prefetch", _no_network)

    cli._ensure_model_downloaded("Runpod/FLUX.2-klein-4B-mflux-4bit")


def test_serve_refuses_partially_downloaded_image_model(tmp_path, monkeypatch, capsys):
    """Exit non-zero naming the model, instead of serving a noise generator.

    Reaching this exit at all is half the contract: it proves the gate sits on
    the real ``serve`` path and reads the alias-resolved repo id, not the alias
    the user typed.
    """
    _seed_cache(tmp_path, monkeypatch, omit=("transformer", "0.safetensors"))

    with pytest.raises(SystemExit) as excinfo:
        _drive_serve(monkeypatch)

    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert _ALIAS in err, "the operator must see which model to re-pull"
    assert "transformer/0.safetensors" in err
