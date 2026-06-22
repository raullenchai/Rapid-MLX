# SPDX-License-Identifier: Apache-2.0
"""R6-H4 (Eva 0.8.7 dogfood) — ``rapid-mlx serve <audio-alias>`` boot guard.

Eva: ``rapid-mlx serve kokoro`` (or whisper/parakeet/...) on a venv
without the ``[audio]`` extra started cleanly, printed the banner, and
only crashed on the FIRST audio request — exact same shape r5-C
fixed for UI-TARS in PR #822. The r6-C fix mirrors that pattern:
``require_audio_or_exit`` probes ``mlx_audio`` BEFORE any banner or
download and exits ``2`` with an actionable install hint.

These tests probe the guard helper directly and confirm the alias
classifier covers the documented audio surface.
"""

from __future__ import annotations

import importlib
import importlib.util

import pytest


def test_is_audio_model_alias_recognises_common_aliases() -> None:
    """The substring classifier should catch every audio alias the
    server actually serves — both bare aliases and HF ids."""
    from vllm_mlx.audio.probe import is_audio_model_alias

    audio_positive = [
        # Bare aliases the route exposes today.
        "kokoro",
        "kokoro-4bit",
        "chatterbox",
        "chatterbox-4bit",
        "vibevoice",
        "voxcpm",
        "whisper-large-v3",
        "whisper-large-v3-turbo",
        "whisper-medium",
        "whisper-small",
        "parakeet",
        "parakeet-v3",
        # HF-style ids — capitalisation must not matter.
        "mlx-community/Kokoro-82M-bf16",
        "mlx-community/Kokoro-82M-4bit",
        "mlx-community/whisper-large-v3-mlx",
        "mlx-community/parakeet-tdt-0.6b-v2",
    ]
    for name in audio_positive:
        assert is_audio_model_alias(name), name


def test_is_audio_model_alias_ignores_non_audio() -> None:
    """Text + vision aliases must NOT trip the audio classifier."""
    from vllm_mlx.audio.probe import is_audio_model_alias

    non_audio = [
        "qwen3.6-27b-4bit",
        "qwen3.5-122b-mxfp4",
        "ui-tars-1.5-7b-4bit",
        "gemma-3-27b-4bit",
        "embeddinggemma-300m-6bit",
        "mlx-community/Qwen3.6-27B-4bit",
        # Edge cases.
        "",
        None,  # type: ignore[arg-type]
    ]
    for name in non_audio:
        assert not is_audio_model_alias(name), name


def test_require_audio_or_exit_exits_2_when_mlx_audio_missing(
    monkeypatch, capsys
) -> None:
    """When ``find_spec("mlx_audio")`` returns None, the helper must
    print the install hint to stderr and ``sys.exit(2)``.

    We monkeypatch ``importlib.util.find_spec`` so the test runs even
    on CI runners that have ``mlx-audio`` installed.
    """
    from vllm_mlx.audio import probe

    real_find_spec = importlib.util.find_spec

    def _find_spec_missing(name, *a, **kw):
        if name == "mlx_audio":
            return None
        return real_find_spec(name, *a, **kw)

    monkeypatch.setattr(importlib.util, "find_spec", _find_spec_missing)

    with pytest.raises(SystemExit) as excinfo:
        probe.require_audio_or_exit("kokoro")

    assert excinfo.value.code == 2, (
        f"Boot guard must exit 2 (argparse usage-error code), got "
        f"{excinfo.value.code!r}"
    )
    captured = capsys.readouterr()
    err = captured.err
    assert "kokoro" in err, err
    assert "[audio]" in err, err
    assert "pip install" in err, err
    assert "rapid-mlx[audio]" in err, err


def test_require_audio_or_exit_no_op_when_mlx_audio_present(monkeypatch) -> None:
    """When ``mlx_audio`` is importable, the guard must return cleanly
    (no exit, no stderr noise)."""
    from vllm_mlx.audio import probe

    real_find_spec = importlib.util.find_spec

    def _find_spec_present(name, *a, **kw):
        if name == "mlx_audio":
            # Return a synthetic spec — only truthiness matters here.
            class _Spec:
                pass

            return _Spec()
        return real_find_spec(name, *a, **kw)

    monkeypatch.setattr(importlib.util, "find_spec", _find_spec_present)
    # Should NOT raise SystemExit.
    probe.require_audio_or_exit("kokoro")


def test_serve_command_triggers_audio_boot_guard(monkeypatch, capsys) -> None:
    """End-to-end: an argparse Namespace with ``model="kokoro"`` and
    no ``mlx_audio`` available must exit 2 from ``serve_command``
    BEFORE the heavy boot path runs.

    The guard fires very early — after embedding + vision probes but
    before any model download, weight load, or banner output. We
    monkeypatch ``find_spec`` so ``mlx_audio`` looks missing while
    keeping every other extra intact.
    """
    from argparse import Namespace

    from vllm_mlx import cli

    # Make mlx_audio look uninstalled to the audio boot guard.
    real_find_spec = importlib.util.find_spec

    def _find_spec(name, *a, **kw):
        if name == "mlx_audio":
            return None
        return real_find_spec(name, *a, **kw)

    monkeypatch.setattr(importlib.util, "find_spec", _find_spec)

    # Build a minimal Namespace that satisfies the early ``serve_command``
    # codepath. The boot guard fires before any of the heavier wiring
    # is consulted (download, registry, parser detection).
    args = Namespace(
        model="kokoro",
        embedding_model=None,
        no_mllm=False,
        mllm=False,
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.serve_command(args)

    assert excinfo.value.code == 2, (
        f"Audio boot guard must exit 2 at serve_command entry, got "
        f"{excinfo.value.code!r}"
    )
    captured = capsys.readouterr()
    err = captured.err
    assert "kokoro" in err
    assert "[audio]" in err
    assert "pip install" in err


def test_serve_command_does_not_audio_guard_text_model(monkeypatch) -> None:
    """A non-audio alias must NOT trip the audio guard, even if
    ``mlx_audio`` happens to be missing.

    This guards against an over-eager substring match that would
    block ``rapid-mlx serve qwen3.6-27b-4bit`` on a base install
    (text models don't need the audio extra).
    """
    from argparse import Namespace

    from vllm_mlx import cli
    from vllm_mlx.audio import probe

    # Spy on the audio guard so we can assert it was NEVER called.
    called: list[str] = []

    def _spy(model_name: str) -> None:
        called.append(model_name)

    monkeypatch.setattr(probe, "require_audio_or_exit", _spy)

    # ``serve_command`` calls a LOT of downstream code that we don't
    # want to exercise here — we only need to confirm the audio guard
    # gate is skipped for text models. Force the embedding / vision
    # guards to no-op and then force an early controlled exit by
    # patching ``prompt_upgrade_if_available`` to raise SystemExit.
    from vllm_mlx import _version_check

    def _early_exit():
        raise SystemExit(0)

    monkeypatch.setattr(_version_check, "prompt_upgrade_if_available", _early_exit)

    # Force the vision guard to no-op too, so we know any SystemExit
    # only comes from our injected hook.
    from vllm_mlx.api import utils as api_utils

    monkeypatch.setattr(api_utils, "is_mllm_model", lambda *_a, **_kw: False)

    args = Namespace(
        model="qwen3.6-27b-4bit",
        embedding_model=None,
        no_mllm=True,  # Skip vision guard.
        mllm=False,
    )
    with pytest.raises(SystemExit):
        cli.serve_command(args)

    assert called == [], (
        "Audio boot guard fired for a text alias — substring "
        f"classifier is over-eager: {called!r}"
    )
