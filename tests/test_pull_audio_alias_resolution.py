# SPDX-License-Identifier: Apache-2.0
"""#991 — ``rapid-mlx pull <audio-alias>`` must resolve to the concrete HF id.

Regression: the CLI dispatch resolves *text* aliases (``resolve_model`` over
``aliases.json``) before handing ``args.model`` to ``pull_command`` /
``rm_command``. Audio aliases live in the separate audio registry and were
never resolved on that path, so ``rapid-mlx pull whisper`` reached
``pull_command`` as the literal string ``"whisper"``, missed the R2 mirror
catalog (keyed by ``hf_path``), and 404'd at HuggingFace — even though
``serve whisper`` and ``pull mlx-community/whisper-large-v3-mlx`` both work.

``serve`` deliberately keeps the short alias (it resolves audio at request
time); only ``pull``/``rm`` — which consume ``args.model`` verbatim — need the
concrete HF id stamped up front. These tests pin both halves of that contract.
"""

from __future__ import annotations

import sys

import pytest

from vllm_mlx import cli
from vllm_mlx.cli import _resolve_audio_download_alias

# --- pure helper -----------------------------------------------------------


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("whisper-tiny", "mlx-community/whisper-tiny-mlx"),
        ("whisper", "mlx-community/whisper-large-v3-mlx"),
        ("kokoro", "mlx-community/Kokoro-82M-bf16"),
        ("parakeet", "mlx-community/parakeet-tdt-0.6b-v2"),
    ],
)
def test_helper_resolves_audio_alias_for_pull(alias: str, expected: str) -> None:
    """``pull`` maps a short audio alias to its registry HF id."""
    assert _resolve_audio_download_alias("pull", alias) == expected


def test_helper_resolves_audio_alias_for_rm() -> None:
    """``rm`` gets the same resolution (cache is scanned by HF id)."""
    assert (
        _resolve_audio_download_alias("rm", "whisper")
        == "mlx-community/whisper-large-v3-mlx"
    )


@pytest.mark.parametrize("command", ["serve", "chat", "run", "bench", "info", None])
def test_helper_leaves_short_alias_for_non_download_commands(command) -> None:
    """Only ``pull``/``rm`` rewrite. ``serve`` (and friends) keep the short
    alias so their request-time audio resolution still fires."""
    assert _resolve_audio_download_alias(command, "whisper") is None


@pytest.mark.parametrize("model", ["qwen3.6-27b-4bit", "not-a-real-alias-xyz", ""])
def test_helper_returns_none_for_non_audio(model: str) -> None:
    """Non-audio names get no rewrite even under ``pull`` — the text/HF path
    handles them (or fails with its own unknown-model help)."""
    assert _resolve_audio_download_alias("pull", model) is None


# --- end-to-end dispatch (main) --------------------------------------------


def _run_main(monkeypatch, argv: list[str]) -> None:
    """Invoke ``cli.main()`` with a fixed argv and telemetry disabled."""
    monkeypatch.setenv("RAPID_MLX_TELEMETRY", "0")
    # Bypass the interactive download-size gate so the dispatch never touches
    # the network regardless of the runner's TTY state.
    monkeypatch.setenv("RAPID_MLX_AUTO_PULL", "1")
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "--no-telemetry", *argv])
    cli.main()


def test_main_pull_rewrites_audio_alias_to_hf_id(monkeypatch) -> None:
    """End-to-end: ``rapid-mlx pull whisper-tiny`` reaches ``pull_command``
    with ``args.model`` rewritten to the HF id and the original alias
    preserved for the banner / summary."""
    captured: dict[str, object] = {}

    def _fake_pull(args) -> None:
        captured["model"] = args.model
        captured["original"] = getattr(args, "_original_alias", None)

    monkeypatch.setattr(cli, "pull_command", _fake_pull)
    _run_main(monkeypatch, ["pull", "whisper-tiny"])

    assert captured["model"] == "mlx-community/whisper-tiny-mlx"
    assert captured["original"] == "whisper-tiny"


def test_main_rm_rewrites_audio_alias_to_hf_id(monkeypatch) -> None:
    """``rapid-mlx rm whisper`` reaches ``rm_command`` with the HF id so the
    cache scan (``models--<owner>--<repo>``) can actually match."""
    captured: dict[str, object] = {}

    def _fake_rm(args) -> None:
        captured["model"] = args.model

    monkeypatch.setattr(cli, "rm_command", _fake_rm)
    _run_main(monkeypatch, ["rm", "whisper"])

    assert captured["model"] == "mlx-community/whisper-large-v3-mlx"


def test_main_serve_keeps_short_audio_alias(monkeypatch) -> None:
    """``rapid-mlx serve whisper`` must NOT be rewritten at dispatch — serve
    resolves audio at request time and relies on the short alias."""
    captured: dict[str, object] = {}

    def _fake_serve(args) -> None:
        captured["model"] = args.model
        captured["original"] = getattr(args, "_original_alias", None)

    monkeypatch.setattr(cli, "serve_command", _fake_serve)
    _run_main(monkeypatch, ["serve", "whisper"])

    assert captured["model"] == "whisper"
    # No alias banner was stamped, so the download-path original is unset.
    assert captured["original"] is None


def test_main_pull_full_hf_path_unchanged(monkeypatch) -> None:
    """A full HF path for an audio repo passes straight through — the mirror
    catalog is keyed by ``hf_path``, so no rewrite is needed or wanted."""
    captured: dict[str, object] = {}

    def _fake_pull(args) -> None:
        captured["model"] = args.model

    monkeypatch.setattr(cli, "pull_command", _fake_pull)
    _run_main(monkeypatch, ["pull", "mlx-community/whisper-tiny-mlx"])

    assert captured["model"] == "mlx-community/whisper-tiny-mlx"
