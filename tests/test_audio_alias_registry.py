# SPDX-License-Identifier: Apache-2.0
"""R10-C1 — audio alias registry + audio-serve-mode dispatch.

Bo r10-R1 found 0/8 audio aliases boot on 0.8.11. Root cause traced
to ``vllm_mlx.cli.serve_command``:

* Short aliases (``kokoro``, ``whisper``, ``parakeet``...) had no
  resolution at all in serve, so ``_ensure_model_downloaded`` queried
  HuggingFace for ``hf.co/kokoro`` and 404'd.
* Full HF ids of audio repos (``mlx-community/Kokoro-82M-bf16``)
  downloaded fine but then crashed inside ``mlx_lm.load_model``
  because audio repos don't ship safetensors.

Codex r8-A r3 predicted this exact regression; Bo r9 + r10 confirmed
it stayed broken across 2 releases.

R10-C1 introduces a single source of truth (``vllm_mlx/audio/aliases.json``)
and an audio-serve-mode fork in ``serve_command`` that routes audio
names to the audio engines and skips the text-LM loader entirely.

These tests pin the contract:

* Registry resolution covers every documented major audio alias.
* HF-id reverse lookup works (full ids are first-class).
* serve_command forks BEFORE the text-LM loader runs.
* ``rapid-mlx models`` advertises the audio alias surface.
* The boot guard still fires for missing-extra installs (no
  regression of R6-H4).
* Text models do NOT trip the audio path (no regression of the
  ~80 text aliases).
"""

from __future__ import annotations

import importlib.util
from argparse import Namespace
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# A) Registry resolution table
# ---------------------------------------------------------------------------


class TestAudioAliasRegistry:
    """The registry is the single source of truth — every documented
    short alias and major HF id must resolve correctly.
    """

    @pytest.mark.parametrize(
        "alias,expected_type,expected_hf_id",
        [
            # Kokoro family — TTS.
            ("kokoro", "tts", "mlx-community/Kokoro-82M-bf16"),
            ("kokoro-82m", "tts", "mlx-community/Kokoro-82M-bf16"),
            ("kokoro-82m-bf16", "tts", "mlx-community/Kokoro-82M-bf16"),
            ("kokoro-82m-4bit", "tts", "mlx-community/Kokoro-82M-4bit"),
            ("kokoro-82m-8bit", "tts", "mlx-community/Kokoro-82M-8bit"),
            ("kokoro-4bit", "tts", "mlx-community/Kokoro-82M-4bit"),
            ("kokoro-8bit", "tts", "mlx-community/Kokoro-82M-8bit"),
            # Chatterbox.
            ("chatterbox", "tts", "mlx-community/chatterbox-turbo-fp16"),
            ("chatterbox-4bit", "tts", "mlx-community/chatterbox-turbo-4bit"),
            # VibeVoice.
            ("vibevoice", "tts", "mlx-community/VibeVoice-Realtime-0.5B-4bit"),
            (
                "vibevoice-realtime",
                "tts",
                "mlx-community/VibeVoice-Realtime-0.5B-4bit",
            ),
            # VoxCPM.
            ("voxcpm", "tts", "mlx-community/VoxCPM1.5"),
            # Dia.
            ("dia", "tts", "mlx-community/Dia-1.6B-4bit"),
            # Whisper family — STT.
            ("whisper", "stt", "mlx-community/whisper-large-v3-mlx"),
            ("whisper-1", "stt", "mlx-community/whisper-large-v3-mlx"),
            ("whisper-large-v3", "stt", "mlx-community/whisper-large-v3-mlx"),
            (
                "whisper-large-v3-turbo",
                "stt",
                "mlx-community/whisper-large-v3-turbo",
            ),
            ("whisper-medium", "stt", "mlx-community/whisper-medium-mlx"),
            ("whisper-small", "stt", "mlx-community/whisper-small-mlx"),
            ("whisper-base", "stt", "mlx-community/whisper-base-mlx"),
            ("whisper-tiny", "stt", "mlx-community/whisper-tiny-mlx"),
            # Parakeet.
            ("parakeet", "stt", "mlx-community/parakeet-tdt-0.6b-v2"),
            ("parakeet-tdt-0.6b", "stt", "mlx-community/parakeet-tdt-0.6b-v2"),
            (
                "parakeet-tdt-0.6b-v2",
                "stt",
                "mlx-community/parakeet-tdt-0.6b-v2",
            ),
            ("parakeet-v3", "stt", "mlx-community/parakeet-tdt-0.6b-v3"),
            (
                "parakeet-tdt-0.6b-v3",
                "stt",
                "mlx-community/parakeet-tdt-0.6b-v3",
            ),
        ],
    )
    def test_alias_resolves_to_expected_hf_id(
        self, alias, expected_type, expected_hf_id
    ):
        from vllm_mlx.audio.registry import resolve_audio_alias

        entry = resolve_audio_alias(alias)
        assert entry is not None, (
            f"R10-C1 regression: alias {alias!r} returned None from "
            f"the registry — the audio-serve-mode dispatcher in "
            f"serve_command relies on this lookup to fork off the "
            f"text path."
        )
        assert entry.type == expected_type, (
            f"alias {alias!r} resolved to type {entry.type!r}, expected "
            f"{expected_type!r}."
        )
        assert entry.hf_id == expected_hf_id, (
            f"alias {alias!r} resolved to hf_id {entry.hf_id!r}, "
            f"expected {expected_hf_id!r}."
        )

    @pytest.mark.parametrize(
        "alias",
        [
            "KOKORO",
            "Kokoro",
            "WHISPER-LARGE-V3",
            "Whisper-Tiny",
            "PARAKEET",
        ],
    )
    def test_alias_lookup_is_case_insensitive(self, alias):
        """SDKs / docs frequently mix the case (the upstream HF repo
        is ``Kokoro-82M-bf16``); both forms must resolve."""
        from vllm_mlx.audio.registry import resolve_audio_alias

        entry = resolve_audio_alias(alias)
        assert entry is not None, alias

    @pytest.mark.parametrize(
        "hf_id",
        [
            "mlx-community/Kokoro-82M-bf16",
            "mlx-community/Kokoro-82M-4bit",
            "mlx-community/Kokoro-82M-8bit",
            "mlx-community/whisper-large-v3-mlx",
            "mlx-community/whisper-large-v3-turbo",
            "mlx-community/whisper-tiny-mlx",
            "mlx-community/whisper-base-mlx",
            "mlx-community/parakeet-tdt-0.6b-v2",
            "mlx-community/parakeet-tdt-0.6b-v3",
            "mlx-community/chatterbox-turbo-fp16",
            "mlx-community/chatterbox-turbo-4bit",
            "mlx-community/VibeVoice-Realtime-0.5B-4bit",
            "mlx-community/VoxCPM1.5",
            "mlx-community/Dia-1.6B-4bit",
        ],
    )
    def test_full_hf_id_resolves_to_audio_entry(self, hf_id):
        """A user pasting ``mlx-community/Kokoro-82M-bf16`` into
        ``rapid-mlx serve`` MUST be routed to audio mode (not text)
        — pre-R10 this downloaded the repo successfully and then
        crashed in ``mlx_lm.load_model`` because there's no
        safetensors. The reverse-index canonical-alias choice is
        intentionally NOT pinned (multiple aliases can map to the
        same HF id; first-alias-wins is implementation detail), only
        that resolution happens AT ALL and lands on a matching entry."""
        from vllm_mlx.audio.registry import resolve_audio_alias

        entry = resolve_audio_alias(hf_id)
        assert entry is not None, (
            f"Full HF id {hf_id!r} did NOT resolve — the audio-serve-"
            f"mode dispatcher would fall through to the text loader, "
            f"which is the exact Bo r10-R1 failure."
        )
        assert entry.hf_id == hf_id, (
            f"HF id {hf_id!r} resolved to entry with hf_id "
            f"{entry.hf_id!r} — reverse index returned the wrong entry."
        )

    @pytest.mark.parametrize(
        "name",
        [
            "",
            None,
            "qwen3.6-27b-4bit",
            "qwen3.5-122b-mxfp4",
            "ui-tars-1.5-7b-4bit",
            "gemma-3-27b-4bit",
            "embeddinggemma-300m-6bit",
            "mlx-community/Qwen3.6-27B-4bit",
            "mlx-community/Llama-3.3-70B-Instruct",
        ],
    )
    def test_non_audio_names_return_none(self, name):
        """Text + vision aliases MUST NOT resolve through the audio
        registry — over-eager matching would route text models to
        the audio loader."""
        from vllm_mlx.audio.registry import resolve_audio_alias

        assert resolve_audio_alias(name) is None, name

    def test_registry_has_expected_minimum_size(self):
        """The audio surface must cover the major models the brief
        called out. Pinning the count prevents an accidental delete
        from silently shrinking the surface."""
        from vllm_mlx.audio.registry import list_audio_aliases

        entries = list_audio_aliases()
        assert len(entries) >= 20, (
            f"audio registry shrunk to {len(entries)} aliases — "
            f"R10-C1 contract requires the major TTS/STT families."
        )

    def test_registry_separates_tts_and_stt(self):
        """Every entry has a ``type`` that's either 'tts' or 'stt'."""
        from vllm_mlx.audio.registry import list_audio_aliases

        kinds = {e.type for e in list_audio_aliases()}
        assert kinds == {"tts", "stt"}, kinds

    def test_route_alias_tables_built_from_registry(self):
        """The STT/TTS alias maps in routes.audio mirror the registry
        — a single JSON edit must reach every consumer.
        """
        from vllm_mlx.audio.registry import stt_aliases, tts_aliases
        from vllm_mlx.routes.audio import STT_MODEL_ALIASES, TTS_MODEL_ALIASES

        assert stt_aliases() == STT_MODEL_ALIASES
        assert tts_aliases() == TTS_MODEL_ALIASES


# ---------------------------------------------------------------------------
# B) is_audio_model_alias — registry-first contract
# ---------------------------------------------------------------------------


class TestIsAudioModelAlias:
    """Boot guard classifier must recognise every registered audio
    alias AND maintain the legacy substring fallback for HF ids of
    third-party audio engines that haven't been added to the registry
    yet."""

    @pytest.mark.parametrize(
        "name",
        [
            # Registry hits.
            "kokoro",
            "kokoro-82m-bf16",
            "kokoro-82m-8bit",
            "whisper",
            "whisper-1",
            "parakeet",
            "parakeet-tdt-0.6b",
            "chatterbox",
            "vibevoice",
            "voxcpm",
            "dia",
            # Full HF ids.
            "mlx-community/Kokoro-82M-bf16",
            "mlx-community/whisper-tiny-mlx",
            "mlx-community/parakeet-tdt-0.6b-v2",
            # Substring-fallback (not in registry but contains an
            # audio token — third-party ports must still trip the
            # boot guard).
            "mlx-community/whisper-japanese-asr",
            "someuser/Kokoro-cantonese-fork",
        ],
    )
    def test_audio_names_recognised(self, name):
        from vllm_mlx.audio.probe import is_audio_model_alias

        assert is_audio_model_alias(name), name

    @pytest.mark.parametrize(
        "name",
        [
            "qwen3.6-27b-4bit",
            "ui-tars-1.5-7b-4bit",
            "mlx-community/Llama-3.3-70B-Instruct",
            "",
            None,
        ],
    )
    def test_non_audio_names_ignored(self, name):
        from vllm_mlx.audio.probe import is_audio_model_alias

        assert not is_audio_model_alias(name), name


# ---------------------------------------------------------------------------
# C) Audio-serve-mode dispatch in serve_command
# ---------------------------------------------------------------------------


def _make_serve_args(model: str) -> Namespace:
    return Namespace(
        model=model,
        embedding_model=None,
        no_mllm=True,
        mllm=False,
        max_tokens=None,
        api_key=None,
        timeout=60,
        max_request_bytes=None,
        cors_origins=None,
        rate_limit=0,
        log_level="INFO",
        host="127.0.0.1",
        port=8000,
        listen_fd=None,
    )


class TestAudioServeModeDispatch:
    """``rapid-mlx serve kokoro`` (and every other audio alias /
    audio HF id) MUST route to the audio path, skipping the text
    loader. This is the headline R10-C1 fix.
    """

    def test_serve_kokoro_short_alias_routes_to_audio_mode(self):
        """``serve kokoro`` -> _serve_audio_mode, NEVER touches
        _ensure_model_downloaded with the unresolved alias."""
        from vllm_mlx import cli

        with (
            patch.object(cli, "_serve_audio_mode") as mock_audio,
            patch.object(cli, "_ensure_model_downloaded") as mock_download,
            patch("vllm_mlx.server.load_model") as mock_load,
        ):
            args = _make_serve_args("kokoro")
            cli.serve_command(args)

        assert mock_audio.called, (
            "R10-C1 regression: serve_command did NOT route the "
            "``kokoro`` alias to _serve_audio_mode. Pre-R10 the alias "
            "fell through to _ensure_model_downloaded (HF 404)."
        )
        assert not mock_download.called, (
            "Audio-mode must skip _ensure_model_downloaded — the audio "
            "loader pulls weights on demand via the route handlers."
        )
        assert not mock_load.called, (
            "Audio-mode must skip the text-LM load_model — calling it "
            "with an audio repo crashes in mlx_lm because audio repos "
            "don't ship safetensors."
        )

    @pytest.mark.parametrize(
        "alias",
        [
            "kokoro",
            "kokoro-82m",
            "kokoro-82m-bf16",
            "kokoro-82m-8bit",
            "whisper",
            "whisper-1",
            "whisper-large-v3",
            "whisper-tiny",
            "parakeet",
            "chatterbox",
            "vibevoice",
            "voxcpm",
            # Full HF ids of audio models — same audio routing.
            "mlx-community/Kokoro-82M-bf16",
            "mlx-community/whisper-tiny-mlx",
            "mlx-community/parakeet-tdt-0.6b-v2",
        ],
    )
    def test_every_audio_name_takes_the_audio_fork(self, alias):
        from vllm_mlx import cli

        with (
            patch.object(cli, "_serve_audio_mode") as mock_audio,
            patch.object(cli, "_ensure_model_downloaded") as mock_download,
        ):
            args = _make_serve_args(alias)
            cli.serve_command(args)

        assert mock_audio.called, alias
        assert not mock_download.called, alias

    def test_serve_audio_mode_resolves_alias_to_hf_id(self):
        """The audio fork stamps args.model with the resolved HF id
        so audio routes / telemetry see a real repo path."""
        from vllm_mlx import cli

        captured = {}

        def _capture(args, entry):
            captured["args_model"] = args.model
            captured["original_alias"] = args._original_alias
            captured["entry_hf_id"] = entry.hf_id
            captured["entry_type"] = entry.type

        with patch.object(cli, "_serve_audio_mode", side_effect=_capture):
            args = _make_serve_args("kokoro")
            cli.serve_command(args)

        assert captured["args_model"] == "mlx-community/Kokoro-82M-bf16"
        assert captured["original_alias"] == "kokoro"
        assert captured["entry_hf_id"] == "mlx-community/Kokoro-82M-bf16"
        assert captured["entry_type"] == "tts"


# ---------------------------------------------------------------------------
# D) Text-model boot path must NOT regress
# ---------------------------------------------------------------------------


class TestTextBootDoesNotRegress:
    """The most important negative test: text aliases must STILL
    reach the text loader. A bug in the audio fork that routed every
    model through audio would silently break ~80 text aliases.
    """

    @pytest.mark.parametrize(
        "model",
        [
            "qwen3.6-27b-4bit",
            "qwen3.5-122b-mxfp4",
            "gemma-3-27b-4bit",
            "ui-tars-1.5-7b-4bit",
            "mlx-community/Qwen3.6-27B-4bit",
            "mlx-community/Llama-3.3-70B-Instruct",
        ],
    )
    def test_text_model_does_not_take_audio_fork(self, model):
        from vllm_mlx import cli

        with (
            patch.object(cli, "_serve_audio_mode") as mock_audio,
            patch(
                "vllm_mlx._version_check.prompt_upgrade_if_available",
                side_effect=SystemExit(0),
            ),
        ):
            args = _make_serve_args(model)
            with pytest.raises(SystemExit):
                cli.serve_command(args)

        assert not mock_audio.called, (
            f"Text alias {model!r} accidentally took the audio fork — "
            f"this would break every text-LM boot."
        )


# ---------------------------------------------------------------------------
# E) Boot guard install hint still fires when [audio] missing
# ---------------------------------------------------------------------------


class TestBootGuardStillFires:
    """R6-H4 contract: when ``mlx_audio`` is missing, an audio alias
    must exit 2 with the install hint BEFORE the audio fork runs.
    R10-C1 must not weaken this guard."""

    def test_missing_extra_exits_2_for_kokoro(self, monkeypatch, capsys):
        real_find_spec = importlib.util.find_spec

        def _find_spec(name, *a, **kw):
            if name == "mlx_audio":
                return None
            return real_find_spec(name, *a, **kw)

        monkeypatch.setattr(importlib.util, "find_spec", _find_spec)
        # Reset the lane cache so the test sees the fresh probe.
        from vllm_mlx.audio import probe

        probe._reset_probe_cache()

        from vllm_mlx import cli

        args = _make_serve_args("kokoro")
        with pytest.raises(SystemExit) as excinfo:
            cli.serve_command(args)

        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "kokoro" in err
        assert "[audio]" in err
        assert "pip install" in err


# ---------------------------------------------------------------------------
# F) rapid-mlx models advertises the audio surface
# ---------------------------------------------------------------------------


class TestModelsCommandListsAudio:
    """``rapid-mlx models`` must surface the audio aliases so users
    can discover them without reading the docs site. Pre-R10 the
    listing was text-only (audio surface was zero rows)."""

    def test_models_lists_kokoro_and_whisper_and_parakeet(self, capsys):
        from argparse import Namespace

        from vllm_mlx import cli

        args = Namespace(cached=False)
        cli.models_command(args)

        out = capsys.readouterr().out
        # Each of the registry's headline aliases shows up.
        for alias in (
            "kokoro",
            "whisper-large-v3",
            "parakeet",
            "chatterbox",
            "vibevoice",
            "voxcpm",
        ):
            assert alias in out, (
                f"R10-C1 regression: ``rapid-mlx models`` did not list "
                f"the audio alias {alias!r}. The audio surface must be "
                f"discoverable in-tool, not docs-only."
            )

        # The audio kind tag is present.
        assert "[audio:tts]" in out
        assert "[audio:stt]" in out
        # The audio section header.
        assert "Audio models" in out

    def test_models_still_lists_text_aliases(self, capsys):
        """Adding the audio section must not displace the text listing."""
        from argparse import Namespace

        from vllm_mlx import cli

        args = Namespace(cached=False)
        cli.models_command(args)

        out = capsys.readouterr().out
        # Sample text alias — pick a well-known one from aliases.json.
        assert "qwen3.6" in out.lower() or "qwen3.5" in out.lower(), (
            "Text alias listing disappeared after the audio section "
            "was added — both must coexist."
        )
