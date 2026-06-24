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
        served_model_name=None,
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
# F) audio-serve-mode must sync server globals into ServerConfig
# ---------------------------------------------------------------------------


class TestAudioServeModeSyncsServerConfig:
    """Codex r1 HIGH #1: ``_serve_audio_mode`` MUST call
    ``server._sync_config()`` so the ``ServerConfig`` singleton that
    the auth middleware reads from is populated with ``_api_key``,
    ``_max_request_bytes``, ``_model_name``, etc. — exactly what
    ``server.load_model`` does on the text path. Skipping this sync
    means ``rapid-mlx serve kokoro --api-key SECRET`` would silently
    accept unauthenticated /v1/audio/* requests because the
    middleware reads ``cfg.api_key`` (still ``None``) instead of
    ``server._api_key``.
    """

    def test_api_key_propagates_to_server_config(self):
        """``--api-key`` must reach ``ServerConfig.api_key`` in audio mode."""
        from vllm_mlx import cli, server
        from vllm_mlx.config import get_config

        # Clear any inherited state.
        cfg = get_config()
        cfg.api_key = None
        server._api_key = None

        with (
            patch.object(cli, "_run_uvicorn"),
            patch.object(cli, "_port_preflight_or_die"),
        ):
            args = _make_serve_args("kokoro")
            args.api_key = "SECRET-r10c1"
            cli.serve_command(args)

        # The middleware reads from get_config(), not from server._api_key.
        assert get_config().api_key == "SECRET-r10c1", (
            "R10-C1 audio-serve-mode regression: ``--api-key`` did "
            "NOT reach ``ServerConfig.api_key`` — the auth middleware "
            "reads from the config singleton, so the audio endpoints "
            "would be effectively unauthenticated. Codex r1 HIGH #1."
        )

    def test_model_name_propagates_to_server_config(self):
        """``/v1/models`` reads ``cfg.model_name`` / ``cfg.model_alias``
        — audio mode must populate both so the audio model shows up
        in the listing."""
        from vllm_mlx import cli, server
        from vllm_mlx.config import get_config

        cfg = get_config()
        cfg.model_name = None
        cfg.model_alias = None
        server._model_name = None
        server._model_alias = None

        with (
            patch.object(cli, "_run_uvicorn"),
            patch.object(cli, "_port_preflight_or_die"),
        ):
            args = _make_serve_args("kokoro")
            cli.serve_command(args)

        # model_name = resolved HF id, model_alias = friendly short name.
        assert get_config().model_name == "mlx-community/Kokoro-82M-bf16"
        assert get_config().model_alias == "kokoro"

    def test_max_request_bytes_propagates_to_server_config(self):
        """``--max-request-bytes`` must reach the config singleton."""
        from vllm_mlx import cli, server
        from vllm_mlx.config import get_config

        cfg = get_config()
        cfg.max_request_bytes = 8 * 1024 * 1024  # default
        server._max_request_bytes = 8 * 1024 * 1024

        with (
            patch.object(cli, "_run_uvicorn"),
            patch.object(cli, "_port_preflight_or_die"),
        ):
            args = _make_serve_args("kokoro")
            args.max_request_bytes = 16 * 1024 * 1024
            cli.serve_command(args)

        assert get_config().max_request_bytes == 16 * 1024 * 1024


# ---------------------------------------------------------------------------
# G) rapid-mlx models advertises the audio surface
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


# ---------------------------------------------------------------------------
# H) R11-K / task #258 — audio serve honors --served-model-name +
#    --embedding-model (deferred R10-I carry).
# ---------------------------------------------------------------------------


class TestAudioServeHonorsServedModelName:
    """Pre-fix ``_serve_audio_mode`` silently dropped
    ``--served-model-name``: ``server._model_name`` was always
    ``entry.hf_id``, so operators wrapping ``rapid-mlx serve kokoro``
    behind a gateway with a fixed ``model_name`` saw the raw HF id on
    ``/v1/models`` and the gateway's allowlist 404'd.

    The text path's contract (see ``server.load_model``:
    ``_model_name = served_model_name or model_name``) must mirror
    1:1 on the audio path. ``_model_alias`` must still surface the
    friendly short alias so ``/v1/models`` lists BOTH the custom name
    and the alias — same wire shape as text.
    """

    def test_served_model_name_overrides_audio_model_name(self):
        """``--served-model-name custom-tts`` -> ``cfg.model_name`` == ``custom-tts``."""
        from vllm_mlx import cli, server
        from vllm_mlx.config import get_config

        cfg = get_config()
        cfg.model_name = None
        cfg.model_alias = None
        cfg.model_path = None
        server._model_name = None
        server._model_alias = None
        server._model_path = None

        with (
            patch.object(cli, "_run_uvicorn"),
            patch.object(cli, "_port_preflight_or_die"),
        ):
            args = _make_serve_args("kokoro")
            args.served_model_name = "custom-tts"
            cli.serve_command(args)

        cfg = get_config()
        # The custom name lands on the primary id (what /v1/models'
        # first row reports) — same contract as text mode.
        assert cfg.model_name == "custom-tts", (
            "R11-K / #258 regression: audio --served-model-name was "
            "ignored. ``cfg.model_name`` should hold the operator's "
            "custom name, not the underlying HF id."
        )
        # Friendly alias still exposed so the alias->custom mapping
        # is discoverable on /v1/models.
        assert cfg.model_alias == "kokoro"
        # The underlying HF id remains the engine's input — used as
        # the cache dir and (not affected by --served-model-name).
        assert cfg.model_path == "mlx-community/Kokoro-82M-bf16"

    def test_no_served_model_name_keeps_hf_id_as_model_name(self):
        """Default (no flag) behavior must NOT regress — alias mapping
        stays as it was before R11-K. This guards the 12-alias Bo r12
        dogfood: every alias must still boot with the same wire shape."""
        from vllm_mlx import cli, server
        from vllm_mlx.config import get_config

        cfg = get_config()
        cfg.model_name = None
        cfg.model_alias = None
        cfg.model_path = None
        server._model_name = None
        server._model_alias = None
        server._model_path = None

        with (
            patch.object(cli, "_run_uvicorn"),
            patch.object(cli, "_port_preflight_or_die"),
        ):
            args = _make_serve_args("kokoro")
            # served_model_name is None (default) — fixture is r11-K aware.
            cli.serve_command(args)

        cfg = get_config()
        # Same pre-fix shape: HF id as the primary, alias as the secondary.
        assert cfg.model_name == "mlx-community/Kokoro-82M-bf16"
        assert cfg.model_alias == "kokoro"

    @pytest.mark.parametrize(
        "alias",
        [
            # 12 Bo r12 dogfood aliases — must keep booting cleanly.
            "kokoro",
            "kokoro-4bit",
            "chatterbox",
            "vibevoice",
            "voxcpm",
            "dia",
            "whisper",
            "whisper-1",
            "whisper-tiny",
            "whisper-base",
            "parakeet",
            "parakeet-v3",
        ],
    )
    def test_all_r12_audio_aliases_boot_with_served_model_name(self, alias):
        """No-regression: all 12 Bo r12 audio aliases must still take
        the audio fork AND honor ``--served-model-name``."""
        from vllm_mlx import cli, server
        from vllm_mlx.config import get_config

        cfg = get_config()
        cfg.model_name = None
        cfg.model_alias = None
        server._model_name = None
        server._model_alias = None

        with (
            patch.object(cli, "_run_uvicorn"),
            patch.object(cli, "_port_preflight_or_die"),
        ):
            args = _make_serve_args(alias)
            args.served_model_name = f"gateway/{alias}"
            cli.serve_command(args)

        assert get_config().model_name == f"gateway/{alias}", alias
        assert get_config().model_alias is not None, alias


class TestTextServeServedModelNameDoesNotRegress:
    """The text-mode ``--served-model-name`` contract is fixed at
    ``server.load_model`` (``_model_name = served_model_name or
    model_name``). The R11-K fix only touches the audio dispatcher;
    the text-mode wiring at ``serve_command`` must NOT shift.

    Brittleness note: running ``serve_command`` end-to-end for a
    text alias requires mocking ~40 SchedulerConfig / engine knobs.
    Instead we use AST parsing to pin the two contract points that
    matter (call-site keyword + assignment target). Codex r2 BLOCKING
    feedback flagged the prior substring-check approach as a false
    pass on comments / dead code — the AST walk below is immune to
    that class of bug because it only matches real ``ast.Call`` /
    ``ast.Assign`` nodes, not string literals.
    """

    @staticmethod
    def _is_args_served_model_name(node) -> bool:
        """Strict structural match for ``args.served_model_name``.

        Codex r3 BLOCKING #2: a substring check on ``ast.unparse``
        would false-pass on ``args.served_model_name_backup``,
        ``not args.served_model_name``, or any expression that
        merely *mentions* the attribute. Enforce ``ast.Attribute``
        whose ``value`` is ``ast.Name("args")`` and whose ``attr``
        is exactly ``"served_model_name"``.
        """
        import ast as _ast

        return (
            isinstance(node, _ast.Attribute)
            and isinstance(node.value, _ast.Name)
            and node.value.id == "args"
            and node.attr == "served_model_name"
        )

    @staticmethod
    def _walk_top_level(func_def):
        """Yield every node inside ``func_def`` reachable WITHOUT
        crossing a nested scope boundary.

        Codex r4 BLOCKING: bare ``ast.walk`` descends into nested
        ``FunctionDef`` / ``Lambda`` / ``ClassDef`` bodies, so a
        copy of the contract expression inside an inner helper
        would false-pass the live forwarding check. This walker
        treats those nodes as opaque — descend INTO control flow
        (``If``/``For``/``Try``/``With``) but NOT into other
        scopes that have their own name resolution.

        It also stops at constant-False ``if False:`` branches —
        codex r4 explicitly called out dead-code matches.
        """
        import ast as _ast

        # Scope boundaries the walker must not cross.
        _SCOPE_NODES = (
            _ast.FunctionDef,
            _ast.AsyncFunctionDef,
            _ast.Lambda,
            _ast.ClassDef,
        )

        def _constant_test(if_node):
            """Return ``True`` for ``if True:``-shaped tests,
            ``False`` for ``if False:`` / ``if 0:`` / ``if None:``-
            shaped tests, and ``None`` for any non-constant test.
            """
            test = if_node.test
            if isinstance(test, _ast.Constant):
                return bool(test.value)
            return None

        stack = list(func_def.body)
        while stack:
            node = stack.pop()
            yield node
            # Skip descending into nested scopes — their bodies are
            # treated as opaque.
            if isinstance(node, _SCOPE_NODES):
                continue
            # Constant-test ``If`` nodes: descend ONLY into the
            # branch the compiler would actually execute. Codex r4
            # BLOCKING flagged that the prior implementation only
            # handled ``if False:`` (skip body, take orelse), which
            # left the symmetric ``if True: ... else: <CONTRACT>``
            # case false-passing — the unreachable ``else`` was
            # still walked.
            if isinstance(node, _ast.If):
                truth = _constant_test(node)
                if truth is True:
                    # ``if True:`` — only the body executes.
                    stack.extend(node.body)
                    continue
                if truth is False:
                    # ``if False:`` — only the orelse executes.
                    stack.extend(node.orelse)
                    continue
                # Non-constant test: both branches reachable, walk
                # both via the generic iter_child_nodes path below.
            # Push the children. Use ``iter_child_nodes`` so the
            # walk visits sibling order; nested scope filtering on
            # the next pop handles the recursion stop.
            stack.extend(_ast.iter_child_nodes(node))

    def test_load_model_call_site_threads_served_model_name(self):
        """``serve_command`` forwards ``args.served_model_name`` to
        ``load_model`` (verbatim) and to ``run_dflash_server`` (with
        the ``_alias_name`` fallback).  Strict structural match per
        codex r3 BLOCKING — no substring fuzziness.

        Pinned contracts:
          * ``load_model(served_model_name=args.served_model_name, ...)``
          * ``run_dflash_server(served_model_name=args.served_model_name or _alias_name, ...)``

        Comments and string literals are not ``Call`` nodes — immune
        to the prior false-pass class.
        """
        import ast
        import inspect

        from vllm_mlx import cli

        tree = ast.parse(inspect.getsource(cli.serve_command))
        # Locate the FunctionDef itself (inspect.getsource returns the
        # decorators + def header + body as a module-level fragment).
        func_def = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
        assert func_def is not None and func_def.name == "serve_command", (
            "serve_command source no longer parses to a FunctionDef "
            "named serve_command — the test scaffold needs an update."
        )

        # Codex r4 BLOCKING: walk ONLY top-level reachable statements,
        # NOT nested helper functions or dead-code branches. A call
        # inside an inner ``def _helper(): load_model(...)`` would
        # false-pass under the previous ``ast.walk`` approach.
        # Codex r5 BLOCKING: collect ALL reachable call sites, not
        # just one — a refactor that adds a second ``load_model``
        # call missing the kwarg would slip past a first-hit overwrite.
        load_model_calls = []
        dflash_calls = []
        for node in self._walk_top_level(func_def):
            if not isinstance(node, ast.Call):
                continue
            callee = ""
            if isinstance(node.func, ast.Name):
                callee = node.func.id
            elif isinstance(node.func, ast.Attribute):
                callee = node.func.attr
            if callee == "load_model":
                load_model_calls.append(node)
            elif callee == "run_dflash_server":
                dflash_calls.append(node)

        # ----- load_model contract: every reachable call must
        # forward ``served_model_name=args.served_model_name``
        # verbatim. Codex r5 BLOCKING #1: an overwriting first-hit
        # check would let a second call site drop the kwarg silently.
        assert load_model_calls, (
            "text-mode regression: serve_command no longer calls "
            "``load_model`` at all. The audio fork must NOT short-"
            "circuit the text path."
        )
        for i, call in enumerate(load_model_calls):
            smn_kw = next(
                (kw for kw in call.keywords if kw.arg == "served_model_name"),
                None,
            )
            assert smn_kw is not None, (
                f"text-mode regression: ``load_model(...)`` call "
                f"#{i + 1} of {len(load_model_calls)} no longer "
                "receives a ``served_model_name`` keyword. The R11-K "
                "audio fix MUST NOT touch this contract — every "
                "reachable call site must forward the kwarg."
            )
            # Strict: value MUST be exactly the ``args.served_model_name``
            # attribute access — no wrapping, no fallback, no rename.
            assert self._is_args_served_model_name(smn_kw.value), (
                f"text-mode regression: ``load_model`` call #{i + 1}: "
                "``served_model_name=...`` is no longer exactly "
                f"``args.served_model_name`` (got {ast.unparse(smn_kw.value)!r}). "
                "Any wrapping expression (e.g. "
                "``not args.served_model_name``, "
                "``args.served_model_name_backup``) is a contract break."
            )

        # ----- run_dflash_server contract: every reachable call must
        # forward ``served_model_name=args.served_model_name or _alias_name``.
        # Codex r5 BLOCKING #2 — same all-call-sites rule as above.
        assert dflash_calls, (
            "text-mode regression: serve_command no longer calls "
            "``run_dflash_server`` — DFlash text serve broken."
        )
        for i, call in enumerate(dflash_calls):
            smn_kw_d = next(
                (kw for kw in call.keywords if kw.arg == "served_model_name"),
                None,
            )
            assert smn_kw_d is not None, (
                f"text-mode (DFlash) regression: ``run_dflash_server(...)`` "
                f"call #{i + 1} of {len(dflash_calls)} no longer "
                "receives a ``served_model_name`` keyword."
            )
            # Strict: must be ``<args.served_model_name> or <_alias_name>``
            # — BoolOp with Or + exactly 2 operands in that order.
            # Codex r3 BLOCKING #1: a permissive check would let the
            # ``_alias_name`` fallback get dropped silently.
            d_val = smn_kw_d.value
            assert isinstance(d_val, ast.BoolOp) and isinstance(d_val.op, ast.Or), (
                f"text-mode (DFlash) regression: ``run_dflash_server`` "
                f"call #{i + 1}: ``served_model_name=...`` is no longer "
                f"an ``or`` expression (got {ast.unparse(d_val)!r}). "
                "The required shape is "
                "``args.served_model_name or _alias_name``."
            )
            assert len(d_val.values) == 2, (
                f"text-mode (DFlash) regression: ``run_dflash_server`` "
                f"call #{i + 1}: served_model_name fallback is no "
                "longer exactly 2 operands (got "
                f"{[ast.unparse(v) for v in d_val.values]!r}). The "
                "required shape is "
                "``args.served_model_name or _alias_name``."
            )
            assert self._is_args_served_model_name(d_val.values[0]), (
                f"text-mode (DFlash) regression: ``run_dflash_server`` "
                f"call #{i + 1}: served_model_name first operand is no "
                f"longer ``args.served_model_name`` (got "
                f"{ast.unparse(d_val.values[0])!r})."
            )
            assert (
                isinstance(d_val.values[1], ast.Name)
                and d_val.values[1].id == "_alias_name"
            ), (
                f"text-mode (DFlash) regression: ``run_dflash_server`` "
                f"call #{i + 1}: served_model_name fallback is no "
                f"longer ``_alias_name`` (got "
                f"{ast.unparse(d_val.values[1])!r}). This fallback is "
                "what surfaces the friendly alias on /v1/models when "
                "the operator did NOT pass --served-model-name."
            )

    def test_server_load_model_still_maps_served_to_model_name(self):
        """``server.load_model``: ``_model_name = served_model_name
        or model_name``. The audio dispatcher mirrors this exact
        expression with ``entry.hf_id`` as the fallback; if the text
        side semantics shift the audio mirror would drift.

        Codex r3 BLOCKING #3 strict check: the RHS MUST be an ``Or``
        BoolOp with EXACTLY two operands in EXACTLY the order
        ``served_model_name``, ``model_name``. A permissive check
        would let ``served_model_name or other_name or model_name``
        (which changes the precedence + adds a new fallback) pass
        silently.
        """
        import ast
        import inspect

        from vllm_mlx import server

        tree = ast.parse(inspect.getsource(server.load_model))
        func_def = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
        assert func_def is not None and func_def.name == "load_model", (
            "server.load_model source no longer parses to a FunctionDef "
            "named load_model — the test scaffold needs an update."
        )

        # Codex r4 BLOCKING: top-level statements only — a copy of
        # the assignment inside a nested helper or dead branch
        # would false-pass under bare ``ast.walk``.
        assign_rhs = None
        for node in self._walk_top_level(func_def):
            if not isinstance(node, ast.Assign):
                continue
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "_model_name":
                    assign_rhs = node.value
                    break
            if assign_rhs is not None:
                break

        assert assign_rhs is not None, (
            "server.load_model no longer assigns to ``_model_name`` "
            "in a recognisable shape. The audio dispatcher's mirror "
            "(``server._model_name = served_name or entry.hf_id``) "
            "depends on this assignment existing on the text side; "
            "both must change in lockstep."
        )
        # Strict: BoolOp + Or + exactly two operands, in order.
        assert isinstance(assign_rhs, ast.BoolOp) and isinstance(
            assign_rhs.op, ast.Or
        ), (
            "server.load_model ``_model_name = ...`` RHS is no longer "
            "an ``or`` expression (got "
            f"{ast.unparse(assign_rhs)!r}). The audio dispatcher's "
            "mirror would drift — both sides must move together."
        )
        assert len(assign_rhs.values) == 2, (
            "server.load_model ``_model_name`` RHS is no longer "
            "exactly 2 operands (got "
            f"{[ast.unparse(v) for v in assign_rhs.values]!r}). "
            "Any extra fallback (e.g. ``served_model_name or "
            "other_name or model_name``) changes the contract — "
            "the audio mirror would need to add the same fallback."
        )
        # Order matters: ``served_model_name`` MUST come first, so a
        # user-supplied value wins over the model_name default.
        assert (
            isinstance(assign_rhs.values[0], ast.Name)
            and assign_rhs.values[0].id == "served_model_name"
        ), (
            "server.load_model ``_model_name`` RHS first operand is "
            f"no longer ``served_model_name`` (got "
            f"{ast.unparse(assign_rhs.values[0])!r}). Reversing the "
            "order would make ``model_name`` always win — "
            "--served-model-name would become inert."
        )
        assert (
            isinstance(assign_rhs.values[1], ast.Name)
            and assign_rhs.values[1].id == "model_name"
        ), (
            "server.load_model ``_model_name`` RHS second operand is "
            f"no longer ``model_name`` (got "
            f"{ast.unparse(assign_rhs.values[1])!r}). The fallback "
            "default must remain the underlying engine model id."
        )


class TestAudioServeHonorsEmbeddingModel:
    """R11-K decision call: audio + embedding compose cleanly via
    ``_load_embedding_model_or_exit`` (intentionally orthogonal to
    the text-LM engine). Audio engines stay lazy on /v1/audio/*,
    the embeddings sidecar serves /v1/embeddings on the same app.
    The helper's docstring explicitly anticipated this audio
    integration."""

    def test_embedding_model_pre_loads_in_audio_mode(self):
        """``--embedding-model bge`` on audio serve calls the shared
        helper exactly the same way text mode does."""
        from vllm_mlx import cli, server

        calls = []

        def _capture(name, lock=False):
            calls.append((name, lock))

        with (
            patch.object(cli, "_run_uvicorn"),
            patch.object(cli, "_port_preflight_or_die"),
            patch.object(server, "load_embedding_model", side_effect=_capture),
            patch("vllm_mlx.embedding.require_mlx_embeddings_or_exit"),
        ):
            args = _make_serve_args("kokoro")
            args.embedding_model = "mlx-community/all-MiniLM-L6-v2-4bit"
            cli.serve_command(args)

        assert calls, (
            "R11-K / #258 regression: audio --embedding-model did "
            "NOT pre-load via server.load_embedding_model. Audio + "
            "embedding compose cleanly; the helper must be invoked "
            "the same way text mode invokes it."
        )
        # First positional arg is the model id; lock=True mirrors text mode.
        assert calls[0][0] == "mlx-community/all-MiniLM-L6-v2-4bit"
        assert calls[0][1] is True

    def test_audio_mode_without_embedding_does_not_call_loader(self):
        """No-regression: ``serve kokoro`` (no --embedding-model) must
        NOT trigger the embeddings install guard or loader. The
        embedding loader must stay strictly opt-in."""
        from vllm_mlx import cli, server

        calls = []

        def _capture(name, lock=False):
            calls.append((name, lock))

        with (
            patch.object(cli, "_run_uvicorn"),
            patch.object(cli, "_port_preflight_or_die"),
            patch.object(server, "load_embedding_model", side_effect=_capture),
        ):
            args = _make_serve_args("kokoro")
            # embedding_model stays None (fixture default).
            cli.serve_command(args)

        assert not calls, (
            "Audio mode pre-loaded an embedding model without "
            "--embedding-model — the loader must be strictly opt-in."
        )


class TestAudioServeArgparseAcceptsBothFlags:
    """Argparse-level smoke test: both flags are REGISTERED on the
    serve subparser (they were before the fix, the bug was downstream
    in ``_serve_audio_mode``), so ``rapid-mlx serve kokoro
    --embedding-model ... --served-model-name ...`` parses without
    an argparse error. Guards against an accidental subparser
    refactor that drops these flags."""

    def test_both_flags_parse_on_audio_alias(self, monkeypatch):
        """``rapid-mlx serve kokoro --embedding-model ... --served-model-name ...``
        must parse without an argparse error and forward both flags
        to ``serve_command``. ``main()`` reads sys.argv, so we patch
        argv and stub the dispatch.

        Both flags WERE registered on the serve subparser before
        R11-K (the bug was downstream in ``_serve_audio_mode``); this
        guards against an accidental subparser refactor that drops
        them and makes the audio invocation fail at argparse time.
        """
        import sys as _sys

        from vllm_mlx import cli as _cli

        captured = {}

        def _capture(args):
            captured["served_model_name"] = args.served_model_name
            captured["embedding_model"] = args.embedding_model
            captured["model"] = args.model

        monkeypatch.setattr(
            _sys,
            "argv",
            [
                "rapid-mlx",
                "serve",
                "kokoro",
                "--embedding-model",
                "mlx-community/embeddinggemma-300m-6bit",
                "--served-model-name",
                "my-tts",
                "--port",
                "8102",
            ],
        )
        with patch.object(_cli, "serve_command", side_effect=_capture):
            try:
                _cli.main()
            except SystemExit:
                # ``main()`` may sys.exit(0) on consent / version
                # paths even when the parse succeeded; we only need
                # the captured-args dict populated.
                pass

        assert captured.get("model") == "kokoro"
        assert captured.get("served_model_name") == "my-tts"
        assert (
            captured.get("embedding_model") == "mlx-community/embeddinggemma-300m-6bit"
        )
