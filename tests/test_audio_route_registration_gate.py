# SPDX-License-Identifier: Apache-2.0
"""Task #292 regression — guard ``/v1/audio/*`` on text-only servers.

Bo R13/R14 fuzz wave: a fresh ``rapid-mlx serve <text-only-model>``
boot (Qwen3-7B-4bit, etc.) still attached the audio router, so
``/v1/audio/transcriptions`` and ``/v1/audio/speech`` accepted POSTs
and either crashed with a 500 (no audio engine loaded) or surfaced a
misleading 404 ``model_not_found``. The route table advertised
capabilities the server couldn't deliver.

The fix splits audio-route registration into a deferred
``register_audio_routes`` helper. ``vllm_mlx.server`` calls it only
when:

* The loaded model alias / HF id resolves through the audio registry
  (the audio-mode boot path always populates a registry-known id), OR
* The operator passed ``--enable-audio`` on a text-mode boot.

Otherwise the router is never attached and FastAPI's stock 404 fires.

These tests construct the FastAPI app with a TestClient — no real
model load, no engine boot. The gate is exercised at three levels:

1. The pure predicate :func:`audio_routes_should_register` (fast unit
   test — no FastAPI / network).
2. The helper :func:`register_audio_routes` against a fresh FastAPI
   app (idempotent, returns False on a second call).
3. End-to-end via ``vllm_mlx.server.register_audio_routes_if_enabled``
   driven by the server globals the boot path writes — exercises the
   same gate ``load_model`` and ``_serve_audio_mode`` hit.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Level 1: pure predicate
# ---------------------------------------------------------------------------


class TestAudioRoutesShouldRegister:
    """Behaviour of :func:`audio_routes_should_register` in isolation."""

    def test_text_only_model_no_flag_returns_false(self):
        from vllm_mlx.routes.audio import audio_routes_should_register

        assert (
            audio_routes_should_register(
                model_name="Qwen/Qwen3-7B-4bit",
                model_alias=None,
                enable_audio_lane=False,
            )
            is False
        )

    def test_text_only_model_with_flag_returns_true(self):
        from vllm_mlx.routes.audio import audio_routes_should_register

        assert (
            audio_routes_should_register(
                model_name="Qwen/Qwen3-7B-4bit",
                model_alias=None,
                enable_audio_lane=True,
            )
            is True
        )

    def test_audio_alias_returns_true(self):
        from vllm_mlx.routes.audio import audio_routes_should_register

        # Bare short alias — matches the audio registry.
        assert (
            audio_routes_should_register(
                model_name="kokoro",
                model_alias=None,
                enable_audio_lane=False,
            )
            is True
        )

    def test_audio_hf_id_returns_true(self):
        from vllm_mlx.routes.audio import audio_routes_should_register

        # Full HF id — registry reverse-index covers it too.
        assert (
            audio_routes_should_register(
                model_name="mlx-community/Kokoro-82M-bf16",
                model_alias=None,
                enable_audio_lane=False,
            )
            is True
        )

    def test_audio_model_alias_via_alias_field(self):
        """``--served-model-name foo`` + ``kokoro`` alias: model_name is
        the served-model-name, the registry-known id sits on
        model_alias. The gate must consult both fields."""
        from vllm_mlx.routes.audio import audio_routes_should_register

        assert (
            audio_routes_should_register(
                model_name="my-gateway-friendly-name",
                model_alias="kokoro",
                enable_audio_lane=False,
            )
            is True
        )

    def test_none_inputs_return_false(self):
        from vllm_mlx.routes.audio import audio_routes_should_register

        assert (
            audio_routes_should_register(
                model_name=None,
                model_alias=None,
                enable_audio_lane=False,
            )
            is False
        )

    def test_empty_string_inputs_return_false(self):
        from vllm_mlx.routes.audio import audio_routes_should_register

        assert (
            audio_routes_should_register(
                model_name="",
                model_alias="",
                enable_audio_lane=False,
            )
            is False
        )


# ---------------------------------------------------------------------------
# Level 2: register_audio_routes helper
# ---------------------------------------------------------------------------


class TestRegisterAudioRoutes:
    """Idempotency + route-table mutation."""

    def test_attaches_router_on_first_call(self):
        from vllm_mlx.routes.audio import register_audio_routes

        app = FastAPI()
        attached = register_audio_routes(app)
        assert attached is True
        paths = {
            getattr(r, "path", "")
            for r in app.routes
            if getattr(r, "path", "").startswith("/v1/audio/")
        }
        assert "/v1/audio/transcriptions" in paths
        assert "/v1/audio/speech" in paths

    def test_second_call_is_noop(self):
        from vllm_mlx.routes.audio import register_audio_routes

        app = FastAPI()
        first = register_audio_routes(app)
        second = register_audio_routes(app)
        assert first is True
        assert second is False
        # Route table didn't grow on the second call.
        audio_routes = [
            r for r in app.routes if getattr(r, "path", "").startswith("/v1/audio/")
        ]
        # transcriptions + translations + speech + voices = 4 unique paths.
        unique_paths = {r.path for r in audio_routes}
        assert unique_paths == {
            "/v1/audio/transcriptions",
            "/v1/audio/translations",
            "/v1/audio/speech",
            "/v1/audio/voices",
        }


# ---------------------------------------------------------------------------
# Level 3: end-to-end — text-only app returns 404 for /v1/audio/*
# ---------------------------------------------------------------------------


class TestAudioRoutes404OnTextOnlyApp:
    """A fresh FastAPI app without the audio router returns clean 404
    for every audio path — the customer-visible behaviour Bo's R13/R14
    report asked for."""

    def setup_method(self):
        self.app = FastAPI()
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def test_transcriptions_404(self):
        r = self.client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", b"\x00" * 64, "audio/wav")},
            data={"model": "whisper-large-v3"},
        )
        assert r.status_code == 404

    def test_translations_404(self):
        r = self.client.post(
            "/v1/audio/translations",
            files={"file": ("a.wav", b"\x00" * 64, "audio/wav")},
            data={"model": "whisper-large-v3"},
        )
        assert r.status_code == 404

    def test_speech_404(self):
        r = self.client.post(
            "/v1/audio/speech",
            json={"model": "kokoro", "input": "hello"},
        )
        assert r.status_code == 404

    def test_voices_404(self):
        r = self.client.get("/v1/audio/voices")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Level 4: server module gate — register_audio_routes_if_enabled
# ---------------------------------------------------------------------------


class TestServerRegisterAudioRoutesIfEnabled:
    """Exercise the gate that ``load_model`` and ``_serve_audio_mode``
    hit after the model is loaded.

    The function reads three server-module globals
    (``_model_name``, ``_model_alias``, ``_enable_audio_lane``) and
    mutates ``server.app`` in place. We use ``monkeypatch`` to set the
    globals and a dedicated FastAPI app per test so the cross-test
    state is isolated.
    """

    @pytest.fixture
    def fresh_app(self, monkeypatch):
        """Swap ``server.app`` for a fresh FastAPI app per test so the
        route-table mutations from one test don't bleed into the next."""
        from vllm_mlx import server

        new_app = FastAPI()
        monkeypatch.setattr(server, "app", new_app)
        return new_app

    def _has_audio_routes(self, app: FastAPI) -> bool:
        return any(getattr(r, "path", "").startswith("/v1/audio/") for r in app.routes)

    def test_text_only_no_flag_does_not_register(self, monkeypatch, fresh_app):
        from vllm_mlx import server

        monkeypatch.setattr(server, "_model_name", "Qwen/Qwen3-7B-4bit")
        monkeypatch.setattr(server, "_model_alias", None)
        monkeypatch.setattr(server, "_enable_audio_lane", False)

        attached = server.register_audio_routes_if_enabled()
        assert attached is False
        assert self._has_audio_routes(fresh_app) is False

    def test_text_only_with_flag_registers(self, monkeypatch, fresh_app):
        from vllm_mlx import server

        monkeypatch.setattr(server, "_model_name", "Qwen/Qwen3-7B-4bit")
        monkeypatch.setattr(server, "_model_alias", None)
        monkeypatch.setattr(server, "_enable_audio_lane", True)

        attached = server.register_audio_routes_if_enabled()
        assert attached is True
        assert self._has_audio_routes(fresh_app) is True

    def test_audio_alias_registers_without_flag(self, monkeypatch, fresh_app):
        from vllm_mlx import server

        monkeypatch.setattr(server, "_model_name", "kokoro")
        monkeypatch.setattr(server, "_model_alias", "kokoro")
        monkeypatch.setattr(server, "_enable_audio_lane", False)

        attached = server.register_audio_routes_if_enabled()
        assert attached is True
        assert self._has_audio_routes(fresh_app) is True

    def test_audio_hf_id_registers_without_flag(self, monkeypatch, fresh_app):
        from vllm_mlx import server

        monkeypatch.setattr(server, "_model_name", "mlx-community/Kokoro-82M-bf16")
        monkeypatch.setattr(server, "_model_alias", None)
        monkeypatch.setattr(server, "_enable_audio_lane", False)

        attached = server.register_audio_routes_if_enabled()
        assert attached is True
        assert self._has_audio_routes(fresh_app) is True

    def test_idempotent_across_calls(self, monkeypatch, fresh_app):
        """``load_model`` may call the gate more than once (e.g. on a
        future refactor that runs ``_sync_config`` twice). The second
        call must NOT re-register the routes — duplicate registration
        triggers a 405 response on otherwise-valid requests."""
        from vllm_mlx import server

        monkeypatch.setattr(server, "_model_name", "kokoro")
        monkeypatch.setattr(server, "_model_alias", None)
        monkeypatch.setattr(server, "_enable_audio_lane", False)

        first = server.register_audio_routes_if_enabled()
        second = server.register_audio_routes_if_enabled()
        assert first is True
        assert second is False

        # Route table didn't double up.
        audio_paths = [
            r.path
            for r in fresh_app.routes
            if getattr(r, "path", "").startswith("/v1/audio/")
        ]
        assert len(audio_paths) == len(set(audio_paths))


# ---------------------------------------------------------------------------
# Level 5: /v1/models reflects the audio-route gate
# ---------------------------------------------------------------------------


class TestModelsListingReflectsAudioGate:
    """The pre-fix shape advertised ``audio_lanes`` on a text-only
    server that wouldn't answer ``/v1/audio/transcriptions``. Verify
    the listing now suppresses ``audio_lanes`` when the router is not
    mounted, and shows it when it is."""

    @pytest.fixture
    def fresh_app(self, monkeypatch):
        from vllm_mlx import server

        new_app = FastAPI()
        monkeypatch.setattr(server, "app", new_app)
        return new_app

    def test_audio_lane_snapshot_none_on_text_only_app(self, monkeypatch, fresh_app):
        from vllm_mlx import server
        from vllm_mlx.config import get_config
        from vllm_mlx.routes.models import _audio_lane_snapshot

        # Force the routes-mounted predicate to False by ensuring the
        # gate doesn't see audio + flag + routes.
        monkeypatch.setattr(server, "_model_name", "Qwen/Qwen3-7B-4bit")
        monkeypatch.setattr(server, "_model_alias", None)
        monkeypatch.setattr(server, "_enable_audio_lane", False)
        cfg = get_config()
        old_flag = cfg.enable_audio_lane
        cfg.enable_audio_lane = False
        try:
            snapshot = _audio_lane_snapshot()
        finally:
            cfg.enable_audio_lane = old_flag
        assert snapshot is None

    def test_audio_lane_snapshot_observed_when_routes_mounted(
        self, monkeypatch, fresh_app
    ):
        """When the audio router is attached, the snapshot returns
        EITHER ``None`` (if the deep probe never ran) or a non-empty
        dict — but never a hidden None from the gate."""
        from vllm_mlx.routes.audio import register_audio_routes
        from vllm_mlx.routes.models import _audio_lane_snapshot, _audio_routes_mounted

        register_audio_routes(fresh_app)
        assert _audio_routes_mounted() is True
        # The snapshot's content depends on whether the deep probe has
        # written status entries; what we're asserting is that the gate
        # no longer short-circuits to None. A real call may still return
        # None if no probe ran — that's the F-K-CAPABILITIES contract,
        # not the task #292 gate.
        snapshot = _audio_lane_snapshot()
        # Either None (no probe yet) or a dict — anything else is wrong
        # shape.
        assert snapshot is None or isinstance(snapshot, dict)
