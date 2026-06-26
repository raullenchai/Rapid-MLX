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


def _walk_routes(app):
    """Yield all routes attached to ``app``, recursing into the
    ``_IncludedRouter`` wrapper FastAPI 0.137+ introduced. Pre-0.137,
    ``include_router`` flattened child routes directly into
    ``app.routes``; from 0.137 onward an opaque wrapper exposes the
    child router via ``.original_router``. Iterating ``app.routes``
    alone misses every route attached via ``include_router`` on a
    fresh-install fastapi, which is why the audio-gate test suite
    needs this helper to stay version-agnostic.
    """
    for r in app.routes:
        if hasattr(r, "original_router"):
            yield from r.original_router.routes
        else:
            yield r


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
            for r in _walk_routes(app)
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
            r
            for r in _walk_routes(app)
            if getattr(r, "path", "").startswith("/v1/audio/")
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
        return any(
            getattr(r, "path", "").startswith("/v1/audio/") for r in _walk_routes(app)
        )

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
            for r in _walk_routes(fresh_app)
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

    def test_routes_mounted_predicate_ignores_config_flag(self, monkeypatch, fresh_app):
        """Codex r0 BLOCKING #1 regression: ``_audio_routes_mounted``
        must NOT return True merely because ``ServerConfig.enable_audio_lane``
        is set. The flag is the gate INPUT; the route table is the gate
        OUTPUT. A boot path that sets the flag but hasn't yet called
        the registration hook (e.g. an earlier ``/v1/models`` hit during
        text-mode boot before ``load_model`` returns) would otherwise
        advertise ``audio_lanes`` while ``/v1/audio/*`` still 404s —
        the exact contradictory state this PR was opened to eliminate.
        """
        from vllm_mlx.config import get_config
        from vllm_mlx.routes.models import _audio_routes_mounted

        cfg = get_config()
        old_flag = cfg.enable_audio_lane
        cfg.enable_audio_lane = True
        try:
            # No call to register_audio_routes — the route table is empty.
            assert _audio_routes_mounted() is False
        finally:
            cfg.enable_audio_lane = old_flag

    def test_register_audio_routes_not_blocked_by_custom_subpath(self):
        """Codex r0 NIT regression: a custom operator-added
        ``/v1/audio/health`` route mounted BEFORE
        :func:`register_audio_routes` must not block the helper from
        attaching the canonical handlers. Pre-fix the idempotency
        check did a prefix scan on ``/v1/audio/`` which collided with
        any operator subroute under that prefix; the fix uses an
        app-local sentinel attribute instead."""
        from vllm_mlx.routes.audio import register_audio_routes

        app = FastAPI()

        @app.get("/v1/audio/health")
        def _audio_health():
            return {"ok": True}

        attached = register_audio_routes(app)
        assert attached is True
        # Canonical handler is present.
        paths = {
            getattr(r, "path", "")
            for r in _walk_routes(app)
            if getattr(r, "path", "").startswith("/v1/audio/")
        }
        assert "/v1/audio/transcriptions" in paths
        assert "/v1/audio/speech" in paths
        # Custom probe survived.
        assert "/v1/audio/health" in paths


# ---------------------------------------------------------------------------
# Level 6: CLI/server entrypoint wire-up
# ---------------------------------------------------------------------------


class TestCliServeCommandWiresEnableAudioFlag:
    """Codex r1/r2 BLOCKING regression — both ``rapid-mlx serve`` and
    ``python -m vllm_mlx.server`` must thread ``--enable-audio`` all
    the way through to ``register_audio_routes_if_enabled``. A future
    refactor that moves the hook out of ``load_model`` (e.g. into a
    FastAPI lifespan event) must not silently drop the flag for either
    entrypoint.

    These tests invoke the REAL entrypoints (parser construction +
    serve_command) with ``load_model`` / ``_run_uvicorn`` stubbed out
    so we observe the actual wire-up without booting a real engine.
    Codex r2 BLOCKING called out that the prior source-level checks
    pass for dead branches / comments — these execution-based checks
    fix that."""

    def test_module_form_parser_accepts_enable_audio(self, monkeypatch):
        """``python -m vllm_mlx.server <model> --enable-audio`` is
        accepted by the REAL ``server.main`` parser and forwarded to
        the ``_enable_audio_lane`` global before ``load_model`` runs.

        Stubs ``load_model`` and ``uvicorn.run`` so the entrypoint
        returns before doing anything expensive, then asserts the
        global was set BEFORE the stubbed ``load_model`` saw a call.

        ``server.main()`` mutates ServerConfig globals
        (``tool_call_parser``, ``reasoning_parser``, etc.) via inline
        auto-detection — snapshot+restore them so the writes don't
        leak into ``tests/test_routes.py``."""
        from vllm_mlx import server
        from vllm_mlx.config import get_config

        cfg = get_config()
        cfg_snapshot = {
            "tool_call_parser": cfg.tool_call_parser,
            "reasoning_parser": cfg.reasoning_parser,
            "reasoning_parser_name": cfg.reasoning_parser_name,
            "enable_auto_tool_choice": cfg.enable_auto_tool_choice,
            "enable_tool_logits_bias": cfg.enable_tool_logits_bias,
        }
        server_snapshot = {
            "_tool_call_parser": getattr(server, "_tool_call_parser", None),
            "_reasoning_parser": getattr(server, "_reasoning_parser", None),
            "_reasoning_parser_name": getattr(server, "_reasoning_parser_name", None),
            "_enable_auto_tool_choice": getattr(
                server, "_enable_auto_tool_choice", False
            ),
            "_enable_tool_logits_bias": getattr(
                server, "_enable_tool_logits_bias", False
            ),
        }

        seen_global: dict[str, bool] = {}

        def _stub_load_model(*_args, **_kwargs):
            seen_global["enable_audio_lane_at_load"] = server._enable_audio_lane

        def _stub_uvicorn(*_args, **_kwargs):
            pass

        # Stub the port preflight before ``server.main`` lazy-imports it
        # from ``cli`` (``server.py`` line ~1971). Otherwise a port-8000
        # collision on the developer's host fails this test for reasons
        # unrelated to the parser wiring it exercises.
        from vllm_mlx import cli as _cli_for_preflight

        monkeypatch.setattr(
            _cli_for_preflight, "_port_preflight_or_die", lambda *_a, **_kw: None
        )
        monkeypatch.setattr(server, "load_model", _stub_load_model)
        # ``uvicorn.run`` is imported as a top-level name in server.main;
        # patch through the module attr to dodge the real network bind.
        import uvicorn

        monkeypatch.setattr(uvicorn, "run", _stub_uvicorn)
        # Pre-zero the global so we observe the boot path setting it.
        monkeypatch.setattr(server, "_enable_audio_lane", False)

        monkeypatch.setattr(
            "sys.argv",
            [
                "vllm_mlx.server",
                "--model",
                "Qwen/Qwen3-7B-4bit",
                "--enable-audio",
            ],
        )
        try:
            server.main()
            assert seen_global.get("enable_audio_lane_at_load") is True
        finally:
            # Restore the parser/reasoner globals that ``main()``'s
            # auto-detection block writes inline. Without this the
            # writes leak into ``tests/test_routes.py`` (and any other
            # test that observes the default ``tool_call_parser=None``
            # shape on ``/v1/models/<id>``).
            for attr, value in server_snapshot.items():
                setattr(server, attr, value)
            for attr, value in cfg_snapshot.items():
                setattr(cfg, attr, value)

    def test_cli_serve_command_wires_enable_audio_to_server_global(self, monkeypatch):
        """``rapid-mlx serve <model> --enable-audio`` writes
        ``server._enable_audio_lane`` from the parsed value BEFORE
        ``load_model`` runs. Codex r3 BLOCKING #1: drive the REAL
        ``cli.main()`` with the same sys.argv ``rapid-mlx serve``
        builds, stub the heavy downstream calls, observe the actual
        write.

        ``cli.main()`` runs the same argparse parser the production
        binary uses, dispatches to ``serve_command``, and exits via
        ``_run_uvicorn``. We stub:

        * ``load_model`` → records the value of ``server._enable_audio_lane``
          at the moment ``serve_command`` would call the loader.
        * ``_run_uvicorn`` → no-op so we don't bind a port.
        * ``_ensure_model_downloaded`` → no-op so CI doesn't hit HF.
        * ``_port_preflight_or_die`` → no-op (no real socket).
        * ``_resolve_audio_model_for_serve`` → returns ``None`` so we
          take the text-mode branch (the audio branch has its own
          test).
        * ``prompt_upgrade_if_available`` → returns ``False``.
        """
        from vllm_mlx import cli, server
        from vllm_mlx.config import get_config

        # Snapshot the globals ``main()`` / ``serve_command`` will
        # mutate so subsequent tests don't see the writes.
        cfg = get_config()
        cfg_snapshot = {
            "tool_call_parser": cfg.tool_call_parser,
            "reasoning_parser": cfg.reasoning_parser,
            "reasoning_parser_name": cfg.reasoning_parser_name,
            "enable_auto_tool_choice": cfg.enable_auto_tool_choice,
            "model_name": cfg.model_name,
            "model_alias": cfg.model_alias,
        }
        server_snapshot = {
            "_tool_call_parser": getattr(server, "_tool_call_parser", None),
            "_reasoning_parser": getattr(server, "_reasoning_parser", None),
            "_reasoning_parser_name": getattr(server, "_reasoning_parser_name", None),
            "_enable_auto_tool_choice": getattr(
                server, "_enable_auto_tool_choice", False
            ),
            "_model_name": getattr(server, "_model_name", None),
            "_model_alias": getattr(server, "_model_alias", None),
            "_api_key": getattr(server, "_api_key", None),
        }

        seen_global: dict[str, object] = {}

        def _stub_load_model(*_a, **_kw):
            seen_global["enable_audio_lane_at_load"] = server._enable_audio_lane

        # Patch every external side-effect serve_command does before
        # reaching the ``server._enable_audio_lane = ...`` block plus
        # the ``load_model`` boundary where we observe the value.
        monkeypatch.setattr(server, "load_model", _stub_load_model)
        monkeypatch.setattr(cli, "_run_uvicorn", lambda *_a, **_kw: None)
        monkeypatch.setattr(cli, "_ensure_model_downloaded", lambda *_a, **_kw: None)
        monkeypatch.setattr(cli, "_port_preflight_or_die", lambda *_a, **_kw: None)
        monkeypatch.setattr(cli, "_resolve_audio_model_for_serve", lambda _n: None)
        # The audio extra-required guard short-circuits to no-op for
        # text models; belt-and-braces stub it.
        from vllm_mlx.audio import probe as _audio_probe

        monkeypatch.setattr(_audio_probe, "is_audio_model_alias", lambda _n: False)
        # The MLLM extra-required guard short-circuits for text models;
        # belt-and-braces stub it.
        monkeypatch.setattr("vllm_mlx.api.utils.is_mllm_model", lambda _n: False)
        # Disable interactive upgrade prompt.
        monkeypatch.setattr(
            "vllm_mlx._version_check.prompt_upgrade_if_available", lambda: False
        )
        # Stub staleness banner so it doesn't print to stderr.
        monkeypatch.setattr(
            "vllm_mlx._version_check.print_staleness_warning_if_any", lambda: None
        )
        # The ``main()`` alias resolver writes ``args._original_alias``;
        # we want to avoid hitting the real alias registry just to keep
        # the test hermetic. The serve subcommand still works on a raw
        # HF id, so passing one bypasses the resolver.
        # Pre-zero the flag so we observe the write.
        monkeypatch.setattr(server, "_enable_audio_lane", False)

        # Drive the real binary entrypoint.
        monkeypatch.setattr(
            "sys.argv",
            [
                "rapid-mlx",
                "serve",
                "mlx-community/Qwen3-7B-4bit",
                "--enable-audio",
                "--port",
                "0",
            ],
        )
        try:
            cli.main()
        except SystemExit:
            # ``main()`` may bail at various points downstream of the
            # ``_enable_audio_lane`` assignment — we only care that the
            # write happened and the stubbed ``load_model`` observed it.
            pass
        finally:
            for attr, value in server_snapshot.items():
                setattr(server, attr, value)
            for attr, value in cfg_snapshot.items():
                setattr(cfg, attr, value)

        assert seen_global.get("enable_audio_lane_at_load") is True, (
            f"serve_command did not set server._enable_audio_lane "
            f"before load_model; observed: {seen_global}"
        )

    def test_cli_serve_command_calls_register_hook_after_load_model(self, monkeypatch):
        """Codex r1/r2/r3 BLOCKING: drive the REAL ``cli.main()`` and
        record the order of calls to ``load_model``,
        ``register_audio_routes_if_enabled``, and ``_run_uvicorn``.

        The hook MUST run AFTER ``load_model`` so it sees the
        populated ``_model_name`` / ``_model_alias`` server globals,
        and BEFORE ``_run_uvicorn`` so the route table is final by
        the time uvicorn binds the port. The combined ordering
        catches both deletions (codex r1 BLOCKING) and wrong-order
        regressions."""
        from vllm_mlx import cli, server
        from vllm_mlx.config import get_config

        cfg = get_config()
        cfg_snapshot = {
            "tool_call_parser": cfg.tool_call_parser,
            "reasoning_parser": cfg.reasoning_parser,
            "reasoning_parser_name": cfg.reasoning_parser_name,
            "enable_auto_tool_choice": cfg.enable_auto_tool_choice,
            "model_name": cfg.model_name,
            "model_alias": cfg.model_alias,
        }
        server_snapshot = {
            "_tool_call_parser": getattr(server, "_tool_call_parser", None),
            "_reasoning_parser": getattr(server, "_reasoning_parser", None),
            "_reasoning_parser_name": getattr(server, "_reasoning_parser_name", None),
            "_enable_auto_tool_choice": getattr(
                server, "_enable_auto_tool_choice", False
            ),
            "_model_name": getattr(server, "_model_name", None),
            "_model_alias": getattr(server, "_model_alias", None),
            "_api_key": getattr(server, "_api_key", None),
        }

        call_order: list[str] = []

        def _stub_load_model(*_a, **_kw):
            call_order.append("load_model")

        def _stub_register_hook():
            call_order.append("register_hook")
            return True

        def _stub_uvicorn(*_a, **_kw):
            call_order.append("uvicorn")

        monkeypatch.setattr(server, "load_model", _stub_load_model)
        monkeypatch.setattr(
            server, "register_audio_routes_if_enabled", _stub_register_hook
        )
        monkeypatch.setattr(cli, "_run_uvicorn", _stub_uvicorn)
        monkeypatch.setattr(cli, "_ensure_model_downloaded", lambda *_a, **_kw: None)
        monkeypatch.setattr(cli, "_port_preflight_or_die", lambda *_a, **_kw: None)
        monkeypatch.setattr(cli, "_resolve_audio_model_for_serve", lambda _n: None)
        from vllm_mlx.audio import probe as _audio_probe

        monkeypatch.setattr(_audio_probe, "is_audio_model_alias", lambda _n: False)
        monkeypatch.setattr("vllm_mlx.api.utils.is_mllm_model", lambda _n: False)
        monkeypatch.setattr(
            "vllm_mlx._version_check.prompt_upgrade_if_available", lambda: False
        )
        monkeypatch.setattr(
            "vllm_mlx._version_check.print_staleness_warning_if_any", lambda: None
        )

        monkeypatch.setattr(
            "sys.argv",
            [
                "rapid-mlx",
                "serve",
                "mlx-community/Qwen3-7B-4bit",
                "--enable-audio",
                "--port",
                "0",
            ],
        )
        try:
            cli.main()
        except SystemExit:
            pass
        finally:
            for attr, value in server_snapshot.items():
                setattr(server, attr, value)
            for attr, value in cfg_snapshot.items():
                setattr(cfg, attr, value)

        assert "load_model" in call_order, (
            f"load_model was not invoked by serve_command; call order: {call_order}"
        )
        assert "register_hook" in call_order, (
            f"register_audio_routes_if_enabled was not invoked by "
            f"serve_command; call order: {call_order}"
        )
        # Ordering: load_model BEFORE register_hook (so the hook sees
        # ``_model_name`` populated). When ``load_model`` is stubbed
        # the inline call inside it doesn't run, so the explicit
        # serve_command call site is the ONE that fires — and it must
        # come AFTER load_model.
        assert call_order.index("load_model") < call_order.index("register_hook"), (
            f"register_hook ran BEFORE load_model — wrong order: {call_order}"
        )
        if "uvicorn" in call_order:
            assert call_order.index("register_hook") < call_order.index("uvicorn"), (
                f"register_hook must run BEFORE uvicorn; call order: {call_order}"
            )

    def test_load_model_invokes_register_hook(self, monkeypatch):
        """``load_model`` is the SHARED loader between
        ``rapid-mlx serve`` and ``python -m vllm_mlx.server``. Stub
        the engine constructors so the function returns quickly, then
        observe that ``register_audio_routes_if_enabled`` was actually
        invoked (not just mentioned in the source).

        Codex r2 BLOCKING #5: a source-level check passes for comments
        and dead branches — this exercise of the real ``load_model``
        through a monkeypatched register hook verifies the hook
        actually runs.

        Snapshot+restore the server's mutable globals because
        ``load_model`` writes to them (``_model_name``, ``_engine``,
        ``_tool_call_parser``, etc.) and these would otherwise leak
        into subsequent tests in the file (e.g.
        ``test_routes.test_retrieve_unknown_id_keeps_baseline_shape``
        observes ``tool_call_parser=hermes`` instead of ``None``)."""
        from unittest import mock

        from vllm_mlx import server
        from vllm_mlx.config import get_config

        # Snapshot every server global ``load_model`` and
        # ``_sync_config`` may write so we can restore at the end.
        # The list mirrors the assignments in ``_sync_config`` — keep
        # them in sync if a future PR adds new globals.
        snapshot_attrs = (
            "_engine",
            "_model_name",
            "_model_alias",
            "_model_path",
            "_default_max_tokens",
            "_default_max_tokens_is_explicit",
            "_tool_parser_instance",
            "_tool_call_parser",
            "_enable_auto_tool_choice",
            "_enable_tool_logits_bias",
            "_reasoning_parser",
            "_reasoning_parser_name",
            "_alias_recommended_sampling",
            "_generation_config_sampling",
            "_cloud_router",
            "_model_registry",
        )
        snapshot = {a: getattr(server, a, None) for a in snapshot_attrs}
        # ``ServerConfig`` is also written via ``_sync_config``;
        # snapshot the fields downstream tests observe.
        cfg = get_config()
        cfg_attrs = (
            "engine",
            "model_name",
            "model_alias",
            "model_path",
            "tool_call_parser",
            "tool_parser_instance",
            "enable_auto_tool_choice",
            "enable_tool_logits_bias",
            "reasoning_parser",
            "reasoning_parser_name",
            "alias_recommended_sampling",
            "generation_config_sampling",
            "model_registry",
        )
        cfg_snapshot = {a: getattr(cfg, a, None) for a in cfg_attrs}

        # Monkeypatch the register hook so we observe its invocation
        # without mutating the real app's route table.
        invocations: list[bool] = []

        def _recording_register_hook():
            invocations.append(True)
            return False

        monkeypatch.setattr(
            server, "register_audio_routes_if_enabled", _recording_register_hook
        )

        # Stub the engine constructor so we don't actually load a model.
        class _FakeEngine:
            is_mllm = False
            preserve_native_tool_format = False

            def __init__(self, *_a, **_kw):
                pass

        monkeypatch.setattr(server, "BatchedEngine", _FakeEngine)

        try:
            with (
                mock.patch(
                    "vllm_mlx.utils.generation_config.load_generation_config_sampling",
                    return_value={},
                ),
                mock.patch(
                    "vllm_mlx._mxfp4_moe_guardrail.check_from_profile",
                    lambda **_kw: None,
                ),
            ):
                server.load_model(
                    "Qwen/Qwen3-7B-4bit",
                    scheduler_config=None,
                    max_tokens=4096,
                )

            assert invocations == [True], (
                f"load_model did not invoke register_audio_routes_if_enabled "
                f"exactly once; got: {invocations}"
            )
        finally:
            # Restore every global we snapshotted so subsequent tests
            # in the file (and other modules) see the same baseline
            # the fixture started with. Without this, the captured
            # tool/reasoning parser writes from
            # ``_detect_native_tool_support`` leak into
            # ``tests/test_routes.py`` and similar.
            for attr, value in snapshot.items():
                setattr(server, attr, value)
            for attr, value in cfg_snapshot.items():
                setattr(cfg, attr, value)
