# SPDX-License-Identifier: Apache-2.0
"""R-03 + R-04: request-time alias resolution on embeddings + audio.

Bugs caught in 0.8.5 dogfood (Priya F-1/F-2, Karim F-K-MODEL-DEFAULT-NO-RESOLVE
/ F-K-OPENAI-ALIAS-GAP):

* **R-03** — ``model="default"`` (the OpenAI-spec placeholder LangChain /
  LlamaIndex / openai-python emit when the caller hasn't picked a
  specific id) was rejected on ``/v1/embeddings`` (400) and on
  ``/v1/audio/transcriptions`` / ``/v1/audio/speech`` (404), breaking
  drop-in OpenAI-SDK compatibility on multiple routes.
* **R-04** — the short embedding alias (e.g. ``embeddinggemma-300m-6bit``)
  was rejected at request-time on ``/v1/embeddings`` even though that
  exact short alias was the CLI flag value. PR #805 / D-EMBED-ALIAS only
  fixed boot-time resolution; the route handler did a literal string
  match against ``cfg.embedding_model_locked`` (set to the resolved HF
  path).

Fix shape: a single source-of-truth helper
(:func:`_resolve_request_alias_or_default`) runs both sides through
``resolve_model()`` so the alias and the HF path compare equal, and
maps ``"default"`` to the configured server-side model id. Every
request-handling route reuses the helper — no per-route regex
band-aids.
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

# All tests run cross-platform. The route-level integration tests below
# import ``vllm_mlx.routes.embeddings`` + ``vllm_mlx.routes.audio``
# directly (which do NOT pull in MLX at module load), and inject a fake
# ``vllm_mlx.server`` module into ``sys.modules`` so the route's lazy
# ``from ..server import load_embedding_model`` lookup never drags in
# the MLX-importing real server module.
#
# Codex r0 BLOCKING on PR #816: pinning the route fix only on
# Apple-Silicon CI silently leaked the regression past Linux CI on
# every prior PR — fix is to make the route tests CI-safe by mocking
# the server module, not by skipping.


@contextmanager
def _fake_server_module(
    embedding_engine=None,
    embedding_model_locked=None,
    load_fn=None,
):
    """Inject a stub ``vllm_mlx.server`` into ``sys.modules``.

    The route handler does lazy imports
    (``from ..server import load_embedding_model``,
    ``from ..server import _embedding_engine``,
    ``from ..server import _embedding_model_locked``) so the only
    surface the test must cover is the ``vllm_mlx.server`` name in
    ``sys.modules``. A real import of that module on Linux CI fails
    with ``ModuleNotFoundError: mlx.core`` — the stub keeps the test
    cross-platform.
    """
    name = "vllm_mlx.server"
    prev = sys.modules.get(name)
    fake = types.ModuleType(name)
    fake._embedding_engine = embedding_engine
    fake._embedding_model_locked = embedding_model_locked
    fake.load_embedding_model = load_fn or (lambda *a, **kw: None)
    sys.modules[name] = fake
    try:
        yield fake
    finally:
        if prev is not None:
            sys.modules[name] = prev
        else:
            sys.modules.pop(name, None)


# ──────────────────────────────────────────────────────────────────
# Unit tests — the alias-resolution helper itself
# ──────────────────────────────────────────────────────────────────


class TestResolveRequestAliasOrDefault:
    """Pin the single-source-of-truth helper behaviour.

    The helper lives in :mod:`vllm_mlx.service.helpers` and is the only
    place that owns the ``"default"`` sentinel + alias-aware equality
    rule. Every route handler is expected to call it; verifying the
    helper directly catches contract breaks without spinning up the
    full route stack.
    """

    def test_returns_locked_when_request_is_none(self):
        from vllm_mlx.service.helpers import _resolve_request_alias_or_default

        assert (
            _resolve_request_alias_or_default(None, "mlx-community/foo")
            == "mlx-community/foo"
        )

    def test_returns_locked_when_request_is_empty_string(self):
        from vllm_mlx.service.helpers import _resolve_request_alias_or_default

        assert (
            _resolve_request_alias_or_default("", "mlx-community/foo")
            == "mlx-community/foo"
        )

    def test_returns_locked_when_request_is_default_sentinel(self):
        """``"default"`` is the OpenAI canonical placeholder. The
        operator note explicitly forbids adversarial validation for
        this sentinel — it MUST map to the configured model id."""
        from vllm_mlx.service.helpers import _resolve_request_alias_or_default

        assert (
            _resolve_request_alias_or_default("default", "mlx-community/foo")
            == "mlx-community/foo"
        )

    def test_short_alias_matches_full_hf_path_via_registry(self):
        """R-04 root cause: the helper must normalize BOTH sides through
        ``resolve_model`` so the short alias the user CLI-passed
        compares equal to the resolved HF id stored in ``cfg``."""
        from vllm_mlx.service.helpers import _resolve_request_alias_or_default

        # Pick an alias that ships in aliases.json — embeddinggemma-300m-6bit
        # was added by PR #805 for the D-EMBED-ALIAS fix.
        result = _resolve_request_alias_or_default(
            "embeddinggemma-300m-6bit",
            "mlx-community/embeddinggemma-300m-6bit",
        )
        assert result == "mlx-community/embeddinggemma-300m-6bit"

    def test_full_hf_path_matches_self(self):
        from vllm_mlx.service.helpers import _resolve_request_alias_or_default

        result = _resolve_request_alias_or_default(
            "mlx-community/embeddinggemma-300m-6bit",
            "mlx-community/embeddinggemma-300m-6bit",
        )
        assert result == "mlx-community/embeddinggemma-300m-6bit"

    def test_full_hf_path_matches_short_alias_locked(self):
        """Symmetry: the helper must also accept the full HF path when
        the locked side happens to be the short alias (defensive — the
        CLI mutates ``args.embedding_model`` to the HF path before
        locking, but a future caller might pre-lock the alias form)."""
        from vllm_mlx.service.helpers import _resolve_request_alias_or_default

        result = _resolve_request_alias_or_default(
            "mlx-community/embeddinggemma-300m-6bit",
            "embeddinggemma-300m-6bit",
        )
        # Returns ``locked`` verbatim per the helper contract — caller
        # echoes back the configured id, not the wire-supplied form.
        assert result == "embeddinggemma-300m-6bit"

    def test_unknown_alias_returns_none(self):
        """Caller decides the rejection envelope — helper just signals
        the miss with None so embeddings can 400/503 and audio can 404."""
        from vllm_mlx.service.helpers import _resolve_request_alias_or_default

        assert (
            _resolve_request_alias_or_default(
                "non-existent-alias-xyz", "mlx-community/foo"
            )
            is None
        )

    def test_returns_none_when_locked_is_none(self):
        """Nothing configured → no match. Caller surfaces the
        embeddings-not-configured 503 via the H-09 guard (R11-G)."""
        from vllm_mlx.service.helpers import _resolve_request_alias_or_default

        assert _resolve_request_alias_or_default("default", None) is None
        assert _resolve_request_alias_or_default("foo", None) is None

    def test_aliases_match_handles_resolve_model_exception_safely(self, monkeypatch):
        """Belt-and-suspenders: an exception inside ``resolve_model``
        (corrupt ``aliases.json``, partial install) must NOT 500 the
        route. The helper should fall back to a literal-equality miss."""
        from vllm_mlx.service.helpers import _aliases_match

        def boom(_name: str) -> str:  # noqa: ARG001
            raise RuntimeError("registry corrupt")

        monkeypatch.setattr("vllm_mlx.model_aliases.resolve_model", boom)
        assert _aliases_match("a", "b") is False
        # Literal equality still works without touching the registry.
        assert _aliases_match("same", "same") is True


# ──────────────────────────────────────────────────────────────────
# Integration — /v1/embeddings (R-03 default + R-04 short alias)
# ──────────────────────────────────────────────────────────────────


class TestEmbeddingsRouteAliasResolution:
    """Pin the route-level wiring: the helper must be called and the
    mock engine must be reached for every accepted ``model`` form.

    Cross-platform: builds a minimal FastAPI app from
    ``vllm_mlx.routes.embeddings.router`` (which does NOT pull in MLX
    at import time) and mocks ``vllm_mlx.server.load_embedding_model``
    so the load path never instantiates a real engine. Codex r0
    BLOCKING on PR #816 — the previous Apple-Silicon-only gate let
    the regression slip past Linux CI.
    """

    EMBED_ALIAS = "embeddinggemma-300m-6bit"
    EMBED_HF = "mlx-community/embeddinggemma-300m-6bit"

    @pytest.fixture()
    def client_with_locked_embed(self):
        """TestClient with the embedding engine mocked and the locked
        id set to the resolved HF path (as the CLI dispatch produces)."""
        from unittest.mock import patch

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from vllm_mlx.config import get_config
        from vllm_mlx.routes.embeddings import router

        mock_engine = MagicMock()
        mock_engine.model_name = self.EMBED_HF
        mock_engine.embed.return_value = [[0.1, 0.2, 0.3]]
        mock_engine.count_tokens.return_value = 1

        cfg = get_config()
        prev_engine = cfg.embedding_engine
        prev_locked = cfg.embedding_model_locked
        prev_api_key = cfg.api_key
        cfg.embedding_engine = mock_engine
        cfg.embedding_model_locked = self.EMBED_HF
        cfg.api_key = None

        # Stub ``check_rate_limit`` with a clean no-arg async callable
        # — ``patch(..., return_value=None)`` replaces with a MagicMock
        # whose ``(*args, **kwargs)`` signature trips FastAPI's
        # introspection (we hit this on test_routes.py too — see the
        # same fix to test_embeddings_locked_model_reject).
        async def _noop_rate_limit():
            return None

        app = FastAPI()
        app.include_router(router)

        try:
            with (
                _fake_server_module(
                    embedding_engine=mock_engine,
                    embedding_model_locked=self.EMBED_HF,
                ),
                patch(
                    "vllm_mlx.middleware.auth.check_rate_limit",
                    new=_noop_rate_limit,
                ),
            ):
                yield TestClient(app), mock_engine
        finally:
            cfg.embedding_engine = prev_engine
            cfg.embedding_model_locked = prev_locked
            cfg.api_key = prev_api_key

    def _assert_embedding_200(self, resp, expected_echo: str) -> None:
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["model"] == expected_echo
        assert body["data"][0]["embedding"] == [0.1, 0.2, 0.3]

    def test_default_sentinel_accepts(self, client_with_locked_embed):
        """R-03: ``model="default"`` must hit the embedding engine, not
        the H-09 ``no_embedding_model`` guard (503 after R11-G)."""
        client, engine = client_with_locked_embed
        resp = client.post(
            "/v1/embeddings", json={"model": "default", "input": "hello"}
        )
        self._assert_embedding_200(resp, self.EMBED_HF)
        engine.embed.assert_called_once()

    def test_short_alias_accepts(self, client_with_locked_embed):
        """R-04: the short alias (``embeddinggemma-300m-6bit``) must
        compare equal to the resolved HF path stored in ``cfg``."""
        client, engine = client_with_locked_embed
        resp = client.post(
            "/v1/embeddings",
            json={"model": self.EMBED_ALIAS, "input": "hello"},
        )
        self._assert_embedding_200(resp, self.EMBED_HF)
        engine.embed.assert_called_once()

    def test_full_hf_path_accepts(self, client_with_locked_embed):
        """Existing path stays working — no regression from the alias
        normalization."""
        client, engine = client_with_locked_embed
        resp = client.post(
            "/v1/embeddings",
            json={"model": self.EMBED_HF, "input": "hello"},
        )
        self._assert_embedding_200(resp, self.EMBED_HF)
        engine.embed.assert_called_once()

    def test_unknown_alias_400_with_param_field(self, client_with_locked_embed):
        """Bogus aliases still 400 — and the envelope MUST carry
        ``error.param == "model"`` so OpenAI-SDK error branches fire
        cleanly. Pre-fix the route returned ``param: None``."""
        client, _ = client_with_locked_embed
        resp = client.post(
            "/v1/embeddings",
            json={"model": "non-existent-alias", "input": "hello"},
        )
        assert resp.status_code == 400
        body = resp.json()
        # FastAPI surfaces a dict ``detail`` verbatim when no exception
        # handler is wired (bare ``include_router`` app). The production
        # server installs ``install_exception_handlers`` which unwraps
        # to ``body["error"]`` directly.
        err = body.get("error") or body.get("detail", {}).get("error")
        assert err is not None, body
        assert err["param"] == "model"
        assert err["code"] == "model_not_found"
        # The error message should still surface the locked id so the
        # operator sees what the server was actually booted with.
        assert self.EMBED_HF in err["message"]

    def test_no_embedding_model_configured_503(self):
        """H-09 invariant preserved: no ``--embedding-model`` →
        every request 503s with the install-hint envelope, regardless
        of whether the client sent ``"default"`` (the bridge MUST NOT
        route ``"default"`` to a chat-only server's hidden states).

        R11-G: status flipped 400 → 503 — a missing ``--embedding-model``
        is a server-configuration gap, not a bad-client-payload, and
        503 lines up with LangChain / LlamaIndex retry semantics.

        Cross-platform path: inject a stub ``vllm_mlx.server`` module
        with ``_embedding_model_locked = None`` so the H-09 bridge
        sees the unconfigured state without dragging in the
        MLX-importing real server module on Linux CI.
        """
        from unittest.mock import patch

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from vllm_mlx.config import get_config
        from vllm_mlx.routes.embeddings import router

        cfg = get_config()
        prev_engine = cfg.embedding_engine
        prev_locked = cfg.embedding_model_locked
        prev_api_key = cfg.api_key
        cfg.embedding_engine = None
        cfg.embedding_model_locked = None
        cfg.api_key = None

        async def _noop_rate_limit():
            return None

        app = FastAPI()
        app.include_router(router)

        try:
            with (
                _fake_server_module(embedding_engine=None, embedding_model_locked=None),
                patch(
                    "vllm_mlx.middleware.auth.check_rate_limit",
                    new=_noop_rate_limit,
                ),
            ):
                client = TestClient(app)
                resp = client.post(
                    "/v1/embeddings", json={"model": "default", "input": "hi"}
                )
        finally:
            cfg.embedding_engine = prev_engine
            cfg.embedding_model_locked = prev_locked
            cfg.api_key = prev_api_key

        assert resp.status_code == 503
        body = resp.json()
        # H-09 envelope — install hint must still appear, and the
        # machine-readable code lets SDKs branch without substring-
        # matching the message.
        err = body.get("error") or (
            body.get("detail", {}) if isinstance(body.get("detail"), dict) else {}
        ).get("error", {})
        msg = err.get("message") if isinstance(err, dict) else None
        if not msg and isinstance(body.get("detail"), str):
            msg = body["detail"]
        assert "embedding" in (msg or "").lower()
        # The R11-G code field must survive both envelope-handler
        # mounted and bare-router-include code paths.
        if isinstance(err, dict):
            assert err.get("code") == "no_embedding_model"


# ──────────────────────────────────────────────────────────────────
# Integration — /v1/audio/* (R-03 default on STT + TTS)
# ──────────────────────────────────────────────────────────────────


class TestAudioRouteAliasResolution:
    """The audio routes have their own alias registry
    (``STT_MODEL_ALIASES`` / TTS ``model_map``) because the engines
    don't share the chat ``aliases.json``. The single source of truth
    here is the ``"default"`` sentinel mapping — R-03 closure on both
    audio surfaces.
    """

    def test_stt_resolver_maps_default_to_whisper(self):
        """R-03 on ``/v1/audio/transcriptions``: ``"default"`` must
        resolve to the same HF path as ``"whisper-large-v3"`` so
        drop-in OpenAI-SDK code works without a manual model override.
        """
        from vllm_mlx.routes.audio import (
            DEFAULT_STT_ALIAS,
            STT_MODEL_ALIASES,
            _resolve_stt_model,
        )

        # The sentinel and the explicit alias resolve identically.
        assert _resolve_stt_model("default") == STT_MODEL_ALIASES[DEFAULT_STT_ALIAS]
        assert _resolve_stt_model("default") == _resolve_stt_model(DEFAULT_STT_ALIAS)

    def test_stt_resolver_still_rejects_bogus_alias(self):
        """``"default"`` is whitelisted but every OTHER unknown bare
        name still 404s with ``model_not_found_error`` — preserves the
        F-167 / F-210 contract pinned by ``test_audio_path_shaped_model``.
        """
        from fastapi import HTTPException

        from vllm_mlx.routes.audio import _resolve_stt_model

        with pytest.raises(HTTPException) as exc:
            _resolve_stt_model("non-existent-stt-alias")
        assert exc.value.status_code == 404
        detail = exc.value.detail
        assert isinstance(detail, dict)
        assert detail["error"]["code"] == "model_not_found"
        assert detail["error"]["param"] == "model"

    def test_stt_resolver_passes_through_known_alias(self):
        """No regression on the known-alias path."""
        from vllm_mlx.routes.audio import STT_MODEL_ALIASES, _resolve_stt_model

        for alias, hf in STT_MODEL_ALIASES.items():
            assert _resolve_stt_model(alias) == hf

    def test_stt_resolver_passes_through_hf_id(self):
        """No regression on the HF-org/name pass-through path."""
        from vllm_mlx.routes.audio import _resolve_stt_model

        assert (
            _resolve_stt_model("mlx-community/whisper-medium-mlx")
            == "mlx-community/whisper-medium-mlx"
        )

    def test_stt_resolver_empty_string_400(self):
        """Empty string still 400 — ``"default"`` is the only sentinel
        recognized; bare ``""`` is a client bug."""
        from fastapi import HTTPException

        from vllm_mlx.routes.audio import _resolve_stt_model

        with pytest.raises(HTTPException) as exc:
            _resolve_stt_model("")
        assert exc.value.status_code == 400


# ──────────────────────────────────────────────────────────────────
# Cross-check — /v1/chat/completions (already works pre-fix)
# ──────────────────────────────────────────────────────────────────


class TestChatRouteDefaultNotRegressed:
    """Cross-check: ``model="default"`` on the chat route already
    works because ``_validate_model_name`` falls through when the
    request model matches ``cfg.model_name`` / ``cfg.model_alias`` /
    ``cfg.model_path``, AND the response-time ``_resolve_model_name``
    maps ``"default"`` to ``cfg.model_name``. Pin the behaviour so a
    future refactor of the shared helper doesn't break chat.
    """

    def test_chat_resolve_model_name_maps_default_to_cfg(self):
        from vllm_mlx.config import get_config
        from vllm_mlx.service.helpers import _resolve_model_name

        cfg = get_config()
        prev = cfg.model_name
        cfg.model_name = "mlx-community/Qwen3-0.6B-8bit"
        try:
            assert _resolve_model_name("default") == "mlx-community/Qwen3-0.6B-8bit"
            assert _resolve_model_name(None) == "mlx-community/Qwen3-0.6B-8bit"
            assert _resolve_model_name("foo") == "foo"
        finally:
            cfg.model_name = prev
