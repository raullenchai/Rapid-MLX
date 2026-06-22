# SPDX-License-Identifier: Apache-2.0
"""Tests for extracted route modules (health, models, embeddings, MCP, audio).

Uses FastAPI TestClient with mocked server globals to test each route
in isolation without needing a real model or server running.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import get_config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_engine():
    """Mock engine with standard attributes."""
    engine = MagicMock()
    engine.is_mllm = False
    engine.get_stats.return_value = {
        "engine_type": "batched",
        "running": False,
        "uptime_seconds": 123.4,
        "steps_executed": 500,
        "num_running": 0,
        "num_waiting": 0,
        "num_requests_processed": 42,
        "total_prompt_tokens": 1000,
        "total_completion_tokens": 2000,
        "metal_active_memory_gb": 8.5,
        "metal_peak_memory_gb": 12.0,
        "metal_cache_memory_gb": 3.0,
    }
    return engine


@pytest.fixture
def mock_registry():
    """Mock model registry."""
    entry = MagicMock()
    entry.model_name = "test-model"
    entry.aliases = {"test-alias", "test-model"}
    registry = MagicMock()
    registry.list_entries.return_value = [entry]
    registry.__contains__ = lambda self, x: x in ("test-model", "test-alias")
    return registry


# ---------------------------------------------------------------------------
# Health routes
# ---------------------------------------------------------------------------


class TestHealthRoutes:
    def _make_app(self):
        from vllm_mlx.routes.health import admin_router, probe_router, router

        app = FastAPI()
        app.include_router(probe_router)
        app.include_router(router)
        # Destructive control-plane routes (F-150 / F-151) live on a separate
        # router with the ``X-Rapid-MLX-Internal: true`` gate. Include it here
        # so the existing test_cache_clear_* / test_health_router_accepts_*
        # cases still resolve the route — they pass the internal header below.
        app.include_router(admin_router)
        return app

    # Convenience: every destructive route now needs ``X-Rapid-MLX-Internal:
    # true`` to even reach the handler (F-150). Tests that care about the
    # handler's behaviour — not the auth gate — pass this dict via ``headers=``.
    _INTERNAL_HEADERS = {"X-Rapid-MLX-Internal": "true"}

    def _patch_config(self, **kwargs):
        """Patch config fields for testing."""
        from vllm_mlx.config import get_config

        cfg = get_config()
        originals = {}
        for k, v in kwargs.items():
            originals[k] = getattr(cfg, k)
            setattr(cfg, k, v)
        return originals

    def _restore_config(self, originals):
        from vllm_mlx.config import get_config

        cfg = get_config()
        for k, v in originals.items():
            setattr(cfg, k, v)

    def test_health_no_engine(self, mock_engine):
        """Health endpoint works when no engine is loaded."""
        orig = self._patch_config(engine=None, mcp_manager=None, model_name=None)
        try:
            app = self._make_app()
            client = TestClient(app)
            r = client.get("/health")
            assert r.status_code == 200
            data = r.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is False
        finally:
            self._restore_config(orig)

    def test_health_with_engine(self, mock_engine):
        """Health endpoint returns engine info."""
        orig = self._patch_config(
            engine=mock_engine, mcp_manager=None, model_name="test-model"
        )
        try:
            app = self._make_app()
            client = TestClient(app)
            r = client.get("/health")
            assert r.status_code == 200
            data = r.json()
            assert data["model_loaded"] is True
            assert data["model_name"] == "test-model"
            assert data["engine_type"] == "batched"
        finally:
            self._restore_config(orig)

    def test_health_includes_ready_field(self, mock_engine):
        """Health response carries cfg.ready so callers can inspect startup
        state without polling /health/ready separately."""
        orig = self._patch_config(
            engine=mock_engine, mcp_manager=None, model_name="test-model", ready=False
        )
        try:
            app = self._make_app()
            client = TestClient(app)
            r = client.get("/health")
            assert r.json()["ready"] is False
        finally:
            self._restore_config(orig)

    def test_health_ready_returns_503_until_ready(self, mock_engine):
        """/health/ready is 503 while lifespan startup is in progress."""
        orig = self._patch_config(
            engine=mock_engine, mcp_manager=None, model_name="test-model", ready=False
        )
        try:
            app = self._make_app()
            client = TestClient(app)
            r = client.get("/health/ready")
            assert r.status_code == 503
        finally:
            self._restore_config(orig)

    def test_health_ready_returns_200_when_ready(self, mock_engine):
        """/health/ready flips to 200 once cfg.ready is set."""
        orig = self._patch_config(
            engine=mock_engine, mcp_manager=None, model_name="test-model", ready=True
        )
        try:
            app = self._make_app()
            client = TestClient(app)
            r = client.get("/health/ready")
            assert r.status_code == 200
            data = r.json()
            assert data["ready"] is True
            assert data["model"] == "test-model"
        finally:
            self._restore_config(orig)

    def test_health_with_mcp(self, mock_engine):
        """Health endpoint includes MCP info."""
        mcp = MagicMock()
        server_status = MagicMock()
        server_status.state.value = "connected"
        mcp.get_server_status.return_value = [server_status]
        mcp.get_all_tools.return_value = [MagicMock(), MagicMock()]

        orig = self._patch_config(
            engine=mock_engine, mcp_manager=mcp, model_name="test-model"
        )
        try:
            app = self._make_app()
            client = TestClient(app)
            r = client.get("/health")
            data = r.json()
            assert data["mcp"]["enabled"] is True
            assert data["mcp"]["servers_connected"] == 1
            assert data["mcp"]["tools_available"] == 2
        finally:
            self._restore_config(orig)

    def test_get_root_returns_200(self):
        """GET / must return 200 so FastAPI auto-generates HEAD / → 200."""
        orig = self._patch_config(engine=None, mcp_manager=None, model_name=None)
        try:
            app = self._make_app()
            client = TestClient(app)
            r = client.get("/")
            assert r.status_code == 200
            assert r.json()["status"] == "ok"
        finally:
            self._restore_config(orig)

    def test_head_root_returns_200(self):
        """HEAD / is explicitly registered on probe_router (no auth) alongside GET /.
        This test pins the contract so a future refactor that moves the route or
        changes its methods doesn't silently break the Claude Code connectivity probe."""
        orig = self._patch_config(engine=None, mcp_manager=None, model_name=None)
        try:
            app = self._make_app()
            client = TestClient(app)
            r = client.head("/")
            assert r.status_code == 200
        finally:
            self._restore_config(orig)

    def test_head_root_bypasses_api_key(self):
        """HEAD / must return 200 even when --api-key is configured."""
        orig = self._patch_config(
            api_key="test-secret",
            engine=None,
            mcp_manager=None,
            model_name=None,
        )
        try:
            app = self._make_app()
            client = TestClient(app)
            r = client.head("/")
            assert r.status_code == 200
        finally:
            self._restore_config(orig)

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("post", "/v1/cache/clear"),
            ("get", "/v1/status"),
            ("get", "/v1/cache/stats"),
            ("delete", "/v1/cache"),
        ],
    )
    def test_management_router_requires_api_key_when_configured(self, method, path):
        """Management routes (cache, status) honor API auth.

        Probe endpoints (/health, /healthz, /livez, /readyz) are NOT in
        this list — they live on a separate no-auth router so k8s/LB
        liveness checks work when --api-key is set. See
        test_probes_bypass_api_key.

        Destructive routes (``/v1/cache/clear``, ``/v1/cache``) additionally
        require ``X-Rapid-MLX-Internal: true`` per F-150 — we pass it here
        so the assertion checks the API-key 401, not the F-150 403. The
        header-only-403 path is exercised in ``test_internal_route_auth.py``.
        """
        orig = self._patch_config(api_key="test-secret", ready=True)
        try:
            app = self._make_app()
            client = TestClient(app)

            r = getattr(client, method)(path, headers=self._INTERNAL_HEADERS)

            assert r.status_code == 401
            assert r.json()["detail"] == "API key required"
        finally:
            self._restore_config(orig)

    @pytest.mark.parametrize(
        "path",
        ["/health", "/healthz", "/health/ready", "/readyz", "/livez"],
    )
    def test_probes_bypass_api_key(self, path, mock_engine):
        """Probe endpoints must be reachable without auth, even when
        --api-key is configured. k8s/AWS ALB/GCP probes cannot send
        Authorization headers by default; without this split, every probe
        marks the pod unhealthy."""
        orig = self._patch_config(
            api_key="test-secret",
            engine=mock_engine,
            mcp_manager=None,
            model_name="test-model",
            ready=True,
        )
        try:
            app = self._make_app()
            client = TestClient(app)

            r = client.get(path)

            assert r.status_code == 200, (
                f"{path} should return 200 with no auth header, got "
                f"{r.status_code}: {r.text}"
            )
        finally:
            self._restore_config(orig)

    @pytest.mark.parametrize(
        ("path", "expected_keys"),
        [
            ("/healthz", {"status", "ready", "model_loaded"}),
            ("/readyz", {"ready", "model"}),
            ("/livez", {"status"}),
        ],
    )
    def test_k8s_probe_aliases_match_canonical_shape(
        self, path, expected_keys, mock_engine
    ):
        """k8s-convention aliases return the same JSON shape as their
        canonical counterparts (/health, /health/ready) so dashboards
        and probe specs can use either path interchangeably."""
        orig = self._patch_config(
            engine=mock_engine,
            mcp_manager=None,
            model_name="test-model",
            ready=True,
        )
        try:
            app = self._make_app()
            client = TestClient(app)
            r = client.get(path)
            assert r.status_code == 200
            assert expected_keys.issubset(r.json().keys())
        finally:
            self._restore_config(orig)

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("get", "/health"),
            ("get", "/health/ready"),
            ("post", "/v1/cache/clear"),
            ("get", "/v1/status"),
            ("get", "/v1/cache/stats"),
            ("delete", "/v1/cache"),
        ],
    )
    def test_health_router_accepts_valid_api_key(self, method, path, mock_engine):
        """Valid Bearer token preserves access to protected management routes.

        Destructive routes (``/v1/cache/clear``, ``/v1/cache`` DELETE) also
        require ``X-Rapid-MLX-Internal: true`` per F-150 — pass it so the
        success path resolves.
        """
        orig = self._patch_config(
            api_key="test-secret",
            engine=mock_engine,
            mcp_manager=None,
            model_name="test-model",
            ready=True,
        )
        try:
            app = self._make_app()
            client = TestClient(app)

            r = getattr(client, method)(
                path,
                headers={
                    "Authorization": "Bearer test-secret",
                    **self._INTERNAL_HEADERS,
                },
            )

            assert r.status_code != 401
        finally:
            self._restore_config(orig)

    def test_status_no_engine(self):
        """Status returns not_loaded when no engine."""
        orig = self._patch_config(engine=None, model_name=None)
        try:
            app = self._make_app()
            client = TestClient(app)
            r = client.get("/v1/status")
            assert r.status_code == 200
            assert r.json()["status"] == "not_loaded"
        finally:
            self._restore_config(orig)

    def test_status_with_engine(self, mock_engine):
        """Status returns engine stats."""
        orig = self._patch_config(engine=mock_engine, model_name="test-model")
        try:
            app = self._make_app()
            client = TestClient(app)
            r = client.get("/v1/status")
            data = r.json()
            assert data["status"] == "idle"
            assert data["model"] == "test-model"
            assert data["steps_executed"] == 500
            assert data["metal"]["active_memory_gb"] == 8.5
            # generation_tps/prompt_tps default to 0 when batch_generator
            # stats are absent (text-only batched engine path).
            assert data["generation_tps"] == 0
            assert data["prompt_tps"] == 0
        finally:
            self._restore_config(orig)

    def test_status_exposes_batch_generator_throughput(self, mock_engine):
        """Status surfaces generation_tps/prompt_tps from batch_generator stats.

        Regression for the upstream bug where these counters existed in the
        batch generator but never reached /v1/status because the engine
        layer didn't forward the 'batch_generator' key.
        """
        mock_engine.get_stats.return_value = {
            **mock_engine.get_stats.return_value,
            "batch_generator": {
                "prompt_tps": 142.7,
                "generation_tps": 38.4,
            },
        }
        orig = self._patch_config(engine=mock_engine, model_name="test-model")
        try:
            app = self._make_app()
            client = TestClient(app)
            data = client.get("/v1/status").json()
            assert data["generation_tps"] == 38.4
            assert data["prompt_tps"] == 142.7
        finally:
            self._restore_config(orig)

    def test_status_handles_non_dict_batch_generator(self, mock_engine):
        """Defensive: malformed batch_generator (not a dict) must not 500.

        Codex flagged that `stats.get(...) or {}` only guards the falsy case;
        a string/list/int would crash on `.get(...)`. Confirm we coerce safely.
        """
        mock_engine.get_stats.return_value = {
            **mock_engine.get_stats.return_value,
            "batch_generator": "unexpected-string",
        }
        orig = self._patch_config(engine=mock_engine, model_name="test-model")
        try:
            data = TestClient(self._make_app()).get("/v1/status").json()
            assert data["generation_tps"] == 0
            assert data["prompt_tps"] == 0
        finally:
            self._restore_config(orig)

    def test_status_coerces_none_throughput_to_zero(self, mock_engine):
        """Defensive: explicit-None throughput values must serialize as 0,
        not null. Monitoring dashboards expect a number."""
        mock_engine.get_stats.return_value = {
            **mock_engine.get_stats.return_value,
            "batch_generator": {"prompt_tps": None, "generation_tps": None},
        }
        orig = self._patch_config(engine=mock_engine, model_name="test-model")
        try:
            data = TestClient(self._make_app()).get("/v1/status").json()
            assert data["generation_tps"] == 0
            assert data["prompt_tps"] == 0
        finally:
            self._restore_config(orig)

    def test_status_preserves_zero_float_throughput(self, mock_engine):
        """A genuine 0.0 idle reading must stay a float. `or 0` would
        collapse it to int 0; downstream schemas that require number-as-
        float would reject the response."""
        mock_engine.get_stats.return_value = {
            **mock_engine.get_stats.return_value,
            "batch_generator": {"prompt_tps": 0.0, "generation_tps": 0.0},
        }
        orig = self._patch_config(engine=mock_engine, model_name="test-model")
        try:
            data = TestClient(self._make_app()).get("/v1/status").json()
            # JSON round-trip preserves int vs float: 0.0 → 0.0, 0 → 0.
            # The stricter assertion is that the raw text contains "0.0".
            r = TestClient(self._make_app()).get("/v1/status")
            assert '"generation_tps":0.0' in r.text.replace(" ", "")
            assert '"prompt_tps":0.0' in r.text.replace(" ", "")
            # Sanity: numeric comparison still holds.
            assert data["generation_tps"] == 0
            assert data["prompt_tps"] == 0
        finally:
            self._restore_config(orig)

    def test_cache_clear_no_engine(self):
        """Cache clear returns 503 when no engine."""
        orig = self._patch_config(engine=None)
        try:
            app = self._make_app()
            # ``TestClient(app)`` defaults to client ``("testclient", 50000)``
            # which is NOT loopback. ``verify_internal_admin`` (codex r1 fix)
            # rejects non-loopback callers when ``--api-key`` is unset, so we
            # pin to 127.0.0.1 here — the auth-gate's loopback branch has its
            # own coverage in ``test_internal_route_auth.py``.
            client = TestClient(app, client=("127.0.0.1", 50000))
            r = client.post("/v1/cache/clear", headers=self._INTERNAL_HEADERS)
            assert r.status_code == 503
        finally:
            self._restore_config(orig)

    def test_cache_clear_no_prompt_cache(self, mock_engine):
        """Cache clear works when no prompt cache exists."""
        mock_engine._model = MagicMock(spec=[])
        orig = self._patch_config(engine=mock_engine)
        try:
            app = self._make_app()
            client = TestClient(app, client=("127.0.0.1", 50000))
            r = client.post("/v1/cache/clear", headers=self._INTERNAL_HEADERS)
            assert r.status_code == 200
            assert "No prompt cache" in r.json()["message"]
        finally:
            self._restore_config(orig)

    def test_cache_stats_no_vlm(self):
        """Cache stats returns fallback when mlx_vlm not available."""
        app = self._make_app()
        # ``/v1/cache/stats`` is a READ route (gated by ``verify_api_key``, no
        # internal-header requirement), so default TestClient host is fine
        # here — this test only verifies the fallback envelope shape.
        client = TestClient(app)
        r = client.get("/v1/cache/stats")
        assert r.status_code == 200
        # Either returns stats or fallback message
        data = r.json()
        assert "multimodal_kv_cache" in data or "model_type" in data

    def test_cache_delete(self):
        """Cache delete endpoint works."""
        app = self._make_app()
        # Destructive route — pin loopback so the codex r1 auth check passes
        # when ``--api-key`` is unset (this test's posture).
        client = TestClient(app, client=("127.0.0.1", 50000))
        r = client.delete("/v1/cache", headers=self._INTERNAL_HEADERS)
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Models routes
# ---------------------------------------------------------------------------


class TestModelsRoutes:
    def _make_app(self):
        from vllm_mlx.routes.models import router

        app = FastAPI()
        app.include_router(router)
        return app

    def _set_config(self, **kwargs):
        from vllm_mlx.config import get_config

        cfg = get_config()
        orig = {}
        for k, v in kwargs.items():
            orig[k] = getattr(cfg, k)
            setattr(cfg, k, v)
        return orig

    def _restore(self, orig):
        from vllm_mlx.config import get_config

        cfg = get_config()
        for k, v in orig.items():
            setattr(cfg, k, v)

    def test_list_models_single(self):
        """List models with single model loaded."""
        orig = self._set_config(
            model_registry=None,
            model_name="test-model",
            model_alias="test-alias",
            api_key=None,
        )
        try:
            client = TestClient(self._make_app())
            r = client.get("/v1/models")
            assert r.status_code == 200
            ids = [m["id"] for m in r.json()["data"]]
            assert "test-model" in ids
            assert "test-alias" in ids
        finally:
            self._restore(orig)

    def test_list_models_registry(self, mock_registry):
        """List models with multi-model registry."""
        orig = self._set_config(
            model_registry=mock_registry,
            model_name="test-model",
            model_alias=None,
            api_key=None,
        )
        try:
            client = TestClient(self._make_app())
            r = client.get("/v1/models")
            assert r.status_code == 200
            assert len(r.json()["data"]) >= 1
        finally:
            self._restore(orig)

    def test_retrieve_model_found(self):
        """Retrieve existing model returns 200."""
        orig = self._set_config(
            model_registry=None,
            model_name="test-model",
            model_alias="test-alias",
            api_key=None,
        )
        try:
            client = TestClient(self._make_app())
            r = client.get("/v1/models/test-model")
            assert r.status_code == 200
            assert r.json()["id"] == "test-model"
        finally:
            self._restore(orig)

    def test_retrieve_model_not_found(self):
        """Retrieve non-existent model returns 404."""
        orig = self._set_config(
            model_registry=None, model_name="test-model", model_alias=None, api_key=None
        )
        try:
            client = TestClient(self._make_app())
            r = client.get("/v1/models/nonexistent")
            assert r.status_code == 404
        finally:
            self._restore(orig)

    # ----- Rapid-MLX vendor extension surface (Sweep autoresearch) -----

    def test_retrieve_known_alias_populates_extensions(self):
        """A known alias (e.g. ``qwen3.5-4b-4bit``) must surface its
        AliasProfile-derived vendor extension fields so rapid-desktop
        can bootstrap calibrated sampling on first chat. Profiles
        without ``recommended_sampling`` MUST still surface
        ``is_hybrid`` + parser pair + modality so the desktop can
        decide whether to show the "Show reasoning" toggle."""
        orig = self._set_config(
            model_registry=None,
            model_name="mlx-community/Qwen3.5-4B-MLX-4bit",
            model_alias="qwen3.5-4b-4bit",
            api_key=None,
        )
        try:
            client = TestClient(self._make_app())
            r = client.get("/v1/models/qwen3.5-4b-4bit")
            assert r.status_code == 200
            body = r.json()
            # OpenAI-canonical baseline preserved.
            assert body["id"] == "qwen3.5-4b-4bit"
            assert body["object"] == "model"
            # Vendor-extension fields surfaced from AliasProfile.
            # The qwen3.5-4b-4bit alias is_hybrid=True per aliases.json
            # (verified by Round-1 autoresearch sweep, 2026-06-14).
            assert body["is_hybrid"] is True, (
                "is_hybrid must be surfaced so the desktop knows to render the "
                "'Show reasoning' toggle for hybrid models"
            )
            # Parser pair surfaced for diagnostics.
            assert body["tool_call_parser"] == "hermes"
            assert body["reasoning_parser"] == "qwen3"
            # Modality defaults to "text" for LLMs.
            assert body["modality"] == "text"
        finally:
            self._restore(orig)

    def test_retrieve_known_alias_surfaces_recommended_sampling(self):
        """An alias whose ``AliasProfile`` carries ``recommended_sampling``
        (curated knobs that beat the model's bare ``generation_config.json``
        on the canonical eval) must surface the values as a JSON object on
        the wire. Pins the ``tuple[(key, value), ...] -> dict`` conversion
        the helper does — a future refactor that forgets to convert would
        either trip Pydantic v2's ``dict_type`` validator at construction
        (current behavior: hard 500) or, if the field type were ever
        widened to accept tuples, silently ship nested arrays and break
        rapid-desktop's slider bootstrap. Either failure mode is caught
        by asserting ``isinstance(sampling, dict)`` below.

        We pick ``gemma-4-12b-4bit`` because gemma-4 family ships curated
        sampling (temperature=1.0, top_k=64, top_p=0.95) — verified via
        ``model_aliases.list_profiles()`` at 2026-06-14."""
        orig = self._set_config(
            model_registry=None,
            model_name="mlx-community/gemma-4-12B-it-4bit",
            model_alias="gemma-4-12b-4bit",
            api_key=None,
        )
        try:
            client = TestClient(self._make_app())
            r = client.get("/v1/models/gemma-4-12b-4bit")
            assert r.status_code == 200
            body = r.json()
            sampling = body["recommended_sampling"]
            # MUST be a JSON object (Python dict round-trip), not a
            # list-of-pairs which is what the dataclass stores natively.
            assert isinstance(sampling, dict), (
                f"recommended_sampling must serialize as a JSON object, "
                f"got {type(sampling).__name__}: {sampling!r}"
            )
            # Spot-check the known gemma-4 curated values.
            assert sampling.get("temperature") == 1.0
            assert sampling.get("top_p") == 0.95
            assert sampling.get("top_k") == 64
            # Modality stays "text" for an LLM, is_hybrid=False for gemma.
            assert body["modality"] == "text"
            assert body["is_hybrid"] is False
        finally:
            self._restore(orig)

    def test_retrieve_unknown_id_keeps_baseline_shape(self):
        """An id that doesn't resolve to an alias (raw HF path that
        isn't in the registry, operator-supplied custom model) must
        keep an OpenAI-compatible shape — id/object/created/owned_by
        carry their canonical values and the vendor-extension keys
        appear as JSON ``null`` (since ``ModelInfo`` does not set
        ``exclude_none``). OpenAI-only clients ignore unknown keys
        per spec whether they appear as ``null`` or are absent, so
        this is additive; the assertions below pin the explicit
        present-with-null contract so a future refactor that flips
        to ``exclude_none=True`` is caught."""
        orig = self._set_config(
            model_registry=None,
            model_name="custom-operator-model-not-in-registry",
            model_alias=None,
            api_key=None,
        )
        try:
            client = TestClient(self._make_app())
            r = client.get("/v1/models/custom-operator-model-not-in-registry")
            assert r.status_code == 200
            body = r.json()
            assert body["id"] == "custom-operator-model-not-in-registry"
            # ``ModelInfo`` does NOT set ``exclude_none``, so the
            # extension keys are PRESENT on the wire body and serialize
            # as JSON ``null`` (not omitted). Pin both presence AND
            # null-value so a future refactor that flips to
            # ``exclude_none=True`` and silently drops the keys is
            # caught here — clients tolerate either shape per the
            # OpenAI spec, but the wire contract should be explicit.
            assert "recommended_sampling" in body
            assert body["recommended_sampling"] is None
            assert "is_hybrid" in body
            assert body["is_hybrid"] is None
            assert "tool_call_parser" in body
            assert body["tool_call_parser"] is None
        finally:
            self._restore(orig)

    def test_list_models_surfaces_extensions_per_entry(self):
        """`/v1/models` list endpoint must surface extensions for each
        alias entry, not just the singleton retrieval path. Without
        this the desktop's catalog pre-fetch (one round trip on
        startup) wouldn't get profile data and would fall back to
        per-alias N+1 calls."""
        orig = self._set_config(
            model_registry=None,
            model_name="mlx-community/Qwen3.5-4B-MLX-4bit",
            model_alias="qwen3.5-4b-4bit",
            api_key=None,
        )
        try:
            client = TestClient(self._make_app())
            r = client.get("/v1/models")
            assert r.status_code == 200
            entries = r.json()["data"]
            alias_entry = next(
                (e for e in entries if e["id"] == "qwen3.5-4b-4bit"), None
            )
            assert alias_entry is not None, (
                "qwen3.5-4b-4bit must appear in the list endpoint"
            )
            assert alias_entry["is_hybrid"] is True
            assert alias_entry["reasoning_parser"] == "qwen3"
        finally:
            self._restore(orig)

    # ----- F-067: modality reporting for VL models -----

    def test_vl_alias_reports_image_modality(self):
        """F-067 regression: aliases that resolve to a Vision-Language
        checkpoint MUST advertise ``modality="image"`` on the wire so
        downstream OpenAI-SDK clients know to send PNG/JPEG/etc. image
        content shapes. Before the fix, all VL aliases (qwen3-vl-2b-4bit,
        gemma-3n-*, qwen3-vl-4b-4bit) reported ``"text"`` because the
        ``AliasProfile.modality`` field is an engine-routing
        discriminator (``text`` = AR lane, ``text-diffusion`` = diffusion
        lane) — the multimodal path is layered on top of the AR lane
        via ``MLLMBatchGenerator`` and so keeps ``modality="text"``
        internally. The route layer now derives the reported value from
        ``is_mllm_model`` so VL aliases surface ``image`` to clients
        without disturbing engine routing.
        """
        for vl_alias in (
            "qwen3-vl-2b-4bit",
            "qwen3-vl-4b-4bit",
            "gemma-3n-e2b-4bit",
            "gemma-3n-e4b-4bit",
        ):
            orig = self._set_config(
                model_registry=None,
                model_name=vl_alias,
                model_alias=None,
                api_key=None,
            )
            try:
                client = TestClient(self._make_app())
                r = client.get(f"/v1/models/{vl_alias}")
                assert r.status_code == 200, f"VL alias {vl_alias!r} should resolve"
                body = r.json()
                assert body["modality"] == "image", (
                    f"F-067 regression: VL alias {vl_alias!r} reports "
                    f"modality={body['modality']!r} (expected 'image')"
                )
            finally:
                self._restore(orig)

    def test_vl_alias_reports_image_modality_on_list_endpoint(self):
        """F-067 regression on the LIST endpoint (the surface clients
        actually consume on catalog pre-fetch). The per-id retrieval
        test above pins the same field at the singleton endpoint;
        this counterpart pins it on ``GET /v1/models`` so a future
        refactor that fixes one path without the other is caught.
        """
        orig = self._set_config(
            model_registry=None,
            model_name="qwen3-vl-2b-4bit",
            model_alias=None,
            api_key=None,
        )
        try:
            client = TestClient(self._make_app())
            r = client.get("/v1/models")
            assert r.status_code == 200
            entries = r.json()["data"]
            vl_entry = next((e for e in entries if e["id"] == "qwen3-vl-2b-4bit"), None)
            assert vl_entry is not None, (
                "qwen3-vl-2b-4bit must appear in the list endpoint"
            )
            assert vl_entry["modality"] == "image", (
                f"F-067 list-endpoint regression: VL alias reports "
                f"modality={vl_entry['modality']!r} (expected 'image')"
            )
        finally:
            self._restore(orig)

    def test_text_alias_still_reports_text_modality(self):
        """Counterpart to F-067: a plain text LLM alias MUST keep
        ``modality="text"`` so the detector doesn't over-trigger and
        mislabel non-VL models as multimodal.
        """
        orig = self._set_config(
            model_registry=None,
            model_name="mlx-community/Qwen3.5-4B-MLX-4bit",
            model_alias="qwen3.5-4b-4bit",
            api_key=None,
        )
        try:
            client = TestClient(self._make_app())
            r = client.get("/v1/models/qwen3.5-4b-4bit")
            assert r.status_code == 200
            assert r.json()["modality"] == "text"
        finally:
            self._restore(orig)

    def test_diffusion_alias_keeps_text_diffusion_modality(self):
        """Counterpart to F-067: the text-diffusion routing
        discriminator on diffusion-gemma-* must pass through unchanged
        — the multimodal detector only kicks in when the profile
        modality is the default ``text``, not for already-non-text
        lanes that have their own dispatch.
        """
        orig = self._set_config(
            model_registry=None,
            model_name="diffusion-gemma-26b-4bit",
            model_alias=None,
            api_key=None,
        )
        try:
            client = TestClient(self._make_app())
            r = client.get("/v1/models/diffusion-gemma-26b-4bit")
            assert r.status_code == 200
            assert r.json()["modality"] == "text-diffusion"
        finally:
            self._restore(orig)

    # ----- Issue #363: context_window on /v1/models -----

    def _engine_with_context(self, max_pos: int):
        """Build a minimal engine stub whose ``get_model_max_context``
        chain resolves to ``max_pos``. Mirrors the lookup priority in
        ``service.helpers.get_model_max_context``: ``engine._model.args
        .max_position_embeddings`` is the first probe, so populating it
        is enough to drive the resolver deterministically without
        spinning up MLX weights.
        """
        engine = MagicMock()
        args = MagicMock()
        args.max_position_embeddings = max_pos
        # Block the text_config nested fallback so the test pins the
        # primary path; otherwise MagicMock's auto-attr would expose a
        # truthy mock there and the helper would prefer the nested
        # branch on the second call.
        args.text_config = None
        model = MagicMock()
        model.args = args
        model.config = None
        engine._model = model
        engine.tokenizer = None
        return engine

    def test_context_window_present_for_loaded_alias(self):
        """Issue #363 primary fix: when an engine is loaded for the
        served alias, ``/v1/models`` MUST emit ``context_window`` so
        rapid-desktop's PR #318 ``max_tokens`` slider auto-scaler
        sees the cap the server will actually enforce. Pre-fix the
        field was absent from the response payload entirely and the
        desktop fell through to a per-family hard-coded heuristic
        that drifted out of sync with every long-context release.
        """
        engine = self._engine_with_context(32768)
        orig = self._set_config(
            model_registry=None,
            model_name="mlx-community/Qwen3.5-4B-MLX-4bit",
            model_alias="qwen3.5-4b-4bit",
            engine=engine,
            api_key=None,
        )
        try:
            client = TestClient(self._make_app())
            r = client.get("/v1/models/qwen3.5-4b-4bit")
            assert r.status_code == 200
            body = r.json()
            assert "context_window" in body, (
                "Issue #363: context_window must be present on the wire "
                "even when the resolver returns None — pre-fix the field "
                "was absent entirely and the desktop consumer had no "
                "signal to even attempt fallback"
            )
            assert body["context_window"] == 32768, (
                f"context_window should reflect the loaded engine's "
                f"max_position_embeddings (32768), got {body['context_window']!r}"
            )
        finally:
            self._restore(orig)

    def test_context_window_present_on_list_endpoint(self):
        """Issue #363 LIST counterpart — the field must surface on the
        bulk endpoint too (where the desktop catalog actually
        pre-fetches on startup). A per-id-only fix would force a
        second N+1 round-trip per alias.
        """
        engine = self._engine_with_context(262144)
        orig = self._set_config(
            model_registry=None,
            model_name="mlx-community/Qwen3.6-35B-A3B-Instruct-MLX-4bit",
            model_alias="qwen3.6-35b-4bit",
            engine=engine,
            api_key=None,
        )
        try:
            client = TestClient(self._make_app())
            r = client.get("/v1/models")
            assert r.status_code == 200
            entries = r.json()["data"]
            alias_entry = next(
                (e for e in entries if e["id"] == "qwen3.6-35b-4bit"), None
            )
            assert alias_entry is not None
            assert alias_entry["context_window"] == 262144
            # Spot-check the canonical hf-path entry inherits the same
            # context window — both ids share the loaded engine.
            hf_entry = next(
                (
                    e
                    for e in entries
                    if e["id"] == "mlx-community/Qwen3.6-35B-A3B-Instruct-MLX-4bit"
                ),
                None,
            )
            assert hf_entry is not None
            assert hf_entry["context_window"] == 262144
        finally:
            self._restore(orig)

    def test_context_window_present_for_three_alias_families(self):
        """Issue #363 cross-family pin: the resolver must work for the
        three alias families covered by the desktop's PR #318
        consumer — qwen3 (32K class), qwen3.6 (long-context class),
        and a gemma alias (128K class). Pins the contract that the
        field is populated as a positive int across the spectrum, not
        a one-off that only fires for one family.
        """
        cases = (
            ("qwen3.5-4b-4bit", "mlx-community/Qwen3.5-4B-MLX-4bit", 40960),
            (
                "qwen3.6-35b-4bit",
                "mlx-community/Qwen3.6-35B-A3B-Instruct-MLX-4bit",
                262144,
            ),
            ("gemma-4-12b-4bit", "mlx-community/gemma-4-12B-it-4bit", 131072),
        )
        for alias, hf_path, ctx in cases:
            engine = self._engine_with_context(ctx)
            orig = self._set_config(
                model_registry=None,
                model_name=hf_path,
                model_alias=alias,
                engine=engine,
                api_key=None,
            )
            try:
                client = TestClient(self._make_app())
                r = client.get(f"/v1/models/{alias}")
                assert r.status_code == 200
                body = r.json()
                assert body["context_window"] == ctx, (
                    f"{alias}: expected context_window={ctx}, got {body['context_window']!r}"
                )
                assert isinstance(body["context_window"], int)
                assert body["context_window"] > 0
            finally:
                self._restore(orig)

    def test_context_window_null_when_no_engine_loaded(self):
        """When no engine is loaded for an alias (cold listing, or an
        unregistered operator id), the field MUST appear as JSON
        ``null`` rather than being omitted. Clients can then fall
        back to their per-family heuristic without ambiguity about
        whether the server-side fix landed.
        """
        orig = self._set_config(
            model_registry=None,
            model_name="qwen3.5-4b-4bit",
            model_alias=None,
            engine=None,
            api_key=None,
        )
        try:
            client = TestClient(self._make_app())
            r = client.get("/v1/models/qwen3.5-4b-4bit")
            assert r.status_code == 200
            body = r.json()
            assert "context_window" in body
            assert body["context_window"] is None
        finally:
            self._restore(orig)

    def test_context_window_suppresses_dos_sentinel(self):
        """``service.helpers.get_model_max_context`` returns its DoS
        sentinel (~4 Mi) when no usable cap can be probed. That
        number is intentionally large for the request-time DoS guard
        — but it's NOT a real context window and the client must not
        advertise it as one. The resolver suppresses any value at or
        above the sentinel floor so the desktop falls back to its
        per-family heuristic instead.
        """
        # Inject an engine whose model exposes NO usable attribute —
        # ``get_model_max_context`` then returns ``_FALLBACK_MAX_CONTEXT_TOKENS``.
        engine = MagicMock()
        engine._model = MagicMock(spec=[])
        engine.tokenizer = None
        orig = self._set_config(
            model_registry=None,
            model_name="qwen3.5-4b-4bit",
            model_alias=None,
            engine=engine,
            api_key=None,
        )
        try:
            client = TestClient(self._make_app())
            r = client.get("/v1/models/qwen3.5-4b-4bit")
            assert r.status_code == 200
            body = r.json()
            assert "context_window" in body
            assert body["context_window"] is None, (
                "DoS sentinel from get_model_max_context must not leak "
                "onto the wire as a real context window"
            )
        finally:
            self._restore(orig)

    def test_context_window_probe_failure_does_not_500(self):
        """A probe that raises mid-resolution (e.g. tokenizer attribute
        access raises) must NOT 500 the listing endpoint — the field
        falls through to ``None`` and the request still completes.
        Pins the defensive ``except`` around the helper call.
        """
        engine = MagicMock()
        # ``getattr(engine, "_model", ...)`` will return this — when
        # the helper then probes ``.args`` we make it raise.
        broken_model = MagicMock()
        type(broken_model).args = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("synthetic"))
        )
        engine._model = broken_model
        orig = self._set_config(
            model_registry=None,
            model_name="qwen3.5-4b-4bit",
            model_alias=None,
            engine=engine,
            api_key=None,
        )
        try:
            client = TestClient(self._make_app())
            r = client.get("/v1/models/qwen3.5-4b-4bit")
            assert r.status_code == 200
            body = r.json()
            assert body.get("context_window") is None
        finally:
            self._restore(orig)


# ---------------------------------------------------------------------------
# MCP routes
# ---------------------------------------------------------------------------


class TestMCPRoutes:
    def _make_app(self):
        from vllm_mlx.routes.mcp_routes import router

        app = FastAPI()
        app.include_router(router)
        return app

    def test_list_tools_no_mcp(self):
        """List tools returns empty when MCP not configured."""
        with (
            patch.object(get_config(), "mcp_manager", None),
            patch.object(get_config(), "api_key", None),
        ):
            app = self._make_app()
            client = TestClient(app)
            r = client.get("/v1/mcp/tools")
            assert r.status_code == 200
            assert r.json()["count"] == 0

    def test_list_tools_with_mcp(self):
        """List tools returns tools when MCP configured."""
        tool = MagicMock()
        tool.full_name = "test_tool"
        tool.description = "A test tool"
        tool.server_name = "test_server"
        tool.input_schema = {"type": "object"}

        mcp = MagicMock()
        mcp.get_all_tools.return_value = [tool]

        with (
            patch.object(get_config(), "mcp_manager", mcp),
            patch.object(get_config(), "api_key", None),
        ):
            app = self._make_app()
            client = TestClient(app)
            r = client.get("/v1/mcp/tools")
            data = r.json()
            assert data["count"] == 1
            assert data["tools"][0]["name"] == "test_tool"

    def test_list_servers_no_mcp(self):
        """List servers returns empty when MCP not configured."""
        with (
            patch.object(get_config(), "mcp_manager", None),
            patch.object(get_config(), "api_key", None),
        ):
            app = self._make_app()
            client = TestClient(app)
            r = client.get("/v1/mcp/servers")
            assert r.status_code == 200
            assert r.json()["servers"] == []

    def test_execute_no_mcp(self):
        """Execute tool returns 503 when MCP not configured."""
        with (
            patch.object(get_config(), "mcp_manager", None),
            patch.object(get_config(), "api_key", None),
        ):
            app = self._make_app()
            client = TestClient(app)
            r = client.post(
                "/v1/mcp/execute", json={"tool_name": "test", "arguments": {}}
            )
            assert r.status_code == 503

    def test_execute_with_mcp(self):
        """Execute tool works with MCP configured."""
        result = MagicMock()
        result.tool_name = "test_tool"
        result.content = "result"
        result.is_error = False
        result.error_message = None

        mcp = MagicMock()
        mcp.execute_tool = AsyncMock(return_value=result)

        with (
            patch.object(get_config(), "mcp_manager", mcp),
            patch.object(get_config(), "api_key", None),
        ):
            app = self._make_app()
            client = TestClient(app)
            r = client.post(
                "/v1/mcp/execute",
                json={"tool_name": "test_tool", "arguments": {"key": "val"}},
            )
            assert r.status_code == 200
            assert r.json()["tool_name"] == "test_tool"


# ---------------------------------------------------------------------------
# Embeddings routes
# ---------------------------------------------------------------------------


class TestEmbeddingsRoutes:
    def _make_app(self):
        from vllm_mlx.routes.embeddings import router

        app = FastAPI()
        app.include_router(router)
        return app

    def test_embeddings_success(self):
        """Embeddings endpoint returns vectors."""
        mock_emb_engine = MagicMock()
        mock_emb_engine.count_tokens.return_value = 5
        mock_emb_engine.embed.return_value = [[0.1, 0.2, 0.3]]

        with (
            patch.object(get_config(), "embedding_engine", mock_emb_engine),
            # H-09 route guard requires an embedding model to be
            # configured before accepting requests; match the test's
            # request model so we exercise the success path.
            patch.object(get_config(), "embedding_model_locked", "test-embed"),
            patch("vllm_mlx.server.load_embedding_model"),
            patch.object(get_config(), "api_key", None),
            patch("vllm_mlx.middleware.auth.check_rate_limit", return_value=None),
        ):
            app = self._make_app()
            client = TestClient(app)
            r = client.post(
                "/v1/embeddings",
                json={
                    "model": "test-embed",
                    "input": "hello world",
                },
            )
            assert r.status_code == 200
            data = r.json()
            assert len(data["data"]) == 1
            assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3]
            assert data["usage"]["prompt_tokens"] == 5

    def test_embeddings_batch(self):
        """Embeddings endpoint handles batch input."""
        mock_emb_engine = MagicMock()
        mock_emb_engine.count_tokens.return_value = 10
        mock_emb_engine.embed.return_value = [[0.1], [0.2], [0.3]]

        with (
            patch.object(get_config(), "embedding_engine", mock_emb_engine),
            # H-09: configure the lock so the route accepts the
            # ``model="test-embed"`` POST instead of 400ing under the
            # new guard.
            patch.object(get_config(), "embedding_model_locked", "test-embed"),
            patch("vllm_mlx.server.load_embedding_model"),
            patch.object(get_config(), "api_key", None),
            patch("vllm_mlx.middleware.auth.check_rate_limit", return_value=None),
        ):
            app = self._make_app()
            client = TestClient(app)
            r = client.post(
                "/v1/embeddings",
                json={
                    "model": "test-embed",
                    "input": ["a", "b", "c"],
                },
            )
            assert r.status_code == 200
            assert len(r.json()["data"]) == 3

    def test_embeddings_locked_model_reject(self):
        """Embeddings rejects wrong model when locked.

        R-03/R-04 follow-up: the rejection envelope was upgraded to the
        OpenAI-canonical shape so SDK error branches (which key on
        ``error.param``) fire cleanly. The message still surfaces the
        locked id so the operator sees what the server was booted with.

        NOTE: ``patch("...", return_value=None)`` replaces the dependency
        with a ``MagicMock``, whose signature FastAPI introspects as
        ``(*args, **kwargs)`` — they get surfaced as query params and
        the request 422s before the route runs. Use a real lambda so
        FastAPI sees a no-arg callable. Same pattern in the success
        tests above already works because their assertions never depend
        on the route's response shape.
        """

        async def _noop():
            return None

        with (
            patch.object(get_config(), "embedding_engine", MagicMock()),
            patch.object(get_config(), "embedding_model_locked", "locked-model"),
            patch.object(get_config(), "api_key", None),
            patch("vllm_mlx.middleware.auth.check_rate_limit", new=_noop),
        ):
            app = self._make_app()
            client = TestClient(app)
            r = client.post(
                "/v1/embeddings",
                json={
                    "model": "wrong-model",
                    "input": "test",
                },
            )
            assert r.status_code == 400, r.text
            body = r.json()
            # FastAPI surfaces a dict ``detail`` verbatim when no
            # exception handler is wired (bare ``_make_app``). The
            # production server installs ``install_exception_handlers``
            # which unwraps to ``body["error"]`` directly.
            err = body.get("error") or body.get("detail", {}).get("error")
            assert err is not None, body
            assert err["param"] == "model"
            assert err["code"] == "model_not_found"
            assert "not available" in err["message"]
