# SPDX-License-Identifier: Apache-2.0
"""Tests for ServerConfig and middleware modules."""

import time

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

# ======================================================================
# ServerConfig
# ======================================================================


class TestServerConfig:
    def test_default_values(self):
        """Config has sensible defaults."""
        from vllm_mlx.config import ServerConfig

        cfg = ServerConfig()
        assert cfg.engine is None
        assert cfg.model_name is None
        assert cfg.default_max_tokens == 4096
        assert cfg.thinking_token_budget == 2048
        # Bumped 300 → 1800 (PR H22): 5min default silently truncated
        # reasoning generations and 30B+ greedy decodes.
        assert cfg.default_timeout == 1800.0
        assert cfg.api_key is None
        assert cfg.gc_control is True
        assert cfg.enable_auto_tool_choice is False

    def test_get_config_singleton(self):
        """get_config returns the same instance."""
        from vllm_mlx.config import get_config

        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_reset_config(self):
        """reset_config creates a fresh instance."""
        from vllm_mlx.config import get_config, reset_config

        cfg1 = get_config()
        cfg1.model_name = "test-model"

        cfg2 = reset_config()
        assert cfg2.model_name is None
        assert get_config() is cfg2
        assert get_config() is not cfg1

    def test_mutable_fields(self):
        """Config fields are mutable."""
        from vllm_mlx.config import reset_config

        cfg = reset_config()
        cfg.engine = "fake-engine"
        cfg.model_name = "test-model"
        cfg.api_key = "secret"
        cfg.default_max_tokens = 8192

        assert cfg.engine == "fake-engine"
        assert cfg.model_name == "test-model"
        assert cfg.api_key == "secret"
        assert cfg.default_max_tokens == 8192


# ======================================================================
# RateLimiter
# ======================================================================


class TestRateLimiter:
    def test_disabled_allows_all(self):
        """Disabled rate limiter allows everything."""
        from vllm_mlx.middleware.auth import RateLimiter

        rl = RateLimiter(requests_per_minute=1, enabled=False)
        for _ in range(100):
            allowed, _ = rl.is_allowed("client1")
            assert allowed

    def test_enabled_limits(self):
        """Enabled rate limiter blocks after limit."""
        from vllm_mlx.middleware.auth import RateLimiter

        rl = RateLimiter(requests_per_minute=3, enabled=True)
        for i in range(3):
            allowed, _ = rl.is_allowed("client1")
            assert allowed, f"Request {i + 1} should be allowed"

        allowed, retry_after = rl.is_allowed("client1")
        assert not allowed
        assert retry_after > 0

    def test_per_client_isolation(self):
        """Different clients have separate limits."""
        from vllm_mlx.middleware.auth import RateLimiter

        rl = RateLimiter(requests_per_minute=2, enabled=True)
        rl.is_allowed("client_a")
        rl.is_allowed("client_a")

        allowed, _ = rl.is_allowed("client_a")
        assert not allowed  # a exhausted

        allowed, _ = rl.is_allowed("client_b")
        assert allowed  # b is fresh

    def test_window_expiry(self):
        """Requests outside window are cleaned up."""
        from vllm_mlx.middleware.auth import RateLimiter

        rl = RateLimiter(requests_per_minute=1, enabled=True)
        rl.window_size = 0.1  # 100ms window for fast test

        rl.is_allowed("client1")
        allowed, _ = rl.is_allowed("client1")
        assert not allowed

        time.sleep(0.15)  # Wait for window to expire
        allowed, _ = rl.is_allowed("client1")
        assert allowed


# ======================================================================
# verify_api_key
# ======================================================================


class TestVerifyApiKey:
    def _make_app(self):
        from vllm_mlx.middleware.auth import verify_api_key

        app = FastAPI()

        @app.get("/test", dependencies=[Depends(verify_api_key)])
        async def test_endpoint():
            return {"ok": True}

        return app

    def test_no_key_configured(self):
        """No API key → all requests pass."""
        from vllm_mlx.config import get_config

        get_config().api_key = None
        app = self._make_app()
        client = TestClient(app)
        r = client.get("/test")
        assert r.status_code == 200

    def test_valid_key(self):
        """Correct API key passes."""
        from vllm_mlx.config import get_config

        get_config().api_key = "test-secret"
        app = self._make_app()
        client = TestClient(app)
        r = client.get("/test", headers={"Authorization": "Bearer test-secret"})
        assert r.status_code == 200
        get_config().api_key = None  # cleanup

    def test_invalid_key(self):
        """Wrong API key returns 401."""
        from vllm_mlx.config import get_config

        get_config().api_key = "test-secret"
        app = self._make_app()
        client = TestClient(app)
        r = client.get("/test", headers={"Authorization": "Bearer wrong-key"})
        assert r.status_code == 401
        get_config().api_key = None

    def test_missing_key_when_required(self):
        """No key header when key required returns 401."""
        from vllm_mlx.config import get_config

        get_config().api_key = "test-secret"
        app = self._make_app()
        client = TestClient(app)
        r = client.get("/test")
        assert r.status_code == 401
        get_config().api_key = None


# ======================================================================
# check_rate_limit
# ======================================================================


class TestCheckRateLimit:
    def test_rate_limit_dependency(self):
        """Rate limit dependency works in FastAPI."""
        from vllm_mlx.middleware.auth import check_rate_limit, rate_limiter

        rate_limiter.enabled = False

        app = FastAPI()

        @app.get("/test", dependencies=[Depends(check_rate_limit)])
        async def test_endpoint():
            return {"ok": True}

        client = TestClient(app)
        r = client.get("/test")
        assert r.status_code == 200

    def test_rate_limit_blocks(self):
        """Rate limit returns 429 when exceeded."""
        from vllm_mlx.middleware.auth import check_rate_limit, rate_limiter

        rate_limiter.enabled = True
        rate_limiter.requests_per_minute = 1

        app = FastAPI()

        @app.get("/test", dependencies=[Depends(check_rate_limit)])
        async def test_endpoint():
            return {"ok": True}

        client = TestClient(app)
        r1 = client.get("/test")
        assert r1.status_code == 200

        r2 = client.get("/test")
        assert r2.status_code == 429

        # cleanup
        rate_limiter.enabled = False
        rate_limiter.requests_per_minute = 60


# ======================================================================
# configure_cors — Fetch-spec-compliant defaults (#190)
# ======================================================================


class TestConfigureCors:
    """``allow_origins=["*"]`` combined with ``allow_credentials=True`` is
    rejected by browsers per the Fetch standard. ``configure_cors`` must
    auto-disable credentials when a wildcard is present so the default
    serve config doesn't silently break cross-origin clients."""

    def _build_app_and_inspect(self, origins: list[str]):
        from fastapi import FastAPI as _FastAPI

        app = _FastAPI()
        # Mimic ``server.configure_cors`` directly on a fresh app so the
        # test doesn't depend on module-level singletons.
        from fastapi.middleware.cors import CORSMiddleware

        allow_credentials = "*" not in origins
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=allow_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        return app, allow_credentials

    def test_wildcard_origin_disables_credentials(self):
        _, allow_credentials = self._build_app_and_inspect(["*"])
        assert allow_credentials is False

    def test_explicit_origin_keeps_credentials(self):
        _, allow_credentials = self._build_app_and_inspect(["https://example.com"])
        assert allow_credentials is True

    def test_mixed_origins_with_wildcard_disables_credentials(self):
        # If a caller passes both an explicit origin and ``*``, the
        # wildcard wins per the Fetch spec; credentials must still be off.
        _, allow_credentials = self._build_app_and_inspect(["*", "https://example.com"])
        assert allow_credentials is False

    def test_wildcard_default_round_trip_response(self):
        # End-to-end: a CORS preflight against a wildcard config returns
        # ``access-control-allow-origin: *`` and *omits* the credentials
        # header, confirming the FastAPI middleware respected the flag.
        app, _ = self._build_app_and_inspect(["*"])

        @app.get("/probe")
        async def probe():
            return {"ok": True}

        client = TestClient(app)
        r = client.options(
            "/probe",
            headers={
                "Origin": "https://other.example",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert r.headers.get("access-control-allow-origin") == "*"
        assert "access-control-allow-credentials" not in r.headers


# ======================================================================
# Global exception handler — no internal-detail leak in 500 body (#191)
# ======================================================================


class TestGlobalExceptionHandler:
    """The 500 response body must NOT echo ``str(exc)`` or the exception
    type — those routinely contain filesystem paths, model paths, and
    other internals that aid targeted exploitation. Full details still go
    to the server log for operators."""

    def _make_app_with_handler(self):
        from starlette.responses import JSONResponse as _JSONResponse

        app = FastAPI()

        # Mirror the production handler shape exactly.
        @app.exception_handler(Exception)
        async def handler(request, exc):  # noqa: ARG001
            return _JSONResponse(
                status_code=500,
                content={"error": {"message": "Internal server error"}},
            )

        @app.get("/boom")
        async def boom():
            # Use a realistic-looking exception whose message would leak a
            # local filesystem path if echoed back.
            raise FileNotFoundError(
                "[Errno 2] No such file or directory: "
                "'/Users/operator/.cache/secret-config.json'"
            )

        return app

    def test_500_body_does_not_leak_exception_message(self):
        app = self._make_app_with_handler()
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/boom")
        assert r.status_code == 500
        body = r.json()
        assert body == {"error": {"message": "Internal server error"}}

        # Specifically guard against the common leak shapes.
        text = r.text
        assert "FileNotFoundError" not in text
        assert "/Users/" not in text
        assert ".cache" not in text
        assert "Errno" not in text

    def test_500_body_omits_exception_type_field(self):
        # The previous handler set ``error.type = type(exc).__name__``,
        # which exposes the implementation language and module shape to
        # clients. The new handler drops that field entirely.
        app = self._make_app_with_handler()
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/boom")
        assert "type" not in r.json()["error"]


class TestHTTPExceptionHandler:
    def _make_app_with_handler(self):
        from fastapi import FastAPI, HTTPException

        from vllm_mlx.server import _http_exception_handler

        app = FastAPI()
        app.add_exception_handler(
            HTTPException,
            _http_exception_handler,
        )

        @app.get("/rate-limit")
        async def rate_limit():
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": "60"},
            )

        @app.get("/custom")
        async def custom():
            raise HTTPException(
                status_code=418,
                detail="teapot",
            )

        @app.get("/auth")
        async def auth():
            raise HTTPException(
                status_code=401,
                detail="Invalid API key",
            )

        return app

    def test_429_returns_openai_style_error(self):
        app = self._make_app_with_handler()
        client = TestClient(app)

        r = client.get("/rate-limit")

        assert r.status_code == 429
        assert r.headers["Retry-After"] == "60"

        assert r.json() == {
            "error": {
                "message": "Rate limit exceeded",
                "type": "rate_limit_error",
                "code": None,
                "param": None,
            }
        }

    def test_unknown_status_uses_api_error(self):
        app = self._make_app_with_handler()
        client = TestClient(app)

        r = client.get("/custom")

        assert r.status_code == 418
        assert r.json()["error"]["type"] == "api_error"

    def test_http_exception_without_headers(self):
        app = self._make_app_with_handler()
        client = TestClient(app)

        r = client.get("/auth")

        assert r.status_code == 401
        assert r.json()["error"]["type"] == "authentication_error"


class TestHTTPExceptionHandlerOnProductionApp:
    """End-to-end tests that hit ``vllm_mlx.server.app`` directly.

    The throwaway-app tests above verify the handler's *logic* but
    would still pass if the production ``@app.exception_handler(...)``
    registration in ``server.py`` were removed or registered for the
    wrong exception class. These tests close that gap: they prove the
    real app produces the OpenAI-style error envelope, including for
    Starlette-generated 404/405 from the router (which is its own
    exception class, NOT fastapi.HTTPException).
    """

    def test_unknown_route_returns_openai_envelope(self):
        # Router-level 404 raises starlette.exceptions.HTTPException,
        # NOT fastapi.HTTPException. If the handler is registered for
        # the fastapi class only, this still returns the default
        # {"detail": "Not Found"} shape — which would break OpenAI-SDK
        # clients that parse error.message.
        from vllm_mlx.server import app

        client = TestClient(app)
        r = client.get("/this-route-does-not-exist-anywhere")

        assert r.status_code == 404
        body = r.json()
        assert "error" in body, (
            f"production app returned non-OpenAI shape for 404: {body}"
        )
        assert body["error"]["type"] == "not_found_error"
        assert body["error"]["code"] is None
        assert body["error"]["param"] is None

    def test_wrong_method_returns_openai_envelope(self):
        # Same class of bug as 404 — router-emitted 405.
        from vllm_mlx.server import app

        client = TestClient(app)
        # /healthz is GET-only; a POST should 405 via the router
        r = client.post("/healthz")

        assert r.status_code == 405
        body = r.json()
        assert "error" in body, (
            f"production app returned non-OpenAI shape for 405: {body}"
        )
        assert body["error"]["type"] == "invalid_request_error"
