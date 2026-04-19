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
        assert cfg.default_timeout == 300.0
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
