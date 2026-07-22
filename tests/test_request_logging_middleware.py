"""Tests for request logging middleware (#51).

Covers:
  1. Middleware logs method, path, status, duration at DEBUG level
  2. Middleware is silent at INFO level (zero-cost)
  3. Health probe paths are excluded
  4. Status code is captured correctly
  5. Duration is measured
"""

from __future__ import annotations

import logging
import re

import pytest

from vllm_mlx.middleware.request_logging import RequestLoggingMiddleware

# ---------------------------------------------------------------------------
# Helpers — minimal ASGI app + scope builders
# ---------------------------------------------------------------------------


def _make_app(status: int = 200, body: bytes = b"ok"):
    """Return a minimal ASGI app that responds with *status*."""

    async def app(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [[b"content-type", b"text/plain"]],
            }
        )
        await send({"type": "http.response.body", "body": body})

    return app


def _http_scope(method: str = "GET", path: str = "/v1/models") -> dict:
    return {"type": "http", "method": method, "path": path}


async def _noop_receive():
    return {"type": "http.request", "body": b""}


_sent_messages: list[dict] = []


async def _collecting_send(message: dict):
    _sent_messages.append(message)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRequestLogging:
    @pytest.fixture(autouse=True)
    def _reset(self):
        _sent_messages.clear()

    @pytest.mark.asyncio
    async def test_logs_at_debug(self, caplog, monkeypatch):
        """Middleware logs method, path, status, duration at DEBUG."""
        inner = _make_app(status=200)
        mw = RequestLoggingMiddleware(inner)

        # Mock perf_counter to verify duration is measured
        call_count = 0

        def _mock_perf_counter():
            nonlocal call_count
            call_count += 1
            return 1.0 if call_count == 1 else 1.234

        monkeypatch.setattr(
            "vllm_mlx.middleware.request_logging.time.perf_counter", _mock_perf_counter
        )

        with caplog.at_level(
            logging.DEBUG, logger="vllm_mlx.middleware.request_logging"
        ):
            await mw(
                _http_scope("POST", "/v1/chat/completions"),
                _noop_receive,
                _collecting_send,
            )

        assert len(caplog.records) == 1
        msg = caplog.records[0].message
        # Format: POST "/v1/chat/completions" 200 0.234s
        assert msg == 'POST "/v1/chat/completions" 200 0.234s'

    @pytest.mark.asyncio
    async def test_silent_at_info(self, caplog):
        """No log output when logger is at INFO level."""
        inner = _make_app(status=200)
        mw = RequestLoggingMiddleware(inner)

        with caplog.at_level(
            logging.INFO, logger="vllm_mlx.middleware.request_logging"
        ):
            await mw(_http_scope("GET", "/v1/models"), _noop_receive, _collecting_send)

        assert len(caplog.records) == 0

    @pytest.mark.asyncio
    async def test_health_excluded(self, caplog):
        """Health probe paths are excluded from logging."""
        inner = _make_app(status=200)
        mw = RequestLoggingMiddleware(inner)

        with caplog.at_level(
            logging.DEBUG, logger="vllm_mlx.middleware.request_logging"
        ):
            await mw(_http_scope("GET", "/health"), _noop_receive, _collecting_send)
            await mw(
                _http_scope("GET", "/health/ready"), _noop_receive, _collecting_send
            )

        assert len(caplog.records) == 0

    @pytest.mark.asyncio
    async def test_captures_error_status(self, caplog):
        """Status code from error responses is captured correctly."""
        inner = _make_app(status=422)
        mw = RequestLoggingMiddleware(inner)

        with caplog.at_level(
            logging.DEBUG, logger="vllm_mlx.middleware.request_logging"
        ):
            await mw(
                _http_scope("POST", "/v1/chat/completions"),
                _noop_receive,
                _collecting_send,
            )

        assert re.match(
            r'POST "/v1/chat/completions" 422 \d+\.\d{3}s',
            caplog.records[0].message,
        )

    @pytest.mark.asyncio
    async def test_non_http_passthrough(self, caplog):
        """Non-HTTP scopes (websocket, lifespan) pass through without logging."""
        inner_called = False

        async def _tracking_app(scope, receive, send):
            nonlocal inner_called
            inner_called = True

        mw = RequestLoggingMiddleware(_tracking_app)

        with caplog.at_level(
            logging.DEBUG, logger="vllm_mlx.middleware.request_logging"
        ):
            await mw({"type": "lifespan"}, _noop_receive, _collecting_send)

        assert inner_called, "inner app must be called for non-HTTP scopes"
        assert len(caplog.records) == 0

    @pytest.mark.asyncio
    async def test_logs_on_exception(self, caplog):
        """Duration is still logged when the inner app raises."""

        async def _failing_app(scope, receive, send):
            raise ValueError("boom")

        mw = RequestLoggingMiddleware(_failing_app)

        with (
            caplog.at_level(
                logging.DEBUG, logger="vllm_mlx.middleware.request_logging"
            ),
            pytest.raises(ValueError, match="boom"),
        ):
            await mw(_http_scope("GET", "/v1/models"), _noop_receive, _collecting_send)

        assert len(caplog.records) == 1
        # Inner app raised before sending a response → logged as 500
        # (matching what Starlette's ServerErrorMiddleware returns)
        assert re.match(r'GET "/v1/models" 500 \d+\.\d{3}s', caplog.records[0].message)

    @pytest.mark.asyncio
    async def test_sanitizes_control_chars(self, caplog):
        """Control characters in path are replaced to prevent log injection."""
        inner = _make_app(status=200)
        mw = RequestLoggingMiddleware(inner)

        with caplog.at_level(
            logging.DEBUG, logger="vllm_mlx.middleware.request_logging"
        ):
            await mw(
                _http_scope("GET", "/v1/models\r\nInjected: evil"),
                _noop_receive,
                _collecting_send,
            )

        msg = caplog.records[0].message
        assert "\r" not in msg
        assert "\n" not in msg
        assert "?" in msg  # control chars replaced with ?
