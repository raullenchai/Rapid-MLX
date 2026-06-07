# SPDX-License-Identifier: Apache-2.0
"""Contract pins for ``vllm_mlx.telemetry.transport``.

The transport must:
- Refuse a non-HTTPS endpoint.
- Cap body size below the Worker's own 256 KB limit.
- Retry transient errors, not 4xx.
- Treat URLError and TimeoutError as distinct (URLError is NOT a
  TimeoutError subclass).
- Never raise — the user must never see a stack trace from telemetry.
"""

from __future__ import annotations

from unittest import mock
from urllib.error import HTTPError, URLError

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("RAPID_MLX_TELEMETRY_DEBUG", raising=False)
    monkeypatch.delenv("RAPID_MLX_TELEMETRY_ENDPOINT", raising=False)


def test_empty_batch_is_success_no_network():
    from vllm_mlx.telemetry import transport

    with mock.patch.object(transport, "urlopen") as urlopen:
        assert transport.post_batch([]) is True
        urlopen.assert_not_called()


def test_post_batch_returns_true_on_2xx():
    from vllm_mlx.telemetry import transport

    resp = mock.MagicMock()
    resp.status = 200
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False

    with mock.patch.object(transport, "urlopen", return_value=resp) as urlopen:
        assert transport.post_batch([{"x": 1}]) is True
        assert urlopen.call_count == 1


def test_4xx_is_immediate_drop_no_retry():
    from vllm_mlx.telemetry import transport

    resp = mock.MagicMock()
    resp.status = 400
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False

    with (
        mock.patch.object(transport, "urlopen", return_value=resp) as urlopen,
        mock.patch.object(transport.time, "sleep") as sleep,
    ):
        assert transport.post_batch([{"x": 1}]) is False
        assert urlopen.call_count == 1  # no retry on schema bug
        sleep.assert_not_called()


def test_5xx_retries_then_gives_up():
    from vllm_mlx.telemetry import transport

    resp = mock.MagicMock()
    resp.status = 503
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False

    with (
        mock.patch.object(transport, "urlopen", return_value=resp) as urlopen,
        mock.patch.object(transport.time, "sleep") as sleep,
    ):
        assert transport.post_batch([{"x": 1}]) is False
        assert urlopen.call_count == 3  # 1 + 2 retries
        assert sleep.call_count == 2  # two backoffs between three attempts


def test_url_error_treated_as_distinct_from_timeout():
    """URLError is NOT a TimeoutError subclass; both must be caught.

    Codex round 4 on PR #504 surfaced this exact gotcha — if you catch
    only TimeoutError, a connection-reset URLError surfaces as an
    unhandled exception and crashes the foreground.
    """
    from vllm_mlx.telemetry import transport

    with (
        mock.patch.object(
            transport, "urlopen", side_effect=URLError("connection reset")
        ),
        mock.patch.object(transport.time, "sleep"),
    ):
        # Test passes if no exception escapes.
        assert transport.post_batch([{"x": 1}]) is False


def test_timeout_error_caught():
    from vllm_mlx.telemetry import transport

    with (
        mock.patch.object(transport, "urlopen", side_effect=TimeoutError("slow")),
        mock.patch.object(transport.time, "sleep"),
    ):
        assert transport.post_batch([{"x": 1}]) is False


def test_os_error_caught():
    """DNS lookup failures and the like surface as bare OSError on some
    platforms (notably macOS during airplane-mode toggling); pin that
    they don't escape."""
    from vllm_mlx.telemetry import transport

    with (
        mock.patch.object(
            transport, "urlopen", side_effect=OSError("name resolution failed")
        ),
        mock.patch.object(transport.time, "sleep"),
    ):
        assert transport.post_batch([{"x": 1}]) is False


def test_http_error_4xx_does_not_retry():
    from vllm_mlx.telemetry import transport

    exc = HTTPError(
        url="https://x",
        code=400,
        msg="bad",
        hdrs=None,
        fp=None,  # type: ignore[arg-type]
    )
    with (
        mock.patch.object(transport, "urlopen", side_effect=exc) as urlopen,
        mock.patch.object(transport.time, "sleep"),
    ):
        assert transport.post_batch([{"x": 1}]) is False
        assert urlopen.call_count == 1


def test_http_error_5xx_retries():
    from vllm_mlx.telemetry import transport

    exc = HTTPError(
        url="https://x",
        code=503,
        msg="busy",
        hdrs=None,
        fp=None,  # type: ignore[arg-type]
    )
    with (
        mock.patch.object(transport, "urlopen", side_effect=exc) as urlopen,
        mock.patch.object(transport.time, "sleep"),
    ):
        assert transport.post_batch([{"x": 1}]) is False
        assert urlopen.call_count == 3


def test_oversized_payload_dropped_locally():
    """Payloads bigger than 200KB get rejected before hitting the network."""
    from vllm_mlx.telemetry import transport

    # Build a payload that crosses the 200KB threshold but isn't gargantuan.
    big = {"x": "a" * (transport.MAX_BODY_BYTES + 100)}
    with mock.patch.object(transport, "urlopen") as urlopen:
        assert transport.post_batch([big]) is False
        urlopen.assert_not_called()


def test_non_https_endpoint_refused(monkeypatch):
    from vllm_mlx.telemetry import transport

    monkeypatch.setenv("RAPID_MLX_TELEMETRY_ENDPOINT", "http://insecure.example/v1")
    with mock.patch.object(transport, "urlopen") as urlopen:
        assert transport.post_batch([{"x": 1}]) is False
        urlopen.assert_not_called()


def test_endpoint_override_via_env(monkeypatch):
    from vllm_mlx.telemetry import transport

    monkeypatch.setenv("RAPID_MLX_TELEMETRY_ENDPOINT", "https://debug.example/v1")
    assert transport.endpoint() == "https://debug.example/v1"


def test_user_agent_is_self_identifying():
    """Cloudflare's bot manager rejects the stdlib default
    ``Python-urllib/*`` UA with HTTP 403 before the request reaches the
    Worker. The transport must therefore set a non-generic UA — and
    the contract is to identify the package + version explicitly so
    the receiver can attribute traffic."""
    from vllm_mlx.telemetry import transport

    ua = transport._user_agent()
    assert "rapid-mlx" in ua
    assert "Python-urllib" not in ua


def test_post_sends_self_identifying_user_agent():
    """End-to-end: ``post_batch`` must build a Request whose ``user-agent``
    header is the self-identifying string, not the urllib default."""
    from vllm_mlx.telemetry import transport

    captured: dict = {}

    def fake_urlopen(req, timeout):
        captured["headers"] = dict(req.headers)
        resp = mock.MagicMock()
        resp.status = 200
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        return resp

    with mock.patch.object(transport, "urlopen", fake_urlopen):
        assert transport.post_batch([{"x": 1}]) is True
    # urllib lowercases header keys when stored on the Request, but
    # the iteration order varies; use a case-insensitive lookup.
    ua = {k.lower(): v for k, v in captured["headers"].items()}["user-agent"]
    assert "rapid-mlx" in ua
    assert "Python-urllib" not in ua


def test_debug_env_truthy_off_by_default(monkeypatch):
    from vllm_mlx.telemetry import transport

    monkeypatch.delenv("RAPID_MLX_TELEMETRY_DEBUG", raising=False)
    assert transport.debug_enabled() is False

    for falsy in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("RAPID_MLX_TELEMETRY_DEBUG", falsy)
        assert transport.debug_enabled() is False

    for truthy in ("1", "true", "yes", "on"):
        monkeypatch.setenv("RAPID_MLX_TELEMETRY_DEBUG", truthy)
        assert transport.debug_enabled() is True
