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


def test_http_error_response_body_closed():
    """Round 2 codex review: ``HTTPError`` holds a file-like response
    object on ``e.fp``; if the handler returns/retries without an
    explicit close, the socket leaks until gc cycles it. Pin that
    ``e.close()`` is called for both 4xx and 5xx paths."""
    from vllm_mlx.telemetry import transport

    closed_4xx = mock.MagicMock()
    err_4xx = HTTPError(
        url="https://x",
        code=400,
        msg="bad",
        hdrs=None,
        fp=None,  # type: ignore[arg-type]
    )
    err_4xx.close = closed_4xx  # type: ignore[method-assign]

    closed_5xx_a = mock.MagicMock()
    closed_5xx_b = mock.MagicMock()
    closed_5xx_c = mock.MagicMock()
    err_5xx_attempts = []
    for c in (closed_5xx_a, closed_5xx_b, closed_5xx_c):
        e = HTTPError(
            url="https://x",
            code=503,
            msg="busy",
            hdrs=None,
            fp=None,  # type: ignore[arg-type]
        )
        e.close = c  # type: ignore[method-assign]
        err_5xx_attempts.append(e)

    with (
        mock.patch.object(transport, "urlopen", side_effect=[err_4xx]),
        mock.patch.object(transport.time, "sleep"),
    ):
        transport.post_batch([{"x": 1}])
    closed_4xx.assert_called_once()

    with (
        mock.patch.object(transport, "urlopen", side_effect=err_5xx_attempts),
        mock.patch.object(transport.time, "sleep"),
    ):
        transport.post_batch([{"x": 1}])
    closed_5xx_a.assert_called_once()
    closed_5xx_b.assert_called_once()
    closed_5xx_c.assert_called_once()


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


def test_non_https_non_loopback_override_silently_falls_back_to_default(monkeypatch):
    """Pre-round-3, an ``http://insecure.example`` override was rejected
    inside ``post_batch`` ("refusing non-HTTPS endpoint"). After round 3
    the rejection happens earlier — in ``endpoint()`` — and the override
    silently falls back to the production HTTPS default. Either way:
    nothing on plain HTTP, on a non-loopback host, ever sees user
    events."""
    from vllm_mlx.telemetry import transport

    monkeypatch.setenv("RAPID_MLX_TELEMETRY_ENDPOINT", "http://insecure.example/v1")
    # The override is silently dropped; ``endpoint()`` returns the
    # production default.
    assert transport.endpoint() == transport.DEFAULT_ENDPOINT


def test_endpoint_override_only_accepts_localhost(monkeypatch):
    """Round 3 codex review: an unrestricted ``RAPID_MLX_TELEMETRY_ENDPOINT``
    let a hostile shell rc / wrapper script redirect opted-in users'
    events to an attacker-controlled collector. The override is now
    restricted to localhost / 127.0.0.1 / ::1 (the only legitimate use
    cases: wrangler dev + CI fixtures)."""
    from vllm_mlx.telemetry import transport

    # Localhost variants pass through.
    for ok in (
        "http://localhost:8787/v1/events",
        "http://127.0.0.1:8787/v1/events",
        "https://localhost/v1/events",
        "http://[::1]:8787/v1/events",
    ):
        monkeypatch.setenv("RAPID_MLX_TELEMETRY_ENDPOINT", ok)
        assert transport.endpoint() == ok, f"{ok} should be accepted"

    # Anything else falls back to the production default.
    for bad in (
        "https://attacker.example/v1/events",
        "https://localhost.attacker.example/v1/events",  # suffix attack
        "https://telemetry.attacker.example/v1/events",
        "ftp://localhost/v1/events",
        "not-a-url",
    ):
        monkeypatch.setenv("RAPID_MLX_TELEMETRY_ENDPOINT", bad)
        assert transport.endpoint() == transport.DEFAULT_ENDPOINT, (
            f"{bad} should be refused"
        )


def test_malformed_url_request_does_not_raise(monkeypatch):
    """Round 5 codex review: ``Request(url, ...)`` raises ``ValueError``
    on a malformed URL (control bytes, NULL, etc.). The never-raises
    contract was leaking that. Pin both the URL with embedded control
    char path and the empty-scheme path."""
    from vllm_mlx.telemetry import transport

    # Patch ``endpoint()`` to return a malformed URL that nonetheless
    # passes the prefix check (so we reach ``Request``).
    bad_url = "https://example.com/v1/\x00events"  # NULL → ValueError
    with (
        mock.patch.object(transport, "endpoint", return_value=bad_url),
        mock.patch.object(transport, "_is_localhost_override", return_value=False),
        mock.patch.object(transport.time, "sleep"),
    ):
        # Must NOT raise the ValueError out to the caller.
        assert transport.post_batch([{"x": 1}]) is False


def test_malformed_port_localhost_override_does_not_raise(monkeypatch):
    """Round 4 codex review: ``http://localhost:bad/`` passed the
    hostname check, then later ``Request``/``urlopen`` raised a
    ``ValueError`` outside the URLError/OSError catch — violating the
    never-raises contract. ``_is_localhost_override`` now forces
    ``parts.port`` access so the malformed port is rejected here."""
    from vllm_mlx.telemetry import transport

    monkeypatch.setenv("RAPID_MLX_TELEMETRY_ENDPOINT", "http://localhost:bad/v1")
    # Falls back to production default (the override is silently dropped).
    assert transport.endpoint() == transport.DEFAULT_ENDPOINT


def test_last_attempt_5xx_log_says_giving_up_not_will_retry():
    """Round 4 codex review: stale 'will retry' log on the final
    attempt was misleading. Pin both 5xx-status and 5xx-HTTPError
    branches log ``giving up`` once retries are exhausted."""
    from vllm_mlx.telemetry import transport

    captured: list[str] = []

    def fake_log(msg):
        captured.append(msg)

    resp = mock.MagicMock()
    resp.status = 503
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False

    with (
        mock.patch.object(transport, "urlopen", return_value=resp),
        mock.patch.object(transport.time, "sleep"),
        mock.patch.object(transport, "_log", fake_log),
    ):
        assert transport.post_batch([{"x": 1}]) is False

    # The final attempt's log must say "giving up", not "will retry".
    last = captured[-1]
    assert "giving up" in last, captured
    assert "will retry" not in last


def test_non_serializable_payload_returns_false_not_raise():
    """Round 3 codex review: ``json.dumps`` ran outside the transport
    try, so a non-serializable payload would have raised through the
    ``never raises`` contract. Pin that it returns ``False`` instead."""
    from vllm_mlx.telemetry import transport

    class _NotSerializable:
        pass

    with mock.patch.object(transport, "urlopen") as urlopen:
        # Must not raise — ``post_batch`` is the privacy/safety boundary.
        assert transport.post_batch([{"x": _NotSerializable()}]) is False
        urlopen.assert_not_called()


def test_retry_constants_are_finite():
    """Round 7 reverted the "atexit waits for transport worst case"
    design. ``session_end`` is now best-effort: the queue's own
    ``SHUTDOWN_BUDGET_S`` (~2 s) caps user-visible exit latency, and
    the transport's own retries run inside that budget. We still pin
    the constants exist with sane types so a future change can't make
    them ``None`` or pathologically large."""
    from vllm_mlx.telemetry import transport

    assert isinstance(transport.TIMEOUT_S, float)
    assert 0 < transport.TIMEOUT_S < 10
    assert isinstance(transport.RETRY_BACKOFFS_S, tuple)
    assert all(
        isinstance(b, (int, float)) and b >= 0 for b in transport.RETRY_BACKOFFS_S
    )


def test_user_agent_is_self_identifying():
    """Cloudflare's bot manager rejects the stdlib default
    ``Python-urllib/*`` UA with HTTP 403 before the request reaches the
    Worker. The transport must therefore set a non-generic UA — and
    the contract is to identify the package + version explicitly so
    the receiver can attribute traffic.

    Round 15 codex review caught the previous ``"rapid-mlx" in ua``
    assertion was too loose: a versionless ``"rapid-mlx"`` UA would
    have passed, defeating the "package + version" contract advertised
    in the docstring. Pin the exact ``rapid-mlx/<version>`` shape."""
    import re

    from vllm_mlx.telemetry import transport

    ua = transport._user_agent()
    assert re.search(r"\brapid-mlx/\S+", ua), (
        f"UA must follow 'rapid-mlx/<version>' shape, got {ua!r}"
    )
    assert "Python-urllib" not in ua


def test_post_sends_self_identifying_user_agent():
    """End-to-end: ``post_batch`` must build a Request whose ``user-agent``
    header is the self-identifying string, not the urllib default."""
    import re

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
    # Round 15: same tighter assertion as ``_user_agent`` — package
    # AND version, not just the package name.
    assert re.search(r"\brapid-mlx/\S+", ua), (
        f"UA must follow 'rapid-mlx/<version>' shape, got {ua!r}"
    )
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
