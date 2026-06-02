# SPDX-License-Identifier: Apache-2.0
"""Unit tests for vllm_mlx.share.session."""

from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from vllm_mlx.share import session


def _ok_response(payload: dict) -> io.BytesIO:
    body = json.dumps(payload).encode()
    resp = io.BytesIO(body)
    resp.status = 200  # type: ignore[attr-defined]
    return resp


def test_relay_base_url_defaults_to_prod():
    with patch.dict("os.environ", {}, clear=False):
        # Pop the override so we see the real default.
        import os

        os.environ.pop("RAPID_MLX_RELAY_URL", None)
        assert session.relay_base_url() == "https://api.rapidmlx.com"


def test_relay_base_url_respects_env_override():
    with patch.dict("os.environ", {"RAPID_MLX_RELAY_URL": "http://localhost:8080/"}):
        assert session.relay_base_url() == "http://localhost:8080"


def test_request_returns_parsed_session():
    payload = {
        "subdomain": "abc123",
        "token": "tk_test",
        "frps_host": "tunnel.rapidmlx.com",
        "frps_port": 7000,
        "public_url": "https://abc123.rapidmlx.com",
        "expires_at": "2026-06-02T12:00:00Z",
    }
    with patch("urllib.request.urlopen") as mock_open:
        mock_open.return_value.__enter__.return_value = _ok_response(payload)
        sess = session.request(model="qwen3.5-4b")

    assert sess.subdomain == "abc123"
    assert sess.token == "tk_test"
    assert sess.frps_port == 7000
    assert sess.public_url == "https://abc123.rapidmlx.com"


def test_request_raises_on_4xx_with_actionable_message():
    err = urllib.error.HTTPError(
        url="https://api.rapidmlx.com/share/session",
        code=429,
        msg="Too Many Requests",
        hdrs=None,  # type: ignore[arg-type]
        fp=None,
    )
    with (
        patch("urllib.request.urlopen", side_effect=err),
        pytest.raises(RuntimeError, match="HTTP 429"),
    ):
        session.request(model="qwen3.5-4b")


def test_request_raises_on_unreachable_with_dev_override_hint():
    err = urllib.error.URLError("connection refused")
    with (
        patch("urllib.request.urlopen", side_effect=err),
        pytest.raises(RuntimeError, match="RAPID_MLX_RELAY_URL"),
    ):
        session.request(model="qwen3.5-4b")


def test_relay_base_url_rejects_plaintext_remote_override():
    """Codex CONCERN: the override must NOT silently send a freshly-minted
    bearer key to an arbitrary plaintext URL. HTTPS-only, except for
    explicit loopback."""
    with (
        patch.dict("os.environ", {"RAPID_MLX_RELAY_URL": "http://attacker.example/"}),
        pytest.raises(RuntimeError, match="unsafe"),
    ):
        session.relay_base_url()


def test_relay_base_url_accepts_https_override():
    with patch.dict(
        "os.environ", {"RAPID_MLX_RELAY_URL": "https://staging.rapidmlx.com/"}
    ):
        assert session.relay_base_url() == "https://staging.rapidmlx.com"


def test_relay_base_url_accepts_loopback_http_override():
    """Local dev convenience — but only loopback, not arbitrary plaintext."""
    with patch.dict("os.environ", {"RAPID_MLX_RELAY_URL": "http://127.0.0.1:8080"}):
        assert session.relay_base_url() == "http://127.0.0.1:8080"


def test_request_raises_runtimeerror_on_invalid_json_body():
    """DeepSeek round-4 BLOCKER #1: a 2xx response with non-JSON body
    (e.g. an upstream proxy serving an HTML error page) must surface
    as RuntimeError, not a raw ValueError traceback."""

    class _NotJsonResp:
        status = 200

        def read(self):
            return b"<html>upstream down</html>"

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    with (
        patch("urllib.request.urlopen", return_value=_NotJsonResp()),
        pytest.raises(RuntimeError, match="non-JSON"),
    ):
        session.request(model="qwen3.5-4b")


def test_relay_base_url_rejects_empty_hostname():
    """DeepSeek round-4 NIT #4: ``https://`` parses but has no hostname.
    Reject it loudly instead of building ``https:///share/session``."""
    with (
        patch.dict("os.environ", {"RAPID_MLX_RELAY_URL": "https://"}),
        pytest.raises(RuntimeError, match="hostname"),
    ):
        session.relay_base_url()


def test_request_raises_runtimeerror_on_missing_field():
    """DeepSeek BLOCKER #4: relay payload missing a field must surface as
    a user-readable error, not a raw KeyError traceback."""
    payload = {  # ``token`` deliberately absent
        "subdomain": "abc123",
        "frps_host": "tunnel.rapidmlx.com",
        "frps_port": 7000,
        "public_url": "https://abc123.rapidmlx.com",
        "expires_at": "2026-06-02T12:00:00Z",
    }
    with patch("urllib.request.urlopen") as mock_open:
        mock_open.return_value.__enter__.return_value = _ok_response(payload)
        with pytest.raises(RuntimeError, match="unexpected response shape"):
            session.request(model="qwen3.5-4b")
