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
