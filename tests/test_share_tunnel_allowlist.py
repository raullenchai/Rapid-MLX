# SPDX-License-Identifier: Apache-2.0
"""The public share tunnel is a model-serving proxy, not a local-admin proxy."""

from __future__ import annotations

import asyncio
import base64
import json
from unittest.mock import MagicMock

import pytest

from rapid_mlx.share import ws_tunnel


def _req_msg(
    *,
    method: str = "POST",
    path: str = "/v1/chat/completions",
    body: bytes = b"{}",
) -> dict:
    return {
        "t": "req",
        "id": "r1",
        "method": method,
        "path": path,
        "headers": {"Content-Type": "application/json"},
        "body": base64.b64encode(body).decode("ascii"),
    }


def _run_serve_request(client: ws_tunnel.TunnelClient, msg: dict) -> list[dict]:
    sent: list[dict] = []

    async def _send(obj):
        sent.append(obj)

    client._send = _send  # type: ignore[method-assign]
    asyncio.run(client._serve_request(msg, msg["id"]))
    return sent


def _response_body(frames: list[dict]) -> bytes:
    return b"".join(
        base64.b64decode(frame["data"]) for frame in frames if frame.get("t") == "chunk"
    )


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/healthz"),
        ("GET", "/v1/models"),
        ("GET", "/v1/models?detail=true"),
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/%63hat/completions"),
        ("POST", "/v1/completions"),
        ("POST", "/v1/responses"),
        ("POST", "/v1/messages"),
        ("POST", "/v1/embeddings"),
        ("POST", "/v1/rerank"),
        ("POST", "/v1/images/generations"),
        ("POST", "/v1/images/edits"),
        ("GET", "/v1/images/progress"),
        ("POST", "/v1/images/cancel"),
        ("POST", "/v1/audio/transcriptions"),
        ("POST", "/v1/audio/translations"),
        ("POST", "/v1/audio/speech"),
        ("POST", "/v1/audio/music"),
        ("GET", "/v1/audio/voices"),
        ("GET", "/v1/videos"),
        ("POST", "/v1/videos"),
        ("POST", "/v1/videos/extend"),
        ("GET", "/v1/videos/capabilities"),
        ("GET", "/v1/videos/video_0123456789abcdef0123456789abcdef"),
        ("GET", "/v1/videos/video_0123456789abcdef0123456789abcdef/content"),
        ("OPTIONS", "/v1/chat/completions"),
        ("OPTIONS", "/v1/models"),
    ],
)
def test_allowed_public_share_endpoints(method: str, path: str) -> None:
    assert ws_tunnel._is_allowed_share_request(method, path)


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/v1/chat/completions"),
        ("POST", "/v1/models"),
        ("DELETE", "/v1/videos/video_0123456789abcdef0123456789abcdef"),
        ("POST", "/v1/cua/runs"),
        ("GET", "/v1/cua/runs"),
        ("POST", "/v1/mcp/execute"),
        ("GET", "/v1/mcp/tools"),
        ("POST", "/v1/cache/import"),
        ("POST", "/v1/cache/prefill"),
        ("GET", "/v1/status"),
        ("GET", "/v1/models/residency"),
        ("POST", "/v1/models/load"),
        ("GET", "/v1/runtime/config"),
        ("GET", "/v1/local-search/status"),
        ("POST", "/v1/local-search/search"),
        ("POST", "/v1/agent/runs"),
        ("POST", "/v1/models/activate"),
        ("DELETE", "/v1/requests/abc"),
        ("POST", "/v1/cache/clear"),
        ("OPTIONS", "/v1/cua/runs"),
        ("GET", "/"),
        ("GET", "/v1/models/"),
    ],
)
def test_rejects_non_serving_and_control_plane_endpoints(
    method: str, path: str
) -> None:
    assert not ws_tunnel._is_allowed_share_request(method, path)


@pytest.mark.parametrize(
    "path",
    [
        "/v1/../v1/cua/runs",
        "/v1/models/../cua/runs",
        "/v1/models%2f..%2fcua%2fruns",
        "/%76%31/cua/runs",
        "/v1/%63ua/runs",
        "//v1/cua/runs",
        "/v1//cua/runs",
        "/v1/cua\\runs",
        "http://127.0.0.1:8000/v1/cua/runs",
        "//evil.example/v1/cua/runs",
        "//[",
        "/v1/%ff/runs",
        "/v1/cua/runs#fragment",
        "/v1/cua/runs?x=1",
    ],
)
def test_path_normalization_prevents_control_plane_bypass(path: str) -> None:
    assert not ws_tunnel._is_allowed_share_request("POST", path)


@pytest.mark.parametrize(
    "path",
    [
        "/v1/../v1/chat/completions",
        "/v1/chat/completions/../models",
        "/v1//chat/completions",
        "//v1/chat/completions",
        "http://127.0.0.1:8000/v1/chat/completions",
    ],
)
def test_path_normalization_does_not_grant_ambiguous_inference_paths(
    path: str,
) -> None:
    # Percent-encoding a *character* inside an otherwise canonical path is
    # harmless and accepted above (``/v1/%63hat/completions``); dot segments,
    # duplicate slashes, and absolute URLs are rejected rather than normalized
    # into a different route.
    assert not ws_tunnel._is_allowed_share_request("POST", path)


def test_rejected_request_returns_synthetic_404_without_touching_loopback() -> None:
    client = ws_tunnel.TunnelClient(local_port=1)
    client._perform_local_fetch = MagicMock(  # type: ignore[method-assign]
        side_effect=AssertionError("loopback must not be touched")
    )
    frames = _run_serve_request(client, _req_msg(path="/v1/cua/runs"))
    client._perform_local_fetch.assert_not_called()
    assert [frame["t"] for frame in frames] == ["head", "chunk", "end"]
    assert frames[0]["status"] == 404
    assert frames[0]["headers"]["content-type"] == "application/json"
    assert json.loads(_response_body(frames))["error"]["message"] == "not found"


def test_allowed_request_still_reaches_loopback_with_original_path() -> None:
    client = ws_tunnel.TunnelClient(local_port=1)
    captured: dict = {}

    def _fake_fetch(req_id, method, path, headers, body):
        captured.update(
            req_id=req_id, method=method, path=path, headers=headers, body=body
        )

    client._perform_local_fetch = _fake_fetch  # type: ignore[method-assign]
    _run_serve_request(
        client,
        _req_msg(
            method="POST", path="/v1/chat/completions?trace=1", body=b'{"stream":true}'
        ),
    )
    assert captured["req_id"] == "r1"
    assert captured["method"] == "POST"
    assert captured["path"] == "/v1/chat/completions?trace=1"
    assert captured["body"] == b'{"stream":true}'


def test_options_preflight_is_forwarded_only_for_allowed_paths() -> None:
    client = ws_tunnel.TunnelClient(local_port=1)
    captured: dict = {}

    def _fake_fetch(req_id, method, path, headers, body):
        captured.update(
            req_id=req_id, method=method, path=path, headers=headers, body=body
        )

    client._perform_local_fetch = _fake_fetch  # type: ignore[method-assign]
    _run_serve_request(
        client,
        _req_msg(method="OPTIONS", path="/v1/chat/completions", body=b""),
    )
    assert captured["method"] == "OPTIONS"

    client._perform_local_fetch = MagicMock(  # type: ignore[method-assign]
        side_effect=AssertionError(
            "preflight for a blocked route must not reach loopback"
        )
    )
    frames = _run_serve_request(
        client,
        _req_msg(method="OPTIONS", path="/v1/cua/runs", body=b""),
    )
    client._perform_local_fetch.assert_not_called()
    assert frames[0]["status"] == 404
