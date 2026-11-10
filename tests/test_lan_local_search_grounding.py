# SPDX-License-Identifier: Apache-2.0
"""LAN inference must not expose this Mac's indexed passages through chat."""

from unittest.mock import AsyncMock

import pytest
from fastapi.responses import JSONResponse

from rapid_mlx.config import get_config
from rapid_mlx.local_search.grounding import prepare_grounding
from rapid_mlx.middleware.exception_handlers import install_exception_handlers
from rapid_mlx.middleware.lan_inference import LANInferenceMiddleware
from tests.test_lan_inference import request
from tests.test_local_search_grounding import body, client_with, service
from tests.test_response_cache_route import _CountingEngine


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("peer", ["192.168.1.42", "::ffff:192.168.1.42", "unknown"])
async def test_lan_grounding_denied_before_preparation(monkeypatch, stream, peer):
    monkeypatch.setenv("RAPID_MLX_LAN_INFERENCE_ONLY", "1")
    engine, search = _CountingEngine(), service([])
    app = client_with(engine, search).app
    get_config().api_key = "secret"
    install_exception_handlers(app)
    app.add_middleware(LANInferenceMiddleware)
    prepare = AsyncMock(wraps=prepare_grounding)
    monkeypatch.setattr("rapid_mlx.local_search.grounding.prepare_grounding", prepare)

    response = await request(
        app,
        "POST",
        "/v1/chat/completions",
        peer=peer,
        json=body(stream=stream),
        extra_headers={"X-Forwarded-For": "127.0.0.1"},
    )
    blocked = await request(app, "POST", "/v1/local-search/search", peer=peer)
    assert response.status_code == blocked.status_code == 403
    assert (
        response.json()
        == blocked.json()
        == {"detail": "This route is available only on this Mac"}
    )
    prepare.assert_not_called()
    search.store.ready.assert_not_awaited()
    search.folder.assert_not_awaited()
    search.search.assert_not_awaited()
    assert engine.chat_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary,peer,grounded",
    [
        ("1", "192.168.1.42", False),
        ("1", "127.0.0.1", True),
        ("1", "::1", True),
        ("1", "::ffff:127.0.0.1", True),
        ("0", "192.168.1.42", True),
        (None, "192.168.1.42", True),
    ],
)
async def test_lan_grounding_preserves_allowed_requests(
    monkeypatch, boundary, peer, grounded
):
    if boundary is None:
        monkeypatch.delenv("RAPID_MLX_LAN_INFERENCE_ONLY", raising=False)
    else:
        monkeypatch.setenv("RAPID_MLX_LAN_INFERENCE_ONLY", boundary)
    engine, search = _CountingEngine(), service([])
    app = client_with(engine, search).app
    get_config().api_key = "secret"
    install_exception_handlers(app)
    app.add_middleware(LANInferenceMiddleware)
    inference = AsyncMock(return_value=JSONResponse({"choices": []}))
    monkeypatch.setattr("rapid_mlx.routes.chat._create_chat_completion_impl", inference)
    payload = body()
    if not grounded:
        payload.pop("local_search")
    response = await request(
        app, "POST", "/v1/chat/completions", peer=peer, json=payload
    )
    assert response.status_code == 200, response.text
    assert engine.chat_calls == 0
    if grounded:
        search.search.assert_awaited_once()
        assert response.json()["local_search"]["status"] == "no_matches"
        inference.assert_not_awaited()
    else:
        search.search.assert_not_awaited()
        inference.assert_awaited_once()
        assert "local_search" not in response.json()
