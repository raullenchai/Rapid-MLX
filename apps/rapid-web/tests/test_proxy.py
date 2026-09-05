# SPDX-License-Identifier: Apache-2.0
"""Tests for the streaming reverse proxy."""

from __future__ import annotations

import httpx
import pytest

from rmlx_web import proxy


class TestHeaderHandling:
    def test_hop_by_hop_headers_are_dropped(self):
        headers = httpx.Headers(
            {
                "content-type": "application/json",
                "content-length": "42",
                "transfer-encoding": "chunked",
                "connection": "keep-alive",
                "x-request-id": "abc",
            }
        )
        filtered = proxy.filtered_response_headers(headers)

        # Content-Length counts the engine's bytes, not ours, and
        # Transfer-Encoding has already been undone by httpx. Copying
        # either corrupts our own framing.
        assert "content-length" not in filtered
        assert "transfer-encoding" not in filtered
        assert "connection" not in filtered
        assert filtered["x-request-id"] == "abc"

    def test_upstream_headers_carry_the_engine_key(self):
        headers = proxy.upstream_headers("engine-key")
        assert headers["Authorization"] == "Bearer engine-key"
        assert headers["Content-Type"] == "application/json"

    def test_upstream_headers_omit_authorization_when_there_is_no_key(self):
        # An engine started without --api-key rejects nothing, but
        # sending "Bearer " would be a malformed credential.
        assert "Authorization" not in proxy.upstream_headers("")


class TestStreamDetection:
    @pytest.mark.parametrize(
        "payload,expected",
        [
            ({"stream": True}, True),
            ({"stream": False}, False),
            ({}, False),
            ({"stream": None}, False),
        ],
    )
    def test_stream_flag(self, payload, expected):
        assert proxy.is_streaming_request(payload) is expected


class TestErrorFrames:
    def test_error_frame_is_terminated_so_the_page_stops_waiting(self):
        frame = proxy._error_frame("something broke").decode()
        assert frame.startswith("data: ")
        # Without the [DONE] sentinel the page's reader would sit open
        # until the connection times out.
        assert frame.endswith("data: [DONE]\n\n")
        assert "something broke" in frame

    def test_engine_error_envelope_is_unwrapped(self):
        body = b'{"error": {"message": "model not found"}}'
        described = proxy._describe_upstream_error(404, body)
        assert "model not found" in described
        assert "404" in described

    def test_non_json_error_body_falls_back_to_raw_text(self):
        described = proxy._describe_upstream_error(500, b"<html>oops</html>")
        assert "500" in described
        assert "oops" in described


class _FakeStream:
    """Minimal stand-in for httpx's streaming context manager."""

    def __init__(self, status_code, chunks=(), body=b""):
        self.status_code = status_code
        self._chunks = list(chunks)
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aiter_raw(self):
        for chunk in self._chunks:
            yield chunk

    async def aread(self):
        return self._body


class _FakeClient:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def stream(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self._response


class TestStreamingProxy:
    @pytest.mark.asyncio
    async def test_chunks_are_relayed_verbatim(self):
        chunks = [b'data: {"choices":[]}\n\n', b"data: [DONE]\n\n"]
        client = _FakeClient(_FakeStream(200, chunks))

        received = [
            chunk
            async for chunk in proxy.proxy_streaming(
                client,
                base_url="http://engine.invalid",
                path="/v1/chat/completions",
                payload={"stream": True},
                api_key="k",
            )
        ]

        # Re-serialising would risk changing the SSE framing for no gain.
        assert received == chunks

    @pytest.mark.asyncio
    async def test_upstream_error_becomes_a_terminal_frame(self):
        client = _FakeClient(
            _FakeStream(400, body=b'{"error": {"message": "bad request"}}')
        )

        received = b"".join(
            [
                chunk
                async for chunk in proxy.proxy_streaming(
                    client,
                    base_url="http://engine.invalid",
                    path="/v1/chat/completions",
                    payload={"stream": True},
                    api_key="k",
                )
            ]
        ).decode()

        # By this point the 200 status is already committed, so raising
        # would only truncate the stream and hang the page.
        assert "bad request" in received
        assert received.endswith("data: [DONE]\n\n")

    @pytest.mark.asyncio
    async def test_transport_failure_becomes_a_terminal_frame(self):
        class _Failing:
            def stream(self, *args, **kwargs):
                raise httpx.ConnectError("refused")

        received = b"".join(
            [
                chunk
                async for chunk in proxy.proxy_streaming(
                    _Failing(),
                    base_url="http://engine.invalid",
                    path="/v1/chat/completions",
                    payload={"stream": True},
                    api_key="k",
                )
            ]
        ).decode()

        assert "connection to the engine failed" in received
        assert received.endswith("data: [DONE]\n\n")
