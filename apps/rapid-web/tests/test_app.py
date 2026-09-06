# SPDX-License-Identifier: Apache-2.0
"""Tests for the HTTP surface.

The engine is replaced by a fake so these run without MLX, a model, or a
`rapid-mlx` install. That is the point of driving the CLI as a
subprocess rather than importing ``vllm_mlx``: the seam is mockable.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from rmlx_web import app as app_module
from rmlx_web.app import MAX_AUDIO_BYTES, MAX_IMAGE_BYTES, WebConfig, create_app
from rmlx_web.catalog import CatalogError, ModelEntry, RemovalError
from rmlx_web.downloads import DownloadError, DownloadJob, DownloadState
from rmlx_web.supervisor import ChildState, ChildStatus, ResidencyOutcome

TOKEN = "test-token-value"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
JSON_CT = {"Content-Type": "application/json"}
UPLOAD = {"X-Rapid-Upload": "1"}


class FakeEngine:
    """Stands in for EngineSupervisor / AttachedEngine."""

    def __init__(self, *, state=ChildState.READY, model="fake-model", can_switch=True):
        self._state = state
        self._model = model
        self.can_switch = can_switch
        self.api_key = "engine-side-key"
        self.stopped = False
        self.started = []
        #: Aliases hot-loaded via `residency_load`, in order.
        self.hot_loaded = []
        #: What the next `residency_load` returns. UNSUPPORTED by default,
        #: so every existing test keeps exercising the respawn path.
        self.residency_outcome = ResidencyOutcome.UNSUPPORTED
        self.resident = []

    @property
    def base_url(self):
        return "http://engine.invalid" if self._state is ChildState.READY else None

    def status(self):
        return ChildStatus(
            state=self._state,
            model=self._model,
            port=1234,
            detail="boom" if self._state is ChildState.FAILED else None,
            recent_output=["line one", "line two"],
            resident=list(self.resident),
        )

    async def residency_load(
        self, model, *, modality, size_bytes=None, image_mode=None
    ):
        self.hot_loaded.append((model, modality, size_bytes, image_mode))
        if self.residency_outcome is ResidencyOutcome.LOADED:
            self.resident.append(model)
            return ResidencyOutcome.LOADED, None
        if self.residency_outcome is ResidencyOutcome.REJECTED:
            return ResidencyOutcome.REJECTED, "would exceed the ceiling"
        return ResidencyOutcome.UNSUPPORTED, None

    async def start(self, model, *, modality="text"):
        self.started.append(model)
        self._model = model
        self.resident = [model]

    async def stop(self):
        self.stopped = True


class FakeCatalog:
    """Stands in for ModelCatalog, with no subprocess."""

    def __init__(self, entries=None, error=None, remove_error=None):
        self.entries = (
            entries
            if entries is not None
            else [
                ModelEntry(
                    alias="qwen3.5-9b-4bit",
                    hf_path="mlx-community/Qwen3.5-9B-4bit",
                    size_bytes=5977075377,
                    cached=True,
                    cached_bytes=5977075377,
                ),
                ModelEntry(
                    alias="bonsai-1.7b-2bit",
                    hf_path="prism-ml/Ternary-Bonsai-1.7B-mlx-2bit",
                    size_bytes=495525300,
                    cached=False,
                ),
            ]
        )
        self.error = error
        self.remove_error = remove_error
        self.forced = []
        self.invalidated = 0
        self.removed = []

    async def list_models(self, *, force=False):
        if self.error:
            raise self.error
        self.forced.append(force)
        return self.entries

    async def list_chat_models(self, *, force=False):
        return [e for e in await self.list_models(force=force) if e.kind == "text"]

    async def is_known_chat_alias(self, alias):
        entry = await self.profile(alias)
        return entry is not None and entry.kind == "text"

    async def profile(self, alias):
        if self.error:
            raise self.error
        for entry in self.entries:
            if entry.alias == alias:
                return entry
        return None

    def invalidate_cache(self):
        self.invalidated += 1

    async def remove(self, alias):
        if self.error:
            raise self.error
        if self.remove_error:
            raise self.remove_error
        for entry in self.entries:
            if entry.alias == alias:
                self.removed.append(alias)
                self.invalidated += 1
                return entry.cached_bytes
        raise RemovalError(f"unknown model alias: {alias}")


def build_client(
    engine=None, catalog="default", downloads=None, token=TOKEN, **config_kwargs
):
    if catalog == "default":
        catalog = FakeCatalog()
    config = WebConfig(
        token=token,
        engine=engine or FakeEngine(),
        catalog=catalog,
        downloads=downloads,
        **config_kwargs,
    )
    return TestClient(create_app(config))


def _await_job(client, job_id, timeout=10.0):
    """Poll a render to a terminal state, exactly as the page does."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/images/jobs/{job_id}", headers=AUTH).json()
        if body["state"] != "running":
            return body
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} never finished")


class TestAuthGate:
    def test_index_is_reachable_without_a_token(self):
        # The page is where the user enters the token, so it cannot
        # itself require one.
        with build_client() as client:
            response = client.get("/")
        assert response.status_code == 200
        assert "Rapid-MLX" in response.text

    def test_api_requires_a_token(self):
        with build_client() as client:
            response = client.get("/api/status")
        assert response.status_code == 401
        assert response.json()["error"]["type"] == "unauthorized"

    def test_api_rejects_a_wrong_token(self):
        with build_client() as client:
            response = client.get(
                "/api/status", headers={"Authorization": "Bearer nope"}
            )
        assert response.status_code == 401

    def test_api_accepts_the_right_token(self):
        with build_client() as client:
            response = client.get("/api/status", headers=AUTH)
        assert response.status_code == 200

    def test_cross_site_request_is_refused_before_the_token_is_checked(self):
        with build_client() as client:
            response = client.get(
                "/api/status",
                headers={
                    **AUTH,
                    "Origin": "https://evil.example",
                    "Sec-Fetch-Site": "cross-site",
                },
            )
        # 403, not 401: the token was valid. A page the user has open
        # could have been given it, so the origin check must still bite.
        assert response.status_code == 403
        assert response.json()["error"]["type"] == "origin_refused"

    def test_post_with_a_simple_content_type_is_refused(self):
        with build_client() as client:
            response = client.post(
                "/api/auth",
                headers={**AUTH, "Content-Type": "text/plain"},
                content="{}",
            )
        # text/plain is a CORS "simple" type and would reach us with no
        # preflight from a cross-origin page.
        assert response.status_code == 415
        assert response.json()["error"]["type"] == "unsupported_media_type"

    def test_multipart_without_the_csrf_header_is_refused(self):
        with build_client() as client:
            response = client.post(
                "/api/audio/transcriptions",
                headers=AUTH,
                files={"file": ("recording.wav", b"RIFFfake", "audio/wav")},
            )
        assert response.status_code == 415
        assert response.json()["error"]["type"] == "unsupported_media_type"


class TestSecurityHeaders:
    def test_index_carries_a_restrictive_csp(self):
        with build_client() as client:
            response = client.get("/")
        csp = response.headers["Content-Security-Policy"]
        assert "default-src 'self'" in csp
        assert "frame-ancestors 'none'" in csp
        assert response.headers["X-Content-Type-Options"] == "nosniff"

    def test_media_src_allows_blob_for_synthesised_speech(self):
        with build_client() as client:
            response = client.get("/")
        csp = response.headers["Content-Security-Policy"]
        # Generated speech reaches `<audio>` as an object URL. Without its
        # own directive `media-src` falls back to `default-src 'self'`,
        # which does not cover `blob:` — the element then fails with
        # MediaError 4 and sits at 0:00/0:00, while the identical URL still
        # downloads fine, so the bytes look correct and the audio takes the
        # blame.
        assert "media-src 'self' blob:" in csp

    def test_img_src_allows_revocable_blob_previews(self):
        with build_client() as client:
            csp = client.get("/").headers["Content-Security-Policy"]
        assert "img-src 'self' data: blob:" in csp


class TestIndexIsSelfContained:
    """Every asset the page references must actually be served.

    ``static/`` is a build output (``apps/rapid-web/frontend/``, ``npm run
    build``) that is committed. The shell references hashed files under
    ``/static/assets/``; if one is missing the page renders blank. This runs
    in the Python suite, so a stale artifact is caught even by someone who
    never touches Node.
    """

    _URL_ATTRIBUTES = re.compile(r'\b(?:src|href)\s*=\s*"([^"]*)"', re.IGNORECASE)

    def _served_html(self) -> str:
        with build_client() as client:
            response = client.get("/")
        assert response.status_code == 200
        return response.text

    def _referenced(self) -> list[str]:
        return [
            url
            for url in self._URL_ATTRIBUTES.findall(self._served_html())
            if not url.startswith("data:")
        ]

    def test_the_page_references_its_assets(self):
        referenced = self._referenced()
        assert referenced, "the built page references no assets at all"
        assert all(url.startswith("/static/") for url in referenced), (
            f"expected every asset under /static/, got {referenced}"
        )

    def test_every_referenced_asset_is_served(self):
        with build_client() as client:
            for url in self._referenced():
                response = client.get(url)
                assert response.status_code == 200, f"{url} -> {response.status_code}"

    def test_assets_are_cached_immutably(self):
        with build_client() as client:
            for url in self._referenced():
                cache_control = client.get(url).headers.get("cache-control", "")
                assert "immutable" in cache_control, f"{url}: {cache_control!r}"

    def test_the_shell_revalidates_rather_than_caching(self):
        with build_client() as client:
            response = client.get("/")
        assert response.headers["Cache-Control"] == "no-cache"
        assert response.headers["ETag"]

    def test_a_matching_etag_gets_a_304(self):
        with build_client() as client:
            etag = client.get("/").headers["ETag"]
            repeat = client.get("/", headers={"If-None-Match": etag})
        assert repeat.status_code == 304
        assert not repeat.content

    def test_the_title_is_contiguous(self):
        # ``test_index_is_reachable_without_a_token`` asserts "Rapid-MLX"
        # appears in the body. The rendered wordmark is
        # ``Rapid<span>-MLX</span>``, which is NOT contiguous, so <title> is the
        # only thing satisfying it. Pinned separately so that dependency is
        # visible rather than incidental.
        assert "<title>Rapid-MLX</title>" in self._served_html()


class TestStatus:
    def test_reports_the_loaded_model(self):
        with build_client() as client:
            body = client.get("/api/status", headers=AUTH).json()
        assert body["state"] == "ready"
        assert body["model"] == "fake-model"
        assert body["can_switch"] is True

    def test_log_tail_is_withheld_unless_the_engine_failed(self):
        with build_client() as client:
            body = client.get("/api/status", headers=AUTH).json()
        # The tail can carry filesystem paths; it is only worth the
        # exposure when it explains a failure.
        assert "recent_output" not in body

    def test_log_tail_is_included_on_failure(self):
        engine = FakeEngine(state=ChildState.FAILED)
        with build_client(engine) as client:
            body = client.get("/api/status", headers=AUTH).json()
        assert body["state"] == "failed"
        assert body["recent_output"] == ["line one", "line two"]

    def test_attach_mode_reports_that_switching_is_unavailable(self):
        engine = FakeEngine(can_switch=False)
        with build_client(engine) as client:
            body = client.get("/api/status", headers=AUTH).json()
        assert body["can_switch"] is False


class TestChatCompletions:
    def test_returns_503_while_the_engine_is_still_loading(self):
        engine = FakeEngine(state=ChildState.STARTING)
        with build_client(engine) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={**AUTH, **JSON_CT},
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
        # 503, not 502: nothing is broken, it is not there yet. The page
        # retries on 503.
        assert response.status_code == 503
        assert response.json()["error"]["type"] == "engine_unavailable"
        assert "loading" in response.json()["error"]["message"]

    def test_failure_detail_is_surfaced(self):
        engine = FakeEngine(state=ChildState.FAILED)
        with build_client(engine) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={**AUTH, **JSON_CT},
                json={"messages": []},
            )
        assert response.status_code == 503
        assert "boom" in response.json()["error"]["message"]

    def test_malformed_json_body_is_rejected(self):
        with build_client() as client:
            response = client.post(
                "/v1/chat/completions",
                headers={**AUTH, **JSON_CT},
                content="not json",
            )
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "invalid_json"

    def test_non_object_json_body_is_rejected(self):
        with build_client() as client:
            response = client.post(
                "/v1/chat/completions",
                headers={**AUTH, **JSON_CT},
                content="[1, 2, 3]",
            )
        assert response.status_code == 400


class TestProxyForwarding:
    """The proxy must swap credentials, not forward them."""

    def test_unary_request_carries_the_engine_key_not_the_web_token(self, monkeypatch):
        captured = {}

        async def fake_post(self, url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers", {})
            captured["json"] = kwargs.get("json")
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "hello"}}]},
                request=httpx.Request("POST", url),
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        with build_client() as client:
            response = client.post(
                "/v1/chat/completions",
                headers={**AUTH, **JSON_CT},
                json={"messages": [{"role": "user", "content": "hi"}]},
            )

        assert response.status_code == 200
        # Forwarding the web token would leak it into the engine's log
        # and would fail the engine's own auth besides.
        assert captured["headers"]["Authorization"] == "Bearer engine-side-key"
        assert TOKEN not in json.dumps(dict(captured["headers"]))
        assert captured["url"].endswith("/v1/chat/completions")

    def test_a_tools_array_reaches_the_engine_untouched(self, monkeypatch):
        """The chat route is a pass-through, so tool calling needs no
        server-side translation — this is what pins that."""
        captured = {}

        async def fake_post(self, url, **kwargs):
            captured["json"] = kwargs.get("json")
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "hi"}}]},
                request=httpx.Request("POST", url),
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        tools_payload = [
            {
                "type": "function",
                "function": {"name": "weather", "description": "d", "parameters": {}},
            }
        ]
        with build_client() as client:
            client.post(
                "/v1/chat/completions",
                headers={**AUTH, **JSON_CT},
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": tools_payload,
                    "tool_choice": "auto",
                },
            )

        assert captured["json"]["tools"] == tools_payload
        assert captured["json"]["tool_choice"] == "auto"

    def test_a_tool_result_turn_reaches_the_engine(self, monkeypatch):
        captured = {}

        async def fake_post(self, url, **kwargs):
            captured["json"] = kwargs.get("json")
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "hi"}}]},
                request=httpx.Request("POST", url),
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        messages = [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": "18C", "tool_call_id": "call_1"},
        ]
        with build_client() as client:
            client.post(
                "/v1/chat/completions",
                headers={**AUTH, **JSON_CT},
                json={"messages": messages},
            )

        assert captured["json"]["messages"] == messages


class TestToolRoutes:
    def test_lists_the_tools_and_which_need_approval(self):
        with build_client() as client:
            response = client.get("/api/tools", headers=AUTH)

        assert response.status_code == 200
        body = response.json()
        assert {t["function"]["name"] for t in body["tools"]} == {
            "weather",
            "web_search",
            "browse",
        }
        assert body["approval_required"] == ["browse"]

    def test_refuses_a_call_for_a_tool_that_was_not_advertised(self):
        with build_client() as client:
            response = client.post(
                "/api/tools/call",
                headers={**AUTH, **JSON_CT},
                json={"name": "browse", "arguments": "{}", "advertised": ["weather"]},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["is_error"]
        assert "isn't available" in body["content"]

    def test_refuses_browsing_a_loopback_address(self):
        with build_client() as client:
            response = client.post(
                "/api/tools/call",
                headers={**AUTH, **JSON_CT},
                json={
                    "name": "browse",
                    "arguments": '{"url": "http://127.0.0.1:8080/admin"}',
                    "advertised": ["browse"],
                    "approved_origins": ["http://127.0.0.1:8080"],
                },
            )

        body = response.json()
        assert body["is_error"]
        assert "private/loopback" in body["content"]

    def test_rejects_a_body_missing_the_advertised_list(self):
        with build_client() as client:
            response = client.post(
                "/api/tools/call",
                headers={**AUTH, **JSON_CT},
                json={"name": "weather", "arguments": "{}"},
            )

        assert response.status_code == 400

    def test_rejects_non_string_arguments(self):
        with build_client() as client:
            response = client.post(
                "/api/tools/call",
                headers={**AUTH, **JSON_CT},
                json={"name": "weather", "arguments": {}, "advertised": ["weather"]},
            )

        assert response.status_code == 400

    def test_requires_the_bearer(self):
        with build_client() as client:
            assert client.get("/api/tools").status_code == 401


class TestImageJobs:
    """Renders run as jobs, not as the request that asked for them.

    The engine answers only once the whole image is finished, so relaying
    it inline held a connection open with no bytes flowing for minutes and
    Cloudflare cut it at 100 s with a 524. The POST starts a job and the
    page polls for the result.
    """

    def test_a_generation_starts_a_job_and_answers_immediately(self, monkeypatch):
        captured = {}
        release = threading.Event()

        async def fake_post(self, url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers", {})
            captured["json"] = kwargs.get("json")
            # Held so the POST provably answers before the render does.
            await asyncio.get_running_loop().run_in_executor(None, release.wait)
            return httpx.Response(
                200,
                json={"created": 1, "data": [{"b64_json": "aGk="}], "cancelled": False},
                request=httpx.Request("POST", url),
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        with build_client() as client:
            started = client.post(
                "/api/images/jobs",
                headers={**AUTH, **JSON_CT},
                json={"prompt": "a cat", "size": "512x512", "model": "flux2-klein-4b"},
            )
            assert started.status_code == 200
            job = started.json()
            assert job["state"] == "running"
            assert job["b64_json"] is None

            release.set()
            body = _await_job(client, job["id"])

        assert body["state"] == "done"
        assert body["b64_json"] == "aGk="
        assert captured["headers"]["Authorization"] == "Bearer engine-side-key"
        assert captured["url"].endswith("/v1/images/generations")
        assert captured["json"]["prompt"] == "a cat"
        assert captured["json"]["size"] == "512x512"
        assert captured["json"]["model"] == "flux2-klein-4b"

    def test_an_edit_is_rebuilt_as_multipart(self, monkeypatch):
        captured = {}

        async def fake_post(self, url, **kwargs):
            captured["url"] = url
            captured["files"] = kwargs.get("files")
            captured["data"] = kwargs.get("data")
            captured["headers"] = kwargs.get("headers", {})
            return httpx.Response(
                200,
                json={"created": 1, "data": [{"b64_json": "aGk="}], "cancelled": False},
                request=httpx.Request("POST", url),
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        with build_client() as client:
            started = client.post(
                "/api/images/jobs",
                headers={**AUTH, **UPLOAD},
                files={"image": ("input.png", b"\x89PNGfake", "image/png")},
                data={"prompt": "make it night", "model": "flux2-klein-4b"},
            )
            assert started.status_code == 200
            body = _await_job(client, started.json()["id"])

        assert body["b64_json"] == "aGk="
        assert captured["url"].endswith("/v1/images/edits")
        # The engine's field is `image`, not the `file` transcription uses.
        assert captured["files"]["image"][1] == b"\x89PNGfake"
        assert captured["data"]["prompt"] == "make it night"
        assert captured["data"]["model"] == "flux2-klein-4b"
        # `size` is deliberately absent: the edit backends derive their canvas
        # from the input image, and the engine discards it anyway.
        assert "size" not in captured["data"]
        # No Content-Type of our own, or httpx cannot set the boundary.
        assert "Content-Type" not in captured["headers"]
        assert captured["headers"]["Authorization"] == "Bearer engine-side-key"

    def test_a_running_job_reports_the_engines_denoise_counter(self, monkeypatch):
        """One poll carries both progress and the result.

        A render therefore occupies a single connection: the separate
        progress feed it replaced was a second one held open beside it.
        """
        release = threading.Event()

        async def fake_post(self, url, **kwargs):
            await asyncio.get_running_loop().run_in_executor(None, release.wait)
            return httpx.Response(
                200, json={"data": []}, request=httpx.Request("POST", url)
            )

        async def fake_get(self, url, **kwargs):
            return httpx.Response(
                200,
                json={"running": True, "step": 3, "total": 8},
                request=httpx.Request("GET", url),
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
        monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

        with build_client() as client:
            job_id = client.post(
                "/api/images/jobs",
                headers={**AUTH, **JSON_CT},
                json={"prompt": "a cat", "model": "flux2-klein-4b"},
            ).json()["id"]

            body = client.get(f"/api/images/jobs/{job_id}", headers=AUTH).json()
            assert body["state"] == "running"
            assert (body["step"], body["total"]) == (3, 8)

            release.set()
            _await_job(client, job_id)

    def test_a_dropped_progress_read_is_not_a_render_failure(self, monkeypatch):
        release = threading.Event()

        async def fake_post(self, url, **kwargs):
            await asyncio.get_running_loop().run_in_executor(None, release.wait)
            return httpx.Response(
                200, json={"data": []}, request=httpx.Request("POST", url)
            )

        async def fake_get(self, url, **kwargs):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
        monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

        with build_client() as client:
            job_id = client.post(
                "/api/images/jobs",
                headers={**AUTH, **JSON_CT},
                json={"prompt": "a cat"},
            ).json()["id"]

            body = client.get(f"/api/images/jobs/{job_id}", headers=AUTH).json()
            # Still running, with no steps yet — the job's own state is what
            # reports a failure, not the progress read.
            assert body["state"] == "running"
            assert (body["step"], body["total"]) == (0, 0)

            release.set()
            _await_job(client, job_id)

    def test_an_engine_error_lands_on_the_job_not_the_start(self, monkeypatch):
        async def fake_post(self, url, **kwargs):
            return httpx.Response(
                409,
                json={
                    "error": {
                        "message": "no image model",
                        "type": "image_model_not_loaded",
                    }
                },
                request=httpx.Request("POST", url),
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        with build_client() as client:
            started = client.post(
                "/api/images/jobs",
                headers={**AUTH, **JSON_CT},
                json={"prompt": "a cat"},
            )
            # The start succeeded; the render is what failed.
            assert started.status_code == 200
            body = _await_job(client, started.json()["id"])

        assert body["state"] == "failed"
        assert body["error"]["type"] == "image_model_not_loaded"
        assert body["error"]["status"] == 409

    def test_a_transport_failure_lands_on_the_job(self, monkeypatch):
        async def fake_post(self, url, **kwargs):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        with build_client() as client:
            started = client.post(
                "/api/images/jobs",
                headers={**AUTH, **JSON_CT},
                json={"prompt": "a cat"},
            )
            body = _await_job(client, started.json()["id"])

        assert body["state"] == "failed"
        assert body["error"]["type"] == "engine_transport"

    def test_an_unknown_job_is_404(self):
        with build_client() as client:
            response = client.get("/api/images/jobs/nope", headers=AUTH)
        # Only the last job is kept. Reporting a vanished one as idle would
        # leave the page waiting on it forever.
        assert response.status_code == 404
        assert response.json()["error"]["type"] == "unknown_image_job"

    def test_a_second_render_is_refused_while_one_runs(self, monkeypatch):
        release = threading.Event()

        async def fake_post(self, url, **kwargs):
            await asyncio.get_running_loop().run_in_executor(None, release.wait)
            return httpx.Response(
                200, json={"data": []}, request=httpx.Request("POST", url)
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        with build_client() as client:
            first = client.post(
                "/api/images/jobs",
                headers={**AUTH, **JSON_CT},
                json={"prompt": "a cat"},
            )
            second = client.post(
                "/api/images/jobs",
                headers={**AUTH, **JSON_CT},
                json={"prompt": "a dog"},
            )
            assert second.status_code == 409
            assert second.json()["error"]["type"] == "image_busy"

            release.set()
            _await_job(client, first.json()["id"])

    def test_a_switch_is_refused_from_the_instant_a_job_starts(self, monkeypatch):
        """The count must be taken before the job's task is scheduled.

        A task does not run until the next tick, and the POST's own response
        is what frees the loop to service the next request — so counting
        inside the task let a load arriving immediately after slip through
        and restart the engine under the render.
        """
        release = threading.Event()

        async def fake_post(self, url, **kwargs):
            await asyncio.get_running_loop().run_in_executor(None, release.wait)
            return httpx.Response(
                200, json={"data": []}, request=httpx.Request("POST", url)
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        engine = FakeEngine()
        with build_client(engine) as client:
            job_id = client.post(
                "/api/images/jobs",
                headers={**AUTH, **JSON_CT},
                json={"prompt": "a cat"},
            ).json()["id"]

            blocked = client.post(
                "/api/models/load",
                headers={**AUTH, **JSON_CT},
                json={"model": "bonsai-1.7b-2bit"},
            )
            assert blocked.status_code == 409
            assert blocked.json()["error"]["type"] == "busy_streaming"
            assert engine.started == []

            release.set()
            _await_job(client, job_id)

    def test_an_empty_prompt_is_refused(self):
        with build_client() as client:
            response = client.post(
                "/api/images/jobs",
                headers={**AUTH, **UPLOAD},
                files={"image": ("input.png", b"x", "image/png")},
                data={"prompt": "  "},
            )
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "invalid_body"

    def test_an_edit_without_a_file_is_refused(self):
        with build_client() as client:
            response = client.post(
                "/api/images/jobs",
                headers={**AUTH, **UPLOAD},
                files={"not_image": ("x.txt", b"x", "text/plain")},
                data={"prompt": "make it night"},
            )
        assert response.status_code == 400

    def test_an_oversize_image_is_refused_here(self):
        with build_client() as client:
            response = client.post(
                "/api/images/jobs",
                headers={**AUTH, **UPLOAD},
                files={
                    "image": ("input.png", b"x" * (MAX_IMAGE_BYTES + 1), "image/png")
                },
                data={"prompt": "make it night"},
            )
        # Refused before the relay rather than after another hop.
        assert response.status_code == 413

    def test_starting_before_a_model_is_loaded_is_503(self):
        engine = FakeEngine(state=ChildState.STARTING)
        with build_client(engine) as client:
            response = client.post(
                "/api/images/jobs",
                headers={**AUTH, **JSON_CT},
                json={"prompt": "a cat"},
            )
        assert response.status_code == 503
        assert response.json()["error"]["type"] == "engine_unavailable"

    @pytest.mark.asyncio
    async def test_a_job_counts_as_a_stream_so_a_switch_is_refused(self, monkeypatch):
        """A render is minutes of GPU work with no resume.

        Switching restarts the engine, so a load arriving mid-render must
        refuse exactly as it does mid-chat — and now for the JOB's whole
        life, not just while a request is being relayed. Patched at
        ``proxy_unary`` rather than on ``httpx.AsyncClient``: the async
        test client is itself an ``AsyncClient``, so patching the class
        would replace the test's own transport.
        """
        release = asyncio.Event()
        started = asyncio.Event()

        async def fake_unary(client, **kwargs):
            started.set()
            await release.wait()
            return httpx.Response(
                200,
                json={"data": []},
                request=httpx.Request("POST", "http://engine.invalid"),
            )

        monkeypatch.setattr(app_module.proxy, "proxy_unary", fake_unary)

        engine = FakeEngine()
        app = create_app(WebConfig(token=TOKEN, engine=engine, catalog=FakeCatalog()))
        app.state.http = httpx.AsyncClient()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            await client.post(
                "/api/images/jobs",
                headers={**AUTH, **JSON_CT},
                json={"prompt": "a cat"},
            )
            await asyncio.wait_for(started.wait(), timeout=10)

            blocked = await client.post(
                "/api/models/load",
                headers={**AUTH, **JSON_CT},
                json={"model": "bonsai-1.7b-2bit"},
            )
            assert blocked.status_code == 409
            assert blocked.json()["error"]["type"] == "busy_streaming"
            # Restarting the engine here would destroy the render.
            assert engine.started == []

            release.set()

        await app.state.http.aclose()

    def test_cancel_sends_the_model_in_the_query_string(self, monkeypatch):
        captured = {}

        async def fake_post(self, url, **kwargs):
            captured["url"] = url
            captured["params"] = kwargs.get("params")
            captured["json"] = kwargs.get("json")
            return httpx.Response(
                200, json={"ok": True}, request=httpx.Request("POST", url)
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        with build_client() as client:
            response = client.post(
                "/api/images/cancel",
                headers={**AUTH, **JSON_CT},
                json={"model": "flux2-klein-4b"},
            )

        assert response.status_code == 200
        # The engine reads only the query here. A JSON body would be
        # discarded silently, which reads as a call that cancels nothing.
        assert captured["params"] == {"model": "flux2-klein-4b"}
        assert captured["json"] is None

    def test_image_routes_require_a_token(self):
        with build_client() as client:
            assert client.get("/api/images/jobs/anything").status_code == 401
            assert (
                client.post(
                    "/api/images/jobs", headers=JSON_CT, json={"prompt": "x"}
                ).status_code
                == 401
            )


class TestAudio:
    """The audio lane rides on whatever model is loaded.

    The child is spawned with ``--enable-audio``, and the engine's gate
    short-circuits on that flag before it looks at the model — so speech
    works while a CHAT model is loaded, with no switch.
    """

    def test_voices_are_relayed(self, monkeypatch):
        async def fake_get(self, url, **kwargs):
            return httpx.Response(
                200,
                json={"voices": ["af_heart", "am_adam"]},
                request=httpx.Request("GET", url),
            )

        monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

        with build_client() as client:
            body = client.get("/api/audio/voices?model=kokoro", headers=AUTH).json()

        assert body["voices"] == ["af_heart", "am_adam"]

    def test_speech_returns_audio_bytes_not_json(self, monkeypatch):
        async def fake_post(self, url, **kwargs):
            return httpx.Response(
                200,
                content=b"RIFF....WAVEfmt ",
                headers={"Content-Type": "audio/wav"},
                request=httpx.Request("POST", url),
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        with build_client() as client:
            response = client.post(
                "/api/audio/speech",
                headers={**AUTH, **JSON_CT},
                json={"model": "kokoro", "input": "hello", "voice": "af_heart"},
            )

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.content.startswith(b"RIFF")

    def test_a_speech_failure_still_arrives_as_json(self, monkeypatch):
        async def fake_post(self, url, **kwargs):
            return httpx.Response(
                503,
                json={"error": {"message": "espeak-ng missing", "type": "api_error"}},
                request=httpx.Request("POST", url),
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        with build_client() as client:
            response = client.post(
                "/api/audio/speech",
                headers={**AUTH, **JSON_CT},
                json={"input": "hello"},
            )

        # Branching on the STATUS, not the content type: the engine's
        # actionable "install X" message must reach the page rather than
        # being handed over as unplayable bytes.
        assert response.status_code == 503
        assert "espeak-ng" in response.json()["error"]["message"]

    def test_transcription_relays_a_bounded_multipart_upload(self, monkeypatch):
        captured = {}

        async def fake_post(self, url, **kwargs):
            captured["url"] = url
            captured["files"] = kwargs.get("files")
            captured["data"] = kwargs.get("data")
            captured["headers"] = kwargs.get("headers", {})
            return httpx.Response(
                200,
                json={"text": "hello there", "language": "en", "duration": 1.2},
                request=httpx.Request("POST", url),
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        with build_client() as client:
            response = client.post(
                "/api/audio/transcriptions",
                headers={**AUTH, **UPLOAD},
                files={"file": ("recording.wav", b"RIFFfake", "audio/wav")},
                data={"model": "whisper-large-v3-turbo"},
            )

        assert response.status_code == 200
        assert response.json()["text"] == "hello there"
        assert captured["files"]["file"][1] == b"RIFFfake"
        assert captured["data"]["model"] == "whisper-large-v3-turbo"
        # No Content-Type of our own, or httpx cannot set the boundary.
        assert "Content-Type" not in captured["headers"]
        assert captured["headers"]["Authorization"] == "Bearer engine-side-key"

    def test_json_audio_is_refused(self):
        with build_client() as client:
            response = client.post(
                "/api/audio/transcriptions",
                headers={**AUTH, **JSON_CT},
                json={"audio": "legacy-base64"},
            )
        assert response.status_code == 415
        assert response.json()["error"]["type"] == "unsupported_media_type"

    def test_an_empty_recording_is_refused(self):
        with build_client() as client:
            response = client.post(
                "/api/audio/transcriptions",
                headers={**AUTH, **UPLOAD},
                files={"file": ("recording.wav", b"", "audio/wav")},
            )
        assert response.status_code == 400

    def test_an_oversize_recording_is_refused_here(self):
        with build_client() as client:
            response = client.post(
                "/api/audio/transcriptions",
                headers={**AUTH, **UPLOAD},
                files={
                    "file": ("recording.wav", b"x" * (MAX_AUDIO_BYTES + 1), "audio/wav")
                },
            )
        # Refused before the relay rather than after another hop.
        assert response.status_code == 413

    def test_a_caller_supplied_filename_cannot_be_arbitrary(self, monkeypatch):
        captured = {}

        async def fake_post(self, url, **kwargs):
            captured["files"] = kwargs.get("files")
            return httpx.Response(
                200, json={"text": ""}, request=httpx.Request("POST", url)
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        with build_client() as client:
            client.post(
                "/api/audio/transcriptions",
                headers={**AUTH, **UPLOAD},
                files={"file": ('evil"; name="model', b"x", "audio/wav")},
            )

        # The name is advisory (the engine spools to a .wav temp file
        # regardless), so a header-splitting attempt falls back rather than
        # being sanitised into something almost-right.
        assert captured["files"]["file"][0] == "recording.wav"

    def test_audio_before_a_model_is_loaded_is_503(self):
        engine = FakeEngine(state=ChildState.STARTING)
        with build_client(engine) as client:
            speech = client.post(
                "/api/audio/speech", headers={**AUTH, **JSON_CT}, json={"input": "x"}
            )
            voices = client.get("/api/audio/voices", headers=AUTH)
        assert speech.status_code == 503
        assert voices.status_code == 503

    def test_audio_routes_require_a_token(self):
        with build_client() as client:
            assert client.get("/api/audio/voices").status_code == 401
            assert (
                client.post(
                    "/api/audio/speech", headers=JSON_CT, json={"input": "x"}
                ).status_code
                == 401
            )


class TestListModels:
    def test_returns_entries_with_the_loaded_alias(self):
        with build_client() as client:
            body = client.get("/api/models", headers=AUTH).json()

        assert [m["alias"] for m in body["models"]] == [
            "qwen3.5-9b-4bit",
            "bonsai-1.7b-2bit",
        ]
        assert body["loaded"] == "fake-model"
        assert body["can_switch"] is True

    def test_cached_state_is_exposed_per_entry(self):
        with build_client() as client:
            models = client.get("/api/models", headers=AUTH).json()["models"]

        by_alias = {m["alias"]: m for m in models}
        assert by_alias["qwen3.5-9b-4bit"]["cached"] is True
        assert by_alias["bonsai-1.7b-2bit"]["cached"] is False

    def test_refresh_query_forces_a_rescan(self):
        catalog = FakeCatalog()
        with build_client(catalog=catalog) as client:
            client.get("/api/models", headers=AUTH)
            client.get("/api/models?refresh=true", headers=AUTH)

        assert catalog.forced == [False, True]

    def test_attach_mode_reports_the_catalog_is_unavailable(self):
        engine = FakeEngine(can_switch=False)
        with build_client(engine, catalog=None) as client:
            response = client.get("/api/models", headers=AUTH)

        assert response.status_code == 501
        assert response.json()["error"]["type"] == "catalog_unavailable"

    def test_catalog_failure_is_surfaced_as_503(self):
        catalog = FakeCatalog(error=CatalogError("rapid-mlx not found"))
        with build_client(catalog=catalog) as client:
            response = client.get("/api/models", headers=AUTH)

        assert response.status_code == 503
        assert "rapid-mlx not found" in response.json()["error"]["message"]

    def test_listing_requires_a_token(self):
        with build_client() as client:
            assert client.get("/api/models").status_code == 401


class TestModelKinds:
    """Image and audio rows reach the picker; only some can be loaded."""

    @staticmethod
    def _catalog():
        return FakeCatalog(
            entries=[
                ModelEntry(
                    alias="qwen3.5-9b-4bit",
                    hf_path="mlx-community/Qwen3.5-9B-4bit",
                    size_bytes=5977075377,
                    cached=True,
                    cached_bytes=5977075377,
                ),
                ModelEntry(
                    alias="flux2-klein-4b",
                    hf_path="Runpod/FLUX.2-klein-4B-mflux-4bit",
                    size_bytes=4619695783,
                    cached=True,
                    kind="image",
                    cached_bytes=4619695783,
                ),
                ModelEntry(
                    alias="whisper-large-v3",
                    hf_path="mlx-community/whisper-large-v3",
                    size_bytes=None,
                    cached=False,
                    kind="audio",
                    audio_kind="stt",
                    family="whisper",
                ),
            ]
        )

    def test_every_kind_is_listed_with_its_tag(self):
        with build_client(catalog=self._catalog()) as client:
            models = client.get("/api/models", headers=AUTH).json()["models"]

        by_alias = {m["alias"]: m for m in models}
        assert by_alias["qwen3.5-9b-4bit"]["kind"] == "text"
        assert by_alias["flux2-klein-4b"]["kind"] == "image"
        assert by_alias["whisper-large-v3"]["kind"] == "audio"

    def test_loadable_is_reported_per_entry(self):
        with build_client(catalog=self._catalog()) as client:
            models = client.get("/api/models", headers=AUTH).json()["models"]

        by_alias = {m["alias"]: m for m in models}
        assert by_alias["flux2-klein-4b"]["loadable"] is True
        # Audio too: the CLI has a dedicated audio-serve fork.
        assert by_alias["whisper-large-v3"]["loadable"] is True

    def test_an_image_model_can_be_loaded(self):
        engine = FakeEngine()
        with build_client(engine, catalog=self._catalog()) as client:
            response = client.post(
                "/api/models/load",
                headers={**AUTH, **JSON_CT},
                json={"model": "flux2-klein-4b"},
            )

        assert response.status_code == 200
        assert response.json()["state"] == "starting"

    def test_an_audio_model_can_be_loaded(self):
        engine = FakeEngine()
        with build_client(engine, catalog=self._catalog()) as client:
            response = client.post(
                "/api/models/load",
                headers={**AUTH, **JSON_CT},
                json={"model": "whisper-large-v3"},
            )

        # `serve <audio-alias>` boots in audio mode and reports ready, so
        # refusing here left the Audio page with no way to make speech work
        # on an idle engine — it could only say "start something else".
        assert response.status_code == 200
        assert response.json()["state"] == "starting"


class TestLoadModel:
    def test_switches_to_a_known_alias(self):
        engine = FakeEngine()
        with build_client(engine) as client:
            response = client.post(
                "/api/models/load",
                headers={**AUTH, **JSON_CT},
                json={"model": "bonsai-1.7b-2bit"},
            )

        assert response.status_code == 200
        body = response.json()
        # Answers immediately with "starting" rather than awaiting the
        # load: a real load takes minutes, far past any phone browser's
        # fetch timeout. The page polls /api/status for the outcome.
        assert body["state"] == "starting"
        assert body["model"] == "bonsai-1.7b-2bit"

    def test_unknown_alias_is_rejected(self):
        with build_client() as client:
            response = client.post(
                "/api/models/load",
                headers={**AUTH, **JSON_CT},
                json={"model": "no-such-model"},
            )

        assert response.status_code == 404
        assert response.json()["error"]["type"] == "unknown_model"

    def test_arbitrary_repo_is_rejected(self):
        with build_client() as client:
            response = client.post(
                "/api/models/load",
                headers={**AUTH, **JSON_CT},
                json={"model": "attacker/arbitrary-repo"},
            )

        # Passing this through would hand a remote caller a
        # general-purpose fetch primitive rather than a model picker.
        assert response.status_code == 404

    def test_reloading_the_current_model_is_a_no_op(self):
        engine = FakeEngine(model="qwen3.5-9b-4bit")
        with build_client(engine) as client:
            response = client.post(
                "/api/models/load",
                headers={**AUTH, **JSON_CT},
                json={"model": "qwen3.5-9b-4bit"},
            )

        assert response.status_code == 200
        assert response.json()["state"] == "ready"
        # Restarting would cost minutes of reload for no change, and a
        # double-tap on a phone list is easy.
        assert engine.started == []

    def test_switching_while_a_model_is_loading_is_refused(self):
        engine = FakeEngine(state=ChildState.STARTING, model="slow-model")
        with build_client(engine) as client:
            response = client.post(
                "/api/models/load",
                headers={**AUTH, **JSON_CT},
                json={"model": "bonsai-1.7b-2bit"},
            )

        assert response.status_code == 409
        assert response.json()["error"]["type"] == "busy_loading"
        assert engine.started == []

    def test_attach_mode_refuses_switching(self):
        engine = FakeEngine(can_switch=False)
        with build_client(engine, catalog=None) as client:
            response = client.post(
                "/api/models/load",
                headers={**AUTH, **JSON_CT},
                json={"model": "bonsai-1.7b-2bit"},
            )

        assert response.status_code == 409
        assert response.json()["error"]["type"] == "switch_unavailable"

    @pytest.mark.parametrize(
        "body", [{}, {"model": ""}, {"model": "   "}, {"model": 7}]
    )
    def test_invalid_bodies_are_rejected(self, body):
        with build_client() as client:
            response = client.post(
                "/api/models/load", headers={**AUTH, **JSON_CT}, json=body
            )

        assert response.status_code == 400

    def test_switching_requires_a_token(self):
        with build_client() as client:
            response = client.post(
                "/api/models/load",
                headers=JSON_CT,
                json={"model": "bonsai-1.7b-2bit"},
            )

        assert response.status_code == 401


class TestSwitchBlockedByActiveStream:
    """Switching mid-stream would kill someone else's generation.

    Two things these tests must avoid, both learned the hard way:

    * **Do not use ``TestClient``.** It drives the app through a single
      portal thread, so a held-open stream and a concurrent synchronous
      request deadlock instead of exercising the guard.
    * **Do not monkeypatch ``httpx.AsyncClient``.** The async test client
      is itself an ``AsyncClient``, so patching the class replaces the
      test's own transport and the request never reaches the app.

    So the seam patched here is ``proxy.proxy_streaming``. The tracker
    wrapping lives outside it, so it is still the real code under test.
    """

    @staticmethod
    def _client(app):
        # ASGITransport does NOT run the lifespan, so the shared client
        # the app opens at startup is absent. The streaming handler reads
        # it before deciding anything, so it has to exist even though the
        # patched proxy never uses it.
        app.state.http = httpx.AsyncClient()
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        )

    @pytest.mark.asyncio
    async def test_load_is_refused_while_a_chat_stream_is_relaying(self, monkeypatch):
        release = asyncio.Event()
        first_chunk = asyncio.Event()

        async def fake_streaming(client, **kwargs):
            yield b'data: {"choices":[]}\n\n'
            first_chunk.set()
            # Hold the relay open so the switch attempt below genuinely
            # overlaps with it.
            await release.wait()
            yield b"data: [DONE]\n\n"

        monkeypatch.setattr(app_module.proxy, "proxy_streaming", fake_streaming)

        engine = FakeEngine()
        app = create_app(WebConfig(token=TOKEN, engine=engine, catalog=FakeCatalog()))

        async with self._client(app) as client:
            # The stream runs as its own task rather than in a nested
            # ``async with``: ASGITransport buffers the whole response
            # before returning, so awaiting it inline would block until
            # the stream ends and the overlap under test would never
            # happen.
            relay = asyncio.create_task(
                client.post(
                    "/v1/chat/completions",
                    headers={**AUTH, **JSON_CT},
                    json={"messages": [], "stream": True},
                )
            )
            await asyncio.wait_for(first_chunk.wait(), timeout=10)

            response = await client.post(
                "/api/models/load",
                headers={**AUTH, **JSON_CT},
                json={"model": "bonsai-1.7b-2bit"},
            )
            assert response.status_code == 409
            assert response.json()["error"]["type"] == "busy_streaming"
            # The engine must be untouched: restarting it here would
            # destroy the generation still being relayed.
            assert engine.started == []

            release.set()
            await asyncio.wait_for(relay, timeout=10)

    @pytest.mark.asyncio
    async def test_the_counter_is_released_after_the_stream_finishes(self, monkeypatch):
        async def fake_streaming(client, **kwargs):
            yield b"data: [DONE]\n\n"

        monkeypatch.setattr(app_module.proxy, "proxy_streaming", fake_streaming)

        app = create_app(
            WebConfig(token=TOKEN, engine=FakeEngine(), catalog=FakeCatalog())
        )

        async with self._client(app) as client:
            await client.post(
                "/v1/chat/completions",
                headers={**AUTH, **JSON_CT},
                json={"messages": [], "stream": True},
            )

            # A leaked count would make switching impossible for the rest
            # of the session, with no way for the user to clear it.
            response = await client.post(
                "/api/models/load",
                headers={**AUTH, **JSON_CT},
                json={"model": "bonsai-1.7b-2bit"},
            )
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_the_counter_is_released_when_the_stream_errors(self, monkeypatch):
        async def fake_streaming(client, **kwargs):
            yield b'data: {"choices":[]}\n\n'
            raise RuntimeError("upstream vanished")

        monkeypatch.setattr(app_module.proxy, "proxy_streaming", fake_streaming)

        app = create_app(
            WebConfig(token=TOKEN, engine=FakeEngine(), catalog=FakeCatalog())
        )

        async with self._client(app) as client:
            with contextlib.suppress(Exception):
                await client.post(
                    "/v1/chat/completions",
                    headers={**AUTH, **JSON_CT},
                    json={"messages": [], "stream": True},
                )

            # The decrement has to survive a mid-stream failure too, or a
            # single dropped upstream permanently wedges switching for
            # the rest of the session.
            response = await client.post(
                "/api/models/load",
                headers={**AUTH, **JSON_CT},
                json={"model": "bonsai-1.7b-2bit"},
            )
            assert response.status_code == 200


class TestAtomicModelTransition:
    @pytest.mark.asyncio
    async def test_load_is_single_flight_and_blocks_new_chat(self, monkeypatch):
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def blocked_switch(config, alias, entry):
            calls.append(alias)
            entered.set()
            await release.wait()

        async def forbidden_unary(*args, **kwargs):
            pytest.fail("chat must not reach the old engine during a transition")

        monkeypatch.setattr(app_module, "_switch", blocked_switch)
        monkeypatch.setattr(app_module.proxy, "proxy_unary", forbidden_unary)

        app = create_app(
            WebConfig(token=TOKEN, engine=FakeEngine(), catalog=FakeCatalog())
        )
        app.state.http = httpx.AsyncClient()
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        )
        try:
            first = await client.post(
                "/api/models/load",
                headers={**AUTH, **JSON_CT},
                json={"model": "bonsai-1.7b-2bit"},
            )
            await asyncio.wait_for(entered.wait(), timeout=10)

            duplicate = await client.post(
                "/api/models/load",
                headers={**AUTH, **JSON_CT},
                json={"model": "bonsai-1.7b-2bit"},
            )
            conflict = await client.post(
                "/api/models/load",
                headers={**AUTH, **JSON_CT},
                json={"model": "qwen3.5-9b-4bit"},
            )
            status = await client.get("/api/status", headers=AUTH)
            chat = await client.post(
                "/v1/chat/completions",
                headers={**AUTH, **JSON_CT},
                json={"messages": [], "stream": False},
            )

            assert first.status_code == 200
            assert duplicate.status_code == 200
            assert duplicate.json()["state"] == "starting"
            assert conflict.status_code == 409
            assert conflict.json()["error"]["type"] == "busy_loading"
            assert status.json()["state"] == "starting"
            assert status.json()["model"] == "bonsai-1.7b-2bit"
            assert chat.status_code == 503
            assert calls == ["bonsai-1.7b-2bit"]
        finally:
            release.set()
            await app.state.lifecycle.shutdown()
            await client.aclose()
            await app.state.http.aclose()


class FakeDownloads:
    """Stands in for DownloadManager, with no subprocess."""

    def __init__(self, *, job=None, start_error=None, cancels=True):
        self._job = job
        self.start_error = start_error
        self.started = []
        self.cancelled = 0
        self._cancels = cancels

    @property
    def job(self):
        return self._job

    def is_running(self):
        return self._job is not None and self._job.state is DownloadState.RUNNING

    async def start(self, alias, *, total_bytes):
        if self.start_error:
            raise self.start_error
        self.started.append((alias, total_bytes))
        self._job = DownloadJob(alias=alias, total_bytes=total_bytes)
        return self._job

    async def cancel(self):
        self.cancelled += 1
        return self._cancels

    async def shutdown(self):
        return None


class TestPullGates:
    def test_downloads_disabled_is_refused(self):
        with build_client(downloads=None) as client:
            response = client.post(
                "/api/models/pull",
                headers={**AUTH, **JSON_CT},
                json={"model": "bonsai-1.7b-2bit"},
            )

        assert response.status_code == 403
        assert response.json()["error"]["type"] == "downloads_disabled"

    def test_unknown_alias_is_refused(self):
        with build_client(downloads=FakeDownloads()) as client:
            response = client.post(
                "/api/models/pull",
                headers={**AUTH, **JSON_CT},
                json={"model": "attacker/arbitrary-repo"},
            )

        # Accepting this would hand a remote caller a general-purpose
        # remote fetch primitive.
        assert response.status_code == 404
        assert response.json()["error"]["type"] == "unknown_model"

    def test_unknown_size_is_refused_rather_than_guessed(self):
        catalog = FakeCatalog(
            entries=[
                ModelEntry(
                    alias="sizeless",
                    hf_path="org/sizeless",
                    size_bytes=None,
                    cached=False,
                )
            ]
        )
        with build_client(catalog=catalog, downloads=FakeDownloads()) as client:
            response = client.post(
                "/api/models/pull",
                headers={**AUTH, **JSON_CT},
                json={"model": "sizeless"},
            )

        # model_sizes.json genuinely lacks entries for some repos.
        # Treating "unknown" as "small" is how a reachable endpoint
        # fills the host's disk.
        assert response.status_code == 507
        assert response.json()["error"]["type"] == "insufficient_storage"

    def test_insufficient_space_is_refused(self, monkeypatch):
        monkeypatch.setattr(
            app_module, "check_disk_budget", lambda size: "not enough free space: ..."
        )
        downloads = FakeDownloads()
        with build_client(downloads=downloads) as client:
            response = client.post(
                "/api/models/pull",
                headers={**AUTH, **JSON_CT},
                json={"model": "bonsai-1.7b-2bit"},
            )

        assert response.status_code == 507
        assert downloads.started == []

    def test_a_valid_pull_starts(self, monkeypatch):
        monkeypatch.setattr(app_module, "check_disk_budget", lambda size: None)
        downloads = FakeDownloads()
        with build_client(downloads=downloads) as client:
            response = client.post(
                "/api/models/pull",
                headers={**AUTH, **JSON_CT},
                json={"model": "bonsai-1.7b-2bit"},
            )

        assert response.status_code == 200
        assert response.json()["state"] == "running"
        assert downloads.started == [("bonsai-1.7b-2bit", 495525300)]

    def test_a_second_concurrent_pull_is_refused(self, monkeypatch):
        monkeypatch.setattr(app_module, "check_disk_budget", lambda size: None)
        downloads = FakeDownloads(
            start_error=DownloadError("a download is already running (x)")
        )
        with build_client(downloads=downloads) as client:
            response = client.post(
                "/api/models/pull",
                headers={**AUTH, **JSON_CT},
                json={"model": "bonsai-1.7b-2bit"},
            )

        assert response.status_code == 409
        assert response.json()["error"]["type"] == "download_conflict"

    def test_pull_requires_a_token(self):
        with build_client(downloads=FakeDownloads()) as client:
            response = client.post(
                "/api/models/pull", headers=JSON_CT, json={"model": "x"}
            )

        assert response.status_code == 401

    @pytest.mark.parametrize("body", [{}, {"model": ""}, {"model": 7}])
    def test_invalid_bodies_are_rejected(self, body):
        with build_client(downloads=FakeDownloads()) as client:
            response = client.post(
                "/api/models/pull", headers={**AUTH, **JSON_CT}, json=body
            )

        assert response.status_code == 400


class TestCancelDownload:
    def test_cancel_stops_a_running_download(self):
        downloads = FakeDownloads()
        with build_client(downloads=downloads) as client:
            response = client.post(
                "/api/downloads/cancel", headers={**AUTH, **JSON_CT}, json={}
            )

        assert response.status_code == 200
        assert downloads.cancelled == 1

    def test_cancel_with_nothing_running_is_a_conflict(self):
        downloads = FakeDownloads(cancels=False)
        with build_client(downloads=downloads) as client:
            response = client.post(
                "/api/downloads/cancel", headers={**AUTH, **JSON_CT}, json={}
            )

        assert response.status_code == 409
        assert response.json()["error"]["type"] == "no_download"

    def test_cancel_is_refused_when_downloads_are_disabled(self):
        with build_client(downloads=None) as client:
            response = client.post(
                "/api/downloads/cancel", headers={**AUTH, **JSON_CT}, json={}
            )

        assert response.status_code == 403


class TestRemoveModel:
    def test_a_cached_model_is_removed(self):
        catalog = FakeCatalog()
        engine = FakeEngine(model="bonsai-1.7b-2bit")
        with build_client(engine=engine, catalog=catalog) as client:
            response = client.post(
                "/api/models/remove",
                headers={**AUTH, **JSON_CT},
                json={"model": "qwen3.5-9b-4bit"},
            )

        assert response.status_code == 200
        assert response.json() == {
            "ok": True,
            "model": "qwen3.5-9b-4bit",
            "freed_bytes": 5977075377,
        }
        assert catalog.removed == ["qwen3.5-9b-4bit"]

    def test_unknown_alias_is_refused(self):
        catalog = FakeCatalog()
        with build_client(catalog=catalog) as client:
            response = client.post(
                "/api/models/remove",
                headers={**AUTH, **JSON_CT},
                json={"model": "attacker/arbitrary-repo"},
            )

        # Same reasoning as pull: an unvalidated alias reaching a
        # subprocess argument is a primitive, and this one deletes.
        assert response.status_code == 409
        assert response.json()["error"]["type"] == "removal_failed"
        assert catalog.removed == []

    def test_the_serving_model_is_refused(self):
        catalog = FakeCatalog()
        engine = FakeEngine(model="qwen3.5-9b-4bit", state=ChildState.READY)
        with build_client(engine=engine, catalog=catalog) as client:
            response = client.post(
                "/api/models/remove",
                headers={**AUTH, **JSON_CT},
                json={"model": "qwen3.5-9b-4bit"},
            )

        # The child has the weights mmap'd; unlinking them underneath it
        # is not a clean operation.
        assert response.status_code == 409
        assert response.json()["error"]["type"] == "model_in_use"
        assert catalog.removed == []

    def test_a_model_that_is_still_loading_is_refused(self):
        catalog = FakeCatalog()
        engine = FakeEngine(model="qwen3.5-9b-4bit", state=ChildState.STARTING)
        with build_client(engine=engine, catalog=catalog) as client:
            response = client.post(
                "/api/models/remove",
                headers={**AUTH, **JSON_CT},
                json={"model": "qwen3.5-9b-4bit"},
            )

        assert response.status_code == 409
        assert response.json()["error"]["type"] == "model_in_use"

    def test_a_model_whose_start_failed_can_be_deleted(self):
        catalog = FakeCatalog()
        engine = FakeEngine(model="qwen3.5-9b-4bit", state=ChildState.FAILED)
        with build_client(engine=engine, catalog=catalog) as client:
            response = client.post(
                "/api/models/remove",
                headers={**AUTH, **JSON_CT},
                json={"model": "qwen3.5-9b-4bit"},
            )

        # The child has already exited and holds nothing. Deleting a
        # checkpoint that just failed to load is exactly what a user
        # does next, so refusing it would strand them.
        assert response.status_code == 200
        assert catalog.removed == ["qwen3.5-9b-4bit"]

    def test_a_model_being_downloaded_is_refused(self):
        catalog = FakeCatalog()
        downloads = FakeDownloads(
            job=DownloadJob(alias="qwen3.5-9b-4bit", total_bytes=1)
        )
        engine = FakeEngine(model=None)
        with build_client(
            engine=engine, catalog=catalog, downloads=downloads
        ) as client:
            response = client.post(
                "/api/models/remove",
                headers={**AUTH, **JSON_CT},
                json={"model": "qwen3.5-9b-4bit"},
            )

        # Unlinking underneath a running pull leaves a half-materialised
        # snapshot, which the next scan reports as "incomplete".
        assert response.status_code == 409
        assert response.json()["error"]["type"] == "model_in_use"
        assert catalog.removed == []

    def test_a_finished_download_does_not_block_removal(self):
        catalog = FakeCatalog()
        downloads = FakeDownloads(
            job=DownloadJob(
                alias="qwen3.5-9b-4bit",
                total_bytes=1,
                state=DownloadState.DONE,
            )
        )
        engine = FakeEngine(model=None)
        with build_client(
            engine=engine, catalog=catalog, downloads=downloads
        ) as client:
            response = client.post(
                "/api/models/remove",
                headers={**AUTH, **JSON_CT},
                json={"model": "qwen3.5-9b-4bit"},
            )

        # The manager retains the last finished job, so keying on the
        # alias alone would permanently block deleting whatever was
        # downloaded most recently.
        assert response.status_code == 200

    def test_removal_failure_is_reported(self):
        catalog = FakeCatalog(remove_error=RemovalError("permission denied"))
        engine = FakeEngine(model=None)
        with build_client(engine=engine, catalog=catalog) as client:
            response = client.post(
                "/api/models/remove",
                headers={**AUTH, **JSON_CT},
                json={"model": "qwen3.5-9b-4bit"},
            )

        assert response.status_code == 409
        assert response.json()["error"]["type"] == "removal_failed"
        assert "permission denied" in response.json()["error"]["message"]

    def test_attach_mode_has_no_catalog_to_remove_from(self):
        engine = FakeEngine(model=None, can_switch=False)
        with build_client(engine=engine, catalog=None) as client:
            response = client.post(
                "/api/models/remove",
                headers={**AUTH, **JSON_CT},
                json={"model": "qwen3.5-9b-4bit"},
            )

        assert response.status_code == 501
        assert response.json()["error"]["type"] == "catalog_unavailable"

    def test_removal_requires_a_token(self):
        with build_client() as client:
            response = client.post(
                "/api/models/remove", headers=JSON_CT, json={"model": "x"}
            )

        assert response.status_code == 401

    def test_a_simple_content_type_is_refused(self):
        # The CSRF control is what makes POST the right method here: a
        # cross-origin page can send text/plain with no preflight.
        with build_client() as client:
            response = client.post(
                "/api/models/remove",
                headers={**AUTH, "Content-Type": "text/plain"},
                content='{"model": "qwen3.5-9b-4bit"}',
            )

        assert response.status_code == 415

    @pytest.mark.parametrize("body", [{}, {"model": ""}, {"model": 7}])
    def test_invalid_bodies_are_rejected(self, body):
        engine = FakeEngine(model=None)
        with build_client(engine=engine) as client:
            response = client.post(
                "/api/models/remove", headers={**AUTH, **JSON_CT}, json=body
            )

        assert response.status_code == 400


class TestDownloadCapabilityReporting:
    def test_auth_reports_whether_downloads_are_allowed(self):
        with build_client(downloads=FakeDownloads()) as client:
            body = client.post("/api/auth", headers={**AUTH, **JSON_CT}, json={}).json()
        assert body["allow_downloads"] is True

        with build_client(downloads=None) as client:
            body = client.post("/api/auth", headers={**AUTH, **JSON_CT}, json={}).json()
        # The page uses this to decide whether tapping an uncached model
        # can do anything, so it must match what the routes enforce.
        assert body["allow_downloads"] is False

    def test_models_reports_whether_downloads_are_allowed(self):
        with build_client(downloads=FakeDownloads()) as client:
            body = client.get("/api/models", headers=AUTH).json()
        assert body["allow_downloads"] is True


class TestDownloadStatus:
    """The polled progress endpoint.

    Replaced an SSE feed that was correct on loopback but unusable through
    a ``trycloudflare`` tunnel: headers in 1.8 s, then no body byte in
    65 s. Chat survives the same tunnel because it emits continuously.
    """

    def test_a_running_job_is_reported(self):
        job = DownloadJob(alias="bonsai-1.7b-2bit", total_bytes=500)
        job.done_bytes = 125
        with build_client(downloads=FakeDownloads(job=job)) as client:
            body = client.get("/api/downloads/status", headers=AUTH).json()

        assert body["alias"] == "bonsai-1.7b-2bit"
        assert body["state"] == "running"
        assert body["done_bytes"] == 125
        assert body["total_bytes"] == 500

    def test_an_idle_manager_reports_idle(self):
        with build_client(downloads=FakeDownloads(job=None)) as client:
            body = client.get("/api/downloads/status", headers=AUTH).json()

        # A synthetic state, not a null job: the page switches on
        # ``state`` alone and should not have to handle a missing body.
        assert body == {"state": "idle"}

    def test_a_finished_job_is_still_reported(self):
        # The manager retains the last finished job, and the page needs
        # to see it: a poll that started after the pull ended must still
        # learn it succeeded, or the strip hangs at its last percentage.
        finished = DownloadJob(alias="already-done", total_bytes=100)
        finished.state = DownloadState.DONE
        finished.done_bytes = 100

        with build_client(downloads=FakeDownloads(job=finished)) as client:
            body = client.get("/api/downloads/status", headers=AUTH).json()

        assert body["state"] == "done"
        assert body["alias"] == "already-done"

    def test_status_requires_a_token(self):
        with build_client(downloads=FakeDownloads()) as client:
            assert client.get("/api/downloads/status").status_code == 401

    def test_status_is_refused_when_downloads_are_disabled(self):
        with build_client(downloads=None) as client:
            response = client.get("/api/downloads/status", headers=AUTH)
        assert response.status_code == 403
        assert response.json()["error"]["type"] == "downloads_disabled"


class TestMandatoryAuthMode:
    def test_an_empty_token_cannot_disable_authentication(self):
        with pytest.raises(ValueError, match="non-empty web access token"):
            WebConfig(token=None, engine=FakeEngine())  # type: ignore[arg-type]


class TestPublicConfig:
    """``/api/config`` is the one unauthenticated JSON endpoint."""

    def test_reports_auth_required_when_a_token_is_set(self):
        with build_client() as client:
            body = client.get("/api/config").json()
        assert body == {"auth_required": True}

    def test_is_reachable_without_a_token(self):
        # The page needs this before it can decide whether to show a
        # login prompt, so it cannot itself be behind the prompt.
        with build_client() as client:
            assert client.get("/api/config").status_code == 200

    def test_leaks_nothing_beyond_the_auth_flag(self):
        engine = FakeEngine(model="secret-model-name")
        with build_client(engine) as client:
            body = client.get("/api/config").json()
        # An unauthenticated caller must not learn which model is loaded,
        # what the catalog holds, or anything about the host.
        assert list(body.keys()) == ["auth_required"]
        assert "secret-model-name" not in json.dumps(body)


class TestResidency:
    """``/api/residency`` — the sidebar's memory panel.

    A read-only relay of the engine's ``/v1/models/residency``.
    """

    def test_the_snapshot_is_relayed(self, monkeypatch):
        snapshot = {
            "memory_limit_bytes": 26843545600,
            "memory_used_bytes": 9750000000,
            "models": [
                {
                    "id": "org/qwen3-4b",
                    "aliases": ["qwen3-4b"],
                    "state": "resident",
                    "pinned": True,
                    "estimated_bytes": 6340000000,
                    "measured_bytes": 5900000000,
                }
            ],
        }

        async def fake_get(self, url, **kwargs):
            assert url.endswith("/v1/models/residency")
            return httpx.Response(200, json=snapshot, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

        with build_client() as client:
            body = client.get("/api/residency", headers=AUTH).json()

        assert body == snapshot

    def test_an_unreachable_engine_reports_an_empty_snapshot(self, monkeypatch):
        """Not an error: the panel describes the machine, and a poll dropped
        during a model switch is not a failure worth surfacing."""

        async def fake_get(self, url, **kwargs):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

        with build_client() as client:
            response = client.get("/api/residency", headers=AUTH)

        assert response.status_code == 200
        assert response.json()["models"] == []

    def test_no_engine_reports_an_empty_snapshot(self):
        engine = FakeEngine(state=ChildState.STOPPED)
        with build_client(engine) as client:
            body = client.get("/api/residency", headers=AUTH).json()

        assert body == {
            "memory_limit_bytes": 0,
            "memory_used_bytes": 0,
            "models": [],
        }

    def test_an_engine_without_the_route_reports_an_empty_snapshot(self, monkeypatch):
        """An older engine 404s here. That is a missing feature, not a fault."""

        async def fake_get(self, url, **kwargs):
            return httpx.Response(
                404, json={"detail": "Not Found"}, request=httpx.Request("GET", url)
            )

        monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

        with build_client() as client:
            body = client.get("/api/residency", headers=AUTH).json()

        assert body["models"] == []

    def test_requires_the_bearer(self):
        with build_client() as client:
            assert client.get("/api/residency").status_code == 401


class TestHotModelLoading:
    """Switching loads into the RUNNING engine before it respawns one.

    This is what lets a chat model and an image model be usable at once:
    the engine keeps text/vision in a single-slot ``assistant`` group and
    gives each media modality its own. A respawn can only ever serve the
    one model it was started for.
    """

    def _catalog(self):
        return FakeCatalog(
            entries=[
                ModelEntry(
                    alias="chat-model",
                    hf_path="org/chat-model",
                    size_bytes=2_000_000_000,
                    cached=True,
                    kind="text",
                ),
                ModelEntry(
                    alias="image-model",
                    hf_path="org/image-model",
                    size_bytes=4_600_000_000,
                    cached=True,
                    kind="image",
                ),
            ]
        )

    def test_a_hot_load_does_not_restart_the_engine(self):
        engine = FakeEngine(model="chat-model")
        engine.residency_outcome = ResidencyOutcome.LOADED

        with build_client(engine, catalog=self._catalog()) as client:
            response = client.post(
                "/api/models/load",
                headers={**AUTH, **JSON_CT},
                json={"model": "image-model"},
            )
            assert response.status_code == 200

        # The whole point: the chat model kept running.
        assert engine.started == []
        assert engine.hot_loaded[0][0] == "image-model"

    def test_the_image_kind_becomes_the_engines_own_modality(self):
        engine = FakeEngine(model="chat-model")
        engine.residency_outcome = ResidencyOutcome.LOADED

        with build_client(engine, catalog=self._catalog()) as client:
            client.post(
                "/api/models/load",
                headers={**AUTH, **JSON_CT},
                json={"model": "image-model"},
            )

        alias, modality, size, image_mode = engine.hot_loaded[0]
        assert modality == "image-gen"
        # The catalog's measured size, not a name-parsed guess.
        assert size == 4_600_000_000
        assert image_mode == "generation"

    def test_a_refused_hot_load_falls_back_to_restarting(self):
        # The engine is over its ceiling. Restarting still gets the user the
        # model they asked for — just without co-residency.
        engine = FakeEngine(model="chat-model")
        engine.residency_outcome = ResidencyOutcome.REJECTED

        with build_client(engine, catalog=self._catalog()) as client:
            response = client.post(
                "/api/models/load",
                headers={**AUTH, **JSON_CT},
                json={"model": "image-model"},
            )
            assert response.status_code == 200

        assert engine.started == ["image-model"]

    def test_an_older_engine_falls_back_to_restarting(self):
        engine = FakeEngine(model="chat-model")
        engine.residency_outcome = ResidencyOutcome.UNSUPPORTED

        with build_client(engine, catalog=self._catalog()) as client:
            client.post(
                "/api/models/load",
                headers={**AUTH, **JSON_CT},
                json={"model": "image-model"},
            )

        assert engine.started == ["image-model"]

    def test_an_already_resident_model_is_a_no_op(self):
        # It is loaded and the engine routes by the request's `model` field,
        # so restarting would throw away a working model for nothing.
        engine = FakeEngine(model="chat-model")
        engine.resident = ["chat-model", "image-model"]

        with build_client(engine, catalog=self._catalog()) as client:
            body = client.post(
                "/api/models/load",
                headers={**AUTH, **JSON_CT},
                json={"model": "image-model"},
            ).json()

        assert body == {"ok": True, "model": "image-model", "state": "ready"}
        assert engine.started == []
        assert engine.hot_loaded == []

    def test_status_reports_every_resident_alias(self):
        # The page needs the whole set: with two models loaded, the surface
        # whose pick is NOT the primary must still read as ready.
        engine = FakeEngine(model="chat-model")
        engine.resident = ["chat-model", "image-model"]

        with build_client(engine) as client:
            body = client.get("/api/status", headers=AUTH).json()

        assert body["model"] == "chat-model"
        assert body["resident"] == ["chat-model", "image-model"]
