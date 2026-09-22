# SPDX-License-Identifier: Apache-2.0
"""Irreversible contracts for retiring the engine telemetry v1 wire."""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import ipaddress
import json
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from rapid_mlx.telemetry.consent_decision import (
    DEFAULT_ON_CUTOFF,
    DISCLOSURE_REVISION,
    REASON_PRE_CUTOFF_RUNTIME,
    ProcessRole,
    StoredConsent,
    decide,
)

_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE = _ROOT / "rapid_mlx"


@pytest.mark.parametrize("name", ("transport", "queue", "emit", "schema", "consent"))
def test_v1_modules_are_not_importable(name):
    assert importlib.util.find_spec(f"rapid_mlx.telemetry.{name}") is None


def test_no_v1_emitter_symbol_survives_in_python_ast():
    forbidden = {
        "_telemetry_emit",
        "fire_session_end_hook",
        "session_start",
        "register_session_end_hook",
    }
    hits: list[str] = []
    for path in _PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in forbidden:
                hits.append(f"{path.relative_to(_ROOT)}:{node.lineno}:{node.id}")
            elif isinstance(node, ast.Attribute) and node.attr in forbidden:
                hits.append(f"{path.relative_to(_ROOT)}:{node.lineno}:{node.attr}")
    assert hits == []


def test_no_v1_collector_host_survives_in_engine_package():
    hits: list[str] = []
    for path in _PACKAGE.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "telemetry.rapidmlx.com" in text or "/v1/events" in text:
            hits.append(str(path.relative_to(_ROOT)))
    assert hits == []


def test_routes_drop_v1_timing_but_keep_v2_inference_inputs():
    for name in ("chat.py", "anthropic.py", "completions.py"):
        text = (_PACKAGE / "routes" / name).read_text(encoding="utf-8")
        assert "_first_token_ts" not in text
        assert "opt-in telemetry" not in text.lower()
        for retained in ("caller_agent", "caller_client", "served_telemetry_id"):
            assert retained in text
    assert "first_token_ts" not in (_PACKAGE / "routes" / "chat.py").read_text(
        encoding="utf-8"
    )
    assert "_stream_start" not in (_PACKAGE / "routes" / "completions.py").read_text(
        encoding="utf-8"
    )


def test_shared_inference_success_predicate_remains_total():
    from rapid_mlx.telemetry.activation_spec import is_successful_inference

    assert is_successful_inference(200, 1) is True
    assert is_successful_inference(500, 1) is False
    assert is_successful_inference(200, 0) is False
    assert is_successful_inference("bad", 1) is False  # type: ignore[arg-type]


@pytest.mark.parametrize("consent", (None, False, True))
@pytest.mark.parametrize("marker", (None, DISCLOSURE_REVISION))
@pytest.mark.parametrize("role", tuple(ProcessRole))
def test_release_cutoff_makes_pre_cutoff_reason_unreachable(
    consent, marker, role, monkeypatch
):
    import rapid_mlx

    assert DEFAULT_ON_CUTOFF == "0.15.0"
    monkeypatch.setattr(rapid_mlx, "__version__", DEFAULT_ON_CUTOFF)
    stored = StoredConsent(consent, "0.14.3", marker)
    decision = decide(
        stored,
        role,
        kill_switch_active=False,
        running_version=rapid_mlx.__version__,
    )
    assert decision.reason != REASON_PRE_CUTOFF_RUNTIME


def test_real_loopback_serve_can_only_flush_to_posthog(monkeypatch, tmp_path):
    """One real HTTP request plus lifecycle shutdown opens no non-loopback wire."""
    import uvicorn
    from fastapi import FastAPI

    import rapid_mlx
    import rapid_mlx.server as server_module
    import rapid_mlx.telemetry as telemetry_package
    from rapid_mlx.config import reset_config
    from rapid_mlx.engine.base import GenerationOutput
    from rapid_mlx.routes.chat import router as chat_router
    from rapid_mlx.telemetry import build_gate, consent_runtime, posthog_sender, state
    from rapid_mlx.telemetry import track as track_module
    from rapid_mlx.telemetry.build_gate import ReleaseStamp
    from rapid_mlx.telemetry.common_props import PlatformFacts

    monkeypatch.setenv("HOME", str(tmp_path))
    for name in (state.ENV_VAR, state.DO_NOT_TRACK_ENV, *state.CI_ENV_VARS):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(rapid_mlx, "__version__", "0.15.0")
    stamp = ReleaseStamp("stable", "phc_" + "a" * 32)
    monkeypatch.setattr(build_gate, "official_build", lambda: stamp)
    monkeypatch.setattr(consent_runtime, "upload_allowed", lambda: True)
    monkeypatch.setattr(
        track_module.common_props,
        "read_platform_facts",
        lambda: PlatformFacts("darwin", "25.0", "arm64", "m3", 16, "3.11"),
    )
    monkeypatch.setattr(
        state,
        "get_or_create_client_id",
        lambda: "6f1b1d3e-4a2b-4c9d-8e7f-0a1b2c3d4e5f",
    )
    monkeypatch.setattr(
        state, "session_id", lambda: "0a1b2c3d-4e5f-6071-8293-a4b5c6d7e8f9"
    )
    monkeypatch.setattr(track_module.store, "days_since_first_run_bucket", lambda: "0")
    track_module._reset_for_tests()
    posthog_sender._reset_for_tests()

    post_urls: list[str] = []
    post_bodies: list[dict[str, object]] = []
    legacy_urls: list[str] = []

    def record_post(url: str, body: bytes, _timeout: float) -> int:
        post_urls.append(url)
        post_bodies.append(json.loads(body))
        return 200

    monkeypatch.setattr(posthog_sender, "default_post", record_post)
    monkeypatch.setattr(posthog_sender, "FLUSH_THRESHOLD", 1)

    # Mutation trap: if a deleted route call such as ``emit.request(...)`` is
    # restored, it resolves this dormant fake and records the retired URL.
    # With the v1 call absent, the fake is never touched.
    legacy_emit = SimpleNamespace(
        request=lambda **_kwargs: legacy_urls.append(
            "https://telemetry.rapidmlx.com/v1/events"
        ),
        is_enabled=lambda: False,
        activation=lambda **_kwargs: None,
        server_surface=lambda: "server",
    )
    monkeypatch.setitem(sys.modules, "rapid_mlx.telemetry.emit", legacy_emit)
    monkeypatch.setattr(telemetry_package, "emit", legacy_emit, raising=False)

    connects: list[object] = []
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def recording_connect(sock, address):
        connects.append(address)
        return real_connect(sock, address)

    def recording_connect_ex(sock, address):
        connects.append(address)
        return real_connect_ex(sock, address)

    monkeypatch.setattr(socket.socket, "connect", recording_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", recording_connect_ex)

    class StubEngine:
        preserve_native_tool_format = False
        is_mllm = False
        supports_guided_generation = False
        tokenizer = None
        _loaded = False

        async def start(self):
            self._loaded = True

        async def stop(self):
            self._loaded = False

        def generate_warmup(self):
            pass

        def build_prompt(self, messages, tools=None, enable_thinking=None):
            return "PROMPT"

        async def chat(self, messages, **kwargs):
            return GenerationOutput(
                text="hello",
                raw_text="hello",
                prompt_tokens=2,
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
            )

    cfg = reset_config()
    cfg.engine = StubEngine()
    monkeypatch.setattr(server_module, "_engine", cfg.engine)
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    app = FastAPI(lifespan=server_module.lifespan)
    app.include_router(chat_router)

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    web_server = uvicorn.Server(
        uvicorn.Config(app, log_level="critical", lifespan="on")
    )
    thread = threading.Thread(
        target=lambda: asyncio.run(web_server.serve(sockets=[listener])), daemon=True
    )
    thread.start()
    deadline = time.monotonic() + 5.0
    while not web_server.started and time.monotonic() < deadline:
        thread.join(timeout=0.01)
    assert web_server.started

    async def exercise() -> None:
        track_module.start_lifecycle("server")
        deadline = time.monotonic() + 2.0
        while not post_bodies and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert post_bodies
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 4,
                }
            ).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        response = await asyncio.to_thread(urllib.request.urlopen, request)
        with response:
            assert json.loads(response.read())["choices"][0]["message"]["content"] == (
                "hello"
            )

    try:
        asyncio.run(exercise())
    finally:
        web_server.should_exit = True
        thread.join(timeout=5.0)
        listener.close()
        posthog_sender._reset_for_tests()
        reset_config()

    assert connects
    for address in connects:
        host = address[0] if isinstance(address, tuple) else address
        assert host == "localhost" or ipaddress.ip_address(host).is_loopback
    assert post_urls
    assert all(url == posthog_sender._resolve_posthog_url() for url in post_urls)
    assert legacy_urls == []
    assert sorted(item["event"] for body in post_bodies for item in body["batch"]) == [
        "app_opened",
        "model_served",
    ]
    assert "rapidmlx.com" not in repr((connects, post_urls, legacy_urls, post_bodies))
