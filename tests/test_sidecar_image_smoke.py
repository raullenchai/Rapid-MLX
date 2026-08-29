from __future__ import annotations

import base64
import importlib.util
import json
import socket
import struct
import time
import zlib
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[1]
_SCRIPT = _ROOT / "apps/rapid-mac/scripts/smoke-sidecar-image.py"
_SPEC = importlib.util.spec_from_file_location("sidecar_image_smoke", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _rgb_png(width: int, height: int, *, uniform: bool = False) -> bytes:
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            value = 40 if uniform else (x * 67 + y * 31) % 256
            row.extend((value, (value + 80) % 256, (value + 160) % 256))
        rows.append(bytes(row))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + _chunk(b"IEND", b"")
    )


def _body(payload: bytes) -> dict:
    return {"data": [{"b64_json": base64.b64encode(payload).decode()}]}


def test_generated_png_must_be_decodable_sized_and_non_uniform() -> None:
    payload = _rgb_png(2, 2)
    assert _MODULE._validate_generated_png(_body(payload), (2, 2)) == len(payload)

    with pytest.raises(RuntimeError, match="uniform"):
        _MODULE._validate_generated_png(_body(_rgb_png(2, 2, uniform=True)), (2, 2))
    with pytest.raises(RuntimeError, match="expected"):
        _MODULE._validate_generated_png(_body(payload), (512, 512))
    with pytest.raises(RuntimeError, match="invalid base64"):
        _MODULE._validate_generated_png({"data": [{"b64_json": "%%%"}]}, (2, 2))


def test_repository_model_without_revision_fails_closed() -> None:
    with pytest.raises(SystemExit, match="requires --revision"):
        _MODULE._resolve_model("owner/model", None)


def test_local_model_path_is_served_without_registry_resolution(tmp_path: Path) -> None:
    assert _MODULE._serve_model(str(tmp_path), tmp_path, None) == str(tmp_path)


def test_bound_listener_holds_port_until_socket_activation_handoff() -> None:
    listener = _MODULE._bound_local_listener()
    host, port = listener.getsockname()
    contender = socket.socket()
    try:
        with pytest.raises(OSError):
            contender.bind((host, port))
    finally:
        contender.close()
        listener.close()


def test_request_deadline_bounds_a_never_finishing_response(monkeypatch) -> None:
    class SlowResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            while True:
                time.sleep(0.01)

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_args, **_kwargs: SlowResponse()
    )

    started = time.monotonic()
    with pytest.raises(_MODULE._RequestDeadlineExceededError, match="wall-clock"):
        _MODULE._request_json("http://127.0.0.1/stream", None, 0.05)
    assert time.monotonic() - started < 0.5


def test_request_deadline_preserves_existing_alarm_budget(monkeypatch) -> None:
    timer_calls = []
    monotonic_values = iter((100.0, 102.0))

    monkeypatch.setattr(_MODULE.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(_MODULE.signal, "signal", lambda *_args: "previous-handler")

    def fake_setitimer(*args):
        timer_calls.append(args)
        return (5.0, 1.0) if len(timer_calls) == 1 else (0.0, 0.0)

    monkeypatch.setattr(_MODULE.signal, "setitimer", fake_setitimer)

    with _MODULE._wall_clock_deadline(10.0):
        pass

    assert timer_calls == [
        (_MODULE.signal.ITIMER_REAL, 10.0),
        (_MODULE.signal.ITIMER_REAL, 0),
        (_MODULE.signal.ITIMER_REAL, 3.0, 1.0),
    ]


_FAKE_SIDECAR = """#!/usr/bin/env python3
import http.server
import json
import os
import socket
import sys

fd = int(sys.argv[sys.argv.index("--listen-fd") + 1])
assert os.environ["HF_HUB_OFFLINE"] == "1"
assert os.environ["TRANSFORMERS_OFFLINE"] == "1"

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        assert self.path == "/health"
        self._send({"ready": True, "engine_type": "image"})

    def do_POST(self):
        assert self.path == "/v1/images/generations"
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        assert payload["size"] == "2x2"
        assert payload["response_format"] == "b64_json"
        self._send({"data": [{"b64_json": os.environ["FAKE_PNG"]}]})

listener = socket.fromfd(fd, socket.AF_INET, socket.SOCK_STREAM)
server = http.server.HTTPServer(("127.0.0.1", 0), Handler, bind_and_activate=False)
server.socket = listener
server.server_address = listener.getsockname()
server.serve_forever()
"""


def test_main_executes_socket_activated_image_generation_journey(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sidecar = tmp_path / "sidecar"
    executable = sidecar / "bin" / "rapid-mlx"
    executable.parent.mkdir(parents=True)
    executable.write_text(_FAKE_SIDECAR)
    executable.chmod(0o755)
    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.setenv("FAKE_PNG", base64.b64encode(_rgb_png(2, 2)).decode())
    monkeypatch.setattr(
        "sys.argv",
        [
            str(_SCRIPT),
            "--sidecar-root",
            str(sidecar),
            "--model",
            str(model),
            "--size",
            "2x2",
            "--startup-timeout",
            "5",
            "--request-timeout",
            "5",
        ],
    )
    assert _MODULE.main() == 0


def test_release_workflow_serializes_flux_after_vision_smoke() -> None:
    workflow = (_ROOT / ".github/workflows/auto-release.yml").read_text()
    build_script = _ROOT / "apps/rapid-mac/scripts/build-sidecar.sh"
    build = build_script.read_text()
    assert "steps.sidecar-pins.outputs.flux_model" in workflow
    assert "steps.sidecar-pins.outputs.flux_revision" in workflow
    assert "SIDECAR_IMAGE_SMOKE_MODEL" in workflow
    assert "SIDECAR_IMAGE_SMOKE_REVISION" in workflow
    assert build.index("smoke-sidecar-vision.py") < build.index(
        "smoke-sidecar-image.py"
    )
    assert '"$REPO_ROOT/scripts/smoke-sidecar-image.py"' in build
    # build-sidecar.sh defines REPO_ROOT as its parent directory
    # (apps/rapid-mac), not the git root. Mirror that resolution and prove the
    # invoked file exists instead of merely accepting a plausible string.
    build_repo_root = build_script.parent.parent
    assert build_repo_root / "scripts/smoke-sidecar-image.py" == _SCRIPT
    assert _SCRIPT.is_file()
    assert '"$SIDE/python/bin/python3.12"' in workflow


def test_flux_pin_is_protected_from_host_hygiene() -> None:
    protected = json.loads((_ROOT / "scripts/protected_models.json").read_text())
    flux = [
        model
        for model in protected["models"]
        if model["repository"] == "Runpod/FLUX.2-klein-4B-mflux-4bit"
    ]
    assert flux == [
        {
            "repository": "Runpod/FLUX.2-klein-4B-mflux-4bit",
            "revision": "7ee1b3aa8178a1240050490072196a57da2bf2a9",
            "sources": ["sidecar"],
        }
    ]
