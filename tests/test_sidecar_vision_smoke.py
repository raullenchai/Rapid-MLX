from __future__ import annotations

import importlib.util
import socket
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "apps/rapid-mac/scripts/smoke-sidecar-vision.py"
_SPEC = importlib.util.spec_from_file_location("sidecar_vision_smoke", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_completion_matches_exact_expected_fixture_class() -> None:
    assert _MODULE._completion_matches("SPOTTED_CAT", "SPOTTED_CAT")
    assert _MODULE._completion_matches(" other. ", "OTHER")


def test_completion_rejects_empty_error_wrong_or_unrelated_output() -> None:
    assert not _MODULE._completion_matches(None, "SPOTTED_CAT")
    assert not _MODULE._completion_matches("", "SPOTTED_CAT")
    assert not _MODULE._completion_matches("Internal server error", "SPOTTED_CAT")
    assert not _MODULE._completion_matches("A blue square is visible.", "OTHER")
    assert not _MODULE._completion_matches("OTHER", "SPOTTED_CAT")
    assert not _MODULE._completion_matches("SPOTTED_CAT", "OTHER")


def test_release_workflow_runs_content_addressed_real_image_gate() -> None:
    workflow = (
        Path(__file__).parents[1] / ".github/workflows/auto-release.yml"
    ).read_text()
    assert "timeout-minutes: 165" in workflow
    assert "SIDECAR_VISION_SMOKE_MODEL: mlx-community/Qwen3.5-9B-4bit" in workflow
    assert (
        "SIDECAR_VISION_SMOKE_REVISION: 8b2b98c00a6b4d291155e4890773ca8f769aee53"
        in workflow
    )
    assert "SIDECAR_GEMMA_SMOKE_MODEL: mlx-community/gemma-4-e2b-it-8bit" in workflow
    assert (
        "SIDECAR_GEMMA_SMOKE_REVISION: 03dcf209f3f549b4075e7191e77cf69b3d48e1b2"
        in workflow
    )
    assert "HF_HUB_OFFLINE=1 bash apps/rapid-mac/scripts/build-sidecar.sh" in workflow
    assert '"$SIDE/python/bin/python3.12"' in workflow
    assert '--model "$SIDECAR_GEMMA_SMOKE_MODEL"' in workflow
    assert (
        "--negative-image apps/rapid-mac/Sources/Rapid/Resources/Assets.xcassets/RapidLogo.imageset/RapidLogo.png"
        in workflow
    )
    assert "apps/rapid-mac/scripts/build-sidecar.sh" in workflow


def test_repository_model_without_revision_fails_closed() -> None:
    with pytest.raises(SystemExit, match="requires --revision"):
        _MODULE._resolve_model("owner/model", None)


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


_FAKE_SIDECAR = """#!/usr/bin/env python3
import http.server
import base64
import json
import socket
import sys

fd = int(sys.argv[sys.argv.index("--listen-fd") + 1])

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        assert self.path == "/v1/models"
        body = json.dumps({"data": []}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        assert self.path == "/v1/chat/completions"
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        content = payload["messages"][0]["content"]
        assert content[0]["type"] == "text"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        encoded = content[1]["image_url"]["url"].split(",", 1)[1]
        image = base64.b64decode(encoded)
        verdict = "SPOTTED_CAT" if image == b"positive-image" else "OTHER"
        body = json.dumps({"choices": [{"message": {"content": verdict}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

listener = socket.fromfd(fd, socket.AF_INET, socket.SOCK_STREAM)
server = http.server.HTTPServer(("127.0.0.1", 0), Handler, bind_and_activate=False)
server.socket = listener
server.server_address = listener.getsockname()
server.serve_forever()
"""


def test_main_executes_socket_activated_http_image_journey(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sidecar = tmp_path / "sidecar"
    executable = sidecar / "bin" / "rapid-mlx"
    executable.parent.mkdir(parents=True)
    executable.write_text(_FAKE_SIDECAR)
    executable.chmod(0o755)
    model = tmp_path / "model"
    model.mkdir()
    image = tmp_path / "fixture.png"
    image.write_bytes(b"positive-image")
    negative_image = tmp_path / "negative.png"
    negative_image.write_bytes(b"negative-image")
    monkeypatch.setattr(
        "sys.argv",
        [
            str(_SCRIPT),
            "--sidecar-root",
            str(sidecar),
            "--model",
            str(model),
            "--image",
            str(image),
            "--negative-image",
            str(negative_image),
            "--startup-timeout",
            "5",
            "--request-timeout",
            "5",
        ],
    )
    assert _MODULE.main() == 0


def test_main_cleans_log_when_process_creation_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sidecar = tmp_path / "sidecar"
    executable = sidecar / "bin" / "rapid-mlx"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    model = tmp_path / "model"
    model.mkdir()
    image = tmp_path / "fixture.png"
    image.write_bytes(b"fixture")
    negative_image = tmp_path / "negative.png"
    negative_image.write_bytes(b"negative")
    original_named_temporary_file = _MODULE.tempfile.NamedTemporaryFile

    def local_temporary_file(**kwargs):
        return original_named_temporary_file(dir=tmp_path, **kwargs)

    def fail_to_start(*args, **kwargs):
        raise OSError("exec failed")

    monkeypatch.setattr(_MODULE.tempfile, "NamedTemporaryFile", local_temporary_file)
    monkeypatch.setattr(_MODULE.subprocess, "Popen", fail_to_start)
    monkeypatch.setattr(
        "sys.argv",
        [
            str(_SCRIPT),
            "--sidecar-root",
            str(sidecar),
            "--model",
            str(model),
            "--image",
            str(image),
            "--negative-image",
            str(negative_image),
        ],
    )
    with pytest.raises(OSError, match="exec failed"):
        _MODULE.main()
    assert list(tmp_path.glob("rapid-sidecar-vision-*.log")) == []


class _FakeProcess:
    pid = 12345

    def __init__(self, poll_result: int | None) -> None:
        self.poll_result = poll_result
        self.wait_calls: list[int] = []

    def poll(self) -> int | None:
        return self.poll_result

    def wait(self, timeout: int) -> int:
        self.wait_calls.append(timeout)
        return 0


def test_stop_process_is_noop_after_server_already_exited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(0)
    monkeypatch.setattr(_MODULE.os, "killpg", lambda *_: pytest.fail("must not signal"))
    _MODULE._stop_process(process)
    assert process.wait_calls == []


def test_stop_process_tolerates_exit_between_poll_and_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(None)

    def process_gone(*_: object) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(_MODULE.os, "killpg", process_gone)
    _MODULE._stop_process(process)
    assert process.wait_calls == [15]
