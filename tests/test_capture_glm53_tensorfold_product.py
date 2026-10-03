from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/capture_glm53_tensorfold_product.py"
SPEC = importlib.util.spec_from_file_location("glm53_product_capture", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
capture = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = capture
SPEC.loader.exec_module(capture)

SOURCE_PROVENANCE = {"commit": "test-commit", "tree": "test-tree", "dirty": False}
LAUNCH_PROVENANCE = {
    "rapid-mlx": {"command": "rapid-mlx", "executable_sha256": "0" * 64},
    "tensorfold": {"command": "tensorfold", "executable_sha256": "1" * 64},
}


def _fake_probe(ports, _sanitizer):
    return {
        "captured_at_utc": "2026-10-01T00:00:00+00:00",
        "memory_pressure": {
            "available": True,
            "exit_code": 0,
            "stdout": "System-wide memory free percentage: 75%",
            "stderr": "",
        },
        "vm_stat": {
            "available": True,
            "exit_code": 0,
            "stdout": "Mach Virtual Memory Statistics: (page size of 16384 bytes)",
            "stderr": "",
        },
        "swap": {
            "available": True,
            "exit_code": 0,
            "stdout": "total = 0.00M  used = 0.00M  free = 0.00M",
            "stderr": "",
            "used_bytes": 0,
        },
        "load_average": {"one": 0.1, "five": 0.2, "fifteen": 0.3},
        "listeners": {str(port): capture.listener_open(port) for port in ports},
    }


def _write_fake_server(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            import argparse
            import hashlib
            import json
            import os
            import signal
            import sys
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            parser = argparse.ArgumentParser()
            parser.add_argument("--mode", choices=("product", "direct"), required=True)
            parser.add_argument("--port", type=int, required=True)
            parser.add_argument("--scenario", default="valid")
            parser.add_argument("--marker")
            args = parser.parse_args()
            signal.signal(signal.SIGTERM, lambda *_args: sys.exit(0))
            if args.marker:
                with open(args.marker, "w") as marker:
                    json.dump({"pid":os.getpid(), "port":args.port}, marker)

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_args):
                    pass

                def send_json(self, value):
                    body = json.dumps(value, separators=(",", ":")).encode()
                    self.send_response(200)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def do_GET(self):
                    if self.path == "/v1/models":
                        if args.scenario == "wait_models":
                            return self.send_error(503)
                        model = (
                            "glm5.3-flash-tensorfold"
                            if args.mode == "product"
                            else "glm53-tf-v06"
                        )
                        record = {"id":model}
                        if args.mode == "product":
                            record["speculative_decoding"] = {
                                "configured":True,"method":"mtp",
                                "runtime_state":"active","backend":"tensorfold",
                            }
                        return self.send_json({"object":"list","data":[record]})
                    if self.path == "/healthz":
                        return self.send_json({
                            "status":"ok",
                            "engine":"tensorfold-glm-5.3-flash",
                            "algorithm":"mtp",
                        })
                    if self.path == "/v1/status":
                        status = {
                            "status":"ready",
                            "model":"glm5.3-flash-tensorfold",
                            "speculative_decoding":{
                                "configured":True,"method":"mtp",
                                "runtime_state":"active","backend":"tensorfold",
                            },
                            "profile":{
                                "id":"glm5.3-flash-tensorfold",
                                "mode":"accelerated",
                                "compatibility":{
                                    "state":"ready","reason":None,"action":None,
                                },
                                "runtime":{
                                    "extra":"manual-source-install","installed":True,
                                },
                                "models":{"target":{
                                    "repository":"Vontra/GLM-5.3-Flash-MLX-4bit-MTP",
                                    "revision":"76add2a341a1cd90ad0e86bb69839ea9c35827c6",
                                    "ready":True,
                                }},
                            },
                        }
                        if args.scenario == "wrong_status":
                            status["status"] = "/Users/private/not-ready"
                        return self.send_json(status)
                    self.send_error(404)

                def do_POST(self):
                    length = int(self.headers.get("content-length", "0"))
                    request = json.loads(self.rfile.read(length))
                    if args.mode == "product":
                        assert request["model"] == "glm5.3-flash-tensorfold"
                        assert request["reasoning_max_tokens"] == 256
                        assert "thinking_budget" not in request
                        token_ids = [11, 22, 33]
                        digest = hashlib.sha256(b"11,22,33").hexdigest()
                        if args.scenario != "missing_audit":
                            with open(os.environ["RAPID_MLX_TENSORFOLD_AUDIT_PATH"], "w") as f:
                                f.write(json.dumps({
                                    "request_id":"rapid-private-id",
                                    "token_ids":token_ids,
                                    "token_sha256":digest,
                                }) + "\\n")
                    else:
                        assert request["model"] == "glm53-tf-v06"
                        assert request["thinking_budget"] == 256
                    response = {
                        "id":"chatcmpl-fixture",
                        "object":"chat.completion",
                        "model":(
                            "glm5.3-flash-tensorfold"
                            if args.mode == "product" else "glm53-tf-v06"
                        ),
                        "choices":[{"index":0,"message":{
                            "role":"assistant",
                            "reasoning_content":"reason",
                            "content":"answer",
                        },"finish_reason":"length"}],
                        "usage":{"prompt_tokens":460,"completion_tokens":3,"total_tokens":463},
                    }
                    if args.mode == "direct":
                        response["tensorfold"] = {"token_sha":"abcdef123456"}
                    self.send_json(response)

            print(
                "pid=12345 /Users/private/secret Authorization: Bearer super-secret "
                "https://private.example/path",
                flush=True,
            )
            ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
            """
        )
    )


def test_capture_contract_with_fake_product_and_direct_servers(tmp_path: Path) -> None:
    fake = tmp_path / "fake_server.py"
    _write_fake_server(fake)
    output = tmp_path / "artifacts"
    target = tmp_path / "snapshots" / ("76add2a341a1cd90ad0e86bb69839ea9c35827c6")
    target.mkdir(parents=True)

    def product_builder(port: int) -> list[str]:
        return [sys.executable, str(fake), "--mode", "product", "--port", str(port)]

    def direct_builder(port: int, _target: Path) -> list[str]:
        return [sys.executable, str(fake), "--mode", "direct", "--port", str(port)]

    manifest = capture.run_capture(
        output,
        target=target,
        load_timeout=10,
        request_timeout=10,
        product_builder=product_builder,
        direct_builder=direct_builder,
        launch_provenance=LAUNCH_PROVENANCE,
        source_provenance=SOURCE_PROVENANCE,
        probe=_fake_probe,
    )

    assert manifest["status"] == "complete"
    assert manifest["final_listeners_gone"] is True
    assert manifest["swap_gate"]["passed"] is True
    product = manifest["phases"]["product"]["completion"]
    assert product["full_token_ids"]["ids"] == [11, 22, 33]
    assert (
        product["full_token_ids"]["sha256"] == hashlib.sha256(b"11,22,33").hexdigest()
    )
    drafted = manifest["phases"]["direct"]["drafted"]
    assert drafted["full_token_ids"]["availability"] == "unavailable"
    assert drafted["full_token_ids"]["sha256"] is None
    assert drafted["opaque_token_fingerprint"] == {
        "value": "abcdef123456",
        "kind": "tensorfold_opaque_token_fingerprint",
        "is_sha256": False,
    }
    assert all(
        manifest["phases"]["direct"]["comparisons"]["direct_drafted_vs_serial"].values()
    )

    frozen = output / "requests/frozen-fixture.json"
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == (
        capture.REQUIRED_FIXTURE_SHA256
    )
    product_request = json.loads(
        (output / "requests/product-submitted.json").read_text()
    )
    assert product_request["reasoning_max_tokens"] == 256
    assert "thinking_budget" not in product_request

    hashes = json.loads((output / capture.HASH_MAP_NAME).read_text())
    retained = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != capture.HASH_MAP_NAME
    }
    assert set(hashes["files"]) == retained
    capture.verify_hash_map(output)

    logs = "\n".join(
        path.read_text() for path in output.rglob("*.log") if path.is_file()
    )
    assert "super-secret" not in logs
    assert "/Users/private" not in logs
    assert "private.example" not in logs
    assert "pid=12345" not in logs
    assert "<redacted>" in logs
    assert "<private-path>" in logs
    assert "pid=<pid>" in logs


def test_production_commands_pin_alias_and_direct_settings(tmp_path: Path) -> None:
    product = capture.product_command(Path("/verified/rapid-mlx"), 18162)
    assert product[:3] == [
        "/verified/rapid-mlx",
        "serve",
        "glm5.3-flash-tensorfold",
    ]
    assert product[-2:] == ["--port", "18162"]

    target = tmp_path / "target"
    direct = capture.direct_command(Path("/verified/tensorfold"), 18163, target)
    assert direct[:2] == ["/verified/tensorfold", "serve"]
    assert direct[2] == str(target)
    assert direct[direct.index("--context") + 1] == "8192"
    assert direct[direct.index("--max-tokens") + 1] == "4096"
    assert direct[direct.index("--parallel") + 1] == "1"
    assert direct[direct.index("--mtp-drafts") + 1] == "3"
    assert direct[direct.index("--prefill-pass") + 1] == "8"
    assert direct[direct.index("--pass-cache-gib") + 1] == "16"
    assert direct[direct.index("--snapshot-dir") + 1] == "none"
    assert "--no-update-check" in direct


def test_stream_request_records_first_visible_delta_ttft() -> None:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["content-length"]))
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n')
            self.wfile.flush()
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n')
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        origin = capture.time.monotonic_ns()
        result = capture.http_request(
            f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
            origin_ns=origin,
            payload=b'{"stream":true}',
            timeout=5,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result.ttft_ns is not None and result.ttft_ns >= 0
    assert result.ttft_basis == "first_visible_sse_delta"
    assert result.started_offset_ns <= result.first_byte_offset_ns
    assert result.first_byte_offset_ns <= result.ended_offset_ns


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("used = 0.00M", 0),
        ("total = 4.00G used = 1.50G free = 2.50G", round(1.5 * 1024**3)),
        ("unparseable", None),
    ],
)
def test_parse_swap_used_bytes(text: str, expected: int | None) -> None:
    assert capture.parse_swap_used_bytes(text) == expected


def test_cli_refuses_to_resolve_model_without_command_lifetime_lock(
    tmp_path: Path,
) -> None:
    env = dict(os.environ)
    env.pop("RAPID_LARGE_MODEL_LOCK_HELD", None)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(tmp_path / "out")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "scripts/large-model-run.py" in result.stderr
    assert not (tmp_path / "out").exists()


def test_nonzero_swap_discards_before_starting_a_server(tmp_path: Path) -> None:
    started = []

    def swapped_probe(ports, sanitizer):
        result = _fake_probe(ports, sanitizer)
        result["swap"]["used_bytes"] = 1024
        result["swap"]["stdout"] = "total = 1.00M used = 0.00M free = 1.00M"
        return result

    def product_builder(_port: int) -> list[str]:
        started.append("product")
        return ["must-not-run"]

    target = tmp_path / "snapshots" / capture.model_contract()["target_revision"]
    target.mkdir(parents=True)
    output = tmp_path / "discarded"
    with pytest.raises(capture.CaptureError, match="pre-capture swap usage"):
        capture.run_capture(
            output,
            target=target,
            product_builder=product_builder,
            direct_builder=lambda _port, _target: ["must-not-run"],
            launch_provenance=LAUNCH_PROVENANCE,
            source_provenance=SOURCE_PROVENANCE,
            probe=swapped_probe,
        )

    assert started == []
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "invalid"
    assert manifest["swap_gate"]["passed"] is False
    capture.verify_hash_map(output)


def test_runtime_contract_requires_exact_noneditable_vcs_revision(monkeypatch) -> None:
    model = capture.model_contract()
    valid = {
        "installed": True,
        "version": "0.6.0",
        "editable": False,
        "vcs": {"type": "git", "commit_id": model["runtime_revision"]},
    }
    monkeypatch.setattr(capture, "package_provenance", lambda _name: valid)
    capture.require_qualified_runtime(model)

    monkeypatch.setattr(
        capture,
        "package_provenance",
        lambda _name: {**valid, "editable": True},
    )
    with pytest.raises(capture.CaptureError, match="exact qualified"):
        capture.require_qualified_runtime(model)


def test_console_binding_requires_current_interpreter_and_distribution_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    executable = tmp_path / "rapid-mlx"
    executable.write_text(
        f"#!{sys.executable}\nfrom rapid_mlx.cli import main\nmain()\n"
    )

    class EntryPoint:
        group = "console_scripts"
        name = "rapid-mlx"
        module = "rapid_mlx.cli"
        attr = "main"
        value = "rapid_mlx.cli:main"

    class Distribution:
        version = "test-version"
        entry_points = [EntryPoint()]

    monkeypatch.setattr(capture.shutil, "which", lambda _command: str(executable))
    monkeypatch.setattr(
        capture.importlib.metadata, "distribution", lambda _name: Distribution()
    )
    binding = capture.resolve_console_binding("rapid-mlx", "rapid-mlx")
    assert binding.executable == executable.resolve()
    assert (
        binding.executable_sha256 == hashlib.sha256(executable.read_bytes()).hexdigest()
    )

    executable.write_text(
        "#!/usr/bin/python3\nfrom rapid_mlx.cli import main\nmain()\n"
    )
    with pytest.raises(capture.CaptureError, match="capture interpreter"):
        capture.resolve_console_binding("rapid-mlx", "rapid-mlx")


def test_capture_fails_closed_on_dirty_source(monkeypatch) -> None:
    monkeypatch.setattr(
        capture,
        "git_provenance",
        lambda: {"commit": "test", "tree": "test", "dirty": True},
    )
    with pytest.raises(capture.CaptureError, match="must be clean"):
        capture.require_clean_source()


def test_sanitizer_redacts_json_credentials_private_urls_and_pids(
    tmp_path: Path,
) -> None:
    sanitizer = capture.Sanitizer(tmp_path)
    cleaned = sanitizer.text(
        '{"api_key":"secret-value","authorization":"Bearer credential",'
        '"pid":1234,"url":"https://private.invalid/path"}'
    )
    assert "secret-value" not in cleaned
    assert "credential" not in cleaned
    assert "1234" not in cleaned
    assert "private.invalid" not in cleaned
    assert cleaned.count("<redacted>") == 2
    assert "<pid>" in cleaned
    assert "<url>" in cleaned


def test_child_environment_is_offline_and_drops_credential_variables(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "private")
    monkeypatch.setenv("SOME_API_KEY", "private")
    monkeypatch.setenv("PYTHONPATH", "/private/shadow-modules")
    monkeypatch.setenv("PYTHONHOME", "/private/shadow-runtime")
    monkeypatch.setenv("RAPID_MLX_TENSORFOLD_AUDIT_PATH", "/private/audit.jsonl")
    monkeypatch.setenv("RAPID_MLX_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("RAPID_MLX_TELEMETRY_DEBUG", "1")
    monkeypatch.setenv("RAPID_MLX_TELEMETRY", "1")
    monkeypatch.setenv("DO_NOT_TRACK", "0")
    monkeypatch.setenv("TENSORFOLD_REQUEST_LOG", "/private/request-log.jsonl")
    monkeypatch.setenv("TENSORFOLD_SNAPSHOT_DIR", "/private/snapshots")
    monkeypatch.setenv("TENSORFOLD_FUTURE_OVERRIDE", "unsafe")
    monkeypatch.setenv("NORMAL_SETTING", "retained")
    environment = capture.offline_child_environment()
    assert "HF_TOKEN" not in environment
    assert "SOME_API_KEY" not in environment
    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert "RAPID_MLX_TENSORFOLD_AUDIT_PATH" not in environment
    assert "RAPID_MLX_LOG_LEVEL" not in environment
    assert "RAPID_MLX_TELEMETRY_DEBUG" not in environment
    assert "TENSORFOLD_REQUEST_LOG" not in environment
    assert "TENSORFOLD_SNAPSHOT_DIR" not in environment
    assert "TENSORFOLD_FUTURE_OVERRIDE" not in environment
    assert environment["NORMAL_SETTING"] == "retained"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["RAPID_MLX_TELEMETRY"] == "0"
    assert environment["RAPID_MLX_DISABLE_VERSION_CHECK"] == "1"
    assert environment["DO_NOT_TRACK"] == "1"


def test_comparison_uses_full_token_hash_when_both_paths_expose_ids() -> None:
    base = {
        "content_sha256": "content",
        "reasoning_sha256": "reasoning",
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        "full_token_ids": {"availability": "available", "sha256": "left"},
        "opaque_token_fingerprint": None,
    }
    comparison = capture.compare_summaries(
        base,
        {
            **base,
            "full_token_ids": {"availability": "available", "sha256": "right"},
        },
    )
    assert comparison["full_token_ids_sha256"] is False
    assert comparison["usage_counts"] is True


def _valid_product_identity():
    model = capture.model_contract()
    health = {
        "status": "ok",
        "engine": "tensorfold-glm-5.3-flash",
        "algorithm": "mtp",
    }
    speculative = capture.expected_speculative_identity()
    status = {
        "status": "ready",
        "model": capture.ALIAS,
        "speculative_decoding": dict(speculative),
        "profile": {
            "id": capture.ALIAS,
            "mode": "accelerated",
            "compatibility": {"state": "ready", "reason": None, "action": None},
            "runtime": {"extra": "manual-source-install", "installed": True},
            "models": {
                "target": {
                    "repository": model["repository"],
                    "revision": model["target_revision"],
                    "ready": True,
                }
            },
        },
    }
    return health, status, model


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda h, _s, _m: h.update(status="bad"), "healthz"),
        (lambda _h, s, _m: s.update(model="wrong"), "identity"),
        (lambda _h, s, _m: s["profile"].update(id="wrong"), "profile"),
        (
            lambda _h, s, _m: s["profile"]["models"]["target"].update(revision="wrong"),
            "provenance",
        ),
        (lambda _h, s, _m: s.update(status="loading"), "readiness"),
        (
            lambda _h, s, _m: s["speculative_decoding"].update(backend="wrong"),
            "TensorFold MTP",
        ),
        (lambda _h, s, _m: s["profile"].update(mode="fallback"), "accelerated"),
        (
            lambda _h, s, _m: s["profile"]["compatibility"].update(state="blocked"),
            "compatibility",
        ),
        (
            lambda _h, s, _m: s["profile"]["models"]["target"].update(ready=False),
            "readiness",
        ),
    ],
)
def test_product_identity_rejects_adversarial_status(mutation, message: str) -> None:
    health, status, model = _valid_product_identity()
    mutation(health, status, model)
    with pytest.raises(capture.CaptureError, match=message):
        capture.require_product_identity(health, status, model)


def test_product_identity_projects_only_validated_constants() -> None:
    health, status, model = _valid_product_identity()
    health["private_path"] = "/Users/private/secret"
    status["private_url"] = "https://private.invalid/path"
    identity = capture.require_product_identity(health, status, model)
    serialized = json.dumps(identity)
    assert "/Users/" not in serialized
    assert "private.invalid" not in serialized
    assert identity["profile"]["target"]["revision"] == model["target_revision"]


def test_models_identity_rejects_wrong_model_backend_and_readiness() -> None:
    valid = {
        "data": [
            {
                "id": capture.ALIAS,
                "speculative_decoding": capture.expected_speculative_identity(),
            }
        ]
    }
    assert (
        capture.require_models_identity(
            valid, capture.ALIAS, require_tensorfold_mtp=True
        )["id"]
        == capture.ALIAS
    )
    wrong_model = json.loads(json.dumps(valid))
    wrong_model["data"][0]["id"] = "wrong"
    with pytest.raises(capture.CaptureError, match="did not expose"):
        capture.require_models_identity(
            wrong_model, capture.ALIAS, require_tensorfold_mtp=True
        )
    wrong_backend = json.loads(json.dumps(valid))
    wrong_backend["data"][0]["speculative_decoding"]["backend"] = "wrong"
    with pytest.raises(capture.CaptureError, match="TensorFold MTP"):
        capture.require_models_identity(
            wrong_backend, capture.ALIAS, require_tensorfold_mtp=True
        )


def _timing() -> object:
    return capture.HTTPResult(
        status=200,
        headers={},
        raw=b"{}",
        started_offset_ns=1,
        first_byte_offset_ns=2,
        ended_offset_ns=3,
        ttft_ns=None,
        ttft_basis=None,
    )


def _valid_completion() -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": capture.DIRECT_SERVED_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        "tensorfold": {"token_sha": "abcdef123456"},
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(choices=[]),
        lambda value: value.update(object="wrong"),
        lambda value: value.update(model="wrong"),
        lambda value: value.update(usage={}),
        lambda value: value["usage"].update(completion_tokens="1"),
        lambda value: value["usage"].update(total_tokens=99),
        lambda value: value["choices"][0].update(finish_reason="private/path"),
        lambda value: value["tensorfold"].update(token_sha="a" * 64),
        lambda value: value.update(token_ids=[1, -2]),
    ],
)
def test_response_summary_rejects_malformed_completion_and_usage(mutation) -> None:
    completion = _valid_completion()
    mutation(completion)
    with pytest.raises(capture.CaptureError):
        capture.response_summary(
            completion,
            _timing(),
            expected_model=capture.DIRECT_SERVED_NAME,
            require_opaque_fingerprint=True,
        )


def test_response_summary_rejects_http_and_audit_token_mismatch() -> None:
    completion = _valid_completion()
    completion["token_ids"] = [1]
    with pytest.raises(capture.CaptureError, match="did not match"):
        capture.response_summary(
            completion,
            _timing(),
            expected_model=capture.DIRECT_SERVED_NAME,
            audit_tokens=([2], "test-audit"),
            require_opaque_fingerprint=True,
        )


def test_start_server_rejects_port_collision(tmp_path: Path) -> None:
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = occupied.getsockname()[1]
        with pytest.raises(capture.CaptureError, match="occupied"):
            capture.start_server(
                "collision",
                [sys.executable, "-c", "raise SystemExit(99)"],
                port=port,
                env=dict(os.environ),
                raw_dir=tmp_path,
                origin_ns=time.monotonic_ns(),
            )


def _write_shutdown_server(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            import argparse
            import signal
            import sys
            from http.server import BaseHTTPRequestHandler, HTTPServer

            parser = argparse.ArgumentParser()
            parser.add_argument("--port", type=int, required=True)
            parser.add_argument("--behavior", choices=("clean", "nonzero", "ignore"))
            args = parser.parse_args()
            if args.behavior == "clean":
                signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
            elif args.behavior == "nonzero":
                signal.signal(signal.SIGTERM, lambda *_: sys.exit(7))
            else:
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_args):
                    pass
            # Keep the shutdown fixture single-threaded. ``listener_open``
            # creates short probe connections; worker threads from
            # ThreadingHTTPServer can still be finalizing when the signal
            # handler raises SystemExit, which makes CPython abort during
            # interpreter teardown and hides the requested 0/7 exit code.
            HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
            """
        )
    )


def _start_shutdown_server(tmp_path: Path, behavior: str):
    script = tmp_path / f"shutdown-{behavior}.py"
    _write_shutdown_server(script)
    port = capture.allocate_loopback_port()
    server = capture.start_server(
        behavior,
        [sys.executable, str(script), "--port", str(port), "--behavior", behavior],
        port=port,
        env=dict(os.environ),
        raw_dir=tmp_path,
        origin_ns=time.monotonic_ns(),
    )
    deadline = time.monotonic() + 5
    while not capture.listener_open(port) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert capture.listener_open(port)
    return server


def test_stop_server_accepts_only_clean_shutdown(tmp_path: Path) -> None:
    facts = capture.stop_server(_start_shutdown_server(tmp_path, "clean"), timeout=10)
    assert facts["exit_code"] == 0
    assert facts["listener_gone"] is True


def test_stop_server_rejects_unexpected_prior_exit(tmp_path: Path) -> None:
    port = capture.allocate_loopback_port()
    process = subprocess.Popen(
        [sys.executable, "-c", "raise SystemExit(0)"], start_new_session=True
    )
    process.wait(timeout=5)
    server = capture.ManagedServer(
        name="prior-exit",
        process=process,
        port=port,
        command=[],
        raw_log=tmp_path / "empty.log",
        started_offset_ns=0,
    )
    with pytest.raises(capture.ShutdownError, match="before requested") as raised:
        capture.stop_server(server)
    assert raised.value.facts["process_was_running_at_shutdown"] is False
    assert raised.value.facts["listener_gone"] is True


@pytest.mark.parametrize(
    ("behavior", "timeout"),
    [("nonzero", 10.0), ("ignore", 0.1)],
)
def test_stop_server_rejects_abnormal_or_forced_shutdown(
    tmp_path: Path, behavior: str, timeout: float
) -> None:
    server = _start_shutdown_server(tmp_path, behavior)
    with pytest.raises(capture.ShutdownError) as raised:
        capture.stop_server(server, timeout=timeout)
    assert raised.value.facts["listener_gone"] is True
    if behavior == "nonzero":
        assert raised.value.facts["exit_code"] == 7
        assert raised.value.facts["forced_kill_of_owned_process_group"] is False
    else:
        assert raised.value.facts["forced_kill_of_owned_process_group"] is True


def test_phase_validation_failure_still_cleans_listener_and_sanitizes_raw_status(
    tmp_path: Path,
) -> None:
    fake = tmp_path / "fake_server.py"
    _write_fake_server(fake)
    output = tmp_path / "invalid-artifacts"
    target = tmp_path / "target"
    target.mkdir()

    def product_builder(port: int) -> list[str]:
        return [
            sys.executable,
            str(fake),
            "--mode",
            "product",
            "--scenario",
            "wrong_status",
            "--port",
            str(port),
        ]

    with pytest.raises(capture.CaptureError, match="readiness"):
        capture.run_capture(
            output,
            target=target,
            product_builder=product_builder,
            direct_builder=lambda _port, _target: ["must-not-run"],
            launch_provenance=LAUNCH_PROVENANCE,
            source_provenance=SOURCE_PROVENANCE,
            load_timeout=5,
            request_timeout=5,
            probe=_fake_probe,
        )

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "invalid"
    assert manifest["phases"]["product"]["process"]["exit_code"] == 0
    assert manifest["final_listeners_gone"] is True
    assert (
        "/Users/private" not in (output / "product/status.sanitized.body").read_text()
    )
    capture.verify_hash_map(output)


def test_product_phase_requires_full_token_audit_evidence(tmp_path: Path) -> None:
    fake = tmp_path / "fake_server.py"
    _write_fake_server(fake)
    output = tmp_path / "missing-audit-artifacts"
    target = tmp_path / "target"
    target.mkdir()

    def product_builder(port: int) -> list[str]:
        return [
            sys.executable,
            str(fake),
            "--mode",
            "product",
            "--scenario",
            "missing_audit",
            "--port",
            str(port),
        ]

    with pytest.raises(capture.CaptureError, match="full-token audit evidence"):
        capture.run_capture(
            output,
            target=target,
            product_builder=product_builder,
            direct_builder=lambda _port, _target: ["must-not-run"],
            launch_provenance=LAUNCH_PROVENANCE,
            source_provenance=SOURCE_PROVENANCE,
            load_timeout=5,
            request_timeout=5,
            probe=_fake_probe,
        )

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "invalid"
    assert manifest["phases"]["product"]["process"]["exit_code"] == 0
    assert manifest["final_listeners_gone"] is True
    capture.verify_hash_map(output)


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP, signal.SIGINT])
def test_termination_signal_cleans_owned_server_and_listener(
    tmp_path: Path, signum: int
) -> None:
    fake = tmp_path / "fake_server.py"
    _write_fake_server(fake)
    marker = tmp_path / "server.json"
    output = tmp_path / "signal-artifacts"
    target = tmp_path / "target"
    target.mkdir()
    runner = tmp_path / "runner.py"
    runner.write_text(
        textwrap.dedent(
            f"""
            import importlib.util
            import json
            import sys
            from pathlib import Path

            script = Path({str(SCRIPT)!r})
            spec = importlib.util.spec_from_file_location("signal_capture", script)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            def probe(ports, _sanitizer):
                return {{
                    "swap": {{"used_bytes": 0}},
                    "listeners": {{str(port): module.listener_open(port) for port in ports}},
                }}

            def product_builder(port):
                return [
                    sys.executable, {str(fake)!r}, "--mode", "product",
                    "--scenario", "wait_models", "--marker", {str(marker)!r},
                    "--port", str(port),
                ]

            try:
                with module.termination_signal_handlers():
                    module.run_capture(
                        Path({str(output)!r}),
                        target=Path({str(target)!r}),
                        product_builder=product_builder,
                        direct_builder=lambda _port, _target: ["must-not-run"],
                        launch_provenance={{}},
                        source_provenance={{"commit":"test","tree":"test","dirty":False}},
                        load_timeout=60,
                        request_timeout=5,
                        probe=probe,
                    )
            except module.TerminationSignalError as exc:
                raise SystemExit(exc.exit_code)
            """
        )
    )
    harness = subprocess.Popen(
        [sys.executable, str(runner)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 10
        while not marker.is_file() and time.monotonic() < deadline:
            if harness.poll() is not None:
                break
            time.sleep(0.02)
        assert marker.is_file(), harness.communicate(timeout=1)
        owned = json.loads(marker.read_text())
        os.kill(harness.pid, signum)
        stdout, stderr = harness.communicate(timeout=10)
        assert harness.returncode == 128 + signum, (stdout, stderr)
        with pytest.raises(ProcessLookupError):
            os.kill(owned["pid"], 0)
        assert capture.listener_open(owned["port"]) is False
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["status"] == "invalid"
        assert manifest["phases"]["product"]["process"]["listener_gone"] is True
        capture.verify_hash_map(output)
    finally:
        if harness.poll() is None:
            harness.kill()
            harness.wait(timeout=5)
