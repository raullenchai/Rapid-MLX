"""Portable regressions for benchmark stream timing and server ownership."""

import io
import itertools
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from scripts import benchmark_moe_decode_fusion as bench


@pytest.fixture
def harness(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench",
            str(model),
            "--workdir",
            str(tmp_path),
            "--contexts",
            "2",
            "--order",
            "A",
        ],
    )
    tokenizer = SimpleNamespace(encode=list, decode=lambda ids: "".join(ids))
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.utils",
        SimpleNamespace(load_tokenizer=lambda path: tokenizer),
    )
    probe = MagicMock()
    probe.__enter__.return_value.connect_ex.return_value = 1
    monkeypatch.setattr(bench.socket, "socket", lambda: probe)
    proc = MagicMock()
    proc.poll.return_value = None
    launch = MagicMock(return_value=proc)
    monkeypatch.setattr(bench.subprocess, "Popen", launch)
    monkeypatch.setattr(bench.time, "sleep", lambda _: None)
    monkeypatch.setattr(bench.time, "perf_counter", itertools.count().__next__)
    return SimpleNamespace(root=tmp_path, proc=proc, launch=launch, probe=probe)


@pytest.mark.parametrize(
    "texts,tokens,rate,ttft",
    [
        (["a", "b", "c"], 3, 1.0, 1),
        (["abc"], 3, None, 1),
        (["ab", "cd"], 4, None, 1),
        ([], 0, None, None),
        (["a"], 1, None, 1),
    ],
)
def test_stream_metrics(harness, monkeypatch, texts, tokens, rate, ttft):
    def urlopen(request, timeout):
        if isinstance(request, str):
            return io.BytesIO(b"{}")
        events = [{"choices": [{"text": text}]} for text in texts]
        events.append({"usage": {"completion_tokens": tokens}})
        wire = b": keepalive\n" + b"".join(
            b"data: " + json.dumps(event).encode() + b"\n" for event in events
        )
        return io.BytesIO(wire + b"data: [DONE]\n")

    monkeypatch.setattr(bench.urllib.request, "urlopen", urlopen)
    bench.main()
    rows = [
        json.loads(line)
        for line in (harness.root / "serve-results.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 3
    for row in rows:
        assert row["tps"] == rate
        assert row["ttft"] == ttft
        assert row["tokens"] == tokens
        assert row["output"] == "".join(texts)
    harness.proc.terminate.assert_called_once()
    assert harness.launch.call_args.kwargs["stdout"].closed


def test_launch_failure_closes_log(harness):
    harness.launch.side_effect = OSError("cannot launch")
    with pytest.raises(OSError, match="cannot launch"):
        bench.main()
    assert harness.launch.call_args.kwargs["stdout"].closed
    harness.proc.terminate.assert_not_called()


def test_server_exit_closes_log(harness):
    harness.proc.poll.return_value = 1
    harness.proc.returncode = 1
    with pytest.raises(RuntimeError, match="server exit 1"):
        bench.main()
    harness.proc.terminate.assert_called_once()
    assert harness.launch.call_args.kwargs["stdout"].closed


def test_readiness_timeout_kills_only_owned_server(harness, monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("not ready")

    monkeypatch.setattr(bench.urllib.request, "urlopen", unavailable)
    harness.proc.wait.side_effect = [subprocess.TimeoutExpired("serve", 30), None]
    with pytest.raises(TimeoutError, match="did not become ready"):
        bench.main()
    harness.proc.terminate.assert_called_once()
    harness.proc.kill.assert_called_once()
    assert harness.launch.call_args.kwargs["stdout"].closed


@pytest.mark.parametrize(
    "flags", [["--model-missing"], ["--concurrency", "0"], ["--port", "8000"]]
)
def test_invalid_arguments(harness, monkeypatch, flags):
    argv = sys.argv.copy()
    if flags == ["--model-missing"]:
        argv[1] = str(harness.root / "absent")
    else:
        argv.extend(flags)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        bench.main()
    assert exc.value.code == 2
    harness.launch.assert_not_called()


def test_busy_port_does_not_launch(harness):
    harness.probe.__enter__.return_value.connect_ex.return_value = 0
    with pytest.raises(RuntimeError, match="already in use"):
        bench.main()
    harness.launch.assert_not_called()


def test_script_entrypoint_and_missing_usage_status(harness, monkeypatch):
    import runpy

    def urlopen(request, timeout):
        if isinstance(request, str):
            if request.endswith("/v1/status"):
                raise OSError("status unavailable")
            return io.BytesIO(b"{}")
        return io.BytesIO(b'data: {"choices": [{"text": "ab"}]}\ndata: [DONE]\n')

    monkeypatch.setattr(bench.urllib.request, "urlopen", urlopen)
    runpy.run_path(bench.__file__, run_name="__main__")
    rows = [
        json.loads(line)
        for line in (harness.root / "serve-results.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 3
    assert all(row["tokens"] == 2 and row["tps"] is None for row in rows)
    assert all(row["status_error"] == "status unavailable" for row in rows)
    assert harness.launch.call_args.kwargs["stdout"].closed
