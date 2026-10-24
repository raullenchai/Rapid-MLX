"""CPU-only protocol and grader regression tests; no model imports."""

from __future__ import annotations

import importlib.util
import json
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "phase6_eval", Path(__file__).parents[1] / "scripts/eval_deepseek_v41_tasks.py"
)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class Tokenizer:
    def encode(self, text):
        return SimpleNamespace(ids=list(range(len(text.split()))))


def call(name, arguments, ident="call_1"):
    return {
        "id": ident,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


@pytest.fixture
def fake_server():
    state = SimpleNamespace(requests=[], failure=None, no_done=False, truncate=False)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.requests.append(payload)
            if state.failure:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error":"unsupported seed"}')
                return
            content = payload["messages"][-1]["content"]
            message = {"role": "assistant", "content": ""}
            finish = "stop"
            if payload["messages"][-1]["role"] == "tool":
                message["content"] = "Paris: 17 C; population 2100000."
            elif payload.get("tools"):
                if payload["tools"][0]["function"]["name"] == "get_temperature":
                    message["tool_calls"] = [
                        call("get_temperature", {"city": "Paris"}),
                        call("get_population", {"city": "Paris"}, "call_2"),
                    ]
                else:
                    message["tool_calls"] = [call("add", {"a": 2, "b": 3})]
                finish = "tool_calls"
            elif "secret access code" in content:
                message["content"] = "SILVER-OTTER-7319"
            elif "中国" in content:
                message["content"] = "中国首都是北京。"
            elif "numeric answer" in content:
                message["content"] = "#### 42"
            elif "correct answer letter" in content:
                message["content"] = "B"
            elif "Complete this Python function" in content:
                message["content"] = "```python\ndef add(a,b):\n return a+b\n```"
            else:
                message["content"] = "4"
            if state.truncate:
                finish = "length"
            usage = {"prompt_tokens": len(content.split()) + 8, "completion_tokens": 23}
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "text/event-stream" if payload["stream"] else "application/json",
            )
            self.end_headers()
            if not payload["stream"]:
                self.wfile.write(
                    json.dumps(
                        {
                            "choices": [{"message": message, "finish_reason": finish}],
                            "usage": usage,
                        }
                    ).encode()
                )
                return

            def event(obj):
                self.wfile.write(("data: " + json.dumps(obj) + "\n\n").encode())
                self.wfile.flush()

            # Deliberately split function names/JSON across SSE events.
            if message.get("tool_calls"):
                for i, tool in enumerate(message["tool_calls"]):
                    fn = tool["function"]
                    event(
                        {
                            "choices": [
                                {
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": i,
                                                "id": tool["id"],
                                                "function": {
                                                    "name": fn["name"][:3],
                                                    "arguments": fn["arguments"][:5],
                                                },
                                            }
                                        ]
                                    }
                                }
                            ]
                        }
                    )
                    event(
                        {
                            "choices": [
                                {
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": i,
                                                "function": {
                                                    "name": fn["name"][3:],
                                                    "arguments": fn["arguments"][5:],
                                                },
                                            }
                                        ]
                                    }
                                }
                            ]
                        }
                    )
            else:
                for part in (message["content"][:2], message["content"][2:]):
                    event({"choices": [{"delta": {"content": part}}]})
            event({"choices": [{"delta": {}, "finish_reason": finish}]})
            event({"choices": [], "usage": usage})
            if not state.no_done:
                self.wfile.write(b"data: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_port}/v1"
    yield state
    server.shutdown()
    server.server_close()
    thread.join()


def test_smoke_all_checks_and_fragmented_two_tool_round_trip(fake_server):
    checks = bench.smoke(bench.Client(fake_server.url, "test"), Tokenizer())
    assert len(checks) == 8
    assert all(r["passed"] for r in checks)
    assert len(fake_server.requests) == 10
    assert {r["stream"] for r in fake_server.requests} == {False, True}
    assert all(
        r["seed"] == 1017 and r["temperature"] == 0 for r in fake_server.requests
    )
    assert all(r["enable_thinking"] is False for r in fake_server.requests)
    assert all(r["body_tokens"] <= 16384 for r in checks if "body_tokens" in r)


def test_speed_uses_usage_not_chunk_count(fake_server):
    rows = bench.speed(bench.Client(fake_server.url, "test"), Tokenizer(), 1)
    assert all(r["passed"] for r in rows)
    assert [r["id"] for r in rows] == ["1024_0", "8192_0"]
    for row in rows:
        response = row["response"]
        assert row["decode_tok_s"] == pytest.approx(
            22 / (response["elapsed"] - response["ttft"])
        )
        assert row["prefill_tok_s_client_estimate"] == pytest.approx(
            response["usage"]["prompt_tokens"] / response["ttft"]
        )


def test_http_and_incomplete_sse_preserve_raw(fake_server):
    client = bench.Client(fake_server.url, "test", mode="thinking")
    fake_server.failure = True
    result = client.chat([{"role": "user", "content": "hi"}])
    assert "HTTP 400" in result["error"] and "unsupported seed" in result["error"]
    assert not bench.valid(result)
    fake_server.failure = False
    fake_server.no_done = True
    result = client.chat([{"role": "user", "content": "hi"}], stream=True)
    assert "incomplete SSE" in result["error"]
    assert result["raw"]
    assert fake_server.requests[-1]["chat_template_kwargs"]["enable_thinking"] is True


def test_numeric_letter_and_truncation():
    assert bench.numeric("Calculation: 5. Final #### -1,234.50") == bench.Decimal(
        "-1234.5"
    )
    assert bench.numeric("no number") is None
    assert bench.letter("Reasoning here.\nAnswer: (J).") == "J"
    assert bench.letter("(B)") == "B"
    assert bench.letter("A misleading explanation without an answer") is None
    result = {"message": {"content": "42"}, "finish_reason": "length"}
    assert not bench.grade(
        {"task": "gsm8k", "data": {"answer": "#### 42"}}, result, None
    )["passed"]


def test_tool_argument_matching_parallel_bijection_and_optional():
    expected = [
        {"add": {"a": [1, 2], "b": [3], "optional": [""]}},
        {"add": {"a": [1], "b": [3]}},
    ]
    message = {
        "tool_calls": [
            call("add", {"a": 1, "b": 3}),
            call("add", {"a": 2, "b": 3}, "two"),
        ]
    }
    assert bench.tool_match(message, expected)
    message["tool_calls"].append(call("add", {"a": 1, "b": 3}, "three"))
    assert not bench.tool_match(message, expected)
    assert not bench.tool_match(
        {"tool_calls": [call("add", {"a": True})]}, [{"add": {"a": [1]}}]
    )
    with pytest.raises(ValueError):
        bench.parse_calls(
            {"tool_calls": [{"function": {"name": "x", "arguments": "{bad"}}]}
        )
    assert (
        bench.schema({"type": "dict", "properties": {"x": {"type": "float"}}})[
            "properties"
        ]["x"]["type"]
        == "number"
    )


def test_sampling_is_fixed_and_stratified():
    rows = [{"category": chr(65 + i // 30), "id": i} for i in range(120)]
    a = bench.stratified(rows, 42, 1017)
    assert a == bench.stratified(rows, 42, 1017)
    assert a != bench.stratified(rows, 42, 1018)
    assert sorted(Counter(r["category"] for r in a).values()) == [10, 10, 11, 11]


def make_manifest(path):
    manifest = {
        "seed": 1017,
        "counts": {"gsm8k": 1, "mmlu_pro": 1, "tools": 1},
        "items": [
            {
                "task": "gsm8k",
                "id": "fixed-gsm",
                "data": {"question": "six times seven?", "answer": "#### 42"},
            },
            {
                "task": "mmlu_pro",
                "id": "fixed-mmlu",
                "data": {
                    "question": "pick B",
                    "options": ["no", "yes"],
                    "answer": "B",
                    "category": "math",
                },
            },
            {
                "task": "tools",
                "id": "fixed-tool",
                "data": {
                    "question": [[{"role": "user", "content": "add 2 and 3"}]],
                    "function": [
                        {
                            "name": "add",
                            "parameters": {
                                "type": "dict",
                                "properties": {
                                    "a": {"type": "integer"},
                                    "b": {"type": "integer"},
                                },
                            },
                        }
                    ],
                    "ground_truth": [{"add": {"a": [2], "b": [3]}}],
                },
            },
        ],
    }
    manifest["sha256"] = bench.digest(manifest)
    bench.dump(path, manifest)
    return manifest


def test_bench_cli_artifacts_and_manifest_tamper(fake_server, tmp_path):
    path, output = tmp_path / "manifest.json", tmp_path / "run"
    manifest = make_manifest(path)
    assert (
        bench.main(
            [
                "bench",
                "--base-url",
                fake_server.url,
                "--manifest",
                str(path),
                "--output",
                str(output),
                "--tasks",
                "gsm8k",
                "mmlu_pro",
                "tools",
                "--build-label",
                "fake-no-draft",
            ]
        )
        == 0
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["complete"]
    assert summary["config"]["manifest_sha256"] == manifest["sha256"]
    assert all(r["score_percent"] == 100 for r in summary["results"].values())
    records = bench.json_lines(output / "items.jsonl")
    assert [r["id"] for r in records] == ["fixed-gsm", "fixed-mmlu", "fixed-tool"]
    assert all(r["response"]["raw"] for r in records)
    with pytest.raises(FileExistsError):
        bench.main(["bench", "--manifest", str(path), "--output", str(output)])
    manifest["items"][0]["id"] = "changed"
    bench.dump(path, manifest)
    with pytest.raises(ValueError, match="digest"):
        bench.load_manifest(path)


def test_bench_errors_abort_and_return_nonzero(fake_server, tmp_path):
    path, output = tmp_path / "manifest.json", tmp_path / "run"
    make_manifest(path)
    fake_server.failure = True
    assert (
        bench.main(
            [
                "bench",
                "--base-url",
                fake_server.url,
                "--manifest",
                str(path),
                "--output",
                str(output),
                "--tasks",
                "gsm8k",
                "mmlu_pro",
                "tools",
                "--max-consecutive-errors",
                "1",
            ]
        )
        == 2
    )
    summary = json.loads((output / "summary.json").read_text())
    assert not summary["complete"] and summary["expected_items"] == 3
    assert len(fake_server.requests) == 1


def test_humaneval_no_sandbox_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(bench.shutil, "which", lambda _: None)
    result = bench.execute_code(
        "def f(): return 1", "def check(f): assert f() == 1", "f", 1, tmp_path
    )
    assert not result["passed"] and "no unsafe fallback" in result["error"]
    assert "def f" in bench.python_code("```python\ndef f():\n return 1\n```", "", "f")
    assert bench.python_code("    return 1", "def f():\n", "f").endswith("return 1")


def test_humaneval_sandbox_execution_and_no_network(tmp_path):
    probe = bench.execute_code(
        "def f(): return 1", "def check(f): assert f() == 1", "f", 1, tmp_path
    )
    if probe.get("error") and (
        "Operation not permitted" in probe["error"] or "unavailable" in probe["error"]
    ):
        pytest.skip(
            "host execution policy denies sandbox-exec; fail-closed path verified separately"
        )
    assert probe["passed"], probe
    denied = bench.execute_code(
        "import socket\ndef f():\n"
        " return socket.create_connection(('1.1.1.1', 80), timeout=1)",
        "def check(f): f()",
        "f",
        1,
        tmp_path,
    )
    assert not denied["passed"]
    timeout = bench.execute_code(
        "def f():\n while True: pass", "def check(f): f()", "f", 1, tmp_path
    )
    assert not timeout["passed"]


@pytest.mark.parametrize("command", ["smoke", "speed"])
def test_cli_smoke_speed_save_raw_and_summary(
    command, fake_server, tmp_path, monkeypatch
):
    import tokenizers

    path, output = tmp_path / "manifest.json", tmp_path / "run"
    manifest = make_manifest(path)
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}")
    manifest.pop("sha256")
    manifest["tokenizer"] = {"file": str(tokenizer)}
    manifest["sha256"] = bench.digest(manifest)
    bench.dump(path, manifest)
    monkeypatch.setattr(
        tokenizers, "Tokenizer", SimpleNamespace(from_file=lambda _: Tokenizer())
    )
    assert (
        bench.main(
            [
                command,
                "--base-url",
                fake_server.url,
                "--manifest",
                str(path),
                "--output",
                str(output),
                "--runs",
                "1",
            ]
        )
        == 0
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["results"][command]["passed"] == (8 if command == "smoke" else 2)
    if command == "speed":
        assert summary["timings"]["8192"]["decode_tok_s"] > 0
    assert all(
        json.loads(p.read_text()).get("response", {}).get("raw")
        or json.loads(p.read_text()).get("responses")
        for p in output.glob("*.json")
        if p.name not in ("summary.json", "config.json")
    )


def test_humaneval_preflight_failure_stops_before_http(
    fake_server, tmp_path, monkeypatch
):
    path, output = tmp_path / "manifest.json", tmp_path / "run"
    make_manifest(path)
    monkeypatch.setattr(
        bench,
        "execute_code",
        lambda *args: {"passed": False, "error": "sandbox unavailable"},
    )
    assert (
        bench.main(
            [
                "bench",
                "--base-url",
                fake_server.url,
                "--manifest",
                str(path),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert not fake_server.requests
    assert json.loads((output / "summary.json").read_text())["complete"] is False


def test_humaneval_http_to_grader(fake_server, tmp_path, monkeypatch):
    seen = []

    def sandbox(code, tests, entry, timeout, scratch):
        seen.append((code, tests, entry))
        return {"passed": "def add" in code and "candidate(2,3)" in tests}

    monkeypatch.setattr(bench, "execute_code", sandbox)
    response = bench.Client(fake_server.url, "test").chat(
        [{"role": "user", "content": "Complete this Python function: def add(a,b)"}]
    )
    item = {
        "task": "humaneval",
        "data": {
            "prompt": "def add(a,b):\n",
            "test": "def check(candidate): assert candidate(2,3)==5",
            "entry_point": "add",
        },
    }
    assert bench.grade(
        item, response, SimpleNamespace(code_timeout=1, scratch=tmp_path)
    )["passed"]
    assert seen[0][2] == "add"


@pytest.mark.parametrize("expire", [False, True])
def test_sandbox_profile_and_timeout_contract(tmp_path, monkeypatch, expire):
    captured = {}
    monkeypatch.setattr(bench.shutil, "which", lambda _: "/usr/bin/sandbox-exec")
    monkeypatch.setattr(
        bench.os, "killpg", lambda pid, sig: captured.update(killed=(pid, sig))
    )

    class Process:
        pid = 1234567
        returncode = 0

        def __init__(self, command, **kwargs):
            captured.update(command=command, kwargs=kwargs)
            self.waits = 0
            import os

            kwargs["stdout"].write(b"candidate output\n")
            program = Path(command[-1]).read_text()
            marker = bench.ast.literal_eval(
                bench.ast.parse(program).body[-1].value.args[-1]
            )
            os.write(kwargs["pass_fds"][0], marker)

        def wait(self, **kwargs):
            self.waits += 1
            if expire and self.waits == 1:
                raise bench.subprocess.TimeoutExpired("candidate", 3)
            return 0

    monkeypatch.setattr(bench.subprocess, "Popen", Process)
    result = bench.execute_code(
        "def f(): return 1", "def check(f): assert f() == 1", "f", 1, tmp_path
    )
    assert result["passed"] is not expire
    profile = captured["command"][2]
    assert "(deny default)" in profile
    assert "(allow signal (target self))" in profile
    assert "(allow signal)" not in profile
    assert (
        "network" not in profile
    )  # no allow-rule: deny-default covers all network operations
    assert "process-fork" not in profile
    assert "(allow file-write-data (literal " in profile
    assert "(allow file-write*" not in profile
    assert captured["kwargs"]["start_new_session"] is True
    assert set(captured["kwargs"]["env"]) == {"PATH", "TMPDIR", "RESULT_FD"}
    if expire:
        assert captured["killed"][0] == 1234567
        assert result["detail"] == "execution timeout"


@pytest.mark.parametrize(
    "code, passed",
    [
        ("print('candidate noise')\ndef f(): return 1", True),
        ("import os\nprint('HUMANEVAL_OK', flush=True)\nos._exit(0)", False),
        ("print('HUMANEVAL_OK')\ndef f(): return 2", False),
    ],
)
def test_humaneval_completion_is_independent_of_stdout(
    tmp_path, monkeypatch, code, passed
):
    popen = bench.subprocess.Popen
    monkeypatch.setattr(bench.shutil, "which", lambda _: "/usr/bin/sandbox-exec")

    def fixed_fixture_process(command, **kwargs):
        return popen(command[3:], **kwargs)

    monkeypatch.setattr(bench.subprocess, "Popen", fixed_fixture_process)
    result = bench.execute_code(code, "def check(f): assert f() == 1", "f", 1, tmp_path)
    assert result["passed"] is passed, result


def test_prefetch_freezes_revisions_and_refuses_replacement(tmp_path, monkeypatch):
    import sys
    import types

    import huggingface_hub

    downloads = []
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}")

    def download(repo, filename, **kwargs):
        downloads.append((repo, filename, kwargs))
        assert kwargs["revision"] == "frozen-revision"
        if filename == "tokenizer.json":
            return str(tokenizer)
        if filename.endswith(".parquet"):
            return repo
        path = tmp_path / filename.replace("/", "_")
        rows = [
            {"id": str(i), "language": "Python", "ground_truth": [{"f": {}}]}
            for i in range(25)
        ]
        path.write_text("\n".join(json.dumps(r) for r in rows))
        return str(path)

    def table(repo):
        if repo == bench.DATASETS["gsm8k"][0]:
            rows = [{"question": "q", "answer": "42"} for _ in range(250)]
        elif repo == bench.DATASETS["mmlu_pro"][0]:
            rows = [{"category": "math", "question_id": i} for i in range(300)]
        else:
            rows = [{"task_id": "HumanEval/0"}]
        return SimpleNamespace(to_pylist=lambda: rows)

    parquet = types.ModuleType("pyarrow.parquet")
    parquet.read_table = table
    arrow = types.ModuleType("pyarrow")
    arrow.parquet = parquet
    monkeypatch.setitem(sys.modules, "pyarrow", arrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet)
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda: SimpleNamespace(
            dataset_info=lambda _: SimpleNamespace(sha="frozen-revision"),
            model_info=lambda _: SimpleNamespace(sha="frozen-revision"),
        ),
    )
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    manifest = tmp_path / "manifest.json"
    assert bench.main(["prefetch", "--manifest", str(manifest)]) == 0
    frozen = bench.load_manifest(manifest)
    assert frozen["counts"] == {
        "gsm8k": 250,
        "mmlu_pro": 300,
        "humaneval": 1,
        "tools": 100,
    }
    assert len(downloads) == 12
    with pytest.raises(ValueError, match="Refusing to replace"):
        bench.prefetch(manifest, 1017)


def test_protocol_and_grader_rejections(tmp_path, monkeypatch):
    import io

    client = bench.Client(
        "http://example.invalid/v1", "fixture", api_key="fixture-token"
    )

    def response(request, **kwargs):
        assert request.get_header("Authorization") == "Bearer fixture-token"
        return io.BytesIO(b'{"choices":[{"message":null}]}')

    monkeypatch.setattr(bench, "urlopen", response)
    assert "missing assistant" in client.chat([])["error"]
    assert bench.schema([{"type": "any"}]) == [{}]
    assert not bench.argument_match({"a": [1]}, {"a": [2]})
    assert bench.argument_match({"a": [1]}, {"a": [1.0]})
    assert not bench.match_call(("f", {"extra": 1}), {"f": {}})
    assert not bench.match_call(("f", {}), {"f": {"required": [1]}})
    assert not bench.match_call(("f", {}), {"g": {}})
    with pytest.raises(ValueError, match="missing"):
        bench.parse_calls({})
    with pytest.raises(ValueError, match="JSON object"):
        bench.parse_calls({"tool_calls": [call("f", [])]})
    result = {"finish_reason": "stop", "message": {"content": "not valid python!"}}
    assert not bench.grade(
        {"task": "tools", "data": {"ground_truth": []}}, result, None
    )["passed"]
    assert (
        "invalid Python"
        in bench.grade(
            {"task": "humaneval", "data": {"prompt": "", "entry_point": "f"}},
            result,
            None,
        )["detail"]
    )
    messages, extra = bench.prompt_item(
        {"task": "humaneval", "data": {"prompt": "def f():"}}
    )
    assert "def f():" in messages[0]["content"] and extra == {}
    monkeypatch.setattr(
        bench, "Decimal", lambda _: (_ for _ in ()).throw(bench.InvalidOperation())
    )
    assert bench.numeric("1") is None


def test_cli_validation_and_failed_grader_artifacts(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    make_manifest(path)
    for args in (["--runs", "0"], ["--seed", "12"]):
        with pytest.raises(SystemExit):
            bench.main(["bench", "--manifest", str(path), *args])
    monkeypatch.setattr(
        bench.Client,
        "chat",
        lambda *a, **k: {
            "usage": {},
            "finish_reason": "stop",
            "message": {"content": "42"},
        },
    )
    monkeypatch.setattr(
        bench, "grade", lambda *a: (_ for _ in ()).throw(ValueError("bad fixture"))
    )
    official = tmp_path / "official.json"
    bench.dump(
        official, {"scores": {"gsm8k": 50}, "comparison_warning": "different protocol"}
    )
    out = tmp_path / "run"
    assert (
        bench.main(
            [
                "bench",
                "--manifest",
                str(path),
                "--output",
                str(out),
                "--tasks",
                "gsm8k",
                "--official-scores",
                str(official),
            ]
        )
        == 2
    )
    summary = json.loads((out / "summary.json").read_text())
    assert summary["published_reference"]["scores"] == {"gsm8k": 50}
    assert summary["results"]["gsm8k"]["errors"] == 1


def test_smoke_records_bad_tool_response():
    client = SimpleNamespace(
        chat=lambda *a, **k: {"message": {}, "usage": {}, "error": "fixture rejection"}
    )
    checks = bench.smoke(client, Tokenizer())
    assert all(not c["passed"] for c in checks)
    assert all("error" in c for c in checks if c["id"].startswith("tools"))


def test_entrypoint_rejects_invalid_arguments(monkeypatch):
    import runpy
    import sys

    monkeypatch.setattr(sys, "argv", [str(SPEC.origin), "speed", "--runs", "0"])
    with pytest.raises(SystemExit) as exit:
        runpy.run_path(str(SPEC.origin), run_name="__main__")
    assert exit.value.code == 2


def test_sandbox_launch_denial_is_infrastructure_error(tmp_path, monkeypatch):
    monkeypatch.setattr(bench.shutil, "which", lambda _: "/usr/bin/sandbox-exec")

    def denied(command, **kwargs):
        kwargs["stderr"].write(b"sandbox-exec: Operation not permitted")
        return SimpleNamespace(wait=lambda **kw: None, returncode=1)

    monkeypatch.setattr(bench.subprocess, "Popen", denied)
    result = bench.execute_code(
        "def f(): return 1", "def check(f): pass", "f", 1, tmp_path
    )
    assert not result["passed"] and "Operation not permitted" in result["error"]
