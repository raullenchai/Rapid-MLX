"""No model, network, or server process required."""

import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "research_server", Path(__file__).parents[1] / "server.py"
)
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)


def test_partial_tail_and_resume(tmp_path):
    path = tmp_path / "events.jsonl"
    assert server.records(path) == ([], 0)
    first = b'{"event":"idea","round":1}\n'
    path.write_bytes(first + b'{"event":"result"')
    rows, offset = server.records(path)
    assert rows == [(len(first), {"event": "idea", "round": 1})]
    with path.open("ab") as stream:
        stream.write(b',"value":0.3}\n')
    later, _ = server.records(path, offset)
    assert later[0][1] == {"event": "result", "value": 0.3}


def test_replay_timing_and_sse_ids(tmp_path):
    path = tmp_path / "events.jsonl"
    events = [
        {"event": "idea", "ts": 100},
        {"event": "curve", "ts": 180},
        {"event": "result", "ts": 196},
        {"event": "done", "ts": 196},
    ]
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    sleeps = []
    frames = list(server.event_stream(path, speed=8, sleep=sleeps.append))
    payloads = [f for f in frames if f.startswith(b"id:")]
    assert len(payloads) == 4
    assert sleeps == [0.15, 2.0]  # Curves compress; verdict timing stays faithful.
    first_id = int(payloads[0].split(b"\n")[0][4:])
    resumed = list(server.event_stream(path, after=first_id, alive=lambda: False))
    assert len([f for f in resumed if f.startswith(b"id:")]) == 3
    assert all(b"\ndata: " in f and f.endswith(b"\n\n") for f in payloads)


def test_tail_sees_append_during_poll(tmp_path):
    path = tmp_path / "events.jsonl"
    path.touch()

    def append(_):
        path.write_text('{"event":"done"}\n')

    frames = list(server.event_stream(path, sleep=append))
    assert b'"event": "done"' in frames[-1]


def test_closed_without_done(tmp_path):
    frames = list(server.event_stream(tmp_path / "missing", alive=lambda: False))
    assert frames == [b"event: closed\ndata: {}\n\n"]


def test_concurrent_inbox_append(tmp_path):
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(
            pool.map(
                lambda i: server.append_idea(
                    tmp_path,
                    {"hypothesis": f"idea {i}\nsecond line", "change": "α → β"},
                ),
                range(30),
            )
        )
    lines = (tmp_path / "inbox.jsonl").read_text().splitlines()
    assert len(lines) == 30
    ideas = [json.loads(line) for line in lines]
    assert len({i["hypothesis"] for i in ideas}) == 30
    assert all(i["change"] == "α → β" for i in ideas)


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"hypothesis": ""},
        {"hypothesis": "  "},
        {"hypothesis": 3},
        {"hypothesis": "x", "change": []},
    ],
)
def test_invalid_idea(tmp_path, data):
    with pytest.raises(ValueError):
        server.append_idea(tmp_path, data)
    assert not (tmp_path / "inbox.jsonl").exists()


def test_exit_drains_final_events(tmp_path):
    path = tmp_path / "events.jsonl"
    path.touch()

    def exited():
        path.write_text('{"event":"done"}\n')
        return False

    frames = list(server.event_stream(path, alive=exited))
    assert b'"event": "done"' in frames[-1]


@pytest.fixture
def app(tmp_path):
    app = server.App(
        ("127.0.0.1", 0),
        projects=tmp_path / "projects",
        models={"q35": "http://127.0.0.1:1/v1", "q27": "http://127.0.0.1:2/v1"},
    )
    yield app
    app.server_close()


def request(app, path, data=None, method=None, headers=None):
    """Exercise the actual handler without binding a public/test-fixed port."""
    from io import BytesIO

    handler = object.__new__(server.Handler)
    body = json.dumps(data).encode() if data is not None else b""
    handler.server, handler.path = app, path
    handler.headers = {
        "Content-Length": str(len(body)),
        "Host": "localhost",
        **(headers or {}),
    }
    handler.rfile, handler.wfile = BytesIO(body), BytesIO()
    result = {"status": None, "headers": {}}
    handler.send_response = lambda status: result.update(status=status)
    handler.send_header = lambda key, value: result["headers"].update({key: value})
    handler.end_headers = lambda: None
    getattr(handler, "do_" + (method or ("POST" if data is not None else "GET")))()
    result["body"] = handler.wfile.getvalue()
    if result["headers"].get("Content-Type") == "application/json":
        result["json"] = json.loads(result["body"])
    return result


def create(app, title="Sorting study", template="sort-speed"):
    response = request(
        app,
        "/api/projects",
        {
            "title": title,
            "template": template,
            "goal": "Reduce runtime without changing the input.",
        },
    )
    assert response["status"] == 201, response
    return response["json"]["project"]["slug"]


def make_run(app, slug, events=None):
    project = app.project_path(slug)
    run = project / "runs" / "example"
    run.mkdir()
    (run / "events.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in (events or []))
    )
    (run / "round0_sort.py").write_text("VALUE = 10\n")
    (run / "round1_sort.py").write_text("VALUE = 8\n")
    (run / "loop.log").write_text("Local experiment log\n")
    return run


def test_projects_templates_and_persistent_goal(app):
    templates = request(app, "/api/templates")["json"]["templates"]
    assert {t["slug"] for t in templates} == {
        "grokking",
        "sort-speed",
        "drug-levels",
        "enzyme-kinetics",
        "lj-cluster",
        "no-three-ap",
        "orbit-integrator",
        "pid-control",
    }
    assert all(t.get("domain") for t in templates)
    slug = create(app)
    detail = request(app, f"/api/projects/{slug}")["json"]
    assert detail["project"]["template"] == "sort-speed"
    assert detail["program"].startswith("# Research goal\n\nReduce runtime")
    assert detail["task"]["direction"] == "min"
    assert detail["discussion"] == detail["ideas"] == detail["runs"] == []
    assert detail["models"] == ["q35", "q27"]
    saved = request(
        app,
        f"/api/projects/{slug}/program",
        {"program": "# New goal\nMeasure carefully."},
        method="PUT",
    )
    assert saved["status"] == 200
    assert (
        app.project_path(slug).joinpath("program.md").read_text()
        == saved["json"]["program"]
    )
    listing = request(app, "/api/projects")["json"]["projects"]
    assert listing[0]["experiments"] == 0 and listing[0]["best"] is None
    assert create(app) != slug  # A duplicate title never overwrites a project.


def test_run_creation_time_is_independent_of_name_and_last_result(app):
    slug = create(app)
    run = make_run(
        app,
        slug,
        [
            {"event": "baseline", "ts": 100, "metric": "runtime_ms", "result": {}},
            {"event": "done", "ts": 500, "best": 8},
        ],
    )
    metadata = run / "run.json"
    metadata.write_text(json.dumps({"created": 90, "model": "q35"}))
    info = request(app, f"/api/projects/{slug}")["json"]["runs"][0]
    assert info["created"] == 90 and info["updated"] == 500
    metadata.unlink()
    # Legacy runs have no creation metadata; their first event is authoritative.
    info = app.info(run)
    assert info["created"] == 100 and info["updated"] == 500
    (run / "events.jsonl").write_text("")
    assert app.info(run)["created"] > 0


def test_create_run_routes_model_and_preserves_current_python(app, monkeypatch):
    from types import SimpleNamespace

    slug = create(app)
    calls = []
    monkeypatch.setattr(
        server.subprocess,
        "Popen",
        lambda args, **kw: calls.append(args) or SimpleNamespace(poll=lambda: None),
    )
    result = request(app, f"/api/projects/{slug}/runs", {"model": "q27", "rounds": 3})
    assert result["status"] == 201
    assert calls[0][0] == server.sys.executable
    assert calls[0][2] == str(app.project_path(slug))
    assert Path(calls[0][3]).is_absolute()
    assert calls[0][4:8] == ["--base-url", app.models["q27"], "--model", "q27"]
    assert calls[0][-2:] == ["--rounds", "3"]
    assert request(app, f"/api/projects/{slug}/runs", {})["status"] == 400
    assert (
        request(
            app, f"/api/projects/{slug}/program", {"program": "change"}, method="PUT"
        )["status"]
        == 400
    )
    run = app.run_path(app.project_path(slug), result["json"]["id"])
    assert (
        request(
            app, f"/api/projects/{slug}/ideas", {"hypothesis": "Use built-in sorting"}
        )["status"]
        == 201
    )
    assert not (run / "inbox.jsonl").exists()
    assert (
        server.hypotheses.collect(
            app.project_path(slug), app.task(app.project_path(slug))
        )[0]["source"]
        == "you"
    )
    (run / "PAUSE").touch()
    assert (
        request(app, f"/api/projects/{slug}/runs/{run.name}/stop", {})["status"] == 200
    )
    assert (run / "STOP").exists() and not (run / "PAUSE").exists()


def test_queue_before_start_and_validation(app):
    slug = create(app)
    base = f"/api/projects/{slug}/ideas"
    assert (
        request(
            app, base, {"hypothesis": "Fewer comparisons", "change": "Use sorting key"}
        )["status"]
        == 201
    )
    idea = request(app, f"/api/projects/{slug}")["json"]["hypotheses"][0]
    assert idea["statement"] == "Fewer comparisons" and idea["source"] == "you"
    for data in (
        {},
        {"hypothesis": ""},
        {"hypothesis": 42},
        {"hypothesis": "x", "change": []},
    ):
        assert request(app, base, data)["status"] == 400


def test_artifacts_diff_summary_and_download(app):
    slug = create(app)
    run = make_run(
        app,
        slug,
        [
            {
                "event": "baseline",
                "metric": "runtime_ms",
                "result": {"runtime_ms": 10},
                "ts": 1,
            },
            {
                "event": "idea",
                "round": 1,
                "parent": 0,
                "change": "Faster sort",
                "ts": 2,
            },
            {
                "event": "result",
                "round": 1,
                "value": 8,
                "verdict": "supported",
                "ts": 3,
            },
            {"event": "done", "best": 8, "ts": 4},
        ],
    )
    (run / "best_sort.py").write_text("VALUE = 8\n")
    summary = request(app, f"/api/projects/{slug}")["json"]
    assert summary["project"]["best"] == 8
    assert summary["project"]["experiments"] == 1
    assert summary["project"]["progress"] == [10, 8]
    assert summary["project"]["status"] == "Finished"
    assert summary["project"]["description"] == (
        "Reduce runtime without changing the input."
    )
    assert request(app, f"/api/projects/{slug}/runs/example")["json"]["files"] == [
        "best_sort.py",
        "events.jsonl",
        "loop.log",
        "round0_sort.py",
        "round1_sort.py",
    ]
    base = f"/api/projects/{slug}/runs/example"
    diff = request(app, base + "/diff/1")["json"]
    assert (
        diff["parent"] == 0
        and "-VALUE = 10" in diff["diff"]
        and "+VALUE = 8" in diff["diff"]
    )
    result = request(app, base + "/files/best_sort.py?download=1")
    assert result["body"] == b"VALUE = 8\n"
    assert (
        result["headers"]["Content-Disposition"]
        == 'attachment; filename="best_sort.py"'
    )
    (run / "best_sort.py").unlink()
    assert (
        request(app, base + "/files/best_sort.py")["body"] == b"VALUE = 8\n"
    )  # Numbered kept-code fallback.
    stream = request(app, base + "/events?replay=1&speed=1000")
    assert b'"event": "done"' in stream["body"]
    for speed in ("nan", "inf", "-1", "1001"):
        assert request(app, base + "/events?replay=1&speed=" + speed)["status"] == 400


@pytest.mark.parametrize(
    "slug", ["../outside", "..", "a/b", "a_b", "A", "%2e%2e", "a--b"]
)
def test_invalid_project_paths(app, slug):
    with pytest.raises(ValueError):
        app.project_path(slug)
    assert request(app, f"/api/projects/{slug}")["status"] in (400, 404)


def test_symlinks_and_file_whitelist(app, tmp_path):
    slug = create(app)
    project = app.project_path(slug)
    run = make_run(app, slug)
    base = f"/api/projects/{slug}/runs/example"
    secret = tmp_path / "secret"
    secret.write_text("do not expose")
    for name in (
        "run.json",
        "task.json",
        "program.md",
        "round1_sort.py.bak",
        "../../secret",
        "%2e%2e",
        "other.py",
    ):
        assert request(app, base + "/files/" + name)["status"] in (400, 404)
    for name in ("best_sort.py", "events.jsonl", "PAUSE", "inbox.jsonl"):
        (run / name).unlink(missing_ok=True)
        (run / name).symlink_to(secret)
        if name == "PAUSE":
            with pytest.raises(ValueError):
                app.info(run)
        elif name == "inbox.jsonl":
            with pytest.raises(ValueError):
                server.append_idea(run, {"hypothesis": "x"})
        else:
            assert request(app, base + "/files/" + name)["status"] == 400
        (run / name).unlink()
    (project / "program.md").unlink()
    (project / "program.md").symlink_to(secret)
    assert request(app, f"/api/projects/{slug}")["status"] == 400
    assert (
        request(
            app, f"/api/projects/{slug}/program", {"program": "overwrite"}, method="PUT"
        )["status"]
        == 400
    )
    assert secret.read_text() == "do not expose"
    (app.projects / "linked").symlink_to(project, target_is_directory=True)
    with pytest.raises(ValueError):
        app.project_path("linked")
    project.joinpath("runs").rename(project / "saved-runs")
    project.joinpath("runs").symlink_to(
        project / "saved-runs", target_is_directory=True
    )
    with pytest.raises(ValueError):
        app.run_path(project, "example")


@pytest.mark.parametrize(
    "method,path,data",
    [
        ("POST", "/api/projects", {"title": "x", "template": "sort-speed"}),
        ("PUT", "/api/projects/sorting-study/program", {"program": "new"}),
    ],
)
def test_cross_origin_writes_rejected(app, method, path, data):
    assert (
        request(app, path, data, method, {"Origin": "https://elsewhere.invalid"})[
            "status"
        ]
        == 403
    )


def test_model_configuration():
    assert list(
        server.parse_models(
            "q35=http://localhost:8482/v1,q27=http://localhost:8481/v1", "", ""
        )
    ) == ["q35", "q27"]
    assert server.parse_models(None, "http://localhost:8000/v1/", "custom-model") == {
        "custom-model": "http://localhost:8000/v1"
    }
    for value in (
        "missing-url",
        "q35=file:///tmp",
        "q35=http://a,q35=http://b",
        "=http://a",
    ):
        with pytest.raises(ValueError):
            server.parse_models(value, "", "")


@pytest.fixture
def upstream():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    received = []

    class Stub(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body["_path"] = self.path
            received.append(body)
            self.send_response(200)
            if body.get("stream"):
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for text in ("A concrete ", "testable idea."):
                    self.wfile.write(
                        (
                            "data: "
                            + json.dumps({"choices": [{"delta": {"content": text}}]})
                            + "\n\n"
                        ).encode()
                    )
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                # The loop should consume two queued ideas without proposing any.
                prompt = body["messages"][-1]["content"]
                assert "Propose the single" not in prompt
                old, new = (
                    ("VALUE = 10", "VALUE = 8")
                    if "VALUE = 10" in prompt
                    else ("VALUE = 8", "VALUE = 8.5")
                )
                import re

                kind = body["response_format"]["json_schema"]["name"]
                if kind == "edit":
                    content = dict(
                        base_sha256=re.search(r"base_sha256: ([a-f0-9]+)", prompt)[1],
                        edits=[dict(search_lines=[old], replace_lines=[new])],
                    )
                elif kind == "conclusion":
                    content = dict(
                        hypothesis_id=re.search(r"conclusion for (h\d+)", prompt)[1],
                        summary_md="Evidence recorded.",
                    )
                else:
                    content = dict(summary_md="Run complete.", next_steps_md="Repeat.")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {
                            "choices": [
                                {
                                    "finish_reason": "stop",
                                    "message": {"content": json.dumps(content)},
                                }
                            ]
                        }
                    ).encode()
                )

    instance = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{instance.server_port}/v1", received
    instance.shutdown()
    instance.server_close()
    thread.join()


def test_streamed_chat_model_routing_persistence_and_context(app, upstream):
    url, received = upstream
    app.models["q27"] = url
    slug = create(app)
    make_run(
        app,
        slug,
        [
            {"event": "baseline", "metric": "runtime_ms", "result": {"runtime_ms": 10}},
            {"event": "idea", "round": 1, "change": "Fewer comparisons"},
            {"event": "result", "round": 1, "value": 8, "verdict": "supported"},
        ],
    )
    response = request(
        app, f"/api/projects/{slug}/chat", {"message": "What next?", "model": "q27"}
    )
    assert response["status"] == 200 and b"testable idea." in response["body"]
    assert request(app, f"/api/projects/{slug}")["json"]["project"]["model"] == "q27"
    body = received[-1]
    assert body["model"] == "q27" and body["stream"] is True
    context = body["messages"][0]["content"]
    assert (
        "Reduce runtime" in context
        and "runtime_ms" in context
        and "supported" in context
        and "Fewer comparisons" in context
    )
    assert "curve" not in context
    messages = request(app, f"/api/projects/{slug}")["json"]["discussion"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert (
        messages[1]["content"] == "A concrete testable idea."
        and messages[1]["model"] == "q27"
    )
    request(app, f"/api/projects/{slug}/chat", {"message": "One more", "model": "q27"})
    assert received[-1]["messages"][-2]["role"] == "assistant"
    app.models["q35"] = url + "/model35"
    request(
        app,
        f"/api/projects/{slug}/chat",
        {"message": "Compare another model", "model": "q35"},
    )
    assert received[-1]["model"] == "q35"
    assert received[-1]["_path"] == "/v1/model35/chat/completions"
    assert received[-2]["_path"] == "/v1/chat/completions"
    assert (
        request(
            app, f"/api/projects/{slug}/chat", {"message": "x", "model": "unknown"}
        )["status"]
        == 400
    )


def test_loop_consumes_project_queue_first_and_keeps_min_verdicts(tmp_path, upstream):
    import subprocess
    import sys

    url, received = upstream
    project = tmp_path / "project"
    project.mkdir()
    (project / "program.md").write_text("Reduce runtime. Keep budget fixed.")
    (project / "task.json").write_text(
        json.dumps(
            {
                "title": "Test runtime",
                "editable": "sort.py",
                "metric": "runtime_ms",
                "direction": "min",
                "noise": 1,
                "fixed": [],
                "seeds": [0, 1],
            }
        )
    )
    (project / "sort.py").write_text(
        'import json\nVALUE = 10\nprint("RESULT " + json.dumps({"runtime_ms": VALUE}))\n'
    )
    (project / "ideas.jsonl").write_text(
        "".join(
            json.dumps({"hypothesis": f"Researcher idea {i}", "change": f"Change {i}"})
            + "\n"
            for i in (1, 2)
        )
    )
    run = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            str(server.ROOT / "loop.py"),
            str(project),
            str(run),
            "--base-url",
            url,
            "--model",
            "q27",
            "--rounds",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    events = [
        json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()
    ]
    ideas = [e for e in events if e["event"] == "idea"]
    assert [e["hypothesis"] for e in ideas] == [
        "Researcher idea 1",
        "Researcher idea 2",
    ]
    assert all(e["source"] == "you" for e in ideas)
    assert [e["verdict"] for e in events if e["event"] == "result"] == [
        "improved",
        "no effect",
    ]
    assert len(server.hypotheses.read_rows(project / "hypotheses.jsonl")) == 2
    assert "VALUE = 8\n" in (run / "best_sort.py").read_text()
    assert [r["response_format"]["json_schema"]["name"] for r in received] == [
        "edit",
        "edit",
        "conclusion",
        "conclusion",
        "run_report",
    ]


def test_baseline_download_when_no_change_was_kept(app):
    slug = create(app)
    make_run(
        app,
        slug,
        [{"event": "baseline", "metric": "runtime_ms", "result": {"runtime_ms": 10}}],
    )
    result = request(
        app, f"/api/projects/{slug}/runs/example/files/best_sort.py?download=1"
    )
    assert result["body"] == b"VALUE = 10\n"
    assert (
        result["headers"]["Content-Disposition"]
        == 'attachment; filename="best_sort.py"'
    )


def test_loop_carries_untried_ideas_to_next_run(tmp_path, upstream):
    import subprocess
    import sys

    url, _ = upstream
    project = tmp_path / "project"
    project.mkdir()
    (project / "program.md").write_text("Reduce runtime.")
    (project / "task.json").write_text(
        json.dumps(
            {
                "title": "Runtime",
                "editable": "sort.py",
                "metric": "runtime_ms",
                "direction": "min",
                "noise": 1,
                "fixed": [],
                "seeds": [0],
            }
        )
    )
    (project / "sort.py").write_text(
        'import json\nVALUE = 10\nprint("RESULT " + json.dumps({"runtime_ms": VALUE}))\n'
    )
    ideas = [{"hypothesis": f"Idea {i}"} for i in (1, 2)]
    (project / "ideas.jsonl").write_text(
        "".join(json.dumps(idea) + "\n" for idea in ideas)
    )
    result = subprocess.run(
        [
            sys.executable,
            str(server.ROOT / "loop.py"),
            str(project),
            str(tmp_path / "run"),
            "--base-url",
            url,
            "--rounds",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    claims = server.hypotheses.collect(
        project,
        json.loads((project / "task.json").read_text()),
        [tmp_path / "run" / "events.jsonl"],
    )
    assert claims[1]["statement"] == "Idea 2" and claims[1]["status"] == "untested"
    assert claims[1]["source"] == "you"


def test_http_asset_and_api_framing(app):
    """A real local HTTP roundtrip covers framing and static font delivery."""
    import threading
    from urllib.request import Request, urlopen

    thread = threading.Thread(target=app.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{app.server_port}"
    try:
        for path, mime in (
            ("/", "text/html"),
            ("/app.js", "text/javascript"),
            ("/style.css", "text/css"),
            ("/fonts/PublicSans.ttf", "font/ttf"),
        ):
            with urlopen(base + path) as response:
                body = response.read()
                assert response.headers["Content-Type"] == mime
                assert int(response.headers["Content-Length"]) == len(body) and body
        data = json.dumps(
            {
                "title": "HTTP project",
                "template": "sort-speed",
                "goal": "Measure runtime",
            }
        ).encode()
        with urlopen(
            Request(base + "/api/projects", data, {"Content-Type": "application/json"})
        ) as response:
            assert response.status == 201
            assert json.load(response)["project"]["slug"] == "http-project"
    finally:
        app.shutdown()
        thread.join()


@pytest.mark.parametrize(
    ("direction", "baseline", "improved", "nearby", "worse"),
    [("min", 100, 80, 79.9, 110), ("max", 0.1, 0.8, 0.801, 0.05)],
)
def test_project_sparkline_summary(
    app, monkeypatch, direction, baseline, improved, nearby, worse
):
    slug = create(app)
    project = app.project_path(slug)
    task_path = project / "task.json"
    task = json.loads(task_path.read_text())
    task["direction"] = direction
    task_path.write_text(json.dumps(task))
    (project / "program.md").write_text(
        "\n# Goal\n\nResearch question\n\nConstraints below.\n"
    )
    empty = app.summary(project)
    assert empty["progress"] == [] and empty["status"] == "No runs"
    assert empty["description"] == "Research question"
    run = make_run(
        app,
        slug,
        [
            {"event": "baseline", "result": {"runtime_ms": baseline}},
            {"event": "result", "value": improved, "verdict": "supported"},
            {"event": "result", "value": nearby, "verdict": "no effect"},
            {"event": "result", "value": worse, "verdict": "refuted"},
            {"event": "result", "verdict": "crashed"},
        ],
    )
    (run / "run.json").write_text(json.dumps({"created": 10}))
    summary = app.summary(project)
    assert summary["progress"] == [baseline, improved, improved, improved, improved]
    assert summary["status"] == "Finished"
    # A later run's weaker baseline cannot reset the project's running best.
    later = project / "runs" / "later"
    later.mkdir()
    (later / "run.json").write_text(json.dumps({"created": 20}))
    (later / "events.jsonl").write_text(
        json.dumps({"event": "baseline", "result": {"runtime_ms": worse}}) + "\n"
    )
    assert app.summary(project)["progress"][-1] == improved
    monkeypatch.setattr(app, "active", lambda path: path == later)
    assert app.summary(project)["status"] == "Running"
    # GET summaries leave all source artifacts unchanged.
    original = (run / "events.jsonl").read_bytes()
    response = request(app, "/api/projects")["json"]["projects"][0]
    assert response["progress"] == [baseline] + [improved] * 5
    assert (run / "events.jsonl").read_bytes() == original


def test_active_project_counts_finished_rounds_across_runs(app, monkeypatch):
    slug = create(app)
    project = app.project_path(slug)
    run = make_run(
        app,
        slug,
        [
            {
                "event": "baseline",
                "metric": "runtime_ms",
                "result": {"runtime_ms": 100},
            },
            {"event": "result", "round": 1, "value": 80, "verdict": "supported"},
            {"event": "result", "round": 2, "verdict": "crashed"},
        ],
    )
    later = project / "runs" / "active"
    later.mkdir()
    (later / "run.json").write_text(json.dumps({"created": 200}))
    events = [
        {"event": "baseline", "metric": "runtime_ms", "result": {"runtime_ms": 100}},
        {"event": "result", "round": 1, "value": 70, "verdict": "supported"},
        {"event": "result", "round": 2, "value": 71, "verdict": "no effect"},
        {"event": "run_start", "round": 3, "seed": 0},
    ]
    (later / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (run / "run.json").write_text(json.dumps({"created": 100}))
    monkeypatch.setattr(app, "active", lambda path: path == later)
    summary = app.summary(project)
    assert (
        summary["experiments"] == 4
    )  # Includes completed crashes, excludes baseline/in-flight.
    assert summary["status"] == "Running"
    assert summary["best"] == 70
    assert summary["progress"] == [100, 80, 80, 80, 70, 70]
    assert summary["task"]["domain"] == "Computer science"


def test_project_hypotheses_api_and_counts_during_legacy_run(app):
    slug = create(app)
    run = make_run(
        app,
        slug,
        [
            {
                "event": "idea",
                "round": 1,
                "hypothesis": "Search multiple basins",
                "change": "Try 20 starts",
            },
            {"event": "result", "round": 1, "verdict": "supported", "value": 8},
        ],
    )
    before = (run / "events.jsonl").read_bytes()
    response = request(
        app,
        f"/api/projects/{slug}/hypotheses",
        {
            "statement": "Regularization reduces error",
            "why": "Less overfitting",
            "prediction": "Lower error",
            "source": "discussion",
            "status": "supported",
        },
    )
    assert response["status"] == 201
    assert response["json"]["hypothesis"]["id"] == "h1"
    claims = response["json"]["hypotheses"]
    assert [h["status"] for h in claims] == ["untested", "supported"]
    assert (
        claims[0]["why"] == "Less overfitting"
        and claims[0]["prediction"] == "Lower error"
    )
    detail = request(app, f"/api/projects/{slug}")["json"]
    assert (
        detail["project"]["hypotheses"] == 2 and detail["project"]["experiments"] == 1
    )
    assert (run / "events.jsonl").read_bytes() == before
    assert not (run / "inbox.jsonl").exists()
    for data in (
        {"statement": "x", "source": "invalid"},
        {"statement": "x", "why": []},
        {"statement": "x", "prediction": 4},
    ):
        assert request(app, f"/api/projects/{slug}/hypotheses", data)["status"] == 400


def test_new_experiment_verdicts_keep_best_summaries_and_download(app):
    slug = create(app)
    run = make_run(
        app,
        slug,
        [
            {
                "event": "baseline",
                "round": 0,
                "metric": "runtime_ms",
                "result": {"runtime_ms": 10},
            },
            {"event": "idea", "round": 1, "hypothesis_id": "h1", "hypothesis": "Claim"},
            {
                "event": "result",
                "round": 1,
                "hypothesis_id": "h1",
                "verdict": "improved",
                "value": 8,
            },
            {"event": "result", "round": 2, "verdict": "worse", "value": 20},
        ],
    )
    assert app.info(run)["best"] == 8
    assert app.summary(app.project_path(slug))["progress"] == [10, 8, 8]
    assert (
        app.artifact(app.project_path(slug), run, "best_sort.py").name
        == "round1_sort.py"
    )


def test_protocol_list_then_subscribe_cursor_and_bundle(app):
    import zipfile
    from io import BytesIO

    import protocol

    slug = create(app)
    run = make_run(app, slug)
    (run / "events.jsonl").write_text("")
    journal = protocol.Journal(run)
    journal.append("baseline", metric="runtime_ms", result=dict(runtime_ms=10))
    owner = dict(scope="run", run="example")
    store = protocol.Artifacts(run)
    artifact = store.register(
        b"x,y\n1,2\n",
        id="data",
        kind="data",
        role="coordinates",
        title="Coordinates",
        path="coords.csv",
        owner=owner,
    )
    base = f"/api/projects/{slug}/runs/example"
    listed = request(app, base + "/artifacts")["json"]
    assert listed["watermark"] == 2 and listed["artifacts"][0] == artifact
    journal.append("done", best=10)
    resumed = request(app, base + "/events?after=2")
    assert b"id: 3\n" in resumed["body"] and b"id: 2\n" not in resumed["body"]
    header = request(app, base + "/events?after=0", headers={"Last-Event-ID": "2"})
    assert header["body"] == resumed["body"]
    rows = request(app, base + "/artifacts/data?rows=999")["json"]
    assert rows["rows"] == [dict(x="1", y="2")] and rows["watermark"] == 3
    downloaded = request(app, base + "/bundle.zip")
    assert downloaded["status"] == 200
    archive = zipfile.ZipFile(BytesIO(downloaded["body"]))
    assert archive.read("artifacts/data-r1.csv") == b"x,y\n1,2\n"
    assert request(app, base + "/records")["json"]["watermark"] == 3


def test_protocol_svg_csp_and_unknown_kind_download(app):
    import protocol

    slug = create(app)
    run = make_run(app, slug)
    (run / "events.jsonl").write_text("")
    store = protocol.Artifacts(run)
    artifact = store.register(
        b'<svg xmlns="http://www.w3.org/2000/svg"/>',
        id="picture",
        kind="image",
        title="Projection",
        path="plot.svg",
        owner=dict(scope="run", run="example"),
    )
    base = f"/api/projects/{slug}/runs/example"
    served = request(app, base + "/artifacts/picture")
    assert served["headers"]["Content-Type"] == "image/svg+xml"
    assert "sandbox" in served["headers"]["Content-Security-Policy"]
    assert served["headers"]["X-Content-Type-Options"] == "nosniff"
    unknown = dict(artifact, id="future", kind="molecule")
    store.journal.append("artifact", artifact=unknown)
    invalid = dict(artifact, id="invalid", protocol=999)
    store.journal.append("artifact", artifact=invalid)
    response = request(app, base + "/artifacts")["json"]
    for row in response["artifacts"][1:]:
        assert row["kind"] == "file" and row["warning"]
        download = request(app, base + "/artifacts/" + row["id"])
        assert download["headers"]["Content-Type"] == "application/octet-stream"
        assert "attachment" in download["headers"]["Content-Disposition"]


def test_invalid_envelope_does_not_mask_other_artifacts_or_break_bundle(app):
    import zipfile
    from io import BytesIO

    import protocol

    slug = create(app)
    run = make_run(app, slug)
    (run / "events.jsonl").write_text("")
    store = protocol.Artifacts(run)
    store.register(
        b"plain log",
        id="valid",
        kind="log",
        title="Log",
        path="log.txt",
        owner=dict(scope="run", run="example"),
    )
    store.journal.append("artifact", artifact=None)
    store.journal.append("artifact", artifact={"id": "bad", "owner": "bad owner"})
    base = f"/api/projects/{slug}/runs/example"
    assert request(app, base + "/artifacts/valid")["body"] == b"plain log"
    assert len(request(app, base + "/artifacts?kind=log")["json"]["artifacts"]) == 1
    archive = zipfile.ZipFile(BytesIO(request(app, base + "/bundle.zip")["body"]))
    assert archive.read("artifacts/valid-r1.txt") == b"plain log"
    assert (
        len([name for name in archive.namelist() if name.startswith("warnings/")]) == 2
    )


def test_legacy_run_with_string_task_uses_project_task():
    import server

    assert server.run_task({"task": "grokking"}) is None
    assert server.run_task({"task": {"editable": "train.py"}}) == {
        "editable": "train.py"
    }
    assert server.run_task(None) is None
