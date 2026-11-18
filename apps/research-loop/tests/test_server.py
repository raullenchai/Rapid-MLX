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


def request_handler(app, path, data):
    from io import BytesIO

    handler = object.__new__(server.Handler)
    body = json.dumps(data).encode()
    handler.server, handler.path = app, path
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile, handler.wfile = BytesIO(body), BytesIO()
    handler.send_response = lambda status: None
    handler.send_header = lambda *args: None
    handler.end_headers = lambda: None
    return handler


def local_app(tmp_path):
    app = object.__new__(server.App)
    app.runs, app.processes = tmp_path, {}
    app.model, app.rounds, app.base_url = "q35", 2, "http://localhost:8482/v1"
    return app


def test_create_run_uses_absolute_paths_and_current_python(tmp_path, monkeypatch):
    from types import SimpleNamespace

    calls = []
    monkeypatch.setattr(
        server.subprocess,
        "Popen",
        lambda args, **kw: calls.append(args) or SimpleNamespace(poll=lambda: None),
    )
    handler = request_handler(local_app(tmp_path), "/api/runs", {"task": "grokking"})
    handler.do_POST()
    meta = json.loads(handler.wfile.getvalue())
    assert meta["rounds"] == 2
    assert calls[0][0] == server.sys.executable
    assert Path(calls[0][2]).is_absolute() and Path(calls[0][3]).is_absolute()
    assert calls[0][-2:] == ["--rounds", "2"]


def test_stop_unpauses_without_killing_process(tmp_path):
    run = tmp_path / "example"
    run.mkdir()
    (run / "PAUSE").touch()
    handler = request_handler(local_app(tmp_path), "/api/runs/example/stop", {})
    handler.do_POST()
    assert json.loads(handler.wfile.getvalue())["ok"]
    assert (run / "STOP").exists()
    assert not (run / "PAUSE").exists()


@pytest.mark.parametrize("name", ["../outside", "/tmp", "", "a/b"])
def test_invalid_run_path(tmp_path, name):
    with pytest.raises(ValueError):
        local_app(tmp_path).run_path(name)
