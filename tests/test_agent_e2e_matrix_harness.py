"""The release harness must not silently invent request or config evidence."""

import asyncio
import io
import json
import signal
from types import SimpleNamespace

from scripts.agent_e2e.instrument.sitecustomize import _Tap
from scripts.agent_e2e.run import (
    approval_handler,
    classified_error,
    handle_first_run_menus,
    output_issues,
    preserved,
    read_until,
    reply_finished,
    request_evidence,
    send_prompt,
    snapshot,
    stop_terminal_process,
)
from scripts.agent_e2e.seed_home import FILES


def test_qwen_tool_approval_answers_once_per_prompt(monkeypatch):
    sent = []
    monkeypatch.setattr("os.write", lambda _fd, data: sent.append(data))
    approve = approval_handler("qwen-code", 10)
    approve("Waiting for user confirmation")
    approve("Waiting for user confirmation")
    approve("Waiting for user confirmation... Waiting for user confirmation")
    assert sent == [b"\r", b"\r"]
    assert approval_handler("pi", 10) is None


def test_crush_prompt_uses_bracketed_paste(monkeypatch):
    sent = []
    monkeypatch.setattr("os.write", lambda _fd, data: sent.append(data))
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    send_prompt("crush", 10, "hello")
    assert sent == [b"\x1b[200~hello\x1b[201~", b"\r"]


def test_crush_tool_approval_is_answered_once_per_modal(monkeypatch):
    sent = []
    monkeypatch.setattr("os.write", lambda _fd, data: sent.append(data))
    monkeypatch.setattr("time.monotonic", lambda: 10.0)
    approve = approval_handler("crush", 10)
    approve("Allow      Allow for Session      Deny")
    approve("Allow      Allow for Session      Deny")
    assert sent == [b"\r"]


def test_aider_file_creation_is_confirmed_once(monkeypatch):
    sent = []
    monkeypatch.setattr("os.write", lambda _fd, data: sent.append(data))
    approve = approval_handler("aider", 10)
    approve("Create new file? (Y)es/(N)o [Yes]:")
    approve("Create new file? (Y)es/(N)o [Yes]:")
    assert sent == [b"y\r"]


def test_terminal_timeout_never_passes_negative_wait_to_select(monkeypatch):
    times = iter((0.0, 0.5, 1.1, 1.2))
    monkeypatch.setattr("time.monotonic", lambda: next(times))
    waits = []

    def select_once(_reads, _writes, _errors, timeout):
        waits.append(timeout)
        return [], [], []

    monkeypatch.setattr("select.select", select_once)
    output, first = read_until(1, SimpleNamespace(poll=lambda: None), 1, io.BytesIO())
    assert (output, first, waits) == ("", None, [0])


def test_snapshot_only_tracks_preexisting_user_config(tmp_path):
    path = tmp_path / ".codex/config.toml"
    path.parent.mkdir()
    path.write_text(FILES[".codex/config.toml"])
    before = snapshot(tmp_path, "codex")
    (path.parent / "session.json").write_text("{}")
    (tmp_path / ".rapid-mlx").mkdir()
    assert snapshot(tmp_path, "codex") == before
    assert preserved(before, dict(before, **{".codex/session.json": "new"}))
    path.write_text(path.read_text() + "# changed\n")
    assert not preserved(before, snapshot(tmp_path, "codex"))


def test_request_evidence_never_substitutes_config_model(tmp_path):
    assert request_evidence(None, "pi", 0, 100)["model_requested"] is None
    log = tmp_path / "requests.jsonl"
    log.write_text(
        json.dumps(
            {
                "agent": "pi",
                "time": 10,
                "path": "/v1/chat/completions",
                "model": "actual-request",
                "status": 200,
                "prompt_tokens": 1234,
            }
        )
        + "\n"
    )
    assert request_evidence(log, "pi", 0, 100) == {
        "model_requested": "actual-request",
        "auth_accepted": True,
        "first_prompt_tokens": 1234,
    }
    log.write_text(
        json.dumps(
            {
                "agent": "codex",
                "time": 10,
                "path": "/v1/responses",
                "model": "actual-request",
                "status": 200,
                "prompt_tokens": 88,
            }
        )
        + "\n"
    )
    assert request_evidence(log, "codex", 0, 100)["model_requested"] == "actual-request"
    log.write_text(
        json.dumps(
            {
                "agent": "codex",
                "time": 11,
                "path": "/v1/responses",
                "model": "later",
                "status": 200,
                "prompt_tokens": 12,
            }
        )
        + "\n"
        + json.dumps(
            {
                "agent": "codex",
                "time": 10,
                "path": "/v1/responses",
                "model": "first",
                "status": 200,
                "prompt_tokens": 34,
            }
        )
        + "\n"
    )
    assert request_evidence(log, "codex", 0, 100)["first_prompt_tokens"] == 34


def test_metadata_tap_forwards_request_and_response_unchanged(tmp_path, monkeypatch):
    log = tmp_path / "requests.jsonl"
    current = tmp_path / "agent"
    current.write_text("pi\n")
    monkeypatch.setenv("AGENT_E2E_REQUEST_LOG", str(log))
    monkeypatch.setenv("AGENT_E2E_CURRENT_AGENT", str(current))
    original = b'{"model":"actual-request","messages":[{"content":"private prompt"}]}'
    response = b'{"usage":{"prompt_tokens":456},"choices":[]}'
    seen = []
    sent = []

    async def app(scope, receive, send):
        seen.append(await receive())
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": response})

    async def receive():
        return {"type": "http.request", "body": original, "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(
        _Tap(app)({"type": "http", "path": "/v1/chat/completions"}, receive, send)
    )
    assert seen[0]["body"] == original
    assert sent[-1]["body"] == response
    logged = log.read_text()
    assert "private prompt" not in logged
    event = json.loads(logged)
    assert event["agent"] == "pi"
    assert event["model"] == "actual-request"
    assert event["prompt_tokens"] == 456
    assert event["status"] == 200


def test_metadata_tap_reads_streamed_responses_usage(tmp_path, monkeypatch):
    log = tmp_path / "requests.jsonl"
    current = tmp_path / "agent"
    current.write_text("codex\n")
    monkeypatch.setenv("AGENT_E2E_REQUEST_LOG", str(log))
    monkeypatch.setenv("AGENT_E2E_CURRENT_AGENT", str(current))

    async def app(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send(
            {
                "type": "http.response.body",
                "body": (
                    b'data: {"type":"response.completed","response":{"usage":{"input_tokens":789}}}\n\n'
                ),
            }
        )

    async def receive():
        return {
            "type": "http.request",
            "body": b'{"model":"actual"}',
            "more_body": False,
        }

    async def send(message):
        pass

    asyncio.run(_Tap(app)({"type": "http", "path": "/v1/responses"}, receive, send))
    assert json.loads(log.read_text())["prompt_tokens"] == 789


def test_output_issue_summary_uses_private_transcript(tmp_path):
    transcript = tmp_path / "agent.pty"
    transcript.write_bytes(b"\x1b[31mWarning: retrying\x1b[0m\n")
    assert output_issues(transcript) == ["warning"]


def test_shareable_error_does_not_copy_credentials():
    assert classified_error("sk-test-secret denied") == "see private transcript"
    assert classified_error("Unrecognized flag: --model") == (
        "unsupported --model flag"
    )


def test_goose_telemetry_choice_is_answered_before_task(monkeypatch):
    writes = []
    monkeypatch.setattr(
        "scripts.agent_e2e.run.os.write", lambda fd, data: writes.append(data)
    )
    monkeypatch.setattr(
        "scripts.agent_e2e.run.read_until",
        lambda *args: ("goose is ready", None),
    )
    opening = "Share anonymous usage data to help improve goose?"
    assert "goose is ready" in handle_first_run_menus("goose", opening, 1, None, None)
    assert writes == [b"\r"]


def test_aider_repo_creation_prompt_is_declined(monkeypatch):
    writes = []
    monkeypatch.setattr(
        "scripts.agent_e2e.run.os.write", lambda fd, data: writes.append(data)
    )
    monkeypatch.setattr(
        "scripts.agent_e2e.run.read_until",
        lambda *args, **kwargs: ("Aider ready", None),
    )
    opening = "No git repo found, create one to track aider's changes?"
    assert "Aider ready" in handle_first_run_menus("aider", opening, 1, None, None)
    assert writes == [b"n\r"]


def test_aider_documentation_prompt_is_declined(monkeypatch):
    writes = []
    monkeypatch.setattr(
        "scripts.agent_e2e.run.os.write", lambda fd, data: writes.append(data)
    )
    monkeypatch.setattr(
        "scripts.agent_e2e.run.read_until",
        lambda *args, **kwargs: ("Open documentation url for more info?", None),
    )
    opening = "No git repo found, create one to track aider's changes?"
    handle_first_run_menus("aider", opening, 1, None, None)
    assert writes == [b"n\r", b"n\r"]


def test_aider_waits_for_delayed_git_question_before_task(monkeypatch):
    writes = []
    screens = iter(
        ["No git repo found, create one to track changes?", "Aider ready", ""]
    )

    def read_screen(*args, **kwargs):
        screen = next(screens)
        if stop_when := kwargs.get("stop_when"):
            if screen:
                assert stop_when(screen)
        return screen, None

    monkeypatch.setattr(
        "scripts.agent_e2e.run.os.write", lambda fd, data: writes.append(data)
    )
    monkeypatch.setattr("scripts.agent_e2e.run.read_until", read_screen)
    opening = "Agent ready."
    assert "Aider ready" in handle_first_run_menus("aider", opening, 1, None, None)
    assert writes == [b"n\r"]


def test_aider_delayed_analytics_question_is_declined(monkeypatch):
    writes = []
    screens = iter(
        [
            "Allow collection of anonymous analytics to help improve aider?",
            "Aider ready",
        ]
    )

    def read_screen(*args, **kwargs):
        screen = next(screens)
        if stop_when := kwargs.get("stop_when"):
            assert stop_when(screen)
        return screen, None

    monkeypatch.setattr(
        "scripts.agent_e2e.run.os.write", lambda fd, data: writes.append(data)
    )
    monkeypatch.setattr("scripts.agent_e2e.run.read_until", read_screen)
    opening = "No git repo found, create one to track aider's changes?"
    assert "Aider ready" in handle_first_run_menus("aider", opening, 1, None, None)
    assert writes == [b"n\r", b"n\r"]


def test_crush_project_initialization_is_declined(monkeypatch):
    writes = []
    monkeypatch.setattr(
        "scripts.agent_e2e.run.os.write", lambda fd, data: writes.append(data)
    )
    monkeypatch.setattr(
        "scripts.agent_e2e.run.read_until",
        lambda *args, **kwargs: ("Crush ready", None),
    )
    opening = "Would you like to initialize now? Yep! Nope"
    assert "Crush ready" in handle_first_run_menus("crush", opening, 1, None, None)
    assert writes == [b"\x1b[C\r"]


def test_goose_waits_for_turn_completion_after_tool_output():
    assert not reply_finished("goose", "tool result: verify-123")
    assert reply_finished("goose", "tool result: verify-123\n⏱ 3.2s")
    assert reply_finished("pi", "tool result: verify-123")


def test_pty_cleanup_falls_back_when_process_group_is_inaccessible(monkeypatch):
    class Process:
        pid = 123
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout):
            return 0

    def denied(*args):
        raise PermissionError("process group changed during shutdown")

    monkeypatch.setattr("scripts.agent_e2e.run.os.killpg", denied)
    proc = Process()
    stop_terminal_process(proc)
    assert proc.terminated


def test_pty_cleanup_signals_group_after_launcher_exits(monkeypatch):
    signals = []

    class Process:
        pid = 12345

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(
        "scripts.agent_e2e.run.os.killpg",
        lambda pid, sig: signals.append((pid, sig)),
    )
    stop_terminal_process(Process())
    assert signals == [
        (12345, signal.SIGTERM),
        (12345, 0),
        (12345, signal.SIGKILL),
    ]
