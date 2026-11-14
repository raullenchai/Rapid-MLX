"""The release harness must not silently invent request or config evidence."""

import asyncio
import io
import json
import signal
import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.agent_e2e import run
from scripts.agent_e2e.instrument.sitecustomize import _Tap
from scripts.agent_e2e.run import (
    approval_handler,
    check_then_open_native,
    classified_error,
    handle_first_run_menus,
    is_opencode_v2,
    opencode_reply_seen,
    output_issues,
    preserved,
    rapid_config_untouched,
    read_until,
    reply_finished,
    request_evidence,
    send_prompt,
    snapshot,
    stop_terminal_process,
)
from scripts.agent_e2e.seed_home import FILES


def test_opencode_evidence_reader_follows_cli_major_version():
    assert is_opencode_v2("opencode v2.1.2")
    assert is_opencode_v2("2.1.2")
    assert not is_opencode_v2("opencode v1.2.27")
    assert not is_opencode_v2("opencode 1.2.3")


@pytest.fixture
def opencode_v1_db(tmp_path):
    db = tmp_path / ".rapid-mlx/connect/agents/opencode/xdg-data/opencode/opencode.db"
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as conn:
        conn.executescript(
            "CREATE TABLE message (id TEXT, session_id TEXT, time_created INTEGER, "
            "time_updated INTEGER, data TEXT);"
            "CREATE TABLE part (id TEXT, message_id TEXT, session_id TEXT, "
            "time_created INTEGER, time_updated INTEGER, data TEXT);"
        )
        conn.execute(
            "INSERT INTO message VALUES (?, ?, ?, ?, ?)",
            (
                "m1",
                "s1",
                11000,
                12000,
                json.dumps(
                    {
                        "role": "assistant",
                        "time": {"created": 11000, "completed": 12000},
                    }
                ),
            ),
        )
        parts = [
            {
                "type": "tool",
                "tool": "read",
                "state": {
                    "status": "completed",
                    "input": {"filePath": f"/work/{path}"},
                    "output": f"<content>1: {value}</content>",
                    "time": {"start": 11000, "end": 12000},
                },
            }
            for path, value in (("hello.txt", "hi-abc"), ("nonce.txt", "nonce-def"))
        ]
        parts.append(
            {
                "type": "text",
                "text": "hi-abc nonce-def",
                "time": {"start": 11000, "end": 12000},
            }
        )
        conn.executemany(
            "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
            [
                (f"p{i}", "m1", "s1", 11000, 12000, json.dumps(part))
                for i, part in enumerate(parts)
            ],
        )
    return db


@pytest.mark.parametrize("separate_reply", [False, True])
def test_opencode_v1_reply_ignores_fragmented_tui(
    tmp_path, opencode_v1_db, separate_reply
):
    if separate_reply:
        with sqlite3.connect(opencode_v1_db) as conn:
            conn.execute(
                "INSERT INTO message SELECT 'm2', session_id, time_created, "
                "time_updated, data FROM message"
            )
            conn.execute("UPDATE part SET message_id = 'm2' WHERE id = 'p2'")
    output = "hi-abc nonce-d\x1b[3D⬝⬝⬝⬝⬝⬝⬝⬝ef"
    assert "nonce-def" not in output
    assert run.opencode_v1_reply_seen(tmp_path, 10, "hi-abc", "nonce-def", output)
    assert not run.opencode_v1_reply_seen(tmp_path, 10, "hi-abc", "nonce-wrong", output)


@pytest.mark.parametrize("major", [1, 2])
@pytest.mark.parametrize(
    "case, expected",
    [
        ("cat", True),
        ("missing_nonce", False),
        ("running", False),
        ("command_only_nonce", False),
        ("non_string_output", False),
        ("read_hello_bash_nonce", True),
        ("bash_hello_read_nonce", True),
        ("separate_bash_outputs", True),
        ("other_session", False),
        ("stale", False),
        ("incomplete_reply", False),
        ("reply_missing_nonce", False),
        ("other_tool", False),
    ],
)
def test_opencode_bash_file_evidence(tmp_path, opencode_v1_db, major, case, expected):
    bash = {
        "type": "tool",
        "tool": "bash",
        "state": {
            "status": "running" if case == "running" else "completed",
            "input": {
                "command": (
                    "printf 'hi-abc nonce-def'"
                    if case == "command_only_nonce"
                    else "cat /work/hello.txt /work/nonce.txt"
                )
            },
            "output": "hi-abc\nnonce-def\n",
        },
    }
    if case in {"missing_nonce", "command_only_nonce", "bash_hello_read_nonce"}:
        bash["state"]["output"] = "hi-abc\n"
    if case == "non_string_output":
        bash["state"]["output"] = ["hi-abc", "nonce-def"]
    if case in {"read_hello_bash_nonce", "separate_bash_outputs"}:
        bash["state"]["output"] = "nonce-def\n"
    if case == "other_tool":
        bash["tool"] = "edit"
    with sqlite3.connect(opencode_v1_db) as conn:
        conn.execute("DELETE FROM part WHERE id = 'p0'")
        conn.execute(
            "UPDATE part SET data = ?, session_id = ?, time_created = ?, "
            "time_updated = ? WHERE id = 'p1'",
            (
                json.dumps(bash),
                "s2" if case == "other_session" else "s1",
                1000 if case == "stale" else 11000,
                2000 if case == "stale" else 12000,
            ),
        )
        conn.execute(
            "INSERT INTO message SELECT 'm2', 's2', time_created, time_updated, "
            "data FROM message WHERE id = 'm1'"
        )
        conn.execute("UPDATE part SET message_id = 'm2' WHERE session_id = 's2'")
        if case in {
            "read_hello_bash_nonce",
            "bash_hello_read_nonce",
            "separate_bash_outputs",
        }:
            path, value = (
                ("nonce.txt", "nonce-def")
                if case == "bash_hello_read_nonce"
                else ("hello.txt", "hi-abc")
            )
            conn.execute(
                "INSERT INTO part VALUES ('p0', 'm1', 's1', 11000, 12000, ?)",
                (
                    json.dumps(
                        {
                            "type": "tool",
                            "tool": "bash"
                            if case == "separate_bash_outputs"
                            else "read",
                            "state": {
                                "status": "completed",
                                "input": {"filePath": f"/work/{path}"},
                                "output": value,
                            },
                        }
                    ),
                ),
            )
        if case == "incomplete_reply":
            conn.execute(
                "UPDATE message SET data = json_remove(data, '$.time.completed')"
            )
        if case == "reply_missing_nonce":
            conn.execute(
                "UPDATE part SET data = json_set(data, '$.text', 'hi-abc') WHERE id = 'p2'"
            )
        if major == 2:
            conn.execute(
                "CREATE TABLE session_message "
                "(session_id TEXT, type TEXT, time_created INTEGER, seq INTEGER, data TEXT)"
            )
            rows = conn.execute(
                "SELECT p.session_id, m.data, p.data, p.time_created FROM part p "
                "JOIN message m ON m.id = p.message_id AND m.session_id = p.session_id"
            ).fetchall()
            for seq, (session_id, raw_message, raw_part, created) in enumerate(rows):
                message, part = json.loads(raw_message), json.loads(raw_part)
                if part["type"] == "tool":
                    part["name"] = part.pop("tool")
                    state = part["state"]
                    state["content"] = [{"text": state.pop("output")}]
                    state["input"]["path"] = state["input"].pop("filePath", "")
                message["content"] = [part]
                conn.execute(
                    "INSERT INTO session_message VALUES (?, 'assistant', ?, ?, ?)",
                    (session_id, created, seq, json.dumps(message)),
                )
    if major == 1:
        actual = run.opencode_v1_reply_seen(tmp_path, 10, "hi-abc", "nonce-def", "")
    else:
        actual = opencode_reply_seen(tmp_path, 10, "hi-abc", "nonce-def")
    assert actual is expected


@pytest.mark.parametrize(
    "change",
    [
        "DELETE FROM part WHERE id = 'p1'",
        "UPDATE part SET data = json_set(data, '$.state.status', 'running') WHERE id = 'p1'",
        "UPDATE part SET data = json_set(data, '$.state.output', 'wrong') WHERE id = 'p1'",
        "UPDATE message SET data = json_remove(data, '$.time.completed')",
        "UPDATE part SET data = json_remove(data, '$.time.end') WHERE id = 'p2'",
        "UPDATE message SET data = json_set(data, '$.role', 'user')",
        "UPDATE part SET data = json_set(data, '$.text', 'hi-abc') WHERE id = 'p2'",
        "UPDATE part SET session_id = 's2', message_id = 'm2' WHERE id = 'p2'",
        "UPDATE part SET time_created = 1000, time_updated = 2000 WHERE id = 'p1'",
        "UPDATE message SET time_created = 1000, time_updated = 2000",
        "UPDATE part SET data = 'invalid json' WHERE id = 'p1'",
        "DROP TABLE part",
    ],
)
def test_opencode_v1_requires_current_completed_evidence(
    tmp_path, opencode_v1_db, change
):
    with sqlite3.connect(opencode_v1_db) as conn:
        conn.execute(
            "INSERT INTO message SELECT 'm2', 's2', time_created, time_updated, data "
            "FROM message WHERE id = 'm1'"
        )
        conn.execute(change)
    # An existing database is authoritative, even if raw PTY contains the nonce.
    assert not run.opencode_v1_reply_seen(
        tmp_path, 10, "hi-abc", "nonce-def", "nonce-def"
    )


def test_opencode_v1_accepts_updated_rows(tmp_path, opencode_v1_db):
    with sqlite3.connect(opencode_v1_db) as conn:
        conn.execute("UPDATE message SET time_created = 1000")
        conn.execute("UPDATE part SET time_created = 1000")
    assert run.opencode_v1_reply_seen(tmp_path, 10, "hi-abc", "nonce-def", "")
    assert not run.opencode_v1_reply_seen(tmp_path, 13, "hi-abc", "nonce-def", "")


def test_opencode_v1_missing_db_falls_back_to_raw_output(tmp_path):
    assert run.opencode_v1_reply_seen(tmp_path, 10, "hi-abc", "nonce-def", "nonce-def")
    assert not run.opencode_v1_reply_seen(
        tmp_path, 10, "hi-abc", "nonce-def", "nonce-d\x1b[3D⬝⬝⬝⬝⬝⬝⬝⬝ef"
    )
    assert not (tmp_path / ".rapid-mlx").exists()


def test_opencode_v1_session_uses_database_for_both_prompts(
    tmp_path, opencode_v1_db, monkeypatch
):
    proc = SimpleNamespace(poll=lambda: None, returncode=0)
    monkeypatch.setattr(run, "version", lambda *_args: "opencode v1.19.5")
    monkeypatch.setattr(run, "open_terminal", lambda: (10, 11))
    monkeypatch.setattr(run.subprocess, "Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(run.os, "close", lambda _fd: None)
    monkeypatch.setattr(run.os, "write", lambda *_args: None)
    monkeypatch.setattr(run.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(run.time, "time", lambda: 10)
    monkeypatch.setattr(run, "stop_terminal_process", lambda _proc: None)
    monkeypatch.setattr(run, "handle_first_run_menus", lambda _agent, out, *args: out)
    replies = []

    def send_reply(_agent, _fd, prompt):
        hello = prompt.split("exactly ")[1].split()[0].rstrip(".")
        nonce = (tmp_path / "nonce.txt").read_text().strip()
        (tmp_path / "hello.txt").write_text(hello)
        with sqlite3.connect(opencode_v1_db) as conn:
            for part_id, value in (("p0", hello), ("p1", nonce)):
                conn.execute(
                    "UPDATE part SET data = json_set(data, '$.state.output', ?) "
                    "WHERE id = ?",
                    (value, part_id),
                )
            conn.execute(
                "UPDATE part SET data = json_set(data, '$.text', ?) WHERE id = 'p2'",
                (f"{hello} {nonce}",),
            )
        replies.append(f"{hello} {nonce[:8]}\x1b[3D⬝⬝⬝⬝⬝⬝⬝⬝{nonce[8:]}")

    def read_reply(_fd, _proc, _timeout, sink, stop_when=None, on_output=None):
        if not replies:
            return "Agent ready.", None
        if stop_when is not None:
            assert stop_when(replies[-1])
        return replies[-1], None

    monkeypatch.setattr(run, "send_prompt", send_reply)
    monkeypatch.setattr(run, "read_until", read_reply)
    result = run.session(
        ["rapid-mlx", "launch", "opencode"],
        {"HOME": str(tmp_path)},
        tmp_path,
        tmp_path / "session.pty",
        1,
        False,
    )
    assert result["tool_ok"] and result["second_ok"]
    assert len(replies) == 2


def test_opencode_v2_reply_uses_completed_session_not_tui_fragments(tmp_path):
    assert opencode_reply_seen(tmp_path, 10, "hi-abc", "nonce-def") is False
    db = tmp_path / ".rapid-mlx/connect/agents/opencode/xdg-data/opencode/opencode.db"
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE session_message "
            "(session_id TEXT, type TEXT, time_created INTEGER, seq INTEGER, data TEXT)"
        )
        parts = [
            {
                "type": "tool",
                "name": "read",
                "state": {
                    "status": "completed",
                    "input": {"path": path},
                    "content": [{"text": content}],
                },
            }
            for path, content in (
                ("hello.txt", "1: hi-abc"),
                ("nonce.txt", "1: nonce-def"),
            )
        ]
        parts.append({"type": "text", "text": "hi-abc and nonce-def"})
        conn.execute(
            "INSERT INTO session_message VALUES (?, ?, ?, ?, ?)",
            (
                "session-a",
                "assistant",
                11000,
                1,
                json.dumps({"time": {"completed": 12}, "content": parts}),
            ),
        )
    assert opencode_reply_seen(tmp_path, 10, "hi-abc", "nonce-def") is True
    assert opencode_reply_seen(tmp_path, 10, "hi-abc", "nonce-wrong") is False
    assert opencode_reply_seen(tmp_path, 12, "hi-abc", "nonce-def") is False
    with sqlite3.connect(db) as conn:
        parts.pop(1)
        conn.execute(
            "UPDATE session_message SET data = ?",
            (json.dumps({"time": {"completed": 12}, "content": parts}),),
        )
    assert opencode_reply_seen(tmp_path, 10, "hi-abc", "nonce-def") is False


def test_opencode_v2_does_not_combine_reads_with_another_session(tmp_path):
    db = tmp_path / ".rapid-mlx/connect/agents/opencode/xdg-data/opencode/opencode.db"
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE session_message "
            "(session_id TEXT, type TEXT, time_created INTEGER, seq INTEGER, data TEXT)"
        )
        reads = [
            {
                "type": "tool",
                "name": "read",
                "state": {
                    "status": "completed",
                    "input": {"path": path},
                    "content": [{"text": value}],
                },
            }
            for path, value in (("hello.txt", "hi-abc"), ("nonce.txt", "nonce-def"))
        ]
        for session_id, content in (
            ("session-a", reads),
            ("session-b", [{"type": "text", "text": "hi-abc nonce-def"}]),
        ):
            conn.execute(
                "INSERT INTO session_message VALUES (?, ?, ?, ?, ?)",
                (
                    session_id,
                    "assistant",
                    11000,
                    1,
                    json.dumps({"time": {"completed": 12}, "content": content}),
                ),
            )
    assert opencode_reply_seen(tmp_path, 10, "hi-abc", "nonce-def") is False


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


AIDER_SHELL = "Run shell command? (Y)es/(N)o/(S)kip all/(D)on't ask again [Yes]:"
AIDER_SHELL_PLURAL = (
    "Run shell commands? (Y)es/(N)o/(S)kip all/(D)on't ask again [Yes]:"
)
AIDER_SHELL_NO_SKIP = "Run shell command? (Y)es/(N)o/(D)on't ask again [Yes]:"
AIDER_SHELL_PLURAL_NO_SKIP = "Run shell commands? (Y)es/(N)o/(D)on't ask again [Yes]:"
AIDER_ADD_OUTPUT = "Add command output to the chat? (Y)es/(N)o/(D)on't ask again [Yes]:"


def crush_modal(workspace, command="cat hello.txt nonce.txt"):
    return (
        "Permission Required ╱╱╱││ Tool bash    ││ Path "
        f"{workspace}    ││ Desc Read files ││ {command} ││ "
        "Allow      Allow for Session      Deny ││ ←/→ choose • enter confirm"
    )


def crush_file_modal(path, tool="write"):
    return (
        crush_modal(path, "@@ -1,0 +1,1 @@ ││ 1 + hi")
        .replace("Tool bash", f"Tool {tool}")
        .replace("Path ", "File ")
    )


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(run.time, "monotonic", lambda: now[0])
    return now


def test_crush_waits_for_dialog_guard_and_quiet(tmp_path, monkeypatch, clock):
    sent = []
    monkeypatch.setattr(run.os, "write", lambda _fd, data: sent.append(data))
    approve = approval_handler("crush", 10, tmp_path)
    prompt = crush_modal(tmp_path)
    approve(prompt)
    assert sent == []
    clock[0] += 1.599
    approve(prompt)
    assert sent == []
    clock[0] = 101.6
    approve(prompt)
    approve(prompt)
    assert sent == [b"s"]


def test_crush_waits_for_output_quiet_without_resetting_guard(
    tmp_path, monkeypatch, clock
):
    sent = []
    monkeypatch.setattr(run.os, "write", lambda _fd, data: sent.append(data))
    approve = approval_handler("crush", 10, tmp_path)
    prompt = crush_modal(tmp_path)
    approve(prompt)
    clock[0] = 101.5
    output = prompt + "\r\n" + prompt
    approve(output)
    clock[0] = 101.6
    approve(output)
    assert sent == []
    clock[0] = 101.999
    approve(output)
    assert sent == []
    clock[0] = 102.0
    approve(output)
    approve(output)
    assert sent == [b"s"]


@pytest.mark.parametrize("new_state", ["working", "unknown", "incomplete", "outside"])
def test_crush_drops_stale_pending_answer(new_state, tmp_path, monkeypatch, clock):
    sent = []
    sink = io.BytesIO()
    monkeypatch.setattr(run.os, "write", lambda _fd, data: sent.append(data))
    approve = approval_handler("crush", 10, tmp_path, sink)
    prompt = crush_modal(tmp_path)
    approve(prompt)
    newer = {
        "working": "\x1b]9;4;state=working:app=crush\x07",
        "unknown": prompt.replace("Tool bash", "Tool read"),
        "incomplete": "Permission Required ││ Tool write ││",
        "outside": crush_modal(tmp_path.parent),
    }[new_state]
    output = prompt + newer
    # Check freshness even if no callback ran between detection and deadline.
    clock[0] += 2
    approve(output)
    clock[0] += 2
    approve(output)
    assert sent == []
    assert sink.getvalue() == b""


def test_crush_different_modal_gets_its_own_deadline(tmp_path, monkeypatch, clock):
    sent = []
    monkeypatch.setattr(run.os, "write", lambda _fd, data: sent.append(data))
    approve = approval_handler("crush", 10, tmp_path)
    prompt = crush_modal(tmp_path)
    approve(prompt)
    clock[0] = 101.0
    output = prompt + crush_file_modal(tmp_path / "hello.txt")
    approve(output)
    clock[0] = 101.6
    approve(output)
    assert sent == []
    clock[0] = 102.6
    approve(output)
    approve(output)
    assert sent == [b"a"]


@pytest.mark.parametrize(
    "gap", ["Permission Required Tool read", "state=working:app=crush"]
)
def test_crush_pending_intervening_state_before_redraw(
    gap, tmp_path, monkeypatch, clock
):
    sent = []
    monkeypatch.setattr(run.os, "write", lambda _fd, data: sent.append(data))
    approve = approval_handler("crush", 10, tmp_path)
    prompt = crush_modal(tmp_path)
    approve(prompt)
    clock[0] += 1
    output = prompt + gap + prompt
    approve(output)
    clock[0] += 2
    approve(output)
    assert sent == ([b"s"] if "working" in gap else [])


@pytest.mark.parametrize("tool", ["bash", "write", "edit", "multiedit"])
def test_crush_resolves_tmp_alias(tool, monkeypatch, clock):
    sent = []
    monkeypatch.setattr("os.write", lambda _fd, data: sent.append(data))
    with tempfile.TemporaryDirectory(
        prefix="website-poshog-agent-e2e-crush-write-", dir="/tmp"
    ) as directory:
        workspace = Path(directory).resolve()
        alias = Path("/tmp") / workspace.name
        assert alias.resolve() == workspace
        prompt = (
            crush_modal(alias)
            if tool == "bash"
            else crush_file_modal(alias / "hello.txt", tool)
        )
        approve = approval_handler("crush", 10, workspace)
        approve(prompt)
        clock[0] += 1.6
        approve(prompt)
    assert sent == [b"s" if tool == "bash" else b"a"]


@pytest.mark.parametrize("tool", ["write", "edit", "multiedit"])
@pytest.mark.parametrize("outside", ["absolute", "parent", "sibling", "symlink"])
def test_crush_file_outside_workspace_is_not_answered(
    tool, outside, tmp_path, monkeypatch, clock
):
    sent = []
    monkeypatch.setattr("os.write", lambda _fd, data: sent.append(data))
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "escape").symlink_to(tmp_path, target_is_directory=True)
    path = {
        "absolute": Path("/etc/hosts"),
        "parent": workspace / "../outside.txt",
        "sibling": tmp_path / "project-other/hello.txt",
        "symlink": workspace / "escape/outside.txt",
    }[outside]
    approve = approval_handler("crush", 10, workspace)
    prompt = crush_file_modal(path, tool)
    approve(prompt)
    clock[0] += 2
    approve(prompt)
    assert sent == []


def test_crush_file_redraws_and_consecutive_prompts(tmp_path, monkeypatch, clock):
    sent = []
    monkeypatch.setattr("os.write", lambda _fd, data: sent.append(data))
    approve = approval_handler("crush", 10, tmp_path)
    prompt = crush_file_modal(tmp_path / "hello.txt")
    approve(prompt[:30])
    assert sent == []
    approve(prompt)
    approve(prompt)
    output = prompt + "\r\n" + prompt
    approve(output)
    clock[0] += 1.6
    approve(output)
    assert sent == [b"a"]
    output += crush_file_modal(tmp_path / "nonce.txt")
    approve(output)
    approve(output)
    clock[0] += 1.6
    approve(output)
    assert sent == [b"a", b"a"]
    output += "\x1b]9;4;state=working:app=crush\x07"
    output += crush_file_modal(tmp_path / "nonce.txt")
    approve(output)
    clock[0] += 1.6
    approve(output)
    assert sent == [b"a", b"a", b"a"]


@pytest.mark.parametrize(
    "agent,aider_prompt",
    [
        ("aider", AIDER_SHELL),
        ("aider", AIDER_SHELL_PLURAL),
        ("aider", AIDER_SHELL_NO_SKIP),
        ("aider", AIDER_SHELL_PLURAL_NO_SKIP),
        ("aider", AIDER_ADD_OUTPUT),
        ("crush", None),
    ],
)
def test_permission_answers_split_reads_and_redraws_once(
    agent, aider_prompt, tmp_path, monkeypatch, clock
):
    sent = []
    sink = io.BytesIO()
    monkeypatch.setattr("os.write", lambda _fd, data: sent.append(data))
    approve = approval_handler(agent, 10, tmp_path, sink)
    prompt = aider_prompt if agent == "aider" else crush_modal(tmp_path)
    prompt = "\x1b[32m" + prompt + "\x1b[0m"
    # Real read_until passes cumulative output, including on idle polls.
    approve(prompt[:30])
    assert sent == []
    approve(prompt)
    approve(prompt)
    approve(prompt + "\r\n   " + prompt)
    approve(prompt + "\r\n   " + prompt)
    clock[0] += 1.6
    approve(prompt + "\r\n   " + prompt)
    keys = b"y\r" if agent == "aider" else b"s"
    assert sent == [keys]
    marker = b"\n[agent_e2e: permission prompt answered]\n"
    assert sink.getvalue() == marker
    next_prompt = (
        "\r\ny\r\ncommand completed\n" + aider_prompt
        if agent == "aider"
        else crush_modal(tmp_path, "cat hello.txt")
    )
    approve(prompt + "\r\n   " + prompt + next_prompt)
    clock[0] += 1.6
    approve(prompt + "\r\n   " + prompt + next_prompt)
    assert sent == [keys, keys]
    assert sink.getvalue() == marker * 2
    transcript = tmp_path / "day-one.pty"
    transcript.write_bytes(sink.getvalue())
    assert output_issues(transcript, transcript) == ["permission prompt answered"]


@pytest.mark.parametrize(
    "agent,prompt",
    [
        ("aider", "Run shell command? [Yes]:"),
        ("aider", "Run shell commands? [Yes]:"),
        ("aider", "Run shell command? (Y)es/(N)o [Yes]:"),
        ("aider", "Run shell command? (Y)es/(N)o/(S)kip all [Yes]:"),
        ("aider", AIDER_SHELL.replace("/(S)kip all", "/(A)ll/(S)kip all")),
        ("aider", AIDER_SHELL_NO_SKIP.replace("[Yes]", "[No]")),
        ("aider", AIDER_SHELL_NO_SKIP.removesuffix(":")),
        ("aider", AIDER_ADD_OUTPUT.replace("/(D)", "/(S)kip all/(D)")),
        ("aider", AIDER_ADD_OUTPUT.replace("/(D)", "/(A)ll/(D)")),
        ("aider", AIDER_ADD_OUTPUT.replace("/(D)", "/(A)ll/(S)kip all/(D)")),
        ("aider", AIDER_ADD_OUTPUT.replace("/(D)on't ask again", "")),
        ("aider", AIDER_ADD_OUTPUT.replace("[Yes]", "[No]")),
        ("aider", AIDER_ADD_OUTPUT.removesuffix(":")),
        ("aider", AIDER_ADD_OUTPUT.replace("output", "outputs")),
        ("aider", "Allow access outside the project?"),
        ("aider", "Select login method"),
        ("crush", "Allow      Allow for Session      Deny"),
        ("crush", "Permission Required Tool bash Path /elsewhere"),
    ],
)
def test_unknown_permission_prompts_are_not_answered(
    agent, prompt, tmp_path, monkeypatch
):
    sent = []
    monkeypatch.setattr("os.write", lambda _fd, data: sent.append(data))
    approval_handler(agent, 10, tmp_path)(prompt)
    assert sent == []


@pytest.mark.parametrize("relative", ["..", "../project-other", "child"])
def test_crush_requires_exact_test_workspace(relative, tmp_path, monkeypatch, clock):
    sent = []
    monkeypatch.setattr("os.write", lambda _fd, data: sent.append(data))
    approve = approval_handler("crush", 10, tmp_path)
    prompt = crush_modal(tmp_path / relative)
    approve(prompt)
    clock[0] += 2
    approve(prompt)
    assert sent == []
    assert approval_handler("crush", 10) is None


def test_permission_prompts_are_agent_specific(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr("os.write", lambda _fd, data: sent.append(data))
    approval_handler("aider", 10)(crush_modal(tmp_path))
    approval_handler("crush", 10, tmp_path)(AIDER_SHELL)
    assert sent == []


@pytest.mark.parametrize(
    "prompt,keys",
    [
        (
            "Would you like to see what's new in this version? (Y)es/(N)o [Yes]:",
            b"n\r",
        ),
        ("Create new file? (Y)es/(N)o [Yes]:", b"y\r"),
    ],
)
def test_aider_full_questions_and_redraws_are_answered_once(prompt, keys, monkeypatch):
    sent = []
    monkeypatch.setattr("os.write", lambda _fd, data: sent.append(data))
    approve = approval_handler("aider", 10)
    approve(prompt.removesuffix(" [Yes]:"))
    assert sent == []
    approve(prompt)
    approve(prompt)
    assert sent == [keys]
    redraw = prompt + "\r\n\x1b[2K" + prompt
    approve(redraw)
    approve(redraw)
    assert sent == [keys]


def test_aider_add_output_once_per_command_group(monkeypatch):
    sent = []
    sink = io.BytesIO()
    monkeypatch.setattr("os.write", lambda _fd, data: sent.append(data))
    approve = approval_handler("aider", 10, sink=sink)
    output = ""
    for shell_prompt in (AIDER_SHELL_PLURAL, AIDER_SHELL_NO_SKIP):
        output += shell_prompt
        approve(output)
        output += "\r\ny\r\nRunning cat hello.txt\r\nhello\r\n"
        output += "Running cat nonce.txt\r\nnonce\r\n" + AIDER_ADD_OUTPUT
        approve(output)
        approve(output)
        output += "\r\n\x1b[2K" + AIDER_ADD_OUTPUT
        approve(output)
        output += "\r\ny\r\nAdded 4 lines of output to the chat.\r\n"
    assert sent == [b"y\r"] * 4
    assert sink.getvalue() == b"\n[agent_e2e: permission prompt answered]\n" * 4


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


def test_codex_native_bookkeeping_does_not_mask_rapid_config_write(tmp_path):
    path = tmp_path / ".codex/config.toml"
    path.parent.mkdir()
    path.write_text(FILES[".codex/config.toml"])
    before = snapshot(tmp_path, "codex")
    assert rapid_config_untouched(tmp_path, "codex", before)

    def native_mode(*_args):
        path.write_text(path.read_text() + "\n[tui]\nseen = true\n")
        return True

    with patch("scripts.agent_e2e.run.normal_mode", side_effect=native_mode):
        assert check_then_open_native(
            tmp_path, "codex", before, {}, tmp_path, tmp_path / "native.pty"
        ) == (True, True)
    path.write_text(FILES[".codex/config.toml"])
    # Negative control: a Rapid write to the seeded user config stays red.
    path.write_text(path.read_text() + "\nmodel = 'rapid-overwrite'\n")
    with patch("scripts.agent_e2e.run.normal_mode", return_value=True):
        assert check_then_open_native(
            tmp_path, "codex", before, {}, tmp_path, tmp_path / "native.pty"
        ) == (False, True)
    with patch("scripts.agent_e2e.run.normal_mode", side_effect=OSError):
        assert check_then_open_native(
            tmp_path, "codex", before, {}, tmp_path, tmp_path / "native.pty"
        ) == (False, False)


def test_claude_folder_trust_menu_selects_yes(monkeypatch):
    writes = []
    monkeypatch.setattr(
        "scripts.agent_e2e.run.os.write", lambda _fd, data: writes.append(data)
    )
    monkeypatch.setattr("scripts.agent_e2e.run.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "scripts.agent_e2e.run.read_until", lambda *args: ("ready", None)
    )
    opening = (
        "Accessing workspace: /private/project\n"
        "Quick safety check: Is this a project you trust?\n"
        "No, exit\nYes, I trust this folder\nEnter to confirm"
    )
    handle_first_run_menus("claude-code", opening, 1, None, None)
    assert writes == [b"\x1b[B", b"\r"]
    writes.clear()
    old_menu = (
        "Do you trust the files in this folder?\n"
        "1. Yes, proceed\n2. No, exit\nEnter to confirm"
    )
    handle_first_run_menus("claude-code", old_menu, 1, None, None)
    assert writes == [b"\r"]


def test_claude_security_page_is_confirmed_before_folder_trust(monkeypatch):
    writes = []
    pages = iter(
        (
            "Accessing workspace /private/project\nNo, exit\n"
            "Yes, I trust this folder\nEnter to confirm",
            "ready",
        )
    )
    monkeypatch.setattr(
        "scripts.agent_e2e.run.os.write", lambda _fd, data: writes.append(data)
    )
    monkeypatch.setattr("scripts.agent_e2e.run.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "scripts.agent_e2e.run.read_until", lambda *args: (next(pages), None)
    )
    opening = (
        "Security notes: Due to prompt injection risks, only use code you trust. "
        "Press Enter to continue"
    )
    handle_first_run_menus("claude-code", opening, 1, None, None)
    assert writes == [b"\r", b"\x1b[B", b"\r"]


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


@pytest.mark.parametrize("day_two", [False, True])
@pytest.mark.parametrize("failed_turn", [1, 2, None])
def test_session_sends_second_prompt_only_after_first_success(
    tmp_path, monkeypatch, failed_turn, day_two
):
    sent = []
    proc = SimpleNamespace(poll=lambda: None, returncode=0)
    monkeypatch.setattr(run, "open_terminal", lambda: (10, 11))
    monkeypatch.setattr(run.subprocess, "Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(run.os, "close", lambda _fd: None)
    monkeypatch.setattr(run.os, "write", lambda fd, data: sent.append((fd, data)))
    monkeypatch.setattr(run.time, "sleep", lambda _seconds: None)
    stopped = []
    monkeypatch.setattr(run, "stop_terminal_process", stopped.append)
    monkeypatch.setattr(run, "handle_first_run_menus", lambda _agent, out, *args: out)
    outputs = []
    nonces = []

    def read_reply(fd, _proc, _timeout, sink, stop_when=None, on_output=None):
        if not sent or stop_when is None:
            return "Agent ready.", None
        assert fd == 10
        task = next(data for _, data in reversed(sent) if b"hello.txt" in data)
        content = task.decode().split("exactly ")[1].split()[0].rstrip(".")
        nonce = (tmp_path / "nonce.txt").read_text()
        nonces.append(nonce)
        if len(nonces) == failed_turn:
            # Timeout leaves the agent alive in an unknown approval dialog.
            output = "Approval required: rm -rf hello.txt"
            assert not stop_when(output)
        else:
            (tmp_path / "hello.txt").write_text(content)
            output = content + "\n" + nonce
            assert stop_when(output)
        outputs.append(output)
        return output, None

    monkeypatch.setattr(run, "read_until", read_reply)
    result = run.session(
        ["rapid-mlx", "launch", "codex"],
        {},
        tmp_path,
        tmp_path / "session.pty",
        1,
        day_two,
    )
    prompts = [data for fd, data in sent if b"hello.txt" in data]
    assert len(prompts) == (1 if failed_turn == 1 else 2)
    assert b"Create hello.txt" in prompts[0]
    if failed_turn == 1:
        assert not any(b"Use a tool to replace hello.txt" in data for _, data in sent)
        assert result["first_reply_s"] is None
        assert result["prompt2_at"] is None
        assert (tmp_path / "nonce.txt").read_text() == nonces[0]
    else:
        assert b"Use a tool to replace hello.txt" in prompts[1]
        assert result["first_reply_s"] is not None
        assert result["prompt2_at"] is not None
        assert nonces[0] != nonces[1]
    assert result["tool_ok"] is (failed_turn != 1)
    assert result["second_ok"] is (failed_turn is None)
    assert (result["second_reply_s"] is None) is (failed_turn is not None)
    assert result["error"] == ("" if failed_turn is None else "".join(outputs)[-500:])
    assert result["day_two"] is day_two
    # After the task prompts, shutdown must write only cancellation controls.
    assert sent == [(10, data) for prompt in prompts for data in (prompt, b"\r")] + [
        (10, b"\x03"),
        (10, b"\x1b"),
        (10, b"\x04"),
    ]
    assert stopped == [proc]
    assert result["exit"] == 0
    assert result["ended_at"] is not None


@pytest.mark.parametrize("agent", ["aider", "crush"])
def test_session_records_permissions_for_both_turns(
    agent, tmp_path, monkeypatch, clock
):
    sent = []
    proc = SimpleNamespace(poll=lambda: None, returncode=0)
    monkeypatch.setattr(run, "open_terminal", lambda: (10, 11))
    monkeypatch.setattr(run.subprocess, "Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(run.os, "close", lambda _fd: None)
    monkeypatch.setattr(run.os, "write", lambda _fd, data: sent.append(data))
    monkeypatch.setattr(run.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(run, "stop_terminal_process", lambda _proc: None)
    monkeypatch.setattr(run, "handle_first_run_menus", lambda _agent, out, *args: out)
    turns = []

    def read_reply(_fd, _proc, _timeout, sink, stop_when=None, on_output=None):
        if on_output is None:
            return "Agent ready.", None
        prompt = AIDER_SHELL if agent == "aider" else crush_modal(tmp_path)
        on_output(prompt)
        on_output(prompt)
        clock[0] += 1.6
        on_output(prompt)
        # Model output arrives only after the confirmation has been answered.
        assert sent[-1] == (b"y\r" if agent == "aider" else b"s")
        task = next(data for data in reversed(sent[:-1]) if b"hello.txt" in data)
        content = task.decode().split("exactly ")[1].split()[0].rstrip(".")
        (tmp_path / "hello.txt").write_text(content)
        output = prompt + "\n" + (tmp_path / "nonce.txt").read_text()
        assert stop_when(output)
        turns.append(content)
        return output, None

    monkeypatch.setattr(run, "read_until", read_reply)
    transcript = tmp_path / "day-one.pty"
    result = run.session(
        ["rapid-mlx", "launch", agent], {}, tmp_path, transcript, 1, False
    )
    assert result["tool_ok"] and result["second_ok"]
    assert len(turns) == 2 and turns[1] == turns[0] + "-second"
    assert transcript.read_bytes().count(b"permission prompt answered") == 2
    assert output_issues(transcript) == ["permission prompt answered"]


def test_read_loop_keeps_waiting_after_answering_permission(tmp_path, monkeypatch):
    sent = []
    sink = io.BytesIO()
    # Include an idle poll and a split question, as in real PTY output.
    chunks = iter(
        [AIDER_SHELL[:25].encode(), AIDER_SHELL[25:].encode(), b"\ny\nreply", b""]
    )
    polls = iter([True, False, True, False, True, True])
    monkeypatch.setattr(run.os, "write", lambda _fd, data: sent.append(data))
    monkeypatch.setattr(run.os, "read", lambda *_args: next(chunks))
    monkeypatch.setattr(
        run.select, "select", lambda *args: ([10] if next(polls) else [], [], [])
    )
    output, first = read_until(
        10,
        SimpleNamespace(poll=lambda: None),
        1,
        sink,
        on_output=approval_handler("aider", 10, tmp_path, sink),
    )
    assert sent == [b"y\r"]
    assert output == AIDER_SHELL + "\ny\nreply"
    assert first is not None
    assert b"permission prompt answered" in sink.getvalue()


def test_crush_new_occurrence_after_working_state(tmp_path, monkeypatch, clock):
    sent = []
    monkeypatch.setattr(run.os, "write", lambda _fd, data: sent.append(data))
    approve = approval_handler("crush", 10, tmp_path)
    prompt = crush_modal(tmp_path)
    approve(prompt)
    clock[0] += 1.6
    approve(prompt)
    # OSC progress establishes that the previous modal was dismissed, even
    # when the next tool requests an identical command.
    output = prompt + "\x1b]9;4;state=working:app=crush\x07" + prompt
    approve(output)
    approve(output)
    clock[0] += 1.6
    approve(output)
    assert sent == [b"s", b"s"]


def test_crush_read_loop_handles_split_utf8_and_ansi(tmp_path, monkeypatch, clock):
    sent = []
    monkeypatch.setattr(run.os, "write", lambda _fd, data: sent.append(data))
    raw = ("\x1b[32m" + crush_modal(tmp_path) + "\x1b[0m").encode()
    # Feed one byte per PTY read, splitting both ANSI escapes and box glyphs.
    chunks = iter([bytes([byte]) for byte in raw] + [b"\xff\xe2", b""])
    reads = [0]

    def select_tick(*_args):
        reads[0] += 1
        if len(raw) + 1 < reads[0] < len(raw) + 10:
            clock[0] += 0.25
            return [], [], []
        return [10], [], []

    monkeypatch.setattr(run.os, "read", lambda *_args: next(chunks))
    monkeypatch.setattr(run.select, "select", select_tick)
    output, _ = read_until(
        10,
        SimpleNamespace(poll=lambda: None),
        5,
        io.BytesIO(),
        on_output=approval_handler("crush", 10, tmp_path),
    )
    assert output == raw.decode() + "\ufffd\ufffd"
    assert sent == [b"s"]


def test_crush_does_not_combine_incomplete_or_other_tool_dialogs(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(run.os, "write", lambda _fd, data: sent.append(data))
    approve = approval_handler("crush", 10, tmp_path)
    other_tool = crush_modal(tmp_path).replace("Tool bash", "Tool read")
    incomplete = "Permission Required ││ Tool bash ││ "
    approve(incomplete + other_tool)
    approve(incomplete + other_tool + crush_modal(tmp_path).split("←/→")[0])
    assert sent == []
