# SPDX-License-Identifier: Apache-2.0
"""Reasoning state and actual PTY-to-terminal scrollback regressions."""

import codecs
import fcntl
import io
import json
import os
import pty
import select
import struct
import subprocess
import sys
import termios
import time

import pyte
import pytest
from rich.console import Console

from rapid_mlx.chat_harness_render import HarnessRenderer
from rapid_mlx.chat_state import TurnState


def event(seq, kind, **data):
    return {
        "wire_version": 1,
        "turn_id": "turn",
        "seq": seq,
        "event": {"type": kind, **data},
    }


def rendered(renderer):
    stream = io.StringIO()
    Console(file=stream, width=80, color_system=None).print(renderer.view())
    return stream.getvalue()


def test_waiting_is_neutral_and_reasoning_elapsed_is_actual(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("rapid_mlx.chat_state.time.monotonic", lambda: clock[0])
    state = TurnState("turn")
    renderer = HarnessRenderer(state, stream=io.StringIO())
    assert "Thinking" not in rendered(renderer)
    clock[0] = 105.0
    state.apply(event(1, "reasoning.delta", part_id="r", text="Consider"))
    assert "Thinking…" in rendered(renderer)
    clock[0] = 107.5
    state.apply(event(2, "text.delta", part_id="answer", text="The answer."))
    assert "Thinking" not in rendered(renderer)
    assert "Thought for 2.5 s" in rendered(renderer)
    assert "Consider" not in rendered(renderer)
    renderer.expanded = True
    assert "Consider" in rendered(renderer)
    clock[0] = 110.0
    state.apply(event(3, "turn.terminal", outcome="completed"))
    assert state.reasoning_seconds == 2.5
    assert rendered(renderer).index("Thought for 2.5 s") < rendered(renderer).index(
        "The answer."
    )


def test_reasoning_replay_deduplication_and_multiple_phases(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("rapid_mlx.chat_state.time.monotonic", lambda: clock[0])
    state = TurnState("turn")
    state.apply(event(1, "reasoning.delta", part_id="r", text="", omitted=True))
    assert state.reasoning_since is None
    delta = event(2, "reasoning.delta", part_id="r", text="one")
    state.apply(delta)
    clock[0] = 2.0
    state.apply(event(3, "tool.started", call_id="call", name="read_attachment"))
    clock[0] = 12.0
    assert not state.apply(delta)
    state.apply(event(4, "reasoning.delta", part_id="r2", text="two"))
    clock[0] = 15.0
    state.apply(event(5, "turn.terminal", outcome="cancelled"))
    assert state.reasoning_seconds == 5.0


def test_no_color_reasoning_status_changes_before_answer(monkeypatch):
    class Terminal(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setenv("NO_COLOR", "1")
    output = Terminal()
    state = TurnState("turn")
    renderer = HarnessRenderer(state, stream=output)
    assert output.getvalue() == "…\n"
    for item in [
        event(1, "reasoning.delta", part_id="r", text="private"),
        event(2, "text.delta", part_id="a", text="Answer"),
    ]:
        state.apply(item)
        renderer.update(item)
    renderer.finish()
    assert "private" not in output.getvalue()
    assert (
        output.getvalue().index("Thinking…")
        < output.getvalue().index("Thought for")
        < output.getvalue().index("Answer")
    )


def terminal_capture(script, *, rows=9, columns=52):
    """Execute on a real PTY, including the kernel's LF → CRLF translation."""
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))
    attributes = termios.tcgetattr(slave)
    attributes[1] |= termios.OPOST | termios.ONLCR
    termios.tcsetattr(slave, termios.TCSANOW, attributes)
    env = dict(os.environ, TERM="xterm-256color", COLUMNS=str(columns), LINES=str(rows))
    env.pop("NO_COLOR", None)
    env.pop("CI", None)
    child = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.DEVNULL,
        stdout=slave,
        stderr=slave,
        env=env,
    )
    os.close(slave)
    chunks = []
    deadline = time.monotonic() + 15
    try:
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.1)
            if ready:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                chunks.append(chunk)
            elif child.poll() is not None:
                break
        assert child.wait(timeout=2) == 0, b"".join(chunks).decode(errors="replace")
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        os.close(master)
    screen = pyte.HistoryScreen(columns, rows, history=10000)
    terminal = pyte.Stream(screen)
    decoder = codecs.getincrementaldecoder("utf-8")()
    for chunk in chunks:
        terminal.feed(decoder.decode(chunk))
    terminal.feed(decoder.decode(b"", final=True))
    lines = [
        "".join(line[column].data for column in range(columns)).rstrip()
        for line in screen.history.top
    ]
    lines.extend(line.rstrip() for line in screen.display)
    return "\n".join(lines), b"".join(chunks)


@pytest.mark.parametrize("format", ["paragraph", "code"])
def test_long_markdown_leaves_exactly_one_copy_in_terminal_scrollback(format):
    script = f"""
from rapid_mlx.chat_harness_render import HarnessRenderer
from rapid_mlx.chat_state import TurnState
print("BEFORE")
state = TurnState("turn")
renderer = HarnessRenderer(state)
seq = 0
def send(text):
    global seq
    seq += 1
    e = {{"wire_version": 1, "turn_id": "turn", "seq": seq, "event": {{"type": "text.delta", "part_id": "answer", "text": text}}}}
    state.apply(e)
    renderer.update(e)
if {format!r} == "code": send("```text\\n")
for row in range(35):
    send(f"ROW_{{row:02d}} ")
    send("final content.")
    send("\\n" if {format!r} == "code" else "\\n\\n")
if {format!r} == "code": send("```\\n")
state.apply({{"wire_version": 1, "turn_id": "turn", "seq": seq + 1, "event": {{"type": "turn.terminal", "outcome": "completed"}}}})
renderer.finish()
renderer.finish()
print("AFTER")
"""
    display, raw = terminal_capture(script)
    assert b"\r\n" in raw
    assert display.count("BEFORE") == 1
    assert display.count("AFTER") == 1
    for row in range(35):
        assert display.count(f"ROW_{row:02d}") == 1, display
        assert f"ROW_{row:02d} final content." in display, display
    assert "Thinking" not in display


@pytest.mark.parametrize("resume", ["", "last"])
def test_prompt_status_and_menu_terminal_attributes(tmp_path, resume):
    """Exercise the real client loop/editor with a deterministic catalog."""
    import pexpect

    from rapid_mlx.chat_client import COMMANDS
    from tests.test_chat_harness_input import PaletteScreen

    script = """
import sys
from types import SimpleNamespace
from rapid_mlx.chat_client import ChatClient
snapshot = {
    'id': '12345678-abcd-1234-abcd-123456789abc',
    'branch_id': 'branch', 'branches': [{'id': 'branch', 'revision': 0}],
    'title': 'New chat', 'display_title': 'Ocean study', 'messages': [],
}
client = ChatClient(SimpleNamespace(resume=sys.argv[1] or None),
                    'http://127.0.0.1:18800', 'fixture')
client.connect = lambda: None
client.heartbeat = lambda: client.stopping.wait()
client.request = lambda method, path, **kwargs: (
    [snapshot] if method == 'GET' and path == '/conversations' else snapshot
)
try:
    client.run()
finally:
    client.close()
"""
    env = dict(os.environ, TERM="xterm-256color", PYTHONIOENCODING="utf-8")
    for key in ("NO_COLOR", "CI", "PROMPT_TOOLKIT_COLOR_DEPTH"):
        env.pop(key, None)
    child = pexpect.spawn(
        sys.executable,
        ["-c", script, resume],
        env=env,
        encoding="utf-8",
        timeout=5,
        dimensions=(24, 80),
    )
    screen = PaletteScreen(child)
    new_status = "New chat · Type / for commands"
    resumed_status = "Resumed: Ocean study · Type / for commands"
    captures = {}
    try:
        child.expect_exact("› ")
        captures["start"] = screen.dump()
        expected = resumed_status if resume else new_status
        assert screen.dump().count(expected) == 1
        assert "12345678" not in screen.dump()
        assert "\x1b[2m" in screen.raw  # Status is secondary/dim.

        child.sendline("/new")
        child.expect_exact("› ")
        assert screen.dump().count(new_status) == (1 if resume else 2)
        child.sendline("/history 1")
        child.expect_exact("› ")
        assert screen.dump().count(resumed_status) == (2 if resume else 1)
        assert "12345678" not in screen.dump()
        captures["resumed"] = screen.dump()

        child.send("/")
        screen.wait_for("Exit (also Ctrl-D)")
        captures["menu"] = screen.dump()

        def menu_rows():
            return [
                (command, y, row.index(command.name))
                for command in COMMANDS
                for y, row in enumerate(screen.screen.display)
                if row.strip().startswith(command.name + " ")
            ]

        rows = menu_rows()
        assert len(rows) == len(COMMANDS)
        for command, y, x in rows:
            cells = screen.screen.buffer[y]
            # Include padding and scrollbar, not just the command glyphs.
            assert all(cells[col].bg == "default" for col in range(80))
            assert not cells[x].reverse and cells[x].fg == "default"
            meta_x = screen.screen.display[y].index(command.description)
            assert cells[meta_x].fg != "default"

        child.send("\x1b[B")
        screen.wait(lambda _: screen.screen.buffer[rows[0][1]][rows[0][2]].reverse)
        for index, (command, y, x) in enumerate(menu_rows()):
            cells = screen.screen.buffer[y]
            assert all(cells[col].bg == "default" for col in range(80))
            assert cells[x].reverse == (index == 0)
            meta_x = screen.screen.display[y].index(command.description)
            assert cells[meta_x].reverse == (index == 0)
        captures["selected_menu"] = screen.dump()

        child.send("\x01\x0b/think on\r")
        child.expect_exact("Thinking on.")
        child.expect_exact("Thinking › ")
        captures["thinking_on"] = screen.dump()
        assert "Thinking ›" in screen.dump()
        assert "›\x1b[0m " in screen.raw  # Accepted repaint is never ready text.
        child.sendline("/think off")
        child.expect_exact("Thinking off.")
        child.expect_exact("› ")
        assert "Thinking ›\x1b[0m " in screen.raw
        assert screen.screen.display[screen.screen.cursor.y].strip() == "›"
        assert "Reasoning" not in screen.dump() and "Fast" not in screen.dump()
        child.sendline("/quit")
        child.expect(pexpect.EOF)
        child.close()
        assert child.exitstatus == 0
        (tmp_path / "chrome-screens.json").write_text(
            json.dumps(captures, ensure_ascii=False, indent=2)
        )
    finally:
        if child.isalive():
            child.terminate(force=True)


def test_tool_progress_finishes_once_and_sources_follow_answer_in_scrollback():
    script = """
from rapid_mlx.chat_harness_render import HarnessRenderer
from rapid_mlx.chat_state import TurnState
state = TurnState("turn")
renderer = HarnessRenderer(state)
events = [
    {"type": "tool.started", "call_id": "weather", "name": "weather", "args": {"location": "San Francisco"}},
    {"type": "tool.result", "call_id": "weather", "outcome": "success"},
    {"type": "tool.started", "call_id": "search", "name": "web_search", "args": {"query": "latest Apple news"}},
    {"type": "source", "source_id": "1", "call_id": "search", "title": "Apple", "url": "https://apple.com/newsroom/"},
    {"type": "tool.result", "call_id": "search", "outcome": "success"},
    {"type": "text.delta", "part_id": "answer", "text": "LIVE_ANSWER [1]."},
    {"type": "turn.terminal", "outcome": "completed"},
]
for sequence, data in enumerate(events, 1):
    envelope = {"wire_version": 1, "turn_id": "turn", "seq": sequence, "event": data}
    state.apply(envelope)
    renderer.update(envelope)
renderer.finish()
renderer.finish()
"""
    display, raw = terminal_capture(script, columns=100)
    assert display.count("Checking weather for San Francisco · done") == 1
    assert display.count("Searching the web: latest Apple news · done") == 1
    assert display.count("LIVE_ANSWER") == 1
    assert display.count("[1] Apple — https://apple.com/newsroom/") == 1
    assert display.index("LIVE_ANSWER") < display.index("https://apple.com")
    assert b"\x1b[2m" in raw


@pytest.mark.parametrize("columns", [80, 120])
def test_owner_rows_preserve_chronology_without_raw_json(columns):
    events = [
        {"type": "text.delta", "part_id": "intro", "text": "BEFORE_TOOLS"},
        {
            "type": "tool.started",
            "call_id": "edit",
            "name": "patch",
            "label": "Edit",
            "target": "calc.py",
        },
        {"type": "tool.output", "call_id": "edit", "text": "RAW_OUTPUT_MUST_NOT_LEAK"},
        {
            "type": "tool.result",
            "call_id": "edit",
            "outcome": "success",
            "summary": "+1 −1",
            "details": ["-    return a-b", "+    return a+b"],
            "details_kind": "diff",
            "result": '{"RAW_JSON_MUST_NOT_LEAK": true}',
        },
        {
            "type": "tool.started",
            "call_id": "run",
            "name": "execute_code",
            "label": "Run Python",
            "target": "",
        },
        {
            "type": "tool.result",
            "call_id": "run",
            "outcome": "success",
            "summary": "add(2, 3) = 5 … (exit 0, 0.1s)",
        },
        {
            "type": "context",
            "used_tokens": 5100,
            "limit_tokens": 8192,
            "notice": "Compacted older conversation into a summary checkpoint.",
        },
        {"type": "text.delta", "part_id": "answer", "text": "AFTER_TOOLS"},
        {"type": "turn.terminal", "outcome": "completed"},
    ]
    script = f"""
from rapid_mlx.chat_harness_render import HarnessRenderer
from rapid_mlx.chat_state import TurnState
state = TurnState('turn')
renderer = HarnessRenderer(state)
for seq, data in enumerate({events!r}, 1):
    e = {{'wire_version': 1, 'turn_id': 'turn', 'seq': seq, 'event': data}}
    state.apply(e)
    renderer.update(e)
renderer.finish()
"""
    display, raw = terminal_capture(script, columns=columns)
    ordered = [
        "BEFORE_TOOLS",
        "⏺ Edit  calc.py",
        "⎿ +1 −1",
        "-    return a-b",
        "+    return a+b",
        "⏺ Run Python",
        "add(2, 3) = 5",
        "Compacted older",
        "AFTER_TOOLS",
    ]
    positions = [display.index(item) for item in ordered]
    assert positions == sorted(positions), display
    assert all(display.count(item) == 1 for item in ordered), display
    assert "RAW_" not in display
    assert b"\x1b[32m" in raw and b"\x1b[31m" in raw


def test_approval_pause_commits_prior_rows_before_card():
    script = """
from rapid_mlx.chat_harness_render import HarnessRenderer
from rapid_mlx.chat_state import TurnState
state = TurnState('turn')
r = HarnessRenderer(state)
events = [
 {'type': 'tool.started', 'call_id': 'call', 'name': 'terminal', 'label': 'Run', 'target': 'dangerous delete'},
 {'type': 'tool.result', 'call_id': 'call', 'summary': 'Denied', 'outcome': 'denied'},
 {'type': 'text.delta', 'part_id': 'answer', 'text': 'Deletion was not run.'},
 {'type': 'turn.terminal', 'outcome': 'completed'},
]
for seq, data in enumerate(events, 1):
 e = {'wire_version': 1, 'turn_id': 'turn', 'seq': seq, 'event': data}
 state.apply(e); r.update(e)
 if seq == 1:
  r.pause(); print('APPROVAL_CARD'); r.resume()
r.finish()
"""
    display, _ = terminal_capture(script, columns=80)
    assert display.count("⏺ Run  dangerous delete") == 1
    assert (
        display.index("dangerous delete")
        < display.index("APPROVAL_CARD")
        < display.index("⎿ Denied")
        < display.index("Deletion was not run.")
    )


def test_parallel_tool_results_identify_their_owner_without_reordering():
    state = TurnState("turn")
    output = io.StringIO()
    renderer = HarnessRenderer(state, stream=output)
    items = [
        event(
            1,
            "tool.started",
            call_id="a",
            name="read_file",
            label="Read",
            target="a.py",
        ),
        event(
            2,
            "tool.started",
            call_id="b",
            name="read_file",
            label="Read",
            target="b.py",
        ),
        event(3, "tool.result", call_id="a", summary="2 lines", outcome="success"),
        event(4, "tool.result", call_id="b", summary="5 lines", outcome="success"),
    ]
    for item in items:
        state.apply(item)
        renderer.update(item)
    for display in (output.getvalue(), rendered(renderer)):
        assert (
            display.index("⏺ Read  a.py")
            < display.index("⏺ Read  b.py")
            < display.index("⎿ Read  a.py — 2 lines")
            < display.index("⎿ 5 lines")
        )


def test_completed_child_is_not_repeated_after_approval_pause():
    script = """
from rapid_mlx.chat_harness_render import HarnessRenderer
from rapid_mlx.chat_state import TurnState
s=TurnState('turn'); r=HarnessRenderer(s)
events = [
 {'type':'tool.started','call_id':'helper','name':'delegate_task','label':'Delegate','target':'Review'},
 {'type':'subagent.finished','id':'child','call_id':'helper','title':'Review files','status':'completed','tool_count':1,'summary':'Done'},
 {'type':'tool.result','call_id':'helper','outcome':'success','summary':'Helper completed'},
 {'type':'text.delta','part_id':'a','text':'Final answer'},
 {'type':'turn.terminal','outcome':'completed'},
]
for seq,data in enumerate(events,1):
 e={'wire_version':1,'turn_id':'turn','seq':seq,'event':data}
 s.apply(e); r.update(e)
 if seq==3:
  r.pause(); print('APPROVAL_CARD'); r.resume()
r.finish()
"""
    display, _ = terminal_capture(script, columns=80)
    assert display.count("↳ Review files · done (1 tool)") == 1, display
