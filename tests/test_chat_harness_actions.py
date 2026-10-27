# SPDX-License-Identifier: Apache-2.0
"""Actual terminal approval keys, safe previews, timeout, and reader reuse."""

import io
import os
import sys
from types import SimpleNamespace

import pexpect
import pyte
import pytest

from rapid_mlx.chat_client import ChatClient
from rapid_mlx.chat_input import ChatInput


class ScreenLog:
    def __init__(self):
        self.screen = pyte.HistoryScreen(100, 30, history=2000)
        self.stream = pyte.Stream(self.screen)
        self.raw = ""

    def write(self, value):
        self.raw += value
        self.stream.feed(value)

    def flush(self):
        pass

    @property
    def text(self):
        return "\n".join(
            [
                "".join(line[i].data for i in range(100))
                for line in self.screen.history.top
            ]
            + self.screen.display
        )


@pytest.mark.parametrize(
    "key,conversation,decision,scope",
    [
        ("y", False, "allow", "once"),
        ("1", False, "allow", "once"),
        ("2", False, "deny", "once"),
        ("\x1b[B\r", False, "deny", "once"),
        ("2", True, "allow", "conversation"),
        ("n", False, "deny", "once"),
        ("a", True, "allow", "conversation"),
        ("an", False, "deny", "once"),
    ],
)
def test_pty_approval_keys_and_diff_preview(key, conversation, decision, scope):
    script = r"""
import json,time,sys
from types import SimpleNamespace
from rapid_mlx.chat_client import ChatClient
from rapid_mlx.chat_harness_render import HarnessRenderer
from rapid_mlx.chat_state import TurnState
client=ChatClient(SimpleNamespace(), 'http://127.0.0.1:18800', 'fixture')
client.request=lambda method,path,**kw: ({'state':'intent','revision':0} if method=='GET' else print('DECISION '+json.dumps(kw['json']),flush=True))
renderer=HarnessRenderer(TurnState('turn'))
try:
    client.permission({'op_id':'op','revision':0,'tool':'patch','preview':'--- calc.py\n+++ calc.py\n@@ -1 +1 @@\n-return a - b\n+return a + b\n\x1b[2J',
      'reason':'Confirm this change','expires_at':time.time()+10,'options':['once','conversation','deny'] if sys.argv[1]=='True' else ['once','deny']},renderer)
    renderer.finish()
    print('NEXT '+client.input.prompt('Next> '),flush=True)
finally:
    client.input.close();client.client.close()
"""
    env = dict(os.environ, TERM="xterm-256color", NO_COLOR="1")
    child = pexpect.spawn(
        sys.executable,
        ["-c", script, str(conversation)],
        env=env,
        encoding="utf-8",
        timeout=10,
        dimensions=(30, 100),
    )
    log = ScreenLog()
    child.logfile_read = log
    try:
        child.expect_exact("No, and tell it why…")
        assert "--- calc.py" in log.text and "+return a + b" in log.text
        assert ("Yes, for this conversation" in log.text) == conversation
        child.send(key)
        child.expect_exact('"decision": "' + decision + '"')
        child.expect_exact('"scope": "' + scope + '"')
        child.expect_exact("Next> ")
        child.sendline("next message")
        child.expect_exact("NEXT next message")
        child.expect(pexpect.EOF)
    finally:
        child.close(force=True)


def test_pty_approval_timeout_fails_closed():
    script = "from rapid_mlx.chat_input import ChatInput; r=ChatInput(); print('RESULT',r.approval(timeout=0.1)); r.close()"
    child = pexpect.spawn(
        sys.executable,
        ["-c", script],
        env=dict(os.environ, TERM="xterm"),
        encoding="utf-8",
        timeout=5,
    )
    try:
        child.expect_exact("RESULT n")
        child.expect(pexpect.EOF)
    finally:
        child.close(force=True)


def test_pty_stop_during_approval_returns_interrupt():
    script = """
from rapid_mlx.chat_input import ChatInput
r=ChatInput()
try: r.approval()
except KeyboardInterrupt: print('STOPPED',flush=True)
finally: r.close()
"""
    child = pexpect.spawn(
        sys.executable,
        ["-c", script],
        env=dict(os.environ, TERM="xterm"),
        encoding="utf-8",
        timeout=5,
    )
    try:
        child.expect_exact("No, and tell it why…")
        child.sendcontrol("c")
        child.expect_exact("STOPPED")
        child.expect(pexpect.EOF)
    finally:
        child.close(force=True)


def test_noninteractive_approval_never_consumes_chat_input(monkeypatch):
    stdin = io.StringIO("y\nnext prompt\n")
    monkeypatch.setattr(sys, "stdin", stdin)
    reader = ChatInput()
    assert reader.approval() == "n"
    assert stdin.read() == "y\nnext prompt\n"


def test_client_undo_uses_stable_request_id(capsys):
    client = ChatClient(SimpleNamespace(), "http://127.0.0.1:18800", "fixture")
    calls = []
    client.conversation_id = "conversation"
    client.request = lambda method, path, **kw: (
        calls.append((method, path, kw))
        or {
            "state": "succeeded",
            "receipt": {"summary": "Restored calc.py"},
        }
    )
    try:
        client.command("/undo")
        assert calls[0][0:2] == ("POST", "/conversations/conversation/undo")
        assert calls[0][2]["json"]["operation_id"]
        assert "Restored calc.py" in capsys.readouterr().out
    finally:
        client.client.close()


def test_pty_optional_denial_reason_does_not_enable_approval_keys_in_reason():
    script = """
from rapid_mlx.chat_input import ChatInput
r=ChatInput()
try:
    print('RESULT',r.approval(),repr(r.approval_reason),flush=True)
finally: r.close()
"""
    child = pexpect.spawn(
        sys.executable,
        ["-c", script],
        env=dict(os.environ, TERM="xterm"),
        encoding="utf-8",
        timeout=5,
    )
    try:
        child.expect_exact("No, and tell it why…")
        child.send("N")
        child.expect_exact("Denial reason (Enter skips)> ")
        child.sendline("need artifacts today")
        child.expect_exact("RESULT n 'need artifacts today'")
        child.expect(pexpect.EOF)
    finally:
        child.close(force=True)


def test_approval_number_admission_does_not_wait_for_output_drain(tmp_path):
    from tests.test_chat_harness_input import wait_for_lines

    marker = tmp_path / "approval.jsonl"
    script = """
import json, sys
from rapid_mlx.chat_input import ChatInput
r = ChatInput()
try:
 decision = r.approval(data={'title':'Run Python code?', 'reason':'Needs network access: example.com', 'preview':'print(1)\\n' * 30, 'preview_language':'python'})
 with open(sys.argv[1], 'w') as f: f.write(json.dumps(decision) + '\\n')
finally: r.close()
"""
    child = pexpect.spawn(
        sys.executable,
        ["-c", script, str(marker)],
        env=dict(os.environ, TERM="xterm"),
        encoding="utf-8",
        timeout=5,
        dimensions=(30, 80),
    )
    try:
        child.expect_exact("v to view all")
        child.expect_exact("No, and tell it why…")
        child.send("1")
        assert wait_for_lines(marker, 1) == ["y"]
    finally:
        child.close(force=True)


def test_approval_v_expands_preview_without_deciding():
    script = """
from rapid_mlx.chat_input import ChatInput
r=ChatInput()
try: print('RESULT', r.approval(data={'title':'Run Python code?', 'preview':'\\n'.join('print('+str(i)+')' for i in range(12)), 'preview_language':'python'}),flush=True)
finally: r.close()
"""
    child = pexpect.spawn(
        sys.executable,
        ["-c", script],
        env=dict(os.environ, TERM="xterm", NO_COLOR="1"),
        encoding="utf-8",
        timeout=5,
        dimensions=(30, 80),
    )
    try:
        child.expect_exact("6 more lines  (v to view all)")
        child.expect_exact("No, and tell it why…")
        child.send("v")
        child.expect_exact("print(11)")
        child.expect_exact("v to collapse")
        child.expect_exact("No, and tell it why…")
        child.send("2")
        child.expect_exact("RESULT n")
        child.expect(pexpect.EOF)
    finally:
        child.close(force=True)
