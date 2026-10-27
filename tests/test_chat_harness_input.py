# SPDX-License-Identifier: Apache-2.0
"""TTY admission must not depend on a terminal continuously draining output."""

from __future__ import annotations

import json
import os
import sys
import time

import pytest

from rapid_mlx.chat_input import ChatInput


def completions(text, cursor=None):
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document

    from rapid_mlx.chat_client import COMMANDS
    from rapid_mlx.chat_input import command_completer

    return list(
        command_completer(COMMANDS).get_completions(
            Document(text, cursor), CompleteEvent(text_inserted=True)
        )
    )


def test_command_completion_registry_descriptions_and_prefix_ranking():
    from rapid_mlx.chat_client import COMMANDS

    menu = completions("/")
    assert [item.display_text for item in menu] == [item.name for item in COMMANDS]
    assert [item.display_meta_text for item in menu] == [
        item.description for item in COMMANDS
    ]
    assert [item.text for item in completions("/th")] == [
        "/think ",
        "/show-thinking",
    ]
    assert [item.text for item in completions("/att")] == ["/attach "]
    assert [item.text for item in completions("/he")] == ["/help"]
    assert completions("/th")[0].start_position == -3
    assert completions("/think", cursor=3)[0].text == "/think "


@pytest.mark.parametrize(
    "text",
    ["", "hello /th", " /th", "/unknown", "/new ", "/edit /th", "/history 1", "/th\n"],
)
def test_command_menu_stays_out_of_prose_and_uncompleted_arguments(text):
    assert completions(text) == []


def test_think_argument_completion():
    assert [item.text for item in completions("/think ")] == ["on", "off"]
    assert [item.text for item in completions("/think o")] == ["on", "off"]
    assert [item.text for item in completions("/think of")] == ["off"]
    assert completions("/think on") == []
    assert completions("/think off ") == []
    assert completions("/think on", cursor=8) == []


def test_attachment_completion_expands_home_and_quotes_paths(tmp_path, monkeypatch):
    import shlex

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.txt").write_text("note")
    (tmp_path / "海洋 notes.txt").write_text("海洋")
    (tmp_path / "notes folder").mkdir()
    menu = completions("/attach ~/no")
    assert [shlex.split(item.text)[0] for item in menu] == [
        "~/notes folder/",
        "~/notes.txt",
    ]
    assert all(item.start_position == -4 for item in menu)
    assert {item.display_text for item in completions("/attach ")} == {
        "notes.txt",
        "海洋 notes.txt",
        "notes folder/",
    }
    for argument in ("海洋", "'海洋", '"海洋', "海洋\\ n"):
        item = completions("/attach " + argument)[0]
        assert shlex.split(item.text) == ["海洋 notes.txt"]
    assert completions("/attach notes.txt") == []
    assert completions("/attach notes.txt second") == []


def wait_for_lines(path, count, timeout=2.5):
    """Observe admission out of band; deliberately never read the PTY here."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        lines = path.read_text().splitlines() if path.exists() else []
        if len(lines) >= count:
            return [json.loads(line) for line in lines]
        time.sleep(0.01)
    pytest.fail(
        f"Input was not admitted while PTY output was paused ({count} expected)"
    )


def spawn_reader(tmp_path):
    pexpect = pytest.importorskip("pexpect")
    marker = tmp_path / "admitted.jsonl"
    script = """
import json, sys
from rapid_mlx.chat_input import ChatInput
reader = ChatInput()
try:
    while True:
        try:
            text = reader.prompt('› ')
        except KeyboardInterrupt:
            print('Interrupted', flush=True)
            continue
        with open(sys.argv[1], 'a') as file:
            file.write(json.dumps(text) + '\\n')
        if text == '/quit':
            break
        if text == '/history':
            print('1. A saved conversation\\nUse /history NUMBER to resume.')
finally:
    reader.close()
"""
    child = pexpect.spawn(
        sys.executable,
        ["-c", script, str(marker)],
        encoding="utf-8",
        timeout=5,
        dimensions=(40, 110),
        env={**os.environ, "TERM": "xterm-256color"},
    )
    child.expect("› ")
    return child, marker


def test_tty_submission_after_history_does_not_wait_for_output_drain(tmp_path):
    child, marker = spawn_reader(tmp_path)
    try:
        child.sendline("/history")
        child.expect("Use /history NUMBER to resume.")
        child.expect("› ")
        text = "写一篇800字的关于海洋的文章"
        child.sendline(text)
        # The prior libedit input() blocks here until echoed output is read;
        # Ctrl-C then discards this unadmitted line. Do not mask it with expect().
        submitted = wait_for_lines(marker, 2)
        assert submitted[-1] == text
        child.sendcontrol("c")
        child.expect("Interrupted")
        child.expect("› ")
        child.sendline("继续用一句话总结海洋")
        assert wait_for_lines(marker, 3)[-1] == "继续用一句话总结海洋"
    finally:
        child.terminate(force=True)


def test_tty_editing_history_bracketed_paste_and_queued_lines(tmp_path):
    child, marker = spawn_reader(tmp_path)
    try:
        child.send("helloXX\x7f\x7f world\r")
        assert wait_for_lines(marker, 1)[-1] == "hello world"
        child.expect("› ")
        child.send("\x1b[A\r")
        assert wait_for_lines(marker, 2)[-1] == "hello world"
        child.expect("› ")
        pasted = "海洋研究\n第二行"
        child.send("\x1b[200~" + pasted + "\x1b[201~\r")
        assert wait_for_lines(marker, 3)[-1] == pasted
        child.expect("› ")
        child.send("first queued\rsecond queued\r")
        assert wait_for_lines(marker, 5)[-2:] == ["first queued", "second queued"]
    finally:
        child.terminate(force=True)


def test_non_tty_uses_standard_input(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    prompts = []
    monkeypatch.setattr("builtins.input", lambda text: prompts.append(text) or "piped")
    reader = ChatInput()
    assert reader.prompt("› ") == "piped"
    assert prompts == ["› "]
    assert reader._session is None
    reader.close()


def test_slash_commands_share_one_reader_without_consuming_next_prompt(tmp_path):
    """Audit every command family, including the one explicit /edit subprompt."""
    pexpect = pytest.importorskip("pexpect")
    marker = tmp_path / "turns.jsonl"
    attached = tmp_path / "attachment.txt"
    attached.write_text("Attachment fixture")
    script = """
import json, sys
from types import SimpleNamespace
from rapid_mlx.chat_client import ChatClient
snapshot = {'id':'conversation','branch_id':'branch','branches':[{'id':'branch','revision':0}], 'messages':[{'id':'user','role':'user','parts':[{'type':'text','text':'Original question'}]}]}
client = ChatClient(SimpleNamespace(profile='input-test',think=False,response_timeout=1), 'http://127.0.0.1:18800', 'fixture')
client.connect = lambda: None
client.heartbeat = lambda: client.stopping.wait()
def request(method, path, **kwargs):
    if path == '/conversations' and method == 'GET':
        return [{**snapshot,'updated_at':'2026-10-25T12:00:00+00:00','message_count':1,'title':'Original question'}]
    if path.endswith('/attachments'): return {'id':'attachment'}
    if path.endswith('/edit') or path.endswith('/regenerate'): return {'id':'branch','branch_id':'branch'}
    return snapshot
client.request = request
def turn(text, **kwargs):
    with open(sys.argv[1], 'a') as file:
        file.write(json.dumps(text) + '\\n')
client.turn = turn
try: client.run()
finally: client.close()
"""
    child = pexpect.spawn(
        sys.executable,
        ["-c", script, str(marker)],
        encoding="utf-8",
        timeout=8,
        dimensions=(40, 110),
        env={**os.environ, "TERM": "xterm-256color", "NO_COLOR": "1"},
    )
    child.expect("› ")
    commands = [
        "/help",
        "/?",
        "/history",
        "/history 1",
        "/branch",
        "/branch 1",
        "/think on",
        "/think off",
        "/show-thinking",
        "/show-thinking",
        "/new",
        "/reset",
        "/clear",
        f"/attach {attached}",
        "/export",
        f"/export {tmp_path / 'export.md'}",
        f"/save {tmp_path / 'saved.md'}",
        "/regen",
        "/edit replacement",
        "/unknown",
    ]
    expected = 0
    try:
        for index, command in enumerate(commands):
            child.sendline(command)
            child.expect(r"(?:Thinking )?› ")
            if command.startswith(("/regen", "/edit")):
                expected += 1
            sentinel = f"下一条消息 {index}"
            child.sendline(sentinel)
            expected += 1
            assert wait_for_lines(marker, expected)[-1] == sentinel
            child.expect(r"(?:Thinking )?› ")
        child.sendline("/edit")
        child.expect("Edited message> ")
        child.sendline("明确的编辑子提示")
        expected += 1
        child.expect("› ")
        child.sendline("编辑后下一条消息")
        expected += 1
        assert wait_for_lines(marker, expected)[-1] == "编辑后下一条消息"
        child.expect("› ")
        child.sendline("/quit")
        child.expect(pexpect.EOF)
    finally:
        child.terminate(force=True)


class PaletteScreen:
    """Feed every PTY read into a terminal emulator, including expect() reads."""

    def __init__(self, child):
        pyte = pytest.importorskip("pyte")
        self.screen = pyte.Screen(80, 24)
        self.stream = pyte.Stream(self.screen)
        self.child = child
        self.raw = ""
        child.logfile_read = self

    def write(self, text):
        self.raw += text
        self.stream.feed(text)

    def flush(self):
        pass

    def dump(self):
        return "\n".join(line.rstrip() for line in self.screen.display)

    def wait_for(self, *present, absent=()):
        def matches(text):
            return all(item in text for item in present) and not any(
                item in text for item in absent
            )

        return self.wait(matches)

    def wait(self, predicate):
        import pexpect

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if predicate(self.dump()):
                return self.dump()
            try:
                self.child.read_nonblocking(65536, timeout=0.1)
            except pexpect.TIMEOUT:
                pass
        pytest.fail("Terminal did not reach expected screen:\n" + self.dump())


def spawn_palette(tmp_path):
    pexpect = pytest.importorskip("pexpect")
    marker = tmp_path / "commands.jsonl"
    script = """
import json, sys
from types import SimpleNamespace
from rapid_mlx.chat_client import ChatClient
client = ChatClient(SimpleNamespace(), 'http://127.0.0.1:18800', 'fixture')
try:
    while True:
        text = client.input.prompt(client.prompt_text())
        with open(sys.argv[1], 'a') as file:
            file.write(json.dumps(text) + '\\n')
        if text.startswith('/') and not client.command(text):
            break
finally:
    client.input.close()
    client.client.close()
"""
    child = pexpect.spawn(
        sys.executable,
        ["-c", script, str(marker)],
        encoding="utf-8",
        timeout=5,
        dimensions=(24, 80),
        env={**os.environ, "TERM": "xterm-256color"},
    )
    screen = PaletteScreen(child)
    child.expect("› ")
    return child, marker, screen


def test_slash_palette_screen_navigation_acceptance_and_execution(tmp_path):
    from rapid_mlx.chat_client import COMMANDS

    child, marker, screen = spawn_palette(tmp_path)
    try:
        child.send("/")
        slash = screen.wait(lambda text: "Exit (also Ctrl-D)" in text)
        for command in COMMANDS:
            assert command.name in slash and command.description in slash
        child.send("th")
        filtered = screen.wait_for(
            "› /th", "Turn thinking on or off", absent=("Start a conversation",)
        )
        assert filtered.index("/think") < filtered.index("/show-thinking")
        (tmp_path / "menu-screens.json").write_text(
            json.dumps({"/": slash, "/th": filtered}, ensure_ascii=False, indent=2)
        )
        child.send("\x1b[B\t")  # Down selects /think; Tab accepts it.
        screen.wait(lambda text: "› /think" in text and "Expand/collapse" not in text)
        assert not marker.exists()  # Neither navigation nor Tab submitted.
        child.send("on\r")
        child.expect("Thinking on.")
        child.expect("Thinking › ")
        assert wait_for_lines(marker, 1) == ["/think on"]
        assert "Turn thinking on or off" not in screen.dump()
        assert "Expand/collapse thinking" not in screen.dump()

        child.send("/th")
        screen.wait(lambda text: "Expand/collapse thinking" in text)
        # Up picks the last match; Down wraps, then Down selects it again.
        child.send("\x1b[A\x1b[B\x1b[B\r")
        screen.wait_for("Thinking › /show-thinking", absent=("Expand/collapse",))
        assert len(marker.read_text().splitlines()) == 1
        child.send("\r")
        child.expect("Thinking expanded.")
        child.expect("Thinking › ")
        assert wait_for_lines(marker, 2)[-1] == "/show-thinking"

        # Fully typed no-argument commands submit with one Enter even with
        # their menu visible; the menu must disappear on the final repaint.
        child.send("/he")
        screen.wait(lambda text: "Show commands" in text)
        child.send("lp\r")
        child.expect("Commands")
        child.expect("Thinking › ")
        assert wait_for_lines(marker, 3)[-1] == "/help"
        child.sendline("海洋研究")
        assert wait_for_lines(marker, 4)[-1] == "海洋研究"
        child.expect("Thinking › ")
        child.sendline("/quit")
        import pexpect

        child.expect(pexpect.EOF)
    finally:
        child.terminate(force=True)


def test_palette_enter_accepts_required_argument_and_edits_command_middle(tmp_path):
    child, marker, screen = spawn_palette(tmp_path)
    try:
        child.send("/think")
        screen.wait(lambda text: "Turn thinking on or off" in text)
        child.send("\x1b[B\r")
        screen.wait_for("› /think", absent=("Turn thinking on or off",))
        assert not marker.exists()
        child.send("of")
        screen.wait(
            lambda text: any(line.strip() == "off" for line in text.splitlines())
        )
        child.send("\x1b[B\r")
        screen.wait(lambda text: "› /think off" in text)
        assert not marker.exists()
        child.send("\r")
        child.expect("Thinking off.")
        child.expect("› ")
        assert wait_for_lines(marker, 1) == ["/think off"]

        child.send("/thXYZ on\x01\x1b[C\x1b[C\x1b[C\t")
        screen.wait_for("› /think on", absent=("Turn thinking on or off",))
        child.send("\x05\r")
        child.expect("Thinking on.")
        assert wait_for_lines(marker, 2)[-1] == "/think on"
    finally:
        child.terminate(force=True)


@pytest.mark.parametrize("percent", [49, 62, 90])
def test_context_indicator_right_of_prompt_only_above_threshold(percent):
    import pexpect

    script = f"""
from rapid_mlx.chat_input import ChatInput
r = ChatInput()
r.context_percent = {percent}
try: r.prompt('› ')
finally: r.close()
"""
    child = pexpect.spawn(
        sys.executable,
        ["-c", script],
        env={**os.environ, "TERM": "xterm-256color"},
        encoding="utf-8",
        timeout=5,
        dimensions=(24, 80),
    )
    screen = PaletteScreen(child)
    try:
        child.expect_exact("› ")
        if percent >= 50:
            screen.wait_for(f"context {percent}%")
            row = screen.screen.display[screen.screen.cursor.y]
            assert row.rstrip().endswith(f"context {percent}%")
            assert row.index("context") > 60
        else:
            assert "context" not in screen.dump()
        child.sendline("done")
        child.expect(pexpect.EOF)
    finally:
        child.close(force=True)
