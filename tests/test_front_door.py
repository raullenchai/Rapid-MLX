# SPDX-License-Identifier: Apache-2.0
"""Bare ``rapid-mlx`` front door (rapid_mlx/front_door.py).

Fully offline: no model load, no download, no network, no real HF cache.
Key presses are injected; the in-process dispatch is replaced by a recorder.
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

import rapid_mlx.cli as cli
import rapid_mlx.first_run as fr
import rapid_mlx.front_door as fd
from rapid_mlx.recommendations import load_recommendation_tiers, recommendation_payload


def _pick(alias, role, *, size=3.0, cached=False, tps=50.0, caveat=None, fit=True):
    return {
        "alias": alias,
        "role": role,
        "download_size_gb": 0.0 if cached else size,
        "cached": cached,
        "tokens_per_sec": tps,
        "caveat": caveat,
        "disk_fit": fit,
    }


def _state(**overrides) -> fd.FrontDoorState:
    picks = overrides.pop(
        "picks",
        [
            _pick("qwen3.5-9b-4bit", "smart", size=5.6, tps=35.7),
            _pick("qwen3.5-4b-4bit", "fast", size=2.9, tps=60.7),
        ],
    )
    overrides.setdefault("ram_gb", 18.0)
    overrides.setdefault("chip", "Apple M3 Pro")
    state = fd.FrontDoorState(version="9.9.9", picks=picks, **overrides)
    if not state.selected:
        state.selected = fd._default_model(state)
    return state


def _keys(*keys):
    seq = iter(keys)
    return lambda: next(seq)


# ======================================================================
# Model choice: the same policy as `rapid-mlx recipe`
# ======================================================================
@pytest.mark.parametrize(
    "ram_gb",
    sorted({float(t.floor_gb) for t in load_recommendation_tiers()} | {0.0, 12.0}),
)
def test_cold_default_is_always_one_of_recipes_picks(ram_gb, monkeypatch):
    payload = recommendation_payload(ram_gb, validate_catalog=False)
    monkeypatch.setattr("rapid_mlx.cli._scan_hf_cache_models", lambda: [])
    monkeypatch.setattr("rapid_mlx.cli._recipe_free_disk_gb", lambda: None)
    picks = fd._picks_for(ram_gb, [])
    assert [p["alias"] for p in picks] == [p["alias"] for p in payload["picks"]]
    state = _state(picks=picks, ram_gb=ram_gb)
    assert state.selected in {p["alias"] for p in payload["picks"]}


def test_quick_start_prefers_no_caveat_then_smaller_download():
    picks = [
        _pick("big", "smart", size=9.0),
        _pick("small-basic", "fast", size=1.0, caveat="Basic chat"),
    ]
    assert fd.quick_start_pick(picks)["alias"] == "big"
    picks = [_pick("a", "smart", size=5.0), _pick("b", "fast", size=2.0)]
    assert fd.quick_start_pick(picks)["alias"] == "b"
    picks = [_pick("a", "smart", size=None), _pick("b", "fast", size=None)]
    assert fd.quick_start_pick(picks)["alias"] == "b"  # tie → the fast pick


def test_18gb_cold_default_matches_recipe_fast_pick(monkeypatch):
    monkeypatch.setattr("rapid_mlx.cli._recipe_free_disk_gb", lambda: None)
    state = _state(picks=fd._picks_for(18.0, []))
    assert state.selected == "qwen3.5-4b-4bit"
    roles = {p["alias"]: p["role"] for p in state.picks}
    assert roles[state.selected] == "fast"


def test_default_priority_server_then_last_used_then_cached_then_cold():
    server = fd.ServerInfo(8000, "qwen3.5-9b-4bit")
    assert _state(server=server).selected == "qwen3.5-9b-4bit"
    # A non-chat server model is shown but never becomes the Enter default.
    emb = fd.ServerInfo(8000, "embeddinggemma-300m-8bit")
    assert _state(server=emb).selected == "qwen3.5-4b-4bit"
    cached = ["lfm2.5-1b-4bit", "qwen3.5-9b-4bit"]
    assert _state(cached=cached, last_used="lfm2.5-1b-4bit").selected == (
        "lfm2.5-1b-4bit"
    )
    # Last used but since deleted → fall back to a cached policy candidate.
    assert _state(cached=["qwen3.5-9b-4bit"], last_used="gone").selected == (
        "qwen3.5-9b-4bit"
    )
    assert _state(cached=["some-unlisted"]).selected == "qwen3.5-4b-4bit"


def test_default_survives_candidate_lookup_failure(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("policy unreadable")

    monkeypatch.setattr("rapid_mlx.recommendations.starter_model_candidates", boom)
    assert _state(cached=["qwen3.5-9b-4bit"]).selected == "qwen3.5-4b-4bit"


def test_can_chat(monkeypatch):
    assert fd._can_chat("qwen3.5-4b-4bit") is True
    assert fd._can_chat("org/unknown-repo") is True
    assert fd._can_chat("embeddinggemma-300m-8bit") is False
    monkeypatch.setattr(
        "rapid_mlx.model_aliases.list_profiles",
        lambda: (_ for _ in ()).throw(RuntimeError("x")),
    )
    assert fd._can_chat("qwen3.5-4b-4bit") is False


# ======================================================================
# Rendering
# ======================================================================
def test_first_run_screen_fits_and_names_every_action():
    text = fd.render_screen(_state(agent="claude-code"))
    lines = text.split("\n")
    assert len(lines) <= 20
    assert lines[0] == "rapid-mlx 9.9.9 · Apple M3 Pro · 18 GB"
    assert lines[1] == fd.PRODUCT_LINE
    assert "▸ qwen3.5-4b-4bit   Fast · ~61 tok/s · 2.9 GB download" in text
    assert "  qwen3.5-9b-4bit   Smart · ~36 tok/s · 5.6 GB download" in text
    assert "Enter  Start chatting with qwen3.5-4b-4bit" in text
    assert "c      Connect Claude Code (detected)" in text
    assert "s      Start the server" in text
    assert "m      Choose another model" in text
    assert "q      Quit" in text
    # No port-collision recovery lines and no ASCII banner on this screen.
    assert "8001" not in text
    assert "busy" not in text
    assert max(len(line) for line in lines) <= 80


def test_screen_without_agent_hides_c():
    text = fd.render_screen(_state(agent=None))
    assert "Connect" not in text
    assert "\n  c " not in text


def test_returning_screen_with_server_and_last_used():
    state = _state(
        server=fd.ServerInfo(8000, "qwen3.5-9b-4bit"),
        cached=["qwen3.5-9b-4bit"],
        last_used="qwen3.5-9b-4bit",
        agent="cline",
    )
    text = fd.render_screen(state)
    assert "Server running on :8000 · qwen3.5-9b-4bit" in text
    assert "Last used: qwen3.5-9b-4bit" in text
    assert "Enter  Chat with qwen3.5-9b-4bit on :8000" in text
    assert "c      Connect Cline to the server on :8000" in text
    assert len(text.split("\n")) <= 20


def test_pick_facts_variants():
    assert fd._pick_facts(_pick("a", "fast", cached=True)) == (
        "Fast · ~50 tok/s · downloaded"
    )
    facts = fd._pick_facts(
        _pick("a", "smart", caveat="Not for coding", tps=None, fit=False)
    )
    assert facts == "Smart · Not for coding · 3.0 GB download · not enough free disk"
    assert fd._pick_facts({"alias": "x", "role": "smart"}) == "Smart"


def test_header_without_chip_or_ram():
    state = _state(chip=None, ram_gb=0.0)
    assert fd._header(state) == "rapid-mlx 9.9.9"
    assert fd._header(_state(ram_gb=36.5)) == (
        "rapid-mlx 9.9.9 · Apple M3 Pro · 36.5 GB"
    )


def test_agent_label_falls_back_to_name():
    assert fd.agent_label("claude-code") == "Claude Code"
    assert fd.agent_label("new-agent") == "new-agent"


def test_picker_lists_policy_picks_then_downloaded_with_fit_marks(monkeypatch):
    state = _state(cached=["qwen3.5-4b-4bit", "qwen3.8-27b-4bit"])
    entries = fd.picker_entries(state)
    assert [a for a, _ in entries] == [
        "qwen3.5-9b-4bit",
        "qwen3.5-4b-4bit",
        "qwen3.8-27b-4bit",
    ]
    assert entries[0][1].endswith("fits this Mac")
    assert entries[2][1] == "downloaded · too big for this Mac"
    text = fd.render_picker(entries)
    assert "  3  qwen3.8-27b-4bit" in text
    assert "More models: rapid-mlx models" in text


def test_picker_caps_entries_and_tolerates_fit_errors(monkeypatch):
    monkeypatch.setattr(
        "rapid_mlx.run.cli._fits_host",
        lambda *_a: (_ for _ in ()).throw(RuntimeError("x")),
    )
    state = _state(cached=[f"m{i}" for i in range(20)])
    entries = fd.picker_entries(state)
    assert len(entries) == fd._PICKER_LIMIT
    assert entries[-1][1] == "downloaded"
    unknown_ram = _state(ram_gb=0.0, cached=["m1"])
    assert fd.picker_entries(unknown_ram)[-1] == ("m1", "downloaded")


def test_non_tty_text_is_short_and_copy_pasteable():
    picks = [_pick("qwen3.5-9b-4bit", "smart"), _pick("qwen3.5-4b-4bit", "fast")]
    text = fd.render_non_tty("9.9.9", 18.0, picks)
    lines = text.split("\n")
    assert len(lines) <= 15
    assert "Recommended for this Mac (18 GB): qwen3.5-4b-4bit (fast)" in text
    assert "  rapid-mlx pull qwen3.5-4b-4bit" in lines
    assert "  rapid-mlx serve qwen3.5-4b-4bit --port 8000" in lines
    assert "  rapid-mlx chat qwen3.5-4b-4bit" in lines
    assert "\x1b[" not in text
    assert "(18 GB)" not in fd.render_non_tty("9.9.9", 0.0, picks)


# ======================================================================
# Actions
# ======================================================================
def test_enter_chats_with_selected_model():
    plan = fd.run_interactive(_state(), key_reader=_keys(fd.KEY_ENTER), out=str)
    assert plan.steps == [["chat", "qwen3.5-4b-4bit"]]


def test_enter_attaches_to_running_server_for_its_model():
    state = _state(server=fd.ServerInfo(8123, "qwen3.5-9b-4bit"))
    plan = fd.run_interactive(state, key_reader=_keys(fd.KEY_ENTER), out=str)
    assert plan.steps == [["chat", "qwen3.5-9b-4bit", "--port", "8123"]]


def test_s_serves_on_a_free_port():
    plan = fd.run_interactive(
        _state(), key_reader=_keys("s"), out=str, free_port=lambda: 8000
    )
    assert plan.steps == [["serve", "qwen3.5-4b-4bit"]]
    assert "http://127.0.0.1:8000/v1" in plan.note
    plan = fd.run_interactive(
        _state(), key_reader=_keys("s"), out=str, free_port=lambda: 8002
    )
    assert plan.steps == [["serve", "qwen3.5-4b-4bit", "--port", "8002"]]


def test_c_pulls_serves_and_launches_only_once_ready():
    plan = fd.run_interactive(
        _state(agent="claude-code"),
        key_reader=_keys("c"),
        out=str,
        free_port=lambda: 8001,
    )
    assert plan.steps == [
        ["pull", "qwen3.5-4b-4bit"],
        ["serve", "qwen3.5-4b-4bit", "--port", "8001"],
    ]
    assert plan.after_ready == [
        "launch",
        "claude-code",
        "--model",
        "qwen3.5-4b-4bit",
        "--server-url",
        "http://127.0.0.1:8001",
    ]
    assert "Claude Code" in plan.note


def test_c_against_running_server_only_launches():
    state = _state(agent="claude-code", server=fd.ServerInfo(8000, "qwen3.5-9b-4bit"))
    plan = fd.run_interactive(
        state,
        key_reader=_keys("c"),
        out=str,
        free_port=lambda: pytest.fail("no port search when attaching"),
    )
    assert plan.steps == [
        [
            "launch",
            "claude-code",
            "--model",
            "qwen3.5-9b-4bit",
            "--server-url",
            "http://127.0.0.1:8000",
        ]
    ]


def test_c_is_ignored_without_a_detected_agent_and_unknown_keys_are_ignored():
    plan = fd.run_interactive(
        _state(agent=None), key_reader=_keys("c", "x", fd.KEY_UNKNOWN, "q"), out=str
    )
    assert plan is None


@pytest.mark.parametrize("key", [fd.KEY_QUIT, "q"])
def test_quit_keys(key):
    assert fd.run_interactive(_state(), key_reader=_keys(key), out=str) is None


def test_ctrl_c_quits():
    def interrupted():
        raise KeyboardInterrupt

    assert fd.run_interactive(_state(), key_reader=interrupted, out=str) is None


def test_m_picks_another_model_and_redraws():
    shown: list[str] = []
    plan = fd.run_interactive(
        _state(),
        key_reader=_keys("m", "9", "x", "1", fd.KEY_ENTER),
        out=shown.append,
    )
    assert plan.steps == [["chat", "qwen3.5-9b-4bit"]]
    assert any("Choose a model" in s for s in shown)
    assert shown[-1].startswith(fd._CLEAR_SCREEN)
    assert "▸ qwen3.5-9b-4bit" in shown[-1]


@pytest.mark.parametrize("back", [fd.KEY_QUIT, "q", fd.KEY_ENTER])
def test_m_back_keeps_selection(back):
    plan = fd.run_interactive(
        _state(), key_reader=_keys("m", back, fd.KEY_ENTER), out=str
    )
    assert plan.steps == [["chat", "qwen3.5-4b-4bit"]]


def test_ctrl_c_inside_picker_quits():
    calls = iter(["m"])

    def reader():
        try:
            return next(calls)
        except StopIteration:
            raise KeyboardInterrupt from None

    assert fd.run_interactive(_state(), key_reader=reader, out=str) is None


def test_first_free_port():
    taken = {8000, 8001}
    assert fd.first_free_port(is_free=lambda p: p not in taken) == 8002
    assert fd.first_free_port(span=2, is_free=lambda p: False) is None


@pytest.mark.parametrize("key", ["s", "c"])
def test_no_free_port_keeps_the_menu_open(key):
    shown: list[str] = []
    plan = fd.run_interactive(
        _state(agent="claude-code"),
        key_reader=_keys(key, "q"),
        out=shown.append,
        free_port=lambda: None,
    )
    assert plan is None
    assert any("No free port in 8000-8099" in line for line in shown)


def test_renamed_server_is_shown_but_never_attached():
    server = fd.ServerInfo(8000, "studio-assistant", attachable=False)
    state = _state(server=server, agent="claude-code")
    assert state.selected == "qwen3.5-4b-4bit"
    assert state.attach_port is None
    text = fd.render_screen(state)
    assert "Server running on :8000 · studio-assistant" in text
    assert "Enter  Start chatting with qwen3.5-4b-4bit" in text


def test_port_is_free_detects_a_bound_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        assert fd._port_is_free(port) is False
    assert isinstance(fd._port_is_free(port), bool)


def test_echo_command():
    assert fd.echo_command(["chat", "m"]) == "→ rapid-mlx chat m"


# ======================================================================
# Terminal detection and key reading
# ======================================================================
def test_interactive_terminal_requires_both_ttys_and_a_real_term(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    assert fd.interactive_terminal() is True
    monkeypatch.setenv("TERM", "dumb")
    assert fd.interactive_terminal() is False
    monkeypatch.setenv("TERM", "xterm")
    monkeypatch.setenv("CI", "true")
    assert fd.interactive_terminal() is False
    monkeypatch.delenv("CI")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    assert fd.interactive_terminal() is False


def test_interactive_terminal_tolerates_closed_streams(monkeypatch):
    def closed():
        raise ValueError("I/O operation on closed file")

    monkeypatch.setattr(sys.stdin, "isatty", closed, raising=False)
    assert fd.interactive_terminal() is False


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"\r", fd.KEY_ENTER),
        (b"\n", fd.KEY_ENTER),
        (b"C", "c"),
        (b"\x04", fd.KEY_QUIT),
        (b"\x1b", fd.KEY_QUIT),
        (b"\x1b[A", fd.KEY_UNKNOWN),
    ],
)
def test_read_key_on_a_real_pty(monkeypatch, data, expected):
    import threading

    master, slave = os.openpty()
    try:
        with os.fdopen(slave, "rb", buffering=0, closefd=False) as stream:
            monkeypatch.setattr(sys, "stdin", stream)
            # read_key's cbreak switch flushes typeahead (TCSAFLUSH), so the
            # key must arrive after it, like a real keypress.
            timer = threading.Timer(0.2, os.write, args=(master, data))
            timer.start()
            try:
                assert fd.read_key() == expected
            finally:
                timer.join()
    finally:
        os.close(master)
        os.close(slave)


def test_read_key_eof_quits(monkeypatch):
    master, slave = os.openpty()
    with os.fdopen(slave, "rb", buffering=0, closefd=False) as stream:
        monkeypatch.setattr(sys, "stdin", stream)
        monkeypatch.setattr(os, "read", lambda _fd, _n: b"")
        assert fd.read_key() == fd.KEY_QUIT
    os.close(master)
    os.close(slave)


def test_styling_respects_no_color(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert fd._styled_screen("a\nb").startswith("\x1b[1ma\x1b[0m")
    monkeypatch.setenv("NO_COLOR", "1")
    assert fd._styled_screen("a\nb") == "a\nb"


def test_narrow_terminal_truncates_instead_of_wrapping(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    text = fd._styled_screen(fd.render_screen(_state(agent="claude-code")), 40)
    lines = text.split("\n")
    assert max(len(line) for line in lines) == 40
    assert len(lines) == len(fd.render_screen(_state(agent="claude-code")).split("\n"))
    assert any(line.endswith("…") for line in lines)


@pytest.mark.parametrize("ram_gb", [8.0, 16.0])
def test_small_mac_screens_fit_the_row_budget(monkeypatch, ram_gb):
    monkeypatch.setattr("rapid_mlx.cli._recipe_free_disk_gb", lambda: 0.0)
    state = _state(
        picks=fd._picks_for(ram_gb, []),
        ram_gb=ram_gb,
        agent="claude-code",
        server=fd.ServerInfo(8000, "x"),
        cached=["lfm2.5-1b-4bit"],
        last_used="lfm2.5-1b-4bit",
    )
    assert len(fd.render_screen(state).split("\n")) <= 20


class _FakeProc:
    def __init__(self, *, alive_polls=10, code=0):
        self._alive = alive_polls
        self.code = code

    def poll(self):
        if self._alive > 0:
            self._alive -= 1
            return None
        return self.code

    def wait(self):
        return self.code


def test_run_server_process_child_keeps_default_sigint_and_parent_restores():
    import signal

    seen = {}

    def fake_popen(cmd):
        seen["cmd"] = cmd
        handler = signal.getsignal(signal.SIGINT)
        # A Python handler (reset to default in the exec'd child), never
        # SIG_IGN, which the child would inherit so Ctrl-C could not stop it.
        seen["handler_is_ignore"] = handler is signal.SIG_IGN
        seen["handler_callable"] = callable(handler)
        handler(signal.SIGINT, None)  # the parent's own copy is swallowed
        return _FakeProc(code=7)

    before = signal.getsignal(signal.SIGINT)
    assert fd.run_server_process(["--no-banner", "serve", "m"], popen=fake_popen) == 7
    assert seen["cmd"] == [*fd._cli_command(), "--no-banner", "serve", "m"]
    assert seen["handler_is_ignore"] is False
    assert seen["handler_callable"] is True
    assert signal.getsignal(signal.SIGINT) is before


def test_run_server_process_runs_on_ready_only_when_ready(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(fd, "wait_until_ready", lambda port, proc: port == 8001)
    code = fd.run_server_process(
        ["serve"],
        port=8001,
        on_ready=lambda: calls.append("ready"),
        popen=lambda cmd: _FakeProc(code=0),
    )
    assert code == 0
    assert calls == ["ready"]
    fd.run_server_process(
        ["serve"],
        port=9999,
        on_ready=lambda: calls.append("never"),
        popen=lambda cmd: _FakeProc(code=1),
    )
    assert calls == ["ready"]


def test_wait_until_ready():
    sleeps: list[float] = []
    probes = iter([False, False, True])
    assert fd.wait_until_ready(
        8000, _FakeProc(), probe=lambda url: next(probes), sleep=sleeps.append
    )
    assert sleeps == [1.0, 1.0]
    # The server exiting first means "not ready" (no launch).
    assert not fd.wait_until_ready(
        8000, _FakeProc(alive_polls=0), probe=lambda url: True, sleep=sleeps.append
    )
    # Timeout.
    assert not fd.wait_until_ready(
        8000, _FakeProc(), timeout_s=0, probe=lambda url: True, sleep=sleeps.append
    )


def test_ready_probe_against_a_real_loopback_server():
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200 if self.path == "/health/ready" else 503)
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        assert fd._ready_probe(f"http://127.0.0.1:{port}/health/ready") is True
        assert fd._ready_probe(f"http://127.0.0.1:{port}/other") is False
    finally:
        server.shutdown()
        server.server_close()
    assert fd._ready_probe(f"http://127.0.0.1:{port}/health/ready") is False


# ======================================================================
# Probes
# ======================================================================
def test_chip_label(monkeypatch):
    monkeypatch.setattr(fd.sys, "platform", "darwin")
    monkeypatch.setattr(
        fd.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="Apple M3 Pro\n"),
    )
    assert fd._chip_label() == "Apple M3 Pro"
    monkeypatch.setattr(
        fd.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="")
    )
    assert fd._chip_label() is None

    def fail(*_a, **_k):
        raise subprocess.TimeoutExpired("sysctl", 1)

    monkeypatch.setattr(fd.subprocess, "run", fail)
    assert fd._chip_label() is None
    monkeypatch.setattr(fd.sys, "platform", "linux")
    assert fd._chip_label() is None


def test_running_server_picks_lowest_port_and_api_name(monkeypatch):
    rows = [
        (1, "8010", "qwen3.5-9b-4bit", "1m"),
        (2, "8001", "served-name (qwen3.5-4b-4bit)", "1m"),
        (3, "bad", "x", "1m"),
        (4, "7000", "(unknown)", "1m"),
    ]
    monkeypatch.setattr("rapid_mlx.cli._scan_running_servers", lambda: rows)
    assert fd._running_server() == fd.ServerInfo(8001, "served-name", False)
    monkeypatch.setattr(
        "rapid_mlx.cli._scan_running_servers", lambda: [(1, "8000", "m", "1m")]
    )
    assert fd._running_server() == fd.ServerInfo(8000, "m", True)
    monkeypatch.setattr("rapid_mlx.cli._scan_running_servers", lambda: [])
    assert fd._running_server() is None

    def boom():
        raise RuntimeError("psutil missing")

    monkeypatch.setattr("rapid_mlx.cli._scan_running_servers", boom)
    assert fd._running_server() is None


def test_gather_state_is_offline_and_uses_one_cache_scan(monkeypatch, tmp_path):
    monkeypatch.setenv("RAPID_MLX_STATE_DIR", str(tmp_path))
    scans = []

    def scan():
        scans.append(1)
        return [("mlx-community/Qwen3.5-9B-4bit", 1, 2.0)]

    monkeypatch.setattr("rapid_mlx.cli._scan_hf_cache_models", scan)
    monkeypatch.setattr("rapid_mlx.cli._cache_entry_is_runnable", lambda _r: True)
    monkeypatch.setattr("rapid_mlx.cli._recipe_free_disk_gb", lambda: 500.0)
    monkeypatch.setattr("rapid_mlx.cli._scan_running_servers", lambda: [])
    monkeypatch.setattr("rapid_mlx.recommendations.physical_ram_gb", lambda: 18.0)
    monkeypatch.setattr(fr, "preferred_agent", lambda: None)
    monkeypatch.setattr(fd, "_chip_label", lambda: "Apple M3 Pro")
    monkeypatch.setattr(
        "socket.socket.connect", lambda *_a, **_k: pytest.fail("network")
    )
    state = fd.gather_state("9.9.9")
    assert len(scans) == 1
    assert state.cached == ["qwen3.5-9b-4bit"]
    assert state.selected == "qwen3.5-9b-4bit"
    assert [p["cached"] for p in state.picks] == [True, False]


def test_gather_state_survives_cache_scan_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("RAPID_MLX_STATE_DIR", str(tmp_path))

    def boom():
        raise OSError("broken cache")

    monkeypatch.setattr("rapid_mlx.cli._scan_hf_cache_models", boom)
    monkeypatch.setattr("rapid_mlx.cli._recipe_free_disk_gb", lambda: None)
    monkeypatch.setattr("rapid_mlx.cli._scan_running_servers", lambda: [])
    monkeypatch.setattr("rapid_mlx.recommendations.physical_ram_gb", lambda: 18.0)
    monkeypatch.setattr(fr, "preferred_agent", lambda: "claude-code")
    state = fd.gather_state("9.9.9")
    assert state.cached == []
    assert state.agent == "claude-code"
    assert state.selected == "qwen3.5-4b-4bit"


# ======================================================================
# run_bare / CLI wiring
# ======================================================================
def test_run_bare_non_tty_prints_to_stderr_and_exits_1(monkeypatch, capsys):
    monkeypatch.setattr(fd, "interactive_terminal", lambda: False)
    monkeypatch.setattr("rapid_mlx.recommendations.physical_ram_gb", lambda: 18.0)
    monkeypatch.setattr("rapid_mlx.cli._recipe_free_disk_gb", lambda: None)
    code = fd.run_bare(
        version="9.9.9",
        top_level_flags=[],
        dispatch=lambda _argv: pytest.fail("non-TTY must not run anything"),
    )
    assert code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "rapid-mlx serve qwen3.5-4b-4bit --port 8000" in captured.err


def test_run_bare_tty_dispatches_each_step_with_flags(monkeypatch, capsys):
    monkeypatch.setattr(fd, "interactive_terminal", lambda: True)
    monkeypatch.setattr(fd, "gather_state", lambda _v: _state(agent="claude-code"))
    monkeypatch.setattr(
        fd, "run_interactive", lambda state: fd.plan_connect(state, port=8001)
    )
    calls: list[list[str]] = []
    served: list[tuple] = []

    def fake_serve(argv, *, port, on_ready):
        served.append((list(argv), port))
        on_ready()
        return 3

    code = fd.run_bare(
        version="9.9.9",
        top_level_flags=["--no-telemetry"],
        dispatch=calls.append,
        serve=fake_serve,
    )
    assert code == 3  # the server child's exit status
    assert served == [
        (
            [
                "--no-banner",
                "--no-telemetry",
                "serve",
                "qwen3.5-4b-4bit",
                "--port",
                "8001",
            ],
            8001,
        )
    ]
    assert calls[0] == ["--no-banner", "--no-telemetry", "pull", "qwen3.5-4b-4bit"]
    assert calls[1][:4] == ["--no-banner", "--no-telemetry", "launch", "claude-code"]
    out = capsys.readouterr().out
    assert "→ rapid-mlx pull qwen3.5-4b-4bit" in out
    assert "→ rapid-mlx serve qwen3.5-4b-4bit --port 8001" in out
    assert "→ rapid-mlx launch claude-code --model qwen3.5-4b-4bit" in out
    assert out.index("serve qwen3.5-4b-4bit") < out.index("launch claude-code")


def test_failed_post_ready_step_keeps_the_server(capsys):
    def failing(_argv):
        raise SystemExit(2)

    fd._run_after_ready(["launch", "x"], ["--no-banner", "launch", "x"], failing)
    out = capsys.readouterr().out
    assert "→ rapid-mlx launch x" in out
    assert "failed (exit 2); the server keeps running" in out
    fd._run_after_ready(["launch", "x"], ["launch", "x"], lambda _a: sys.exit(0))
    assert "failed" not in capsys.readouterr().out


def test_run_bare_tty_quit_runs_nothing(monkeypatch):
    monkeypatch.setattr(fd, "interactive_terminal", lambda: True)
    monkeypatch.setattr(fd, "gather_state", lambda _v: _state())
    monkeypatch.setattr(fd, "run_interactive", lambda state: None)
    code = fd.run_bare(
        version="9.9.9",
        top_level_flags=[],
        dispatch=lambda _argv: pytest.fail("quit must not dispatch"),
    )
    assert code == 0


def test_run_bare_probe_failure_raises_unavailable(monkeypatch):
    monkeypatch.setattr(fd, "interactive_terminal", lambda: True)

    def boom(_version):
        raise RuntimeError("policy file missing")

    monkeypatch.setattr(fd, "gather_state", boom)
    with pytest.raises(fd.FrontDoorUnavailableError):
        fd.run_bare(version="9.9.9", top_level_flags=[], dispatch=print)


def _run_main_bare(monkeypatch, argv=("rapid-mlx",)):
    monkeypatch.setattr(sys, "argv", list(argv))
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return exc.value.code


def test_cli_bare_non_tty_exit_code_matches_main_contract(monkeypatch, capsys):
    # Golden: origin/main exits 1 for a bare non-interactive invocation;
    # scripts may depend on it.
    monkeypatch.setattr(fd, "interactive_terminal", lambda: False)
    monkeypatch.setattr("rapid_mlx.recommendations.physical_ram_gb", lambda: 18.0)
    assert _run_main_bare(monkeypatch) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "no command given" in captured.err


def test_cli_bare_falls_back_to_help_when_unavailable(monkeypatch, capsys):
    def unavailable(**_kwargs):
        raise fd.FrontDoorUnavailableError("x")

    monkeypatch.setattr(fd, "run_bare", unavailable)
    assert _run_main_bare(monkeypatch) == 1
    assert "usage: rapid-mlx" in capsys.readouterr().out


def test_cli_bare_forwards_no_telemetry_and_dispatches_in_process(monkeypatch):
    seen = {}

    def fake_run_bare(*, version, top_level_flags, dispatch):
        seen["flags"] = list(top_level_flags)
        seen["dispatch"] = dispatch
        return 0

    monkeypatch.setattr(fd, "run_bare", fake_run_bare)
    assert _run_main_bare(monkeypatch, ["rapid-mlx", "--no-telemetry"]) == 0
    assert seen["flags"] == ["--no-telemetry"]
    assert seen["dispatch"] is cli._dispatch_in_process


def test_dispatch_in_process_runs_main_with_argv_and_restores(monkeypatch):
    seen = {}
    monkeypatch.setattr(sys, "argv", ["/usr/bin/rapid-mlx"])
    monkeypatch.setattr(cli, "main", lambda: seen.setdefault("argv", list(sys.argv)))
    cli._dispatch_in_process(["--no-banner", "chat", "m"])
    assert seen["argv"] == ["/usr/bin/rapid-mlx", "--no-banner", "chat", "m"]
    assert sys.argv == ["/usr/bin/rapid-mlx"]
    monkeypatch.setattr(sys, "argv", [])
    cli._dispatch_in_process(["version"])
    assert sys.argv == []


# ======================================================================
# Last-used model (first_run helpers + chat hook)
# ======================================================================
def test_record_and_read_last_model(monkeypatch, tmp_path):
    monkeypatch.setenv("RAPID_MLX_STATE_DIR", str(tmp_path / "s"))
    assert fr.last_used_model() is None
    fr.record_last_model("qwen3.5-4b-4bit")
    assert fr.last_used_model() == "qwen3.5-4b-4bit"
    # Paths, raw repo ids and non-chat aliases are never stored.
    for name in ("/tmp/model", "org/repo", "embeddinggemma-300m-8bit", None, ""):
        fr.record_last_model(name)
        assert fr.last_used_model() == "qwen3.5-4b-4bit"
    (tmp_path / "s" / "last_chat_model").write_text("not-an-alias\n")
    assert fr.last_used_model() is None


def test_record_last_model_is_fail_silent(monkeypatch, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    monkeypatch.setenv("RAPID_MLX_STATE_DIR", str(blocker / "state"))
    fr.record_last_model("qwen3.5-4b-4bit")
    assert fr.last_used_model() is None
    monkeypatch.setattr(
        "rapid_mlx.model_aliases.list_profiles",
        lambda: (_ for _ in ()).throw(RuntimeError("x")),
    )
    assert fr._is_chat_alias("qwen3.5-4b-4bit") is False


def _run_chat_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", *argv])
    monkeypatch.setattr(cli, "chat_command", lambda _args: None)
    monkeypatch.setattr("rapid_mlx.telemetry.consent_runtime.startup", lambda **_: None)
    monkeypatch.setattr(cli, "_start_v2_lifecycle", lambda _command: None)
    monkeypatch.setattr("rapid_mlx._download_gate.is_repo_cached", lambda *_a: True)
    cli.main()


def test_local_chat_records_last_model(monkeypatch, tmp_path):
    monkeypatch.setenv("RAPID_MLX_STATE_DIR", str(tmp_path))
    _run_chat_main(monkeypatch, ["chat", "qwen3.5-9b-4bit"])
    assert fr.last_used_model() == "qwen3.5-9b-4bit"


def test_attached_chat_does_not_record_last_model(monkeypatch, tmp_path):
    monkeypatch.setenv("RAPID_MLX_STATE_DIR", str(tmp_path))
    _run_chat_main(monkeypatch, ["chat", "qwen3.5-9b-4bit", "--port", "8123"])
    assert fr.last_used_model() is None


def test_recipe_without_detectable_ram_asks_for_max_ram(monkeypatch):
    # recipe_command was split around the shared _annotate_recipe_picks helper;
    # its unknown-RAM refusal is unchanged.
    monkeypatch.setattr("rapid_mlx.recommendations.physical_ram_gb", lambda: 0.0)
    with pytest.raises(SystemExit, match="--max-ram"):
        cli.recipe_command(SimpleNamespace(max_ram=None, json=False))


def test_cli_command_prefers_the_sibling_entry_point(monkeypatch, tmp_path):
    script = tmp_path / "rapid-mlx"
    script.write_text("#!/bin/sh\n")
    script.chmod(0o755)
    monkeypatch.setattr(fd.sys, "executable", str(tmp_path / "python"))
    assert fd._cli_command() == [str(script)]
    script.chmod(0o644)
    assert fd._cli_command() == [str(tmp_path / "python"), "-m", "rapid_mlx.cli"]


def test_run_bare_chat_runs_in_process_and_exits_0(monkeypatch, capsys):
    monkeypatch.setattr(fd, "interactive_terminal", lambda: True)
    monkeypatch.setattr(fd, "gather_state", lambda _v: _state())
    monkeypatch.setattr(fd, "run_interactive", fd.plan_chat)
    calls: list[list[str]] = []
    code = fd.run_bare(
        version="9.9.9",
        top_level_flags=[],
        dispatch=calls.append,
        serve=lambda _argv: pytest.fail("chat must not spawn a server child"),
    )
    assert code == 0
    assert calls == [["--no-banner", "chat", "qwen3.5-4b-4bit"]]
    assert "→ rapid-mlx chat qwen3.5-4b-4bit" in capsys.readouterr().out
