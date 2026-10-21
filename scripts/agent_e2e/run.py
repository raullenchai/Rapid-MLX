#!/usr/bin/env python3
"""Black-box, fail-closed terminal-agent acceptance on a leased Mac.

The caller supplies a dedicated HOME and port. Raw transcripts stay in the
private output directory; only a redacted matrix is suitable for sharing.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import pty
import re
import select
import shutil
import signal
import socket
import sqlite3
import struct
import subprocess
import termios
import time
import uuid
from pathlib import Path

AGENTS = (
    "codex",
    "claude-code",
    "pi",
    "opencode",
    "hermes",
    "goose",
    "qwen-code",
    "crush",
    "aider",
)
BINARIES = {
    "codex": "codex",
    "claude-code": "claude",
    "pi": "pi",
    "opencode": "opencode",
    "hermes": "hermes",
    "goose": "goose",
    "qwen-code": "qwen",
    "crush": "crush",
    "aider": "aider",
}
USER_CONFIG = {
    "codex": (".codex/config.toml",),
    "claude-code": (".claude/settings.json", ".claude.json"),
    "pi": (".pi/agent/models.json", ".pi/agent/auth.json"),
    "opencode": (".config/opencode/opencode.json",),
    "hermes": (".hermes/config.yaml",),
    "goose": (".config/goose/config.yaml",),
    "qwen-code": (".qwen/settings.json", ".qwen/oauth_creds.json"),
    "crush": (".config/crush/crush.json",),
    "aider": (".aider.conf.yml", ".aider.model.settings.yml"),
}


def snapshot(home: Path, agent: str) -> dict[str, str]:
    """Include file bytes, symlink targets and modes; exclude Rapid's subtree."""
    result = {}
    for relative in USER_CONFIG[agent]:
        path = home / relative
        if not path.exists() and not path.is_symlink():
            continue
        key = str(path.relative_to(home))
        payload = os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
        result[key] = (
            hashlib.sha256(payload).hexdigest() + f":{path.lstat().st_mode & 0o777:o}"
        )
    return result


def preserved(before: dict[str, str], after: dict[str, str]) -> bool:
    """New session files are allowed; every pre-existing config byte must stay."""
    return bool(before) and all(
        after.get(path) == digest for path, digest in before.items()
    )


def rapid_config_untouched(home: Path, agent: str, before: dict[str, str]) -> bool:
    """Check immediately after Rapid launch/removal, before opening native mode."""
    return preserved(before, snapshot(home, agent))


def classified_error(value: str) -> str:
    """Use only fixed labels in the shareable matrix; raw errors stay private."""
    if not value:
        return "—"
    screen = _ANSI.sub("", value).lower()
    if "unrecognized flag: --model" in screen:
        return "unsupported --model flag"
    if "select login method" in screen or "paste code here" in screen:
        return "login required"
    if "choice [default" in screen:
        return "first-run menu"
    return "see private transcript"


def output_issues(*transcripts: Path) -> list[str]:
    patterns = ("traceback", "fatal error", "warning", "deprecated", "login required")
    found = []
    for path in transcripts:
        if not path.exists():
            continue
        screen = _ANSI.sub("", path.read_bytes().decode(errors="replace")).lower()
        for pattern in patterns:
            if pattern in screen and pattern not in found:
                found.append(pattern)
    return found


def version(binary: str, env: dict[str, str]) -> str:
    if not shutil.which(binary, path=env["PATH"]):
        return "NOT INSTALLED"
    try:
        p = subprocess.run(
            [binary, "--version"],
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return (p.stdout or p.stderr).strip().splitlines()[0][:160]
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return "UNKNOWN"


def read_until(
    fd: int,
    process: subprocess.Popen,
    seconds: float,
    sink,
    stop_when=None,
    on_output=None,
) -> tuple[str, float | None]:
    end = time.monotonic() + seconds
    pieces = []
    first = None
    last = None
    while time.monotonic() < end and process.poll() is None:
        if stop_when is not None and last is not None and time.monotonic() - last > 5:
            if stop_when("".join(pieces)):
                break
        ready, _, _ = select.select(
            [fd], [], [], max(0, min(0.25, end - time.monotonic()))
        )
        if not ready:
            if on_output is not None and pieces:
                on_output("".join(pieces))
            continue
        try:
            data = os.read(fd, 65536)
        except OSError:
            break
        if not data:
            break
        if first is None:
            first = time.monotonic()
        last = time.monotonic()
        sink.write(data)
        sink.flush()
        pieces.append(data.decode("utf-8", errors="replace"))
        if on_output is not None:
            on_output("".join(pieces))
    return "".join(pieces), first


def open_terminal() -> tuple[int, int]:
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
    return master, slave


def stop_terminal_process(proc: subprocess.Popen) -> None:
    """Stop the session created for this terminal, including child CLIs."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        try:
            proc.terminate()
        except (ProcessLookupError, PermissionError):
            pass
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        proc.wait(timeout=3)
    # A shell launcher can exit while its agent remains in the same session.
    # Kill that owned group before starting the next row or releasing a lease.
    try:
        os.killpg(proc.pid, 0)
    except (ProcessLookupError, PermissionError):
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


_ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")


def _compact_screen(output: str) -> str:
    return "".join(_ANSI.sub("", output).lower().split())


def handle_first_run_menus(
    agent: str, output: str, master: int, proc: subprocess.Popen, sink
) -> str:
    """Answer only known local onboarding prompts, never a cloud login."""
    if agent == "crush":
        if "Would you like to initialize now?" in _ANSI.sub("", output):
            os.write(master, b"\x1b[C\r")  # Select Nope; keep the project untouched.
            more, _ = read_until(master, proc, 8, sink)
            output += more
        if "Ready for instructions" not in _ANSI.sub("", output):
            more, _ = read_until(
                master,
                proc,
                20,
                sink,
                stop_when=lambda screen: (
                    "Ready for instructions" in _ANSI.sub("", screen)
                ),
            )
            output += more
        return output
    if agent == "goose":
        if "Share anonymous usage data to help improve goose?" in _ANSI.sub("", output):
            os.write(master, b"\r")  # Keep the default No choice.
            more, _ = read_until(master, proc, 8, sink)
            return output + more
        return output
    if agent == "aider":
        # Rapid can print its own ready line before Aider reaches the Git
        # onboarding question. Wait for that question before sending a task.
        analytics = "allowcollectionofanonymousanalyticstohelpimproveaider"
        if "nogitrepofound,createone" not in _compact_screen(output):
            more, _ = read_until(
                master,
                proc,
                20,
                sink,
                stop_when=lambda screen: (
                    "nogitrepofound,createone" in _compact_screen(screen)
                ),
            )
            output += more
        answered = set()
        for _ in range(5):
            screen = _ANSI.sub("", output)
            if (
                "nogitrepofound,createone" in _compact_screen(screen)
                and "git" not in answered
            ):
                answered.add("git")
            elif (
                "Open documentation url for more info?" in screen
                and "docs" not in answered
            ):
                answered.add("docs")
            elif (
                "Would you like to see what's new in this version?" in screen
                and "news" not in answered
            ):
                answered.add("news")
            elif analytics in _compact_screen(screen) and "analytics" not in answered:
                answered.add("analytics")
            else:
                if "git" in answered and "analytics" not in answered:
                    more, _ = read_until(
                        master,
                        proc,
                        20,
                        sink,
                        stop_when=lambda later: analytics in _compact_screen(later),
                    )
                    output += more
                    if analytics in _compact_screen(more):
                        continue
                break
            os.write(master, b"n\r")
            more, _ = read_until(master, proc, 8, sink)
            output += more
        return output
    if agent == "hermes":
        if "Choice [default 3]:" in _ANSI.sub("", output):
            os.write(master, b"\r")  # Decline optional telemetry.
            more, _ = read_until(master, proc, 8, sink)
            return output + more
        return output
    if agent != "claude-code":
        return output
    answered = set()
    for _ in range(5):
        screen = _compact_screen(output)
        if "choosethetextstyle" in screen and "theme" not in answered:
            answered.add("theme")
            os.write(master, b"\r")
        elif "detectedacustomapikey" in screen and "local-key" not in answered:
            answered.add("local-key")
            os.write(master, b"\x1b[A")  # default is No; select Yes
            time.sleep(0.3)
            os.write(master, b"\r")
        elif (
            "trust" in screen
            and "folder" in screen
            and "yes," in screen
            and "no,exit" in screen
            and "trust" not in answered
        ):
            answered.add("trust")
            if screen.rfind("no,exit") < screen.rfind("yes,"):
                os.write(master, b"\x1b[B")  # New menu highlights No first.
                time.sleep(0.3)
            os.write(master, b"\r")
        elif (
            "promptinjection" in screen
            and "tocontinue" in screen
            and "security" not in answered
        ):
            answered.add("security")
            os.write(master, b"\r")
        else:
            break
        more, _ = read_until(master, proc, 8, sink)
        output += more
    return output


def reply_finished(agent: str, output: str) -> bool:
    """Some TUIs expose tool output before the agent completes the turn."""
    if agent == "goose":
        return "⏱" in output
    return True


def opencode_reply_seen(home: Path, since: float, hello: str, nonce: str) -> bool:
    """Read one v2 conversation instead of fragmented TUI redraws."""
    db = home / ".rapid-mlx/connect/agents/opencode/xdg-data/opencode/opencode.db"
    if not db.is_file():
        return False
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=0.2) as conn:
            rows = conn.execute(
                "SELECT session_id, data FROM session_message WHERE type = 'assistant' "
                "AND time_created >= ? ORDER BY time_created, seq",
                (int(since * 1000),),
            ).fetchall()
    except sqlite3.Error:
        return False
    sessions: dict[str, tuple[set[str], list[str]]] = {}
    for session_id, raw in rows:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            return False
        reads, replies = sessions.setdefault(session_id, (set(), []))
        for part in message.get("content", []):
            if part.get("type") == "tool" and part.get("name") == "read":
                state = part.get("state", {})
                if state.get("status") == "completed":
                    path = Path(state.get("input", {}).get("path", "")).name
                    contents = " ".join(
                        item.get("text", "") for item in state.get("content", [])
                    )
                    if path == "hello.txt" and hello in contents:
                        reads.add("hello.txt")
                    if path == "nonce.txt" and nonce in contents:
                        reads.add("nonce.txt")
            elif part.get("type") == "text" and message.get("time", {}).get(
                "completed"
            ):
                replies.append(part.get("text", ""))
    return any(
        reads == {"hello.txt", "nonce.txt"}
        and any(hello in reply and nonce in reply for reply in replies)
        for reads, replies in sessions.values()
    )


def is_opencode_v2(version_text: str) -> bool:
    """Route by CLI major version, not a matching minor component."""
    return bool(re.match(r"^(?:opencode\s+)?v?2\.", version_text, re.IGNORECASE))


def approval_handler(agent: str, master: int):
    """Accept an explicit local tool approval, as a user at the terminal would."""
    if agent == "aider":
        answered = False
        answered_news = False

        def answer_aider(output: str) -> None:
            nonlocal answered, answered_news
            tail = _ANSI.sub("", output[-1200:])
            if not answered_news and "Would you like to see what's new" in tail:
                os.write(master, b"n\r")
                answered_news = True
            elif not answered and "Create new file? (Y)es/(N)o" in tail:
                os.write(master, b"y\r")
                answered = True

        return answer_aider
    if agent == "crush":
        waiting = False
        last_answered = 0.0
        retries = 0

        def answer_crush(output: str) -> None:
            nonlocal waiting, last_answered, retries
            raw_tail = output[-3000:]
            tail = _ANSI.sub("", raw_tail)
            menu = "Allow for Session" in tail and "Deny" in tail
            if menu and (
                not waiting or (retries < 3 and time.monotonic() - last_answered > 2)
            ):
                os.write(master, b"\r")  # Highlighted Allow, for this action only.
                waiting = True
                last_answered = time.monotonic()
                retries += 1
            elif (
                waiting
                and not menu
                and "state=working:app=crush" in raw_tail
                and time.monotonic() - last_answered > 1
            ):
                waiting = False
                retries = 0

        return answer_crush
    if agent != "qwen-code":
        return None
    count = 0

    def answer(output: str) -> None:
        nonlocal count
        seen = output.count("Waiting for user confirmation")
        if seen > count:
            count = seen
            os.write(master, b"\r")  # The highlighted choice is "allow once".

    return answer


def send_prompt(agent: str, master: int, prompt: str) -> None:
    data = prompt.encode()
    if agent == "crush":
        # The TUI enables bracketed paste. Raw bulk input loses its first
        # characters while the screen is redrawn after onboarding.
        data = b"\x1b[200~" + data + b"\x1b[201~"
    os.write(master, data)
    time.sleep(0.5)
    os.write(master, b"\r")


def session(
    command: list[str],
    env: dict[str, str],
    workspace: Path,
    transcript: Path,
    timeout: float,
    day_two: bool,
) -> dict:
    opencode_v2 = command[2] == "opencode" and is_opencode_v2(version("opencode", env))
    marker = uuid.uuid4().hex[:12]
    target = workspace / "hello.txt"
    nonce1 = f"nonce-{uuid.uuid4().hex[:12]}"
    nonce2 = f"nonce-{uuid.uuid4().hex[:12]}"
    nonce_file = workspace / "nonce.txt"
    nonce_file.write_text(nonce1 + "\n")
    if target.exists():
        target.unlink()
    first_prompt = (
        f"Create hello.txt containing exactly hi-{marker}. Use a tool to read "
        "both hello.txt and nonce.txt. Quote each file's exact line in your "
        "reply. nonce.txt is independent of hello.txt; do not edit it."
    )
    second_prompt = (
        f"Use a tool to replace hello.txt with exactly hi-{marker}-second "
        "(one trailing newline is okay). Read both hello.txt and nonce.txt "
        "again with a tool. Quote each file's exact line in your reply. "
        "nonce.txt is independent of hello.txt; do not edit it."
    )
    master, slave = open_terminal()
    started = time.monotonic()
    proc = subprocess.Popen(
        command,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        cwd=workspace,
        env=env,
        start_new_session=True,
    )
    os.close(slave)
    answer = {
        "open_s": None,
        "first_reply_s": None,
        "second_reply_s": None,
        "tool_ok": False,
        "second_ok": False,
        "exit": None,
        "error": "",
        "day_two": day_two,
        "prompt1_at": None,
        "prompt2_at": None,
        "ended_at": None,
    }

    def reply_seen(output: str, hello: str, nonce: str, since: float) -> bool:
        if opencode_v2:
            return opencode_reply_seen(Path(env["HOME"]), since, hello, nonce)
        return nonce in output and reply_finished(command[2], output)

    with transcript.open("wb") as sink:
        try:
            opening, _ = read_until(
                master,
                proc,
                timeout,
                sink,
                stop_when=lambda output: "Agent ready." in output,
            )
            opening = handle_first_run_menus(command[2], opening, master, proc, sink)
            # Startup text is not proof that the agent opened. The launch
            # readiness line is a lower bound; a failed launch stays blank.
            if "Agent ready." in opening:
                answer["open_s"] = round(time.monotonic() - started, 2)
            if proc.poll() is not None or answer["open_s"] is None:
                answer["error"] = opening[-500:]
                return answer
            sent = time.monotonic()
            answer["prompt1_at"] = time.time()
            approve = approval_handler(command[2], master)
            send_prompt(command[2], master, first_prompt)
            output, _ = read_until(
                master,
                proc,
                timeout,
                sink,
                stop_when=lambda output: (
                    target.is_file()
                    and target.read_text().strip() == f"hi-{marker}"
                    and nonce_file.is_file()
                    and nonce_file.read_text().strip() == nonce1
                    and reply_seen(output, f"hi-{marker}", nonce1, answer["prompt1_at"])
                ),
                on_output=approve,
            )
            answer["tool_ok"] = (
                target.is_file()
                and target.read_text().strip() == f"hi-{marker}"
                and nonce_file.is_file()
                and nonce_file.read_text().strip() == nonce1
                and reply_seen(output, f"hi-{marker}", nonce1, answer["prompt1_at"])
            )
            if answer["tool_ok"]:
                # The terminal first echoes our input. Completion is the first
                # reliable upper bound for a reply that used a tool.
                answer["first_reply_s"] = round(time.monotonic() - sent, 2)
            if proc.poll() is None:
                nonce_file.write_text(nonce2 + "\n")
                sent = time.monotonic()
                answer["prompt2_at"] = time.time()
                approve = approval_handler(command[2], master)
                send_prompt(command[2], master, second_prompt)
                output2, _ = read_until(
                    master,
                    proc,
                    timeout,
                    sink,
                    stop_when=lambda output: (
                        target.is_file()
                        and target.read_text().strip() == f"hi-{marker}-second"
                        and nonce_file.is_file()
                        and nonce_file.read_text().strip() == nonce2
                        and reply_seen(
                            output,
                            f"hi-{marker}-second",
                            nonce2,
                            answer["prompt2_at"],
                        )
                    ),
                    on_output=approve,
                )
                answer["second_ok"] = (
                    target.is_file()
                    and target.read_text().strip() == f"hi-{marker}-second"
                    and nonce_file.is_file()
                    and nonce_file.read_text().strip() == nonce2
                    and reply_seen(
                        output2,
                        f"hi-{marker}-second",
                        nonce2,
                        answer["prompt2_at"],
                    )
                )
                if answer["second_ok"]:
                    answer["second_reply_s"] = round(time.monotonic() - sent, 2)
            if not answer["tool_ok"] or not answer["second_ok"]:
                answer["error"] = (
                    output + output2 if "output2" in locals() else output
                )[-500:]
            if proc.poll() is None:
                os.write(master, b"\x03")
                time.sleep(0.3)
                os.write(master, b"\x1b")
                os.write(master, b"\x04")
                read_until(master, proc, 2, sink)
        finally:
            stop_terminal_process(proc)
            answer["exit"] = proc.returncode
            answer["ended_at"] = time.time()
            os.close(master)
    return answer


def request_evidence(path: Path | None, agent: str, start: float, end: float) -> dict:
    """Only accept server-side JSONL; never infer the requested model from config."""
    if path is None or not path.exists():
        return {
            "model_requested": None,
            "auth_accepted": None,
            "first_prompt_tokens": None,
        }
    requests = []
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
            if event.get("agent") == agent and start <= event["time"] <= end:
                requests.append(event)
        except (ValueError, KeyError, TypeError):
            continue
    requests.sort(key=lambda event: event["time"])
    first = next(
        (
            r
            for r in requests
            if r.get("path", "").endswith(
                ("/chat/completions", "/messages", "/responses")
            )
        ),
        None,
    )
    return {
        "model_requested": first.get("model") if first else None,
        "auth_accepted": first.get("status") in (200, 201) if first else None,
        "first_prompt_tokens": first.get("prompt_tokens") if first else None,
    }


def normal_mode(
    binary: str, env: dict[str, str], workspace: Path, transcript: Path
) -> bool:
    """Start the unwrapped CLI from the same HOME after Rapid profile removal."""
    master, slave = open_terminal()
    proc = subprocess.Popen(
        [binary],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        cwd=workspace,
        env=env,
        start_new_session=True,
    )
    os.close(slave)
    with transcript.open("wb") as sink:
        try:
            output, _ = read_until(master, proc, 8, sink)
            fatal = ("traceback", "fatal error", "invalid config", "command not found")
            return (
                bool(output.strip())
                and proc.poll() is None
                and not any(word in output.lower() for word in fatal)
            )
        finally:
            stop_terminal_process(proc)
            os.close(master)


def check_then_open_native(
    home: Path,
    agent: str,
    before: dict[str, str],
    env: dict[str, str],
    workspace: Path,
    transcript: Path,
) -> tuple[bool, bool]:
    """Attribute user-config changes to Rapid before native CLI bookkeeping."""
    untouched = rapid_config_untouched(home, agent, before)
    try:
        opened = normal_mode(BINARIES[agent], env, workspace, transcript)
    except OSError:
        opened = False
    return untouched, opened


def live_server_pid(home: Path) -> int | None:
    path = home / ".rapid-mlx/launch.pid"
    try:
        pid = int(path.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--home", type=Path, required=True, help="dedicated, pre-seeded macOS user HOME"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="private evidence directory"
    )
    parser.add_argument(
        "--rapid",
        type=Path,
        required=True,
        help="installed current-main rapid-mlx binary",
    )
    parser.add_argument("--model", default="qwen3.5-9b-4bit")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--agents", nargs="+", choices=AGENTS, default=list(AGENTS))
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    args.home = args.home.resolve()
    args.rapid = args.rapid.resolve()
    args.output = args.output.resolve()
    if not args.home.is_dir() or not args.rapid.is_file() or args.port == 7659:
        parser.error(
            "use an existing dedicated HOME, installed binary, and a non-owner port"
        )
    with socket.socket() as probe:
        probe.settimeout(0.5)
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            parser.error(
                "reserved port already has a listener; cold start requires a stopped server"
            )
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output.chmod(0o700)
    env = {
        key: os.environ[key]
        for key in ("PATH", "LANG", "LC_ALL", "TERM", "SHELL", "TMPDIR")
        if key in os.environ
    }
    env["HOME"] = str(args.home)
    env["XDG_CONFIG_HOME"] = str(args.home / ".config")
    env["XDG_DATA_HOME"] = str(args.home / ".local/share")
    env["XDG_CACHE_HOME"] = str(args.home / ".cache")
    env["PATH"] = os.pathsep.join(
        (
            str(args.home / ".local/bin"),
            str(args.home / ".rapid-mlx/bin"),
            str(args.rapid.parent),
            env.get("PATH", ""),
        )
    )
    env["RAPID_MLX_AUTO_PULL"] = "1"
    log_path = args.output / "requests.jsonl"
    current_agent = args.output / "current-agent"
    env["AGENT_E2E_REQUEST_LOG"] = str(log_path)
    env["AGENT_E2E_CURRENT_AGENT"] = str(current_agent)
    env["PYTHONPATH"] = (
        str(Path(__file__).parent / "instrument")
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    rows = []
    original_server_pid = None
    for agent in args.agents:
        current_agent.write_text(agent + "\n")
        work = args.output / agent / "project"
        work.mkdir(parents=True, exist_ok=True)
        before = snapshot(args.home, agent)
        row = {
            "agent": agent,
            "version": version(BINARIES[agent], env),
            "config_seeded": bool(before),
            "config_untouched": None,
        }
        row["cold_server"] = agent == args.agents[0]
        cmd = [
            str(args.rapid),
            "launch",
            agent,
            "--model",
            args.model,
            "--port",
            str(args.port),
        ]
        row["day_one"] = session(
            cmd, env, work, args.output / agent / "day-one.pty", args.timeout, False
        )
        server_pid = live_server_pid(args.home)
        if original_server_pid is None:
            original_server_pid = server_pid
        row["server_reused"] = bool(
            server_pid is not None and server_pid == original_server_pid
        )
        # This subprocess has a new shell environment and no live TTY state.
        row["day_two"] = session(
            [str(args.rapid), "launch", agent],
            env,
            work,
            args.output / agent / "day-two.pty",
            args.timeout,
            True,
        )
        row["server_reused"] &= live_server_pid(args.home) == original_server_pid
        one = row["day_one"]
        two = row["day_two"]
        windows = (
            (one["prompt1_at"], one["prompt2_at"]),
            (one["prompt2_at"], one["ended_at"]),
            (two["prompt1_at"], two["prompt2_at"]),
            (two["prompt2_at"], two["ended_at"]),
        )
        row["requests"] = [
            request_evidence(log_path, agent, start, end)
            if start is not None and end is not None
            else request_evidence(None, agent, 0, 0)
            for start, end in windows
        ]
        row.update(row["requests"][0])
        removed = subprocess.run(
            [str(args.rapid), "launch", agent, "--remove"],
            env=env,
            cwd=work,
            capture_output=True,
            text=True,
            timeout=30,
        )
        row["remove_ok"] = (
            removed.returncode == 0
            and not (args.home / ".rapid-mlx/connect/agents" / agent).exists()
        )
        # Rapid's private launch and removal must not change a user config.
        # The native CLI may add its own first-run bookkeeping when opened
        # below, so compare before starting that separate native session.
        row["config_untouched"], row["normal_mode_opens"] = check_then_open_native(
            args.home,
            agent,
            before,
            env,
            work,
            args.output / agent / "normal-mode.pty",
        )
        row["ugly_output"] = output_issues(
            args.output / agent / "day-one.pty",
            args.output / agent / "day-two.pty",
            args.output / agent / "normal-mode.pty",
        )
        row["works"] = bool(
            row["config_seeded"]
            and row["config_untouched"]
            and row["day_one"]["tool_ok"]
            and row["day_one"]["second_ok"]
            and row["day_two"]["tool_ok"]
            and row["day_two"]["second_ok"]
            and row["remove_ok"]
            and row["normal_mode_opens"]
            and row["server_reused"]
            and all(
                r["model_requested"] == args.model and r["auth_accepted"]
                for r in row["requests"]
            )
        )
        rows.append(row)
        (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    header = "| Agent | Version | Works | Open s | Reply 1/2 s | Requested model | Auth | Prompt tokens | Tools | Config | Normal | Remove | Error | Ugly output |\n"
    divider = "|---|---|---|---:|---:|---|---|---:|---|---|---|---|---|---|\n"
    lines = ["# Terminal agent real-user acceptance\n\n", header, divider]
    for row in rows:
        one = row["day_one"]
        fmt = lambda value: "—" if value is None else str(value)
        lines.append(
            f"| {row['agent']} | {row['version']} | {'Y' if row['works'] else 'N'} | "
            f"{fmt(one['open_s'])} | {fmt(one['first_reply_s'])}/{fmt(one['second_reply_s'])} | "
            f"{fmt(row['model_requested'])} | {fmt(row['auth_accepted'])} | "
            f"{fmt(row['first_prompt_tokens'])} | {one['tool_ok']}/{one['second_ok']} | "
            f"{'Y' if row['config_untouched'] else 'N'} | "
            f"{'Y' if row['normal_mode_opens'] else 'N'} | "
            f"{'Y' if row['remove_ok'] else 'N'} | "
            f"{classified_error(one['error'] or row['day_two']['error'])} | "
            f"{', '.join(row['ugly_output']) or '—'} |\n"
        )
    (args.output / "matrix.md").write_text("".join(lines))
    print(args.output / "matrix.md")
    return 0 if all(row["works"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
