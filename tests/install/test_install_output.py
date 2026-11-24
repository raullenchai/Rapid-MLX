"""Regression checks for installer output."""

import fcntl
import json
import os
import pty
import select
import shlex
import signal
import subprocess
import sys
import termios
import time
import tomllib
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "install.sh"


def run_bash(code, *args, env=None, **kwargs):
    clean_env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"UV_CONFIG_FILE", "UV_NO_CONFIG"}
    }
    return subprocess.run(
        [
            "bash",
            "-c",
            'RAPID_INSTALL_LIB=1 source "$1"; shift; ' + code,
            "test",
            str(SCRIPT),
            *map(str, args),
        ],
        env={**clean_env, **(env or {})},
        **kwargs,
    )


@pytest.mark.parametrize("telemetry_env", [None, "1"])
@pytest.mark.parametrize("terminal", [False, True])
def test_internal_commands_preserve_first_use_disclosure(
    tmp_path, monkeypatch, telemetry_env, terminal
):
    from rapid_mlx.telemetry import state

    home = tmp_path / "home"
    binaries = home / ".rapid-mlx" / "bin"
    binaries.mkdir(parents=True)
    python = binaries / "python"
    # A relocated symlink loses pyvenv.cfg discovery and the test runner's deps.
    # Invoke the original interpreter path without resolving its venv symlink.
    python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    python.chmod(0o755)
    events = tmp_path / "events.jsonl"
    cli = binaries / "rapid-mlx"
    # Exercise the real CLI/consent/tracking path as an official release, with
    # a recording sender: no network or model discovery is needed.
    cli.write_text(
        f"#!{sys.executable}\n"
        f"import sys; sys.path.insert(0, {str(SCRIPT.parent)!r})\n"
        """
import json, os
from pathlib import Path
import rapid_mlx
from rapid_mlx import cli
from rapid_mlx.telemetry import build_gate, posthog_sender
rapid_mlx.__version__ = "0.15.1"
stamp = build_gate.ReleaseStamp(channel="stable", posthog_key="phc_" + "a" * 32)
build_gate.official_build = lambda: stamp
class Sender:
    def capture(self, item):
        with Path(os.environ["TEST_EVENTS"]).open("a") as stream:
            stream.write(json.dumps(dict(item)) + "\\n")
        return True
posthog_sender.get_sender = lambda: Sender()
posthog_sender.install_atexit = lambda: None
cli.models_command = lambda args: print('{"cached":[]}')
cli.main()
"""
    )
    cli.chmod(0o755)
    for name in (
        state.ENV_VAR,
        state.DO_NOT_TRACK_ENV,
        *state.CI_ENV_VARS,
        "RAPID_MLX_PROCESS_ROLE",
        "RAPID_MLX_WATCHDOG_PPID",
    ):
        monkeypatch.delenv(name, raising=False)
    env = {**os.environ, "HOME": str(home), "TEST_EVENTS": str(events)}
    if telemetry_env is not None:
        env[state.ENV_VAR] = telemetry_env
    master, slave = pty.openpty() if terminal else (None, None)
    try:
        result = run_bash(
            """
VENV_BACKUP=fixture
commit_rebuild() { :; }
finish_rebuild
installed_cached_aliases
run_install_command "$INSTALL_DIR/bin/rapid-mlx" --version
run_install_command "$INSTALL_DIR/bin/python" "$INSTALL_DIR/bin/rapid-mlx" models --cached --json
printf 'parent-telemetry=%s\\n' "${RAPID_MLX_TELEMETRY-unset}"
""",
            env=env,
            stdin=slave if terminal else subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
        )
    finally:
        if terminal:
            os.close(master)
            os.close(slave)
    assert result.returncode == 0, result.stderr
    assert f"parent-telemetry={telemetry_env or 'unset'}" in result.stdout
    assert "anonymous usage reporting" not in result.stderr
    assert not events.exists()
    assert not (home / ".rapid-mlx" / "telemetry-client-id").exists()
    assert not (home / ".rapid-mlx" / "telemetry-consent.yaml").exists()
    assert not (home / ".rapid-mlx" / "telemetry-consent.yaml.lock").exists()
    assert sorted(path.name for path in (home / ".rapid-mlx").iterdir()) == ["bin"]

    normal = subprocess.run(
        [str(cli), "models", "--cached", "--json"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert normal.returncode == 0, normal.stderr
    assert "anonymous usage reporting is ON" in normal.stderr
    assert (home / ".rapid-mlx" / "telemetry-client-id").exists()
    assert (
        "notice_revision_seen: 1"
        in (home / ".rapid-mlx" / "telemetry-consent.yaml").read_text()
    )
    captured = [json.loads(line) for line in events.read_text().splitlines()]
    assert any(
        item["event"] == "app_opened" and item["properties"]["command"] == "models"
        for item in captured
    )


@pytest.mark.parametrize("audio_present", [True, False])
def test_audio_preparation_only_runs_for_installed_audio(audio_present):
    result = run_bash(
        """
INSTALLER=(fake-installer)
run_install_command() {
    case "$*" in
        *find_spec*) return "$1" ;;
        *prepare_runtime_requirement*) printf 'pipeline prepared\\n' ;;
    esac
}
""".replace('return "$1"', "return 0" if audio_present else "return 1")
        + """
run_package_command() { printf '%s\\n' "$@"; }
prepare_installed_audio
""",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    if audio_present:
        assert "misaki>=0.5.0" in result.stdout
        assert "spacy>=3.7,<3.9" in result.stdout
        assert "pipeline prepared" in result.stdout
    else:
        assert not result.stdout


@pytest.mark.parametrize("failure", ["dependencies", "pipeline"])
def test_audio_preparation_failure_prints_one_fix(failure):
    result = run_bash(
        """
INSTALLER=(fake-installer)
run_install_command() {
    case "$*" in
        *find_spec*) return 0 ;;
        *) return 1 ;;
    esac
}
run_package_command() { return "$1"; }
""".replace('return "$1"', "return 1" if failure == "dependencies" else "return 0")
        + "prepare_installed_audio",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    warning = result.stdout + result.stderr
    assert warning.count("unavailable.") == 1
    assert (
        "misaki>=0.5.0" in warning
        if failure == "dependencies"
        else "pull kokoro" in warning
    )
    assert "Traceback" not in warning


@pytest.mark.parametrize("term", ["dumb", "xterm"])
def test_no_colour_on_pipe(term):
    result = run_bash(
        "ok ok; warn warning; err error; info info; dim dim",
        env={"TERM": term},
        capture_output=True,
    )
    assert b"\x1b" not in result.stdout + result.stderr


def test_stderr_colour_independent_of_stdout():
    master, slave = pty.openpty()
    try:
        result = run_bash(
            "err error; dim_err detail",
            env={"TERM": "xterm", "NO_COLOR": ""},
            stdout=slave,
            stderr=subprocess.PIPE,
        )
        assert result.returncode == 0
        assert b"\x1b" not in result.stderr
    finally:
        os.close(master)
        os.close(slave)


def test_stderr_terminal_colour_with_stdout_captured():
    master, slave = pty.openpty()
    try:
        result = run_bash(
            "err error; dim_err detail",
            env={"TERM": "xterm", "NO_COLOR": ""},
            stdout=subprocess.PIPE,
            stderr=slave,
        )
        assert result.returncode == 0
        assert os.read(master, 4096).count(b"\x1b") == 4
    finally:
        os.close(master)
        os.close(slave)


@pytest.mark.parametrize(
    "term,no_color,expected",
    [("xterm", "", True), ("dumb", "", False), ("xterm", "1", False), ("", "", False)],
)
def test_terminal_colour(term, no_color, expected):
    master, slave = pty.openpty()
    try:
        result = run_bash(
            "ok hello",
            env={"TERM": term, "NO_COLOR": no_color},
            stdout=slave,
            stderr=subprocess.PIPE,
        )
        output = os.read(master, 4096)
        assert result.returncode == 0
        assert (b"\x1b" in output) == expected
    finally:
        os.close(master)
        if slave != -1:
            os.close(slave)


@pytest.mark.parametrize("status", [0, 3])
def test_package_output_filter(status):
    result = run_bash(
        """
stub() {
    echo 'warning: The package `transformers==5.15.1` does not have an extra named `tokenizers`' >&2
    echo "WARNING: transformers 5.16.0 does not provide the extra 'tokenizers'" >&2
    echo 'warning: unrelated warning' >&2
    echo 'error: installation problem' >&2
    echo "WARNING: another-package 1.0 does not provide the extra 'tokenizers'" >&2
    echo 'stdout remains unchanged'
    return "$1"
}
run_package_command stub "$1"
""",
        status,
        capture_output=True,
        text=True,
    )
    assert result.returncode == status
    assert result.stdout == "stdout remains unchanged\n"
    assert result.stderr == (
        "warning: The package `transformers==5.15.1` does not have an extra named `tokenizers`\n"
        "warning: unrelated warning\n"
        "error: installation problem\n"
        "WARNING: another-package 1.0 does not provide the extra 'tokenizers'\n"
    )


def test_package_filter_all_stderr():
    result = run_bash(
        """
stub() { echo "WARNING: transformers 5.15.1 does not provide the extra 'tokenizers'" >&2; }
run_package_command stub
""",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stderr == ""


EXPECTED_METADATA = {
    "name": "mlx-video-with-audio",
    "version": "0.1.36",
    "requires-dist": [
        "mlx>=0.22.0",
        "numpy",
        "safetensors",
        "huggingface_hub",
        "transformers",
        "tqdm",
        "opencv-python>=4.12.0.88",
        "Pillow>=10.3.0",
        "mlx-vlm",
        "soundfile>=0.12.0",
        'pytest; extra == "dev"',
    ],
}


@pytest.mark.parametrize("status", [0, 7])
def test_uv_metadata_and_stderr(tmp_path, status):
    stderr = tmp_path / "stderr"
    record = tmp_path / "record"
    config_copy = tmp_path / "config-copy"
    result = run_bash(
        """
unset UV_CONFIG_FILE UV_NO_CONFIG XDG_CONFIG_DIRS
export XDG_CONFIG_HOME="$1/config-home"
cd "$1"
RECORD=$2 CONFIG_COPY=$3 EXPECTED_STDERR=$4 STATUS=$5 PYTHON=$6
uv() {
    "$PYTHON" -c 'import os, sys; a = os.fstat(2); b = os.stat(sys.argv[1]); sys.exit((a.st_dev, a.st_ino) != (b.st_dev, b.st_ino))' "$EXPECTED_STDERR" || return 99
    printf '%s\n' "$UV_CONFIG_FILE" "$@" > "$RECORD"
    cat "$UV_CONFIG_FILE" > "$CONFIG_COPY"
    return "$STATUS"
}
run_package_command uv pip install
""",
        tmp_path,
        record,
        config_copy,
        stderr,
        status,
        sys.executable,
        stdout=subprocess.PIPE,
        stderr=stderr.open("w"),
        text=True,
    )
    assert result.returncode == status
    config_path, *args = record.read_text().splitlines()
    assert args == ["pip", "install"]
    assert tomllib.loads(config_copy.read_text()) == {
        "dependency-metadata": [EXPECTED_METADATA]
    }
    assert not Path(config_path).parent.exists()
    assert stderr.read_text() == ""


def test_metadata_version_matches_project_pin():
    project = tomllib.loads((SCRIPT.parent / "pyproject.toml").read_text())
    requirements = project["project"]["dependencies"] + [
        requirement
        for group in project["project"].get("optional-dependencies", {}).values()
        for requirement in group
    ]
    pins = {
        requirement.split("==", 1)[1].split(";", 1)[0].strip()
        for requirement in requirements
        if requirement.startswith("mlx-video-with-audio==")
    }
    assert pins == {EXPECTED_METADATA["version"]}


@pytest.mark.parametrize(
    "configuration",
    ["xdg_home", "preset", "ancestor", "xdg_dirs", "no_config", "pyproject"],
)
def test_uv_preserves_existing_configuration(tmp_path, configuration):
    work = tmp_path / "parent" / "child"
    work.mkdir(parents=True)
    config_home = tmp_path / "config-home"
    config_dirs = tmp_path / "config-dirs"
    preset = tmp_path / "preset.toml"
    env = {"XDG_CONFIG_HOME": str(config_home), "XDG_CONFIG_DIRS": str(config_dirs)}
    if configuration == "xdg_home":
        (config_home / "uv").mkdir(parents=True)
        (config_home / "uv" / "uv.toml").touch()
    elif configuration == "preset":
        env["UV_CONFIG_FILE"] = str(preset)
    elif configuration == "ancestor":
        (work.parent / "uv.toml").touch()
    elif configuration == "xdg_dirs":
        (config_dirs / "uv").mkdir(parents=True)
        (config_dirs / "uv" / "uv.toml").touch()
    elif configuration == "no_config":
        env["UV_NO_CONFIG"] = "1"
    else:
        (work.parent / "pyproject.toml").write_text("[tool.uv]\n")
    result = run_bash(
        """
uv() { printf '%s|%s|%s\n' "${UV_CONFIG_FILE-unset}" "${UV_NO_CONFIG-unset}" "$*"; }
run_package_command uv pip install
""",
        env=env,
        cwd=work,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert (
        result.stdout
        == f"{env.get('UV_CONFIG_FILE', 'unset')}|{env.get('UV_NO_CONFIG', 'unset')}|pip install\n"
    )


@pytest.mark.parametrize(
    "content,absent",
    [
        ('[tool."uv"]\n', False),
        ("[tool.'uv']\n", False),
        ("[ tool . uv ]\n", False),
        ('[[ tool . "uv" . index ]]\n', False),
        ('tool.uv.index = "example"\n', False),
        ('[tool]\nuv = { index-url = "https://example.invalid/simple" }\n', False),
        ("[tool.uv\n", False),
        ("[tool.ruff]\n", True),
    ],
)
def test_uv_pyproject_spellings(tmp_path, content, absent):
    (tmp_path / "pyproject.toml").write_text(content)
    bin_dir = tmp_path / "install" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").symlink_to(sys.executable)
    result = run_bash(
        'INSTALL_DIR="$1"; uv_config_absent',
        tmp_path / "install",
        env={"XDG_CONFIG_HOME": str(tmp_path / "config"), "XDG_CONFIG_DIRS": ""},
        cwd=tmp_path,
        capture_output=True,
    )
    assert (result.returncode == 0) == absent


def test_reinstall_extras_filters_pip_warning(tmp_path):
    result = run_bash(
        """
INSTALL_DIR="$1" PRESERVED_EXTRAS=chat TARGET=stable
run_install_command() {
    if [ "$1" = "$INSTALL_DIR/bin/python" ]; then echo 1.0; else "$@"; fi
}
pip_stub() {
    printf '%s\n' "$*" >> "$INSTALL_DIR/pip-calls"
    echo "WARNING: transformers 5.16.0 does not provide the extra 'tokenizers'" >&2
}
INSTALLER=(pip_stub)
reinstall_preserved_extras
""",
        tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stderr == ""
    assert "rapid-mlx[chat]==1.0" in (tmp_path / "pip-calls").read_text()


def test_reinstall_extras_uses_uv_metadata(tmp_path):
    result = run_bash(
        """
unset UV_CONFIG_FILE UV_NO_CONFIG XDG_CONFIG_DIRS
export XDG_CONFIG_HOME="$1/config"
cd "$1"
INSTALL_DIR="$1" PRESERVED_EXTRAS=chat TARGET=stable
run_install_command() {
    if [ "$1" = "$INSTALL_DIR/bin/python" ]; then echo 1.0; else "$@"; fi
}
uv() { cat "$UV_CONFIG_FILE"; }
INSTALLER=(uv pip install)
reinstall_preserved_extras
""",
        tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert 'name = "mlx-video-with-audio"' in result.stdout


@pytest.mark.parametrize("available", [True, False])
def test_quick_start_chat_hint(tmp_path, available):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(f"#!/bin/bash\nexit {0 if available else 1}\n")
    python.chmod(0o755)
    result = run_bash(
        'INSTALL_DIR="$1"; RECOMMENDED_MODEL=model RECOMMENDED_FLAGS= RAM_GB=16; print_quick_start_commands',
        tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    line = next(line for line in result.stdout.splitlines() if "rapid-mlx-chat" in line)
    assert ("first:" not in line) == available


def test_quick_start_detects_coding_agents_on_path(tmp_path):
    install_bin = tmp_path / "bin"
    install_bin.mkdir()
    python = install_bin / "python"
    python.write_text("#!/bin/bash\nexit 1\n")
    python.chmod(0o755)

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    # Clean PATH with no coding agents -> no detected line.
    empty = run_bash(
        'INSTALL_DIR="$1"; RECOMMENDED_MODEL=model RECOMMENDED_FLAGS= RAM_GB=16; print_quick_start_commands',
        tmp_path,
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert empty.returncode == 0
    assert "Detected coding agents" not in empty.stdout

    # Multiple agents on PATH -> lists detected agents in order, recommends first launch target.
    for name in ("codex", "claude", "hermes"):
        stub = agents_dir / name
        stub.write_text("#!/bin/sh\nexit 0\n")
        stub.chmod(0o755)
    found = run_bash(
        'INSTALL_DIR="$1"; RECOMMENDED_MODEL=model RECOMMENDED_FLAGS= RAM_GB=16; print_quick_start_commands',
        tmp_path,
        env={"PATH": f"{agents_dir}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert found.returncode == 0
    assert (
        "Detected coding agents (codex, claude, hermes) — connect with: rapid-mlx launch codex"
        in found.stdout
    )
    assert found.stdout.index("Detected coding agents") < found.stdout.index(
        "Quick start:"
    )

    # When claude is the first detected binary, map it to `rapid-mlx launch claude-code`.
    (agents_dir / "codex").unlink()
    claude_first = run_bash(
        'INSTALL_DIR="$1"; RECOMMENDED_MODEL=model RECOMMENDED_FLAGS= RAM_GB=16; print_quick_start_commands',
        tmp_path,
        env={"PATH": f"{agents_dir}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert claude_first.returncode == 0
    assert (
        "Detected coding agents (claude, hermes) — connect with: rapid-mlx launch claude-code"
        in claude_first.stdout
    )


def test_detect_coding_agents_ignores_functions_aliases_nonexec_and_dirs(tmp_path):
    agents_dir = tmp_path / "agents with spaces"
    agents_dir.mkdir()
    # Directory and non-executable file on PATH must not count as installed agents.
    (agents_dir / "opencode").mkdir()
    non_exec = agents_dir / "hermes"
    non_exec.write_text("#!/bin/sh\nexit 0\n")
    non_exec.chmod(0o644)

    # Shell functions and aliases must not trigger detection either.
    none_found = run_bash(
        """
shopt -s expand_aliases
alias claude='echo fake-alias'
codex() { echo fake-func; }
detect_coding_agents
""",
        env={"PATH": f"{agents_dir}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert none_found.returncode == 1
    assert none_found.stdout == ""

    # A real executable file in a PATH directory (including with spaces) is detected,
    # while functions, aliases, directories, and non-executable files stay ignored.
    pi_bin = agents_dir / "pi"
    pi_bin.write_text("#!/bin/sh\nexit 0\n")
    pi_bin.chmod(0o755)
    only_exec = run_bash(
        """
shopt -s expand_aliases
alias claude='echo fake-alias'
codex() { echo fake-func; }
detect_coding_agents
""",
        env={"PATH": f"{agents_dir}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert only_exec.returncode == 0
    assert only_exec.stdout.strip() == "pi|pi"


def run_first_chat(
    tmp_path,
    *,
    answer=None,
    env=None,
    controlling_tty=True,
    stdout_tty=True,
    model=None,
    chat_status=0,
    interrupt_chat=False,
):
    """Run the actual installer footer offline, with piped stdin and a PTY."""
    bin_dir = tmp_path / ".rapid-mlx" / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.write_text("#!/bin/bash\nexit 1\n")
    python.chmod(0o755)
    binary = bin_dir / "rapid-mlx"
    binary.write_text(
        "#!/bin/bash\nset -eu\n"
        "[ -t 0 ] && [ -t 1 ] && [ -t 2 ]\n"
        '[ ! -e "$HOME/.rapid-mlx.install-lock" ]\n'
        'printf "stub invoked: %s; telemetry=%s\\n" "$*" "$RAPID_MLX_TELEMETRY"\n'
        'IFS= read -r message\nprintf "stub received: %s\\n" "$message"\n'
        + ("while :; do :; done\n" if interrupt_chat else f"exit {chat_status}\n")
    )
    binary.chmod(0o755)
    # Include the real final call so tests catch missing or misplaced wiring.
    footer = SCRIPT.read_text().split("VERSION=$(run_install_command", 1)[1]
    footer = footer[footer.index('\necho ""') :]
    code = (
        'RAPID_INSTALL_LIB=1 source "$1"; '
        'RAM_GB=16 RAM_TIER="16 GB" RECOMMENDED_FLAGS= VERSION=stub; '
        'RECOMMENDED_MODEL="$(select_starter_model "$RAM_GB" "${CACHED_MODEL:-}")"; '
        "NEED_PATH_HINT=true; acquire_install_lock; "
        'trap on_installer_exit EXIT; trap "exit 130" INT; ' + footer
    )
    master, slave = pty.openpty()

    def setup_terminal():
        os.setsid()
        if controlling_tty:
            fcntl.ioctl(slave, termios.TIOCSCTTY, 0)

    process = subprocess.Popen(
        ["/bin/bash", "-c", code, "test", str(SCRIPT)],
        env={
            **os.environ,
            "HOME": str(tmp_path),
            "CI": "",
            "NO_COLOR": "1",
            "RAPID_INSTALL_NO_CHAT": "0",
            "RAPID_MLX_TELEMETRY": "1",
            "CACHED_MODEL": model or "",
            **(env or {}),
        },
        stdin=subprocess.PIPE,
        stdout=slave if stdout_tty else subprocess.PIPE,
        stderr=slave,
        pass_fds=(slave,),
        preexec_fn=setup_terminal,
    )
    # Leave stdin open: accidentally reading it must time out, not appear to
    # work because EOF was supplied. The chat must read its own terminal input.
    output = bytearray()
    try:
        deadline = time.monotonic() + 10
        sent = False
        sent_message = False
        interrupted = False
        while time.monotonic() < deadline:
            if select.select([master], [], [], 0.05)[0]:
                output.extend(os.read(master, 65536))
            if answer is not None and not sent and b"[Y/n] " in output:
                os.write(master, (answer + "\n").encode())
                sent = True
            if not sent_message and b"stub invoked:" in output:
                os.write(master, b"hello\n")
                sent_message = True
            if interrupt_chat and not interrupted and b"stub received:" in output:
                os.write(master, b"\x03")
                interrupted = True
            if process.poll() is not None:
                while select.select([master], [], [], 0)[0]:
                    chunk = os.read(master, 65536)
                    if not chunk:
                        break
                    output.extend(chunk)
                if not stdout_tty:
                    output.extend(process.stdout.read())
                return process.returncode, output.decode().replace("\r\n", "\n")
        pytest.fail("installer footer blocked waiting for input")
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()
        os.close(master)
        os.close(slave)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"controlling_tty": False},
        {"stdout_tty": False},
        {"env": {"RAPID_INSTALL_NO_CHAT": "1"}},
        {"env": {"CI": "true"}},
    ],
)
def test_first_chat_skips_without_blocking(tmp_path, kwargs):
    status, output = run_first_chat(tmp_path, **kwargs)
    assert status == 0, output
    assert "Start a chat" not in output
    assert "Quick start:" in output
    assert "rapid-mlx chat lfm2.5-2.6b-4bit" in output
    assert "Restart your terminal or run:" in output
    assert "stub invoked:" not in output


@pytest.mark.parametrize("answer", ["", "y", "Y", "yes"])
@pytest.mark.parametrize("model", [None, "qwen3.5-4b-4bit"])
def test_first_chat_yes_uses_installed_binary_and_recommended_model(
    tmp_path, answer, model
):
    status, output = run_first_chat(tmp_path, answer=answer, model=model)
    recommended = model or "lfm2.5-2.6b-4bit"
    assert status == 0, output
    assert output.count(f"Start a chat with {recommended} now? [Y/n]") == 1
    assert "Quick start:" not in output
    assert "The model will download first if it is not already cached." in output
    assert f"stub invoked: chat {recommended}; telemetry=1" in output
    assert output.index("Restart your terminal or run:") < output.index("stub invoked:")
    assert output.endswith("stub received: hello\n")


@pytest.mark.parametrize("answer", ["n", "N", "no", "invalid"])
def test_first_chat_no_prints_existing_quick_start(tmp_path, answer):
    status, output = run_first_chat(tmp_path, answer=answer)
    assert status == 0, output
    assert "Quick start:" in output
    assert "rapid-mlx chat lfm2.5-2.6b-4bit" in output
    assert "stub invoked:" not in output


def test_first_chat_preserves_command_exit_status(tmp_path):
    status, output = run_first_chat(tmp_path, answer="y", chat_status=7)
    assert status == 7, output
    assert output.endswith("stub received: hello\n")


def test_first_chat_ctrl_c_returns_to_shell(tmp_path):
    status, output = run_first_chat(tmp_path, answer="y", interrupt_chat=True)
    assert status in {-signal.SIGINT, 130}, output
    assert "stub received: hello" in output


def test_reusable_standalone_takes_precedence(tmp_path):
    source = SCRIPT.read_text()
    flow = source[
        source.index('PYTHON="$(select_python || true)"') : source.index(
            '\ndim "Python:', source.index('PYTHON="$(select_python || true)"')
        )
    ]
    result = run_bash(
        """
select_python() { return 1; }
standalone_python_reusable() { return 0; }
brew() { echo unexpected-brew; return 1; }
install_standalone_python() { echo unexpected-download; return 1; }
eval "$2"
printf 'selected=%s\\n' "$PYTHON"
""",
        tmp_path,
        flow,
        env={"HOME": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Using Python 3.12.13 from ~/.rapid-mlx-python" in result.stdout
    assert f"selected={tmp_path}/.rapid-mlx-python/bin/python3" in result.stdout
    assert "not found" not in result.stdout
    assert "unexpected" not in result.stdout


def test_concurrent_installers_serialize_before_python_setup(tmp_path):
    if not os.access("/usr/bin/perl", os.X_OK):
        pytest.skip("perl is required for flock")
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    scripts = {
        "uname": 'case "$1" in -m) echo arm64 ;; *) echo Darwin ;; esac',
        "sw_vers": "echo 15.0",
        "sysctl": 'case "$2" in hw.memsize) echo 17179869184 ;; hw.optional.arm64) echo 1 ;; esac',
        "python3": 'case "$1" in -c) echo 3.9 ;; --version) echo "Python 3.9.0" ;; esac',
        "curl": 'echo "$$" >> "$HOME/curl-calls"; for ((i=0; i<300; i++)); do [ ! -e "$HOME/gate" ] || exit 1; sleep 0.1; done; exit 1',
    }
    scripts["python3.12"] = scripts["python3.13"] = scripts["python3"]
    for name, body in scripts.items():
        path = stubs / name
        path.write_text("#!/bin/bash\n" + body + "\n")
        path.chmod(0o755)
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "PATH": f"{stubs}:/usr/bin:/bin",
        "RAPID_MLX_TELEMETRY": "0",
        "DO_NOT_TRACK": "1",
    }
    first = subprocess.Popen(
        ["bash", str(SCRIPT)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 20
        while not (tmp_path / "curl-calls").exists() and time.monotonic() < deadline:
            if first.poll() is not None:
                break
            time.sleep(0.1)
        assert (tmp_path / "curl-calls").exists(), "first installer did not reach curl"
        second = subprocess.run(
            ["bash", str(SCRIPT)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20,
        )
        assert second.returncode != 0
        assert "Another Rapid-MLX install is already running" in second.stdout
        assert len((tmp_path / "curl-calls").read_text().splitlines()) == 1
    finally:
        (tmp_path / "gate").touch()
        try:
            first.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            first.kill()
            first.communicate()
