"""Regression checks for installer output."""

import os
import pty
import shutil
import subprocess
import sys
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
    if not shutil.which("perl", path="/usr/bin:/bin"):
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
