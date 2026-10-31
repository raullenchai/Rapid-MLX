"""Regression checks for installer output."""
import os
import pty
import subprocess
import sys
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "install.sh"


def run_bash(code, *args, env=None, **kwargs):
    return subprocess.run(["bash", "-c", 'RAPID_INSTALL_LIB=1 source "$1"; shift; ' + code,
                           "test", str(SCRIPT), *map(str, args)],
                          env={**os.environ, **(env or {})}, **kwargs)


@pytest.mark.parametrize("term", ["dumb", "xterm"])
def test_no_colour_on_pipe(term):
    result = run_bash('ok ok; warn warning; err error; info info; dim dim',
                      env={"TERM": term}, capture_output=True)
    assert b"\x1b" not in result.stdout + result.stderr


@pytest.mark.parametrize("term,no_color,expected", [("xterm", "", True), ("dumb", "", False), ("xterm", "1", False), ("", "", False)])
def test_terminal_colour(term, no_color, expected):
    master, slave = pty.openpty()
    try:
        result = run_bash('ok hello', env={"TERM": term, "NO_COLOR": no_color},
                          stdout=slave, stderr=subprocess.PIPE)
        output = os.read(master, 4096)
        assert result.returncode == 0
        assert (b"\x1b" in output) == expected
    finally:
        os.close(master)
        if slave != -1:
            os.close(slave)


@pytest.mark.parametrize("status", [0, 3])
def test_package_output_filter(status):
    result = run_bash('''
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
''', status, capture_output=True, text=True)
    assert result.returncode == status
    assert result.stdout == "stdout remains unchanged\n"
    assert result.stderr == ("warning: The package `transformers==5.15.1` does not have an extra named `tokenizers`\n"
                             "warning: unrelated warning\n"
                             "error: installation problem\n"
                             "WARNING: another-package 1.0 does not provide the extra 'tokenizers'\n")


def test_package_filter_all_stderr():
    result = run_bash('''
stub() { echo "WARNING: transformers 5.15.1 does not provide the extra 'tokenizers'" >&2; }
run_package_command stub
''', capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stderr == ""


@pytest.mark.parametrize("requirements,expected", [
    (["transformers>=5.0,<6", "transformers[tokenizers]>=5.1,!=5.2; extra == 'chat'"],
     "transformers>=5.0,<6,>=5.1,!=5.2"),
    (["other<2"], ""),
    (["transformers>=5,<6", "transformers>=oops"], ""),
    (["transformers>=5"], ""),
])
@pytest.mark.parametrize("target", ["stable", "1.2.3", "latest"])
def test_transformers_override(tmp_path, requirements, expected, target):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "python").symlink_to(sys.executable)
    if target == "latest":
        content = '[project]\ndependencies = ' + json.dumps(requirements[:1]) + '\n'
        content += '[project.optional-dependencies]\nchat = ' + json.dumps(requirements[1:]) + '\n'
        expected_url = "https://raw.githubusercontent.com/raullenchai/Rapid-MLX/" + "a" * 40 + "/pyproject.toml"
    else:
        content = json.dumps({"info": {"requires_dist": requirements}})
        expected_url = "https://pypi.org/pypi/rapid-mlx/" + ("json" if target == "stable" else target + "/json")
    metadata = tmp_path / "metadata"
    metadata.write_text(content)
    result = run_bash('''
INSTALL_DIR=$1 TARGET=$2 RAPID_MLX_GIT_COMMIT=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
# Pass fixture paths through globals; download has its own positional parameters.
FIXTURE=$3 EXPECTED_URL=$4
download() { [ "$1" = "$EXPECTED_URL" ] || return 1; cat "$FIXTURE"; }
transformers_override
''', tmp_path, target, metadata, expected_url, capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout == expected
    assert result.stderr == ""


@pytest.mark.parametrize("status", [0, 7])
@pytest.mark.parametrize("fetch_ok", [True, False])
def test_uv_override_and_stderr(tmp_path, status, fetch_ok):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "python").symlink_to(sys.executable)
    stderr = tmp_path / "stderr"
    record = tmp_path / "record"
    result = run_bash('''
INSTALL_DIR=$1 TARGET=stable RECORD=$2 EXPECTED_STDERR=$3 FETCH_OK=$4 STATUS=$5
download() {
    [ "$FETCH_OK" = yes ] || return 1
    printf '%s' '{"info":{"requires_dist":["transformers>=5,<6"]}}'
}
run_install_command() {
    "$INSTALL_DIR/bin/python" -I -c 'import os, sys; a = os.fstat(2); b = os.stat(sys.argv[1]); sys.exit((a.st_dev, a.st_ino) != (b.st_dev, b.st_ino))' "$EXPECTED_STDERR" || return 99
    printf '%s\\n' "$@" > "$RECORD"
    if [ "$FETCH_OK" = yes ]; then
        [ "$4" = --overrides ] || return 98
        cat "$5" >> "$RECORD"
    fi
    return "$STATUS"
}
run_package_command uv pip install
''', tmp_path, record, stderr, "yes" if fetch_ok else "no", status,
                      stdout=subprocess.PIPE, stderr=stderr.open("w"), text=True)
    assert result.returncode == status
    args = record.read_text().splitlines()
    assert args[:3] == ["uv", "pip", "install"]
    if fetch_ok:
        assert args[3] == "--overrides"
        assert args[5] == "transformers>=5,<6"
        assert not Path(args[4]).parent.exists()
    else:
        assert len(args) == 3
    assert stderr.read_text() == ""
