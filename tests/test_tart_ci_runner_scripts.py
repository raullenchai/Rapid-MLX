"""Offline checks for the host and guest CI runner scripts."""

import os
import plistlib
import re
import select
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TART = ROOT / "tools/ci-runners/tart"
SCRIPTS = sorted(TART.rglob("*.sh"))
# launchd caps GUI LaunchAgent ExitTimeOut at 60s; larger values silently
# clamp to 60 and mislead operators.
EXIT_TIMEOUT = 60


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: str(p.relative_to(TART)))
def test_bash_syntax(script):
    subprocess.run(["/bin/bash", "-n", str(script)], check=True)


def test_shellcheck():
    shellcheck = shutil.which("shellcheck")
    if not shellcheck:
        pytest.skip("shellcheck unavailable")
    subprocess.run(
        [shellcheck, "-x", "-s", "bash", "-P", "tools/ci-runners/tart/bin"]
        + [str(p.relative_to(ROOT)) for p in SCRIPTS],
        cwd=ROOT,
        check=True,
    )


def test_node_checksum():
    provision = (TART / "guest/provision.sh").read_text()
    assert re.search(r"\n  NODE_SHA256=[0-9a-f]{64}\n", provision)
    assert 'echo "$NODE_SHA256  /tmp/node.tar.xz" | sha256sum -c -' in provision
    assert "SHASUMS256.txt" not in provision


def test_launchd_timeout():
    template = (TART / "com.rapidmlx.ci-runners.plist.template").read_text()
    plist = plistlib.loads(template.replace("__HOME__", "/Users/test").encode())
    assert plist["ExitTimeOut"] == EXIT_TIMEOUT


@pytest.fixture
def fake_launchctl(tmp_path):
    fake = tmp_path / "launchctl"
    fake.write_text(
        "#!/bin/bash\n"
        'printf "%s\\n" "$@" >> "$CALLS"\n'
        'count=0; [ ! -f "$COUNTER" ] || read -r count < "$COUNTER"\n'
        'count=$((count + 1)); echo "$count" > "$COUNTER"\n'
        '[ "$count" -le "$ACTIVE_CALLS" ] || exit 113\n'
    )
    fake.chmod(0o755)
    return {
        **os.environ,
        "LAUNCHCTL": str(fake),
        "LABEL": "com.rapidmlx.ci-runners",
        "POLL_SECS": "0",
        "CALLS": str(tmp_path / "calls"),
        "COUNTER": str(tmp_path / "counter"),
        "ACTIVE_CALLS": "2",
    }


def run_wait_stopped(env, *args):
    return subprocess.run(
        ["/bin/bash", str(TART / "bin/wait-stopped.sh"), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_wait_stopped(fake_launchctl):
    result = run_wait_stopped(fake_launchctl)
    assert result.returncode == 0, result.stderr
    assert "stopped" in result.stdout
    assert Path(fake_launchctl["COUNTER"]).read_text().strip() == "3"
    assert Path(fake_launchctl["CALLS"]).read_text().splitlines() == [
        "print",
        f"gui/{os.getuid()}/com.rapidmlx.ci-runners",
    ] * 3


def test_wait_stopped_timeout(fake_launchctl):
    Path(fake_launchctl["LAUNCHCTL"]).write_text("#!/bin/bash\nexit 0\n")
    result = run_wait_stopped(fake_launchctl, "0")
    assert result.returncode == 1
    assert "com.rapidmlx.ci-runners" in result.stderr


def test_wait_stopped_invalid_timeout(fake_launchctl):
    result = run_wait_stopped(fake_launchctl, "abc")
    assert result.returncode == 2
    assert "Usage:" in result.stderr
    assert not Path(fake_launchctl["CALLS"]).exists()


def test_build_lock_and_retention():
    build = (TART / "bin/build-golden.sh").read_text()
    assert "/usr/bin/lockf -k -s -t 0" in build
    assert '"$CI_ROOT/state/build-golden.lock"' in build
    assert 'RMX_BUILD_GOLDEN_LOCKED=1 /bin/bash "$0" "$@"' in build
    # The initial delete replaces stale -prev; only the final delete is gated.
    assert re.search(
        r'if \[ "\$\{KEEP_GOLDEN_PREV:-0\}" = 1 \]; then\n'
        r"  log [^\n]+\nelse\n"
        r'  "\$TART" delete "\$PREV" [^\n]+\nfi',
        build,
    )


def test_slot_exit_trap():
    slot = (TART / "bin/slot.sh").read_text()
    assert "trap on_exit EXIT" in slot
    assert re.search(r"on_exit\(\).*?destroy_vm", slot, re.DOTALL)
    assert "''|*[!0-9]*)" in slot


@pytest.mark.parametrize(
    ("response", "runner_id", "expected"),
    [
        ('{"runner":{"id":42}}', "", "42\n"),
        ('{"runner":{"id":"x"}}', "", ""),
        ('{"runner":{}}', "", ""),
        ("invalid json", "", ""),
        ("", "", ""),
        ('{"runner":{"id":42}}', "17", "17\n"),
    ],
)
def test_slot_destroy_recovers_runner_id(tmp_path, response, runner_id, expected):
    if not shutil.which("jq"):
        pytest.skip("jq unavailable")
    slot = (TART / "bin/slot.sh").read_text()
    definitions = []
    for name in ("destroy_vm", "runner_removed"):
        match = re.search(rf"^{name}\(\) \{{\n.*?^\}}", slot, re.MULTILINE | re.DOTALL)
        assert match, f"missing {name} definition"
        definitions.append(match.group())
    harness = (
        "\n".join(definitions)
        + """
VM_PID=""; EXEC_PID=""; RUNNER_ID="$2"; resp="$3"
VM=test; TART=true; STATE="$1/slot"
delete_runner_if_idle() { printf '%s\\n' "$1"; }
destroy_vm
[ -z "$RUNNER_ID" ] && [ -z "$resp" ] || exit 1
destroy_vm
"""
    )
    result = subprocess.run(
        ["/bin/bash", "-uc", harness, "test", str(tmp_path), runner_id, response],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert result.stdout == expected
    assert result.stderr == ""


@pytest.mark.parametrize("active", [True, False], ids=["cleanup", "empty"])
def test_slot_exit_trap_behavior(tmp_path, active):
    if not shutil.which("jq"):
        pytest.skip("jq unavailable")
    slot = (TART / "bin/slot.sh").read_text()
    definitions = []
    for name in ("destroy_vm", "runner_removed", "on_exit"):
        match = re.search(rf"^{name}\(\) \{{\n.*?^\}}", slot, re.MULTILINE | re.DOTALL)
        assert match, f"missing {name} definition"
        definitions.append(match.group())
    tart = tmp_path / "tart"
    tart.write_text("#!/bin/bash\nexit 1\n")
    tart.chmod(0o755)
    harness = (
        "\n".join(definitions)
        + """
tmp="$1"; VM=test; STATE="$tmp/slot"; TART="$tmp/tart"
VM_PID=""; EXEC_PID=""; RUNNER_ID=""; resp=""
delete_runner_if_idle() { printf '%s\\n' "$1" >> "$tmp/deleted"; return 0; }
trap on_exit EXIT
"""
    )
    if active:
        harness += """
sleep 30 >/dev/null 2>&1 &
VM_PID=$!
printf '%s\\n' "$VM_PID" > "$tmp/vm.pid"
EXEC_PID=999999; resp='{"runner":{"id":42}}'
echo registered > "$STATE.runner"
exit 3
"""
    try:
        # Match slot.sh: nounset only, so cleanup continues after command failures.
        result = subprocess.run(
            ["/bin/bash", "-u", "-c", harness, "test", str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == (3 if active else 0), result.stderr
        if active:
            assert (tmp_path / "deleted").read_text() == "42\n"
            assert not (tmp_path / "slot.runner").exists()
            with pytest.raises(ProcessLookupError):
                os.kill(int((tmp_path / "vm.pid").read_text()), 0)
        else:
            assert not (tmp_path / "deleted").exists()
    finally:
        # Also stop the child if a cleanup regression makes an assertion fail.
        if (tmp_path / "vm.pid").exists():
            try:
                os.kill(int((tmp_path / "vm.pid").read_text()), 15)
            except ProcessLookupError:
                pass


def test_concurrent_build_exits_before_tart(tmp_path):
    lockf = shutil.which("lockf")
    if not lockf or not Path("/usr/bin/lockf").is_file():
        pytest.skip("macOS lockf unavailable")
    ci_root = tmp_path / "ci-runners"
    bin_dir = ci_root / "bin"
    bin_dir.mkdir(parents=True)
    state = ci_root / "state"
    state.mkdir()
    for script in (TART / "bin").glob("*.sh"):
        shutil.copy2(script, bin_dir)
    (ci_root / "host.env").write_text("MACHINE_ID=test\n")
    calls = ci_root / "tart.calls"
    stub = bin_dir / "tart"
    stub.write_text('#!/bin/bash\necho called >> "$CI_ROOT/tart.calls"\nexit 0\n')
    stub.chmod(0o755)
    # Readiness is emitted only after lockf has acquired the lock.
    holder = subprocess.Popen(
        [
            lockf,
            "-s",
            "-t",
            "0",
            str(state / "build-golden.lock"),
            sys.executable,
            "-c",
            'import sys; print("locked", flush=True); sys.stdin.read()',
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert select.select([holder.stdout], [], [], 10)[0], "lock holder not ready"
        assert holder.stdout.readline() == b"locked\n"
        env = {**os.environ, "CI_ROOT": str(ci_root)}
        env.pop("RMX_BUILD_GOLDEN_LOCKED", None)
        result = subprocess.run(
            ["/bin/bash", str(bin_dir / "build-golden.sh")],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0
        assert "build lock busy" in result.stdout
        assert not calls.exists()
    finally:
        holder.communicate(timeout=10)
