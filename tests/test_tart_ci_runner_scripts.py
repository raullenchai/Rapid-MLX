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
# Per slot: 120s deferred TERM + 10s stop + 120s removal + 110s margin.
EXIT_TIMEOUT = 360


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
    assert re.search(r'NODE_SHA256="\$\{NODE_SHA256:-[0-9a-f]{64}\}"', provision)
    assert 'echo "$NODE_SHA256  /tmp/node.tar.xz" | sha256sum -c -' in provision
    assert "SHASUMS256.txt" not in provision


def test_launchd_timeout():
    template = (TART / "com.rapidmlx.ci-runners.plist.template").read_text()
    plist = plistlib.loads(template.replace("__HOME__", "/Users/test").encode())
    assert plist["ExitTimeOut"] >= EXIT_TIMEOUT


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
