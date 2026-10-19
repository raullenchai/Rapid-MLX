"""notarize.sh retries a transiently failing staple or validate."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "apps/rapid-mac/scripts/notarize.sh"

XCRUN_STUB = """#!/bin/bash
echo "xcrun $*" >> "$STUB_LOG"
case "$1 $2" in
  "notarytool submit")
    echo '{"status":"Accepted","id":"stub-id","message":"Processing complete"}'
    ;;
  "stapler staple")
    count_file="$STUB_DIR/staple-count"
    n=$(( $(cat "$count_file" 2>/dev/null || echo 0) + 1 ))
    echo "$n" > "$count_file"
    if (( n <= STAPLE_FAILURES )); then
      echo "The staple and validate action failed! Error 68." >&2
      exit 68
    fi
    ;;
  "stapler validate")
    count_file="$STUB_DIR/validate-count"
    n=$(( $(cat "$count_file" 2>/dev/null || echo 0) + 1 ))
    echo "$n" > "$count_file"
    if (( n <= VALIDATE_FAILURES )); then
      echo "The validate action failed! Error 68." >&2
      exit 68
    fi
    ;;
esac
exit 0
"""

SLEEP_STUB = """#!/bin/bash
echo "sleep $*" >> "$STUB_LOG"
"""

SPCTL_STUB = """#!/bin/bash
echo "spctl $*" >> "$STUB_LOG"
"""


def _write_stub(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _run(
    tmp_path: Path, staple_failures: int, validate_failures: int = 0
) -> tuple[subprocess.CompletedProcess, list[str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_stub(bin_dir / "xcrun", XCRUN_STUB)
    _write_stub(bin_dir / "sleep", SLEEP_STUB)
    _write_stub(bin_dir / "spctl", SPCTL_STUB)
    submit = tmp_path / "app.zip"
    submit.write_bytes(b"zip")
    key = tmp_path / "AuthKey.p8"
    key.write_text("key")
    log = tmp_path / "stub.log"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "AC_API_KEY_ID": "id",
        "AC_API_ISSUER_ID": "issuer",
        "AC_API_KEY_PATH": str(key),
        "STUB_LOG": str(log),
        "STUB_DIR": str(tmp_path),
        "STAPLE_FAILURES": str(staple_failures),
        "VALIDATE_FAILURES": str(validate_failures),
    }
    proc = subprocess.run(
        ["bash", str(SCRIPT), str(submit), str(tmp_path / "Rapid.app")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc, log.read_text().splitlines() if log.exists() else []


@pytest.mark.parametrize("failures", [0, 1, 4])
def test_staple_retries_then_validates(tmp_path: Path, failures: int) -> None:
    proc, calls = _run(tmp_path, failures)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert calls.count(f"xcrun stapler staple {tmp_path / 'Rapid.app'}") == failures + 1
    assert [c for c in calls if c.startswith("sleep")] == [
        f"sleep {d}" for d in (15, 30, 60, 120)[:failures]
    ]
    assert calls.index(f"xcrun stapler validate {tmp_path / 'Rapid.app'}") > max(
        i for i, c in enumerate(calls) if c.startswith("xcrun stapler staple")
    )


def test_staple_gives_up_without_validating(tmp_path: Path) -> None:
    proc, calls = _run(tmp_path, 5)
    assert proc.returncode != 0
    assert "stapler staple FAILED after 5 attempts" in proc.stdout
    assert sum(c.startswith("xcrun stapler staple") for c in calls) == 5
    assert not any(c.startswith("xcrun stapler validate") for c in calls)
    assert not any(c.startswith("spctl") for c in calls)


@pytest.mark.parametrize("failures", [1, 4])
def test_validate_retries_after_staple(tmp_path: Path, failures: int) -> None:
    proc, calls = _run(tmp_path, 0, failures)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    target = tmp_path / "Rapid.app"
    assert calls.count(f"xcrun stapler staple {target}") == 1
    assert calls.count(f"xcrun stapler validate {target}") == failures + 1
    assert [c for c in calls if c.startswith("sleep")] == [
        f"sleep {d}" for d in (15, 30, 60, 120)[:failures]
    ]
    assert any(c.startswith("spctl") for c in calls)


def test_validate_gives_up_without_assessing(tmp_path: Path) -> None:
    proc, calls = _run(tmp_path, 0, 5)
    assert proc.returncode != 0
    assert "stapler validate FAILED after 5 attempts" in proc.stdout
    assert sum(c.startswith("xcrun stapler validate") for c in calls) == 5
    assert not any(c.startswith("spctl") for c in calls)
