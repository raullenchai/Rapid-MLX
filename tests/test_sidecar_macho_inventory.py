from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "apps/rapid-mac/scripts/list-sidecar-machos.py"
_SPEC = importlib.util.spec_from_file_location("sidecar_macho_inventory", INVENTORY)
assert _SPEC and _SPEC.loader
_INVENTORY_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_INVENTORY_MODULE)
BUILD = ROOT / "apps/rapid-mac/scripts/build-sidecar.sh"


def test_inventory_covers_extensionless_regular_machos_and_ignores_text(tmp_path: Path):
    stage = tmp_path / "rapid-mlx"
    torch_bin = stage / "site-packages/torch/bin"
    torch_bin.mkdir(parents=True)
    source = Path("/usr/bin/true")
    assert "Mach-O" in subprocess.check_output(["file", "-b", source], text=True)

    expected = []
    for name in ("torch_shm_manager", "protoc-3.21.12.0", "protoc"):
        target = torch_bin / name
        shutil.copyfile(source, target)
        target.chmod(0o755)
        expected.append(str(target))

    # Extension and executable mode are not trusted as binary classifiers.
    disguised = stage / "lib" / "native.so"
    disguised.parent.mkdir()
    shutil.copyfile(source, disguised)
    disguised.chmod(0o644)
    expected.append(str(disguised))
    spaced = stage / "bin" / "tool with space"
    spaced.parent.mkdir()
    shutil.copyfile(source, spaced)
    spaced.chmod(0o755)
    expected.append(str(spaced))
    fat = stage / "lib" / "synthetic-fat"
    fat.write_bytes(b"\xca\xfe\xba\xbe" + b"fixture")
    expected.append(str(fat))
    text_executable = stage / "bin" / "helper"
    text_executable.parent.mkdir(exist_ok=True)
    text_executable.write_text("#!/bin/sh\nexit 0\n")
    text_executable.chmod(0o755)

    output = tmp_path / "machos.txt"
    subprocess.run([sys.executable, INVENTORY, stage, output], check=True)
    assert output.read_text().splitlines() == sorted(expected)
    assert (torch_bin / "protoc").stat().st_ino != (
        torch_bin / "protoc-3.21.12.0"
    ).stat().st_ino


def test_inventory_fails_closed_when_a_regular_file_cannot_be_read(tmp_path: Path):
    stage = tmp_path / "stage"
    stage.mkdir()
    unreadable = stage / "unreadable"
    unreadable.write_bytes(b"fixture")
    unreadable.chmod(0)
    output = tmp_path / "machos.txt"
    try:
        result = subprocess.run(
            [sys.executable, INVENTORY, stage, output], text=True, capture_output=True
        )
        assert result.returncode != 0
    finally:
        unreadable.chmod(0o600)


def test_inventory_propagates_traversal_errors(monkeypatch, tmp_path: Path):
    stage = tmp_path / "stage"
    stage.mkdir()

    def broken_walk(root, *, followlinks, onerror):
        assert root == stage
        assert followlinks is False
        onerror(PermissionError("blocked subtree"))
        return iter(())

    monkeypatch.setattr(_INVENTORY_MODULE.os, "walk", broken_walk)
    with pytest.raises(PermissionError, match="blocked subtree"):
        _INVENTORY_MODULE.list_machos(stage)


def test_inventory_rejects_newline_path_before_writing_line_protocol(tmp_path: Path):
    stage = tmp_path / "stage"
    stage.mkdir()
    newline = stage / "bad\nname"
    newline.write_bytes(b"\xcf\xfa\xed\xfe" + b"fixture")
    output = tmp_path / "machos.txt"
    result = subprocess.run(
        [sys.executable, INVENTORY, stage, output], text=True, capture_output=True
    )
    assert result.returncode != 0
    assert "contains a newline" in result.stderr


def test_signing_loop_strictly_verifies_every_inventory_entry():
    script = BUILD.read_text()
    assert (
        '"$STAGE/python/bin/python3.12" "$REPO_ROOT/scripts/list-sidecar-machos.py" "$STAGE" "$MACHOS_LIST"'
        in script
    )
    signing_loop = script.split('echo "==> codesigning', 1)[1].split(
        'done < "$MACHOS_LIST"', 1
    )[0]
    assert "codesign --force --options runtime --timestamp" in signing_loop
    assert 'verify-sidecar-signature.sh" "$f"' in signing_loop
    assert "--official" in signing_loop
    assert 'MACHO_BASELINE_COUNT="${MACHO_BASELINE_COUNT:-231}"' in script


def _fake_codesign(tmp_path: Path) -> Path:
    binary = tmp_path / "bin"
    binary.mkdir(exist_ok=True)
    script = binary / "codesign"
    script.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = --verify ]; then exit "${VERIFY_EXIT:-0}"; fi\n'
        "printf '%s\\n' \"${DETAIL_OUTPUT:-}\" >&2\n"
        'exit "${DETAIL_EXIT:-0}"\n'
    )
    script.chmod(0o755)
    return binary


def _verify(
    tmp_path: Path, details: str, *, official: bool = True, verify_exit: int = 0
):
    verifier = ROOT / "apps/rapid-mac/scripts/verify-sidecar-signature.sh"
    fake_bin = _fake_codesign(tmp_path)
    env = {
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "DETAIL_OUTPUT": details,
        "VERIFY_EXIT": str(verify_exit),
    }
    command = [verifier, tmp_path / "payload"]
    if official:
        command.append("--official")
    return subprocess.run(command, env=env, text=True, capture_output=True)


def test_official_signature_verification_requires_all_notary_properties(tmp_path: Path):
    complete = "\n".join(
        [
            "CodeDirectory v=20500 size=1 flags=0x10000(runtime) hashes=1+0 location=embedded",
            "Authority=Developer ID Application: Example (TEAMID)",
            "Timestamp=Oct 7, 2026 at 10:00:00 PM",
        ]
    )
    assert _verify(tmp_path, complete).returncode == 0
    for missing in ("Authority=", "Timestamp=", "runtime"):
        details = "\n".join(
            line for line in complete.splitlines() if missing not in line
        )
        result = _verify(tmp_path, details)
        assert result.returncode == 1, (missing, result.stderr)


def test_strict_integrity_failure_is_fatal_but_adhoc_skips_release_properties(
    tmp_path: Path,
):
    assert _verify(tmp_path, "", verify_exit=1).returncode == 1
    assert _verify(tmp_path, "", official=False).returncode == 0
