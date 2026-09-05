# SPDX-License-Identifier: Apache-2.0
"""Hermetic unit tests for apps/rapid-mac/scripts/verify-dmg-layout.py.

The script structurally validates a .DS_Store against the deterministic
Rapid-MLX DMG layout (window bounds, icon view + volume-relative background
alias, icon positions, and no build-host/mount strings). These tests build
synthetic .DS_Store fixtures byte-by-byte with the standard library so they
are hermetic and Linux-runnable (no mac-only calls), then invoke the validator
exactly the way the release scripts do: ``python3 ... .DS_Store``.
"""

from __future__ import annotations

import plistlib
import struct
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER = REPO_ROOT / "apps" / "rapid-mac" / "scripts" / "verify-dmg-layout.py"

# Canonical layout contract (mirrors Verify's EXPECTED_* constants).
EXPECTED_APP_POSITION = (180, 228)
EXPECTED_APPLICATIONS_POSITION = (540, 228)

HFS_PATH_NULL = b"Rapid-MLX Desktop:.background:\x00background.png"
POSIX_PATH = b"/.background/background.png"


def _pascal(value: str, capacity: int) -> bytes:
    """Encode a Pascal string (length-prefixed) for an alias field."""
    raw = value.encode("utf-8")
    assert len(raw) < capacity
    return bytes([len(raw)]) + raw


def make_alias(posix_path: str = "/.background/background.png") -> bytes:
    """Build a v2 Alias Manager record mirroring the Swift makeFinderAlias.

    The fixed 150-byte header carries the record size, version + target kind,
    the volume name and file name. Extension records then carry the parent,
    HFS and POSIX path tags.
    """
    alias = bytearray(b"\x00" * 150)
    alias[6] = 0
    alias[7] = 2  # Alias Manager record version.
    alias[8] = 0
    alias[9] = 0  # File target.
    head = _pascal("Rapid-MLX Desktop", 28)
    alias[10 : 10 + len(head)] = head
    fname = _pascal("background.png", 64)
    alias[50 : 50 + len(fname)] = fname
    alias = bytes(alias)

    body = b""
    for tag, value in (
        (0x0000, b".background"),
        (0x0002, HFS_PATH_NULL),
        (0x0012, posix_path.encode("utf-8")),
    ):
        body += struct.pack(">HH", tag, len(value)) + value
        body += b"\x00" if len(value) % 2 else b""
    alias += body + struct.pack(">HH", 0xFFFF, 0)
    size = struct.pack(">H", len(alias))
    return alias[:4] + size + alias[6:]


def flat_bplist(value: object) -> bytes:
    """Serialize a value as a flat binary bplist payload."""
    return plistlib.dumps(value, fmt=plistlib.FMT_BINARY)


def nested_bplist(value: object) -> bytes:
    """Wrap ``value`` as a ds_store-style nested bplist ``<data>`` record.

    ``ds_store`` nests the real icvp/bwsp bplist inside an outer bplist whose
    single object is a ``<data>`` blob, so ``plistlib.loads`` yields bytes of
    the inner bplist. The validator must unwrap it.
    """
    return plistlib.dumps(flat_bplist(value), fmt=plistlib.FMT_BINARY)


def _record(marker: bytes, payload: bytes) -> bytes:
    """Encode a ``<marker><>I len</><payload>`` record."""
    return marker + struct.pack(">I", len(payload)) + payload


def make_iloc(name: str, x: int, y: int) -> bytes:
    """Encode a B-tree leaf Iloc record owned by ``name``.

    Leaf layout: ``<nlen:>I><filename:utf16be><code:4s><type:4s><value>``.
    For icon positions code/type are ``Iloc``/``blob`` and the value is the
    16-byte (x, y, flags) payload the validator expects.
    """
    name_utf16 = name.encode("utf-16-be")
    nchars = len(name_utf16) // 2
    payload = struct.pack(">IIII", x, y, 0xFFFFFFFF, 0xFFFF0000)
    return (
        struct.pack(">I", nchars)
        + name_utf16
        + b"Ilocblob"
        + struct.pack(">I", len(payload))
        + payload
    )


def make_icvp(
    alias: bytes, *, background_type: int = 2, icon_size: float = 96.0
) -> dict:
    return {
        "backgroundImageAlias": alias,
        "backgroundType": background_type,
        "iconSize": icon_size,
        "textSize": 13.0,
        "showIconPreview": True,
        "showItemInfo": False,
        "labelOnBottom": True,
        "arrangeBy": "none",
    }


MAKE_BOUNDS = {"left": 180, "top": 120, "right": 900, "bottom": 580}

HAPPY_ILOCS = [
    ("Rapid-MLX Desktop.app", *EXPECTED_APP_POSITION),
    ("Applications", *EXPECTED_APPLICATIONS_POSITION),
]


def build_store(
    *,
    alias: bytes | None = None,
    bounds: dict | None = None,
    _icvp: dict | None = None,
    ilocs: list[tuple[str, int, int]] | None = None,
    extra: bytes = b"",
    nested: bool = False,
) -> bytes:
    """Assemble a raw .DS_Store from bwsp + icvp + Iloc records.

    ``_icvp``/``bounds`` let callers inject deliberately broken records; when
    omitted the canonical values are used. ``nested`` wraps the bplist payloads
    the way ``ds_store`` does. ``extra`` appends raw bytes (e.g. a forbidden
    substring) to the end of the file.
    """
    ipayload = _icvp if _icvp is not None else make_icvp(alias or make_alias())
    bpayload = bounds if bounds is not None else MAKE_BOUNDS
    enc = nested_bplist if nested else flat_bplist
    records = [
        _record(b"bwspblob", enc(bpayload)),
        _record(b"icvpblob", enc(ipayload)),
    ]
    for name, x, y in ilocs or []:
        records.append(make_iloc(name, x, y))
    return b"".join(records) + extra


def run_verifier(fixture: bytes, *extra_args: str) -> tuple[int, str, str]:
    """Run the validator against ``fixture``, returning (rc, stdout, stderr)."""
    proc = subprocess.run(
        [sys.executable, str(VERIFIER), *extra_args],
        input=fixture,
        capture_output=True,
    )
    return proc.returncode, proc.stdout.decode(), proc.stderr.decode()


def _write_fixture(fixture: bytes, tmp_path: Path) -> Path:
    path = tmp_path / ".DS_Store"
    path.write_bytes(fixture)
    return path


def _run_on_file(path: Path) -> tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, str(VERIFIER), str(path)],
        capture_output=True,
    )
    return proc.returncode, proc.stdout.decode(), proc.stderr.decode()


class TestHappyPath:
    def test_flat_records_pass(self, tmp_path: Path) -> None:
        fixture = build_store(
            ilocs=[
                ("Rapid-MLX Desktop.app", *EXPECTED_APP_POSITION),
                ("Applications", *EXPECTED_APPLICATIONS_POSITION),
            ]
        )
        rc, out, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 0, err
        assert "verify-dmg-layout: OK" in out
        assert "FAIL" not in err

    def test_nested_ds_store_wrapped_records_still_pass(self, tmp_path: Path) -> None:
        fixture = build_store(
            nested=True,
            ilocs=[
                ("Rapid-MLX Desktop.app", *EXPECTED_APP_POSITION),
                ("Applications", *EXPECTED_APPLICATIONS_POSITION),
            ],
        )
        rc, out, _ = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 0, "nested ds_store-wrapped bplists must unwrap"
        assert "verify-dmg-layout: OK" in out


class TestStructuralFailures:
    def test_credentials_missing_bwsp_fails(self, tmp_path: Path) -> None:
        fixture = _record(b"icvpblob", flat_bplist(make_icvp(make_alias())))
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "FAIL" in err
        assert "bwsp" in err

    def test_wrong_window_bounds_fails(self, tmp_path: Path) -> None:
        fixture = build_store(
            bounds={"left": 181, "top": 120, "right": 900, "bottom": 580},
            ilocs=[
                ("Rapid-MLX Desktop.app", *EXPECTED_APP_POSITION),
                ("Applications", *EXPECTED_APPLICATIONS_POSITION),
            ],
        )
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "FAIL" in err
        assert "bounds" in err

    def test_missing_background_alias_fails(self, tmp_path: Path) -> None:
        icvp = make_icvp(make_alias())
        del icvp["backgroundImageAlias"]
        fixture = build_store(
            _icvp=icvp,
            ilocs=[
                ("Rapid-MLX Desktop.app", *EXPECTED_APP_POSITION),
                ("Applications", *EXPECTED_APPLICATIONS_POSITION),
            ],
        )
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "FAIL" in err

    def test_wrong_icon_size_fails(self, tmp_path: Path) -> None:
        fixture = build_store(
            _icvp=make_icvp(make_alias(), icon_size=128.0),
            ilocs=[
                ("Rapid-MLX Desktop.app", *EXPECTED_APP_POSITION),
                ("Applications", *EXPECTED_APPLICATIONS_POSITION),
            ],
        )
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "iconSize" in err

    def test_non_image_background_type_fails(self, tmp_path: Path) -> None:
        fixture = build_store(
            _icvp=make_icvp(make_alias(), background_type=0),
            ilocs=[
                ("Rapid-MLX Desktop.app", *EXPECTED_APP_POSITION),
                ("Applications", *EXPECTED_APPLICATIONS_POSITION),
            ],
        )
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "backgroundType" in err

    def test_bad_alias_posix_path_fails(self, tmp_path: Path) -> None:
        fixture = build_store(
            alias=make_alias(posix_path="/wrong/background.png"),
            ilocs=[
                ("Rapid-MLX Desktop.app", *EXPECTED_APP_POSITION),
                ("Applications", *EXPECTED_APPLICATIONS_POSITION),
            ],
        )
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "FAIL" in err


class TestIconPositions:
    def test_missing_iloc_filename_fails(self, tmp_path: Path) -> None:
        fixture = build_store(ilocs=[("Applications", *EXPECTED_APPLICATIONS_POSITION)])
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "FAIL" in err
        assert "positions" in err

    def test_extra_iloc_filename_fails(self, tmp_path: Path) -> None:
        fixture = build_store(
            ilocs=[
                ("Rapid-MLX Desktop.app", *EXPECTED_APP_POSITION),
                ("Applications", *EXPECTED_APPLICATIONS_POSITION),
                ("Some Other.app", 10, 10),
            ]
        )
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "FAIL" in err

    def test_wrong_app_position_fails(self, tmp_path: Path) -> None:
        fixture = build_store(
            ilocs=[("Rapid-MLX Desktop.app", 10, 228), ("Applications", 540, 228)]
        )
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "positions" in err


class TestNoHostEmbedding:
    def test_forbidden_abs_mount_string_fails(self, tmp_path: Path) -> None:
        fixture = build_store(ilocs=HAPPY_ILOCS) + b"/Volumes/Rapid-MLX Desktop"
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "FAIL" in err

    def test_forbidden_temp_mount_string_fails(self, tmp_path: Path) -> None:
        fixture = build_store(ilocs=HAPPY_ILOCS) + b"rapid-dmg-layout-abc123"
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "FAIL" in err

    def test_forbidden_tmpdir_string_fails(self, tmp_path: Path) -> None:
        fixture = build_store(ilocs=HAPPY_ILOCS) + b"/tmp/"
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "FAIL" in err


class TestCLI:
    def test_no_argument_usage(self) -> None:
        rc, _, err = run_verifier(b"")
        assert rc != 0
        assert "usage" in err

    def test_extra_positional_arg_usage(self, tmp_path: Path) -> None:
        fixture = build_store(ilocs=HAPPY_ILOCS)
        path = _write_fixture(fixture, tmp_path)
        rc, _, err = run_verifier(b"", str(path), "extra")
        assert rc != 0
        assert "usage" in err
