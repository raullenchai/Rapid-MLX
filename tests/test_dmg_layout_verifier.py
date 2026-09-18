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

import importlib.util
import plistlib
import struct
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER = REPO_ROOT / "apps" / "rapid-mac" / "scripts" / "verify-dmg-layout.py"

# Canonical layout contract (mirrors Verify's EXPECTED_* constants).
EXPECTED_APP_POSITION = (180, 228)
EXPECTED_APPLICATIONS_POSITION = (540, 228)
# Parked below the 460pt window fold so Finder never draws it on the install
# page, even for a viewer running with hidden files shown.
EXPECTED_BACKGROUND_POSITION = (100, 560)

HFS_PATH_NULL = b"Rapid-MLX Desktop:.background:\x00background.png"
POSIX_PATH = b"/.background/background.png"
# Alias tag 0x0013: Finder resolves the background as mount point + POSIX
# path, so the record must carry the canonical /Volumes mount point.
MOUNT_POINT = b"/Volumes/Rapid-MLX Desktop"
# The bwsp/icvp/icvl records belong to the root "." entry; a real record's
# structure id is preceded by this leaf header (nlen=1 + "." in UTF-16BE).
# The verifier anchors its marker scan to it, so fixtures must build the
# records with the header too (see _dot_blob / ICVL_RECORD).
DOT_LEAF_HEADER = b"\x00\x00\x00\x01\x00."
# icvl is a 4-byte OSType ``type`` record (code + type + value), not a blob.
ICVL_RECORD = DOT_LEAF_HEADER + b"icvltypeicnv"


def _pascal(value: str, capacity: int) -> bytes:
    """Encode a Pascal string (length-prefixed) for an alias field."""
    raw = value.encode("utf-8")
    assert len(raw) < capacity
    return bytes([len(raw)]) + raw


def make_alias(
    posix_path: str = "/.background/background.png",
    *,
    include_mount_point: bool = True,
) -> bytes:
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

    tags = [
        (0x0000, b".background"),
        (0x0002, HFS_PATH_NULL),
        (0x0012, posix_path.encode("utf-8")),
    ]
    if include_mount_point:
        tags.append((0x0013, MOUNT_POINT))
    body = b""
    for tag, value in tags:
        body += struct.pack(">HH", tag, len(value)) + value
        body += b"\x00" if len(value) % 2 else b""
    alias += body + struct.pack(">HH", 0xFFFF, 0)
    size = struct.pack(">H", len(alias))
    return alias[:4] + size + alias[6:]


def flat_bplist(value: object) -> bytes:
    """Serialize a value as a flat binary bplist payload."""
    return plistlib.dumps(value, fmt=plistlib.FMT_BINARY)


def nested_bplist(value: object) -> bytes:
    """Wrap ``value`` as a bplist whose root is a ``<data>`` blob of a bplist.

    This is what ``ds_store`` writes when handed pre-serialised bytes instead
    of a dict, and what the 0.14.0–0.14.2 template shipped. Finder reads the
    record into an NSDictionary and aborts with ``-[__NSCFData count]:
    unrecognized selector`` on macOS 14 and 26.6 (#3468). The validator must
    reject it.
    """
    return plistlib.dumps(flat_bplist(value), fmt=plistlib.FMT_BINARY)


def _record(marker: bytes, payload: bytes) -> bytes:
    """Encode a ``<marker><>I len</><payload>`` record."""
    return marker + struct.pack(">I", len(payload)) + payload


def _dot_blob(code: bytes, payload: bytes) -> bytes:
    """Encode a root-``.`` ``<code>blob`` record, anchored to the leaf header
    exactly as Finder writes it (and as the verifier now requires)."""
    return DOT_LEAF_HEADER + _record(code + b"blob", payload)


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
    alias: bytes,
    *,
    background_type: int = 2,
    icon_size: float = 96.0,
    include_colors: bool = True,
) -> dict:
    icvp = {
        "backgroundImageAlias": alias,
        "backgroundType": background_type,
        "iconSize": icon_size,
        "textSize": 13.0,
        "showIconPreview": True,
        "showItemInfo": False,
        "labelOnBottom": True,
        "arrangeBy": "none",
    }
    if include_colors:
        # Finder drops the whole icvp record without the colour triple, even
        # in image-background mode (measured on macOS 26.5).
        icvp["backgroundColorRed"] = 1.0
        icvp["backgroundColorGreen"] = 1.0
        icvp["backgroundColorBlue"] = 1.0
    return icvp


# Finder's own bwsp schema: WindowBounds "{{x, y}, {w, h}}" plus chrome
# switches, all off so the install page is a bare icon window.
MAKE_BOUNDS = {
    "ContainerShowSidebar": False,
    "PreviewPaneVisibility": False,
    "ShowPathbar": False,
    "ShowSidebar": False,
    "ShowStatusBar": False,
    "ShowTabView": False,
    "ShowToolbar": False,
    "SidebarWidth": 0,
    "WindowBounds": "{{180, 120}, {720, 460}}",
}

HAPPY_ILOCS = [
    ("Rapid-MLX Desktop.app", *EXPECTED_APP_POSITION),
    ("Applications", *EXPECTED_APPLICATIONS_POSITION),
    (".background", *EXPECTED_BACKGROUND_POSITION),
]


def build_store(
    *,
    alias: bytes | None = None,
    bounds: object | None = None,
    _icvp: dict | None = None,
    ilocs: list[tuple[str, int, int]] | None = None,
    extra: bytes = b"",
    nested: bool = False,
    include_icvl: bool = True,
) -> bytes:
    """Assemble a raw .DS_Store from bwsp + icvp + Iloc records.

    ``_icvp``/``bounds`` let callers inject deliberately broken records; when
    omitted the canonical values are used. ``nested`` wraps the bplist payloads
    as ``<data>``-rooted bplists (the Finder-crashing encoding). ``extra``
    appends raw bytes (e.g. a forbidden substring) to the end of the file.
    """
    ipayload = _icvp if _icvp is not None else make_icvp(alias or make_alias())
    bpayload = bounds if bounds is not None else MAKE_BOUNDS
    enc = nested_bplist if nested else flat_bplist
    records = [
        _dot_blob(b"bwsp", enc(bpayload)),
        _dot_blob(b"icvp", enc(ipayload)),
    ]
    if include_icvl:
        records.append(ICVL_RECORD)
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
                (".background", *EXPECTED_BACKGROUND_POSITION),
            ]
        )
        rc, out, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 0, err
        assert "verify-dmg-layout: OK" in out
        assert "FAIL" not in err

    def test_committed_template_passes(self) -> None:
        """The shipped template must be exactly what the validator demands."""
        template = (
            REPO_ROOT / "apps" / "rapid-mac" / "Resources" / "finder-layout.DS_Store"
        )
        rc, out, err = _run_on_file(template)
        assert rc == 0, err
        assert "verify-dmg-layout: OK" in out

    def test_committed_template_records_are_flat_dicts(self) -> None:
        """Regression for #3468: both plist records must have a dict root.

        Checked directly with plistlib, independent of the validator, so a
        future validator change cannot silently re-admit the nested form.
        """
        template = (
            REPO_ROOT / "apps" / "rapid-mac" / "Resources" / "finder-layout.DS_Store"
        )
        data = template.read_bytes()
        for marker in (b"bwspblob", b"icvpblob"):
            pos = data.find(marker)
            assert pos >= 0, marker
            length = struct.unpack(">I", data[pos + 8 : pos + 12])[0]
            root = plistlib.loads(data[pos + 12 : pos + 12 + length])
            assert isinstance(root, dict), f"{marker!r} root is {type(root).__name__}"
        bwsp_pos = data.find(b"bwspblob")
        bwsp_len = struct.unpack(">I", data[bwsp_pos + 8 : bwsp_pos + 12])[0]
        bwsp = plistlib.loads(data[bwsp_pos + 12 : bwsp_pos + 12 + bwsp_len])
        assert bwsp["WindowBounds"] == "{{180, 120}, {720, 460}}"


class TestFinderCrashingEncodings:
    """Encodings Finder rejects at open time must fail the gate (#3468)."""

    def test_nested_bplist_records_fail(self, tmp_path: Path) -> None:
        fixture = build_store(nested=True, ilocs=HAPPY_ILOCS)
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "FAIL" in err
        assert "<data> wrapping another bplist" in err
        assert "3468" in err

    def test_nested_icvp_only_fails(self, tmp_path: Path) -> None:
        records = [
            _dot_blob(b"bwsp", flat_bplist(MAKE_BOUNDS)),
            _dot_blob(b"icvp", nested_bplist(make_icvp(make_alias()))),
        ]
        fixture = b"".join(records) + b"".join(make_iloc(*i) for i in HAPPY_ILOCS)
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "icvp record root is <data>" in err

    def test_plain_data_root_fails(self, tmp_path: Path) -> None:
        records = [
            _dot_blob(b"bwsp", flat_bplist(b"not a plist at all")),
            _dot_blob(b"icvp", flat_bplist(make_icvp(make_alias()))),
        ]
        fixture = b"".join(records) + b"".join(make_iloc(*i) for i in HAPPY_ILOCS)
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "bwsp record root is <data>, not a dict" in err

    def test_array_root_fails(self, tmp_path: Path) -> None:
        fixture = build_store(bounds=[180, 120, 900, 580], ilocs=HAPPY_ILOCS)
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "bwsp record root is list, not a dict" in err

    def test_garbage_record_fails(self, tmp_path: Path) -> None:
        records = [
            _dot_blob(b"bwsp", b"\x00\x01\x02 definitely not a bplist"),
            _dot_blob(b"icvp", flat_bplist(make_icvp(make_alias()))),
        ]
        fixture = b"".join(records) + b"".join(make_iloc(*i) for i in HAPPY_ILOCS)
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "bwsp record is not a valid bplist" in err


class TestStructuralFailures:
    def test_credentials_missing_bwsp_fails(self, tmp_path: Path) -> None:
        fixture = _dot_blob(b"icvp", flat_bplist(make_icvp(make_alias())))
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "FAIL" in err
        assert "bwsp" in err

    def test_wrong_window_bounds_fails(self, tmp_path: Path) -> None:
        fixture = build_store(
            bounds={**MAKE_BOUNDS, "WindowBounds": "{{181, 120}, {720, 460}}"},
            ilocs=[
                ("Rapid-MLX Desktop.app", *EXPECTED_APP_POSITION),
                ("Applications", *EXPECTED_APPLICATIONS_POSITION),
                (".background", *EXPECTED_BACKGROUND_POSITION),
            ],
        )
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "FAIL" in err
        assert "bounds" in err

    def test_legacy_edge_dict_bounds_fails(self, tmp_path: Path) -> None:
        """The pre-#3468 {left,top,right,bottom} schema is not Finder's."""
        fixture = build_store(
            bounds={"left": 180, "top": 120, "right": 900, "bottom": 580},
            ilocs=HAPPY_ILOCS,
        )
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "unexpected window bounds None" in err

    def test_finder_chrome_switched_on_fails(self, tmp_path: Path) -> None:
        fixture = build_store(
            bounds={**MAKE_BOUNDS, "ShowToolbar": True}, ilocs=HAPPY_ILOCS
        )
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "bwsp ShowToolbar is True" in err

    def test_preview_pane_enabled_fails(self, tmp_path: Path) -> None:
        # A bare install window has the preview pane off; the generator writes
        # PreviewPaneVisibility=False, so the gate must reject it enabled.
        fixture = build_store(
            bounds={**MAKE_BOUNDS, "PreviewPaneVisibility": True}, ilocs=HAPPY_ILOCS
        )
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "bwsp PreviewPaneVisibility is True" in err

    def test_nonzero_sidebar_width_fails(self, tmp_path: Path) -> None:
        # SidebarWidth is an int (0 for a bare window); a nonzero width would
        # reserve a sidebar column, so it must be rejected too.
        fixture = build_store(
            bounds={**MAKE_BOUNDS, "SidebarWidth": 180}, ilocs=HAPPY_ILOCS
        )
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "bwsp SidebarWidth is 180" in err

    def test_missing_background_alias_fails(self, tmp_path: Path) -> None:
        icvp = make_icvp(make_alias())
        del icvp["backgroundImageAlias"]
        fixture = build_store(
            _icvp=icvp,
            ilocs=[
                ("Rapid-MLX Desktop.app", *EXPECTED_APP_POSITION),
                ("Applications", *EXPECTED_APPLICATIONS_POSITION),
                (".background", *EXPECTED_BACKGROUND_POSITION),
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
                (".background", *EXPECTED_BACKGROUND_POSITION),
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
                (".background", *EXPECTED_BACKGROUND_POSITION),
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
                (".background", *EXPECTED_BACKGROUND_POSITION),
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
                (".background", *EXPECTED_BACKGROUND_POSITION),
                ("Some Other.app", 10, 10),
            ]
        )
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "FAIL" in err

    def test_duplicate_background_iloc_fails(self, tmp_path: Path) -> None:
        """A second record must not overwrite an earlier visible position."""
        fixture = build_store(
            ilocs=[
                ("Rapid-MLX Desktop.app", *EXPECTED_APP_POSITION),
                ("Applications", *EXPECTED_APPLICATIONS_POSITION),
                (".background", 105, 64),
                (".background", *EXPECTED_BACKGROUND_POSITION),
            ]
        )
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "duplicate Iloc record" in err

    def test_background_inside_window_fails(self, tmp_path: Path) -> None:
        """A .background parked on the visible page must be rejected.

        This is the regression the position exists for: with Finder set to
        show hidden files there is no flag that suppresses the icon, so the
        only defence is keeping it below the window fold.
        """
        fixture = build_store(
            ilocs=[
                ("Rapid-MLX Desktop.app", *EXPECTED_APP_POSITION),
                ("Applications", *EXPECTED_APPLICATIONS_POSITION),
                (".background", 105, 64),
            ]
        )
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "window fold" in err

    def test_background_below_fold_is_not_pinned(self, tmp_path: Path) -> None:
        """Any below-fold parking spot is accepted, not just the shipped one.

        The guarantee is "Finder cannot draw it on the install page", so the
        validator must gate the property rather than one coordinate pair.
        """
        fixture = build_store(
            ilocs=[
                ("Rapid-MLX Desktop.app", *EXPECTED_APP_POSITION),
                ("Applications", *EXPECTED_APPLICATIONS_POSITION),
                (".background", 400, 700),
            ]
        )
        rc, out, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 0, err
        assert "verify-dmg-layout: OK" in out

    def test_wrong_app_position_fails(self, tmp_path: Path) -> None:
        fixture = build_store(
            ilocs=[
                ("Rapid-MLX Desktop.app", 10, 228),
                ("Applications", 540, 228),
                (".background", *EXPECTED_BACKGROUND_POSITION),
            ]
        )
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "positions" in err


class TestNoHostEmbedding:
    def test_canonical_mount_point_is_allowed(self, tmp_path: Path) -> None:
        """The exact /Volumes/<volume name> mount point is required, not forbidden.

        The background alias resolves as mount point + POSIX path, so the
        canonical mount point (derived from the volume name, not the build
        host) must be present and must not trip the no-host-embedding gate.
        """
        fixture = build_store(ilocs=HAPPY_ILOCS)
        assert b"/Volumes/Rapid-MLX Desktop" in fixture
        rc, out, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 0, err
        assert "verify-dmg-layout: OK" in out

    def test_forbidden_suffixed_mount_string_fails(self, tmp_path: Path) -> None:
        """A suffixed mount (macOS assigns "... 1" when the name is taken)
        means the alias was captured against a build-time remount and must
        be rejected."""
        fixture = build_store(ilocs=HAPPY_ILOCS) + b"/Volumes/Rapid-MLX Desktop 1"
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


class TestMarkerAnchoring:
    """A marker occurring inside another record's payload must not be mistaken
    for a structural record (#3468 review hardening). Finder reads the B-tree,
    so a .DS_Store with no real icvp/icvl record must be rejected even if the
    marker bytes appear elsewhere.
    """

    def test_icvpblob_embedded_in_bwsp_payload_is_not_a_record(
        self, tmp_path: Path
    ) -> None:
        # No real "." icvp record; the icvp marker + a canonical icvp plist is
        # buried inside a data value of the (anchored, real, still-valid) bwsp
        # plist, so bwsp parses cleanly and only the anchoring stops it.
        fake_icvp = _record(b"icvpblob", flat_bplist(make_icvp(make_alias())))
        bwsp = {**MAKE_BOUNDS, "ignoredData": fake_icvp}
        fixture = (
            _dot_blob(b"bwsp", flat_bplist(bwsp))
            + ICVL_RECORD
            + b"".join(make_iloc(*i) for i in HAPPY_ILOCS)
        )
        assert fixture.count(b"icvpblob") == 1  # the buried one
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "icvp record, found 0" in err

    def test_icvl_marker_embedded_in_icvp_payload_is_not_a_record(
        self, tmp_path: Path
    ) -> None:
        # No real "." icvl record; the icvl marker is buried in a bytes value
        # inside the (anchored, real) icvp record's payload.
        icvp = make_icvp(make_alias())
        icvp["ignoredData"] = b"icvltypeicnv"
        fixture = (
            _dot_blob(b"bwsp", flat_bplist(MAKE_BOUNDS))
            + _dot_blob(b"icvp", flat_bplist(icvp))
            + b"".join(make_iloc(*i) for i in HAPPY_ILOCS)
        )
        assert b"icvltypeicnv" in fixture  # only the buried one
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "icvl record, found 0" in err

    def test_leaf_prefixed_icvp_embedded_in_bwsp_payload_is_rejected(
        self, tmp_path: Path
    ) -> None:
        # Round-3 hardening: unlike the unprefixed marker above, this buries a
        # FULLY leaf-header-prefixed icvp record (DOT_LEAF_HEADER + "icvpblob"
        # + len + a valid, complete icvp plist) inside a bwsp data value, with
        # no real "." icvp record. The anchored scan *does* find this marker
        # (count == 1) and the buried plist would pass every content check, so
        # only the top-level boundary check rejects it: Finder, walking the
        # real B-tree, never reaches the interior of a value.
        buried = _dot_blob(b"icvp", flat_bplist(make_icvp(make_alias())))
        bwsp = {**MAKE_BOUNDS, "ignoredData": buried}
        fixture = (
            _dot_blob(b"bwsp", flat_bplist(bwsp))
            + ICVL_RECORD
            + b"".join(make_iloc(*i) for i in HAPPY_ILOCS)
        )
        assert fixture.count(DOT_LEAF_HEADER + b"icvpblob") == 1  # the buried one
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "embedded inside another record's payload" in err

    def test_leaf_prefixed_icvl_embedded_in_icvp_payload_is_rejected(
        self, tmp_path: Path
    ) -> None:
        # Same construction for the icvl view-mode record: a full, real
        # ICVL_RECORD (DOT_LEAF_HEADER + "icvltypeicnv") is buried inside the
        # real icvp record's payload, with no top-level icvl. The scan finds
        # exactly one icvl marker, so only the boundary check stops it.
        icvp = make_icvp(make_alias())
        icvp["ignoredData"] = ICVL_RECORD
        fixture = (
            _dot_blob(b"bwsp", flat_bplist(MAKE_BOUNDS))
            + _dot_blob(b"icvp", flat_bplist(icvp))
            + b"".join(make_iloc(*i) for i in HAPPY_ILOCS)
        )
        assert fixture.count(ICVL_RECORD) == 1  # the buried one
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "embedded inside another record's payload" in err

    def test_leaf_prefixed_icvp_embedded_in_unrecognized_blob_is_rejected(
        self, tmp_path: Path
    ) -> None:
        # Round-4 hardening: the fabricated icvp lives inside an UNRECOGNIZED
        # root "." blob ("Junk") -- a record the verifier does not parse for
        # content -- with no real icvp present. The marker scan still finds
        # exactly one icvp and it would pass every content check, so only the
        # every-"."-payload overlap check catches it. Finder, reading the
        # B-tree, ignores "Junk" and opens without the icon-view settings.
        fake_icvp = _dot_blob(b"icvp", flat_bplist(make_icvp(make_alias())))
        fixture = (
            _dot_blob(b"bwsp", flat_bplist(MAKE_BOUNDS))
            + _dot_blob(b"Junk", fake_icvp)
            + ICVL_RECORD
            + b"".join(make_iloc(*i) for i in HAPPY_ILOCS)
        )
        assert fixture.count(DOT_LEAF_HEADER + b"icvpblob") == 1  # the buried one
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "embedded inside another record's payload" in err


class TestRenderCriticalRecords:
    """Records that, if wrong, leave Finder showing a blank/default window
    even though the volume opens — the second class of #3468 symptom.
    """

    def test_missing_color_triple_fails(self, tmp_path: Path) -> None:
        icvp = make_icvp(make_alias(), include_colors=False)
        fixture = build_store(_icvp=icvp, ilocs=HAPPY_ILOCS)
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "backgroundColorRed" in err

    def test_nonfinite_or_out_of_range_color_fails(self, tmp_path: Path) -> None:
        # A colour channel that is present and a float but NaN, infinite, or
        # outside [0, 1] is not a value Finder renders; the gate must reject it
        # (an isinstance-float check alone would pass NaN/inf).
        for bad in (float("nan"), float("inf"), -0.5, 1.5):
            icvp = make_icvp(make_alias())
            icvp["backgroundColorGreen"] = bad
            fixture = build_store(_icvp=icvp, ilocs=HAPPY_ILOCS)
            rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
            assert rc == 1, f"expected failure for backgroundColorGreen={bad!r}"
            assert "finite colour component" in err

    def test_missing_alias_mount_point_fails(self, tmp_path: Path) -> None:
        icvp = make_icvp(make_alias(include_mount_point=False))
        fixture = build_store(_icvp=icvp, ilocs=HAPPY_ILOCS)
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "mount point" in err or "0x0013" in err

    def test_missing_icvl_record_fails(self, tmp_path: Path) -> None:
        fixture = build_store(ilocs=HAPPY_ILOCS, include_icvl=False)
        rc, _, err = _run_on_file(_write_fixture(fixture, tmp_path))
        assert rc == 1
        assert "icvl" in err


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


class TestGeneratorDeterminism:
    """The committed template is a build artifact of make-dmg-layout.py; the
    generator is the source of truth. Guard against it drifting or becoming
    nondeterministic while the binary-only tests above stay green. Skipped
    where ds_store/mac_alias are absent (the generator needs them; the rest of
    this suite is stdlib-only)."""

    def test_generator_reproduces_committed_template_byte_for_byte(
        self, tmp_path: Path
    ) -> None:
        pytest.importorskip("ds_store")
        pytest.importorskip("mac_alias")
        gen_path = VERIFIER.parent / "make-dmg-layout.py"
        spec = importlib.util.spec_from_file_location("make_dmg_layout", gen_path)
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)

        first = tmp_path / "first.DS_Store"
        second = tmp_path / "second.DS_Store"
        gen.write_template(first)
        gen.write_template(second)

        committed = (
            REPO_ROOT / "apps" / "rapid-mac" / "Resources" / "finder-layout.DS_Store"
        ).read_bytes()
        # Deterministic across runs, and identical to what is committed: a
        # generator change that alters the bytes must land with a regenerated
        # template, and one that is nondeterministic fails here.
        assert first.read_bytes() == second.read_bytes()
        assert first.read_bytes() == committed
