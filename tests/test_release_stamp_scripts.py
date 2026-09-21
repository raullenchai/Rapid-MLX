# SPDX-License-Identifier: Apache-2.0
"""Pin the release-stamp writers/verifiers used by the publish workflow.

Telemetry v2 transmits ONLY from official builds, and the ONLY writer of
``rapid_mlx/telemetry/_release_stamp.json`` is
``scripts/write_release_stamp.py`` invoked by the release workflows before
``python -m build``. These tests pin:

* the channel derivation table (including hostile spellings that must
  fail the release job instead of writing a mislabelled stamp),
* PostHog key resolution (public project token + validated env override),
* no-write-on-error, overwrite refusal / ``--force`` / idempotence, and
  the post-write round-trip through ``build_gate._parse_stamp``,
* ``scripts/verify_release_stamp.py`` against synthetic wheel (zip) and
  sdist (tar.gz) archives built in ``tmp_path``: stamp present, missing,
  malformed, wrong channel, unreadable archives, and the dist/ shape
  contract shared with ``release_manifest.release_files``.

The final test performs a REAL ``python -m build`` and is marked ``slow``
(the repo's established marker — deselect with ``-m "not slow"``), so it
never runs in the default CI lanes where ``build`` is not installed.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
import yaml

from rapid_mlx.telemetry import build_gate
from rapid_mlx.telemetry.build_gate import ReleaseStamp
from scripts import verify_release_stamp as vrs
from scripts import write_release_stamp as wrs
from scripts.verify_release_stamp import (
    _artifact_version,
    verify_artifact,
    verify_dist,
)
from scripts.write_release_stamp import (
    DEFAULT_POSTHOG_KEY,
    POSTHOG_KEY_ENV,
    StampError,
    default_dest,
    derive_channel,
    stamp_document,
    write_stamp,
)
from scripts.write_release_stamp import (
    main as write_main,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: A syntactically valid PostHog key different from the shipped default.
VALID_KEY = "phc_" + "b" * 20

#: Filename of the stamp inside a wheel, and the trailing member path
#: inside an sdist (which nests the same path under a top-level dir).
WHEEL_STAMP_MEMBER = f"rapid_mlx/telemetry/{build_gate.RELEASE_STAMP_NAME}"


# ------------------------------------------------------- channel derivation


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        # Plain stable versions, with and without the tag's leading v.
        ("0.15.0", "stable"),
        ("v0.15.0", "stable"),
        ("0.14.3", "stable"),
        ("v0.0.0", "stable"),
        # PEP 440 / tag pre-releases -> rc.
        ("0.15.0rc1", "rc"),
        ("v0.15.0rc1", "rc"),
        ("0.15.0-rc1", "rc"),
        ("v0.15.0-rc1", "rc"),
        ("0.15.0rc12", "rc"),
        ("1.2.3a1", "rc"),
        ("1.2.3a", "rc"),  # PEP 440 implicit pre-release number
        ("1.2.3b2", "rc"),
        ("1.2.3c1", "rc"),  # c is a PEP 440 alias of rc
        ("0.15.0.dev0", "rc"),
        ("0.15.0-dev1", "rc"),
        ("1.2.3dev1", "rc"),
        ("0.15.0-rc", "rc"),  # implicit number
    ],
)
def test_channel_derivation(version: str, expected: str):
    assert derive_channel(version) == expected


@pytest.mark.parametrize(
    "version",
    [
        "",  # empty
        "latest",
        "v",  # bare prefix
        "1.2",  # two components
        "v1.2",
        "v1.2.3.4",  # four components
        "1.2.3.4",
        "1.2.3+local",  # local version segment
        "1.2.3+local.dev",
        "0.15.0\n",  # trailing newline on a stable version
        "0.15.0 ",  # trailing space on a stable version
        "0.15.0rc1\n",  # trailing newline
        " 0.15.0",  # leading space
        "0.15.0rc1 ",  # trailing space
        "01.2.3",  # leading zero component
        "1.02.3",
        "0.15.0-RC1",  # uppercase pre-release: the publish lane binds lowercase only
        "V0.15.0",  # uppercase v
        "0.15.0.post1",  # post release is neither plain stable nor pre-release
        "0.15.0-alpha1",  # alpha spelled out does not normalize away here
        "١.٢.٣",  # unicode digits
        "1.2.٣",  # mixed ascii/unicode digits
        "１.２.３",  # fullwidth digits
        "release-0.15.0",
        "rapid-mac-v0.11.0",  # a desktop tag, not an engine version
    ],
)
def test_channel_derivation_rejects_hostile_versions(version: str):
    with pytest.raises(StampError, match="invalid release version"):
        derive_channel(version)


# ------------------------------------------------------------ PostHog key


def test_default_key_is_the_public_write_only_project_token():
    # Pin the exact constant: it is PostHog's PUBLIC, write-only project
    # token ("safe to use in public apps") — swapping it silently would
    # send official-build telemetry to the wrong project.
    assert DEFAULT_POSTHOG_KEY == ("phc_pnhbZbU8pZKysBtXPkd2qmF56i3ARfnY5bt9mZFdeEEW")
    # And it must be a key the runtime gate accepts on read.
    assert build_gate._POSTHOG_KEY_RE.fullmatch(DEFAULT_POSTHOG_KEY) is not None


def test_default_key_is_used_without_an_override(tmp_path):
    dest = tmp_path / "stamp.json"
    _, written = write_stamp("0.15.0", dest=dest)
    assert written
    assert json.loads(dest.read_text(encoding="utf-8"))["posthog_key"] == (
        DEFAULT_POSTHOG_KEY
    )


def test_env_override_supplies_a_different_valid_key(tmp_path):
    dest = tmp_path / "stamp.json"
    write_stamp("0.15.0", dest=dest, env={POSTHOG_KEY_ENV: VALID_KEY})
    assert json.loads(dest.read_text(encoding="utf-8"))["posthog_key"] == VALID_KEY


def test_env_override_empty_string_falls_back_to_the_default(tmp_path):
    dest = tmp_path / "stamp.json"
    write_stamp("0.15.0", dest=dest, env={POSTHOG_KEY_ENV: ""})
    assert json.loads(dest.read_text(encoding="utf-8"))["posthog_key"] == (
        DEFAULT_POSTHOG_KEY
    )


@pytest.mark.parametrize("bad_key", ["nope", "phc_short", "phc_" + "a" * 19 + "!", 42])
def test_invalid_env_override_writes_nothing(tmp_path, bad_key):
    dest = tmp_path / "stamp.json"
    with pytest.raises(StampError, match="not a valid PostHog project key"):
        write_stamp("0.15.0", dest=dest, env={POSTHOG_KEY_ENV: bad_key})  # type: ignore[dict-item]
    assert not dest.exists()


# --------------------------------------------------------- stamp writing


def test_stamp_document_has_the_gate_shape():
    parsed = build_gate._parse_stamp(stamp_document("rc", VALID_KEY))
    assert parsed == ReleaseStamp(channel="rc", posthog_key=VALID_KEY)


def test_default_dest_is_this_worktrees_package_tree():
    expected = (
        Path(wrs.__file__).resolve().parents[1]
        / "rapid_mlx"
        / "telemetry"
        / build_gate.RELEASE_STAMP_NAME
    )
    assert default_dest() == expected
    assert expected.parents[2] == REPO_ROOT


def test_default_dest_is_independent_of_the_current_working_directory(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    assert default_dest() == (
        REPO_ROOT / "rapid_mlx" / "telemetry" / build_gate.RELEASE_STAMP_NAME
    )


def test_release_stamp_is_declared_as_setuptools_package_data():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = re.search(
        r"(?ms)^\[tool\.setuptools\.package-data\]\s*$"
        r"(?P<body>.*?)(?=^\[|\Z)",
        pyproject,
    )
    assert section is not None
    rapid_mlx = re.search(
        r"(?ms)^rapid_mlx\s*=\s*\[(?P<items>.*?)^\]",
        section.group("body"),
    )
    assert rapid_mlx is not None
    items = ast.literal_eval("[" + rapid_mlx.group("items") + "]")
    assert "telemetry/_release_stamp.json" in items


@pytest.mark.parametrize("version", ["latest", "", "1.2.3+local"])
def test_bad_version_writes_nothing(tmp_path, version):
    dest = tmp_path / "nested" / "stamp.json"
    with pytest.raises(StampError, match="invalid release version"):
        write_stamp(version, dest=dest, force=True)
    assert not dest.exists()


def test_unreadable_existing_stamp_fails_without_overwrite(tmp_path):
    dest = tmp_path / "stamp.json"
    dest.mkdir()  # a directory: read_text must fail
    with pytest.raises(StampError, match="cannot read existing stamp"):
        write_stamp("0.15.0", dest=dest)
    assert dest.is_dir()


def test_overwrite_refusal_force_and_idempotence(tmp_path):
    dest = tmp_path / "stamp.json"
    write_stamp("0.15.0rc1", dest=dest)
    rc_text = dest.read_text(encoding="utf-8")

    # Different content, no --force: refused, file untouched.
    with pytest.raises(StampError, match="refusing to overwrite"):
        write_stamp("0.15.0", dest=dest)
    assert dest.read_text(encoding="utf-8") == rc_text

    # Identical content: idempotent no-op, no error.
    target, written = write_stamp("0.15.0rc1", dest=dest)
    assert (target, written) == (dest, False)
    assert dest.read_text(encoding="utf-8") == rc_text

    # --force replaces the content.
    _, written = write_stamp("0.15.0", dest=dest, force=True)
    assert written
    assert json.loads(dest.read_text(encoding="utf-8"))["channel"] == "stable"


def test_unparsable_stamp_is_removed(monkeypatch, tmp_path):
    # Force the post-write round-trip to fail: the half-written stamp must
    # be removed so a broken stamp can never linger in the package tree.
    dest = tmp_path / "stamp.json"
    monkeypatch.setattr(wrs.build_gate, "_parse_stamp", lambda raw: None)
    with pytest.raises(StampError, match="round-trip"):
        write_stamp("0.15.0", dest=dest)
    assert not dest.exists()


def test_written_stamp_round_trips_through_read_release_stamp(monkeypatch, tmp_path):
    # The contract the publish job relies on: a stamp the script wrote is
    # exactly what build_gate.read_release_stamp() returns at runtime.
    dest = tmp_path / "stamp.json"
    write_stamp("v0.15.0-rc1", dest=dest)
    monkeypatch.setattr(build_gate, "_stamp_path", lambda: dest)
    assert build_gate.read_release_stamp() == ReleaseStamp(
        channel="rc", posthog_key=DEFAULT_POSTHOG_KEY
    )


# ------------------------------------------------------------ writer CLI


def test_write_cli_success(tmp_path):
    dest = tmp_path / "stamp.json"
    assert write_main(["--version", "v0.15.0", "--dest", str(dest)]) == 0
    assert json.loads(dest.read_text(encoding="utf-8"))["channel"] == "stable"


def test_write_cli_failure_is_silent_about_the_file(tmp_path, capsys):
    dest = tmp_path / "stamp.json"
    assert write_main(["--version", "latest", "--dest", str(dest)]) == 1
    assert not dest.exists()
    assert "invalid release version" in capsys.readouterr().err


def test_write_cli_requires_version():
    with pytest.raises(SystemExit) as excinfo:
        write_main([])
    assert excinfo.value.code == 2


def test_write_cli_end_to_end_via_subprocess(tmp_path):
    # A fresh interpreter: exercises the direct-CLI import fallback (the
    # script puts the repo root on sys.path itself when `rapid_mlx` is not
    # importable) and the exact command line the publish workflow runs.
    dest = tmp_path / "stamp.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "write_release_stamp.py"),
            "--version",
            "v0.15.0-rc1",
            "--dest",
            str(dest),
        ],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env={**os.environ, "PYTHONPATH": ""},
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    stamp = build_gate._parse_stamp(dest.read_text(encoding="utf-8"))
    assert stamp == ReleaseStamp("rc", DEFAULT_POSTHOG_KEY)

    bad = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "write_release_stamp.py"),
            "--version",
            "1.2",
            "--dest",
            str(tmp_path / "never.json"),
        ],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env={**os.environ, "PYTHONPATH": ""},
        check=False,
    )
    assert bad.returncode != 0
    assert not (tmp_path / "never.json").exists()


# --------------------------------------------- verify: artifact filename


def test_artifact_version_parsing():
    assert (
        _artifact_version("rapid_mlx-0.15.0rc1-cp312-cp312-macosx_11_0_arm64.whl")
        == "0.15.0rc1"
    )
    assert _artifact_version("rapid_mlx-0.15.0.tar.gz") == "0.15.0"


@pytest.mark.parametrize(
    "filename",
    [
        "rapid_mlx-.whl",  # no version segment
        "other-0.15.0-py3-none-any.whl",  # wrong distribution name
        "not-ours.tar.gz",  # wrong sdist prefix
        "rapid_mlx-0.15.0.zip",  # unsupported extension
    ],
)
def test_artifact_version_parsing_rejects_garbage(filename):
    with pytest.raises(ValueError, match="unrecognized"):
        _artifact_version(filename)


# ------------------------------------- verify: synthetic wheel/sdist archives


def _stamp_text(channel: str = "rc", key: str = VALID_KEY) -> str:
    return stamp_document(channel, key)


def _make_wheel(
    dist: Path,
    version: str = "0.15.0rc1",
    stamp: str | bytes | None = None,
) -> Path:
    """A synthetic (structurally valid) wheel with/without a stamp."""
    path = dist / f"rapid_mlx-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("rapid_mlx/__init__.py", "")
        if stamp is not None:
            archive.writestr(WHEEL_STAMP_MEMBER, stamp)
    return path


def _add_wheel_symlink(archive: zipfile.ZipFile, member: str) -> None:
    info = zipfile.ZipInfo(member)
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    archive.writestr(info, "missing-target")


def _make_sdist(
    dist: Path,
    version: str = "0.15.0rc1",
    stamp: str | bytes | None = None,
    *,
    prefixed: bool = True,
    stamp_as_directory: bool = False,
    stamp_as_symlink: bool = False,
    decoy_stamp: bool = False,
) -> Path:
    """A synthetic sdist; *prefixed=False* stores members at the tar root."""
    path = dist / f"rapid_mlx-{version}.tar.gz"
    prefix = f"rapid_mlx-{version}/" if prefixed else ""

    def add(member: str, data: bytes | None, *, is_dir: bool = False) -> None:
        info = tarfile.TarInfo(f"{prefix}{member}")
        info.type = tarfile.DIRTYPE if is_dir else tarfile.REGTYPE
        if data is not None:
            info.size = len(data)
        tf.addfile(info, io.BytesIO(data) if data is not None else None)

    with tarfile.open(path, "w:gz") as tf:
        add("pyproject.toml", b"[project]\nname = 'rapid-mlx'\n")
        add("rapid_mlx/__init__.py", b"")
        if decoy_stamp:
            add(f"docs/{build_gate.RELEASE_STAMP_NAME}", _stamp_text().encode())
        if stamp is not None:
            payload = stamp if isinstance(stamp, bytes) else stamp.encode("utf-8")
            if stamp_as_directory:
                add(
                    f"rapid_mlx/telemetry/{build_gate.RELEASE_STAMP_NAME}",
                    None,
                    is_dir=True,
                )
            else:
                add(f"rapid_mlx/telemetry/{build_gate.RELEASE_STAMP_NAME}", payload)
        elif stamp_as_symlink:
            info = tarfile.TarInfo(
                f"{prefix}rapid_mlx/telemetry/{build_gate.RELEASE_STAMP_NAME}"
            )
            info.type = tarfile.SYMTYPE
            info.linkname = "missing-target"
            tf.addfile(info)
    return path


def _populated_dist(tmp_path: Path, **kwargs) -> tuple[Path, Path, Path]:
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = _make_wheel(dist, **kwargs)
    sdist = _make_sdist(dist, **kwargs)
    return dist, wheel, sdist


def test_verify_happy_path_both_artifacts_carry_a_matching_stamp(tmp_path):
    dist, wheel, sdist = _populated_dist(tmp_path, stamp=_stamp_text("rc"))
    results = verify_dist(dist)
    assert results == [
        (wheel, vrs.build_gate.ReleaseStamp("rc", VALID_KEY)),
        (sdist, vrs.build_gate.ReleaseStamp("rc", VALID_KEY)),
    ]
    # Cross-checking against the tag the artifacts claim to be: passes.
    assert verify_dist(dist, expected_version="v0.15.0-rc1") == results


def test_verify_accepts_root_level_sdist_members(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    _make_wheel(dist, stamp=_stamp_text("rc"))
    _make_sdist(dist, stamp=_stamp_text("rc"), prefixed=False)
    assert len(verify_dist(dist)) == 2


def test_verify_missing_stamp_in_wheel(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    _make_wheel(dist, stamp=None)
    _make_sdist(dist, stamp=_stamp_text("rc"))
    with pytest.raises(ValueError, match="wheel is missing"):
        verify_dist(dist)


def test_verify_missing_stamp_in_sdist(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    _make_wheel(dist, stamp=_stamp_text("rc"))
    _make_sdist(dist, stamp=None)
    with pytest.raises(ValueError, match="sdist is missing"):
        verify_dist(dist)


def test_verify_sdist_ignores_same_filename_outside_package_path(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    _make_wheel(dist, stamp=_stamp_text("rc"))
    _make_sdist(dist, stamp=None, decoy_stamp=True)
    with pytest.raises(ValueError, match="sdist is missing"):
        verify_dist(dist)


def test_verify_wheel_ignores_same_filename_outside_package_path(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = _make_wheel(dist, stamp=None)
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr(f"docs/{build_gate.RELEASE_STAMP_NAME}", _stamp_text("rc"))
    _make_sdist(dist, stamp=_stamp_text("rc"))
    with pytest.raises(ValueError, match="wheel is missing"):
        verify_dist(dist)


def test_verify_sdist_stamp_member_that_is_a_directory_is_not_a_stamp(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    _make_wheel(dist, stamp=_stamp_text("rc"))
    _make_sdist(dist, stamp=None, stamp_as_directory=True)
    with pytest.raises(ValueError, match="sdist is missing"):
        verify_dist(dist)


def test_verify_sdist_stamp_member_that_is_a_symlink_is_not_a_stamp(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    _make_wheel(dist, stamp=_stamp_text("rc"))
    _make_sdist(dist, stamp=None, stamp_as_symlink=True)
    with pytest.raises(ValueError, match="sdist is missing"):
        verify_dist(dist)


def test_verify_wheel_stamp_member_that_is_a_symlink_is_not_a_stamp(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = _make_wheel(dist, stamp=None)
    with zipfile.ZipFile(wheel, "a") as archive:
        _add_wheel_symlink(archive, WHEEL_STAMP_MEMBER)
    _make_sdist(dist, stamp=_stamp_text("rc"))
    with pytest.raises(ValueError, match="wheel is missing"):
        verify_dist(dist)


def test_verify_malformed_stamp(tmp_path):
    dist, _, _ = _populated_dist(tmp_path, stamp="not json at all")
    with pytest.raises(ValueError, match="does not parse"):
        verify_dist(dist)


def test_verify_stamp_with_invalid_utf8_in_wheel(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    _make_wheel(dist, stamp=b"\xff\xfe\x00\x9b")
    _make_sdist(dist, stamp=_stamp_text("rc"))
    with pytest.raises(ValueError, match="not valid UTF-8"):
        verify_dist(dist)


def test_verify_stamp_channel_contradicts_artifact_version(tmp_path):
    dist, _, _ = _populated_dist(tmp_path, stamp=_stamp_text("stable"))
    with pytest.raises(ValueError, match="does not match the artifact version"):
        verify_dist(dist)


def test_verify_stamp_channel_contradicts_the_release_tag(tmp_path):
    dist, _, _ = _populated_dist(tmp_path, stamp=_stamp_text("rc"))
    with pytest.raises(ValueError, match="does not match the release tag"):
        verify_dist(dist, expected_version="v0.15.0")


def test_verify_unreadable_wheel(tmp_path):
    garbage = tmp_path / "rapid_mlx-0.15.0rc1-py3-none-any.whl"
    garbage.write_bytes(b"this is not a zip archive")
    with pytest.raises(ValueError, match="not a readable wheel"):
        verify_artifact(garbage)


def test_verify_unreadable_sdist(tmp_path):
    garbage = tmp_path / "rapid_mlx-0.15.0rc1.tar.gz"
    garbage.write_bytes(b"this is not a tar.gz")
    with pytest.raises(ValueError, match="not a readable sdist"):
        verify_artifact(garbage)


def test_verify_unsupported_artifact_type(tmp_path):
    stranger = tmp_path / "rapid_mlx-0.15.0.zip"
    stranger.write_bytes(b"whatever")
    with pytest.raises(ValueError, match="unsupported artifact type"):
        verify_artifact(stranger)


def test_verify_rejects_a_dist_dir_with_the_wrong_shape(tmp_path):
    dist, _, _ = _populated_dist(tmp_path, stamp=_stamp_text("rc"))
    (dist / "release-manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one rapid_mlx wheel"):
        verify_dist(dist)


def test_verify_rejects_a_dist_dir_missing_the_sdist(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    _make_wheel(dist, stamp=_stamp_text("rc"))
    with pytest.raises(ValueError, match="exactly one rapid_mlx wheel"):
        verify_dist(dist)


def test_verify_dist_dir_that_is_not_a_directory(tmp_path, capsys):
    assert vrs.main([str(tmp_path / "nope")]) == 1
    assert "not a directory" in capsys.readouterr().err


def test_verify_cli_success(tmp_path, capsys):
    dist, _, _ = _populated_dist(tmp_path, stamp=_stamp_text("rc"))
    assert vrs.main([str(dist), "--version", "v0.15.0-rc1"]) == 0
    out = capsys.readouterr().out
    assert "release stamp verified in wheel and sdist" in out


def test_verify_cli_failure(tmp_path, capsys):
    dist, _, _ = _populated_dist(tmp_path, stamp=_stamp_text("stable"))
    assert vrs.main([str(dist), "--version", "v0.15.0-rc1"]) == 1
    assert "does not match" in capsys.readouterr().err


# --------------------------------------- bare-interpreter workflow contract


def test_bare_venv_runs_both_release_stamp_scripts_from_outside_repo(tmp_path):
    """The publish scripts need only stdlib, even before package install.

    The release job invokes the writer after installing only ``build`` and
    ``twine``. A no-pip venv plus ``-I -S`` pins the stronger contract: neither
    script may obtain Rapid-MLX or any dependency from the test environment.
    """
    venv = tmp_path / "bare-venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venv)],
        capture_output=True,
        text=True,
        check=True,
    )
    python = venv / "bin" / "python"
    outside = tmp_path / "outside-repo"
    outside.mkdir()
    env = os.environ.copy()
    env.pop(POSTHOG_KEY_ENV, None)

    writer = REPO_ROOT / "scripts" / "write_release_stamp.py"
    for index, (version, channel) in enumerate(
        [
            ("v0.15.0", "stable"),
            ("v0.15.0-rc1", "rc"),
            ("0.15.0rc2", "rc"),
        ]
    ):
        dest = outside / f"valid-{index}.json"
        proc = subprocess.run(
            [
                str(python),
                "-I",
                "-S",
                str(writer),
                "--version",
                version,
                "--dest",
                str(dest),
            ],
            cwd=outside,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert json.loads(dest.read_text(encoding="utf-8"))["channel"] == channel

    for index, version in enumerate(["latest", "v1.2", "0.15.0+local", ""]):
        dest = outside / f"invalid-{index}.json"
        proc = subprocess.run(
            [
                str(python),
                "-I",
                "-S",
                str(writer),
                "--version",
                version,
                "--dest",
                str(dest),
            ],
            cwd=outside,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode != 0
        assert not dest.exists()

    dist = outside / "dist"
    dist.mkdir()
    _make_wheel(dist, version="0.15.0", stamp=_stamp_text("stable"))
    _make_sdist(dist, version="0.15.0", stamp=_stamp_text("stable"))
    verifier = REPO_ROOT / "scripts" / "verify_release_stamp.py"
    proc = subprocess.run(
        [
            str(python),
            "-I",
            "-S",
            str(verifier),
            str(dist),
            "--version",
            "v0.15.0",
        ],
        cwd=outside,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "release stamp verified in wheel and sdist" in proc.stdout


# ------------------------------------------------------ workflow contracts


def _workflow_job(path: Path, job: str) -> dict:
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    return workflow["jobs"][job]


def _step_named(steps: list[dict], name: str) -> tuple[int, dict]:
    matches = [
        (index, step) for index, step in enumerate(steps) if step.get("name") == name
    ]
    assert len(matches) == 1, f"expected exactly one workflow step named {name!r}"
    return matches[0]


def test_release_artifact_matrix_stamps_and_verifies_publishable_builds():
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "release-artifact-matrix.yml"
    ).read_text(encoding="utf-8")
    stamp_step = workflow.index("- name: Write the telemetry release stamp")
    build_step = workflow.index("- name: Build and validate distributions")
    verify_step = workflow.index("- name: Verify the telemetry release stamp")
    manifest_step = workflow.index("- name: Create release artifact manifest")
    assert stamp_step < build_step < verify_step < manifest_step
    stamp_block = workflow[stamp_step:build_step]
    verify_block = workflow[verify_step:manifest_step]
    assert "if: inputs.publish && startsWith(inputs.ref, 'v')" in stamp_block
    assert "REF: ${{ inputs.ref }}" in stamp_block
    assert 'write_release_stamp.py --version "$REF"' in stamp_block
    assert "if: inputs.publish" in verify_block
    assert "${{" not in "\n".join(
        line for line in stamp_block.splitlines() if line.lstrip().startswith("run:")
    )


def test_release_artifact_matrix_has_a_final_stamp_gate_before_pypi():
    workflow_path = REPO_ROOT / ".github" / "workflows" / "release-artifact-matrix.yml"
    steps = _workflow_job(workflow_path, "publish")["steps"]
    download_index, _ = _step_named(steps, "Download candidate distribution")
    checkout_index, checkout = _step_named(steps, "Checkout the tested release source")
    verify_index, verify = _step_named(
        steps, "Verify the telemetry release stamp immediately before PyPI"
    )
    publish_index, _ = _step_named(
        steps, "Publish exact candidate to PyPI with attestations"
    )

    assert download_index < checkout_index < verify_index < publish_index
    checkouts_after_download = [
        step
        for step in steps[download_index + 1 :]
        if str(step.get("uses", "")).startswith("actions/checkout@")
    ]
    assert checkouts_after_download
    assert all(step.get("with", {}).get("path") for step in checkouts_after_download)
    assert checkout["with"]["persist-credentials"] is False
    assert set(checkout["with"]["sparse-checkout"].splitlines()) == {
        "rapid_mlx/telemetry/build_gate.py",
        "scripts/release_manifest.py",
        "scripts/release_version.py",
        "scripts/verify_release_stamp.py",
        "scripts/write_release_stamp.py",
    }
    assert checkout["with"]["sparse-checkout-cone-mode"] is False
    checkout_path = checkout["with"]["path"]
    assert verify["env"] == {
        "EXPECTED_VERSION": "${{ needs.build-candidate.outputs.version }}"
    }
    assert verify["run"] == (
        f"python3 {checkout_path}/scripts/verify_release_stamp.py "
        'candidate/dist/ --version "$EXPECTED_VERSION"'
    )


def test_legacy_publish_workflow_writes_and_verifies_the_release_stamp():
    workflow_path = REPO_ROOT / ".github" / "workflows" / "publish.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    assert workflow["env"]["RAPID_MLX_TELEMETRY"] == "0"
    steps = _workflow_job(workflow_path, "build")["steps"]
    write_index, write = _step_named(steps, "Write the telemetry release stamp")
    build_index, _ = _step_named(steps, "Build package (retry on magic-byte collision)")
    verify_index, verify = _step_named(
        steps, "Verify the telemetry release stamp in the built artifacts"
    )
    manifest_index, _ = _step_named(
        steps, "Validate distribution metadata and create release manifest"
    )
    upload_index, _ = _step_named(steps, "Upload build artifacts")

    assert write_index < build_index < verify_index < manifest_index < upload_index
    assert write["env"] == {"TAG": "${{ github.event.release.tag_name }}"}
    assert write["run"] == 'python scripts/write_release_stamp.py --version "$TAG"'
    assert verify["env"] == {"TAG": "${{ github.event.release.tag_name }}"}
    assert verify["run"] == (
        'python scripts/verify_release_stamp.py dist/ --version "$TAG"'
    )
    assert "${{" not in write["run"]
    assert "${{" not in verify["run"]


def test_legacy_publish_workflow_only_runs_for_engine_release_tags():
    workflow = (REPO_ROOT / ".github" / "workflows" / "publish.yml").read_text(
        encoding="utf-8"
    )
    build_job = workflow.split("jobs:\n", 1)[1].split("    steps:\n", 1)[0]
    assert "if: startsWith(github.event.release.tag_name, 'v')" in build_job


def test_ci_runs_the_real_release_stamp_build_once():
    steps = _workflow_job(
        REPO_ROOT / ".github" / "workflows" / "ci.yml", "test-matrix"
    )["steps"]
    _, install = _step_named(
        steps, "Install package builder for release-stamp integration test"
    )
    _, run = _step_named(steps, "Run real release-stamp package build")
    expected_if = "matrix.python-version == '3.11' && matrix.shard == 1"
    assert install["if"] == expected_if
    assert install["run"] == "python -m pip install build"
    assert run["if"] == expected_if
    assert run["run"] == (
        "pytest \\\n"
        "  tests/test_release_stamp_scripts.py::"
        "test_real_build_ships_the_stamp_in_wheel_and_sdist \\\n"
        "  --run-slow -m slow \\\n"
        "  -v --tb=short\n"
    )


# ------------------------------------------- real build (slow, opt-in)


def _project_version() -> str:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project = re.search(r"(?ms)^\[project\]\s*$(?P<body>.*?)(?=^\[|\Z)", pyproject)
    assert project is not None
    version = re.search(r'(?m)^version\s*=\s*"(?P<version>[^"]+)"\s*$', project["body"])
    assert version is not None
    return version["version"]


def _copy_tracked_tree(dest: Path) -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout.split(b"\0")
    for raw_path in tracked:
        if not raw_path:
            continue
        relative = Path(os.fsdecode(raw_path))
        source = REPO_ROOT / relative
        if not source.exists():
            continue
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _build_package_available() -> bool:
    return importlib.util.find_spec("build.__main__") is not None


@pytest.mark.parametrize(("spec", "expected"), [(None, False), (object(), True)])
def test_build_package_probe_requires_module_entrypoint(monkeypatch, spec, expected):
    probed = []

    def find_spec(name):
        probed.append(name)
        return spec

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)

    assert _build_package_available() is expected
    assert probed == ["build.__main__"]


def test_copy_tracked_tree_skips_a_tracked_file_deleted_from_the_worktree(
    monkeypatch, tmp_path
):
    missing = Path("tracked-but-deleted.txt")
    assert not (REPO_ROOT / missing).exists()
    result = subprocess.CompletedProcess(
        args=["git", "ls-files", "-z"],
        returncode=0,
        stdout=os.fsencode(missing) + b"\0",
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: result)

    _copy_tracked_tree(tmp_path / "copy")

    assert not (tmp_path / "copy" / missing).exists()


@pytest.mark.slow
def test_real_build_ships_the_stamp_in_wheel_and_sdist(tmp_path):
    """A REAL ``python -m build`` carries the stamp in BOTH artifacts.

    Proves the packaging chain end to end: the stamp written into
    ``rapid_mlx/telemetry/`` reaches the sdist (plain setuptools includes
    package_data globs when no MANIFEST.in exists) and the wheel built
    from that sdist. It builds from a temporary copy containing only
    Git-tracked files, so an interrupted test can never leave an official-build
    stamp in the checkout. Needs the ``build`` package and network access for
    build isolation, so it is marked slow and skipped when ``build`` is absent.
    """
    if not _build_package_available():
        pytest.skip("build package is not installed")
    real_stamp = default_dest()
    assert not real_stamp.exists()
    source = tmp_path / "source"
    _copy_tracked_tree(source)
    version = _project_version()
    copied_stamp = source / "rapid_mlx" / "telemetry" / build_gate.RELEASE_STAMP_NAME
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "write_release_stamp.py"),
            "--version",
            version,
            "--dest",
            str(copied_stamp),
        ],
        cwd=source,
        capture_output=True,
        text=True,
        check=True,
    )
    out = tmp_path / "dist"
    subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(out)],
        cwd=source,
        capture_output=True,
        text=True,
        check=True,
    )
    results = verify_dist(out, expected_version=version)
    expected_channel = derive_channel(version)
    assert [stamp.channel for _, stamp in results] == [
        expected_channel,
        expected_channel,
    ]
    assert not real_stamp.exists()
