# SPDX-License-Identifier: Apache-2.0
"""Pin the official-build gate contract.

Telemetry v2 is default-on, so ``official_build()`` is the one thing
standing between a developer machine and PostHog. These tests pin every
input to the decision: the release stamp file (missing, unreadable,
malformed, or valid), the PEP 610 ``direct_url.json`` verdict, git
checkout detection (clone-style directory AND worktree-style file), the
fail-closed paths, the truth table, and the process cache. No real
install metadata is consulted — everything is monkeypatched onto
``tmp_path`` — except two deliberate real-filesystem tests at the end.
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

from rapid_mlx.telemetry import build_gate
from rapid_mlx.telemetry.build_gate import ReleaseStamp

#: A syntactically valid PostHog key for stamp fixtures.
VALID_KEY = "phc_" + "a" * 20

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_stamp(directory: Path, channel: str, key: str) -> Path:
    """Write a valid-shaped stamp file and return its path."""
    stamp = directory / "stamp.json"
    stamp.write_text(
        json.dumps({"channel": channel, "posthog_key": key}),
        encoding="utf-8",
    )
    return stamp


def _direct_url_payload(dir_info: dict[str, object] | None) -> str:
    """A PEP 610 ``direct_url.json`` body, ``dir_info`` omitted when None."""
    payload: dict[str, object] = {"url": "file:///tmp/example"}
    if dir_info is not None:
        payload["dir_info"] = dir_info
    return json.dumps(payload)


class _FakeDistribution:
    """Stand-in for ``importlib.metadata.Distribution`` with a canned body."""

    def __init__(self, direct_url: str | None) -> None:
        self._direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        if filename == "direct_url.json":
            return self._direct_url
        return None


@pytest.fixture(autouse=True)
def _fresh_official_build_cache():
    """Reset the process-wide ``official_build`` cache around every test.

    ``official_build`` is ``lru_cache``-d per process by design; without
    a reset, one test's world (stamp present, install kind) would leak
    into the next through the cached answer.
    """
    build_gate._reset_for_tests()
    yield
    build_gate._reset_for_tests()


@pytest.fixture
def outside_checkout(monkeypatch, tmp_path):
    """Make the module believe its package dir is under ``tmp_path``.

    The real package dir of THIS worktree sits inside a git checkout, so
    every test that expects ``is_editable_or_source_install() is False``
    must relocate the walk first.
    """
    monkeypatch.setattr(
        build_gate,
        "__file__",
        str(tmp_path / "rapid_mlx" / "telemetry" / "build_gate.py"),
    )


# --------------------------------------------------------------- the stamp


def test_missing_stamp_file_reads_as_none(monkeypatch):
    # No monkeypatch of _stamp_path: the real package dir of this repo
    # never contains the stamp (it is not committed) — same verdict.
    assert build_gate.read_release_stamp() is None


def test_unreadable_stamp_file_reads_as_none(monkeypatch, tmp_path):
    # Point the locator at a directory: reading it raises OSError, which
    # must be swallowed into None, never propagated.
    monkeypatch.setattr(build_gate, "_stamp_path", lambda: tmp_path)
    assert build_gate.read_release_stamp() is None


@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        "",  # empty file
        "[]",  # JSON array — a stamp is an object
        '"just a string"',
        "42",
        "null",
        f'{{"channel": "beta", "posthog_key": "{VALID_KEY}"}}',  # bad channel
        f'{{"channel": 42, "posthog_key": "{VALID_KEY}"}}',  # non-str channel
        '{"channel": "stable"}',  # key missing
        f'{{"posthog_key": "{VALID_KEY}"}}',  # channel missing
    ],
)
def test_malformed_stamp_reads_as_none(monkeypatch, tmp_path, raw):
    stamp = tmp_path / "stamp.json"
    stamp.write_text(raw, encoding="utf-8")
    monkeypatch.setattr(build_gate, "_stamp_path", lambda: stamp)
    assert build_gate.read_release_stamp() is None


@pytest.mark.parametrize(
    "key",
    [
        "phc_short",  # too short
        "phc_" + "a" * 19,  # one below the 20-char floor
        "phc_" + "a" * 81,  # one above the 80-char ceiling
        "nothc_" + "a" * 20,  # wrong prefix
        "phc_" + "a" * 19 + "!",  # illegal character
        42,  # not a str
        None,  # not a str
    ],
)
def test_stamp_with_bad_posthog_key_reads_as_none(monkeypatch, tmp_path, key):
    stamp = tmp_path / "stamp.json"
    stamp.write_text(
        json.dumps({"channel": "stable", "posthog_key": key}),
        encoding="utf-8",
    )
    monkeypatch.setattr(build_gate, "_stamp_path", lambda: stamp)
    assert build_gate.read_release_stamp() is None


def test_valid_stable_stamp_round_trips(monkeypatch, tmp_path):
    stamp = _write_stamp(tmp_path, "stable", VALID_KEY)
    monkeypatch.setattr(build_gate, "_stamp_path", lambda: stamp)
    assert build_gate.read_release_stamp() == ReleaseStamp(
        channel="stable", posthog_key=VALID_KEY
    )


def test_valid_rc_stamp_round_trips(monkeypatch, tmp_path):
    stamp = _write_stamp(tmp_path, "rc", VALID_KEY)
    monkeypatch.setattr(build_gate, "_stamp_path", lambda: stamp)
    assert build_gate.read_release_stamp() == ReleaseStamp(
        channel="rc", posthog_key=VALID_KEY
    )


# ------------------------------------------- PEP 610 direct_url.json check


def test_direct_url_editable_true_is_editable(monkeypatch):
    payload = _direct_url_payload({"editable": True})
    monkeypatch.setattr(
        build_gate, "distribution", lambda _name: _FakeDistribution(payload)
    )
    assert build_gate.is_editable_or_source_install() is True


@pytest.mark.usefixtures("outside_checkout")
def test_direct_url_editable_false_is_not_editable(monkeypatch):
    payload = _direct_url_payload({"editable": False})
    monkeypatch.setattr(
        build_gate, "distribution", lambda _name: _FakeDistribution(payload)
    )
    assert build_gate.is_editable_or_source_install() is False


@pytest.mark.usefixtures("outside_checkout")
def test_direct_url_malformed_json_is_not_proven_editable(monkeypatch):
    monkeypatch.setattr(
        build_gate, "distribution", lambda _name: _FakeDistribution("{not json")
    )
    assert build_gate.is_editable_or_source_install() is False


@pytest.mark.usefixtures("outside_checkout")
def test_direct_url_json_array_is_not_proven_editable(monkeypatch):
    monkeypatch.setattr(
        build_gate, "distribution", lambda _name: _FakeDistribution("[]")
    )
    assert build_gate.is_editable_or_source_install() is False


@pytest.mark.usefixtures("outside_checkout")
def test_direct_url_missing_dir_info_is_not_proven_editable(monkeypatch):
    monkeypatch.setattr(
        build_gate,
        "distribution",
        lambda _name: _FakeDistribution(_direct_url_payload(None)),
    )
    assert build_gate.is_editable_or_source_install() is False


@pytest.mark.usefixtures("outside_checkout")
def test_direct_url_editable_non_bool_is_not_proven_editable(monkeypatch):
    payload = _direct_url_payload({"editable": "yes"})
    monkeypatch.setattr(
        build_gate, "distribution", lambda _name: _FakeDistribution(payload)
    )
    assert build_gate.is_editable_or_source_install() is False


@pytest.mark.usefixtures("outside_checkout")
def test_direct_url_file_absent_is_not_proven_editable(monkeypatch):
    # The normal PyPI-wheel case: distribution metadata readable, but no
    # direct_url.json at all — not editable, and not unknown provenance.
    monkeypatch.setattr(
        build_gate, "distribution", lambda _name: _FakeDistribution(None)
    )
    assert build_gate.is_editable_or_source_install() is False


def test_distribution_not_found_fails_closed(monkeypatch):
    def raise_pnfe(_name):
        raise PackageNotFoundError("rapid-mlx")

    monkeypatch.setattr(build_gate, "distribution", raise_pnfe)
    monkeypatch.setattr(build_gate, "_inside_git_checkout", lambda _start: False)
    # Unknown provenance must read as "editable or source" — never send.
    assert build_gate.is_editable_or_source_install() is True


def test_distribution_read_failure_fails_closed(monkeypatch):
    class _HostileDist:
        def read_text(self, filename: str) -> str | None:
            raise OSError("metadata directory unreadable")

    monkeypatch.setattr(build_gate, "distribution", lambda _name: _HostileDist())
    monkeypatch.setattr(build_gate, "_inside_git_checkout", lambda _start: False)
    assert build_gate.is_editable_or_source_install() is True


# -------------------------------------------------- git checkout detection


def test_git_directory_marks_a_checkout(tmp_path):
    package = tmp_path / "pkg" / "rapid_mlx"
    package.mkdir(parents=True)
    (tmp_path / ".git").mkdir()  # clone-style: .git is a directory
    assert build_gate._inside_git_checkout(package) is True


def test_git_file_marks_a_checkout(tmp_path):
    # git worktrees keep a .git FILE pointing at the real gitdir; the
    # walk must not insist on a directory.
    package = tmp_path / "wt" / "rapid_mlx"
    package.mkdir(parents=True)
    (tmp_path / ".git").write_text("gitdir: /somewhere/else/.git/worktrees/wt\n")
    assert build_gate._inside_git_checkout(package) is True


def test_git_not_found_within_level_bound(tmp_path):
    # .git exists 15 levels up, but the walk stops after 12 — the bound
    # wins and the verdict is False.
    deep = tmp_path
    for i in range(15):
        deep = deep / f"level_{i}"
    deep.mkdir(parents=True)
    (tmp_path / ".git").mkdir()
    assert build_gate._inside_git_checkout(deep) is False


def test_walk_reaching_root_without_git_is_not_a_checkout(tmp_path):
    # tmp_path sits a handful of levels under the filesystem root with no
    # .git anywhere above it: the walk hits the root and stops early.
    assert build_gate._inside_git_checkout(tmp_path) is False


def test_hostile_filesystem_fails_closed(monkeypatch, tmp_path):
    # A readable distribution that is not editable, but a filesystem
    # that refuses the checkout walk (e.g. unreadable parents): the
    # question cannot be answered, so the answer is True (never send).
    monkeypatch.setattr(
        build_gate,
        "distribution",
        lambda _name: _FakeDistribution(_direct_url_payload({"editable": False})),
    )
    real_exists = Path.exists

    def hostile_exists(self: Path, *args: object, **kwargs: object) -> bool:
        if self.name == ".git":
            raise PermissionError("hostile filesystem")
        return real_exists(self)

    monkeypatch.setattr(Path, "exists", hostile_exists)
    assert build_gate.is_editable_or_source_install() is True


# ------------------------------------------------- official_build decisions


@pytest.mark.parametrize(
    ("stamp_present", "source_like", "expected"),
    [
        (False, False, None),  # no stamp -> nothing to transmit behind
        (False, True, None),  # no stamp and a dev install -> doubly silent
        (True, True, None),  # stamp but editable/source install -> silent
        (True, False, ReleaseStamp("stable", VALID_KEY)),  # official build
    ],
)
def test_official_build_truth_table(
    monkeypatch, tmp_path, stamp_present, source_like, expected
):
    if stamp_present:
        stamp = _write_stamp(tmp_path, "stable", VALID_KEY)
    else:
        stamp = tmp_path / "absent.json"
    monkeypatch.setattr(build_gate, "_stamp_path", lambda: stamp)
    monkeypatch.setattr(
        build_gate,
        "distribution",
        lambda _name: _FakeDistribution(_direct_url_payload({"editable": False})),
    )
    monkeypatch.setattr(build_gate, "_inside_git_checkout", lambda _start: source_like)
    build_gate._reset_for_tests()
    assert build_gate.official_build() == expected


def test_official_build_result_is_cached_until_reset(monkeypatch, tmp_path):
    stamp = _write_stamp(tmp_path, "stable", VALID_KEY)
    monkeypatch.setattr(build_gate, "_stamp_path", lambda: stamp)
    monkeypatch.setattr(
        build_gate,
        "distribution",
        lambda _name: _FakeDistribution(_direct_url_payload({"editable": False})),
    )
    monkeypatch.setattr(build_gate, "_inside_git_checkout", lambda _start: False)
    build_gate._reset_for_tests()
    assert build_gate.official_build() is not None
    # The stamp disappears from disk, yet the cached per-process answer
    # is still served: the decision is taken once per process.
    stamp.unlink()
    assert build_gate.official_build() is not None
    # After the reset the new reality is observed.
    build_gate._reset_for_tests()
    assert build_gate.official_build() is None


def test_reset_for_tests_clears_a_stale_negative(monkeypatch, tmp_path):
    monkeypatch.setattr(build_gate, "_stamp_path", lambda: tmp_path / "absent.json")
    build_gate._reset_for_tests()
    assert build_gate.official_build() is None
    # A stamp now exists (as it would after the release workflow runs):
    # only because _reset_for_tests cleared the cache does the fresh
    # call observe it.
    monkeypatch.setattr(
        build_gate, "_stamp_path", lambda: _write_stamp(tmp_path, "rc", VALID_KEY)
    )
    monkeypatch.setattr(
        build_gate,
        "distribution",
        lambda _name: _FakeDistribution(_direct_url_payload({"editable": False})),
    )
    monkeypatch.setattr(build_gate, "_inside_git_checkout", lambda _start: False)
    build_gate._reset_for_tests()
    assert build_gate.official_build() == ReleaseStamp("rc", VALID_KEY)


# ----------------------------------------------------------- never raises


@pytest.mark.parametrize(
    "raw_bytes",
    [
        b"",
        b"\xff\xfe\x00\x9b",  # invalid UTF-8
        b"{broken",
        b'{"channel": "stable", "posthog_key": "phc_" + "a" * 20}',
        b'{"channel": ["stable"], "posthog_key": 3.5}',
        b'{"channel": {"nested": true}, "posthog_key": ["phc_" + "a" * 20]}',
        b'{"channel": "stable", "posthog_key": "phc_" + "a" * 500}',
    ],
)
def test_read_release_stamp_survives_hostile_bytes(monkeypatch, tmp_path, raw_bytes):
    stamp = tmp_path / "stamp.json"
    stamp.write_bytes(raw_bytes)
    monkeypatch.setattr(build_gate, "_stamp_path", lambda: stamp)
    result = build_gate.read_release_stamp()
    assert result is None or isinstance(result, ReleaseStamp)


def test_public_functions_survive_a_hostile_world(monkeypatch, tmp_path):
    class _ExplosiveDist:
        def read_text(self, filename: str) -> str | None:
            raise RuntimeError("boom")

    monkeypatch.setattr(build_gate, "distribution", lambda _name: _ExplosiveDist())
    monkeypatch.setattr(build_gate, "_stamp_path", lambda: tmp_path / "nope.json")
    assert build_gate.read_release_stamp() is None
    assert build_gate.is_editable_or_source_install() is True
    assert build_gate.official_build() is None


# ----------------------------------------------- real-world anchor + guard


def test_this_checkout_is_not_an_official_build():
    """This repo's own checkout IS a git checkout — the gate must stay shut.

    No monkeypatch: the real filesystem must yield ``None`` both because
    the stamp is not committed and because the walk up from
    ``rapid_mlx/telemetry`` finds this repo's ``.git`` within the bound.
    """
    build_gate._reset_for_tests()
    assert build_gate.official_build() is None
    assert build_gate.read_release_stamp() is None
    assert build_gate.is_editable_or_source_install() is True


def test_release_stamp_is_never_committed_and_is_gitignored():
    """The stamp only exists on release machines; the repo must not ship it."""
    stamp_path = REPO_ROOT / "rapid_mlx" / "telemetry" / build_gate.RELEASE_STAMP_NAME
    assert not stamp_path.exists(), (
        f"{build_gate.RELEASE_STAMP_NAME} must never be committed to the repo"
    )
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "rapid_mlx/telemetry/_release_stamp.json" in gitignore, (
        ".gitignore must cover rapid_mlx/telemetry/_release_stamp.json"
    )
