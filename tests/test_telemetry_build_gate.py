# SPDX-License-Identifier: Apache-2.0
"""Pin the official-build gate contract.

Telemetry v2 is default-on, so ``official_build()`` is the one thing
standing between a developer machine and PostHog. These tests pin every
input to the decision: the release stamp file (location, missing,
unreadable, malformed, or valid), the PEP 610 ``direct_url.json`` verdict
bound to the RUNNING package (absent → known-not-editable,
present-but-unparseable → fail closed), source-tree detection via THIS
project's ``pyproject.toml`` beside the package (and NOT via any
``.git`` above it — real installs live inside unrelated git checkouts:
Homebrew's ``/opt/homebrew``, ``~/.pyenv``, project venvs), the
fail-closed paths, the truth table, and the process cache. Install
metadata is faked by patching the ``distributions`` iterable; the module
location is faked by patching ``__file__``. Everything lands on
``tmp_path`` — except deliberate real-filesystem tests at the end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rapid_mlx.telemetry import build_gate
from rapid_mlx.telemetry.build_gate import ReleaseStamp

#: A syntactically valid PostHog key for stamp fixtures.
VALID_KEY = "phc_" + "a" * 20

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Site-packages shape of every real wheel install (pip, Homebrew, pyenv,
#: project venv), relative to some ancestor directory.
_SITE_PACKAGES = Path(".venv") / "lib" / "python3.12" / "site-packages"


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


class _FakeDist:
    """Stand-in for ``importlib.metadata.Distribution``.

    ``base_dir`` is what ``locate_file("rapid_mlx")`` joins onto — the
    real analogue of a dist-info's parent (site-packages). ``None``
    base_dir makes ``locate_file`` return ``None`` (a hostile/unusable
    distribution). ``name`` defaults to our distribution name.
    """

    def __init__(
        self,
        base_dir: Path | None,
        direct_url: str | None,
        name: str = "rapid-mlx",
    ) -> None:
        self._base_dir = base_dir
        self._direct_url = direct_url
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def locate_file(self, path: str) -> Path | None:
        if self._base_dir is None:
            return None
        return self._base_dir / path

    def read_text(self, filename: str) -> str | None:
        if filename == "direct_url.json":
            return self._direct_url
        return None


class _ExplosiveDist:
    """A distribution whose every accessor raises (hostile metadata)."""

    @property
    def name(self) -> str:
        raise RuntimeError("boom")

    def locate_file(self, path: str) -> Path:
        raise RuntimeError("boom")

    def read_text(self, filename: str) -> str | None:
        raise RuntimeError("boom")


def _dist_bound_to_running_package(payload: str | None) -> _FakeDist:
    """A fake dist whose ``locate_file`` binds it to THIS module's package."""
    package_dir = Path(build_gate.__file__).resolve().parent.parent
    return _FakeDist(package_dir.parent, payload)


def _point_module_at(monkeypatch, tmp_path) -> Path:
    """Relocate the module into a fake ``.../site-packages/rapid_mlx`` tree."""
    package_dir = tmp_path / "site-packages" / "rapid_mlx"
    (package_dir / "telemetry").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        build_gate,
        "__file__",
        str(package_dir / "telemetry" / "build_gate.py"),
    )
    return package_dir


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
def site_packages_install(monkeypatch, tmp_path):
    """A fake wheel-install world under ``tmp_path``.

    The module sits in site-packages (so the pyproject lookup finds
    nothing — the shape of a real wheel install) and exactly one
    distribution is bound to it, reporting a non-editable direct_url.
    Tests that expect ``is_editable_or_source_install() is False`` are
    thereby isolated from THIS worktree's real source tree and from the
    test venv's own (editable, other-worktree) metadata.
    """
    package_dir = _point_module_at(monkeypatch, tmp_path)
    monkeypatch.setattr(
        build_gate,
        "distributions",
        lambda: iter(
            [_dist_bound_to_running_package(_direct_url_payload({"editable": False}))]
        ),
    )


# --------------------------------------------------------------- the stamp


def test_stamp_path_is_pinned_to_the_package_directory():
    # The stamp lives NEXT TO build_gate.py, inside the installed
    # package — never the current directory, where any user could forge
    # one. Pins the locator against the module-dir seam.
    assert build_gate._stamp_path() == Path(build_gate.__file__).with_name(
        build_gate.RELEASE_STAMP_NAME
    )


def test_real_stamp_location_round_trips_without_patching_the_locator(
    monkeypatch, tmp_path
):
    # Write a valid stamp into a fake package dir and read it back with
    # the REAL _stamp_path: only __file__ is relocated. This is the
    # positive end-to-end for the locator — a stamp the release workflow
    # would have written beside the module is actually found there.
    package_dir = tmp_path / "site-packages" / "rapid_mlx"
    (package_dir / "telemetry").mkdir(parents=True)
    stamp = package_dir / "telemetry" / build_gate.RELEASE_STAMP_NAME
    stamp.write_text(
        json.dumps({"channel": "stable", "posthog_key": VALID_KEY}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        build_gate, "__file__", str(package_dir / "telemetry" / "build_gate.py")
    )
    assert build_gate.read_release_stamp() == ReleaseStamp("stable", VALID_KEY)


def test_missing_stamp_file_reads_as_none():
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
        build_gate,
        "distributions",
        lambda: iter([_dist_bound_to_running_package(payload)]),
    )
    assert build_gate.is_editable_or_source_install() is True


@pytest.mark.usefixtures("site_packages_install")
def test_direct_url_editable_false_is_not_editable():
    assert build_gate.is_editable_or_source_install() is False


@pytest.mark.usefixtures("site_packages_install")
def test_direct_url_malformed_json_fails_closed(monkeypatch):
    # A PRESENT direct_url.json that does not parse is UNKNOWN
    # provenance, not proof of a wheel install: fail closed (True).
    monkeypatch.setattr(
        build_gate,
        "distributions",
        lambda: iter([_dist_bound_to_running_package("{not json")]),
    )
    assert build_gate.is_editable_or_source_install() is True


@pytest.mark.usefixtures("site_packages_install")
def test_direct_url_json_array_fails_closed(monkeypatch):
    # Valid JSON, but not an object: same unknown-provenance verdict.
    monkeypatch.setattr(
        build_gate,
        "distributions",
        lambda: iter([_dist_bound_to_running_package("[]")]),
    )
    assert build_gate.is_editable_or_source_install() is True


@pytest.mark.parametrize(
    "dir_info",
    [None, "not-a-mapping"],
    ids=["dir_info-missing", "dir_info-not-a-dict"],
)
@pytest.mark.usefixtures("site_packages_install")
def test_direct_url_without_dir_info_is_known_not_editable(monkeypatch, dir_info):
    # dir_info absent (or not a mapping) in a VALID direct_url.json is
    # exactly how pip records a direct-URL install: known, not editable.
    monkeypatch.setattr(
        build_gate,
        "distributions",
        lambda: iter([_dist_bound_to_running_package(_direct_url_payload(dir_info))]),
    )
    assert build_gate.is_editable_or_source_install() is False


@pytest.mark.usefixtures("site_packages_install")
def test_direct_url_editable_non_bool_is_not_proven_editable(monkeypatch):
    monkeypatch.setattr(
        build_gate,
        "distributions",
        lambda: iter(
            [_dist_bound_to_running_package(_direct_url_payload({"editable": "yes"}))]
        ),
    )
    assert build_gate.is_editable_or_source_install() is False


@pytest.mark.usefixtures("site_packages_install")
def test_direct_url_file_absent_is_not_proven_editable(monkeypatch):
    # The normal PyPI-wheel case: distribution metadata readable, but no
    # direct_url.json at all — pip omits it for registry installs. Known
    # provenance, not editable, and NOT unknown: fail open to False.
    monkeypatch.setattr(
        build_gate,
        "distributions",
        lambda: iter([_dist_bound_to_running_package(None)]),
    )
    assert build_gate.is_editable_or_source_install() is False


def test_matching_dist_wins_over_a_stale_editable_one(monkeypatch, tmp_path):
    # Metadata resolves by NAME through sys.path order and can describe a
    # DIFFERENT install than the one being imported. A stale editable
    # rapid-mlx dist-info earlier in the list must NOT silence a genuine
    # wheel install of the running package.
    package_dir = _point_module_at(monkeypatch, tmp_path)
    stale = _FakeDist(tmp_path / "elsewhere", _direct_url_payload({"editable": True}))
    real = _FakeDist(package_dir.parent, _direct_url_payload({"editable": False}))
    monkeypatch.setattr(build_gate, "distributions", lambda: iter([stale, real]))
    assert build_gate.is_editable_or_source_install() is False


def test_no_matching_distribution_fails_closed(monkeypatch, tmp_path):
    # No distribution's locate_file maps onto the running package dir —
    # whether wrong name or wrong location — so provenance is unknown.
    package_dir = _point_module_at(monkeypatch, tmp_path)
    wrong_name = _FakeDist(
        package_dir.parent, _direct_url_payload({"editable": True}), name="other-pkg"
    )
    wrong_location = _FakeDist(
        tmp_path / "elsewhere", _direct_url_payload({"editable": True})
    )
    monkeypatch.setattr(
        build_gate, "distributions", lambda: iter([wrong_name, wrong_location])
    )
    assert build_gate.is_editable_or_source_install() is True


def test_bound_distribution_read_failure_fails_closed(monkeypatch, tmp_path):
    # A distribution IS bound to the running package, but its metadata
    # refuses to be read: unknown provenance, fail closed.
    package_dir = _point_module_at(monkeypatch, tmp_path)

    class _UnreadableDist:
        name = "rapid-mlx"

        def locate_file(self, path: str) -> Path:
            return package_dir

        def read_text(self, filename: str) -> str | None:
            raise OSError("metadata directory unreadable")

    monkeypatch.setattr(build_gate, "distributions", lambda: iter([_UnreadableDist()]))
    assert build_gate.is_editable_or_source_install() is True


def test_distribution_scan_failure_fails_closed(monkeypatch):
    def boom():
        raise OSError("metadata scan failed")

    monkeypatch.setattr(build_gate, "distributions", boom)
    assert build_gate.is_editable_or_source_install() is True


def test_hostile_or_unusable_distributions_are_skipped(monkeypatch, tmp_path):
    # Distributions that raise, or whose locate_file yields nothing (or a
    # non-path), must be skipped — the well-behaved match further down
    # the list still gets its verdict honored.
    package_dir = _point_module_at(monkeypatch, tmp_path)
    nonlocatable = _FakeDist(None, _direct_url_payload({"editable": True}))
    good = _FakeDist(package_dir.parent, _direct_url_payload({"editable": False}))
    monkeypatch.setattr(
        build_gate,
        "distributions",
        lambda: iter([_ExplosiveDist(), nonlocatable, _IntDist(), good]),
    )
    assert build_gate.is_editable_or_source_install() is False


class _IntDist:
    """A hostile distribution whose locate_file returns a non-path."""

    @property
    def name(self) -> str:
        return "rapid-mlx"

    def locate_file(self, path: str) -> int:
        return 42  # type: ignore[return-value]


class _SpyDist:
    """A distribution that records whether its METADATA (name) was read."""

    def __init__(self, base_dir: Path, direct_url: str | None) -> None:
        self._base_dir = base_dir
        self._direct_url = direct_url
        self.name_reads = 0

    @property
    def name(self) -> str:
        self.name_reads += 1
        return "rapid-mlx"

    def locate_file(self, path: str) -> Path:
        return self._base_dir / path

    def read_text(self, filename: str) -> str | None:
        if filename == "direct_url.json":
            return self._direct_url
        return None


@pytest.mark.parametrize("dist_name", ["rapid_mlx", "Rapid-MLX", "rapid.mlx"])
def test_pep503_equivalent_distribution_names_are_honored(
    monkeypatch, tmp_path, dist_name
):
    # METADATA rewritten to a PEP 503-equivalent Name (e.g. by a repacker)
    # must not blind the gate: both sides are normalized before compare.
    package_dir = _point_module_at(monkeypatch, tmp_path)
    dist = _FakeDist(
        package_dir.parent, _direct_url_payload({"editable": False}), name=dist_name
    )
    monkeypatch.setattr(build_gate, "distributions", lambda: iter([dist]))
    # False proves the variant-named dist was SELECTED: a skipped dist
    # would leave provenance unknown (True).
    assert build_gate.is_editable_or_source_install() is False


def test_pep503_non_equivalent_distribution_name_is_skipped(monkeypatch, tmp_path):
    package_dir = _point_module_at(monkeypatch, tmp_path)
    dist = _FakeDist(
        package_dir.parent,
        _direct_url_payload({"editable": False}),
        name="rapid-mlx-extra",
    )
    monkeypatch.setattr(build_gate, "distributions", lambda: iter([dist]))
    assert build_gate.is_editable_or_source_install() is True


def test_name_metadata_is_read_only_for_a_location_match(monkeypatch, tmp_path):
    # METADATA parsing (dist.name) is the expensive half of the scan: it
    # must run only for a distribution whose package dir matches.
    package_dir = _point_module_at(monkeypatch, tmp_path)
    elsewhere = _SpyDist(
        tmp_path / "elsewhere", _direct_url_payload({"editable": True})
    )
    here = _SpyDist(package_dir.parent, _direct_url_payload({"editable": False}))
    monkeypatch.setattr(build_gate, "distributions", lambda: iter([elsewhere, here]))
    assert build_gate.is_editable_or_source_install() is False
    assert elsewhere.name_reads == 0
    assert here.name_reads >= 1


def test_location_match_with_hostile_name_is_skipped_fail_closed(monkeypatch, tmp_path):
    # The location matches, but reading METADATA explodes: skip the dist
    # (never raise) — with no other match, provenance is unknown.
    package_dir = _point_module_at(monkeypatch, tmp_path)

    class _HostileNameDist:
        @property
        def name(self) -> str:
            raise RuntimeError("boom")

        def locate_file(self, path: str) -> Path:
            return package_dir

    monkeypatch.setattr(build_gate, "distributions", lambda: iter([_HostileNameDist()]))
    assert build_gate.is_editable_or_source_install() is True


# ------------------------------------------------- source-tree detection


@pytest.mark.parametrize("git_style", ["directory", "file"])
def test_site_packages_inside_a_git_checkout_is_not_source(tmp_path, git_style):
    # THE acceptance case: Homebrew (/opt/homebrew/.git with the package
    # ~10 levels below), pyenv (~/.pyenv) and the classic project venv
    # (~/code/myproject/.venv) all nest site-packages inside a git
    # checkout. A .git anywhere above proves nothing — and indeed no
    # pyproject.toml sits beside the package here, so: not source.
    if git_style == "directory":
        (tmp_path / ".git").mkdir()  # clone-style: .git is a directory
    else:
        (tmp_path / ".git").write_text("gitdir: /somewhere/else/.git/worktrees/wt\n")
    package = tmp_path / _SITE_PACKAGES / "rapid_mlx"
    package.mkdir(parents=True)
    assert build_gate._package_is_in_source_tree(package) is False


def test_package_beside_matching_pyproject_is_source(tmp_path):
    package = tmp_path / "rapid_mlx"
    (package / "telemetry").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "rapid-mlx"\n',
        encoding="utf-8",
    )
    assert build_gate._package_is_in_source_tree(package) is True


@pytest.mark.parametrize(
    "pyproject_text",
    [
        # A monorepo vendoring us: rapid_mlx/ beside someone else's project.
        '[project]\nname = "some-monorepo"\n',
        # Near-miss names must not match: the quotes anchor the value.
        '[project]\nname = "rapid-mlx-extra"\n',
        '[project]\nname = "not-rapid-mlx"\n',
        # The authors array mentions rapid-mlx, but the project does not.
        '[project]\nname = "vendor-app"\n'
        'authors = [{name = "rapid-mlx contributors"}]\n',
        # No name line at all.
        '[project]\nversion = "0.1.0"\n',
    ],
)
def test_pyproject_for_another_or_vaguely_named_project_is_not_source(
    tmp_path, pyproject_text
):
    package = tmp_path / "rapid_mlx"
    package.mkdir()
    (tmp_path / "pyproject.toml").write_text(pyproject_text, encoding="utf-8")
    assert build_gate._package_is_in_source_tree(package) is False


@pytest.mark.parametrize(
    "name_line",
    [
        'name = "rapid-mlx"',  # canonical
        "name = 'rapid-mlx'",  # single quotes
        'name="rapid-mlx"',  # no spaces
        'name   =   "rapid-mlx"',  # extra spaces
        '  name = "rapid-mlx"',  # indented (inside [project])
        'name = "rapid-mlx"  # ours',  # trailing comment
    ],
)
def test_pyproject_name_line_is_matched_tolerantly(tmp_path, name_line):
    package = tmp_path / "rapid_mlx"
    package.mkdir()
    (tmp_path / "pyproject.toml").write_text(
        f"[project]\n{name_line}\n",
        encoding="utf-8",
    )
    assert build_gate._package_is_in_source_tree(package) is True


def test_unreadable_pyproject_fails_closed(monkeypatch, tmp_path):
    # A pyproject.toml exists but refuses to be read: the source-tree
    # question cannot be answered, so the answer is True (never send).
    package = tmp_path / "rapid_mlx"
    package.mkdir()
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "rapid-mlx"\n')
    real_read_text = Path.read_text

    def hostile_read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self.name == "pyproject.toml":
            raise PermissionError("hostile filesystem")
        return real_read_text(self, *args, **kwargs)  # type: ignore[arg-type,return-value]

    monkeypatch.setattr(Path, "read_text", hostile_read_text)
    assert build_gate._package_is_in_source_tree(package) is True


def test_missing_pyproject_is_not_source(tmp_path):
    # Plain directory without pyproject.toml and without a checkout
    # interpretation: the site-packages verdict, reached via the same
    # FileNotFoundError path as any real install.
    package = tmp_path / "rapid_mlx"
    package.mkdir()
    assert build_gate._package_is_in_source_tree(package) is False


def test_site_packages_inside_git_is_not_source_via_public_api(monkeypatch, tmp_path):
    # End-to-end shape of the misclassification the git-walk draft had:
    # direct_url says wheel-install, .git sits above the venv, and the
    # verdict must still be "not editable or source" (False = may send).
    (tmp_path / ".git").mkdir()
    _point_module_at(monkeypatch, tmp_path)
    monkeypatch.setattr(
        build_gate,
        "distributions",
        lambda: iter(
            [_dist_bound_to_running_package(_direct_url_payload({"editable": False}))]
        ),
    )
    assert build_gate.is_editable_or_source_install() is False


def test_source_tree_via_public_api(monkeypatch, tmp_path):
    # The checkout/sdist shape: rapid_mlx/ directly beside the matching
    # pyproject.toml — even with a .git present anywhere, the pyproject
    # is what identifies OUR tree.
    (tmp_path / ".git").mkdir()
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "rapid-mlx"\n')
    monkeypatch.setattr(
        build_gate,
        "__file__",
        str(tmp_path / "rapid_mlx" / "telemetry" / "build_gate.py"),
    )
    monkeypatch.setattr(
        build_gate,
        "distributions",
        lambda: iter(
            [_dist_bound_to_running_package(_direct_url_payload({"editable": False}))]
        ),
    )
    assert build_gate.is_editable_or_source_install() is True


# ------------------------------------------------- never-raise hard guards


def test_missing_dunder_file_fails_closed(monkeypatch):
    # Frozen/embedded interpreters can lack __file__ entirely: the
    # package directory is unknowable -> fail closed, never NameError.
    monkeypatch.delattr(build_gate, "__file__")
    assert build_gate.is_editable_or_source_install() is True


def test_none_dunder_file_fails_closed(monkeypatch):
    # A __file__ of None (as some embedded embedders set it) must not
    # explode Path(): fail closed, never TypeError.
    monkeypatch.setattr(build_gate, "__file__", None)
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
    _point_module_at(monkeypatch, tmp_path)
    monkeypatch.setattr(build_gate, "_stamp_path", lambda: stamp)
    monkeypatch.setattr(
        build_gate,
        "distributions",
        lambda: iter(
            [_dist_bound_to_running_package(_direct_url_payload({"editable": False}))]
        ),
    )
    monkeypatch.setattr(
        build_gate, "_package_is_in_source_tree", lambda _dir: source_like
    )
    build_gate._reset_for_tests()
    assert build_gate.official_build() == expected


def test_official_build_ignores_runtime_kill_switches(monkeypatch, tmp_path):
    """The gate identifies release bytes; consent separately controls uploads."""

    stamp = _write_stamp(tmp_path, "stable", VALID_KEY)
    _point_module_at(monkeypatch, tmp_path)
    monkeypatch.setattr(build_gate, "_stamp_path", lambda: stamp)
    monkeypatch.setattr(
        build_gate,
        "distributions",
        lambda: iter(
            [_dist_bound_to_running_package(_direct_url_payload({"editable": False}))]
        ),
    )
    monkeypatch.setattr(build_gate, "_package_is_in_source_tree", lambda _dir: False)
    monkeypatch.setenv("RAPID_MLX_TELEMETRY", "0")
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    assert build_gate.official_build() == ReleaseStamp("stable", VALID_KEY)


def test_official_build_result_is_cached_until_reset(monkeypatch, tmp_path):
    stamp = _write_stamp(tmp_path, "stable", VALID_KEY)
    _point_module_at(monkeypatch, tmp_path)
    monkeypatch.setattr(build_gate, "_stamp_path", lambda: stamp)
    monkeypatch.setattr(
        build_gate,
        "distributions",
        lambda: iter(
            [_dist_bound_to_running_package(_direct_url_payload({"editable": False}))]
        ),
    )
    monkeypatch.setattr(build_gate, "_package_is_in_source_tree", lambda _dir: False)
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
    _point_module_at(monkeypatch, tmp_path)
    monkeypatch.setattr(
        build_gate,
        "distributions",
        lambda: iter(
            [_dist_bound_to_running_package(_direct_url_payload({"editable": False}))]
        ),
    )
    monkeypatch.setattr(build_gate, "_package_is_in_source_tree", lambda _dir: False)
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
    def boom():
        raise RuntimeError("metadata scan exploded")

    monkeypatch.setattr(build_gate, "distributions", boom)
    monkeypatch.setattr(build_gate, "_stamp_path", lambda: tmp_path / "nope.json")
    assert build_gate.read_release_stamp() is None
    assert build_gate.is_editable_or_source_install() is True
    assert build_gate.official_build() is None


# ----------------------------------------------- real-world anchor + guard


def test_this_checkout_is_not_an_official_build():
    """This repo's own checkout is OUR source tree — the gate must stay shut.

    No monkeypatch: the real filesystem must yield ``None`` both because
    the stamp is not committed and because ``rapid_mlx/`` sits directly
    beside this repo's ``pyproject.toml`` (``[project]`` name
    ``rapid-mlx``).
    """
    build_gate._reset_for_tests()
    assert build_gate.official_build() is None
    assert build_gate.read_release_stamp() is None
    assert build_gate.is_editable_or_source_install() is True


def test_real_package_dir_sits_beside_this_projects_pyproject():
    """The real ``rapid_mlx`` package dir is directly beside the real pyproject.

    Pins the ``parent.parent`` arithmetic against the real filesystem and
    asserts the module under test is loaded from THIS worktree — the test
    venv's own rapid-mlx metadata is an EDITABLE install pointing at a
    different worktree, exactly the stale-metadata mismatch the
    distribution binding exists to ignore.
    """
    package_dir = Path(build_gate.__file__).resolve().parent.parent
    assert package_dir == (REPO_ROOT / "rapid_mlx").resolve()
    assert (package_dir.parent / "pyproject.toml").is_file()
    assert build_gate._package_is_in_source_tree(package_dir) is True


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
