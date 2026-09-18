# SPDX-License-Identifier: Apache-2.0
"""Three fail-open paths in the provenance chain, closed.

1. A packaged app with no build stamp resolved to ``distribution: release``.
   ``_source_checkout_revision`` returns ``None`` for an ``.app`` or sidecar
   layout *by design*, so a packaging omission became an official-release
   claim that nothing downstream could contradict.
2. Every reader applied its own partial rules, so a document one refused
   another repaired: ``resolve_provenance`` silently dropped a revision from a
   release stamp, and the stored-provenance reader accepted unknown fields.
3. ``git status ... || true`` turned a failed status command into empty output,
   which is indistinguishable from a clean tree.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from rapid_mlx.community_bench import provenance_schema, run_builder
from rapid_mlx.community_bench.provenance_schema import ProvenanceInvalid
from rapid_mlx.community_bench.workspace import (
    LocalRunArchive,
    ProvenanceUnreadable,
    validate_provenance,
)

REPO = Path(__file__).resolve().parents[1]
BUILD_SIDECAR = REPO / "apps/rapid-mac/scripts/build-sidecar.sh"


@pytest.fixture(autouse=True)
def _forget_cached_provenance():
    run_builder._reset_provenance_cache()
    yield
    run_builder._reset_provenance_cache()


def _script(name: str):
    path = REPO / "apps/rapid-mac/scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 1. A packaged runtime without a stamp
# ---------------------------------------------------------------------------


def _packaged_module(root: Path) -> Path:
    package = root / "rapid-mlx" / "site-packages" / "rapid_mlx" / "community_bench"
    package.mkdir(parents=True)
    (root / "rapid-mlx" / "VERSION").write_text("0.13.4\n")
    module = package / "run_builder.py"
    module.write_text("")
    return module


def test_a_packaged_sidecar_without_a_stamp_fails_before_measuring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fail-open this replaces: no stamp + no checkout revision → release."""

    module = _packaged_module(tmp_path)
    monkeypatch.setattr(run_builder, "_BUILD_STAMP", tmp_path / "absent.json")
    monkeypatch.setattr(run_builder, "__file__", str(module))
    with pytest.raises(RuntimeError, match="missing its build provenance stamp"):
        run_builder.resolve_provenance()


def test_a_bundled_app_without_a_stamp_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "Rapid.app" / "Contents" / "Resources" / "lib" / "rapid_mlx"
    package.mkdir(parents=True)
    module = package / "run_builder.py"
    module.write_text("")
    monkeypatch.setattr(run_builder, "_BUILD_STAMP", tmp_path / "absent.json")
    monkeypatch.setattr(run_builder, "__file__", str(module))
    with pytest.raises(RuntimeError, match="missing its build provenance stamp"):
        run_builder.resolve_provenance()


def test_a_packaged_release_with_a_valid_stamp_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Finder-launched packaged release still works — with its stamp."""

    module = _packaged_module(tmp_path)
    stamp = tmp_path / "_build_stamp.json"
    stamp.write_text(json.dumps({"distribution": "release"}))
    monkeypatch.setattr(run_builder, "_BUILD_STAMP", stamp)
    monkeypatch.setattr(run_builder, "__file__", str(module))
    monkeypatch.setenv("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")

    def no_git(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("a stamped packaged build probed Git")

    monkeypatch.setattr(run_builder.subprocess, "run", no_git)
    assert run_builder.resolve_provenance() == {"distribution": "release"}


def test_an_unpackaged_checkout_without_a_stamp_still_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The legacy path this must not disturb: running from a source tree."""

    monkeypatch.setattr(run_builder, "_BUILD_STAMP", tmp_path / "absent.json")
    monkeypatch.setattr(run_builder, "_source_checkout_revision", lambda *_: None)
    assert run_builder.resolve_provenance() == {"distribution": "release"}


# ---------------------------------------------------------------------------
# 2. One closed schema, applied everywhere
# ---------------------------------------------------------------------------

VALID = [
    {"distribution": "release"},
    {"distribution": "source", "revision": "a" * 40},
    {"distribution": "source", "revision": "a" * 40, "dirty": True},
    {"distribution": "source", "revision": "a" * 40, "dirty": False},
]

INVALID = [
    pytest.param({}, id="empty"),
    pytest.param({"distribution": "nightly"}, id="unknown-distribution"),
    pytest.param({"revision": "a" * 40}, id="no-distribution"),
    pytest.param(
        {"distribution": "release", "revision": "a" * 40}, id="release+revision"
    ),
    pytest.param({"distribution": "release", "dirty": True}, id="release+dirty"),
    pytest.param({"distribution": "release", "dirty": False}, id="release+dirty-false"),
    pytest.param({"distribution": "release", "note": "x"}, id="release+unknown"),
    pytest.param({"distribution": "source"}, id="source-no-revision"),
    pytest.param({"distribution": "source", "revision": "abc"}, id="short-revision"),
    pytest.param(
        {"distribution": "source", "revision": "A" * 40}, id="uppercase-revision"
    ),
    pytest.param(
        {"distribution": "source", "revision": "z" * 40}, id="non-hex-revision"
    ),
    pytest.param(
        {"distribution": "source", "revision": "a" * 40, "dirty": "true"},
        id="dirty-string",
    ),
    pytest.param(
        {"distribution": "source", "revision": "a" * 40, "dirty": 1}, id="dirty-int"
    ),
    pytest.param(
        {"distribution": "source", "revision": "a" * 40, "note": "x"},
        id="source+unknown",
    ),
    pytest.param([1, 2, 3], id="not-an-object"),
    pytest.param("release", id="a-string"),
]


@pytest.mark.parametrize("document", VALID)
def test_the_schema_accepts_valid_documents(document: dict) -> None:
    assert provenance_schema.validate(document) == document


@pytest.mark.parametrize("document", INVALID)
def test_the_schema_rejects_invalid_documents(document: Any) -> None:
    with pytest.raises(ProvenanceInvalid):
        provenance_schema.validate(document)


@pytest.mark.parametrize("document", INVALID)
def test_resolve_provenance_rejects_every_invalid_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, document: Any
) -> None:
    """Every reader, same answer. This one used to sanitize instead."""

    stamp = tmp_path / "_build_stamp.json"
    stamp.write_text(json.dumps(document))
    monkeypatch.setattr(run_builder, "_BUILD_STAMP", stamp)
    with pytest.raises(RuntimeError):
        run_builder.resolve_provenance()


@pytest.mark.parametrize("document", INVALID)
def test_stored_provenance_rejects_every_invalid_document(document: Any) -> None:
    with pytest.raises(ProvenanceUnreadable):
        validate_provenance(document)


@pytest.mark.parametrize("document", INVALID)
def test_the_verifier_rejects_every_invalid_stamp(document: Any) -> None:
    verify = _script("verify-sidecar-stamp.py").verify
    if not isinstance(document, dict):
        pytest.skip("the verifier rejects non-objects before calling verify()")
    official = document.get("distribution") == "release"
    assert verify(document, official=official)


def test_a_release_stamp_with_a_revision_is_rejected_not_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replaces the old behaviour, which dropped the revision and returned a
    clean release. Two producers disagreeing is not something to silently
    resolve in favour of the more permissive reading."""

    stamp = tmp_path / "_build_stamp.json"
    stamp.write_text(json.dumps({"distribution": "release", "revision": "a" * 40}))
    monkeypatch.setattr(run_builder, "_BUILD_STAMP", stamp)
    with pytest.raises(RuntimeError, match="release"):
        run_builder.resolve_provenance()


def test_a_source_stamp_with_dirty_true_string_cannot_become_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`dirty: "true"` is truthy to a lenient reader and not `True` to a strict
    one — the worst kind of value. It must never resolve to clean provenance."""

    stamp = tmp_path / "_build_stamp.json"
    stamp.write_text(
        json.dumps({"distribution": "source", "revision": "a" * 40, "dirty": "true"})
    )
    monkeypatch.setattr(run_builder, "_BUILD_STAMP", stamp)
    with pytest.raises(RuntimeError, match="dirty"):
        run_builder.resolve_provenance()

    # And it never reaches the per-run store as clean provenance either.
    with pytest.raises(ProvenanceUnreadable, match="dirty"):
        validate_provenance(
            {"distribution": "source", "revision": "a" * 40, "dirty": "true"}
        )
    # …nor past the verifier.
    assert _script("verify-sidecar-stamp.py").verify(
        {"distribution": "source", "revision": "a" * 40, "dirty": "true"},
        official=False,
    )


def test_the_verifier_rejects_unknown_fields() -> None:
    verify = _script("verify-sidecar-stamp.py").verify
    assert verify(
        {"distribution": "source", "revision": "a" * 40, "x": 1}, official=False
    )
    assert verify({"distribution": "release", "x": 1}, official=True)


def test_the_stamp_writer_only_produces_valid_documents() -> None:
    writer = _script("write-sidecar-stamp.py")
    for official, revision, dirty in [
        ("1", "a" * 40, "0"),
        ("0", "a" * 40, "0"),
        ("0", "a" * 40, "1"),
    ]:
        provenance_schema.validate(
            writer.validated(writer.build_stamp(official, revision, dirty))
        )


def test_a_valid_dirty_stamp_survives_resolution_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Carried, not dropped — the fix from the previous round, still holding."""

    document = {"distribution": "source", "revision": "b" * 40, "dirty": True}
    stamp = tmp_path / "_build_stamp.json"
    stamp.write_text(json.dumps(document))
    monkeypatch.setattr(run_builder, "_BUILD_STAMP", stamp)
    assert run_builder.resolve_provenance() == document


# ---------------------------------------------------------------------------
# 3. Git cleanliness that cannot be established
# ---------------------------------------------------------------------------


def _fake_git(directory: Path, *, status_rc: int, status_out: str = "") -> Path:
    """A git whose ``rev-parse`` succeeds and whose ``status`` behaves to order."""

    directory.mkdir(parents=True, exist_ok=True)
    git = directory / "git"
    git.write_text(
        "#!/bin/sh\n"
        'for arg in "$@"; do\n'
        '  case "$arg" in\n'
        "    rev-parse) echo 60be718538e44dd5497a75be419501080cbfff32; exit 0 ;;\n"
        f'    status) printf %s "{status_out}"; exit {status_rc} ;;\n'
        "  esac\n"
        "done\n"
        "exit 0\n"
    )
    git.chmod(0o755)
    # The directory, because it is what goes on PATH.
    return directory


def _run_stamp_step(tmp_path: Path, fake_git_dir: Path, official: str):
    """Execute the real step-6b block from build-sidecar.sh."""

    text = BUILD_SIDECAR.read_text()
    start = text.index('STAMP="$STAGE/site-packages/rapid_mlx/_build_stamp.json"')
    end = text.index("# Recompile so the stamped package")
    fragment = tmp_path / "step6b.sh"
    fragment.write_text(text[start:end])

    stage = tmp_path / "stage" / "rapid-mlx"
    (stage / "site-packages" / "rapid_mlx").mkdir(parents=True)
    env = {
        **os.environ,
        "PATH": f"{fake_git_dir}:{os.environ['PATH']}",
        "STAGE": str(stage),
        "RAPID_MLX_SOURCE": str(tmp_path),
        "REPO_ROOT": str(REPO / "apps/rapid-mac"),
        "RAPID_MLX_OFFICIAL_RELEASE": official,
    }
    result = subprocess.run(
        ["bash", "-c", f"set -euo pipefail; source {fragment}"],
        capture_output=True,
        text=True,
        env=env,
    )
    stamp_path = stage / "site-packages" / "rapid_mlx" / "_build_stamp.json"
    return result, stamp_path


def test_an_official_build_fails_when_git_status_fails(tmp_path: Path) -> None:
    """`rev-parse` succeeds, `status` fails. The old `|| true` read that as a
    clean tree and stamped an official release."""

    git_dir = _fake_git(tmp_path / "bin", status_rc=128, status_out="fatal: bad object")
    result, stamp_path = _run_stamp_step(tmp_path, git_dir, official="1")

    assert result.returncode != 0, result.stdout
    assert "could not determine whether" in result.stderr
    assert not stamp_path.exists(), "a release stamp was written despite unknown state"


def test_a_source_build_also_fails_when_git_status_fails(tmp_path: Path) -> None:
    git_dir = _fake_git(tmp_path / "bin", status_rc=128, status_out="fatal: bad object")
    result, stamp_path = _run_stamp_step(tmp_path, git_dir, official="0")
    assert result.returncode != 0
    assert not stamp_path.exists()


def test_a_clean_tree_with_a_working_git_stamps_a_release(tmp_path: Path) -> None:
    git_dir = _fake_git(tmp_path / "bin", status_rc=0, status_out="")
    result, stamp_path = _run_stamp_step(tmp_path, git_dir, official="1")
    assert result.returncode == 0, result.stderr
    assert json.loads(stamp_path.read_text()) == {"distribution": "release"}


def test_a_dirty_tree_still_blocks_an_official_build(tmp_path: Path) -> None:
    git_dir = _fake_git(
        tmp_path / "bin", status_rc=0, status_out="?? rapid_mlx/patch.py"
    )
    result, stamp_path = _run_stamp_step(tmp_path, git_dir, official="1")
    assert result.returncode != 0
    # The message wraps, so match a phrase that survives the line break.
    assert "refusing to build an OFFICIAL RELEASE from a modified" in result.stderr
    assert "?? rapid_mlx/patch.py" in result.stderr
    assert not stamp_path.exists()


def test_a_dirty_tree_stamps_a_source_build(tmp_path: Path) -> None:
    git_dir = _fake_git(
        tmp_path / "bin", status_rc=0, status_out="?? rapid_mlx/patch.py"
    )
    result, stamp_path = _run_stamp_step(tmp_path, git_dir, official="0")
    assert result.returncode == 0, result.stderr
    stamp = json.loads(stamp_path.read_text())
    assert stamp["distribution"] == "source"
    assert stamp["dirty"] is True
    provenance_schema.validate(stamp)


def test_the_build_script_no_longer_swallows_a_status_failure() -> None:
    text = BUILD_SIDECAR.read_text()
    code = [
        line
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert not any("status --porcelain" in line and "|| true" in line for line in code)
    assert "SIDECAR_STATUS_RC" in text


# ---------------------------------------------------------------------------
# The legacy policy, explicitly preserved
# ---------------------------------------------------------------------------


def test_an_absent_per_run_provenance_file_remains_a_legacy_result(
    tmp_path: Path,
) -> None:
    """Separate from a missing packaged build stamp, and deliberately kept.

    An old archived run has no provenance file because the feature did not
    exist; refusing it would retroactively block every result measured before
    today. A missing *build stamp* is different: that file is written by the
    build, so its absence means the build is broken.
    """

    archive = LocalRunArchive(tmp_path)
    assert archive.provenance("00000000-0000-4000-8000-000000000003") is None


# ---------------------------------------------------------------------------
# The writer's own argument boundary
# ---------------------------------------------------------------------------
#
# The writer validated only the document it had already built, which cannot
# catch an input that was silently coerced on the way in. `dirty="bogus"` made
# a perfectly valid *clean* source stamp, and the shared schema had nothing to
# object to. These exercise the raw arguments, not the finished document.


def _writer():
    return _script("write-sidecar-stamp.py")


@pytest.mark.parametrize(
    "official", ["true", "True", "TRUE", "yes", "y", "", " 1", "1 ", "01", "2", "-1"]
)
def test_the_writer_refuses_a_malformed_official_flag(official: str) -> None:
    """Anything but "0"/"1" used to fall through to the source branch, so a
    release lane spelling it "true" silently produced a source build."""

    with pytest.raises(SystemExit, match="official must be exactly"):
        _writer().build_stamp(official, "a" * 40, "0")


@pytest.mark.parametrize(
    "dirty", ["true", "True", "bogus", "", " 1", "1 ", "yes", "01", "2"]
)
def test_the_writer_refuses_a_malformed_dirty_flag(dirty: str) -> None:
    """The dangerous direction: every one of these was silently *clean*."""

    with pytest.raises(SystemExit, match="dirty must be exactly"):
        _writer().build_stamp("0", "a" * 40, dirty)


def test_a_malformed_dirty_flag_is_refused_on_the_release_lane_too() -> None:
    with pytest.raises(SystemExit, match="dirty must be exactly"):
        _writer().build_stamp("1", "a" * 40, "true")


@pytest.mark.parametrize(
    "revision",
    [
        "A" * 40,
        "AbC" + "d" * 37,
        " " + "a" * 40,
        "a" * 40 + " ",
        "\t" + "a" * 39,
        "a" * 40 + "\n",
        "a" * 39,
        "a" * 41,
        "z" * 40,
        "",
    ],
)
def test_the_writer_refuses_a_revision_it_would_have_had_to_repair(
    revision: str,
) -> None:
    """Uppercase and surrounding whitespace were silently normalised.

    A repaired revision is a value the caller never produced, written into a
    document that claims the caller produced it.
    """

    with pytest.raises(SystemExit, match="not exactly 40 lowercase hexadecimal"):
        _writer().build_stamp("0", revision, "0")


def test_the_writer_does_not_normalise_a_valid_revision() -> None:
    revision = "60be718538e44dd5497a75be419501080cbfff32"
    stamp = _writer().build_stamp("0", revision, "0")
    assert stamp == {"distribution": "source", "revision": revision}


def test_a_valid_release_input_produces_only_a_distribution() -> None:
    """A *valid* revision is required, and then deliberately not carried.

    This test used to pass ``""`` and assert success, which contradicted the
    rule it was meant to protect: the official branch returned its fixed
    document before looking at the revision at all, so any garbage was
    accepted. Validation and content are separate questions — the argument
    must be a real commit, and the document still must not contain it.
    """

    writer = _writer()
    for revision in [
        "a" * 40,
        "60be718538e44dd5497a75be419501080cbfff32",
        "0" * 40,
    ]:
        assert writer.build_stamp("1", revision, "0") == {"distribution": "release"}


#: Revisions no build may be stamped on, official or source.
BAD_REVISIONS = [
    pytest.param("", id="empty"),
    pytest.param("garbage", id="arbitrary"),
    pytest.param("not-a-sha-at-all", id="arbitrary-hyphenated"),
    pytest.param("A" * 40, id="uppercase"),
    pytest.param("60BE718538E44DD5497A75BE419501080CBFFF32", id="uppercase-real"),
    pytest.param("AbC" + "d" * 37, id="mixed-case"),
    pytest.param(" " + "a" * 40, id="leading-space"),
    pytest.param("a" * 40 + " ", id="trailing-space"),
    pytest.param("\t" + "a" * 39, id="leading-tab"),
    pytest.param("a" * 40 + "\n", id="trailing-newline"),
    pytest.param("a" * 39, id="too-short"),
    pytest.param("a" * 41, id="too-long"),
    pytest.param("z" * 40, id="non-hex"),
    pytest.param("60be718538e44dd5497a75be419501080cbfff3g", id="one-non-hex-char"),
    pytest.param("HEAD", id="a-ref-name"),
]


@pytest.mark.parametrize("revision", BAD_REVISIONS)
def test_an_official_build_refuses_every_bad_revision(revision: str) -> None:
    """The defect this closes: the official branch never looked.

    ``build_stamp("1", "garbage", "0")`` returned a clean release stamp, so a
    caller whose ``git rev-parse`` produced nonsense still shipped an official
    build — and the output looked perfect, because the output ignores the
    argument.
    """

    with pytest.raises(SystemExit, match="not exactly 40 lowercase hexadecimal"):
        _writer().build_stamp("1", revision, "0")


@pytest.mark.parametrize("revision", BAD_REVISIONS)
def test_a_source_build_refuses_every_bad_revision(revision: str) -> None:
    """The same rule, so the two branches cannot drift apart again."""

    with pytest.raises(SystemExit, match="not exactly 40 lowercase hexadecimal"):
        _writer().build_stamp("0", revision, "0")


def test_the_revision_is_validated_before_the_official_branch_is_chosen() -> None:
    """Order matters: validating inside each branch is what let one skip it."""

    writer = _writer()
    # A bad revision is refused for a dirty official build too — and with the
    # revision's message, because that check runs first.
    with pytest.raises(SystemExit, match="not exactly 40 lowercase hexadecimal"):
        writer.build_stamp("1", "garbage", "1")


def test_a_valid_clean_source_input_produces_no_dirty_key() -> None:
    stamp = _writer().build_stamp("0", "b" * 40, "0")
    assert stamp == {"distribution": "source", "revision": "b" * 40}
    assert "dirty" not in stamp


def test_a_valid_dirty_source_input_records_the_flag_as_a_boolean() -> None:
    stamp = _writer().build_stamp("0", "c" * 40, "1")
    assert stamp == {"distribution": "source", "revision": "c" * 40, "dirty": True}
    assert stamp["dirty"] is True


def test_an_official_dirty_build_is_still_refused() -> None:
    with pytest.raises(SystemExit, match="modified working tree"):
        _writer().build_stamp("1", "a" * 40, "1")


def test_the_writer_main_refuses_malformed_arguments_and_writes_nothing(
    tmp_path: Path,
) -> None:
    """Through ``main``, because that is what the build script calls."""

    writer = _writer()
    target = tmp_path / "_build_stamp.json"
    for argv in [
        ["w", str(target), "true", "a" * 40, "0"],
        ["w", str(target), "0", "a" * 40, "bogus"],
        ["w", str(target), "0", "A" * 40, "0"],
    ]:
        with pytest.raises(SystemExit):
            writer.main(argv)
        assert not target.exists(), f"a stamp was written for {argv[2:]}"


@pytest.mark.parametrize("revision", BAD_REVISIONS)
def test_main_writes_no_file_for_a_malformed_official_revision(
    tmp_path: Path, revision: str
) -> None:
    """The end-to-end shape of the defect: `main` produced a release stamp on
    disk for any revision at all, so the file that gates the whole provenance
    chain was written from an argument nobody had checked."""

    writer = _writer()
    target = tmp_path / "_build_stamp.json"
    with pytest.raises(SystemExit, match="not exactly 40 lowercase hexadecimal"):
        writer.main(["w", str(target), "1", revision, "0"])
    assert not target.exists(), f"an official stamp was written for {revision!r}"


@pytest.mark.parametrize("revision", BAD_REVISIONS)
def test_main_writes_no_file_for_a_malformed_source_revision(
    tmp_path: Path, revision: str
) -> None:
    writer = _writer()
    target = tmp_path / "_build_stamp.json"
    with pytest.raises(SystemExit, match="not exactly 40 lowercase hexadecimal"):
        writer.main(["w", str(target), "0", revision, "1"])
    assert not target.exists()


def test_main_writes_an_official_stamp_for_a_valid_revision(tmp_path: Path) -> None:
    """And the valid case still produces exactly the release document."""

    writer = _writer()
    target = tmp_path / "_build_stamp.json"
    revision = "60be718538e44dd5497a75be419501080cbfff32"
    assert writer.main(["w", str(target), "1", revision, "0"]) == 0
    written = json.loads(target.read_text())
    assert written == {"distribution": "release"}
    # The validated revision is required as input and absent from the output.
    assert revision not in target.read_text()
    provenance_schema.validate(written)


def test_main_does_not_overwrite_a_good_stamp_with_a_bad_call(
    tmp_path: Path,
) -> None:
    """A second, malformed invocation must not damage what is already there."""

    writer = _writer()
    target = tmp_path / "_build_stamp.json"
    writer.main(["w", str(target), "1", "a" * 40, "0"])
    before = target.read_text()
    with pytest.raises(SystemExit):
        writer.main(["w", str(target), "1", "garbage", "0"])
    assert target.read_text() == before


def test_the_writer_main_writes_a_valid_document(tmp_path: Path) -> None:
    writer = _writer()
    target = tmp_path / "_build_stamp.json"
    assert writer.main(["w", str(target), "0", "d" * 40, "1"]) == 0
    written = json.loads(target.read_text())
    assert written == {"distribution": "source", "revision": "d" * 40, "dirty": True}
    # And the readers accept exactly what was written.
    provenance_schema.validate(written)


def test_the_writer_main_rejects_the_wrong_argument_count(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="usage:"):
        _writer().main(["w", str(tmp_path / "s.json"), "0", "a" * 40])


def test_the_build_script_passes_zero_or_one_for_both_flags() -> None:
    """The writer is now strict, so the caller must be exact."""

    text = BUILD_SIDECAR.read_text()
    assert 'OFFICIAL_RELEASE="${RAPID_MLX_OFFICIAL_RELEASE:-0}"' in text
    assert "SIDECAR_DIRTY=0" in text
    assert "SIDECAR_DIRTY=1" in text
    # The invocation passes the two flags positionally.
    assert '"$STAMP" "$OFFICIAL_RELEASE" "$SIDECAR_REVISION" "$SIDECAR_DIRTY"' in text


def test_the_build_script_validates_the_official_flag_before_branching(
    tmp_path: Path,
) -> None:
    """Otherwise it announces "source", builds, and only then is refused."""

    git_dir = _fake_git(tmp_path / "bin", status_rc=0, status_out="")
    result, stamp_path = _run_stamp_step(tmp_path, git_dir, official="true")
    assert result.returncode != 0
    assert "must be exactly 0 or 1" in result.stderr
    assert not stamp_path.exists()
    # And it does not first claim to be building a source stamp.
    assert "distribution: source" not in result.stdout


def test_the_build_script_rejects_a_malformed_rev_parse_answer(tmp_path: Path) -> None:
    """A `git` whose rev-parse answers with something that is not a sha.

    The writer is the authoritative boundary, but failing here names the cause
    — an unusable `rev-parse` — instead of surfacing as a stamp-writer error
    three steps later.
    """

    directory = tmp_path / "bin"
    directory.mkdir(parents=True)
    git = directory / "git"
    git.write_text(
        "#!/bin/sh\n"
        'for arg in "$@"; do\n'
        '  case "$arg" in\n'
        "    rev-parse) echo 'ref: refs/heads/main'; exit 0 ;;\n"
        "    status) exit 0 ;;\n"
        "  esac\n"
        "done\n"
        "exit 0\n"
    )
    git.chmod(0o755)

    result, stamp_path = _run_stamp_step(tmp_path, directory, official="1")
    assert result.returncode != 0, result.stdout
    assert "not a 40-character lowercase" in result.stderr
    assert not stamp_path.exists()


def test_the_build_script_accepts_a_real_sha_from_rev_parse(tmp_path: Path) -> None:
    git_dir = _fake_git(tmp_path / "bin", status_rc=0, status_out="")
    result, stamp_path = _run_stamp_step(tmp_path, git_dir, official="1")
    assert result.returncode == 0, result.stderr
    assert json.loads(stamp_path.read_text()) == {"distribution": "release"}
