# SPDX-License-Identifier: Apache-2.0
"""Provenance must be settled before a benchmark spends minutes measuring.

`execution_config` used to probe Git only while assembling the finished
record, so a Mac whose `git` could not answer lost a completed benchmark to:

    benchmark completed but result could not be constructed:
    could not resolve the Rapid-MLX source revision

Two things were wrong. The probe ran at the worst possible moment, and it ran
at all for a packaged app that has no checkout to ask about.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from rapid_mlx.community_bench import local_runner, run_builder


@pytest.fixture(autouse=True)
def _forget_cached_provenance() -> None:
    """Each test decides provenance for itself."""

    run_builder._reset_provenance_cache()
    yield
    run_builder._reset_provenance_cache()


# ---------------------------------------------------------------------------
# A packaged release never probes Git
# ---------------------------------------------------------------------------


def _packaged_sidecar(root: Path) -> Path:
    """The layout `build-sidecar-tarball.sh` produces and `build.sh` copies
    into `Rapid.app/Contents/Resources/rapid-mlx/`."""

    package = root / "rapid-mlx" / "site-packages" / "rapid_mlx" / "community_bench"
    package.mkdir(parents=True)
    (root / "rapid-mlx" / "VERSION").write_text("0.13.4\n")
    module = package / "run_builder.py"
    module.write_text("")
    return module


def test_packaged_sidecar_is_a_release_without_touching_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _packaged_sidecar(tmp_path)

    def explode(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("a packaged sidecar probed Git")

    monkeypatch.setattr(run_builder.subprocess, "run", explode)
    assert run_builder._source_checkout_revision(module) is None


def test_packaged_sidecar_inside_a_git_checkout_is_still_a_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Users really do drop Rapid.app into a checkout, and the dev build stages
    # the sidecar inside this repository.
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    module = _packaged_sidecar(tmp_path)

    def explode(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("a packaged sidecar probed Git")

    monkeypatch.setattr(run_builder.subprocess, "run", explode)
    assert run_builder._source_checkout_revision(module) is None


def test_module_inside_an_app_bundle_is_a_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "Rapid.app" / "Contents" / "Resources" / "lib" / "rapid_mlx"
    package.mkdir(parents=True)
    module = package / "run_builder.py"
    module.write_text("")

    def explode(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("a bundled runtime probed Git")

    monkeypatch.setattr(run_builder.subprocess, "run", explode)
    assert run_builder._source_checkout_revision(module) is None


def test_an_unstamped_packaged_runtime_refuses_rather_than_claiming_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This test used to assert the opposite, and the opposite was the bug.

    ``_source_checkout_revision`` returns ``None`` for a packaged layout by
    design, so "no stamp and no revision" fell through to
    ``distribution: release``. A packaging omission therefore produced an
    official-release claim from an app that could not say what built it.
    """

    module = _packaged_sidecar(tmp_path)
    monkeypatch.setattr(run_builder, "_BUILD_STAMP", tmp_path / "absent.json")
    monkeypatch.setattr(run_builder, "__file__", str(module))
    # PATH as Finder gives it, and no usable git anywhere on it.
    monkeypatch.setenv("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")

    def no_git(*args, **kwargs):
        raise OSError(2, "No such file or directory: 'git'")

    monkeypatch.setattr(run_builder.subprocess, "run", no_git)

    with pytest.raises(RuntimeError, match="missing its build provenance stamp"):
        run_builder.resolve_provenance()


def test_a_stamped_packaged_release_survives_a_finder_path_with_no_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The case that must keep working: a real release, launched from Finder
    on a Mac with no command line tools."""

    module = _packaged_sidecar(tmp_path)
    stamp = tmp_path / "_build_stamp.json"
    stamp.write_text(json.dumps({"distribution": "release"}))
    monkeypatch.setattr(run_builder, "_BUILD_STAMP", stamp)
    monkeypatch.setattr(run_builder, "__file__", str(module))
    monkeypatch.setenv("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")

    def no_git(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("a stamped packaged release probed Git")

    monkeypatch.setattr(run_builder.subprocess, "run", no_git)

    config = run_builder.execution_config("text_generation")
    assert config["runtime"]["distribution"] == "release"
    assert "rapid_mlx_revision" not in config["runtime"]


# ---------------------------------------------------------------------------
# A build stamp is consumed instead of probing
# ---------------------------------------------------------------------------


def test_an_official_release_stamp_is_a_release_without_probing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stamp = tmp_path / "_build_stamp.json"
    stamp.write_text(json.dumps({"distribution": "release"}))
    monkeypatch.setattr(run_builder, "_BUILD_STAMP", stamp)

    def explode(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("a stamped build probed Git")

    monkeypatch.setattr(run_builder.subprocess, "run", explode)

    assert run_builder.resolve_provenance() == {"distribution": "release"}
    runtime = run_builder.execution_config("text_generation")["runtime"]
    # `execution-config.schema.json` FORBIDS `rapid_mlx_revision` on a release.
    assert runtime["distribution"] == "release"
    assert "rapid_mlx_revision" not in runtime


def test_a_locally_packaged_source_build_stays_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The blocker: `bash scripts/build.sh` on a branch used to stamp
    `release`, so a developer's numbers were published as an official build
    and the commit that produced them was dropped."""

    revision = "a" * 40
    stamp = tmp_path / "_build_stamp.json"
    stamp.write_text(json.dumps({"distribution": "source", "revision": revision}))
    monkeypatch.setattr(run_builder, "_BUILD_STAMP", stamp)

    def explode(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("a stamped build probed Git")

    monkeypatch.setattr(run_builder.subprocess, "run", explode)

    assert run_builder.resolve_provenance() == {
        "distribution": "source",
        "revision": revision,
    }
    runtime = run_builder.execution_config("text_generation")["runtime"]
    assert runtime["distribution"] == "source"
    # A source runtime REQUIRES the revision, and it comes from the stamp —
    # no `git` is invoked, which is the whole point of stamping.
    assert runtime["rapid_mlx_revision"] == revision


def test_a_dirty_source_stamp_still_names_its_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision = "b" * 40
    stamp = tmp_path / "_build_stamp.json"
    stamp.write_text(
        json.dumps({"distribution": "source", "revision": revision, "dirty": True})
    )
    monkeypatch.setattr(run_builder, "_BUILD_STAMP", stamp)
    provenance = run_builder.resolve_provenance()
    assert provenance["distribution"] == "source"
    assert provenance["revision"] == revision


def test_a_source_stamp_without_a_revision_fails_before_measuring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refused, not downgraded. Calling it a release is the failure mode this
    whole change exists to remove."""

    stamp = tmp_path / "_build_stamp.json"
    stamp.write_text(json.dumps({"distribution": "source"}))
    monkeypatch.setattr(run_builder, "_BUILD_STAMP", stamp)
    with pytest.raises(RuntimeError, match="missing|source build"):
        run_builder.resolve_provenance()


@pytest.mark.parametrize(
    "revision",
    [
        "",
        "abc",
        "z" * 40,
        "a" * 39,
        "a" * 41,
        None,
        12345,
        "  " + "a" * 38,
        # Uppercase too: the schema is closed on the exact form, and quietly
        # lowercasing would repair a document its producer never wrote.
        "A" * 40,
        "AbC" + "d" * 37,
    ],
)
def test_a_source_stamp_with_a_bad_revision_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, revision: object
) -> None:
    stamp = tmp_path / "_build_stamp.json"
    stamp.write_text(json.dumps({"distribution": "source", "revision": revision}))
    monkeypatch.setattr(run_builder, "_BUILD_STAMP", stamp)
    with pytest.raises(RuntimeError):
        run_builder.resolve_provenance()


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (None, None),
        (123, None),
        ("abc", None),
        ("z" * 40, None),
        ("A" * 40, "a" * 40),
        ("  " + "b" * 40 + "\n", "b" * 40),
    ],
)
def test_revision_normalizer_accepts_only_a_full_hex_sha(
    candidate: object, expected: str | None
) -> None:
    assert run_builder._valid_revision(candidate) == expected


def test_an_unknown_distribution_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stamp = tmp_path / "_build_stamp.json"
    stamp.write_text(json.dumps({"distribution": "nightly"}))
    monkeypatch.setattr(run_builder, "_BUILD_STAMP", stamp)
    with pytest.raises(RuntimeError, match="unreadable"):
        run_builder.resolve_provenance()


def test_a_malformed_stamp_is_refused_rather_than_guessed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stamp that exists but cannot be read is not the same as no stamp.

    Something wrote it and we cannot tell what it says. Defaulting to
    "release" would be a guess about provenance, which is the one thing this
    file must never do.
    """

    stamp = tmp_path / "_build_stamp.json"
    stamp.write_text("{not json")
    monkeypatch.setattr(run_builder, "_BUILD_STAMP", stamp)
    with pytest.raises(RuntimeError, match="unreadable"):
        run_builder.resolve_provenance()


def test_a_stamp_that_is_not_an_object_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stamp = tmp_path / "_build_stamp.json"
    stamp.write_text("[1, 2, 3]")
    monkeypatch.setattr(run_builder, "_BUILD_STAMP", stamp)
    with pytest.raises(RuntimeError, match="unreadable"):
        run_builder.resolve_provenance()


def test_no_stamp_at_all_falls_back_to_the_checkout_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unpackaged runtime — running from a checkout — is the case the
    probe exists for, and a missing stamp must not be an error."""

    monkeypatch.setattr(run_builder, "_BUILD_STAMP", tmp_path / "absent.json")
    monkeypatch.setattr(run_builder, "_source_checkout_revision", lambda *_: None)
    assert run_builder.resolve_provenance() == {"distribution": "release"}


# ---------------------------------------------------------------------------
# A real source checkout is never silently downgraded
# ---------------------------------------------------------------------------


def test_a_tracked_source_checkout_still_reports_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision = "b" * 40
    responses = iter(
        [
            SimpleNamespace(returncode=0, stdout="", stderr=""),  # ls-files
            SimpleNamespace(returncode=0, stdout=f"{revision}\n", stderr=""),
        ]
    )
    (tmp_path / ".git").mkdir()
    module = tmp_path / "rapid_mlx" / "run_builder.py"
    module.parent.mkdir()
    module.write_text("")
    monkeypatch.setattr(run_builder.subprocess, "run", lambda *a, **k: next(responses))
    assert run_builder._source_checkout_revision(module) == revision


def test_an_unresolvable_source_checkout_is_an_error_not_a_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Calling a developer's working tree a "release" would attribute their
    build to the published runtime, so this must stay a hard failure."""

    (tmp_path / ".git").mkdir()
    module = tmp_path / "rapid_mlx" / "run_builder.py"
    module.parent.mkdir()
    module.write_text("")

    def no_git(*args, **kwargs):
        raise OSError(2, "No such file or directory: 'git'")

    monkeypatch.setattr(run_builder.subprocess, "run", no_git)
    with pytest.raises(RuntimeError, match="could not resolve"):
        run_builder._source_checkout_revision(module)


def test_the_error_names_the_cause_and_the_fix() -> None:
    assert "Git checkout" in run_builder._UNRESOLVED_REVISION
    assert "xcode-select --install" in run_builder._UNRESOLVED_REVISION


# ---------------------------------------------------------------------------
# Resolution happens BEFORE the measurement
# ---------------------------------------------------------------------------


def test_unresolved_provenance_fails_before_any_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    measured = []

    def never(*args, **kwargs):  # pragma: no cover - must not be reached
        measured.append(True)
        raise AssertionError("measurement started despite unresolved provenance")

    monkeypatch.setattr(local_runner, "_run_local_measured", never)
    monkeypatch.setattr(
        local_runner,
        "plan_for_alias",
        lambda *a, **k: {
            "model": {
                "repo_id": "mlx-community/Qwen3.5-9B-4bit",
                "task_type": "text_generation",
            }
        },
    )
    monkeypatch.setattr(
        local_runner,
        "resolve_provenance",
        lambda: (_ for _ in ()).throw(RuntimeError(run_builder._UNRESOLVED_REVISION)),
    )

    with pytest.raises(RuntimeError, match="could not resolve"):
        local_runner.run_local("qwen3.5-9b-4bit", archive=object())
    assert measured == [], "the benchmark ran before provenance was settled"


def test_provenance_is_resolved_once_per_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(run_builder, "_BUILD_STAMP", tmp_path / "absent.json")

    def probe(*_args, **_kwargs):
        calls.append(True)
        return None

    monkeypatch.setattr(run_builder, "_source_checkout_revision", probe)
    run_builder.resolve_provenance()
    run_builder.resolve_provenance()
    run_builder.execution_config("text_generation")
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# The post-measurement error says whether anything was saved
# ---------------------------------------------------------------------------


def test_assembly_failure_says_nothing_was_saved() -> None:
    source = Path(local_runner.__file__).read_text()
    assert "NOTHING was saved to this Mac" in source
    # And the old ambiguous wording is gone.
    assert "result could not be constructed" not in source


def test_save_failure_already_distinguishes_itself() -> None:
    source = Path(local_runner.__file__).read_text()
    assert "result could not be saved" in source


# ---------------------------------------------------------------------------
# The schema is the authority on which fields may coexist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stamp",
    [
        None,
        {"distribution": "release"},
        {"distribution": "source", "revision": "c" * 40},
    ],
)
def test_execution_config_validates_against_the_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stamp: dict | None
) -> None:
    """Every provenance shape must produce a document the validator accepts.

    `runtime.distribution: source` requires `rapid_mlx_revision`; `release`
    forbids it. Asserting the field directly is not enough — the rule is
    conditional in both directions, and getting it backwards only surfaced
    when a real packaged benchmark finished and was then thrown away.
    """

    from rapid_mlx.catalog.validation import ContractValidator

    if stamp is None:
        monkeypatch.setattr(run_builder, "_BUILD_STAMP", tmp_path / "absent.json")
        monkeypatch.setattr(run_builder, "_source_checkout_revision", lambda *_: None)
    else:
        path = tmp_path / "_build_stamp.json"
        path.write_text(json.dumps(stamp))
        monkeypatch.setattr(run_builder, "_BUILD_STAMP", path)

    config = run_builder.execution_config("text_generation")
    ContractValidator().validate("execution_config", config)


def test_a_source_runtime_carries_its_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rapid_mlx.catalog.validation import ContractValidator

    revision = "d" * 40
    monkeypatch.setattr(run_builder, "_BUILD_STAMP", tmp_path / "absent.json")
    monkeypatch.setattr(run_builder, "_source_checkout_revision", lambda *_: revision)

    config = run_builder.execution_config("text_generation")
    assert config["runtime"]["distribution"] == "source"
    assert config["runtime"]["rapid_mlx_revision"] == revision
    ContractValidator().validate("execution_config", config)


# ---------------------------------------------------------------------------
# The stamp writer the build script calls
# ---------------------------------------------------------------------------


def _stamp_writer():
    """Import ``apps/rapid-mac/scripts/write-sidecar-stamp.py`` by path."""

    import importlib.util

    path = (
        Path(__file__).resolve().parents[1]
        / "apps/rapid-mac/scripts/write-sidecar-stamp.py"
    )
    spec = importlib.util.spec_from_file_location("write_sidecar_stamp", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_the_build_script_stamps_a_local_build_as_source() -> None:
    writer = _stamp_writer()
    revision = "c" * 40
    assert writer.build_stamp("0", revision, "0") == {
        "distribution": "source",
        "revision": revision,
    }


def test_the_build_script_records_a_dirty_tree() -> None:
    writer = _stamp_writer()
    stamp = writer.build_stamp("0", "d" * 40, "1")
    assert stamp["distribution"] == "source"
    assert stamp["dirty"] is True


def test_only_an_explicit_official_release_is_stamped_release() -> None:
    writer = _stamp_writer()
    # A real revision is required even for a release — the argument is
    # validated, then deliberately not carried, because the contract forbids
    # `rapid_mlx_revision` on a release.
    assert writer.build_stamp("1", "e" * 40, "0") == {"distribution": "release"}
    assert "revision" not in writer.build_stamp("1", "e" * 40, "0")
    # An empty revision is not a release shortcut; it is a caller that could
    # not resolve a commit.
    with pytest.raises(SystemExit, match="not exactly 40 lowercase hexadecimal"):
        writer.build_stamp("1", "", "0")


def test_the_build_script_refuses_a_source_build_with_no_revision() -> None:
    writer = _stamp_writer()
    with pytest.raises(SystemExit, match="not exactly 40 lowercase hexadecimal"):
        writer.build_stamp("0", "", "0")


def test_the_build_scripts_stamp_round_trips_through_resolve_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the build writes is what the runtime reads."""

    writer = _stamp_writer()
    revision = "f" * 40
    stamp = tmp_path / "_build_stamp.json"
    stamp.write_text(json.dumps(writer.build_stamp("0", revision, "0")))
    monkeypatch.setattr(run_builder, "_BUILD_STAMP", stamp)
    assert run_builder.resolve_provenance() == {
        "distribution": "source",
        "revision": revision,
    }


# ---------------------------------------------------------------------------
# A Finder-like PATH with no git, for both build kinds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stamp", "expected_distribution", "expects_revision"),
    [
        ({"distribution": "release"}, "release", False),
        ({"distribution": "source", "revision": "a" * 40}, "source", True),
    ],
)
def test_a_packaged_build_never_needs_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stamp: dict,
    expected_distribution: str,
    expects_revision: bool,
) -> None:
    """A Finder-launched app inherits a bare PATH, and a Mac without the
    command line tools cannot run `git` at all. Neither build kind may care."""

    from rapid_mlx.catalog.validation import ContractValidator

    path = tmp_path / "_build_stamp.json"
    path.write_text(json.dumps(stamp))
    monkeypatch.setattr(run_builder, "_BUILD_STAMP", path)
    monkeypatch.setenv("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")

    def no_git(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("a packaged build invoked git at run time")

    monkeypatch.setattr(run_builder.subprocess, "run", no_git)

    config = run_builder.execution_config("text_generation")
    runtime = config["runtime"]
    assert runtime["distribution"] == expected_distribution
    assert ("rapid_mlx_revision" in runtime) is expects_revision
    # And the document the service will be asked to accept is valid either way.
    ContractValidator().validate("execution_config", config)
