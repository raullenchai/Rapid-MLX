# SPDX-License-Identifier: Apache-2.0
"""Contract between the vendored LTX-2.5 runtime and rapid-mlx's pins.

The runtime is copied verbatim from a pinned upstream archive by
``scripts/vendor_ltx25.py``. These tests hold four things together:

* the upstream dependency floors and rapid-mlx's own pins still leave a
  non-empty constraint intersection (a floor bump must not make the two
  requirement sets mutually unsatisfiable; this is a specifier check, not a
  resolver run);
* the vendoring script's constants match the runtime adapter's constants;
* the checked-in tree still matches the pin (``--check``), so a hand edit
  cannot drift silently;
* ``--update`` reproduces the same layout from a verified archive.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import re
import shutil
import sys
import tarfile
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

from rapid_mlx.video import ltx25

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_SCRIPT = REPO_ROOT / "scripts" / "vendor_ltx25.py"

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on the 3.10 floor
    import tomli as tomllib

# Floors recorded from the pinned upstream source, commit
# 57952288076766abe27dda3a774b2c24f7346977:
#   packages/ltx-core-mlx/pyproject.toml    -> mlx, mlx-arsenal, mlx-lm,
#                                              safetensors, huggingface-hub, numpy
#   packages/ltx-pipelines-mlx/pyproject.toml -> mlx-arsenal, pillow, tqdm
# ``ltx-core-mlx`` itself is vendored here, so it is not a rapid-mlx pin.
UPSTREAM_REQUIREMENTS = {
    "mlx": ">=0.31.0",
    "mlx-arsenal": ">=0.2.4",
    "mlx-lm": ">=0.31.3",
    "safetensors": ">=0.4.0",
    "huggingface-hub": ">=0.26.0",
    "numpy": ">=1.26.0",
    "pillow": ">=10.4.0",
    "tqdm": ">=4.66.0",
}
UPSTREAM_REQUIRES_PYTHON = ">=3.11"


def _load_vendor_script():
    spec = importlib.util.spec_from_file_location("vendor_ltx25", VENDOR_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _rapid_pins() -> dict[str, SpecifierSet]:
    pins: dict[str, SpecifierSet] = {}
    for raw in _pyproject()["project"]["dependencies"]:
        requirement = Requirement(raw)
        pins[canonicalize_name(requirement.name)] = requirement.specifier
    return pins


def _candidate_versions(*specifiers: SpecifierSet) -> list[Version]:
    """Versions worth probing for a non-empty intersection."""
    candidates: set[Version] = set()
    for specifier in specifiers:
        for item in specifier:
            if item.operator in (">=", "=="):
                try:
                    candidates.add(Version(item.version))
                except ValueError:
                    continue
    candidates.add(Version("1.0"))
    return sorted(candidates)


def _intersection_is_nonempty(left: SpecifierSet, right: SpecifierSet) -> bool:
    combined = left & right
    return any(
        combined.contains(version) for version in _candidate_versions(left, right)
    )


def test_upstream_floors_and_rapid_mlx_pins_intersect() -> None:
    """The two requirement sets must share at least one viable version.

    This is a constraint-intersection check over specifier sets: it proves
    rapid-mlx's pins do not contradict an upstream floor. It does not run the
    resolver, prove the locked environment is installable, or verify the
    versions actually installed in any lane.
    """
    pins = _rapid_pins()
    missing = sorted(set(UPSTREAM_REQUIREMENTS) - set(pins))
    assert not missing, f"rapid-mlx no longer declares {missing}"

    unsatisfiable = []
    for name, floor in UPSTREAM_REQUIREMENTS.items():
        if not _intersection_is_nonempty(pins[name], SpecifierSet(floor)):
            unsatisfiable.append(
                f"{name}: rapid-mlx {pins[name]} has no version also satisfying {floor}"
            )
    assert not unsatisfiable, "\n".join(unsatisfiable)


def test_safetensors_is_declared_in_base_and_ci_linux() -> None:
    """The runtime imports safetensors directly, so both lanes declare it."""
    project = _pyproject()
    base = {
        canonicalize_name(Requirement(raw).name)
        for raw in project["project"]["dependencies"]
    }
    ci_linux = {
        canonicalize_name(Requirement(raw).name)
        for raw in project["project"]["optional-dependencies"]["ci-linux"]
    }
    assert "safetensors" in base
    assert "safetensors" in ci_linux


def test_python_floor_intersects_upstream() -> None:
    """The runtime needs 3.11+; rapid-mlx's floor must not exclude it."""
    requires_python = SpecifierSet(_pyproject()["project"]["requires-python"])
    assert _intersection_is_nonempty(
        requires_python, SpecifierSet(UPSTREAM_REQUIRES_PYTHON)
    )


def test_vendor_script_constants_match_runtime_adapter() -> None:
    vendor = _load_vendor_script()
    assert vendor.LTX25_RUNTIME_COMMIT == ltx25.LTX25_RUNTIME_COMMIT
    assert vendor.LTX25_RUNTIME_REPOSITORY == ltx25.LTX25_RUNTIME_REPOSITORY
    assert vendor.LTX25_RUNTIME_VERSION == ltx25.LTX25_RUNTIME_VERSION
    assert vendor.PROVENANCE_FILENAME == ltx25.LTX25_PROVENANCE_FILE
    assert tuple(vendor.LTX25_PACKAGES) == ltx25.LTX25_VENDORED_PACKAGES
    assert ltx25.LTX25_RUNTIME_COMMIT in vendor.LTX25_RUNTIME_ARCHIVE_URL
    assert ltx25.LTX25_RUNTIME_COMMIT in vendor.LTX25_ARCHIVE_ROOT
    assert re.fullmatch(r"[0-9a-f]{64}", vendor.LTX25_RUNTIME_SHA256)


def test_vendored_tree_matches_the_pin() -> None:
    vendor = _load_vendor_script()
    assert vendor.check(REPO_ROOT) == []
    assert vendor.main(["--check", "--root", str(REPO_ROOT)]) == 0


def test_vendored_provenance_licenses_and_digests() -> None:
    vendor = _load_vendor_script()
    for package in ltx25.LTX25_VENDORED_PACKAGES:
        directory = REPO_ROOT / package
        provenance = vendor.read_provenance(directory)
        assert provenance is not None, f"{package} has no provenance"
        assert provenance["REPOSITORY"] == ltx25.LTX25_RUNTIME_REPOSITORY
        assert provenance["COMMIT"] == ltx25.LTX25_RUNTIME_COMMIT
        assert provenance["VERSION"] == ltx25.LTX25_RUNTIME_VERSION
        assert provenance["PACKAGE"] == package
        assert provenance["TREE_SHA256"] == vendor.tree_digest(directory)
        # The runtime adapter must compute the exact same digest from the same
        # file set and ordering, or a trusted install would be rejected.
        assert ltx25._tree_digest(directory) == provenance["TREE_SHA256"]

        license_text = (directory / "LICENSE").read_text(encoding="utf-8")
        assert "MIT License" in license_text
        assert "dgrauet" in license_text

        notice = (directory / "NOTICE").read_text(encoding="utf-8")
        assert package in notice
        assert ltx25.LTX25_RUNTIME_COMMIT in notice
        assert vendor.LTX25_RUNTIME_SHA256 in notice
        assert "unmodified" in notice


def test_tree_digest_ignores_generated_metadata_and_caches(tmp_path: Path) -> None:
    vendor = _load_vendor_script()
    package = tmp_path / "ltx_core_mlx"
    package.mkdir()
    (package / "module.py").write_text("VALUE = 1\n")
    (package / "_vendored.py").write_text("COMMIT = 'x'\n")
    (package / "LICENSE").write_text("license\n")
    (package / "NOTICE").write_text("notice\n")
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "module.cpython-312.pyc").write_bytes(b"cache")
    (package / "stale.pyc").write_bytes(b"cache")
    digest = vendor.tree_digest(package)

    (package / "_vendored.py").write_text("COMMIT = 'y'\n")
    (package / "LICENSE").write_text("changed\n")
    (package / "NOTICE").write_text("changed\n")
    assert vendor.tree_digest(package) == digest

    (package / "module.py").write_text("VALUE = 2\n")
    assert vendor.tree_digest(package) != digest


def _copy_vendored_tree(destination: Path) -> Path:
    for package in ltx25.LTX25_VENDORED_PACKAGES:
        shutil.copytree(REPO_ROOT / package, destination / package)
    return destination


def test_check_rejects_drifted_source_file(tmp_path: Path) -> None:
    vendor = _load_vendor_script()
    root = _copy_vendored_tree(tmp_path / "root")
    target = root / "ltx_core_mlx" / "__init__.py"
    target.write_text(target.read_text(encoding="utf-8") + "\n# drift\n")

    problems = vendor.check(root)
    assert any("digest changed" in problem for problem in problems)
    assert vendor.main(["--check", "--root", str(root)]) == 1


def test_check_reports_missing_metadata(tmp_path: Path) -> None:
    vendor = _load_vendor_script()
    root = _copy_vendored_tree(tmp_path / "root")
    (root / "ltx_core_mlx" / "_vendored.py").unlink()
    (root / "ltx_pipelines_mlx" / "LICENSE").unlink()
    (root / "ltx_pipelines_mlx" / "NOTICE").unlink()

    problems = vendor.check(root)
    assert any("missing or invalid" in problem for problem in problems)
    assert any("LICENSE is missing" in problem for problem in problems)
    assert any("NOTICE is missing" in problem for problem in problems)


def _synthetic_archive(tmp_path: Path) -> Path:
    """Build a tarball with the pinned archive's root and package layout."""
    vendor = _load_vendor_script()
    source = tmp_path / "source"
    root = source / vendor.LTX25_ARCHIVE_ROOT
    (root / "packages" / "ltx-core-mlx" / "src" / "ltx_core_mlx").mkdir(parents=True)
    (root / "packages" / "ltx-pipelines-mlx" / "src" / "ltx_pipelines_mlx").mkdir(
        parents=True
    )
    (root / "LICENSE").write_text("MIT License\nCopyright (c) 2025 dgrauet\n")
    (
        root / "packages" / "ltx-core-mlx" / "src" / "ltx_core_mlx" / "__init__.py"
    ).write_text("")
    (
        root / "packages" / "ltx-core-mlx" / "src" / "ltx_core_mlx" / "core.py"
    ).write_text("VALUE = 1\n")
    (
        root
        / "packages"
        / "ltx-pipelines-mlx"
        / "src"
        / "ltx_pipelines_mlx"
        / "__init__.py"
    ).write_text("")
    (
        root / "packages" / "ltx-pipelines-mlx" / "src" / "ltx_pipelines_mlx" / "cli.py"
    ).write_text("def main():\n    return 0\n")

    archive = tmp_path / "ltx-2-mlx.tar.gz"
    with tarfile.open(archive, "w:gz") as tarball:
        tarball.add(root, arcname=root.name)
    return archive


def test_update_reproduces_the_pinned_layout(tmp_path: Path, monkeypatch) -> None:
    vendor = _load_vendor_script()
    archive = _synthetic_archive(tmp_path)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setattr(vendor, "LTX25_RUNTIME_SHA256", digest)

    root = tmp_path / "root"
    assert (
        vendor.main(["--update", "--root", str(root), "--tarball", str(archive)]) == 0
    )

    assert vendor.check(root) == []
    for package in ltx25.LTX25_VENDORED_PACKAGES:
        directory = root / package
        assert (directory / "_vendored.py").is_file()
        assert (directory / "LICENSE").is_file()
        assert (directory / "NOTICE").is_file()
        provenance = vendor.read_provenance(directory)
        assert provenance is not None
        assert provenance["COMMIT"] == ltx25.LTX25_RUNTIME_COMMIT
        assert provenance["PACKAGE"] == package
        assert provenance["TREE_SHA256"] == vendor.tree_digest(directory)
    assert (root / "ltx_pipelines_mlx" / "cli.py").is_file()


def test_update_rejects_a_tarball_with_the_wrong_digest(
    tmp_path: Path, monkeypatch
) -> None:
    vendor = _load_vendor_script()
    archive = _synthetic_archive(tmp_path)
    monkeypatch.setattr(vendor, "LTX25_RUNTIME_SHA256", "0" * 64)

    with pytest.raises(vendor.VendorError, match="integrity verification"):
        vendor.update(tmp_path / "root", archive)
    assert not (tmp_path / "root" / "ltx_core_mlx").exists()


def test_update_rejects_unsafe_archive_paths(tmp_path: Path, monkeypatch) -> None:
    vendor = _load_vendor_script()
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as tarball:
        payload = b"escape\n"
        info = tarfile.TarInfo(name="../escape.txt")
        info.size = len(payload)
        tarball.addfile(info, io.BytesIO(payload))
    monkeypatch.setattr(
        vendor, "LTX25_RUNTIME_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )

    with pytest.raises(vendor.VendorError, match="unsafe path"):
        vendor.update(tmp_path / "root", archive)


def test_vendor_script_is_self_contained() -> None:
    """The script must not import ``rapid_mlx`` (which pulls the MLX engine)."""
    source = VENDOR_SCRIPT.read_text(encoding="utf-8")
    assert "import rapid_mlx" not in source
    assert "from rapid_mlx" not in source
    assert "from .video" not in source


def test_sidecar_build_keeps_the_vendored_provenance_sources() -> None:
    """The Desktop source-drop must not strip ``_vendored.py``.

    ``ltx25_runtime()`` reads the provenance file from disk; a .pyc-only
    vendored package is reported as damaged, so the sidecar build has to
    keep these two package trees as sources.
    """
    script = (
        REPO_ROOT / "apps" / "rapid-mac" / "scripts" / "build-sidecar.sh"
    ).read_text(encoding="utf-8")
    assert '-not -path "*/site-packages/ltx_core_mlx/*"' in script
    assert '-not -path "*/site-packages/ltx_pipelines_mlx/*"' in script


def test_apple_lane_runs_the_runtime_and_packaged_artifact_tests() -> None:
    """The bundled runtime's unit and artifact-install tests must reach CI."""
    import yaml

    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["test-apple-silicon"]["steps"]
    runtime_run = next(
        step["run"] for step in steps if step.get("name") == "Run MLX-dependent tests"
    )
    artifact_run = next(
        step["run"]
        for step in steps
        if step.get("name") == "Run LTX-2.5 packaged-artifact install test"
    )

    for path in (
        "tests/test_ltx25_video.py",
        "tests/test_video_runtime_preflight.py",
        "tests/test_ltx25_vendor_contract.py",
    ):
        assert path in runtime_run
    assert "tests/test_ltx25_packaged_artifact.py" in artifact_run
    assert "--run-slow -m slow" in artifact_run
