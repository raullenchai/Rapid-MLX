from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[1]
    / "apps"
    / "rapid-mac"
    / "scripts"
    / "check-sidecar-distributions.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("sidecar_constraints", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def constraints():
    return _load_module()


def test_emitted_constraints_are_stable_and_complete(constraints) -> None:
    emitted = constraints.emit_constraints().splitlines()
    assert emitted[0].startswith("# Release-tested versions")
    assert emitted[1:22] == [
        "mlx==0.32.3",
        "tensorfold==0.6.6",
        "transformers==5.15.1",
        "mlx-vlm==0.7.2",
        "mlx-audio==0.5.3",
        "mflux==0.20.0",
        "sentencepiece==0.2.2",
        "mlx-video-with-audio==0.1.36",
        "mlx-arsenal==0.12.1",
        "pyobjc-core==12.2.2",
        "pyobjc-framework-ApplicationServices==12.2.2",
        "pyobjc-framework-Cocoa==12.2.2",
        "pyobjc-framework-CoreText==12.2.2",
        "pyobjc-framework-Quartz==12.2.2",
        "torch==2.14.1",
        "torchvision==0.29.1",
        "opencv-python==4.14.0.94",
        "matplotlib==3.11.2",
        "imageio==2.38.0",
        "imageio-ffmpeg==0.6.0",
        "pillow==12.3.0",
    ]


def test_emit_constraints_needs_only_the_python_standard_library() -> None:
    result = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "--emit-constraints"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "mlx-vlm==0.7.2" in result.stdout
    assert "mlx-audio==0.5.3" in result.stdout
    assert "mlx-video-with-audio==0.1.36" in result.stdout


def _write_metadata(
    root: Path, name: str, version: str, requires: tuple[str, ...] = ()
) -> None:
    dist_info = root / f"{name.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir()
    lines = ["Metadata-Version: 2.1", f"Name: {name}", f"Version: {version}"]
    lines.extend(f"Requires-Dist: {requirement}" for requirement in requires)
    (dist_info / "METADATA").write_text("\n".join(lines) + "\n")


def test_find_errors_accepts_coherent_distribution_metadata(
    constraints, tmp_path: Path
) -> None:
    _write_metadata(tmp_path, "transformers", "5.15.1")
    _write_metadata(
        tmp_path,
        "mlx-vlm",
        "0.7.2",
        ("transformers>=5.14.0; python_version >= '3'",),
    )
    assert constraints.find_errors(tmp_path) == []


def test_find_errors_rejects_nonexception_conflict_from_metadata(
    constraints, tmp_path: Path
) -> None:
    _write_metadata(tmp_path, "transformers", "5.12.1")
    _write_metadata(tmp_path, "other-vision", "1.0", ("transformers>=5.14.0",))
    assert constraints.find_errors(tmp_path) == [
        "other-vision requires transformers>=5.14.0, but bundled transformers==5.12.1"
    ]


def test_find_errors_rejects_empty_distribution_directory(
    constraints, tmp_path: Path
) -> None:
    assert constraints.find_errors(tmp_path) == [
        f"no installed distributions found in {tmp_path}"
    ]


def test_tensorfold_repair_matches_sidecar_and_running_python(monkeypatch):
    import runpy
    import shlex

    from rapid_mlx.speculative import tensorfold_runtime as policy

    pins = (SCRIPT.parent / "sidecar-constraints.txt").read_text().splitlines()
    assert f"mlx=={policy.SUPPORTED_MLX_VERSION}" in pins
    assert f"mlx-audio=={policy.SUPPORTED_MLX_AUDIO_VERSION}" in pins
    # A launcher path containing shell syntax must remain a single argument.
    interpreter = "/tmp/Rapid user's $venv/bin/python"
    monkeypatch.setattr(sys, "executable", interpreter)
    loaded = runpy.run_path(policy.__file__)
    command = shlex.split(loaded["INSTALL_HINT"].splitlines()[1])
    assert command == [
        interpreter,
        "-m",
        "pip",
        "install",
        f"mlx=={policy.SUPPORTED_MLX_VERSION}",
        f"mlx-audio=={policy.SUPPORTED_MLX_AUDIO_VERSION}",
        f"tensorfold @ git+{policy.SUPPORTED_RUNTIME_URL}@{policy.SUPPORTED_REVISION}",
    ]


def test_arsenal_dependency_matches_release_tested_sidecar(constraints):
    from packaging.requirements import Requirement

    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    project = tomllib.loads((SCRIPT.parents[3] / "pyproject.toml").read_text())[
        "project"
    ]
    for requirements in (
        project["dependencies"],
        project["optional-dependencies"]["video"],
    ):
        requirement = next(
            Requirement(value)
            for value in requirements
            if value.startswith("mlx-arsenal")
        )
        assert str(requirement.specifier) == "==0.12.1"
        assert "mlx-arsenal==0.12.1" in constraints.emit_constraints().splitlines()
        assert requirement.marker.evaluate(
            {"platform_system": "Darwin", "python_version": "3.11"}
        )
        assert not requirement.marker.evaluate(
            {"platform_system": "Darwin", "python_version": "3.10"}
        )
        assert not requirement.marker.evaluate(
            {"platform_system": "Linux", "python_version": "3.12"}
        )


def test_all_base_requirements_have_reviewed_upper_bounds():
    from importlib import import_module

    tomllib = import_module("tomllib" if sys.version_info >= (3, 11) else "tomli")
    from packaging.requirements import Requirement

    project = tomllib.loads((SCRIPT.parents[3] / "pyproject.toml").read_text())[
        "project"
    ]
    for raw in project["dependencies"]:
        requirement = Requirement(raw)
        assert any(
            s.operator in ("<", "<=", "==", "~=") for s in requirement.specifier
        ), raw
        if requirement.name == "torch":
            assert "2.10.0" in requirement.specifier
            assert "2.9.1" not in requirement.specifier


def test_hash_lock_covers_base_and_desktop_extras():
    import re
    from importlib import import_module

    tomllib = import_module("tomllib" if sys.version_info >= (3, 11) else "tomli")
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    lock = (SCRIPT.parent / "sidecar-requirements.lock").read_text()
    entries = lock.replace("\\\n", "").splitlines()
    pins = {}
    for entry in entries:
        if not entry.strip() or entry.startswith("#"):
            continue
        requirement, *hashes = entry.split("--hash=sha256:")
        parsed = Requirement(requirement.strip())
        specs = list(parsed.specifier)
        assert len(specs) == 1 and specs[0].operator == "=="
        assert hashes and all(re.fullmatch(r"[0-9a-f]{64}", h.strip()) for h in hashes)
        name = canonicalize_name(parsed.name)
        assert name not in pins
        pins[name] = specs[0].version

    constraints = {}
    for line in (SCRIPT.parent / "sidecar-constraints.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==")
            constraints[canonicalize_name(name)] = version
    assert {k: v for k, v in constraints.items() if k != "tensorfold"} == pins
    assert pins["tiktoken"] == "0.13.0"
    assert pins["safetensors"] == "0.8.0"
    assert {
        "regex",
        "requests",
        "charset-normalizer",
        "idna",
        "urllib3",
        "certifi",
    } <= pins.keys()

    project = tomllib.loads((SCRIPT.parents[3] / "pyproject.toml").read_text())[
        "project"
    ]
    environment = {
        "platform_system": "Darwin",
        "sys_platform": "darwin",
        "platform_machine": "arm64",
        "python_version": "3.12",
    }
    for raw in (
        project["dependencies"]
        + project["optional-dependencies"]["audio-desktop"]
        + project["optional-dependencies"]["computer-use"]
    ):
        requirement = Requirement(raw)
        if requirement.marker is None or requirement.marker.evaluate(environment):
            assert pins[canonicalize_name(requirement.name)] in requirement.specifier, (
                raw
            )


def test_sidecar_source_installs_cannot_resolve_outside_hash_lock():
    import shlex

    script = (SCRIPT.parent / "build-sidecar.sh").read_text().replace("\\\n", "")
    installs = [
        shlex.split(line) for line in script.splitlines() if "-m pip install" in line
    ]
    assert len(installs) == 2
    locked, source = installs
    assert "--require-hashes" in locked
    assert locked[locked.index("--requirement") + 1] == "$SIDECAR_LOCK"
    assert "--only-binary=:all:" in locked
    assert locked[locked.index("--platform") + 1] == "macosx_14_0_arm64"
    assert "--no-deps" in source
    assert "${RAPID_MLX_INSTALL_TARGET}[audio-desktop,computer-use]" in source
    assert "$TENSORFOLD_RUNTIME_SPEC" in source
