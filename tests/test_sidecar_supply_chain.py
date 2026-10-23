"""Exercise the build's actual PBS gate without installing or running Python."""

import hashlib
import io
import os
import subprocess
import tarfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "apps/rapid-mac/scripts/build-sidecar.sh"


def archive(path, payload):
    with tarfile.open(path, "w:gz") as output:
        entry = tarfile.TarInfo("python/bin/python3.12")
        entry.mode = 0o755
        entry.size = len(payload)
        output.addfile(entry, io.BytesIO(payload))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_gate(tmp_path, cached=None, download=b"trusted interpreter"):
    trusted = tmp_path / "trusted.tar.gz"
    digest = archive(trusted, b"trusted interpreter")
    downloaded = tmp_path / "download.tar.gz"
    archive(downloaded, download)
    # gzip headers include filename; reuse the exact trusted bytes for a good download.
    if download == b"trusted interpreter":
        downloaded.write_bytes(trusted.read_bytes())
    cache = tmp_path / f"cpython-test-test-{digest}.tar.gz"
    if cached == "valid":
        cache.write_bytes(trusted.read_bytes())
    elif cached == "tampered":
        archive(cache, b"attacker interpreter")
    stage = tmp_path / "stage"
    stage.mkdir()
    source = BUILD.read_text()
    block = source.split("# ----- step 1: embedded python interpreter", 1)[1]
    block = block.split("PBS_URL=", 1)[1]
    block = "PBS_URL=" + block.split("# ----- step 1.5:", 1)[0]
    # Replace transport, not validation: the real script must reject a valid tar
    # carrying unexpected bytes, before extraction or any interpreter execution.
    script = """set -euo pipefail
curl() {
    touch "$OUT_DIR/download-attempted"
    while [ "$1" != "-o" ]; do shift; done
    cp "$OUT_DIR/download.tar.gz" "$2"
}
""" + block
    env = dict(
        os.environ,
        PBS_TAG="test",
        PBS_VERSION="test",
        PBS_SHA256=digest,
        OUT_DIR=str(tmp_path),
        STAGE=str(stage),
    )
    return subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True
    )


def test_valid_cache_is_verified_and_reused(tmp_path):
    result = run_gate(tmp_path, cached="valid", download=b"bad")
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "download-attempted").exists()
    assert (
        tmp_path / "stage/python/bin/python3.12"
    ).read_bytes() == b"trusted interpreter"


def test_tampered_but_well_formed_cache_is_replaced(tmp_path):
    result = run_gate(tmp_path, cached="tampered")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "download-attempted").exists()
    assert (
        tmp_path / "stage/python/bin/python3.12"
    ).read_bytes() == b"trusted interpreter"


def test_tampered_but_well_formed_download_is_never_extracted(tmp_path):
    result = run_gate(tmp_path, download=b"attacker interpreter")
    assert result.returncode != 0
    assert "SHA-256/integrity validation" in result.stderr
    assert not (tmp_path / "stage/python").exists()
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "tag,version,digest",
    [
        ("other", "3.12.13", ""),
        ("20260610", "other", ""),
        ("20260610", "3.12.13", "bad"),
    ],
)
def test_interpreter_overrides_require_reviewed_digest(tag, version, digest):
    source = BUILD.read_text()
    configuration = source[source.index("PBS_TAG=") : source.index("# compileall uses")]
    result = subprocess.run(
        ["bash", "-c", configuration],
        capture_output=True,
        text=True,
        env=dict(os.environ, PBS_TAG=tag, PBS_VERSION=version, PBS_SHA256=digest),
    )
    assert result.returncode != 0
    assert "PBS_SHA256 must be a reviewed" in result.stderr


def test_arsenal_base_and_legacy_extra_match_release_pin():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    constraints = (BUILD.parent / "sidecar-constraints.txt").read_text().splitlines()
    pin = next(
        Requirement(line) for line in constraints if line.startswith("mlx-arsenal==")
    )
    for dependencies in (
        project["dependencies"],
        project["optional-dependencies"]["video"],
    ):
        requirement = next(
            Requirement(line) for line in dependencies if line.startswith("mlx-arsenal")
        )
        assert requirement.specifier == pin.specifier
        assert not requirement.specifier.contains("0.17.0")
        assert requirement.marker.evaluate(
            {"platform_system": "Darwin", "python_version": "3.12"}
        )
        assert not requirement.marker.evaluate(
            {"platform_system": "Linux", "python_version": "3.12"}
        )
