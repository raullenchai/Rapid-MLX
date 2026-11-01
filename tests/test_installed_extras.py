# SPDX-License-Identifier: Apache-2.0
"""Keep installed-extra inference aligned with the one-line installer."""

import builtins
import importlib.metadata as metadata
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rapid_mlx.runtime.installed_extras import installed_extras, recorded_tool_extras


@pytest.mark.parametrize("name", ["rapid-mlx", "Rapid_MLX", "vllm.mlx"])
def test_recorded_uv_extras(tmp_path, name):
    (tmp_path / "uv-receipt.toml").write_text(
        f'[tool]\nrequirements = [{{name = "{name}", '
        'extras = ["Audio_Feature", "audio.feature", "embeddings", '
        '"dev", "test", "all", "CI_GPU"], specifier = "==1.0.0"}]\n'
    )
    assert recorded_tool_extras(tmp_path) == [
        "all",
        "audio-feature",
        "ci-gpu",
        "dev",
        "embeddings",
        "test",
    ]


def test_recorded_uv_extras_ignores_other_packages(tmp_path):
    (tmp_path / "uv-receipt.toml").write_text(
        '[tool]\nrequirements = [{name = "other", extras = ["audio"]}]\n'
    )
    assert recorded_tool_extras(tmp_path) == []


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("rapid-mlx[audio]==1.0.0", ["audio"]),
        (
            "rapid-mlx[audio,embeddings] @ git+https://example.invalid/r.git@abc",
            ["audio", "embeddings"],
        ),
        ("git+https://example.invalid/r.git@abc", []),
        ("/tmp/rapid-mlx", []),
        ("other[audio]==1.0.0", []),
        (
            "VLLM_MLX[Audio_Feature,all]==1.0.0",
            ["all", "audio-feature"],
        ),
        ("rapid-mlx[audio] @", ["audio"]),
    ],
)
def test_recorded_pipx_extras(tmp_path, spec, expected):
    (tmp_path / "pipx_metadata.json").write_text(
        json.dumps({"main_package": {"package_or_url": spec}})
    )
    assert recorded_tool_extras(tmp_path) == expected


@pytest.mark.parametrize("filename", ["uv-receipt.toml", "pipx_metadata.json"])
def test_recorded_extras_malformed_receipt(tmp_path, filename):
    (tmp_path / filename).write_text("{invalid")
    assert recorded_tool_extras(tmp_path) == []


def test_recorded_extras_no_receipts(tmp_path):
    assert recorded_tool_extras(tmp_path) == []


def test_recorded_extras_defaults_to_running_prefix(tmp_path, monkeypatch):
    (tmp_path / "uv-receipt.toml").write_text(
        '[tool]\nrequirements = [{name = "rapid-mlx", extras = ["audio"]}]\n'
    )
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    assert recorded_tool_extras() == ["audio"]


def test_installed_extras_matches_installer():
    installer = (Path(__file__).parents[1] / "install.sh").read_text()
    function = installer.split("installed_extras_of_venv() {", 1)[1]
    script = function.split("-I -c '", 1)[1].split("' 2>/dev/null)", 1)[0]
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == ",".join(installed_extras())


@pytest.fixture
def fake_metadata(monkeypatch):
    def configure(requirements, installed, dependencies=None, *, legacy=False):
        def distribution(name):
            if legacy and name == "rapid-mlx":
                raise metadata.PackageNotFoundError(name)
            return SimpleNamespace(requires=requirements)

        monkeypatch.setattr(metadata, "distribution", distribution)
        monkeypatch.setattr(
            metadata,
            "distributions",
            lambda: [
                SimpleNamespace(metadata={"Name": name}, version=version)
                for name, version in installed.items()
            ],
        )
        monkeypatch.setattr(
            metadata, "requires", lambda name: (dependencies or {}).get(name)
        )

    return configure


def test_selects_installed_distinct_dependencies_and_normalizes(fake_metadata):
    fake_metadata(
        ['audio_dep>=1; extra == "Audio_Feature"', 'other; extra == "embeddings"'],
        {"Audio.Dep": "1.1", "other": "2"},
    )
    assert installed_extras() == ["audio-feature", "embeddings"]


@pytest.mark.parametrize("installed", [{}, {"audio-dep": "0.9"}, {"audio-dep": "1.1"}])
def test_requires_every_distinct_dependency(fake_metadata, installed):
    fake_metadata(
        ['audio-dep>=1; extra == "audio"', 'missing; extra == "audio"'], installed
    )
    assert installed_extras() == []


def test_rejects_incompatible_version(fake_metadata):
    fake_metadata(['audio-dep>=2; extra == "audio"'], {"audio-dep": "1"})
    assert installed_extras() == []


def test_base_dependency_closure_is_not_an_extra(fake_metadata):
    fake_metadata(
        ["base", 'transitive; extra == "audio"'],
        {"base": "1", "transitive": "1"},
        {"base": ["middle"], "middle": ["transitive"], "transitive": ["base"]},
    )
    assert installed_extras() == []


def test_shared_base_version_does_not_disqualify_extra(fake_metadata):
    fake_metadata(
        ["base", 'base>=2; extra == "audio"', 'audio-dep>=1; extra == "audio"'],
        {"base": "1", "audio-dep": "1.1a1"},
    )
    assert installed_extras() == ["audio"]


def test_excludes_development_extras(fake_metadata):
    fake_metadata(
        [
            f'dep; extra == "{extra}"'
            for extra in ("dev", "test", "all", "ci_GPU", "audio")
        ],
        {"dep": "1"},
    )
    assert installed_extras() == ["audio"]


def test_legacy_distribution_fallback(fake_metadata):
    fake_metadata(['dep; extra == "audio"'], {"dep": "1"}, legacy=True)
    assert installed_extras() == ["audio"]


@pytest.mark.parametrize("vendored", [False, True])
def test_packaging_fallbacks(fake_metadata, monkeypatch, vendored):
    from packaging.requirements import Requirement

    real_import = builtins.__import__

    def import_without_packaging(name, *args, **kwargs):
        if name == "packaging.requirements":
            raise ImportError(name)
        if name == "pip._vendor.packaging.requirements":
            if vendored:
                return SimpleNamespace(Requirement=Requirement)
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    fake_metadata(
        [
            'dep; extra == "audio"',
            'dep; extra == "gated" and python_version >= "3.0"',
            'dep; extra == "windows" and sys_platform == "win32"',
        ],
        {"dep": "1"},
    )
    monkeypatch.setattr(builtins, "__import__", import_without_packaging)
    assert installed_extras() == (["audio", "gated"] if vendored else ["audio"])


@pytest.mark.parametrize("operation", ["distribution", "distributions", "requires"])
def test_errors_return_empty_list(fake_metadata, monkeypatch, operation):
    fake_metadata(["base", 'dep; extra == "audio"'], {"dep": "1"})

    def fail(*args):
        raise RuntimeError("broken metadata")

    monkeypatch.setattr(metadata, operation, fail)
    assert installed_extras() == []


def test_no_rapid_mlx_distribution_returns_empty(monkeypatch):
    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "distribution", missing)
    assert installed_extras() == []


def test_skips_unparseable_requirements_and_dependencies(fake_metadata):
    fake_metadata(
        [
            "base",
            "@not-a-name",
            'broken[; extra == "audio"',
            'dep; extra == "audio"',
        ],
        {"base": "1", "dep": "1"},
        {"base": ["@not-a-name", "invalid[", "missing-dist"]},
    )
    assert installed_extras() == ["audio"]
