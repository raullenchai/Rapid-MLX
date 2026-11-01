# SPDX-License-Identifier: Apache-2.0
"""Keep installed-extra inference aligned with the one-line installer."""

import builtins
import importlib.metadata as metadata
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rapid_mlx.runtime.installed_extras import installed_extras


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
