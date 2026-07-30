# SPDX-License-Identifier: Apache-2.0
"""Packaging contract for the bundled CogVideoX-Fun MLX runtime."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "videox_fun_mlx"


def _pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_cogvideox_runtime_is_in_package_discovery() -> None:
    includes = _pyproject()["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "videox_fun_mlx*" in includes


def test_cogvideox_runtime_has_complete_import_surface_and_attribution() -> None:
    required = {
        "models/cogvideox_transformer3d.py",
        "models/cogvideox_vae.py",
        "models/t5_encoder.py",
        "models/tokenizer.py",
        "pipeline/pipeline_cogvideox_fun_inpaint.py",
        "pipeline/scheduler.py",
        "LICENSE",
        "NOTICE",
    }
    assert not {path for path in required if not (RUNTIME / path).is_file()}


def test_video_extra_has_tokenizer_runtime() -> None:
    specs = _pyproject()["project"]["optional-dependencies"]["video"]
    assert any(spec.lower().startswith("sentencepiece") for spec in specs)


def test_cogvideox_docs_require_no_source_checkout() -> None:
    guide = (ROOT / "docs/guides/video-generation.md").read_text()
    section = guide.split("## CogVideoX-Fun", 1)[1]
    assert "git clone" not in section
    assert "export PYTHONPATH" not in section
    assert "pip install 'rapid-mlx[video]'" in section
