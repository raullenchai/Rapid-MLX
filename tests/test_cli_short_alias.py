"""Packaging contract for the first-class short CLI command."""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


def test_rmlx_uses_the_canonical_cli_entrypoint():
    with (Path(__file__).resolve().parents[1] / "pyproject.toml").open("rb") as handle:
        scripts = tomllib.load(handle)["project"]["scripts"]

    assert scripts["rmlx"] == scripts["rapid-mlx"] == "vllm_mlx.cli:main"
