"""Packaging contract for the first-class short CLI command."""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


def test_rmlx_uses_the_canonical_cli_entrypoint():
    repo_root = Path(__file__).resolve().parents[1]
    with (repo_root / "pyproject.toml").open("rb") as handle:
        scripts = tomllib.load(handle)["project"]["scripts"]

    assert scripts["rmlx"] == scripts["rapid-mlx"] == "vllm_mlx.cli:cli_entrypoint"


def test_curl_installer_exposes_rmlx_on_path():
    installer = (Path(__file__).resolve().parents[1] / "install.sh").read_text()

    assert '"$INSTALL_DIR/bin/rmlx"' in installer
    assert '"$BIN_DIR/rmlx"' in installer
