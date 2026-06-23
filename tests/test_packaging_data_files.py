# SPDX-License-Identifier: Apache-2.0
"""r10-I — packaging regression: data files must ship in installed wheel.

Codex r10-B caught that ``vllm_mlx/audio/aliases.json`` was added under
``vllm_mlx/audio/`` but never declared in ``[tool.setuptools.package-data]``.
A sdist/wheel built from that state silently dropped the file — source-tree
tests passed (the file is right there on disk), but ``pip install rapid-mlx``
and then ``rapid-mlx serve kokoro`` would raise ``FileNotFoundError`` inside
``vllm_mlx/audio/registry.py::resolve_audio_alias`` before any audio engine
loaded.

This test pins two invariants:

1. Every non-Python data file that ``vllm_mlx`` reads at runtime via
   ``importlib.resources`` must be reachable through that API in the
   current installed/source layout.
2. Every such file must be listed in
   ``[tool.setuptools.package-data].vllm_mlx`` in ``pyproject.toml`` so
   it actually ends up in the built wheel and sdist.

If a future contributor adds a new JSON/YAML registry file but forgets the
package-data entry, this test fails loudly at PR time instead of silently
shipping a broken wheel to PyPI.
"""

from __future__ import annotations

import importlib.resources
import sys
from pathlib import Path

import pytest

try:  # pragma: no cover - py311+ has tomllib in stdlib
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


# Files that must be present in every installed/source layout.
# Format: (package, file-name-relative-to-package, package-data-glob).
# The third element is what we expect to find in pyproject's package-data
# entry (either an exact match or a glob that covers it).
REQUIRED_DATA_FILES: list[tuple[str, str, str]] = [
    # Text-model alias registry — has always shipped.
    ("vllm_mlx", "aliases.json", "aliases.json"),
    # r10-A: audio alias registry. Codex r10-B caught this missing from
    # package-data; this entry locks the fix in place.
    ("vllm_mlx", "audio/aliases.json", "audio/aliases.json"),
]


def _pyproject_path() -> Path:
    # tests/ is a sibling of pyproject.toml in the source tree, and the
    # installed wheel layout never ships pyproject.toml — so this test
    # only runs meaningfully against a source checkout. Skip otherwise.
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        candidate = parent / "pyproject.toml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("pyproject.toml not found from tests dir")


def _package_data_entries() -> list[str]:
    with _pyproject_path().open("rb") as fh:
        data = tomllib.load(fh)
    return list(
        data.get("tool", {})
        .get("setuptools", {})
        .get("package-data", {})
        .get("vllm_mlx", [])
    )


@pytest.mark.parametrize(
    ("package", "relpath", "_glob"),
    REQUIRED_DATA_FILES,
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_required_data_file_resolvable_via_importlib_resources(
    package: str, relpath: str, _glob: str
) -> None:
    """The file must be reachable through ``importlib.resources``.

    This mirrors how production code reads the registry (see
    ``vllm_mlx/audio/registry.py`` and ``vllm_mlx/aliases.py``), so the
    test fails the same way a real install would if the file were
    missing.
    """
    try:
        traversable = importlib.resources.files(package).joinpath(relpath)
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive
        pytest.fail(f"package {package!r} not importable: {exc}")

    assert traversable.is_file(), (
        f"{package}/{relpath} is not reachable via importlib.resources — "
        f"the installed wheel will be broken at runtime even if the "
        f"source tree happens to have the file on disk."
    )


@pytest.mark.parametrize(
    ("_package", "_relpath", "glob"),
    REQUIRED_DATA_FILES,
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_required_data_file_declared_in_pyproject_package_data(
    _package: str, _relpath: str, glob: str
) -> None:
    """The file must appear in ``[tool.setuptools.package-data].vllm_mlx``.

    setuptools only bundles files explicitly listed (or matched by a glob)
    in package-data. A file that exists on disk but is missing from this
    list will be silently dropped during ``python -m build`` — exactly the
    failure mode codex r10-B caught for ``audio/aliases.json``.
    """
    if not _pyproject_path().exists():  # pragma: no cover
        pytest.skip("pyproject.toml not available (installed wheel layout)")

    entries = _package_data_entries()
    assert glob in entries, (
        f"Expected {glob!r} in [tool.setuptools.package-data].vllm_mlx, "
        f"got entries={entries!r}. Without this declaration the file will "
        f"be missing from the built wheel/sdist."
    )
