# SPDX-License-Identifier: Apache-2.0
"""Regression tests for L-07 — ``rapid-mlx[vision]`` extra wiring.

Pre-fix probe (Kai r2): a fresh-venv ``pip install rapid-mlx==0.8.0``
did not pull ``mlx-vlm``, and hitting a VL route subsequently 500'd with
``ModuleNotFoundError: No module named 'mlx_vlm'``. The expected workaround
was an undocumented ``pip install 'mlx-vlm>=0.4.4'``.

State at 0.8.0 release (``a56a39b``): ``mlx-vlm`` is intentionally NOT in
core ``[project].dependencies`` (saves ~322 MB for text-only users) but
IS declared under ``[project.optional-dependencies].vision``, alongside
``opencv-python`` / ``torch`` / ``torchvision`` / ``pillow``. The
README quickstart documents ``pip install 'rapid-mlx[vision]'``.

These tests parse ``pyproject.toml`` via ``tomllib`` and lock in the
shape so a future refactor can't silently drop the extra (re-introducing
the bare ``500 ModuleNotFoundError`` on the VL route — same failure
shape as H-08's embeddings crash).
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path


def _load_pyproject() -> dict:
    """Locate and parse the repo's ``pyproject.toml``. Walks up from
    this test file so the lookup survives both ``pytest tests/`` and
    a ``pytest path/to/tests/...`` invocation."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "pyproject.toml"
        if candidate.is_file():
            with candidate.open("rb") as fp:
                return tomllib.load(fp)
    raise RuntimeError(  # pragma: no cover — sanity, repo always has one
        f"pyproject.toml not found above {here}"
    )


def _extra_specs(pyproject: dict, extra: str) -> list[str]:
    """Return the dep-spec list for a given ``[project.optional-dependencies]``
    extra. PEP 621 keeps these under ``project.optional-dependencies``."""
    return (
        pyproject.get("project", {})
        .get("optional-dependencies", {})
        .get(extra, [])
    )


def _split_spec(spec: str) -> tuple[str, str]:
    """Split ``"mlx-vlm>=0.6.3"`` into ``("mlx-vlm", ">=0.6.3")``. The
    PEP 508 surface is broader than this (markers, environment
    qualifiers) but every spec under our extras is the plain
    ``name<op><version>`` shape — keep the parse simple to avoid pulling
    ``packaging`` into the test runtime."""
    for op in (">=", "==", "~=", "!=", "<=", ">", "<"):
        if op in spec:
            name, _, ver = spec.partition(op)
            return name.strip(), op + ver.strip()
    return spec.strip(), ""


# ──────────────────────────────────────────────────────────────────────
# Core: vision extra exists, lists mlx-vlm, and the version floor is
# real (not a placeholder).
# ──────────────────────────────────────────────────────────────────────


def test_vision_extra_exists() -> None:
    """``[project.optional-dependencies].vision`` MUST be declared so
    ``pip install 'rapid-mlx[vision]'`` resolves. The README quickstart
    documents this exact invocation — its disappearance would be a
    documentation-vs-code drift."""
    py = _load_pyproject()
    extras = py.get("project", {}).get("optional-dependencies", {})
    assert "vision" in extras, (
        f"`[vision]` extra missing from pyproject. Available extras: "
        f"{sorted(extras)!r}. README quickstart references "
        f"`pip install 'rapid-mlx[vision]'` and would 404 without this."
    )


def test_vision_extra_lists_mlx_vlm() -> None:
    """The whole point of ``[vision]`` is to install ``mlx-vlm``. If
    this assertion fails, hitting a VL route still 500s with
    ``ModuleNotFoundError: No module named 'mlx_vlm'`` — i.e. L-07
    regressed."""
    py = _load_pyproject()
    specs = _extra_specs(py, "vision")
    names = {_split_spec(s)[0].lower() for s in specs}
    assert "mlx-vlm" in names, (
        f"mlx-vlm missing from `[vision]` extra. Got specs={specs!r}. "
        f"This is the dep that prevents the ``ModuleNotFoundError: No "
        f"module named 'mlx_vlm'`` on VL routes per L-07."
    )


def test_vision_mlx_vlm_floor_is_recognizable() -> None:
    """The minimum-version floor must be a concrete version, not an
    empty pin or a wildcard. ``0.8.0`` was tested against ``>=0.6.3``
    (Gemma 4 DLM PR #1347 + long-context prefill PR #1348). The L-07
    TODO references the older ``>=0.4.4`` floor as a workaround — we
    keep the stricter floor since downgrading would unwire
    DiffusionGemma support."""
    py = _load_pyproject()
    specs = _extra_specs(py, "vision")
    for spec in specs:
        name, ver = _split_spec(spec)
        if name.lower() == "mlx-vlm":
            assert ver.startswith(">="), (
                f"mlx-vlm spec should pin a minimum version with ``>=``; "
                f"got {spec!r}. A non-floor pin (``==`` / ``~=``) breaks "
                f"forward compatibility with mlx-vlm patch releases."
            )
            # The floor must be a real PEP-440 version string, not a
            # placeholder. Cheap-check: at least one dot.
            floor = ver[2:].strip()
            assert "." in floor, (
                f"mlx-vlm floor {floor!r} doesn't look like a real version. "
                f"Expected something like ``>=0.6.3``."
            )
            return
    raise AssertionError(  # pragma: no cover — guarded by the test above
        "mlx-vlm spec not found in `[vision]` extra"
    )


# ──────────────────────────────────────────────────────────────────────
# Defense in depth: mlx-vlm must NOT be in core deps (would defeat the
# whole point of the extra — the ~322 MB save for text-only users).
# ──────────────────────────────────────────────────────────────────────


def test_mlx_vlm_not_in_core_dependencies() -> None:
    """If ``mlx-vlm`` slips into ``[project].dependencies`` the
    text-only ``pip install rapid-mlx`` jumps from ~460 MB to ~782 MB.
    The point of the ``[vision]`` extra is to keep that surface
    opt-in. This is the second half of L-07's contract."""
    py = _load_pyproject()
    core = py.get("project", {}).get("dependencies", [])
    core_names = {_split_spec(s)[0].lower() for s in core}
    assert "mlx-vlm" not in core_names, (
        f"mlx-vlm leaked into core deps={core!r}. Move it back under "
        f"`[project.optional-dependencies].vision` so the text-only "
        f"`pip install rapid-mlx` stays slim (L-07)."
    )


# ──────────────────────────────────────────────────────────────────────
# README quickstart references `pip install 'rapid-mlx[vision]'`. The
# docs-code drift would silently break the documented opt-in path.
# ──────────────────────────────────────────────────────────────────────


def test_readme_quickstart_mentions_vision_extra() -> None:
    """The README quickstart MUST surface the ``[vision]`` opt-in so a
    user reading top-down learns how to install for VL routes BEFORE
    hitting a 500. Detection is intentionally loose: any line carrying
    both ``rapid-mlx`` and ``[vision]`` in the README counts."""
    here = Path(__file__).resolve()
    readme_path = None
    for parent in [here.parent, *here.parents]:
        candidate = parent / "README.md"
        if candidate.is_file():
            readme_path = candidate
            break
    assert readme_path is not None, (
        "README.md not found above the test file — repo layout regressed?"
    )
    text = readme_path.read_text(encoding="utf-8")
    has_vision_install = any(
        ("rapid-mlx" in line and "[vision]" in line) for line in text.splitlines()
    )
    assert has_vision_install, (
        "README.md no longer documents `pip install 'rapid-mlx[vision]'`. "
        "Users hitting a VL route now get a bare 500 with no install hint "
        "(L-07 — Kai r2 probe surfaced this on fresh-venv 0.8.0 installs)."
    )


# ──────────────────────────────────────────────────────────────────────
# Self-consistency: the ``all`` extra (advertised as union-of-everything)
# also pulls mlx-vlm. Otherwise ``pip install 'rapid-mlx[all]'`` silently
# skips vision and we get the L-07 failure mode under a different name.
# ──────────────────────────────────────────────────────────────────────


def test_all_extra_includes_mlx_vlm() -> None:
    """``[all]`` is documented as the union of vision + chat + embeddings.
    A user who installs ``rapid-mlx[all]`` expecting "everything"
    must NOT discover at request-time that mlx-vlm isn't there."""
    py = _load_pyproject()
    all_specs = _extra_specs(py, "all")
    names = {_split_spec(s)[0].lower() for s in all_specs}
    assert "mlx-vlm" in names, (
        f"`[all]` extra is documented as union-of-everything but does "
        f"not include mlx-vlm. Got specs={all_specs!r}."
    )


# ──────────────────────────────────────────────────────────────────────
# Tooling sanity — tomllib was stdlib starting in 3.11 (we require
# Python >=3.10 in pyproject). On 3.10 we'd need ``tomli`` as a
# fallback — skip this assertion there. ``sys.version_info`` is the
# right gate (the import would already have failed at module load).
# ──────────────────────────────────────────────────────────────────────


def test_python_version_supports_tomllib() -> None:
    """We use stdlib ``tomllib`` for the pyproject parse — it's the
    Python ≥3.11 path. The repo's ``requires-python = ">=3.10"`` means
    a 3.10 contributor would see this test fail at import-time. Pin
    the version requirement here so the failure mode is obvious."""
    assert sys.version_info >= (3, 11), (
        "tomllib is stdlib on Python ≥3.11. On 3.10 install ``tomli`` "
        "and import-fallback if you need to backport these tests."
    )
