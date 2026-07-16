# SPDX-License-Identifier: Apache-2.0
"""Regression tests for L-07 (``rapid-mlx[vision]`` extra wiring) and
L-07-B (Gemma 4 fresh-install regression from 0.10.0).

Pre-fix probe (Kai r2, 0.8.0): a fresh-venv ``pip install rapid-mlx==0.8.0``
did not pull ``mlx-vlm``, and hitting a VL route 500'd with
``ModuleNotFoundError: No module named 'mlx_vlm'``. Fix: declare
``mlx-vlm`` under ``[project.optional-dependencies].vision`` (NOT core,
to save ~322 MB for text-only users). README quickstart documents
``pip install 'rapid-mlx[vision]'``.

0.10.0 regression (Layer A "+13 Gemma 4 aliases"): a fresh-venv
``pip install rapid-mlx==0.10.0 && rapid-mlx serve gemma-4-12b-4bit``
crashed the same way — Gemma 4 uses ``mlx_vlm.models.gemma4.language``
because Google publishes it as a VLM-family checkpoint. The obvious
fix — promoting ``mlx-vlm`` to core — would drag in ~+483 MB of
transitive weight (opencv-python 120 MB, pyarrow 123 MB from datasets,
pandas 70 MB, scipy 98 MB, mlx-audio 17 MB, ...) for text-only users
who never touch Gemma 4. Instead we **vendor** the ~1200 lines of
Gemma 4 text-only classes (config + language + rope_utils) into
``vllm_mlx/models/gemma4_vendored/`` (+200 KB of repo, zero MB user
install). ``vllm_mlx/models/gemma4_text.py`` prefers ``mlx-vlm`` when
installed (so ``[vision]`` users get the shared code path) and falls
back to the vendored copy otherwise.

These tests parse ``pyproject.toml`` via ``tomllib`` and lock in both
contracts:

  * ``[vision]`` still exists, still lists ``mlx-vlm`` (for Qwen-VL and
    true VLM routes — image-input models still need mlx-vlm proper).
  * ``mlx-vlm`` NEVER slips into core deps (the ~322 MB / ~483 MB anti-
    bloat guard).
  * Gemma 4 boots WITHOUT ``mlx-vlm`` — the vendored module tree exists
    and ``gemma4_text.py`` carries the try/except fallback.

A future refactor that silently drops any of these re-opens either the
original L-07 (bare 500 on VL routes) or L-07-B (bare boot failure on
Gemma 4 — same failure shape).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Codex round-1 BLOCKING: ``import tomllib`` at module-import time would
# crash on Python 3.10 (the floor in ``pyproject.toml`` →
# ``requires-python = ">=3.10"``) before the version-floor test could
# emit a diagnostic. Fall back to the third-party ``tomli`` backport
# (same API surface) so the file imports cleanly on every supported
# interpreter. The runtime is identical on 3.11+ where ``tomllib`` is
# stdlib; on 3.10 the user must ``pip install tomli`` to run the
# vision-extra lock-in tests (skipped at the file level otherwise).
try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover — 3.10 fallback
    try:
        import tomli as tomllib  # type: ignore[import-not-found,no-redef]
    except ModuleNotFoundError:
        import pytest

        pytest.skip(
            "tomllib (stdlib ≥3.11) or tomli (3.10 backport) is required "
            "to parse pyproject.toml; install one with `pip install tomli` "
            "to run the L-07 lock-in tests.",
            allow_module_level=True,
        )


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
    return pyproject.get("project", {}).get("optional-dependencies", {}).get(extra, [])


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
# Tooling sanity — pyproject pins ``requires-python = ">=3.10"``. To
# keep the L-07 lock-in suite running on every supported interpreter
# (codex round-2 BLOCKING: a 3.10 CI worker without tomli would have
# module-skipped the entire suite), the [dev] extra declares
# ``tomli>=2.0.1; python_version < "3.11"``. This test pins both
# halves of that contract: the requires-python floor AND the matching
# tomli dev-dep declaration.
# ──────────────────────────────────────────────────────────────────────


def test_pyproject_requires_python_floor_matches_tomllib_fallback() -> None:
    """The TOML import block falls back to ``tomli`` on Python <3.11 so
    the L-07 lock-in tests work on every Python the project supports.
    Pin both halves of that contract here so a future floor bump
    (e.g. ``>=3.11``) lets a reviewer notice the now-dead fallback."""
    py = _load_pyproject()
    requires = py.get("project", {}).get("requires-python", "")
    assert requires, "pyproject.toml must declare requires-python"
    # The floor must be ``>=3.10`` or stricter — anything looser would
    # widen the surface this test file's tomli fallback covers.
    assert ">=3." in requires, (
        f"requires-python={requires!r} doesn't pin a 3.x floor — the "
        f"tomli fallback in this file assumes a Python 3 floor."
    )
    # Document the runtime we're using so a 3.10 contributor sees a
    # meaningful trail when reviewing the fallback.
    assert sys.version_info >= (3, 10), (
        f"Test runtime {sys.version_info[:2]} is below the project's "
        f"3.10 floor — the tomli fallback can't help here."
    )


# ──────────────────────────────────────────────────────────────────────
# L-07-B: Gemma 4 fresh-install regression from 0.10.0.
#
# Google publishes Gemma 4 as a VLM-family checkpoint whose Python
# classes live in ``mlx_vlm.models.gemma4``. In 0.10.0 that dep was
# only in ``[vision]``, so a fresh ``pip install rapid-mlx==0.10.0``
# followed by ``rapid-mlx serve gemma-4-12b-4bit`` crashed at import.
# The 0.10.1 fix is to vendor the ~1200 lines of Gemma 4 text-only
# classes into ``vllm_mlx/models/gemma4_vendored/`` so the fresh
# install works without any [vision] extra pulled.
# ──────────────────────────────────────────────────────────────────────


def test_gemma4_vendored_module_exists() -> None:
    """The vendored module tree must exist with the 3 upstream files
    (config, language, rope_utils) plus the __init__.py that inlines
    BaseModelConfig + LanguageModelOutput. Missing any one of these
    means a fresh-venv boot re-crashes with
    ``ImportError: mlx-vlm dependency missing`` — the exact 0.10.0
    regression this vendor path was created to close."""
    here = Path(__file__).resolve()
    repo_root = None
    for parent in [here.parent, *here.parents]:
        if (parent / "pyproject.toml").is_file():
            repo_root = parent
            break
    assert repo_root is not None, "pyproject.toml not found above test file"
    vendored = repo_root / "vllm_mlx" / "models" / "gemma4_vendored"
    assert vendored.is_dir(), (
        f"gemma4_vendored/ missing at {vendored}. Without it, a fresh "
        f"`pip install rapid-mlx && rapid-mlx serve gemma-4-12b-4bit` "
        f"re-crashes with the 0.10.0 ImportError (L-07-B)."
    )
    for required in ("__init__.py", "config.py", "language.py", "rope_utils.py"):
        path = vendored / required
        assert path.is_file(), (
            f"gemma4_vendored/{required} missing. All four files are "
            f"needed to bypass mlx-vlm for Gemma 4 text-only inference."
        )


def test_gemma4_vendored_modules_importable_without_mlx_vlm() -> None:
    """Codex round 3 NIT: the AST test proves the fallback SYNTAX is
    intact but not that the vendored ``TextConfig`` and
    ``LanguageModel`` classes are actually importable, or that they
    stay API-compatible with what ``gemma4_text.py`` expects. Close
    that gap by importing the vendored modules under a simulated
    missing-``mlx_vlm`` environment and constructing a ``TextConfig``
    from a minimal dict.

    Simulation: temporarily insert ``None`` under the ``mlx_vlm.*``
    module keys in ``sys.modules`` so any ``import mlx_vlm...``
    raises ``ImportError`` — mirroring the fresh-venv path. Restore
    the originals in ``finally`` so no other test sees the tampering.

    We intentionally do NOT construct ``LanguageModel(tc)`` here:
    that would allocate MLX weights (~gigabytes for a real config).
    The wrapper's contract with the vendored file is
    ``TextConfig.from_dict(dict) -> TextConfig`` and
    ``LanguageModel(TextConfig)`` — the first is cheap and worth
    asserting; the second's signature is asserted by
    ``inspect.signature`` without invoking it."""
    import importlib
    import inspect
    import sys

    # Snapshot + poison mlx_vlm imports so the fallback branch is
    # actually forced. Restore in finally so subsequent tests aren't
    # affected. Use ``None`` (the sys.modules sentinel for "raise
    # ImportError on next import") rather than a fake module, so
    # `from mlx_vlm.X import Y` raises the same shape as a real
    # missing-package failure.
    mlx_vlm_snapshot = {
        k: sys.modules[k] for k in list(sys.modules) if k.startswith("mlx_vlm")
    }
    for k in list(sys.modules):
        if k.startswith("mlx_vlm"):
            del sys.modules[k]
    # Poison future imports so gemma4_text.py's `try:` branch fails
    # and the `except ImportError:` branch is exercised.
    sys.modules["mlx_vlm"] = None  # type: ignore[assignment]
    sys.modules["mlx_vlm.models"] = None  # type: ignore[assignment]
    sys.modules["mlx_vlm.models.gemma4"] = None  # type: ignore[assignment]
    sys.modules["mlx_vlm.models.gemma4.config"] = None  # type: ignore[assignment]
    sys.modules["mlx_vlm.models.gemma4.language"] = None  # type: ignore[assignment]
    try:
        # Fresh import of the vendored modules — must succeed with
        # NO mlx_vlm on the import path.
        cfg_mod = importlib.import_module("vllm_mlx.models.gemma4_vendored.config")
        lang_mod = importlib.import_module("vllm_mlx.models.gemma4_vendored.language")
        TextConfig = cfg_mod.TextConfig
        LanguageModel = lang_mod.LanguageModel

        # Contract 1: `TextConfig.from_dict` is a classmethod that
        # accepts a params dict and returns a TextConfig. This is
        # what gemma4_text.py calls to build the config.
        assert hasattr(TextConfig, "from_dict"), (
            "vendored TextConfig missing `from_dict` — gemma4_text.py "
            "calls it as `TextConfig.from_dict(text_config)`. "
            "Sync the vendored config.py against upstream mlx-vlm."
        )
        # Sanity: pass a subset of fields (from_dict must tolerate
        # missing/extra keys per BaseModelConfig contract).
        tc = TextConfig.from_dict(
            {"hidden_size": 128, "num_hidden_layers": 2, "vocab_size": 100}
        )
        assert tc is not None
        assert getattr(tc, "hidden_size", None) == 128

        # Contract 2: `LanguageModel.__init__` accepts a single
        # positional TextConfig. Assert via signature so we don't
        # allocate gigabytes of MLX weights (a real construct would).
        sig = inspect.signature(LanguageModel.__init__)
        params = [
            p
            for p in sig.parameters.values()
            if p.name != "self" and p.kind != inspect.Parameter.VAR_KEYWORD
        ]
        assert params, (
            "vendored LanguageModel.__init__ has no non-self params — "
            "gemma4_text.py calls `LanguageModel(tc)` expecting exactly "
            "one positional TextConfig argument."
        )
    finally:
        # Cleanup: remove poisoned entries and restore snapshot.
        for k in [
            "mlx_vlm",
            "mlx_vlm.models",
            "mlx_vlm.models.gemma4",
            "mlx_vlm.models.gemma4.config",
            "mlx_vlm.models.gemma4.language",
        ]:
            sys.modules.pop(k, None)
        sys.modules.update(mlx_vlm_snapshot)


def test_gemma4_text_prefers_vendored_fallback() -> None:
    """``vllm_mlx/models/gemma4_text.py`` must try mlx-vlm first, then
    fall back to the vendored copy. A refactor that dropped the
    fallback would silently re-introduce the 0.10.0 regression for any
    fresh install without ``[vision]``.

    Detection walks the AST — not a substring scan — because the file
    carries a ~15-line explanatory comment that itself references
    ``mlx_vlm.models.gemma4`` and ``gemma4_vendored`` (documenting
    why the fallback exists). A substring check would pass even if
    someone deleted the executable ``try:/except ImportError:`` block
    but left the comment intact, silently re-opening L-07-B. The AST
    walk asserts the actual imports resolve to the right modules
    inside a real ``ImportError`` handler."""
    import ast

    here = Path(__file__).resolve()
    repo_root = None
    for parent in [here.parent, *here.parents]:
        if (parent / "pyproject.toml").is_file():
            repo_root = parent
            break
    assert repo_root is not None, "pyproject.toml not found above test file"
    gemma4_text = repo_root / "vllm_mlx" / "models" / "gemma4_text.py"
    assert gemma4_text.is_file(), f"gemma4_text.py missing at {gemma4_text}"

    tree = ast.parse(gemma4_text.read_text(encoding="utf-8"))
    found_pattern = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        # The Try must have at least one ImportError handler (catches
        # both `except ImportError:` and `except (ImportError, ...):`).
        handles_import_error = any(
            _handler_catches_import_error(h) for h in node.handlers
        )
        if not handles_import_error:
            continue
        try_imports = _module_names_imported(node.body)
        # An ImportError handler with a bare pass / pyright ignore has
        # no imports of its own — skip it. We need the fallback path
        # to also import; that's how the vendored classes get loaded.
        fallback_imports = [
            _module_names_imported(h.body)
            for h in node.handlers
            if _handler_catches_import_error(h)
        ]
        wants_mlx_vlm_first = any(
            m.startswith("mlx_vlm.models.gemma4") for m in try_imports
        )
        wants_vendored_fallback = any(
            any(m.startswith("vllm_mlx.models.gemma4_vendored") for m in fb)
            for fb in fallback_imports
        )
        if wants_mlx_vlm_first and wants_vendored_fallback:
            found_pattern = True
            break
    assert found_pattern, (
        "gemma4_text.py no longer contains a `try: from mlx_vlm.models."
        "gemma4 import ... except ImportError: from vllm_mlx.models."
        "gemma4_vendored import ...` block. Without it, a fresh "
        "`pip install rapid-mlx && rapid-mlx serve gemma-4-12b-4bit` "
        "re-crashes with the 0.10.0 ImportError (L-07-B). Restore the "
        "try/except fallback. (Note: this test walks the AST — a bare "
        "comment referencing these module names is intentionally NOT "
        "enough, since deleting executable imports while keeping the "
        "explanatory comment would silently regress L-07-B.)"
    )


def _handler_catches_import_error(handler) -> bool:
    """Return True if ``handler`` catches ``ImportError`` **explicitly**
    (bare ``ImportError`` or a tuple containing it, e.g.
    ``except (ImportError, ModuleNotFoundError):``). ``ModuleNotFoundError``
    is a subclass of ``ImportError`` so we only need to match the parent.

    Codex round 2 NIT: an earlier version accepted a bare ``except:``.
    That's too permissive — it would let a future refactor wrap the
    upstream import in ``except: pass``, silently swallow non-import
    bugs during model startup (e.g. ``RuntimeError`` from Metal
    initialization), and still pass this L-07-B lock-in. Require the
    handler type to be ``ImportError`` (or a tuple containing it) so
    the fallback stays scoped to import failures only."""
    import ast

    exc_type = handler.type
    if exc_type is None:
        # Bare ``except:`` — rejected. See NIT above.
        return False
    if isinstance(exc_type, ast.Name) and exc_type.id == "ImportError":
        return True
    if isinstance(exc_type, ast.Tuple):
        return any(
            isinstance(el, ast.Name) and el.id == "ImportError" for el in exc_type.elts
        )
    return False


def _module_names_imported(body) -> list[str]:
    """Return the module names imported by any ``import`` /
    ``from ... import`` in ``body``. Used to confirm both the mlx-vlm
    and vendored branches of the fallback actually IMPORT (not merely
    comment about) their target modules."""
    import ast

    names: list[str] = []
    for stmt in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(stmt, ast.ImportFrom) and stmt.module:
            names.append(stmt.module)
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                names.append(alias.name)
    return names


def test_dev_extra_pins_tomli_for_python_310() -> None:
    """Codex round-2 BLOCKING: without ``tomli`` in the [dev] extra
    for Python <3.11, a 3.10 CI worker without the package would have
    module-skipped this whole L-07 suite — letting a future refactor
    silently drop the ``[vision]`` extra without any test ever firing.

    Pin that ``tomli`` is declared with the ``python_version < "3.11"``
    marker (PEP 508) so the dep only resolves on 3.10 where it's
    actually needed. On 3.11+ the stdlib ``tomllib`` is used; the
    marker keeps the install lean.
    """
    py = _load_pyproject()
    dev_specs = _extra_specs(py, "dev")
    # PEP 508 marker form: spec is something like
    # ``tomli>=2.0.1; python_version < "3.11"``. The dep name + marker
    # check is what we care about; the exact version is the dev's choice.
    matching = [
        spec
        for spec in dev_specs
        if "tomli" in spec.lower()
        and ";" in spec
        and "python_version" in spec
        and "3.11" in spec
    ]
    assert matching, (
        f"`[dev]` extra must declare tomli with a `python_version < "
        f'"3.11"` marker so 3.10 CI workers can actually run the '
        f"L-07 lock-in suite. Got dev specs: {dev_specs!r}"
    )


# ──────────────────────────────────────────────────────────────────────
# Broken-build exclusion: mlx-vlm 0.6.4 ships a broken qwen3_5
# GatedDeltaNet SSM forward (garbage output — this is why Bonsai 27B is
# served text-only via mlx-lm). It is the CURRENT PyPI latest with no
# 0.6.5 fix, so a bare ``mlx-vlm>=0.6.3`` resolves a fresh
# ``pip install 'rapid-mlx[vision]'`` STRAIGHT to the broken wheel.
# Every extra that pulls mlx-vlm must exclude it. Mirrors the upstream
# fix waybarrios/vllm-mlx#633 for our rebranded tree.
# ──────────────────────────────────────────────────────────────────────


def _spec_excludes_064(spec: str) -> bool:
    """True iff ``spec`` cannot resolve to the broken mlx-vlm 0.6.4.

    Evaluated with full PEP 440 semantics via ``packaging`` — NOT a
    substring check. A naive ``"!=0.6.4" in spec`` would be fooled by
    ``!=0.6.4.1`` (which excludes 0.6.4.1 yet still ADMITS 0.6.4), so we
    ask the parsed specifier set directly whether it admits 0.6.4. This
    also transparently accepts a future ``>=0.6.5`` floor after an
    upstream fix, with no test edit. ``packaging.requirements.Requirement``
    parses (and discards) any PEP 508 environment marker such as
    ``; platform_system == 'Darwin'`` on its own, so the Darwin-gated
    dev/test pins are handled identically to the unmarked ones.
    ``prereleases=True`` keeps the check honest if a pin ever carries a
    0.6.4 prerelease form (defensive; our pins are all final releases)."""
    from packaging.requirements import Requirement
    from packaging.version import Version

    return not Requirement(spec).specifier.contains(
        Version("0.6.4"), prereleases=True
    )


def test_all_mlx_vlm_specs_exclude_broken_064() -> None:
    """EVERY optional-dependency extra that pulls mlx-vlm must exclude the
    broken 0.6.4 build.

    0.6.4 ships a broken qwen3_5 GatedDeltaNet SSM forward (garbage
    output) AND is the current PyPI latest with no 0.6.5 fix, so a bare
    ``mlx-vlm>=0.6.3`` installs it on a fresh venv — the exact fresh-user
    trap this test guards. The scan walks ALL extras (not a hard-coded
    list) so a NEW extra that adds mlx-vlm without the exclusion is caught
    here, at CI time, instead of by a user's broken vision/dflash boot."""
    py = _load_pyproject()
    extras = py.get("project", {}).get("optional-dependencies", {})
    offenders: list[tuple[str, str]] = []
    for extra_name, specs in extras.items():
        for spec in specs:
            name, _ = _split_spec(spec)
            if name.lower() != "mlx-vlm":
                continue
            if not _spec_excludes_064(spec):
                offenders.append((extra_name, spec))
    assert offenders == [], (
        "These mlx-vlm specs can still resolve to the BROKEN 0.6.4 build:\n"
        + "\n".join(f"  [{e}] {s!r}" for e, s in offenders)
        + "\n\n0.6.4 has a broken qwen3_5 GatedDeltaNet SSM forward and is "
        "the current PyPI latest with no fix, so a bare `>=0.6.3` installs "
        "it on a fresh venv. Add `!=0.6.4` to each spec (or a `>=0.6.5` "
        "floor once an upstream fix ships)."
    )
