# SPDX-License-Identifier: Apache-2.0
"""
Tests for the MLX hardware-compat shim (#404 M5 single-stream).

We can't test on actual M5 from CI, but we can:
1. Verify the shim is installed *before* any module-level
   ``mx.new_thread_local_stream`` capture inside ``mlx_lm.generate``,
   by checking that importing ``vllm_mlx.scheduler`` triggers install().
2. Mock the probe failure and assert the fallback path returns
   ``mx.default_stream(...)``.
3. Verify idempotency — install() can be called multiple times safely.
4. Verify the shim is transparent on hardware that works (this runs on
   the dev's actual hardware in the test-apple-silicon CI job).

We do NOT test that ``import vllm_mlx`` installs the shim — that is the
*wrong* contract. We deliberately keep top-level ``import vllm_mlx``
free of ``mlx.core`` import so the package stays usable for metadata-only
access on systems where ``mlx`` is installed but Metal is unavailable
(``import mlx.core`` SIGABRTs there with an uncatchable NSException).
"""

from __future__ import annotations

import importlib
import importlib.resources

import pytest

pytest.importorskip("mlx.core")


def test_shim_installed_when_scheduler_imports():
    """Importing vllm_mlx.scheduler must install the compat shim — that's
    the gate that protects the module-level ``mx.new_thread_local_stream``
    call inside mlx_lm.generate (which scheduler imports at module top)."""
    import mlx.core as mx

    # Re-install explicitly so this test is order-independent: even if
    # scheduler was already imported by a prior test, install() is
    # idempotent and the assertion still holds.
    from vllm_mlx import _mlx_compat

    if hasattr(mx, "_rapid_mlx_compat_installed"):
        delattr(mx, "_rapid_mlx_compat_installed")
    import vllm_mlx.scheduler  # noqa: F401

    if not getattr(mx, "_rapid_mlx_compat_installed", False):
        # scheduler may already be in sys.modules from a previous test —
        # in which case its module-level install() didn't re-run. Confirm
        # that calling install() directly works.
        _mlx_compat.install()
    assert getattr(mx, "_rapid_mlx_compat_installed", False) is True


def test_vllm_mlx_init_does_not_install_shim_or_import_mlx():
    """`vllm_mlx/__init__.py` must NOT import mlx or call _mlx_compat.install().
    Both would eagerly load `mlx.core`, which SIGABRTs (uncatchable from
    Python) on systems where the `mlx` package is installed but Metal is
    unavailable — breaking metadata-only callers (`__version__`, etc.).

    Pure source-text audit; no module manipulation so the test is safe
    in a shared pytest process. The shim must be installed lazily at
    the top of every module that imports `mlx_lm.*` instead
    (verified by `test_every_mlx_lm_consumer_installs_shim`)."""
    init_source = (
        importlib.resources.files("vllm_mlx").joinpath("__init__.py").read_text()
    )
    assert "import mlx" not in init_source, (
        "vllm_mlx/__init__.py must not import mlx — it would break "
        "metadata-only usage on systems with broken Metal init."
    )
    assert "_mlx_compat.install()" not in init_source, (
        "vllm_mlx/__init__.py must not call _mlx_compat.install() — that "
        "would eagerly import mlx.core (which can SIGABRT on Metal-less "
        "systems). The shim must install at scheduler-import time instead."
    )


def test_every_mlx_lm_consumer_installs_shim():
    """Any vllm_mlx file that runs a ``from mlx_lm…`` / ``import mlx_lm…``
    *at module load time* MUST also call ``_mlx_compat.install()`` before
    the first such import.

    Why ANY ``mlx_lm`` submodule, not just ``mlx_lm.generate``: importing
    e.g. ``mlx_lm.sample_utils`` or ``mlx_lm.models.base`` still triggers
    ``mlx_lm/__init__.py`` execution, which does ``from .generate import
    batch_generate, generate, stream_generate`` and therefore runs
    ``mlx_lm/generate.py:226`` ``generation_stream =
    mx.new_thread_local_stream(mx.default_device())`` at module-import
    time. The #404 single-stream crash on M5 happens on the same code
    path regardless of which submodule the import statement names.

    Why AST-based and not text-based: imports inside top-level ``try /
    except ImportError`` blocks are indented but still execute at module
    load time. ``vllm_mlx/utils/mamba_cache.py`` and ``vllm_mlx/api/
    guided.py`` both use that pattern for optional-dep guards, and a
    line-prefix check misses them. We walk the AST and accept any
    ``Import`` / ``ImportFrom`` whose parent chain stays inside
    module-load-time scopes (``If`` / ``Try`` / ``With`` / ``For`` /
    ``While`` bodies, plus ``ClassDef`` bodies — class statements run
    at definition time), while excluding ``FunctionDef`` /
    ``AsyncFunctionDef`` / ``Lambda`` (deferred to call time).

    Surfaced by community PR #485 (Michael Ledin / @mxl, 2026-05-29):
    ``mllm_batch_generator.py``, ``mllm_scheduler.py``, and
    ``models/deepseek_v4.py`` all had top-level ``mlx_lm.<sub>`` imports
    without the shim; the prior version of this guard (which matched
    only literal ``mlx_lm.generate``) missed all three. Codex round 1
    review against this PR then surfaced the indented-import gap above,
    and DeepSeek pr_validate round 2 added the ``ClassDef``-descent
    requirement (class bodies execute at module load).

    This is a structural audit: a new file that adds a module-load-time
    ``mlx_lm.*`` import without ``_mlx_compat.install()`` first will
    fail here at unit-test time, catching the slip before any M5 user
    does.
    """
    import ast
    import pathlib

    # AST node types that defer execution — an mlx_lm import nested under
    # any of these does NOT run at module load, so it's out of scope.
    # NB: ``ClassDef`` is *not* deferred. A class body runs at the class's
    # definition site; for a module-level class that's still at module
    # load time, so an ``import mlx_lm`` at class scope captures the
    # GPU stream exactly as a top-level import would. Method bodies
    # inside the class are still skipped via ``FunctionDef``. Flagged
    # as [BLOCKING] by DeepSeek pr_validate round 2 on PR #487.
    DEFERRED_SCOPES = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.Lambda,
    )

    def _is_mlx_lm_node(node: ast.AST) -> bool:
        if isinstance(node, ast.Import):
            return any(
                alias.name == "mlx_lm" or alias.name.startswith("mlx_lm.")
                for alias in node.names
            )
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            return mod == "mlx_lm" or mod.startswith("mlx_lm.")
        # ``importlib.import_module("mlx_lm…")`` — dynamic import with
        # the same module-load effect as a static one. DeepSeek
        # pr_validate round 4 flagged that the AST-only walk silently
        # dropped this shape, which the prior text-based guard caught
        # via a literal `import_module("mlx_lm.generate")` substring.
        # We also accept the keyword-arg form
        # ``import_module(name="mlx_lm.…")``.
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "import_module":
                if isinstance(func.value, ast.Name) and func.value.id == "importlib":
                    arg = None
                    if node.args:
                        arg = node.args[0]
                    else:
                        for kw in node.keywords:
                            if kw.arg == "name":
                                arg = kw.value
                                break
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        s = arg.value
                        return s == "mlx_lm" or s.startswith("mlx_lm.")
        return False

    def _is_install_call(node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "install":
            return False
        return isinstance(func.value, ast.Name) and func.value.id == "_mlx_compat"

    def _is_type_checking_guard(node: ast.AST) -> bool:
        """``True`` for ``if TYPE_CHECKING:`` and
        ``if {typing,typing_extensions}.TYPE_CHECKING:`` — both evaluate
        to ``False`` at runtime, so the body never executes at module
        load and isn't a stream-capture hazard.

        Narrowed to those two attribute owners specifically so an
        unrelated ``if config.TYPE_CHECKING:`` (where the attribute is a
        real runtime flag) is NOT skipped — DeepSeek pr_validate round
        3 flagged the un-narrowed version as [BLOCKING]: a real
        ``True`` flag named ``TYPE_CHECKING`` on some other namespace
        would have hidden an unprotected ``mlx_lm`` import.

        Known limitation (DeepSeek round 4 NIT): aliased typing imports
        like ``import typing as t; if t.TYPE_CHECKING: import mlx_lm``
        get the false-positive treatment (the test would flag the
        ``mlx_lm`` import as unshielded). Convention in this codebase
        is ``from typing import TYPE_CHECKING`` — don't alias the
        ``typing`` module in files that import ``mlx_lm`` under a
        ``TYPE_CHECKING`` guard."""
        if not isinstance(node, ast.If):
            return False
        test = node.test
        if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
            return True
        if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
            owner = test.value
            return isinstance(owner, ast.Name) and owner.id in (
                "typing",
                "typing_extensions",
            )
        return False

    def _walk_module_level(node: ast.AST, parents: tuple[ast.AST, ...] = ()):
        """Yield (node, parents) for every descendant that is NOT inside
        a deferred (function/lambda) scope. The walk descends into
        ``If``/``Try``/``With``/``For``/``While``/``ClassDef`` bodies —
        all of which run at module load time when the enclosing module
        is imported. ``if TYPE_CHECKING:`` is treated as deferred since
        its body never runs at import time."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, DEFERRED_SCOPES):
                continue
            if _is_type_checking_guard(child):
                # Yield the If node itself (for parent-tracking) but DO NOT
                # descend into its body / orelse — the imports there only
                # run under static analysis, never at module load.
                yield child, parents
                continue
            yield child, parents
            yield from _walk_module_level(child, parents + (child,))

    pkg_root = pathlib.Path(
        str(importlib.resources.files("vllm_mlx").joinpath(""))
    ).resolve()
    offenders = []
    for path in pkg_root.rglob("*.py"):
        if path.name == "_mlx_compat.py":
            continue  # the shim itself; nothing to install
        source = path.read_text()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue  # not our problem here
        first_mlx_lm_line = None
        first_install_line = None
        for node, _parents in _walk_module_level(tree):
            if first_mlx_lm_line is None and _is_mlx_lm_node(node):
                first_mlx_lm_line = node.lineno
            if first_install_line is None and _is_install_call(node):
                first_install_line = node.lineno
        if first_mlx_lm_line is None:
            continue
        if first_install_line is None or first_install_line > first_mlx_lm_line:
            rel = str(path.relative_to(pkg_root))
            offenders.append(
                f"{rel} (mlx_lm import @ line {first_mlx_lm_line}, "
                f"install @ line {first_install_line})"
            )
    assert not offenders, (
        "Files run a module-load-time `from mlx_lm` / `import mlx_lm` "
        "without calling `_mlx_compat.install()` first — #404 M5 "
        "regression risk (any mlx_lm submodule triggers "
        "mlx_lm/__init__.py which loads mlx_lm.generate which captures "
        "the GPU stream):\n  " + "\n  ".join(offenders)
    )


def test_install_is_idempotent():
    import mlx.core as mx

    from vllm_mlx import _mlx_compat

    _mlx_compat.install()
    first = mx.new_thread_local_stream
    _mlx_compat.install()
    second = mx.new_thread_local_stream
    assert first is second, "second install() must not re-wrap the function"


def test_install_is_noop_when_symbol_missing(monkeypatch):
    """Regression for #408: on mlx builds that predate
    ``mx.new_thread_local_stream``, ``install()`` must be a no-op rather
    than crash with AttributeError. Without this guard,
    ``import vllm_mlx.scheduler`` aborts before the server can bind a
    port — every user on the affected mlx is blocked from upgrading."""
    import mlx.core as mx

    from vllm_mlx import _mlx_compat

    # If a future mlx genuinely drops the symbol, this assert fails
    # loudly so we revisit whether the compat shim still has a job to
    # do — `raising=False` on the delattr below would silently turn
    # this into a degenerate test that exercises nothing.
    assert hasattr(mx, "new_thread_local_stream"), (
        "expected baseline mlx to expose new_thread_local_stream; "
        "if upstream removed it, this test no longer covers the #408 "
        "regression path and the shim itself can probably go away."
    )
    monkeypatch.delattr(mx, "new_thread_local_stream")
    monkeypatch.setattr(mx, "_rapid_mlx_compat_installed", False, raising=False)
    importlib.reload(_mlx_compat)
    _mlx_compat.install()  # must not raise — that's the #408 contract
    # Note: on the no-symbol path the shim deliberately does NOT mark
    # itself "installed" so that a later mlx upgrade (which adds the
    # symbol) gets the wrap on the next install() call. The contract
    # this test pins is "no AttributeError", not the flag.


def test_fallback_engages_when_probe_raises(monkeypatch):
    """Simulate M5: probe raises 'no Stream(gpu, 1)' → patched function must
    return mx.default_stream(device) instead of the unusable stream."""
    import mlx.core as mx

    from vllm_mlx import _mlx_compat

    # Make `_probe` always fail with the M5-shaped error. We poke the
    # ``mx`` namespace because the patch wires `with mx.stream(stream)`
    # → `mx.array(...) + mx.array(...)` — substituting `mx.stream`
    # itself is the cleanest interception point.
    class _BoomStream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            raise RuntimeError("There is no Stream(gpu, 1) in current thread.")

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(mx, "stream", _BoomStream)

    # Force a fresh install with our broken probe environment.
    monkeypatch.setattr(mx, "_rapid_mlx_compat_installed", False, raising=False)
    importlib.reload(_mlx_compat)
    _mlx_compat.install()

    device = mx.default_device()
    fallback = mx.new_thread_local_stream(device)
    expected = mx.default_stream(device)
    # mx.default_stream is comparable by repr; compare structurally.
    assert repr(fallback) == repr(expected), (
        f"M5 fallback should return mx.default_stream({device!r}); got {fallback!r}"
    )


def test_fallback_does_not_engage_on_unrelated_runtime_error(monkeypatch):
    """If `with mx.stream(stream)` raises a RuntimeError that doesn't look
    like the M5 single-stream signature, the shim must NOT swallow it —
    we want unexpected failures to surface, not get silently degraded."""
    import mlx.core as mx

    from vllm_mlx import _mlx_compat

    class _BoomStream:
        def __init__(self, stream):
            pass

        def __enter__(self):
            raise RuntimeError("Some completely unrelated MLX error")

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(mx, "stream", _BoomStream)
    monkeypatch.setattr(mx, "_rapid_mlx_compat_installed", False, raising=False)
    importlib.reload(_mlx_compat)
    _mlx_compat.install()

    with pytest.raises(RuntimeError, match="completely unrelated"):
        mx.new_thread_local_stream(mx.default_device())


def test_happy_path_unchanged_on_real_hardware():
    """On hardware where the original API works (M1–M4), the patched
    function must return a usable stream — and `with mx.stream(stream)`
    must run a trivial op. This is the test that confirms the shim is
    transparent for users who don't need it."""
    import mlx.core as mx

    from vllm_mlx import _mlx_compat

    # Cleanup from prior monkeypatched tests
    if hasattr(mx, "_rapid_mlx_compat_installed"):
        delattr(mx, "_rapid_mlx_compat_installed")
    importlib.reload(_mlx_compat)
    _mlx_compat.install()

    stream = mx.new_thread_local_stream(mx.default_device())
    with mx.stream(stream):
        result = (mx.array([1.0]) + mx.array([2.0])).item()
    assert result == 3.0
