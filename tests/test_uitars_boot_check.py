# SPDX-License-Identifier: Apache-2.0
"""R-10 — boot-time guard for UI-TARS / VLM aliases when ``mlx-vlm`` missing.

PyPI 0.8.6 dogfood: a first-time ``pip install rapid-mlx`` user (no
``[vision]`` extra) running ``rapid-mlx serve ui-tars-1.5-7b-4bit``
crashed deep inside the engine's MLLM load path with a confusing
``ImportError: No module named 'mlx_vlm'`` traceback — minutes of
wall-clock noise (alias resolution + weight download via the R2
mirror) before the actionable error surfaced.

Fix shape (matches the ``--embedding-model`` /
:func:`require_mlx_embeddings_or_exit` pattern at the top of
``serve_command``):

* Add :func:`mlx_vlm_available` (importlib-spec probe, no actual import).
* Add :func:`require_mlx_vlm_or_exit` — print actionable install hint to
  stderr and ``sys.exit(2)`` (argparse usage-error code).
* Wire it into ``cli.serve_command`` BEFORE the multi-minute model
  download, gated on ``is_mllm_model(args.model) or args.mllm`` and
  bypassable via the existing ``--no-mllm`` escape hatch.

This module pins:

* The probe returns False when ``mlx_vlm`` is masked from importlib.
* The probe returns True when ``mlx_vlm`` is installed (no-op in
  installed envs).
* The exit helper prints the actionable hint AND raises SystemExit(2).
* The hint message names ``rapid-mlx[vision]`` so the user can copy-
  paste straight from stderr.
* The hint message names the offending model id so the user knows
  WHY the boot failed (not just "vision needed").
* :func:`_require_mlx_vlm` (the engine-side last-line-of-defence)
  still raises a friendly :class:`ImportError` when called directly,
  so library callers that skipped the CLI guard still get an
  actionable message.
"""

from __future__ import annotations

import importlib
import sys

import pytest


def _mask_mlx_vlm(monkeypatch):
    """Make ``importlib.util.find_spec("mlx_vlm")`` return None and
    any direct ``import mlx_vlm`` raise ``ImportError`` — the exact
    state of a fresh ``pip install rapid-mlx`` without the
    ``[vision]`` extra."""
    real_find_spec = importlib.util.find_spec

    def _fake_find_spec(name, *args, **kwargs):
        if name == "mlx_vlm" or name.startswith("mlx_vlm."):
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)
    # Also poison sys.modules so a direct import would fail. We can't
    # just delete mlx_vlm from sys.modules (it may not be present), so
    # we install a finder that refuses it.

    class _BlockMlxVlm:
        def find_spec(self, name, path=None, target=None):
            if name == "mlx_vlm" or name.startswith("mlx_vlm."):
                raise ImportError("mocked: mlx_vlm masked for R-10 test")
            return None

    # Insert before any other finders so our refusal wins.
    monkeypatch.setattr(
        sys, "meta_path", [_BlockMlxVlm(), *sys.meta_path], raising=False
    )
    # Drop any cached entry so a fresh import is forced.
    sys.modules.pop("mlx_vlm", None)


def test_mlx_vlm_available_returns_false_when_missing(monkeypatch):
    """R-10 probe must return False on a base install with no extras."""
    from vllm_mlx.models.mllm import mlx_vlm_available

    _mask_mlx_vlm(monkeypatch)
    assert mlx_vlm_available() is False


def test_require_mlx_vlm_or_exit_prints_hint_and_exits(
    monkeypatch, capsys
):
    """R-10 fix: boot guard must emit the actionable install hint to
    stderr and ``sys.exit(2)`` — same shape as
    :func:`require_mlx_embeddings_or_exit`."""
    from vllm_mlx.models.mllm import require_mlx_vlm_or_exit

    _mask_mlx_vlm(monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        require_mlx_vlm_or_exit("ui-tars-1.5-7b-4bit")

    assert exc_info.value.code == 2

    captured = capsys.readouterr()
    err = captured.err
    # User-facing hint must include the canonical install command,
    # the offending model name, and a "vision" marker so the message
    # is searchable in support threads.
    assert "ui-tars-1.5-7b-4bit" in err, (
        f"hint must name the offending model id, got: {err!r}"
    )
    assert "rapid-mlx[vision]" in err or "rapid-mlx[vision]" in err.replace(
        "'", ""
    ), f"hint must name the [vision] extra install path, got: {err!r}"
    assert "mlx-vlm" in err, f"hint must name the dep, got: {err!r}"


def test_require_mlx_vlm_or_exit_is_noop_when_installed(monkeypatch):
    """When ``mlx_vlm`` IS installed (or its spec is fakable), the
    guard returns silently — no SystemExit, no stderr output."""
    from vllm_mlx.models.mllm import require_mlx_vlm_or_exit

    # Force the probe to report True without touching sys.modules.
    monkeypatch.setattr(
        "vllm_mlx.models.mllm.mlx_vlm_available", lambda: True
    )

    # Must not raise SystemExit.
    require_mlx_vlm_or_exit("ui-tars-1.5-7b-4bit")


def test_engine_side_require_mlx_vlm_still_raises_importerror(
    monkeypatch,
):
    """The pre-existing engine-side guard (:func:`_require_mlx_vlm`)
    must remain an ``ImportError`` raiser — library callers that
    skip the CLI guard (test harnesses, direct ``MLXMultimodalLM``
    use) still need an actionable message rather than a raw
    ``ModuleNotFoundError`` from deep inside the load path."""
    from vllm_mlx.models.mllm import _require_mlx_vlm

    _mask_mlx_vlm(monkeypatch)

    with pytest.raises(ImportError) as exc_info:
        _require_mlx_vlm()

    # Same actionable hint surface as the CLI guard.
    msg = str(exc_info.value)
    assert "rapid-mlx[vision]" in msg
    assert "mlx-vlm" in msg


def test_vision_extra_pulls_in_mlx_vlm():
    """L-07 lock-in: the ``[vision]`` extra in pyproject.toml MUST
    declare ``mlx-vlm`` so ``pip install rapid-mlx[vision]`` is a
    complete fix without needing a separate ``pip install mlx-vlm``."""
    try:
        import tomllib
    except ImportError:  # pragma: no cover — Python 3.10 path
        import tomli as tomllib

    from pathlib import Path

    pyproject_path = (
        Path(__file__).resolve().parent.parent / "pyproject.toml"
    )
    data = tomllib.loads(pyproject_path.read_text())

    extras = data["project"]["optional-dependencies"]
    assert "vision" in extras, "[vision] extra missing entirely"
    vision_deps = " ".join(extras["vision"])
    assert "mlx-vlm" in vision_deps, (
        "[vision] extra must declare mlx-vlm so the install hint in "
        "require_mlx_vlm_or_exit actually fixes the missing dep — "
        f"got vision = {extras['vision']!r}"
    )
    # `[all]` mirrors `[vision]` content directly (the comment in
    # pyproject explains why a recursive self-dep breaks editable
    # installs); the `[all]` extra must also include mlx-vlm so the
    # ``pip install rapid-mlx[all]`` shortcut covers UI-TARS users.
    all_deps = " ".join(extras["all"])
    assert "mlx-vlm" in all_deps, (
        "[all] extra must also include mlx-vlm — pre-fix this drift "
        "was the secondary footgun (users following the docs to "
        "`pip install rapid-mlx[all]` got everything EXCEPT the vision "
        f"runtime). got all = {extras['all']!r}"
    )
