# SPDX-License-Identifier: Apache-2.0
"""Regression tests for #1126 (honest vision-runtime detection — the
Homebrew ``pip install --no-deps mlx-vlm`` half).

Root cause: Homebrew installs mlx-vlm with ``pip install --no-deps
mlx-vlm``, so mlx-vlm's source + dist metadata exist but its runtime dep
Pillow (PIL) does NOT. Three separate "is vision available?" checks all
answered YES from metadata/spec:

* ``importlib.metadata.version("mlx-vlm")`` returns a version (doctor's
  ``_safe_version`` → green ``✓ mlx-vlm``),
* ``importlib.util.find_spec("mlx_vlm")`` is not None
  (``mlx_vlm_available`` → boot guard passes),

while the REAL ``import mlx_vlm`` crashes on ``from PIL import Image``
(mlx_vlm/utils.py) — deep inside FastAPI lifespan, with a message telling
the user to install mlx-vlm (which IS installed).

The invariant these tests pin: **when mlx-vlm's metadata/spec is present
but its import chain is broken, every "is vision usable?" surface reports
not-usable and names the actually-missing module (PIL), distinct from the
"mlx-vlm truly absent" message.** They simulate the PIL-missing state via
monkeypatch (find_spec present, ``import mlx_vlm`` raising
``ModuleNotFoundError("No module named 'PIL'")``) so they run on a normal
dev machine where mlx-vlm + PIL are actually installed.

These assertions FAIL against the pre-fix code (find_spec-only /
metadata-only checks report "available"/green) and PASS after.
"""

from __future__ import annotations

import builtins
import importlib.util as _ilu
import sys

import pytest


@pytest.fixture(autouse=True)
def clean_runtime_probe_state(clean_doctor_runtime_state):
    """Keep host server processes out of doctor's runtime selection."""
    yield


# ─────────────────────────────────────────────────────────────────────────
# Simulators
# ─────────────────────────────────────────────────────────────────────────


def _simulate_mlx_vlm_present_but_pil_missing(monkeypatch):
    """Homebrew ``--no-deps`` state: ``find_spec('mlx_vlm')`` is not None
    (dist present) but ``import mlx_vlm`` raises
    ``ModuleNotFoundError("No module named 'PIL'")``."""
    real_find_spec = _ilu.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "mlx_vlm":
            # mlx-vlm's spec/metadata IS present (installed --no-deps).
            spec = real_find_spec(name, *args, **kwargs)
            return spec if spec is not None else object()
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(_ilu, "find_spec", fake_find_spec)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mlx_vlm" or name.startswith("mlx_vlm."):
            # Mirror the real crash inside mlx_vlm/utils.py:
            # ``from PIL import Image`` with PIL uninstalled.
            raise ModuleNotFoundError("No module named 'PIL'", name="PIL")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # ``delitem`` (not ``sys.modules.pop``) so monkeypatch restores any
    # pre-existing cached ``mlx_vlm`` on teardown → order-independent tests.
    monkeypatch.delitem(sys.modules, "mlx_vlm", raising=False)


def _simulate_mlx_vlm_absent(monkeypatch):
    """Plain ``pip install rapid-mlx`` state: mlx-vlm not installed at all
    (``find_spec`` None, import raises for the package itself)."""
    real_find_spec = _ilu.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "mlx_vlm" or name.startswith("mlx_vlm."):
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(_ilu, "find_spec", fake_find_spec)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mlx_vlm" or name.startswith("mlx_vlm."):
            raise ModuleNotFoundError("No module named 'mlx_vlm'", name="mlx_vlm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "mlx_vlm", raising=False)


def _spec_present_for_mlx_vlm(monkeypatch):
    """Make ``find_spec('mlx_vlm')`` report the top-level package present
    (installed) without importing it — shared setup for the "installed but
    the import chain is broken" simulators below."""
    real_find_spec = _ilu.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "mlx_vlm":
            spec = real_find_spec(name, *args, **kwargs)
            return spec if spec is not None else object()
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(_ilu, "find_spec", fake_find_spec)


def _simulate_mlx_vlm_installed_but_import_raises(monkeypatch, exc):
    """mlx-vlm's top-level package IS installed (``find_spec`` present) but
    ``import mlx_vlm`` raises ``exc``.

    Covers the states the pre-fix name-heuristic mishandled:
    * a damaged/incomplete install whose OWN submodule is missing
      (``ModuleNotFoundError(name='mlx_vlm.<sub>')`` → was wrongly ABSENT),
    * a missing shared lib / broken native ext
      (``OSError``/``RuntimeError`` → previously ESCAPED and crashed the guard).
    """
    _spec_present_for_mlx_vlm(monkeypatch)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mlx_vlm" or name.startswith("mlx_vlm."):
            raise exc
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "mlx_vlm", raising=False)


# ─────────────────────────────────────────────────────────────────────────
# Tri-state runtime status: the single source of truth.
# ─────────────────────────────────────────────────────────────────────────


def test_status_broken_when_pil_missing(monkeypatch):
    """mlx-vlm present but PIL missing ⇒ status is the "present-but-broken"
    state and names the missing module (PIL), NOT "absent"."""
    from vllm_mlx.models.mllm import VisionRuntimeStatus, vision_runtime_status

    _simulate_mlx_vlm_present_but_pil_missing(monkeypatch)

    status, missing = vision_runtime_status()
    assert status is VisionRuntimeStatus.BROKEN, (
        f"expected BROKEN (metadata present, import chain broken), got {status}"
    )
    assert missing == "PIL", f"missing module should be PIL, got {missing!r}"


def test_status_absent_when_mlx_vlm_missing(monkeypatch):
    """mlx-vlm not installed at all ⇒ status is "absent" (distinct from
    broken)."""
    from vllm_mlx.models.mllm import VisionRuntimeStatus, vision_runtime_status

    _simulate_mlx_vlm_absent(monkeypatch)

    status, _missing = vision_runtime_status()
    assert status is VisionRuntimeStatus.ABSENT, (
        f"expected ABSENT (mlx-vlm not installed), got {status}"
    )


def test_mlx_vlm_available_false_when_pil_missing(monkeypatch):
    """The boot-guard presence probe must be honest: metadata/spec present
    but a broken import chain ⇒ NOT available. Pre-fix this returned True
    (find_spec-only), so the boot guard passed and the crash surfaced deep
    in FastAPI lifespan."""
    from vllm_mlx.models.mllm import mlx_vlm_available

    _simulate_mlx_vlm_present_but_pil_missing(monkeypatch)

    assert mlx_vlm_available() is False


def test_status_broken_when_internal_submodule_missing(monkeypatch):
    """A damaged/incomplete mlx-vlm whose OWN internal submodule is missing
    (``ModuleNotFoundError(name='mlx_vlm.<sub>')``) is installed-but-broken,
    NOT absent. Pre-fix the ``name.startswith('mlx_vlm')`` heuristic
    mislabelled it ABSENT → misleading "install mlx-vlm" for an install that
    is already present."""
    from vllm_mlx.models.mllm import VisionRuntimeStatus, vision_runtime_status

    _simulate_mlx_vlm_installed_but_import_raises(
        monkeypatch,
        ModuleNotFoundError(
            "No module named 'mlx_vlm.trainer'", name="mlx_vlm.trainer"
        ),
    )

    status, detail = vision_runtime_status()
    assert status is VisionRuntimeStatus.BROKEN, (
        f"present-but-damaged mlx-vlm must be BROKEN, not ABSENT; got {status}"
    )
    assert detail, "BROKEN must carry an actionable diagnostic, not None/empty"


def test_status_broken_when_import_raises_oserror(monkeypatch):
    """A missing shared library surfaces as ``OSError`` from ``import
    mlx_vlm``. Pre-fix this ESCAPED ``vision_runtime_status`` (only
    ModuleNotFoundError/ImportError were caught) and crashed the boot guard;
    it must now be classified BROKEN without raising."""
    from vllm_mlx.models.mllm import VisionRuntimeStatus, vision_runtime_status

    _simulate_mlx_vlm_installed_but_import_raises(
        monkeypatch, OSError("dlopen(libmlx.dylib): image not found")
    )

    status, detail = vision_runtime_status()  # must NOT raise
    assert status is VisionRuntimeStatus.BROKEN, (
        f"OSError on import ⇒ installed-but-broken, got {status}"
    )
    assert detail, "BROKEN must retain a diagnostic for the OSError failure"


def test_status_broken_when_import_raises_runtimeerror(monkeypatch):
    """A broken native extension / ABI mismatch can raise ``RuntimeError``
    on import — also non-control-flow, also previously escaping. Must be
    BROKEN, not a crash."""
    from vllm_mlx.models.mllm import VisionRuntimeStatus, vision_runtime_status

    _simulate_mlx_vlm_installed_but_import_raises(
        monkeypatch, RuntimeError("incompatible mlx ABI")
    )

    status, detail = vision_runtime_status()  # must NOT raise
    assert status is VisionRuntimeStatus.BROKEN
    assert detail


def test_boot_guard_exit_2_when_import_raises_oserror(monkeypatch, capsys):
    """The tri-state contract's whole point: a missing-shared-lib ``OSError``
    must be caught and turned into a clean exit-code-2 boot guard with an
    honest diagnostic — NOT propagate as an uncaught crash, and NOT misdirect
    the user to "install mlx-vlm" (which IS installed)."""
    from vllm_mlx.models.mllm import require_mlx_vlm_or_exit

    _simulate_mlx_vlm_installed_but_import_raises(
        monkeypatch, OSError("dlopen(libmlx.dylib): image not found")
    )

    with pytest.raises(SystemExit) as exc_info:
        require_mlx_vlm_or_exit("gemma-4-26b-a4b-it-4bit")
    assert exc_info.value.code == 2

    err = capsys.readouterr().err
    assert "vision runtime cannot load" in err, (
        f"broken-runtime boot hint must say the runtime can't load, got: {err!r}"
    )
    # Honest: the primary directive must NOT be the misleading bare
    # "install mlx-vlm" absent message — it IS installed.
    assert "requires the optional `mlx-vlm` dependency" not in err


# ─────────────────────────────────────────────────────────────────────────
# Engine-side guard: message names PIL, distinct from the absent message.
# ─────────────────────────────────────────────────────────────────────────


def test_require_mlx_vlm_message_names_pil_when_broken(monkeypatch):
    """``_require_mlx_vlm`` (engine-side last line of defence) must raise an
    ImportError that names the actually-missing module (PIL) rather than
    telling the user to install mlx-vlm (which IS installed)."""
    from vllm_mlx.models.mllm import _require_mlx_vlm

    _simulate_mlx_vlm_present_but_pil_missing(monkeypatch)

    with pytest.raises(ImportError) as exc_info:
        _require_mlx_vlm()
    msg = str(exc_info.value)
    assert "PIL" in msg, f"broken-runtime message must name PIL, got: {msg!r}"


def test_broken_and_absent_messages_are_distinct(monkeypatch):
    """The two failure modes must NOT collapse into the same message: the
    broken message names the missing runtime dep (PIL); the absent message
    tells the user to install mlx-vlm. A user staring at 'install mlx-vlm'
    while mlx-vlm IS installed is the exact #1126 confusion."""
    from vllm_mlx.models.mllm import _require_mlx_vlm

    _simulate_mlx_vlm_present_but_pil_missing(monkeypatch)
    with pytest.raises(ImportError) as broken_exc:
        _require_mlx_vlm()
    broken_msg = str(broken_exc.value)

    _simulate_mlx_vlm_absent(monkeypatch)
    with pytest.raises(ImportError) as absent_exc:
        _require_mlx_vlm()
    absent_msg = str(absent_exc.value)

    assert broken_msg != absent_msg, (
        "broken and absent messages must differ so the user can tell "
        "'install pillow' apart from 'install mlx-vlm'"
    )
    assert "PIL" in broken_msg and "PIL" not in absent_msg, (
        f"only the broken message should name PIL. broken={broken_msg!r} "
        f"absent={absent_msg!r}"
    )


def test_boot_guard_exit_message_names_pil_when_broken(monkeypatch, capsys):
    """``require_mlx_vlm_or_exit`` must preserve its exit-code-2 shape AND
    name PIL in the broken case so the operator sees the actionable hint on
    stderr before any download starts."""
    from vllm_mlx.models.mllm import require_mlx_vlm_or_exit

    _simulate_mlx_vlm_present_but_pil_missing(monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        require_mlx_vlm_or_exit("gemma-4-26b-a4b-it-4bit")
    assert exc_info.value.code == 2

    err = capsys.readouterr().err
    assert "PIL" in err or "Pillow" in err, (
        f"broken-runtime boot hint must name Pillow/PIL, got: {err!r}"
    )
    # Still points at the fix.
    assert "rapid-mlx[vision]" in err or "pillow" in err.lower()


# ─────────────────────────────────────────────────────────────────────────
# Doctor: the vision + dflash rows must not be green when mlx-vlm metadata
# is present but PIL is missing.
# ─────────────────────────────────────────────────────────────────────────


def _find_row(section, *needles):
    for check in section.checks:
        low = check.label.lower()
        if all(n.lower() in low for n in needles):
            return check
    return None


def test_doctor_vision_row_not_ok_and_names_pil_when_pil_missing(monkeypatch):
    """``rapid-mlx doctor`` must NOT show a green ``✓ mlx-vlm (vision
    extras)`` when mlx-vlm metadata is present but the import chain is
    broken (PIL missing). It must WARN/FAIL and name PIL/Pillow.

    The check stays FAST (≤5 s doctor contract) — a lightweight
    ``find_spec('PIL')`` probe, not a heavy ``import mlx_vlm`` that would
    pull torch."""
    from vllm_mlx.doctor import env_health
    from vllm_mlx.doctor.env_health import CheckStatus

    # mlx-vlm metadata present (Homebrew --no-deps), PIL not importable.
    def fake_safe_version(dist, runtime=None):
        if dist == "mlx-vlm":
            return "0.6.17"
        return None

    monkeypatch.setattr(env_health, "_safe_version", fake_safe_version)
    monkeypatch.setattr(
        env_health,
        "_pil_importable",
        lambda runtime=None, packages=None: False,
    )

    section = env_health.section_optional_packages()

    vision_row = _find_row(section, "mlx-vlm", "vision")
    assert vision_row is not None, "vision (mlx-vlm) row missing from doctor"
    assert vision_row.status is not CheckStatus.OK, (
        f"vision row must NOT be green when PIL is missing, got {vision_row.status}: "
        f"{vision_row.label!r}"
    )
    assert "pil" in vision_row.label.lower() or "pillow" in vision_row.label.lower(), (
        f"vision row must name PIL/Pillow so the user knows the real gap, "
        f"got: {vision_row.label!r}"
    )

    dflash_row = _find_row(section, "dflash")
    assert dflash_row is not None, "dflash row missing from doctor"
    assert dflash_row.status is not CheckStatus.OK, (
        f"dflash row must NOT be green when PIL missing, got {dflash_row.status}"
    )
    assert "pil" in dflash_row.label.lower() or "pillow" in dflash_row.label.lower(), (
        f"dflash row must name PIL/Pillow, got: {dflash_row.label!r}"
    )


def test_doctor_vision_row_ok_when_pil_present(monkeypatch):
    """Guard against over-warning: when BOTH mlx-vlm metadata AND PIL are
    present the vision + dflash rows stay green — the honest-detection fix
    must not turn a healthy vision install red."""
    from vllm_mlx.doctor import env_health
    from vllm_mlx.doctor.env_health import CheckStatus

    def fake_safe_version(dist, runtime=None):
        if dist == "mlx-vlm":
            return "0.6.17"
        return None

    monkeypatch.setattr(env_health, "_safe_version", fake_safe_version)
    monkeypatch.setattr(
        env_health,
        "_pil_importable",
        lambda runtime=None, packages=None: True,
    )
    monkeypatch.setattr(
        env_health,
        "_module_visibility",
        lambda dist, runtime=None: (True, True),
    )

    section = env_health.section_optional_packages()
    vision_row = _find_row(section, "mlx-vlm", "vision")
    assert vision_row is not None
    assert vision_row.status is CheckStatus.OK, (
        f"vision row should be green when mlx-vlm + PIL both present, got "
        f"{vision_row.status}: {vision_row.label!r}"
    )
    dflash_row = _find_row(section, "dflash")
    assert dflash_row is not None
    assert dflash_row.status is CheckStatus.OK


def test_doctor_vision_row_warns_when_mlx_vlm_truly_absent(monkeypatch):
    """The truly-absent path must still WARN as before (not silently pass,
    not spuriously name PIL). Preserves the existing text-only install
    messaging."""
    from vllm_mlx.doctor import env_health
    from vllm_mlx.doctor.env_health import CheckStatus

    # mlx-vlm not installed at all.
    monkeypatch.setattr(env_health, "_safe_version", lambda dist, runtime=None: None)
    # PIL state is irrelevant when mlx-vlm itself is absent, but pin it
    # present so a spurious PIL mention can't sneak in via the absent path.
    monkeypatch.setattr(
        env_health,
        "_pil_importable",
        lambda runtime=None, packages=None: True,
    )
    monkeypatch.setattr(
        env_health,
        "_visible_without_metadata",
        lambda dist, runtime=None: False,
    )
    monkeypatch.setattr(
        env_health,
        "_module_visibility",
        lambda dist, runtime=None: (False, False),
    )

    section = env_health.section_optional_packages()
    vision_row = _find_row(section, "mlx-vlm", "vision")
    assert vision_row is not None
    assert vision_row.status is CheckStatus.WARN, (
        f"truly-absent mlx-vlm must WARN, got {vision_row.status}"
    )
    assert "not installed" in vision_row.label.lower()


# ─────────────────────────────────────────────────────────────────────────
# Doctor's PIL probe must reflect the REAL import, not just find_spec.
# ─────────────────────────────────────────────────────────────────────────


def _simulate_pillow_damaged(monkeypatch):
    """Shadowed/damaged Pillow: a ``PIL`` spec is discoverable but the real
    ``from PIL import Image`` raises — the exact false-green a
    ``find_spec('PIL')``-only probe would miss.

    BOTH the spec presence AND the failing import are simulated so the test is
    hermetic on the supported base install that intentionally omits the
    ``[vision]`` extra (no host Pillow required)."""
    real_find_spec = _ilu.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "PIL":
            # Discoverable even where Pillow isn't installed — a find_spec-only
            # probe would WRONGLY pass here.
            spec = real_find_spec(name, *args, **kwargs)
            return spec if spec is not None else object()
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(_ilu, "find_spec", fake_find_spec)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("cannot import name 'Image' from 'PIL' (shadowed)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "PIL", raising=False)
    monkeypatch.delitem(sys.modules, "PIL.Image", raising=False)


def test_pil_importable_false_when_pillow_damaged(monkeypatch):
    """``_pil_importable`` must return False when the real ``from PIL import
    Image`` raises even though something named PIL is discoverable. Pre-fix it
    used ``find_spec('PIL')`` only and returned True (false green)."""
    from vllm_mlx.doctor import env_health

    _simulate_pillow_damaged(monkeypatch)

    # A find_spec-only probe (the pre-fix behaviour) would pass here — the
    # simulated spec is discoverable — yet the real import is broken.
    assert _ilu.find_spec("PIL") is not None
    assert env_health._pil_importable() is False


def _simulate_pillow_native_backend_broken(monkeypatch):
    """Pillow's Python layer imports fine but its native ``_imaging`` backend
    is broken/ABI-mismatched, so a real image op raises. ``find_spec('PIL')``
    passes and ``from PIL import Image`` succeeds — only touching the native
    core reveals the breakage.

    Hermetic: inject a stand-in ``PIL`` / ``PIL.Image`` whose ``new`` raises,
    so no host Pillow (or a real broken build) is required."""
    import types

    fake_pil = types.ModuleType("PIL")
    fake_image = types.ModuleType("PIL.Image")

    def _broken_new(*_a, **_k):
        raise ImportError("The _imaging C module is not installed or ABI-mismatched")

    fake_image.new = _broken_new
    fake_pil.Image = fake_image

    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_image)


def test_pil_importable_false_when_native_backend_broken(monkeypatch):
    """``_pil_importable`` must return False when Pillow's Python layer imports
    but its native ``_imaging`` backend is broken — i.e. ``from PIL import
    Image`` succeeds yet a real op (``Image.new``) raises. A probe that only
    did the import (no native-backed op) could false-green this."""
    from vllm_mlx.doctor import env_health

    _simulate_pillow_native_backend_broken(monkeypatch)

    # The import itself succeeds against the stand-in …
    from PIL import Image  # noqa: F401

    # … but the native-backed probe must still report broken.
    assert env_health._pil_importable() is False


def test_doctor_vision_row_red_when_pillow_damaged(monkeypatch):
    """End-to-end: a damaged/shadowed Pillow must turn the doctor vision +
    dflash rows red, exercising the REAL ``_pil_importable`` (not a stub) so
    the find_spec-only false-green regression stays pinned."""
    from vllm_mlx.doctor import env_health
    from vllm_mlx.doctor.env_health import CheckStatus

    def fake_safe_version(dist, runtime=None):
        return "0.6.17" if dist == "mlx-vlm" else None

    monkeypatch.setattr(env_health, "_safe_version", fake_safe_version)
    _simulate_pillow_damaged(monkeypatch)

    section = env_health.section_optional_packages()

    vision_row = _find_row(section, "mlx-vlm", "vision")
    assert vision_row is not None, "vision (mlx-vlm) row missing from doctor"
    assert vision_row.status is not CheckStatus.OK, (
        f"vision row must NOT be green when Pillow is damaged, got "
        f"{vision_row.status}: {vision_row.label!r}"
    )
    assert "pil" in vision_row.label.lower() or "pillow" in vision_row.label.lower()

    dflash_row = _find_row(section, "dflash")
    assert dflash_row is not None
    assert dflash_row.status is not CheckStatus.OK
