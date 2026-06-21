# SPDX-License-Identifier: Apache-2.0
"""Single source of truth for ``mlx_audio`` availability.

D-CAPABILITIES-DETECTION (F-D05): pre-fix, ``/v1/audio/voices`` and
``/v1/audio/speech`` used *different* probes — the voices route never
touched ``mlx_audio`` at all (returned a static voice list) while the
speech route did a lazy ``import mlx_audio.tts.generate.load_model``
inside the request handler. Result: when the runtime import broke
(transitive-dep mismatch, partial reinstall, etc.) the two endpoints
disagreed — ``voices`` said "yes" with a full list while ``speech``
503'd "mlx-audio not installed". Users couldn't tell whether TTS
was actually wired up.

The fix routes EVERY audio endpoint through the same probe so they
agree. Two complementary checks:

1. ``importlib.util.find_spec("mlx_audio")`` — cheap presence check;
   answers "is the top-level package even installed?".
2. ``import mlx_audio.tts.generate as _probe`` — runtime check;
   answers "does the package actually import cleanly?". Caches the
   verdict so we don't pay the import on every request.

The probe is purely lazy — ``vllm_mlx.audio.probe`` itself never
imports ``mlx_audio`` at module top level, so the base install
(without the ``[audio]`` extra) can ``from vllm_mlx.audio.probe import
require_mlx_audio`` without crashing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _Verdict:
    """Outcome of the ``mlx_audio`` probe.

    ``ok`` is the headline answer the routes branch on. ``reason``
    carries the original failure string so the 503 envelope can point
    operators at the actual root cause (e.g. ``ModuleNotFoundError:
    No module named 'mlx_audio.tts.foo'`` — *not* the generic "install
    the [audio] extra" hint that lies when the extra IS installed but
    the runtime import is broken).
    """

    ok: bool
    reason: str | None = None


# Module-level cache. Set on first call to :func:`mlx_audio_available`
# and re-used by every audio route. Cleared by :func:`_reset_probe_cache`
# in tests so monkeypatched import failures take effect on the next
# call without leaking across tests.
_cached_verdict: _Verdict | None = None


def _reset_probe_cache() -> None:
    """Test hook: clear the cached verdict.

    The audio routes call :func:`mlx_audio_available` on every request,
    so a stale cache from a previous test (where the import succeeded)
    would mask a monkeypatched failure in the next test. Tests that
    swap ``builtins.__import__`` or otherwise simulate a broken
    ``mlx_audio`` call this helper in their fixture to force re-probe.
    """
    global _cached_verdict
    _cached_verdict = None


def mlx_audio_available() -> _Verdict:
    """Probe whether ``mlx_audio`` is usable.

    Returns the same :class:`_Verdict` for every caller within a
    single process — the import check runs at most once, then the
    cached answer is re-used. ``find_spec`` covers the "not
    installed" case; the late ``import mlx_audio.tts.generate``
    covers the "installed but transitively broken" case (the failure
    mode Diego hit on a fresh ``[vision]`` install where ``mlx-audio``
    was pulled in as a transitive but actually breaks at runtime).

    The route handlers consult this through
    :func:`require_mlx_audio` so the failure surface is uniform
    across ``/v1/audio/speech``, ``/v1/audio/voices``, and
    ``/v1/audio/transcriptions``.
    """
    global _cached_verdict
    if _cached_verdict is not None:
        return _cached_verdict

    import importlib.util

    if importlib.util.find_spec("mlx_audio") is None:
        _cached_verdict = _Verdict(
            ok=False,
            reason="mlx-audio is not installed",
        )
        return _cached_verdict

    # find_spec said the top-level package is reachable. Now confirm
    # the sub-modules actually used by the TTS/STT engines import
    # cleanly. A torn install (e.g. a transitive dep version
    # mismatch) shows up here — and we want the route to advertise
    # the real ``ImportError`` reason rather than the generic
    # "install the extra" hint that would mislead the operator.
    try:
        import mlx_audio.tts.generate  # noqa: F401
    except Exception as e:  # noqa: BLE001
        _cached_verdict = _Verdict(
            ok=False,
            reason=f"mlx-audio import failed at runtime: {type(e).__name__}: {e}",
        )
        return _cached_verdict

    _cached_verdict = _Verdict(ok=True, reason=None)
    return _cached_verdict


def require_mlx_audio() -> None:
    """Raise an HTTP 503 when ``mlx_audio`` is not usable.

    Centralizes the failure envelope so every audio route returns the
    SAME body and status when the dep is missing or broken. Pre-fix
    the speech and voices routes had divergent behavior; that's the
    cross-endpoint inconsistency this helper closes.

    Imports ``HTTPException`` lazily so the base install — which
    doesn't necessarily reach the audio routes at all — doesn't pay
    the FastAPI cost just to wire the probe.
    """
    verdict = mlx_audio_available()
    if verdict.ok:
        return
    from fastapi import HTTPException

    detail = verdict.reason or "mlx-audio is not available"
    raise HTTPException(
        status_code=503,
        detail=(f"{detail}. Install with: pip install 'rapid-mlx[audio]'"),
    )
