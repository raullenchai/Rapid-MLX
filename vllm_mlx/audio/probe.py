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

The fix routes EVERY audio endpoint through the same probe surface
so they agree, but per-lane so a torn install in one lane doesn't
503 the other. Two complementary checks per lane:

1. ``importlib.util.find_spec("mlx_audio")`` — cheap presence check;
   answers "is the top-level package even installed?". Shared across
   lanes (cached once per process).
2. Lane-specific late-import — for the TTS lane, probe
   ``mlx_audio.tts.generate``; for the STT lane, probe
   ``mlx_audio.stt.utils``. Caches per-lane verdicts so we don't
   pay the import on every request.

Lane separation rationale (codex r2 BLOCKING on PR #804): a single
combined probe would 503 the TTS routes when only STT is broken (or
vice versa) — a regression for TTS-only callers on installs where
STT happens to be torn. Each route probes ONLY the lane it needs.
The base ``find_spec("mlx_audio")`` failure (extra missing entirely)
still 503s all three routes with the same envelope because that's
the genuinely-shared failure mode.

The probe is purely lazy — ``vllm_mlx.audio.probe`` itself never
imports ``mlx_audio`` at module top level, so the base install
(without the ``[audio]`` extra) can ``from vllm_mlx.audio.probe
import require_mlx_audio_tts`` without crashing.
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


# Lane-keyed cache. ``"tts"`` and ``"stt"`` map to the most recent
# verdict for each lane; the empty ``""`` key holds the result of the
# shared ``find_spec`` presence check so the cheap "extra missing"
# path doesn't repeat the syscall on every request.
_cached_verdict: dict[str, _Verdict] = {}


def _reset_probe_cache() -> None:
    """Test hook: clear every cached verdict.

    The audio routes call the lane-specific probes on every request,
    so a stale cache from a previous test (where the import succeeded)
    would mask a monkeypatched failure in the next test. Tests that
    swap ``builtins.__import__`` or otherwise simulate a broken
    ``mlx_audio`` call this helper in their fixture to force re-probe.
    """
    _cached_verdict.clear()


# Sub-modules each route actually needs to load. Split per lane so a
# torn install in one lane doesn't 503 the other (codex r2 BLOCKING
# on PR #804). ``find_spec("mlx_audio")`` is the shared presence
# check; the lane-specific entries cover runtime sub-module breakage.
_LANE_SUBMODULES: dict[str, str] = {
    "tts": "mlx_audio.tts.generate",  # /v1/audio/speech, voices
    "stt": "mlx_audio.stt.utils",  # /v1/audio/transcriptions
}


def _probe_lane(lane: str) -> _Verdict:
    """Internal: probe a single lane (``"tts"`` or ``"stt"``).

    Shared first step — ``importlib.util.find_spec("mlx_audio")`` —
    is cached under the empty-string key so the cheap presence check
    only runs once per process. If the top-level package is missing,
    the verdict for the lane folds back to that "extra not installed"
    answer so callers see a uniform envelope across lanes for the
    common case.

    Sub-module probe uses bare ``__import__`` (rather than
    ``importlib.import_module``) so a torn install is detected even
    when an earlier successful import has already populated
    ``sys.modules``. ``import_module`` short-circuits to the cached
    entry; ``__import__`` re-resolves the import machinery — which is
    what tests need to validate the broken-install path AND what
    production needs when a runtime force-reload (plugin hot-reload,
    config reload mid-process) cleared the cache.
    """
    if lane in _cached_verdict:
        return _cached_verdict[lane]

    # Shared presence check: cached separately so a TTS probe doesn't
    # re-pay the find_spec syscall for STT.
    if "" not in _cached_verdict:
        import importlib.util

        if importlib.util.find_spec("mlx_audio") is None:
            _cached_verdict[""] = _Verdict(
                ok=False, reason="mlx-audio is not installed"
            )
        else:
            _cached_verdict[""] = _Verdict(ok=True, reason=None)
    presence = _cached_verdict[""]
    if not presence.ok:
        _cached_verdict[lane] = presence
        return presence

    submod = _LANE_SUBMODULES.get(lane)
    if submod is None:
        # Programmer error — unknown lane.
        _cached_verdict[lane] = _Verdict(
            ok=False, reason=f"unknown audio lane {lane!r}"
        )
        return _cached_verdict[lane]
    try:
        __import__(submod)
    except Exception as e:  # noqa: BLE001
        _cached_verdict[lane] = _Verdict(
            ok=False,
            reason=(
                f"mlx-audio {lane} import failed at runtime: "
                f"{type(e).__name__}: {e} (probing {submod})"
            ),
        )
        return _cached_verdict[lane]

    _cached_verdict[lane] = _Verdict(ok=True, reason=None)
    return _cached_verdict[lane]


def mlx_audio_available(lane: str = "tts") -> _Verdict:
    """Probe whether ``mlx_audio`` is usable for ``lane``.

    ``lane`` is one of ``"tts"`` (default) or ``"stt"`` — selects
    which sub-module gets the runtime late-import check. The shared
    ``find_spec`` presence check is cached across lanes so a missing
    extra reports a uniform envelope from every route.

    The route handlers consult this through
    :func:`require_mlx_audio_tts` / :func:`require_mlx_audio_stt`
    so the failure surface is uniform within a lane while a torn
    STT install can no longer 503 the TTS routes.
    """
    return _probe_lane(lane)


def _raise_503(verdict: _Verdict) -> None:
    """Translate a failed :class:`_Verdict` into the HTTP 503 envelope.

    Imports ``HTTPException`` lazily so the base install — which
    doesn't necessarily reach the audio routes at all — doesn't pay
    the FastAPI cost just to wire the probe.
    """
    from fastapi import HTTPException

    detail = verdict.reason or "mlx-audio is not available"
    raise HTTPException(
        status_code=503,
        detail=(f"{detail}. Install with: pip install 'rapid-mlx[audio]'"),
    )


def require_mlx_audio_tts() -> None:
    """Raise an HTTP 503 when the TTS lane of ``mlx_audio`` isn't usable.

    Used by ``/v1/audio/speech`` and ``/v1/audio/voices``. A torn
    STT install does NOT trip this probe — pre-codex-r2 the
    combined probe did, which masked TTS-usable installs as broken.
    """
    verdict = _probe_lane("tts")
    if verdict.ok:
        return
    _raise_503(verdict)


def require_mlx_audio_stt() -> None:
    """Raise an HTTP 503 when the STT lane of ``mlx_audio`` isn't usable.

    Used by ``/v1/audio/transcriptions``. A torn TTS install does
    NOT trip this probe.
    """
    verdict = _probe_lane("stt")
    if verdict.ok:
        return
    _raise_503(verdict)


# Backwards-compat shim — earlier PR #804 commits exported
# ``require_mlx_audio`` as a single combined probe. Kept as an alias
# for the TTS lane so any in-flight code that imported the old name
# still works; the TTS lane is the more common probe target (speech
# + voices vs. transcriptions alone). Re-aliased through the
# explicit lane name so call sites read clearly.
require_mlx_audio = require_mlx_audio_tts
