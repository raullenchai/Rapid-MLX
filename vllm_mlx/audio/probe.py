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

    Also clears the per-lane deep-probe status recorded by
    :func:`deep_probe_audio_lane` so tests don't leak ``degraded``
    state across cases.
    """
    _cached_verdict.clear()
    _LANE_STATUS.clear()
    _LANE_REASON.clear()


# Sub-modules each route actually needs to load. Split per lane so a
# torn install in one lane doesn't 503 the other (codex r2 BLOCKING
# on PR #804). ``find_spec("mlx_audio")`` is the shared presence
# check; the lane-specific entries cover runtime sub-module breakage.
_LANE_SUBMODULES: dict[str, str] = {
    "tts": "mlx_audio.tts.generate",  # /v1/audio/speech, voices
    "stt": "mlx_audio.stt.utils",  # /v1/audio/transcriptions
}

# F-K-KOKORO-MISAKI: Kokoro's tokenizer transitively depends on
# ``misaki`` (the G2P / phonemizer package). ``mlx_audio.tts.generate``
# imports cleanly without ``misaki`` because the dependency is loaded
# lazily inside ``KokoroPipeline``; the failure only surfaces on the
# FIRST ``generate()`` call.
#
# We expose a Kokoro-specific helper that's called from the speech
# route when the requested model is the Kokoro family. The lane probe
# stays Kokoro-agnostic so installs that only use Chatterbox/VibeVoice/
# VoxCPM aren't 503'd by a missing G2P package they don't need.
#
# The check IS NOT gated by a config flag — missing ``misaki`` is a
# deterministic hard failure for every Kokoro request, so a probe
# that lets the request through and 503s deep inside the engine
# leaks a stack-trace-shaped envelope. Catching the missing-extra at
# the route boundary keeps the envelope clean and the failure cheap
# (no model load, no audio synthesis kicked off).
_KOKORO_EXTRA_DEP = "misaki"
_KOKORO_EXTRA_HINT = (
    "Kokoro TTS requires the optional `misaki` G2P package, which is "
    "not installed. Reinstall with `pip install 'rapid-mlx[audio]'` "
    "to pull every audio dep, or `pip install misaki` for a "
    "minimal Kokoro-only install."
)


# F-K-CAPABILITIES-OMIT-AUDIO: D-CAPABILITIES-DETECTION's existing
# per-lane probe (``mlx_audio_available``) only checks that the
# sub-module imports — it doesn't validate that the engine can
# generate output. A model loadable at boot can still 500/503 on
# the first inference (F-K-WHISPER-500 was exactly this shape).
#
# ``deep_probe_audio_lane`` runs a tiny dry-run BEYOND the import
# check: for STT, decode 1 s of silence; for TTS, synthesize a
# single character. If the dry-run raises, the lane is marked
# ``degraded`` and that fact is surfaced via :func:`audio_lane_status`
# so the ``/v1/models`` listing (and any operator-side observability)
# can advertise the broken backend without a real user having to be
# the canary.
#
# Cost: STT dry-run is ~1 s on M2 (mostly model-load); TTS Kokoro
# dry-run is ~2 s. Gated behind the ``deep`` probe-depth setting so
# operators on tight cold-start budgets can opt out via
# ``RAPID_MLX_AUDIO_PROBE_DEPTH=shallow``. Default is ``deep`` —
# the goal of D-CAPABILITIES-DETECTION is to catch backend defects
# at boot, not at first user request.

_LANE_STATUS: dict[str, str] = {}
# Status values: "ok" | "degraded" | "missing" | "unknown"
_LANE_REASON: dict[str, str] = {}


def audio_lane_status(lane: str) -> dict[str, str | None]:
    """Return the current status snapshot for ``lane``.

    Used by ``/v1/models`` to decorate audio models with a
    capability tag. ``status`` is one of:

    * ``"ok"`` — import succeeded AND the deep dry-run (if it ran)
      produced output;
    * ``"degraded"`` — import succeeded but the dry-run failed —
      the route will 503 on real requests;
    * ``"missing"`` — ``mlx_audio`` (or the lane sub-module) is not
      importable;
    * ``"unknown"`` — no probe has run yet (deep probe disabled,
      lane never exercised).

    ``reason`` carries the failure string when status != ``"ok"``.
    """
    status = _LANE_STATUS.get(lane, "unknown")
    reason = _LANE_REASON.get(lane)
    return {"status": status, "reason": reason}


def _record_lane_status(lane: str, status: str, reason: str | None) -> None:
    _LANE_STATUS[lane] = status
    if reason is None:
        _LANE_REASON.pop(lane, None)
    else:
        _LANE_REASON[lane] = reason


def deep_probe_audio_lane(
    lane: str, model_name: str | None = None
) -> dict[str, str | None]:
    """Run a deeper-than-import dry-run for ``lane`` and record the result.

    F-K-CAPABILITIES-OMIT-AUDIO: callers (the boot-time capability
    probe, the test harness, the ``/v1/models`` capability decorator
    refresh) use this to validate that the configured audio engine
    can actually generate output, not just import. Returns the
    recorded :func:`audio_lane_status` snapshot.

    ``model_name`` is the engine the operator configured for the
    lane (defaults to the package-level defaults). Failures during
    the dry-run are CAUGHT — the function never raises. This is
    deliberate: the boot-time probe must not crash the server if
    one audio lane is broken; the only side effect is a recorded
    ``degraded`` status that downstream callers act on.

    The function is idempotent — re-calling it re-runs the dry-run.
    Tests use that to validate degraded-status surfacing without
    polluting the per-process import cache.
    """
    # First, the shallow probe — if the lane fails to import, deep
    # probing is meaningless. Record the same verdict and return.
    verdict = _probe_lane(lane)
    if not verdict.ok:
        _record_lane_status(lane, "missing", verdict.reason)
        return audio_lane_status(lane)

    if lane == "stt":
        ok, reason = _dry_run_stt(model_name)
    elif lane == "tts":
        ok, reason = _dry_run_tts(model_name)
    else:
        _record_lane_status(lane, "unknown", f"unknown audio lane {lane!r}")
        return audio_lane_status(lane)

    if ok:
        _record_lane_status(lane, "ok", None)
    else:
        _record_lane_status(lane, "degraded", reason)
    return audio_lane_status(lane)


def _dry_run_stt(model_name: str | None) -> tuple[bool, str | None]:
    """Decode 1 s of silence through the STT engine.

    Catches the F-K-WHISPER-500 shape: a Whisper model that loads
    but has no processor wired. The dry-run reaches the same
    ``get_tokenizer()`` branch the real request hits.

    Codex r2 BLOCKING #1+#2: defaults to the Whisper engine, not
    Parakeet — the WHOLE POINT of the deep probe is to catch the
    Whisper-specific processor wiring failure. Parakeet bypasses
    the broken code path (its tokenizer is bundled), so probing it
    would always report ``ok`` even when ``whisper-large-v3``
    requests are silently 500'ing. Operators serving a non-Whisper
    STT model can pass ``model_name`` explicitly to probe their
    configured engine instead.
    """
    try:
        import tempfile
        import wave

        from ..audio.stt import DEFAULT_WHISPER_MODEL, STTEngine

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            with wave.open(wav_path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(b"\x00\x00" * 16000)
            engine = STTEngine(model_name or DEFAULT_WHISPER_MODEL)
            engine.load()
            result = engine.transcribe(wav_path)
            # An empty string is a valid transcription of silence.
            if not hasattr(result, "text"):
                return False, "STT result missing `text` attribute"
        finally:
            import os as _os

            try:
                _os.unlink(wav_path)
            except OSError:
                pass
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, f"STT dry-run failed: {type(e).__name__}: {e}"


def _dry_run_tts(model_name: str | None) -> tuple[bool, str | None]:
    """Synthesize a single character through the TTS engine.

    Catches the F-K-KOKORO-MISAKI shape: Kokoro loads cleanly but
    the misaki G2P pulls in lazily and fails on first generate.
    """
    try:
        from ..audio.tts import DEFAULT_TTS_MODEL, TTSEngine

        engine = TTSEngine(model_name or DEFAULT_TTS_MODEL)
        engine.load()
        # Synthesizing a single character keeps the probe fast (<1 s).
        # Failure modes (missing misaki, broken pipeline) raise inside
        # ``generate()`` — we catch them and report degraded.
        result = engine.generate("a", voice="af_heart")
        if not hasattr(result, "audio") or len(result.audio) == 0:
            return False, "TTS result is empty (no audio produced)"
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, f"TTS dry-run failed: {type(e).__name__}: {e}"


def _reset_lane_status() -> None:
    """Test hook: clear recorded lane statuses."""
    _LANE_STATUS.clear()
    _LANE_REASON.clear()


def require_kokoro_runtime() -> None:
    """Raise an HTTP 503 when the Kokoro TTS runtime is incomplete.

    F-K-KOKORO-MISAKI: ``mlx_audio.tts.generate.load_model`` succeeds
    for Kokoro even when ``misaki`` is absent (the G2P import happens
    lazily inside the pipeline's first ``generate`` call), so the
    shared TTS-lane probe can't catch this. Called explicitly by
    ``/v1/audio/speech`` when the requested model resolves to a
    Kokoro family member. Surfaces the missing extra as a clean 503
    BEFORE we load weights, attempt synthesis, or hit the runtime
    ``Kokoro requires the optional 'misaki' package`` error inside
    mlx_audio.

    The check is intentionally narrow — Chatterbox / VibeVoice /
    VoxCPM don't depend on misaki, so this helper isn't called for
    those families. The TTS lane probe still gates them all the same.
    """
    import importlib.util

    from fastapi import HTTPException

    if importlib.util.find_spec(_KOKORO_EXTRA_DEP) is not None:
        return
    raise HTTPException(
        status_code=503,
        detail=f"{_KOKORO_EXTRA_HINT}",
    )


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


# ---------------------------------------------------------------------------
# R6-H4: CLI-side boot guard for audio model aliases.
#
# Mirrors the r5-C ``require_mlx_vlm_or_exit`` shape (PR #822) that fires
# from ``vllm_mlx.cli.serve_command`` when the operator asks the server to
# serve a vision alias on a base install missing the ``[vision]`` extra.
# Audio aliases (``kokoro``, ``whisper-large-v3``, ``parakeet``, ...) had
# no equivalent guard — ``rapid-mlx serve kokoro`` on a fresh
# ``pip install rapid-mlx`` would boot, print the startup banner, and
# only crash on the FIRST audio request (a 503 envelope from the
# in-route probe). That looked like "successful boot, broken
# inference" instead of the obvious "you need the [audio] extra".
#
# The fix probes ``mlx_audio`` at boot whenever the model alias hits
# the audio family and exits cleanly (rc=2, conventional argparse
# usage-error code) with the same install-hint copy the route's 503
# uses — so the operator sees the same actionable line whether they
# tripped the guard at boot or at first request.
# ---------------------------------------------------------------------------

#: Canonical install-hint copy — shared with the route probes via
#: :func:`_raise_503` so a torn install reports the same one-liner
#: whether the operator hit it at boot or mid-request.
AUDIO_EXTRA_INSTALL_HINT = "Install with: pip install 'rapid-mlx[audio]'"


# Known audio alias surface — kept narrow on purpose. The boot guard
# is a CONVENIENCE for the common-case alias path; an explicit HF
# ``mlx-community/...`` repo id intentionally falls through to the
# per-route probe (where the operator may legitimately be running the
# server in a hybrid setup with mlx-audio injected outside the
# extra). The list is deliberately a prefix/substring match — any
# ``whisper``/``parakeet``/``kokoro``/``chatterbox``/``vibevoice``/
# ``voxcpm`` alias counts, including future quantised variants
# (``kokoro-4bit``, ``parakeet-v3`` etc.) that drop suffix on the
# canonical name.
_AUDIO_ALIAS_TOKENS: tuple[str, ...] = (
    "whisper",
    "parakeet",
    "kokoro",
    "chatterbox",
    "vibevoice",
    "voxcpm",
)


def is_audio_model_alias(model_name: str | None) -> bool:
    """Return True iff ``model_name`` looks like an audio alias.

    Substring match against :data:`_AUDIO_ALIAS_TOKENS` so the guard
    fires for the common aliases (``kokoro``, ``whisper-large-v3``,
    ``parakeet``), their quantised siblings (``kokoro-4bit``), and
    HF-style ids that contain the engine token
    (``mlx-community/Kokoro-82M-bf16``). The match is case-insensitive
    so capitalised HF repo names (``Kokoro``, ``Whisper``) trip it the
    same way the lowercase aliases do.

    A non-string / empty value short-circuits to False so the boot
    guard never crashes the CLI on a missing ``args.model`` (the
    serve command rejects that case earlier with its own error).
    """
    if not isinstance(model_name, str) or not model_name:
        return False
    lc = model_name.lower()
    return any(tok in lc for tok in _AUDIO_ALIAS_TOKENS)


def require_audio_or_exit(model_name: str) -> None:
    """CLI-side boot guard: bail out cleanly when an audio alias is
    served on an install missing the ``[audio]`` extra.

    Mirrors :func:`vllm_mlx.models.mllm.require_mlx_vlm_or_exit` and
    :func:`vllm_mlx.embedding.require_mlx_embeddings_or_exit` — probe
    ``importlib.util.find_spec("mlx_audio")`` so we only answer "no"
    for the specific case the install hint is meant to address
    (top-level package missing). A broken transitive dependency raising
    deep inside the package would surface as a real exception via the
    in-route probe instead, not get masked behind the install hint.

    Exits ``2`` (argparse usage-error code) with the canonical install
    hint on stderr. ``vllm_mlx.cli.serve_command`` calls this after
    embedding + vision guards so a single ``rapid-mlx serve`` command
    that requests audio-only sees the audio hint, not the (irrelevant)
    embedding/vision one.
    """
    import importlib.util
    import sys

    if importlib.util.find_spec("mlx_audio") is not None:
        return
    print(
        f"error: model {model_name!r} is an audio alias and requires the "
        f"optional `mlx-audio` dependency (shipped with the [audio] "
        f"extra).\n" + AUDIO_EXTRA_INSTALL_HINT,
        file=sys.stderr,
    )
    sys.exit(2)
