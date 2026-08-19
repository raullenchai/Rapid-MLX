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

import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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
    :func:`deep_probe_audio_lane` and the cached Kokoro espeak +
    spaCy-G2P-model readiness verdicts so tests don't leak state
    across cases.
    """
    _cached_verdict.clear()
    _LANE_STATUS.clear()
    _LANE_REASON.clear()
    _reset_espeak_state()
    _reset_g2p_model_state()


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

# F-K-KOKORO-ESPEAK: ``misaki`` being importable is necessary but not
# sufficient for Kokoro — its English G2P falls back to espeak-ng (via
# ``phonemizer`` + ``espeakng-loader``) for any out-of-dictionary word.
# On a fresh ``pip install 'rapid-mlx[audio]'`` the espeak-ng dylib
# shipped by ``espeakng-loader`` can fail to locate its ``espeak-ng-data``
# (the compiled data path points at the wheel's CI build dir), and the
# resulting failure is a C-level ``exit()``/``abort()`` INSIDE
# ``espeak_Initialize`` — it takes down the whole uvicorn worker and
# every in-flight request, and Python ``try/except`` cannot catch it.
#
# Two-part mitigation, both proven necessary by dogfooding a fresh
# ``[audio]`` venv:
#   1. CONTAINMENT — validate espeak in a throwaway SUBPROCESS before we
#      let the real synthesis touch it. A broken dylib kills the child,
#      not the server; we read the child's exit code.
#   2. REPAIR — if the bundled dylib is broken but a system espeak-ng is
#      installed (e.g. ``brew install espeak-ng``), point ``phonemizer``
#      at the system library + data so Kokoro actually works. Only when
#      neither the bundled nor a system espeak-ng can initialize do we
#      surface a clean 503 (the worker stays up either way).
#
# The check is Kokoro-specific and cached per-process: the first Kokoro
# request pays a one-time ~1-2 s subprocess cost; every later request
# hits the cached verdict. Chatterbox / VibeVoice / VoxCPM / F5 don't use
# espeak and never reach this path.
_ESPEAK_BROKEN_HINT = (
    "Kokoro TTS could not initialize its espeak-ng phonemizer backend. "
    "The espeak-ng data bundled by `espeakng-loader` failed to load on "
    "this machine and no working system espeak-ng was found. Install one "
    "with `brew install espeak-ng` and restart the server, or use a "
    "non-espeak TTS model such as `chatterbox` or `f5-tts-zh`."
)

# Child-process espeak self-test. Constructs a ``phonemizer`` espeak
# backend (this is where ``espeak_Initialize`` runs and where a broken
# dylib aborts) and phonemizes a short string. ``RAPID_MLX_ESPEAK_LIB`` /
# ``RAPID_MLX_ESPEAK_DATA`` (when set) redirect it at a system install;
# absent, it exercises whatever ``misaki`` wired up at import (bundled).
# Exit 0 == espeak works; any non-zero (including a C-level abort that
# never returns to Python) == broken.
_ESPEAK_SELFTEST_SRC = (
    "import os, sys\n"
    "try:\n"
    "    import misaki.espeak  # noqa: F401 (import wires bundled espeak)\n"
    "    from phonemizer.backend.espeak.wrapper import EspeakWrapper\n"
    "    _lib = os.environ.get('RAPID_MLX_ESPEAK_LIB')\n"
    "    _data = os.environ.get('RAPID_MLX_ESPEAK_DATA')\n"
    "    if _lib:\n"
    "        EspeakWrapper.set_library(_lib)\n"
    "    if _data:\n"
    "        EspeakWrapper.set_data_path(_data)\n"
    "    import phonemizer\n"
    "    _be = phonemizer.backend.EspeakBackend(\n"
    "        language='en-us', preserve_punctuation=True,\n"
    "        with_stress=True, tie='^')\n"
    "    _out = _be.phonemize(['phonemizer selftest'])\n"
    "except Exception:\n"
    "    sys.exit(6)\n"
    # ``sys.exit`` raises ``SystemExit`` (a BaseException, not Exception),
    # so the success/empty exit lives OUTSIDE the ``except Exception`` above
    # — otherwise a clean ``sys.exit(0)`` would be swallowed and reported as
    # a failure. A broken dylib aborts at the C level and never reaches
    # Python, so ``except Exception`` is the right (and only catchable) net.
    "sys.exit(0 if _out and _out[0].strip() else 5)\n"
)

# Per-process espeak readiness cache: None = not yet probed, True = ready
# (bundled works or repaired to system), False = unfixable (clean 503).
# The lock coalesces concurrent first-request probes (two Kokoro requests
# racing on a cold worker must not both spawn the subprocess sweep).
_ESPEAK_READY: bool | None = None
_ESPEAK_REASON: str | None = None
_ESPEAK_LOCK = threading.Lock()

# Hard bound on the readiness sweep: at most this many system (library,
# data-dir) self-tests, so a badly-broken host can't spin through an unbounded
# number of subprocesses. This is a TOTAL pair budget (not a per-axis cap):
# discovery schedules candidates fairly (anti-diagonal over the library x data
# grid) so the budget still samples multiple libraries AND multiple data dirs
# before exhausting either axis, then truncates the tail.
_MAX_ESPEAK_CANDIDATES = 8


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
        import wave

        from .._tempfile_safe import managed_tempfile_path
        from ..audio.stt import DEFAULT_WHISPER_MODEL, STTEngine

        # GH #719 — the old pattern was ``NamedTemporaryFile(delete=False)``
        # inside a ``with`` block (which only closes the FD, not the
        # file), followed by a manual ``try/finally`` for unlink. The
        # uncovered window was any exception between path allocation
        # (when ``NamedTemporaryFile`` returned the open handle) and
        # the start of the manual ``try`` block — including the
        # ``with`` body itself, ``wave.open`` errors, or anything that
        # raised before the manual ``finally`` could fire.
        # ``managed_tempfile_path`` registers the path in a process-wide
        # set the moment ``mkstemp`` returns, so the atexit fallback
        # reaps it even on the bypass paths the old shape couldn't see.
        with managed_tempfile_path(suffix=".wav") as wav_handle:
            wav_path = wav_handle.path
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
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, f"STT dry-run failed: {type(e).__name__}: {e}"


def _dry_run_tts(model_name: str | None) -> tuple[bool, str | None]:
    """Synthesize a single character through the TTS engine.

    Catches the F-K-KOKORO-MISAKI shape: Kokoro loads cleanly but
    the misaki G2P pulls in lazily and fails on first generate.

    ``SystemExit`` is caught alongside ``Exception`` (#1254): misaki's
    spaCy-model download shells out to ``uv pip``, which ``sys.exit()``s
    under a uv-tool / console-script launcher with no active venv. That is
    a ``BaseException``, so a bare ``except Exception`` would let it abort
    server startup when ``RAPID_MLX_AUDIO_DEEP_PROBE`` is enabled — this
    probe MUST only ever report degraded, never take the process down.
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
    except (Exception, SystemExit) as e:  # noqa: BLE001
        return False, f"TTS dry-run failed: {type(e).__name__}: {e}"


def _espeak_selftest_subprocess(
    lib: str | None = None, data: str | None = None, timeout: float = 30.0
) -> bool:
    """Run the espeak self-test in a throwaway child; True == espeak works.

    F-K-KOKORO-ESPEAK containment: a broken ``espeakng-loader`` dylib
    aborts the process at the C level inside ``espeak_Initialize`` —
    uncatchable from Python. Running the probe in a subprocess means
    that abort kills the CHILD; we recover the verdict from its exit
    code (0 == espeak initialized and phonemized; anything else — a
    non-zero return, a signal, or a timeout — == broken).

    ``lib`` / ``data`` (when given) redirect the child at a system
    espeak-ng install via the ``RAPID_MLX_ESPEAK_*`` env vars the
    self-test source reads; omit both to exercise the bundled dylib.
    """
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    if lib:
        env["RAPID_MLX_ESPEAK_LIB"] = lib
    else:
        env.pop("RAPID_MLX_ESPEAK_LIB", None)
    if data:
        env["RAPID_MLX_ESPEAK_DATA"] = data
    else:
        env.pop("RAPID_MLX_ESPEAK_DATA", None)
    try:
        result = subprocess.run(
            [sys.executable, "-c", _ESPEAK_SELFTEST_SRC],
            env=env,
            capture_output=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


def _resolve_lib_path(name_or_path: str) -> str | None:
    """Resolve a library name or bare soname to an absolute existing path.

    ``ctypes.util.find_library`` returns an absolute path on macOS but often
    just a bare soname (``libespeak-ng.so.1``) on Linux. ``EspeakWrapper``
    needs a real filesystem path, so resolve the basename against the
    standard library directories — including the Debian/Ubuntu multiarch
    dir (``/usr/lib/<triplet>``) that ``find_library`` reports only by
    name. Returns None when nothing resolves.
    """
    import os
    import sysconfig

    if os.path.isabs(name_or_path):
        return name_or_path if os.path.exists(name_or_path) else None

    base = os.path.basename(name_or_path)
    search_dirs = [
        "/opt/homebrew/lib",
        "/usr/local/lib",
        "/usr/lib",
        "/lib",
        "/usr/lib64",  # Fedora / RHEL / SUSE
        "/lib64",
    ]
    libdir = sysconfig.get_config_var("LIBDIR")
    if libdir:
        search_dirs.append(libdir)
    multiarch = sysconfig.get_config_var("MULTIARCH")
    if multiarch:
        search_dirs += [f"/usr/lib/{multiarch}", f"/lib/{multiarch}"]
    for directory in search_dirs:
        cand = os.path.join(directory, base)
        if os.path.exists(cand):
            return cand
    return None


def _discover_system_espeak() -> list[tuple[str, str]]:
    """Return candidate ``(library_path, data_parent_dir)`` pairs for a
    system espeak-ng, best-first (may be empty).

    Covers the layouts that show up in practice: the install prefix derived
    from the ``espeak-ng`` executable on ``PATH`` (Homebrew's
    ``/opt/homebrew/bin/espeak-ng`` → ``/opt/homebrew/{lib,share}``), a few
    well-known prefixes, and ``ctypes.util.find_library`` for the library
    (which resolves Linux multiarch dirs such as
    ``/usr/lib/x86_64-linux-gnu`` that a bare ``<prefix>/lib`` guess would
    miss — codex MAJOR).

    Only pairs whose library file exists and whose data dir actually
    contains ``espeak-ng-data/phontab`` are returned (espeak appends
    ``/espeak-ng-data`` to the data parent itself). The caller still
    self-tests each pair in a subprocess before trusting it, so a wrong
    pairing degrades to the next candidate or a clean 503 — never a crash.
    """
    import ctypes.util
    import os
    import shutil

    lib_names = (
        "libespeak-ng.1.dylib",
        "libespeak-ng.dylib",
        "libespeak-ng.so.1",
        "libespeak-ng.so",
    )

    prefixes: list[str] = []
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if exe:
        prefixes.append(os.path.dirname(os.path.dirname(os.path.realpath(exe))))
    prefixes += ["/opt/homebrew", "/usr/local", "/usr"]

    libs: list[str] = []
    data_parents: list[str] = []
    for prefix in prefixes:
        for name in lib_names:
            cand = os.path.join(prefix, "lib", name)
            if os.path.exists(cand):
                libs.append(cand)
        share = os.path.join(prefix, "share")
        if os.path.exists(os.path.join(share, "espeak-ng-data", "phontab")):
            data_parents.append(share)

    # Loader-resolved library — handles multiarch dirs the prefix guess
    # misses. ``find_library`` returns an absolute path on macOS but often a
    # bare soname (``libespeak-ng.so.1``) on Linux, which ``set_library``
    # can't load — resolve it to an absolute path first (codex MAJOR r2).
    found = ctypes.util.find_library("espeak-ng")
    if found:
        resolved = _resolve_lib_path(found)
        if resolved:
            libs.append(resolved)

    def _dedup(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    libs = _dedup(libs)
    data_parents = _dedup(data_parents)
    # Fair (anti-diagonal) schedule over the library x data-dir grid: order
    # pairs by ``library_rank + data_rank`` so the total budget samples several
    # libraries AND several data dirs before exhausting either axis. A
    # library-major or data-major flatten would spend the whole budget on one
    # axis and hide a valid pairing on the other — e.g. every probe against a
    # single wrong data dir, or against a single broken library (codex, both
    # directions). Best-first within each diagonal (lower ranks first); the
    # caller truncates the tail to :data:`_MAX_ESPEAK_CANDIDATES`.
    pairs: list[tuple[str, str]] = []
    for diag in range(len(libs) + len(data_parents)):
        for i, lib in enumerate(libs):
            j = diag - i
            if 0 <= j < len(data_parents):
                pairs.append((lib, data_parents[j]))
    return pairs[:_MAX_ESPEAK_CANDIDATES]


def _apply_system_espeak(lib: str, data: str) -> None:
    """Point ``phonemizer`` at a system espeak-ng in THIS worker process.

    ``misaki`` sets the bundled library/data on the ``EspeakWrapper``
    class at import; overriding those class attributes AFTER the import
    (but before Kokoro's pipeline constructs its espeak backend on the
    first ``generate``) makes the real synthesis use the working system
    install. ``set_data_path`` receives the parent dir because espeak
    appends ``/espeak-ng-data`` itself.
    """
    import misaki.espeak  # noqa: F401 (ensure the import-time wiring ran)
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    EspeakWrapper.set_library(lib)
    EspeakWrapper.set_data_path(data)


def _probe_espeak_readiness() -> tuple[bool, str | None]:
    """Run the (blocking) espeak readiness sweep. Returns ``(ready, reason)``.

    Bundled espeak first — preserves existing behaviour on platforms where
    the shipped dylib loads correctly (no override, no repair). If bundled is
    broken, self-tests each discovered system espeak-ng candidate in a
    subprocess and repairs this worker to the first that initializes. No
    candidate works → not ready. Discovery bounds the sweep to at most
    :data:`_MAX_ESPEAK_CANDIDATES` self-tests, so a badly-broken host can't
    spin through an unbounded number of installs.

    Errors are contained so the caller always returns a clean 503 (never a
    500) and the verdict is cached — an escaping exception would leave
    ``_ESPEAK_READY`` unresolved, re-probing (and re-spawning subprocesses) on
    every request (codex). Containment is per-candidate: one malformed
    candidate (e.g. ``_apply_system_espeak`` raising) is skipped, never
    aborting the sweep while a later candidate could still work (codex).
    """
    try:
        if _espeak_selftest_subprocess():
            return True, None
    except Exception:
        logger.warning("bundled espeak self-test failed unexpectedly", exc_info=True)

    try:
        candidates = _discover_system_espeak()
    except Exception:
        logger.warning("espeak discovery failed unexpectedly", exc_info=True)
        return False, _ESPEAK_BROKEN_HINT

    for lib, data in candidates:
        try:
            if _espeak_selftest_subprocess(lib=lib, data=data):
                _apply_system_espeak(lib, data)
                return True, None
        except Exception:
            logger.warning(
                "espeak candidate %s / %s failed unexpectedly", lib, data, exc_info=True
            )
            continue

    return False, _ESPEAK_BROKEN_HINT


# ---------------------------------------------------------------------------
# F-K-KOKORO-SPACY (#1254): misaki's English G2P tokenizer needs the spaCy
# ``en_core_web_sm`` model, which misaki downloads LAZILY on the first
# ``generate`` via ``spacy.cli.download``. spaCy's installer shells out to
# ``sys.executable -m pip`` when ``pip`` is importable, else ``uv pip`` (the
# uv-tool / bundled console-script case). ``uv pip`` aborts with a
# ``SystemExit`` — "No virtual environment found" — when no venv is active,
# which the route's ``except Exception`` cannot catch → an opaque HTTP 500 on
# the FIRST speech request. We pre-resolve the model at the route boundary
# (offloaded to a worker thread, like the espeak sweep) so a missing model
# 503s cleanly BEFORE weight load and can never reach misaki's SystemExit-y
# runtime download. Cached + lock-coalesced: the one-time install runs once
# per process even under concurrent cold-start requests.
_KOKORO_G2P_SPACY_MODEL = "en_core_web_sm"

# Per-process readiness cache + single-flight lock. rapid-mlx serves from a
# SINGLE uvicorn worker (``server.py`` calls ``uvicorn.run(app)`` with no
# ``workers=``), so a process-local lock fully coalesces the one-time install;
# a hypothetical multi-worker deployment would need a cross-process lock, but
# that isn't a supported topology here.
#
# SUCCESS is cached for the process lifetime; a FAILURE is cached only for a
# short cooldown (``_G2P_MODEL_RETRY_COOLDOWN_S``) so a transient index outage
# or timeout doesn't disable Kokoro until restart — after the window we
# re-probe, and once the dependency is present we cache success.
_G2P_MODEL_READY: bool | None = None
_G2P_MODEL_REASON: str | None = None
_G2P_MODEL_RETRY_AFTER: float = 0.0
_G2P_MODEL_RETRY_COOLDOWN_S = 60.0
_G2P_MODEL_LOCK = threading.Lock()


def _g2p_model_install_hint() -> str:
    """Fixed, operator-facing 503 message.

    Carries NO subprocess output and no interpreter path: raw installer stderr
    can leak authenticated package-index URLs, internal hostnames, usernames,
    and filesystem paths, so it is logged server-side and NEVER returned in the
    HTTP response. The recovery text covers both a normal venv and the uv-tool
    / no-venv launcher (where a bare ``spacy download`` can't resolve a target)
    by pointing at a venv reinstall of the audio extras.
    """
    return (
        f"Kokoro TTS needs the spaCy G2P model '{_KOKORO_G2P_SPACY_MODEL}', "
        "which could not be prepared automatically. Reinstall the audio extras "
        "with 'pip install \"rapid-mlx[audio]\"' inside a virtual environment "
        f"(or run 'python -m spacy download {_KOKORO_G2P_SPACY_MODEL}' against "
        "the server's interpreter), then retry — the server re-checks "
        "automatically; restart only if it still can't find the model."
    )


def _g2p_installer_env(
    environ: dict[str, str], prefix: str, prefix_is_venv: bool
) -> dict[str, str]:
    """Env for a child spaCy-model install so it targets THIS interpreter.

    spaCy's ``uv pip`` fallback installs into ``$VIRTUAL_ENV``. Two cases:

    * The running interpreter IS a venv (the uv-tool / bundled-sidecar case):
      force ``VIRTUAL_ENV`` to ``prefix`` — even if one is already set —
      because an inherited ``VIRTUAL_ENV`` from an outer activated shell can
      point at a DIFFERENT environment; installing the model there would leave
      it invisible to the running spaCy and keep misaki's SystemExit-y runtime
      download reachable.
    * The running interpreter is NOT a venv (system Python): DROP any inherited
      ``VIRTUAL_ENV`` rather than let a child ``uv pip`` install into that
      unrelated outer env. With no venv, ``uv pip`` errors cleanly (→ our
      caught failure → 503) instead of silently polluting the wrong env.

    Harmless for the ``pip`` path either way (pip installs into
    ``sys.executable`` regardless of ``VIRTUAL_ENV``). Pure + no FS access →
    unit-testable.
    """
    env = dict(environ)
    if prefix_is_venv:
        env["VIRTUAL_ENV"] = prefix
    else:
        env.pop("VIRTUAL_ENV", None)
    return env


def _spacy_download_subprocess(
    cmd: list[str], env: dict[str, str], timeout: int
) -> None:
    """Run the spaCy-model install in its OWN process group; kill the WHOLE
    group on timeout.

    ``subprocess.run(timeout=...)`` SIGKILLs only the direct child, so a pip/uv
    grandchild would survive a timeout, keep mutating the environment after the
    single-flight lock is released, and race a later retry. ``start_new_session``
    makes the child a group leader whose grandchildren inherit the group, so one
    ``killpg`` reaps the whole tree. Raises ``CalledProcessError`` on non-zero
    exit (stderr on the exception only) or ``TimeoutExpired`` on timeout — both
    already handled by the caller, which logs a sanitized shape, never raw text.
    """
    import os
    import signal
    import subprocess

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        _, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            proc.kill()  # fall back to killing at least the direct child
        # BOUNDED best-effort reap: SIGKILL is uncatchable so this normally
        # returns at once; cap it so a wedged descendant still holding a pipe
        # can't hang the worker past the timeout. Close our pipe ends regardless
        # so we never leak fds; any residual process is reaped by the subprocess
        # module's own cleanup on the next ``Popen``.
        try:
            proc.communicate(timeout=10)
        except Exception:  # noqa: BLE001 — never mask the timeout re-raised below
            pass
        for _pipe in (proc.stdout, proc.stderr, proc.stdin):
            if _pipe is not None:
                try:
                    _pipe.close()
                except Exception:  # noqa: BLE001
                    pass
        raise
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, stderr=stderr)


def _probe_kokoro_g2p_model() -> tuple[bool, str | None]:
    """Ensure misaki's spaCy G2P model is importable; install it once if not.

    Returns ``(ready, reason)`` and NEVER raises — its contract is that ANY
    failure (a spaCy import error, ``spacy.util.is_package`` raising on corrupt
    metadata, a subprocess or ``SystemExit`` from the installer) becomes a
    clean 503 verdict, not an exception the route would collapse into the
    opaque 500 this fix exists to prevent.
    """
    try:
        return _probe_kokoro_g2p_model_impl()
    except (Exception, SystemExit) as e:  # noqa: BLE001
        # Log only the exception TYPE — an arbitrary exception's repr can carry
        # subprocess stderr / index URLs / paths (secret-in-logs), matching the
        # inner installer-error handler's sanitized contract.
        logger.error(
            "Kokoro TTS: spaCy G2P readiness probe failed: %s", type(e).__name__
        )
        return False, _g2p_model_install_hint()


def _probe_kokoro_g2p_model_impl() -> tuple[bool, str | None]:
    import importlib
    import os
    import subprocess
    import sys

    try:
        import spacy.util
    except Exception as e:  # noqa: BLE001
        # Reached only for a Kokoro request past the misaki-present gate, and
        # misaki hard-depends on spaCy — so an unimportable spaCy (absent, torn
        # submodule, or broken C-ext) is a broken install. Fail closed with a
        # 503 rather than mask it as ready (which would resurface as the opaque
        # 500 this fix prevents). Log only the exception TYPE (a broken-C-ext
        # ImportError repr can carry a dylib path); never return it.
        logger.error(
            "Kokoro TTS: spaCy G2P runtime is not importable: %s", type(e).__name__
        )
        return False, _g2p_model_install_hint()
    # ``is_package`` is misaki's OWN download predicate — misaki/en.py does
    # ``if not spacy.util.is_package(name): spacy.cli.download(name)`` then
    # ``spacy.load(name)``. Matching that predicate exactly is what makes this
    # gate sufficient: when it's True, misaki's ``if not is_package`` is False,
    # so misaki NEVER reaches ``spacy.cli.download`` and the uv-pip SystemExit
    # cannot fire. (A corrupt-but-installed model would fail later in
    # ``spacy.load`` as a normal ``OSError`` — a plain Exception the route 500s
    # cleanly — never the #1254 SystemExit, because misaki gates the download
    # on presence, not on load success.)
    if spacy.util.is_package(_KOKORO_G2P_SPACY_MODEL):
        return True, None

    env = _g2p_installer_env(
        dict(os.environ),
        sys.prefix,
        os.path.exists(os.path.join(sys.prefix, "pyvenv.cfg")),
    )
    logger.info(
        "Kokoro TTS: installing spaCy G2P model %s (first-use bootstrap)…",
        _KOKORO_G2P_SPACY_MODEL,
    )
    try:
        _spacy_download_subprocess(
            [sys.executable, "-m", "spacy", "download", _KOKORO_G2P_SPACY_MODEL],
            env,
            300,
        )
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
        OSError,
    ) as e:
        # Log only the failure SHAPE (exception type + exit status) — never the
        # raw installer stderr, which can echo authenticated package-index URLs,
        # credentials, hostnames, and paths (secret-in-logs). The operator
        # reproduces with the exact command surfaced in the 503 hint.
        rc = getattr(e, "returncode", None)
        logger.error(
            "Kokoro TTS: spaCy G2P model install failed: %s%s",
            type(e).__name__,
            f" (exit {rc})" if rc is not None else "",
        )
        return False, _g2p_model_install_hint()

    importlib.invalidate_caches()
    # Verify the model is importable in THIS interpreter. A child ``uv pip``
    # honouring a mismatched ``VIRTUAL_ENV`` (or any silent no-op) could report
    # success while the wheel landed elsewhere; without this re-check misaki
    # would still hit its runtime download. Fail closed → clean 503, never 500.
    if not spacy.util.is_package(_KOKORO_G2P_SPACY_MODEL):
        logger.error(
            "Kokoro TTS: spaCy G2P model install reported success but '%s' is "
            "still not importable in this interpreter (%s)",
            _KOKORO_G2P_SPACY_MODEL,
            sys.prefix,
        )
        return False, _g2p_model_install_hint()
    return True, None


def _ensure_kokoro_g2p_model_ready() -> None:
    """Raise a clean 503 if misaki's spaCy G2P model can't be made available.

    Single-flight, NON-blocking: the first cold request acquires
    ``_G2P_MODEL_LOCK`` and runs the one-time install (up to the subprocess's
    300 s timeout); concurrent cold requests do NOT block waiting on the lock.
    Because the ``/v1/audio/speech`` route offloads this to the shared route
    thread pool, a first-use burst that all blocked here would tie up worker
    threads for the whole install window and stall unrelated endpoints — so a
    request that finds the install already in flight returns a transient 503
    ("retry shortly") and frees its worker immediately.

    SUCCESS is cached for the process lifetime; a FAILURE is cached only for
    ``_G2P_MODEL_RETRY_COOLDOWN_S`` so a transient index outage / timeout can
    recover WITHOUT a server restart (re-probe after the cooldown; cache
    success once the dependency is present).
    """
    global _G2P_MODEL_READY, _G2P_MODEL_REASON, _G2P_MODEL_RETRY_AFTER

    import time

    from fastapi import HTTPException

    if _G2P_MODEL_READY:
        return  # success is sticky for the process lifetime

    # A recent failure is served from cache until its cooldown expires — don't
    # re-run the (up to 300 s) install on every request while it's still broken.
    if _G2P_MODEL_REASON is not None and time.monotonic() < _G2P_MODEL_RETRY_AFTER:
        raise HTTPException(status_code=503, detail=_G2P_MODEL_REASON)

    if _G2P_MODEL_LOCK.acquire(blocking=False):
        try:
            # Re-check under the lock: another request may have just resolved it
            # or set a fresh cooldown while we were racing to acquire.
            if not _G2P_MODEL_READY and time.monotonic() >= _G2P_MODEL_RETRY_AFTER:
                ready, reason = _probe_kokoro_g2p_model()
                if ready:
                    _G2P_MODEL_READY, _G2P_MODEL_REASON = True, None
                else:
                    _G2P_MODEL_REASON = reason
                    _G2P_MODEL_RETRY_AFTER = (
                        time.monotonic() + _G2P_MODEL_RETRY_COOLDOWN_S
                    )
        finally:
            _G2P_MODEL_LOCK.release()
    else:
        # Another request holds the lock and is running the install; don't
        # occupy a worker thread blocked on it. Fail fast + retryable.
        raise HTTPException(
            status_code=503,
            detail=(
                f"Kokoro TTS is preparing its spaCy G2P model "
                f"('{_KOKORO_G2P_SPACY_MODEL}', one-time first-use setup). "
                "Retry in a few seconds."
            ),
        )

    if _G2P_MODEL_READY:
        return
    raise HTTPException(
        status_code=503, detail=_G2P_MODEL_REASON or _g2p_model_install_hint()
    )


def _reset_g2p_model_state() -> None:
    """Test hook: forget the cached spaCy-G2P-model readiness verdict."""
    global _G2P_MODEL_READY, _G2P_MODEL_REASON, _G2P_MODEL_RETRY_AFTER
    _G2P_MODEL_READY = None
    _G2P_MODEL_REASON = None
    _G2P_MODEL_RETRY_AFTER = 0.0


def _ensure_kokoro_g2p_ready() -> None:
    """Ensure Kokoro's espeak G2P can initialize; repair or 503 if not.

    Cached per-process and coalesced under a lock so concurrent first
    requests on a cold worker probe only once. The sweep spawns
    subprocesses and can block, so callers on the event loop MUST offload
    this to a worker thread (the ``/v1/audio/speech`` route does). If
    neither the bundled nor any system espeak-ng can initialize, raises a
    clean 503 — the worker survives; the raw C abort never reaches the
    request path.
    """
    global _ESPEAK_READY, _ESPEAK_REASON

    from fastapi import HTTPException

    if _ESPEAK_READY is None:
        with _ESPEAK_LOCK:
            # Another request may have resolved the verdict while we waited.
            if _ESPEAK_READY is None:
                _ESPEAK_READY, _ESPEAK_REASON = _probe_espeak_readiness()

    if _ESPEAK_READY:
        return
    raise HTTPException(status_code=503, detail=_ESPEAK_REASON)


def _reset_espeak_state() -> None:
    """Test hook: forget the cached espeak readiness verdict."""
    global _ESPEAK_READY, _ESPEAK_REASON
    _ESPEAK_READY = None
    _ESPEAK_REASON = None


# Kokoro voice ids encode language in the first letter. These are the KNOWN
# NON-English prefixes (misaki.ja/zh/es/fr/hi/it/pt), which use their own
# tokenizers and don't need spaCy ``en_core_web_sm``. English is ``a``
# (American) / ``b`` (British); anything ELSE — including a custom/unknown id —
# is treated as English so it can't slip past the gate into the #1254
# first-request ``SystemExit``.
_KOKORO_NON_ENGLISH_VOICE_PREFIXES = frozenset("jzefhip")


def _kokoro_voice_needs_en_g2p(voice: str | None) -> bool:
    """True when a Kokoro voice uses misaki's ENGLISH G2P (``misaki.en``) — the
    only path that needs the spaCy ``en_core_web_sm`` model (#1254).

    FAIL-SAFE: only an explicitly-recognized non-English language prefix
    (:data:`_KOKORO_NON_ENGLISH_VOICE_PREFIXES`) skips the gate. English
    (``a``/``b``), an omitted voice, AND any unrecognized/custom id all default
    to True — the canonical default voice (``af_heart``) is English, and a
    custom English voice must not bypass the gate.
    """
    if not voice:
        return True
    return voice[:1].lower() not in _KOKORO_NON_ENGLISH_VOICE_PREFIXES


def require_kokoro_runtime(voice: str | None = None) -> None:
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

    F-K-KOKORO-SPACY (#1254): misaki's English G2P tokenizer needs the
    spaCy ``en_core_web_sm`` model, which it downloads lazily on the
    first ``generate`` in a way that ``SystemExit``s under uv-tool /
    console-script launchers (an opaque 500). This pre-resolves it here
    (installed once, in a contained subprocess) so a missing model 503s
    cleanly instead. Gated on ``voice`` — only ENGLISH voices use
    ``misaki.en``; Japanese/Mandarin/etc. voices don't need this model.

    F-K-KOKORO-ESPEAK: beyond the missing-extra check, this also
    validates that misaki's espeak-ng G2P backend can actually
    initialize (in a subprocess, so a broken bundled dylib can't abort
    the worker) and repairs it to a system espeak-ng when possible —
    otherwise a Kokoro request with any out-of-dictionary word would
    hard-crash the whole server.

    The check is intentionally narrow — Chatterbox / VibeVoice /
    VoxCPM don't depend on misaki, so this helper isn't called for
    those families. The TTS lane probe still gates them all the same.
    """
    import importlib.util

    from fastapi import HTTPException

    if importlib.util.find_spec(_KOKORO_EXTRA_DEP) is None:
        raise HTTPException(
            status_code=503,
            detail=f"{_KOKORO_EXTRA_HINT}",
        )
    # Run the CHEAP espeak readiness check first (a ~1-2 s subprocess self-test)
    # so a host missing espeak fails fast, BEFORE the spaCy model bootstrap —
    # which can spend up to 300 s on a network install.
    _ensure_kokoro_g2p_ready()
    # F-K-KOKORO-SPACY (#1254): misaki's English G2P also needs the spaCy
    # ``en_core_web_sm`` model, downloaded lazily on first ``generate`` in a
    # way that SystemExits under uv-tool launchers → opaque 500. Resolve it
    # here (503 on failure) — but ONLY for English voices; other languages
    # don't use misaki.en / spaCy.
    if _kokoro_voice_needs_en_g2p(voice):
        _ensure_kokoro_g2p_model_ready()


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


# Known audio alias surface — kept narrow on purpose. The list is
# deliberately a substring match — any ``whisper``/``parakeet``/
# ``kokoro``/``chatterbox``/``vibevoice``/``voxcpm`` alias counts,
# INCLUDING HF-style ids that embed the engine name
# (``mlx-community/Kokoro-82M-bf16``, ``mlx-community/whisper-large-v3-mlx``)
# — those are the canonical pass-through cases the STT/TTS routes
# accept, so the boot guard MUST recognise them as audio too. Codex
# raised in review: the previous comment claimed HF ids "fell through"; the
# code (and test ``test_is_audio_model_alias_recognises_common_aliases``)
# disagree — they ARE matched, and intentionally so.
#
# Bare strings that don't contain any of these tokens fall through
# unchanged — the boot guard is silent for text/vision/embedding
# aliases.
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

    R10-C1: registry-first. :mod:`vllm_mlx.audio.registry` is the
    authoritative source of truth (every entry's HF id was verified
    on hf.co at registry-introduction time). The legacy substring
    match against :data:`_AUDIO_ALIAS_TOKENS` is preserved as a
    fallback for HF ids of audio engines that haven't yet been added
    to the registry (third-party Whisper / Parakeet ports, future
    mlx-community uploads). Both checks are case-insensitive.
    """
    if not isinstance(model_name, str) or not model_name:
        return False
    # Registry hit — authoritative, no substring guessing required.
    # Late-import so a base install that never reaches the audio path
    # doesn't pay the JSON read at module import time.
    try:
        from .registry import is_audio_name

        if is_audio_name(model_name):
            return True
    except Exception:
        # Registry load failed (malformed JSON, missing file in a
        # partial install). Fall through to the substring fallback so
        # a busted registry doesn't deafen the boot guard — the legacy
        # surface still catches the common aliases.
        pass
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
