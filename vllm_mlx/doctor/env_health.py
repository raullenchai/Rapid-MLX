# SPDX-License-Identifier: Apache-2.0
"""Environment-health probes for ``rapid-mlx doctor``.

The whole module is a tree of cheap, side-effect-free checks: chip / OS / disk,
Python interpreter, packages installed, HF cache, network, shell integration,
optional dev tools. The user runs ``rapid-mlx doctor`` to answer one question
— "is my install/env broken?" — so every probe must:

* run in well under a second (no model load, no engine init, no server boot);
* never escalate to sudo or read user data outside ``~/.cache/huggingface``;
* report a deterministic status (✓ / ⚠ / ✗) with a one-line label.

Total wall-clock for ``rapid-mlx doctor`` ≤ 5 s on a warm cache, dominated by
the single 2-second network HEAD against ``huggingface.co`` (which downgrades
to ⚠ on timeout — never ✗).

The CLI in ``doctor/cli.py`` consumes ``run_all()`` and renders the report.
Tests in ``tests/test_doctor_env_health.py`` cover each section's probe.
"""

from __future__ import annotations

import importlib.metadata as _im
import importlib.util as _iu
import os
import platform
import plistlib
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class CheckStatus(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class Check:
    """One row in a section. ``detail`` is shown under ``--verbose``."""

    label: str
    status: CheckStatus
    detail: str = ""


@dataclass
class Section:
    title: str
    checks: list[Check] = field(default_factory=list)

    def add(self, label: str, status: CheckStatus, detail: str = "") -> None:
        self.checks.append(Check(label=label, status=status, detail=detail))


@dataclass
class Report:
    sections: list[Section] = field(default_factory=list)

    def all_checks(self) -> list[Check]:
        return [c for s in self.sections for c in s.checks]

    @property
    def n_ok(self) -> int:
        return sum(1 for c in self.all_checks() if c.status is CheckStatus.OK)

    @property
    def n_warn(self) -> int:
        return sum(1 for c in self.all_checks() if c.status is CheckStatus.WARN)

    @property
    def n_fail(self) -> int:
        return sum(1 for c in self.all_checks() if c.status is CheckStatus.FAIL)

    @property
    def exit_code(self) -> int:
        # Spec: warnings never fail the exit code — only ✗ items do. CI scripts
        # that gate on doctor want a strict "is anything broken" signal, not
        # "is anything not perfect".
        return 1 if self.n_fail else 0


# ---------------------------------------------------------------------------
# Required + optional package matrices
# ---------------------------------------------------------------------------

# Each tuple: (PyPI distribution name, human label). The doctor doesn't
# enforce version pins here — pyproject already does that at install time.
# Showing the version is enough for the user to grep "old transformers?".
REQUIRED_PACKAGES: list[tuple[str, str]] = [
    ("mlx", "mlx"),
    ("mlx-lm", "mlx-lm"),
    ("transformers", "transformers"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("rapid-mlx", "rapid-mlx"),
]

# Each tuple: (distribution, label, install hint). Missing optionals are ⚠
# (warning) not ✗ — that's the whole point of "optional". The hint is
# echoed verbatim in the report so the user can copy-paste.
OPTIONAL_PACKAGES: list[tuple[str, str, str]] = [
    ("mlx-vlm", "mlx-vlm (vision extras)", "pip install 'rapid-mlx[vision]'"),
    ("mlx-audio", "mlx-audio (audio extras)", "pip install 'rapid-mlx[audio]'"),
    (
        "mlx-embeddings",
        "mlx-embeddings (embeddings extras)",
        "pip install 'rapid-mlx[embeddings]'",
    ),
]

# ``find_spec`` checks discoverability without importing heavyweight audio
# packages such as scipy, numba, and spaCy.  Keep this aligned with the
# runtime dependencies declared by the ``audio`` extra in pyproject.toml.
_AUDIO_IMPORTS: tuple[tuple[str, str], ...] = (
    ("mlx-audio", "mlx_audio"),
    ("f5-tts-mlx", "f5_tts_mlx"),
    ("sounddevice", "sounddevice"),
    ("soundfile", "soundfile"),
    ("scipy", "scipy"),
    ("numba", "numba"),
    ("tiktoken", "tiktoken"),
    ("misaki", "misaki"),
    ("spacy", "spacy"),
    ("num2words", "num2words"),
    ("loguru", "loguru"),
    ("espeakng-loader", "espeakng_loader"),
    ("phonemizer-fork", "phonemizer"),
    ("cn2an", "cn2an"),
)

# The macOS desktop sidecar deliberately installs ``rapid-mlx[audio-desktop]``
# (``apps/rapid-mac/scripts/build-sidecar.sh``), a bounded extra that is just
# ``mlx-audio`` + ``soundfile`` — the general-purpose TTS stack (f5-tts-mlx,
# spaCy, misaki, numba, …) is excluded to stay under the bundle size gate.
# Grading that bundle against ``_AUDIO_IMPORTS`` reported a perfectly healthy
# signed 0.12.18 build as "incomplete — missing: f5-tts-mlx, numba, tiktoken,
# …", i.e. the exact set difference between the two extras. Keep this aligned
# with the ``audio-desktop`` extra in pyproject.toml (locked by
# ``tests/test_audio_desktop_extra.py``).
_AUDIO_DESKTOP_IMPORTS: tuple[tuple[str, str], ...] = (
    ("mlx-audio", "mlx_audio"),
    ("soundfile", "soundfile"),
)

# Distributions the desktop sidecar deliberately does NOT ship. The bundle is
# built to a hard size cap (``.github/workflows/rapid-mac-release.yml`` gates on
# BUNDLE_SIZE_CAP_MB), so ``build-sidecar.sh`` installs the bounded
# ``[audio-desktop]`` extra plus ``mlx-vlm --no-deps`` + Pillow, and nothing
# else. ``mlx-embeddings`` is simply not part of the desktop product surface.
# Reporting it as "not installed — reinstall the app" was doubly wrong: the
# install is healthy, and reinstalling reproduces the identical warning
# forever. These rows are informational, never ⚠.
_DESKTOP_EXCLUDED_DISTS: frozenset[str] = frozenset({"mlx-embeddings"})

# Where a bundled sidecar came from decides what "repair" even means, so the
# two sources must not share one hint.
#
# * ``embedded`` — ``Rapid-MLX Desktop.app/Contents/Resources/rapid-mlx/``.
#   Inside the code-signed, notarized app; pip-installing into it breaks the
#   signature seal and Gatekeeper then rejects the app ("a sealed resource is
#   missing or invalid"). Reinstalling the app genuinely replaces it.
# * ``runtime-override`` — ``~/Library/Application Support/Rapid/
#   runtime-override/rapid-mlx/``, written by the bootstrapper on first launch
#   of the slim DMG. It lives OUTSIDE the app bundle and deliberately "survives
#   desktop upgrades" (``ServerLocator.swift``), and the bootstrapper
#   short-circuits its download while the cache is present
#   (``build-bootstrapper-dmg.sh``). So "reinstall the .app" is actively wrong
#   here — it typically changes nothing.
#
#   There is no in-app "repair the runtime" action to point at, and no
#   automatic re-download either. Settings → check for updates hands off to
#   Sparkle, which updates the *app* bundle, and ``UpdateChecker`` explicitly
#   does not act on the manifest's ``sidecar_*`` fields. The bootstrapper that
#   originally populated this slot is not part of the source tree any more, and
#   the release workflow builds only the full DMG. So "remove it and relaunch"
#   is NOT self-repairing on its own: with no bundled sidecar the desktop just
#   lands on the missing-runtime overlay, whose only actions are Recheck and an
#   app-update download.
#
#   What does work today, in this order: install the current
#   Rapid-MLX Desktop.app — its DMG ships a sidecar at
#   ``Contents/Resources/rapid-mlx/`` (``build.sh``) — and only then remove the
#   override, so ``ServerLocator.find()`` resolves the bundled slot. Doing it
#   the other way round leaves a slim install stranded. Note that reinstalling
#   alone is not enough when the override's VERSION is equal or newer: it keeps
#   winning over the bundled copy (``shouldPreferBundled``), which is exactly
#   why the removal step is required.
_EMBEDDED_REPAIR_HINT = (
    "reinstall Rapid-MLX Desktop.app — the bundled sidecar's Python "
    "environment is code-signed and must not be pip-installed into"
)
_RUNTIME_OVERRIDE_REPAIR_HINT_TEMPLATE = (
    "this runtime at {root} lives outside the app bundle and no app update "
    "replaces it — install the current Rapid-MLX Desktop.app (its DMG ships a "
    "sidecar), then remove {root} and relaunch so the bundled sidecar is used"
)


def _module_available(module: str) -> bool:
    """Return whether *module* is discoverable, without importing it."""
    try:
        return _iu.find_spec(module) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _bundled_sidecar_root() -> Path | None:
    """Return the sidecar bundle root when doctor is running from a managed
    sidecar's embedded interpreter, else ``None``.

    The bundle layout produced by ``build-sidecar.sh`` is fixed, and both
    managed slots share it (``ServerLocator.swift`` resolves each through the
    same ``rapid-mlx/bin/rapid-mlx`` suffix)::

        <root>/bin/rapid-mlx        # shell shim (sidecar-shim.sh)
        <root>/python/bin/python3.12
        <root>/site-packages/

    We fingerprint that shape off ``sys.executable`` rather than trusting an
    env var: the desktop is not the only thing that spawns the sidecar (the
    user can run the shim by hand), so any env-var marker would be absent in
    exactly the cases that matter.  The shape alone is not provenance,
    though: a custom Python installation can reproduce it.  Require one of
    the two locations ``ServerLocator`` owns before changing doctor contracts.
    """
    try:
        exe = Path(sys.executable).resolve()
    except OSError:
        return None
    if len(exe.parents) < 3:
        return None
    if exe.parent.name != "bin" or exe.parents[1].name != "python":
        return None
    root = exe.parents[2]
    if not (root / "site-packages").is_dir():
        return None
    if not (root / "bin" / "rapid-mlx").exists():
        return None

    # Full DMG: <any app name>.app/Contents/Resources/rapid-mlx.  Users may
    # rename the app after installation, so the stable macOS bundle suffix is
    # the provenance signal rather than the display name.
    embedded_shape = (
        root.parent.name == "Resources"
        and root.parents[1].name == "Contents"
        and root.parents[2].suffix == ".app"
    )
    embedded = False
    if embedded_shape:
        try:
            with (root.parents[1] / "Info.plist").open("rb") as handle:
                plist = plistlib.load(handle)
        except (OSError, plistlib.InvalidFileException, ValueError, TypeError):
            plist = None
        bundle_id = plist.get("CFBundleIdentifier") if isinstance(plist, dict) else None
        # Production is exact; dogfood isolation appends a suffix so multiple
        # personas can coexist without sharing preferences.
        embedded = isinstance(bundle_id, str) and (
            bundle_id == "com.rapidmlx.rapid"
            or bundle_id.startswith("com.rapidmlx.rapid.dogfood-")
        )

    # Runtime override: ApplicationSupportLocator may root this under a
    # dogfood HOME, but the suffix below is invariant.  Match the complete
    # product-owned suffix so an arbitrary ``runtime-override/rapid-mlx``
    # directory is not enough to suppress ordinary CLI diagnostics.
    runtime_override = False
    home = os.environ.get("HOME", "").strip()
    if home:
        try:
            expected_override = (
                Path(home).expanduser().resolve()
                / "Library"
                / "Application Support"
                / "Rapid"
                / "runtime-override"
                / "rapid-mlx"
            ).resolve()
        except OSError:
            expected_override = None
        runtime_override = root == expected_override
    return root if embedded or runtime_override else None


def _sidecar_repair_hint(root: Path) -> str:
    """Pick the repair hint matching which managed slot *root* sits in.

    ``runtime-override`` is detected by its parent directory name rather than
    by a full absolute-path match against ``~/Library/Application Support`` —
    the desktop resolves Application Support through ``ApplicationSupportLocator``
    which honours an overridden ``$HOME`` (dogfood / test launches), so a
    hardcoded home-relative path would misclassify exactly those runs. Anything
    that is not the override slot is the in-bundle copy.

    The override hint embeds the resolved path because doctor already knows it
    exactly and the recovery is a manual removal — a generic "remove the cached
    runtime" would leave the user guessing at a location that moves with
    ``$HOME``.
    """
    if root.parent.name == "runtime-override":
        return _RUNTIME_OVERRIDE_REPAIR_HINT_TEMPLATE.format(root=root)
    return _EMBEDDED_REPAIR_HINT


# ---------------------------------------------------------------------------
# Section: System
# ---------------------------------------------------------------------------


def _detect_apple_silicon() -> tuple[str | None, int | None]:
    """Return (chip_brand, ram_gb) or (None, None) on non-Mac / sysctl failure.

    ``sysctl`` is a system binary that's always present on macOS; we use it
    instead of ``platform.processor()`` because the latter returns ``''`` on
    arm64 macOS in some Python builds (CPython issue #97965, present in 3.10+).
    """
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None, None
    try:
        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        memsize = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    brand_str = brand.stdout.strip() or None
    try:
        ram_gb: int | None = round(int(memsize.stdout.strip()) / (1024**3))
    except (TypeError, ValueError):
        ram_gb = None
    return brand_str, ram_gb


def _disk_free_gb(path: Path) -> float | None:
    try:
        usage = shutil.disk_usage(path)
        return usage.free / (1024**3)
    except (OSError, FileNotFoundError):
        return None


def _hf_cache_dir() -> Path:
    """Return the HuggingFace **hub** cache dir.

    Resolution order matches huggingface_hub itself:

      1. ``$HF_HUB_CACHE`` (the most specific override; some users point this
         at an external SSD while leaving ``HF_HOME`` alone).
      2. ``$HF_HOME/hub`` (the canonical sub-path under a custom HF_HOME).
      3. ``~/.cache/huggingface/hub`` (the default the hub library writes to).

    Earlier revisions returned ``~/.cache/huggingface`` (no ``hub`` suffix).
    That was wrong: real downloads land in the ``hub`` subdir, so a missing
    or unwritable hub would have been masked by a probe that checked the
    parent. Codex-review round 1 caught this; the env-var fall-through plus
    the trailing ``hub`` segment fix both problems at once.
    """
    env_hub_cache = os.environ.get("HF_HUB_CACHE")
    if env_hub_cache:
        return Path(env_hub_cache).expanduser()
    env_home = os.environ.get("HF_HOME")
    if env_home:
        return Path(env_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


# Wall-clock budget for the recursive HF-cache size walk. The whole doctor
# run is contracted at ≤ 5 s and the network probe alone can spend 2 s, so
# the cache walk must finish in ~1 s on a hot FS / abort cleanly on a cold
# or network-mounted cache. Codex-review round 1 flagged the previous
# unbounded walk as a contract violation on TB-scale caches.
_CACHE_WALK_BUDGET_S = 1.5


def _dir_size_gb(path: Path, *, budget_s: float = _CACHE_WALK_BUDGET_S) -> float | None:
    """Sum file sizes under ``path``.

    Returns:
        ``None`` if the directory doesn't exist OR the walk hit the wall-clock
        budget (which is itself useful signal — "cache is too large/slow to
        size", which the caller renders as "unknown").

        Otherwise the total in GB.

    Walks with ``os.walk(..., followlinks=False)`` so LM-Studio-style symlinks
    don't double-count. The deadline is checked **inside** the per-file loop,
    not just between directories: HF cache's ``blobs/`` subdir is flat with
    thousands of entries, so a per-directory deadline would let a single
    cold-cache stat() storm blow past the 1.5 s budget. Codex review round 2
    caught the per-directory variant as a contract violation; this version
    aborts on the very next file once the deadline expires.
    """
    import time as _time

    if not path.exists():
        return None
    deadline = _time.monotonic() + budget_s
    total = 0
    try:
        for root, _dirs, files in os.walk(path, followlinks=False):
            for f in files:
                if _time.monotonic() >= deadline:
                    # Budget exhausted mid-directory; partial total isn't
                    # useful (lower bound only). Caller renders "unknown".
                    return None
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    # Broken symlink, permission denied — skip silently;
                    # this probe is "is the cache enormous?", not "audit
                    # every file".
                    continue
            if _time.monotonic() >= deadline:
                return None
    except OSError:
        return None
    return total / (1024**3)


def section_system() -> Section:
    """Hardware + OS section.

    ⚠ on:
      * non-Apple-Silicon (rapid-mlx targets M-series; works elsewhere but
        with Metal-fallback caveats).
      * < 20 GB free disk (model weights are big).
      * HF cache > 100 GB (suggest ``rapid-mlx rm`` cleanup).

    ✗ on:
      * < 5 GB free disk (next download will fail).
    """
    s = Section("System")

    chip, ram_gb = _detect_apple_silicon()
    if chip:
        ram_str = f"{ram_gb} GB" if ram_gb else "unknown RAM"
        s.add(
            f"Apple Silicon ({chip}, {ram_str})",
            CheckStatus.OK,
            detail=f"chip={chip} ram_gb={ram_gb}",
        )
    elif platform.system() == "Darwin":
        s.add(
            "Non-Apple-Silicon Mac — MLX requires arm64",
            CheckStatus.WARN,
            detail=f"machine={platform.machine()}",
        )
    else:
        s.add(
            f"Not macOS ({platform.system()}) — MLX is Apple-only",
            CheckStatus.WARN,
            detail=f"system={platform.system()} machine={platform.machine()}",
        )

    mac_ver = platform.mac_ver()[0]
    if mac_ver:
        s.add(
            f"macOS {mac_ver} (Darwin {platform.release()})",
            CheckStatus.OK,
            detail=f"mac_ver={mac_ver} darwin={platform.release()}",
        )
    else:
        s.add(
            f"OS: {platform.system()} {platform.release()}",
            CheckStatus.OK,
            detail=f"system={platform.system()} release={platform.release()}",
        )

    free_gb = _disk_free_gb(Path.home())
    if free_gb is None:
        s.add(
            "Free disk: unknown",
            CheckStatus.WARN,
            detail="shutil.disk_usage($HOME) failed",
        )
    elif free_gb < 5:
        s.add(
            f"Free disk: {free_gb:.0f} GB (very low — next download may fail)",
            CheckStatus.FAIL,
            detail=f"free_gb={free_gb:.1f}",
        )
    elif free_gb < 20:
        s.add(
            f"Free disk: {free_gb:.0f} GB (low — large models need 20+ GB)",
            CheckStatus.WARN,
            detail=f"free_gb={free_gb:.1f}",
        )
    else:
        s.add(
            f"Free disk: {free_gb:.0f} GB",
            CheckStatus.OK,
            detail=f"free_gb={free_gb:.1f}",
        )

    cache = _hf_cache_dir()
    if not cache.exists():
        s.add(
            f"HF cache: not present ({cache})",
            CheckStatus.OK,
            detail=f"path={cache}",
        )
    else:
        cache_size_gb = _dir_size_gb(cache)
        if cache_size_gb is None:
            # Walk hit the time budget — likely a very large or network-
            # mounted cache. Don't penalise the user; just say so.
            s.add(
                f"HF cache size: too large to size in {_CACHE_WALK_BUDGET_S:.1f}s "
                "(consider `rapid-mlx rm` if unused models accumulated)",
                CheckStatus.WARN,
                detail=f"path={cache} budget_s={_CACHE_WALK_BUDGET_S}",
            )
        elif cache_size_gb > 100:
            s.add(
                f"HF cache size: {cache_size_gb:.0f} GB "
                "(consider `rapid-mlx rm` for unused models)",
                CheckStatus.WARN,
                detail=f"cache_gb={cache_size_gb:.1f} path={cache}",
            )
        else:
            s.add(
                f"HF cache size: {cache_size_gb:.1f} GB",
                CheckStatus.OK,
                detail=f"cache_gb={cache_size_gb:.1f} path={cache}",
            )

    return s


# ---------------------------------------------------------------------------
# Section: Python
# ---------------------------------------------------------------------------


def _install_location() -> tuple[str, Path]:
    """Classify where ``rapid-mlx`` is installed: ``uv tool``, ``pipx``,
    ``virtualenv``, ``system``. Returned label is for display; the path
    is shown in --verbose."""
    exe = Path(sys.executable).resolve()
    parts = exe.parts
    lower = str(exe).lower()
    if "uv/tools" in lower or "/uv/tools/" in lower:
        return "uv tool", exe
    if "pipx" in lower:
        return "pipx", exe
    # site-packages under a venv-style structure
    if (
        sys.prefix != getattr(sys, "base_prefix", sys.prefix)
        or "VIRTUAL_ENV" in os.environ
    ):
        return "virtualenv", exe
    if "Cellar" in parts or "/homebrew/" in lower:
        return "Homebrew", exe
    return "system", exe


def section_python() -> Section:
    s = Section("Python")

    py_ver = ".".join(str(x) for x in sys.version_info[:3])
    # Defensive: pyproject pins ``requires-python = ">=3.10"`` so install-
    # time pip would already have refused — but doctor should still tell the
    # user clearly if they somehow got rapid-mlx onto an older interpreter
    # (e.g. a hand-copied wheel). Ruff's UP036 flags this as dead under our
    # support matrix; that's the point of the defensive branch.
    if sys.version_info >= (3, 10):  # noqa: UP036
        s.add(
            f"Python {py_ver}",
            CheckStatus.OK,
            detail=f"executable={sys.executable}",
        )
    else:  # pragma: no cover — only reachable on unsupported interpreters
        s.add(
            f"Python {py_ver} (rapid-mlx requires >= 3.10)",
            CheckStatus.FAIL,
            detail=f"executable={sys.executable}",
        )

    label, path = _install_location()
    s.add(
        f"Install location: {label} ({path})",
        CheckStatus.OK,
        detail=f"sys.executable={path}",
    )

    return s


# ---------------------------------------------------------------------------
# Section: packages
# ---------------------------------------------------------------------------


def _safe_version(dist: str) -> str | None:
    try:
        return _im.version(dist)
    except _im.PackageNotFoundError:
        return None


def _pil_importable() -> bool:
    """Lightweight probe: does mlx-vlm's ``from PIL import Image`` actually
    work?

    mlx-vlm does ``from PIL import Image`` at module load, so a present
    mlx-vlm with an absent PIL — the Homebrew ``pip install --no-deps
    mlx-vlm`` state (#1126) — is a FALSE positive: metadata says installed,
    but every vision path crashes on import.

    We perform the EXACT lightweight import mlx-vlm uses rather than a
    ``find_spec('PIL')`` probe: ``find_spec`` only proves *something named
    PIL is discoverable*, so a shadowed namespace dir or a damaged Pillow
    whose real ``from PIL import Image`` raises would still get a green row.
    We then run a minimal native-backed op (``Image.new``) so a Pillow whose
    Python layer imports but whose ``_imaging`` C extension is missing or
    ABI-mismatched — present-but-broken, not merely absent — is caught too,
    without relying on the version-specific moment Pillow first touches
    ``_imaging``. ``PIL.Image`` + a 1×1 allocation is microsecond-cheap and
    does NOT pull torch the way a real ``import mlx_vlm`` would, so it stays
    well within doctor's ≤5 s budget. ANY failure (missing, shadowed, broken
    native ext) ⇒ not importable."""
    try:
        from PIL import Image

        # Force the native ``_imaging`` backend to actually initialise — a
        # broken/ABI-mismatched C ext otherwise slips through as healthy.
        Image.new("RGB", (1, 1))
    except Exception:
        # ImportError (absent/shadowed), OSError (broken native ext), or any
        # other load-time failure — all mean the real vision import can't run.
        return False
    return True


def _version_at_least(ver: str, minimum: tuple[int, ...]) -> bool:
    """Compare a PEP 440 version string against a ``(major, minor, patch)``
    floor using leading-numeric tuple semantics.

    Pre-release/local-version suffixes are stripped — ``"0.5.0rc1"`` and
    ``"0.5.0+local"`` both compare equal to ``(0, 5, 0)``. We don't need
    full PEP 440 ordering here; the only floor we care about is the DFlash
    bump (``mlx-vlm >= 0.5.0``). A malformed version returns ``False`` —
    safer to nudge the user to upgrade than to silently pass.
    """
    parts: list[int] = []
    for raw in ver.split("."):
        # Stop at the first non-numeric chunk (e.g. "0rc1", "5+local").
        digits = ""
        for ch in raw:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    if not parts:
        return False
    # Pad to the shape of ``minimum`` so the tuple comparison is well-defined.
    while len(parts) < len(minimum):
        parts.append(0)
    return tuple(parts[: len(minimum)]) >= minimum


def section_required_packages() -> Section:
    s = Section("Required Packages")
    for dist, label in REQUIRED_PACKAGES:
        ver = _safe_version(dist)
        if ver:
            s.add(
                f"{label} {ver}",
                CheckStatus.OK,
                detail=f"distribution={dist} version={ver}",
            )
        else:
            s.add(
                f"{label} not installed",
                CheckStatus.FAIL,
                detail=f"distribution={dist} missing",
            )
    return s


def section_updates(
    *,
    installed: Callable[[], str | None] | None = None,
    fetch_latest: Callable[[], str | None] | None = None,
    install_info: object | None = None,
) -> Section:
    """Is the installed rapid-mlx the latest release?

    Best-effort and network-bounded — reuses ``_version_check``'s 2 s timeout and
    cache-first fetch, and warms the same cache the CLI's staleness banner reads.
    Never fatal: an unknown installed version or an unreachable endpoint
    downgrades to ⚠ (like the HF network probe), so ``doctor`` never exits
    non-zero just because the machine is offline.
    """
    s = Section("Updates")
    from vllm_mlx import _version_check as vc

    cur = (installed or vc._installed_version)()
    if not cur:
        s.add(
            "installed rapid-mlx version unknown",
            CheckStatus.WARN,
            detail="_installed_version() returned None",
        )
        return s

    latest = (fetch_latest or vc.get_latest_version)()
    if latest is None:
        s.add(
            f"rapid-mlx {cur} — couldn't reach the update endpoint (offline?)",
            CheckStatus.WARN,
            detail="version endpoint unreachable; freshness check skipped",
        )
        return s

    pc, pl = vc._parse_version(cur), vc._parse_version(latest)
    if pc is None or pl is None:
        # One side is a dev/rc/git-describe build ``_parse_version`` won't
        # touch (e.g. ``0.10.15.dev3``, ``0.11.0rc1``, ``0.10``). We can't
        # order them, so DON'T fall through to a green "up to date" — that
        # would falsely reassure a user who might well be behind. Downgrade
        # to ⚠ like every other uncertain branch in this section.
        s.add(
            f"rapid-mlx {cur} — can't compare against latest {latest} "
            "(unrecognized version format); freshness check skipped",
            CheckStatus.WARN,
            detail=(
                f"installed={cur} latest={latest} "
                f"parsed_installed={pc} parsed_latest={pl}"
            ),
        )
    elif pl > pc:
        info = install_info if install_info is not None else vc.detect_install_method()
        cmd = getattr(info, "upgrade_command", None) or "rapid-mlx upgrade"
        s.add(
            f"update available: {latest} (installed {cur}) — run `{cmd}`",
            CheckStatus.WARN,
            detail=(
                f"installed={cur} latest={latest} method={getattr(info, 'method', '?')}"
            ),
        )
    else:
        s.add(
            f"rapid-mlx {cur} is up to date",
            CheckStatus.OK,
            detail=f"installed={cur} latest={latest}",
        )
    return s


def section_optional_packages() -> Section:
    s = Section("Optional Packages")
    # RC 0.12.18: the signed desktop bundle installs the bounded
    # ``[audio-desktop]`` extra, but doctor graded it against the full
    # ``[audio]`` contract and reported a healthy build as "incomplete", then
    # told the user to pip-install into a code-signed app. Detect the managed
    # sidecar once and swap the contract, the remediation hint, and the
    # treatment of extras the bundle intentionally omits.
    sidecar_root = _bundled_sidecar_root()
    bundled = sidecar_root is not None
    audio_contract = _AUDIO_DESKTOP_IMPORTS if bundled else _AUDIO_IMPORTS
    repair_hint = _sidecar_repair_hint(sidecar_root) if sidecar_root else None
    for dist, label, install_hint in OPTIONAL_PACKAGES:
        hint = repair_hint if repair_hint else install_hint
        ver = _safe_version(dist)
        if bundled and not ver and dist in _DESKTOP_EXCLUDED_DISTS:
            # Not a defect: this extra is outside the desktop product surface,
            # so there is nothing for the user to repair. ⚠ + "reinstall" here
            # would survive any reinstall and train users to ignore the
            # section.
            s.add(
                f"{label} not bundled with Rapid-MLX Desktop (not required)",
                CheckStatus.OK,
                detail=f"distribution={dist} bundled=false reason=excluded-from-desktop",
            )
            continue
        if ver:
            # #1255: mlx-vlm can install mlx-audio transitively without
            # respecting Rapid-MLX's supported audio range. Presence alone
            # must not make doctor report that environment as healthy.
            if dist == "mlx-audio" and not (
                _version_at_least(ver, (0, 2, 9))
                and not _version_at_least(ver, (0, 4, 4))
            ):
                s.add(
                    f"{label} {ver} unsupported — rapid-mlx requires "
                    f"mlx-audio>=0.2.9,<0.4.4 (`{hint}`)",
                    CheckStatus.WARN,
                    detail=(
                        f"distribution={dist} version={ver} "
                        f"supported=>=0.2.9,<0.4.4 hint={hint}"
                    ),
                )
                continue
            if dist == "mlx-audio":
                missing = [
                    distribution
                    for distribution, module in audio_contract
                    if not _module_available(module)
                ]
                if missing:
                    missing_text = ", ".join(missing)
                    s.add(
                        f"{label} {ver} incomplete — missing: {missing_text} "
                        f"(`{hint}`)",
                        CheckStatus.WARN,
                        detail=(
                            f"distribution={dist} version={ver} "
                            f"missing={missing_text} hint={hint} "
                            f"contract={'audio-desktop' if bundled else 'audio'}"
                        ),
                    )
                    continue
            # #1126: mlx-vlm imports Pillow (PIL) at load. A present
            # mlx-vlm with an absent PIL (Homebrew `pip install --no-deps
            # mlx-vlm`) is a FALSE positive — metadata says "installed" but
            # every vision path crashes on `from PIL import Image` deep in
            # the FastAPI lifespan. Report it honestly and name the real
            # gap so the user fixes the right thing.
            if dist == "mlx-vlm" and not _pil_importable():
                s.add(
                    f"{label} {ver} present but Pillow (PIL) missing or "
                    f"broken — vision paths will fail (`{hint}`)",
                    CheckStatus.WARN,
                    detail=(
                        f"distribution={dist} version={ver} "
                        f"pil=missing-or-broken hint={hint}"
                    ),
                )
                continue
            s.add(
                f"{label} {ver}",
                CheckStatus.OK,
                detail=f"distribution={dist} version={ver}",
            )
        else:
            s.add(
                f"{label} not installed (`{hint}`)",
                CheckStatus.WARN,
                detail=f"distribution={dist} hint={hint}",
            )

    # DFlash is the headline 0.9.x feature and is gated on mlx-vlm >= 0.5.0.
    # The plain optional-package row above only reports presence/version —
    # it cannot say "you have mlx-vlm but it's too old for [dflash]". This
    # extra row makes the gate explicit so a fresh-install user knows whether
    # `pip install 'rapid-mlx[dflash]'` will actually work.
    #
    # The desktop bundle DOES ship mlx-vlm (pinned 0.6.3, ``--no-deps`` + Pillow
    # — the gemma-4 family needs it even in text-only mode), so unlike
    # mlx-embeddings this row is a real contract on a bundled sidecar and stays
    # gradeable; only the remediation wording changes.
    dflash_min = (0, 5, 0)
    dflash_hint = repair_hint or "pip install 'rapid-mlx[dflash]'"
    vision_hint = repair_hint or "pip install 'rapid-mlx[vision]'"
    vlm_ver = _safe_version("mlx-vlm")
    if vlm_ver and _version_at_least(vlm_ver, dflash_min):
        # #1126: same PIL honesty as the vision row — a version-adequate
        # mlx-vlm whose Pillow dep is missing can't actually run the
        # dflash/vision runtime, so don't paint it green.
        if not _pil_importable():
            s.add(
                "mlx-vlm 0.5.0+ (dflash extras) present but Pillow (PIL) "
                "missing or broken — dflash/vision paths will fail",
                CheckStatus.WARN,
                detail=(
                    f"distribution=mlx-vlm version={vlm_ver} "
                    f"pil=missing-or-broken hint={vision_hint}"
                ),
            )
        else:
            s.add(
                "mlx-vlm 0.5.0+ (dflash extras)",
                CheckStatus.OK,
                detail=f"distribution=mlx-vlm version={vlm_ver}",
            )
    else:
        current = vlm_ver or "not installed"
        s.add(
            f"mlx-vlm 0.5.0+ (dflash extras) not installed or too old "
            f"(current: {current}, need: 0.5.0+)",
            CheckStatus.WARN,
            detail=dflash_hint,
        )

    return s


# ---------------------------------------------------------------------------
# Section: HuggingFace cache
# ---------------------------------------------------------------------------


def _nearest_existing_parent(p: Path) -> Path | None:
    """Walk up ``p``'s ancestors until we find one that exists, or return
    ``None`` if even the filesystem root has somehow disappeared."""
    for ancestor in (p, *p.parents):
        if ancestor.exists():
            return ancestor
    return None


def section_hf_cache() -> Section:
    s = Section("HuggingFace Cache")

    cache = _hf_cache_dir()
    if cache.exists():
        # Codex review round 2: ``os.access`` returns True for a writable
        # regular file too, so a user who set ``HF_HUB_CACHE`` to a path
        # that's now a file (typo, mv accident, …) would see ✓ here and
        # then fail on the first download with a confusing error.
        if not cache.is_dir():
            s.add(
                f"{cache} exists but is NOT a directory",
                CheckStatus.FAIL,
                detail=f"path={cache} type=non-directory",
            )
        elif os.access(cache, os.W_OK):
            s.add(
                f"{cache} exists, writable",
                CheckStatus.OK,
                detail=f"path={cache}",
            )
        else:
            s.add(
                f"{cache} exists but NOT writable",
                CheckStatus.FAIL,
                detail=f"path={cache}",
            )
    else:
        # Missing cache isn't *always* a soft warning — if the nearest
        # existing parent isn't writable either, the first download will
        # fail trying to create ``cache``. Codex review round 2 caught
        # the previous unconditional WARN as silently green for
        # ``HF_HOME=/readonly/hf``.
        parent = _nearest_existing_parent(cache.parent)
        if parent is None or not os.access(parent, os.W_OK):
            s.add(
                f"{cache} does not exist and parent {parent} is NOT writable "
                "— next download will fail",
                CheckStatus.FAIL,
                detail=f"path={cache} parent={parent}",
            )
        else:
            s.add(
                f"{cache} does not exist yet (will be created on first download)",
                CheckStatus.WARN,
                detail=f"path={cache} parent={parent}",
            )

    # Disk free for the partition the cache lives on (or would live on).
    probe_dir = cache if cache.exists() else cache.parent
    if not probe_dir.exists():
        probe_dir = Path.home()
    free_gb = _disk_free_gb(probe_dir)
    if free_gb is None:
        s.add(
            "Free space on cache partition: unknown",
            CheckStatus.WARN,
            detail=f"probe_dir={probe_dir}",
        )
    elif free_gb < 5:
        s.add(
            f"Free space: {free_gb:.0f} GB (very low — model downloads will fail)",
            CheckStatus.FAIL,
            detail=f"free_gb={free_gb:.1f} probe_dir={probe_dir}",
        )
    else:
        s.add(
            f"Free space: {free_gb:.0f} GB",
            CheckStatus.OK,
            detail=f"free_gb={free_gb:.1f} probe_dir={probe_dir}",
        )

    return s


# ---------------------------------------------------------------------------
# Section: Network
# ---------------------------------------------------------------------------


# Single, time-boxed network probe. The whole point is to catch "user is
# behind a proxy / offline / DNS broken" early — not to audit reachability
# of every endpoint we ever talk to. A 2 s budget keeps the worst-case
# doctor runtime under the 5 s contract even when the resolver hangs.
_HF_PROBE_URL = "https://huggingface.co"
_HF_PROBE_TIMEOUT_S = 2.0


def _probe_hf(timeout: float = _HF_PROBE_TIMEOUT_S) -> tuple[CheckStatus, str]:
    """Return (status, detail) for the huggingface.co HEAD probe."""
    req = urllib.request.Request(_HF_PROBE_URL, method="HEAD")  # noqa: S310 — https only
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return (
                CheckStatus.OK,
                f"HEAD {_HF_PROBE_URL} → HTTP {resp.status}",
            )
    except urllib.error.HTTPError as e:
        # HF returns 405 to HEAD on some routes — still "reachable".
        if e.code in (200, 301, 302, 405):
            return CheckStatus.OK, f"HEAD {_HF_PROBE_URL} → HTTP {e.code}"
        return CheckStatus.WARN, f"HTTP {e.code} (rate-limited?)"
    except (urllib.error.URLError, TimeoutError) as e:
        # Spec rule: network timeout is a WARNING, never a FAIL — we don't
        # want CI runners in air-gapped environments to fail a doctor run
        # just because they can't talk to the public internet.
        return (
            CheckStatus.WARN,
            f"unreachable ({type(e).__name__}: {e})",
        )
    except OSError as e:
        return CheckStatus.WARN, f"OSError: {e}"


def section_network(
    *, probe: Callable[[], tuple[CheckStatus, str]] | None = None
) -> Section:
    """Network reachability probe.

    ``probe`` is injected by tests to avoid hitting the real internet.
    """
    s = Section("Network")
    fn = probe or _probe_hf
    status, detail = fn()
    if status is CheckStatus.OK:
        s.add("huggingface.co reachable", CheckStatus.OK, detail=detail)
    else:
        s.add(
            f"huggingface.co not reachable ({detail})",
            CheckStatus.WARN,
            detail=detail,
        )

    return s


# ---------------------------------------------------------------------------
# Section: Shell integration
# ---------------------------------------------------------------------------


_ARGCOMPLETE_HOOK_NEEDLE = "register-python-argcomplete rapid-mlx"

# Bound per-rc read so a 50 MB hand-edited zshrc, a named pipe, or a
# block device pointed-to via symlink can't make doctor hang or eat RAM.
# 256 KB is roughly 4000 lines of shell config, which is far above any
# real-world rc file's footprint. Codex review round 2 caught the
# previous unbounded ``read_text`` as a DoS / hang vector.
_RC_READ_LIMIT_BYTES = 256 * 1024


def _candidate_shell_rcs() -> list[Path]:
    """Return the rc files we look at for argcomplete activation."""
    home = Path.home()
    return [
        home / ".zshrc",
        home / ".bashrc",
        home / ".bash_profile",
        home / ".profile",
    ]


def _read_rc_prefix(rc: Path, limit: int = _RC_READ_LIMIT_BYTES) -> str | None:
    """Read up to ``limit`` bytes from ``rc``. Skips non-regular files
    (pipes / devices / symlinks-to-non-files) and decoding errors."""
    try:
        # ``stat`` follows symlinks, which is the right behavior for shell
        # rc files (people symlink their dotfiles all the time) — but we
        # refuse non-regular targets (S_IFREG missing).
        st = rc.stat()
    except OSError:
        return None
    import stat as _stat

    if not _stat.S_ISREG(st.st_mode):
        return None
    try:
        with rc.open("rb") as f:
            return f.read(limit).decode("utf-8", errors="replace")
    except OSError:
        return None


def _argcomplete_hook_present(
    rcs: list[Path] | None = None,
) -> tuple[bool, Path | None]:
    """Return (present, rc_file_with_hook). ``rcs`` is injected by tests."""
    rcs = rcs if rcs is not None else _candidate_shell_rcs()
    for rc in rcs:
        content = _read_rc_prefix(rc)
        if content and _ARGCOMPLETE_HOOK_NEEDLE in content:
            return True, rc
    return False, None


def _rapid_mlx_on_path(path_env: str | None = None) -> list[str]:
    """Every ``rapid-mlx`` executable on PATH, in PATH order, deduped by resolved
    target. More than one distinct target means competing installs — e.g. a
    ``curl | bash`` copy in ``~/.local/bin`` shadowing a Homebrew one — where the
    first on PATH silently wins and upgrades to the others do nothing.
    """
    raw = os.environ.get("PATH", "") if path_env is None else path_env
    seen: set[str] = set()
    out: list[str] = []
    for d in raw.split(os.pathsep):
        # An empty PATH component means the current directory on POSIX, which
        # is exactly how ``shutil.which()`` (the function that picks the
        # *active* rapid-mlx above) resolves it. Skipping it would let a
        # ``./rapid-mlx`` shadow slip past this very check — the kind of
        # competing install it exists to surface — so map "" → os.curdir
        # instead of dropping it.
        directory = d or os.curdir
        cand = os.path.join(directory, "rapid-mlx")
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            target = os.path.realpath(cand)
            if target not in seen:
                seen.add(target)
                out.append(cand)
    return out


def section_shell_integration(
    *,
    which: Callable[[str], str | None] | None = None,
    rcs: list[Path] | None = None,
    find_all: Callable[[], list[str]] | None = None,
) -> Section:
    """Verify the CLI is on PATH and argcomplete is wired up.

    ``which``, ``rcs`` and ``find_all`` are dependency-injected for tests.
    """
    s = Section("Shell Integration")
    which_fn = which or shutil.which

    cli_path = which_fn("rapid-mlx")
    if cli_path:
        s.add(
            f"rapid-mlx in $PATH ({cli_path})",
            CheckStatus.OK,
            detail=f"path={cli_path}",
        )
        installs = (find_all or _rapid_mlx_on_path)()
        if len(installs) > 1:
            active, shadowed = installs[0], installs[1:]
            s.add(
                f"rapid-mlx installed in {len(installs)} places — {active} wins on "
                f"$PATH, {', '.join(shadowed)} shadowed; remove the extra install(s)",
                CheckStatus.WARN,
                detail=f"active={active} shadowed={shadowed}",
            )
    else:
        s.add(
            "rapid-mlx NOT on $PATH",
            CheckStatus.FAIL,
            detail="shutil.which('rapid-mlx') returned None",
        )

    present, rc = _argcomplete_hook_present(rcs=rcs)
    if present:
        s.add(
            f"argcomplete activated in {rc.name if rc else 'shell rc'}",
            CheckStatus.OK,
            detail=f"hook found in {rc}",
        )
    else:
        s.add(
            "argcomplete not activated — add "
            '`eval "$(register-python-argcomplete rapid-mlx)"` to your shell rc',
            CheckStatus.WARN,
            detail="no shell rc contains the activation snippet",
        )

    return s


# ---------------------------------------------------------------------------
# Section: Optional Tools
# ---------------------------------------------------------------------------


def section_optional_tools(
    *, which: Callable[[str], str | None] | None = None
) -> Section:
    """Probe for development tools that improve the rapid-mlx experience but
    are never required to run inference. Missing → ✗ (issue) because the user
    explicitly opted into a workflow that needs them — phrasing makes it
    clear they're only relevant if you're using those harnesses."""
    s = Section("Optional Tools")
    which_fn = which or shutil.which

    codex = which_fn("codex")
    if codex:
        s.add(
            f"codex CLI ({codex})",
            CheckStatus.OK,
            detail="@openai/codex on PATH",
        )
    else:
        s.add(
            "codex CLI not installed (relevant if using codex agent harness)",
            CheckStatus.WARN,
            detail="npm install -g @openai/codex",
        )

    return s


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


# Sections fixed in spec order. Adding a new probe means appending to one
# of these lists, not adding a new section midway — keeps the user's mental
# model stable across rapid-mlx versions.
_SECTION_BUILDERS = (
    section_system,
    section_python,
    section_required_packages,
    section_updates,
    section_optional_packages,
    section_hf_cache,
    section_network,
    section_shell_integration,
    section_optional_tools,
)


def run_all() -> Report:
    """Run every section and return the aggregate report.

    Each section builder is wrapped in a try/except so a single buggy probe
    cannot abort the whole report. If a section crashes, it lands in the
    report as a single ✗ row labelled with the exception class — that's
    still a useful signal ("doctor is broken, file a bug").
    """
    report = Report()
    for builder in _SECTION_BUILDERS:
        try:
            report.sections.append(builder())
        except Exception as e:  # noqa: BLE001 — see docstring above
            crashed = Section(builder.__name__.replace("section_", "").title())
            crashed.add(
                f"probe crashed: {type(e).__name__}: {e}",
                CheckStatus.FAIL,
                detail=f"{type(e).__module__}.{type(e).__name__}: {e}",
            )
            report.sections.append(crashed)
    return report
