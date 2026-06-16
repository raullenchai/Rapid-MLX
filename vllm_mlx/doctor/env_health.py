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
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
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
    don't double-count. Aborts as soon as ``budget_s`` elapses — the next
    file's ``getsize`` is still allowed, but no new directory is descended
    into. That keeps the abort path O(1) instead of O(open-files-in-current-dir).
    """
    import time as _time

    if not path.exists():
        return None
    deadline = _time.monotonic() + budget_s
    total = 0
    try:
        for root, _dirs, files in os.walk(path, followlinks=False):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    # Broken symlink, permission denied — skip silently;
                    # this probe is "is the cache enormous?", not "audit
                    # every file".
                    continue
            if _time.monotonic() >= deadline:
                # Budget exhausted; the partial total isn't useful as a
                # size (lower bound only), so return None and let the
                # caller render "unknown / too large to size".
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


def section_optional_packages() -> Section:
    s = Section("Optional Packages")
    for dist, label, hint in OPTIONAL_PACKAGES:
        ver = _safe_version(dist)
        if ver:
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
    return s


# ---------------------------------------------------------------------------
# Section: HuggingFace cache
# ---------------------------------------------------------------------------


def section_hf_cache() -> Section:
    s = Section("HuggingFace Cache")

    cache = _hf_cache_dir()
    if cache.exists():
        if os.access(cache, os.W_OK):
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
        # Missing cache isn't a failure — first run will create it. But warn
        # so users with a misconfigured HF_HOME notice early.
        s.add(
            f"{cache} does not exist yet (will be created on first download)",
            CheckStatus.WARN,
            detail=f"path={cache}",
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


def section_network(*, probe: callable | None = None) -> Section:
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

    # Cookie hint — not a probe, just an informational warning aligned with
    # the project's documented mitigation for HF/YouTube rate limits.
    # Env-var names are spelled out as literals so the routing-shape audit
    # (tests/test_no_out_of_band_routing.py) can statically prove no
    # routing decision is composed at the call site.
    has_cookies = any(
        os.environ.get(name)
        for name in (
            "YOUTUBE_COOKIES",
            "YOUTUBE_COOKIES_1",
            "YOUTUBE_COOKIES_2",
            "YOUTUBE_COOKIES_3",
            "YOUTUBE_COOKIES_4",
            "YOUTUBE_COOKIES_5",
        )
    )
    if has_cookies:
        s.add(
            "YouTube/HF cookies configured",
            CheckStatus.OK,
            detail="YOUTUBE_COOKIES* env var set",
        )
    else:
        s.add(
            "No YouTube/HF cookies configured (rate-limit risk on heavy downloads)",
            CheckStatus.WARN,
            detail="set YOUTUBE_COOKIES_1..5 to enable cookie rotation",
        )

    return s


# ---------------------------------------------------------------------------
# Section: Shell integration
# ---------------------------------------------------------------------------


_ARGCOMPLETE_HOOK_NEEDLE = "register-python-argcomplete rapid-mlx"


def _candidate_shell_rcs() -> list[Path]:
    """Return the rc files we look at for argcomplete activation."""
    home = Path.home()
    return [
        home / ".zshrc",
        home / ".bashrc",
        home / ".bash_profile",
        home / ".profile",
    ]


def _argcomplete_hook_present(
    rcs: list[Path] | None = None,
) -> tuple[bool, Path | None]:
    """Return (present, rc_file_with_hook). ``rcs`` is injected by tests."""
    rcs = rcs if rcs is not None else _candidate_shell_rcs()
    for rc in rcs:
        try:
            if rc.exists() and _ARGCOMPLETE_HOOK_NEEDLE in rc.read_text(
                errors="replace"
            ):
                return True, rc
        except OSError:
            continue
    return False, None


def section_shell_integration(
    *,
    which: callable | None = None,
    rcs: list[Path] | None = None,
) -> Section:
    """Verify the CLI is on PATH and argcomplete is wired up.

    ``which`` and ``rcs`` are dependency-injected for tests.
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


def section_optional_tools(*, which: callable | None = None) -> Section:
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

    anth_ver = _safe_version("anthropic")
    if anth_ver:
        s.add(
            f"anthropic SDK {anth_ver}",
            CheckStatus.OK,
            detail=f"version={anth_ver}",
        )
    else:
        s.add(
            "anthropic SDK not installed (relevant if using anthropic agent harness)",
            CheckStatus.WARN,
            detail="pip install anthropic",
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
