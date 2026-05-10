# SPDX-License-Identifier: Apache-2.0
"""Redaction primitives — every "could this leak PII?" decision lives here.

Phase 1 ships these as pure functions with thorough unit tests. Phase 2
event sites call them; the contract is "if it didn't go through redact,
it doesn't leave the machine".

Bucketing rationale: exact counts (token totals, TTFT in ms) are a soft
fingerprint when joined with other fields. Bucketing into a small fixed
set of strings collapses the join surface to something an aggregation
pipeline can actually count without re-identifying a session.
"""

from __future__ import annotations

import hashlib
import platform
import re
import traceback
from pathlib import Path

# ---------------------------------------------------------------- bucketing


_TOKEN_BUCKETS: tuple[tuple[int, str], ...] = (
    (256, "0-256"),
    (1024, "256-1k"),
    (4096, "1k-4k"),
    (16384, "4k-16k"),
    (65536, "16k-64k"),
)
_TOKEN_OVERFLOW = "64k+"


def bucket_tokens(n: int) -> str:
    """Map a token count to one of 6 fixed buckets.

    Edges go to the *upper* bucket — exactly 256 tokens lands in
    ``"256-1k"``, not ``"0-256"`` — because the buckets read as
    half-open intervals ``[lower, upper)``.
    """
    if n < 0:
        return "0-256"
    for upper, label in _TOKEN_BUCKETS:
        if n < upper:
            return label
    return _TOKEN_OVERFLOW


_TTFT_BUCKETS: tuple[tuple[float, str], ...] = (
    (100, "<100ms"),
    (500, "100-500ms"),
    (1500, "500-1500ms"),
    (5000, "1.5-5s"),
)
_TTFT_OVERFLOW = ">5s"


def bucket_ttft_ms(ms: float) -> str:
    if ms < 0:
        return "<100ms"
    for upper, label in _TTFT_BUCKETS:
        if ms < upper:
            return label
    return _TTFT_OVERFLOW


_TPS_BUCKETS: tuple[tuple[float, str], ...] = (
    (10, "<10"),
    (30, "10-30"),
    (50, "30-50"),
    (100, "50-100"),
)
_TPS_OVERFLOW = ">100"


def bucket_tps(tps: float) -> str:
    if tps < 0:
        return "<10"
    for upper, label in _TPS_BUCKETS:
        if tps < upper:
            return label
    return _TPS_OVERFLOW


def bucket_memory_gb(bytes_: int) -> int:
    """Round a byte count to the nearest GB. Negatives clamp to 0."""
    if bytes_ <= 0:
        return 0
    return round(bytes_ / (1024**3))


# ---------------------------------------------------------------- model paths

# A HuggingFace repo ID is roughly ``org/name`` with letters, digits,
# dot, dash, and underscore. We deliberately keep this strict — anything
# else gets redacted, even at the cost of losing the model identity for
# users who downloaded directly via git clone or symlinked HF cache.
_HF_REPO_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


def normalize_model_path(path: str) -> str:
    """Pass through ``org/name`` repo IDs; redact local paths to ``"<local>"``.

    A local ``./qwen3.5-9b`` checkout that resolve_model() prefers over
    the alias would otherwise leak the user's home-directory layout via
    the model name.
    """
    if not path:
        return "<empty>"
    # Local-path heuristics: anything that looks like a filesystem path.
    if path.startswith(("/", "./", "../", "~/")) or "\\" in path:
        return "<local>"
    if path.startswith("file://"):
        return "<local>"
    # Any ``/`` that survives must be the org/name separator and the
    # whole string must match the repo-ID pattern.
    if "/" in path:
        if _HF_REPO_RE.match(path):
            return path
        return "<local>"
    # Bare alias names (``qwen3.5-9b``) are public + harmless.
    return path


# ---------------------------------------------------------------- argv flags

# Captures ``--flag``, ``--flag-name``, and short ``-x`` (single letter
# only — multi-char short opts like ``-xy`` are uncommon enough we don't
# bother). Dashes inside the name are kept; everything after ``=`` is
# dropped before we ever see it because we only return names.
_LONG_FLAG_RE = re.compile(r"^--([A-Za-z][A-Za-z0-9-]*)(?:=.*)?$")
_SHORT_FLAG_RE = re.compile(r"^-([A-Za-z])$")


def hash_flag_names(argv: list[str]) -> list[str]:
    """Extract flag names from argv. Values are NEVER returned.

    Only the name of each flag survives. ``--api-key sk-xxx`` becomes
    ``["api-key"]`` (note: ``sk-xxx`` is never even read by this
    function — we simply skip non-flag tokens). Returns sorted unique
    list so the output is order-independent.
    """
    names: set[str] = set()
    for token in argv:
        if not isinstance(token, str):
            continue
        m = _LONG_FLAG_RE.match(token)
        if m:
            names.add(m.group(1))
            continue
        m = _SHORT_FLAG_RE.match(token)
        if m:
            names.add(m.group(1))
    return sorted(names)


# ---------------------------------------------------------------- traceback


def fingerprint_traceback(exc: BaseException) -> str:
    """Hash a traceback's *module paths only* — never message text.

    We use the qualified frame paths (filename + function name + line
    number) joined with the exception's class name. Crucially, we do NOT
    include ``str(exc)`` — exception messages routinely contain user
    input ("could not open /Users/alice/secret.txt").

    Returns a 16-hex-char prefix of sha256 — enough to distinguish ~10^9
    distinct sites with negligible collision risk, short enough to
    eyeball in a dashboard.
    """
    tb = traceback.TracebackException.from_exception(exc)
    parts: list[str] = [exc.__class__.__module__ + ":" + exc.__class__.__name__]
    for frame in tb.stack:
        # frame.filename is an absolute path — strip the directory part
        # to avoid leaking the user's home. We keep the basename + the
        # function name + line number, which is enough to identify the
        # site without revealing where rapid-mlx is installed.
        basename = Path(frame.filename).name
        parts.append(f"{basename}:{frame.name}:{frame.lineno}")
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


# ---------------------------------------------------------------- platform


def _read_chip_brand() -> str:
    """Best-effort Apple Silicon chip name (e.g. ``"Apple M3 Ultra"``).

    Falls back to ``platform.processor()`` on non-Darwin or when sysctl
    is unavailable. Never raises.
    """
    if platform.system() != "Darwin":
        return platform.processor() or "unknown"
    try:
        import subprocess

        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
        brand = result.stdout.strip()
        return brand or platform.processor() or "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return platform.processor() or "unknown"


def _read_total_memory_bytes() -> int:
    """Best-effort total RAM in bytes. Returns 0 if unknown."""
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        return 0


def platform_info() -> dict:
    """Coarse platform fingerprint — schema-stable, no full kernel string.

    Notes:
    - ``os_version`` is major.minor only (Darwin 25.3.0 → "25.3"). The
      patch number changes weekly and is a soft fingerprint.
    - ``python_version`` is also major.minor only.
    - ``memory_gb`` is rounded — exact byte counts can identify
      individual machines.
    """
    os_release = platform.release() or ""
    os_version = ".".join(os_release.split(".")[:2]) or os_release
    py_version = "{}.{}".format(*platform.python_version_tuple()[:2])
    return {
        "os": platform.system().lower(),
        "os_version": os_version,
        "arch": platform.machine(),
        "chip": _read_chip_brand(),
        "memory_gb": bucket_memory_gb(_read_total_memory_bytes()),
        "python_version": py_version,
    }
