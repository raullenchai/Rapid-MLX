# SPDX-License-Identifier: Apache-2.0
"""Apple Silicon hardware fingerprint for community benchmark submissions.

**Privacy contract** (enforced by the explicit allowlist at module top
and the type returned by ``collect()``):

- Only the listed ``/usr/bin`` tools are invoked. No new process is
  spawned that isn't in ``_PERMITTED_BINARIES``.
- Each probe reads only the specific field it claims to read. Tools
  like ``system_profiler`` would happily emit the user's name and
  hostname if queried with ``SPSoftwareDataType``; we never do.
- No environment variables, file paths, or unrelated sysctls. No
  network. No privileged operations — every probe runs as the
  invoking user with zero entitlement.

If a probe fails (e.g. ``system_profiler`` times out on a slow disk),
the corresponding field is set to ``None`` and the submission proceeds
without it. We never block the user on an optional field.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass

# The COMPLETE list of external programs this module will invoke. Any
# expansion goes through code review precisely because new programs
# expand the privacy surface.
_PERMITTED_BINARIES: frozenset[str] = frozenset(
    {
        "/usr/sbin/sysctl",
        "/usr/bin/sw_vers",
        "/usr/sbin/system_profiler",
    }
)

# Per-binary timeouts. ``system_profiler`` is slow on first call (3-10s
# cold), the others are <50 ms.
_SYSCTL_TIMEOUT_S: float = 2.0
_SWVERS_TIMEOUT_S: float = 2.0
_SYSTEM_PROFILER_TIMEOUT_S: float = 15.0


@dataclass(frozen=True)
class Hardware:
    """The subset of hardware info shipped in a submission.

    Fields map 1:1 onto ``schema.json#/properties/hardware``. Keep this
    dataclass narrow — every new field expands what we collect from
    user machines.
    """

    chip: str
    ram_gb: int
    cpu_cores: int
    gpu_cores: int | None  # may be None if system_profiler probe failed


@dataclass(frozen=True)
class Software:
    """``schema.json#/properties/software`` mirror."""

    macos: str
    rapid_mlx: str
    mlx: str
    python: str


def _run(cmd: list[str], timeout: float) -> str:
    """Run an allowlisted command and return stripped stdout.

    Raises ``RuntimeError`` if the binary isn't on the allowlist (so a
    future contributor can't quietly add ``ioreg`` etc.), or if the
    call fails / times out.
    """
    if cmd[0] not in _PERMITTED_BINARIES:
        raise RuntimeError(
            f"hardware probe attempted disallowed binary: {cmd[0]!r}. "
            f"Add to _PERMITTED_BINARIES with review."
        )
    try:
        result = subprocess.run(  # noqa: S603 — input is the allowlist itself
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"probe {cmd!r} failed: {e}") from e
    return result.stdout.strip()


def _chip() -> str:
    """`sysctl -n machdep.cpu.brand_string` → 'Apple M4 Pro'."""
    return _run(
        ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
        _SYSCTL_TIMEOUT_S,
    )


def _ram_gb() -> int:
    """`sysctl -n hw.memsize` → bytes → round to GB."""
    raw = _run(
        ["/usr/sbin/sysctl", "-n", "hw.memsize"],
        _SYSCTL_TIMEOUT_S,
    )
    bytes_ = int(raw)
    # Use 1<<30 (GiB) — that's what Apple's product pages mean by "GB".
    return round(bytes_ / (1 << 30))


def _cpu_cores() -> int:
    """`sysctl -n hw.ncpu` → integer count."""
    return int(_run(["/usr/sbin/sysctl", "-n", "hw.ncpu"], _SYSCTL_TIMEOUT_S))


def _gpu_cores() -> int | None:
    """`system_profiler SPDisplaysDataType` → 'Total Number of Cores: N'.

    Returns ``None`` if the probe times out or the line is absent.
    Never raises — GPU cores is a nice-to-have, not a blocker. The
    submission still ships without it (schema allows null).
    """
    try:
        out = _run(
            ["/usr/sbin/system_profiler", "SPDisplaysDataType"],
            _SYSTEM_PROFILER_TIMEOUT_S,
        )
    except RuntimeError:
        return None
    # Lines look like: "      Total Number of Cores: 20"
    m = re.search(r"Total Number of Cores:\s*(\d+)", out)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _macos_version() -> str:
    """`sw_vers -productVersion` → '26.5.1'."""
    return _run(
        ["/usr/bin/sw_vers", "-productVersion"],
        _SWVERS_TIMEOUT_S,
    )


def _rapid_mlx_version() -> str:
    """`vllm_mlx.__version__` — imported lazily so the hardware
    module can be unit-tested without the full engine."""
    try:
        from vllm_mlx import __version__

        return str(__version__)
    except ImportError:
        return "unknown"


def _mlx_version() -> str:
    """MLX version.

    ``mlx`` itself is a namespace package whose ``__version__`` lives
    under ``mlx.core``, not the top-level module. Use that first; fall
    back to ``importlib.metadata`` for the ``mlx`` distribution
    (covers the case where ``mlx.core`` couldn't import — e.g. on a
    non-Apple-Silicon dev box mocking the probe).
    """
    try:
        import mlx.core

        v = getattr(mlx.core, "__version__", None)
        if v:
            return str(v)
    except ImportError:
        pass
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("mlx")
    except (ImportError, PackageNotFoundError):
        return "unknown"


def _python_version() -> str:
    """First three components of sys.version_info as 'X.Y.Z'."""
    v = sys.version_info
    return f"{v.major}.{v.minor}.{v.micro}"


def is_apple_silicon() -> bool:
    """True iff we're running on Apple Silicon (arm64 Darwin).

    The submission flow refuses to run on anything else — a submission
    from a non-Apple-Silicon machine has no place in this database.
    """
    return sys.platform == "darwin" and platform.machine() == "arm64"


def collect() -> tuple[Hardware, Software]:
    """Collect the full whitelisted hardware + software fingerprint.

    Each probe runs sequentially because they're all fast except
    ``system_profiler``; parallelism would complicate error attribution
    without a real wall-clock win at this scale.

    Raises ``RuntimeError`` if any *required* field (chip, ram_gb,
    cpu_cores, macos) fails to probe — those are the bucketing keys
    and the submission is meaningless without them. Optional fields
    (gpu_cores) silently fall back to ``None``.
    """
    if not is_apple_silicon():
        raise RuntimeError(
            "community benchmark submissions are Apple-Silicon-only "
            f"(detected platform={sys.platform!r} machine={platform.machine()!r})"
        )
    # Pre-flight: every required binary must exist before we start
    # measuring. Fail fast with a readable message rather than mid-way
    # through a 60-second bench.
    for bin_path in ("/usr/sbin/sysctl", "/usr/bin/sw_vers"):
        if not shutil.which(bin_path):
            raise RuntimeError(
                f"required probe binary not found: {bin_path}. "
                "Is this really a macOS install?"
            )

    hardware = Hardware(
        chip=_chip(),
        ram_gb=_ram_gb(),
        cpu_cores=_cpu_cores(),
        gpu_cores=_gpu_cores(),
    )
    software = Software(
        macos=_macos_version(),
        rapid_mlx=_rapid_mlx_version(),
        mlx=_mlx_version(),
        python=_python_version(),
    )
    return hardware, software


def as_dicts(hw: Hardware, sw: Software) -> tuple[dict, dict]:
    """Serialize to plain dicts for JSON encoding."""
    return asdict(hw), asdict(sw)
