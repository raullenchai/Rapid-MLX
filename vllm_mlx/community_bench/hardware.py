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
from dataclasses import dataclass

# The COMPLETE list of external programs this module will invoke. Any
# expansion goes through code review precisely because new programs
# expand the privacy surface.
_PERMITTED_BINARIES: frozenset[str] = frozenset(
    {
        "/usr/sbin/sysctl",
        "/usr/bin/sw_vers",
        "/usr/sbin/system_profiler",
        # ``pmset -g batt`` for the volatile power-source condition (AC vs
        # battery). Read-only, unprivileged.
        "/usr/bin/pmset",
    }
)

# Per-binary timeouts. ``system_profiler`` is slow on first call (3-10s
# cold), the others are <50 ms.
_SYSCTL_TIMEOUT_S: float = 2.0
_SWVERS_TIMEOUT_S: float = 2.0
_SYSTEM_PROFILER_TIMEOUT_S: float = 15.0
_PMSET_TIMEOUT_S: float = 2.0


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
    # Empty argv would crash ``cmd[0]`` with ``IndexError`` rather
    # than the documented ``RuntimeError`` from the allowlist guard.
    # Tightening the precondition turns a programmer error into the
    # same explicit failure mode as a non-allowlisted binary. (Codex
    # PR #582 round-7 BLOCKING.)
    if not cmd or cmd[0] not in _PERMITTED_BINARIES:
        raise RuntimeError(
            f"hardware probe attempted disallowed binary: {cmd[0] if cmd else '<empty>'!r}. "
            f"Add to _PERMITTED_BINARIES with review."
        )
    try:
        result = subprocess.run(  # noqa: S603 — input is the allowlist itself
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
            # Defensive: explicit ``shell=False`` matches the privacy
            # contract's "no shell interpretation". The default is
            # already False when args is a list, but pinning makes
            # the invariant resilient to a future refactor that
            # accidentally accepts a string. (Codex PR #582 round-7
            # NIT.)
            shell=False,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        # ``OSError`` covers process-creation failures (EAGAIN, EMFILE, a
        # missing executable). Every caller treats ``RuntimeError`` as
        # "this probe is unavailable"; letting ``OSError`` escape would turn
        # an optional post-measurement probe into a lost benchmark result.
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


def host_memory_gib() -> int | None:
    """Best-effort unified-memory size of this Mac for planning output.

    Returns ``None`` when the sysctl probe is unavailable (non-macOS, sandbox)
    so callers can degrade to "unknown" instead of failing a read-only command.
    """
    try:
        return _ram_gb()
    except (RuntimeError, ValueError, OSError):
        return None


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


# ---------------------------------------------------------------------------
# Volatile run conditions (``machine-observation.schema.json#/$defs/runConditions``)
# ---------------------------------------------------------------------------
#
# The atomic ``MachineObservation`` separates the stable hardware profile from
# the conditions a run happened under. Until 0.13.4 every producer wrote
# ``unknown``/``null`` here, which made "was this Mac on battery / throttled /
# swapping?" unanswerable on the board. Each probe below is best-effort and
# independent: a failure degrades exactly one field to ``unknown``/``None``
# and never blocks the benchmark.

#: ``NSProcessInfoThermalState`` raw values → schema enum.
_THERMAL_STATES: dict[int, str] = {
    0: "nominal",
    1: "fair",
    2: "serious",
    3: "critical",
}

#: ``kern.memorystatus_vm_pressure_level`` raw values → schema enum. The
#: kernel reports 1/2/4 (normal/warning/critical); anything else is unknown.
_MEMORY_PRESSURE_LEVELS: dict[int, str] = {
    1: "normal",
    2: "warning",
    4: "critical",
}

_MIB = 1024 * 1024


def _power_source() -> str:
    """``ac`` / ``battery`` from the first line of ``pmset -g batt``."""
    try:
        first = _run(["/usr/bin/pmset", "-g", "batt"], _PMSET_TIMEOUT_S).splitlines()[0]
    except (RuntimeError, IndexError):
        return "unknown"
    lowered = first.lower()
    if "ac power" in lowered:
        return "ac"
    if "battery power" in lowered:
        return "battery"
    return "unknown"


def _process_info(selector: bytes, restype):
    """Send one no-argument message to ``NSProcessInfo.processInfo``.

    Read in-process through the Objective-C runtime — no subprocess, no
    entitlement, nothing beyond one scalar leaves the call. Returns ``None``
    on any failure (non-Darwin, missing runtime, unexpected value) so each
    caller degrades its own field.
    """
    if sys.platform != "darwin":
        return None
    try:
        import ctypes

        # NSProcessInfo lives in Foundation; make sure the framework is
        # mapped into a clean process before asking the runtime for the class.
        ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
        objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        send_id = ctypes.cast(
            objc.objc_msgSend,
            ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p),
        )
        send = ctypes.cast(
            objc.objc_msgSend,
            ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p),
        )
        process_info = send_id(
            objc.objc_getClass(b"NSProcessInfo"),
            objc.sel_registerName(b"processInfo"),
        )
        if not process_info:
            return None
        # Older macOS releases lack some selectors (``isLowPowerModeEnabled``
        # arrived in 12.0); an unrecognised selector would raise an
        # Objective-C exception that ctypes cannot catch, so ask first.
        responds = ctypes.cast(
            objc.objc_msgSend,
            ctypes.CFUNCTYPE(
                ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
            ),
        )
        target = objc.sel_registerName(selector)
        if not responds(
            process_info, objc.sel_registerName(b"respondsToSelector:"), target
        ):
            return None
        return send(process_info, target)
    except (OSError, AttributeError, ValueError):
        return None


def _low_power_mode() -> bool | None:
    """``NSProcessInfo.isLowPowerModeEnabled`` — the live setting.

    ``pmset -g`` was tried first and rejected: its output differs between
    hosts (some print the ``lowpowermode`` row, some only the header), so it
    silently reported ``null`` on machines where Low Power Mode was on.
    """
    import ctypes

    value = _process_info(b"isLowPowerModeEnabled", ctypes.c_bool)
    return None if value is None else bool(value)


def _thermal_state() -> str:
    """``NSProcessInfo.thermalState`` — the signal macOS and the Desktop app use."""
    import ctypes

    value = _process_info(b"thermalState", ctypes.c_long)
    if value is None:
        return "unknown"
    return _THERMAL_STATES.get(int(value), "unknown")


def _memory_pressure() -> str:
    try:
        raw = int(
            _run(
                ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                _SYSCTL_TIMEOUT_S,
            )
        )
    except (RuntimeError, ValueError):
        return "unknown"
    return _MEMORY_PRESSURE_LEVELS.get(raw, "unknown")


def _available_memory_mib() -> int | None:
    """Free + speculative + purgeable pages, in MiB.

    A conservative reading of what a model load could claim without paging
    anything out; the numbers come from the same ``vm.*`` counters
    ``vm_stat`` prints, read through the already-allowlisted ``sysctl``.
    """
    try:
        out = _run(
            [
                "/usr/sbin/sysctl",
                "-n",
                "vm.page_free_count",
                "vm.page_speculative_count",
                "vm.page_purgeable_count",
                "hw.pagesize",
            ],
            _SYSCTL_TIMEOUT_S,
        )
        free, speculative, purgeable, page_size = (int(v) for v in out.split())
    except (RuntimeError, ValueError):
        return None
    return (free + speculative + purgeable) * page_size // _MIB


def run_conditions() -> dict[str, bool | int | str | None]:
    """Snapshot the volatile conditions a run happens under.

    Returns a schema-valid ``runConditions`` object. Take one snapshot before
    loading the model and one after the last measurement so readers can see
    whether the machine was on battery, throttled, or under memory pressure
    while the numbers were produced.
    """
    return {
        "power_source": _power_source(),
        "low_power_mode": _low_power_mode(),
        "thermal_state": _thermal_state(),
        "memory_pressure": _memory_pressure(),
        "available_memory_mib": _available_memory_mib(),
    }
