# SPDX-License-Identifier: Apache-2.0
"""macOS Unified Buffer Cache (UBC) eviction helper — Defect 4.

Background
----------
On macOS (Darwin), ``mmap(MAP_SHARED, PROT_READ)`` pages of a regular
file remain resident in the Unified Buffer Cache (UBC) after the caller
``munmap``'s and ``close``'s the file. That retention is by design — it
amortises read costs across processes — but it kills the GLM-5.2 boot
chain on a 256 GB M3 Ultra: ``mlx_lm.utils.load_model`` is in the
middle of materialising the same tensors into UMA, so the effective
memory pressure for the load window is ``(mmap mirror) + (materialised
weights) ~= 2x model size``. For GLM-5.2 (~200 GB) that ~400 GB burst
trips macOS Jetsam and the process is SIGTERM'd before the load
finishes.

The fix is to *actively* evict the file-backed pages once MLX has
materialised the tensors. On Darwin:

* ``madvise(MADV_DONTNEED)`` is **advisory** and does NOT release
  file-backed pages from UBC. Verified at runtime in the Step-1
  reproduction (see d4_design.md).

* ``msync(addr, len, MS_INVALIDATE)`` on a ``MAP_SHARED|PROT_READ``
  mapping is the documented Darwin path that flushes a UBC mirror.
  The XNU source chain is::

      bsd/kern/kern_mman.c:1073-1075     msync(2) entry — flags include MS_INVALIDATE
      osfmk/vm/vm_user.c:1129            mach_vm_msync_callback
      osfmk/vm/vm_map.c:21333-21391      vm_object_deactivate_pages(..., kill_pages=TRUE, ...)

  ``vm_object_deactivate_pages`` with ``kill_pages=TRUE`` removes the
  pages from the active queue and discards their UBC backing. For a
  PROT_READ-only mapping there is no dirty data to write back, so the
  call is purely an eviction — no I/O, no risk of corrupting another
  consumer's view of the file.

* ``posix_fadvise``: not implemented by Darwin libc — wrong tool.
* ``fcntl(F_NOCACHE)``: disables caching for *future* reads through that
  descriptor only; does not evict already-cached pages — wrong tool.

Safety
------
Calling ``msync(MS_INVALIDATE)`` on a read-only mapping while another
process (or MLX itself) holds a *separate* mapping of the same file is
safe. We open our own short-lived ``mmap``, issue ``msync``, and
``munmap``. The MLX path that materialised the tensor data into UMA has
already copied what it needs by the time we run — we never touch the
materialised ``mx.array`` storage. The kernel will simply re-page the
file from disk if another reader touches it later.

The helper:

* Is a no-op on non-Darwin platforms (returns 0).
* Never raises — every error path logs at WARNING/DEBUG and returns 0.
* Surfaces a process-monotonic ``ubc_evicted_bytes_total`` counter via
  :func:`snapshot` / :func:`render_prometheus_lines` (consumed by
  ``vllm_mlx.routes.metrics``), labelled ``path_kind="safetensors"``
  because the only current caller is the load path.

Integration
-----------
The intended caller is :mod:`vllm_mlx.utils.tokenizer` (the
``load_model_with_fallback`` wrapper) - after ``mlx_lm.load`` has
materialised every tensor into MLX (and run ``mx.eval`` implicitly via
``sanitize``), the source safetensors shards are pure UBC shadow and
can be evicted. See d4_design.md for the precise integration design.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import sys
import threading
import time
from collections.abc import Iterable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Darwin syscall constants
# ---------------------------------------------------------------------
# /usr/include/sys/mman.h on macOS 25.5.x:
#   #define PROT_READ      0x01
#   #define MAP_SHARED     0x0001
#   #define MS_INVALIDATE  0x0002
#
# These are stable across every macOS release Apple has shipped since
# Mac OS X 10.0 — they are inherited from 4.4 BSD and any change would
# break every existing binary on the system. Hard-coding them avoids a
# dependency on the ``mmap`` stdlib module (which exposes ``PROT_READ``
# but not ``MS_INVALIDATE``).
_PROT_READ: int = 0x01
_MAP_SHARED: int = 0x0001
_MS_INVALIDATE: int = 0x0002

# Sentinel returned by ``mmap(2)`` on failure — Darwin's mmap returns
# ``MAP_FAILED == (void *)-1`` rather than NULL. ``ctypes.c_void_p(-1)``
# canonicalises the bit pattern across 32/64-bit ABIs.
_MMAP_FAILED: int = ctypes.c_void_p(-1).value


# ---------------------------------------------------------------------
# libc resolution — lazy, locked, macOS-only
# ---------------------------------------------------------------------
_libc_lock = threading.Lock()
_libc: ctypes.CDLL | None = None


def _get_libc() -> ctypes.CDLL | None:
    """Return the cached libc handle with the three syscalls typed, or None.

    Returns None on non-Darwin platforms or when libc cannot be loaded
    (e.g. exotic OS X builds without ``libSystem.dylib`` on the standard
    path). Idempotent.
    """
    global _libc
    if sys.platform != "darwin":
        return None
    if _libc is not None:
        return _libc
    with _libc_lock:
        if _libc is not None:
            return _libc
        try:
            libname = ctypes.util.find_library("c") or "libSystem.dylib"
            lib = ctypes.CDLL(libname, use_errno=True)
            # Full argtype/restype on every syscall so a wrong-arity
            # call fails with a Python TypeError instead of a segfault.
            lib.mmap.argtypes = [
                ctypes.c_void_p,  # addr (NULL — let kernel pick)
                ctypes.c_size_t,  # length
                ctypes.c_int,  # prot
                ctypes.c_int,  # flags
                ctypes.c_int,  # fd
                ctypes.c_longlong,  # off_t (Darwin: 64-bit even on 32-bit ABIs)
            ]
            lib.mmap.restype = ctypes.c_void_p
            lib.msync.argtypes = [
                ctypes.c_void_p,  # addr
                ctypes.c_size_t,  # length
                ctypes.c_int,  # flags
            ]
            lib.msync.restype = ctypes.c_int
            lib.munmap.argtypes = [
                ctypes.c_void_p,  # addr
                ctypes.c_size_t,  # length
            ]
            lib.munmap.restype = ctypes.c_int
            _libc = lib
        except OSError as e:  # pragma: no cover — defensive
            logger.debug("ubc_evict: libc load failed: %s", e)
            _libc = None
    return _libc


# ---------------------------------------------------------------------
# Process-monotonic counter — exposed to /metrics via snapshot()
# ---------------------------------------------------------------------
_counter_lock = threading.Lock()
_ubc_evicted_bytes_total: int = 0
_ubc_evict_calls_total: int = 0
_ubc_evict_failed_total: int = 0


def _bump_counter(evicted: int, *, failed: bool) -> None:
    global _ubc_evicted_bytes_total, _ubc_evict_calls_total, _ubc_evict_failed_total
    with _counter_lock:
        _ubc_evict_calls_total += 1
        if failed:
            _ubc_evict_failed_total += 1
        elif evicted > 0:
            _ubc_evicted_bytes_total += int(evicted)


# ---------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------
def ubc_evict(path: str) -> int:
    """Evict the UBC mirror of ``path`` via ``msync(MS_INVALIDATE)``.

    Args:
        path: Filesystem path of the file whose UBC pages should be
            released. May point at any regular file; the typical caller
            passes a safetensors shard whose tensors have already been
            materialised into ``mx.array`` storage.

    Returns:
        Number of bytes the kernel evicted, on a best-effort basis.
        Returns 0 in every error path and on non-Darwin platforms — the
        counter exposed to /metrics ticks regardless so operators can
        still see ``calls_total`` move during boot. The "bytes" return
        is conservative: we report the file size when ``msync`` succeeds
        and 0 otherwise. The kernel does not give us a precise eviction
        count, but the file size is the worst case — the upper bound on
        what we asked it to discard.

    Errors are NEVER raised. Every exception path logs at DEBUG (for
    expected no-ops) or WARNING (for unexpected libc errors) and
    returns 0. The intent is operator visibility, not enforcement: a
    failure here should never prevent a model from loading.

    No-op on non-Darwin platforms. Linux/Windows model loads do not
    suffer the UBC retention bug (Linux ``mmap`` pages are released by
    the kernel's page-cache eviction policy under memory pressure
    without needing an explicit syscall).
    """
    if sys.platform != "darwin":
        logger.debug("ubc_evict no-op on %s", sys.platform)
        _bump_counter(0, failed=False)
        return 0

    libc = _get_libc()
    if libc is None:
        logger.debug("ubc_evict: libc unavailable, no-op")
        _bump_counter(0, failed=True)
        return 0

    # File-existence / size check FIRST so a missing or zero-byte file
    # never reaches the syscall (mmap of length 0 is EINVAL on Darwin).
    try:
        size = os.path.getsize(path)
    except (FileNotFoundError, NotADirectoryError, PermissionError) as e:
        logger.warning("ubc_evict: cannot stat %s: %s", path, e)
        _bump_counter(0, failed=True)
        return 0
    except OSError as e:
        logger.warning("ubc_evict: stat %s failed: %s", path, e)
        _bump_counter(0, failed=True)
        return 0
    if size <= 0:
        logger.debug("ubc_evict: %s is empty, no-op", path)
        _bump_counter(0, failed=False)
        return 0

    # Open / mmap / msync / munmap / close, with errno preserved on every
    # libc failure for the WARNING log. The fd is opened with O_RDONLY
    # so we cannot accidentally dirty the file — even if a future Darwin
    # version interpreted MS_INVALIDATE differently, the read-only fd
    # acts as a belt for the suspenders.
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError as e:
        logger.warning("ubc_evict: open %s failed: %s", path, e)
        _bump_counter(0, failed=True)
        return 0

    try:
        ctypes.set_errno(0)
        addr = libc.mmap(None, size, _PROT_READ, _MAP_SHARED, fd, 0)
        if addr is None or addr == 0 or addr == _MMAP_FAILED:
            err = ctypes.get_errno()
            logger.warning(
                "ubc_evict: mmap %s failed errno=%d (%s)",
                path,
                err,
                os.strerror(err) if err else "unknown",
            )
            _bump_counter(0, failed=True)
            return 0
        try:
            ctypes.set_errno(0)
            rc = libc.msync(addr, size, _MS_INVALIDATE)
            if rc != 0:
                err = ctypes.get_errno()
                logger.warning(
                    "ubc_evict: msync(MS_INVALIDATE) %s rc=%d errno=%d (%s)",
                    path,
                    rc,
                    err,
                    os.strerror(err) if err else "unknown",
                )
                _bump_counter(0, failed=True)
                return 0
        finally:
            # Always release the mapping even if msync failed. pr_validate
            # codex NIT #2: check the return code so a cleanup leak is at
            # least visible in the logs (the helper handles large mappings,
            # so a silent munmap failure would be invisible RSS pressure).
            ctypes.set_errno(0)
            munmap_rc = libc.munmap(addr, size)
            if munmap_rc != 0:
                err = ctypes.get_errno()
                logger.warning(
                    "ubc_evict: munmap %s rc=%d errno=%d (%s)",
                    path,
                    munmap_rc,
                    err,
                    os.strerror(err) if err else "unknown",
                )
    finally:
        try:
            os.close(fd)
        except OSError:  # pragma: no cover — defensive
            pass

    _bump_counter(size, failed=False)
    return size


def ubc_evict_paths(paths: Iterable[str]) -> int:
    """Evict each path; aggregate the bytes evicted; never raise.

    Helper for the load-path integration: takes an iterable of safetensors
    shard paths and returns the total bytes evicted across all of them.
    Logs one INFO line per shard summarising the eviction so operators
    can correlate boot-time RSS drops with specific files.
    """
    if sys.platform != "darwin":
        logger.debug("ubc_evict_paths no-op on %s", sys.platform)
        return 0
    total = 0
    t0 = time.monotonic()
    for p in paths:
        bytes_evicted = ubc_evict(str(p))
        if bytes_evicted > 0:
            logger.info(
                "ubc_evict: evicted %.1f MB from UBC for %s",
                bytes_evicted / (1024 * 1024),
                p,
            )
        total += bytes_evicted
    if total > 0:
        logger.info(
            "ubc_evict: pass complete total_mb=%.1f elapsed_s=%.2f",
            total / (1024 * 1024),
            time.monotonic() - t0,
        )
    return total


# ---------------------------------------------------------------------
# Prometheus surface — consumed by routes/metrics.py
# ---------------------------------------------------------------------
def snapshot() -> dict[str, int]:
    """Return a thread-safe snapshot of the UBC counters."""
    with _counter_lock:
        return {
            "ubc_evicted_bytes_total": _ubc_evicted_bytes_total,
            "ubc_evict_calls_total": _ubc_evict_calls_total,
            "ubc_evict_failed_total": _ubc_evict_failed_total,
        }


_UBC_EVICTED_HELP = (
    "Cumulative bytes that the macOS Unified Buffer Cache eviction "
    "helper (Defect 4) has asked the kernel to discard via "
    "msync(MS_INVALIDATE). Reported per shard file evicted by the "
    "load path. Non-zero only on Darwin; Linux / Windows builds keep "
    "the series flat at 0 since they don't suffer the UBC retention "
    "cliff that motivated this counter."
)


def render_prometheus_lines() -> list[str]:
    """Render the UBC counter as Prometheus text-exposition lines.

    Single counter, single label (``path_kind="safetensors"``). The
    label is fixed today because the only caller is the safetensors
    load path; the label slot is reserved so a future caller (e.g.
    tokenizer.json eviction, sidecar weights) can land without
    breaking the existing series name.
    """
    stats = snapshot()
    evicted = int(stats.get("ubc_evicted_bytes_total", 0))
    return [
        f"# HELP rapid_mlx_ubc_evicted_bytes_total {_UBC_EVICTED_HELP}",
        "# TYPE rapid_mlx_ubc_evicted_bytes_total counter",
        f'rapid_mlx_ubc_evicted_bytes_total{{path_kind="safetensors"}} {evicted}',
    ]


def reset_for_tests() -> None:
    """Test-only hook: zero the counters between cases.

    Production code MUST NOT call this — Prometheus counters are
    contractually monotonic for the process lifetime.
    """
    global _ubc_evicted_bytes_total, _ubc_evict_calls_total, _ubc_evict_failed_total
    with _counter_lock:
        _ubc_evicted_bytes_total = 0
        _ubc_evict_calls_total = 0
        _ubc_evict_failed_total = 0


__all__ = [
    "ubc_evict",
    "ubc_evict_paths",
    "snapshot",
    "render_prometheus_lines",
    "reset_for_tests",
]
