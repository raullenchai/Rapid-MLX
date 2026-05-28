# SPDX-License-Identifier: Apache-2.0
"""Auto-pull confirmation gate for large model downloads.

Persona-3 ("Ollama switcher") feedback (2026-05): running

    rapid-mlx chat qwen3-coder

against an alias that wasn't yet cached silently kicked off a 41.8 GB
download with no ``[y/N]`` prompt. The download itself ran fine, but
because the spawned ``serve`` subprocess captured stdout to a logfile,
the user saw a blank screen and assumed the CLI was hung.

This module is the user-visible safety net. It is intentionally
self-contained (no rapid-mlx imports) so it stays cheap to import from
``cli.main()`` on every invocation.

Public API:

* :func:`estimate_repo_size_bytes` — best-effort HF API size lookup.
* :func:`confirm_or_abort`         — prompt + abort path used by the CLI.
* :func:`is_repo_cached`           — cache-presence probe (so callers can
  short-circuit without re-implementing the path dance).

Design choices:

* The HF call is wrapped in a hard 5-second timeout and a blanket
  ``except Exception`` — a flaky metadata query must never block a
  perfectly-good cached load, and we already gate on cache presence
  upstream of the size estimate.
* The threshold defaults to 10 GiB. Anything under that is too small
  to warrant interrupting the user's flow.
* The env override (``RAPID_MLX_AUTO_PULL=1``) is the documented escape
  hatch for non-interactive CI usage and ``--yes``-style workflows.
* Non-TTY stdin → auto-confirm. Scripts that pipe input into
  ``rapid-mlx`` must not deadlock on a missing terminal.
"""

from __future__ import annotations

import os
import sys
import threading

# File suffixes that contribute to "model weight + tokenizer" footprint.
# Anything outside this set (e.g. ``.gitattributes``, ``README.md``) is a
# rounding error and is excluded so the prompt size matches what the user
# actually waits on.
_WEIGHT_SUFFIXES: tuple[str, ...] = (
    ".safetensors",
    ".bin",
    ".gguf",
    ".json",
    ".txt",
    ".model",
    ".tiktoken",
)

# 5-second cap on the HF metadata call. Anything slower than this is a
# signal we should fall through silently rather than block startup.
_HF_API_TIMEOUT_SECONDS: float = 5.0


def _format_size(num_bytes: int) -> str:
    """Render ``num_bytes`` as a human-friendly string (e.g. ``42.3 GB``).

    Uses base-1024 units to match the way HF Hub and macOS Finder
    report file sizes. Picks the largest unit where the value is ``>= 1``.
    """
    if num_bytes < 0:
        num_bytes = 0
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:.1f} TB"  # unreachable, keeps mypy happy


def _is_weight_file(name: str) -> bool:
    """True if ``name`` should be counted in the download-size estimate."""
    if not name or name.startswith(".git"):
        return False
    lower = name.lower()
    return any(lower.endswith(s) for s in _WEIGHT_SUFFIXES)


def _sibling_size(sibling) -> int:
    """Extract the on-disk size of an HF ``RepoSibling``.

    Newer huggingface_hub releases expose the LFS pointer's true size
    under ``sibling.lfs.size``; older ones store it directly on
    ``sibling.size``. Try both, prefer the LFS value when both are
    populated (the regular ``size`` may report the pointer-file size,
    not the resolved blob size).
    """
    lfs = getattr(sibling, "lfs", None)
    if lfs is not None:
        lfs_size = getattr(lfs, "size", None)
        if isinstance(lfs_size, int) and lfs_size > 0:
            return lfs_size
    raw = getattr(sibling, "size", None)
    if isinstance(raw, int) and raw > 0:
        return raw
    return 0


def _model_info_with_timeout(repo_id: str, timeout: float):
    """Call ``HfApi().model_info`` with a hard timeout.

    huggingface_hub itself doesn't accept a timeout argument on
    ``model_info`` in every release we support, so we run it in a worker
    thread and join with a deadline. Worst case (network hang) the
    daemon thread is leaked and reaped at interpreter exit — acceptable
    for a CLI that's about to exit anyway one way or the other.
    """
    from huggingface_hub import HfApi

    result: dict = {}

    def _call() -> None:
        try:
            result["info"] = HfApi().model_info(repo_id, files_metadata=True)
        except Exception as exc:  # pragma: no cover - defensive
            result["error"] = exc

    worker = threading.Thread(target=_call, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(f"model_info({repo_id!r}) exceeded {timeout}s")
    if "error" in result:
        raise result["error"]
    return result.get("info")


def estimate_repo_size_bytes(repo_id: str) -> int | None:
    """Best-effort total size of weight + tokenizer files in ``repo_id``.

    Returns the sum of ``sibling.size`` (preferring LFS-reported size
    when available) across files whose extension marks them as weight
    or tokenizer payload. ``None`` on any failure (network down, gated
    repo, HF outage, timeout) — callers should fall through silently.
    """
    try:
        info = _model_info_with_timeout(repo_id, _HF_API_TIMEOUT_SECONDS)
    except Exception:
        return None

    siblings = getattr(info, "siblings", None) or []
    total = 0
    for sib in siblings:
        name = getattr(sib, "rfilename", "") or ""
        if not _is_weight_file(name):
            continue
        total += _sibling_size(sib)
    return total if total > 0 else None


def is_repo_cached(repo_id: str) -> bool:
    """True if ``repo_id`` already has at least a config.json in the HF cache.

    Two probes:

    1. ``huggingface_hub.try_to_load_from_cache(repo_id, "config.json")``
       — the official "is this snapshot resolvable" hook.
    2. Snapshot-directory fallback — handles repos whose layout is
       non-standard (e.g. ``chat_template.jinja`` shipped without a
       ``config.json`` at the same revision).

    Returns ``False`` on any internal exception so the caller defaults
    to the safe path (prompting, if the size warrants it).
    """
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(repo_id, "config.json")
        if isinstance(cached, str) and os.path.exists(cached):
            return True
    except Exception:
        pass

    # Snapshot-directory fallback.
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        snap_root = os.path.join(
            HF_HUB_CACHE,
            f"models--{repo_id.replace('/', '--')}",
            "snapshots",
        )
        if os.path.isdir(snap_root):
            for entry in os.listdir(snap_root):
                snap_dir = os.path.join(snap_root, entry)
                if os.path.isdir(snap_dir) and any(os.scandir(snap_dir)):
                    return True
    except Exception:
        pass

    return False


def confirm_or_abort(
    repo_id: str,
    estimated_bytes: int | None,
    *,
    threshold_bytes: int = 10 * 1024**3,  # 10 GiB
    auto_yes_env: str = "RAPID_MLX_AUTO_PULL",
    logfile_hint: str | None = None,
) -> bool:
    """Interactive gate before a large model download begins.

    Returns ``True`` (proceed) without prompting when:

    * the env var ``auto_yes_env`` is set to a truthy value
      (``"1"``/``"true"``/``"yes"``, case-insensitive), OR
    * ``sys.stdin`` is not a TTY (scripts/CI), OR
    * ``estimated_bytes`` is below ``threshold_bytes``.

    When the size estimate is ``None`` (HF lookup failed) we print a
    heads-up but proceed — blocking on a transient API failure would be
    worse than the silent-download problem we're trying to fix.

    When the user types anything other than ``y``/``yes``, we print a
    one-line abort hint and call ``sys.exit(1)``.
    """
    # Env override always wins.
    env_val = os.environ.get(auto_yes_env, "").strip().lower()
    if env_val in {"1", "true", "yes"}:
        return True

    # Non-interactive: never block; we already burned the user's time
    # if they piped a script that didn't pass --auto-pull.
    if not sys.stdin.isatty():
        return True

    # Unknown size → noisy heads-up but proceed. We get here when the
    # HF metadata API is down or the repo is gated; either way the
    # actual download will surface its own error if there is one.
    if estimated_bytes is None:
        print()
        print(f"  About to download {repo_id}")
        print("    Estimated size: unknown (HF metadata lookup failed)")
        print(
            "    Proceeding without confirmation. Set "
            f"{auto_yes_env}=1 to silence this notice."
        )
        print()
        return True

    # Small downloads don't deserve interruption.
    if estimated_bytes < threshold_bytes:
        return True

    size_str = _format_size(estimated_bytes)
    print()
    print(f"  About to download {repo_id}")
    print(f"    Estimated size: {size_str} (this may take a while on first run)")
    if logfile_hint:
        print(f"    Download progress will appear in {logfile_hint}; tail it to watch.")
    print()
    try:
        answer = input("  Continue? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""

    if answer in {"y", "yes"}:
        return True

    print(
        f"  Aborted. Use 'rapid-mlx pull {repo_id}' to download separately, "
        f"or set {auto_yes_env}=1 to skip this prompt."
    )
    sys.exit(1)
