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

# Suffixes treated as cache-proving by ``is_repo_cached``. mlx-lm's
# high-level loader (the path rapid-mlx serve takes) only globs
# ``model*.safetensors`` — see ``mlx_lm/utils.py:316``.
#
# Codex round-4 BLOCKING #2 trimmed this list to ``.safetensors`` only:
#   * ``.bin``  — PyTorch shards, never loaded by mlx-lm.
#   * ``.gguf`` — mlx-lm has *export* support (convert_to_gguf) but
#                 no load path; ``mx.save_gguf`` is one-way.
#   * ``.npz``  — older mlx-lm convert format; current mlx-lm load
#                 doesn't reach it either.
#
# Keeping these in the cache-proving set lets a non-loadable cache
# (e.g. cached ``weights.npz`` from a 2024-era mlx-community fork) pass
# the gate and route the user back into "silent download in the
# spawned serve subprocess" — which is exactly what B2 exists to fix.
_WEIGHT_ONLY_SUFFIXES: tuple[str, ...] = (".safetensors",)

# 5-second cap on the HF metadata call. Anything slower than this is a
# signal we should fall through silently rather than block startup.
_HF_API_TIMEOUT_SECONDS: float = 5.0


def _format_size(num_bytes: int) -> str:
    """Render ``num_bytes`` as a human-friendly string (e.g. ``42.3 GiB``).

    Uses base-1024 units (KiB/MiB/GiB) to match the way HF Hub and
    macOS Finder report file sizes. The ``iB`` suffix is the IEC
    standard for base-1024 — clearer than bare ``KB``/``GB`` which is
    ambiguous (powers of 10 in some tools, 2 in others) and matters
    here because the confirmation threshold is denominated in 1024**3.
    """
    if num_bytes < 0:
        num_bytes = 0
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:.1f} TiB"  # unreachable, keeps mypy happy


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


def _is_model_weight_filename(name: str) -> bool:
    """True if ``name`` matches mlx-lm's loader glob ``model*.safetensors``.

    mlx-lm's high-level load path (``mlx_lm/utils.py:316``) is literally
    ``glob.glob(str(model_path / "model*.safetensors"))``. Adapter /
    sidecar files (``adapter.safetensors``, LoRA fine-tunes,
    ``embeddings.safetensors``, etc.) DON'T match this pattern and
    aren't loaded by rapid-mlx's text path — so they must NOT count
    as cache-proof either. Codex round-5 BLOCKING #2.
    """
    lower = name.lower()
    if not lower.endswith(".safetensors"):
        return False
    return lower.startswith("model")


def _snapshot_is_complete(snap_dir: str) -> bool:
    """True if ``snap_dir`` looks like a fully-downloaded model snapshot.

    Mirrors ``vllm_mlx.doctor.discovery._is_complete_snapshot`` so the
    two cache-completeness checks (doctor pre-flight + B2 gate) stay in
    sync. The only behavior difference is suffix scope: B2 only cares
    about formats mlx-lm's loader actually consumes (``model*.safetensors``).

    Strategy:
      1. ``model.safetensors.index.json`` present → parse ``weight_map``
         and require every referenced shard to exist with non-zero
         size. Codex round-4 BLOCKING #1.
      2. Otherwise, a single non-empty ``model*.safetensors`` is
         sufficient (covers single-file non-sharded models).

    Index-but-no-shards (Codex round-5 BLOCKING #1): when an
    ``model.safetensors.index.json`` exists but yields no shard names
    (corrupt schema, alternate-key layout, metadata-only index), DO
    NOT fall back to the single-file probe. The presence of the index
    itself is the loader's signal that this is a sharded model — a
    single stray ``model.safetensors`` next to a non-standard index
    is incomplete by definition.
    """
    index_path = os.path.join(snap_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        import json

        try:
            with open(index_path) as fh:
                index = json.load(fh)
        except (OSError, json.JSONDecodeError):
            # Truncated index → safer to treat as incomplete and re-prompt.
            return False
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            # Index exists but doesn't yield a usable shard list. We
            # know SOMETHING expects shards here; refuse to fall
            # through to the lax single-file probe.
            return False
        shard_names = set(weight_map.values())
        for shard in shard_names:
            target = os.path.join(snap_dir, shard)
            try:
                if os.path.getsize(target) <= 0:
                    return False
            except OSError:
                return False
        return True

    # Single-file (non-sharded) model. Match mlx-lm's actual loader
    # glob — ``adapter.safetensors`` / ``embeddings.safetensors`` and
    # other sidecars don't count, only ``model*.safetensors``.
    for root, _dirs, files in os.walk(snap_dir):
        for name in files:
            if not _is_model_weight_filename(name):
                continue
            try:
                if os.path.getsize(os.path.join(root, name)) > 0:
                    return True
            except OSError:
                continue
    return False


def is_repo_cached(repo_id: str) -> bool:
    """True if ``repo_id`` has a usable model snapshot in the HF cache.

    Codex review round 1 caught that an earlier "config.json exists →
    cached" check let a partial cache (config + tokenizer only, weight
    shards missing) bypass the gate. The serve subprocess would then
    silently download the weights inside its log file. Round 4 then
    caught that even "any one safetensors file present" let a partial
    sharded cache (shard 1/2 present, shard 2/2 missing) bypass the
    gate; mlx-lm's loader globs every shard and fails halfway through.

    The check delegates to ``_snapshot_is_complete`` so the doctor
    pre-flight and the B2 gate share one source of truth.

    Returns ``False`` on any internal exception so the caller defaults
    to the safe path (prompting, if the size warrants it).
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        snap_root = os.path.join(
            HF_HUB_CACHE,
            f"models--{repo_id.replace('/', '--')}",
            "snapshots",
        )
        if not os.path.isdir(snap_root):
            return False
        for entry in os.listdir(snap_root):
            snap_dir = os.path.join(snap_root, entry)
            if not os.path.isdir(snap_dir):
                continue
            if _snapshot_is_complete(snap_dir):
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
