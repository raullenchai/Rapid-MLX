# SPDX-License-Identifier: Apache-2.0
"""Disk-backed KV-cache checkpointing at 256-token boundaries (R15-P1 task #296).

This module is the long-context partner of the in-process radix prefix cache
(task #303): instead of holding the full KV tail in RAM for the lifetime of a
session, the scheduler snapshots the cache to disk at fixed token boundaries
(default 256, matching upstream MLX-LM's ``step=256`` allocator and LMCache's
external-chunk size). When the same prefix shows up later — same session
resumed, same shared system prompt, or a long-running agent walking up to its
context cap — the on-disk snapshot is reloaded instead of re-prefilled, which
is the headline 体感 row that unlocks "Mac users can run all day" (82% peak
RAM reduction at long context + 2.2× parallel-chat throughput).

The design is a deliberate port of LM Studio MLX-engine PR #326's prompt-
cache layer (specifically the ``prompt_cache/`` package at commit
``ea1a6bb16``), narrowed to the rapid-mlx use case:

- **Boundary granularity**: 256 tokens (``DEFAULT_CHECKPOINT_INTERVAL``),
  same as ``mlx_engine/.../types.py::DEFAULT_PREFIX_CHUNK_SIZE``. The MLX-LM
  KV cache allocates in 256-token steps, so writing at a multiple of 256 keeps
  the on-disk shape aligned with the in-memory shape — important because the
  loader uses ``mlx_lm.load_prompt_cache`` which reads the cache class name
  out of the safetensors metadata, then constructs a fresh
  ``KVCache``/``QuantizedKVCache`` whose step rounding has to match.
- **On-disk format**: ``mlx_lm.models.cache.save_prompt_cache`` /
  ``load_prompt_cache`` directly. Same path the in-process radix already uses
  for its store/fetch round-trip (memory_cache.py:2017). Pre-existing
  round-trip / corruption / dedup guards (R10-D, R12-T1) apply here too with
  zero extra code.
- **Atomic writes**: write to ``<token_offset>.safetensors.tmp`` + fsync +
  rename to ``<token_offset>.safetensors``. Mirrors the prefix-cache
  ``cache_dir.new/`` → ``cache_dir`` rename in ``MemoryAwarePrefixCache``.
- **Disk-budget eviction**: oldest-first across all checkpoints in
  ``~/.cache/rapid-mlx/kv_checkpoints/``, capped at a configurable byte cap
  (default 20 GiB, env override ``RAPID_MLX_KV_CHECKPOINT_MAX_BYTES``).
  ``mtime``-ordered LRU rather than the size-aware policy in PR #326 because
  the rapid-mlx scheduler is single-tenant per process and the cap exists
  primarily to keep a runaway agent from filling the disk, not to optimize
  hit rate across a vision-mixed workload.
- **Special-model handling**: a small registry
  (:data:`MODELS_REQUIRING_FULL_CHECKPOINT`) marks families whose attention
  cache cannot be sliced — Gemma 4 sliding-window (the cache holds the live
  window state and the offset alone can't reconstruct it) and Qwen3.5 hybrid
  attention (full + sliding layers alternate). For these we write the WHOLE
  ``prompt_cache`` list at the boundary; for everything else we write the
  whole list too (we don't slice — rapid-mlx loads checkpoints "as a
  resumable suspension point" and the writer doesn't have to know which
  layers are sliceable). The registry is exposed for the loader because a
  partial restore policy could be added later: today both paths converge.

Deviations from LM Studio PR #326 (documented for the PR body):

- **No record-kind slicing**. The upstream code separates ``kv_delta`` /
  ``rotating_delta`` / ``state_checkpoint`` and writes per-layer per-chunk.
  We write the whole cache list at one boundary because (a) rapid-mlx's
  in-process radix already handles cross-tenant prefix dedup, so disk
  checkpoints don't need to dedup against each other, and (b) the upstream
  delta path requires fine-grained slicing that doesn't compose with the
  ``QuantizedKVCache`` (whose triple of (packed, scales, biases) can't be
  sliced cheaply on the seq axis without dequantizing first — the prefix
  cache already learnt this the hard way at memory_cache.py:2014).
- **No image-span hashing**. We index by request hash, not by chunk hash.
  Image / vision is handled by the rapid-mlx ``mllm_*`` lane on a separate
  cache.
- **No blob-store coalescing**. Each checkpoint is its own safetensors file
  under a per-request directory; the disk-cap eviction policy is
  mtime-ordered across all files. The upstream
  ``TemporarySafetensorBlobStore`` uses a packed temp-file with extent
  coalescing because the upstream coordinator may write hundreds of small
  delta records per request; we write one per 256-token boundary per
  request, where the disk pressure simply doesn't justify a packed store.

Integration touchpoints:

- The scheduler calls :func:`maybe_write_checkpoint` after every step that
  pushes ``request.num_computed_tokens`` past the next 256-token boundary.
  Cheap when disabled (``interval=0`` short-circuits with no I/O).
- ``vllm_mlx.runtime.cache`` loads checkpoints during startup via
  :func:`scan_checkpoints` — the radix index gets a metadata flag so the
  next lookup knows the entry's source was disk, not RAM. Hand-off to the
  radix is best-effort: a missing index entry just means the next request
  will re-prefill, not crash.
- ``vllm_mlx.runtime.disk_kv_checkpoint.get_stats`` returns a stats dict
  the scheduler folds into ``get_stats()`` so ``/metrics`` can render the
  four ``rapid_mlx_kv_checkpoint_*`` series (writes, loads, bytes,
  evictions).

Concurrency: every public function takes a per-checkpoint-root ``RLock``
(module-level). Writers serialise on the lock; readers ``scan_checkpoints``
also takes the lock to avoid racing the disk-cap eviction. The lock is
cheap because writes happen at most once per 256 generated tokens — way
below the per-step scheduler cadence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

# Default checkpoint interval — matches MLX-LM's KVCache.step (256) and
# LMCache's external chunk size. Picked because round-tripping a checkpoint
# whose token count isn't a multiple of the underlying cache step would
# force the loader to allocate a non-step-aligned buffer on first reuse,
# which then trips the same allocation-noise path the in-process radix is
# already careful to avoid (see _grow_kv_cache step rounding in
# vllm_mlx/positioned_kv_cache.py).
DEFAULT_CHECKPOINT_INTERVAL = 256

# Disk cap default: 20 GiB. The env override is honoured at scan-time so an
# operator can shrink/grow without restarting the server. Picked to match the
# headroom rapid-desktop reserves under ~/.cache/rapid-mlx (#194).
DEFAULT_MAX_DISK_BYTES = 20 * 1024 * 1024 * 1024
_DISK_CAP_ENV = "RAPID_MLX_KV_CHECKPOINT_MAX_BYTES"

# Models that require a FULL cache-state snapshot at each boundary — i.e. the
# attention cache cannot be reconstructed from a position offset alone:
#
# - **Gemma 4 sliding-window**: every layer holds a fixed-size window that
#   rolls forward; rewinding by N tokens requires the actual window contents
#   at that position, not just the offset. Our writer captures the entire
#   ``prompt_cache`` list, which includes the live window state, so the
#   loader gets a faithful resume point.
# - **Qwen3.5 hybrid attention**: full-attention and sliding layers
#   alternate. The sliding layers have the same constraint as Gemma 4; we
#   therefore checkpoint both layer types together — there's no benefit to
#   per-layer slicing because the cache state has to round-trip in
#   lock-step.
#
# Pattern match is case-insensitive substring over BOTH the alias key and
# the resolved HF path (mirrors ``kv_cache_dtype._is_sliding_window``).
# Entries are family globs ("gemma-4-*") rather than exact aliases so newly-
# uploaded quants (e.g. ``mlx-community/gemma-4-12b-int4``) auto-pick the
# right policy without an aliases.json edit.
MODELS_REQUIRING_FULL_CHECKPOINT: frozenset[str] = frozenset(
    {
        "gemma-4",
        "gemma_4",
        "gemma4",
        "qwen3.5",
        "qwen3_5",
        "qwen35",
    }
)

# Filename suffix on the persisted safetensors blob. Kept short because a
# busy disk root accumulates one file per boundary per request and Linux
# ``readdir`` cost scales with path length.
_CHECKPOINT_EXT = ".safetensors"
_METADATA_EXT = ".json"
# Tmp file name shape: ``<basename>.tmp.safetensors``. The trailing
# ``.safetensors`` is REQUIRED because ``mlx.core.save_safetensors``
# silently auto-appends ``.safetensors`` when the path does not already
# end in it (mlx.core 0.31.3 behaviour, verified empirically). Without
# this shape, calling ``save_safetensors('foo.safetensors.tmp', ...)``
# actually writes ``foo.safetensors.tmp.safetensors`` and the subsequent
# rename fails. The ``.tmp.`` infix is what ``scan_checkpoints`` strips
# from on rescan.
_TMP_INFIX = ".tmp"


# ---------------------------------------------------------------------------
# Stats dataclass — folded into Scheduler.get_stats() for /metrics
# ---------------------------------------------------------------------------


@dataclass
class CheckpointStats:
    """Process-monotonic counters surfaced via ``/metrics``.

    Attributes:
        writes: cumulative ``write_checkpoint`` calls that committed (renamed
            the .tmp into place). Failed writes do NOT increment.
        loads: cumulative ``load_checkpoint`` calls that returned a non-None
            cache list.
        bytes: live total byte count across every committed checkpoint under
            the root, refreshed on every ``write_checkpoint`` / scan / evict.
            Gauge (not counter) because the value goes down on eviction.
        evictions: cumulative oldest-first evictions performed because the
            byte total crossed the cap. One per evicted file (so a single
            scan that releases 5 files bumps the counter by 5).
        hook_errors: cumulative unexpected exceptions caught by the
            scheduler's disk-KV hook wrapper (``Scheduler.``
            ``_process_batch_responses`` ``try/except`` around
            ``_maybe_disk_checkpoint``, plus the ``enforce_disk_cap``
            catch inside the hook itself). Counts wrong-attribute typos
            and similarly silent-shipped regressions. **Operators expect
            this to stay 0.** Added 2026-06-29 after PR #919's
            ``self.scheduler_config`` / ``self.batch_gen`` typos shipped
            for two releases without any signal — see the parent commit
            of this PR for the root-cause writeup.
    """

    writes: int = 0
    loads: int = 0
    bytes: int = 0
    evictions: int = 0
    hook_errors: int = 0


# Module-level stats (process-monotonic). Mutated under the lock below.
_STATS = CheckpointStats()
_STATS_LOCK = threading.Lock()
_DISK_LOCK = threading.RLock()


def get_stats() -> dict[str, int]:
    """Snapshot the process-monotonic counters as a dict.

    Called by ``Scheduler.get_stats()`` so ``/metrics`` can fold the four
    ``rapid_mlx_kv_checkpoint_*`` series next to the existing prefix-cache
    series. Snapshotting under the lock keeps a concurrent
    ``write_checkpoint`` from publishing a torn (writes, bytes) pair.
    """
    with _STATS_LOCK:
        return {
            "writes": _STATS.writes,
            "loads": _STATS.loads,
            "bytes": _STATS.bytes,
            "evictions": _STATS.evictions,
            "hook_errors": _STATS.hook_errors,
        }


def record_hook_error() -> None:
    """Bump the ``hook_errors`` counter under the stats lock.

    The scheduler's wrapper at ``Scheduler._process_batch_responses``
    calls this every time the disk-KV hook raises an unexpected
    exception (every *expected* skip path is an early-return inside
    ``_maybe_disk_checkpoint`` and never reaches the wrapper). The
    `enforce_disk_cap` catch inside the hook itself also calls this.
    Surfaces silent regressions like the wrong-attribute typos shipped
    in #919 — see the ``hook_errors`` field doc.
    """
    with _STATS_LOCK:
        _STATS.hook_errors += 1


def reset_stats_for_tests() -> None:
    """Test-only hook: zero the module-level counters.

    Prod code never calls this; the counters are process-monotonic by
    contract (matches every other Prometheus client library).
    """
    global _STATS
    with _STATS_LOCK:
        _STATS = CheckpointStats()


# ---------------------------------------------------------------------------
# Helpers — config resolution + path layout
# ---------------------------------------------------------------------------


def get_default_root() -> str:
    """Return the on-disk root for KV checkpoints.

    ``~/.cache/rapid-mlx/kv_checkpoints/`` — sibling of the existing
    ``prefix_cache/`` directory used by the in-process radix. The dir is
    created lazily by the first ``write_checkpoint`` so operators who never
    enable disk checkpointing don't see an empty directory show up.
    """
    return os.path.join(
        os.path.expanduser("~"), ".cache", "rapid-mlx", "kv_checkpoints"
    )


def resolve_max_disk_bytes(default: int = DEFAULT_MAX_DISK_BYTES) -> int:
    """Resolve the disk cap, honouring the env override.

    Returns 0 (cap disabled) when the env var is explicitly set to ``0``
    or a negative integer. Matches the convention the prefix cache uses
    for ``RAPID_MLX_PREFIX_CACHE_MAX_BYTES``: an explicit ``0`` is the
    escape hatch, not "use default".
    """
    raw = os.environ.get(_DISK_CAP_ENV)
    if raw is None:
        return max(0, int(default))
    try:
        n = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            f"[disk_kv_checkpoint] invalid {_DISK_CAP_ENV}={raw!r}; "
            f"falling back to default {default}"
        )
        return max(0, int(default))
    return max(0, n)


def request_hash(request_id: str, model_name: str | None = None) -> str:
    """Return a short stable hash that pins a request to its checkpoint dir.

    Includes the model name so the same ``request_id`` against two
    different models can't collide (the on-disk safetensors carries the
    cache class names and would error out at load time, but a hash
    collision in the directory layer is the cleaner failure mode).
    """
    raw = f"{model_name or ''}::{request_id}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def checkpoint_path(root: str, req_hash: str, token_offset: int) -> str:
    """Return the absolute safetensors path for one checkpoint."""
    return os.path.join(root, req_hash, f"checkpoint-{token_offset}{_CHECKPOINT_EXT}")


def metadata_path(root: str, req_hash: str, token_offset: int) -> str:
    """Return the absolute metadata JSON path for one checkpoint.

    The JSON sits next to the .safetensors and records the model name,
    KV dtype, sliding/hybrid flags, token offset, and write timestamp.
    Useful for the scan path and for the radix-index hand-off, which
    needs to know "where did this loaded entry come from".
    """
    return os.path.join(root, req_hash, f"checkpoint-{token_offset}{_METADATA_EXT}")


def model_requires_full_checkpoint(
    model_name: str | None,
    hf_path: str | None = None,
    alias_metadata: dict[str, Any] | None = None,
    hf_config: dict[str, Any] | None = None,
) -> bool:
    """Detect whether this model family must checkpoint the WHOLE cache.

    Detection order (cheapest first):
    1. ``alias_metadata['requires_full_checkpoint'] is True`` — explicit
       operator pin via aliases.json (works for verified-tier aliases
       whose family doesn't match a substring pattern). Does NOT touch
       the closed-key fields ``architecture`` / ``family`` /
       ``quantization`` / ``notes`` per the aliases.json schema rule —
       this is a new boolean key only.
    2. ``hf_config['sliding_window']`` populated — the canonical HF
       signal for sliding-window attention. Catches Gemma 4 + sliding
       Mistral variants without name matching.
    3. ``hf_config['hybrid_attention']`` populated truthy — Qwen3.5
       hybrid layer toggle.
    4. Substring match against :data:`MODELS_REQUIRING_FULL_CHECKPOINT`
       over both ``model_name`` and ``hf_path`` (case-insensitive).
       Picks up freshly-quantized community uploads that don't have
       an alias entry yet.

    Returns False on ``None``/empty inputs — disk checkpointing is
    best-effort and a "don't know, assume sliceable" answer just means
    we write the same full snapshot anyway (today both branches converge
    to a full write; the registry gates a future partial path).
    """
    if alias_metadata is not None:
        flag = alias_metadata.get("requires_full_checkpoint")
        if isinstance(flag, bool) and flag:
            return True

    if hf_config is not None:
        sw = hf_config.get("sliding_window")
        if isinstance(sw, int) and sw > 0:
            return True
        if hf_config.get("hybrid_attention"):
            return True

    needle = f"{model_name or ''} {hf_path or ''}".lower()
    return any(pat in needle for pat in MODELS_REQUIRING_FULL_CHECKPOINT)


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------


def should_checkpoint(
    num_tokens: int,
    last_checkpoint_at: int,
    interval: int = DEFAULT_CHECKPOINT_INTERVAL,
) -> bool:
    """Return True when ``num_tokens`` has crossed the next boundary.

    Boundary semantics (locked by ``test_disk_kv_checkpoint.py``):

    - ``interval=0`` → never checkpoint. The CLI flag uses 0 as the
      disable sentinel; the helper honours it so callers don't have to
      add a separate gate at every call site.
    - ``num_tokens < interval`` → no checkpoint yet. The first boundary
      lands AT ``interval`` (so for the default 256: offsets 0..255 do
      nothing, 256 fires the first checkpoint, 257..511 stay quiet,
      512 fires the second, …).
    - ``num_tokens >= last_checkpoint_at + interval`` → fire. Using
      ``last_checkpoint_at`` rather than a strict ``% interval == 0``
      keeps the trigger correct even when the scheduler skips token
      counts (spec decode can advance by multiple tokens per step).
    - Negative / NaN tokens are floored to 0 (defensive — the scheduler
      caller already validates, but the unit test exercises this).
    """
    if interval <= 0:
        return False
    if not isinstance(num_tokens, int):
        # Be paranoid — a stray float from a user-supplied SamplingParams
        # field that survived Field validation could end up here.
        try:
            num_tokens = int(num_tokens)
        except (TypeError, ValueError):
            return False
    if num_tokens < 0:
        return False
    if num_tokens < interval:
        return False
    return num_tokens >= last_checkpoint_at + interval


def write_checkpoint(
    cache: list[Any],
    *,
    root: str,
    req_hash: str,
    token_offset: int,
    kv_dtype: str = "bf16",
    requires_full_checkpoint: bool = False,
    model_name: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
    radix_index: Any | None = None,
) -> str | None:
    """Write one cache snapshot to disk at ``token_offset`` atomically.

    Path layout:
        <root>/<req_hash>/checkpoint-<token_offset>.safetensors
        <root>/<req_hash>/checkpoint-<token_offset>.json

    Atomicity contract:
    - The safetensors body is written to ``<...>.safetensors.tmp``,
      fsync'd, then atomically renamed into place. A SIGKILL between
      ``open`` and ``rename`` leaves only the .tmp file, which
      ``scan_checkpoints`` ignores AND clears on first visit.
    - The metadata JSON is written + fsync'd + renamed AFTER the
      safetensors rename so a partial commit can never expose a JSON
      that points at a missing body.

    Returns the safetensors path on success, or None when:
    - ``interval <= 0`` (caller already short-circuited via
      :func:`should_checkpoint`, but defensive)
    - the write failed before rename (logged, counters untouched —
      this is the "best-effort persistence" contract the in-process
      radix already uses)

    Args:
        cache: ``list`` of MLX-LM cache layers (KVCache /
            QuantizedKVCache / hybrid). Must round-trip through
            ``mlx_lm.save_prompt_cache``. The caller is responsible for
            using :func:`vllm_mlx.positioned_kv_cache.positioned_update_and_fetch`
            for any pre-checkpoint writes; passing a ``PositionedKVCache``
            subclass instance here would WORK at write time but FAIL at
            load time because ``mlx_lm.load_prompt_cache`` looks the
            class name up in the upstream module globals.
        root: directory containing per-request subdirs. Created on
            demand; survives across restarts.
        req_hash: short stable hash (see :func:`request_hash`).
        token_offset: number of tokens already in the cache. Used as
            both the filename suffix and the metadata field for the
            radix-index hand-off.
        kv_dtype: ``"bf16"``/``"int8"``/``"int4"`` — recorded in
            metadata for the loader's bookkeeping. Does NOT change the
            on-disk format; ``save_prompt_cache`` writes the cache
            class names regardless.
        requires_full_checkpoint: pre-resolved via
            :func:`model_requires_full_checkpoint`. Recorded in the
            metadata so the loader can refuse to restore a partial
            snapshot from a model family that needs full state.
        model_name: alias key or HF path. Recorded for observability.
        extra_metadata: free-form dict added to the JSON. Used by the
            radix hand-off to record the source token sequence hash.
        radix_index: optional radix-index handle; when provided AND
            the metadata carries a ``tokens_key`` list, the radix is
            notified via ``radix_index.insert(tokens_key)`` so the
            next prefix lookup can find the on-disk entry without a
            re-scan. Best-effort: any radix exception is logged and
            the write succeeds anyway.
    """
    # The CLI / scheduler caller already gates on
    # :func:`should_checkpoint`; the guard here is a belt for the
    # in-process radix path that pokes ``write_checkpoint`` directly.
    if not isinstance(token_offset, int) or token_offset < 0:
        return None
    if cache is None or not cache:
        return None

    with _DISK_LOCK:
        try:
            from mlx_lm.models.cache import save_prompt_cache
        except ImportError:  # pragma: no cover — every prod env has mlx_lm
            logger.warning("[disk_kv_checkpoint] mlx_lm not importable; skipping")
            return None

        dst_dir = os.path.join(root, req_hash)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = checkpoint_path(root, req_hash, token_offset)
        meta_path = metadata_path(root, req_hash, token_offset)
        # See ``_TMP_INFIX`` comment: the tmp path must end in
        # ``.safetensors`` or ``mx.save_safetensors`` will rewrite the
        # filename and the rename will fail.
        tmp_path = dst_path.replace(_CHECKPOINT_EXT, _TMP_INFIX + _CHECKPOINT_EXT)
        meta_tmp = meta_path + _TMP_INFIX

        # Build the safetensors metadata that ships INSIDE the file.
        # ``save_prompt_cache`` requires str→str — JSON-encode the
        # boolean / int fields so the round-trip is faithful.
        st_meta = {
            "token_offset": str(token_offset),
            "kv_dtype": kv_dtype,
            "requires_full_checkpoint": "true" if requires_full_checkpoint else "false",
        }
        if model_name:
            st_meta["model_name"] = str(model_name)

        try:
            save_prompt_cache(tmp_path, cache, metadata=st_meta)
            # Durably commit the body BEFORE the rename. Same rationale as
            # ``memory_cache.py`` R8-M7 codex r1 BLOCKING #3 — without
            # the fsync a SIGTERM-driven shutdown could leave a renamed
            # file with empty/partial contents on hard reset.
            _fsync_file(tmp_path)
            os.replace(tmp_path, dst_path)
            _fsync_dir(dst_dir)
        except Exception as e:
            logger.warning(
                f"[disk_kv_checkpoint] safetensors write failed at {dst_path!r}: {e}",
                exc_info=True,
            )
            # Best-effort cleanup so the next ``scan_checkpoints`` doesn't
            # see a stale .tmp.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            return None

        # Sidecar JSON — written AFTER the safetensors so a torn shutdown
        # can never leave a JSON pointing at a missing body.
        meta_payload: dict[str, Any] = {
            "schema_version": 1,
            "token_offset": int(token_offset),
            "kv_dtype": str(kv_dtype),
            "requires_full_checkpoint": bool(requires_full_checkpoint),
            "model_name": model_name,
            "created_at": time.time(),
            "size_bytes": _safe_filesize(dst_path),
        }
        if extra_metadata:
            for k, v in extra_metadata.items():
                if k in meta_payload:
                    # Don't let extra_metadata clobber the fields we own;
                    # silently skip rather than raise so a buggy caller
                    # can't tear the write down.
                    continue
                meta_payload[k] = v

        try:
            with open(meta_tmp, "w", encoding="utf-8") as fh:
                json.dump(meta_payload, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(meta_tmp, meta_path)
            _fsync_dir(dst_dir)
        except Exception as e:
            logger.warning(
                f"[disk_kv_checkpoint] metadata write failed at "
                f"{meta_path!r}: {e}; body left in place at {dst_path!r}"
            )
            try:
                os.unlink(meta_tmp)
            except OSError:
                pass
            # Body is still valid; the loader tolerates a missing
            # metadata sidecar (treats it as "unknown source").

        # Stats: writes++, bytes refreshed against the live filesystem so
        # we never report stale totals after an eviction.
        with _STATS_LOCK:
            _STATS.writes += 1
            _STATS.bytes = _measure_root_bytes(root)

        # Radix hand-off — best-effort, mirrors the in-process store path.
        if radix_index is not None and extra_metadata is not None:
            tokens_key = extra_metadata.get("tokens_key")
            if isinstance(tokens_key, (list, tuple)) and tokens_key:
                try:
                    radix_index.insert(list(tokens_key))
                except Exception as e:  # pragma: no cover — radix is optional
                    logger.debug(f"[disk_kv_checkpoint] radix.insert failed: {e}")

        return dst_path


def maybe_write_checkpoint(
    cache: list[Any],
    *,
    root: str,
    req_hash: str,
    num_tokens: int,
    last_checkpoint_at: int,
    interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    kv_dtype: str = "bf16",
    requires_full_checkpoint: bool = False,
    model_name: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
    radix_index: Any | None = None,
) -> tuple[int, str | None]:
    """Convenience wrapper: gate via :func:`should_checkpoint`, then write.

    Returns ``(new_last_checkpoint_at, path_or_None)``:
    - ``new_last_checkpoint_at`` is the largest multiple of ``interval``
      that is ``<= num_tokens``. The scheduler stores this on the
      request so the next call doesn't re-fire.
    - ``path_or_None`` is the safetensors path on success, None when
      the gate was open but the write failed.

    Called once per scheduler step from the hook in
    :mod:`vllm_mlx.scheduler`.
    """
    if not should_checkpoint(num_tokens, last_checkpoint_at, interval):
        return last_checkpoint_at, None

    # Snap the new boundary to the largest multiple of ``interval`` that
    # is still ``<= num_tokens``. Without snapping, a step that advances
    # by N>interval (e.g. spec decode) would fire one checkpoint and
    # then re-fire on the next step because ``last_checkpoint_at`` only
    # bumped by interval, not by the actual gap.
    new_boundary = (num_tokens // interval) * interval

    path = write_checkpoint(
        cache,
        root=root,
        req_hash=req_hash,
        token_offset=new_boundary,
        kv_dtype=kv_dtype,
        requires_full_checkpoint=requires_full_checkpoint,
        model_name=model_name,
        extra_metadata=extra_metadata,
        radix_index=radix_index,
    )
    if path is None:
        # Don't advance the watermark when the write failed — the next
        # boundary still gets a try.
        return last_checkpoint_at, None
    return new_boundary, path


# ---------------------------------------------------------------------------
# Load + scan path
# ---------------------------------------------------------------------------


@dataclass
class LoadedCheckpoint:
    """Result of a successful :func:`load_checkpoint` call.

    Attributes:
        cache: ``list`` of MLX-LM cache layers ready to feed into the
            BatchGenerator's prompt cache slot.
        token_offset: number of tokens already in ``cache``.
        kv_dtype: ``"bf16"``/``"int8"``/``"int4"`` recorded at write
            time. Loader uses this to refuse a mismatched re-load if
            the operator switched ``--kv-cache-dtype`` between runs.
        requires_full_checkpoint: True when the source model is in
            :data:`MODELS_REQUIRING_FULL_CHECKPOINT`. The scheduler can
            use this to refuse a partial restore.
        metadata: sidecar JSON contents — free-form, useful for the
            radix-index hand-off.
        path: absolute safetensors path the cache came from.
    """

    cache: list[Any]
    token_offset: int
    kv_dtype: str
    requires_full_checkpoint: bool
    metadata: dict[str, Any]
    path: str


def load_checkpoint(path: str) -> LoadedCheckpoint | None:
    """Load one checkpoint by absolute safetensors path.

    Returns ``None`` and logs a warning when:
    - the file is missing / unreadable
    - ``mlx_lm.load_prompt_cache`` raises (corrupt body, class-name
      mismatch — the latter is the trap the disk format inherits from
      ``save_prompt_cache``)
    - the sidecar JSON is missing AND the safetensors metadata fails to
      decode

    Calls the ``loads`` counter on success.
    """
    with _DISK_LOCK:
        try:
            from mlx_lm.models.cache import load_prompt_cache
        except ImportError:  # pragma: no cover — every prod env has mlx_lm
            logger.warning("[disk_kv_checkpoint] mlx_lm not importable; skipping")
            return None

        if not os.path.isfile(path):
            return None

        try:
            cache, st_meta = load_prompt_cache(path, return_metadata=True)
        except Exception as e:
            logger.warning(
                f"[disk_kv_checkpoint] load_prompt_cache failed at {path!r}: {e}"
            )
            return None

        # Sidecar metadata is the source of truth; fall back to the
        # safetensors metadata if the sidecar went missing.
        meta_path_str = path.replace(_CHECKPOINT_EXT, _METADATA_EXT)
        sidecar: dict[str, Any] = {}
        if os.path.isfile(meta_path_str):
            try:
                with open(meta_path_str, encoding="utf-8") as fh:
                    sidecar = json.load(fh)
            except Exception as e:
                logger.warning(
                    f"[disk_kv_checkpoint] sidecar load failed at "
                    f"{meta_path_str!r}: {e}; falling back to embedded metadata"
                )

        # The embedded metadata is str→str; coerce safely.
        embedded = st_meta or {}
        token_offset = int(
            sidecar.get("token_offset")
            if sidecar.get("token_offset") is not None
            else embedded.get("token_offset", 0) or 0
        )
        kv_dtype = str(
            sidecar.get("kv_dtype") or embedded.get("kv_dtype", "bf16") or "bf16"
        )
        requires_full = bool(
            sidecar.get("requires_full_checkpoint")
            if "requires_full_checkpoint" in sidecar
            else (
                str(embedded.get("requires_full_checkpoint", "false")).lower() == "true"
            )
        )

        with _STATS_LOCK:
            _STATS.loads += 1

        return LoadedCheckpoint(
            cache=cache,
            token_offset=token_offset,
            kv_dtype=kv_dtype,
            requires_full_checkpoint=requires_full,
            metadata=sidecar,
            path=path,
        )


def scan_checkpoints(root: str) -> list[tuple[str, float, int]]:
    """Return ``[(path, mtime, size_bytes), …]`` for every committed checkpoint.

    Cleans up stale ``.tmp`` files as a side effect (a SIGKILL between
    the safetensors write and rename leaves them; they're never
    recoverable so erasing them is strictly safe).

    Used by:
    - The disk-cap eviction loop in :func:`enforce_disk_cap`.
    - The startup loader hand-off in
      :mod:`vllm_mlx.runtime.cache` (a future iteration; today the loader
      is gated on memory-aware cache presence and disk checkpoints aren't
      auto-loaded back into a fresh engine).
    """
    with _DISK_LOCK:
        if not os.path.isdir(root):
            return []

        out: list[tuple[str, float, int]] = []
        # Tmp suffix shapes:
        #   <name>.tmp.safetensors  — safetensors body tmp
        #   <name>.json.tmp         — sidecar JSON tmp
        # Both are stale on rescan and must be cleaned up.
        tmp_body_marker = _TMP_INFIX + _CHECKPOINT_EXT  # e.g. ".tmp.safetensors"
        tmp_json_marker = _METADATA_EXT + _TMP_INFIX  # e.g. ".json.tmp"
        for entry in os.scandir(root):
            if not entry.is_dir(follow_symlinks=False):
                continue
            try:
                for child in os.scandir(entry.path):
                    name = child.name
                    if name.endswith(tmp_body_marker) or name.endswith(tmp_json_marker):
                        # Stale tmp from a torn write — best-effort cleanup.
                        try:
                            os.unlink(child.path)
                        except OSError:
                            pass
                        continue
                    if not name.endswith(_CHECKPOINT_EXT):
                        continue
                    try:
                        stat = child.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    out.append((child.path, stat.st_mtime, stat.st_size))
            except OSError:
                # Per-request dir vanished mid-scan — fine, move on.
                continue

        out.sort(key=lambda row: row[1])
        return out


def enforce_disk_cap(root: str, *, max_bytes: int | None = None) -> tuple[int, int]:
    """Evict oldest checkpoints until the on-disk total fits in ``max_bytes``.

    Returns ``(num_evicted, bytes_remaining)`` for the caller's log line.
    ``max_bytes`` defaults to :func:`resolve_max_disk_bytes`; pass ``0``
    to skip the cap (escape hatch — operators on a big disk who don't
    want eviction at all). NaN-safe: any non-finite float is clamped to
    the default.
    """
    if max_bytes is None:
        max_bytes = resolve_max_disk_bytes()
    elif isinstance(max_bytes, float) and not math.isfinite(max_bytes):
        # NaN/Inf coercion — Pydantic Field(ge=) does NOT reject these,
        # so the validation has to happen here for any user-input float
        # that survived the schema layer.
        max_bytes = resolve_max_disk_bytes()
    max_bytes = max(0, int(max_bytes))

    with _DISK_LOCK:
        entries = scan_checkpoints(root)
        total = sum(size for _, _, size in entries)
        if max_bytes == 0 or total <= max_bytes:
            with _STATS_LOCK:
                _STATS.bytes = total
            return 0, total

        evicted = 0
        for path, _mtime, size in entries:
            if total <= max_bytes:
                break
            try:
                os.unlink(path)
            except OSError as e:
                logger.warning(
                    f"[disk_kv_checkpoint] eviction unlink({path!r}) failed: {e}"
                )
                continue
            sidecar = path.replace(_CHECKPOINT_EXT, _METADATA_EXT)
            try:
                os.unlink(sidecar)
            except OSError:
                pass
            total -= size
            evicted += 1
            # Best-effort prune of the parent directory if the eviction
            # just emptied it. Keeps the scan loop cheap on long-running
            # servers.
            parent = os.path.dirname(path)
            try:
                if not os.listdir(parent):
                    os.rmdir(parent)
            except OSError:
                pass

        with _STATS_LOCK:
            _STATS.evictions += evicted
            _STATS.bytes = total

        return evicted, total


# ---------------------------------------------------------------------------
# Per-request bookkeeping helpers (for the scheduler)
# ---------------------------------------------------------------------------


@dataclass
class RequestCheckpointState:
    """In-memory bookkeeping the scheduler stores per-request.

    Carries the last successfully-written boundary so
    :func:`should_checkpoint` can stay stateless. Optional fields are
    populated by the boot path:

    Attributes:
        req_hash: stable hash from :func:`request_hash`. Cached so the
            hot path doesn't re-hash on every step.
        interval: per-request override (defaults to the CLI flag value).
            ``0`` disables disk checkpointing for THIS request only.
        last_checkpoint_at: number of tokens already on disk for this
            request. Bumped by :func:`maybe_write_checkpoint`.
        requires_full_checkpoint: pre-resolved via
            :func:`model_requires_full_checkpoint`. Passed through to
            the writer at every boundary.
        kv_dtype: ``"bf16"`` / ``"int8"`` / ``"int4"`` — recorded in
            metadata so the loader can refuse a mismatched re-load.
    """

    req_hash: str
    interval: int = DEFAULT_CHECKPOINT_INTERVAL
    last_checkpoint_at: int = 0
    requires_full_checkpoint: bool = False
    kv_dtype: str = "bf16"
    model_name: str | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def _fsync_file(path: str) -> None:
    """fsync a file by path, swallowing errors the caller will retry."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: str) -> None:
    """fsync a directory so the rename is durable on hard reset.

    Linux requires an explicit dir-fsync after a rename for the new
    name to survive power loss; macOS (HFS+/APFS) handles this within
    the rename syscall but the extra call is cheap and matches the
    cross-platform contract the prefix cache already uses.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        # macOS sometimes refuses fsync on a dir descriptor — non-fatal.
        pass
    finally:
        os.close(fd)


def _safe_filesize(path: str) -> int:
    """Return the byte size of ``path``, or 0 if the stat fails.

    Used only for the sidecar metadata, so a missed read just means
    the JSON records 0 — the actual disk-cap accounting uses
    ``scan_checkpoints`` and never trusts the sidecar value.
    """
    try:
        return os.stat(path).st_size
    except OSError:
        return 0


def _measure_root_bytes(root: str) -> int:
    """Return the total live bytes under the checkpoint root.

    Cheap O(N) scan via ``scan_checkpoints``; called only after a
    successful write or eviction so it's amortized across the
    256-token boundary cadence, not the per-step hot path.
    """
    try:
        return sum(size for _, _, size in scan_checkpoints(root))
    except Exception:  # pragma: no cover — defensive
        return 0


def cleanup_request(root: str, req_hash: str) -> int:
    """Drop every checkpoint for one request (e.g. on completion).

    Returns the number of files removed. Best-effort — partial cleanup
    is fine, the next ``enforce_disk_cap`` pass will mop up.

    Called by the scheduler when a request finishes / errors out so the
    on-disk footprint matches the live request set.
    """
    with _DISK_LOCK:
        dir_path = os.path.join(root, req_hash)
        if not os.path.isdir(dir_path):
            return 0
        try:
            n = sum(1 for _ in os.scandir(dir_path))
        except OSError:
            n = 0
        try:
            shutil.rmtree(dir_path, ignore_errors=True)
        except Exception:  # pragma: no cover — defensive
            return 0
        with _STATS_LOCK:
            _STATS.bytes = _measure_root_bytes(root)
        return n


# ---------------------------------------------------------------------------
# Test-only: deterministic root override
# ---------------------------------------------------------------------------


def temporary_root() -> str:
    """Return a fresh temporary checkpoint root (unit-test helper).

    Used by the disk-checkpoint tests so they don't pollute
    ``~/.cache/rapid-mlx/`` and don't race against any other agent. The
    caller is responsible for ``shutil.rmtree`` cleanup; using
    ``tempfile.TemporaryDirectory`` is cleaner in test fixtures.
    """
    return tempfile.mkdtemp(prefix="rapid-mlx-kv-checkpoint-")


__all__ = [
    "CheckpointStats",
    "DEFAULT_CHECKPOINT_INTERVAL",
    "DEFAULT_MAX_DISK_BYTES",
    "LoadedCheckpoint",
    "MODELS_REQUIRING_FULL_CHECKPOINT",
    "RequestCheckpointState",
    "checkpoint_path",
    "cleanup_request",
    "enforce_disk_cap",
    "get_default_root",
    "get_stats",
    "load_checkpoint",
    "maybe_write_checkpoint",
    "metadata_path",
    "model_requires_full_checkpoint",
    "record_hook_error",
    "request_hash",
    "reset_stats_for_tests",
    "resolve_max_disk_bytes",
    "scan_checkpoints",
    "should_checkpoint",
    "temporary_root",
    "write_checkpoint",
]
