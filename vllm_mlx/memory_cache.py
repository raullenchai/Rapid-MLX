# SPDX-License-Identifier: Apache-2.0
"""
Memory-aware prefix cache for rapid-mlx.

This module provides a prefix cache implementation that tracks memory usage
and evicts entries based on memory pressure rather than entry count.

Key features:
- Automatic memory limit detection based on available system RAM
- Accurate memory tracking for MLX array caches
- LRU eviction triggered by memory thresholds
- Deep copies on fetch to prevent mutation of stored cache entries

Example:
    config = MemoryCacheConfig(max_memory_percent=0.25)
    cache = MemoryAwarePrefixCache(model, config)

    # Fetch returns reference (no copy) - safe because MLX arrays are immutable
    kv_cache, remaining = cache.fetch(tokens)

    # Store tracks memory automatically
    cache.store(tokens, kv_cache)
"""

from __future__ import annotations

import bisect
import copy
import json
import logging
import math
import os
import struct
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Constants
_BYTES_PER_MB = 1024 * 1024
_DEFAULT_MEMORY_PERCENT = 0.20  # 20% of available RAM
_MIN_MEMORY_BYTES = 100 * _BYTES_PER_MB  # Minimum 100MB
_MAX_ENTRIES_FALLBACK = 50  # Fallback if memory detection fails

# ---------------------------------------------------------------------------
# Persist-format pinning (R10-D, Talia r10-R1)
# ---------------------------------------------------------------------------
# Multi-cycle SIGTERM round-trips were dropping ~30% of reloaded entries
# (29/42 with 13 corrupt) because the on-disk format had no per-entry
# integrity stamp. ``index.json`` claimed "entry K has N tokens", but a
# previous-cycle orphan or a partial mid-write could leave ``entry_K_tokens.bin``
# carrying a DIFFERENT entry's payload at the same int-array byte length
# — the size cross-check at load passed and the loader registered a
# mismatched key. The fix below pins three invariants per save:
#
#   1. A file-level ``save_uuid`` written into ``index.json`` AND embedded
#      in every ``entry_K_tokens.bin`` — guarantees the entry file came
#      from the same save as the index that references it. Any orphan
#      from a previous save fails this check at load and gets skipped
#      with a structured WARN + a metric increment.
#
#   2. A per-entry magic header ``RMTKBIN1`` + version byte so the loader
#      can tell at-a-glance whether a tokens.bin came from a v3-aware
#      writer. Legacy v2 files (no magic, no save_uuid) are detected by
#      the absence of the magic and fall through to the pre-R10 path:
#      size cross-check only, no uuid guard. This preserves the
#      single-cycle PASS path (Talia's 12/12 round-trip) byte-exact.
#
#   3. An explicit ``token_count`` length prefix INSIDE tokens.bin
#      (uint32 LE) — must equal both ``index["entries"][i]["num_tokens"]``
#      AND ``(file_size - header_len) / 4``. Any disagreement is the
#      length-prefix off-by-one signature the spec called out.
#
# Bumping the file-level ``index["version"]`` from 2 → 3 lets old loaders
# refuse a new file cleanly. New loaders accept both 2 and 3 so the
# upgrade path is one-way: a v3 writer can read a v2 file from a
# previous deploy (legacy-mode tokens.bin), then re-save under v3.
_TOKENS_MAGIC = b"RMTKBIN1"  # 8 bytes — "rapid-mlx tokens-bin v1"
# Header layout for v3 tokens.bin:
#   [0..8)   magic       — b"RMTKBIN1"
#   [8..12)  token_count — uint32 LE
#   [12..16) save_uuid_len — uint32 LE (length of uuid string in bytes, ≤64)
#   [16..16+save_uuid_len) save_uuid — utf-8 hex string
#   then aligned to 4-byte boundary, then token_count * 4 bytes of int32 LE
# A short fixed-prefix (magic + 2 lengths) reads in a single ``f.read(16)``;
# the variable uuid is a hex string so an operator can grep / diff snapshots
# without binary tooling.
_TOKENS_HEADER_FIXED_LEN = 16
_TOKENS_FORMAT_VERSION_IN_INDEX = 3  # bumped from 2

# Per-token serialization width is fixed at 4 bytes (int32 LE) regardless
# of ``array.array("i").itemsize`` — that attribute is platform-dependent
# (4 on POSIX 64-bit, 2 on some legacy Windows builds), and pinning a
# wire-format width is the only sound way to make tokens.bin portable
# across the heterogeneous fleet (codex r10-D systematic fix). Writer
# uses struct.pack into bytes; reader uses struct.unpack_from a slice.
_TOKEN_STRUCT_FMT = "<i"
_TOKEN_BYTES = 4


def _write_tokens_bin_v3(path: str, tokens: list[int], save_uuid: str) -> None:
    """Write tokens.bin with magic + length + save_uuid + int32 LE tokens.

    File layout pinned at _TOKENS_HEADER_FIXED_LEN-byte fixed prefix
    followed by a variable-length uuid string and the token payload —
    see module-level "Persist-format pinning" comment for the full
    layout description.

    Atomic on the fd close + os.fsync — the caller is responsible for
    fsyncing the containing directory after this returns. Raises on any
    write error (caller's outer try/except converts to a per-entry
    skip + WARN).
    """
    import struct as _struct

    uuid_bytes = save_uuid.encode("ascii")
    if len(uuid_bytes) > 64:
        # Defensive: uuid is always 32 hex chars in our writer, but cap
        # at 64 so a future widening can't blow past a sane reader bound.
        raise ValueError(
            f"save_uuid too long for tokens.bin header ({len(uuid_bytes)} > 64)"
        )
    header = _TOKENS_MAGIC + _struct.pack("<II", len(tokens), len(uuid_bytes))
    assert len(header) == _TOKENS_HEADER_FIXED_LEN, "fixed-prefix length drift"
    # Pack tokens via struct so the wire format is independent of
    # ``array.array("i").itemsize`` on this host. A single struct.pack
    # with ``count * 4`` bytes is faster than per-token packing for the
    # common 100-10K-token range — uses CPython's tightly-vectorized
    # bulk path.
    payload = _struct.pack(f"<{len(tokens)}i", *tokens)
    with open(path, "wb") as f:
        f.write(header)
        f.write(uuid_bytes)
        f.write(payload)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            # Same non-fatal logic as the legacy path — the verify loop
            # in save_to_disk catches files that vanished after we wrote
            # them.
            pass


def _peek_tokens_bin_header(path: str) -> tuple[int | None, str | None, str]:
    """Read just the v3 header of a tokens.bin (magic + length + uuid).

    Returns ``(token_count, save_uuid, reject_reason)``. On any
    structural / IO problem returns ``(None, None, reason)``.

    Used by :meth:`MemoryAwarePrefixCache.save_to_disk`'s post-write
    self-verify pass — see R12-T1 in the docstring there. The fast
    path reads at most ``_TOKENS_HEADER_FIXED_LEN + 64`` bytes (16
    fixed + bounded uuid) and skips the int32 LE token payload so
    a 100-entry verify against a 2.6 GB on-disk snapshot completes
    in milliseconds rather than seconds.

    Pinned to the v3 wire format only — legacy v2 sidecars never
    carried magic / length / uuid, so the post-write check is moot
    for them (they're never written by the current writer). Returns
    ``(None, None, "missing v3 magic")`` for a v2-shaped file; the
    caller treats that as a structural reject in the verify pass.

    Kept structurally close to ``_read_tokens_bin`` so a future tweak
    to the wire format only has to touch the per-field offsets once.
    """
    import struct as _struct

    try:
        with open(path, "rb") as f:
            head = f.read(_TOKENS_HEADER_FIXED_LEN)
            if len(head) < _TOKENS_HEADER_FIXED_LEN:
                return None, None, "tokens.bin truncated at fixed prefix"
            if head[: len(_TOKENS_MAGIC)] != _TOKENS_MAGIC:
                return None, None, "tokens.bin missing v3 magic"
            token_count, uuid_len = _struct.unpack("<II", head[len(_TOKENS_MAGIC) :])
            if uuid_len > 64:
                return None, None, (f"save_uuid_len {uuid_len} exceeds bound 64")
            uuid_bytes = f.read(uuid_len)
            if len(uuid_bytes) != uuid_len:
                return (
                    None,
                    None,
                    (f"save_uuid short read ({len(uuid_bytes)}/{uuid_len})"),
                )
            return token_count, uuid_bytes.decode("ascii", errors="replace"), ""
    except OSError as exc:
        return None, None, f"open/read failed: {exc}"


def _read_tokens_bin(
    path: str, expected_num_tokens: int, expected_save_uuid: str | None
) -> tuple[list[int] | None, str]:
    """Read tokens.bin. Returns (tokens, reject_reason).

    Detects v3 magic at offset 0:
      * magic present → enforce length prefix + save_uuid (when provided)
        + int32-LE payload size; any mismatch returns (None, reason).
      * magic absent → fall through to legacy path: read
        ``expected_num_tokens`` ints via ``array.array("i")``. Only used
        for v2 index.json files (no save_uuid claim). Preserves
        byte-exact single-cycle round-trip behavior for in-flight
        upgrades.

    ``reject_reason`` is empty string on success. Never raises for
    structural mismatches — the caller treats reject_reason as a
    skip + bump-metric signal. Raises only on truly unexpected I/O
    failures.
    """
    import struct as _struct

    with open(path, "rb") as f:
        head = f.read(_TOKENS_HEADER_FIXED_LEN)
        if len(head) < len(_TOKENS_MAGIC):
            return None, "tokens.bin shorter than magic length"
        if head[: len(_TOKENS_MAGIC)] != _TOKENS_MAGIC:
            # Magic absent. ONLY legitimate when the file-level index
            # advertised no save_uuid (v2 legacy layout). If the index
            # claimed v3 but this tokens.bin lacks the magic, that's a
            # mid-rewrite or external clobber — fail closed instead of
            # silently falling through to legacy int-array decode,
            # which would feed the loader 60 bytes of header content
            # as "tokens" (codex r10-D round-2 BLOCKING — fall-through
            # could re-expose the silent-mis-decode bug R10-D exists
            # to prevent).
            if expected_save_uuid is not None:
                return None, (
                    "tokens.bin missing v3 magic but index.json declared "
                    f"save_uuid={expected_save_uuid!r}"
                )
            # Legacy v2 path — enforce the same exact-size invariant
            # the pre-R10-D loader did (codex round 2 HIGH: a v2
            # sidecar with trailing garbage was silently accepted by
            # `array.fromfile`; preserve the historic "size mismatch
            # = corruption" contract so existing BUG A defenses don't
            # regress).
            try:
                actual_size = os.fstat(f.fileno()).st_size
            except OSError as exc:
                return None, f"legacy tokens.bin stat failed: {exc}"
            expected_size = expected_num_tokens * _TOKEN_BYTES
            if actual_size != expected_size:
                return None, (
                    f"legacy tokens.bin size mismatch "
                    f"(expected {expected_size} bytes for "
                    f"{expected_num_tokens} tokens, got {actual_size})"
                )
            f.seek(0)
            import array as _array

            arr = _array.array("i")
            try:
                arr.fromfile(f, expected_num_tokens)
            except (EOFError, OSError) as exc:
                return None, f"legacy tokens.bin short read: {exc}"
            return list(arr), ""

        if len(head) < _TOKENS_HEADER_FIXED_LEN:
            return None, "tokens.bin truncated after magic"
        token_count, uuid_len = _struct.unpack("<II", head[len(_TOKENS_MAGIC) :])
        if uuid_len > 64:
            return None, f"save_uuid_len {uuid_len} exceeds bound 64"
        if token_count != expected_num_tokens:
            # Length-prefix off-by-one / drift — exactly the failure
            # mode the spec called out (entry's payload was rewritten
            # with a different cycle's tokens; index.json never caught
            # up).
            return None, (
                f"length prefix mismatch: tokens.bin says {token_count}, "
                f"index.json says {expected_num_tokens}"
            )
        uuid_bytes = f.read(uuid_len)
        if len(uuid_bytes) != uuid_len:
            return None, f"save_uuid short read ({len(uuid_bytes)}/{uuid_len})"
        on_disk_uuid = uuid_bytes.decode("ascii", errors="replace")
        if expected_save_uuid is not None and on_disk_uuid != expected_save_uuid:
            # Orphan from a previous cycle — exactly the multi-cycle
            # drift scenario. Skip cleanly so the loader doesn't pair
            # the index's "entry K" claim with another save's bytes.
            return None, (
                f"save_uuid mismatch: tokens.bin={on_disk_uuid!r} "
                f"index={expected_save_uuid!r}"
            )
        payload_bytes_expected = token_count * _TOKEN_BYTES
        payload = f.read(payload_bytes_expected)
        if len(payload) != payload_bytes_expected:
            return None, (
                f"payload short read ({len(payload)}/{payload_bytes_expected})"
            )
        # R10-D codex round 2 HIGH: enforce "no trailing bytes" so the
        # v3 wire format is unambiguous. A v3 sidecar that decoded
        # cleanly through the header + payload but carried EXTRA bytes
        # past EOF would otherwise be silently accepted; that's
        # exactly the per-entry corruption signature R10-D was built
        # to catch (a length-prefix that disagrees with the on-disk
        # size). Read one more byte — any non-empty trailing read is
        # a structural reject.
        trailing = f.read(1)
        if trailing:
            return None, (
                "v3 tokens.bin has unexpected trailing bytes past "
                f"declared payload (token_count={token_count}, "
                f"payload={payload_bytes_expected} bytes)"
            )
        try:
            tokens = list(_struct.unpack_from(f"<{token_count}i", payload, 0))
        except _struct.error as exc:
            return None, f"struct.unpack_from failed: {exc}"
        return tokens, ""


def _fsync_file(path: str) -> None:
    """Flush a file's contents to disk.

    R8-M7 codex r1 BLOCKING #3: ``_fsync_dir`` alone is insufficient —
    the dir fsync only commits directory metadata (file entries +
    names), not the file body. A file whose contents are still in the
    page cache can survive a dir-fsync rename and surface as empty
    / partial on a hard reset. This helper opens the file read-only
    and calls ``os.fsync`` to force the body durable BEFORE the
    rename commits.

    Opens read-only so we don't disturb the file's mtime / atime;
    fsync on a read-only fd is allowed on POSIX (some platforms
    require write but Linux/macOS accept either). Errors propagate
    so the caller can decide non-fatal vs hard-fail.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: str) -> None:
    """Flush directory metadata to disk.

    R8-M7: the persist commit phase relies on ``os.rename`` being
    atomic relative to a subsequent crash. POSIX guarantees
    rename-atomicity at the metadata level, but the contents of the
    files INSIDE the staging dir (entry safetensors + index.json) may
    still be buffered in the kernel page cache when the rename
    commits. Without ``fsync`` on the directory, a power loss / OOM
    kill / kernel panic between the rename and the periodic flush
    can leave the renamed dir pointing at empty / partial files —
    observed on Linux ext4 with ``data=writeback`` mount option;
    macOS APFS is more conservative but the fsync is still a
    correctness invariant.

    POSIX-only; on Windows there is no equivalent directory fsync
    (the filesystem journals dir metadata differently) and we
    silently no-op. The caller catches OSError, so an unsupported
    platform doesn't break the save.

    Implementation detail: open with ``O_RDONLY`` because ``O_DIRECTORY``
    is not available on every platform (and Python's ``os.open`` does
    accept opening a dir RDONLY on POSIX). ``os.fsync`` on the
    returned fd is what flushes the dir's metadata journal entry.
    """
    if not hasattr(os, "O_DIRECTORY") and os.name == "nt":
        # Windows: no directory-fsync equivalent. The recovery path
        # in load_from_disk is the fall-back; skip silently.
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _adapt_should_abort(predicate):
    """Adapt a ``should_abort`` predicate to the one-arg contract.

    The forward-looking shape ``Callable[[float], bool]`` is the new
    contract, but external callers / older fixtures may pass a
    zero-arg ``Callable[[], bool]`` from the round-1 docstring.
    Inspect the signature once and return a normalized
    ``Callable[[float], bool]`` that calls the inner predicate
    correctly. ``None`` passes through unchanged.

    Codex PR #667 round 3 BLOCKING-2: round-2 unconditionally called
    ``should_abort(predicted_sec)`` which raises ``TypeError`` against
    zero-arg predicates documented in the previous contract.
    """
    if predicate is None:
        return None

    import inspect

    try:
        sig = inspect.signature(predicate)
    except (TypeError, ValueError):
        # Builtin / C-extension / partial — assume positional one-arg
        # shape (it's the contract going forward); a runtime TypeError
        # on invocation is no worse than what callers got before.
        return lambda predicted_sec: predicate(predicted_sec)

    # Classify the predicate's calling convention. Codex PR #667 round
    # 4 BLOCKING-1: a naive "accepts ANY arg" check sent keyword-only
    # and ``**kwargs``-only callables down the positional path, which
    # raises ``TypeError`` on the very first call — defeating the
    # whole point of the adapter. We have to distinguish the call
    # shape, not just "accepts something".
    accepts_positional = False
    accepts_keyword_only_predicted_sec = False
    has_var_kwargs = False
    for p in sig.parameters.values():
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            accepts_positional = True
        elif p.kind == inspect.Parameter.KEYWORD_ONLY:
            if p.name == "predicted_sec":
                accepts_keyword_only_predicted_sec = True
        elif p.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_kwargs = True

    if accepts_positional:
        # ``def pred(p)`` / ``def pred(p, **kw)`` / ``def pred(*args)``
        # — positional is the natural shape.
        return lambda predicted_sec: predicate(predicted_sec)
    if accepts_keyword_only_predicted_sec or has_var_kwargs:
        # ``def pred(*, predicted_sec=0.0)`` — must use the keyword.
        # ``def pred(**kw)`` — keyword is the only shape it accepts;
        # the predicate may or may not look for ``predicted_sec`` in
        # ``kw``, but passing it by name is the contract.
        return lambda predicted_sec: predicate(predicted_sec=predicted_sec)
    # Zero parameters → call with no args (round-1 documented shape).
    return lambda predicted_sec: predicate()


def _safetensors_is_complete(path: str) -> bool:
    """Validate a safetensors file is at least as long as its header claims.

    Catches the body-truncated case that ``mx.load`` happily mmaps over —
    a partial KV file that returns zeros at the missing positions and only
    blows up much later with a wrong-output bug. Cheap: reads ≤ a few KB.

    File layout (per safetensors spec):
        [8 bytes LE uint64: header_len]
        [header_len bytes: JSON header with data_offsets per tensor]
        [tensor data]

    Returns False on any structural problem (caller should drop the entry).
    """
    parsed = _read_safetensors_header(path)
    if parsed is None:
        return False
    header, header_len = parsed
    try:
        size = os.path.getsize(path)
        max_end = 0
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            offsets = meta.get("data_offsets") if isinstance(meta, dict) else None
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(x, int) for x in offsets)
                or offsets[0] < 0
                or offsets[1] < offsets[0]
            ):
                return False
            if offsets[1] > max_end:
                max_end = offsets[1]
        return size >= 8 + header_len + max_end
    except (OSError, ValueError, struct.error, AttributeError, TypeError):
        return False


def _read_safetensors_header(path: str) -> tuple[dict, int] | None:
    """Parse a safetensors header without loading any tensor data.

    Returns ``(header_dict, header_len_bytes)`` on success, or ``None`` if
    the file is structurally invalid. Both values are needed by
    :func:`_safetensors_is_complete` to compute the absolute end-of-data
    offset; :func:`_safetensors_cache_classes` ignores ``header_len``.
    Returning both from one read avoids opening the file twice.
    """
    try:
        size = os.path.getsize(path)
        if size < 8:
            return None
        with open(path, "rb") as f:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) != 8:
                return None
            header_len = struct.unpack("<Q", header_len_bytes)[0]
            if header_len <= 0 or 8 + header_len > size:
                return None
            header_bytes = f.read(header_len)
            if len(header_bytes) != header_len:
                return None
        header = json.loads(header_bytes)
        if not isinstance(header, dict):
            return None
        return header, header_len
    except (OSError, ValueError, struct.error, AttributeError, TypeError):
        return None


def _safetensors_cache_classes(path: str) -> list[str]:
    """Read mlx-lm cache class names from a safetensors prompt-cache file.

    ``mlx_lm.models.cache.save_prompt_cache`` writes per-layer class names
    under metadata keys of the form ``"2.{layer_idx}"``. This reads them
    back without instantiating the cache — needed to gate disk-cache
    loading on cache-type compatibility (see Bug B in #198).

    Returns ``[]`` if the file is unreadable, has no metadata, or has no
    ``2.*`` keys. The caller treats ``[]`` as "permissive — assume
    ``KVCache``" for backward compat with files saved before the
    in-index ``cache_types`` field existed; that's safe today because
    every mlx-lm version we depend on writes the ``2.*`` metadata, so
    an actually-quantized file always yields a non-empty list. If a
    future mlx-lm changes the metadata key layout this will silently
    misclassify quantized files as KVCache and re-expose Bug A — emit
    a one-time WARNING when the metadata block exists but yields no
    ``2.*`` keys, so format drift is at least visible in logs.
    """
    parsed = _read_safetensors_header(path)
    if parsed is None:
        return []
    header, _ = parsed
    meta = header.get("__metadata__")
    if not isinstance(meta, dict):
        return []
    layer_classes: dict[int, str] = {}
    for k, v in meta.items():
        if not isinstance(k, str) or not k.startswith("2."):
            continue
        try:
            idx = int(k.split(".", 1)[1])
        except (IndexError, ValueError):
            continue
        if isinstance(v, str) and v:
            layer_classes[idx] = v
    if not layer_classes and meta:
        # Header has metadata but no recognizable per-layer class keys —
        # likely a future mlx-lm format change. Surface it so the cause
        # is diagnosable without parsing the safetensors by hand.
        logger.warning(
            f"[cache_persist] {path}: safetensors __metadata__ present "
            f"but no '2.*' cache-class keys found "
            f"(meta_keys={sorted(meta)[:8]}...) — assuming plain KVCache; "
            f"if this file was actually quantized, the entry may crash "
            f"the scheduler at fetch (#198 BUG A)"
        )
    return [layer_classes[i] for i in sorted(layer_classes)]


def _cache_classes_compatible(
    class_names: list[str], config: MemoryCacheConfig
) -> tuple[bool, str]:
    """Check whether a persisted cache is loadable under the current config.

    Reasoning (see #198 BUG B):

    * ``KVCache`` / ``MambaCache`` / etc. — always loadable. Under
      ``kv_quantize`` or ``kv_turboquant`` the next ``store()`` call
      will recompress; until then they pass through fetch unchanged.
    * ``QuantizedKVCache`` — only loadable when ``kv_quantize=True``.
      The dequantize path in ``_decompress_cache`` is guarded on the
      flag; under any other config the tuple-form ``keys`` reach the
      scheduler and crash (#198 BUG A's downstream symptom).
    * ``TurboQuantKVCache`` — only loadable when ``kv_turboquant=True``.
      In practice never persisted (no ``state`` attribute), so this
      branch is defensive.

    Returns ``(is_compatible, reason)``. ``reason`` is empty when ok.
    """
    if not class_names:
        # Backward compat: pre-cache_type files have no class info. Assume
        # KVCache (the only thing all earlier rapid-mlx versions wrote).
        # Always compatible.
        return True, ""
    for cn in class_names:
        if cn == "QuantizedKVCache" and not config.kv_quantize:
            return (
                False,
                f"persisted {cn} requires --kv-cache-quantization "
                "(current config does not enable it)",
            )
        if cn == "TurboQuantKVCache" and not config.kv_turboquant:
            return (
                False,
                f"persisted {cn} requires --kv-cache-turboquant "
                "(current config does not enable it)",
            )
    return True, ""


def _get_available_memory() -> int:
    """
    Get available system memory in bytes.

    Returns:
        Available memory in bytes, or 0 if detection fails.
    """
    try:
        import psutil

        return psutil.virtual_memory().available
    except ImportError:
        logger.warning("psutil not installed, using fallback memory limit")
        return 0
    except Exception as e:
        logger.warning(f"Failed to detect available memory: {e}")
        return 0


# Name of the env var operators set to bound the prefix-cache memory.
# Exported so the metrics route, config dumps, and the tests can refer
# to a single canonical string.
PREFIX_CACHE_MAX_BYTES_ENV = "RAPID_MLX_PREFIX_CACHE_MAX_BYTES"


def _resolve_env_cache_max_bytes() -> int:
    """Read ``RAPID_MLX_PREFIX_CACHE_MAX_BYTES`` from the environment.

    Returns the parsed integer when the env var is set to a positive
    integer; ``0`` for any other shape (unset, blank, non-integer, or
    non-positive). The caller treats ``0`` as "no override" and falls
    through to the legacy heuristic. Out-of-shape values are logged
    once per process so a misconfigured operator gets a visible
    diagnostic without flooding subsequent reconfigs.
    """
    raw = os.environ.get(PREFIX_CACHE_MAX_BYTES_ENV)
    if raw is None:
        return 0
    raw = raw.strip()
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError:
        global _ENV_CACHE_MAX_BYTES_PARSE_WARNED
        if not _ENV_CACHE_MAX_BYTES_PARSE_WARNED:
            _ENV_CACHE_MAX_BYTES_PARSE_WARNED = True
            logger.warning(
                "%s=%r is not a valid integer; ignoring and falling back "
                "to the heuristic limit.",
                PREFIX_CACHE_MAX_BYTES_ENV,
                raw,
            )
        return 0
    if value <= 0:
        return 0
    return value


# Once-per-process flag so an operator who set ``RAPID_MLX_PREFIX_CACHE_MAX_BYTES``
# to garbage (e.g. ``"5GB"`` instead of bytes) sees the warning once and
# the cache silently falls through to the heuristic instead of spamming
# the log on every reconfig.
_ENV_CACHE_MAX_BYTES_PARSE_WARNED = False


def _array_memory(arr) -> int:
    """
    Estimate array memory from shape+dtype without triggering lazy eval.

    Accessing .nbytes on a lazy MLX array forces evaluation of the entire
    computation graph, causing a VRAM spike. This function uses shape and
    dtype metadata (which are always available without eval) to compute
    the same value.

    Args:
        arr: An MLX array or similar object.

    Returns:
        Estimated memory in bytes.
    """
    if arr is None:
        return 0
    if hasattr(arr, "shape") and hasattr(arr, "dtype"):
        dtype = arr.dtype
        if hasattr(dtype, "size"):
            return math.prod(arr.shape) * dtype.size
    # Fallback for non-MLX arrays or objects without shape/dtype
    if hasattr(arr, "nbytes"):
        return arr.nbytes
    return 0


def estimate_kv_cache_memory(cache: list[Any]) -> int:
    """
    Estimate memory usage of a KV cache in bytes.

    This function inspects MLX arrays in the cache and calculates their
    total memory footprint using shape+dtype metadata to avoid triggering
    lazy evaluation (which would cause a VRAM spike).

    Args:
        cache: List of layer cache objects, each containing keys/values tensors.

    Returns:
        Estimated memory usage in bytes.
    """
    if not cache:
        return 0

    total_bytes = 0

    for layer_cache in cache:
        if layer_cache is None:
            continue
        # TurboQuantKVCache: has values_compressed instead of values
        from .turboquant import TurboQuantKVCache

        if isinstance(layer_cache, TurboQuantKVCache):
            total_bytes += layer_cache.memory_bytes
            continue
        # Handle different cache object types
        # Check dict first since dicts have .keys() method that would match below
        if isinstance(layer_cache, dict) and "state" in layer_cache:
            # Extracted state dict
            keys, values = layer_cache["state"]
            total_bytes += _array_memory(keys)
            total_bytes += _array_memory(values)
        # Handle QuantizedKVCache: keys/values are tuples of (data, scales, biases)
        elif hasattr(layer_cache, "keys") and isinstance(
            getattr(layer_cache, "keys", None), (list, tuple)
        ):
            for arr in layer_cache.keys:
                total_bytes += _array_memory(arr)
            for arr in layer_cache.values:
                total_bytes += _array_memory(arr)
            continue
        elif hasattr(layer_cache, "state") and not isinstance(layer_cache, dict):
            # Cache with state property returning (keys, values)
            try:
                keys, values = layer_cache.state
                total_bytes += _array_memory(keys)
                total_bytes += _array_memory(values)
            except (TypeError, ValueError):
                pass
        elif hasattr(layer_cache, "keys") and hasattr(layer_cache, "values"):
            # Standard KVCache with keys/values attributes (not dict)
            keys_attr = layer_cache.keys
            values_attr = layer_cache.values
            # Ensure these are arrays, not methods
            if not callable(keys_attr):
                total_bytes += _array_memory(keys_attr)
            if not callable(values_attr):
                total_bytes += _array_memory(values_attr)

    return total_bytes


@dataclass(frozen=True)
class MemoryCacheConfig:
    """
    Configuration for memory-aware prefix cache.

    Attributes:
        max_memory_mb: Maximum memory in MB. If None, auto-detects.
        max_memory_percent: Fraction of available RAM to use (0.0-1.0).
        max_entries: Hard limit on number of entries (safety net).
        enable_memory_tracking: Whether to track per-entry memory.
        kv_quantize: Whether to quantize KV cache layers for reduced memory.
        kv_bits: Number of bits for KV cache quantization.
        kv_group_size: Group size for KV cache quantization.
        kv_min_quantize_tokens: Minimum sequence length for quantization to apply.
    """

    max_memory_mb: int | None = None
    max_memory_percent: float = _DEFAULT_MEMORY_PERCENT
    max_entries: int = 1000  # Safety limit
    enable_memory_tracking: bool = True
    kv_quantize: bool = False
    kv_bits: int = 8
    kv_group_size: int = 64
    kv_min_quantize_tokens: int = 256
    # TurboQuant KV cache compression. ``kv_turboquant_mode`` selects
    # between the legacy ``"v4"`` (V-only) and ``"k8v4"`` (K-8bit +
    # V-4bit) schemes shipped in R15 Phase 4.
    kv_turboquant: bool = False
    kv_turboquant_bits: int | None = None  # None = auto-select by head_dim
    kv_turboquant_group_size: int = 32
    kv_turboquant_mode: str = "v4"

    def __post_init__(self) -> None:
        if not 0.0 < self.max_memory_percent <= 1.0:
            raise ValueError(
                f"max_memory_percent must be in (0, 1], got {self.max_memory_percent}"
            )
        if self.max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {self.max_entries}")
        if self.kv_min_quantize_tokens < 0:
            raise ValueError(
                f"kv_min_quantize_tokens must be >= 0, got {self.kv_min_quantize_tokens}"
            )

    def compute_memory_limit(self) -> int:
        """
        Compute the memory limit in bytes.

        Resolution order (first hit wins):
          1. ``RAPID_MLX_PREFIX_CACHE_MAX_BYTES`` env var — operator
             override for ops who need to bound the cache to a known
             ceiling regardless of system RAM (R6-H6 fix from the
             0.8.7 dogfood: the default 20% of available RAM let the
             cache balloon to 31 GB on a large-memory host before any
             eviction fired). Accepts a plain integer (bytes); invalid
             / non-positive values fall through to the next step so a
             misconfigured operator gets the legacy default rather
             than a hard server failure.
          2. ``MemoryCacheConfig.max_memory_mb`` — programmatic
             override set by callers (CLI / config plumbing).
          3. ``max_memory_percent`` × available RAM (default 20%).
          4. ``max_memory_percent`` × 8 GiB fallback when psutil is
             unavailable.

        Returns:
            Memory limit in bytes.
        """
        env_override = _resolve_env_cache_max_bytes()
        if env_override > 0:
            # The env override is an OPERATOR ceiling — we trust it
            # verbatim. NOT clamped to ``_MIN_MEMORY_BYTES`` because
            # that floor only exists to keep the heuristic 20% × RAM
            # path from underestimating on a memory-starved host. An
            # operator who explicitly set a small value wants the
            # small value (e.g. test fixtures that drive eviction
            # against a deterministic cap).
            return env_override

        if self.max_memory_mb is not None:
            return self.max_memory_mb * _BYTES_PER_MB

        available = _get_available_memory()
        if available > 0:
            limit = int(available * self.max_memory_percent)
            return max(limit, _MIN_MEMORY_BYTES)

        # Fallback: assume 8GB system, use configured percent
        fallback_total = 8 * 1024 * _BYTES_PER_MB
        return int(fallback_total * self.max_memory_percent)


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    tokens_saved: int = 0
    current_memory_bytes: int = 0
    max_memory_bytes: int = 0
    entry_count: int = 0
    # R10-D (Talia r10-R1): persist-format drift across multi-cycle
    # round-trips can corrupt ~30% of entries when the on-disk index
    # disagrees with the tokens.bin / safetensors that ship beside it.
    # Each per-entry rejection at load (magic mismatch, length-prefix
    # mismatch, save-uuid mismatch, body-truncated safetensors, or
    # an mlx_lm.load_prompt_cache exception) bumps this counter so the
    # operator can see the dropout rate in /metrics rather than buried
    # in WARNING logs. Survives ``cache.clear()`` so the Prometheus
    # counter contract holds — see ``reset_stats`` for the carry-over.
    load_skipped: int = 0
    # R12-T1 (dogfood-0815 Talia r12 SEVERE): save-side mirror of
    # ``load_skipped``. Counts entries dropped by the post-write
    # self-verify pass in ``save_to_disk`` — a tokens.bin in our
    # ``.new`` staging dir that disagrees with the index.json we're
    # about to commit (save_uuid drift or length-prefix mismatch).
    # Pre-R12-T1 such drift survived into ``cache_dir`` and the
    # NEXT boot's loader refused the whole snapshot via R10-D's
    # integrity guard ("LOADED 0 entries ... SKIPPED N corrupt").
    # The post-write verify catches the drift before the rename so
    # the corruption never reaches the committed snapshot — this
    # counter surfaces the rate so operators see the rescue rather
    # than a silent on-disk loss. Cumulative, carries across
    # ``cache.clear()`` / ``reset_stats``, same contract as
    # ``load_skipped``.
    save_drift_drops: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def memory_utilization(self) -> float:
        if self.max_memory_bytes == 0:
            return 0.0
        return self.current_memory_bytes / self.max_memory_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "evictions": self.evictions,
            "tokens_saved": self.tokens_saved,
            "current_memory_mb": round(self.current_memory_bytes / _BYTES_PER_MB, 2),
            "max_memory_mb": round(self.max_memory_bytes / _BYTES_PER_MB, 2),
            # R7-M1 (dogfood-088 Talia r2): raw-byte fields surface the
            # cap + current usage in the unit the Prometheus gauges
            # ``rapid_mlx_prefix_cache_cap_bytes`` and
            # ``rapid_mlx_prefix_cache_current_bytes`` consume. The
            # MB-rounded fields above stay (existing dashboards depend
            # on them) but Prometheus prefers raw bytes for byte-unit
            # series (see "Base units" in the Prometheus naming
            # conventions doc). Both rows are static-cost — they read
            # ints already tracked on this stats object.
            "current_memory_bytes": int(self.current_memory_bytes),
            "max_memory_bytes": int(self.max_memory_bytes),
            "memory_utilization": round(self.memory_utilization, 4),
            "entry_count": self.entry_count,
            # R10-D: cumulative count of entries the loader rejected for
            # any per-entry corruption signal — drives the
            # ``rapid_mlx_prefix_cache_load_skipped_total`` Prometheus
            # counter and closes the R9-L4 observability gap.
            "load_skipped": int(self.load_skipped),
            # R12-T1: cumulative count of entries the save-side
            # post-write verify rejected — drives the
            # ``rapid_mlx_prefix_cache_save_drift_drops_total``
            # Prometheus counter. Cumulative same as ``load_skipped``.
            "save_drift_drops": int(self.save_drift_drops),
        }


@dataclass
class _CacheEntry:
    """Internal cache entry with memory tracking."""

    tokens: tuple[int, ...]
    cache: list[Any]
    memory_bytes: int

    @classmethod
    def create(cls, tokens: list[int], cache: list[Any]) -> _CacheEntry:
        """Create a cache entry with memory estimation."""
        memory = estimate_kv_cache_memory(cache)
        return cls(
            tokens=tuple(tokens),
            cache=cache,
            memory_bytes=memory,
        )


def _trim_cache_offset(cache: list[Any], trim_by: int) -> list[Any]:
    """Create shallow copies of KVCache/QuantizedKVCache layers with offset reduced.

    This is used when returning a cached KV state to the scheduler so that
    the last N positions are "freed" and the model will recompute them on the
    next forward pass (preventing duplicate KV entries).

    Supports both KVCache (keys/values are arrays) and QuantizedKVCache
    (keys/values are 3-tuples of arrays).
    """
    from mlx_lm.models.cache import KVCache

    try:
        from mlx_lm.models.cache import QuantizedKVCache
    except ImportError:
        QuantizedKVCache = None  # noqa: N806

    trimmed: list[Any] = []
    for layer_cache in cache:
        if layer_cache is None:
            trimmed.append(layer_cache)
            continue
        if QuantizedKVCache is not None and isinstance(layer_cache, QuantizedKVCache):
            tc = QuantizedKVCache.__new__(QuantizedKVCache)
            tc.keys = layer_cache.keys
            tc.values = layer_cache.values
            tc.offset = max(layer_cache.offset - trim_by, 0)
            tc.group_size = layer_cache.group_size
            tc.bits = layer_cache.bits
            trimmed.append(tc)
        elif hasattr(layer_cache, "values_compressed"):
            # TurboQuantKVCache — use its trim method on a copy
            tc = copy.copy(layer_cache)
            tc.trim(trim_by)
            trimmed.append(tc)
        elif (
            hasattr(layer_cache, "offset")
            and hasattr(layer_cache, "keys")
            and not isinstance(layer_cache.keys, (list, tuple))
        ):
            tc = KVCache.__new__(KVCache)
            tc.keys = layer_cache.keys
            tc.values = layer_cache.values
            tc.offset = max(layer_cache.offset - trim_by, 0)
            trimmed.append(tc)
        else:
            # Deep copy unknown/wrapper layers (e.g. CacheList) to prevent
            # aliasing the stored cache entry — generation mutates in-place.
            trimmed.append(copy.deepcopy(layer_cache))
    return trimmed


def _needs_kv_trim(layer: Any) -> bool:
    """Check if a cache layer has oversized KV arrays (duck-typed, no MLX import)."""
    if layer is None:
        return False
    keys = getattr(layer, "keys", None)
    offset = getattr(layer, "offset", None)
    if keys is None or offset is None:
        return False
    if isinstance(keys, (list, tuple)):
        return False  # QuantizedKVCache — skip
    shape = getattr(keys, "shape", None)
    if shape is None or len(shape) < 3:
        return False
    return 0 < offset < shape[2]


def _trim_to_offset(cache: list[Any]) -> list[Any]:
    """Trim KV arrays to their actual used size (offset) before storage.

    KV arrays are often pre-allocated larger than needed (e.g. 4096 slots
    when only 100 are used).  This slices them down to ``offset`` and
    evaluates the result so the original large buffer can be freed.

    Args:
        cache: List of cache layer objects (KVCache or other types).

    Returns:
        New list with KVCache layers trimmed to their offset.
        Non-KVCache layers are passed through unchanged.
    """
    if not any(_needs_kv_trim(layer) for layer in cache):
        return cache

    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    trimmed = []
    eval_targets = []
    for layer in cache:
        if isinstance(layer, KVCache) and layer.keys is not None:
            offset = layer.offset
            if offset <= 0 or offset >= layer.keys.shape[2]:
                trimmed.append(layer)
                continue
            tc = KVCache()
            tc.keys = layer.keys[:, :, :offset, :]
            tc.values = layer.values[:, :, :offset, :]
            tc.offset = offset
            eval_targets.extend([tc.keys, tc.values])
            trimmed.append(tc)
        else:
            trimmed.append(layer)

    if eval_targets:
        mx.eval(*eval_targets)

    return trimmed


def _quantize_cache(cache: list[Any], bits: int = 8, group_size: int = 64) -> list[Any]:
    """Quantize KVCache layers to reduce memory. Non-KVCache layers are kept as-is."""
    from mlx_lm.models.cache import KVCache

    quantized = []
    for layer in cache:
        if layer is None:
            quantized.append(layer)
            continue
        if isinstance(layer, KVCache) and layer.keys is not None:
            quantized.append(layer.to_quantized(group_size=group_size, bits=bits))
        else:
            quantized.append(layer)
    return quantized


def _dequantize_cache(cache: list[Any]) -> list[Any]:
    """Dequantize QuantizedKVCache layers back to regular KVCache."""
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache, QuantizedKVCache

    result = []
    for layer in cache:
        if layer is None:
            result.append(layer)
            continue
        if isinstance(layer, QuantizedKVCache) and layer.keys is not None:
            kv = KVCache()
            kv.keys = mx.dequantize(
                *layer.keys, group_size=layer.group_size, bits=layer.bits
            )
            kv.values = mx.dequantize(
                *layer.values, group_size=layer.group_size, bits=layer.bits
            )
            kv.offset = layer.offset
            result.append(kv)
        else:
            result.append(layer)
    return result


def _turboquant_compress_cache(
    cache: list[Any],
    bits: int | None,
    group_size: int,
    mode: str = "v4",
) -> list[Any]:
    """Compress KVCache tensors using TurboQuant.

    Args:
        cache: Per-layer KV cache list.
        bits: V-side bit width (3 or 4). ``None`` triggers auto-select
            by ``head_dim`` — 3-bit for head_dim>=96, 4-bit for 64.
        group_size: V-side group size.
        mode: Compression mode — ``"v4"`` (V-only, legacy) or
            ``"k8v4"`` (K-8bit + V-4bit). When set to ``"k8v4"`` the
            V-side bit width is forced to 4 (the K8 path is only
            validated against V4).
    """
    from mlx_lm.models.cache import KVCache

    from .turboquant import TurboQuantConfig, TurboQuantKVCache, auto_select_bits

    compressed_count = 0
    result = []
    for layer in cache:
        if layer is None:
            result.append(layer)
            continue
        if isinstance(layer, KVCache) and layer.keys is not None:
            head_dim = layer.values.shape[-1] if layer.values is not None else 128
            if mode == "k8v4":
                # K8V4 is V4-pinned; ignore the V-side ``bits`` arg so
                # operators who left the legacy default (``None``) on
                # head_dim=96+ don't fall back to V=3-bit silently.
                actual_bits = 4
            else:
                actual_bits = bits if bits is not None else auto_select_bits(head_dim)
            config = TurboQuantConfig(
                bits=actual_bits, group_size=group_size, mode=mode
            )
            result.append(TurboQuantKVCache.from_kv_cache(layer, config))
            compressed_count += 1
        else:
            result.append(layer)

    if compressed_count > 0:
        logger.debug(
            f"TurboQuant compressed {compressed_count}/{len(cache)} layers "
            f"(mode={mode}, {bits or 'auto'}-bit V, group_size={group_size})"
        )
    return result


def _turboquant_decompress_cache(cache: list[Any]) -> list[Any]:
    """Decompress TurboQuantKVCache layers back to regular KVCache."""
    from .turboquant import TurboQuantKVCache

    result = []
    for layer in cache:
        if layer is None:
            result.append(layer)
            continue
        if isinstance(layer, TurboQuantKVCache) and layer.keys is not None:
            result.append(layer.to_kv_cache())
        else:
            result.append(layer)
    return result


class MemoryAwarePrefixCache:
    """
    Prefix cache with memory-based eviction.

    This cache tracks memory usage per entry and evicts based on memory
    pressure rather than entry count. It uses LRU (Least Recently Used)
    ordering for eviction decisions.

    Key design decisions:
    - No deep copies on fetch: MLX arrays are immutable, so sharing is safe
    - Memory tracking per entry: Accurate accounting for eviction
    - Auto-detection of available RAM: Adapts to different systems
    - OrderedDict for O(1) LRU operations

    Thread Safety:
        ``fetch``, ``store``, ``remove`` and ``clear`` hold an internal lock,
        so it is safe to call them from different threads (e.g. the asyncio
        event loop calling ``fetch`` while the mlx-step worker calls
        ``store``). Read-only attribute access (``__contains__``, ``__len__``,
        ``get_stats``) is single-op and relies on the GIL — no lock needed.
    """

    def __init__(
        self,
        model: Any,
        config: MemoryCacheConfig | None = None,
        radix_index: Any = None,
    ) -> None:
        """
        Initialize the memory-aware prefix cache.

        Args:
            model: The MLX model (used for identification).
            config: Cache configuration. Uses defaults if None.
            radix_index: Optional ``RadixPrefixIndex`` (R15-P1, task #303).
                When supplied, all store/remove/evict/clear mutations also
                keep the radix in sync, AND the prefix-match path inside
                ``fetch`` consults the radix first. The radix is the
                source-of-truth for prefix lookup but NOT for entry
                storage — that stays in ``_entries``. Pass ``None`` to
                fall back to the legacy bisect-over-sorted-keys path.
        """
        self._model_id = id(model)
        self._config = config or MemoryCacheConfig()

        # OrderedDict maintains insertion order for LRU
        # Key: tuple(tokens), Value: _CacheEntry
        self._entries: OrderedDict[tuple[int, ...], _CacheEntry] = OrderedDict()

        # Sorted index of token keys for efficient prefix/supersequence lookup.
        # Tuple lexicographic ordering means a prefix key P is always < any
        # extension of P, so bisect gives O(log N) range scans instead of O(N).
        self._sorted_keys: list[tuple[int, ...]] = []

        # R15-P1 radix-tree prefix-cache index. When set, ``fetch`` uses
        # the radix's O(prefix_len) walk instead of bisect+LCP scan for
        # exact / prefix matches; supersequence and LCP fallbacks still
        # run through the sorted index (those require a "give me an entry
        # LONGER than my query" lookup which the radix doesn't accelerate
        # over the bisect). Keeping both paths live means the radix is
        # additive and can be flipped off via the CLI without code change.
        self._radix_index = radix_index

        # Memory tracking
        self._max_memory = self._config.compute_memory_limit()
        self._current_memory = 0

        # Statistics
        self._stats = CacheStats(max_memory_bytes=self._max_memory)

        # Track the match type from the last fetch() call
        self._last_match_type: str | None = None

        # Guards _entries / _sorted_keys mutations against concurrent
        # fetch/store/evict from multiple threads (asyncio loop + mlx-step).
        self._lock = threading.Lock()

        logger.info(
            f"MemoryAwarePrefixCache initialized: "
            f"max_memory={self._max_memory / _BYTES_PER_MB:.1f}MB, "
            f"max_entries={self._config.max_entries}, "
            f"radix_index={'on' if radix_index is not None else 'off'}"
        )

    def _decompress_cache(self, cache: list[Any]) -> list[Any]:
        """Decompress cache layers (TurboQuant or standard quantization)."""
        if self._config.kv_turboquant:
            return _turboquant_decompress_cache(cache)
        elif self._config.kv_quantize:
            return _dequantize_cache(cache)
        return cache

    def fetch(self, tokens: list[int]) -> tuple[list[Any] | None, list[int]]:
        """
        Find cached KV state for the given tokens.

        This method searches for exact matches, prefix matches, supersequence
        matches, and longest-common-prefix (LCP) matches.  Uses a sorted key
        index for O(log N) lookup instead of scanning all entries.

        Returns the cached KV state directly (no copy) since MLX arrays
        are immutable and safe to share.

        Args:
            tokens: Input token sequence.

        Returns:
            Tuple of (cache, remaining_tokens):
            - cache: Cached KV state if found, None otherwise
            - remaining_tokens: Tokens that still need processing
        """
        if not tokens:
            self._stats.misses += 1
            self._last_match_type = "miss"
            return None, tokens

        tokens_key = tuple(tokens)

        with self._lock:
            return self._fetch_locked(tokens, tokens_key)

    def _fetch_locked(
        self, tokens: list[int], tokens_key: tuple[int, ...]
    ) -> tuple[list[Any] | None, list[int]]:
        # --- O(1) exact match ---
        if tokens_key in self._entries:
            entry = self._entries[tokens_key]
            self._entries.move_to_end(tokens_key)
            self._stats.hits += 1
            self._stats.tokens_saved += len(tokens)
            self._last_match_type = "exact"
            # Deep copy: cache objects have mutable offset/state that
            # generation modifies in-place, corrupting the stored entry.
            cache_out = copy.deepcopy(entry.cache)
            cache_out = self._decompress_cache(cache_out)
            return cache_out, []

        # --- O(log N) prefix & supersequence match via sorted index ---
        best_match: _CacheEntry | None = None
        best_length = 0
        best_super: _CacheEntry | None = None

        # R15-P1: radix fast-path. The trie walk gives us the longest
        # stored prefix of ``tokens`` in O(prefix_len) — strictly faster
        # than the bisect+backscan for any non-trivial cache. We only use
        # the result for the prefix match; the supersequence and LCP
        # paths still run through the sorted index (a supersequence means
        # "a stored entry LONGER than my query that starts with my
        # query", which the radix's "stop at terminal" semantics don't
        # produce — that path is a separate traversal we can ship in a
        # follow-up if profiling shows it dominates). When the prefix
        # match here resolves, we skip the entire backwards bisect scan
        # below, which is where the 3-5× lookup-latency win comes from
        # on shared-system-prompt workloads.
        if self._radix_index is not None:
            try:
                matched_tokens, matched_key = self._radix_index.longest_prefix(
                    tokens_key
                )
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning(f"[radix] longest_prefix failed: {exc}")
                matched_key = None
            if matched_key is not None and matched_key in self._entries:
                # An exact match would have been caught by the O(1) dict
                # lookup above, so any terminal we find here is strictly
                # shorter than the query — i.e. a prefix match.
                best_match = self._entries[matched_key]
                best_length = len(matched_key)

        # Skip the backwards-scan for prefix matches when the radix
        # already returned an answer — the radix is exact for the longest-
        # stored-prefix question, so anything the bisect would find would
        # be a redundant (and strictly equal-or-shorter) match.
        radix_resolved_prefix = self._radix_index is not None and best_match is not None

        sorted_keys = self._sorted_keys
        if sorted_keys and not radix_resolved_prefix:
            # Find insertion point for tokens_key in the sorted list.
            # Keys that are prefixes of tokens_key or supersequences will be
            # clustered around this position due to lexicographic ordering.
            idx = bisect.bisect_left(sorted_keys, tokens_key)

            # Scan backwards from idx to find cached keys that are PREFIXES
            # of tokens_key (shorter cached sequences).  A prefix P of T
            # satisfies P <= T lexicographically, so P is at idx-1 or earlier.
            for i in range(idx - 1, -1, -1):
                cached_key = sorted_keys[i]
                cached_len = len(cached_key)
                if cached_len >= len(tokens_key):
                    continue  # Not a prefix (same length or longer)
                # Check if cached_key is a prefix of tokens_key
                if tokens_key[:cached_len] == cached_key:
                    if cached_len > best_length:
                        best_match = self._entries[cached_key]
                        best_length = cached_len
                    # Found best prefix — shorter entries can't be longer
                    break
                # Once we go past the prefix range, stop
                if cached_key[0] != tokens_key[0]:
                    break
        if sorted_keys:
            idx = bisect.bisect_left(sorted_keys, tokens_key)

            # Scan forward from idx to find cached keys that are SUPERSEQUENCES
            # of tokens_key (longer cached sequences starting with tokens_key).
            for i in range(idx, len(sorted_keys)):
                cached_key = sorted_keys[i]
                cached_len = len(cached_key)
                if cached_len < len(tokens_key):
                    continue
                # Check if tokens_key is a prefix of cached_key
                if cached_key[: len(tokens_key)] == tokens_key:
                    if best_super is None or cached_len > len(best_super.tokens):
                        best_super = self._entries[cached_key]
                else:
                    # Past the supersequence range
                    break

        # --- Supersequence match handling ---
        if best_super is not None:
            n_cached = len(best_super.tokens)
            n_requested = len(tokens)
            excess = n_cached - n_requested

            has_non_trimmable = any(
                not (
                    lc.is_trimmable()
                    if hasattr(lc, "is_trimmable")
                    else hasattr(lc, "trim")
                )
                for lc in best_super.cache
            )

            if excess > 0 and has_non_trimmable:
                logger.debug(
                    "[cache_fetch] supersequence match skipped: "
                    "non-trimmable cache layers (hybrid model)"
                )
            elif excess > 0:
                trimmed_cache = _trim_cache_offset(best_super.cache, excess)
                self._entries.move_to_end(best_super.tokens)
                self._stats.hits += 1
                self._stats.tokens_saved += n_requested
                self._last_match_type = "supersequence"
                trimmed_cache = self._decompress_cache(trimmed_cache)
                return trimmed_cache, []
            else:
                self._entries.move_to_end(best_super.tokens)
                self._stats.hits += 1
                self._stats.tokens_saved += n_requested
                self._last_match_type = "supersequence"
                cache_out = copy.deepcopy(best_super.cache)
                cache_out = self._decompress_cache(cache_out)
                return cache_out, []

        # --- Prefix match ---
        if best_match is not None:
            self._entries.move_to_end(best_match.tokens)
            self._stats.hits += 1
            self._stats.tokens_saved += best_length
            remaining = tokens[best_length:]
            self._last_match_type = "prefix"
            cache_out = copy.deepcopy(best_match.cache)
            cache_out = self._decompress_cache(cache_out)
            return cache_out, remaining

        # --- LCP (Longest Common Prefix) for divergent sequences ---
        # This handles the agentic pattern: same system+context prefix
        # but different final user message.  Use the sorted index to find
        # the nearest neighbor which likely shares the longest prefix.
        best_lcp_entry: _CacheEntry | None = None
        best_lcp_length = 0

        if sorted_keys:
            idx = bisect.bisect_left(sorted_keys, tokens_key)
            # Check neighbors around insertion point (they share the most
            # common prefix due to lexicographic ordering).
            for i in (idx - 1, idx):
                if i < 0 or i >= len(sorted_keys):
                    continue
                cached_key = sorted_keys[i]
                if cached_key == tokens_key:
                    continue  # Skip exact (already handled)
                min_len = min(len(cached_key), len(tokens_key))
                if min_len <= best_lcp_length:
                    continue
                # Compute LCP length
                lcp = 0
                for j in range(min_len):
                    if cached_key[j] != tokens_key[j]:
                        break
                    lcp = j + 1
                if lcp > best_lcp_length:
                    best_lcp_entry = self._entries[cached_key]
                    best_lcp_length = lcp
                    logger.debug(
                        f"[cache_fetch] LCP scan: cached_len={len(cached_key)} "
                        f"req_len={len(tokens_key)} lcp={lcp}"
                    )

        if best_lcp_entry is not None and best_lcp_length > 0:
            excess = len(best_lcp_entry.tokens) - best_lcp_length

            has_non_trimmable = any(
                not (
                    lc.is_trimmable()
                    if hasattr(lc, "is_trimmable")
                    else hasattr(lc, "trim")
                )
                for lc in best_lcp_entry.cache
            )
            logger.debug(
                f"[cache_fetch] LCP candidate: lcp={best_lcp_length} "
                f"entry_len={len(best_lcp_entry.tokens)} excess={excess} "
                f"non_trimmable={has_non_trimmable} "
                f"cache_layers={len(best_lcp_entry.cache)} "
                f"layer_types={[type(lc).__name__ for lc in best_lcp_entry.cache[:3]]}"
            )

            if not has_non_trimmable:
                trimmed_cache = _trim_cache_offset(best_lcp_entry.cache, excess)
                self._entries.move_to_end(best_lcp_entry.tokens)
                self._stats.hits += 1
                self._stats.tokens_saved += best_lcp_length
                remaining = tokens[best_lcp_length:]
                logger.debug(
                    f"[cache_fetch] LCP hit: shared={best_lcp_length} "
                    f"trimmed={excess} remaining={len(remaining)}"
                )
                self._last_match_type = "lcp"
                trimmed_cache = self._decompress_cache(trimmed_cache)
                return trimmed_cache, remaining

        self._stats.misses += 1
        self._last_match_type = "miss"

        return None, tokens

    def store(
        self, tokens: list[int], cache: list[Any], evict_prefixes: bool = True
    ) -> bool:
        """
        Store KV cache for future reuse.

        This method stores the cache reference directly (no copy) and
        tracks memory usage. If memory limit is exceeded, LRU entries
        are evicted until there's room.

        Args:
            tokens: Token sequence that was processed.
            cache: The computed KV cache to store.
            evict_prefixes: If True, evict existing entries whose token
                sequence is a strict prefix of ``tokens``.  Set to False
                when storing prompt+output entries to preserve prompt-only
                entries created by prompt_cache_save (those are the entries
                that future requests will actually match).

        Returns:
            True if stored successfully, False if rejected.
        """
        if not tokens or not cache:
            return False

        tokens_key = tuple(tokens)

        # Fast path: already cached — bump LRU and skip expensive trim/quantize.
        # Holds the lock briefly so the bump is consistent with concurrent fetch.
        with self._lock:
            if tokens_key in self._entries:
                self._entries.move_to_end(tokens_key)
                return True

        # Trim oversized KV arrays to actual used size (pure compute, no shared
        # state — kept outside the lock so concurrent fetch isn't blocked).
        cache = _trim_to_offset(cache)

        # Compress cache for storage (TurboQuant or standard quantization)
        if (
            self._config.kv_turboquant
            and len(tokens) >= self._config.kv_min_quantize_tokens
        ):
            cache = _turboquant_compress_cache(
                cache,
                self._config.kv_turboquant_bits,
                self._config.kv_turboquant_group_size,
                self._config.kv_turboquant_mode,
            )
        elif (
            self._config.kv_quantize
            and len(tokens) >= self._config.kv_min_quantize_tokens
        ):
            cache = _quantize_cache(
                cache, self._config.kv_bits, self._config.kv_group_size
            )

        # Create entry and estimate memory (pure compute, no shared state).
        entry = _CacheEntry.create(tokens, cache)

        # Check if single entry exceeds limit
        if entry.memory_bytes > self._max_memory:
            logger.warning(
                f"Cache entry too large: {entry.memory_bytes / _BYTES_PER_MB:.1f}MB "
                f"exceeds limit {self._max_memory / _BYTES_PER_MB:.1f}MB"
            )
            return False

        with self._lock:
            # Re-check exact match: a concurrent store may have inserted
            # the same key while we were trimming/compressing outside the
            # lock. Just bump LRU and bail.
            if tokens_key in self._entries:
                self._entries.move_to_end(tokens_key)
                return True

            # Prefix-subset eviction: remove entries whose token sequence
            # is a strict prefix of the new entry.  Uses sorted index for
            # O(log N + K) lookup instead of O(N) scan.
            if evict_prefixes and self._sorted_keys:
                to_remove = []
                idx = bisect.bisect_left(self._sorted_keys, tokens_key)
                # Scan backwards — prefixes of tokens_key are immediately before idx
                for i in range(idx - 1, -1, -1):
                    key = self._sorted_keys[i]
                    klen = len(key)
                    if klen >= len(tokens_key):
                        continue
                    if tokens_key[:klen] == key:
                        to_remove.append(key)
                    elif key[0] != tokens_key[0]:
                        break
                for key in to_remove:
                    # Remove from sorted index FIRST so a concurrent fetch
                    # never sees a key in the index that's missing from
                    # _entries (was the source of issue #163's KeyError
                    # under the higher store() rate from PR #165).
                    self._remove_from_sorted(key)
                    if self._radix_index is not None:
                        try:
                            self._radix_index.remove(key)
                        except Exception as exc:  # pragma: no cover
                            logger.warning(
                                f"[radix] remove failed for {len(key)} tokens: {exc}"
                            )
                    old = self._entries.pop(key)
                    self._current_memory -= old.memory_bytes
                    self._stats.evictions += 1
                    logger.debug(
                        f"[prefix_evict] removed {len(key)} tokens, "
                        f"freed {old.memory_bytes / _BYTES_PER_MB:.2f}MB, "
                        f"new_entry={len(tokens_key)} tokens"
                    )
                if to_remove:
                    self._stats.entry_count = len(self._entries)
                    self._stats.current_memory_bytes = self._current_memory

            # Evict until we have room
            while (
                self._current_memory + entry.memory_bytes > self._max_memory
                or len(self._entries) >= self._config.max_entries
            ) and self._entries:
                self._evict_lru()

            # Store entry. Insert into _entries before _sorted_keys so
            # that even if a future change drops the lock, fetch never
            # observes a key in sorted_keys that's missing from entries.
            self._entries[tokens_key] = entry
            self._current_memory += entry.memory_bytes
            bisect.insort(self._sorted_keys, tokens_key)
            self._stats.entry_count = len(self._entries)
            self._stats.current_memory_bytes = self._current_memory
            # R15-P1: keep the radix index in sync. Insert happens INSIDE
            # the cache lock so the radix and ``_entries`` are observed
            # coherently from concurrent fetchers. Skips silently if the
            # radix is not wired (``hash`` mode) — the bisect path stays
            # the source of truth in that case.
            if self._radix_index is not None:
                try:
                    self._radix_index.insert(tokens_key)
                except Exception as exc:  # pragma: no cover — defensive
                    logger.warning(
                        f"[radix] insert failed for {len(tokens_key)} tokens: {exc}"
                    )

        logger.debug(
            f"Stored cache: {len(tokens)} tokens, "
            f"{entry.memory_bytes / _BYTES_PER_MB:.2f}MB, "
            f"total={self._current_memory / _BYTES_PER_MB:.1f}MB"
        )

        return True

    def _remove_from_sorted(self, key: tuple[int, ...]) -> None:
        """Remove a key from the sorted index using bisect for O(log N)."""
        idx = bisect.bisect_left(self._sorted_keys, key)
        if idx < len(self._sorted_keys) and self._sorted_keys[idx] == key:
            self._sorted_keys.pop(idx)

    def _evict_lru(self) -> None:
        """Evict the least recently used entry.

        Caller must hold ``self._lock``.
        """
        if not self._entries:
            return

        # Peek the oldest key, drop sorted-index entry first so a fetch
        # without the lock can't trip the orphaned-sorted-key KeyError.
        tokens_key = next(iter(self._entries))
        self._remove_from_sorted(tokens_key)
        if self._radix_index is not None:
            try:
                self._radix_index.remove(tokens_key)
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning(
                    f"[radix] lru-evict remove failed for {len(tokens_key)} tokens: {exc}"
                )
        entry = self._entries.pop(tokens_key)
        self._current_memory -= entry.memory_bytes
        self._stats.evictions += 1
        self._stats.entry_count = len(self._entries)
        self._stats.current_memory_bytes = self._current_memory

        logger.debug(
            f"[lru_evict] removed {len(tokens_key)} tokens, "
            f"freed {entry.memory_bytes / _BYTES_PER_MB:.2f}MB"
        )

    def remove(self, tokens: list[int]) -> bool:
        """
        Remove a specific cache entry.

        Args:
            tokens: Token sequence to remove.

        Returns:
            True if entry was found and removed.
        """
        tokens_key = tuple(tokens)
        with self._lock:
            if tokens_key not in self._entries:
                return False
            self._remove_from_sorted(tokens_key)
            if self._radix_index is not None:
                try:
                    self._radix_index.remove(tokens_key)
                except Exception as exc:  # pragma: no cover — defensive
                    logger.warning(
                        f"[radix] explicit-remove failed for {len(tokens_key)} tokens: {exc}"
                    )
            entry = self._entries.pop(tokens_key)
            self._current_memory -= entry.memory_bytes
            self._stats.entry_count = len(self._entries)
            self._stats.current_memory_bytes = self._current_memory
        return True

    def clear(self) -> None:
        """Clear all cached entries.

        R10-D codex round 2 HIGH: ``load_skipped`` is cumulative —
        it backs the ``rapid_mlx_prefix_cache_load_skipped_total``
        Prometheus counter, which by contract must never decrease.
        The metrics route's sticky accumulator catches a process-
        global reset, but if ``clear()`` runs BETWEEN two scrapes
        the value the route sees drops and the accumulator pins
        the reset to its own monotonic floor — losing the skip
        delta. Carry it over here so the in-process counter never
        regresses either.

        R12-T1: the same carry-over applies to ``save_drift_drops``
        — it backs the ``rapid_mlx_prefix_cache_save_drift_drops_total``
        Prometheus counter and must be monotonic.
        """
        with self._lock:
            self._entries.clear()
            self._sorted_keys.clear()
            if self._radix_index is not None:
                try:
                    self._radix_index.clear()
                except Exception as exc:  # pragma: no cover — defensive
                    logger.warning(f"[radix] clear failed: {exc}")
            self._current_memory = 0
            carried_load_skipped = self._stats.load_skipped
            carried_save_drift_drops = self._stats.save_drift_drops
            self._stats = CacheStats(
                max_memory_bytes=self._max_memory,
                load_skipped=carried_load_skipped,
                save_drift_drops=carried_save_drift_drops,
            )
        logger.debug("Cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        When a radix index is attached, its counters are nested under
        ``radix`` so the /metrics route can surface them as
        ``rapid_mlx_prefix_cache_radix_*`` without colliding with the
        legacy cache counters.
        """
        out = self._stats.to_dict()
        if self._radix_index is not None:
            try:
                out["radix"] = self._radix_index.stats()
            except Exception as exc:  # pragma: no cover — defensive
                logger.debug(f"[radix] stats() failed: {exc}")
        return out

    def reset_stats(self) -> None:
        """Reset statistics while preserving cache contents.

        R10-D codex round 2 HIGH: same monotonic-counter rationale as
        ``clear`` — ``load_skipped`` must carry across a stats reset
        so the Prometheus counter never loses a delta.

        R12-T1: ``save_drift_drops`` likewise must carry across so its
        backing Prometheus counter never regresses.
        """
        self._stats = CacheStats(
            max_memory_bytes=self._max_memory,
            current_memory_bytes=self._current_memory,
            entry_count=len(self._entries),
            load_skipped=self._stats.load_skipped,
            save_drift_drops=self._stats.save_drift_drops,
        )

    @property
    def memory_usage_mb(self) -> float:
        """Current memory usage in MB."""
        return self._current_memory / _BYTES_PER_MB

    @property
    def memory_limit_mb(self) -> float:
        """Memory limit in MB."""
        return self._max_memory / _BYTES_PER_MB

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._entries)

    def __contains__(self, tokens: list[int]) -> bool:
        """Check if tokens are cached."""
        return tuple(tokens) in self._entries

    # -----------------------------------------------------------------
    # Disk persistence — survives server restarts
    # -----------------------------------------------------------------

    def save_to_disk(
        self,
        cache_dir: str,
        should_abort=None,
    ) -> bool:
        """Save all cache entries to disk using mlx_lm's safetensors format.

        The snapshot is committed via a directory-rename to make it
        all-or-nothing: writes go to ``<cache_dir>.new/``, then a
        three-step swap (``cache_dir → .old``, ``.new → cache_dir``,
        ``rm .old``) atomically replaces the previous snapshot. A crash
        anywhere during the writes leaves the previous snapshot intact;
        :meth:`load_from_disk` recovers from a crash mid-swap.

        Directory layout (committed)::

            cache_dir/
              index.json          # token keys + metadata per entry
              entry_0.safetensors # KV arrays for entry 0
              entry_0_tokens.bin
              entry_1.safetensors
              entry_1_tokens.bin
              ...

        Args:
            cache_dir: Final committed directory path (``.new`` / ``.old``
                staging dirs are siblings).
            should_abort: Optional ``Callable[[float], bool]`` that returns
                True when the caller wants the save loop to stop early. The
                ``float`` arg is ``predicted_sec`` — the per-entry loop's
                estimate of how long the NEXT entry's write will take, so
                the predicate can answer "would starting that operation
                push us past the deadline?" rather than only firing AFTER
                wall-clock has already crossed it (codex PR #667 round 1
                BLOCKING-2 — a single uninterruptible 300 MB
                ``save_prompt_cache`` call can straddle the deadline and
                still get SIGKILL'd mid-write if the check is at-now only).

                Used by the lifespan shutdown to enforce a SIGTERM-grace
                deadline so a multi-GB save doesn't get SIGKILLed mid-
                flight and leave ``cache_dir.new/`` orphaned (rapid-
                desktop only gives the sidecar ~5s before SIGKILL). When
                the callable trips, the loop stops, the entries that did
                finish are verified, and the staging dir is committed via
                the same atomic rename as a normal save — so a partial
                result is preferable to the previous behavior (truncated
                mid-entry → orphaned ``.new`` → lost cache on next
                launch).

                Backwards-compatible: a zero-arg ``Callable[[], bool]``
                (the round-1 documented shape) is auto-adapted via
                ``_adapt_should_abort`` so external callers / older
                fixtures don't break — see codex round 3 BLOCKING-2.
                A ``None`` value preserves the pre-existing "save
                everything, no deadline" behavior used by tests and the
                offline ``rapid-mlx`` CLI.

        Returns True if at least one entry was committed to disk.
        """
        import shutil
        import time as _time

        if not self._entries:
            logger.info("[cache_persist] nothing to save (0 entries)")
            return False

        t0 = _time.monotonic()

        try:
            from mlx_lm.models.cache import save_prompt_cache
        except ImportError:
            logger.warning("[cache_persist] mlx_lm not available, cannot save")
            return False

        # Strip trailing separators so ``<cache_dir>.new`` is a sibling of
        # cache_dir, not a child. A child path silently breaks the swap.
        cache_dir = cache_dir.rstrip(os.sep)
        new_dir = cache_dir + ".new"
        old_dir = cache_dir + ".old"

        # Pre-clean stale staging dirs from a previous interrupted save.
        #
        # R12-T1 (dogfood-0815 Talia r12): the old pre-clean used
        # ``ignore_errors=True`` blindly — a partial rmtree (file locked
        # by an external process, EACCES under a chmod, ENOTEMPTY race
        # with a concurrent reader) would silently leave stale entry
        # files in ``.new``, and the subsequent ``os.makedirs(...,
        # exist_ok=True)`` would adopt them. Those orphan files then
        # ride the atomic rename into ``cache_dir`` alongside the
        # current save's writes — producing the (uuid_A tokens.bin,
        # uuid_B index.json) mismatch Talia r12 caught. Belt-and-
        # suspenders: keep ``ignore_errors=True`` for the rmtree (so
        # one bad file doesn't kill the whole save) but VERIFY the
        # post-rmtree state with ``os.listdir`` and surface a hard
        # error if anything survived. The post-write self-verify pass
        # below would catch any survivor that did manage to ride along,
        # but failing fast here gives a structured signal directly
        # tied to the mechanism rather than a "drift drop" downstream.
        for stale in (new_dir, old_dir):
            if not os.path.exists(stale):
                continue
            logger.info(f"[cache_persist] removing stale staging dir: {stale}")
            shutil.rmtree(stale, ignore_errors=True)
            # If anything survived, escalate. We don't try harder than
            # rmtree did — a stuck file means an external process has it
            # open / locked, which a retry won't fix and a second rmtree
            # call could merely race with whatever holds it. Better to
            # ABORT the save (cache_dir keeps the previous good snapshot)
            # than commit a snapshot that we know contains foreign files.
            if os.path.exists(stale):
                try:
                    survivors = os.listdir(stale)
                except OSError:
                    survivors = ["<listdir failed>"]
                logger.warning(
                    f"[cache_persist] R12-T1 pre-clean could not fully remove "
                    f"{stale}; {len(survivors)} entries survived "
                    f"(first 5: {survivors[:5]}); aborting save to keep "
                    f"cache_dir consistent — operator should investigate "
                    f"and remove the staging dir manually"
                )
                return False

        os.makedirs(new_dir, exist_ok=True)

        # Single source of truth for per-entry on-disk filenames. Used
        # by both the save loop and the post-loop "did the files
        # actually survive?" filter — keep them in lockstep so a future
        # rename only has one place to edit.
        def _entry_paths(idx: int) -> tuple[str, str]:
            return (
                os.path.join(new_dir, f"entry_{idx}.safetensors"),
                os.path.join(new_dir, f"entry_{idx}_tokens.bin"),
            )

        # R10-D: stamp this save with a fresh uuid so the loader can
        # detect orphans from a previous cycle. uuid4 is 128-bit, hex
        # form is 32 ASCII chars — short enough to grep through
        # snapshots, wide enough that two saves never collide. Embedded
        # in BOTH index.json (file level) AND each tokens.bin (per
        # entry) — see _write_tokens_bin_v3.
        import uuid as _uuid

        save_uuid = _uuid.uuid4().hex
        index = {
            "version": _TOKENS_FORMAT_VERSION_IN_INDEX,
            "save_uuid": save_uuid,
            "num_entries": len(self._entries),
            "total_memory_bytes": self._current_memory,
            "entries": [],
        }

        saved = 0
        aborted_early = False
        total_entries = len(self._entries)
        # Track observed disk throughput so we can predict whether the
        # NEXT entry's write will fit within the shutdown budget. The
        # predicate fires forward-looking — if we don't predict, a
        # single in-flight ``save_prompt_cache`` call can run past the
        # deadline and get SIGKILL'd mid-write (leaves ``cache_dir.new/``
        # orphaned; this is the bug the deadline gate exists to prevent).
        #
        # Bootstrap floor for entry 0 (no observed sample yet) is
        # 150 MB/s — calibrated so:
        #   - typical Gemma 4 26B entry (~250 MB) predicts ~1.7 s,
        #     comfortably fitting the 3.1 s safe budget (3.5 s budget −
        #     0.4 s commit headroom);
        #   - genuinely oversized entry (~600 MB+) predicts >4 s and
        #     correctly trips before write starts — would straddle
        #     deadline either way.
        # Round 1 used 50 MB/s and over-predicted typical entries
        # (codex round 2 BLOCKING-1). Round 2 used 0 and let huge
        # entries straddle the deadline (codex round 3 BLOCKING-1).
        # 150 MB/s is the goldilocks middle ground: 3× round 1, gives
        # real-world observed throughput (~875 MB/s during the
        # original incident) ~6× safety margin while still catching
        # genuinely-too-large entries.
        _BOOTSTRAP_BYTES_PER_SEC = 150 * _BYTES_PER_MB
        # Support BOTH zero-arg and one-arg ``should_abort`` predicates
        # at the per-entry layer. The new contract is
        # ``Callable[[float], bool]`` (forward-looking) but external
        # callers may still pass a zero-arg shape from the round 1
        # docstring contract — auto-detect and adapt instead of
        # raising TypeError. Codex PR #667 round 3 BLOCKING-2.
        check_abort = _adapt_should_abort(should_abort)
        total_bytes_written = 0
        total_write_seconds = 0.0
        for i, (tokens_key, entry) in enumerate(self._entries.items()):
            if total_write_seconds > 0:
                observed_bps = total_bytes_written / total_write_seconds
            else:
                observed_bps = _BOOTSTRAP_BYTES_PER_SEC
            predicted_sec = entry.memory_bytes / observed_bps
            # Deadline-aware early exit: the lifespan handler installs a
            # ``should_abort`` predicate driven by the SIGTERM-grace budget.
            # Once it trips we stop persisting NEW entries but still run
            # the verify + index + atomic-rename steps below so the
            # partial snapshot we already have on disk gets COMMITTED
            # rather than left in ``cache_dir.new/``. ``saved >= 1`` is
            # the gate that controls whether the rename happens —
            # nothing else changes from the full-flush path.
            if check_abort is not None and check_abort(predicted_sec):
                aborted_early = True
                bps_label = "observed" if total_write_seconds > 0 else "bootstrap floor"
                logger.warning(
                    f"[cache_persist] shutdown budget would not fit "
                    f"entry {i}/{total_entries} "
                    f"(predicted {predicted_sec * 1000:.0f}ms write at "
                    f"{observed_bps / _BYTES_PER_MB:.0f}MB/s "
                    f"[{bps_label}]) — committing {saved} entries that "
                    f"finished before deadline"
                )
                break
            entry_path, tokens_path = _entry_paths(i)
            entry_t0 = _time.monotonic()
            try:
                # Dequantize QuantizedKVCache layers before saving.
                # save_prompt_cache requires .state and .meta_state which
                # the wrapper does not provide; dequantizing restores the
                # original cache types that do.
                from mlx_lm.models.cache import QuantizedKVCache

                persist_cache = (
                    _dequantize_cache(entry.cache)
                    if any(isinstance(c, QuantizedKVCache) for c in entry.cache)
                    else entry.cache
                )
                save_prompt_cache(
                    entry_path,
                    persist_cache,
                    metadata={"num_tokens": str(len(tokens_key))},
                )
                # R8-M7 codex r1 BLOCKING #3: durably commit the
                # safetensors file. ``mx.save_safetensors`` (which
                # ``save_prompt_cache`` calls) writes through to the
                # kernel page cache but does not fsync — under
                # SIGTERM-driven shutdown the body can still be in
                # cache when the dir rename publishes its name,
                # leaving a renamed entry with empty/partial contents
                # on a hard reset / power loss. Open + fsync after
                # the write so the file body hits durable storage
                # BEFORE the rename. Open errors are non-fatal —
                # they may indicate the file already vanished, and
                # the subsequent verify loop already catches that.
                try:
                    _fsync_file(entry_path)
                except OSError as fs_err:
                    logger.debug(
                        f"[cache_persist] fsync({entry_path}) failed: {fs_err}; "
                        "continuing — verify-loop will catch real file loss"
                    )
                # R10-D: write tokens.bin via the v3 helper — magic +
                # length prefix + save_uuid + int32 LE payload. Every
                # branch above (fsync, dir-fsync, size cross-check) keeps
                # working unchanged because the helper writes a
                # self-describing file whose total length is deterministic
                # given (token_count, uuid_len). Width pinned to 4 bytes
                # per token regardless of host ``array.array("i").itemsize``
                # — wire format must be portable.
                _write_tokens_bin_v3(tokens_path, list(tokens_key), save_uuid)

                # Record the per-layer cache class names so loaders can
                # gate on cache-type compatibility (#198 BUG B). Read from
                # ``persist_cache`` (post-dequantize), not ``entry.cache``,
                # so the index reflects what's actually on disk — otherwise
                # a saved-while-quantized entry would be rejected on a
                # subsequent unquantized startup despite being loadable.
                cache_types = [
                    type(layer).__name__ for layer in persist_cache if layer is not None
                ]

                index["entries"].append(
                    {
                        "index": i,
                        "num_tokens": len(tokens_key),
                        "memory_bytes": entry.memory_bytes,
                        "cache_types": cache_types,
                    }
                )
                saved += 1
                # Feed the throughput estimator. We measure including
                # both the safetensors write and the tokens sidecar so
                # the next entry's prediction reflects the full per-
                # entry cost, not just the KV blob.
                elapsed = _time.monotonic() - entry_t0
                if elapsed > 0:
                    total_bytes_written += entry.memory_bytes
                    total_write_seconds += elapsed
                logger.info(
                    f"[cache_persist] saved entry {i}: "
                    f"{len(tokens_key)} tokens, "
                    f"{entry.memory_bytes / _BYTES_PER_MB:.1f}MB KV, "
                    f"file={entry_path}"
                )
            except Exception as e:
                logger.warning(f"[cache_persist] failed to save entry {i}: {e}")

        if saved == 0:
            shutil.rmtree(new_dir, ignore_errors=True)
            logger.warning("[cache_persist] no entries saved successfully, aborting")
            return False

        # R12-T1 (dogfood-0815 Talia r12 SEVERE): post-write self-verify.
        # The save_uuid + length-prefix invariants this writer claims must
        # hold on the just-written ``.new/`` snapshot BEFORE we publish it
        # via the atomic rename. Talia r12 caught a deterministic 2-cycle-
        # SIGTERM repro where cache_dir ended up with index.json from save B
        # but several entry_K_tokens.bin files carrying save A's uuid and
        # length-prefix — the loader's R10-D integrity guard refused to
        # load the whole 100-entry / 2.6 GB snapshot the next boot. The
        # mechanism (concurrent writers, mmap coherence window, fs-event
        # external clobber, partial pre-clean of a stale ``.new``, …) is
        # noisy in production but the consequence is invariant: a tokens.bin
        # in our staging dir that disagrees with what we just claimed in
        # index.json. Make THIS check the source of truth so the corrupt
        # state cannot survive into ``cache_dir`` regardless of mechanism.
        #
        # Three passes, cheapest first:
        #   1. ``_both_exist`` — the legacy filter for staging-dir clobber
        #      under disk pressure / Spotlight. Preserved verbatim.
        #   2. Header-only verify — read each tokens.bin's fixed prefix
        #      + uuid (≤80 bytes per entry; cheap even at 100 entries)
        #      and confirm (save_uuid, token_count) match what we'd write
        #      into index.json for that entry. Mismatches drop the entry
        #      and bump a per-save metric so the operator sees the rate.
        #   3. Orphan-file sweep — any ``entry_*`` file in ``.new`` that
        #      isn't covered by the (now-filtered) index.json gets removed
        #      so the committed dir contains exactly what the index claims.
        #
        # If pass (2) drops every entry the save aborts — we'd otherwise
        # commit an empty index that would unnecessarily clobber the
        # known-good ``cache_dir``.
        def _both_exist(e: dict) -> bool:
            sf, tk = _entry_paths(e["index"])
            return os.path.exists(sf) and os.path.exists(tk)

        existing = [e for e in index["entries"] if _both_exist(e)]
        if not existing:
            shutil.rmtree(new_dir, ignore_errors=True)
            logger.warning(
                "[cache_persist] staging dir vanished mid-save, no entries survived "
                f"(saved {saved}/{len(self._entries)} but 0 files remain on disk)"
            )
            return False
        if len(existing) < len(index["entries"]):
            logger.warning(
                f"[cache_persist] {len(index['entries']) - len(existing)} of "
                f"{len(index['entries'])} entry files vanished mid-save, "
                f"persisting {len(existing)} that survived"
            )

        # Pass 2 — header-only self-verify. Catches the R12-T1 drift class
        # regardless of mechanism: if the file we ASSUMED we wrote does not
        # carry (save_uuid, token_count) we declared for it, drop the entry
        # rather than commit a snapshot that the loader will refuse en bloc.
        verified: list[dict] = []
        drift_drops = 0
        for e in existing:
            _, tk = _entry_paths(e["index"])
            on_disk_count, on_disk_uuid, reject_reason = _peek_tokens_bin_header(tk)
            if reject_reason:
                drift_drops += 1
                logger.warning(
                    f"[cache_persist] R12-T1 post-write self-verify dropped "
                    f"entry {e['index']}: {reject_reason}"
                )
                continue
            if on_disk_uuid != save_uuid:
                drift_drops += 1
                logger.warning(
                    f"[cache_persist] R12-T1 post-write self-verify dropped "
                    f"entry {e['index']}: tokens.bin save_uuid "
                    f"{on_disk_uuid!r} != current save {save_uuid!r}"
                )
                continue
            if on_disk_count != e["num_tokens"]:
                drift_drops += 1
                logger.warning(
                    f"[cache_persist] R12-T1 post-write self-verify dropped "
                    f"entry {e['index']}: tokens.bin length-prefix "
                    f"{on_disk_count} != index num_tokens {e['num_tokens']}"
                )
                continue
            verified.append(e)
        if drift_drops:
            # Sticky in-process counter so /metrics can surface "% of
            # entries dropped by post-write verify per save" without
            # parsing logs. Mirrors the load_skipped contract for the
            # save side.
            self._stats.save_drift_drops += drift_drops
        if not verified:
            shutil.rmtree(new_dir, ignore_errors=True)
            logger.warning(
                f"[cache_persist] R12-T1 post-write verify rejected ALL "
                f"{len(existing)} entries (save_uuid / length-prefix drift); "
                f"aborting save so cache_dir keeps the previous good snapshot"
            )
            return False
        if len(verified) < len(existing):
            logger.warning(
                f"[cache_persist] R12-T1 post-write verify dropped "
                f"{len(existing) - len(verified)} of {len(existing)} entries "
                f"(save_uuid / length-prefix drift); committing "
                f"{len(verified)} that round-tripped cleanly"
            )

        index["entries"] = verified
        # Always pin num_entries to the actually-verified count. The initial
        # value was ``total_entries`` (set before the save loop) which is
        # wrong both when we aborted early AND when some entry files
        # vanished mid-save — index.json must agree with the entry list it
        # ships alongside, or load_from_disk's ``num_entries`` read drifts
        # from reality and downstream callers report a phantom count.
        index["num_entries"] = len(index["entries"])

        # Pass 3 — orphan-file sweep. The atomic rename publishes the
        # entire ``.new/`` directory tree, so any file we wrote but the
        # post-verify filter dropped must be physically removed from the
        # staging dir BEFORE the rename. Otherwise a subsequent recovery
        # path could still match it via ``entry_<index>_tokens.bin``
        # naming and re-introduce the very drift the verify pass caught.
        # We also catch orphan files left behind by ANY other source
        # (e.g. a previous interrupted save's ``.new`` that survived the
        # pre-clean's ``ignore_errors=True``) — the committed dir must
        # contain exactly { index.json } ∪ { entry_K.safetensors,
        # entry_K_tokens.bin : K ∈ index["entries"] }.
        keep_paths = {os.path.join(new_dir, "index.json")}
        for e in index["entries"]:
            sf, tk = _entry_paths(e["index"])
            keep_paths.add(sf)
            keep_paths.add(tk)
        try:
            for name in os.listdir(new_dir):
                full = os.path.join(new_dir, name)
                if full in keep_paths:
                    continue
                # Only sweep regular files — leave any unexpected dirs
                # alone so we don't recurse into something weird.
                try:
                    if os.path.isfile(full):
                        os.remove(full)
                        logger.info(
                            f"[cache_persist] R12-T1 orphan sweep removed {name}"
                        )
                except OSError as sweep_err:
                    logger.debug(
                        f"[cache_persist] orphan sweep failed on {name}: "
                        f"{sweep_err}; continuing"
                    )
        except OSError as listdir_err:
            # If listdir itself fails the dir is gone — the recheck
            # below catches it and we abort cleanly.
            logger.debug(
                f"[cache_persist] orphan sweep listdir({new_dir}) failed: "
                f"{listdir_err}; deferring to TOCTOU recheck"
            )

        # Defensively recreate new_dir before the index.json write — the
        # filter above proves at least one entry's files exist, so the
        # dir must too, but a stat-cache delay or NFS-style coherence
        # window could still trip the open() below. Cheap insurance.
        os.makedirs(new_dir, exist_ok=True)

        # TOCTOU re-check: between the filter above and the index.json
        # write below, the same external process could clobber new_dir
        # again. If that happens, makedirs recreates an EMPTY dir, and
        # we'd commit an index.json pointing to entry files that no
        # longer exist (load_from_disk's _has_valid_index() would then
        # reject the snapshot — recoverable, but a wasted swap). Verify
        # the first entry still exists right before we write; if not,
        # abort cleanly.
        first_sf, first_tk = _entry_paths(index["entries"][0]["index"])
        if not (os.path.exists(first_sf) and os.path.exists(first_tk)):
            shutil.rmtree(new_dir, ignore_errors=True)
            logger.warning(
                "[cache_persist] staging dir vanished after filter — entry "
                "files gone before index.json could be written, aborting"
            )
            return False

        # Write index.json LAST inside the staging dir. Its presence is the
        # signal to load_from_disk that .new contains a complete snapshot.
        # Catch FileNotFoundError as a final guard against the recheck
        # above missing the dir-loss window — the file or dir could still
        # vanish in the microseconds between the recheck and the open().
        index_path = os.path.join(new_dir, "index.json")
        try:
            with open(index_path, "w") as f:
                json.dump(index, f, indent=2)
                # R8-M7 codex r1 BLOCKING #3: fsync the file fd so its
                # contents (not just metadata) hit durable storage before
                # the rename commits. Without this, the rename can
                # publish a name that points at an empty/partial file
                # because the page cache hasn't flushed yet. The dir
                # fsync below covers directory-entry durability; this
                # fsync covers the file body itself.
                f.flush()
                os.fsync(f.fileno())
        except OSError as e:
            # Catch the broader OSError (FileNotFoundError if dir vanished,
            # PermissionError if cache_dir was suddenly chmod'd, ENOSPC if
            # disk filled up mid-shutdown). All of these should log
            # cleanly, not raise a traceback up to the lifespan handler.
            shutil.rmtree(new_dir, ignore_errors=True)
            logger.warning(
                f"[cache_persist] could not write index.json ({e}), aborting"
            )
            return False

        # R8-M7 (dogfood-089 Talia r1/r2): the commit phase is the
        # narrow window where a SIGTERM-driven shutdown can leave
        # ``cache_dir`` absent + ``.new/`` orphaned + load_from_disk
        # then fails to recover because the load was never called
        # (e.g. SIGKILL hit before next boot's lifespan, or the next
        # boot raced its own save-to-disk and pre-cleaned ``.new``).
        # Wrap the two-rename swap in a try/except that, on ANY
        # failure mid-commit, attempts a self-recovery promotion of
        # ``.new`` to ``cache_dir`` so the on-disk state is never
        # left in the "cache_dir missing, .new present, .old present"
        # window for longer than this method's own scope. Without
        # this, a transient OSError between the two renames (e.g.
        # PermissionError from a fs-event-driven antivirus touching
        # cache_dir mid-rename, observed on macOS Spotlight rebuilds)
        # would silently lose the just-saved snapshot the next time
        # save_to_disk runs and pre-cleans ``.new`` + ``.old``.
        #
        # The fsync on the staging dir before the rename forces the
        # index.json + entry files' metadata into the journal so the
        # rename commits the right contents. Without it, a kernel
        # crash between the rename and the periodic fs flush can
        # leave the renamed dir referencing not-yet-written contents
        # (observed on Linux ext4 with ``data=writeback``; macOS APFS
        # is more conservative but the fsync is still a correctness
        # invariant). We fsync the dir, not individual files, because
        # the entry-file write already returns from
        # ``mx.save_safetensors`` only after the kernel queues the
        # write — and the dir fsync covers the index.json + the
        # entries' rename-into-place metadata.
        try:
            _fsync_dir(new_dir)
        except OSError as fsync_err:
            # fsync failures on the staging dir are a soft signal —
            # log + proceed. The rename below may still work; if it
            # doesn't, the recovery branch picks up the pieces.
            logger.debug(
                f"[cache_persist] fsync({new_dir}) failed: {fsync_err}; "
                "continuing with rename"
            )

        # Atomic-ish directory swap. If we crash between the two renames,
        # load_from_disk's recovery path (see below) handles it.
        rename_committed = False
        try:
            if os.path.exists(cache_dir):
                os.rename(cache_dir, old_dir)
            os.rename(new_dir, cache_dir)
            rename_committed = True
        except OSError as rename_err:
            # R8-M7: one of the two renames raised. Attempt
            # in-process recovery so the next load_from_disk
            # — which may not happen for hours if the operator
            # doesn't reboot — doesn't have to. Three cases:
            #   * cache_dir absent, .new present (rename 2 failed
            #     before cache_dir was created): retry rename 2.
            #   * cache_dir absent, .old present, .new present
            #     (rename 1 succeeded, rename 2 failed): same retry,
            #     and clean .old if rename 2 then succeeds.
            #   * cache_dir present, .new present (rename 1 failed):
            #     keep cache_dir, drop .new (already-committed
            #     snapshot is the safer choice; the next save
            #     attempt will rebuild .new from current state).
            logger.warning(
                f"[cache_persist] commit-phase rename raised "
                f"({rename_err}); attempting in-process recovery"
            )
            if not os.path.exists(cache_dir) and os.path.exists(new_dir):
                try:
                    os.rename(new_dir, cache_dir)
                    rename_committed = True
                    logger.warning("[cache_persist] recovered: .new -> cache_dir")
                except OSError as retry_err:
                    logger.error(
                        f"[cache_persist] recovery rename failed "
                        f"({retry_err}); cache_dir absent, .new orphan "
                        f"— next load_from_disk will promote .new"
                    )
            elif os.path.exists(cache_dir) and os.path.exists(new_dir):
                # cache_dir survived rename 1 failure; keep it and
                # drop the staging dir so a future save doesn't
                # pre-clean a still-meaningful .new.
                shutil.rmtree(new_dir, ignore_errors=True)
                logger.warning(
                    "[cache_persist] kept existing cache_dir, dropped "
                    "stale .new from failed rename"
                )

        # Drop the now-redundant .old — but ONLY if the new snapshot
        # made it into cache_dir. Codex round-1 BLOCKING: pre-fix this
        # rmtree ran unconditionally, so a commit-phase failure that
        # left ``.new`` orphan + ``.old`` valid would then DESTROY the
        # last known-good snapshot before returning False — silently
        # downgrading "recoverable via load_from_disk" to "no cache
        # at all". Keep ``.old`` when rename did not commit so the
        # standard recovery path (``_has_valid_index(old_dir)``) can
        # restore it on next boot. Errors are non-fatal: next save's
        # pre-clean will catch anything we leave behind.
        if rename_committed and os.path.exists(old_dir):
            shutil.rmtree(old_dir, ignore_errors=True)

        dt = _time.monotonic() - t0
        tail = " (partial — shutdown deadline hit)" if aborted_early else ""
        if rename_committed:
            logger.info(
                f"[cache_persist] SAVED {saved}/{total_entries} entries "
                f"to {cache_dir} in {dt:.1f}s "
                f"({self._current_memory / _BYTES_PER_MB:.0f}MB total){tail}"
            )
        else:
            logger.warning(
                f"[cache_persist] partial commit after {dt:.1f}s — "
                f"{saved}/{total_entries} entries written but rename "
                f"did not complete; load_from_disk recovery required"
            )
        return saved > 0 and rename_committed

    def load_from_disk(self, cache_dir: str) -> int:
        """Load cache entries from disk.

        Recovers from a save interrupted between the two directory
        renames in :meth:`save_to_disk`:

        * if ``cache_dir`` is missing but ``cache_dir.new/index.json``
          exists, the snapshot was fully written but never swapped in
          → promote ``.new`` to ``cache_dir``;
        * else if ``cache_dir.old`` is present and ``cache_dir`` is
          missing, restore ``.old``.

        Each entry is validated before insertion: the on-disk
        ``tokens.bin`` size must match ``num_tokens * 4``, the
        ``.safetensors`` file size must cover the data range declared
        in its header (``mx.load`` mmaps lazily and returns zeros past
        EOF, so a body-truncated KV would otherwise slip through), and
        ``cache.offset`` must equal ``len(tokens)``. Any entry that
        fails validation is dropped with a warning.

        Returns the number of entries successfully loaded.
        """
        import shutil
        import time as _time

        # Strip trailing separators (see save_to_disk for rationale).
        cache_dir = cache_dir.rstrip(os.sep)
        new_dir = cache_dir + ".new"
        old_dir = cache_dir + ".old"

        def _has_valid_index(d: str) -> bool:
            """Cheap sanity check: index.json exists, is valid JSON, has the
            expected version, AND at least one referenced entry file exists
            on disk. The last check defends against the pathological case
            where index.json survives but its entry files don't (manual
            deletion, fs corruption, partial restore from backup) — without
            it, recovery would promote a "valid index, no data" snapshot
            and discard the previous good `.old` snapshot for nothing."""
            p = os.path.join(d, "index.json")
            if not os.path.exists(p):
                return False
            try:
                with open(p) as f:
                    obj = json.load(f)
            except (OSError, ValueError):
                return False
            if not (isinstance(obj, dict) and obj.get("version", 0) >= 2):
                return False
            entries = obj.get("entries") or []
            if not entries:
                # An index claiming zero entries is degenerate; nothing to
                # promote. Treat as missing so recovery can fall through
                # to a real snapshot in the other staging dir.
                return False
            first_idx = entries[0].get("index")
            if first_idx is None:
                return False
            sf = os.path.join(d, f"entry_{first_idx}.safetensors")
            tk = os.path.join(d, f"entry_{first_idx}_tokens.bin")
            return os.path.exists(sf) and os.path.exists(tk)

        # Crash-recovery for an interrupted save_to_disk.
        if not os.path.exists(cache_dir):
            if _has_valid_index(new_dir):
                logger.info(
                    f"[cache_persist] recovering interrupted save: "
                    f"promoting {new_dir} → {cache_dir}"
                )
                os.rename(new_dir, cache_dir)
                if os.path.exists(old_dir):
                    shutil.rmtree(old_dir, ignore_errors=True)
            elif _has_valid_index(old_dir):
                logger.info(
                    f"[cache_persist] recovering interrupted save: "
                    f"restoring {old_dir} → {cache_dir}"
                )
                os.rename(old_dir, cache_dir)
                if os.path.exists(new_dir):
                    shutil.rmtree(new_dir, ignore_errors=True)
        else:
            # cache_dir exists — clean up any orphan staging dirs that a
            # previous interrupted save may have left behind.
            for stale in (new_dir, old_dir):
                if os.path.exists(stale):
                    logger.info(f"[cache_persist] cleaning orphan staging dir: {stale}")
                    shutil.rmtree(stale, ignore_errors=True)

        index_path = os.path.join(cache_dir, "index.json")
        if not os.path.exists(index_path):
            logger.info(f"[cache_persist] no index at {index_path}, nothing to load")
            return 0

        t0 = _time.monotonic()

        try:
            from mlx_lm.models.cache import load_prompt_cache
        except ImportError:
            logger.warning("[cache_persist] mlx_lm not available, cannot load")
            return 0

        with open(index_path) as f:
            index = json.load(f)

        # Accept v2 (legacy: int-array tokens.bin, no save_uuid) and v3
        # (R10-D: magic-prefixed tokens.bin with save_uuid). A future
        # writer that bumps past v3 will be refused cleanly here — the
        # operator sees one structured WARN instead of silent garbage,
        # closing the "schema evolution without version stamp" gap the
        # spec called out. Anything below v2 is the pre-#198 layout
        # whose entry files lacked cache_types metadata; skip.
        version = index.get("version", 1)
        if version < 2:
            logger.warning(
                f"[cache_persist] unsupported version {version} "
                f"(known: 2 legacy, {_TOKENS_FORMAT_VERSION_IN_INDEX} current); "
                f"skipping load"
            )
            return 0
        if version > _TOKENS_FORMAT_VERSION_IN_INDEX:
            # Newer file from a future deploy. Refuse cleanly so we
            # never reach into a format we don't know — silent
            # mis-decode would re-open the R10-D drift class.
            logger.warning(
                f"[cache_persist] index.json version {version} is newer than "
                f"this build supports (max {_TOKENS_FORMAT_VERSION_IN_INDEX}); "
                f"skipping load to avoid silent corruption"
            )
            return 0
        # File-level uuid (v3+ only). Used by ``_read_tokens_bin`` to
        # detect a tokens.bin that was clobbered by a previous-cycle
        # orphan — the index's claim and the entry file's claim must
        # agree on whose save they came from.
        #
        # R10-D codex round 2 HIGH: a malformed v3 index with a missing
        # / non-string save_uuid would otherwise pass ``None`` here and
        # silently re-enable the v2 legacy fallback in ``_read_tokens_bin``
        # — defeating the whole point of the format pin. Refuse the
        # load if the writer didn't stamp a valid uuid string on a v3+
        # index. (v2 indices have no save_uuid by design — that's the
        # only legitimate ``None``.)
        if version >= 3:
            raw_uuid = index.get("save_uuid")
            if not isinstance(raw_uuid, str) or not raw_uuid:
                logger.warning(
                    f"[cache_persist] index.json version {version} is missing "
                    f"a valid save_uuid (got {type(raw_uuid).__name__}); "
                    f"refusing load to avoid orphan-pair mis-decode"
                )
                return 0
            expected_save_uuid = raw_uuid
        else:
            expected_save_uuid = None

        loaded = 0
        corrupt_skipped = 0
        duplicate_skipped = 0
        incompatible_skipped = 0
        for entry_meta in index.get("entries", []):
            i = entry_meta["index"]
            expected_num_tokens = entry_meta["num_tokens"]
            entry_path = os.path.join(cache_dir, f"entry_{i}.safetensors")
            tokens_path = os.path.join(cache_dir, f"entry_{i}_tokens.bin")

            if not os.path.exists(entry_path) or not os.path.exists(tokens_path):
                logger.warning(f"[cache_persist] missing files for entry {i}, skipping")
                corrupt_skipped += 1
                continue

            # Cache-type compatibility check (#198 BUG B). Reject entries
            # whose persisted cache class doesn't match what the current
            # config can dequantize at fetch time — otherwise tuple-form
            # keys reach the scheduler. Done early to skip the safetensors
            # body validation work for entries we'd discard anyway.
            cache_types = entry_meta.get("cache_types") or []
            if not cache_types:
                # Backward compat with index.json from before cache_types
                # existed: peek at safetensors __metadata__.
                cache_types = _safetensors_cache_classes(entry_path)
            ok, reason = _cache_classes_compatible(cache_types, self._config)
            if not ok:
                logger.info(
                    f"[cache_persist] entry {i} skipped — {reason}; "
                    f"persisted types={cache_types}"
                )
                incompatible_skipped += 1
                continue

            # R10-D: size cross-check is now version-aware. Legacy v2
            # tokens.bin is exactly ``num_tokens * 4`` bytes; v3 has a
            # variable-length header (magic + lengths + save_uuid) on
            # top. The downstream ``_read_tokens_bin`` enforces the
            # actual byte invariants of the right format and returns a
            # structured reject reason — but a fast pre-check on the
            # absolute floor (tokens.bin must hold at least the legacy
            # payload length OR the v3 fixed-prefix length) catches
            # severely truncated files before the open() syscall.
            actual_bytes = os.path.getsize(tokens_path)
            legacy_expected_bytes = expected_num_tokens * _TOKEN_BYTES
            v3_min_bytes = _TOKENS_HEADER_FIXED_LEN  # uuid + payload come after
            if actual_bytes < min(legacy_expected_bytes, v3_min_bytes):
                logger.warning(
                    f"[cache_persist] entry {i} tokens.bin too short "
                    f"({actual_bytes} bytes) for "
                    f"{expected_num_tokens} tokens — corruption, skipping"
                )
                corrupt_skipped += 1
                continue

            # mx.load mmaps safetensors lazily and will silently return
            # zeros for positions past EOF. Verify the body is fully on
            # disk via the safetensors header before trusting the entry
            # (BUG D — body-truncated file slips through load otherwise).
            if not _safetensors_is_complete(entry_path):
                logger.warning(
                    f"[cache_persist] entry {i} safetensors body is short of "
                    f"its header's declared data range — corruption, skipping"
                )
                corrupt_skipped += 1
                continue

            try:
                # R10-D: read tokens via the format-aware helper. Detects
                # the v3 magic and enforces (magic + length prefix +
                # save_uuid) round-trip invariants; falls back to the
                # pre-R10 array.array("i") path when magic is absent so
                # a legacy v2 file is still readable in-place. Any
                # mismatch returns (None, reason) — we bump the
                # corruption metric and surface a structured WARN.
                tokens, reject_reason = _read_tokens_bin(
                    tokens_path, expected_num_tokens, expected_save_uuid
                )
                if tokens is None:
                    logger.warning(
                        f"[cache_persist] entry {i} tokens.bin rejected: "
                        f"{reject_reason} — corruption, skipping"
                    )
                    corrupt_skipped += 1
                    continue

                # Skip duplicates (e.g. an entry that warmup already
                # populated). Done BEFORE load_prompt_cache so a duplicate
                # entry doesn't pay the safetensors mmap cost only to be
                # discarded. Without this guard, bisect.insort would also
                # create duplicate keys in _sorted_keys and memory would
                # double-count. Benign — not a corruption signal.
                tokens_key = tuple(tokens)
                if tokens_key in self._entries:
                    logger.debug(
                        f"[cache_persist] entry {i} already present in cache "
                        f"(len={len(tokens)}), skipping disk copy"
                    )
                    duplicate_skipped += 1
                    continue

                # Load KV cache (header completeness already validated above).
                cache = load_prompt_cache(entry_path)

                # Invariant: a well-formed entry has cache.offset == len(tokens).
                # Any deviation means BUG A poisoning slipped through earlier
                # checks; drop it rather than risk corrupting fetch output.
                if cache:
                    head_offset = getattr(cache[0], "offset", None)
                    if head_offset is not None and head_offset != len(tokens):
                        logger.warning(
                            f"[cache_persist] entry {i} cache offset "
                            f"({head_offset}) != tokens length ({len(tokens)}) "
                            f"— corruption, skipping"
                        )
                        corrupt_skipped += 1
                        continue

                # Estimate memory
                memory = estimate_kv_cache_memory(cache)

                # Check if it fits
                if self._current_memory + memory > self._max_memory:
                    logger.info(
                        f"[cache_persist] entry {i} would exceed memory limit "
                        f"({(self._current_memory + memory) / _BYTES_PER_MB:.0f}MB > "
                        f"{self._max_memory / _BYTES_PER_MB:.0f}MB), stopping load"
                    )
                    break

                entry = _CacheEntry(
                    tokens=tokens_key,
                    cache=cache,
                    memory_bytes=memory,
                )
                self._entries[tokens_key] = entry
                self._current_memory += memory
                bisect.insort(self._sorted_keys, tokens_key)
                loaded += 1

                logger.info(
                    f"[cache_persist] loaded entry {i}: "
                    f"{len(tokens)} tokens, "
                    f"{memory / _BYTES_PER_MB:.1f}MB KV"
                )

            except Exception as e:
                logger.warning(f"[cache_persist] failed to load entry {i}: {e}")
                corrupt_skipped += 1

        self._stats.entry_count = len(self._entries)
        self._stats.current_memory_bytes = self._current_memory
        # R10-D / R9-L4: surface the corrupt-skip count as a sticky
        # cumulative counter so /metrics can graph "% of disk-load
        # rejected per startup" without re-scraping logs. We only
        # bump for CORRUPTION skips — duplicate and incompatible skips
        # are benign (a deliberate config change or in-memory dedup)
        # and would dilute the corruption signal an operator is
        # actually looking for.
        self._stats.load_skipped += corrupt_skipped

        dt = _time.monotonic() - t0
        summary = (
            f"[cache_persist] LOADED {loaded} entries from {cache_dir} "
            f"in {dt:.1f}s ({self._current_memory / _BYTES_PER_MB:.0f}MB total)"
        )
        if duplicate_skipped:
            summary += (
                f", {duplicate_skipped} skipped as duplicates of in-memory entries"
            )
        if incompatible_skipped:
            summary += (
                f", {incompatible_skipped} skipped as incompatible with "
                f"current cache config (e.g. config changed between runs)"
            )
        if corrupt_skipped:
            logger.warning(
                f"{summary}, SKIPPED {corrupt_skipped} corrupt entries — "
                f"disk cache may need cleanup"
            )
        else:
            logger.info(summary)
        return loaded
