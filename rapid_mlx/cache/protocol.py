# SPDX-License-Identifier: Apache-2.0
"""Wire protocol for the KV cache export/import HTTP API (issue #476).

Two concepts that look similar but are NOT the same — keep them straight:

* ``PROTOCOL_VERSION`` (this module) — the version of the *manifest* the
  HTTP API agrees on. Bumped when the manifest schema below changes shape
  in a way old clients can't read.
* ``index["version"]`` (in ``rapid_mlx/memory_cache.py``) — the on-disk
  format of the engine's prefix-cache directory itself (entry layout,
  safetensors keys, etc.). Bumped independently when the engine changes
  how it serializes entries.

A manifest sits **alongside** the engine's ``index.json`` at the export
root and describes "what model produced this blob, with what quant /
paged-cache / turboquant-kv settings, and at what protocol version" so a
peer instance can refuse a mismatched import before touching tensors.
"""

from __future__ import annotations

import datetime as _datetime
import json
import logging
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "1"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class SaveOutcome:
    """Operation-specific result of one cache save (#1100 codex round 4 #2).

    Returned DIRECTLY from the engine save call so the export route never
    reads a cache-global ``_last_save_outcome`` field that a concurrent
    save to ANOTHER destination (distinct per-destination lock) could
    overwrite between the op and the read.

    ``outcome`` is one of:
      * ``"empty"``     — the cache held 0 entries (a legit no-op export)
      * ``"committed"`` — >=1 entry committed + atomically published
      * ``"failed"``    — the cache had entries but nothing committed
    """

    outcome: str


@dataclass(frozen=True)
class LoadResult:
    """Operation-specific result of one cache load (#1100 codex round 4 #2).

    ``entries`` is the count loaded; ``bytes_loaded`` is the exact KV byte
    total this load installed (0 on an empty load or an aborted replace).
    Returned directly from the engine load call so the import route never
    reads a cache-global ``_last_load_bytes`` field a concurrent load to
    another path could clobber.
    """

    entries: int
    bytes_loaded: int


# Default sandbox root for export/import paths. Overridable via the
# ``RAPID_MLX_CACHE_EXPORT_DIR`` env var. All caller-supplied paths must
# resolve inside this directory after symlink expansion — otherwise a
# bearer-token holder could write arbitrary files anywhere on disk.
_DEFAULT_EXPORT_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "rapid-mlx", "cache_exports"
)
_EXPORT_DIR_ENV = "RAPID_MLX_CACHE_EXPORT_DIR"


class InvalidExportPathError(ValueError):
    """Raised when a caller-supplied path escapes the sandbox root."""


class ManifestNotFoundError(FileNotFoundError):
    """Raised when ``read_manifest`` is called on a path without one."""


class MalformedManifestError(ValueError):
    """Raised when ``manifest.json`` exists but isn't a valid JSON object.

    Distinct from ``ManifestNotFoundError`` (missing file) and
    ``ManifestMismatchError`` (well-formed but fails compatibility) so
    route handlers can map each to its correct HTTP status — malformed
    payload is a caller error (400), missing is 404, mismatch is 409.
    """


class CommittedIndexUnreadableError(RuntimeError):
    """Raised when a COMMITTED export's ``index.json`` can't be read back.

    #1100 codex round 6 (#2): ``build_manifest_from_engine_state`` normally
    reads the entry/byte counts from the committed on-disk ``index.json`` and
    falls back to the live cache ledger only when there is NO index (a
    legitimately empty export / disabled cache, both 0/0). But that fallback
    also fired on an index that EXISTS yet is unreadable or malformed — so a
    save that reported ``"committed"`` but left a torn index would publish a
    manifest describing the (nonzero) live ledger for a snapshot no peer can
    import. When the caller asserts the save committed
    (``require_committed_index=True``), a ``None`` committed read is therefore
    an ERROR, not an empty-export signal: the export route catches this and
    discards the blob instead of publishing a lying manifest.
    """


class EngineNotReadyError(RuntimeError):
    """Raised by a cache op when the engine exists but isn't fully loaded.

    #1100 codex round 8 (#4): the route's ``_engine_or_503`` only checks
    ``get_config().engine is None``. But a ``BatchedEngine`` can be installed
    (non-None) while its INNER engine (``_engine``) is still absent — model not
    yet started, or torn down. Its cache forwards previously masked that state
    as a successful EMPTY save / ZERO load (``SaveOutcome("empty")`` /
    ``LoadResult(0, 0)``), so export/import returned 200 instead of the
    advertised 503 "engine not loaded". The forwarders now raise this instead;
    the cache route catches it and maps to the same 503 an absent engine gets,
    so "engine not loaded" is one consistent signal whether the outer engine is
    None OR its inner engine hasn't come up.
    """


class ManifestMismatchError(ValueError):
    """Raised when a manifest doesn't match caller expectations.

    Carries both sides so routes can surface a structured 409 body.
    """

    def __init__(self, field: str, expected: str, actual: str) -> None:
        super().__init__(
            f"manifest {field} mismatch: expected {expected!r}, got {actual!r}"
        )
        self.field = field
        self.expected = expected
        self.actual = actual


# Field-type registry for ``Manifest.from_dict`` runtime validation.
# Kept inline rather than reflected from the dataclass because
# ``from __future__ import annotations`` makes ``field.type`` a string
# at class-definition time, and runtime typing.get_type_hints would
# need this module to be fully importable — easier to enumerate.
_FIELD_TYPES: dict[str, type] = {
    "protocol_version": str,
    "model_id": str,
    "quantization": str,
    "paged_cache": bool,
    "turboquant_kv": bool,
    "index_format_version": int,
    "entries": int,
    "total_bytes": int,
    "rapid_mlx_version": str,
    "created_at": str,
    "extra": dict,
}


def _is_expected_type(value: object, expected: type) -> bool:
    """Strict isinstance check that rejects bool when int is expected.

    Python's ``isinstance(True, int)`` is True (bool subclasses int), but
    JSON ``true`` was clearly not meant as the integer 1 — so reject it.
    Symmetric: ``isinstance(1, bool)`` is False, which is what we want.
    """
    if expected is int and isinstance(value, bool):
        return False
    return isinstance(value, expected)


@dataclass
class Manifest:
    """Header describing the engine-cache blob at an export root.

    Additive-only: new fields MUST default to a value old readers will
    treat as "unknown / unset". Removing or renaming a field is a
    breaking change and requires bumping ``PROTOCOL_VERSION``.
    """

    protocol_version: str = PROTOCOL_VERSION
    model_id: str = ""
    quantization: str = ""
    paged_cache: bool = False
    turboquant_kv: bool = False
    # The engine's on-disk index format version (``index["version"]``)
    # at the time of export. Separate from protocol_version above —
    # see the module docstring.
    index_format_version: int = 0
    entries: int = 0
    total_bytes: int = 0
    # Free-form provenance — exporting instance's rapid-mlx version
    # and a timestamp. Importers MUST NOT gate on these (they're
    # informational), but they're invaluable for debugging.
    rapid_mlx_version: str = ""
    created_at: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Manifest:
        """Build a ``Manifest`` from a parsed JSON dict, with type checks.

        Drops unknown keys (additive-evolution semantics) but rejects any
        known key whose value is the wrong type. Without this a peer could
        serve ``{"entries": "not-an-int"}`` and the route would return 200
        for a structurally invalid manifest — codex round-3 BLOCKING.
        """
        filtered = {}
        for key, value in data.items():
            if key not in _FIELD_TYPES:
                continue  # additive evolution
            expected = _FIELD_TYPES[key]
            if not _is_expected_type(value, expected):
                raise MalformedManifestError(
                    f"manifest field {key!r}: expected {expected.__name__}, "
                    f"got {type(value).__name__}"
                )
            filtered[key] = value
        return cls(**filtered)


def write_manifest(root: Path, manifest: Manifest) -> Path:
    """Atomically write ``manifest.json`` under ``root``. Returns the path.

    Crash/disk-full mid-write to ``manifest.json`` would otherwise leave
    a truncated file that subsequent reads surface as a 400 instead of
    preserving the last successful export. Mitigation: write to a temp
    file in the same directory (same fs → atomic ``os.replace``), fsync
    the data, then rename onto ``manifest.json``. If anything fails the
    temp is cleaned up and the prior manifest is untouched.
    """
    root.mkdir(parents=True, exist_ok=True)
    target = root / MANIFEST_FILENAME
    fd, tmp_name = tempfile.mkstemp(prefix=".manifest-", suffix=".json", dir=str(root))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    return target


def read_manifest(root: Path) -> Manifest:
    """Read and parse ``manifest.json`` at ``root``.

    Three distinct failure modes — each maps to a different HTTP status:

    * ``ManifestNotFoundError`` — the file doesn't exist (caller picked
      a path that hasn't been exported to). Routes → 404.
    * ``MalformedManifestError`` — file exists but isn't valid JSON, or
      decodes to something other than an object (a list, a string, etc.).
      A peer could have written garbage, or an old v0 layout slipped in.
      Routes → 400. Without this branch the JSONDecodeError / TypeError
      would surface as a 500 and hide a caller-controlled bug.
    """
    target = root / MANIFEST_FILENAME
    if not target.is_file():
        # Path-free exception text — the route layer logs the resolved
        # path server-side, but the HTTP body stays caller-oriented so a
        # bearer-token holder can't probe the server's export-root layout
        # by enumerating 404s. The path lives in the chained exception's
        # `filename` for callers that want it programmatically.
        exc = ManifestNotFoundError("manifest.json not found")
        exc.filename = str(target)
        raise exc
    try:
        data = json.loads(target.read_text())
    except json.JSONDecodeError as exc:
        raise MalformedManifestError(
            f"manifest.json is not valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(data, dict):
        raise MalformedManifestError(
            f"manifest.json must decode to a JSON object, got {type(data).__name__}"
        )
    return Manifest.from_dict(data)


def _rapid_mlx_version() -> str:
    """Best-effort installed ``rapid-mlx`` version for manifest provenance.

    Falls back to ``""`` (never raises): a source checkout or an editable
    install without dist metadata must still be able to export — the field
    is informational (importers MUST NOT gate on it, see ``Manifest``).
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("rapid-mlx")
        except PackageNotFoundError:
            return ""
    except Exception:  # pragma: no cover — defensive; metadata API unavailable
        return ""


# On-disk prefix-cache ``index.json`` schema window this build understands.
# Mirrors ``memory_cache._TOKENS_FORMAT_VERSION_IN_INDEX`` (kept here rather
# than imported to avoid a protocol→memory_cache import cycle — memory_cache
# imports THIS module). v2 = legacy int-array tokens.bin (no save_uuid);
# v3 = magic-prefixed tokens.bin with save_uuid. Anything outside [min, max]
# is refused (older = missing metadata; newer = format we can't decode).
_COMMITTED_INDEX_MIN_VERSION = 2
_COMMITTED_INDEX_MAX_VERSION = 3


def _valid_nonneg_int(v: object) -> bool:
    """True iff ``v`` is a nonnegative integer value.

    Rejects ``bool`` (a JSON ``true``/``false`` is an ``int`` subclass) and
    negatives; accepts a whole-valued ``float`` (json may parse ``5.0``).
    ``index`` / ``num_tokens`` are indices/counts the importer dereferences,
    so they MUST be nonnegative ints.
    """
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return v >= 0
    if isinstance(v, float):
        return v >= 0 and v.is_integer()
    return False


def validate_committed_index_data(
    data: object,
) -> tuple[bool, str, int, int]:
    """Validate a parsed ``index.json`` mapping FAIL-CLOSED over the whole file.

    Single source of truth for "is this committed cache index safe to trust?"
    Used by BOTH the export/manifest side (``_read_committed_cache_counts`` →
    ``build_manifest_from_engine_state(require_committed_index=True)``) and the
    IMPORT side (``memory_cache.load_from_disk`` gate) so the two can't drift on
    what counts as a well-formed index — the exact drift codex round 8 (#1/#3)
    flagged (export published a manifest for an index the importer then crashed
    on at ``entry_meta["index"]``).

    Checks, in order:

    * ``data`` is a dict.
    * ``version`` (default 1) is within [``_COMMITTED_INDEX_MIN_VERSION``,
      ``_COMMITTED_INDEX_MAX_VERSION``] — a wrong-version index is refused, not
      silently parsed against a format we don't know (round 8 #3).
    * ``entries`` is a list.
    * EVERY entry is a dict carrying nonneg-int ``index`` + ``num_tokens`` and a
      nonneg (int|float) ``memory_bytes`` (defaulting to 0 when absent). One
      malformed entry rejects the WHOLE index — no partial trust.

    Returns ``(ok, reason, entries_count, total_bytes)``. On failure ``ok`` is
    ``False`` and ``reason`` names the first problem (for a single structured
    WARN); ``entries_count``/``total_bytes`` are 0. On success the count is
    ``len(entries)`` and total is the summed per-entry ``memory_bytes``.
    """
    if not isinstance(data, dict):
        return False, "index.json is not a JSON object", 0, 0
    version = data.get("version", 1)
    if isinstance(version, bool) or not isinstance(version, int):
        return False, f"index version has non-int type {type(version).__name__}", 0, 0
    if not (_COMMITTED_INDEX_MIN_VERSION <= version <= _COMMITTED_INDEX_MAX_VERSION):
        return (
            False,
            f"index version {version} outside supported "
            f"[{_COMMITTED_INDEX_MIN_VERSION}, {_COMMITTED_INDEX_MAX_VERSION}]",
            0,
            0,
        )
    entries_list = data.get("entries")
    if not isinstance(entries_list, list):
        return False, "index 'entries' is not a list", 0, 0
    total = 0
    for pos, entry in enumerate(entries_list):
        if not isinstance(entry, dict):
            return False, f"entry {pos} is not an object", 0, 0
        if not _valid_nonneg_int(entry.get("index")):
            return False, f"entry {pos} has invalid 'index'", 0, 0
        if not _valid_nonneg_int(entry.get("num_tokens")):
            return False, f"entry {pos} has invalid 'num_tokens'", 0, 0
        mb = entry.get("memory_bytes", 0)
        # #1100 codex round 9 (#2): reject non-finite floats. Python's json
        # parses ``NaN``/``Infinity`` into float('nan')/float('inf'); a bare
        # ``mb < 0`` check passes them through, and the ``int(mb)`` below then
        # raises (crashing the import) instead of rejecting the malformed index.
        # Require a finite, integral, nonnegative value (memory_bytes is a byte
        # COUNT — fractional/NaN/Inf is meaningless).
        if (
            isinstance(mb, bool)
            or not isinstance(mb, (int, float))
            or (
                isinstance(mb, float) and (not math.isfinite(mb) or not mb.is_integer())
            )
            or mb < 0
        ):
            return False, f"entry {pos} has invalid 'memory_bytes'", 0, 0
        total += int(mb)
    return True, "", len(entries_list), total


def _read_committed_cache_counts(cache_dir: str | Path) -> tuple[int, int] | None:
    """Read ``(entries, total_bytes)`` from the committed ``index.json``.

    BLOCKING-3 (#1100): the manifest counters must reflect what
    ``save_to_disk`` actually committed to disk, NOT the live in-memory
    ledger (``len(cache._entries)`` / ``cache._current_memory``). The live
    ledger drifts from the snapshot when the cache mutates in the window
    between the save and this manifest build (a concurrent store, or an
    entry ``save_to_disk`` dropped in its post-write self-verify pass) —
    producing a manifest whose ``entries``/``total_bytes`` disagree with
    the on-disk index a peer will actually load.

    Reads the per-entry ``memory_bytes`` the save loop wrote into each
    ``index["entries"][i]`` and sums them (this is the post-verify entry
    list — entries whose files vanished mid-save are already filtered), so
    the total matches the bytes a peer can actually load, not the pre-
    verify ``total_memory_bytes`` snapshot.

    Returns ``None`` when there is no readable, WELL-FORMED committed index —
    an empty export (``save_to_disk`` writes no ``index.json`` for 0 entries),
    a disabled prefix cache, OR a torn/malformed index. The caller uses the
    ``None``/count distinction to decide fallback-vs-fail:
    ``build_manifest_from_engine_state(require_committed_index=True)`` (a
    committed save) treats ``None`` as an error and discards the export; the
    live-ledger fallback (empty / disabled cache) correctly reports 0/0.

    #1100 codex round 7 (#3) → round 8 (#3): validation is FAIL-CLOSED over the
    WHOLE index via the shared ``validate_committed_index_data`` — one malformed
    entry (not a dict, missing / wrong-typed / negative ``index`` /
    ``num_tokens`` / ``memory_bytes``) OR a wrong ``version`` rejects the entire
    index (``None``). The old loop ``continue``d past malformed entries yet
    returned ``len(entries_list)`` — counting entries it hadn't validated — so a
    partially-torn index could pass ``require_committed_index`` here and then
    CRASH the importer at ``entry_meta["index"]`` / ``["num_tokens"]``. Round 8
    adds the version gate so a manifest is never published for a snapshot whose
    on-disk format a peer's importer would refuse. The importer
    (``memory_cache.load_from_disk``) applies the SAME validator so export and
    import agree byte-for-byte on what "loadable" means.
    """
    try:
        index_path = Path(cache_dir) / "index.json"
        if not index_path.is_file():
            return None
        data = json.loads(index_path.read_text())
        ok, _reason, count, total = validate_committed_index_data(data)
        if not ok:
            return None
        return count, total
    except Exception:  # pragma: no cover — defensive against a torn index
        return None


def resolve_engine_model_id(engine: Any) -> str:
    """Resolve the model id for a loaded engine, server-singleton first.

    BLOCKING-2 (#1100 codex round 2): the import route's model-identity
    gate and the manifest builder MUST derive the loaded model's id the
    SAME way, or they disagree on embedded engines. The manifest builder
    fell back to ``engine.config.model_name`` when the server singleton's
    ``model_name`` was empty (which it is for an embedded / unit-test
    engine that never populated ``ServerConfig``); the gate read only the
    empty singleton and therefore skipped the check — letting an embedded
    engine import another model's KV state. This single helper is the one
    source of truth both call sites use, so they can't drift again.

    Resolution order (first non-empty wins):

    1. ``get_config().model_name`` — the operator-facing HF id / alias the
       server booted with (empty for embedded engines).
    2. ``engine.config.model_name`` — the engine's own config.
    3. ``engine.config.model_path`` — last-resort path form.

    Returns ``""`` only when none of the above yields an id.
    """
    try:
        from ..config import get_config

        model_id = get_config().model_name or ""
    except Exception:  # pragma: no cover — config singleton import guard
        model_id = ""
    if not model_id:
        engine_cfg = getattr(engine, "config", None)
        if engine_cfg is not None:
            model_id = (
                getattr(engine_cfg, "model_name", None)
                or getattr(engine_cfg, "model_path", None)
                or ""
            )
    return model_id


def resolve_engine_cache_geometry(engine: Any) -> tuple[str, bool, bool]:
    """Return the loaded engine's KV-cache geometry ``(quantization,
    paged_cache, turboquant_kv)``.

    #1100 codex round 9 (#1): the manifest carries these knobs and the import
    gate must validate the imported blob's geometry AGAINST the loaded engine,
    not just the model_id string — two checkpoints sharing an alias but built
    with different KV-cache dtype / paged / turboquant settings produce
    byte-incompatible KV tensors, and loading one into the other corrupts
    inference. This is the single source of truth both the manifest BUILDER (on
    export) and the import GATE use, so they can't drift on how geometry is read
    (the same one-source-of-truth discipline ``resolve_engine_model_id`` gives
    the id gate). Reads ``scheduler.config`` via ``_resolve_scheduler`` (which
    unwraps the production BatchedEngine nesting); returns the conservative
    ``("", False, False)`` only when no scheduler config is reachable.

    NOTE: this is a NECESSARY-not-SUFFICIENT geometry check — it covers the KV
    knobs the manifest already records. A full immutable fingerprint (model
    revision hash, tokenizer, architecture) is tracked as a follow-up; it needs
    engine-side fingerprint computation + a manifest schema addition beyond this
    PR's export/import wiring scope.
    """
    from ..runtime.cache import _resolve_scheduler

    scheduler = _resolve_scheduler(engine)
    sched_cfg = getattr(scheduler, "config", None) if scheduler is not None else None
    if sched_cfg is None:
        return "", False, False
    quantization = getattr(sched_cfg, "kv_cache_dtype", "") or ""
    paged_cache = bool(getattr(sched_cfg, "use_paged_cache", False))
    turboquant_kv = bool(getattr(sched_cfg, "kv_cache_turboquant", False))
    return quantization, paged_cache, turboquant_kv


def build_manifest_from_engine_state(
    engine: Any,
    cache_dir: str | Path | None = None,
    require_committed_index: bool = False,
) -> Manifest:
    """Build a :class:`Manifest` describing the engine's current cache blob.

    Reads the model id + cache-config knobs (quantization / paged / turbo-
    quant) off the live engine so a peer instance can gate an import before
    touching tensors. Every field access is defensive: a partially-built or
    prefix-cache-disabled engine must still produce a well-formed manifest
    (with ``entries``/``total_bytes`` = 0) rather than raise — the export
    route already 503s when the engine is absent, so by the time we get here
    the *engine* exists but its individual sub-objects may not.

    Field provenance (verified against ``SchedulerConfig`` / the server
    ``ServerConfig`` at 0.10.9):

    * ``model_id`` — ``get_config().model_name`` (the operator-facing HF id
      / alias the server booted with). Falls back to the engine's own
      ``config.model_name`` / ``config.model_path`` if the server singleton
      isn't populated (unit-test engines).
    * ``quantization`` — ``scheduler.config.kv_cache_dtype`` (the canonical
      R15 #300 operator-facing dtype string: ``bf16`` / ``int8`` / ``int4``).
    * ``paged_cache`` — ``scheduler.config.use_paged_cache``.
    * ``turboquant_kv`` — ``scheduler.config.kv_cache_turboquant``.
    * ``index_format_version`` — the engine's on-disk prefix-cache index
      format (``memory_cache._TOKENS_FORMAT_VERSION_IN_INDEX``), NOT the
      manifest protocol version (see module docstring).
    * ``entries`` / ``total_bytes`` — when ``cache_dir`` is given, read
      from the COMMITTED ``index.json`` at that export root (the source of
      truth a peer will load), via ``_read_committed_cache_counts``. Falls
      back to the live prefix-cache ledger (``len(cache._entries)`` /
      ``cache._current_memory``) only when there's no committed index —
      an empty export or a disabled prefix cache, both of which the live
      ledger reports as 0/0. Deriving from the committed index (not the
      live ledger) is BLOCKING-3 (#1100): the ledger drifts from the
      snapshot if the cache mutates between save and manifest-build.

    ``cache_dir`` is the export root just written by ``save_to_disk``.
    Omitting it (the pre-#1100 call shape) keeps the live-ledger behavior
    — still correct for callers that only want the config knobs.

    ``require_committed_index`` (#1100 codex round 6 #2): when the caller
    KNOWS the preceding save committed (>=1 entry atomically published), pass
    ``True`` so a missing/unreadable ``index.json`` raises
    :class:`CommittedIndexUnreadableError` instead of silently falling back to
    the live ledger. The live-ledger fallback is only correct for the genuine
    no-index cases (empty export / disabled cache); on a committed save a
    ``None`` committed read means a TORN index, and publishing the live
    ledger's nonzero counts would advertise a snapshot no peer can import.
    """
    # Model id — prefer the server singleton (what the operator typed), fall
    # back to the engine's own config for unit-test / embedded engines. The
    # import route's identity gate shares this exact resolution via
    # ``resolve_engine_model_id`` so the two can't drift (#1100 BLOCKING-2).
    model_id = resolve_engine_model_id(engine)

    # Cache-config knobs off the scheduler's SchedulerConfig. Read via the
    # SHARED ``resolve_engine_cache_geometry`` (which unwraps the production
    # BatchedEngine nesting the same way ``_resolve_scheduler`` does — a plain
    # ``getattr(engine, "scheduler")`` collapsed to None under BatchedEngine,
    # zeroing every field: real-hardware smoke saw 70 entries / 1.4 GB on disk
    # but manifest reported entries=0). Using the same helper the import gate
    # uses keeps builder and gate from drifting on how geometry is read.
    quantization, paged_cache, turboquant_kv = resolve_engine_cache_geometry(engine)

    # The unwrapped scheduler is also the source of the live-ledger entry-count
    # fallback below (when there's no committed index). Resolve it the same way.
    from ..runtime.cache import _resolve_scheduler

    scheduler = _resolve_scheduler(engine)

    # On-disk index format version — read the engine's own constant so the
    # manifest tracks whatever the engine actually writes into index.json.
    try:
        from ..memory_cache import _TOKENS_FORMAT_VERSION_IN_INDEX

        index_format_version = int(_TOKENS_FORMAT_VERSION_IN_INDEX)
    except Exception:  # pragma: no cover — memory_cache import guard
        index_format_version = 0

    # entries / total_bytes — prefer the COMMITTED on-disk index (#1100
    # BLOCKING-3). When ``cache_dir`` points at a just-written export root,
    # read the counts the save loop actually committed; only fall back to
    # the live ledger when there's no readable index (empty export /
    # disabled cache, both correctly 0/0 on the live path).
    entries = 0
    total_bytes = 0
    committed = (
        _read_committed_cache_counts(cache_dir) if cache_dir is not None else None
    )
    if committed is None and require_committed_index:
        # #1100 codex round 6 (#2): the caller asserts a save just COMMITTED
        # (>=1 entry atomically published), so a readable ``index.json`` MUST
        # exist at ``cache_dir``. A ``None`` here means the index is missing or
        # torn/malformed — do NOT silently fall back to the live ledger and
        # publish a manifest for a snapshot no peer can import. Fail loudly so
        # the export route discards the blob and 500s.
        raise CommittedIndexUnreadableError(
            "committed export index.json is missing or unreadable at the "
            "destination — refusing to publish a manifest for a non-importable "
            "snapshot"
        )
    if committed is not None:
        entries, total_bytes = committed
    else:
        # Live prefix-cache ledger fallback. ``memory_aware_cache`` is None
        # when the prefix cache is disabled (--disable-prefix-cache) —
        # export an empty snapshot rather than raising. ``entries`` and
        # ``total_bytes`` are read in SEPARATE try/excepts so a fault
        # reading one (a partially-torn cache) doesn't zero the other.
        prefix_cache = (
            getattr(scheduler, "memory_aware_cache", None)
            if scheduler is not None
            else None
        )
        if prefix_cache is not None:
            try:
                entries = len(prefix_cache._entries)  # noqa: SLF001 — ledger read
            except Exception:  # pragma: no cover — defensive against partial cache
                entries = 0
            try:
                total_bytes = int(prefix_cache._current_memory)  # noqa: SLF001
            except Exception:  # pragma: no cover — defensive against partial cache
                total_bytes = 0

    return Manifest(
        protocol_version=PROTOCOL_VERSION,
        model_id=model_id,
        quantization=quantization,
        paged_cache=paged_cache,
        turboquant_kv=turboquant_kv,
        index_format_version=index_format_version,
        entries=entries,
        total_bytes=total_bytes,
        rapid_mlx_version=_rapid_mlx_version(),
        created_at=_datetime.datetime.now(_datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    )


def default_export_root() -> Path:
    """The sandbox root all caller-supplied paths must resolve inside."""
    raw = os.environ.get(_EXPORT_DIR_ENV) or _DEFAULT_EXPORT_DIR
    # realpath here too — if the operator points the env at a symlink,
    # we sandbox to its target so commonpath comparisons stay sound.
    return Path(os.path.realpath(os.path.expanduser(raw)))


def resolve_cache_dir(caller_path: str | None) -> Path:
    """Resolve ``caller_path`` into a vetted absolute path under the sandbox.

    Rules (all must hold):

    1. ``None`` or empty → the sandbox root itself.
    2. The literal ``..`` segment is rejected pre-realpath as defense in
       depth against ``realpath`` CVEs that may not normalize correctly.
    3. Relative paths are resolved against the sandbox root.
    4. ``os.path.realpath`` collapses every symlink (including
       transitive chains) to its final target. The result must share
       the sandbox root as its ``commonpath`` — any escape, whether via
       absolute path or symlink-to-outside, fails here.

    Raises ``InvalidExportPathError`` on any violation.
    """
    root = default_export_root()
    root.mkdir(parents=True, exist_ok=True)

    if caller_path is None or caller_path == "":
        return root

    if ".." in Path(caller_path).parts:
        raise InvalidExportPathError(
            f"path component '..' is not allowed: {caller_path!r}"
        )

    candidate = Path(caller_path)
    if not candidate.is_absolute():
        candidate = root / candidate

    resolved = Path(os.path.realpath(candidate))

    try:
        common = Path(os.path.commonpath([str(root), str(resolved)]))
    except ValueError as exc:
        # commonpath raises on cross-drive paths (e.g. Windows). Treat
        # as escape — we're not crossing drives intentionally on macOS.
        raise InvalidExportPathError(
            f"path {caller_path!r} could not be compared to sandbox root"
        ) from exc

    if common != root:
        raise InvalidExportPathError(
            f"path {caller_path!r} resolves outside sandbox {root}"
        )

    return resolved
