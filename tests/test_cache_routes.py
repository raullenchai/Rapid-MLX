# SPDX-License-Identifier: Apache-2.0
"""Wire + engine-body tests for the KV cache export/import HTTP API (#476).

Covers the full surface: auth, path sandbox, manifest validation, AND the
engine-touching save/load bodies — including the production BatchedEngine
nesting (scheduler at ``engine._engine.engine.scheduler``), the ArraysCache
round-trip, and the #1100 hardening: the unconditional model-id gate,
committed-size ``max_bytes`` enforcement with blob discard, the atomic
``replace`` clear-inside-load, and precise ``bytes_loaded`` accounting.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.requires_mlx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.cache.protocol import (
    PROTOCOL_VERSION,
    InvalidExportPathError,
    MalformedManifestError,
    Manifest,
    read_manifest,
    resolve_cache_dir,
    write_manifest,
)


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    """Point the export sandbox at an isolated tmp dir for the test."""
    export_root = tmp_path / "exports"
    monkeypatch.setenv("RAPID_MLX_CACHE_EXPORT_DIR", str(export_root))
    return export_root


class _FakeCache:
    """Minimal stand-in for ``MemoryAwarePrefixCache`` — just the ledger
    fields the routes read (``_entries`` / ``_current_memory``) plus a
    ``clear()`` that the ``merge_strategy="replace"`` path calls.

    #1100 codex round 4: the routes no longer read a pre/post ``_current_
    memory`` snapshot around the op — they read the AUTHORITATIVE outcome the
    serialized op records on the cache: ``_last_save_outcome`` (empty /
    committed / failed) and ``_last_load_bytes``. The fake engine's
    save/load stubs below set these the same way the real op does.
    """

    def __init__(self, entries: int = 0, current_memory: int = 0):
        # The routes read ``len(cache._entries)`` and ``cache._current_memory``.
        self._entries = {f"k{i}": i for i in range(entries)}
        self._current_memory = current_memory
        self.clear_calls = 0
        # Mirror the real cache's authoritative op-outcome attributes.
        self._last_save_outcome = "empty"
        self._last_load_bytes = 0

    def clear(self) -> None:
        self.clear_calls += 1
        self._entries = {}
        self._current_memory = 0

    def __len__(self) -> int:
        # Mirror ``MemoryAwarePrefixCache.__len__`` so the export route's
        # entry-count reads exercise the SAME code path the real cache does.
        return len(self._entries)


class _FakeEngine:
    """Faithful fake of the engine surface the cache routes touch.

    Mirrors ``EngineCore``: ``scheduler.memory_aware_cache`` (the prefix
    cache), ``scheduler.config`` (a real-ish ``SchedulerConfig`` shape) plus
    ``save_cache_to_disk`` / ``load_cache_from_disk``. The save/load bodies
    are stubs that write/read a sentinel file so tests can assert the route
    plumbing without a real model; the round-trip suite below exercises the
    REAL ``MemoryAwarePrefixCache.save_to_disk`` / ``load_from_disk`` instead.
    """

    def __init__(
        self,
        *,
        entries: int = 0,
        current_memory: int = 0,
        prefix_cache: bool = True,
        kv_cache_dtype: str = "bf16",
        use_paged_cache: bool = False,
        kv_cache_turboquant: bool = False,
        load_returns: int = 0,
        loaded_bytes: int = 0,
    ):
        cache = (
            _FakeCache(entries=entries, current_memory=current_memory)
            if prefix_cache
            else None
        )
        self.scheduler = SimpleNamespace(
            memory_aware_cache=cache,
            config=SimpleNamespace(
                kv_cache_dtype=kv_cache_dtype,
                use_paged_cache=use_paged_cache,
                kv_cache_turboquant=kv_cache_turboquant,
            ),
        )
        self.config = SimpleNamespace(model_name="test-model", model_path=None)
        self._load_returns = load_returns
        self._loaded_bytes = loaded_bytes
        self.saved_to: str | None = None
        self.loaded_from: str | None = None
        self.load_replace: bool | None = None
        # #1100 codex round 6 (#2): serializes save/load+outcome-read so the
        # fake matches the real engine's single ``max_workers=1`` step-thread
        # atomicity when the route drives it from concurrent to_thread handlers.
        import threading as _threading

        self._step_lock = _threading.Lock()

    def save_cache_to_disk(self, cache_dir: str, should_abort=None) -> bool:
        self.saved_to = cache_dir
        cache = self.scheduler.memory_aware_cache
        committed = bool(cache and self._entry_count())
        # #1100 codex round 4 (#1): record the authoritative outcome the same
        # way the real ``save_to_disk`` does — empty cache is a legit no-op,
        # a non-empty cache that commits >=1 entry is "committed". (The
        # failed-save path is simulated by overriding this stub in the
        # dedicated failure test.)
        if cache is not None:
            cache._last_save_outcome = "committed" if committed else "empty"
        # #1100 codex round 6 (#2): the REAL ``save_to_disk`` always writes a
        # committed ``index.json`` when it commits (>=1 entry), and the manifest
        # builder now REQUIRES that index to be readable on a committed save
        # (``require_committed_index=True``) — else it refuses to publish. Honor
        # that post-condition here so the fake's committed saves produce an
        # importable-shaped blob whose entry count + total memory MATCH the live
        # ledger the tests set up (the manifest reads counts from this index,
        # not the ledger). Emitted only on commit (empty saves write no index).
        if committed:
            n = self._entry_count()
            total = int(getattr(cache, "_current_memory", 0)) if cache else 0
            self._write_committed_index(cache_dir, entries=n, total_bytes=total)
        return committed

    @staticmethod
    def _write_committed_index(
        cache_dir: str, entries: int = 1, total_bytes: int | None = None
    ) -> None:
        """Write a minimal valid committed ``index.json`` mirroring the real
        ``save_to_disk`` post-condition. ``total_bytes`` is split evenly across
        the entries' ``memory_bytes`` so the manifest's committed-index counts
        match the live ledger the caller staged (any remainder lands on the
        last entry). Defaults to 512 B/entry when ``total_bytes`` is omitted."""
        import json as _json
        import os as _os

        entries = max(entries, 0)
        if total_bytes is None:
            per = [512] * entries
        elif entries == 0:
            per = []
        else:
            base = total_bytes // entries
            per = [base] * entries
            per[-1] += total_bytes - base * entries
        _os.makedirs(cache_dir, exist_ok=True)
        with open(_os.path.join(cache_dir, "index.json"), "w") as _fh:
            _json.dump(
                {
                    "version": 3,
                    "entries": [
                        {"index": i, "num_tokens": 4, "memory_bytes": per[i]}
                        for i in range(entries)
                    ],
                },
                _fh,
            )

    def _resolve_cache(self):
        """Resolve the prefix cache like the route's ``_prefix_cache`` does —
        overridden by ``_NestedFakeEngine`` for the nested BatchedEngine
        shape (where ``self.scheduler`` was deleted)."""
        return self.scheduler.memory_aware_cache

    def load_cache_from_disk(self, cache_dir: str, replace: bool = False) -> int:
        self.loaded_from = cache_dir
        self.load_replace = replace
        cache = self._resolve_cache()
        if cache is not None:
            # New contract (#1100 BLOCKING-4): the "replace" clear happens
            # INSIDE the load (atomic on the step thread), not in the route.
            if replace:
                cache.clear()
            # Simulate the loaded footprint. #1100 codex round 4 (#3): the
            # route now reads the AUTHORITATIVE ``_last_load_bytes`` the op
            # records (summed over installed entries), NOT a before/after
            # ``_current_memory`` diff. Record it the same way the real
            # ``load_from_disk`` does: the bytes this load installed, or 0
            # when nothing loaded (aborted replace / empty source).
            cache._current_memory += self._loaded_bytes
            cache._last_load_bytes = self._loaded_bytes if self._load_returns else 0
        return self._load_returns

    def save_cache_with_outcome(self, cache_dir: str, should_abort=None):
        """#1100 codex round 4 (#2): run the save stub then return the
        authoritative outcome as a value — mirrors the real engine wrapper that
        captures the outcome in the SAME step-thread task as the save.

        #1100 codex round 6 (#2): the REAL engine runs save+outcome-read inside
        one task on the single ``max_workers=1`` mlx-step thread, so they are
        atomic w.r.t. any OTHER save. The route calls this via
        ``anyio.to_thread`` from concurrent handlers, so two exports would run
        it on DIFFERENT pool threads and their shared-``_last_save_outcome``
        reads could interleave (a "good" save's read seeing a concurrent "bad"
        save's field write — exactly the cross-contamination this fake exists to
        rule out). Serialize save+read under a per-engine lock so the fake has
        the same single-writer atomicity the real step thread provides.
        """
        from vllm_mlx.cache.protocol import SaveOutcome

        with self._step_lock:
            self.save_cache_to_disk(cache_dir, should_abort=should_abort)
            cache = self._resolve_cache()
            outcome = (
                getattr(cache, "_last_save_outcome", "empty")
                if cache is not None
                else "empty"
            )
        return SaveOutcome(outcome=outcome)

    def load_cache_with_result(self, cache_dir: str, replace: bool = False):
        """#1100 codex round 4 (#2): run the load stub then return the entries
        count + loaded bytes as a value (same step-thread capture semantics)."""
        from vllm_mlx.cache.protocol import LoadResult

        with self._step_lock:
            entries = self.load_cache_from_disk(cache_dir, replace=replace)
            cache = self._resolve_cache()
            bytes_loaded = (
                int(getattr(cache, "_last_load_bytes", 0)) if cache is not None else 0
            )
        return LoadResult(entries=entries, bytes_loaded=bytes_loaded)

    def _entry_count(self) -> int:
        cache = self.scheduler.memory_aware_cache
        return len(cache._entries) if cache is not None else 0


class _NestedFakeEngine(_FakeEngine):
    """Fake of the PRODUCTION ``BatchedEngine`` nesting.

    The real ``BatchedEngine`` does NOT expose ``.scheduler`` — the
    scheduler lives at ``engine._engine.engine.scheduler`` (``._engine`` =
    AsyncEngineCore wrapper, ``.engine`` = inner EngineCore). This fake
    reparents ``_FakeEngine``'s scheduler under that same nesting and DELETES
    the top-level ``.scheduler`` attribute so a route that reaches the cache
    via a naive ``getattr(engine, "scheduler")`` gets None — reproducing the
    real-hardware smoke bug (export reported entries=0 despite 70 real
    entries on disk). ``save_cache_to_disk`` / ``load_cache_from_disk`` stay
    inherited (BatchedEngine forwards them to the inner engine in prod).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        real_scheduler = self.scheduler
        # Reparent under the AsyncEngineCore→EngineCore chain and remove the
        # direct attribute so ONLY the nested lookup succeeds.
        del self.scheduler
        inner_engine = SimpleNamespace(scheduler=real_scheduler)  # EngineCore
        self._engine = SimpleNamespace(engine=inner_engine)  # AsyncEngineCore

    @property
    def _nested_scheduler(self):
        return self._engine.engine.scheduler

    def _resolve_cache(self):
        # Nested shape: the cache is only reachable through the
        # AsyncEngineCore→EngineCore chain (self.scheduler was deleted).
        return self._nested_scheduler.memory_aware_cache

    def save_cache_to_disk(self, cache_dir: str, should_abort=None) -> bool:
        # Prod BatchedEngine forwards to the inner engine; the inner
        # EngineCore runs the real save. Mirror the inherited stub but read
        # the nested cache for the "did anything commit?" signal.
        self.saved_to = cache_dir
        cache = self._nested_scheduler.memory_aware_cache
        committed = bool(cache and len(cache._entries))
        if cache is not None:
            cache._last_save_outcome = "committed" if committed else "empty"
        # #1100 codex round 6 (#2): honor the real committed-save post-condition
        # (a readable index.json whose counts match the nested ledger) the
        # manifest builder now requires.
        if committed:
            self._write_committed_index(
                cache_dir,
                entries=len(cache._entries),
                total_bytes=int(getattr(cache, "_current_memory", 0)),
            )
        return committed


@pytest.fixture
def cache_client(monkeypatch, sandbox):
    """FastAPI TestClient with the cache router + auth + a fake engine.

    The engine now has to be faithful (unlike the pre-#476 stub, where the
    handlers never touched it): the export/import handlers read
    ``scheduler.memory_aware_cache`` + ``scheduler.config`` and call
    ``save_cache_to_disk`` / ``load_cache_from_disk``. ``_FakeEngine``
    provides exactly that surface. Tests that need a specific engine shape
    (no prefix cache, non-empty cache, a load that skips entries) install
    their own via ``get_config().engine = _FakeEngine(...)``.
    """
    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.cache import router

    cfg = reset_config()
    cfg.api_key = "test-secret"
    cfg.engine = _FakeEngine()
    cfg.model_name = "test-model"

    app = FastAPI()
    app.include_router(router)
    yield SimpleNamespace(
        client=TestClient(app),
        sandbox=sandbox,
        cfg=cfg,
        FakeEngine=_FakeEngine,
        NestedFakeEngine=_NestedFakeEngine,
    )

    reset_config()


def _auth() -> dict:
    return {"Authorization": "Bearer test-secret"}


# ---------------------------------------------------------------------------
# protocol.resolve_cache_dir — unit tests at the helper level
# ---------------------------------------------------------------------------


def test_resolve_cache_dir_returns_sandbox_root_for_none(sandbox):
    """``None`` resolves to the sandbox root itself, which is created."""
    resolved = resolve_cache_dir(None)
    assert resolved == Path(sandbox).resolve()
    assert resolved.is_dir()


def test_resolve_cache_dir_relative_path_is_joined(sandbox):
    """Relative paths resolve under the sandbox root."""
    resolved = resolve_cache_dir("session-a")
    assert resolved == (Path(sandbox).resolve() / "session-a")


def test_resolve_cache_dir_rejects_dotdot_segment(sandbox):
    """``..`` in any segment is rejected before realpath even runs."""
    with pytest.raises(InvalidExportPathError, match="not allowed"):
        resolve_cache_dir("../etc/passwd")


def test_resolve_cache_dir_rejects_absolute_outside(sandbox):
    """An absolute path outside the sandbox is rejected by commonpath."""
    with pytest.raises(InvalidExportPathError, match="outside sandbox"):
        resolve_cache_dir("/tmp/anywhere-else")


def test_resolve_cache_dir_rejects_symlink_escape(sandbox):
    """A symlink whose realpath leaves the sandbox is rejected.

    ``os.path.realpath`` follows the link to ``outside_dir``, and the
    subsequent ``commonpath`` check sees the result is no longer a
    descendant of the sandbox root. Without realpath the literal path
    ``sandbox/escape/anything`` would look safe — this is the case
    that justifies the realpath step.
    """
    sandbox.mkdir(parents=True, exist_ok=True)
    outside = sandbox.parent / "outside_dir"
    outside.mkdir()
    link = sandbox / "escape"
    link.symlink_to(outside)

    with pytest.raises(InvalidExportPathError, match="outside sandbox"):
        resolve_cache_dir("escape/anything")


# ---------------------------------------------------------------------------
# protocol.Manifest — roundtrip + additive evolution
# ---------------------------------------------------------------------------


def test_manifest_roundtrip(tmp_path):
    """``write_manifest`` then ``read_manifest`` recovers every field."""
    original = Manifest(
        protocol_version=PROTOCOL_VERSION,
        model_id="mlx-community/Qwen3.5-9B-4bit",
        quantization="4bit",
        paged_cache=True,
        turboquant_kv=False,
        index_format_version=2,
        entries=42,
        total_bytes=12_345_678,
        rapid_mlx_version="0.7.29",
        created_at="2026-06-18T00:00:00Z",
    )
    write_manifest(tmp_path, original)
    recovered = read_manifest(tmp_path)
    assert recovered == original


def test_write_manifest_failed_rename_preserves_prior_manifest(tmp_path, monkeypatch):
    """A crash mid-rename must not corrupt the prior manifest.

    Atomic write idiom: write tmp → fsync → ``os.replace``. If ``replace``
    fails (here we monkeypatch it to ValueError), the prior manifest.json
    must be untouched and the tmp file cleaned up. Without this the
    next ``read_manifest`` would 400 against a truncated file even
    though a valid one existed before.
    """
    original = Manifest(model_id="qwen3.5-9b-4bit", entries=18)
    write_manifest(tmp_path, original)
    assert (tmp_path / "manifest.json").is_file()

    import vllm_mlx.cache.protocol as protocol_mod

    def _boom(*args, **kwargs):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(protocol_mod.os, "replace", _boom)

    with pytest.raises(OSError, match="simulated rename failure"):
        write_manifest(tmp_path, Manifest(model_id="will-not-land", entries=99))

    # Prior manifest intact: same fields, no truncation.
    recovered = read_manifest(tmp_path)
    assert recovered.model_id == "qwen3.5-9b-4bit"
    assert recovered.entries == 18

    # No temp file left behind.
    leaked = list(tmp_path.glob(".manifest-*.json"))
    assert leaked == [], f"leaked temp files: {leaked}"


def test_read_manifest_rejects_invalid_json(tmp_path):
    """Malformed JSON at the manifest path surfaces a typed exception.

    Without this branch the JSONDecodeError would propagate as a 500 in
    the routes — a caller-controlled bug masquerading as a server fault.
    """
    (tmp_path / "manifest.json").write_text("not even close to JSON {")
    with pytest.raises(MalformedManifestError, match="not valid JSON"):
        read_manifest(tmp_path)


def test_read_manifest_rejects_non_object_payload(tmp_path):
    """A JSON list at the manifest path is structurally malformed."""
    (tmp_path / "manifest.json").write_text('["this", "is", "a", "list"]')
    with pytest.raises(MalformedManifestError, match="JSON object"):
        read_manifest(tmp_path)


def test_manifest_from_dict_rejects_wrong_type(tmp_path):
    """A known field with the wrong JSON type → MalformedManifestError.

    Codex round-3 BLOCKING: ``"entries": "not-an-int"`` previously
    constructed the dataclass blindly, so a peer could serve a manifest
    that violated its own advertised schema and the route would return
    200 anyway. Now each known field's value is checked against its
    expected Python type at read time.
    """
    (tmp_path / "manifest.json").write_text(
        json.dumps({"protocol_version": "1", "entries": "not-an-int"})
    )
    with pytest.raises(MalformedManifestError, match="entries"):
        read_manifest(tmp_path)


def test_manifest_from_dict_rejects_bool_for_int_field(tmp_path):
    """``isinstance(True, int)`` is True in Python — but JSON ``true`` is
    clearly not the integer 1. The strict check rejects this."""
    (tmp_path / "manifest.json").write_text(
        json.dumps({"protocol_version": "1", "entries": True})
    )
    with pytest.raises(MalformedManifestError, match="entries"):
        read_manifest(tmp_path)


def test_manifest_from_dict_rejects_string_for_bool_field(tmp_path):
    """``"paged_cache": "yes"`` is structurally wrong even if intuitive."""
    (tmp_path / "manifest.json").write_text(
        json.dumps({"protocol_version": "1", "paged_cache": "yes"})
    )
    with pytest.raises(MalformedManifestError, match="paged_cache"):
        read_manifest(tmp_path)


def test_manifest_from_dict_drops_unknown_fields(tmp_path):
    """An older reader handling a newer writer's extra fields just ignores them."""
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "model_id": "x",
        "future_field_v2": "something the current reader doesn't know about",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(payload))
    m = read_manifest(tmp_path)
    assert m.model_id == "x"
    assert m.protocol_version == PROTOCOL_VERSION


# ---------------------------------------------------------------------------
# auth — every route requires the bearer when --api-key is set
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("post", "/v1/cache/export", {}),
        ("post", "/v1/cache/import", {"source": "anywhere"}),
        ("get", "/v1/cache/info", None),
    ],
)
def test_routes_require_auth(cache_client, method, path, body):
    """No bearer → 401 on every route."""
    client = cache_client.client
    if method == "post":
        resp = client.post(path, json=body)
    else:
        resp = client.get(path)
    assert resp.status_code == 401, resp.text


def test_info_requires_auth_even_with_valid_manifest(cache_client):
    """An unauthenticated ``GET /v1/cache/info`` against a path with a
    valid manifest must still return 401, not 200.

    Codex round-3 NIT: the parametrized auth check uses an empty default
    path, where auth-fires-before-handler is indistinguishable from
    auth-fires-after-handler by 404 vs 401 ordering. With a real manifest
    in place, a bypassed auth dependency would surface as a 200 — this
    test catches that exact regression.
    """
    _write_export_root(
        cache_client.sandbox,
        "valid",
        Manifest(protocol_version=PROTOCOL_VERSION, model_id="x", entries=1),
    )
    resp = cache_client.client.get("/v1/cache/info?path=valid")
    assert resp.status_code == 401


def test_routes_reject_wrong_bearer(cache_client):
    resp = cache_client.client.post(
        "/v1/cache/export",
        json={},
        headers={"Authorization": "Bearer wrong"},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# /v1/cache/export — engine-backed 200 after passing the sandbox check
# ---------------------------------------------------------------------------


def test_export_default_destination_returns_200(cache_client):
    """No destination → uses sandbox root, calls the engine, returns a
    caller-oriented summary. Resolved destination must NOT appear in the
    response body (sibling concern to F-180 — no operator-home-dir leak)."""
    # Fake engine with a small non-empty cache so bytes/entries are non-zero.
    cache_client.cfg.engine = cache_client.FakeEngine(
        entries=3, current_memory=2048, kv_cache_dtype="int8"
    )
    resp = cache_client.client.post("/v1/cache/export", json={}, headers=_auth())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["protocol_version"] == PROTOCOL_VERSION
    assert body["entries_exported"] == 3
    assert body["bytes_written"] == 2048
    assert body["quantization"] == "int8"
    assert body["manifest_path"] == "manifest.json"
    # The engine's save was actually invoked with the resolved destination.
    assert cache_client.cfg.engine.saved_to is not None

    # H-02/H-12: the resolved sandbox path must NOT ride the wire. Only the
    # caller-relative ``manifest.json`` filename is echoed.
    serialized = resp.text
    sandbox_real = str(Path(cache_client.sandbox).resolve())
    assert sandbox_real not in serialized


def test_export_empty_cache_returns_200_with_zero_entries(cache_client):
    """A prefix cache that's disabled (None) still exports a valid empty
    snapshot — 200 with entries_exported=0 and a manifest on disk."""
    cache_client.cfg.engine = cache_client.FakeEngine(prefix_cache=False)
    resp = cache_client.client.post(
        "/v1/cache/export", json={"destination": "empty-snap"}, headers=_auth()
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["entries_exported"] == 0
    assert body["bytes_written"] == 0
    # Manifest was written under the resolved destination.
    manifest_root = Path(cache_client.sandbox).resolve() / "empty-snap"
    assert (manifest_root / "manifest.json").is_file()


def test_export_engine_not_loaded_returns_503(cache_client):
    """No model loaded (``cfg.engine is None``) → 503, matching the
    ``/v1/cache/clear`` idiom in routes.health."""
    cache_client.cfg.engine = None
    resp = cache_client.client.post("/v1/cache/export", json={}, headers=_auth())
    assert resp.status_code == 503, resp.text
    assert resp.json()["detail"] == "engine not loaded"


def test_export_over_max_bytes_returns_413(cache_client):
    """A cache whose in-memory footprint exceeds ``max_bytes`` is rejected
    with 413 BEFORE any write — the engine's save is never called."""
    engine = cache_client.FakeEngine(entries=5, current_memory=10_000)
    cache_client.cfg.engine = engine
    resp = cache_client.client.post(
        "/v1/cache/export",
        json={"destination": "too-big", "max_bytes": 4096},
        headers=_auth(),
    )
    assert resp.status_code == 413, resp.text
    # save_cache_to_disk must NOT have run (no write started).
    assert engine.saved_to is None
    # Nothing written to disk either.
    assert not (Path(cache_client.sandbox).resolve() / "too-big").exists()


def test_export_under_max_bytes_returns_200(cache_client):
    """Footprint at/under the cap exports normally."""
    engine = cache_client.FakeEngine(entries=2, current_memory=4096)
    cache_client.cfg.engine = engine
    resp = cache_client.client.post(
        "/v1/cache/export",
        json={"destination": "fits", "max_bytes": 4096},
        headers=_auth(),
    )
    assert resp.status_code == 200, resp.text
    assert engine.saved_to is not None


# ---------------------------------------------------------------------------
# REGRESSION: production BatchedEngine nesting (real-hardware smoke bug)
#
# BatchedEngine has NO top-level ``.scheduler`` — the cache lives at
# ``engine._engine.engine.scheduler.memory_aware_cache``. The routes reached
# it via ``getattr(engine, "scheduler")`` which returned None under the prod
# engine, so: manifest fields collapsed to 0/""; the max_bytes 413 gate read
# _current_memory=0 (inert); merge_strategy="replace" never cleared. These
# tests drive the SAME handlers with the nested engine shape to lock the fix.
# ---------------------------------------------------------------------------


def test_export_nested_engine_reports_real_entries(cache_client):
    """The bug: export with a BatchedEngine-shaped engine reported
    entries_exported=0 despite a populated nested cache. Now it must see the
    real ledger + cache-config off ``engine._engine.engine.scheduler``."""
    engine = cache_client.NestedFakeEngine(
        entries=70,
        current_memory=1_400_000_000,
        kv_cache_dtype="int4",
        use_paged_cache=True,
        kv_cache_turboquant=True,
    )
    # Sanity: the engine really does NOT expose a top-level scheduler.
    assert not hasattr(engine, "scheduler")
    cache_client.cfg.engine = engine
    resp = cache_client.client.post(
        "/v1/cache/export", json={"destination": "nested"}, headers=_auth()
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["entries_exported"] == 70  # was 0 pre-fix
    assert body["bytes_written"] == 1_400_000_000  # was 0 pre-fix
    assert body["quantization"] == "int4"  # was "" pre-fix
    assert body["paged_cache"] is True  # was False pre-fix
    assert body["turboquant_kv"] is True  # was False pre-fix
    assert engine.saved_to is not None


def test_export_nested_engine_max_bytes_gate_fires(cache_client):
    """The bug: with the cache seen as None, ``_current_memory`` read 0, so a
    ``max_bytes:1`` export was NOT rejected (a second 1.4 GB blob got
    written — H-04 gate inert). Now the 413 fires from the real footprint."""
    engine = cache_client.NestedFakeEngine(entries=70, current_memory=1_400_000_000)
    cache_client.cfg.engine = engine
    resp = cache_client.client.post(
        "/v1/cache/export",
        json={"destination": "nested-big", "max_bytes": 1},
        headers=_auth(),
    )
    assert resp.status_code == 413, resp.text
    assert engine.saved_to is None  # write never started
    assert not (Path(cache_client.sandbox).resolve() / "nested-big").exists()


def test_import_nested_engine_replace_clears_real_cache(cache_client):
    """The bug: replace never cleared under the nested engine (cache None), so
    a re-import collided all entries as duplicates and loaded 0. Now
    ``clear()`` runs on the REAL nested cache before load."""
    _write_export_root(
        cache_client.sandbox,
        "nested-ready",
        Manifest(protocol_version=PROTOCOL_VERSION, model_id="test-model", entries=70),
    )
    engine = cache_client.NestedFakeEngine(
        entries=70, current_memory=1_400_000_000, load_returns=70
    )
    cache_client.cfg.engine = engine
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={
            "source": "nested-ready",
            "expected_model_id": "test-model",
            "merge_strategy": "replace",
        },
        headers=_auth(),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["entries_loaded"] == 70
    # The REAL nested cache was cleared exactly once before the load.
    nested_cache = engine._engine.engine.scheduler.memory_aware_cache
    assert nested_cache.clear_calls == 1


def test_build_manifest_nested_engine(cache_client):
    """Unit-level: ``build_manifest_from_engine_state`` unwraps the nested
    BatchedEngine directly (not just through the HTTP handler)."""
    from vllm_mlx.cache.protocol import build_manifest_from_engine_state

    engine = cache_client.NestedFakeEngine(
        entries=12,
        current_memory=999,
        kv_cache_dtype="int8",
        use_paged_cache=True,
    )
    m = build_manifest_from_engine_state(engine)
    assert m.entries == 12
    assert m.total_bytes == 999
    assert m.quantization == "int8"
    assert m.paged_cache is True


def test_export_rejects_path_traversal(cache_client):
    resp = cache_client.client.post(
        "/v1/cache/export",
        json={"destination": "../../../etc"},
        headers=_auth(),
    )
    assert resp.status_code == 403
    # H-02: 403 body is sanitized — error.code identifies the failure
    # mode, the caller-supplied path stays in server logs only.
    detail = resp.json()["detail"]
    assert detail["error"]["code"] == "sandbox_escape"
    assert detail["error"]["type"] == "invalid_request_error"


def test_export_rejects_absolute_outside(cache_client):
    resp = cache_client.client.post(
        "/v1/cache/export",
        json={"destination": "/tmp/escape-target"},
        headers=_auth(),
    )
    assert resp.status_code == 403


def test_export_rejects_invalid_max_bytes(cache_client):
    """pydantic catches the ge=1 violation as 422 before the handler runs."""
    resp = cache_client.client.post(
        "/v1/cache/export",
        json={"max_bytes": 0},
        headers=_auth(),
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /v1/cache/import — manifest mismatches surface as 409 before engine work
# ---------------------------------------------------------------------------


def _write_export_root(sandbox: Path, name: str, manifest: Manifest) -> Path:
    root = sandbox / name
    root.mkdir(parents=True, exist_ok=True)
    write_manifest(root, manifest)
    return root


def test_import_malformed_manifest_returns_400(cache_client):
    """Corrupt manifest.json at the source → 400, not 500.

    Without the dedicated mapping in ``_read_manifest_or_http``, the
    underlying ``json.JSONDecodeError`` would escape and FastAPI would
    surface it as an opaque 500 — hiding a caller-supplied bad blob
    inside a server-fault status. Codex blocking-finding regression.
    """
    bad = cache_client.sandbox / "corrupt"
    bad.mkdir(parents=True)
    (bad / "manifest.json").write_text("{ not valid json")
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={"source": "corrupt"},
        headers=_auth(),
    )
    assert resp.status_code == 400
    assert "not valid JSON" in resp.json()["detail"]


def test_info_malformed_manifest_returns_400(cache_client):
    bad = cache_client.sandbox / "corrupt-info"
    bad.mkdir(parents=True)
    (bad / "manifest.json").write_text('"a bare JSON string is not an object"')
    resp = cache_client.client.get(
        "/v1/cache/info?path=corrupt-info",
        headers=_auth(),
    )
    assert resp.status_code == 400
    assert "JSON object" in resp.json()["detail"]


def test_info_400_detail_does_not_leak_resolved_path(cache_client):
    """The 400 body must not include the server's resolved cache root.

    Codex round-3 NIT: leaking ``/Users/raullen/.cache/rapid-mlx/...`` to
    any bearer-token holder is unnecessary information disclosure.
    """
    bad = cache_client.sandbox / "leak-probe"
    bad.mkdir(parents=True)
    (bad / "manifest.json").write_text("{ syntax error here")
    resp = cache_client.client.get(
        "/v1/cache/info?path=leak-probe",
        headers=_auth(),
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert str(cache_client.sandbox) not in detail
    assert "/" not in detail or "JSON" in detail  # may mention syntax but no path


def test_info_404_detail_does_not_leak_resolved_path(cache_client):
    """The 404 body must not include the server's resolved cache root."""
    (cache_client.sandbox / "no-such").mkdir(parents=True)
    resp = cache_client.client.get(
        "/v1/cache/info?path=no-such",
        headers=_auth(),
    )
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert str(cache_client.sandbox) not in detail
    assert detail == "no manifest.json at the requested cache path"


def test_import_missing_manifest_returns_404(cache_client):
    """Source path exists but has no manifest.json."""
    (cache_client.sandbox / "no-manifest").mkdir(parents=True)
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={"source": "no-manifest"},
        headers=_auth(),
    )
    assert resp.status_code == 404


def test_import_protocol_version_mismatch_returns_409(cache_client):
    _write_export_root(
        cache_client.sandbox,
        "v999",
        Manifest(protocol_version="999", model_id="any"),
    )
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={"source": "v999", "expected_protocol_version": PROTOCOL_VERSION},
        headers=_auth(),
    )
    assert resp.status_code == 409
    assert "protocol_version" in resp.json()["detail"]


def test_import_model_id_mismatch_returns_409(cache_client):
    _write_export_root(
        cache_client.sandbox,
        "qwen",
        Manifest(protocol_version=PROTOCOL_VERSION, model_id="qwen3.5-9b-4bit"),
    )
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={
            "source": "qwen",
            "expected_model_id": "gpt-oss-20b-mxfp4-q8",
        },
        headers=_auth(),
    )
    assert resp.status_code == 409
    assert "model_id" in resp.json()["detail"]


def test_import_kv_geometry_quant_mismatch_returns_409(cache_client):
    """#1100 codex round 9 (#1): matching model_id is necessary but NOT
    sufficient — a blob exported under a different KV ``quantization``
    (``kv_cache_dtype``) is byte-incompatible with the loaded engine even for
    the SAME model, so it must 409. The gate reads the engine's geometry via the
    SAME helper the manifest builder uses."""
    _write_export_root(
        cache_client.sandbox,
        "geo",
        Manifest(
            protocol_version=PROTOCOL_VERSION,
            model_id="test-model",
            quantization="int4",  # blob was exported under int4 KV cache
            entries=1,
        ),
    )
    # Engine loaded with a DIFFERENT KV dtype (bf16) — incompatible geometry.
    engine = cache_client.FakeEngine(
        entries=1, current_memory=100, load_returns=1, kv_cache_dtype="bf16"
    )
    engine.config.model_name = "test-model"
    cache_client.cfg.model_name = "test-model"
    cache_client.cfg.engine = engine
    resp = cache_client.client.post(
        "/v1/cache/import", json={"source": "geo"}, headers=_auth()
    )
    assert resp.status_code == 409, resp.text
    assert "kv_cache_dtype" in resp.json()["detail"]
    assert engine.loaded_from is None  # gate fired before load

    # Same geometry (both int4) → the gate passes (load runs, 200).
    engine2 = cache_client.FakeEngine(
        entries=1, current_memory=100, load_returns=1, kv_cache_dtype="int4"
    )
    engine2.config.model_name = "test-model"
    cache_client.cfg.engine = engine2
    ok = cache_client.client.post(
        "/v1/cache/import", json={"source": "geo"}, headers=_auth()
    )
    assert ok.status_code == 200, ok.text
    assert engine2.loaded_from is not None


def test_import_legacy_manifest_without_quant_still_imports(cache_client):
    """#1100 codex round 9 (#1): the geometry gate must NOT break importing an
    older-but-valid export. A legacy manifest predating the ``quantization``
    field carries ``quantization=""`` (unknown); unknown-vs-known is treated as
    unverifiable-yet-permitted (the model_id + protocol gates already passed) so
    the manifest contract stays additive/back-compatible."""
    _write_export_root(
        cache_client.sandbox,
        "legacy",
        Manifest(
            protocol_version=PROTOCOL_VERSION,
            model_id="test-model",
            quantization="",  # legacy: field absent → empty
            entries=1,
        ),
    )
    engine = cache_client.FakeEngine(
        entries=1, current_memory=100, load_returns=1, kv_cache_dtype="bf16"
    )
    engine.config.model_name = "test-model"
    cache_client.cfg.model_name = "test-model"
    cache_client.cfg.engine = engine
    resp = cache_client.client.post(
        "/v1/cache/import", json={"source": "legacy"}, headers=_auth()
    )
    assert resp.status_code == 200, resp.text
    assert engine.loaded_from is not None


def test_import_validated_request_returns_200(cache_client):
    """All wire checks pass → engine.load_cache_from_disk runs, 200 summary.

    ``entries_skipped`` derives from ``manifest.entries - entries_loaded``.
    Here the manifest claims 18 entries and the fake loader returns 15, so
    3 were dropped (per-entry validation) → entries_skipped == 3."""
    manifest = Manifest(
        protocol_version=PROTOCOL_VERSION,
        model_id="qwen3.5-9b-4bit",
        entries=18,
        total_bytes=4_096_000,
    )
    _write_export_root(cache_client.sandbox, "ready", manifest)
    # The server must be running the SAME model the manifest was exported
    # from, else the #1100 BLOCKING-1 unconditional gate 409s before load.
    cache_client.cfg.model_name = "qwen3.5-9b-4bit"
    # ``loaded_bytes`` simulates the footprint the load hydrates; under
    # "replace" the cache is cleared first so the post-load footprint IS the
    # loaded bytes → the route reports bytes_loaded == loaded_bytes.
    engine = cache_client.FakeEngine(
        entries=2, current_memory=99, load_returns=15, loaded_bytes=4_096_000
    )
    cache_client.cfg.engine = engine
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={
            "source": "ready",
            "expected_model_id": "qwen3.5-9b-4bit",
            "merge_strategy": "replace",
        },
        headers=_auth(),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["protocol_version"] == PROTOCOL_VERSION
    assert body["entries_loaded"] == 15
    assert body["entries_skipped"] == 3  # 18 claimed − 15 loaded
    # #1100 BLOCKING-5: bytes_loaded is the ACTUAL loaded footprint (replace
    # cleared first, so post-load footprint == loaded), not manifest.total.
    assert body["bytes_loaded"] == 4_096_000
    # The engine's load actually ran, with the resolved source dir.
    assert engine.loaded_from is not None
    # merge_strategy="replace" cleared the in-memory cache before loading.
    assert engine.scheduler.memory_aware_cache.clear_calls == 1


def test_import_replace_abort_reports_zero_bytes_loaded(cache_client):
    """#1100 BLOCKING-1 × BLOCKING-5 interaction: a ``replace`` that ABORTS on
    a corrupt entry blob returns 0 loaded WITHOUT clearing — so the preserved
    cache's footprint must NOT be reported as ``bytes_loaded``. Nothing loaded
    → bytes_loaded == 0 (the round-1 accounting reported ``after_bytes``,
    which is the untouched existing cache under an aborted replace)."""
    _write_export_root(
        cache_client.sandbox,
        "abort-src",
        Manifest(protocol_version=PROTOCOL_VERSION, model_id="test-model", entries=9),
    )

    # A fake whose load simulates a replace-abort: returns 0 and does NOT
    # clear — the existing cache footprint stays put.
    engine = cache_client.FakeEngine(entries=4, current_memory=5000)

    def _load_aborts(cache_dir, replace=False):
        engine.loaded_from = cache_dir
        engine.load_replace = replace
        # replace aborted on corruption: no clear, footprint unchanged, 0 loaded.
        return 0

    engine.load_cache_from_disk = _load_aborts
    cache_client.cfg.engine = engine
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={"source": "abort-src", "merge_strategy": "replace"},
        headers=_auth(),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["entries_loaded"] == 0
    # NOT the preserved 5000-byte cache footprint — nothing was loaded.
    assert body["bytes_loaded"] == 0
    # The existing cache was left intact (never cleared).
    assert engine.scheduler.memory_aware_cache.clear_calls == 0
    assert engine.scheduler.memory_aware_cache._current_memory == 5000


def test_import_merge_does_not_clear_cache(cache_client):
    """``merge_strategy="merge"`` (default) must NOT clear the in-memory
    cache — existing entries are kept and the new blob is layered on top."""
    _write_export_root(
        cache_client.sandbox,
        "ready-merge",
        Manifest(protocol_version=PROTOCOL_VERSION, model_id="test-model", entries=4),
    )
    engine = cache_client.FakeEngine(entries=2, load_returns=4)
    cache_client.cfg.engine = engine
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={"source": "ready-merge", "merge_strategy": "merge"},
        headers=_auth(),
    )
    assert resp.status_code == 200, resp.text
    assert engine.scheduler.memory_aware_cache.clear_calls == 0
    assert resp.json()["entries_loaded"] == 4


def test_import_entries_skipped_floored_at_zero(cache_client):
    """If the loader returns MORE entries than the manifest claimed (a
    merge that hydrated pre-existing on-disk entries too), entries_skipped
    floors at 0 rather than going negative."""
    _write_export_root(
        cache_client.sandbox,
        "ready-over",
        Manifest(protocol_version=PROTOCOL_VERSION, model_id="test-model", entries=3),
    )
    engine = cache_client.FakeEngine(load_returns=5)
    cache_client.cfg.engine = engine
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={"source": "ready-over"},
        headers=_auth(),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["entries_skipped"] == 0


def test_import_rejects_mismatched_model_without_expected_id(cache_client):
    """#1100 BLOCKING-1: the unconditional model-id gate can NOT be bypassed
    by omitting ``expected_model_id``. A manifest whose model_id differs from
    the server's loaded model is rejected (409) even with no caller-side
    expectation — loading another model's KV geometry would corrupt
    inference. The load must not run, and the 409 must not leak the server's
    model id."""
    _write_export_root(
        cache_client.sandbox,
        "wrong-model",
        Manifest(
            protocol_version=PROTOCOL_VERSION,
            model_id="some-other-model-70b",  # != server "test-model"
            entries=5,
        ),
    )
    engine = cache_client.FakeEngine(entries=3, current_memory=500, load_returns=5)
    cache_client.cfg.engine = engine
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={"source": "wrong-model"},  # NO expected_model_id — must still 409
        headers=_auth(),
    )
    assert resp.status_code == 409, resp.text
    # The gate fires before the load — load never ran.
    assert engine.loaded_from is None
    # 409 detail must not name either model id (server or manifest).
    assert "test-model" not in resp.text
    assert "some-other-model-70b" not in resp.text


def test_import_gate_uses_engine_config_when_server_singleton_empty(cache_client):
    """#1100 BLOCKING-2: the model-id gate must derive the loaded model's id
    the SAME way the manifest builder does — server singleton FIRST, then
    ``engine.config.model_name``. For an embedded engine the server singleton's
    ``model_name`` is empty, but the engine's own config carries a real id.
    The round-1 gate read only the empty singleton and skipped the check,
    letting an embedded engine import ANOTHER model's KV state. Here the
    singleton is empty and the engine.config id is ``embedded/model-a``; a
    manifest for ``embedded/model-b`` must still 409 (load never runs)."""
    # Server singleton empty (embedded engine that never populated ServerConfig).
    cache_client.cfg.model_name = ""
    _write_export_root(
        cache_client.sandbox,
        "embedded-wrong",
        Manifest(
            protocol_version=PROTOCOL_VERSION,
            model_id="embedded/model-b",  # != engine.config "embedded/model-a"
            entries=4,
        ),
    )
    engine = cache_client.FakeEngine(entries=2, current_memory=100, load_returns=4)
    engine.config.model_name = "embedded/model-a"
    cache_client.cfg.engine = engine
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={"source": "embedded-wrong"},  # NO expected_model_id
        headers=_auth(),
    )
    assert resp.status_code == 409, resp.text
    assert engine.loaded_from is None  # gate fired — load never ran
    # No-leak discipline: neither model id in the 409 body.
    assert "embedded/model-a" not in resp.text
    assert "embedded/model-b" not in resp.text


def test_import_gate_engine_config_match_allows_load(cache_client):
    """Complement to the gate test: when the server singleton is empty but the
    manifest's model_id MATCHES the engine.config id, the import proceeds
    (proving the fallback isn't over-rejecting — it resolves the right id)."""
    cache_client.cfg.model_name = ""
    _write_export_root(
        cache_client.sandbox,
        "embedded-ok",
        Manifest(
            protocol_version=PROTOCOL_VERSION,
            model_id="embedded/model-a",  # == engine.config
            entries=4,
        ),
    )
    engine = cache_client.FakeEngine(entries=0, load_returns=4, loaded_bytes=10)
    engine.config.model_name = "embedded/model-a"
    cache_client.cfg.engine = engine
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={"source": "embedded-ok", "merge_strategy": "merge"},
        headers=_auth(),
    )
    assert resp.status_code == 200, resp.text
    assert engine.loaded_from is not None  # gate passed — load ran
    assert resp.json()["entries_loaded"] == 4


def test_export_failed_save_of_nonempty_cache_returns_500(cache_client):
    """#1100 BLOCKING-3 + codex round 3 (#2): a non-empty cache whose save
    commits NOTHING is a FAILURE (500). A real failing ``save_to_disk`` uses
    its own ``.new``/``.old`` staging and leaves the PUBLISHED destination
    untouched — so the route must NOT ``_discard_export`` the destination
    (that would destroy a previously-valid snapshot the failed save never
    modified). It sweeps only the orphaned staging siblings. Here a prior
    valid snapshot sits at the destination and the save fails leaving a
    ``.new`` staging dir; the route 500s, sweeps ``.new``, and PRESERVES the
    prior published snapshot."""
    from pathlib import Path as _Path

    engine = cache_client.FakeEngine(entries=5, current_memory=2048)
    blob_dir = _Path(cache_client.sandbox) / "failed-save"
    # A previously-valid published snapshot at the destination.
    blob_dir.mkdir(parents=True, exist_ok=True)
    (blob_dir / "index.json").write_text('{"version":3,"entries":[]}')
    (blob_dir / "manifest.json").write_text('{"protocol_version":"1"}')

    def _save_commits_nothing(cache_dir, should_abort=None):
        # Real failing save: writes into <dest>.new, never renames onto <dest>,
        # then aborts — leaving the staging dir but not touching the published
        # destination.
        engine.saved_to = cache_dir
        staging = _Path(str(cache_dir) + ".new")
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "index.json").write_text("{}")  # orphaned staging artifact
        # #1100 codex round 4 (#1): the real ``save_to_disk`` records "failed"
        # on the cache when a non-empty cache commits nothing. The route reads
        # THIS (not a pre-write len snapshot) to classify the outcome.
        cache = engine.scheduler.memory_aware_cache
        if cache is not None:
            cache._last_save_outcome = "failed"
        return False  # nothing committed

    engine.save_cache_to_disk = _save_commits_nothing
    cache_client.cfg.engine = engine
    resp = cache_client.client.post(
        "/v1/cache/export",
        json={"destination": "failed-save"},
        headers=_auth(),
    )
    assert resp.status_code == 500, resp.text
    # The prior published snapshot is PRESERVED (the failed save never touched
    # it — discarding it would be data loss, codex round 3 #2).
    assert (blob_dir / "index.json").exists()
    assert (blob_dir / "manifest.json").exists()
    # The orphaned staging dir was swept.
    assert not _Path(str(blob_dir) + ".new").exists()


def test_export_classifies_failed_save_by_authoritative_outcome(cache_client):
    """#1100 codex round 4 (#1): the export route classifies the save result by
    the AUTHORITATIVE ``_last_save_outcome`` the op records, NOT by a pre-write
    ``len(cache)`` snapshot that a racing store/evict could skew. Here the
    op reports 'failed' (a non-empty cache that committed nothing) while the
    route is given NO pre-write len discriminator at all — the classification
    must come purely from the recorded outcome (→ 500 failed save), proving the
    route no longer depends on the racy snapshot it used to sample."""
    engine = cache_client.FakeEngine(entries=2, current_memory=999)

    def _save_fails(cache_dir, should_abort=None):
        # Authoritative outcome: had entries but committed nothing.
        engine.saved_to = cache_dir
        cache = engine.scheduler.memory_aware_cache
        cache._last_save_outcome = "failed"
        return False

    engine.save_cache_to_disk = _save_fails
    cache_client.cfg.engine = engine
    resp = cache_client.client.post(
        "/v1/cache/export",
        json={"destination": "outcome-failed"},
        headers=_auth(),
    )
    # Trusted the authoritative 'failed' outcome → 500.
    assert resp.status_code == 500, resp.text
    assert "commit" in resp.json()["detail"].lower()


def test_export_empty_cache_save_false_is_200_not_error(cache_client):
    """The flip side of BLOCKING-3: a genuinely-EMPTY cache whose save returns
    False is legitimate — 200 with zero counts, NOT a 500. The default
    ``_FakeEngine`` with 0 entries returns False from ``save_cache_to_disk``;
    the route must treat that as an empty export, not a failure."""
    engine = cache_client.FakeEngine(entries=0, current_memory=0)
    cache_client.cfg.engine = engine
    resp = cache_client.client.post(
        "/v1/cache/export",
        json={"destination": "empty-ok"},
        headers=_auth(),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["entries_exported"] == 0


def test_export_empty_cache_clears_stale_prior_snapshot(cache_client):
    """#1100 codex round 3 (#1): an EMPTY export to a destination that already
    holds a STALE prior snapshot must NOT silently re-export the old entries.
    ``build_manifest_from_engine_state`` reads counts from the committed
    ``index.json`` — a leftover one would make an 'empty' export report the
    old blob. The route clears the destination's blob artifacts first so the
    manifest honestly reports 0/0 and no importable stale entry files remain."""
    from pathlib import Path as _Path

    blob_dir = _Path(cache_client.sandbox) / "stale-then-empty"
    blob_dir.mkdir(parents=True, exist_ok=True)
    # A stale prior snapshot: index.json claiming 7 entries + entry files.
    stale_index = {
        "version": 3,
        "entries": [
            {
                "index": i,
                "num_tokens": 4,
                "memory_bytes": 100,
                "cache_types": ["KVCache"],
            }
            for i in range(7)
        ],
    }
    (blob_dir / "index.json").write_text(json.dumps(stale_index))
    for i in range(7):
        (blob_dir / f"entry_{i}.safetensors").write_text("x")
        (blob_dir / f"entry_{i}_tokens.bin").write_text("x")
    (blob_dir / "manifest.json").write_text('{"protocol_version":"1","entries":7}')

    # Empty cache → save_cache_to_disk returns False (nothing to commit).
    engine = cache_client.FakeEngine(entries=0, current_memory=0)
    cache_client.cfg.engine = engine
    resp = cache_client.client.post(
        "/v1/cache/export",
        json={"destination": "stale-then-empty"},
        headers=_auth(),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Honest empty export — NOT the stale 7 entries.
    assert body["entries_exported"] == 0
    assert body["bytes_written"] == 0
    # The stale entry files are gone (not importable).
    assert not (blob_dir / "index.json").exists()
    assert not (blob_dir / "entry_0.safetensors").exists()
    # A fresh empty manifest was written.
    assert (blob_dir / "manifest.json").is_file()
    fresh = read_manifest(blob_dir)
    assert fresh.entries == 0


def test_export_manifest_write_failure_discards_committed_blob(
    cache_client, monkeypatch
):
    """#1100 codex round 3 (#3): if post-save processing (manifest write /
    size gate) raises after a multi-GB snapshot commits — e.g. ENOSPC — the
    just-committed blob must be DISCARDED before the error propagates, not
    orphaned to worsen disk exhaustion on retry."""
    from pathlib import Path as _Path

    import vllm_mlx.routes.cache as cache_mod

    engine = cache_client.FakeEngine(entries=2, current_memory=100)

    def _save_commits(cache_dir, should_abort=None):
        engine.saved_to = cache_dir
        d = _Path(cache_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.json").write_text(
            '{"version":3,"entries":[{"index":0,"num_tokens":4,'
            '"memory_bytes":50,"cache_types":["KVCache"]}]}'
        )
        (d / "entry_0.safetensors").write_text("x")
        (d / "entry_0_tokens.bin").write_text("x")
        cache = engine.scheduler.memory_aware_cache
        if cache is not None:
            cache._last_save_outcome = "committed"
        return True

    engine.save_cache_to_disk = _save_commits
    cache_client.cfg.engine = engine

    # write_manifest raises (simulate ENOSPC after the snapshot committed).
    def _boom_write(*a, **k):
        raise OSError("simulated ENOSPC writing manifest")

    monkeypatch.setattr(cache_mod, "write_manifest", _boom_write)

    resp = cache_client.client.post(
        "/v1/cache/export",
        json={"destination": "enospc"},
        headers=_auth(),
    )
    assert resp.status_code == 500, resp.text
    # The committed blob was discarded — not orphaned.
    blob_dir = _Path(cache_client.sandbox) / "enospc"
    assert not (blob_dir / "index.json").exists()
    assert not (blob_dir / "entry_0.safetensors").exists()


def test_import_fails_closed_when_engine_model_id_unresolvable(cache_client):
    """#1100 codex round 3 (#4) → round 7 (#1): the model-identity gate must
    FAIL CLOSED — HARD — when the loaded engine's model id can't be resolved.

    Round 3 rejected an id-less engine UNLESS the caller pinned a matching
    ``expected_model_id``. Codex round 7 (#1) flagged that as still-open: BOTH
    ``expected_model_id`` AND the manifest's ``model_id`` are caller-controlled
    and untrusted, so comparing one against the other proves NOTHING about the
    geometry of the model the server actually loaded — a caller can echo the
    manifest id it shipped and slip an incompatible KV blob past the gate.
    There is no trusted server-side signal to compare against when the engine
    can't identify itself, so the import is now rejected UNCONDITIONALLY (422).
    The remediation is operational: give the engine a resolvable model id.
    """
    cache_client.cfg.model_name = ""  # server singleton empty
    _write_export_root(
        cache_client.sandbox,
        "idless",
        Manifest(
            protocol_version=PROTOCOL_VERSION, model_id="foreign/model", entries=3
        ),
    )
    # Engine whose own config also has no resolvable id.
    engine = cache_client.FakeEngine(entries=1, load_returns=3)
    engine.config.model_name = None
    engine.config.model_path = None
    cache_client.cfg.engine = engine
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={"source": "idless"},  # NO expected_model_id
        headers=_auth(),
    )
    assert resp.status_code == 422, resp.text
    assert engine.loaded_from is None  # load never ran

    # Round 7 (#1): echoing the manifest's own model_id via expected_model_id
    # does NOT open the gate — both values are caller-controlled, so the server
    # still has no trustworthy identity assertion. Reject unconditionally.
    resp2 = cache_client.client.post(
        "/v1/cache/import",
        json={"source": "idless", "expected_model_id": "foreign/model"},
        headers=_auth(),
    )
    assert resp2.status_code == 422, resp2.text
    assert engine.loaded_from is None  # load still never ran

    # The correct fix is operational: once the engine has a resolvable id that
    # MATCHES the manifest, the server can make the identity assertion itself
    # and the import proceeds. (Model id lives on ``engine.config.model_name``,
    # read via ``resolve_engine_model_id`` when the server singleton is empty.)
    engine.config.model_name = "foreign/model"
    resp3 = cache_client.client.post(
        "/v1/cache/import",
        json={"source": "idless"},
        headers=_auth(),
    )
    assert resp3.status_code == 200, resp3.text
    assert engine.loaded_from is not None


@pytest.mark.asyncio
async def test_export_concurrent_same_destination_serialized(cache_client):
    """#1100 codex round 3 (#5): concurrent exports to the SAME destination
    must be serialized so one request never writes a manifest for another's
    snapshot or discards a newer one. Fire N concurrent exports at one
    destination on a SINGLE event loop (exactly how uvicorn schedules them in
    production — not raw OS threads) and assert (a) each returns a coherent
    200, and (b) the per-destination lock actually serialized the critical
    section (no two save→publish transactions overlapped)."""
    import asyncio as _asyncio

    # Instrument the save to detect overlap: the save runs in a threadpool
    # (anyio.to_thread), so WITHOUT the per-destination lock multiple saves to
    # the same destination would overlap (peak in-flight > 1). A short real
    # sleep widens the window; a threading.Lock-guarded counter records the
    # peak. With the route's per-destination lock, saves serialize → peak==1.
    import threading
    import time as _time

    import httpx
    from fastapi import FastAPI

    import vllm_mlx.routes.cache as cache_mod
    from vllm_mlx.routes.cache import router

    engine = cache_client.FakeEngine(entries=3, current_memory=1024)
    state = {"in_flight": 0, "peak": 0}
    counter_lock = threading.Lock()

    def _save(cache_dir, should_abort=None):
        with counter_lock:
            state["in_flight"] += 1
            state["peak"] = max(state["peak"], state["in_flight"])
        _time.sleep(0.05)  # widen the overlap window
        with counter_lock:
            state["in_flight"] -= 1
        engine.saved_to = cache_dir
        cache = engine.scheduler.memory_aware_cache
        if cache is not None:
            cache._last_save_outcome = "committed"
        # #1100 codex round 6 (#2): write the committed index (3 entries /
        # 1024 B, matching the ledger) the manifest builder now requires.
        engine._write_committed_index(cache_dir, entries=3, total_bytes=1024)
        return True

    engine.save_cache_to_disk = _save

    # #1100 codex round 7 (#5): the default ``_FakeEngine.save_cache_with_outcome``
    # wraps save+outcome-read in ``self._step_lock`` to mimic the real single
    # ``max_workers=1`` step thread. But that lock ALSO serializes concurrent
    # exports — so this test would observe ``peak == 1`` even if the ROUTE's
    # per-destination lock were deleted, i.e. it would prove nothing about the
    # thing it exists to verify. Install an UNSYNCHRONIZED outcome wrapper on
    # THIS engine (no ``_step_lock``) so the ONLY lock that can hold ``peak``
    # at 1 is the route's per-destination lock we are actually testing.
    def _unsync_save_with_outcome(cache_dir, should_abort=None):
        from vllm_mlx.cache.protocol import SaveOutcome

        engine.save_cache_to_disk(cache_dir, should_abort=should_abort)
        cache = engine.scheduler.memory_aware_cache
        outcome = getattr(cache, "_last_save_outcome", "empty") if cache else "empty"
        return SaveOutcome(outcome=outcome)

    engine.save_cache_with_outcome = _unsync_save_with_outcome
    cache_client.cfg.engine = engine

    # Reset the module-level per-destination lock registry so this loop owns
    # freshly-created locks (asyncio.Lock binds to the loop that first awaits
    # it; a lock cached from another test's loop would be foreign here).
    cache_mod._export_dest_locks.clear()
    cache_mod._export_locks_guard = _asyncio.Lock()

    app = FastAPI()
    app.include_router(router)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

        async def _do_export():
            return await client.post(
                "/v1/cache/export",
                json={"destination": "shared"},
                headers=_auth(),
            )

        responses = await _asyncio.gather(*[_do_export() for _ in range(6)])

    for r in responses:
        assert r.status_code == 200, r.text
        assert r.json()["entries_exported"] == 3
    # The per-destination lock serialized the save critical section — the
    # in-flight save count never exceeded 1.
    assert state["peak"] == 1, f"lock did not serialize; peak in-flight={state['peak']}"


@pytest.mark.asyncio
async def test_import_holds_dest_lock_against_concurrent_export(cache_client):
    """#1100 codex round 4 (#4) + round 6 (#5) + round 8 (#5): the import handler
    holds the SAME per-destination lock export uses, from manifest read THROUGH
    load, so a concurrent export to the same path can't swap ``index.json`` /
    ``manifest.json`` between our manifest validation and the load.

    Round-8 strengthening (codex #5): the round-6 version relied on
    ``asyncio.sleep(0)`` to order import-before-export and on overlapping
    ``time.sleep`` windows for contention — neither DETERMINISTICALLY put the
    export's swap into the validate->load gap, so an implementation that
    released and immediately re-took the lock in that gap could still pass. This
    version drives the ordering with explicit events:

      1. ``read_manifest`` is wrapped so the import's validation call fires
         ``import_validated`` — the import has validated the ORIGINAL manifest
         and (if correct) still holds the lock.
      2. The export coroutine waits for ``import_validated`` before issuing its
         request, so it contends for the lock strictly AFTER the import
         validated — precisely in the validate->load gap.
      3. The import's load signals ``export_may_run`` and gives the export a
         bounded window to attempt its swap. If the lock spans validate->load
         the export stays BLOCKED and the load still observes ``test-model``; if
         the lock were dropped in the gap the swap lands and the load observes
         ``EXPORT-swapped-model`` — failing the assertion."""
    import asyncio as _asyncio
    import json as _json
    import threading
    import time as _time
    from pathlib import Path as _Path

    import httpx
    from fastapi import FastAPI

    import vllm_mlx.routes.cache as cache_mod
    from vllm_mlx.routes.cache import router

    # A shared path holding a valid manifest the import will read+validate.
    manifest = Manifest(
        protocol_version=PROTOCOL_VERSION, model_id="test-model", entries=1
    )
    _write_export_root(cache_client.sandbox, "shared-io", manifest)
    shared_dir = _Path(cache_client.sandbox) / "shared-io"

    engine = cache_client.FakeEngine(
        entries=1, current_memory=1024, load_returns=1, loaded_bytes=512
    )
    engine.config.model_name = "test-model"
    cache_client.cfg.model_name = "test-model"

    state = {"in_flight": 0, "peak": 0}
    # Ordering witnesses: what model_id the LOAD saw on disk, and whether the
    # export's manifest-swap landed before or after the load ran.
    witness = {"load_saw_model_id": None, "manifest_swapped_before_load": None}
    counter_lock = threading.Lock()

    # #1100 codex round 8 (#5): cross-thread coordination. The save/load run on
    # anyio worker threads and the route handlers on the loop; threading.Event
    # is safe to set from the loop and wait on from a worker.
    import_validated = threading.Event()
    export_may_run = threading.Event()

    # Wrap read_manifest so the import's validation of the ORIGINAL manifest
    # fires ``import_validated`` (the export waits on this before contending, so
    # it lands strictly in the validate->load gap).
    _orig_read_manifest = cache_mod.read_manifest

    def _wrapped_read_manifest(cache_dir):
        m = _orig_read_manifest(cache_dir)
        if getattr(m, "model_id", None) == "test-model":
            import_validated.set()
        return m

    def _instrumented_save(cache_dir, should_abort=None):
        # The export's save runs INSIDE the dest lock. Publish a FOREIGN
        # snapshot: swap the on-disk manifest+index to a foreign model_id. If
        # the import's lock discipline is correct, this can only run once the
        # import has fully released the lock (after its load).
        with counter_lock:
            state["in_flight"] += 1
            state["peak"] = max(state["peak"], state["in_flight"])
        # Publish the export's foreign manifest over the shared path, and a
        # committed index.json (the real committed-save post-condition the
        # manifest builder now requires — round 6 #2).
        (shared_dir / "index.json").write_text(
            _json.dumps(
                {
                    "version": 3,
                    "entries": [{"index": 0, "num_tokens": 4, "memory_bytes": 512}],
                }
            )
        )
        (shared_dir / "manifest.json").write_text(
            _json.dumps(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "model_id": "EXPORT-swapped-model",
                    "entries": 1,
                }
            )
        )
        with counter_lock:
            state["in_flight"] -= 1
        cache = engine.scheduler.memory_aware_cache
        if cache is not None:
            cache._last_save_outcome = "committed"
            cache._last_load_bytes = 512
        return True

    def _instrumented_load(cache_dir, replace=False):
        with counter_lock:
            state["in_flight"] += 1
            state["peak"] = max(state["peak"], state["in_flight"])
        # We are now in the validate->load gap, holding the dest lock (if the
        # implementation is correct). Release the export to attempt its swap and
        # give it a real window to try. A correct lock keeps the export blocked,
        # so the swap never lands before we read the manifest below.
        export_may_run.set()
        _time.sleep(0.15)
        # Read the manifest model_id the import is loading against, ON DISK, at
        # LOAD time — the blob a correct lock guarantees is still the one the
        # import validated, not one a concurrent export swapped in.
        try:
            on_disk = _json.loads((shared_dir / "manifest.json").read_text())
            witness["load_saw_model_id"] = on_disk.get("model_id")
            witness["manifest_swapped_before_load"] = (
                on_disk.get("model_id") == "EXPORT-swapped-model"
            )
        except Exception:
            witness["load_saw_model_id"] = "<read-failed>"
        with counter_lock:
            state["in_flight"] -= 1
        cache = engine.scheduler.memory_aware_cache
        if cache is not None:
            cache._last_load_bytes = 512
        return 1

    engine.save_cache_to_disk = _instrumented_save
    engine.load_cache_from_disk = _instrumented_load

    # #1100 codex round 9 (#3): the default ``_FakeEngine.save_cache_with_outcome``
    # / ``load_cache_with_result`` wrap their op under ``self._step_lock`` (to
    # mimic the real single step thread). But that lock ALSO serializes a
    # concurrent export's save against this import's load — so the test would
    # stay green even if the ROUTE dropped its per-destination lock in the
    # validate->load gap (the fake's lock, not the route's, would keep the swap
    # out). Install UNSYNCHRONIZED wrappers so the ONLY thing that can keep the
    # export blocked during the load is the route's dest lock under test.
    from vllm_mlx.cache.protocol import LoadResult, SaveOutcome

    def _unsync_save_with_outcome(cache_dir, should_abort=None):
        engine.save_cache_to_disk(cache_dir, should_abort=should_abort)
        cache = engine.scheduler.memory_aware_cache
        outcome = getattr(cache, "_last_save_outcome", "empty") if cache else "empty"
        return SaveOutcome(outcome=outcome)

    def _unsync_load_with_result(cache_dir, replace=False):
        entries = engine.load_cache_from_disk(cache_dir, replace=replace)
        cache = engine.scheduler.memory_aware_cache
        bytes_loaded = int(getattr(cache, "_last_load_bytes", 0)) if cache else 0
        return LoadResult(entries=entries, bytes_loaded=bytes_loaded)

    engine.save_cache_with_outcome = _unsync_save_with_outcome
    engine.load_cache_with_result = _unsync_load_with_result
    cache_client.cfg.engine = engine
    cache_mod.read_manifest = _wrapped_read_manifest

    cache_mod._export_dest_locks.clear()
    cache_mod._export_locks_guard = _asyncio.Lock()

    app = FastAPI()
    app.include_router(router)
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # Start the IMPORT; it wins the lock and validates the ORIGINAL
            # manifest (firing ``import_validated``).
            importr = _asyncio.ensure_future(
                client.post(
                    "/v1/cache/import",
                    json={"source": "shared-io", "merge_strategy": "merge"},
                    headers=_auth(),
                )
            )

            async def _export_after_import_validates():
                # Contend ONLY after the import validated, so the export lands
                # strictly in the validate->load gap. Poll the threading.Event
                # off the loop without blocking it.
                while not import_validated.is_set():
                    await _asyncio.sleep(0.005)
                return await client.post(
                    "/v1/cache/export",
                    json={"destination": "shared-io"},
                    headers=_auth(),
                )

            exportr = _asyncio.ensure_future(_export_after_import_validates())
            r_imp, r_exp = await _asyncio.gather(importr, exportr)
    finally:
        cache_mod.read_manifest = _orig_read_manifest

    assert r_imp.status_code == 200, r_imp.text
    assert r_exp.status_code == 200, r_exp.text
    # Export's save and import's load share the destination lock → their
    # instrumented critical sections never overlapped.
    assert state["peak"] == 1, (
        f"import/export on one path overlapped; peak in-flight={state['peak']}"
    )
    # The export was released to run DURING the import's validate->load gap, yet
    # the load still observed the ORIGINAL manifest — proving the lock spanned
    # validate->load (a release-and-reacquire in the gap would let the swap land
    # and the load would see EXPORT-swapped-model).
    assert witness["load_saw_model_id"] == "test-model", (
        "import loaded against a manifest a concurrent export swapped in "
        f"(TOCTOU): load saw model_id={witness['load_saw_model_id']!r}"
    )
    assert witness["manifest_swapped_before_load"] is False
    # export_may_run must have been set (the load ran) — sanity that the
    # coordination fired rather than the test silently no-op'ing.
    assert export_may_run.is_set()


@pytest.mark.asyncio
async def test_interprocess_lock_serializes_and_uses_sibling_path(tmp_path):
    """#1100 codex round 4 (#5): the cross-process ``flock`` (a) is held
    exclusively — a second acquire blocks until the first releases — and (b)
    lives on a ``.txlock`` SIBLING of the destination, never inside it (a child
    would ride the export blob / count against max_bytes)."""
    import asyncio as _asyncio

    from vllm_mlx.routes.cache import _InterProcessLock

    # #1100 codex round 4 (#4): the parent dir does NOT exist yet — the lock
    # must create it, not silently degrade to no-exclusion on ENOENT.
    dest = tmp_path / "sub" / "snap"
    assert not dest.parent.exists()

    lock_a = _InterProcessLock(dest)
    async with lock_a:
        # The lock actually acquired an fd (not the degraded None path).
        assert lock_a._fd is not None, "flock degraded on a nested path (ENOENT)"
        # The lockfile is a sibling of ``dest``, not a child under it.
        sibling = tmp_path / "sub" / "snap.txlock"
        assert sibling.exists()
        assert not (dest / "snap.txlock").exists()

        # A second lock on the SAME dest must not acquire while A holds it.
        # ``flock`` is per-open-file-description, so a second fd from THIS
        # process still blocks on LOCK_EX — assert the acquire doesn't
        # complete within a short window.
        lock_b = _InterProcessLock(dest)
        acquire_b = _asyncio.ensure_future(lock_b.__aenter__())
        done, _pending = await _asyncio.wait({acquire_b}, timeout=0.3)
        assert not done, "second flock acquired while the first was held"

    # Once A releases, B acquires promptly.
    await _asyncio.wait_for(acquire_b, timeout=2.0)
    await lock_b.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_dest_lock_registry_evicts_idle_entries():
    """#1100 codex round 4 (#5): the per-destination lock registry is
    reference-counted — an entry is evicted once the last holder releases it,
    so a stream of unique paths can't grow it without bound. A concurrent
    waiter keeps the entry alive; it disappears only when fully idle."""
    import asyncio as _asyncio

    import vllm_mlx.routes.cache as cache_mod
    from vllm_mlx.routes.cache import _dest_lock

    cache_mod._export_dest_locks.clear()
    cache_mod._export_locks_guard = _asyncio.Lock()

    # A single acquire+release leaves the registry empty (idle → evicted).
    async with _dest_lock("path-a"):
        assert "path-a" in cache_mod._export_dest_locks
        assert cache_mod._export_dest_locks["path-a"].refs == 1
    assert "path-a" not in cache_mod._export_dest_locks

    # Overlapping holders keep the entry until the LAST releases.
    started = _asyncio.Event()
    release = _asyncio.Event()

    async def _hold():
        async with _dest_lock("path-b"):
            started.set()
            await release.wait()

    holder = _asyncio.ensure_future(_hold())
    await started.wait()
    assert cache_mod._export_dest_locks["path-b"].refs == 1
    release.set()
    await holder
    assert "path-b" not in cache_mod._export_dest_locks


def test_import_missing_source_does_not_leak_lock_entry(cache_client):
    """#1100 codex round 4 (#5): a missing-source import 404s, and its lock
    entry must NOT linger in the registry afterward (the round-3 version
    inserted an entry for every path, so 404-spamming unique paths grew the
    dict unboundedly).

    #1100 codex round 5 (#1): the 404 must also fire BEFORE
    ``_InterProcessLock`` runs, so no ``<source>.txlock`` file (nor its parent
    dirs) gets created on disk — otherwise a stream of unique nonexistent
    import sources would consume unbounded inodes."""
    import vllm_mlx.routes.cache as cache_mod

    cache_client.cfg.engine = cache_client.FakeEngine(entries=0)
    before = dict(cache_mod._export_dest_locks)
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={"source": "no-such-export", "merge_strategy": "merge"},
        headers=_auth(),
    )
    assert resp.status_code == 404, resp.text
    # No new permanent entry for the missing path.
    assert cache_mod._export_dest_locks == before
    # And no lockfile artifact leaked onto disk for the missing source.
    assert not (cache_client.sandbox / "no-such-export.txlock").exists()
    assert not (cache_client.sandbox / "no-such-export").exists()


def test_sweep_staging_dirs_recovers_old_snapshot(tmp_path):
    """#1100 codex round 4 (#1): if a save failed mid-swap leaving ``<dest>``
    MISSING and the last valid PUBLISHED snapshot in ``<dest>.old``,
    ``_sweep_staging_dirs`` must RESTORE ``.old`` → ``<dest>`` before removing
    staging — not blindly delete both (destroying the only recoverable copy).

    #1100 codex round 5 (#5): a recoverable snapshot must be IMPORTABLE — index
    AND manifest. ``.old`` is a prior PUBLISHED snapshot so it carries both."""
    from pathlib import Path as _Path

    from vllm_mlx.routes.cache import _sweep_staging_dirs

    dest = _Path(tmp_path / "snap")
    old_dir = _Path(str(dest) + ".old")
    new_dir = _Path(str(dest) + ".new")
    # Simulate a save interrupted after ``dest → .old`` but before ``.new →
    # dest``: dest is gone, .old holds the valid prior PUBLISHED snapshot
    # (index + manifest), .new holds raw save_to_disk staging (index, NO
    # manifest — the route writes it only after the atomic publish).
    old_dir.mkdir(parents=True)
    (old_dir / "index.json").write_text('{"version":3,"entries":[{"index":0}]}')
    (old_dir / "entry_0.safetensors").write_text("published-data")
    (old_dir / "manifest.json").write_text('{"protocol_version":"1"}')
    new_dir.mkdir(parents=True)
    (new_dir / "index.json").write_text('{"version":3,"entries":[{"index":0}]}')
    (new_dir / "entry_0.safetensors").write_text("staging-only")  # no manifest

    _sweep_staging_dirs(dest)

    # The importable .old snapshot was restored to dest — NOT the manifest-less
    # .new (which would be non-importable). Staging siblings are gone.
    assert dest.is_dir()
    assert (dest / "index.json").exists()
    assert (dest / "manifest.json").exists()
    assert (dest / "entry_0.safetensors").read_text() == "published-data"
    assert not old_dir.exists()
    assert not new_dir.exists()


def test_sweep_staging_dirs_does_not_promote_manifestless_new(tmp_path):
    """#1100 codex round 5 (#5): ``.new`` (raw save_to_disk staging — valid
    index but NO manifest) must NOT be promoted over a complete ``.old``. If
    ONLY a manifest-less ``.new`` exists (no ``.old``), it is non-importable
    and must be swept, NOT published to ``<dest>`` (which would leave a broken,
    non-importable snapshot a peer would 404 on)."""
    from pathlib import Path as _Path

    from vllm_mlx.routes.cache import _sweep_staging_dirs

    dest = _Path(tmp_path / "snap")
    new_dir = _Path(str(dest) + ".new")
    new_dir.mkdir(parents=True)
    (new_dir / "index.json").write_text('{"version":3,"entries":[{"index":0}]}')
    (new_dir / "entry_0.safetensors").write_text("staging-only")  # no manifest

    _sweep_staging_dirs(dest)

    # Non-importable .new was swept, not promoted — dest stays absent.
    assert not dest.exists()
    assert not new_dir.exists()


def test_sweep_staging_dirs_preserves_valid_published_dest(tmp_path):
    """#1100 codex round 4 (#1) complement: when ``<dest>`` already holds a
    valid published snapshot, ``_sweep_staging_dirs`` leaves it untouched and
    only removes the orphaned staging siblings."""
    from pathlib import Path as _Path

    from vllm_mlx.routes.cache import _sweep_staging_dirs

    dest = _Path(tmp_path / "snap")
    dest.mkdir(parents=True)
    (dest / "index.json").write_text('{"version":3,"entries":[{"index":0}]}')
    (dest / "entry_0.safetensors").write_text("published")
    new_dir = _Path(str(dest) + ".new")
    new_dir.mkdir(parents=True)
    (new_dir / "index.json").write_text("{}")  # orphan staging

    _sweep_staging_dirs(dest)

    assert (dest / "entry_0.safetensors").read_text() == "published"
    assert not new_dir.exists()


@pytest.mark.asyncio
async def test_export_outcome_isolated_across_concurrent_paths(cache_client):
    """#1100 codex round 4 (#2): the save outcome is returned as a VALUE from
    the step-thread op, not read from a cache-global field on the asyncio
    thread — so two concurrent exports to DIFFERENT paths (distinct locks,
    same underlying cache) can't cross-contaminate each other's classification.
    One path's save legitimately commits; another's legitimately fails; each
    handler must report ITS OWN outcome."""
    import asyncio as _asyncio

    import httpx
    from fastapi import FastAPI

    import vllm_mlx.routes.cache as cache_mod
    from vllm_mlx.routes.cache import router

    engine = cache_client.FakeEngine(entries=3, current_memory=1024)

    # The save's outcome depends on the destination: 'committed' for "good",
    # 'failed' for "bad". Both mutate the SAME shared cache field; the route
    # must NOT read that field (it reads the returned value instead).
    def _save(cache_dir, should_abort=None):
        engine.saved_to = cache_dir
        cache = engine.scheduler.memory_aware_cache
        if str(cache_dir).endswith("bad"):
            cache._last_save_outcome = "failed"
            return False
        cache._last_save_outcome = "committed"
        # #1100 codex round 6 (#2): committed save writes a readable index.json
        # (the manifest builder now requires it on a committed save).
        engine._write_committed_index(cache_dir)
        return True

    engine.save_cache_to_disk = _save
    cache_client.cfg.engine = engine

    cache_mod._export_dest_locks.clear()
    cache_mod._export_locks_guard = _asyncio.Lock()

    app = FastAPI()
    app.include_router(router)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        good = client.post(
            "/v1/cache/export", json={"destination": "good"}, headers=_auth()
        )
        bad = client.post(
            "/v1/cache/export", json={"destination": "bad"}, headers=_auth()
        )
        r_good, r_bad = await _asyncio.gather(good, bad)

    # Each handler reported its OWN outcome despite sharing the cache field.
    assert r_good.status_code == 200, r_good.text
    assert r_bad.status_code == 500, r_bad.text


def test_export_committed_but_unreadable_index_500s_and_discards(cache_client):
    """#1100 codex round 6 (#2): when a save reports ``"committed"`` but its
    on-disk ``index.json`` is missing/torn, the manifest builder must NOT fall
    back to the live ledger and publish a manifest for a non-importable
    snapshot. The export 500s and the blob is discarded — never a 200 that
    advertises entries a peer can't load."""
    from pathlib import Path as _Path

    engine = cache_client.FakeEngine(entries=3, current_memory=4096)

    def _save_committed_torn_index(cache_dir, should_abort=None):
        # Report a committed save but write a MALFORMED index.json (valid save
        # would have written a well-formed one; simulate a torn commit).
        engine.saved_to = cache_dir
        d = _Path(cache_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.json").write_text("{ this is not valid json")
        (d / "entry_0.safetensors").write_text("x")
        cache = engine.scheduler.memory_aware_cache
        if cache is not None:
            cache._last_save_outcome = "committed"
        return True

    engine.save_cache_to_disk = _save_committed_torn_index
    cache_client.cfg.engine = engine

    resp = cache_client.client.post(
        "/v1/cache/export", json={"destination": "torn"}, headers=_auth()
    )
    assert resp.status_code == 500, resp.text
    assert "unreadable" in resp.json()["detail"]
    # The blob was discarded — no importable index/manifest left behind that
    # would let a peer load a snapshot the save never validly committed.
    torn = _Path(cache_client.sandbox) / "torn"
    assert not (torn / "manifest.json").is_file()
    assert not (torn / "index.json").is_file()


def test_export_discard_quarantines_when_delete_fails(cache_client, monkeypatch):
    """#1100 BLOCKING-5: if the oversized-blob cleanup can't DELETE the blob
    (permission / fs failure), it must QUARANTINE the import-critical files
    (rename manifest.json + index.json to ``.rejected``) so the import path
    refuses the blob. A 413 must guarantee the blob is not importable — a
    stray manifest.json/index.json left readable would let a peer import an
    over-cap blob."""
    from pathlib import Path as _Path

    import vllm_mlx.routes.cache as cache_mod

    engine = cache_client.FakeEngine(entries=1, current_memory=10)

    def _save_writes_big_blob(cache_dir, should_abort=None):
        engine.saved_to = cache_dir
        d = _Path(cache_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.json").write_text(
            '{"version":3,"entries":[{"index":0,"num_tokens":4,'
            '"memory_bytes":42,"cache_types":["KVCache"]}]}'
        )
        (d / "entry_0.safetensors").write_text("x" * 5000)  # on-disk over cap
        (d / "entry_0_tokens.bin").write_text("x")
        cache = engine.scheduler.memory_aware_cache
        if cache is not None:
            cache._last_save_outcome = "committed"
        return True

    engine.save_cache_to_disk = _save_writes_big_blob
    cache_client.cfg.engine = engine

    # Make rmtree a no-op so deletion "fails" (the dedicated-subdir discard
    # path uses rmtree) — forcing the quarantine path. os.replace (the
    # quarantine rename) is left working.
    monkeypatch.setattr(cache_mod.shutil, "rmtree", lambda *a, **k: None)

    resp = cache_client.client.post(
        "/v1/cache/export",
        json={"destination": "quarantine-me", "max_bytes": 1000},
        headers=_auth(),
    )
    # Still a 413 (the invariant is honored via quarantine, not deletion).
    assert resp.status_code == 413, resp.text
    blob_dir = _Path(cache_client.sandbox) / "quarantine-me"
    # Import-critical files renamed to .rejected → import path refuses them.
    assert not (blob_dir / "index.json").exists()
    assert not (blob_dir / "manifest.json").exists()
    assert (blob_dir / "index.json.rejected").exists()
    assert (blob_dir / "manifest.json.rejected").exists()

    # Prove the blob is no longer importable: an import against it 404s
    # (read_manifest finds no manifest.json).
    imp = cache_client.client.post(
        "/v1/cache/import",
        json={"source": "quarantine-me"},
        headers=_auth(),
    )
    assert imp.status_code == 404, imp.text


def test_export_discard_500_when_neither_delete_nor_quarantine_works(
    cache_client, monkeypatch
):
    """#1100 BLOCKING-5: if cleanup can neither DELETE nor QUARANTINE the
    over-cap blob, the route must NOT claim a 413 discard (which promises the
    blob is gone) — it returns 500 so the lie ("discarded") is never told and
    an operator sees the stray blob signal."""
    from pathlib import Path as _Path

    import vllm_mlx.routes.cache as cache_mod

    engine = cache_client.FakeEngine(entries=1, current_memory=10)

    def _save_writes_big_blob(cache_dir, should_abort=None):
        engine.saved_to = cache_dir
        d = _Path(cache_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.json").write_text(
            '{"version":3,"entries":[{"index":0,"num_tokens":4,'
            '"memory_bytes":42,"cache_types":["KVCache"]}]}'
        )
        (d / "entry_0.safetensors").write_text("x" * 5000)
        (d / "entry_0_tokens.bin").write_text("x")
        # #1100 codex round 4 (#1): committed save — record the authoritative
        # outcome the route reads to route into the size-gate branch.
        cache = engine.scheduler.memory_aware_cache
        if cache is not None:
            cache._last_save_outcome = "committed"
        return True

    engine.save_cache_to_disk = _save_writes_big_blob
    cache_client.cfg.engine = engine

    # Both deletion (rmtree no-op) AND quarantine (returns False = couldn't
    # remove the import-critical files) fail. Patch the quarantine helper
    # directly rather than os.replace — os.replace is shared with
    # write_manifest, which runs BEFORE the gate and must succeed.
    monkeypatch.setattr(cache_mod.shutil, "rmtree", lambda *a, **k: None)
    monkeypatch.setattr(
        cache_mod, "_quarantine_import_critical", lambda destination: False
    )

    resp = cache_client.client.post(
        "/v1/cache/export",
        json={"destination": "stuck-blob", "max_bytes": 1000},
        headers=_auth(),
    )
    # Neither delete nor quarantine worked → 500, NOT a 413 that lies.
    assert resp.status_code == 500, resp.text
    assert "discard" in resp.json()["detail"].lower()


def test_load_from_disk_replace_preserves_cache_on_missing_source():
    """#1100 BLOCKING-4: ``replace=True`` clears the cache ONLY after the
    source index validates. A missing/unreadable source returns 0 WITHOUT
    clearing — so a failed replace-import preserves the existing cache
    instead of destroying it (the route used to clear before it knew the
    load would fail)."""
    import mlx.core as mx

    src_cache, _ = _build_cache_with_arrays_layer()
    # Materialize arrays on the main thread (see the roundtrip test's stream
    # rationale) so touching them from the default-stream context is safe.
    for entry in src_cache._entries.values():
        for layer in entry.cache:
            state = getattr(layer, "state", None)
            if state is not None:
                mx.eval(state)
    before = len(src_cache._entries)
    assert before > 0

    with mx.stream(mx.default_stream(mx.default_device())):
        loaded = src_cache.load_from_disk("/nonexistent/replace/source", replace=True)

    assert loaded == 0
    # NOT cleared — a failed replace leaves the existing cache intact.
    assert len(src_cache._entries) == before


def test_load_from_disk_replace_preserves_cache_on_corrupt_entry_blob(tmp_path):
    """#1100 BLOCKING-1: a VALID index.json paired with a corrupt/missing
    entry_*.safetensors must NOT destroy the existing cache under
    ``replace=True``. The round-1 fix cleared the live cache after the index
    validated but BEFORE any entry blob was read, so this exact case wiped the
    cache and loaded nothing. The stage-then-swap fix reads every entry FIRST
    and aborts (existing cache intact) if any entry blob is corrupt."""
    import mlx.core as mx

    from vllm_mlx.memory_cache import MemoryAwarePrefixCache

    # 1. Build + save a real 1-entry snapshot to disk (valid index + blobs).
    src_cache, load_cfg = _build_cache_with_arrays_layer()
    for entry in src_cache._entries.values():
        for layer in entry.cache:
            state = getattr(layer, "state", None)
            if state is not None:
                mx.eval(state)
    target = str(tmp_path / "snap")
    with mx.stream(mx.default_stream(mx.default_device())):
        assert src_cache.save_to_disk(target) is True

    # 2. Corrupt the entry safetensors body while leaving index.json VALID —
    #    truncate it to a few bytes so the header/body completeness check
    #    rejects it as corrupt (not merely incompatible).
    entry_sf = Path(target) / "entry_0.safetensors"
    assert entry_sf.is_file()
    entry_sf.write_bytes(b"\x00\x00\x00\x00")  # garbage — valid index, dead blob

    # 3. A DIFFERENT populated destination cache attempts replace-import.
    dst_cache = MemoryAwarePrefixCache(model=object(), config=load_cfg)
    dst_tokens = (90, 91, 92)
    dst_kv = _second_kvcache()
    from vllm_mlx.memory_cache import _CacheEntry

    entry = _CacheEntry.create(list(dst_tokens), dst_kv)
    with dst_cache._lock:
        dst_cache._entries[dst_tokens] = entry
        dst_cache._current_memory += entry.memory_bytes
    for layer in dst_kv:
        state = getattr(layer, "state", None)
        if state is not None:
            mx.eval(state)
    before_entries = dict(dst_cache._entries)
    before_mem = dst_cache._current_memory
    assert len(before_entries) == 1

    with mx.stream(mx.default_stream(mx.default_device())):
        loaded = dst_cache.load_from_disk(target, replace=True)

    # Replace aborted on the corrupt entry — nothing loaded, cache preserved.
    assert loaded == 0
    assert dst_cache._entries == before_entries
    assert dst_cache._current_memory == before_mem
    # #1100 codex round 4 (#3): the authoritative loaded-byte total is 0 — the
    # preserved existing footprint must NOT be reported as loaded by this call.
    assert dst_cache._last_load_bytes == 0


def test_load_from_disk_replace_records_authoritative_loaded_bytes(tmp_path):
    """#1100 codex round 4 (#3): a COMMITTED replace records the exact KV byte
    total it installed on ``_last_load_bytes`` (summed under the lock over the
    entries it staged), so the import route reports the loaded footprint
    without a racy before/after ``_current_memory`` diff. And (#2) the
    clear+install is a single atomic swap — the post-load footprint equals the
    recorded loaded bytes, never a half-rebuilt intermediate."""
    import mlx.core as mx

    from vllm_mlx.memory_cache import MemoryAwarePrefixCache, _CacheEntry

    # 1. Build + save a real 1-entry snapshot (valid index + blobs).
    src_cache, load_cfg = _build_cache_with_arrays_layer()
    for entry in src_cache._entries.values():
        for layer in entry.cache:
            state = getattr(layer, "state", None)
            if state is not None:
                mx.eval(state)
    target = str(tmp_path / "snap")
    with mx.stream(mx.default_stream(mx.default_device())):
        assert src_cache.save_to_disk(target) is True
    # The source's save recorded a "committed" outcome (>=1 entry landed).
    assert src_cache._last_save_outcome == "committed"

    # 2. A populated destination cache does a replace-import of the snapshot.
    dst_cache = MemoryAwarePrefixCache(model=object(), config=load_cfg)
    dst_tokens = (90, 91, 92)
    dst_kv = _second_kvcache()
    entry = _CacheEntry.create(list(dst_tokens), dst_kv)
    with dst_cache._lock:
        dst_cache._entries[dst_tokens] = entry
        dst_cache._current_memory += entry.memory_bytes
    for layer in dst_kv:
        state = getattr(layer, "state", None)
        if state is not None:
            mx.eval(state)

    with mx.stream(mx.default_stream(mx.default_device())):
        loaded = dst_cache.load_from_disk(target, replace=True)

    # The pre-existing entry was replaced; the snapshot's single entry loaded.
    assert loaded == 1
    assert dst_tokens not in dst_cache._entries
    # Authoritative loaded bytes == the installed footprint == post-load
    # ``_current_memory`` (replace cleared first, so no residue skews it).
    assert dst_cache._last_load_bytes > 0
    assert dst_cache._last_load_bytes == dst_cache._current_memory


def test_save_to_disk_records_empty_outcome_on_empty_cache(tmp_path):
    """#1100 codex round 4 (#1): ``save_to_disk`` on a genuinely EMPTY cache
    records ``_last_save_outcome == 'empty'`` (a legit no-op), distinct from a
    ``'failed'`` save of a non-empty cache — so the export route can 200 an
    empty export vs 500 a failed one WITHOUT sampling ``len(cache)`` before the
    op (a racy pre-write snapshot)."""
    _src, load_cfg = _build_cache_with_arrays_layer()
    from vllm_mlx.memory_cache import MemoryAwarePrefixCache

    empty = MemoryAwarePrefixCache(model=object(), config=load_cfg)
    assert len(empty) == 0
    assert empty.save_to_disk(str(tmp_path / "empty-snap")) is False
    assert empty._last_save_outcome == "empty"


def _second_kvcache():
    """A small standalone KVCache list for the destination-cache fixture."""
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    kv = KVCache()
    kv.update_and_fetch(mx.zeros((1, 2, 3, 8)), mx.ones((1, 2, 3, 8)))
    return [kv]


def test_export_post_write_max_bytes_discards_oversized_blob(cache_client):
    """#1100 BLOCKING-2 + BLOCKING-4: max_bytes is enforced against the ACTUAL
    COMMITTED ON-DISK size (sum of real file sizes via os.stat), NOT the
    logical ``memory_bytes`` ledger. A cache that grows between the pre-check
    and the snapshot — OR whose on-disk overhead (tokens.bin + index.json +
    manifest.json + safetensors headers) pushes the real footprint over the
    cap — still can't produce an over-cap export: the oversized blob is
    discarded from disk rather than left for a peer to import.

    Here the pre-check sees a tiny live footprint (10 B ≤ cap) but the save
    writes a real 5000-byte entry file, so the committed directory size
    exceeds the 1000 B cap and the export is rejected + discarded."""
    import json as _json
    from pathlib import Path as _Path

    engine = cache_client.FakeEngine(entries=1, current_memory=10, load_returns=0)

    def _save_writes_big_blob(cache_dir, should_abort=None):
        # Pre-check saw a tiny live footprint (10 B ≤ cap); the committed
        # blob is genuinely large ON DISK (a 5000-byte entry file) — this is
        # what the post-write committed-size gate must catch.
        engine.saved_to = cache_dir
        d = _Path(cache_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.json").write_text(
            _json.dumps(
                {
                    "version": 3,
                    "total_memory_bytes": 42,
                    "entries": [
                        {
                            "index": 0,
                            "num_tokens": 4,
                            "memory_bytes": 42,
                            "cache_types": ["KVCache"],
                        }
                    ],
                }
            )
        )
        # A genuinely large on-disk entry file (5000 bytes) — even though the
        # logical memory_bytes above is a mere 42, the REAL committed size
        # blows the 1000 B cap. This is the exact class the old logical-only
        # gate missed.
        (d / "entry_0.safetensors").write_text("x" * 5000)
        (d / "entry_0_tokens.bin").write_text("x")
        # #1100 codex round 4 (#1): committed save — the route reads this
        # authoritative outcome to reach the post-write size gate.
        cache = engine.scheduler.memory_aware_cache
        if cache is not None:
            cache._last_save_outcome = "committed"
        return True

    engine.save_cache_to_disk = _save_writes_big_blob
    cache_client.cfg.engine = engine

    resp = cache_client.client.post(
        "/v1/cache/export",
        json={"destination": "grew", "max_bytes": 1000},  # on-disk 5000+ > 1000
        headers=_auth(),
    )
    assert resp.status_code == 413, resp.text
    # The oversized blob was discarded — nothing left on disk for a peer.
    blob_dir = _Path(cache_client.sandbox) / "grew"
    assert not (blob_dir / "index.json").exists()
    assert not (blob_dir / "entry_0.safetensors").exists()
    # No manifest was written for the rejected export (it was discarded too).
    assert not (blob_dir / "manifest.json").exists()


def test_import_engine_not_loaded_returns_503(cache_client):
    """No model loaded → 503, AFTER the manifest checks pass (so a bad
    manifest still 409s even without an engine — order matters)."""
    _write_export_root(
        cache_client.sandbox,
        "ready-noeng",
        Manifest(protocol_version=PROTOCOL_VERSION, model_id="test-model"),
    )
    cache_client.cfg.engine = None
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={"source": "ready-noeng"},
        headers=_auth(),
    )
    assert resp.status_code == 503, resp.text
    assert resp.json()["detail"] == "engine not loaded"


def test_import_409_precedes_503(cache_client):
    """A protocol mismatch must 409 even when no engine is loaded — the
    manifest gate runs before the engine gate."""
    _write_export_root(
        cache_client.sandbox,
        "v999-noeng",
        Manifest(protocol_version="999", model_id="test-model"),
    )
    cache_client.cfg.engine = None
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={"source": "v999-noeng"},
        headers=_auth(),
    )
    assert resp.status_code == 409, resp.text


def test_import_rejects_path_traversal(cache_client):
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={"source": "../etc"},
        headers=_auth(),
    )
    assert resp.status_code == 403


def test_import_missing_source_returns_422(cache_client):
    """``source`` is required — pydantic rejects the missing field."""
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={},
        headers=_auth(),
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /v1/cache/info — fully implemented (the only non-stub endpoint)
# ---------------------------------------------------------------------------


def test_info_returns_manifest(cache_client):
    manifest = Manifest(
        protocol_version=PROTOCOL_VERSION,
        model_id="qwen3.5-9b-4bit",
        quantization="4bit",
        entries=18,
    )
    _write_export_root(cache_client.sandbox, "ready", manifest)
    resp = cache_client.client.get(
        "/v1/cache/info?path=ready",
        headers=_auth(),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["protocol_version"] == PROTOCOL_VERSION
    assert body["manifest"]["model_id"] == "qwen3.5-9b-4bit"
    assert body["manifest"]["entries"] == 18


def test_info_missing_manifest_returns_404(cache_client):
    (cache_client.sandbox / "empty").mkdir(parents=True)
    resp = cache_client.client.get(
        "/v1/cache/info?path=empty",
        headers=_auth(),
    )
    assert resp.status_code == 404


def test_info_rejects_path_traversal(cache_client):
    resp = cache_client.client.get(
        "/v1/cache/info?path=../etc",
        headers=_auth(),
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# build_manifest_from_engine_state — reads model id + cache-config knobs
# ---------------------------------------------------------------------------


def test_build_manifest_from_engine_reads_config(monkeypatch):
    """The manifest builder pulls quantization/paged/turboquant off
    ``scheduler.config`` and entries/bytes off the live prefix cache."""
    from vllm_mlx.cache.protocol import build_manifest_from_engine_state
    from vllm_mlx.config import reset_config
    from vllm_mlx.memory_cache import _TOKENS_FORMAT_VERSION_IN_INDEX

    cfg = reset_config()
    cfg.model_name = "mlx-community/Qwen3.5-9B-4bit"
    try:
        engine = _FakeEngine(
            entries=7,
            current_memory=8192,
            kv_cache_dtype="int4",
            use_paged_cache=True,
            kv_cache_turboquant=True,
        )
        m = build_manifest_from_engine_state(engine)
        assert m.protocol_version == PROTOCOL_VERSION
        assert m.model_id == "mlx-community/Qwen3.5-9B-4bit"  # from server config
        assert m.quantization == "int4"
        assert m.paged_cache is True
        assert m.turboquant_kv is True
        assert m.index_format_version == _TOKENS_FORMAT_VERSION_IN_INDEX
        assert m.entries == 7
        assert m.total_bytes == 8192
        assert m.created_at.endswith("Z")  # ISO-8601 UTC
    finally:
        reset_config()


def test_build_manifest_prefix_cache_none_is_empty(monkeypatch):
    """A disabled prefix cache (None) yields entries/bytes = 0, no raise."""
    from vllm_mlx.cache.protocol import build_manifest_from_engine_state
    from vllm_mlx.config import reset_config

    cfg = reset_config()
    cfg.model_name = "test-model"
    try:
        engine = _FakeEngine(prefix_cache=False)
        m = build_manifest_from_engine_state(engine)
        assert m.entries == 0
        assert m.total_bytes == 0
        assert m.model_id == "test-model"
    finally:
        reset_config()


def test_build_manifest_falls_back_to_engine_model_id(monkeypatch):
    """When the server singleton has no ``model_name`` (embedded engine),
    fall back to the engine's own ``config.model_name``."""
    from vllm_mlx.cache.protocol import build_manifest_from_engine_state
    from vllm_mlx.config import reset_config

    reset_config()  # model_name stays None
    try:
        engine = _FakeEngine()
        engine.config.model_name = "embedded/model-id"
        m = build_manifest_from_engine_state(engine)
        assert m.model_id == "embedded/model-id"
    finally:
        reset_config()


# ---------------------------------------------------------------------------
# REAL round-trip — MemoryAwarePrefixCache.save_to_disk → load_from_disk
#
# This is the decisive test the #476 follow-up owes: a hand-built cache with
# BOTH a plain KVCache layer AND a recurrent-state ``ArraysCache`` layer must
# survive save→load. ``_cache_classes_compatible`` returns loadable=True for
# ArraysCache under a no-quant config, so the entry is NOT dropped as
# incompatible — this asserts that end to end against the real engine cache.
# ---------------------------------------------------------------------------


def _build_cache_with_arrays_layer():
    """A real ``MemoryAwarePrefixCache`` holding one entry whose layers are
    ``[KVCache, ArraysCache]``. The ArraysCache is injected directly into
    ``_entries`` (bypassing ``store()``'s non-trimmable-drop gate) because
    the point here is the SAVE/LOAD persistence path, not the reuse gate."""
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache, KVCache

    from vllm_mlx.memory_cache import (
        MemoryAwarePrefixCache,
        MemoryCacheConfig,
        _CacheEntry,
    )

    cfg = MemoryCacheConfig(max_memory_percent=0.5)
    cache = MemoryAwarePrefixCache(model=object(), config=cfg)

    kv = KVCache()
    kv.update_and_fetch(mx.zeros((1, 2, 4, 8)), mx.ones((1, 2, 4, 8)))
    arrays = ArraysCache(size=1)
    arrays[0] = mx.ones((1, 3, 5))

    tokens = (10, 11, 12, 13)
    entry = _CacheEntry.create(list(tokens), [kv, arrays])
    with cache._lock:
        cache._entries[tokens] = entry
        cache._current_memory += entry.memory_bytes
    return cache, MemoryCacheConfig(max_memory_percent=0.5)


def test_arrays_cache_layer_is_loadable_under_no_quant():
    """Unit-level proof of the frozen-design claim: ArraysCache round-trips
    (the QuantizedKVCache/TurboQuantKVCache gates do NOT apply to it)."""
    from vllm_mlx.memory_cache import MemoryCacheConfig, _cache_classes_compatible

    ok, reason = _cache_classes_compatible(
        ["KVCache", "ArraysCache"], MemoryCacheConfig(max_memory_percent=0.5)
    )
    assert ok is True, reason
    assert reason == ""


def test_real_cache_roundtrip_survives_arrays_layer(tmp_path):
    """Save a real cache with a KVCache + ArraysCache entry, load it into a
    fresh cache, and assert the ArraysCache layer survived (not dropped as
    incompatible). This is the correctness backbone the HTTP handlers ride."""
    import json

    cache, load_cfg = _build_cache_with_arrays_layer()
    target = str(tmp_path / "snap")

    assert cache.save_to_disk(target) is True
    assert len(cache._entries) == 1

    # The on-disk index records the REAL per-layer class names — proving
    # the recurrent-state layer is persisted, not silently converted.
    with open(Path(target) / "index.json") as f:
        index = json.load(f)
    persisted_types = index["entries"][0]["cache_types"]
    assert "ArraysCache" in persisted_types, persisted_types
    assert "KVCache" in persisted_types, persisted_types

    # Load into a FRESH cache under a default (no-quant) config.
    from vllm_mlx.memory_cache import MemoryAwarePrefixCache

    fresh = MemoryAwarePrefixCache(model=object(), config=load_cfg)
    loaded = fresh.load_from_disk(target)
    assert loaded == 1, "the ArraysCache entry must NOT be dropped on load"
    assert len(fresh._entries) == 1

    # The reconstructed entry still carries the ArraysCache layer.
    (entry,) = fresh._entries.values()
    layer_classes = [type(layer).__name__ for layer in entry.cache]
    assert "ArraysCache" in layer_classes, layer_classes
    assert "KVCache" in layer_classes, layer_classes


def test_load_from_disk_repopulates_radix_index(tmp_path):
    """#1100 codex round 4 (#3): a load must insert the loaded keys into the
    radix lookup index, not just ``_entries``/``_sorted_keys``. The round-4
    atomic-swap clears the radix (replace) / leaves it stale (merge) and the
    round-4 bug installed entries WITHOUT re-inserting them into the radix —
    so a radix-backed fetch would MISS every imported entry. Uses the real
    save/load path (real MLX arrays) into a radix-WIRED destination cache."""
    import mlx.core as mx

    from vllm_mlx.memory_cache import MemoryAwarePrefixCache
    from vllm_mlx.runtime.radix_index import RadixPrefixIndex

    # Real 1-entry snapshot on disk (KVCache + ArraysCache, real arrays).
    src_cache, load_cfg = _build_cache_with_arrays_layer()
    for entry in src_cache._entries.values():
        for layer in entry.cache:
            state = getattr(layer, "state", None)
            if state is not None:
                mx.eval(state)
    target = str(tmp_path / "radix-snap")
    with mx.stream(mx.default_stream(mx.default_device())):
        assert src_cache.save_to_disk(target) is True
    (src_key,) = src_cache._entries.keys()

    # Destination cache WITH a radix index, replace-import the snapshot.
    dst = MemoryAwarePrefixCache(
        model=object(), config=load_cfg, radix_index=RadixPrefixIndex()
    )
    with mx.stream(mx.default_stream(mx.default_device())):
        loaded = dst.load_from_disk(target, replace=True)

    assert loaded == 1
    # The radix index mirrors ``_entries`` — the imported key is findable via
    # radix, not just the bisect fallback.
    assert len(dst._radix_index) == len(dst._entries) == 1
    assert src_key in dst._radix_index


def test_load_from_disk_refuses_malformed_index_without_crashing(tmp_path):
    """#1100 codex round 8 (#1): the IMPORT path (``load_from_disk``) reads
    ``entry_meta["index"]`` / ``["num_tokens"]`` raw in its staging loop. A
    malformed / torn / hand-crafted index (non-dict entry, missing or wrong-
    typed required field, wrong version) must be REJECTED up front — returning 0
    with the live cache intact — not raise ``TypeError``/``KeyError`` mid-load
    (which in replace mode could land after the cache was already cleared). The
    importer applies the SAME validator the export/manifest side uses."""
    import json as _json

    import mlx.core as mx

    from vllm_mlx.memory_cache import MemoryAwarePrefixCache

    _src, load_cfg = _build_cache_with_arrays_layer()

    def _run_case(index_obj):
        d = tmp_path / "malformed"
        d.mkdir(exist_ok=True)
        (d / "index.json").write_text(_json.dumps(index_obj))
        dst = MemoryAwarePrefixCache(model=object(), config=load_cfg)
        # Seed a prior entry so we can prove replace leaves it intact.
        prior_key = (7, 8, 9)
        dst._entries[prior_key] = SimpleNamespace(memory_bytes=55, cache=[])
        dst._sorted_keys = [prior_key]
        dst._current_memory = 55
        with mx.stream(mx.default_stream(mx.default_device())):
            loaded = dst.load_from_disk(str(d), replace=True)
        # No crash, nothing loaded, prior cache preserved (replace never
        # reached the clear because the index was refused up front).
        assert loaded == 0
        assert dst._entries == {prior_key: dst._entries[prior_key]}
        assert dst._current_memory == 55

    # A grab-bag of malformed indices the raw deref would have crashed on:
    _run_case({"version": 3, "entries": [{"num_tokens": 4}]})  # missing index
    _run_case({"version": 3, "entries": [{"index": 0}]})  # missing num_tokens
    _run_case({"version": 3, "entries": ["not-a-dict"]})  # non-dict entry
    _run_case({"version": 3, "entries": [{"index": "x", "num_tokens": 4}]})  # str index
    _run_case(
        {"version": 99, "entries": [{"index": 0, "num_tokens": 4}]}
    )  # bad version


def test_load_from_disk_rolls_back_entry_on_radix_insert_failure(tmp_path, monkeypatch):
    """#1100 codex round 6 (#4): if the radix insert FAILS while committing a
    loaded entry, that entry must be rolled back out of ``_entries`` /
    ``_sorted_keys`` / memory accounting — never left reported as loaded but
    unreachable through the radix-backed lookup. A radix-present ``_entries``
    entry absent from the radix is silently unreachable AND inflates the loaded
    total; rolling it back keeps ``_entries`` and the radix in lockstep."""
    import mlx.core as mx

    from vllm_mlx.memory_cache import MemoryAwarePrefixCache
    from vllm_mlx.runtime.radix_index import RadixPrefixIndex

    src_cache, load_cfg = _build_cache_with_arrays_layer()
    for entry in src_cache._entries.values():
        for layer in entry.cache:
            state = getattr(layer, "state", None)
            if state is not None:
                mx.eval(state)
    target = str(tmp_path / "radix-fail-snap")
    with mx.stream(mx.default_stream(mx.default_device())):
        assert src_cache.save_to_disk(target) is True

    radix = RadixPrefixIndex()

    def _boom_insert(_key):
        raise RuntimeError("radix insert blew up")

    monkeypatch.setattr(radix, "insert", _boom_insert)

    dst = MemoryAwarePrefixCache(model=object(), config=load_cfg, radix_index=radix)
    with mx.stream(mx.default_stream(mx.default_device())):
        loaded = dst.load_from_disk(target, replace=True)

    # The only entry's radix insert failed → it was rolled back: nothing
    # loaded, no orphaned ``_entries`` / ``_sorted_keys`` / memory, and the
    # authoritative loaded-byte total is 0 (not the rolled-back entry's bytes).
    assert loaded == 0
    assert len(dst._entries) == 0
    assert dst._sorted_keys == []
    assert dst._current_memory == 0
    assert dst._last_load_bytes == 0


def test_load_from_disk_replace_restores_prior_cache_on_radix_failure(
    tmp_path, monkeypatch
):
    """#1100 codex round 8 (#2): replace mode clears the live cache BEFORE
    installing the staged set. Round 6 (#4) rolled back only the ONE entry whose
    radix insert failed — but by then the PRIOR cache was already destroyed, so a
    mid-commit radix failure left a partial cache and permanently lost the prior
    one while still returning a "successful" load. The fix snapshots the prior
    cache and RESTORES it wholesale on any radix failure (true all-or-nothing
    replace). This test seeds a real prior entry, forces the staged entry's radix
    insert to blow up, and asserts the prior cache is intact (not wiped) and 0
    loaded — the replace was atomic."""
    import mlx.core as mx

    from vllm_mlx.memory_cache import MemoryAwarePrefixCache
    from vllm_mlx.runtime.radix_index import RadixPrefixIndex

    src_cache, load_cfg = _build_cache_with_arrays_layer()
    for entry in src_cache._entries.values():
        for layer in entry.cache:
            state = getattr(layer, "state", None)
            if state is not None:
                mx.eval(state)
    target = str(tmp_path / "replace-restore-snap")
    with mx.stream(mx.default_stream(mx.default_device())):
        assert src_cache.save_to_disk(target) is True

    radix = RadixPrefixIndex()

    # A destination cache holding a REAL prior entry (with a valid radix
    # membership), so a botched replace has something concrete to lose.
    dst = MemoryAwarePrefixCache(model=object(), config=load_cfg, radix_index=radix)
    prior_key = (11, 22, 33)
    prior_entry = SimpleNamespace(memory_bytes=777, cache=[])
    dst._entries[prior_key] = prior_entry
    dst._sorted_keys = [prior_key]
    dst._current_memory = 777
    radix.insert(prior_key)

    # Now make the NEXT radix insert (the staged entry, during commit) fail —
    # AFTER the prior cache was cleared.
    _orig_insert = radix.insert

    def _boom_on_new(_key):
        if _key == prior_key:
            return _orig_insert(_key)  # allow the restore-path re-insert
        raise RuntimeError("radix insert blew up on staged entry")

    monkeypatch.setattr(radix, "insert", _boom_on_new)

    with mx.stream(mx.default_stream(mx.default_device())):
        loaded = dst.load_from_disk(target, replace=True)

    # Atomic replace: the staged entry's radix failure restored the PRIOR cache
    # wholesale — nothing loaded, prior entry + accounting intact, prior radix
    # membership rebuilt.
    assert loaded == 0
    assert dst._entries == {prior_key: prior_entry}
    assert dst._sorted_keys == [prior_key]
    assert dst._current_memory == 777
    assert dst._last_load_bytes == 0


def test_load_from_disk_replace_aborts_on_insufficient_physical_headroom(
    tmp_path, monkeypatch
):
    """#1100 codex round 6 (#3): replace mode holds the existing cache AND the
    staged blob until the swap (~2× peak). If admitting the next staged entry
    would blow past the physical-headroom budget, the replace ABORTS with the
    existing cache intact — never OOM-ing the host mid-import."""
    import mlx.core as mx

    import vllm_mlx.memory_cache as mc
    from vllm_mlx.memory_cache import MemoryAwarePrefixCache

    src_cache, load_cfg = _build_cache_with_arrays_layer()
    for entry in src_cache._entries.values():
        for layer in entry.cache:
            state = getattr(layer, "state", None)
            if state is not None:
                mx.eval(state)
    target = str(tmp_path / "headroom-snap")
    with mx.stream(mx.default_stream(mx.default_device())):
        assert src_cache.save_to_disk(target) is True

    # A destination cache that already holds a (sentinel) live entry, so the
    # replace has an existing cache to PRESERVE on abort.
    dst = MemoryAwarePrefixCache(model=object(), config=load_cfg)
    dst._entries[(1, 2, 3)] = object()
    dst._sorted_keys = [(1, 2, 3)]
    dst._current_memory = 999
    before_entries = dict(dst._entries)

    # Force a tiny-but-nonzero physical-headroom budget so ANY real staged
    # entry exceeds it (100 B available → 75 B budget after the headroom
    # fraction; a real KV entry is far larger). A 0 budget would DISABLE the
    # check (psutil-missing fallback), so we use a small positive value.
    monkeypatch.setattr(mc, "_get_available_memory", lambda: 100)

    with mx.stream(mx.default_stream(mx.default_device())):
        loaded = dst.load_from_disk(target, replace=True)

    # Replace aborted: nothing loaded, existing cache fully preserved.
    assert loaded == 0
    assert dst._entries == before_entries
    assert dst._sorted_keys == [(1, 2, 3)]
    assert dst._current_memory == 999
    assert dst._last_load_bytes == 0


def test_http_export_import_roundtrip_end_to_end(monkeypatch, sandbox):
    """Full HTTP round-trip against the REAL engine cache primitives.

    Wires a live ``MemoryAwarePrefixCache`` (with a KVCache + ArraysCache
    entry) into a fake engine whose ``save_cache_to_disk`` /
    ``load_cache_from_disk`` delegate to the real cache methods, then drives
    ``POST /v1/cache/export`` followed by ``POST /v1/cache/import`` over the
    TestClient. Asserts the ArraysCache entry survives the full wire path."""
    from vllm_mlx.config import reset_config
    from vllm_mlx.memory_cache import MemoryAwarePrefixCache
    from vllm_mlx.routes.cache import router

    export_root = sandbox
    monkeypatch.setenv("RAPID_MLX_CACHE_EXPORT_DIR", str(export_root))

    src_cache, load_cfg = _build_cache_with_arrays_layer()
    dst_cache = MemoryAwarePrefixCache(model=object(), config=load_cfg)

    # The route runs save/load in an ``anyio.to_thread`` worker. MLX arrays
    # are stream-bound, so a worker thread that touches main-thread arrays
    # raises "There is no Stream(gpu, 0) in current thread" — exactly why the
    # REAL ``EngineCore`` routes these through its mlx-step thread which owns
    # the generation_stream (engine_core.py:73 binds
    # ``mx.default_stream(mx.default_device())``). We mimic that here: eval
    # the arrays on the main thread, then wrap the worker-side cache call in
    # the shared device default-stream context. This is test-harness plumbing
    # standing in for the step thread — production doesn't need it because
    # ``_run_on_step_thread`` already provides the stream.
    import mlx.core as mx

    for entry in src_cache._entries.values():
        for layer in entry.cache:
            state = getattr(layer, "state", None)
            if state is not None:
                mx.eval(state)

    def _on_default_stream(fn, *args):
        with mx.stream(mx.default_stream(mx.default_device())):
            return fn(*args)

    class _RealEngine:
        """Engine whose save/load delegate to real cache methods. The
        exporter uses ``src_cache``; the importer hydrates ``dst_cache``."""

        def __init__(self):
            self.scheduler = SimpleNamespace(
                memory_aware_cache=src_cache,
                config=SimpleNamespace(
                    kv_cache_dtype="bf16",
                    use_paged_cache=False,
                    kv_cache_turboquant=False,
                ),
            )
            self.config = SimpleNamespace(model_name="test-model", model_path=None)

        def save_cache_to_disk(self, cache_dir, should_abort=None):
            return _on_default_stream(src_cache.save_to_disk, cache_dir)

        def load_cache_from_disk(self, cache_dir, replace=False):
            # Importer loads into the DESTINATION cache — the real
            # load_from_disk path (compat gate included). ``replace`` forwards
            # to the real clear-inside-load contract (#1100 BLOCKING-4).
            return _on_default_stream(dst_cache.load_from_disk, cache_dir, replace)

        def save_cache_with_outcome(self, cache_dir, should_abort=None):
            # #1100 codex round 4 (#2): run the real save + capture the
            # authoritative outcome the SAME way the real engine wrapper does.
            from vllm_mlx.cache.protocol import SaveOutcome

            self.save_cache_to_disk(cache_dir, should_abort=should_abort)
            return SaveOutcome(outcome=src_cache._last_save_outcome)

        def load_cache_with_result(self, cache_dir, replace=False):
            from vllm_mlx.cache.protocol import LoadResult

            entries = self.load_cache_from_disk(cache_dir, replace=replace)
            return LoadResult(
                entries=entries, bytes_loaded=int(dst_cache._last_load_bytes)
            )

    cfg = reset_config()
    cfg.api_key = "test-secret"
    cfg.model_name = "test-model"
    cfg.engine = _RealEngine()

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    try:
        # Export → 200, entries_exported == 1, ArraysCache persisted.
        exp = client.post(
            "/v1/cache/export",
            json={"destination": "rt"},
            headers=_auth(),
        )
        assert exp.status_code == 200, exp.text
        assert exp.json()["entries_exported"] == 1

        # Import the same blob back → 200, entries_loaded == 1.
        imp = client.post(
            "/v1/cache/import",
            json={"source": "rt", "expected_model_id": "test-model"},
            headers=_auth(),
        )
        assert imp.status_code == 200, imp.text
        assert imp.json()["entries_loaded"] == 1
        assert imp.json()["entries_skipped"] == 0

        # The destination cache now holds the entry, ArraysCache intact.
        assert len(dst_cache._entries) == 1
        (entry,) = dst_cache._entries.values()
        layer_classes = [type(layer).__name__ for layer in entry.cache]
        assert "ArraysCache" in layer_classes, layer_classes
    finally:
        reset_config()


# ---------------------------------------------------------------------------
# #1100 codex round 7 direct-unit regressions for the non-route fixes.
# ---------------------------------------------------------------------------


def test_base_engine_save_outcome_default_distinguishes_failed_from_empty():
    """#1100 codex round 7 (#2): the ``BaseEngine.save_cache_with_outcome``
    DEFAULT must NOT collapse every ``save_cache_to_disk() is False`` into
    ``"empty"``. ``save_cache_to_disk`` returns ``False`` for BOTH a genuine
    empty no-op AND a NON-empty cache that failed to commit anything, so a
    subclass overriding only ``save_cache_to_disk`` (relying on the default
    outcome wrapper) would otherwise report a FAILED export as a successful
    empty snapshot — the route would then publish an empty manifest instead of
    500ing. The default disambiguates via authoritative ``get_cache_stats``:
    False + entries>0 → ``"failed"``; False + no entries → ``"empty"``.
    """
    from vllm_mlx.engine.base import BaseEngine

    # Concretize every abstract member with an inert stub so the ABC can be
    # instantiated; we only exercise the concrete ``save_cache_with_outcome``
    # default (NOT overridden) driving ``save_cache_to_disk`` + ``get_cache_stats``.
    _abstract_stubs = {
        name: (lambda self, *a, **k: None) for name in BaseEngine.__abstractmethods__
    }

    class _MinimalEngine(BaseEngine):
        """Overrides ONLY ``save_cache_to_disk`` (+ stats) — exactly the
        subclass shape the default outcome wrapper must serve correctly."""

        locals().update(_abstract_stubs)

        def __init__(self, *, save_returns: bool, entry_count: int):
            self._save_returns = save_returns
            self._entry_count = entry_count

        def get_cache_stats(self):
            return {"entry_count": self._entry_count}

        def save_cache_to_disk(self, cache_dir, should_abort=None):
            return self._save_returns

    # Committed save → "committed".
    committed = _MinimalEngine(save_returns=True, entry_count=3)
    assert committed.save_cache_with_outcome("/tmp/whatever").outcome == "committed"

    # Non-empty cache that committed nothing → "failed" (the bug this closes:
    # the old default returned "empty" here, hiding a failed export).
    failed = _MinimalEngine(save_returns=False, entry_count=3)
    assert failed.save_cache_with_outcome("/tmp/whatever").outcome == "failed"

    # Genuinely empty cache, save is a legit no-op → "empty".
    empty = _MinimalEngine(save_returns=False, entry_count=0)
    assert empty.save_cache_with_outcome("/tmp/whatever").outcome == "empty"

    # #1100 codex round 10 (#4): a subclass that overrides save_cache_to_disk
    # but NOT get_cache_stats (inheriting the None default) must FAIL CLOSED on
    # a False save — reporting "failed", not "empty" (an unauthoritative count
    # cannot prove emptiness, and a false "empty" ships a lying export).
    class _NoStatsEngine(BaseEngine):
        locals().update(_abstract_stubs)

        def save_cache_to_disk(self, cache_dir, should_abort=None):
            return False  # non-committing save, stats unavailable

        # get_cache_stats NOT overridden → BaseEngine default returns None.

    assert _NoStatsEngine().save_cache_with_outcome("/tmp/x").outcome == "failed"

    # A malformed stats shape (missing entry_count / non-numeric) is also
    # unauthoritative → "failed".
    class _BadStatsEngine(BaseEngine):
        locals().update(_abstract_stubs)

        def save_cache_to_disk(self, cache_dir, should_abort=None):
            return False

        def get_cache_stats(self):
            return {"entry_count": "not-a-number"}

    assert _BadStatsEngine().save_cache_with_outcome("/tmp/x").outcome == "failed"


def test_base_engine_load_result_default_tolerates_old_one_arg_signature():
    """#1100 codex round 9 (#4) → round 10 (#5): the ``load_cache_with_result``
    default must not break a pre-existing engine that overrides only the OLD
    one-arg ``load_cache_from_disk(self, cache_dir)`` (pre-#476), AND must decide
    the call shape by INTROSPECTION rather than catching ``TypeError`` — so a
    ``TypeError`` raised from INSIDE the method body is never mistaken for a
    signature mismatch and re-invoked (which would duplicate partial
    mutations)."""
    from vllm_mlx.engine.base import BaseEngine

    _stubs = {
        name: (lambda self, *a, **k: None) for name in BaseEngine.__abstractmethods__
    }

    class _OldMergeEngine(BaseEngine):
        locals().update(_stubs)

        def __init__(self):
            self.calls = []

        def load_cache_from_disk(self, cache_dir):  # OLD one-arg signature
            self.calls.append(("one-arg", cache_dir))
            return 7

    e = _OldMergeEngine()
    # Merge path must work with the legacy signature (no replace= keyword).
    res = e.load_cache_with_result("/tmp/x", replace=False)
    assert res.entries == 7
    assert e.calls == [("one-arg", "/tmp/x")]

    # A legacy one-arg engine CANNOT honor a replace — surface it, don't degrade.
    with pytest.raises(TypeError):
        _OldMergeEngine().load_cache_with_result("/tmp/x", replace=True)

    # A new-style engine accepting replace= still gets the keyword on replace.
    class _NewEngine(BaseEngine):
        locals().update(_stubs)

        def __init__(self):
            self.saw_replace = None

        def load_cache_from_disk(self, cache_dir, replace: bool = False):
            self.saw_replace = replace
            return 3

    n = _NewEngine()
    assert n.load_cache_with_result("/tmp/x", replace=True).entries == 3
    assert n.saw_replace is True

    # #1100 codex round 10 (#5): a TypeError raised from the method BODY (not a
    # signature mismatch) must propagate ONCE — never be swallowed and the load
    # re-invoked (which the old ``except TypeError`` retry would have done,
    # duplicating partial mutations). Introspection sees a compatible signature,
    # so the body error is surfaced as-is and the method runs exactly once.
    class _BodyRaisesEngine(BaseEngine):
        locals().update(_stubs)

        def __init__(self):
            self.invocations = 0

        def load_cache_from_disk(self, cache_dir, replace: bool = False):
            self.invocations += 1
            raise TypeError("boom from inside the body, not a signature mismatch")

    b = _BodyRaisesEngine()
    with pytest.raises(TypeError, match="boom from inside the body"):
        b.load_cache_with_result("/tmp/x", replace=False)
    assert b.invocations == 1  # called EXACTLY once — no double-invocation

    # A ``**kwargs`` signature is also recognized as replace-capable.
    class _KwargsEngine(BaseEngine):
        locals().update(_stubs)

        def load_cache_from_disk(self, cache_dir, **kwargs):
            return 1 if kwargs.get("replace") else 2

    assert _KwargsEngine().load_cache_with_result("/tmp/x", replace=True).entries == 1

    # #1111 codex r4: forwarding of ``protected_import`` must be DECOUPLED from
    # ``replace`` acceptance. An engine that accepts ``protected_import`` but NOT
    # ``replace`` must still receive a non-default ``protected_import=False`` —
    # the old nested-under-``accepts_replace`` code silently dropped it.
    class _ProtectedOnlyEngine(BaseEngine):
        locals().update(_stubs)

        def __init__(self):
            self.saw_protected = "unset"

        def load_cache_from_disk(self, cache_dir, protected_import: bool = True):
            self.saw_protected = protected_import
            return 5

    p = _ProtectedOnlyEngine()
    # replace left at default (False) → dropped silently; protected_import=False
    # is non-default and MUST be forwarded even though replace isn't accepted.
    assert (
        p.load_cache_with_result(
            "/tmp/x", replace=False, protected_import=False
        ).entries
        == 5
    )
    assert p.saw_protected is False, (
        "protected_import must forward independently of replace acceptance"
    )

    # Symmetric case: accepts ``replace`` but NOT ``protected_import``. A
    # non-default protected_import=False can't be honored → fail loudly rather
    # than silently upgrade to protected (which would re-open the growth bug).
    class _ReplaceOnlyEngine(BaseEngine):
        locals().update(_stubs)

        def load_cache_from_disk(self, cache_dir, replace: bool = False):
            return 9

    with pytest.raises(TypeError, match="protected_import"):
        _ReplaceOnlyEngine().load_cache_with_result(
            "/tmp/x", replace=True, protected_import=False
        )
    # But the DEFAULT protected_import=True is fine to drop silently.
    assert (
        _ReplaceOnlyEngine()
        .load_cache_with_result("/tmp/x", replace=True, protected_import=True)
        .entries
        == 9
    )


def test_read_committed_cache_counts_rejects_malformed_index_fail_closed(tmp_path):
    """#1100 codex round 7 (#3): ``_read_committed_cache_counts`` must FAIL
    CLOSED over the WHOLE index. The old loop skipped malformed entries yet
    returned ``len(entries_list)`` — counting entries it never validated — so a
    partially-torn index could satisfy ``require_committed_index=True`` and then
    CRASH the importer at ``entry_meta["index"]`` / ``["num_tokens"]``. Now ANY
    malformed entry (non-dict, missing / wrong-typed / negative required field)
    rejects the ENTIRE index (→ ``None``); a fully well-formed index returns the
    ``(count, total_bytes)`` summed from per-entry ``memory_bytes``.
    """
    from vllm_mlx.cache.protocol import _read_committed_cache_counts

    def _write_index(entries, version=3):
        (tmp_path / "index.json").write_text(
            json.dumps({"version": version, "entries": entries})
        )
        return tmp_path

    def _good(i):
        return {"index": i, "num_tokens": 4, "memory_bytes": 100}

    # Fully well-formed → (count, summed bytes).
    _write_index([_good(0), _good(1)])
    assert _read_committed_cache_counts(tmp_path) == (2, 200)

    # #1100 codex round 8 (#3): wrong index-format version is refused even when
    # every entry is well-formed — a manifest must never be published for a
    # snapshot whose on-disk format a peer's importer would reject.
    _write_index([_good(0)], version=1)  # below min (legacy pre-#198)
    assert _read_committed_cache_counts(tmp_path) is None
    _write_index([_good(0)], version=99)  # above max (future format)
    assert _read_committed_cache_counts(tmp_path) is None
    _write_index([_good(0)], version="3")  # non-int version
    assert _read_committed_cache_counts(tmp_path) is None

    # A single malformed entry poisons the whole index (fail-closed → None):
    _missing_mb = {"index": 1, "num_tokens": 4}  # memory_bytes defaults to 0 → VALID
    malformed_cases = [
        [_good(0), "not-a-dict"],  # non-dict entry
        [_good(0), {"num_tokens": 4, "memory_bytes": 100}],  # missing index
        [_good(0), {"index": 1, "memory_bytes": 100}],  # missing num_tokens
        [_good(0), _missing_mb],  # missing memory_bytes → default 0 (VALID)
        [_good(0), {"index": -1, "num_tokens": 4, "memory_bytes": 100}],  # negative
        [_good(0), {"index": True, "num_tokens": 4, "memory_bytes": 100}],  # bool≠int
        [_good(0), {"index": "1", "num_tokens": 4, "memory_bytes": 100}],  # str
        [_good(0), {"index": 1, "num_tokens": 4, "memory_bytes": -5}],  # neg bytes
        # #1100 codex round 9 (#2): non-finite / non-integral memory_bytes.
        # json parses NaN/Infinity; a bare ``mb < 0`` check would pass them and
        # int(NaN) would then CRASH the import instead of rejecting the index.
        [_good(0), {"index": 1, "num_tokens": 4, "memory_bytes": float("nan")}],
        [_good(0), {"index": 1, "num_tokens": 4, "memory_bytes": float("inf")}],
        [_good(0), {"index": 1, "num_tokens": 4, "memory_bytes": 1.5}],  # fractional
    ]
    for entries in malformed_cases:
        _write_index(entries)
        result = _read_committed_cache_counts(tmp_path)
        if entries[1] is _missing_mb:
            # memory_bytes absent defaults to 0 → this index is well-formed.
            assert result == (2, 100), (
                f"missing memory_bytes should default 0: {result}"
            )
        else:
            assert result is None, (
                f"malformed index not rejected: {entries!r} → {result!r}"
            )

    # No index.json at all → None (empty export).
    (tmp_path / "index.json").unlink()
    assert _read_committed_cache_counts(tmp_path) is None


def test_batched_engine_cache_ops_raise_when_inner_engine_absent():
    """#1100 codex round 8 (#4): a ``BatchedEngine`` whose inner engine hasn't
    started (``_engine is None``) must raise ``EngineNotReadyError`` from the
    export/import outcome forwarders — NOT mask it as a successful empty save /
    zero-entry load (which made export/import return 200 instead of the
    advertised 503 "engine not loaded"). The bare ``save_cache_to_disk`` keeps
    its no-op-False for lifespan persistence, where "no engine, nothing to
    persist" is legitimate."""
    from vllm_mlx.cache.protocol import EngineNotReadyError
    from vllm_mlx.engine.batched import BatchedEngine

    engine = BatchedEngine("fake-model")
    assert engine._engine is None  # not started — the finding's condition

    with pytest.raises(EngineNotReadyError):
        engine.save_cache_with_outcome("/tmp/whatever")
    with pytest.raises(EngineNotReadyError):
        engine.load_cache_with_result("/tmp/whatever")

    # The bare persistence methods stay quiet no-ops (lifespan startup/shutdown
    # persistence must not crash when there's no engine to persist).
    assert engine.save_cache_to_disk("/tmp/whatever") is False
    assert engine.load_cache_from_disk("/tmp/whatever") == 0


def test_export_import_503_when_inner_engine_not_ready(cache_client):
    """#1100 codex round 8 (#4): the cache route maps ``EngineNotReadyError``
    (raised when a non-None engine's inner engine hasn't started) to the SAME
    503 an absent engine gets — so "engine not loaded" is one consistent signal
    whether the OUTER engine is None or its INNER engine is down."""
    from vllm_mlx.cache.protocol import EngineNotReadyError

    class _NotReadyEngine(cache_client.FakeEngine):
        def save_cache_with_outcome(self, cache_dir, should_abort=None):
            raise EngineNotReadyError("inner engine not loaded")

        def load_cache_with_result(self, cache_dir, replace: bool = False):
            raise EngineNotReadyError("inner engine not loaded")

    engine = _NotReadyEngine(entries=1, current_memory=1024)
    engine.config.model_name = "test-model"
    cache_client.cfg.model_name = "test-model"
    cache_client.cfg.engine = engine

    exp = cache_client.client.post(
        "/v1/cache/export", json={"destination": "nr"}, headers=_auth()
    )
    assert exp.status_code == 503, exp.text

    # Import needs a manifest to reach the load call; write a matching one.
    _write_export_root(
        cache_client.sandbox,
        "nr-src",
        Manifest(protocol_version=PROTOCOL_VERSION, model_id="test-model", entries=1),
    )
    imp = cache_client.client.post(
        "/v1/cache/import", json={"source": "nr-src"}, headers=_auth()
    )
    assert imp.status_code == 503, imp.text
