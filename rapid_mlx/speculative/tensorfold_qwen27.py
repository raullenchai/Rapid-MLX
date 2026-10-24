# SPDX-License-Identifier: Apache-2.0
"""Experimental, opt-in bridge to TensorFold's Qwen3.8-27B decoder.

TensorFold owns its model, scheduler and caches.  This module owns only the
request/lifecycle boundary used by Rapid's existing HTTP and output layers.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import importlib.metadata
import json
import logging
import platform
import sys
import threading
import weakref
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .tensorfold_runtime import (
    INSTALL_HINT as INSTALL_HINT,
)
from .tensorfold_runtime import (
    SUPPORTED_MLX_VERSION,
    SUPPORTED_REVISION,
    SUPPORTED_RUNTIME_URL,
    SUPPORTED_VERSION,
)

logger = logging.getLogger(__name__)

SUPPORTED_MODEL_TYPE = "qwen3_5"
SUPPORTED_TARGET = "Vontra/Qwen3.8-27B-MLX-4bit"
SUPPORTED_DRAFTER = "z-lab/Qwen3.8-27B-DFlash2"
SUPPORTED_TARGET_REVISIONS = frozenset({"70ae7fac63274ff2eac54152031433374cb80f2f"})
SUPPORTED_DRAFTER_REVISIONS = frozenset({"50307d4c4cde6860d4eee73e2547cd786fe8e8a4"})


@dataclass(frozen=True)
class QualifiedPairArtifacts:
    target_path: str
    drafter_path: str


def download_qualified_pair() -> QualifiedPairArtifacts:
    """Resolve the product pair into the default HF cache at immutable SHAs."""

    from .._mirror import pinned_snapshot_download

    target = pinned_snapshot_download(
        SUPPORTED_TARGET, next(iter(SUPPORTED_TARGET_REVISIONS))
    )
    drafter = pinned_snapshot_download(
        SUPPORTED_DRAFTER, next(iter(SUPPORTED_DRAFTER_REVISIONS))
    )
    validate_pair(Path(target), Path(drafter))
    return QualifiedPairArtifacts(target_path=target, drafter_path=drafter)


SAMPLING_FIELDS = frozenset({"temperature", "top_p", "top_k", "min_p", "seed", "draft"})


class TensorFoldUnavailable(RuntimeError):  # noqa: N818 - public adapter API
    """The explicitly requested experimental backend cannot be started."""


class UnsupportedRequest(ValueError):  # noqa: N818 - public adapter API
    """A request cannot enter the text-only proof backend."""


@dataclass(frozen=True)
class BackendEvent:
    delta: str | dict[str, Any] | None = None
    reply: dict[str, Any] | None = None
    error: BaseException | None = None

    @property
    def terminal(self) -> bool:
        return self.reply is not None or self.error is not None


def validate_request(
    *,
    images: Any = None,
    videos: Any = None,
    grammar: Any = None,
    response_format: Any = None,
    tools: Any = None,
    sampling: dict[str, Any] | None = None,
) -> None:
    unsupported = [
        name
        for name, value in (
            ("images", images),
            ("videos", videos),
            ("grammar", grammar),
            ("response_format", response_format),
            ("tools", tools),
        )
        if value not in (None, [], {})
    ]
    if unsupported:
        raise UnsupportedRequest(
            "TensorFold text profiles do not support: " + ", ".join(unsupported)
        )
    unknown = set(sampling or ()) - SAMPLING_FIELDS
    if unknown:
        raise UnsupportedRequest(
            "unsupported sampling fields: " + ", ".join(sorted(unknown))
        )


def _snapshot_revision(path: Any) -> str:
    resolved = path.resolve()
    parts = resolved.parts
    try:
        return cast(str, parts[parts.index("snapshots") + 1])
    except (ValueError, IndexError) as exc:
        raise TensorFoldUnavailable(
            f"checkpoint is not a pinned Hugging Face snapshot: {resolved}"
        ) from exc


def validate_pair(target: Any, drafter: Any) -> None:
    """Reject every checkpoint outside the one measured pair before loading weights."""
    if _snapshot_revision(target) not in SUPPORTED_TARGET_REVISIONS:
        raise TensorFoldUnavailable("unqualified Qwen3.8-27B target revision")
    if _snapshot_revision(drafter) not in SUPPORTED_DRAFTER_REVISIONS:
        raise TensorFoldUnavailable("unqualified Qwen3.8-27B DFlash2 revision")
    try:
        target_config = json.loads((target / "config.json").read_text())
        draft_config = json.loads((drafter / "config.json").read_text())
    except (OSError, ValueError) as exc:
        raise TensorFoldUnavailable(
            "target and drafter require readable config.json files"
        ) from exc
    text = target_config.get("text_config") or {}
    quant = (
        target_config.get("quantization")
        or target_config.get("quantization_config")
        or {}
    )
    target_shape = (
        target_config.get("model_type") == SUPPORTED_MODEL_TYPE
        and text.get("model_type") == "qwen3_5_text"
        and text.get("hidden_size") == 5120
        and text.get("num_hidden_layers") == 64
        and text.get("vocab_size") == 248320
        and target_config.get("tie_word_embeddings") is False
        and quant.get("bits") == 4
        and quant.get("group_size") == 64
        and quant.get("mode") == "affine"
    )
    draft_shape = (
        draft_config.get("architectures") == ["DFlash2DraftModel"]
        and draft_config.get("hidden_size") == 5120
        and draft_config.get("num_hidden_layers") == 5
        and draft_config.get("vocab_size") == 248320
        and not draft_config.get("quantization")
        and not draft_config.get("quantization_config")
    )
    if not target_shape:
        raise TensorFoldUnavailable(
            "target is not the qualified Qwen3.8-27B 4-bit/group-64 layout"
        )
    if not draft_shape:
        raise TensorFoldUnavailable(
            "drafter is not the qualified Qwen3.8-27B DFlash2 layout"
        )


def _runtime_direct_url() -> dict[str, Any]:
    """Return pip's immutable VCS provenance for the installed runtime."""

    try:
        distribution = importlib.metadata.distribution("tensorfold")
        raw = distribution.read_text("direct_url.json")
        parsed = json.loads(raw) if raw else {}
        return parsed if isinstance(parsed, dict) else {}
    except (importlib.metadata.PackageNotFoundError, OSError, ValueError):
        return {}


def require_runtime(
    version: str | None = None, *, direct_url: dict[str, Any] | None = None
) -> None:
    try:
        found = version or importlib.metadata.version("tensorfold")
    except importlib.metadata.PackageNotFoundError as exc:
        raise TensorFoldUnavailable(
            f"tensorfold-qwen27 requires the optional tensorfold=={SUPPORTED_VERSION} runtime"
        ) from exc
    if found != SUPPORTED_VERSION:
        raise TensorFoldUnavailable(
            f"tensorfold-qwen27 requires tensorfold=={SUPPORTED_VERSION}; found {found}"
        )
    provenance = _runtime_direct_url() if direct_url is None else direct_url
    vcs = provenance.get("vcs_info") or {}
    if (
        provenance.get("url") != SUPPORTED_RUNTIME_URL
        or vcs.get("vcs") != "git"
        or vcs.get("commit_id") != SUPPORTED_REVISION
        or (provenance.get("dir_info") or {}).get("editable") is True
    ):
        raise TensorFoldUnavailable(
            "tensorfold-qwen27 requires the exact qualified TensorFold git revision"
        )


def require_environment(
    *, mlx_version: str | None = None, machine: str | None = None
) -> None:
    machine = machine or platform.machine()
    if sys.platform != "darwin" or machine != "arm64":
        raise TensorFoldUnavailable(
            "tensorfold-qwen27 proof requires Apple Silicon arm64/macOS"
        )
    try:
        found = mlx_version or importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError as exc:
        raise TensorFoldUnavailable(
            f"tensorfold-qwen27 requires mlx=={SUPPORTED_MLX_VERSION}"
        ) from exc
    if found != SUPPORTED_MLX_VERSION:
        raise TensorFoldUnavailable(
            f"tensorfold-qwen27 requires mlx=={SUPPORTED_MLX_VERSION}; found {found}"
        )


def _checkpoint_headroom() -> int:
    import mlx.core as mx
    import psutil
    from tensorfold.server.memory_budget import PROCESS_BYTES, memory_limit_bytes

    return int(
        min(
            psutil.virtual_memory().available,
            memory_limit_bytes(mx) - PROCESS_BYTES - mx.get_active_memory(),
        )
    )


def _checkpoint_engine(
    engine_type: Any,
    store_getter: Callable[[], Any],
    checkpoint_context: dict[str, tuple[int, tuple[int, ...]]],
) -> Any:
    """Keep one optional cut unless native selection without it would retain it."""

    class CheckpointEngine(engine_type):
        interior_tokens = None
        interior_releases = 0

        def _release_interior(self, store):
            tokens = self.interior_tokens
            self.interior_tokens = None
            if tokens is None:
                return
            # Public take transfers precisely this owned prefix; it does not
            # choose a victim. Never remove a state promoted to a system pin.
            probe = [*tokens, 0]
            usable = lambda n: n == len(tokens)
            entry = store.peek(probe, usable=usable)
            if entry is not None and not entry.pinned:
                if store.match(probe, usable=usable, take=True) is not None:
                    self.interior_releases += 1

        def _retention_fits(self, store, snapshots, pending=0):
            return len(store) + len(snapshots) + pending <= store.slots and (
                store.budget_bytes is None
                or store.nbytes
                + sum(int(self.cache_nbytes(cache)) for _, cache in snapshots)
                <= store.budget_bytes
            )

        def _guarded_steps(self, steps, stream, store, added, selections, cached):
            from tensorfold.server.cancellation import Cancellation, PrefillGuard

            engine = self
            boundaries = iter(
                sorted(n for n in selections if cached < n < len(stream.prompt_ids))
            )

            class Guard:
                def __init__(self, previous):
                    self.previous = previous

                def __getattr__(self, name):
                    return getattr(self.previous, name)

                def allow_checkpoint(self, cache):
                    boundary = next(boundaries)
                    if not self.previous.allow_checkpoint(cache):
                        return False
                    if boundary != added:
                        return True
                    # Size the populated cache before copy_single_cache allocates.
                    # Reserve a full-size copy for each remaining native cut too.
                    from tensorfold.server.memory_budget import CacheMemory

                    profile = CacheMemory.from_cache(cache)
                    required = sum(
                        profile.cache_bytes(n) for n in selections if n >= added
                    )
                    pending = sum(
                        engine.cache_nbytes(c) for _, c in stream.history_checkpoints
                    )
                    return (
                        store.budget_bytes is None
                        or store.nbytes + pending + required <= store.budget_bytes
                    ) and required <= _checkpoint_headroom()

            class Steps:
                def __iter__(self):
                    return self

                def _call(self, method, *args):
                    previous = getattr(engine, "prefill_guard", None)
                    engine.prefill_guard = Guard(
                        previous or PrefillGuard(Cancellation())
                    )
                    try:
                        return getattr(steps, method)(*args)
                    finally:
                        engine.prefill_guard = previous

                def __next__(self):
                    return self._call("__next__")

                def send(self, value):
                    return self._call("send", value)

                def throw(self, *args):
                    return self._call("throw", *args)

                def close(self):
                    return self._call("close")

            return Steps()

        def _native_selection_without_owned(self, store, stream, owned):
            from tensorfold.server.checkpoints import choose_checkpoints

            ids = stream.prompt_ids
            starts = self.prompt_chunks(ids)
            # prompt_fill.py:80-99: usable is chunk membership; take only changes
            # cache transfer, not selection. peek returns match's prior metadata
            # without copying, touching LRU, or changing last_prompt
            # (checkpoints.py:175-208). Reject just our compatible exact key.
            hit = store.peek(ids, usable=lambda n: n != len(owned) and n in starts)
            cached = 0 if hit is None else len(hit.tokens)
            last_prompt = None if hit is None else hit.last_prompt
            history, shared = checkpoint_context.get(stream.stream_id, (0, ()))
            chosen = choose_checkpoints(history, cached, last_prompt, ids)
            # prompt_fill.py:57-58,97-99: shared cuts use the same floor, then
            # both sources are filtered strictly beyond the counterfactual hit.
            shared_at = {starts.floor(n) for n in shared} - {0}
            return {
                at
                for at in {*(starts.floor(n) for n in chosen), *shared_at}
                if cached < at < len(ids)
            }

        def begin_stream(
            self, stream, *, cache=None, cached_tokens=0, checkpoints_at=()
        ):
            store = store_getter()
            if store is None:
                return super().begin_stream(
                    stream,
                    cache=cache,
                    cached_tokens=cached_tokens,
                    checkpoints_at=checkpoints_at,
                )
            original = set(checkpoints_at)
            ids = stream.prompt_ids
            owned = self.interior_tokens
            if owned is not None:
                entry = store.peek([*owned, 0], usable=lambda n: n == len(owned))
                compatible = ids[: len(owned)] == owned
                if entry is None or entry.pinned:
                    self.interior_tokens = None
                elif (
                    compatible
                    and cached_tokens == len(owned)
                    and len(owned)
                    in self._native_selection_without_owned(store, stream, owned)
                ):
                    # Public insert replaces this exact key without copying.
                    # Match refreshed last_prompt, but only insert refreshes born,
                    # which native eviction uses (checkpoints.py:244,275-283).
                    shared = checkpoint_context.get(stream.stream_id, (0, ()))[1]
                    store.insert(
                        owned,
                        entry.cache,
                        last_prompt=ids,
                        pinned=len(owned)
                        in {self.prompt_chunks(ids).floor(n) for n in shared},
                    )
                    self.interior_tokens = None
                elif not compatible or cached_tokens > len(owned):
                    # A later native boundary supersedes this optional tail.
                    self._release_interior(store)
            # Reserve original insertions before considering another optional cut.
            if not self._retention_fits(store, (), len(original)):
                self._release_interior(store)
            if not original and self.interior_tokens is None:
                return super().begin_stream(
                    stream,
                    cache=cache,
                    cached_tokens=cached_tokens,
                    checkpoints_at=checkpoints_at,
                )
            starts = [
                i for i, token in enumerate(ids) if token in self.prefill_plan.openers
            ]
            candidate = None
            if len(starts) >= 2 and starts[-1] - starts[-2] >= self.prefill_step:
                cuts = self.prompt_chunks(ids).starts or []
                boundary = min(starts[-1], max(original, default=starts[-1]))
                grid = [
                    at
                    for previous, at in zip(cuts, cuts[1:])
                    if at - previous == self.prefill_step
                    and at < boundary
                    and at not in original
                ]
                if grid:
                    candidate = grid[-1]
            added = None
            if (
                original
                and self.interior_tokens is None
                and candidate is not None
                and candidate > cached_tokens
                and len(store) + len(original) + 1 <= store.slots
            ):
                added = candidate
                checkpoints_at = sorted(original | {added})
            steps = super().begin_stream(
                stream,
                cache=cache,
                cached_tokens=cached_tokens,
                checkpoints_at=checkpoints_at,
            )
            if added is not None:
                steps = self._guarded_steps(
                    steps, stream, store, added, checkpoints_at, cached_tokens
                )
            if added is None and self.interior_tokens is None:
                return steps

            def retain():
                completed = False
                try:
                    result = yield from steps
                    completed = True
                    return result
                finally:
                    snapshots = stream.history_checkpoints
                    # Exhaustion/throw/close runs this before the scheduler takes
                    # history_checkpoints. Drop resident surplus before any insert.
                    if not self._retention_fits(store, snapshots):
                        self._release_interior(store)
                    if added is not None:
                        extra = next(
                            (pair for pair in snapshots if len(pair[0]) == added), None
                        )
                        progress = (
                            not completed
                            and extra is not None
                            and added == max(len(tokens) for tokens, _ in snapshots)
                        )
                        if (
                            extra is not None
                            and completed
                            and self._retention_fits(store, snapshots)
                        ):
                            self.interior_tokens = extra[0]
                        elif not progress:
                            # Only remove our extra. Keep every runtime history,
                            # shared-system and interrupted-prefill selection.
                            stream.history_checkpoints = [
                                pair for pair in snapshots if len(pair[0]) != added
                            ]

            return retain()

    return CheckpointEngine


class TensorFoldQwen27Backend:
    """Request adapter around a pinned TensorFold ``ChatApp``.

    ``app`` is injectable so lifecycle and protocol contracts are testable
    without loading model weights.  Production construction is deliberately
    centralized in :meth:`load` because TensorFold's builder is internal.
    """

    def __init__(
        self, app: Any, *, executor: concurrent.futures.Executor | None = None
    ):
        self._app = app
        self._checkpoint_context: dict[str, tuple[int, tuple[int, ...]]] = {}
        self._executor = executor or concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, int(getattr(app, "max_batch_size", 1))),
            thread_name_prefix="tensorfold-qwen27",
        )
        self._owns_executor = executor is None
        self._closed = False
        self._active: dict[str, Any] = {}
        self._provider_iterators: set[Any] = set()
        self._lock = threading.Lock()

    @classmethod
    def load(
        cls,
        target_dir: str,
        drafter_dir: str,
        *,
        served_name: str,
        context_window: int = 8192,
        max_tokens: int = 4096,
        lanes: int = 1,
        enable_prefix_cache: bool = True,
    ) -> TensorFoldQwen27Backend:
        """Load the pinned family/app boundary used by the proof."""
        from .tensorfold_concurrency import request_lanes

        lanes = request_lanes("qwen3.8-27b-tensorfold", lanes)
        require_runtime()
        require_environment()
        from tensorfold.engine.lane_engine import LaneEngine
        from tensorfold.engine.prefill_plan import PrefillPlan, message_markers
        from tensorfold.families import detect
        from tensorfold.server.app import ChatApp

        target = Path(target_dir)
        drafter = Path(drafter_dir)
        validate_pair(target, drafter)
        family = detect(target)
        package = family.package
        if getattr(package, "DRAFTER", None) != SUPPORTED_DRAFTER:
            raise TensorFoldUnavailable(
                "pinned Qwen3.8 DFlash2 family declaration is missing"
            )
        # The measured checkpoint stores source drafter weights without a
        # quantization block. TensorFold performs the qualified 4-bit packing
        # here; passing any other width is outside this adapter's contract.
        model, tokenizer = package.load(
            target,
            lane_kernels="auto",
            drafter=str(drafter),
            drafter_bits=4,
            **({"parallel": lanes} if lanes > 1 else {}),
        )
        # The runtime's bare engine defaults to a 2048-token grid. Without
        # message cuts, a shorter prompt's reusable history floors to zero
        # and no checkpoint is saved. Use the same reply-boundary plan as
        # TensorFold serving so recurrent state resumes at an exact cut.
        openers, assistant = message_markers(tokenizer)
        plan = PrefillPlan(LaneEngine.prefill_step, openers, 256, assistant)
        fraction = None
        if lanes > 1:
            from tensorfold.server.memory_budget import model_fraction

            fraction = model_fraction(package)

        checkpoint_context: dict[str, tuple[int, tuple[int, ...]]] = {}
        store_ref: Callable[[], Any] = lambda: None
        # The scheduler interleaves unfinished prefills when lanes > 1.
        # Admission and later transfer are not one transaction: the optional
        # cut's ownership/headroom policy therefore requires a single lane.
        engine_type = (
            _checkpoint_engine(LaneEngine, lambda: store_ref(), checkpoint_context)
            if lanes == 1
            else LaneEngine
        )
        app = ChatApp(
            model,
            tokenizer,
            served_name=served_name,
            engine_factory=functools.partial(
                engine_type,
                prefill_plan=plan,
            ),
            lanes=lanes,
            **({"memory_fraction": fraction} if lanes > 1 else {}),
            context_window=int(context_window),
            default_max_tokens=int(max_tokens),
            snapshot_dir=None,
            # Disk snapshots and in-memory prompt checkpoints are independent.
            checkpoint_slots=None if enable_prefix_cache else 0,
        )
        if app.checkpoints is not None:
            store_ref = weakref.ref(app.checkpoints)
        backend = cls(app)
        backend._checkpoint_context = checkpoint_context
        return backend

    async def stream(
        self,
        request_id: str,
        prompt_ids: list[int],
        *,
        max_tokens: int,
        sampling: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        **features: Any,
    ) -> AsyncIterator[BackendEvent]:
        if self._closed:
            raise RuntimeError("TensorFold backend is closed")
        validate_request(tools=tools, sampling=sampling, **features)
        from tensorfold.server.cancellation import Cancellation

        cancellation = Cancellation()
        with self._lock:
            if request_id in self._active:
                raise ValueError(f"duplicate request id: {request_id}")
            self._active[request_id] = cancellation
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[BackendEvent] = asyncio.Queue(maxsize=64)

        def emit(delta: str | dict[str, Any]) -> None:
            future = asyncio.run_coroutine_threadsafe(
                queue.put(BackendEvent(delta=delta)), loop
            )
            while True:
                try:
                    future.result(timeout=0.05)
                    return
                except concurrent.futures.TimeoutError:
                    if cancellation.cancelled:
                        future.cancel()
                        raise RuntimeError(
                            "request cancelled during stream backpressure"
                        )

        def terminal(event: BackendEvent) -> None:
            try:
                asyncio.run_coroutine_threadsafe(queue.put(event), loop)
            except RuntimeError:
                cancellation.cancel()

        def run() -> None:
            final_event: BackendEvent
            try:
                reply = self._app.chat(
                    [],
                    prompt=list(prompt_ids),
                    max_tokens=int(max_tokens),
                    temperature=float((sampling or {}).get("temperature", 0.0)),
                    sampling=dict(sampling or {}),
                    tools=None,
                    cancellation=cancellation,
                    on_delta=emit,
                )
                final_event = BackendEvent(
                    reply={
                        "content": reply.get("content", ""),
                        "reasoning": reply.get("reasoning"),
                        "finish_reason": reply.get("finish_reason") or "stop",
                        "prompt_tokens": int(
                            reply.get("prompt_tokens", len(prompt_ids))
                        ),
                        "completion_tokens": int(reply.get("completion_tokens", 0)),
                        "cached_tokens": int(reply.get("cached_tokens", 0)),
                        "speculative": reply.get("speculative"),
                    }
                )
            except BaseException as exc:  # terminal error must reach Rapid's route
                final_event = BackendEvent(error=exc)
            finally:
                with self._lock:
                    self._active.pop(request_id, None)
            terminal(final_event)

        try:
            self._executor.submit(run)
        except BaseException:
            with self._lock:
                self._active.pop(request_id, None)
            raise
        terminal_seen = False
        try:
            while True:
                event = await queue.get()
                yield event
                if event.terminal:
                    terminal_seen = True
                    break
        finally:
            # Closing the async iterator is Rapid's disconnect signal.  The
            # TensorFold worker may still be between scheduler checks, so set
            # its cancellation immediately and let it own cache disposal.
            if not terminal_seen:
                self.cancel(request_id)

    def cancel(self, request_id: str) -> bool:
        with self._lock:
            cancellation = self._active.get(request_id)
        if cancellation is None:
            return False
        cancellation.cancel()
        return True

    def close(self) -> None:
        with self._lock:
            self._closed = True
            active = list(self._active.values())
        for cancellation in active:
            cancellation.cancel()
        close = getattr(self._app, "close", None)
        if close is not None:
            close()
        else:
            scheduler = getattr(self._app, "scheduler", None)
            if scheduler is not None:
                scheduler.stop()
        if self._owns_executor:
            self._executor.shutdown(wait=True)
        # Finalize suspended providers even when the executor is borrowed.
        # A borrowed executor may still be pulling a running generator; its
        # close raises, but cancellation lets its own finally clean up later.
        with self._lock:
            iterators = list(self._provider_iterators)
            self._provider_iterators.clear()
        for iterator in iterators:
            try:
                iterator.close()
            except Exception:
                logger.warning(
                    "Failed to close TensorFold provider iterator", exc_info=True
                )
        self._app = None
