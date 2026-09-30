# SPDX-License-Identifier: Apache-2.0
"""Experimental, opt-in bridge to TensorFold's Qwen3.8-27B decoder.

TensorFold owns its model, scheduler and caches.  This module owns only the
request/lifecycle boundary used by Rapid's existing HTTP and output layers.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import importlib.metadata
import json
import platform
import sys
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

SUPPORTED_VERSION = "0.5.0"
SUPPORTED_MLX_VERSION = "0.32.3"
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

    from huggingface_hub import snapshot_download

    target = snapshot_download(
        SUPPORTED_TARGET, revision=next(iter(SUPPORTED_TARGET_REVISIONS))
    )
    drafter = snapshot_download(
        SUPPORTED_DRAFTER, revision=next(iter(SUPPORTED_DRAFTER_REVISIONS))
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
            "tensorfold-qwen27 proof supports text chat only; unsupported: "
            + ", ".join(unsupported)
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


def require_runtime(version: str | None = None) -> None:
    try:
        found = version or importlib.metadata.version("tensorfold")
    except importlib.metadata.PackageNotFoundError as exc:
        raise TensorFoldUnavailable(
            "tensorfold-qwen27 requires the optional tensorfold==0.5.0 runtime"
        ) from exc
    if found != SUPPORTED_VERSION:
        raise TensorFoldUnavailable(
            f"tensorfold-qwen27 requires tensorfold=={SUPPORTED_VERSION}; found {found}"
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
        raise TensorFoldUnavailable("tensorfold-qwen27 requires mlx==0.32.3") from exc
    if found != SUPPORTED_MLX_VERSION:
        raise TensorFoldUnavailable(
            f"tensorfold-qwen27 requires mlx=={SUPPORTED_MLX_VERSION}; found {found}"
        )


class TensorFoldQwen27Backend:
    """One-lane adapter around a pinned TensorFold ``ChatApp``.

    ``app`` is injectable so lifecycle and protocol contracts are testable
    without loading model weights.  Production construction is deliberately
    centralized in :meth:`load` because TensorFold 0.5.0's builder is internal.
    """

    def __init__(
        self, app: Any, *, executor: concurrent.futures.Executor | None = None
    ):
        self._app = app
        self._executor = executor or concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tensorfold-qwen27"
        )
        self._owns_executor = executor is None
        self._closed = False
        self._active: dict[str, Any] = {}
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
    ) -> TensorFoldQwen27Backend:
        """Load the exact 0.5.0 family/app boundary used by the proof."""
        require_runtime()
        require_environment()
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
            target, lane_kernels="auto", drafter=str(drafter), drafter_bits=4
        )
        app = ChatApp(
            model,
            tokenizer,
            served_name=served_name,
            lanes=1,
            context_window=int(context_window),
            default_max_tokens=int(max_tokens),
            snapshot_dir=None,
        )
        return cls(app)

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
            raise RuntimeError("tensorfold-qwen27 backend is closed")
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
        self._closed = True
        with self._lock:
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
        self._app = None
