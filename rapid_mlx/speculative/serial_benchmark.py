# SPDX-License-Identifier: Apache-2.0
"""Thread-affine serial runtime interface for in-process benchmarks."""

from __future__ import annotations

import gc
from collections.abc import Callable, Iterator
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenChunk:
    token: int
    prompt_tokens: int


@dataclass
class SerialRuntimeHandle:
    lane: str
    tokenizer: Any
    executor: Executor
    token_stream: Callable[[list[int], int], Iterator[TokenChunk]]
    speculative: dict
    target_path: str | None
    context_length: int | None
    close: Callable[..., None]
    prefill_chunk_size: int | None = None


class _NeverStop:
    def __call__(self, *args, **kwargs):
        return False

    def reset(self, *args, **kwargs):
        pass

    def add_eos_token_ids(self, *args, **kwargs):
        pass


def mlx_vlm_token_stream(model, processor, *, stream_generate_fn, generation_kwargs):
    """Build a generator factory; create and iterate it on the runtime executor."""

    def token_stream(prompt_ids: list[int], max_tokens: int) -> Iterator[TokenChunk]:
        if max_tokens <= 0:
            return
        import mlx.core as mx

        if stream_generate_fn is None:
            from mlx_vlm import stream_generate
        else:
            stream_generate = stream_generate_fn
        tokenizer = getattr(processor, "tokenizer", processor)
        original = tokenizer.stopping_criteria
        tokenizer.stopping_criteria = _NeverStop()
        inner = None
        try:
            inner = stream_generate(
                model,
                processor,
                tokenizer.decode(prompt_ids),
                input_ids=mx.array([prompt_ids]),
                **generation_kwargs(max_tokens),
            )
            count = 0
            for result in inner:
                if result.token is None:
                    continue
                yield TokenChunk(int(result.token), result.prompt_tokens)
                count += 1
                if count >= max_tokens:
                    break
        finally:
            try:
                if inner is not None:
                    inner.close()
            finally:
                tokenizer.stopping_criteria = original

    return token_stream


def pinned_cached_snapshot(repo: str, revision: str | None = None) -> str | None:
    """Find a cached snapshot directly, without validating its contents or networking."""
    from pathlib import Path

    from huggingface_hub import constants

    from rapid_mlx import model_aliases

    if Path(repo).is_dir():
        return str(Path(repo).resolve())
    pin = model_aliases.resolve_revision(repo)
    hub_repo = model_aliases.resolve_hub_model(repo) if pin else repo
    cache = Path(constants.HF_HUB_CACHE) / ("models--" + hub_repo.replace("/", "--"))
    revision = pin if revision is None else revision
    if revision is None:
        try:
            revision = (cache / "refs" / "main").read_text().strip()
        except OSError:
            return None
        if (
            not revision
            or revision in {".", ".."}
            or "/" in revision
            or "\\" in revision
        ):
            return None
    if (
        not revision
        or "\\" in revision
        or Path(revision).is_absolute()
        or any(part in {"", ".", ".."} for part in revision.split("/"))
    ):
        return None
    ref = cache / "refs" / revision
    if ref.is_file():
        try:
            revision = ref.read_text().strip()
        except OSError:
            return None
    if not revision or revision in {".", ".."} or "/" in revision or "\\" in revision:
        return None
    snapshot = cache / "snapshots" / revision
    if snapshot.resolve().parent != (cache / "snapshots").resolve():
        return None
    return str(snapshot) if snapshot.is_dir() else None


def cached_snapshot_path(repo, revision):
    """Return the local snapshot when it can be resolved without a download."""
    return pinned_cached_snapshot(repo, revision)


def _runtime_handle(lane, model, processor, runtime, kwargs_fn, facts, target_path):
    from rapid_mlx.speculative.dflash.server import _dflash_executor

    holder = {
        "model": model,
        "processor": processor,
        "runtime": runtime,
        "kwargs_fn": kwargs_fn,
    }

    def stream(prompt, maximum):
        return mlx_vlm_token_stream(
            holder["model"],
            holder["processor"],
            stream_generate_fn=None,
            generation_kwargs=lambda n: holder["kwargs_fn"](
                holder["runtime"], max_tokens=n, temperature=0.0, top_p=1.0
            ),
        )(prompt, maximum)

    kwargs = holder["kwargs_fn"](
        holder["runtime"], max_tokens=1, temperature=0.0, top_p=1.0
    )
    prefill = kwargs.get("prefill_step_size")
    if prefill is None:
        from mlx_vlm.generate.common import DEFAULT_PREFILL_STEP_SIZE

        prefill = DEFAULT_PREFILL_STEP_SIZE
    handle = SerialRuntimeHandle(
        lane=lane,
        tokenizer=getattr(processor, "tokenizer", processor),
        executor=_dflash_executor,
        token_stream=stream,
        speculative=facts,
        target_path=target_path,
        context_length=None,
        close=lambda *, wait=True: None,
        prefill_chunk_size=prefill,
    )

    closed = False

    def close(*, wait: bool = True):
        nonlocal closed
        if closed:
            return
        closed = True

        def cleanup():
            import mlx.core as mx

            handle.token_stream = None
            handle.tokenizer = None
            holder.clear()
            gc.collect()
            mx.clear_cache()

        # After a timeout, wait=False queues cleanup behind the in-flight MLX
        # step, which cannot be interrupted from another thread. The benchmark
        # reports immediately; the worker finishes or is reclaimed at exit.
        future = _dflash_executor.submit(cleanup)
        if wait:
            future.result()

    handle.close = close
    return handle


def dflash_runtime_handle(
    model,
    processor,
    runtime,
    *,
    lane="dflash",
    drafter_repo,
    drafter_revision,
    target_path,
):
    from mlx_vlm.speculative.common import _dflash_block_total

    from rapid_mlx.speculative.dflash.server import dflash_generation_kwargs

    # The runtime block includes one anchor, leaving block_total - 1 proposals.
    count = _dflash_block_total(runtime.drafter, None) - 1
    return _runtime_handle(
        lane,
        model,
        processor,
        runtime,
        dflash_generation_kwargs,
        {
            "method": "dflash",
            "max_draft_tokens": count,
            "draft_repo": runtime.drafter_repo,
            "draft_revision": runtime.drafter_revision,
            "draft_path": getattr(runtime, "drafter_path", None)
            or cached_snapshot_path(runtime.drafter_repo, runtime.drafter_revision),
            "algorithm": runtime.algorithm,
            "backend": "mlx_vlm",
        },
        target_path,
    )


def native_mtp_runtime_handle(
    model, processor, runtime, generation_kwargs_fn, *, pair, target_path
):
    # The configured block includes the anchor token, not a draft proposal.
    count = runtime.block_size - 1
    return _runtime_handle(
        "native_mtp",
        model,
        processor,
        runtime,
        generation_kwargs_fn,
        {
            "method": "mtp",
            "max_draft_tokens": count,
            "draft_repo": runtime.drafter_repo,
            "draft_revision": runtime.drafter_revision,
            "draft_path": getattr(runtime, "drafter_path", None)
            or cached_snapshot_path(runtime.drafter_repo, runtime.drafter_revision),
            "algorithm": runtime.algorithm,
            "backend": "mlx_vlm",
        },
        target_path,
    )


def companion_dspark_runtime_handle(
    model, processor, runtime, generation_kwargs_fn, *, pair, artifacts
):
    # This field directly counts proposals, excluding the block's anchor.
    count = runtime.num_speculative_tokens
    return _runtime_handle(
        "companion_dspark",
        model,
        processor,
        runtime,
        generation_kwargs_fn,
        {
            "method": "dspark",
            "max_draft_tokens": count,
            "draft_repo": runtime.drafter_repo,
            "draft_revision": runtime.drafter_revision,
            "draft_path": artifacts.drafter_path if artifacts else None,
            "algorithm": runtime.algorithm,
            "backend": "mlx_vlm",
        },
        artifacts.target_path if artifacts else None,
    )


def tensorfold_runtime_handle(
    backend,
    provider,
    *,
    lane,
    method,
    algorithm,
    draft_repo,
    draft_revision,
    draft_path,
    target_path,
    max_draft_tokens,
    lanes,
):
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tensorfold-bench")

    holder = {"backend": backend, "provider": provider}

    def token_stream(prompt_ids, n):
        iterator = holder["provider"]._outputs(
            "",
            prompt_token_ids=prompt_ids,
            ignore_eos=True,
            temperature=0.0,
            max_tokens=n,
        )
        count = 0
        try:
            for output in iterator:
                for token in output.new_token_ids:
                    yield TokenChunk(int(token), len(prompt_ids))
                    count += 1
                    if count >= n:
                        return
        finally:
            iterator.close()

    closed = False

    def close(*, wait: bool = True):
        nonlocal closed
        if closed:
            return
        closed = True

        def cleanup():
            import mlx.core as mx

            try:
                holder["backend"].close()
            finally:
                handle.token_stream = None
                handle.tokenizer = None
                holder.clear()
                gc.collect()
                mx.clear_cache()

        # After a timeout, wait=False queues cleanup behind the in-flight MLX
        # step, which cannot be interrupted from another thread. The benchmark
        # reports immediately; the worker finishes or is reclaimed at exit.
        try:
            future = executor.submit(cleanup)
            if wait:
                future.result()
        finally:
            executor.shutdown(wait=wait)

    context_window = getattr(backend._app, "context_window", None)
    if (
        isinstance(context_window, bool)
        or not isinstance(context_window, int)
        or context_window <= 0
    ):
        context_window = None

    handle = SerialRuntimeHandle(
        lane=lane,
        tokenizer=backend._app.tokenizer,
        executor=executor,
        token_stream=token_stream,
        speculative={
            "method": method,
            "algorithm": algorithm,
            "draft_repo": draft_repo,
            "draft_revision": draft_revision,
            "draft_path": draft_path,
            "max_draft_tokens": max_draft_tokens,
            "lanes": lanes,
        },
        target_path=target_path,
        context_length=context_window,
        close=close,
    )
    return handle


def tensorfold_live_max_draft(backend) -> int:
    """Read the actual engine cap and any loaded model draft limit."""
    engine = getattr(getattr(backend, "_app", None), "engine", None)
    cap = getattr(engine, "max_draft", None)
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise RuntimeError("Cannot determine TensorFold live max_draft")
    model = getattr(engine, "model", None)
    head = getattr(model, "head_drafts", None)
    drafter = getattr(head, "drafter", None)
    block = getattr(drafter, "block_size", None)
    if block is not None:
        if isinstance(block, bool) or not isinstance(block, int) or block < 2:
            raise RuntimeError("Cannot determine TensorFold drafter block size")
        cap = min(cap, block - 1)
    drafts = getattr(model, "drafts", None)
    if drafts is not None:
        if isinstance(drafts, bool) or not isinstance(drafts, int) or drafts < 1:
            raise RuntimeError("Cannot determine TensorFold model draft count")
        cap = min(cap, drafts)
    return cap
