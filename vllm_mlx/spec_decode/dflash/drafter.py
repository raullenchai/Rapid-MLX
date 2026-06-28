# SPDX-License-Identifier: Apache-2.0
"""Block-diffusion drafter wrapper for DFlash spec decode (R15 task #313).

Defines the abstract interface the generator and the verifier expect
from a "block diffusion drafter": a small auxiliary model that emits
``block_size`` candidate tokens per forward pass given a prefix and
the current position.

Three concrete pieces live in this module:

1. :class:`BlockDiffusionDrafter` — the per-block ``Protocol`` consumed
   by :mod:`vllm_mlx.spec_decode.dflash.generator` and ...verifier.
   This per-block contract is the **0.10 interface** — the standardized
   ``--spec-decode dflash`` integration that pairs with rapid-mlx's own
   generator/verifier. It is NOT what mlx-vlm 0.6.3's DFlash drafter
   actually exposes (see #3 below); the rapid-mlx generator/verifier
   pair around this Protocol remains scaffolding until the BatchedEngine
   integration lands.
2. :class:`StubBlockDiffusionDrafter` — deterministic, MLX-allocation-
   free stub that satisfies the Protocol. Used by the generator /
   verifier unit tests and by ``--dry-run`` smoke runs. Not a model —
   no diffusion math, no weights, no GPU.
3. :class:`MlxVlmDFlashDriver` — production driver. Wraps mlx-vlm
   0.6.3's full DFlash round loop (target prefill → hidden capture →
   :func:`mlx_vlm.speculative.dflash._dflash_rounds`) behind a single
   :meth:`generate` method. Unlike #1 it is **not** a per-block adapter:
   mlx-vlm 0.6.3's ``DFlashDraftModel.draft_block`` requires the target
   model's captured hidden states as input (it is a hidden-state
   conditioned diffusion model), so a per-block adapter cannot satisfy
   it without owning the verify forward too. Rather than re-implement
   the full DFlash control flow in rapid-mlx (the previous adapter at
   this site tried and silently broke on any real call — the bench
   could never produce a number), this class delegates the whole loop
   to mlx-vlm's reference implementation and surfaces accept-rate /
   tok-saved stats through the underlying drafter's ``accept_lens`` /
   ``draft_lens`` lists.

Why the architecture split
--------------------------

The :class:`BlockDiffusionDrafter` Protocol describes "what an ideal
per-block drafter looks like" from rapid-mlx's BatchedEngine point of
view — one ``draft_block`` call returns the next block's candidate
tokens, the verifier owns the rest. Today that contract is satisfied
ONLY by the stub: the production mlx-vlm drafter cannot fit it because
it needs target hidden states as input. When the BatchedEngine
integration lands (0.10), we'll either:

* extend the Protocol to thread hidden states through ``draft_block``
  AND teach the verifier to capture + pass them, OR
* re-architect around a "drive the whole loop" driver model that
  matches mlx-vlm's contract more directly (which is what
  :class:`MlxVlmDFlashDriver` already does).

Until then, the bench script and the dflash-server-mode use
:class:`MlxVlmDFlashDriver`; unit tests of the
:mod:`...spec_decode.dflash.generator` / .verifier code path use
:class:`StubBlockDiffusionDrafter`.

Why not a duck-typed callable
-----------------------------

The interface is a Protocol rather than a free function so:

* The drafter can carry state across calls (the mlx-vlm drafter caches
  its hidden state across rounds; resetting per request is a
  :meth:`reset` call on the same instance).
* Tests can swap a stub with an identical surface without monkey-patching
  the import path.
* A future "tree-DFlash" drafter (variant of the paper that branches
  the block into multiple candidate paths) gets a clean place to land
  via a new method on the same Protocol without disturbing the chain
  caller.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any, Protocol

import mlx.core as mx

logger = logging.getLogger(__name__)


class BlockDiffusionDrafter(Protocol):
    """Abstract block-diffusion drafter interface.

    The verifier calls :meth:`draft_block` once per outer loop
    iteration. The drafter is responsible for its own internal cache
    state (the verifier never touches it); :meth:`reset` clears that
    state between requests so per-request acceptance metrics aren't
    pooled across sessions.

    Attributes:
        block_size: Number of draft tokens emitted per ``draft_block``
            call. Fixed for the drafter instance (the underlying model
            is typically trained for a specific block size). The
            verifier reads this to size its position-id tensor.
    """

    block_size: int

    def draft_block(
        self,
        prefix_tokens: mx.array,
        current_position: int,
    ) -> mx.array:
        """Emit ``block_size`` candidate tokens for the next block.

        Args:
            prefix_tokens: 1-D ``mx.uint32`` token IDs for the
                already-emitted sequence so far. The drafter is free
                to consume only the last few — the full prefix is
                passed so a stateless or context-windowed drafter can
                pick its own cutoff.
            current_position: The position index of the FIRST token in
                the block being drafted. Equal to ``len(emitted_so_far)``
                in the simple case. The verifier passes this so the
                drafter can advance its internal positional state
                (rotary embeddings, position-aware drafter heads) to
                the correct slot.

        Returns:
            1-D ``mx.uint32`` array of shape ``(block_size,)`` with the
            candidate token IDs. The verifier reads them in order.
        """
        ...

    def reset(self) -> None:
        """Clear per-request drafter state.

        Called by the generator at the START of each request. Concrete
        implementations should clear any KV cache the drafter built
        for previous prompts.
        """
        ...


# ---------------------------------------------------------------------------
# StubBlockDiffusionDrafter — deterministic, MLX-allocation-free
# ---------------------------------------------------------------------------


class StubBlockDiffusionDrafter:
    """Deterministic stub drafter used by unit tests and bench dry-runs.

    Takes a Python list of "scripted blocks" at construction. Each
    :meth:`draft_block` call consumes the next block from the script
    and returns it as an ``mx.uint32`` array. When the script runs out,
    raises ``IndexError`` — tests should pad with sentinel blocks.

    The stub is NOT a model. It does not learn from the prefix, it
    does not implement diffusion math, it does not hold weights. It
    exists so the verifier / generator wiring can be exercised end-to-
    end without holding the GPU and without depending on the mlx-vlm
    drafter being cached locally.

    Args:
        scripted_blocks: List of length-``block_size`` lists of token
            IDs. Each outer-list entry is consumed in order by
            successive :meth:`draft_block` calls.
        block_size: Block size the verifier should expect. Must match
            ``len(scripted_blocks[i])`` for every entry.
    """

    def __init__(
        self,
        scripted_blocks: list[list[int]],
        block_size: int = 16,
    ) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be >= 1; got {block_size}")
        for i, block in enumerate(scripted_blocks):
            if len(block) != block_size:
                raise ValueError(
                    f"scripted_blocks[{i}] has length {len(block)}; "
                    f"expected block_size={block_size}"
                )
        self.block_size = block_size
        self._script: list[list[int]] = [list(b) for b in scripted_blocks]
        self._cursor = 0
        self._draft_calls = 0
        self._reset_calls = 0

    def draft_block(
        self,
        prefix_tokens: mx.array,
        current_position: int,
    ) -> mx.array:
        """Emit the next scripted block as an ``mx.uint32`` array."""
        if self._cursor >= len(self._script):
            raise IndexError(
                f"StubBlockDiffusionDrafter script exhausted after "
                f"{self._cursor} blocks; pad the script if the verifier "
                "needs more attempts."
            )
        block = self._script[self._cursor]
        self._cursor += 1
        self._draft_calls += 1
        # current_position / prefix_tokens are intentionally unused —
        # the stub is scripted. Reading them here is enough for the
        # type-checker to track that the Protocol is satisfied.
        _ = prefix_tokens
        _ = current_position
        return mx.array(block, dtype=mx.uint32)

    def reset(self) -> None:
        """Reset the script cursor and call counters.

        Lets the same stub instance be reused across multiple test
        requests without re-constructing.
        """
        self._cursor = 0
        self._draft_calls = 0
        self._reset_calls += 1


# ---------------------------------------------------------------------------
# MlxVlmDFlashDriver — production driver around mlx-vlm 0.6.3
# ---------------------------------------------------------------------------


class MlxVlmDFlashDriver:
    """Production DFlash driver — wraps mlx-vlm 0.6.3's ``stream_generate``.

    Background — why this is NOT a per-block adapter
    ------------------------------------------------

    mlx-vlm 0.6.3 ships ``DFlashDraftModel.draft_block`` with the
    signature::

        draft_block(self, last_bonus, hidden, cache, block_size,
                    sampler, token_dtype=mx.int32) -> mx.array

    The ``hidden`` arg is the TARGET model's captured hidden states for
    the previously-emitted positions — DFlash is a hidden-state-
    conditioned diffusion drafter, not a token-only one. A per-block
    adapter that knows only ``(prefix_tokens, current_position)`` (the
    previous adapter contract here, and the
    :class:`BlockDiffusionDrafter` Protocol above) cannot synthesize
    those hidden states; the verifier forward owns them, and the
    drafter's first call needs the prefill hidden state. Wiring this
    into rapid-mlx's own verifier/generator pair would require
    re-implementing mlx-vlm's :func:`_dflash_rounds` control flow —
    explicitly out of scope for 0.9.

    Instead, this driver wraps the WHOLE round loop. It loads target
    + drafter via mlx-vlm's loaders, then exposes
    :meth:`generate` which delegates to ``mlx_vlm.stream_generate``
    with ``draft_model`` and ``draft_kind`` set. The same code path the
    DFlash server mode uses (see :mod:`vllm_mlx.speculative.dflash.server`).
    The bench script consumes the wrapper's generator and reads accept-
    rate / draft-len telemetry from the underlying drafter's
    ``accept_lens`` and ``draft_lens`` lists, which mlx-vlm populates as
    a side effect of the loop.

    Concurrency
    -----------

    mlx-vlm's DFlash path is single-stream (mlx-vlm 0.6.3 has no batched
    DFlash kernel). Callers that need concurrency should serialize
    through an external lock — see
    :mod:`vllm_mlx.speculative.dflash.server` for the pattern.

    Args:
        target_repo: HF path / local path of the target model. Loaded
            via :func:`mlx_vlm.utils.load`.
        drafter_repo: HF path / local path of the DFlash drafter.
            Loaded via :func:`vllm_mlx.speculative.dflash.load_runtime`.
        block_size: Default block size override passed to mlx-vlm's
            ``draft_block_size`` kwarg. Defaults to ``None`` so mlx-vlm
            uses the drafter checkpoint's trained value (typically 16).
            When set, mlx-vlm uses it as the CEILING; the adaptive
            ``_dflash_next_block_size`` heuristic still scales down on
            poor acceptance unless the drafter sets
            ``prefer_requested_block_size=True``.
    """

    def __init__(
        self,
        target_repo: str,
        drafter_repo: str,
        *,
        block_size: int | None = None,
    ) -> None:
        if not target_repo:
            raise ValueError("target_repo must be a non-empty string")
        if not drafter_repo:
            raise ValueError("drafter_repo must be a non-empty string")
        if block_size is not None and block_size <= 0:
            raise ValueError(f"block_size must be >= 1 or None; got {block_size}")
        self.target_repo = target_repo
        self.drafter_repo = drafter_repo
        self.block_size = block_size
        self._target: Any | None = None
        self._processor: Any | None = None
        self._runtime: Any | None = None

    @property
    def loaded(self) -> bool:
        """True once :meth:`load` has materialized target + drafter."""
        return self._target is not None and self._runtime is not None

    @property
    def target(self) -> Any:
        """The loaded mlx-vlm target model. Raises if :meth:`load` hasn't run."""
        if self._target is None:
            raise RuntimeError("MlxVlmDFlashDriver.load() must be called first")
        return self._target

    @property
    def processor(self) -> Any:
        """The loaded mlx-vlm processor / tokenizer. Raises if not loaded."""
        if self._processor is None:
            raise RuntimeError("MlxVlmDFlashDriver.load() must be called first")
        return self._processor

    @property
    def runtime(self) -> Any:
        """The :class:`vllm_mlx.speculative.dflash.DFlashRuntime` handle."""
        if self._runtime is None:
            raise RuntimeError("MlxVlmDFlashDriver.load() must be called first")
        return self._runtime

    def load(self) -> None:
        """Materialize the target + drafter via mlx-vlm loaders.

        Idempotent — a second call is a no-op. Defers all heavy I/O
        until first use so test harnesses can construct the driver
        without paying the GB-scale weight load just to verify wiring.

        Caller is responsible for thread affinity: mlx-lm 0.31.3+ keeps
        GPU streams in thread-local storage. Loading on thread A and
        generating on thread B will crash. The DFlash server pins both
        to a single-worker executor (see
        :mod:`vllm_mlx.speculative.dflash.server` for the pattern).
        """
        if self.loaded:
            return
        # Lazy imports — leave mlx-vlm optional at module import time so
        # an installation without ``[dflash]`` extras keeps unit tests
        # of the Stub working.
        from mlx_vlm import load as _mlx_vlm_load

        from vllm_mlx.speculative.dflash import load_runtime

        logger.info("[dflash.driver] Loading target via mlx-vlm: %s", self.target_repo)
        self._target, self._processor = _mlx_vlm_load(self.target_repo)
        logger.info(
            "[dflash.driver] Loading DFlash drafter: %s (block_size=%r)",
            self.drafter_repo,
            self.block_size,
        )
        self._runtime = load_runtime(self.drafter_repo)

    def adopt(
        self,
        *,
        target: Any,
        processor: Any,
        runtime: Any,
    ) -> None:
        """Inject already-loaded target + processor + runtime objects.

        For callers that have already paid the mlx-vlm load cost
        elsewhere (e.g. a bench script that wants ONE shared target
        instance across baseline and DFlash conditions, instead of
        loading 28+ GB of weights twice). After :meth:`adopt`,
        :meth:`generate` and :meth:`accept_stats` work the same way
        they would after :meth:`load`.

        Args:
            target: An ``mlx_vlm.load``-style model object — must
                expose ``language_model`` and the DFlash hooks
                (``capture_layer_ids``, ``rollback_speculative_cache``).
                The bench script gets one of these via ``mlx_vlm.load``.
            processor: The matching mlx-vlm processor / tokenizer.
            runtime: A
                :class:`vllm_mlx.speculative.dflash.DFlashRuntime` — the
                loaded drafter + kind. Get one from
                :func:`vllm_mlx.speculative.dflash.load_runtime`.

        Raises:
            ValueError: If the driver was already loaded or adopted.
        """
        if self.loaded:
            raise ValueError(
                "MlxVlmDFlashDriver is already loaded; adopt() is for "
                "fresh instances. Construct a new driver."
            )
        if target is None or processor is None or runtime is None:
            raise ValueError("adopt() requires non-None target, processor, runtime")
        self._target = target
        self._processor = processor
        self._runtime = runtime

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Generator[Any, None, None]:
        """Drive the full DFlash round loop and yield mlx-vlm chunks.

        Args:
            prompt: The full prompt string. mlx-vlm's
                ``stream_generate`` handles tokenization via the
                processor.
            max_tokens: Generation budget. Forwarded to
                ``stream_generate``.
            temperature: Sampling temperature. ``0.0`` selects greedy.
            top_p: Nucleus sampling threshold. ``1.0`` disables it.

        Yields:
            ``mlx_vlm`` ``GenerationResult`` chunks, one per emitted
            token. Each carries ``.text`` (incremental decode),
            ``.token`` (the int id), ``.generation_tokens`` (cumulative
            counter), and ``.prompt_tokens``. The bench script reads
            ``.generation_tokens`` for tok/s; the accept-rate is
            sourced from :meth:`accept_stats` after the generator is
            exhausted.

        Raises:
            RuntimeError: If :meth:`load` hasn't been called.

        Notes:
            Resets the drafter's per-request ``accept_lens`` and
            ``draft_lens`` so accept-rate isn't pooled across calls.
        """
        if not self.loaded:
            raise RuntimeError(
                "MlxVlmDFlashDriver.generate() requires load() to be called first"
            )
        # Clear per-request state so the accept-rate snapshot below
        # only reflects this prompt's rounds. (mlx-vlm's
        # ``DFlashDraftModel.reset`` is called by ``_dflash_rounds`` at
        # the start of every loop, but a small belt-and-suspenders.)
        self.runtime.reset_accept_lens()
        drafter = self.runtime.drafter
        if hasattr(drafter, "draft_lens") and isinstance(drafter.draft_lens, list):
            drafter.draft_lens.clear()

        # Lazy import — same rationale as in load().
        from mlx_vlm import stream_generate

        gen_kwargs: dict[str, Any] = dict(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            draft_model=drafter,
            draft_kind=self.runtime.kind,
        )
        if self.block_size is not None:
            gen_kwargs["draft_block_size"] = self.block_size
        # NOTE: deliberately not ``yield from`` here — a bare ``yield
        # from`` would leak the upstream generator on early caller
        # break (e.g. the bench's ``if n >= max_tokens: break``). The
        # try/finally ensures ``stream_generate``'s GeneratorExit
        # cleanup runs and the drafter's KV cache is released before
        # the next prompt starts. mlx-vlm transiently doubles GPU
        # memory if a generator's KV state isn't released — without
        # this close() the next prompt would race a half-freed cache.
        upstream = stream_generate(self.target, self.processor, prompt, **gen_kwargs)
        try:
            for chunk in upstream:
                yield chunk
        finally:
            close = getattr(upstream, "close", None)
            if callable(close):
                close()

    def accept_stats(self) -> dict[str, Any]:
        """Return per-request DFlash accept stats from the underlying drafter.

        ``mlx_vlm`` stores per-round acceptance as two lists on the
        drafter: ``accept_lens[i]`` = positions accepted in round ``i``,
        ``draft_lens[i]`` = positions drafted in that round (always
        equal to ``block_size - 1`` since the first slot is the
        always-accepted bonus). Both are reset by :meth:`generate`.

        Returns:
            Dict with:
              - ``attempts`` — number of verify rounds (``len(accept_lens)``).
              - ``accepted_tokens`` — sum over accepted positions.
              - ``drafted_tokens`` — sum of drafted positions.
              - ``accept_rate`` — ``accepted / drafted`` (0.0 when no rounds).
              - ``mean_accepted_per_attempt`` — ``accepted / attempts``.
              - ``accept_lens`` — a copy of the per-round list.
              - ``draft_lens`` — a copy of the per-round list.
        """
        if self._runtime is None:
            return {
                "attempts": 0,
                "accepted_tokens": 0,
                "drafted_tokens": 0,
                "accept_rate": 0.0,
                "mean_accepted_per_attempt": 0.0,
                "accept_lens": [],
                "draft_lens": [],
            }
        drafter = self._runtime.drafter
        accept_lens = list(getattr(drafter, "accept_lens", []) or [])
        draft_lens = list(getattr(drafter, "draft_lens", []) or [])
        attempts = len(accept_lens)
        accepted = int(sum(accept_lens))
        drafted = int(sum(draft_lens))
        accept_rate = accepted / drafted if drafted > 0 else 0.0
        mean_accept = accepted / attempts if attempts > 0 else 0.0
        return {
            "attempts": attempts,
            "accepted_tokens": accepted,
            "drafted_tokens": drafted,
            "accept_rate": accept_rate,
            "mean_accepted_per_attempt": mean_accept,
            "accept_lens": accept_lens,
            "draft_lens": draft_lens,
        }


__all__ = [
    "BlockDiffusionDrafter",
    "StubBlockDiffusionDrafter",
    "MlxVlmDFlashDriver",
]
