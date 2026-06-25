# SPDX-License-Identifier: Apache-2.0
"""Block-diffusion drafter wrapper for DFlash spec decode (R15 task #313).

Defines the abstract interface the generator and the verifier expect
from a "block diffusion drafter": a small auxiliary model that emits
``block_size`` candidate tokens per forward pass given a prefix and
the current position.

Two concrete implementations live in this module:

1. :class:`MlxVlmBlockDiffusionDrafter` — a thin adapter around the
   mlx-vlm 0.5.0+ DFlash drafter loader (the same backend
   :mod:`vllm_mlx.speculative.dflash` uses). When the operator passes
   ``--spec-decode dflash`` against a real Qwen3.5/3.6 deployment and
   the matching drafter checkpoint is bound in the side-registry, this
   adapter is what loads and runs the forward.
2. :class:`StubBlockDiffusionDrafter` — a deterministic, MLX-allocation-
   free stub used by the generator unit tests and by the bench script
   dry-run. The stub takes a Python list of "scripted blocks" and
   emits them one at a time on each :meth:`draft_block` call. This is
   not the real drafter (no diffusion math, no weights) — it exists so
   the verifier / generator wiring can be exercised without holding
   the GPU.

The interface is intentionally narrow: one ``draft_block(prefix_tokens,
current_position)`` method returning a length-``block_size`` ``mx.array``
of candidate token IDs. The verifier owns the position math, the cache
write, and the longest-accepted-prefix decision — see
:mod:`vllm_mlx.spec_decode.dflash.verifier`.

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
# MlxVlmBlockDiffusionDrafter — production adapter
# ---------------------------------------------------------------------------


class MlxVlmBlockDiffusionDrafter:
    """Production adapter wrapping mlx-vlm 0.5.0+'s DFlash drafter.

    The adapter loads the drafter through the SAME
    :mod:`vllm_mlx.speculative.dflash.runtime` module the existing
    mlx-vlm bridge uses, so a deployment that has the drafter cached
    for the mlx-vlm bridge doesn't re-download for spec_decode.

    Construction is deferred until first :meth:`draft_block` so the
    CLI boot path doesn't pay the drafter-load cost if the operator
    asks for ``--spec-decode none``. The first call pays the load;
    subsequent calls hit the warm drafter.

    Args:
        drafter_hf_path: HF path / local path passed to mlx-vlm's
            :func:`load_drafter`. Examples: ``"z-lab/Qwen3.5-9B-DFlash"``
            (HF), ``"/path/to/drafter"`` (local).
        block_size: Number of tokens the drafter is configured to emit
            per forward. Defaults to 16 (the paper bench value).
    """

    def __init__(
        self,
        drafter_hf_path: str,
        block_size: int = 16,
    ) -> None:
        if not drafter_hf_path:
            raise ValueError("drafter_hf_path must be a non-empty string")
        if block_size <= 0:
            raise ValueError(f"block_size must be >= 1; got {block_size}")
        self.drafter_hf_path = drafter_hf_path
        self.block_size = block_size
        self._runtime: Any | None = None

    def _ensure_loaded(self) -> Any:
        """Lazy-load the underlying mlx-vlm drafter runtime.

        Defers the import + load until the first ``draft_block`` so
        a CLI that doesn't actually use DFlash never touches mlx-vlm.
        """
        if self._runtime is None:
            # Reuse the existing speculative bridge so a deployment
            # that has the drafter cached for mlx-vlm doesn't re-download.
            from vllm_mlx.speculative.dflash import load_runtime

            logger.info(
                "[dflash.drafter] Loading mlx-vlm DFlash drafter: %s (block_size=%d)",
                self.drafter_hf_path,
                self.block_size,
            )
            self._runtime = load_runtime(self.drafter_hf_path)
        return self._runtime

    def draft_block(
        self,
        prefix_tokens: mx.array,
        current_position: int,
    ) -> mx.array:
        """Run one drafter forward pass and return the block.

        Delegates to the mlx-vlm drafter's ``_dflash_rounds`` call
        adapter (the same surface the bridge uses). The mlx-vlm
        drafter returns a ``(block_size,)`` array of token IDs at
        every successful round; we forward that unchanged.

        On a drafter failure (mlx-vlm not installed, drafter
        download blocked, GPU OOM) we propagate the exception — the
        generator's outer loop catches it and falls back to the
        no-spec-decode path for that step. We don't catch + suppress
        here because the upper-layer fallback is the right place to
        decide between "skip this attempt and continue" vs "disable
        spec-decode for the rest of the session".
        """
        runtime = self._ensure_loaded()
        drafter = runtime.drafter
        # The mlx-vlm drafter exposes ``draft_block(prefix, position)``
        # in v0.5.0+ per the bridge's surface contract. Older versions
        # use a different name; an AttributeError here would surface as
        # a clear "drafter doesn't implement DFlash block interface"
        # message to the operator.
        block = drafter.draft_block(prefix_tokens, current_position)
        # Defensive shape check: the verifier downstream relies on the
        # block having the right size; a drafter that silently returns
        # a shorter block would corrupt the position-id contract.
        if block.shape[0] != self.block_size:
            raise RuntimeError(
                f"mlx-vlm drafter returned block of size "
                f"{int(block.shape[0])}, expected {self.block_size}. "
                "Check drafter checkpoint's block_size config matches "
                "the rapid-mlx spec_decode/dflash default."
            )
        return block.astype(mx.uint32)

    def reset(self) -> None:
        """Reset per-request drafter state.

        Delegates to the mlx-vlm bridge's ``reset_accept_lens`` so the
        accept-len list doesn't pool across requests; it also clears
        the drafter's internal KV cache via the mlx-vlm
        ``DFlashRuntime`` adapter when the cache attribute is present.
        """
        if self._runtime is None:
            return
        self._runtime.reset_accept_lens()
        # Clear the drafter's KV cache if the mlx-vlm drafter exposes
        # one. Tolerant of versions that don't.
        drafter = self._runtime.drafter
        if hasattr(drafter, "reset_cache"):
            drafter.reset_cache()
