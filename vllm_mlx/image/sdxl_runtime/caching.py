"""First-Block Cache (FBCache): skip redundant transformer compute across steps.

Adjacent denoising steps feed the diffusion transformer almost-identical inputs,
so its output changes slowly. FBCache exploits this: each step it computes only the
*first* transformer block (cheap), and if that block's output barely moved since the
last full step, it reuses the cached residual of all the remaining blocks instead of
recomputing them — turning a full forward into ~1/N of the work on cacheable steps.

This is lossy in principle but near-lossless at small thresholds, and it is opt-in
(``threshold = 0`` disables it). It's the same idea as TeaCache / ParaAttention's
first-block cache, kept model-agnostic: the only signal is the relative change of the
first block's hidden state.
"""

from __future__ import annotations

import mlx.core as mx


class DeepCache:
    """DeepCache for U-Net diffusion: skip the deep blocks on most steps.

    A U-Net's deep (bottleneck) features change slowly across denoising steps, while
    the shallow blocks carry the high-frequency detail. DeepCache recomputes the full
    network only every ``interval`` steps; in between it runs just the shallowest
    down/up blocks and reuses the cached deep feature — skipping the most expensive
    levels (for SDXL, the 1280-channel / 10-transformer-layer blocks).

    ``interval = 1`` disables caching (every step full). ``interval = 2`` caches every
    other step (~1.5-1.8x). The first step is always full (no cache yet).
    """

    def __init__(self, interval: int = 1):
        self.interval = interval
        self.deep: mx.array | None = None
        self.steps = 0
        self.skipped = 0

    @property
    def enabled(self) -> bool:
        return self.interval > 1

    def reset(self) -> None:
        self.deep = None
        self.steps = 0
        self.skipped = 0

    def should_reuse(self) -> bool:
        """Return whether this step should reuse the cached deep feature."""
        reuse = (
            self.enabled and self.deep is not None and (self.steps % self.interval != 0)
        )
        self.steps += 1
        if reuse:
            self.skipped += 1
        return reuse


class FirstBlockCache:
    """Decide, per step, whether to reuse the cached transformer residual.

    Args:
        threshold: accumulated relative first-block change that triggers a full
            recompute. ``0`` disables caching (always full / exact). On WAN 2.1 1.3B
            (256px, 25 steps) ``0.1`` ≈ 1.5x and ``0.2`` ≈ 2.2x with no visible quality
            change (the sample differs but stays sharp and coherent); ``>= 0.3``
            starts to degrade. Higher = faster, lower fidelity.
    """

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
        self.prev_first: mx.array | None = None
        self.residual: mx.array | None = None
        self._accum = 0.0
        self.steps = 0
        self.skipped = 0

    def reset(self) -> None:
        """Clear state before a new generation."""
        self.prev_first = None
        self.residual = None
        self._accum = 0.0
        self.steps = 0
        self.skipped = 0

    @property
    def enabled(self) -> bool:
        return self.threshold > 0.0

    def should_reuse(self, first_residual: mx.array) -> bool:
        """Given this step's first-block *contribution*, return whether to reuse.

        Accumulates the relative L1 change of the first block's residual (its output
        minus its input); while the running total stays under ``threshold`` we reuse,
        and we force a recompute (resetting the accumulator) once it crosses. The
        first step, and any step before a residual exists, always recomputes.
        """
        self.steps += 1
        if not self.enabled or self.prev_first is None or self.residual is None:
            self.prev_first = first_residual
            return False
        rel = mx.mean(mx.abs(first_residual - self.prev_first)) / (
            mx.mean(mx.abs(self.prev_first)) + 1e-8
        )
        self.prev_first = first_residual
        self._accum += float(rel)  # one tiny scalar sync per step (~nothing vs a block)
        if self._accum < self.threshold:
            self.skipped += 1
            return True
        self._accum = 0.0
        return False
