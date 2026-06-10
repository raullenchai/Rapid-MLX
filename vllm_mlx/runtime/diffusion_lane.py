"""Diffusion lane — discrete text-diffusion inference path.

This module is a **skeleton placeholder** for the DiffusionGemma
integration. It is intentionally not wired into any active code path:
no alias in ``aliases.json`` currently sets ``modality="text-diffusion"``,
so ``get_diffusion_runner`` is never reached at runtime. The skeleton
lets us land the modality field + dispatch shape in one PR (#TBD)
while the real implementation waits on:

  1. The active environment's ``mlx-vlm`` being upgraded to **v0.6.3 or
     later** — that release includes Blaizzy/mlx-vlm#1347 (Gemma 4
     DLM model files) and #1348 (DiffusionGemma long-context prefill
     fix). Confirmed import-OK as of 2026-06-10.
  2. ``pyproject.toml`` bumping the ``mlx-vlm`` dependency floor to
     ``>=0.6.3`` so a fresh ``pip install rapid-mlx`` doesn't land on a
     pre-DLM build and surface a confusing import error at request time.

Once the pyproject pin lands, fill in the bodies of ``load_runner``
and ``DiffusionRunner.generate`` to delegate to mlx-vlm's
block-streaming denoising loop (see ``mlx_vlm.server.generation``
upstream) and the SSE adapter in ``routes/chat.py`` will already have
the dispatch hook in place.

Why this lane exists at all
---------------------------
DiffusionGemma denoises a fixed-size canvas (default 256 tokens, MoE
3.8B-active) for K steps and emits the whole block at once, then
slides the window. This is incompatible with the auto-regressive
``Scheduler.step()`` loop in three ways:

  * No per-token logits stream — emission is block-granular.
  * No KV cache mutation per token — the canvas is overwritten in place.
  * Spec-decode + DFlash are silently meaningless (no draft tokens
    to verify when the whole block lands at once).

So instead of trying to shoehorn the diffusion path into ``Scheduler``,
we route at the ``modality`` boundary. The AR lane stays exactly as it
is today; this lane owns its own loop + its own SSE adapter.

Status: 2026-06-10 — skeleton landed, real implementation pending.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# Bumped when the wire format we expose to ``routes/chat.py`` changes
# (e.g. block-vs-token deltas, finish_reason semantics). Keeps the
# adapter contract explicit so a future mlx-vlm version bump that
# changes the upstream generator surface doesn't silently reshape our
# SSE output.
DIFFUSION_LANE_VERSION = "0.0-skeleton"


@dataclass(frozen=True)
class DiffusionGenerationConfig:
    """Sampling / decoding knobs for the diffusion lane.

    Mirrors the subset of mlx-vlm's diffusion generator parameters we
    intend to surface through ``/v1/chat/completions``. The real
    implementation will translate from ``ChatCompletionRequest`` →
    this dataclass at the dispatch boundary (``routes/chat.py``) so
    the diffusion runner never sees the OpenAI schema directly.
    """

    # Number of denoising steps per block. Lower = faster but lossier;
    # mlx-vlm default is 16. Surface to the API as ``max_tokens`` /
    # ``n`` analog once the real loop is wired.
    diffusion_steps: int = 16
    # Block size (tokens) — the canvas the denoiser operates on.
    # DiffusionGemma trains on 256; smaller values are valid but waste
    # the trained denoiser's capacity. Surfaced to the API only via a
    # ``stream`` flag in the request extras dict.
    block_tokens: int = 256
    # Temperature applied at the per-token argmax inside the denoiser.
    # 0.0 means greedy; matches the AR lane's convention.
    temperature: float = 0.0


class DiffusionRunner:
    """Wraps mlx-vlm's discrete-text-diffusion generator.

    NOT YET IMPLEMENTED. Holds the contract surface so callers
    (``routes/chat.py``, ``service/text.py``) can be written against
    a stable shape while the body waits on the mlx-vlm dependency.
    """

    def __init__(self, model: object, tokenizer: object, hf_path: str) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._hf_path = hf_path

    def generate(
        self,
        prompt_ids: list[int],
        config: DiffusionGenerationConfig,
    ) -> Any:
        """Block-stream denoised text.

        Real implementation will yield ``(block_text, is_final)`` tuples
        so the chat route can wrap each into an SSE ``delta.content``
        event. The non-streaming branch will fold blocks into a single
        completion string.

        Until the mlx-vlm dependency is released, calling this raises
        explicitly so a misconfigured alias surfaces at request time
        with a clear remediation hint rather than a confusing
        AttributeError deep in the engine loop.
        """
        raise NotImplementedError(
            "DiffusionRunner.generate is not implemented yet — this lane "
            "is gated on mlx-vlm >= 0.6.3 containing PRs Blaizzy/mlx-vlm"
            "#1347 + #1348. No alias should currently be routing here."
        )


def load_runner(hf_path: str) -> DiffusionRunner:
    """Construct a ``DiffusionRunner`` for the given HF path.

    Skeleton: refuses to load until the mlx-vlm dependency contains
    the upstream diffusion-gemma PRs. The dispatch tree in
    ``routes/chat.py`` will call this only for aliases with
    ``modality="text-diffusion"`` — and no such alias exists today,
    so this is dead code until the unblock arrives.
    """
    raise NotImplementedError(
        f"diffusion lane is gated on mlx-vlm >= 0.6.3 (hf_path={hf_path!r}). "
        "Verify with: "
        '`python -c "from mlx_vlm.models.diffusion_gemma.language import *"` '
        "and then remove this guard."
    )


__all__ = [
    "DIFFUSION_LANE_VERSION",
    "DiffusionGenerationConfig",
    "DiffusionRunner",
    "load_runner",
]
