# SPDX-License-Identifier: Apache-2.0
"""Shared MLLM prefill chunk sizing."""

# Upper bound on the per-forward chunk size used when prefilling a long
# text-only prompt on the MLLM path (issue #1187, Problem B). The actual
# chunk is ``min(prefill_step_size, MLLM_PREFILL_CHUNK_TOKENS)`` so an
# operator who deliberately set a *smaller* ``--prefill-step-size`` (a
# memory-constrained box) still gets chunks no larger than they asked for,
# while the large values needed to clear the per-batch cap (the #682
# reporter runs ``--prefill-step-size 22000``) no longer force the whole
# prompt through a single forward. 2048 measured the lowest peak on a
# gemma-4-26b 20k-token prefill (18.4 GB vs 20.0 GB at 4096, 35.2 GB
# single-shot) with no throughput cost — it is also mlx-lm's own text-lane
# default (``generate_step``/``PromptProcessingBatch`` prefill_step_size).
MLLM_PREFILL_CHUNK_TOKENS = 2048


def effective_mllm_prefill_chunk(prefill_step_size: int) -> int:
    return max(1, min(int(prefill_step_size), MLLM_PREFILL_CHUNK_TOKENS))
