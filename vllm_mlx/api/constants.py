# SPDX-License-Identifier: Apache-2.0
"""
Wire-level constants shared across the api / service layers.

Lives in ``vllm_mlx.api`` to preserve the layering rule that
``vllm_mlx.api`` is the lower layer (no engine / MLX imports) and
``vllm_mlx.service`` may depend on it. Constants placed here must
remain pure-literal so that importing this module never pulls in
MLX, Metal, or any engine code — adapter unit tests in headless
environments depend on this.
"""

#: Literal sentinel surfaced to ``content`` on ``finish_reason="length"``
#: when generation is cut short mid-think (default ON; opt out via
#: ``RAPID_MLX_REASONING_CUTOFF_NOTICE=disabled``). Defined here so the
#: ``api/responses_adapter`` can exclude it from its
#: ``downstream_output_seen`` check (issue #858 → PR #860 follow-up)
#: without dragging the engine into the adapter's import graph.
#: ``service.helpers`` re-exports this name as the public symbol so
#: existing callers (route helpers + their tests) need not change.
REASONING_CUTOFF_SENTINEL = "[truncated — reasoning incomplete; raise max_tokens]"
