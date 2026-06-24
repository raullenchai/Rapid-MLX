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


def is_rescue_payload(content: str | None) -> bool:
    """Return True iff ``content`` is a rescue payload produced by
    ``_build_reasoning_rescue_payload`` (R12-8 / H-01).

    The rescue payload has exactly one of two literal shapes:

    1. ``REASONING_CUTOFF_SENTINEL`` (bare) — sanitizer stripped the
       entire reasoning tail; only the sentinel remains.
    2. ``REASONING_CUTOFF_SENTINEL + "\\n\\n" + <sanitized tail>`` —
       normal case. The separator is a literal ``\\n\\n`` because
       ``_build_reasoning_rescue_payload`` joins the two parts with
       exactly that string.

    Tighter than ``content.startswith(SENTINEL)`` because a real model
    response starting with the sentinel literal but NOT followed by
    ``\\n\\n`` (e.g. ``"[truncated ...] also see notes."`` typed by a
    model echoing the sentinel as part of a longer reply) is correctly
    classified as real downstream output. Codex pr_validate r1 raised
    the false-positive risk; this shape gate addresses it without
    needing a structured per-request flag plumbed through the wire.

    ``None`` and empty strings are NOT rescue payloads (the rescue
    helper always returns at least the sentinel).
    """
    if not content:
        return False
    if content == REASONING_CUTOFF_SENTINEL:
        return True
    prefix = REASONING_CUTOFF_SENTINEL + "\n\n"
    return content.startswith(prefix) and len(content) > len(prefix)
