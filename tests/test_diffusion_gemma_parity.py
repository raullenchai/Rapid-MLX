# SPDX-License-Identifier: Apache-2.0
"""End-to-end parity test — the rapid backend must produce coherent
DiffusionGemma output that's at least as good as mlx-vlm's upstream
loop on the canonical eval prompts.

Marked ``slow`` because it loads the real ~1.2 GB checkpoint and
exercises the GPU. CI skips by default; run on M3 Ultra with:

    python3.12 -m pytest tests/test_diffusion_gemma_parity.py -m slow -v

The 8-prompt set mirrors ``research/diffusion-gemma/scripts/quality_eval.py``
so the bench data on disk stays comparable with what CI/SOP asserts.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow

MODEL_ID = "mlx-community/diffusiongemma-26B-A4B-it-4bit"

CANONICAL_PROMPTS: list[tuple[str, str, int, list[str]]] = [
    (
        "longform-coherence",
        "Explain what a diffusion language model is and how it differs from an "
        "autoregressive language model. Cover: token emission order, parallelism, "
        "training objective, and one concrete strength + one concrete weakness.",
        256,
        ["diffusion", "autoregressive"],
    ),
    (
        "code-python",
        "Write a Python function `quicksort(arr)` that sorts a list of integers in "
        "ascending order, in place. Include a 1-line docstring. Just the code, no "
        "surrounding explanation.",
        256,
        # ``pivot`` is the only word every quicksort variant emits (in-place
        # vs out-of-place differ on whether they ``return arr``; some
        # implementations emit ``mid`` instead of ``pivot`` only when
        # using merge sort, not quicksort).
        ["def quicksort", "pivot"],
    ),
    (
        "structured-json",
        "Return ONLY a JSON object (no markdown, no prose) describing a book "
        "with: title (string), author (string), year (int), tags (array of "
        "strings, 3 tags), in_stock (bool).",
        256,
        ['"title"', '"author"'],
    ),
]


@pytest.fixture(scope="module")
def loaded_model():
    """Load DiffusionGemma once for the whole module — the checkpoint
    is 1.2 GB and warmup costs ~5s; sharing across the 3 prompts is
    necessary for the suite to finish under 30s."""
    import mlx.core as mx
    from mlx_vlm.utils import load

    model, processor = load(MODEL_ID)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Warmup once so the first measured generation isn't cold-shaded.
    from vllm_mlx.runtime.diffusion_loop import rapid_stream_diffusion_generate
    warm_chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        add_generation_prompt=True, tokenize=False,
    )
    warm_ids = mx.array([tokenizer.encode(warm_chat)])
    for _ in rapid_stream_diffusion_generate(
        model, processor, tokenizer, warm_ids,
        max_tokens=8, fixed_steps=4, sc_every=1, temperature=0.0,
    ):
        pass

    return model, processor, tokenizer


@pytest.mark.parametrize(
    "prompt_id,prompt,max_tokens,expected_keywords",
    CANONICAL_PROMPTS,
    ids=[p[0] for p in CANONICAL_PROMPTS],
)
def test_rapid_backend_keyword_quality(
    loaded_model, prompt_id: str, prompt: str,
    max_tokens: int, expected_keywords: list[str],
) -> None:
    """Rapid backend at ``fixed_steps=8, sc_every=1, temperature=0`` must
    contain every expected keyword in the decoded output. This is the
    same gate the hand-graded quality eval uses (see
    ``research/diffusion-gemma/quality/eval-20260611-1001.md``)."""
    import mlx.core as mx

    from vllm_mlx.runtime.diffusion_loop import rapid_stream_diffusion_generate

    model, processor, tokenizer = loaded_model
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=False,
    )
    input_ids = mx.array([tokenizer.encode(chat)])

    pieces: list[str] = []
    terminated = False
    for r in rapid_stream_diffusion_generate(
        model, processor, tokenizer, input_ids,
        max_tokens=max_tokens, fixed_steps=8, sc_every=1, temperature=0.0,
    ):
        if r.text:
            pieces.append(r.text)
        if r.finish_reason is not None:
            terminated = True
    text = "".join(pieces).strip()

    assert terminated, "rapid loop must end with a finish_reason"
    assert text, f"{prompt_id}: rapid backend produced empty output"
    text_lc = text.lower()
    missing = [k for k in expected_keywords if k.lower() not in text_lc]
    assert not missing, (
        f"{prompt_id}: rapid output missing expected keywords {missing}. "
        f"Output head: {text[:200]!r}"
    )


def test_rapid_backend_throughput_floor(loaded_model) -> None:
    """Rapid backend on the canonical long-form prompt must do at
    least 80 tok/s on M3 Ultra. The hand-measured number is ~127 tok/s
    (see quality eval 2026-06-11); 80 is the regression floor — if
    we drop below this, somebody changed something that broke the
    fast path (e.g. removed precise=True softmax, lost dequantize
    optimization, made a fixed_steps default change)."""
    import time

    import mlx.core as mx

    from vllm_mlx.runtime.diffusion_loop import rapid_stream_diffusion_generate

    model, processor, tokenizer = loaded_model
    prompt = CANONICAL_PROMPTS[0][1]  # longform-coherence
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=False,
    )
    input_ids = mx.array([tokenizer.encode(chat)])

    t0 = time.perf_counter()
    last = None
    for r in rapid_stream_diffusion_generate(
        model, processor, tokenizer, input_ids,
        max_tokens=256, fixed_steps=8, sc_every=1, temperature=0.0,
    ):
        last = r
    e2e = time.perf_counter() - t0

    assert last is not None
    assert last.generation_tokens >= 200, (
        f"rapid backend generated only {last.generation_tokens} tokens "
        "for max_tokens=256 — early stop hides a bug if no EOS was hit"
    )
    tps = last.generation_tokens / e2e
    assert tps >= 80.0, (
        f"rapid backend throughput regressed: {tps:.1f} tok/s "
        f"(floor: 80 tok/s, hand-measured: ~127 tok/s on M3 Ultra)"
    )
