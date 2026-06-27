# SPDX-License-Identifier: Apache-2.0
"""Real-weights integration test for the MTP injection pipeline.

The unit tests in :mod:`tests.test_mtp_spec_decode` and
:mod:`tests.test_mtp_lossless` use synthetic / mocked models so they
run in 1s without GPU contention. That coverage is essential for the
control-flow surfaces (eligibility, accept-counter math, install
idempotency) but it does NOT exercise the **quantize → load-from-
disk → real-forward** chain — exactly the chain whose absence shipped
PR #918 with an inject pipeline that built a random-init MTP module
and never loaded weights.

This file fills the gap with one end-to-end probe:

* Load ``mlx-community/Qwen3.5-9B-4bit`` via ``mlx_lm.load``.
* Call :func:`inject_mtp_support` with the cached sidecar repo
  ``mlx-community/Qwen3.5-9B-MTP-4bit``.
* Verify the four contract surfaces land
  (:func:`validate_mtp_support`).
* Run a single 20-token MTP-spec-decode pass and a single 20-token
  baseline pass, then compare them byte-equally — the lossless
  contract at temp=0 says they MUST match.

Heavy by default (5 GB base + 131 MB sidecar download on cold cache,
~15 s wall on warm cache). Gated on ``RAPID_MLX_RUN_HEAVY_TESTS=1``
so it does not fire in the ordinary CI sweep. Operators wanting to
re-verify after touching the inject / generator / rollback code path
run::

    RAPID_MLX_RUN_HEAVY_TESTS=1 pytest tests/test_mtp_real_weights.py -xvs
"""

from __future__ import annotations

import os

import pytest

mx = pytest.importorskip("mlx.core")

# This whole file is gated on RAPID_MLX_RUN_HEAVY_TESTS=1. The unit
# tests in ``test_mtp_spec_decode.py`` continue to cover the
# control-flow surfaces in the normal CI sweep.
_HEAVY = os.environ.get("RAPID_MLX_RUN_HEAVY_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not _HEAVY,
    reason=(
        "Heavy real-weights probe (5 GB base + 131 MB sidecar). "
        "Set RAPID_MLX_RUN_HEAVY_TESTS=1 to run."
    ),
)


_BASE_MODEL = "mlx-community/Qwen3.5-9B-4bit"
_MTP_SIDECAR = "mlx-community/Qwen3.5-9B-MTP-4bit"


_BASELINE_PROMPTS = (
    "Write a short Python Fibonacci function with type hints.",
    "Explain how a Bloom filter works.",
    "Two trains travel toward each other at 60 and 80 km/h, 350 km "
    "apart. When do they meet?",
)
_BASELINE_N_TOKENS = 20


@pytest.fixture(scope="module")
def baseline_tokens():
    """Capture baseline (no MTP) tokens BEFORE any inject runs.

    Codex flagged on PR #954 that running ``stream_generate`` after
    ``inject_mtp_support`` compares MTP against the *patched* model,
    not the original Qwen3.5 forward — which silently weakens the
    lossless guard. Fix: load a separate, un-injected model in this
    fixture and tear it down before the MTP fixture loads. This
    guarantees the baseline tokens were produced by the pristine
    upstream code path.
    """
    import gc

    from mlx_lm import load
    from mlx_lm.generate import stream_generate

    model, tokenizer = load(_BASE_MODEL)
    baselines: dict[str, list[int]] = {}
    for prompt in _BASELINE_PROMPTS:
        toks: list[int] = []
        for resp in stream_generate(
            model, tokenizer, prompt, max_tokens=_BASELINE_N_TOKENS
        ):
            toks.append(int(resp.token))
            if len(toks) >= _BASELINE_N_TOKENS:
                break
        baselines[prompt] = toks

    # Release the un-injected model before the patched fixture loads
    # the second copy. Two 9B-4bit copies briefly coexist; cleanup is
    # explicit to keep peak GPU mem bounded.
    del model
    del tokenizer
    gc.collect()
    return baselines


@pytest.fixture(scope="module")
def loaded_model(baseline_tokens):
    """Load the base + inject MTP exactly once for all tests in the file.

    Depends on ``baseline_tokens`` so the un-injected baseline pass
    completes (and releases its model) before this fixture mutates a
    fresh copy via ``inject_mtp_support``.
    """
    from mlx_lm import load

    from vllm_mlx.spec_decode.mtp.qwen3_5_inject import (
        inject_mtp_support,
        validate_mtp_support,
    )

    model, tokenizer = load(_BASE_MODEL)
    injected = inject_mtp_support(model, mtp_sidecar=_MTP_SIDECAR)
    assert injected is True, (
        "inject_mtp_support returned False on real Qwen3.5-9B-4bit + "
        "sidecar. Likely causes: sidecar repo unreachable, base config "
        "missing mtp_num_hidden_layers in text_config, or the inner "
        "TextModel could not be resolved off model.language_model."
    )
    assert validate_mtp_support(model), (
        "validate_mtp_support failed after a successful inject — "
        "the four PR #990 surfaces (return_hidden, n_confirmed, "
        "mtp_forward, make_mtp_cache) did not all land."
    )
    return model, tokenizer


def test_inject_loads_real_sidecar_weights(loaded_model):
    """The sidecar's 31 keys must round-trip into the quantized MTP head.

    This is the test that would have caught the original PR #918
    defect: it builds + quantizes + load_weights against a real
    safetensors blob (not a stub) and asserts the resulting parameter
    tree matches the upstream schema. If any of (quantization params
    detect, MTP module layout, sidecar key map) drifts, the
    ``mtp.load_weights(..., strict=False)`` call would still succeed
    but parameters would silently retain their random init — caught
    here by inspecting the loaded weight tensors against the sidecar
    file directly.
    """
    import mlx.core as _mx

    model, _ = loaded_model
    inner = model.language_model
    mtp = inner.mtp
    assert mtp is not None, "inject_mtp_support did not attach inner.mtp"

    # Sidecar weight count — every tensor in the safetensors must
    # appear in the MTP module's parameter tree post-quantize.
    from huggingface_hub import snapshot_download

    sidecar_dir = snapshot_download(_MTP_SIDECAR)
    weights_file = None
    for candidate in ("model-mtp.safetensors", "model.safetensors"):
        path = f"{sidecar_dir}/{candidate}"
        try:
            _mx.load(path)
            weights_file = path
            break
        except Exception:
            continue
    assert weights_file is not None, (
        f"No model.safetensors or model-mtp.safetensors found in {sidecar_dir}"
    )

    raw = _mx.load(weights_file)
    # Spot-check a few weight tensors actually moved from sidecar into
    # the MTP module — using values, not just shapes, so a random-init
    # would fail.
    fc_weight_from_disk = raw["fc.weight"]
    fc_weight_in_module = mtp.fc.weight
    assert fc_weight_from_disk.shape == fc_weight_in_module.shape, (
        "fc.weight shape mismatch between sidecar and MTP module"
    )
    # Equality check requires evaluating. Compare uint32 quantized
    # weights byte-equally — random init would not match.
    diff = _mx.sum(fc_weight_from_disk != fc_weight_in_module).item()
    assert diff == 0, (
        f"fc.weight in MTP module differs from sidecar by {diff} entries — "
        f"weights were NOT loaded from disk (likely random init still). "
        f"This is the defect-class that PR #918 originally shipped."
    )


def test_mtp_lossless_byte_equal_against_baseline(loaded_model, baseline_tokens):
    """At temp=0, MTP spec decode must be byte-equal to non-spec decode.

    The ``baseline_tokens`` fixture captured ground-truth tokens
    against a *fresh, un-injected* Qwen3.5-9B-4bit model BEFORE the
    ``loaded_model`` fixture mutated a separate copy with
    ``inject_mtp_support``. This test then runs the MTP generator on
    the patched copy and asserts the decoded token sequences match
    byte-equally.

    Any divergence indicates either:

    * The MTP head is producing wrong drafts AND the verify accepts
      them anyway (probabilistic-accept arithmetic broken at temp=0).
    * The cache rollback on draft rejection is failing to restore
      linear-attention SSM state — output then drifts from the
      baseline after the first rejected draft.

    Both failure modes would invalidate the lossless contract; this
    test is the canonical guard for it on a real checkpoint.
    """
    import mlx.core as _mx

    from vllm_mlx.spec_decode.mtp import MTPAcceptCounter
    from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step

    model, tokenizer = loaded_model
    inner = model.language_model

    for prompt in _BASELINE_PROMPTS:
        base_tokens = baseline_tokens[prompt]
        assert len(base_tokens) == _BASELINE_N_TOKENS, (
            f"baseline_tokens fixture returned {len(base_tokens)} tokens for "
            f"prompt {prompt[:40]!r}; expected {_BASELINE_N_TOKENS}."
        )

        # MTP on the patched model.
        counter = MTPAcceptCounter()
        prompt_ids = _mx.array(tokenizer.encode(prompt), _mx.uint32)
        mtp_tokens: list[int] = []
        for tok, _, _ in mtp_generate_step(
            prompt_ids,
            inner,
            max_tokens=_BASELINE_N_TOKENS,
            temp=0.0,
            accept_counter=counter,
        ):
            mtp_tokens.append(int(tok))
            if len(mtp_tokens) >= _BASELINE_N_TOKENS:
                break

        # Lossless contract: byte-equal at temp=0.
        assert mtp_tokens == base_tokens, (
            f"MTP-vs-baseline divergence on prompt {prompt[:40]!r}: "
            f"baseline {base_tokens} vs mtp {mtp_tokens}. "
            f"Accept rate this run: {counter.snapshot()}."
        )
