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

This file fills the gap with end-to-end probes:

* Load ``mlx-community/Qwen3.5-9B-4bit`` via ``mlx_lm.load``.
* Call :func:`inject_mtp_support` with the cached sidecar repo
  ``mlx-community/Qwen3.5-9B-MTP-4bit``.
* Verify the four contract surfaces land
  (:func:`validate_mtp_support`).
* Retain an independent pristine-model byte-regression oracle on three short,
  known-stable prompt prefixes without claiming that parity is universal.
* Exercise the same Rapid generator parked at K=0 and at fixed greedy MTP
  depths K=1/2/3. Byte equality across those shapes is not the contract: a
  batched verify forward can flip a near-tied argmax under quantized weights.

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

from bench.bench_spec_decode_mtp import _BENCH_PROMPTS, _tokenizer_stop_tokens

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
_CONSISTENCY_N_TOKENS = 128


@pytest.fixture(scope="module")
def baseline_tokens():
    """Capture a small known-stable pristine-model regression baseline.

    These prompts are not evidence of universal byte parity: wider prompts can
    hit quantized near ties where single-token and batched forwards differ.
    They remain useful independent sentinels for cache/rollback drift because
    this checkpoint is known to preserve exact output on these short cases.
    """
    import gc

    from mlx_lm import load
    from mlx_lm.generate import stream_generate

    model, tokenizer = load(_BASE_MODEL)
    baselines: dict[str, list[int]] = {}
    for prompt in _BASELINE_PROMPTS:
        tokens = [
            int(response.token)
            for response in stream_generate(
                model,
                tokenizer,
                prompt,
                max_tokens=_BASELINE_N_TOKENS,
            )
        ]
        baselines[prompt] = tokens[:_BASELINE_N_TOKENS]

    del model
    del tokenizer
    gc.collect()
    return baselines


@pytest.fixture(scope="module")
def loaded_model(baseline_tokens):
    """Load the base + inject MTP after the pristine baseline is released."""
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


@pytest.mark.real_hf_cache
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

    # ------------------------------------------------------------------
    # Coverage-check: codex flagged on PR #954 that only checking
    # ``fc.weight`` would let the test pass even if every other MTP
    # tensor stayed random-init. Walk the MTP module's full parameter
    # tree and assert every expected key is present in the sidecar AND
    # byte-equal to the loaded module. This catches both partial-load
    # (some key map drift on one layer) and key-name drift (e.g. norm
    # vs norm_pre vs pre_norm).
    # ------------------------------------------------------------------
    from mlx.utils import tree_flatten

    flat_module = dict(tree_flatten(mtp.parameters()))
    expected_keys = set(flat_module.keys())
    # ``mtp.``-prefix tolerance — same rewrite the inject does.
    sidecar_norm = {
        (k.removeprefix("mtp.") if k.startswith("mtp.") else k): v
        for k, v in raw.items()
    }
    sidecar_norm_keys = set(sidecar_norm.keys())

    missing_in_sidecar = expected_keys - sidecar_norm_keys
    assert not missing_in_sidecar, (
        f"Sidecar {weights_file} is missing {len(missing_in_sidecar)} required "
        f"MTP parameter(s): {sorted(missing_in_sidecar)[:8]}. "
        f"inject_mtp_support should have already refused this load."
    )

    # Byte-equal every shared key. uint32 packed quant weights compare
    # exactly; bf16/fp16 norms compare exactly; bias tensors compare
    # exactly. Random init on ANY tensor would surface here.
    mismatched: list[tuple[str, int]] = []
    for k in sorted(expected_keys):
        on_disk = sidecar_norm[k]
        in_module = flat_module[k]
        assert on_disk.shape == in_module.shape, (
            f"{k}: shape mismatch (disk {on_disk.shape} vs module {in_module.shape})"
        )
        diff = _mx.sum(on_disk != in_module).item()
        if diff != 0:
            mismatched.append((k, int(diff)))
    assert not mismatched, (
        f"{len(mismatched)} MTP tensor(s) differ between sidecar and module "
        f"(first 5): {mismatched[:5]}. Weights were NOT loaded from disk — "
        f"this is the defect-class PR #918 originally shipped. Sidecar key "
        f"presence: {len(sidecar_norm_keys)} total in file, {len(expected_keys)} "
        f"expected by module, {len(sidecar_norm_keys & expected_keys)} matched."
    )

    # Explicit representative-key smoke check — codex suggested
    # asserting coverage across ``fc.*``, ``layers.*``, ``norm`` (and
    # the linear-attention pre-norms inside layer 0). Re-state the
    # successful matches by category so a future regression that
    # silently shrinks the parameter tree (e.g. a refactor that
    # removes the pre-norms) shows up loud here, not just as a
    # mysteriously-smaller expected_keys count.
    categories = {
        "fc.*": [k for k in expected_keys if k.startswith("fc.")],
        "layers.0.*": [k for k in expected_keys if k.startswith("layers.0.")],
        "norm": [k for k in expected_keys if k == "norm.weight" or k == "norm"],
        "pre_norms (layers.0.*norm*)": [
            k for k in expected_keys if k.startswith("layers.0.") and "norm" in k
        ],
    }
    for label, keys in categories.items():
        assert keys, (
            f"Coverage gap: no parameters under category {label!r} in the "
            f"MTP module's parameter tree. Either the upstream layout drifted "
            f"or the inject failed to wire the sub-module. expected_keys "
            f"sample: {sorted(expected_keys)[:8]}"
        )


def test_mtp_lossless_byte_equal_against_baseline(loaded_model, baseline_tokens):
    """Catch real cache/rollback drift on known byte-stable prompt prefixes.

    This deliberately narrow regression signature complements, rather than
    defines, the general MTP contract. It retains an independently advanced
    pristine target-model oracle without claiming byte parity for prompts that
    encounter cross-shape quantized near ties.
    """
    import mlx.core as _mx

    from vllm_mlx.spec_decode.mtp import MTPAcceptCounter
    from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step

    model, tokenizer = loaded_model
    inner = model.language_model
    for prompt in _BASELINE_PROMPTS:
        counter = MTPAcceptCounter()
        prompt_ids = _mx.array(tokenizer.encode(prompt), _mx.uint32)
        tokens = [
            int(token)
            for token, _logprobs, _from_draft in mtp_generate_step(
                prompt_ids,
                inner,
                max_tokens=_BASELINE_N_TOKENS,
                temp=0.0,
                accept_counter=counter,
                disable_auto_k=True,
                max_k=1,
            )
        ][:_BASELINE_N_TOKENS]
        snapshot = counter.snapshot()
        assert snapshot.attempts > 0, (
            f"Known-stable regression did not exercise MTP for {prompt[:40]!r}"
        )
        assert snapshot.accepts < snapshot.attempts, (
            f"Known-stable regression did not exercise rejection/rollback for "
            f"{prompt[:40]!r}: {snapshot}"
        )
        assert tokens == baseline_tokens[prompt], (
            f"Known-stable MTP regression on {prompt[:40]!r}: "
            f"baseline={baseline_tokens[prompt]}, mtp={tokens}, "
            f"counter={snapshot}"
        )


def test_mtp_greedy_fixed_depth_real_weight_activity(loaded_model):
    """Every fixed greedy depth must engage and complete on real weights.

    K=0 proves that the parked control does not draft. K=1/2/3 prove that every
    supported fixed depth performs real draft attempts and target verification
    over the full eight-prompt, 128-token horizon.

    This is an integration/activity smoke, not a token-equality correctness
    gate. K=0 and K>0 use different target-forward shapes, and K=1/2/3 use
    different batched shapes from each other; quantized near ties can therefore
    produce legitimate byte differences. Deterministic tests cover accept,
    reject, and rollback invariants, while the sampled-distribution suite covers
    target-distribution preservation. The standalone fixed-K diagnostic reports
    token differences for investigation without converting them into an invalid
    pass/fail rule. See #3295.
    """
    import mlx.core as _mx

    from vllm_mlx.spec_decode.mtp import MTPAcceptCounter
    from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step

    model, tokenizer = loaded_model
    inner = model.language_model
    stop_tokens = _tokenizer_stop_tokens(tokenizer)

    def run(prompt: str, max_k: int):
        prompt_ids = _mx.array(tokenizer.encode(prompt), _mx.uint32)
        counter = MTPAcceptCounter()
        timing: dict[str, float] = {}
        tokens: list[int] = []
        for tok, _logprobs, _from_draft in mtp_generate_step(
            prompt_ids,
            inner,
            max_tokens=_CONSISTENCY_N_TOKENS,
            temp=0.0,
            accept_counter=counter,
            disable_auto_k=True,
            max_k=max_k,
            stop_tokens=stop_tokens,
            timing_stats=timing,
        ):
            tokens.append(int(tok))
            if len(tokens) >= _CONSISTENCY_N_TOKENS:
                break
        return tokens, counter.snapshot(), timing

    for prompt in _BENCH_PROMPTS:
        k0_tokens, k0_counter, k0_timing = run(prompt, 0)
        assert k0_counter.attempts == 0
        assert int(k0_timing.get("verify_calls", 0.0)) == 0
        assert len(k0_tokens) == _CONSISTENCY_N_TOKENS or (
            k0_tokens and k0_tokens[-1] in stop_tokens
        )

        for max_k in (1, 2, 3):
            mtp_tokens, mtp_counter, mtp_timing = run(prompt, max_k)
            assert mtp_counter.attempts > 0, (
                f"Fixed K={max_k} did not attempt speculation for {prompt[:40]!r}"
            )
            assert int(mtp_timing.get("verify_calls", 0.0)) > 0
            assert len(mtp_tokens) == _CONSISTENCY_N_TOKENS or (
                mtp_tokens and mtp_tokens[-1] in stop_tokens
            )


# Sampled-smoke settings. temp>0 with a top_p filter is what ordinary
# chat traffic uses; the temp=0 test above cannot reach the
# probabilistic accept/residual arithmetic at all, because at temp=0
# the verify accepts iff the argmaxes match and the random draw is
# ignored outright.
_SAMPLED_TEMP = 0.8
_SAMPLED_TOP_P = 0.95
_SAMPLED_N_TOKENS = 120
_SAMPLED_PROMPT = "Explain how a Bloom filter works, in three sentences."


def _longest_draft_run(from_draft_flags: list[bool]) -> int:
    """Longest run of consecutive draft-sourced tokens.

    A run of length N means some round had N draft positions accepted
    back-to-back, which is direct evidence that a chain of depth >= N
    was verified — the property the K>=2 accept arithmetic depends on.
    """
    best = current = 0
    for flag in from_draft_flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


@pytest.mark.real_hf_cache
@pytest.mark.parametrize("max_k", [2, 3])
def test_mtp_nongreedy_real_sampled_smoke(loaded_model, max_k):
    """Sampled (temp>0) MTP decode on real weights must reach depth K and accept.

    Why this exists alongside the temp=0 consistency test: at temp=0 the
    verify path never consults its random draw, so the entire
    probabilistic accept / residual-resample branch — the branch every
    non-greedy chat request now takes — is untested on real weights by
    that test. This one drives it.

    Why the depth is pinned rather than left to the EV controller:
    measured on this checkpoint, the controller settles on K=1 for this
    prompt, giving a longest-consecutive-draft-run of exactly 1. A
    K=1 chain has no cross-position accept arithmetic to get wrong, so
    an auto-K run would report a healthy acceptance rate while never
    executing the K>=2 code path at all. ``disable_auto_k=True`` with
    ``max_k=K`` fixes the chain depth at K every round so the path is
    actually exercised. Both settings are ordinary generator kwargs;
    nothing about the sampling math changes.

    Observed on main @ 0.14.0 (Qwen3.5-9B-4bit + sidecar, 120 tokens,
    M3 Ultra) across seeds 1234/7/99/2024: K=2 accepted 61-66 of
    106-116 draft positions (0.53-0.62) and K=3 accepted 65-70 of
    147-165 (0.39-0.48). The longest consecutive draft-sourced run came
    out exactly equal to max_k in all eight runs, and every run produced
    coherent prose. The thresholds below sit well under the measured
    accept rates so ordinary sampling variance does not flake the test.
    """
    import mlx.core as _mx

    from vllm_mlx.spec_decode.mtp import MTPAcceptCounter
    from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step

    model, tokenizer = loaded_model
    inner = model.language_model

    _mx.random.seed(1234)
    counter = MTPAcceptCounter()
    prompt_ids = _mx.array(tokenizer.encode(_SAMPLED_PROMPT), _mx.uint32)

    tokens: list[int] = []
    from_draft: list[bool] = []
    for tok, _lp, fd in mtp_generate_step(
        prompt_ids,
        inner,
        max_tokens=_SAMPLED_N_TOKENS,
        temp=_SAMPLED_TEMP,
        top_p=_SAMPLED_TOP_P,
        accept_counter=counter,
        disable_auto_k=True,
        max_k=max_k,
    ):
        tokens.append(int(tok))
        from_draft.append(bool(fd))
        if len(tokens) >= _SAMPLED_N_TOKENS:
            break

    snap = counter.snapshot()

    assert len(tokens) == _SAMPLED_N_TOKENS, (
        f"Sampled MTP run produced {len(tokens)} tokens, expected "
        f"{_SAMPLED_N_TOKENS}. The generator terminated early."
    )

    # The spec path must actually have run. A zero here means MTP
    # silently degraded to plain autoregressive decode and every other
    # assertion in this test would pass vacuously.
    assert snap.attempts > 0, (
        f"No draft positions were attempted at max_k={max_k}: {snap}. "
        f"MTP spec decode did not engage."
    )

    accept_rate = snap.accepts / snap.attempts
    assert accept_rate > 0.15, (
        f"Sampled accept rate {accept_rate:.3f} ({snap.accepts}/"
        f"{snap.attempts}) at max_k={max_k} is far below the ~0.5 "
        f"measured on this checkpoint. Either the draft head regressed "
        f"or the accept arithmetic is rejecting valid proposals."
    )

    # The accept rate needs a ceiling as well as a floor. A verifier
    # that accepted every proposal would satisfy every other assertion
    # in this test while never entering the reject-and-resample-from-
    # residual branch — which is half of what the sampled path does and
    # the more delicate half. Measured rejections on this checkpoint are
    # 41-55 at K=2 and 77-93 at K=3 across four seeds, so requiring 10
    # keeps four-fold margin over the smallest observed run.
    rejections = snap.attempts - snap.accepts
    assert rejections >= 10, (
        f"Only {rejections} of {snap.attempts} draft positions were "
        f"rejected at max_k={max_k} (accept rate {accept_rate:.3f}). "
        f"The residual-resample branch is essentially unexercised, so "
        f"this run does not cover it. A verifier that accepts "
        f"everything reaches this line."
    )

    # The point of the test: prove a chain of depth ``max_k`` was
    # verified and accepted, i.e. the multi-position accept path really
    # ran to the pinned depth. Asserting a fixed >= 2 instead would let
    # the max_k=3 case pass on a run that never accepted a third
    # position, making that parametrization prove nothing K=2 did not.
    longest = _longest_draft_run(from_draft)
    assert longest >= max_k, (
        f"Longest consecutive draft-sourced run was {longest} at "
        f"max_k={max_k}; expected >= {max_k}. The full depth-{max_k} "
        f"accept path was never exercised, so this run proves nothing "
        f"about it."
    )

    text = tokenizer.decode(tokens)
    assert text.strip(), (
        f"Sampled MTP decode at max_k={max_k} produced no printable text "
        f"from {len(tokens)} tokens."
    )
