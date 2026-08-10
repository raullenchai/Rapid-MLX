# SPDX-License-Identifier: Apache-2.0
"""``vllm_mlx.disk_stream_patch`` — install-time glue tying tickets 01/02
into an end-to-end disk-streaming generation run.

Tickets: ``.scratch/rapid-mlx-disk-stream/issues/03-patch-glue-lfm25-e2e.md``
(LFM2.5, ``"stacked"`` layout) and
``.scratch/rapid-mlx-disk-stream/issues/05-qwen2-moe-shared-expert-arch.md``
(qwen2_moe, ``"direct"`` layout, shared+routed).

Tiers, per the PRD's testing decision:

1. ``test_install_unregistered_model_type_raises_immediately`` — fast, no
   model/checkpoint, a lightweight dummy object stands in for ``model``.
   This is the one legitimate exception to "no mock-based unit test for
   the monkeypatch-installation logic": it tests registry-miss handling
   (a pure lookup + raise, before ``model`` is ever touched), not whether
   streaming actually works.
2. ``test_install_streams_lfm25_token_exact_with_lower_peak_memory`` —
   ``@pytest.mark.slow`` (needs ``--run-slow``), the LFM2.5 real-checkpoint
   integration test. No mock model here: it loads the real, already
   locally-cached LFM2.5-8B-A1B-MLX-4bit checkpoint lazily and exercises
   the actual streaming math end-to-end, because a mock model would
   validate the wiring without validating the memory-win guarantee this
   feature exists to provide.
3. ``test_install_streams_qwen2_moe_token_exact_with_lower_peak_memory`` —
   ticket 05's second ``@pytest.mark.slow`` integration test, same shape as
   #2 but for qwen2_moe (shared+routed, sharded checkpoint, ``checkpoint_path``
   passed as the snapshot *directory* — the shape the real CLI flow hands
   ``install`` via ``_resolve_model_path``, exercising
   ``offset_reader._resolve_shard_path``'s index.json resolution too).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_mlx.disk_stream_patch import UnsupportedModelTypeError, install

# Same fixed prompt/seed/greedy/token-count as the spike's ticket 01
# baseline and ticket 04 e2e script (.scratch/moe-disk-stream/scripts/
# 01_baseline_harness.py, 04_e2e_streaming_generation.py) — required for
# the token-exact comparison below to be meaningful. Ticket 05's qwen2_moe
# baseline (``baseline_output_qwen_moe.json``, see below) used this exact
# same prompt/seed/token-count too, so these constants are shared rather
# than duplicated with a QWEN_ prefix.
HF_REPO = "mlx-community/LFM2.5-8B-A1B-MLX-4bit"
PROMPT = "Explain what a mixture-of-experts model is in one paragraph."
SEED = 1234
NUM_TOKENS = 32
BASELINE_PATH = (
    Path(__file__).resolve().parents[2]
    / ".scratch"
    / "moe-disk-stream"
    / "baseline_output.json"
)


def test_install_unregistered_model_type_raises_immediately():
    """No silent fallback to resident loading (PRD user story 6): an
    unregistered ``model_type`` must raise before ``model`` is touched at
    all. A bare ``object()`` stands in for the model — if the raise
    happened after the registry check, this dummy would blow up on the
    first real attribute access instead of cleanly asserting the message.
    """
    dummy_model = object()

    with pytest.raises(
        UnsupportedModelTypeError, match="totally_unregistered_model_type"
    ):
        install(
            dummy_model,
            "totally_unregistered_model_type",
            checkpoint_path="/nonexistent",
        )

    # The error type is a ValueError subclass (documented in the module) so
    # callers who only catch ValueError still see it.
    with pytest.raises(ValueError):
        install(
            dummy_model,
            "totally_unregistered_model_type",
            checkpoint_path="/nonexistent",
        )


def _local_lfm25_checkpoint() -> Path | None:
    snapshots = (
        Path.home()
        / ".cache/huggingface/hub/models--mlx-community--LFM2.5-8B-A1B-MLX-4bit"
        / "snapshots"
    )
    if not snapshots.is_dir():
        return None
    for snapshot_dir in snapshots.iterdir():
        candidate = snapshot_dir / "model.safetensors"
        if candidate.is_file():
            return candidate
    return None


@pytest.mark.slow
def test_install_streams_lfm25_token_exact_with_lower_peak_memory():
    """The full end-to-end guarantee: install() on a lazily-loaded LFM2.5
    produces token-exact output vs. a fully resident run of the same
    prompt/seed/sampler, with measurably lower peak memory, and
    ``mx.get_peak_memory()`` right after the lazy load (before install or
    any forward pass) is near-zero.
    """
    checkpoint_path = _local_lfm25_checkpoint()
    if checkpoint_path is None:
        pytest.skip(
            "LFM2.5-8B-A1B-MLX-4bit not found in local HF cache -- this "
            "test reuses the spike's already-downloaded checkpoint and "
            "never downloads one itself"
        )
    if not BASELINE_PATH.is_file():
        pytest.skip(f"spike baseline not found at {BASELINE_PATH}")

    baseline = json.loads(BASELINE_PATH.read_text())
    assert baseline["prompt"] == PROMPT
    assert baseline["seed"] == SEED
    assert baseline["num_tokens_requested"] == NUM_TOKENS
    # Baseline was generated via mlx_lm.sample_utils.make_sampler(temp=0.0)
    # (verified against .scratch/moe-disk-stream/scripts/01_baseline_harness.py)
    # -- greedy decoding, deterministic regardless of seed. Prompt/seed/
    # token-count above match this test's run, so the recorded token_ids
    # are a valid non-streaming comparison point without loading a second
    # full model instance in-process.

    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(HF_REPO, lazy=True)
    peak_after_load_bytes = mx.get_peak_memory()
    assert peak_after_load_bytes < 1e7, (
        f"lazy load should leave peak memory near-zero (no MoE weights "
        f"materialized yet), got {peak_after_load_bytes} bytes"
    )

    result = install(
        model, "lfm2_moe", checkpoint_path=checkpoint_path, cache_budget_gb=1.0
    )
    assert result.num_moe_layers_patched > 0

    mx.random.seed(SEED)
    sampler = make_sampler(temp=0.0)
    messages = [{"role": "user", "content": PROMPT}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    token_ids = []
    for response in stream_generate(
        model, tokenizer, prompt, max_tokens=NUM_TOKENS, sampler=sampler
    ):
        token_ids.append(response.token)

    streaming_peak_gb = mx.get_peak_memory() / 1e9
    baseline_peak_gb = baseline["peak_memory_gb_mx_get_peak_memory"]

    assert token_ids == baseline["token_ids"], (
        "streaming generation must be token-exact vs. the non-streaming "
        f"baseline. streaming={token_ids} baseline={baseline['token_ids']}"
    )
    assert streaming_peak_gb < baseline_peak_gb, (
        f"streaming peak memory ({streaming_peak_gb:.4f} GB) must be "
        f"measurably lower than the full resident baseline "
        f"({baseline_peak_gb:.4f} GB)"
    )
    assert result.cache.misses > 0  # cache actually did streaming fetches


# ---------------------------------------------------------------------------
# qwen2_moe (shared+routed, ticket 05) — same shape as the LFM2.5 test above.
# ---------------------------------------------------------------------------

QWEN_HF_REPO = "mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit"
QWEN_BASELINE_PATH = (
    Path(__file__).resolve().parents[2]
    / ".scratch"
    / "moe-disk-stream"
    / "baseline_output_qwen_moe.json"
)


def _local_qwen2_moe_checkpoint_dir() -> Path | None:
    snapshots = (
        Path.home()
        / ".cache/huggingface/hub/models--mlx-community--Qwen1.5-MoE-A2.7B-Chat-4bit"
        / "snapshots"
    )
    if not snapshots.is_dir():
        return None
    for snapshot_dir in snapshots.iterdir():
        if (snapshot_dir / "model.safetensors.index.json").is_file():
            return snapshot_dir
    return None


@pytest.mark.slow
def test_install_streams_qwen2_moe_token_exact_with_lower_peak_memory():
    """Ticket 05's second end-to-end guarantee: install() on a lazily-loaded
    qwen2_moe model (shared+routed, sharded checkpoint) produces token-exact
    output vs. a fully resident run of the same prompt/seed/sampler, with
    measurably lower peak memory. ``checkpoint_path`` is passed as the
    snapshot *directory* (matching the real ``--disk-stream`` CLI flow's
    ``_resolve_model_path`` -> ``install`` call shape, unlike LFM2.5's
    single-file test above) so this also exercises
    ``offset_reader._resolve_shard_path``'s ``model.safetensors.index.json``
    resolution against the real 2-shard checkpoint.
    """
    checkpoint_dir = _local_qwen2_moe_checkpoint_dir()
    if checkpoint_dir is None:
        pytest.skip(
            "Qwen1.5-MoE-A2.7B-Chat-4bit not found in local HF cache -- this "
            "test reuses the spike's already-downloaded checkpoint and "
            "never downloads one itself"
        )
    if not QWEN_BASELINE_PATH.is_file():
        pytest.skip(f"spike baseline not found at {QWEN_BASELINE_PATH}")

    baseline = json.loads(QWEN_BASELINE_PATH.read_text())
    assert baseline["prompt"] == PROMPT
    assert baseline["seed"] == SEED
    assert baseline["num_tokens_requested"] == NUM_TOKENS

    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(QWEN_HF_REPO, lazy=True)
    peak_after_load_bytes = mx.get_peak_memory()
    assert peak_after_load_bytes < 1e7, (
        f"lazy load should leave peak memory near-zero (no MoE weights "
        f"materialized yet), got {peak_after_load_bytes} bytes"
    )

    result = install(
        model, "qwen2_moe", checkpoint_path=checkpoint_dir, cache_budget_gb=1.0
    )
    assert result.num_moe_layers_patched > 0

    mx.random.seed(SEED)
    sampler = make_sampler(temp=0.0)
    messages = [{"role": "user", "content": PROMPT}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    token_ids = []
    for response in stream_generate(
        model, tokenizer, prompt, max_tokens=NUM_TOKENS, sampler=sampler
    ):
        token_ids.append(response.token)

    streaming_peak_gb = mx.get_peak_memory() / 1e9
    baseline_peak_gb = baseline["peak_memory_gb_mx_get_peak_memory"]

    assert token_ids == baseline["token_ids"], (
        "streaming generation must be token-exact vs. the non-streaming "
        f"baseline. streaming={token_ids} baseline={baseline['token_ids']}"
    )
    assert streaming_peak_gb < baseline_peak_gb, (
        f"streaming peak memory ({streaming_peak_gb:.4f} GB) must be "
        f"measurably lower than the full resident baseline "
        f"({baseline_peak_gb:.4f} GB)"
    )
    assert result.cache.misses > 0  # cache actually did streaming fetches
