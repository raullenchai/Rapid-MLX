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

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx


import gc
import inspect
from pathlib import Path

from vllm_mlx import registry
from vllm_mlx.disk_stream_patch import (
    DiskStreamInstallError,
    UnsupportedModelTypeError,
    install,
    is_installed,
    uninstall,
)

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


def _eager_baseline(hf_repo: str) -> tuple[list[int], float]:
    """Generate the comparison on this host, then release resident weights."""
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    mx.reset_peak_memory()
    model, tokenizer = load(hf_repo)
    mx.random.seed(SEED)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True
    )
    token_ids = [
        response.token
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=NUM_TOKENS,
            sampler=make_sampler(temp=0.0),
        )
    ]
    peak_gb = mx.get_peak_memory() / 1e9
    del model, tokenizer
    gc.collect()
    mx.clear_cache()
    return token_ids, peak_gb


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


class _FakeMoeBlock:
    """Lightweight stand-in MoE block for install/is_installed/uninstall
    bookkeeping tests below. Deliberately NOT a real mlx_lm class and NOT
    used to test streaming math -- this stays inside the PRD's "no mock-
    based unit test for the STREAMING logic itself" boundary because
    is_installed()/uninstall() are pure bookkeeping around the class-level
    monkeypatch, not the streaming forward.
    """

    def __call__(self, x):
        return "orig-call"


def _fake_streaming_forward(block, x, layer_idx, cache):  # pragma: no cover
    """Never invoked -- install() reads ``adapter.streaming_forward`` eagerly
    (to build the ``streaming_call`` closure) even though this test never
    calls the patched ``__call__``, so the adapter needs a resolvable
    streaming_forward_module/fn_name pair or ``install()`` itself raises.
    """
    raise AssertionError("not expected to be called by this bookkeeping test")


def test_install_sets_marker_and_uninstall_restores_original_call():
    """install() sets the ``is_installed()`` marker and patches
    ``__call__``; uninstall() restores both to pre-install state.

    Mirrors ``test_uninstall_restores_originals_across_module_reload`` in
    ``tests/test_deepseek_v32_indexer_gate.py`` (this repo's precedent for
    the same class-level-``__call__``-monkeypatch shape), scoped to a fake
    class registered just for this test so it doesn't touch the real
    lfm2_moe/qwen2_moe adapters other tests in this file rely on.
    """
    fake_adapter = registry.StreamingAdapter(
        model_type="_disk_stream_patch_test_fake",
        moe_block_module=__name__,
        moe_block_class_name="_FakeMoeBlock",
        tensor_template=registry.ExpertTensorTemplate(
            layout="stacked", name_template="unused.{layer}.{proj}.{component}"
        ),
        num_experts=1,
        streaming_forward_module=__name__,
        streaming_forward_fn_name="_fake_streaming_forward",
    )
    registry._register(fake_adapter)

    class _FakeModel:
        layers = [type("Layer", (), {"feed_forward": _FakeMoeBlock()})()]

    orig_call = _FakeMoeBlock.__call__
    assert not is_installed(_FakeMoeBlock)

    result = install(
        _FakeModel(),
        "_disk_stream_patch_test_fake",
        checkpoint_path="/nonexistent",
    )

    assert is_installed(_FakeMoeBlock)
    assert _FakeMoeBlock.__call__ is not orig_call
    assert result.moe_block_cls is _FakeMoeBlock
    assert result.orig_call is orig_call

    uninstall(result.moe_block_cls, result.orig_call)

    assert not is_installed(_FakeMoeBlock)
    assert _FakeMoeBlock.__call__ is orig_call


def test_install_rejects_zero_matching_layers_and_duplicate_install():
    fake_adapter = registry.StreamingAdapter(
        model_type="_disk_stream_patch_guard_test",
        moe_block_module=__name__,
        moe_block_class_name="_FakeMoeBlock",
        tensor_template=registry.ExpertTensorTemplate(
            layout="stacked", name_template="unused.{layer}.{proj}.{component}"
        ),
        num_experts=1,
        streaming_forward_module=__name__,
        streaming_forward_fn_name="_fake_streaming_forward",
    )
    registry._register(fake_adapter)

    empty_model = type("EmptyModel", (), {"layers": []})()
    with pytest.raises(DiskStreamInstallError, match="found no"):
        install(empty_model, fake_adapter.model_type, "/nonexistent")
    assert not is_installed(_FakeMoeBlock)

    layer = type("Layer", (), {"feed_forward": _FakeMoeBlock()})()
    model = type("FakeModel", (), {"layers": [layer]})()
    result = install(model, fake_adapter.model_type, "/nonexistent")
    try:
        with pytest.raises(DiskStreamInstallError, match="already installed"):
            install(model, fake_adapter.model_type, "/nonexistent")
    finally:
        uninstall(result.moe_block_cls, result.orig_call)


def test_streaming_forwards_use_each_projections_quantization_parameters():
    """Mixed-quantization checkpoints must not reuse gate settings."""
    from vllm_mlx.disk_stream_patch import _streaming_moe_forward
    from vllm_mlx.qwen2_moe_forward import qwen2_moe_streaming_forward
    from vllm_mlx.qwen3_next_forward import qwen3_next_streaming_forward

    for forward in (
        _streaming_moe_forward,
        qwen2_moe_streaming_forward,
        qwen3_next_streaming_forward,
    ):
        source = inspect.getsource(forward)
        for projection in ("gate_proj", "up_proj", "down_proj"):
            assert f"group_size={projection}.group_size" in source
            assert f"bits={projection}.bits" in source
            assert f"mode={projection}.mode" in source


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
    baseline_token_ids, baseline_peak_gb = _eager_baseline(HF_REPO)

    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    # mx.get_peak_memory() is a process-wide running maximum, not scoped
    # per-test -- reset it before the lazy load so the "near-zero peak"
    # assertion below is meaningful regardless of what ran earlier in this
    # pytest session (e.g. the qwen2_moe slow test below, if collection/
    # run order ever changes).
    mx.reset_peak_memory()
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

    try:
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
        assert token_ids == baseline_token_ids, (
            "streaming generation must be token-exact vs. the non-streaming "
            f"baseline. streaming={token_ids} baseline={baseline_token_ids}"
        )
        assert streaming_peak_gb < baseline_peak_gb, (
            f"streaming peak memory ({streaming_peak_gb:.4f} GB) must be "
            f"measurably lower than the full resident baseline "
            f"({baseline_peak_gb:.4f} GB)"
        )
        assert result.cache.misses > 0  # cache actually did streaming fetches
    finally:
        # ``install()`` patches Lfm2MoeSparseMoeBlock.__call__ at the CLASS
        # level; without tearing it back down, the streaming_call closure
        # keeps this test's ~1GB ExpertCache alive for the rest of the
        # pytest session (confirmed empirically: active memory stayed at
        # ~997MB after del+gc.collect() alone, until uninstall() dropped
        # the class's reference to it) -- which is exactly what made the
        # qwen2_moe test's own near-zero-peak-after-lazy-load assertion
        # below fail when both slow tests ran together, even with
        # mx.reset_peak_memory() at that test's top: reset zeroes the
        # *peak* counter, but get_peak_memory() immediately reports
        # whatever is still actively allocated, and this leaked cache was
        # part of that. gc.collect() + mx.clear_cache() then release the
        # now-unreferenced buffers so the next test starts from a clean
        # slate.
        uninstall(result.moe_block_cls, result.orig_call)
        del model, tokenizer, result
        gc.collect()
        mx.clear_cache()


# ---------------------------------------------------------------------------
# qwen2_moe (shared+routed, ticket 05) — same shape as the LFM2.5 test above.
# ---------------------------------------------------------------------------

QWEN_HF_REPO = "mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit"


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
    baseline_token_ids, baseline_peak_gb = _eager_baseline(QWEN_HF_REPO)

    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    # See the matching comment in the LFM2.5 test above: mx.get_peak_memory()
    # is a process-wide running maximum left over from whatever ran earlier
    # in this pytest session (notably the LFM2.5 slow test above, which by
    # its own success criterion drives the peak up to ~1GB) -- reset it here
    # so this test's own near-zero-peak-after-lazy-load assertion is
    # trustworthy when both slow tests run together in one session.
    mx.reset_peak_memory()
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

    try:
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
        assert token_ids == baseline_token_ids, (
            "streaming generation must be token-exact vs. the non-streaming "
            f"baseline. streaming={token_ids} baseline={baseline_token_ids}"
        )
        assert streaming_peak_gb < baseline_peak_gb, (
            f"streaming peak memory ({streaming_peak_gb:.4f} GB) must be "
            f"measurably lower than the full resident baseline "
            f"({baseline_peak_gb:.4f} GB)"
        )
        assert result.cache.misses > 0  # cache actually did streaming fetches
    finally:
        # Same leak-prevention teardown as the LFM2.5 test above -- keeps
        # this test's own ExpertCache from leaking into whatever slow test
        # runs after it if this file's suite grows a third one.
        uninstall(result.moe_block_cls, result.orig_call)
        del model, tokenizer, result
        gc.collect()
        mx.clear_cache()
