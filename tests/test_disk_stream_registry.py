# SPDX-License-Identifier: Apache-2.0
"""``vllm_mlx.registry`` — architecture lookup for disk-streaming
MoE weight loading.

Ticket: ``.scratch/rapid-mlx-disk-stream/issues/01-registry-offset-reader-lfm25.md``.

No checkpoint download or model load required — ``get_adapter`` is a pure
lookup, and resolving ``StreamingAdapter.moe_block_cls`` only imports the
``mlx_lm`` package (already an installed floor dependency), not any
weights.
"""

from __future__ import annotations

from vllm_mlx.registry import get_adapter


def test_get_adapter_returns_lfm2_moe_adapter():
    """``lfm2_moe`` is LFM2.5's real ``model_type`` (confirmed against the
    downloaded checkpoint's ``config.json`` and ``mlx_lm/models/lfm2_moe.py``,
    not guessed).
    """
    adapter = get_adapter("lfm2_moe")
    assert adapter is not None
    assert adapter.model_type == "lfm2_moe"
    assert adapter.num_experts == 32
    assert adapter.tensor_template.layout == "stacked"

    from mlx_lm.models.lfm2_moe import Lfm2MoeSparseMoeBlock

    assert adapter.moe_block_cls is Lfm2MoeSparseMoeBlock


def test_lfm2_moe_tensor_template_matches_lfm25_safetensors_naming():
    """Pin the exact tensor-naming template against the real on-disk key
    format (``model.layers.<i>.feed_forward.switch_mlp.<proj>.<component>``
    — verified in the spike's ``02_offset_reader.py`` self-check).
    """
    adapter = get_adapter("lfm2_moe")
    name = adapter.tensor_template.tensor_name(
        layer_idx=2, proj="gate_proj", component="weight"
    )
    assert name == "model.layers.2.feed_forward.switch_mlp.gate_proj.weight"


def test_get_adapter_returns_qwen2_moe_adapter():
    """``qwen2_moe`` is Qwen1.5-MoE's real ``model_type`` (confirmed against
    the downloaded ``Qwen1.5-MoE-A2.7B-Chat-4bit`` checkpoint's
    ``config.json`` and against ``mlx_lm/models/qwen2_moe.py``). Ticket 05.
    """
    adapter = get_adapter("qwen2_moe")
    assert adapter is not None
    assert adapter.model_type == "qwen2_moe"
    assert adapter.num_experts == 60
    assert adapter.tensor_template.layout == "direct"
    assert adapter.moe_block_attr == "mlp"  # not "feed_forward" — LFM2.5's name

    from mlx_lm.models.qwen2_moe import Qwen2MoeSparseMoeBlock

    assert adapter.moe_block_cls is Qwen2MoeSparseMoeBlock


def test_qwen2_moe_tensor_template_matches_direct_named_expert_tensors():
    """Pin the exact tensor-naming template against the real on-disk key
    format (``model.layers.<i>.mlp.experts.<e>.<proj>.<component>`` —
    verified in the spike's ``09_shared_expert_offset_reader.py`` and in
    ``mlx_lm``'s ``Model.sanitize()``, which un-stacks these exact names).
    """
    adapter = get_adapter("qwen2_moe")
    name = adapter.tensor_template.tensor_name(
        layer_idx=2, proj="gate_proj", component="weight", expert_id=5
    )
    assert name == "model.layers.2.mlp.experts.5.gate_proj.weight"


def test_qwen2_moe_streaming_forward_resolves_and_lfm2_moe_unaffected():
    """qwen2_moe's ``streaming_forward`` resolves to the new
    ``vllm_mlx.qwen2_moe_forward`` module, and registering it did not
    change lfm2_moe's resolution (still ``disk_stream_patch._streaming_moe_forward``)
    — the two adapters' streaming math stays fully independent.
    """
    from vllm_mlx.disk_stream_patch import _streaming_moe_forward
    from vllm_mlx.qwen2_moe_forward import qwen2_moe_streaming_forward

    qwen_adapter = get_adapter("qwen2_moe")
    lfm2_adapter = get_adapter("lfm2_moe")
    assert qwen_adapter.streaming_forward is qwen2_moe_streaming_forward
    assert lfm2_adapter.streaming_forward is _streaming_moe_forward


def test_get_adapter_unknown_model_type_returns_none():
    """Unregistered ``model_type`` returns ``None`` (documented lookup-miss
    contract — see the module docstring for why ``None`` was chosen over
    raising: ``patch.py``, a later ticket, is responsible for turning a
    ``None`` into a loud, clear "unsupported architecture" error).
    """
    assert get_adapter("totally_unregistered_architecture") is None
    assert get_adapter("") is None
