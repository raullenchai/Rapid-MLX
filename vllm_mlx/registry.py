# SPDX-License-Identifier: Apache-2.0
"""Architecture registry for disk-streaming MoE weight loading.

Ticket: ``.scratch/rapid-mlx-disk-stream/issues/01-registry-offset-reader-lfm25.md``.

Maps a checkpoint's ``model_type`` string (the same string ``config.json``
carries and ``mlx_lm`` dispatches on) to a :class:`StreamingAdapter`
describing two things a downstream monkeypatch installer (``patch.py``,
a later ticket) needs to stream that architecture's routed experts off
disk instead of holding them resident:

1. the MoE block class to patch (``moe_block_cls``), and
2. a tensor-naming template (:class:`ExpertTensorTemplate`) describing
   where one routed expert's 9-tensor bundle (gate/up/down projections x
   weight/scales/biases) lives in that architecture's safetensors layout.

LFM2.5 (``lfm2_moe``, ``"stacked"`` layout) and qwen2_moe
(``qwen2_moe``, ``"direct"`` layout, shared+routed) are registered.
Both ``model_type`` strings are confirmed against their respective
downloaded checkpoints' ``config.json`` and against ``mlx_lm/models/``.
Adding a third architecture is one more call to ``_register`` (plus,
if its streaming math differs, one small new module for its
streaming-forward function — see ``vllm_mlx/qwen2_moe_forward.py`` for
the pattern) — no change to :func:`get_adapter` or its callers.

Lookup-miss contract
---------------------
:func:`get_adapter` returns ``None`` for an unregistered ``model_type``
rather than raising. This is a deliberate, documented choice (the PRD
flags it as something a later ticket — the ``patch.py`` monkeypatch
installer — depends on being unambiguous):

* ``None`` matches the conventional Python lookup-miss signal (``dict.get``,
  ``re.match``, ...) rather than an exception type callers must know to
  catch.
* It keeps this module a pure lookup with zero opinion on error handling.
  The PRD's user story 6 requires ``--disk-stream`` to *fail loudly* for an
  unsupported architecture — but that's ``patch.py``'s job (it is the
  module that knows it's about to install a patch and can raise a clear,
  actionable error); ``registry.get_adapter`` merely reports "not found".
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

# The 3 routed-expert sub-projections and 3 per-projection tensors that
# make up one expert's 9-tensor bundle. Shared by every "stacked" layout
# adapter; a future "direct" (qwen2_moe-style, per-expert-named-tensor)
# adapter would reuse the same constants.
SUB_PROJECTIONS: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")
TENSOR_COMPONENTS: tuple[str, ...] = ("weight", "scales", "biases")


@dataclass(frozen=True)
class ExpertTensorTemplate:
    """Describes how to locate one routed expert's weight tensors in a
    checkpoint's safetensors layout.

    ``layout``:

    * ``"stacked"`` — every sub-projection's ``weight``/``scales``/``biases``
      is a *single* on-disk tensor holding all experts stacked on axis 0
      (LFM2.5's layout). One expert is a contiguous byte-range slice of
      that tensor, computed from ``data_offsets`` and ``expert_id`` — see
      ``offset_reader.fetch_expert_bundle``.
    * ``"direct"`` — each expert has its own separately-named tensor per
      component (qwen2_moe-style). Both layouts are implemented by
      ``offset_reader.fetch_expert_bundle``: ``"direct"`` resolves each
      expert's per-component tensor name directly (optionally via a
      sharded checkpoint's ``model.safetensors.index.json``) rather than
      slicing a stacked tensor — see qwen2_moe's registration below for a
      real ``"direct"`` adapter.

    ``name_template`` is a ``str.format`` template with ``{layer}``,
    ``{proj}`` (one of ``SUB_PROJECTIONS``) and ``{component}`` (one of
    ``TENSOR_COMPONENTS``) placeholders, e.g. for LFM2.5:
    ``"model.layers.{layer}.feed_forward.switch_mlp.{proj}.{component}"``.
    A ``"direct"`` template additionally uses an ``{expert}`` placeholder
    (e.g. qwen2_moe: ``"model.layers.{layer}.mlp.experts.{expert}.{proj}.{component}"``)
    — ``expert_id`` is optional on :meth:`tensor_name` (unused/ignored by
    ``"stacked"`` templates, which don't reference ``{expert}``) so this
    stays a backward-compatible, additive extension of ticket 01's method.
    """

    layout: str
    name_template: str

    def tensor_name(
        self,
        layer_idx: int,
        proj: str,
        component: str,
        expert_id: int | None = None,
    ) -> str:
        if self.layout == "direct" and expert_id is None:
            raise ValueError(
                f"{self.layout!r}-layout tensor template requires "
                f"expert_id (got None): each expert has its own "
                f"separately-named tensor, so omitting expert_id would "
                f"silently format a malformed tensor name"
            )
        return self.name_template.format(
            layer=layer_idx, proj=proj, component=component, expert=expert_id
        )


@dataclass(frozen=True)
class StreamingAdapter:
    """Everything a disk-streaming monkeypatch installer needs for one
    MoE architecture.

    ``moe_block_module`` / ``moe_block_class_name`` name the class to
    patch (``mlx_lm.models.lfm2_moe.Lfm2MoeSparseMoeBlock`` for LFM2.5)
    without importing it at module load time — resolved lazily by the
    ``moe_block_cls`` property so importing ``vllm_mlx.registry``
    never requires ``mlx``/``mlx_lm`` to be installed (matches the rest of
    ``vllm_mlx``: mlx imports stay lazy so the package loads on platforms
    without Metal, e.g. Linux CI).

    ``moe_block_attr`` is the decoder layer's attribute name holding the
    MoE block instance (``"feed_forward"`` for LFM2.5, ``"mlp"`` for
    qwen2_moe) — architectures disagree on this, so it's per-adapter data
    rather than a hardcoded string in the installer.

    ``streaming_forward_module`` / ``streaming_forward_fn_name`` name the
    ``(block, x, layer_idx, cache) -> mx.array`` streaming-forward
    function this architecture needs, resolved lazily by the
    ``streaming_forward`` property for the same reason ``moe_block_cls``
    is lazy. Left unset (``None``) on adapters that predate this field —
    :func:`~vllm_mlx.disk_stream_patch.install` dispatches through it, so
    a real adapter used for disk-streaming must set both.
    """

    model_type: str
    moe_block_module: str
    moe_block_class_name: str
    tensor_template: ExpertTensorTemplate
    num_experts: int
    # ponytail: kept defaulted (not made required) — tests/test_disk_stream_
    # offset_reader.py constructs StreamingAdapter directly in 3 fixtures
    # without passing moe_block_attr; dropping the default would break them,
    # and that file is out of scope for this fix. Reviewer nit acknowledged.
    moe_block_attr: str = "feed_forward"
    streaming_forward_module: str | None = None
    streaming_forward_fn_name: str | None = None
    sub_projections: tuple[str, ...] = SUB_PROJECTIONS
    tensor_components: tuple[str, ...] = TENSOR_COMPONENTS

    @property
    def moe_block_cls(self):
        module = importlib.import_module(self.moe_block_module)
        return getattr(module, self.moe_block_class_name)

    @property
    def streaming_forward(self):
        if (
            self.streaming_forward_module is None
            or self.streaming_forward_fn_name is None
        ):
            raise NotImplementedError(
                f"{self.model_type!r} adapter has no streaming_forward "
                f"configured (streaming_forward_module/streaming_forward_fn_name "
                f"unset)"
            )
        module = importlib.import_module(self.streaming_forward_module)
        return getattr(module, self.streaming_forward_fn_name)


_REGISTRY: dict[str, StreamingAdapter] = {}


def _register(adapter: StreamingAdapter) -> None:
    _REGISTRY[adapter.model_type] = adapter


_register(
    StreamingAdapter(
        model_type="lfm2_moe",
        moe_block_module="mlx_lm.models.lfm2_moe",
        moe_block_class_name="Lfm2MoeSparseMoeBlock",
        tensor_template=ExpertTensorTemplate(
            layout="stacked",
            name_template="model.layers.{layer}.feed_forward.switch_mlp.{proj}.{component}",
        ),
        num_experts=32,
        moe_block_attr="feed_forward",
        streaming_forward_module="vllm_mlx.disk_stream_patch",
        streaming_forward_fn_name="_streaming_moe_forward",
    )
)

# qwen2_moe (e.g. mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit) — shared+routed
# MoE. Real model_type confirmed against the downloaded checkpoint's
# config.json and mlx_lm/models/qwen2_moe.py. Unlike LFM2.5, routed experts
# are NOT axis-0-stacked on disk: each is its own directly-named tensor
# (Model.sanitize() in mlx_lm's qwen2_moe.py is what stacks them into the
# runtime switch_mlp arrays at load time — on disk there is no stacking to
# undo), hence layout="direct" with an {expert} placeholder. The MoE block
# lives at the decoder layer's `.mlp` attribute (not `.feed_forward`, LFM2.5's
# name) and always sums a resident shared_expert/shared_expert_gate term —
# see vllm_mlx/qwen2_moe_forward.py for the streaming-forward math.
_register(
    StreamingAdapter(
        model_type="qwen2_moe",
        moe_block_module="mlx_lm.models.qwen2_moe",
        moe_block_class_name="Qwen2MoeSparseMoeBlock",
        tensor_template=ExpertTensorTemplate(
            layout="direct",
            name_template="model.layers.{layer}.mlp.experts.{expert}.{proj}.{component}",
        ),
        num_experts=60,
        moe_block_attr="mlp",
        streaming_forward_module="vllm_mlx.qwen2_moe_forward",
        streaming_forward_fn_name="qwen2_moe_streaming_forward",
    )
)


def get_adapter(model_type: str) -> StreamingAdapter | None:
    """Look up the :class:`StreamingAdapter` for ``model_type``.

    Returns ``None`` when ``model_type`` isn't registered — see the
    module docstring for why this returns rather than raises.
    """
    return _REGISTRY.get(model_type)
