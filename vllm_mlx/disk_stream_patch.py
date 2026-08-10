# SPDX-License-Identifier: Apache-2.0
"""Install-time glue for disk-streaming MoE weight loading.

Ticket: ``.scratch/rapid-mlx-disk-stream/issues/03-patch-glue-lfm25-e2e.md``.

Ties tickets 01 (``registry.py`` + ``offset_reader.py``) and 02
(``expert_cache.py``) together into a working generation run: given a
*lazily*-loaded model (``mlx_lm.load(..., lazy=True)``, so its MoE
parameters are never evaluated/materialized) and its checkpoint's
``model_type`` string, :func:`install` looks up the architecture's
:class:`~vllm_mlx.registry.StreamingAdapter`, wraps
:func:`~vllm_mlx.offset_reader.fetch_expert_bundle` in a byte-budgeted
:class:`~vllm_mlx.expert_cache.ExpertCache`, and installs a class-level
``__call__`` monkeypatch on the adapter's MoE block class so routed-expert
compute streams each selected expert's weights fresh off disk through the
cache instead of touching the model's resident (never-materialized)
``switch_mlp`` stack.

Deliberate deviation from the ticket's shorthand ``install(model,
model_type, cache_budget_gb)`` signature: a checkpoint path is structurally
required (:func:`~vllm_mlx.offset_reader.fetch_expert_bundle` reads
straight off a safetensors file) and nothing on a loaded ``mlx_lm`` model
object records where it was loaded from (verified empirically — a lazily
loaded ``mlx_lm.models.lfm2_moe.Model`` carries no ``model_path`` /
``name_or_path`` attribute). ``checkpoint_path`` is therefore an explicit,
required parameter; the CLI wiring (a later ticket) already has this path
in hand at ``load(path_or_hf_repo, lazy=True)`` call time and can pass it
straight through.

Streaming math is architecture-specific and lives with each adapter, not
here: ``streaming_call`` below dispatches to
``adapter.streaming_forward(self, x, layer_idx, cache)``, a
``(block, x, layer_idx, cache) -> mx.array`` function resolved lazily off
the :class:`~vllm_mlx.registry.StreamingAdapter` (see that class's
``streaming_forward`` property). ``_streaming_moe_forward`` below is
LFM2.5's: router (dense, resident) -> top-k -> per selected expert
``silu(gate_proj(x)) * up_proj(x) -> down_proj(...)`` via
``mx.quantized_matmul`` against that one expert's cached slice -> score-
weighted sum, no shared-expert term (LFM2.5 has none) — registered as
``lfm2_moe``'s ``streaming_forward`` in ``registry.py``. qwen2_moe's
shared+routed math (ticket 05) lives in ``vllm_mlx/qwen2_moe_forward.py``
instead, registered the same way — adding it required exactly two changes
here (this dispatch, and ``moe_block_attr`` below), both driven entirely
by adapter data with zero architecture-specific logic added; see that
ticket's report for why the ticket 03 interface couldn't stay untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from vllm_mlx import registry
from vllm_mlx.expert_cache import ExpertCache
from vllm_mlx.offset_reader import fetch_expert_bundle


class UnsupportedModelTypeError(ValueError):
    """Raised by :func:`install` when ``model_type`` has no registered
    :class:`~vllm_mlx.registry.StreamingAdapter`.

    No silent fallback to resident loading and no downstream crash mid-
    forward (PRD user story 6) — this fires immediately, at install time,
    naming the unsupported architecture.
    """


class DiskStreamInstallError(RuntimeError):
    """Raised when a registered adapter cannot be installed safely."""


@dataclass
class InstallResult:
    """What :func:`install` hands back: enough to report what happened and
    to inspect cache behavior (hit rate etc.) after a generation run.

    ``moe_block_cls``/``orig_call`` are the class :func:`install` patched
    and its pre-patch ``__call__`` — pass both to :func:`uninstall` to
    restore it (see that function's docstring). Additive fields; existing
    callers that only read the first four are unaffected.
    """

    model_type: str
    checkpoint_path: Path
    num_moe_layers_patched: int
    cache: ExpertCache
    moe_block_cls: type
    orig_call: Any


def install(
    model: Any,
    model_type: str,
    checkpoint_path: str | Path,
    cache_budget_gb: float = 1.0,
) -> InstallResult:
    """Install the disk-streaming monkeypatch on every MoE layer of
    ``model``.

    Raises :class:`UnsupportedModelTypeError` immediately (before touching
    ``model`` at all) if ``model_type`` isn't registered in
    :mod:`vllm_mlx.registry` — no silent fallback.
    """
    adapter = registry.get_adapter(model_type)
    if adapter is None:
        raise UnsupportedModelTypeError(
            f"disk-streaming MoE weight loading does not support "
            f"model_type={model_type!r}: no StreamingAdapter is registered "
            f"for it in vllm_mlx.registry. Register one (see "
            f"vllm_mlx/registry.py) before enabling disk-streaming for this "
            f"architecture; falling back to resident loading is not done "
            f"automatically."
        )

    checkpoint_path = Path(checkpoint_path)
    moe_block_cls = adapter.moe_block_cls

    if is_installed(moe_block_cls):
        raise DiskStreamInstallError(
            f"disk-streaming is already installed on {moe_block_cls.__name__}; "
            "a second class-level installation would mix model instances and "
            "expert caches"
        )

    cache = ExpertCache(
        fetch_fn=lambda layer_idx, expert_id: fetch_expert_bundle(
            adapter, checkpoint_path, layer_idx, expert_id
        ),
        budget_bytes=int(cache_budget_gb * 1e9),
    )

    # id(block) -> layer_idx, built once. Needed because the patched
    # __call__ is installed at the *class* level (an instance-attribute
    # override of __call__ does not intercept `block(x)` call syntax in
    # Python) and must recover which layer a given block instance is.
    layer_of_block: dict[int, int] = {}
    for i, layer in enumerate(model.layers):
        block = getattr(layer, adapter.moe_block_attr, None)
        if isinstance(block, moe_block_cls):
            layer_of_block[id(block)] = i

    if not layer_of_block:
        raise DiskStreamInstallError(
            f"disk-streaming found no {moe_block_cls.__name__} instances in "
            f"model_type={model_type!r}; refusing to report a successful install"
        )

    orig_call = moe_block_cls.__call__
    streaming_forward = adapter.streaming_forward

    def streaming_call(self, x):
        layer_idx = layer_of_block.get(id(self))
        if layer_idx is None:
            return orig_call(self, x)  # not one of the patched layers
        return streaming_forward(self, x, layer_idx, cache)

    moe_block_cls.__call__ = streaming_call
    # Upstream-class marker (mirrors deepseek_v32_indexer_gate.py's
    # ``_RAPID_MLX_INDEXER_GATE_INSTALLED`` convention) so an argument-free,
    # out-of-process check (e.g. a subprocess wiring test per CONTRIBUTING.md's
    # "Testing install-time patches" section) can confirm the patch actually
    # fired without needing this function's return value in hand.
    moe_block_cls._RAPID_MLX_DISK_STREAM_INSTALLED = True

    return InstallResult(
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        num_moe_layers_patched=len(layer_of_block),
        cache=cache,
        moe_block_cls=moe_block_cls,
        orig_call=orig_call,
    )


def is_installed(moe_block_cls: type) -> bool:
    """Return whether :func:`install` has patched ``moe_block_cls.__call__``.

    Reads the class-level marker :func:`install` sets, so this works from
    a fresh process/import with no reference to the ``InstallResult`` the
    original ``install()`` call returned.
    """
    return bool(getattr(moe_block_cls, "_RAPID_MLX_DISK_STREAM_INSTALLED", False))


def uninstall(moe_block_cls: type, orig_call: Any) -> None:
    """Undo :func:`install`'s class-level ``__call__`` monkeypatch.

    ``orig_call`` is the pre-install callable captured by
    :attr:`InstallResult.orig_call` (``install(...).orig_call``). Restores
    ``moe_block_cls.__call__`` to it and clears the marker so
    :func:`is_installed` reports ``False`` again. Test-only today (like
    ``deepseek_v32_indexer_gate.uninstall_deepseek_v32_indexer_gate``,
    the same restore shape for the same class-level-monkeypatch pattern);
    production code installs once per served model and never tears down.
    """
    moe_block_cls.__call__ = orig_call
    moe_block_cls._RAPID_MLX_DISK_STREAM_INSTALLED = False


def _streaming_moe_forward(block, x, layer_idx: int, cache: ExpertCache):
    """LFM2.5-shaped streaming replacement for
    ``Lfm2MoeSparseMoeBlock.__call__`` — router stays resident (small,
    dense), only the selected experts' gate/up/down weights come from
    ``cache`` (disk-streamed, LRU-cached) instead of the resident
    ``switch_mlp`` stack. Bit-exact reproduction of
    ``.scratch/moe-disk-stream/scripts/03_streaming_moe_forward.py``'s
    verified math, generalized to any layer index via ``cache``.
    """
    gates = block.gate(x).astype(mx.float32)
    gates = mx.softmax(gates, axis=-1)
    if block.use_expert_bias:
        gates = gates + block.expert_bias

    k = block.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if block.norm_topk_prob:
        scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
    scores = scores.astype(x.dtype)

    # Quantization params are small resident scalars on the switch layer
    # (not the big weight tensor) — read them off the real module instead
    # of hardcoding, so a differently-quantized checkpoint is handled
    # correctly rather than silently mismatched.
    gate_proj = block.switch_mlp.gate_proj
    up_proj = block.switch_mlp.up_proj
    down_proj = block.switch_mlp.down_proj

    B, L, _D = x.shape
    inds_list = inds.tolist()  # [B][L][k] expert ids selected per token

    rows_b = []
    for b in range(B):
        rows_l = []
        for pos in range(L):
            tok_x = x[b, pos][None, :]  # (1, D)
            acc = None
            for kk, expert_id in enumerate(inds_list[b][pos]):
                bundle = cache.get(layer_idx, expert_id)
                gp = bundle["gate_proj"]
                up = bundle["up_proj"]
                dp = bundle["down_proj"]

                gate_out = mx.quantized_matmul(
                    tok_x,
                    gp["weight"],
                    scales=gp["scales"],
                    biases=gp["biases"],
                    transpose=True,
                    group_size=gate_proj.group_size,
                    bits=gate_proj.bits,
                    mode=gate_proj.mode,
                )
                up_out = mx.quantized_matmul(
                    tok_x,
                    up["weight"],
                    scales=up["scales"],
                    biases=up["biases"],
                    transpose=True,
                    group_size=up_proj.group_size,
                    bits=up_proj.bits,
                    mode=up_proj.mode,
                )
                h = nn.silu(gate_out) * up_out
                down_out = mx.quantized_matmul(
                    h,
                    dp["weight"],
                    scales=dp["scales"],
                    biases=dp["biases"],
                    transpose=True,
                    group_size=down_proj.group_size,
                    bits=down_proj.bits,
                    mode=down_proj.mode,
                )
                term = down_out[0] * scores[b, pos, kk]
                acc = term if acc is None else acc + term
            rows_l.append(acc)
        rows_b.append(mx.stack(rows_l, axis=0))
    return mx.stack(rows_b, axis=0)


if __name__ == "__main__":
    # Smallest runnable self-check (ponytail rule): the registry-miss error
    # path, no model/checkpoint needed. Full streaming-math coverage is the
    # @pytest.mark.slow integration test in tests/test_disk_stream_patch.py.
    try:
        install(object(), "totally_unregistered_model_type", "/nonexistent")
    except UnsupportedModelTypeError as e:
        assert "totally_unregistered_model_type" in str(e)
        print("disk_stream_patch self-check OK:", e)
    else:
        raise AssertionError("expected UnsupportedModelTypeError")
