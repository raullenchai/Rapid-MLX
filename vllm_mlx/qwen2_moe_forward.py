# SPDX-License-Identifier: Apache-2.0
"""Streaming forward for qwen2_moe-style shared+routed MoE blocks.

Ticket: ``.scratch/rapid-mlx-disk-stream/issues/05-qwen2-moe-shared-expert-arch.md``.

Selected by ``vllm_mlx.registry``'s ``qwen2_moe`` ``StreamingAdapter`` (its
lazily-resolved ``streaming_forward`` property) and dispatched by
``vllm_mlx.disk_stream_patch.install``'s ``streaming_call`` closure —
see that module's docstring for why the forward function had to become
adapter-selected instead of the single hardcoded ``_streaming_moe_forward``
ticket 03 shipped. A third architecture with its own streaming math drops
its own module here (or reuses this one, or ``disk_stream_patch``'s, if the
math matches) and one ``_register`` call in ``registry.py`` — no further
``disk_stream_patch.py`` changes needed.

Reproduces ``mlx_lm.models.qwen2_moe.Qwen2MoeSparseMoeBlock.__call__``
exactly — verified against the installed ``mlx_lm`` package's real source
and bit/token-exact per the spike's
``.scratch/moe-disk-stream/scripts/10_shared_expert_streaming_forward.py``
(single-layer) and
``.scratch/moe-disk-stream/scripts/11_shared_expert_e2e_generation.py``
(full 24-layer generation, 32/32 token-exact):

    gates = softmax(self.gate(x), axis=-1, precise=True)        # router:
                                                                    # small
                                                                    # dense
                                                                    # nn.Linear,
                                                                    # resident
    inds = stop_gradient(argpartition(-gates, kth=top_k-1)[..., :top_k])
    scores = take_along_axis(gates, inds, axis=-1)                # RAW gate
                                                                    # probs —
                                                                    # this
                                                                    # architecture
                                                                    # has no
                                                                    # norm_topk_prob
                                                                    # renormalization
                                                                    # (confirmed:
                                                                    # no such
                                                                    # field on
                                                                    # ModelArgs)
    y = switch_mlp(x, inds)                                        # per
                                                                    # (token, k):
                                                                    # silu(gate_proj(x))
                                                                    # * up_proj(x)
                                                                    # -> down_proj(...)
    y = (y * scores[..., None]).sum(axis=-2)

    shared_expert_output = self.shared_expert(x)                   # resident
                                                                    # dense MLP,
                                                                    # always
                                                                    # active
    shared_expert_output = sigmoid(self.shared_expert_gate(x)) * shared_expert_output
    return y + shared_expert_output

``switch_mlp(x, inds)`` (``SwitchGLU`` + ``mx.gather_qmm`` under the hood)
is reproduced here per selected expert via ``mx.quantized_matmul`` against
that one expert's ``(weight, scales, biases)`` — the same math, restricted
to a single gathered index — fetched from ``cache`` (disk-streamed,
LRU-cached by ``expert_cache.ExpertCache``) instead of touching the
resident ``(60, ...)`` ``switch_mlp`` stack. ``shared_expert`` /
``shared_expert_gate`` run via the block's own already-loaded resident
parameters — never streamed, per the PRD's round-3 decision (always
active every token, so streaming buys nothing).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from vllm_mlx.expert_cache import ExpertCache


def qwen2_moe_streaming_forward(
    block, x: mx.array, layer_idx: int, cache: ExpertCache
) -> mx.array:
    """Streaming replacement for ``Qwen2MoeSparseMoeBlock.__call__``.

    ``block`` is the real block instance (so ``block.gate``,
    ``block.shared_expert``, ``block.shared_expert_gate`` and
    ``block.switch_mlp``'s quantization params are all read off the
    already-loaded resident module — only the selected experts' gate/up/down
    weights come from ``cache``).
    """
    gates = block.gate(x)
    gates = mx.softmax(gates, axis=-1, precise=True)

    k = block.top_k
    inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
    scores = mx.take_along_axis(gates, inds, axis=-1)  # no renorm for this arch

    # Quantization params are small resident scalars on the switch layer
    # (not the big weight tensor) — read off the real module instead of
    # hardcoding, so a differently-quantized checkpoint is handled
    # correctly rather than silently mismatched.
    gate_proj = block.switch_mlp.gate_proj
    up_proj = block.switch_mlp.up_proj
    down_proj = block.switch_mlp.down_proj

    B, L, D = x.shape
    inds_list = inds.tolist()  # [B][L][k] expert ids selected per token

    rows_b = []
    for b in range(B):
        rows_l = []
        for pos in range(L):
            tok_x = x[b, pos][None, :]  # (1, D)
            acc = mx.zeros((D,), dtype=x.dtype)
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
                acc = acc + down_out[0] * scores[b, pos, kk]
            rows_l.append(acc)
        rows_b.append(mx.stack(rows_l, axis=0))
    y = mx.stack(rows_b, axis=0)

    shared_expert_output = block.shared_expert(x)
    shared_expert_output = (
        mx.sigmoid(block.shared_expert_gate(x)) * shared_expert_output
    )
    return y + shared_expert_output


if __name__ == "__main__":
    # Smallest runnable self-check (ponytail rule): confirms the module
    # imports cleanly and the function has the expected 4-arg signature
    # disk_stream_patch.install dispatches through. Full math coverage is
    # the @pytest.mark.slow bit-exact/token-exact tests in
    # tests/test_disk_stream_offset_reader.py and tests/test_disk_stream_patch.py.
    import inspect

    sig = inspect.signature(qwen2_moe_streaming_forward)
    assert list(sig.parameters) == ["block", "x", "layer_idx", "cache"]
    print("qwen2_moe_forward self-check OK:", sig)
