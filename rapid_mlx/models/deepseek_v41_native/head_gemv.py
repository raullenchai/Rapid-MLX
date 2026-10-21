"""Exact FP32-activation / BF16-weight vocabulary-head GEMV for V4.1.

The head keeps the reference's FP32 logits: ``head(h.astype(float32))`` with a
BF16 weight.  The stock path promotes the whole 129280 x 5120 weight to FP32 on
every call, so one decode step reads ~1.3 GB, writes ~2.6 GB and reads it back
before the GEMV even starts.  This kernel converts each weight element in
registers instead and reproduces the stock FP32 GEMV lane geometry for this
shape (32 lanes across K, four contiguous values per lane, one simdgroup per
output, shuffle-down tree reduction), so a single row is bit-exact with the
stock path.  Up to ``MAX_ROWS`` rows (batched lanes, speculative verify) run
the same per-row arithmetic, i.e. each row equals its own single-row result;
for the vocabulary head those rows also share one read of the weight.
Larger batches (prefill) stay on the stock path.

Exactness preconditions mirror the stock dispatch (mlx 0.32 ``gemv_axbpy``):
the weight is the row-major ``[N, K]`` array the loader produces (the stock
path then runs the non-transposed GEMV), ``K > 64`` and ``K < 16 * N`` (other
sizes select a different lane split), and ``K % 128 == 0`` (no tail block).
MLX exposes no stride query, so a transposed view assigned to ``head.weight``
is outside this contract; V4.1 only ever assigns checkpoint arrays.

The stock operation order is an implementation detail of the installed MLX
and GPU, so the first call for each head weight runs a one-time self-check:
the kernel evaluates ``MAX_ROWS`` seeded rows against that weight and each
row must equal its single-row stock projection; any bit difference (or a
compile/launch failure) keeps that head on the stock path with a warning. A
replaced weight is validated again.
"""

from __future__ import annotations

import logging
import weakref
from functools import lru_cache

import mlx.core as mx
from .kernels import custom_kernels_enabled
import mlx.nn as nn

logger = logging.getLogger(__name__)

# Module switch: False routes every call through the stock projection.
ENABLED = True
MAX_ROWS = 8
# Output count from which multi-row calls share one weight read (see
# ``_shares_weight_read``): the vocabulary head, not the 384-expert routers.
SHARED_READ_MIN_OUTPUTS = 16384
_LANES = 32
_VALUES_PER_LANE = 4
_BLOCK_K = _LANES * _VALUES_PER_LANE
_OUTPUTS_PER_THREADGROUP = 4

_ROW_SOURCE = r"""
    // Stock FP32 GEMV order for K % 128 == 0: lane-strided blocks of four,
    // fused multiply-add in element order, then a shuffle-down tree.
    constexpr int BLOCK_K = 128;
    const uint lane = thread_index_in_simdgroup;
    const int out_col = int(threadgroup_position_in_grid.x) * OUTPUTS
        + int(simdgroup_index_in_threadgroup);
    const int row = int(threadgroup_position_in_grid.y);
    if (out_col >= N) {
        return;
    }
    const device bfloat16_t* weight_row = weight + size_t(out_col) * K;
    const device float* input_row = input + size_t(row) * K;
    float result = 0.0f;
    int offset = int(lane) * 4;
    for (int block = 0; block < K / BLOCK_K; ++block) {
        float values[4];
        float weights[4];
        for (int index = 0; index < 4; ++index) {
            values[index] = input_row[offset + index];
        }
        for (int index = 0; index < 4; ++index) {
            weights[index] = float(weight_row[offset + index]);
        }
        for (int index = 0; index < 4; ++index) {
            result += weights[index] * values[index];
        }
        offset += BLOCK_K;
    }
    for (ushort delta = 16; delta >= 1; delta >>= 1) {
        result += simd_shuffle_down(result, delta);
    }
    if (lane == 0) {
        output[size_t(row) * N + out_col] = result;
    }
"""

_SHARED_SOURCE = r"""
    // Stock FP32 GEMV order for K % 128 == 0: lane-strided blocks of four,
    // fused multiply-add in element order, then a shuffle-down tree. Every
    // row of the call shares one weight read: each weight block is loaded
    // once and accumulated into each row's own sum in that same order, so a
    // row's arithmetic is identical to a single-row launch.
    constexpr int BLOCK_K = 128;
    const uint lane = thread_index_in_simdgroup;
    const int out_col = int(threadgroup_position_in_grid.x) * OUTPUTS
        + int(simdgroup_index_in_threadgroup);
    const int rows = int(input_shape[0]);
    if (out_col >= N) {
        return;
    }
    const device bfloat16_t* weight_row = weight + size_t(out_col) * K;
    float result[MAX_ROWS];
    for (int row = 0; row < MAX_ROWS; ++row) {
        result[row] = 0.0f;
    }
    int offset = int(lane) * 4;
    for (int block = 0; block < K / BLOCK_K; ++block) {
        float weights[4];
        for (int index = 0; index < 4; ++index) {
            weights[index] = float(weight_row[offset + index]);
        }
        for (int row = 0; row < MAX_ROWS; ++row) {
            if (row < rows) {
                const device float* input_row = input + size_t(row) * K + offset;
                float values[4];
                for (int index = 0; index < 4; ++index) {
                    values[index] = input_row[index];
                }
                for (int index = 0; index < 4; ++index) {
                    result[row] += weights[index] * values[index];
                }
            }
        }
        offset += BLOCK_K;
    }
    for (int row = 0; row < MAX_ROWS; ++row) {
        if (row < rows) {
            float sum = result[row];
            for (ushort delta = 16; delta >= 1; delta >>= 1) {
                sum += simd_shuffle_down(sum, delta);
            }
            if (lane == 0) {
                output[size_t(row) * N + out_col] = sum;
            }
        }
    }
"""


@lru_cache(maxsize=2)
def _kernel(shared: bool = False):
    return mx.fast.metal_kernel(
        name="rapid_v41_fp32_bf16_head_gemv" + ("_shared" if shared else ""),
        input_names=["input", "weight"],
        output_names=["output"],
        source=_SHARED_SOURCE if shared else _ROW_SOURCE,
        ensure_row_contiguous=True,
    )


def _shares_weight_read(weight: mx.array) -> bool:
    """Whether multi-row calls on ``weight`` read it once for all rows.

    The vocabulary head (1.3 GB) is bandwidth bound, so re-reading it per row
    costs a full weight pass each; a router weight (a few MB) stays cached and
    is latency bound, where one threadgroup per row finishes sooner.
    """
    return int(weight.shape[0]) >= SHARED_READ_MIN_OUTPUTS


# Per-head attribute holding (weakref to the validated weight, kernel matches stock).
_CHECK_ATTR = "_rapid_exact_head_check"


def _gpu_default() -> bool:
    return bool(mx.default_device() == mx.gpu)


def supports(head, x: mx.array) -> bool:
    """True when ``head(x)`` can take the exact GEMV path."""
    return (
        type(head) is nn.Linear
        and "bias" not in head
        and _supports_weight(getattr(head, "weight", None), x)
    )


def _supports_weight(weight, x: mx.array) -> bool:
    """True when ``x @ weight.T`` can take the exact GEMV path."""
    if not ENABLED or not custom_kernels_enabled() or not _gpu_default():
        return False
    if (
        weight is None
        or weight.ndim != 2
        or weight.dtype != mx.bfloat16
        or x.dtype != mx.float32
        or x.ndim < 1
        or x.shape[-1] != weight.shape[1]
        or weight.shape[1] % _BLOCK_K
        or weight.shape[1] >= 16 * weight.shape[0]
    ):
        return False
    rows = x.size // x.shape[-1] if x.shape[-1] else 0
    return 1 <= rows <= MAX_ROWS


def _launch(weight: mx.array, x: mx.array, shared: bool | None = None) -> mx.array:
    outputs, inputs = int(weight.shape[0]), int(weight.shape[1])
    rows = x.size // inputs
    if shared is None:
        shared = rows > 1 and _shares_weight_read(weight)
    template = [("K", inputs), ("N", outputs), ("OUTPUTS", _OUTPUTS_PER_THREADGROUP)]
    if shared:
        template.append(("MAX_ROWS", MAX_ROWS))
    (logits,) = _kernel(shared)(
        inputs=[x.reshape(rows, inputs), weight],
        template=template,
        # The shared kernel serves every row from one threadgroup column, so
        # the weight is read once per call instead of once per row.
        grid=(
            -(-outputs // _OUTPUTS_PER_THREADGROUP) * _LANES * _OUTPUTS_PER_THREADGROUP,
            1 if shared else rows,
            1,
        ),
        threadgroup=(_LANES * _OUTPUTS_PER_THREADGROUP, 1, 1),
        output_shapes=[(rows, outputs)],
        output_dtypes=[mx.float32],
    )
    return logits.reshape(*x.shape[:-1], outputs)


def _matches_stock(owner) -> bool:
    weight = owner.weight
    cached = getattr(owner, _CHECK_ATTR, None)
    if cached is None or cached[0]() is not weight:
        try:
            # Cover the full row grid: each row must equal its stock single-row
            # projection, which is the contract for every supported row count.
            probe = mx.random.normal(
                (MAX_ROWS, int(weight.shape[1])), key=mx.random.key(0x41)
            )
            # Every kernel variant a later call on this weight can select.
            variants = [False, True] if _shares_weight_read(weight) else [False]
            actual = [_launch(weight, probe, shared) for shared in variants]
            ok = True
            for row in range(MAX_ROWS):
                # One stock row at a time: each call promotes the whole weight
                # to FP32, so a single graph would hold MAX_ROWS copies.
                expected = probe[row : row + 1] @ weight.T
                for result in actual:
                    ok = ok and bool(
                        mx.array_equal(result[row : row + 1], expected).item()
                    )
        except Exception:  # compile/launch failure on this runtime: stay stock
            logger.warning("exact V4.1 GEMV unavailable", exc_info=True)
            ok = False
        if not ok:
            logger.warning("exact V4.1 GEMV disabled: stock path for this weight")
        # Plain attribute (not a parameter); the weak reference re-validates a
        # replaced weight without keeping the old 1.3 GB array alive.
        cached = (weakref.ref(weight), ok)
        object.__setattr__(owner, _CHECK_ATTR, cached)
    return bool(cached[1])


def head_logits(head, x: mx.array) -> mx.array:
    """FP32 logits for ``head(x)``; exact GEMV when supported, stock otherwise."""
    if not supports(head, x) or not _matches_stock(head):
        return head(x)
    # No runtime fallback past this point: the self-check above already
    # evaluated every kernel specialization this weight can select (the
    # template is K, N, OUTPUTS and the fixed MAX_ROWS, the shared kernel reads
    # its row count at run time and the support checks fix the dtypes), so the
    # Metal pipeline that runs lazily here has compiled and launched once.
    # ENABLED is the switch back to stock.
    return _launch(head.weight, x)


def project(owner, x: mx.array) -> mx.array:
    """FP32 ``x @ owner.weight.T`` for a BF16 ``owner.weight`` (router gates).

    Same contract as :func:`head_logits` for weights that are not held by a
    bias-free ``nn.Linear``; the stock expression promotes the weight first.
    """
    weight = owner.weight

    def stock() -> mx.array:
        out: mx.array = x @ weight.astype(mx.float32).T
        return out

    if not _supports_weight(weight, x) or not _matches_stock(owner):
        return stock()
    return _launch(weight, x)
