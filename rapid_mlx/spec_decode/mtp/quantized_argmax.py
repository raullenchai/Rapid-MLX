# SPDX-License-Identifier: Apache-2.0
"""Quantized lm-head argmax adapted from mlx-vlm's Qwen3.5 backend.

The Metal kernel computes per-tile maxima directly from affine-packed weights,
so greedy MTP drafting does not materialize a 248k-token logits row.
"""

import functools

import mlx.core as mx
import mlx.nn as nn


def _target_verify_qlinear_header(bits: int, group_size: int) -> str:
    return r"""
    using namespace metal;

    constant constexpr int SIMD_SIZE = 32;
    constant constexpr int BITS = __BITS__;
    constant constexpr int GS = __GS__;
    constant constexpr int PACK_FACTOR = (BITS == 5 ? 8 : 32 / BITS);
    constant constexpr int BYTES_PER_PACK = (BITS == 5 ? 5 : 32 / 8);
    constant constexpr int PACKS_PER_THREAD = 2;
    constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
    constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
    constant constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;
    constant constexpr int RESULTS_PER_SIMDGROUP = 4;
    constant constexpr int NUM_SIMDGROUPS = 2;
    constant constexpr int BN = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;

    template <typename T>
    inline float load_vector_exact(const device T* x, thread float* x_thread) {
      float sum = 0.0f;
      if (BITS == 4) {
        for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
          sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
          x_thread[i] = x[i];
          x_thread[i + 1] = x[i + 1] / 16.0f;
          x_thread[i + 2] = x[i + 2] / 256.0f;
          x_thread[i + 3] = x[i + 3] / 4096.0f;
        }
      } else if (BITS == 5) {
        for (int i = 0; i < VALUES_PER_THREAD; i += 8) {
          sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
              x[i + 6] + x[i + 7];
          x_thread[i] = x[i];
          x_thread[i + 1] = x[i + 1] / 32.0f;
          x_thread[i + 2] = x[i + 2] / 4.0f;
          x_thread[i + 3] = x[i + 3] / 128.0f;
          x_thread[i + 4] = x[i + 4] / 16.0f;
          x_thread[i + 5] = x[i + 5] / 2.0f;
          x_thread[i + 6] = x[i + 6] / 64.0f;
          x_thread[i + 7] = x[i + 7] / 8.0f;
        }
      }
      return sum;
    }

    inline float qdot_exact(
        const device uint8_t* w,
        const thread float* x_thread,
        float scale,
        float bias,
        float sum) {
      float accum = 0.0f;
      if (BITS == 4) {
        const device uint16_t* ws = (const device uint16_t*)w;
        for (int i = 0; i < (VALUES_PER_THREAD / 4); i++) {
          accum +=
              (x_thread[4 * i] * (ws[i] & 0x000f) +
               x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
               x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
               x_thread[4 * i + 3] * (ws[i] & 0xf000));
        }
      } else if (BITS == 5) {
        for (int i = 0; i < (VALUES_PER_THREAD / 8); i++) {
          const thread float* xt = x_thread + 8 * i;
          const device uint8_t* wb = w + 5 * i;

          accum += (wb[0] & 0x1f) * xt[0];
          accum += (wb[0] & 0xe0) * xt[1];
          accum += (wb[1] & 0x3) * (xt[1] * 256.0f);
          accum += (wb[1] & 0x7c) * xt[2];
          accum += (wb[1] & 0x80) * xt[3];
          accum += (wb[2] & 0xf) * (xt[3] * 256.0f);
          accum += (wb[2] & 0xf0) * xt[4];
          accum += (wb[3] & 0x1) * (xt[4] * 256.0f);
          accum += (wb[3] & 0x3e) * xt[5];
          accum += (wb[3] & 0xc0) * xt[6];
          accum += (wb[4] & 0x7) * (xt[6] * 256.0f);
          accum += (wb[4] & 0xf8) * xt[7];
        }
      }
      return scale * accum + sum * bias;
    }
""".replace("__BITS__", str(bits)).replace("__GS__", str(group_size))


_TARGET_VERIFY_QARGMAX_SOURCE = r"""
    uint n_tile = threadgroup_position_in_grid.y;
    uint b_idx = threadgroup_position_in_grid.z;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
    int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    int in_vec_size_g = K_SIZE / GS;

    threadgroup float tile_best_values[VERIFY_T][NUM_SIMDGROUPS];
    threadgroup int tile_best_indices[VERIFY_T][NUM_SIMDGROUPS];

    const device uint8_t* ws_base =
        (const device uint8_t*)w + out_row * in_vec_size_w +
        int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* scales_base =
        scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* biases_base =
        biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* x_base =
        x + int(b_idx) * VERIFY_T * K_SIZE + int(simd_lid) * VALUES_PER_THREAD;

    float result[VERIFY_T][RESULTS_PER_SIMDGROUP];
    float x_thread[VERIFY_T][VALUES_PER_THREAD];
    for (int t = 0; t < VERIFY_T; ++t) {
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        result[t][row] = 0.0f;
      }
    }

    const device uint8_t* ws = ws_base;
    const device T* sc = scales_base;
    const device T* bs = biases_base;
    const device T* xk = x_base;

    for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
      float sums[VERIFY_T];
      for (int t = 0; t < VERIFY_T; ++t) {
        sums[t] = load_vector_exact<T>(xk + t * K_SIZE, x_thread[t]);
      }

      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        const device T* sl = sc + row * in_vec_size_g;
        const device T* bl = bs + row * in_vec_size_g;
        float s = float(sl[0]);
        float b = float(bl[0]);
        for (int t = 0; t < VERIFY_T; ++t) {
          result[t][row] += qdot_exact(wl, x_thread[t], s, b, sums[t]);
        }
      }

      ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      sc += BLOCK_SIZE / GS;
      bs += BLOCK_SIZE / GS;
      xk += BLOCK_SIZE;
    }

    for (int t = 0; t < VERIFY_T; ++t) {
      float best_value = -3.4028234663852886e38f;
      int best_index = 0;
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        int n = out_row + row;
        if (n < N_SIZE) {
          float rounded = float(T(simd_sum(result[t][row])));
          if (rounded > best_value) {
            best_value = rounded;
            best_index = n;
          }
        }
      }

      if (simd_lid == 0) {
        tile_best_values[t][simd_gid] = best_value;
        tile_best_indices[t][simd_gid] = best_index;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_gid == 0 && simd_lid == 0) {
      for (int t = 0; t < VERIFY_T; ++t) {
        float best = tile_best_values[t][0];
        int best_idx = tile_best_indices[t][0];
        for (int i = 1; i < NUM_SIMDGROUPS; ++i) {
          float candidate = tile_best_values[t][i];
          int candidate_idx = tile_best_indices[t][i];
          if (candidate > best) {
            best = candidate;
            best_idx = candidate_idx;
          }
        }
        int offset = (int(b_idx) * VERIFY_T + t) * NUM_TILES + int(n_tile);
        tile_values[offset] = T(best);
        tile_indices[offset] = best_idx;
      }
    }
"""


@functools.cache
def _target_verify_qargmax_kernel(bits, group_size, dtype, verify_t, k_size, n_size):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=(
            "qwen3_5_target_verify_qargmax_"
            f"b{bits}_gs{group_size}_t{verify_t}_k{k_size}_n{n_size}_{dtype_name}"
        ),
        input_names=["x", "w", "scales", "biases"],
        output_names=["tile_values", "tile_indices"],
        header=_target_verify_qlinear_header(bits, group_size),
        source=_TARGET_VERIFY_QARGMAX_SOURCE,
    )


def _can_target_verify_quantized_head(linear) -> bool:
    if (
        not isinstance(linear, nn.QuantizedLinear)
        or linear.bits not in (4, 5)
        or linear.mode != "affine"
        or linear.biases is None
        or linear.scales.dtype not in (mx.bfloat16, mx.float16)
        or linear.biases.dtype != linear.scales.dtype
    ):
        return False

    K = linear.weight.shape[1] * 32 // linear.bits
    N = linear.weight.shape[0]
    return K % 512 == 0 and N % 8 == 0


def _can_target_verify_quantized(linear, x: mx.array) -> bool:
    if (
        not _can_target_verify_quantized_head(linear)
        or x.ndim != 3
        or x.shape[1] < 1
        or x.dtype != linear.scales.dtype
    ):
        return False

    K = linear.weight.shape[1] * 32 // linear.bits
    return x.shape[-1] == K


def quantized_argmax(linear, x: mx.array) -> mx.array | None:
    if not _can_target_verify_quantized(linear, x) or "bias" in linear:
        return None

    B, T, K = x.shape
    if T == 1 and 1 < B <= 4:
        out = quantized_argmax(linear, x.transpose(1, 0, 2))
        if out is not None:
            return out.transpose(1, 0)

    N = linear.weight.shape[0]
    num_tiles = N // 8

    x = mx.contiguous(x)
    kernel = _target_verify_qargmax_kernel(
        linear.bits, linear.group_size, x.dtype, T, K, N
    )
    inputs = [x, linear.weight, linear.scales, linear.biases]
    tile_values, tile_indices = kernel(
        inputs=inputs,
        template=[
            ("T", x.dtype),
            ("VERIFY_T", int(T)),
            ("K_SIZE", int(K)),
            ("N_SIZE", int(N)),
            ("NUM_TILES", int(num_tiles)),
        ],
        grid=(32, 2 * num_tiles, B),
        threadgroup=(32, 2, 1),
        output_shapes=[(B, T, num_tiles), (B, T, num_tiles)],
        output_dtypes=[x.dtype, mx.int32],
    )
    best_tile = mx.argmax(tile_values, axis=-1)
    return mx.take_along_axis(tile_indices, best_tile[..., None], axis=-1).squeeze(-1)
