# Portions Copyright © 2025-2026 Apple Inc., MIT License:
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Copyright © 2026 KRAFTON Inc.
"""Raon-1 nano (raon1_nano) for mlx-lm, in one file. See NOTICE for the sources it includes."""
"""Metal kernels for Raon single-token bfloat16 inference."""

import mlx.core as mx


def cached_array(module, name, make):
    """Per-module derived constant, computed once and kept out of the parameter tree."""
    if module.training:
        return make()
    value = module.__dict__.get(name)
    if value is None:
        value = make()
        mx.eval(value)
        object.__setattr__(module, name, value)
    return value


_FUSED_TG = 512


_FUSED_BLK = 1024


_FUSED_SRC1 = r"""
    const uint row = threadgroup_position_in_grid.y;
    const uint blk = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint T = threads_per_threadgroup.x;
    const int H = H_DIM;
    const int S = 4;
    const int K = 17;
    const int D = S * H;
    const int B = BLK;
    threadgroup float red[32][K + 1];
    float acc[K + 1];
    for (int k = 0; k <= K; ++k) acc[k] = 0.0f;
    const device bfloat16_t* rin = rows + row * D;
    device bfloat16_t* rout = new_rows + row * D;
    const device bfloat16_t* br = branch + row * H;
    const int start = blk * B;
    const int stop = metal::min(start + B, D);
    for (int idx = start + tid; idx < stop; idx += T) {
        const int s = idx / H;
        const int h = idx - s * H;
        float v;
        if (DO_MIX) {
            float m = 0.0f;
            for (int j = 0; j < S; ++j) m += res_in[row * 16 + s * 4 + j] * (float)rin[j * H + h];
            m += post_in[row * 4 + s] * (float)br[h];
            bfloat16_t mb = (bfloat16_t)m;
            rout[idx] = mb;
            v = (float)mb;
        } else {
            v = (float)rin[idx];
            rout[idx] = rin[idx];
        }
        acc[K] += v * v;
        for (int k = 0; k < K; ++k) acc[k] += v * (float)phi[k * D + idx];
    }
    const uint lane = tid % 32;
    const uint sg = tid / 32;
    for (int k = 0; k <= K; ++k) {
        float t = simd_sum(acc[k]);
        if (lane == 0) red[sg][k] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid <= (uint)K) {
        float t = 0.0f;
        for (uint g = 0; g < (T + 31) / 32; ++g) t += red[g][tid];
        partial[(row * NBLK + blk) * (K + 1) + tid] = t;
    }
"""


_FUSED_SRC2 = r"""
    const uint row = threadgroup_position_in_grid.y;
    const uint hb = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint T = threads_per_threadgroup.x;
    const int H = H_DIM;
    const int S = 4;
    const int K = 17;
    const int D = S * H;
    threadgroup float tail[4 + 4 + 16];
    if (tid == 0) {
        float tot[K + 1];
        for (int k = 0; k <= K; ++k) {
            float t = 0.0f;
            for (int b = 0; b < NBLK; ++b) t += partial[(row * NBLK + b) * (K + 1) + k];
            tot[k] = t;
        }
        const float inv = metal::rsqrt(tot[K] / (float)D + prm[0]);
        float z[K];
        for (int k = 0; k < K; ++k) z[k] = (float)(bfloat16_t)tot[k] * inv;
        for (int i = 0; i < 4; ++i) {
            tail[i] = 1.0f / (1.0f + metal::exp(-(prm[2] * z[i] + prm[3 + i])));
            tail[4 + i] = prm[1] / (1.0f + metal::exp(-(prm[7] * z[4 + i] + prm[8 + i])));
        }
        float c[9];
        for (int k = 0; k < 9; ++k) c[k] = prm[12] * z[8 + k] + prm[13 + k];
        float e[16];
        float rho = 0.0f;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                float t = 0.0f;
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        t += prm[23 + i * 3 + a] * c[a * 3 + b] * prm[23 + j * 3 + b];
                e[i * 4 + j] = t;
                rho = metal::max(rho, -t / prm[35 + i * 4 + j]);
            }
        const float g = prm[22];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                tail[8 + i * 4 + j] = (1.0f - g) * (i == j ? 1.0f : 0.0f)
                                      + g * (prm[35 + i * 4 + j] + e[i * 4 + j] / (1.0f + rho));
        if (hb == 0) {
            for (int i = 0; i < 4; ++i) post_out[row * 4 + i] = tail[4 + i];
            for (int k = 0; k < 16; ++k) res_out[row * 16 + k] = tail[8 + k];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int h = hb * T + tid;
    if (h < H) {
        float a = 0.0f;
        for (int s = 0; s < S; ++s) a += (float)new_rows[row * D + s * H + h] * tail[s];
        agg[row * H + h] = (bfloat16_t)a;
    }
"""


_FUSED_KERNELS = {}


def _fused_kernels():
    if not _FUSED_KERNELS:
        _FUSED_KERNELS["k1"] = mx.fast.metal_kernel(
            name="raon_mhc_pass1",
            input_names=["rows", "branch", "post_in", "res_in", "phi"],
            output_names=["new_rows", "partial"],
            source=_FUSED_SRC1,
        )
        _FUSED_KERNELS["k2"] = mx.fast.metal_kernel(
            name="raon_mhc_pass2",
            input_names=["partial", "new_rows", "prm"],
            output_names=["agg", "post_out", "res_out"],
            source=_FUSED_SRC2,
        )
    return _FUSED_KERNELS["k1"], _FUSED_KERNELS["k2"]


def fused_mhc(rows, branch, post, res, mhc, do_mix):
    """rows (N,4,H) bf16; branch (N,H) bf16 or None; post (N,4) f32; res (N,4,4) f32.
    Returns new_rows, agg (N,H), post_next, res_next for `mhc`."""
    n, s, h = rows.shape
    if branch is None:
        branch = mx.zeros((n, h), mx.bfloat16)
        post = mx.zeros((n, 4), mx.float32)
        res = mx.zeros((n, 4, 4), mx.float32)
    d = s * h
    nblk = (d + _FUSED_BLK - 1) // _FUSED_BLK
    k1, k2 = _fused_kernels()
    new_rows, partial = k1(
        inputs=[
            rows,
            branch,
            post.astype(mx.float32),
            res.astype(mx.float32),
            mhc.fused_phi(),
        ],
        template=[
            ("H_DIM", h),
            ("DO_MIX", int(bool(do_mix))),
            ("BLK", _FUSED_BLK),
            ("NBLK", nblk),
        ],
        grid=(_FUSED_TG * nblk, n, 1),
        threadgroup=(_FUSED_TG, 1, 1),
        output_shapes=[(n, s, h), (n, nblk, 18)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )
    hb = (h + _FUSED_TG - 1) // _FUSED_TG
    agg, post_out, res_out = k2(
        inputs=[partial, new_rows, mhc.fused_params()],
        template=[("H_DIM", h), ("NBLK", nblk)],
        grid=(_FUSED_TG * hb, n, 1),
        threadgroup=(_FUSED_TG, 1, 1),
        output_shapes=[(n, h), (n, 4), (n, 4, 4)],
        output_dtypes=[mx.bfloat16, mx.float32, mx.float32],
    )
    return new_rows, agg, post_out, res_out


_FUSED_SRC2_NORM = r"""
    const uint row = threadgroup_position_in_grid.y;
    const uint hb = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint T = threads_per_threadgroup.x;
    const int H = H_DIM;
    const int S = 4;
    const int K = 17;
    const int D = S * H;
    threadgroup float tail[4 + 4 + 16];
    threadgroup float tots[K + 1];
    if (tid <= (uint)K) {
        float t = 0.0f;
        for (int b = 0; b < NBLK; ++b) t += partial[(row * NBLK + b) * (K + 1) + tid];
        tots[tid] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float zs[K];
    threadgroup float cs[9];
    threadgroup float es[16];
    threadgroup float rhos[1];
    const float inv_t = metal::rsqrt(tots[K] / (float)D + prm[0]);
    if (tid < (uint)K) zs[tid] = (float)(bfloat16_t)tots[tid] * inv_t;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 4) {
        tail[tid] = 1.0f / (1.0f + metal::exp(-(prm[2] * zs[tid] + prm[3 + tid])));
    } else if (tid < 8) {
        const int i = tid - 4;
        tail[4 + i] = prm[1] / (1.0f + metal::exp(-(prm[7] * zs[4 + i] + prm[8 + i])));
    } else if (tid < 17) {
        const int k = tid - 8;
        cs[k] = prm[12] * zs[8 + k] + prm[13 + k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16) {
        const int i = tid / 4, j = tid % 4;
        float t = 0.0f;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                t += prm[23 + i * 3 + a] * cs[a * 3 + b] * prm[23 + j * 3 + b];
        es[tid] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float rho = 0.0f;
        for (int ij = 0; ij < 16; ++ij) rho = metal::max(rho, -es[ij] / prm[35 + ij]);
        rhos[0] = rho;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16) {
        const int i = tid / 4, j = tid % 4;
        const float g = prm[22];
        tail[8 + tid] = (1.0f - g) * (i == j ? 1.0f : 0.0f) + g * (prm[35 + tid] + es[tid] / (1.0f + rhos[0]));
        res_out[row * 16 + tid] = tail[8 + tid];
        if (tid < 4) post_out[row * 4 + tid] = tail[4 + tid];
    }
    threadgroup float local_sums[32];
    threadgroup float local_inv[1];
    const uint lane = tid % 32;
    const uint sg = tid / 32;
    if (sg == 0) local_sums[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float xs[4];
    float acc = 0.0f;
    for (int i = 0; i < 4; ++i) {
        const int h = tid * 4 + i;
        float x = 0.0f;
        if (h < H) {
            float a = 0.0f;
            for (int s = 0; s < S; ++s) a += (float)new_rows[row * D + s * H + h] * tail[s];
            x = (float)(bfloat16_t)a;
        }
        xs[i] = x;
        acc += x * x;
    }
    acc = simd_sum(acc);
    if (lane == 0) local_sums[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        acc = simd_sum(local_sums[lane]);
        if (lane == 0) local_inv[0] = metal::precise::rsqrt(acc / (uint)H + eps[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < 4; ++i) {
        const int h = tid * 4 + i;
        if (h < H) {
            const bfloat16_t o = w[h] * static_cast<bfloat16_t>(xs[i] * local_inv[0]);
            normed[row * H + h] = o;
            if (WITH_F32) normed32[row * H + h] = static_cast<float>(o);
        }
    }
"""


_FUSED_SRC1_MOE = r"""
    const uint row = threadgroup_position_in_grid.y;
    const uint blk = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint T = threads_per_threadgroup.x;
    const int H = H_DIM;
    const int S = 4;
    const int K = 17;
    const int D = S * H;
    const int B = BLK;
    threadgroup float red[32][K + 1];
    float acc[K + 1];
    for (int k = 0; k <= K; ++k) acc[k] = 0.0f;
    const device bfloat16_t* rin = rows + row * D;
    device bfloat16_t* rout = new_rows + row * D;
    const int start = blk * B;
    const int stop = metal::min(start + B, D);
    for (int idx = start + tid; idx < stop; idx += T) {
        const int s = idx / H;
        const int h = idx - s * H;
        float v;
        if (DO_MIX) {
            float m = 0.0f;
            for (int j = 0; j < S; ++j) m += res_in[row * 16 + s * 4 + j] * (float)rin[j * H + h];
            float t = 0.0f;
            for (int kk = 0; kk < TOPK; ++kk) {
                const float pr = (float)mo[(row * TOPK + kk) * H + h] * mw[row * TOPK + kk];
                t += as_type<float>(as_type<uint>(pr));
            }
            const bfloat16_t bv = static_cast<bfloat16_t>(t) + sh[row * H + h];
            m += post_in[row * 4 + s] * (float)bv;
            bfloat16_t mb = (bfloat16_t)m;
            rout[idx] = mb;
            v = (float)mb;
        } else {
            v = (float)rin[idx];
            rout[idx] = rin[idx];
        }
        acc[K] += v * v;
        for (int k = 0; k < K; ++k) acc[k] += v * (float)phi[k * D + idx];
    }
    const uint lane = tid % 32;
    const uint sg = tid / 32;
    for (int k = 0; k <= K; ++k) {
        float t = simd_sum(acc[k]);
        if (lane == 0) red[sg][k] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid <= (uint)K) {
        float t = 0.0f;
        for (uint g = 0; g < (T + 31) / 32; ++g) t += red[g][tid];
        partial[(row * NBLK + blk) * (K + 1) + tid] = t;
    }
"""


_FINAL_SRC = r"""
    const uint row = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = tid % 32;
    const uint sg = tid / 32;
    const int H = H_DIM;
    const int S = 4;
    threadgroup float local_sums[32];
    threadgroup float local_inv[1];
    if (sg == 0) local_sums[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device bfloat16_t* rin = rows + row * S * H;
    float xs[4];
    float acc = 0.0f;
    for (int i = 0; i < 4; ++i) {
        const int h = tid * 4 + i;
        float x = 0.0f;
        if (h < H) {
            float t = 0.0f;
            for (int kk = 0; kk < TOPK; ++kk) {
                const float pr = (float)mo[(row * TOPK + kk) * H + h] * mw[row * TOPK + kk];
                t += as_type<float>(as_type<uint>(pr));
            }
            const bfloat16_t bv = static_cast<bfloat16_t>(t) + sh[row * H + h];
            float sm = 0.0f;
            for (int s = 0; s < S; ++s) {
                float m = 0.0f;
                for (int j = 0; j < S; ++j) m += res_in[row * 16 + s * 4 + j] * (float)rin[j * H + h];
                const float pb = post_in[row * 4 + s] * (float)bv;
                m += as_type<float>(as_type<uint>(pb));
                sm += (float)(bfloat16_t)m;
            }
            x = (float)static_cast<bfloat16_t>(sm * 0.25f);
        }
        xs[i] = x;
        acc += x * x;
    }
    acc = simd_sum(acc);
    if (lane == 0) local_sums[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        acc = simd_sum(local_sums[lane]);
        if (lane == 0) local_inv[0] = metal::precise::rsqrt(acc / (uint)H + eps[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < 4; ++i) {
        const int h = tid * 4 + i;
        if (h < H) normed[row * H + h] = w[h] * static_cast<bfloat16_t>(xs[i] * local_inv[0]);
    }
"""


_ROUTER_SRC = r"""
    const uint row = threadgroup_position_in_grid.x;
    const uint e = thread_position_in_threadgroup.x;
    const int E = N_EXPERTS;
    const int K = TOP_K;
    threadgroup float sv[N_EXPERTS];
    threadgroup float bv[32];
    threadgroup int bi[32];
    threadgroup int chosen[TOP_K];
    auto sigmoid = [](float z) { float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(z))); return (z < 0.0f) ? y : 1.0f - y; };
    const float s = sigmoid(logits[row * E + e]);
    float r = -(s + bias[e]);
    sv[e] = s;
    const uint lane = e % 32, sg = e / 32;
    for (int j = 0; j < K; ++j) {
        float v = r;
        int i = (int)e;
        for (uint off = 16; off > 0; off >>= 1) {
            float v2 = simd_shuffle_down(v, off);
            int i2 = simd_shuffle_down(i, off);
            if (v2 < v || (v2 == v && i2 < i)) { v = v2; i = i2; }
        }
        if (lane == 0) { bv[sg] = v; bi[sg] = i; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (e == 0) {
            float bvv = bv[0];
            int bii = bi[0];
            for (int g = 1; g < E / 32; ++g) {
                if (bv[g] < bvv || (bv[g] == bvv && bi[g] < bii)) { bvv = bv[g]; bii = bi[g]; }
            }
            chosen[j] = bii;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((int)e == chosen[j]) r = INFINITY;
    }
    if (e == 0) {
        float total = 0.0f;
        for (int j = 0; j < K; ++j) total = sv[chosen[j]] + total;
        const float denom = total + 1e-20f;
        for (int j = 0; j < K; ++j) {
            indices[row * K + j] = (uint32_t)chosen[j];
            weights[row * K + j] = metal::precise::divide(sv[chosen[j]], denom) * scale[0];
        }
    }
"""


# Adapted from the gated delta-rule kernel in mlx-lm (MIT License), with the output RMS norm and
# sigmoid gate fused in.
_RDA_OUT_SRC = r"""
    auto n = threadgroup_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;
    threadgroup float ys[Dv];
    auto q_ = q + b_idx * Hk * Dk + hk_idx * Dk;
    auto k_ = k + b_idx * Hk * Dk + hk_idx * Dk;
    auto v_ = v + b_idx * Hv * Dv + hv_idx * Dv;
    auto g_ = g + (b_idx * Hv + hv_idx) * Dk;
    const float bt = beta[b_idx * Hv + hv_idx];
    auto dk_idx = thread_position_in_threadgroup.x;
    auto rg = thread_position_in_threadgroup.y;
    float qv[n_per_t], kv[n_per_t], gv[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      qv[i] = q_[s_idx];
      kv[i] = k_[s_idx];
      gv[i] = g_[s_idx];
    }
    for (int r = 0; r < ROWS; ++r) {
      auto dv_idx = rg * ROWS + r;
      auto i_state = state_in + (n * Dv + dv_idx) * Dk;
      auto o_state = state_out + (n * Dv + dv_idx) * Dk;
      float state[n_per_t];
      for (int i = 0; i < n_per_t; ++i)
        state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        state[i] = state[i] * gv[i];
        kv_mem += state[i] * kv[i];
      }
      kv_mem = simd_sum(kv_mem);
      auto delta = (v_[dv_idx] - kv_mem) * bt;
      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        state[i] = state[i] + kv[i] * delta;
        out += state[i] * qv[i];
      }
      out = simd_sum(out);
      if (dk_idx == 0)
        ys[dv_idx] = static_cast<float>(static_cast<InT>(out));
      for (int i = 0; i < n_per_t; ++i)
        o_state[n_per_t * dk_idx + i] = static_cast<StT>(state[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (rg == 0) {
      float xs[4];
      float acc = 0.0f;
      for (int i = 0; i < 4; ++i) {
        const int h = dk_idx * 4 + i;
        xs[i] = (h < Dv) ? ys[h] : 0.0f;
        acc += xs[i] * xs[i];
      }
      acc = simd_sum(acc);
      acc = simd_sum(dk_idx == 0 ? acc : 0.0f);
      const float inv = metal::precise::rsqrt(acc / (uint)Dv + eps[0]);
      auto gate_ = gate + (b_idx * Hv + hv_idx) * Dv;
      auto o_ = gated + (b_idx * Hv + hv_idx) * Dv;
      for (int i = 0; i < 4; ++i) {
        const int h = dk_idx * 4 + i;
        if (h < Dv) {
          const float nv = norm_w[h] * (xs[i] * inv);
          const float z = static_cast<float>(gate_[h]);
          const float e = 1.0f / (1.0f + metal::precise::exp(metal::abs(z)));
          o_[h] = static_cast<bfloat16_t>(nv * ((z < 0.0f) ? e : 1.0f - e));
        }
      }
    }
"""


_FUSION_KERNELS = {}


def decode_kernel(name):
    if name not in _FUSION_KERNELS:
        spec = {
            "pass1_moe": (
                ["rows", "mo", "mw", "sh", "post_in", "res_in", "phi"],
                ["new_rows", "partial"],
                _FUSED_SRC1_MOE,
            ),
            "pass2_norm": (
                ["partial", "new_rows", "prm", "w", "eps"],
                ["normed", "post_out", "res_out", "normed32"],
                _FUSED_SRC2_NORM,
            ),
            "final": (
                ["rows", "mo", "mw", "sh", "post_in", "res_in", "w", "eps"],
                ["normed"],
                _FINAL_SRC,
            ),
            "router": (
                ["logits", "bias", "scale"],
                ["indices", "weights"],
                _ROUTER_SRC,
            ),
            "rda_out": (
                ["q", "k", "v", "g", "beta", "state_in", "norm_w", "gate", "eps"],
                ["gated", "state_out"],
                _RDA_OUT_SRC,
            ),
        }[name]
        _FUSION_KERNELS[name] = mx.fast.metal_kernel(
            name=f"raon_{name}",
            input_names=spec[0],
            output_names=spec[1],
            source=spec[2],
        )
    return _FUSION_KERNELS[name]


def _norm_threads(h):
    return 32 * ((h + 127) // 128)


def fused_mhc_norm(rows, branch, post, res, mhc, do_mix, norm, with_f32=False):
    """fused_mhc followed by `norm` on the aggregate. `branch` is a bf16 (N,H) array, or the MoE parts
    (routed out (N,K,H) bf16, weights (N,K) f32, shared (N,H) bf16) to combine in pass1.
    Returns new_rows, normed (N,H) bf16, post_next, res_next and, with `with_f32`, the f32 copy of normed.
    """
    n, s, h = rows.shape
    if branch is None:
        branch = mx.zeros((n, h), mx.bfloat16)
        post = mx.zeros((n, 4), mx.float32)
        res = mx.zeros((n, 4, 4), mx.float32)
    nblk = (s * h + _FUSED_BLK - 1) // _FUSED_BLK
    template = [
        ("H_DIM", h),
        ("DO_MIX", int(bool(do_mix))),
        ("BLK", _FUSED_BLK),
        ("NBLK", nblk),
    ]
    tail = [post.astype(mx.float32), res.astype(mx.float32), mhc.fused_phi()]
    if isinstance(branch, tuple):
        mo, mw, sh = branch
        k = mw.shape[-1]
        kernel, inputs = decode_kernel("pass1_moe"), [
            rows,
            mo.reshape(n, k, h),
            mw.reshape(n, k),
            sh.reshape(n, h),
        ]
        template.append(("TOPK", k))
    else:
        kernel, inputs = _fused_kernels()[0], [rows, branch]
    new_rows, partial = kernel(
        inputs=inputs + tail,
        template=template,
        grid=(_FUSED_TG * nblk, n, 1),
        threadgroup=(_FUSED_TG, 1, 1),
        output_shapes=[(n, s, h), (n, nblk, 18)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )
    eps = cached_array(norm, "_eps32", lambda: mx.array([norm.eps], mx.float32))
    tg = _norm_threads(h)
    normed, post_out, res_out, normed32 = decode_kernel("pass2_norm")(
        inputs=[partial, new_rows, mhc.fused_params(), norm.weight, eps],
        template=[("H_DIM", h), ("NBLK", nblk), ("WITH_F32", int(with_f32))],
        grid=(tg, n, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(n, h), (n, 4), (n, 4, 4), (n, h) if with_f32 else (1,)],
        output_dtypes=[mx.bfloat16, mx.float32, mx.float32, mx.float32],
    )
    return (new_rows, normed, post_out, res_out) + ((normed32,) if with_f32 else ())


def fused_final(rows, parts, post, res, norm):
    """Last MoE combine, last mHC mix, mean over the streams and `norm`: (N,H) bf16."""
    mo, mw, sh = parts
    n, s, h = rows.shape
    k = mw.shape[-1]
    eps = cached_array(norm, "_eps32", lambda: mx.array([norm.eps], mx.float32))
    tg = _norm_threads(h)
    return decode_kernel("final")(
        inputs=[
            rows,
            mo.reshape(n, k, h),
            mw.reshape(n, k),
            sh.reshape(n, h),
            post.astype(mx.float32),
            res.astype(mx.float32),
            norm.weight,
            eps,
        ],
        template=[("H_DIM", h), ("TOPK", k)],
        grid=(tg, n, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(n, h)],
        output_dtypes=[mx.bfloat16],
    )[0]


_RDA_PREP_SOURCE = r"""

    const uint b = threadgroup_position_in_grid.y;
    const uint h = threadgroup_position_in_grid.x;
    const uint d = thread_position_in_threadgroup.x;
    const int D = HEAD_DIM;
    const int NH = N_HEADS;
    const int P = NH * D;
    const int KS = KSIZE;
    const int KEEP = KS - 1;

    float outv[3];
    for (int part = 0; part < 3; ++part) {
        const int c = part * P + h * D + d;               // channel in the merged q|k|v conv
        float acc = 0.0f;
        for (int t = 0; t < KEEP; ++t)
            acc += (float)state[(b * KEEP + t) * 3 * P + c] * w32[c * KS + t];
        const float xin = (float)x[b * 3 * P + c];
        acc += xin * w32[c * KS + KEEP];
        for (int t = 0; t < KEEP - 1; ++t)
            new_state[(b * KEEP + t) * 3 * P + c] = state[(b * KEEP + t + 1) * 3 * P + c];
        new_state[(b * KEEP + KEEP - 1) * 3 * P + c] = x[b * 3 * P + c];
        const float ys = 1.0f / (1.0f + metal::precise::exp(metal::abs(acc)));
        const float s = acc * ((acc < 0.0f) ? ys : 1.0f - ys);
        outv[part] = (float)(bfloat16_t)s;
    }
    // Sum in the order of the MLX row reduction (32 lanes, D / 32 values each) to match _rda_qk.
    threadgroup float sq[2][HEAD_DIM];
    threadgroup float tot[2];
    sq[0][d] = outv[0] * outv[0];
    sq[1][d] = outv[1] * outv[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d < 32) {
      float pq = 0.0f, pk = 0.0f;
      for (int i = 0; i < D / 32; ++i) { pq = sq[0][(D / 32) * d + i] + pq; pk = sq[1][(D / 32) * d + i] + pk; }
      pq = simd_sum(pq);
      pk = simd_sum(pk);
      if (d == 0) { tot[0] = pq; tot[1] = pk; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float qs = tot[0], ks = tot[1];
    const float qn = (float)(bfloat16_t)metal::precise::divide(outv[0], metal::max(metal::precise::sqrt(qs), 1e-6f));
    const float kn = (float)(bfloat16_t)metal::precise::divide(outv[1], metal::max(metal::precise::sqrt(ks), 1e-6f));
    const int o = (b * NH + h) * D + d;
    q_out[o] = (bfloat16_t)(qn * (float)(bfloat16_t)fp[0]);
    k_out[o] = (bfloat16_t)kn;
    const float br = (float)braw[b * NH + h];
    const float dl = (float)delta[b * NH + h];
    // Same formulas as the MLX ops, so the results match _rda_gates.
    auto sigmoid = [](float z) { float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(z))); return (z < 0.0f) ? y : 1.0f - y; };
    auto log1p_mlx = [](float z) { float xp1 = 1.0f + z; if (xp1 == 1.0f) return z; return z * (metal::log(xp1) / (xp1 - 1.0f)); };
    auto softplus = [&](float z) {
      float mx_ = metal::max(z, 0.0f), mn_ = metal::min(z, 0.0f);
      return (metal::isinf(mn_) || metal::isinf(mx_)) ? mx_ : mx_ + log1p_mlx(metal::exp(mn_ - mx_)); };
    const float ratio = metal::precise::exp(-softplus(-br - dl) + softplus(-br));
    v_out[o] = (bfloat16_t)(outv[2] * ratio);
    const float dec = ((float)decay[b * P + h * D + d] + dt32[h * D + d]) * a_exp[h];
    g_out[o] = metal::precise::exp(fp[1] * sigmoid(dec));
    if (d == 0) beta_out[b * NH + h] = (bfloat16_t)sigmoid(br);
"""


_RDA_PREP_KERNEL = {}


def rda_prep_kernel():
    if "k" not in _RDA_PREP_KERNEL:
        _RDA_PREP_KERNEL["k"] = mx.fast.metal_kernel(
            name="raon_rda_decode_prep",
            input_names=[
                "x",
                "state",
                "w32",
                "braw",
                "delta",
                "decay",
                "dt32",
                "a_exp",
                "fp",
            ],
            output_names=["q_out", "k_out", "v_out", "g_out", "beta_out", "new_state"],
            source=_RDA_PREP_SOURCE,
        )
    return _RDA_PREP_KERNEL["k"]


# Copyright © 2026 KRAFTON Inc.
"""Raon with merged projections and Metal decode kernels.

Prefill and unsupported devices or dtypes use MLX operations. The Metal
mHC path changes float32 accumulation order relative to the operation path.
"""

import math
from dataclasses import dataclass
from functools import partial
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.gated_delta import gated_delta_kernel, gated_delta_ops
from mlx_lm.models.mla import MultiLinear
from mlx_lm.models.switch_layers import (
    QuantizedSwitchLinear,
    SwitchGLU,
    SwitchLinear,
    _gather_sort,
    _scatter_unsort,
)


# rda_config keys this model reads; any other key is an error.
_RDA_CONFIG_KEYS = frozenset(
    {"num_heads", "head_dim", "conv_kernel_size", "full_rank_gate", "gate_lower_bound", "rda_layers",
     "mla_layers"}
)

# Adapted from mlx-lm (MIT License).
# SiTU activation: beta * tanh(gate / beta) * sigmoid(gate) times linear_beta * tanh(x / linear_beta), computed in float32
# and rounded once to the input dtype, as one compiled kernel.
@partial(mx.compile, shapeless=True)
def _situ(x, gate, beta, linear_beta):
    dtype = x.dtype
    gate = gate.astype(mx.float32)
    x = x.astype(mx.float32)
    a = beta * mx.tanh(gate / beta) * mx.sigmoid(gate)
    if linear_beta is not None:
        x = linear_beta * mx.tanh(x / linear_beta)
    return (a * x).astype(dtype)


class SiTU(nn.Module):
    def __init__(self, beta: float = 1.0, linear_beta: Optional[float] = None):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        return _situ(x, gate, self.beta, self.linear_beta)


@dataclass
class MiniModelArgs(BaseModelArgs):
    model_type: str = "raon1_mini"
    hidden_size: int = 2560
    num_hidden_layers: int = 25
    num_attention_heads: int = 40
    head_dim: int = 128
    vocab_size: int = 259840
    intermediate_size: int = 12032
    rms_norm_eps: float = 1e-5
    mla_latent_norm_eps: float = 1e-6
    raon_mhc_selector_eps: float = 1e-5
    raon_mhc_streams: int = 4
    raon_mhc_post_factor: float = 2.0
    hidden_act: str = "situ"
    activation_situ_beta: float = 4.0
    activation_situ_linear_beta: float = 25.0
    num_experts: Optional[int] = 128
    num_experts_per_token: int = 6
    moe_intermediate_size: int = 1088
    num_shared_experts: int = 1
    first_k_dense_replace: int = 1
    moe_layer_freq: int = 1
    moe_router_activation_func: str = "sigmoid"
    moe_renormalize: bool = True
    routed_scaling_factor: float = 1.0
    use_grouped_topk: bool = False
    routed_expert_hidden_size: Optional[int] = None
    latent_moe_use_norm: bool = False
    q_lora_rank: int = 576
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    mla_use_nope: bool = True
    mla_use_output_gate: bool = True
    rda_config: Optional[dict] = None
    rda_decay_rank: int = 128
    num_nextn_predict_layers: int = 1
    attn_res_block_size: Optional[int] = None
    tie_word_embeddings: bool = False
    raon_fp32_export_head: bool = False
    raon_fused_mhc_decode: bool = True
    raon_global_scaled_linears: Optional[list] = None
    raon_merged_gate_up: bool = True

    def __post_init__(self):
        if self.hidden_act != "situ" or self.moe_router_activation_func != "sigmoid":
            raise ValueError("RAON requires SiTU activation and sigmoid routing")
        if (
            self.use_grouped_topk
            or self.routed_expert_hidden_size is not None
            or self.latent_moe_use_norm
        ):
            raise ValueError("Grouped routing and latent MoE are not supported")
        if not self.mla_use_nope or self.attn_res_block_size is not None:
            raise ValueError("RAON requires NoPE MLA and mHC residual streams")
        if self.raon_mhc_streams < 2 or self.moe_layer_freq < 1:
            raise ValueError("Invalid stream count or MoE layer frequency")
        if self.num_experts and not 0 < self.num_experts_per_token <= self.num_experts:
            raise ValueError("Invalid number of active experts")
        if self.rda_config is None:
            raise ValueError("config must set rda_config")
        unknown = set(self.rda_config) - _RDA_CONFIG_KEYS
        missing = {"rda_layers", "mla_layers"} - set(self.rda_config)
        if unknown or missing:
            raise ValueError(f"rda_config keys: unknown {sorted(unknown)}, missing {sorted(missing)}")
        if not self.rda_config.get("full_rank_gate", True):
            raise ValueError("Only full-rank RDA gates are supported")
        linear = self.rda_config["rda_layers"]
        full = self.rda_config["mla_layers"]
        if sorted(linear + full) != list(range(1, self.num_hidden_layers + 1)):
            raise ValueError("Attention layer lists must partition all layers")


_DECODE_ROWS = 8


def _qparams(m):
    """(kind, group_size, bits, mode) of a quantized linear without bias, or None."""
    if isinstance(m, GlobalScaledLinear):
        return ("gs", 16, 4, "nvfp4")
    if isinstance(m, nn.QuantizedLinear) and "bias" not in m:
        return ("q", m.group_size, m.bits, m.mode)
    return None


def _merged_decode(module, names):
    """A cached merged copy of `names` (same input, same quantization) for decode, or None."""
    d = module.__dict__
    if "_raon_merged" not in d:
        mods = [getattr(module, n) for n in names]
        kinds = {_qparams(m) for m in mods}
        merged = None
        if len(kinds) == 1 and None not in kinds:
            kind, gs, bits, mode = kinds.pop()
            w = mx.concatenate([m["weight"] for m in mods], axis=0)
            sc = mx.concatenate([m["scales"] for m in mods], axis=0)
            b = (
                mx.concatenate([m["biases"] for m in mods], axis=0)
                if all(m.get("biases") is not None for m in mods)
                else None
            )
            col = None
            if kind == "gs":
                col = mx.concatenate(
                    [
                        mx.broadcast_to(
                            m["global_scale"].astype(mx.float32),
                            (m["weight"].shape[0],),
                        )
                        for m in mods
                    ]
                )
            sizes, off = [], 0
            for m in mods[:-1]:
                off += m["weight"].shape[0]
                sizes.append(off)
            mx.eval(w, sc, *[a for a in (b, col) if a is not None])
            merged = (w, sc, b, col, gs, bits, mode, sizes)
            _share_parts(mods, merged)
        object.__setattr__(module, "_raon_merged", merged)
    return d["_raon_merged"]


def _share_parts(mods, merged):
    """Decode fusion: point the part linears at row slices of the merged arrays, so the original copies are freed.
    The slices hold the same bytes with the same shapes, so the per-part (prefill) products do not change.
    """
    w, sc, b, _, _, _, _, sizes = merged
    bounds = [0] + list(sizes) + [w.shape[0]]
    parts = []
    for m, lo, hi in zip(mods, bounds[:-1], bounds[1:]):
        m.weight, m.scales = w[lo:hi], sc[lo:hi]
        parts += [m.weight, m.scales]
        if b is not None:
            m.biases = b[lo:hi]
            parts.append(m.biases)
    mx.eval(*parts)


def _project(module, names, x):
    """Outputs of the linears `names` on x; one merged quantized matmul for decode-sized inputs."""
    if not module.training and x.size // x.shape[-1] <= _DECODE_ROWS:
        merged = _merged_decode(module, names)
        if merged is not None:
            w, sc, b, col, gs, bits, mode, sizes = merged
            y = mx.quantized_matmul(
                x, w, sc, b, transpose=True, group_size=gs, bits=bits, mode=mode
            )
            if col is not None:
                y = _scale_cast(y, col)
            return mx.split(y, sizes, axis=-1)
    return [getattr(module, n)(x) for n in names]


@mx.compile
def _scale_cast(y, g):
    return (y.astype(mx.float32) * g).astype(y.dtype)


@partial(mx.compile, shapeless=False)
def _route_tail(
    z,
    alpha_pre,
    b_pre,
    alpha_post,
    b_post,
    alpha_z,
    b_z,
    eta,
    helmert,
    chart_c,
    eye,
    post_factor,
):
    n = eye.shape[0]
    f32 = mx.float32
    pre = mx.sigmoid(alpha_pre.astype(f32) * z[:, :n] + b_pre.astype(f32))
    post = post_factor * mx.sigmoid(
        alpha_post.astype(f32) * z[:, n : 2 * n] + b_post.astype(f32)
    )
    chart = alpha_z.astype(f32) * z[:, 2 * n :] + b_z.astype(f32)
    e = helmert @ chart.reshape(-1, n - 1, n - 1) @ helmert.T
    rho = mx.maximum((-e / chart_c).max(axis=(-2, -1)), 0)
    phi = chart_c + e / (1 + rho[:, None, None])
    gamma = mx.sigmoid(eta.astype(f32))
    return pre, post, (1 - gamma) * eye + gamma * phi


@mx.compile
def _route_head(flat, weights, eps):
    inv = mx.rsqrt(mx.square(flat.astype(mx.float32)).mean(-1, keepdims=True) + eps)
    return (flat @ weights.astype(flat.dtype).T).astype(mx.float32) * inv


@mx.compile
def _aggregate(stream, pre):
    return (stream.astype(mx.float32) * pre[..., None]).sum(-2).astype(stream.dtype)


@mx.compile
def _mix(stream, branch, post, residual):
    return (
        residual @ stream.astype(mx.float32)
        + post[..., None] * branch.astype(mx.float32)[:, None]
    ).astype(stream.dtype)


class StreamMix(nn.Module):
    def __init__(self, streams, hidden):
        super().__init__()
        size = (streams - 1) ** 2
        self.phi_z = mx.zeros((size, streams * hidden))
        self.alpha_z = mx.array(1.0)
        self.b_z = mx.zeros((size,))
        self.eta = mx.array(0.0)


class MHC(nn.Module):
    def __init__(self, args):
        super().__init__()
        n, hidden = args.raon_mhc_streams, args.hidden_size
        self.num_streams = n
        self.eps = args.raon_mhc_selector_eps
        self.post_factor = args.raon_mhc_post_factor
        self.phi_pre = mx.zeros((n, n * hidden))
        self.phi_post = mx.zeros((n, n * hidden))
        self.alpha_pre = mx.array(1.0)
        self.alpha_post = mx.array(1.0)
        self.b_pre = mx.zeros((n,))
        self.b_post = mx.zeros((n,))
        self.res = StreamMix(n, hidden)
        self._helmert = mx.array(
            [
                [
                    (1.0 if i < k else -float(k) if i == k else 0.0)
                    / math.sqrt(k * (k + 1))
                    for k in range(1, n)
                ]
                for i in range(n)
            ]
        )
        self._eye = mx.eye(n)
        self._chart = (self._eye + mx.ones((n, n)) / n) * 0.5

    def phi(self):
        return cached_array(
            self,
            "_phi_cat",
            lambda: mx.concatenate([self.phi_pre, self.phi_post, self.res.phi_z]),
        )

    def route(self, stream):
        n = self.num_streams
        flat = stream.reshape(-1, n * stream.shape[-1])
        z = _route_head(flat, self.phi(), mx.array(self.eps, mx.float32))
        return _route_tail(
            z,
            self.alpha_pre,
            self.b_pre,
            self.alpha_post,
            self.b_post,
            self.res.alpha_z,
            self.res.b_z,
            self.res.eta,
            self._helmert,
            self._chart,
            self._eye,
            mx.array(self.post_factor, mx.float32),
        )

    aggregate = staticmethod(_aggregate)
    mix = staticmethod(_mix)

    # fused mHC kernel inputs
    def fused_phi(self):
        return cached_array(self, "_fused_phi", lambda: self.phi().astype(mx.bfloat16))

    def fused_params(self):
        def make():
            f = lambda a: a.astype(mx.float32).reshape(-1)
            gamma = mx.sigmoid(self.res.eta.astype(mx.float32)).reshape(-1)
            out = mx.concatenate(
                [
                    mx.array([self.eps, self.post_factor], mx.float32),
                    f(self.alpha_pre),
                    f(self.b_pre),
                    f(self.alpha_post),
                    f(self.b_post),
                    f(self.res.alpha_z),
                    f(self.res.b_z),
                    gamma,
                    f(self._helmert),
                    f(self._chart),
                ]
            )
            if out.size != 51:
                raise ValueError("fused mHC kernel expects 4 streams")
            return out

        return cached_array(self, "_fused_prm", make)


_CACHE_EVAL_EVERY = 64


def _bound_cache_graph(cache, names):
    """Evaluate batched cache counters periodically to bound lazy graph growth."""
    d = cache.__dict__
    n = d.get("_raon_cf_n", 0) + 1
    d["_raon_cf_n"] = n
    if n % _CACHE_EVAL_EVERY == 0:
        arrs = [
            a
            for a in (getattr(cache, k, None) for k in names)
            if isinstance(a, mx.array)
        ]
        if arrs:
            mx.eval(*arrs)


class ShortConv(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.weight = mx.zeros((channels, kernel_size, 1))
        self.kernel_size = kernel_size

    def __call__(self, x, state, mask, lengths):
        if mask is not None:
            x = mx.where(mask[..., None], x, 0)
        keep = self.kernel_size - 1
        if state is None:
            state = mx.zeros((x.shape[0], keep, x.shape[-1]), dtype=x.dtype)
        signal = mx.concatenate([state, x], axis=1)
        w32 = cached_array(self, "_w32", lambda: self.weight.astype(mx.float32))
        out = nn.silu(
            mx.conv1d(signal.astype(mx.float32), w32, groups=x.shape[-1])
        ).astype(x.dtype)
        if lengths is not None:
            ends = mx.clip(lengths, 0, x.shape[1])
            indices = (ends[:, None] + mx.arange(keep))[..., None]
            state = mx.take_along_axis(signal, indices, axis=1)
        else:
            state = mx.contiguous(signal[:, -keep:, :]) if keep else signal[:, :0, :]
        return out, state


@mx.compile
def _rda_qk(q, k, inv_sqrt_d):
    def normalize(value):
        v32 = value.astype(mx.float32)
        return (
            v32 / mx.maximum(mx.sqrt(mx.square(v32).sum(-1, keepdims=True)), 1e-6)
        ).astype(value.dtype)

    return normalize(q) * inv_sqrt_d.astype(q.dtype), normalize(k)


@mx.compile
def _rda_gates(beta_raw, delta, v, decay, dt32, a_exp, lower_bound):
    f32 = mx.float32
    beta_raw, delta = beta_raw.astype(f32), delta.astype(f32)
    ratio = mx.exp(-nn.softplus(-beta_raw - delta) + nn.softplus(-beta_raw))
    v = (v.astype(f32) * ratio[..., None]).astype(v.dtype)
    decay = (decay.astype(f32) + dt32) * a_exp[:, None]
    log_decay = (
        -nn.softplus(decay) if lower_bound is None else lower_bound * mx.sigmoid(decay)
    )
    g = mx.exp(log_decay)
    return v, g, mx.sigmoid(beta_raw)


@mx.compile
def _rda_out(out, norm_w, gate_raw, eps):
    normed = mx.fast.rms_norm(out.astype(mx.float32), norm_w.astype(mx.float32), eps)
    return normed * mx.sigmoid(gate_raw.astype(mx.float32))


_RDA_PARTS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "g_proj",
    "f_a_proj",
    "b_proj",
    "rda_w_proj",
)


class RDA(nn.Module):
    def __init__(self, args):
        super().__init__()
        cfg = args.rda_config
        self.num_heads = cfg.get("num_heads", args.num_attention_heads)
        self.head_dim = cfg.get("head_dim", args.head_dim)
        self.proj_size = self.num_heads * self.head_dim
        self.lower_bound = cfg.get("gate_lower_bound", -5.0)
        self.eps = args.rms_norm_eps
        hidden, size = args.hidden_size, self.proj_size
        for name in ["q_proj", "k_proj", "v_proj", "g_proj"]:
            setattr(self, name, nn.Linear(hidden, size, bias=False))
        self.qkv_conv1d = ShortConv(3 * size, cfg.get("conv_kernel_size", 4))
        self.f_a_proj = nn.Linear(hidden, args.rda_decay_rank, bias=False)
        self.f_b_proj = nn.Linear(args.rda_decay_rank, size, bias=False)
        self.b_proj = nn.Linear(hidden, self.num_heads, bias=False)
        self.rda_w_proj = nn.Linear(hidden, self.num_heads, bias=False)
        self.A_log = mx.zeros((self.num_heads,))
        self.dt_bias = mx.zeros((size,))
        self.o_norm = nn.RMSNorm(self.head_dim, eps=self.eps)
        self.o_proj = nn.Linear(size, hidden, bias=False)

    def _fused_decode(self, x, cache):
        return (
            x.shape[1] == 1
            and cache is not None
            and cache.lengths is None
            and x.dtype == mx.bfloat16
            and self.head_dim == 128
            and self.lower_bound is not None
            and not self.training
            and mx.default_device() == mx.gpu
            and mx.metal.is_available()
        )

    def _fused_prep(self, qkv, state, braw, delta, decay, dt32, a_exp):
        batch = qkv.shape[0]
        nh, hd, conv = self.num_heads, self.head_dim, self.qkv_conv1d
        keep = conv.kernel_size - 1
        if state is None:
            state = mx.zeros((batch, keep, qkv.shape[-1]), dtype=qkv.dtype)
        w32 = cached_array(conv, "_w32", lambda: conv.weight.astype(mx.float32))
        fp = cached_array(
            self, "_prep_fp", lambda: mx.array([hd**-0.5, self.lower_bound], mx.float32)
        )
        q, k, v, g, beta, new_state = rda_prep_kernel()(
            inputs=[
                qkv.reshape(batch, -1),
                state,
                w32.reshape(-1),
                braw.reshape(batch, nh),
                delta.reshape(batch, nh),
                decay.reshape(batch, -1),
                dt32.reshape(-1),
                a_exp,
                fp,
            ],
            template=[("HEAD_DIM", hd), ("N_HEADS", nh), ("KSIZE", conv.kernel_size)],
            grid=(nh * hd, batch, 1),
            threadgroup=(hd, 1, 1),
            output_shapes=[(batch, nh, hd)] * 4 + [(batch, nh), state.shape],
            output_dtypes=[
                mx.bfloat16,
                mx.bfloat16,
                mx.bfloat16,
                mx.float32,
                mx.bfloat16,
                qkv.dtype,
            ],
        )
        shape = (batch, 1, nh, hd)
        return (
            q.reshape(shape),
            k.reshape(shape),
            v.reshape(shape),
            g.reshape(shape),
            beta.reshape(batch, 1, nh),
            new_state,
        )

    def __call__(self, x, mask=None, cache=None):
        batch, length, _ = x.shape
        dtype = x.dtype
        if mask is not None:
            x = mx.where(mask[..., None], x, 0)
        states = [None, None] if cache is None else cache.cache
        lengths = None if cache is None else cache.lengths
        qp, kp, vp, gp, fa, braw, delta = _project(self, _RDA_PARTS, x)
        shape = (batch, length, self.num_heads, self.head_dim)
        a_exp = cached_array(
            self, "_a_exp", lambda: mx.exp(self.A_log.astype(mx.float32))
        )
        dt32 = cached_array(
            self,
            "_dt32",
            lambda: self.dt_bias.astype(mx.float32).reshape(
                self.num_heads, self.head_dim
            ),
        )
        if self._fused_decode(x, cache):
            q, k, v, g, beta, conv_state = self._fused_prep(
                mx.concatenate([qp, kp, vp], axis=-1),
                states[0],
                braw,
                delta,
                self.f_b_proj(fa),
                dt32,
                a_exp,
            )
        else:
            qkv, conv_state = self.qkv_conv1d(
                mx.concatenate([qp, kp, vp], axis=-1), states[0], mask, lengths
            )
            q, k, v = mx.split(qkv, 3, axis=-1)
            q, k, v = q.reshape(shape), k.reshape(shape), v.reshape(shape)
            q, k = _rda_qk(
                q,
                k,
                cached_array(
                    self,
                    "_inv_sqrt_d",
                    lambda: mx.array(self.head_dim**-0.5, mx.float32),
                ),
            )
            # RDA may exceed float16 range before beta compensates it.
            update_dtype = mx.float32 if dtype == mx.float16 else dtype
            q, k, v = (
                q.astype(update_dtype),
                k.astype(update_dtype),
                v.astype(update_dtype),
            )
            decay = self.f_b_proj(fa).reshape(shape)
            v, g, beta = _rda_gates(
                braw,
                delta,
                v,
                decay,
                dt32,
                a_exp,
                (
                    None
                    if self.lower_bound is None
                    else cached_array(
                        self, "_lower32", lambda: mx.array(self.lower_bound, mx.float32)
                    )
                ),
            )
            beta = beta.astype(update_dtype)
        state = states[1]
        if state is None:
            state = mx.zeros(
                (batch, self.num_heads, self.head_dim, self.head_dim), dtype=mx.float32
            )
        if mask is None and self._fused_decode(x, cache):
            gated, state = self._fused_out(q, k, v, g, beta, state, gp.reshape(shape))
        else:
            use_kernel = (
                not self.training
                and mx.default_device() == mx.gpu
                and mx.metal.is_available()
                and self.head_dim >= 32
                and self.head_dim % 32 == 0
            )
            update = gated_delta_kernel if use_kernel else gated_delta_ops
            out, state = update(q, k, v, g, beta, state, mask)
            gated = _rda_out(
                out, self.o_norm.weight, gp.reshape(shape), self.eps
            ).astype(dtype)
        if cache is not None:
            cache.cache = [conv_state, state]
            cache.advance(length)
            _bound_cache_graph(cache, ("lengths", "left_padding"))
        return self.o_proj(gated.reshape(batch, length, self.proj_size))

    def _fused_out(self, q, k, v, g, beta, state, gate):
        """Decode fusion: recurrence, output norm and gate in one kernel; returns bf16 gated output and state."""
        batch, _, nh, hd = q.shape
        rows = 4
        norm_w = cached_array(
            self, "_norm_w32", lambda: self.o_norm.weight.astype(mx.float32)
        )
        eps = cached_array(self, "_eps32", lambda: mx.array([self.eps], mx.float32))
        return decode_kernel("rda_out")(
            inputs=[q, k, v, g, beta, state, norm_w, gate, eps],
            template=[
                ("InT", q.dtype),
                ("StT", state.dtype),
                ("Dk", hd),
                ("Dv", hd),
                ("Hk", nh),
                ("Hv", nh),
                ("ROWS", rows),
            ],
            grid=(32, hd // rows, batch * nh),
            threadgroup=(32, hd // rows, 1),
            output_shapes=[q.shape, state.shape],
            output_dtypes=[mx.bfloat16, state.dtype],
        )


class MLA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.nope = args.qk_nope_head_dim
        self.extra = args.qk_rope_head_dim
        self.value_dim = args.v_head_dim
        self.latent_dim = args.kv_lora_rank
        self.scale = (self.nope + self.extra) ** -0.5
        self.use_gate = args.mla_use_output_gate
        hidden = args.hidden_size
        self.q_a_proj = nn.Linear(hidden, args.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.RMSNorm(args.q_lora_rank, eps=args.mla_latent_norm_eps)
        self.q_b_proj = nn.Linear(
            args.q_lora_rank, self.num_heads * (self.nope + self.extra), bias=False
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden, self.latent_dim + self.extra, bias=False
        )
        self.kv_a_layernorm = nn.RMSNorm(self.latent_dim, eps=args.mla_latent_norm_eps)
        self.embed_q = MultiLinear(self.nope, self.latent_dim, self.num_heads)
        self.unembed_out = MultiLinear(self.latent_dim, self.value_dim, self.num_heads)
        self.o_proj = nn.Linear(self.num_heads * self.value_dim, hidden, bias=False)
        if self.use_gate:
            self.g_proj = nn.Linear(hidden, self.num_heads * self.value_dim, bias=False)

    def __call__(self, x, mask=None, cache=None):
        batch, length, _ = x.shape
        if self.use_gate:
            qa, kva, gate = _project(
                self, ("q_a_proj", "kv_a_proj_with_mqa", "g_proj"), x
            )
        else:
            qa, kva = _project(self, ("q_a_proj", "kv_a_proj_with_mqa"), x)
        q = self.q_b_proj(self.q_a_layernorm(qa))
        q = q.reshape(batch, length, self.num_heads, self.nope + self.extra).transpose(
            0, 2, 1, 3
        )
        q_nope, q_extra = mx.split(q, [self.nope], axis=-1)
        latent, extra = mx.split(kva, [self.latent_dim], axis=-1)
        latent = self.kv_a_layernorm(latent)[:, None]
        extra = extra[:, None]
        if cache is not None:
            latent, extra = cache.update_and_fetch(latent, extra)
            _bound_cache_graph(cache, ("offset", "left_padding"))
        quantized = hasattr(cache, "bits")
        if quantized:
            pe_scores = mx.quantized_matmul(
                q_extra * self.scale,
                *extra,
                transpose=True,
                group_size=cache.group_size,
                bits=cache.bits,
            )
        else:
            pe_scores = (q_extra * self.scale) @ extra.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(mask, pe_scores, mx.finfo(pe_scores.dtype).min)
        attn_cache = cache
        if length == 1:
            q_nope = self.embed_q(q_nope)
            key = value = latent
        else:
            if quantized:
                latent = mx.dequantize(
                    *latent, group_size=cache.group_size, bits=cache.bits
                )
                attn_cache = None
            key = self.embed_q(latent, transpose=False)
            value = self.unembed_out(latent)
        if quantized and length == 1:
            scores = mx.quantized_matmul(
                q_nope * self.scale,
                *key,
                transpose=True,
                group_size=cache.group_size,
                bits=cache.bits,
            )
            scores = mx.softmax(scores + pe_scores, axis=-1, precise=True)
            out = mx.quantized_matmul(
                scores,
                *value,
                transpose=False,
                group_size=cache.group_size,
                bits=cache.bits,
            )
        else:
            out = scaled_dot_product_attention(
                q_nope, key, value, cache=attn_cache, scale=self.scale, mask=pe_scores
            )
        if length == 1:
            out = self.unembed_out(out)
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        if self.use_gate:
            out = out * mx.sigmoid(gate)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, args, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, args.hidden_size, bias=False)
        self.activation = SiTU(
            args.activation_situ_beta, args.activation_situ_linear_beta
        )

    def __call__(self, x):
        gate, up = _project(self, ("gate_proj", "up_proj"), x)
        return self.down_proj(self.activation(up, gate))


@mx.compile
def _router_scores(x, weight, bias):
    scores = mx.sigmoid(x.astype(mx.float32) @ weight.T)
    return scores, scores + bias.astype(mx.float32)


@mx.compile
def _router_weights(scores, indices, scale):
    weights = mx.take_along_axis(scores, indices, axis=-1)
    return weights / (weights.sum(-1, keepdims=True) + 1e-20) * scale


class Router(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.weight = mx.zeros((args.num_experts, args.hidden_size))
        self.e_score_correction_bias = mx.zeros((args.num_experts,))
        self.top_k = args.num_experts_per_token
        self.renormalize = args.moe_renormalize
        self.scale = args.routed_scaling_factor

    def __call__(self, x):
        w32 = cached_array(self, "_w32", lambda: self.weight.astype(mx.float32))
        scores, ranked = _router_scores(x, w32, self.e_score_correction_bias)
        indices = mx.stop_gradient(mx.argsort(-ranked, axis=-1)[..., : self.top_k])
        if self.renormalize:
            return indices, _router_weights(
                scores,
                indices,
                cached_array(
                    self, "_scale32", lambda: mx.array(self.scale, mx.float32)
                ),
            )
        return indices, mx.take_along_axis(scores, indices, axis=-1) * self.scale

    def route_f32(self, x32):
        """Decode fusion: renormalized routing from the f32 input in one tail kernel (same result as __call__)."""
        w32 = cached_array(self, "_w32", lambda: self.weight.astype(mx.float32))
        bias = cached_array(
            self, "_bias32", lambda: self.e_score_correction_bias.astype(mx.float32)
        )
        scale = cached_array(
            self, "_scale1", lambda: mx.array([self.scale], mx.float32)
        )
        logits = x32 @ w32.T
        e = logits.shape[-1]
        rows = logits.size // e
        indices, weights = decode_kernel("router")(
            inputs=[logits.reshape(rows, e), bias, scale],
            template=[("N_EXPERTS", e), ("TOP_K", self.top_k)],
            grid=(rows * e, 1, 1),
            threadgroup=(e, 1, 1),
            output_shapes=[(rows, self.top_k)] * 2,
            output_dtypes=[mx.uint32, mx.float32],
        )
        shape = (*x32.shape[:-1], self.top_k)
        return indices.reshape(shape), weights.reshape(shape)


@mx.compile
def _moe_combine(out, weights, shared):
    return (out.astype(mx.float32) * weights[..., None]).sum(-2).astype(
        shared.dtype
    ) + shared


@mx.compile
def _moe_sum(out, weights, like):
    return (out.astype(mx.float32) * weights[..., None]).sum(-2).astype(like.dtype)


class MergedSwitchGLU(nn.Module):
    """SwitchGLU with up|gate in one switch linear (`gate_up`, up rows first)."""

    def __init__(self, input_dims, hidden_dims, num_experts, activation):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.gate_up = SwitchLinear(
            input_dims, 2 * hidden_dims, num_experts, bias=False
        )
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=False)
        self.activation = activation

    def _half(self, x, idx, sort, lo, hi):
        gu = self.gate_up
        if isinstance(gu, QuantizedSwitchLinear):
            return mx.gather_qmm(
                x,
                gu["weight"][:, lo:hi],
                gu["scales"][:, lo:hi],
                None if gu.get("biases") is None else gu["biases"][:, lo:hi],
                rhs_indices=idx,
                transpose=True,
                group_size=gu.group_size,
                bits=gu.bits,
                mode=gu.mode,
                global_scale=gu.global_scale,
                sorted_indices=sort,
            )
        return mx.gather_mm(
            x,
            gu["weight"][:, lo:hi].swapaxes(-1, -2),
            rhs_indices=idx,
            sorted_indices=sort,
        )

    def __call__(self, x, indices):
        decode = x.size // x.shape[-1] <= _DECODE_ROWS
        x = mx.expand_dims(x, (-2, -3))
        indices = mx.stop_gradient(indices)
        sort = indices.size >= 64
        idx, inv_order = indices, None
        if sort:
            x, idx, inv_order = _gather_sort(x, indices)
        h = self.hidden_dims
        if decode:
            x_up, x_gate = mx.split(
                self.gate_up(x, idx, sorted_indices=sort), 2, axis=-1
            )
        else:
            # the matrix kernels tile by output width: keep the shapes of the separate products
            x_up, x_gate = self._half(x, idx, sort, 0, h), self._half(
                x, idx, sort, h, 2 * h
            )
        x = self.down_proj(self.activation(x_up, x_gate), idx, sorted_indices=sort)
        if sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)


class MoE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gate = Router(args)
        glu = MergedSwitchGLU if args.raon_merged_gate_up else SwitchGLU
        self.switch_mlp = glu(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
            activation=SiTU(
                args.activation_situ_beta, args.activation_situ_linear_beta
            ),
        )
        self.shared_experts = (
            MLP(args, args.moe_intermediate_size * args.num_shared_experts)
            if args.num_shared_experts
            else None
        )

    def __call__(self, x):
        indices, weights = self.gate(x)
        out = self.switch_mlp(x, indices)
        if self.shared_experts is not None:
            return _moe_combine(out, weights, self.shared_experts(x))
        return _moe_sum(out, weights, x)

    def fusable(self):
        return (
            self.shared_experts is not None
            and self.gate.renormalize
            and self.gate.weight.shape[0] % 32 == 0
            and self.gate.weight.shape[0] <= 1024
        )

    def parts(self, x, x32):
        """Decode fusion: (routed out, weights, shared out) before the combine, which the next kernel does."""
        indices, weights = self.gate.route_f32(x32)
        return self.switch_mlp(x, indices), weights, self.shared_experts(x)


class GlobalScaledLinear(nn.Module):
    """NVFP4 weight-only linear from the converted checkpoint: quantized_matmul(x, codes, E4M3 block scales) * FP32 global.
    quantized_matmul has no global_scale argument, so the per-tensor scale is applied to the output (exact).
    """

    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.weight = mx.zeros((output_dims, input_dims // 8), mx.uint32)
        self.scales = mx.zeros((output_dims, input_dims // 16), mx.uint8)
        self.global_scale = mx.array(1.0, mx.float32)
        self.freeze()

    def __call__(self, x):
        y = mx.quantized_matmul(
            x,
            self["weight"],
            self["scales"],
            None,
            transpose=True,
            group_size=16,
            bits=4,
            mode="nvfp4",
        )
        return _scale_cast(y, self["global_scale"])


class DecoderLayer(nn.Module):
    def __init__(self, args, index):
        super().__init__()
        self.is_linear = index + 1 in args.rda_config["rda_layers"]
        self.is_moe = (
            bool(args.num_experts)
            and index >= args.first_k_dense_replace
            and index % args.moe_layer_freq == 0
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.self_attn = RDA(args) if self.is_linear else MLA(args)
        self.self_attn_raon_mhc = MHC(args)
        self.mlp_raon_mhc = MHC(args)
        if self.is_moe:
            self.moe = MoE(args)
        else:
            self.mlp = MLP(args, args.intermediate_size)

    def __call__(self, stream, mask=None, cache=None):
        batch, length, streams, hidden = stream.shape
        rows = stream.reshape(-1, streams, hidden)
        pre, post, residual = self.self_attn_raon_mhc.route(stream)
        x = self.input_layernorm(MHC.aggregate(rows, pre)).reshape(
            batch, length, hidden
        )
        branch = self.self_attn(x, mask, cache).reshape(-1, hidden)
        rows = MHC.mix(rows, branch, post, residual)
        pre, post, residual = self.mlp_raon_mhc.route(rows.reshape(stream.shape))
        x = self.post_attention_layernorm(MHC.aggregate(rows, pre)).reshape(
            batch, length, hidden
        )
        branch = (self.moe(x) if self.is_moe else self.mlp(x)).reshape(
            -1, hidden
        )
        return MHC.mix(rows, branch, post, residual).reshape(stream.shape)


class RaonModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.streams = args.raon_mhc_streams
        self._linear_index = next(
            (i for i, layer in enumerate(self.layers) if layer.is_linear), None
        )
        self._full_index = next(
            (i for i, layer in enumerate(self.layers) if not layer.is_linear), None
        )

    def _masks(self, h, cache):
        linear_mask = (
            create_ssm_mask(h, cache[self._linear_index])
            if self._linear_index is not None
            else None
        )
        full_mask = (
            create_attention_mask(h, cache[self._full_index], return_array=True)
            if self._full_index is not None
            else None
        )
        return linear_mask, full_mask

    def __call__(self, inputs, cache=None):
        h = self.embed_tokens(inputs)
        cache = [None] * len(self.layers) if cache is None else cache
        if len(cache) != len(self.layers):
            raise ValueError("Cache must have one entry per layer")
        linear_mask, full_mask = self._masks(h, cache)
        # the fused kernels take bf16 rows/branch: require bf16 activations end to end (norm weights bf16)
        fused = (
            not self.training
            and self.args.raon_fused_mhc_decode
            and inputs.shape[1] == 1
            and h.dtype == mx.bfloat16
            and self.layers[0].input_layernorm.weight.dtype == mx.bfloat16
            and self.streams == 4
            and mx.default_device() == mx.gpu
            and mx.metal.is_available()
        )
        if fused and self.norm.weight.dtype == mx.bfloat16:
            return self._decode_fusion(h, cache, linear_mask, full_mask)
        if fused:
            return self._decode_fused(h, cache, linear_mask, full_mask)
        stream = mx.repeat(h[:, :, None], self.streams, axis=2)
        for layer, layer_cache in zip(self.layers, cache):
            stream = layer(
                stream, linear_mask if layer.is_linear else full_mask, layer_cache
            )
        return self.norm(stream.astype(mx.float32).mean(axis=2).astype(h.dtype))

    def _decode_fused(self, h, cache, linear_mask, full_mask):
        batch, length, hidden = h.shape
        rows = mx.repeat(h[:, :, None], self.streams, axis=2).reshape(
            -1, self.streams, hidden
        )
        branch = post = res = None
        for layer, lc in zip(self.layers, cache):
            rows, agg, post, res = fused_mhc(
                rows, branch, post, res, layer.self_attn_raon_mhc, branch is not None
            )
            x = layer.input_layernorm(agg).reshape(batch, length, hidden)
            branch = layer.self_attn(
                x, linear_mask if layer.is_linear else full_mask, lc
            ).reshape(-1, hidden)
            rows, agg, post, res = fused_mhc(
                rows, branch, post, res, layer.mlp_raon_mhc, True
            )
            x = layer.post_attention_layernorm(agg).reshape(batch, length, hidden)
            branch = (
                layer.moe(x) if layer.is_moe else layer.mlp(x)
            ).reshape(-1, hidden)
        rows = MHC.mix(rows, branch, post, res)
        stream = rows.reshape(batch, length, self.streams, hidden)
        return self.norm(stream.astype(mx.float32).mean(axis=2).astype(h.dtype))

    def _decode_fusion(self, h, cache, linear_mask, full_mask):
        """Fuse norms and MoE combination into the mHC decode kernels."""
        batch, length, hidden = h.shape
        rows = mx.repeat(h[:, :, None], self.streams, axis=2).reshape(
            -1, self.streams, hidden
        )
        branch = post = res = None
        for layer, lc in zip(self.layers, cache):
            rows, x, post, res = fused_mhc_norm(
                rows,
                branch,
                post,
                res,
                layer.self_attn_raon_mhc,
                branch is not None,
                layer.input_layernorm,
            )
            mask = linear_mask if layer.is_linear else full_mask
            branch = layer.self_attn(
                x.reshape(batch, length, hidden), mask, lc
            ).reshape(-1, hidden)
            if layer.is_moe and layer.moe.fusable():
                rows, x, post, res, x32 = fused_mhc_norm(
                    rows,
                    branch,
                    post,
                    res,
                    layer.mlp_raon_mhc,
                    True,
                    layer.post_attention_layernorm,
                    with_f32=True,
                )
                shape = (batch, length, hidden)
                branch = layer.moe.parts(
                    x.reshape(shape), x32.reshape(shape)
                )
            else:
                rows, x, post, res = fused_mhc_norm(
                    rows,
                    branch,
                    post,
                    res,
                    layer.mlp_raon_mhc,
                    True,
                    layer.post_attention_layernorm,
                )
                x = x.reshape(batch, length, hidden)
                branch = (
                    layer.moe(x) if layer.is_moe else layer.mlp(x)
                ).reshape(-1, hidden)
        if isinstance(branch, tuple):
            return fused_final(rows, branch, post, res, self.norm).reshape(
                batch, length, hidden
            )
        rows = MHC.mix(rows, branch, post, res)
        stream = rows.reshape(batch, length, self.streams, hidden)
        return self.norm(stream.astype(mx.float32).mean(axis=2).astype(h.dtype))


class MiniModel(nn.Module):
    def __init__(self, args: MiniModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = RaonModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        for path in args.raon_global_scaled_linears or []:
            self._install_global_scaled(path)

    def _install_global_scaled(self, path):
        parent = self
        parts = path.split(".")
        for p in parts[:-1]:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        lin = getattr(parent, parts[-1])
        if not isinstance(lin, nn.Linear):
            raise ValueError(f"{path} is not a Linear")
        out_dims, in_dims = lin.weight.shape
        setattr(parent, parts[-1], GlobalScaledLinear(in_dims, out_dims))

    def __call__(self, inputs, cache=None):
        out = self.model(inputs, cache)
        return (
            self.model.embed_tokens.as_linear(out)
            if self.args.tie_word_embeddings
            else self.lm_head(out)
        )

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            ArraysCache(2) if layer.is_linear else KVCache() for layer in self.layers
        ]

    def sanitize(self, weights):
        weights = dict(weights)
        args = self.args
        for k, v in weights.items():
            if k.endswith("self_attn.f_a_proj.weight") and v.shape[0] != args.rda_decay_rank:
                raise ValueError(f"{k} has {v.shape[0]} rows but rda_decay_rank is {args.rda_decay_rank}")
        for i in range(
            args.num_hidden_layers,
            args.num_hidden_layers + args.num_nextn_predict_layers,
        ):
            prefix = f"model.layers.{i}."
            weights = {k: v for k, v in weights.items() if not k.startswith(prefix)}
        for i, layer in enumerate(self.layers):
            prefix = f"model.layers.{i}."
            if layer.is_linear:
                convs = [
                    prefix + f"self_attn.{n}_conv1d.weight" for n in ("q", "k", "v")
                ]
                if all(c in weights for c in convs):
                    ws = [weights.pop(c) for c in convs]
                    ws = [w[..., None] if w.ndim == 2 else w for w in ws]
                    weights[prefix + "self_attn.qkv_conv1d.weight"] = mx.concatenate(
                        ws, axis=0
                    )
                key = prefix + "self_attn.qkv_conv1d.weight"
                if key in weights and weights[key].ndim == 2:
                    weights[key] = weights[key][..., None]
            else:
                key = prefix + "self_attn.kv_b_proj.weight"
                if key in weights:
                    w = weights.pop(key).reshape(
                        args.num_attention_heads,
                        args.qk_nope_head_dim + args.v_head_dim,
                        args.kv_lora_rank,
                    )
                    weights[prefix + "self_attn.embed_q.weight"] = w[
                        :, : args.qk_nope_head_dim
                    ].swapaxes(-1, -2)
                    weights[prefix + "self_attn.unembed_out.weight"] = w[
                        :, args.qk_nope_head_dim :
                    ]
            if layer.is_moe:
                moe = prefix + "moe."
                if moe + "experts.0.gate_proj.weight" in weights:
                    for old, new in [
                        ("gate_proj", "gate_proj"),
                        ("up_proj", "up_proj"),
                        ("down_proj", "down_proj"),
                    ]:
                        weights[moe + f"switch_mlp.{new}.weight"] = mx.stack(
                            [
                                weights.pop(moe + f"experts.{e}.{old}.weight")
                                for e in range(args.num_experts)
                            ]
                        )
            if layer.is_moe and args.raon_merged_gate_up:
                sw = prefix + "moe.switch_mlp."
                if sw + "up_proj.weight" in weights:
                    gs_up = weights.pop(sw + "up_proj.global_scale", None)
                    gs_gate = weights.pop(sw + "gate_proj.global_scale", None)
                    if (gs_up is None) != (gs_gate is None) or (
                        gs_up is not None and not mx.array_equal(gs_up, gs_gate)
                    ):
                        raise ValueError(
                            f"{sw}: up and gate need the same global scale to merge"
                        )
                    for name in ("weight", "scales", "biases"):
                        up, gate = sw + f"up_proj.{name}", sw + f"gate_proj.{name}"
                        if up in weights:
                            weights[sw + f"gate_up.{name}"] = mx.concatenate(
                                [weights.pop(up), weights.pop(gate)], axis=1
                            )
                    if gs_up is not None:
                        weights[sw + "gate_up.global_scale"] = gs_up
        if args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def cast_predicate(self):
        return lambda path: not (
            path.endswith(
                (
                    "A_log",
                    "dt_bias",
                    "e_score_correction_bias",
                    "res.eta",
                    "global_scale",
                )
            )
            or (self.args.raon_fp32_export_head and path == "lm_head.weight")
        )

    @property
    def quant_predicate(self):
        return lambda path, module: not (
            path.endswith("moe.gate")
            or isinstance(module, GlobalScaledLinear)
            or (self.args.raon_fp32_export_head and path == "lm_head")
        )


"""Raon-1 nano PLE layout 2 on the existing Raon Mini attention and Metal kernels."""
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn



@dataclass
class ModelArgs(MiniModelArgs):
    hidden_size: int = 1536
    num_hidden_layers: int = 24
    num_attention_heads: int = 32
    intermediate_size: int = 6144
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    num_experts: int | None = None
    num_nextn_predict_layers: int = 0
    q_lora_rank: int = 512
    tie_word_embeddings: bool = True
    per_layer_embedding_dim: int = 256
    per_layer_embedding_layout_version: int = 2
    ple_norm_backend: str = "quack"
    nano_optimized: bool = False
    nano_decode_optimized: bool = False
    raon_weight_format: str | None = None
    raon_variants: str = 'all'

    def __post_init__(self):
        if self.hidden_act != "silu" or self.num_experts is not None:
            raise ValueError("Raon-1 nano requires dense SiLU FFNs")
        if self.per_layer_embedding_layout_version != 2:
            raise ValueError("Only post-FFN PLE layout 2 is supported")
        if self.ple_norm_backend not in ("quack", "eager"):
            raise ValueError("Unknown PLE normalization backend")
        if not self.mla_use_nope or self.attn_res_block_size is not None:
            raise ValueError("Raon-1 nano requires NoPE MLA and mHC")
        if self.rda_config is None:
            raise ValueError("Explicit RDA/MLA layer partition is required")
        rda, mla = self.rda_config["rda_layers"], self.rda_config["mla_layers"]
        if sorted(rda + mla) != list(range(1, self.num_hidden_layers + 1)):
            raise ValueError("RDA and MLA must partition all layers")

    @classmethod
    def from_hf(cls, config):
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in config.items() if k in known})


class PleNorm(nn.Module):
    def __init__(self, width, args):
        super().__init__()
        self.weight = mx.ones((width,))
        self.eps = args.rms_norm_eps
        self.backend = args.ple_norm_backend

    def __call__(self, x):
        xf = x.astype(mx.float32)
        norm = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + self.eps)
        if self.backend == "quack":
            return (norm * self.weight.astype(mx.float32)).astype(x.dtype)
        return self.weight * norm.astype(x.dtype)


class PerLayerInputs(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layers, self.width = args.num_hidden_layers, args.per_layer_embedding_dim
        self.table = nn.Embedding(args.vocab_size, self.layers * self.width)
        self.embedding_projection = nn.Linear(args.hidden_size, self.layers * self.width, bias=False)
        self.projection_norm = PleNorm(self.width, args)

    def __call__(self, tokens, embeddings):
        shape = (*tokens.shape, self.layers, self.width)
        projected = self.projection_norm(self.embedding_projection(embeddings).reshape(shape))
        rows = self.table(tokens).reshape(shape) * math.sqrt(self.width)
        return (projected + rows) * (1.0 / math.sqrt(2.0))


class PerLayerInputGate(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gate = nn.Linear(args.hidden_size, args.per_layer_embedding_dim, bias=False)
        self.projection = nn.Linear(args.per_layer_embedding_dim, args.hidden_size, bias=False)
        self.output_norm = PleNorm(args.hidden_size, args)

    def __call__(self, streams, layer_input):
        hidden = streams.astype(mx.float32).mean(axis=2).astype(streams.dtype)
        gate = self.gate(hidden)
        activated = nn.gelu_approx(gate.astype(mx.float32)).astype(gate.dtype)
        delta = self.output_norm(self.projection(activated * layer_input))
        return streams + delta.astype(streams.dtype)[:, :, None, :]


class SiLUFFN(nn.Module):
    def __init__(self, args):
        super().__init__()
        h, inner = args.hidden_size, args.intermediate_size
        self.gate_proj = nn.Linear(h, inner, bias=False)
        self.up_proj = nn.Linear(h, inner, bias=False)
        self.down_proj = nn.Linear(inner, h, bias=False)

    def __call__(self, x):
        gate, up = _project(self, ("gate_proj", "up_proj"), x)
        return self.down_proj(nn.silu(gate) * up)


class Raon1NanoMLA(MLA):
    """Query-blocked prefill adapted from the existing Raon prefill port."""
    def __init__(self, args):
        super().__init__(args)
        self.optimized = args.nano_optimized

    def __call__(self, x, mask=None, cache=None):
        if not self.optimized or x.shape[1] <= 128 or hasattr(cache, 'bits'):
            return super().__call__(x, mask, cache)
        batch, length, _ = x.shape
        qa, kva, gate = _project(self, ('q_a_proj','kv_a_proj_with_mqa','g_proj'), x)
        q = self.q_b_proj(self.q_a_layernorm(qa)).reshape(
            batch,length,self.num_heads,self.nope+self.extra).transpose(0,2,1,3)
        q_nope,q_extra=mx.split(q,[self.nope],axis=-1)
        latent,extra=mx.split(kva,[self.latent_dim],axis=-1)
        latent,extra=self.kv_a_layernorm(latent)[:,None],extra[:,None]
        if cache is not None:
            latent,extra=cache.update_and_fetch(latent,extra)
            _bound_cache_graph(cache,('offset','left_padding'))
        groups=4 if latent.shape[-2]>=16384 and self.num_heads%4==0 else 1
        heads=self.num_heads//groups
        outputs=[]
        for h0 in range(0,self.num_heads,heads):
            h1=h0+heads
            key=latent@self.embed_q.weight[h0:h1]
            value=latent@self.unembed_out.weight[h0:h1].swapaxes(-1,-2)
            pieces=[]
            starts=list(range(0,length,128))
            if len(starts)>1 and length-starts[-1]<=8:
                starts.pop()
            for index,start in enumerate(starts):
                end=starts[index+1] if index+1<len(starts) else length
                scores=(q_extra[:,h0:h1,start:end]*self.scale)@extra.swapaxes(-1,-2)
                if mask is not None:
                    scores=mx.where(mask[...,start:end,:],scores,mx.finfo(scores.dtype).min)
                pieces.append(mx.fast.scaled_dot_product_attention(
                    q_nope[:,h0:h1,start:end],key,value,scale=self.scale,mask=scores))
                mx.async_eval(pieces[-1])
                if index:
                    mx.eval(pieces[index-1])
            output=mx.concatenate(pieces,axis=2)
            if groups>1:
                mx.eval(output)
            outputs.append(output)
        out=mx.concatenate(outputs,axis=1).transpose(0,2,1,3).reshape(batch,length,-1)
        return self.o_proj(out*mx.sigmoid(gate))


class Raon1NanoBackbone(RaonModel):
    def __init__(self, args):
        super().__init__(args)
        self.per_layer_inputs = PerLayerInputs(args)
        for layer in self.layers:
            layer.mlp = SiLUFFN(args)
            layer.per_layer_input_gate = PerLayerInputGate(args)
            if not layer.is_linear:
                layer.self_attn = Raon1NanoMLA(args)

    @staticmethod
    def decode_layer(layer, streams, mask, state):
        shape=streams.shape
        rows=streams.reshape(-1,shape[-2],shape[-1])
        rows,x,post,residual=fused_mhc_norm(
            rows,None,None,None,layer.self_attn_raon_mhc,False,layer.input_layernorm)
        branch=layer.self_attn(x.reshape(shape[0],shape[1],-1),mask,state).reshape(-1,shape[-1])
        rows,x,post,residual=fused_mhc_norm(
            rows,branch,post,residual,layer.mlp_raon_mhc,True,layer.post_attention_layernorm)
        branch=layer.mlp(x.reshape(shape[0],shape[1],-1)).reshape(-1,shape[-1])
        return MHC.mix(rows,branch,post,residual).reshape(shape)

    def __call__(self, inputs, cache=None):
        h = self.embed_tokens(inputs)
        cache = [None] * len(self.layers) if cache is None else cache
        if len(cache) != len(self.layers):
            raise ValueError("Cache must have one entry per layer")
        linear_mask, full_mask = self._masks(h, cache)
        per_layer = self.per_layer_inputs(inputs, h)
        streams = mx.broadcast_to(h[:, :, None, :], (*h.shape[:2], self.streams, h.shape[-1]))
        fast=(self.args.nano_decode_optimized and inputs.shape[1]==1 and self.streams==4
              and h.dtype==mx.bfloat16 and mx.default_device()==mx.gpu)
        for index, (layer, state) in enumerate(zip(self.layers, cache)):
            mask=linear_mask if layer.is_linear else full_mask
            streams = self.decode_layer(layer,streams,mask,state) if fast else layer(streams,mask,state)
            streams = layer.per_layer_input_gate(streams, per_layer[:, :, index, :])
        return self.norm(streams.astype(mx.float32).mean(axis=2).astype(h.dtype))


class Model(MiniModel):
    def __init__(self, args):
        nn.Module.__init__(self)
        self.args, self.model_type = args, "raon1_nano"
        self.model = Raon1NanoBackbone(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def sanitize(self, weights):
        remapped = {}
        for name, weight in weights.items():
            name = name.replace(".self_attn_mhc.", ".self_attn_raon_mhc.")
            name = name.replace(".mlp_mhc.", ".mlp_raon_mhc.")
            if name.endswith(("per_layer_inputs.table", "per_layer_inputs.embedding_projection",
                              "per_layer_input_gate.gate", "per_layer_input_gate.projection")):
                name += ".weight"
            remapped[name] = weight
        if "lm_head.weight" in remapped and self.args.tie_word_embeddings:
            if not mx.array_equal(remapped["lm_head.weight"], remapped["model.embed_tokens.weight"]):
                raise ValueError("Materialized head differs from tied embedding")
        return super().sanitize(remapped)


def load_model(path, dtype=None):
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    args = ModelArgs.from_hf(config)
    model = Model(args)
    if config.get('raon_weight_format') == 'mlx':
        q = config.get('quantization')
        if q:
            quantize_model(model, q['bits'], q['embedding_bits'], q['group_size'])
    weights = {}
    for shard in sorted(path.glob("*.safetensors")):
        weights.update(mx.load(shard))
    if dtype is not None:
        if config.get('raon_weight_format') == 'mlx':
            raise ValueError('Explicit dtype is supported only for floating HF weights')
        weights={name:value.astype(dtype) for name,value in weights.items()}
    if config.get('raon_weight_format') != 'mlx':
        weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    return model


def quantize_model(model, bits=4, embedding_bits=8, group_size=64):
    def predicate(path, module):
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            return False
        return {'bits':embedding_bits if isinstance(module, nn.Embedding) else bits,
                'group_size':group_size, 'mode':'affine'}
    nn.quantize(model, class_predicate=predicate)


def set_optimized(model, enabled=True, enable_decode=False):
    model.args.nano_optimized=enabled
    model.args.nano_decode_optimized=enabled and enable_decode
    for layer in model.layers:
        if isinstance(layer.self_attn,Raon1NanoMLA):
            layer.self_attn.optimized=enabled


"""Opt-in decode variants for Raon-1 nano MLX (one token, bf16, Metal GPU).

`install(model, variants)` swaps classes on a loaded model; `install(model, 'off')` restores the default path.
Every variant except those in BIT_CHANGING reproduces the default decode logits bit for bit
RAON1_NANO_FUSED=0 forces the default path.
"""
import os
from functools import lru_cache, partial

import mlx.core as mx
import mlx.nn as nn


def _metal_kernel(name, input_names, output_names, source, header='', ensure_row_contiguous=True):
    return mx.fast.metal_kernel(name=name, input_names=input_names, output_names=output_names, header=header,
                                source=source, ensure_row_contiguous=ensure_row_contiguous)


VARIANTS = ('ple', 'route_w', 'mhc_route', 'mhc_agg', 'mhc_mix', 'ple4', 'mhc2', 'pf_mhc', 'pf_gd', 'pf_skiplast', 'pf_pe', 'mla_exact', 'mixmean', 'ple_route', 'pf_grprows', 'rda_fold', 'pf_layer_eval', 'mhc_ptail', 'mhc_ptail2', 'pf_kvcache', 'mla_flash', 'pf_ple')
DEFAULT_ALL = ('ple', 'mhc2', 'mhc_ptail', 'mhc_ptail2', 'mhc_mix', 'ple4', 'mixmean', 'ple_route', 'mla_exact', 'rda_fold', 'pf_mhc', 'pf_gd', 'pf_skiplast', 'pf_pe', 'pf_grprows', 'mla_flash', 'pf_ple')  # 'all': the default bundle
# Parts of 'all' whose decode logits are not bit-identical to the default MLX ops (category II: closer to an fp32
# reference). 'exact' = 'all' without them; 'all,-mla_flash' drops one part.
BIT_CHANGING = {'mla_flash'}
EXACT_MLX = '0.32.4'
_RDA_ON = set()
_MHC_EXACT = set()  # parts that run as exact Metal kernels in decode (set by install)
_EXACT_NAMES = {'mhc_route', 'mhc_agg', 'mhc_mix', 'ple4', 'mhc2', 'pf_mhc', 'pf_gd', 'mla_exact', 'mixmean', 'ple_route', 'pf_grprows', 'rda_fold', 'pf_layer_eval', 'mhc_ptail', 'mhc_ptail2', 'pf_kvcache', 'mla_flash', 'pf_ple'}


def _decode_ok(x):
    return x.shape[:2] == (1, 1) and x.dtype == mx.bfloat16 and mx.default_device() == mx.gpu


def _qlinear(m):
    """(weight, scales, biases, group_size, bits) of an affine quantized linear without bias."""
    if not isinstance(m, nn.QuantizedLinear) or 'bias' in m or m.mode != 'affine':
        raise ValueError('expected an affine quantized linear without bias')
    return m['weight'], m['scales'], m['biases'], m.group_size, m.bits


@lru_cache(maxsize=None)
def _ple_gate_fn(g_group, g_bits, p_group, p_bits, eps):
    # PerLayerInputGate.__call__ and PleNorm (quack backend) with the same operations in the same order, traced once.
    f32 = mx.float32

    def gate(streams, layer_input, gw, gs, gb, pw, ps, pb, nw):
        hidden = streams.astype(f32).mean(axis=2).astype(streams.dtype)
        g = mx.quantized_matmul(hidden, gw, gs, gb, transpose=True, group_size=g_group, bits=g_bits)
        activated = nn.gelu_approx(g.astype(f32)).astype(g.dtype)
        y = mx.quantized_matmul(activated * layer_input, pw, ps, pb, transpose=True, group_size=p_group, bits=p_bits)
        xf = y.astype(f32)
        norm = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
        delta = (norm * nw.astype(f32)).astype(y.dtype)
        return streams + delta.astype(streams.dtype)[:, :, None, :]

    return mx.compile(gate)


# pf_ple: the PLE gate of prompt rows as one mean kernel and two compiled tails instead of about 20 single dispatches.
_PF_MEAN_SRC = r"""
    uint e = thread_position_in_grid.x;
    uint h = e % HH, n = e / HH;
    float t = 0.0f;
    for (int j = 0; j < 4; ++j) t = static_cast<float>(S[(n * 4 + j) * HH + h]) + t;
    float m = t * 0.25f;
    hid[e] = static_cast<bfloat16_t>(m);
"""


@lru_cache(maxsize=None)
def _pf_mean_kernel():
    return _metal_kernel(name='raon1_nano_prompt_ple_mean', input_names=['S'], output_names=['hid'], source=_PF_MEAN_SRC)


@partial(mx.compile, shapeless=True)
def _pf_ple_activate(gate, layer_input):
    return nn.gelu_approx(gate.astype(mx.float32)).astype(gate.dtype) * layer_input


@partial(mx.compile, shapeless=True)
def _pf_ple_tail(prj, streams, weight, eps):
    xf = prj.astype(mx.float32)
    norm = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return streams + (norm * weight).astype(streams.dtype)[:, :, None, :]


def _pf_ple(gate, streams, layer_input):
    batch, length, _, hidden = streams.shape
    hid = _pf_mean_kernel()(inputs=[streams], template=[('HH', hidden)], grid=(batch * length * hidden, 1, 1), threadgroup=(256, 1, 1),
                            output_shapes=[(batch, length, hidden)], output_dtypes=[mx.bfloat16])[0]
    prj = gate.projection(_pf_ple_activate(gate.gate(hid), layer_input))
    return _pf_ple_tail(prj, streams, cached_array(gate.output_norm, '_w32', lambda: gate.output_norm.weight.astype(mx.float32)),
                        cached_array(gate.output_norm, '_pf_eps32', lambda: mx.array(gate.output_norm.eps, mx.float32)))


class FusedPerLayerInputGate(PerLayerInputGate):
    def __call__(self, streams, layer_input):
        if ('pf_ple' in _MHC_EXACT and streams.shape[1] > 64 and streams.shape[2] == 4 and streams.dtype == mx.bfloat16
                and self.output_norm.backend == 'quack' and mx.default_device() == mx.gpu):
            return _pf_ple(self, streams, layer_input)
        if (not _decode_ok(streams) or self.output_norm.backend != 'quack'
                or not isinstance(self.gate, nn.QuantizedLinear)):  # unquantized (BF16) weights: plain MLX layer
            return super().__call__(streams, layer_input)
        if 'ple4' in _MHC_EXACT and _ple_shapes_ok(self, streams, layer_input):
            return _ple4(self, streams, layer_input)
        gw, gs, gb, gg, gbits = _qlinear(self.gate)
        pw, ps, pb, pg, pbits = _qlinear(self.projection)
        fn = _ple_gate_fn(gg, gbits, pg, pbits, self.output_norm.eps)
        return fn(streams, layer_input, gw, gs, gb, pw, ps, pb, self.output_norm.weight)


@mx.compile
def _route_head_bf16(flat, weights, eps):
    # _route_head with the weight cast to bf16 done once at install time instead of every call.
    inv = mx.rsqrt(mx.square(flat.astype(mx.float32)).mean(-1, keepdims=True) + eps)
    return (flat @ weights.T).astype(mx.float32) * inv


class FusedMHC(MHC):
    def route(self, stream):
        if stream.dtype != mx.bfloat16:
            return super().route(stream)
        n = self.num_streams
        flat = stream.reshape(-1, n * stream.shape[-1])
        weights = cached_array(self, '_phi_bf16', lambda: self.phi().astype(mx.bfloat16))
        eps = cached_array(self, '_eps32', lambda: mx.array(self.eps, mx.float32))
        z = _route_head_bf16(flat, weights, eps)
        return _route_tail(
            z, self.alpha_pre, self.b_pre, self.alpha_post, self.b_post, self.res.alpha_z, self.res.b_z, self.res.eta,
            self._helmert, self._chart, self._eye,
            cached_array(self, '_post_factor32', lambda: mx.array(self.post_factor, mx.float32)))


# --- exact mHC decode kernels -----------------------------------------------------------------------------------------
# One decode row (4 streams x 1,536, bf16) with MLX 0.32.4's arithmetic in MLX's order, one threadgroup per kernel:
#   route : f32 square sum of the 6,144-wide row as MLX's row_reduce_looped (1,024 threads x 4, two simd_sums),
#           the 17 x 6,144 bf16 gemv as MLX gemv (TM 4, TN 4, SN 32, BN 8: lanes over K, shuffle-down, then simdgroups
#           in order) rounded to bf16, then _route_head's scale and _route_tail with the small helmert products as fma
#           chains over ascending k.
#   agg   : _aggregate (4-term f32 sum from 0, rounded to bf16) followed by rms_single_row (384 threads x 4 values).
#   mix   : _mix as the released Raon-1 mini prompt kernel (fma chain over the 4 streams, then + post * branch).
# Products that MLX writes to memory before adding are kept out of fma contraction with as_type.
_SIGMOID = """
inline float sigm(float x) { float y = 1 / (1 + metal::precise::exp(metal::abs(x))); return (x < 0) ? y : 1 - y; }
inline float hold(float x) { return as_type<float>(as_type<uint>(x)); }
"""

_ROUTE_SRC = r"""
    // prm: eps, post_factor, 1/n, alpha_pre, b_pre[4], alpha_post, b_post[4], alpha_z, b_z[9], eta, helmert[12], chart[16]
    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    threadgroup float shared_vals[32];
    threadgroup float part[NOUT * 8];
    threadgroup float total_s[1];
    threadgroup float zs[NOUT];
    {   // MLX row_reduce_looped, 1,024 threads, N_READS 4, on the f32 squares
        const int blocks = KK / 4096, extra = KK - blocks * 4096;
        const device bfloat16_t* in = x + lid * 4;
        float rt = 0.0f;
        for (int b = 0; b < blocks; b++) {
            for (int i = 0; i < 4; i++) { float v = static_cast<float>(in[i]); rt = hold(v * v) + rt; }
            in += 4096;
        }
        if ((int)lid * 4 + 4 <= extra) {
            for (int i = 0; i < 4; i++) { float v = static_cast<float>(in[i]); rt = hold(v * v) + rt; }
        } else {
            for (int i = 0; (int)lid * 4 + i < extra; i++) { float v = static_cast<float>(in[i]); rt = hold(v * v) + rt; }
        }
        float t = 0.0f + rt;
        t = simd_sum(t);
        if (lane == 0) shared_vals[sg] = t;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        t = simd_sum(lid < 32 ? shared_vals[lid] : 0.0f);
        if (lid == 0) total_s[0] = t;
    }
    for (uint task = sg; task < NOUT * 8; task += 32) {
        const uint j = task / 8, s = task % 8;
        int bn = (32 * s + lane) * 4;
        float result = 0.0f;
        float v[4];
        for (int i = 0; i < KK / 1024; ++i) {
            for (int t = 0; t < 4; t++) v[t] = static_cast<float>(x[bn + t]);
            const device bfloat16_t* wr = w + (size_t)j * KK + bn;
            for (int t = 0; t < 4; t++) result += wr[t] * v[t];
            bn += 1024;
        }
        for (ushort sn = 16; sn >= 1; sn >>= 1) result += simd_shuffle_down(result, sn);
        if (lane == 0) part[j * 8 + s] = result;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < NOUT) {
        float acc = part[lid * 8];
        for (int s = 1; s < 8; s++) acc += part[lid * 8 + s];
        const float y = static_cast<float>(static_cast<bfloat16_t>(acc));
        const float m = total_s[0] * prm[2];
        const float a = m + prm[0];
        const float inv = metal::precise::rsqrt(a);
        zs[lid] = y * inv;
        dbg[1 + lid] = y;
        if (lid == 0) dbg[0] = total_s[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        const device float* H = prm + 24;
        const device float* C = prm + 36;
        for (int i = 0; i < 4; i++) {
            float m1 = prm[3] * zs[i]; float a1 = m1 + prm[4 + i]; pre[i] = sigm(a1);
            float m2 = prm[8] * zs[4 + i]; float a2 = m2 + prm[9 + i]; float s2 = sigm(a2); post[i] = prm[1] * s2;
        }
        float ch[9];
        for (int k = 0; k < 9; k++) { float m3 = prm[13] * zs[8 + k]; ch[k] = m3 + prm[14 + k]; }
        float e1[12];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 3; j++) {
                float acc = 0.0f;
                for (int k = 0; k < 3; k++) acc = fma(H[i * 3 + k], ch[k * 3 + j], acc);
                e1[i * 3 + j] = acc;
            }
        float e[16];
        float mx = -INFINITY;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) {
                float acc = 0.0f;
                for (int k = 0; k < 3; k++) acc = fma(e1[i * 3 + k], H[j * 3 + k], acc);
                e[i * 4 + j] = acc;
                float ng = -acc; float q = ng / C[i * 4 + j]; mx = metal::max(mx, q);
            }
        const float rho = metal::max(mx, 0.0f);
        const float gamma = sigm(prm[23]);
        const float omg = 1 - gamma;
        const float d = 1 + rho;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) {
                float t2 = omg * (i == j ? 1.0f : 0.0f);
                float q = e[i * 4 + j] / d;
                float phi = C[i * 4 + j] + q;
                float g = gamma * phi;
                res[i * 4 + j] = t2 + g;
            }
    }
"""

_AGG_NORM_SRC = r"""
    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const uint axis_size = HH;
    threadgroup float local_sums[32];
    threadgroup float local_inv[1];
    float tx[4];
    float acc = 0;
    for (int i = 0; i < 4; i++) {
        const uint h = lid * 4 + i;
        float a = 0.0f;
        for (int j = 0; j < 4; j++) a = hold(P[j] * static_cast<float>(X[j * HH + h])) + a;
        tx[i] = static_cast<float>(static_cast<bfloat16_t>(a));
        acc += tx[i] * tx[i];
    }
    acc = simd_sum(acc);
    if (sg == 0) local_sums[lane] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) local_sums[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        acc = simd_sum(local_sums[lane]);
        if (lane == 0) local_inv[0] = metal::precise::rsqrt(acc / axis_size + eps[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < 4; i++) out[lid * 4 + i] = W[lid * 4 + i] * static_cast<bfloat16_t>(tx[i] * local_inv[0]);
"""

_MIX_SRC = r"""
    uint e = thread_position_in_grid.x;
    uint h = e % HH, i = e / HH;
    float acc = 0.0f;
    for (int j = 0; j < 4; ++j) acc = fma(R[i * 4 + j], static_cast<float>(X[j * HH + h]), acc);
    float pb = post[i] * static_cast<float>(Bt[h]);
    out[e] = static_cast<bfloat16_t>(hold(acc) + pb);
"""


_MIXMEAN_SRC = r"""
    // _MIX_SRC for the 4 streams of one h, then the PLE stream mean of the 4 bf16 results (as _PLE4 'mean').
    uint h = thread_position_in_grid.x;
    float t = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < 4; ++j) acc = fma(R[i * 4 + j], static_cast<float>(X[j * HH + h]), acc);
        float pb = post[i] * static_cast<float>(Bt[h]);
        const bfloat16_t o = static_cast<bfloat16_t>(hold(acc) + pb);
        out[i * HH + h] = o;
        t = static_cast<float>(o) + t;
    }
    float m = t * 0.25f;
    hid[h] = static_cast<bfloat16_t>(m);
"""
_DEFER = {}  # id(streams) -> (streams, prj, PleNorm): a PLE gate left for the next layer's route kernel
_HID = {}  # id(layer output) -> PLE stream mean computed by the mix kernel (consumed by the next PLE gate call)


def _mix_mean(rows, branch, post, res):
    hidden = rows.shape[-1]
    out, hid = _kernel('mixmean')(inputs=[res, rows, branch, post], template=[('HH', hidden)], grid=(hidden, 1, 1),
                                  threadgroup=(256, 1, 1), output_shapes=[rows.shape, (hidden,)],
                                  output_dtypes=[mx.bfloat16, mx.bfloat16])
    return out, hid


@lru_cache(maxsize=None)
def _kernel(name):
    src, inputs, outputs = {
        'route': (_ROUTE_SRC, ['x', 'w', 'prm'], ['pre', 'post', 'res', 'dbg']),
        'agg': (_AGG_NORM_SRC, ['P', 'X', 'W', 'eps'], ['out']),
        'mix': (_MIX_SRC, ['R', 'X', 'Bt', 'post'], ['out']),
        'mixmean': (_MIXMEAN_SRC, ['R', 'X', 'Bt', 'post'], ['out', 'hid']),
    }[name]
    return _metal_kernel(name=f'raon1_nano_exact_{name}', input_names=inputs, output_names=outputs,
                                header=_SIGMOID, source=src, ensure_row_contiguous=True)


def _route_params(mhc):
    def make():
        f = lambda a: a.astype(mx.float32).reshape(-1)
        n = mhc.phi().shape[1]
        out = mx.concatenate([mx.array([mhc.eps, mhc.post_factor], mx.float32), mx.array([1.0 / n], mx.float32),
                              f(mhc.alpha_pre), f(mhc.b_pre), f(mhc.alpha_post), f(mhc.b_post), f(mhc.res.alpha_z),
                              f(mhc.res.b_z), f(mhc.res.eta), f(mhc._helmert), f(mhc._chart)])
        if out.size != 52:
            raise ValueError('exact mHC route expects 4 streams')
        return out
    return cached_array(mhc, '_exact_prm', make)


def _route_exact(mhc, rows, debug=False):
    weights = cached_array(mhc, '_phi_bf16', lambda: mhc.phi().astype(mx.bfloat16))
    nout, k = weights.shape
    pre, post, res, dbg = _kernel('route')(
        inputs=[rows.reshape(-1), weights, _route_params(mhc)], template=[('KK', k), ('NOUT', nout)],
        grid=(1024, 1, 1), threadgroup=(1024, 1, 1), output_shapes=[(1, 4), (1, 4), (1, 4, 4), (1 + nout,)],
        output_dtypes=[mx.float32] * 4)
    return (pre, post, res, dbg) if debug else (pre, post, res)


def _agg_norm_exact(rows, pre, norm):
    hidden = rows.shape[-1]
    eps = cached_array(norm, '_eps32', lambda: mx.array([norm.eps], mx.float32))
    return _kernel('agg')(inputs=[pre, rows, norm.weight, eps], template=[('HH', hidden)],
                          grid=(hidden // 4, 1, 1), threadgroup=(hidden // 4, 1, 1),
                          output_shapes=[(1, hidden)], output_dtypes=[mx.bfloat16])[0]


def _mix_exact(rows, branch, post, res):
    hidden = rows.shape[-1]
    return _kernel('mix')(inputs=[res, rows, branch, post], template=[('HH', hidden)], grid=(4 * hidden, 1, 1),
                          threadgroup=(256, 1, 1), output_shapes=[rows.shape], output_dtypes=[mx.bfloat16])[0]

_ROUTE2_SRC = r"""
    // grid: NOUT + 1 threadgroups x 1,024 threads. Threadgroup j < NOUT: gemv row j (MLX's 8 simdgroups over K, then in
    // order); threadgroup NOUT: the square sum. Outputs zr[0..NOUT) = bf16-rounded gemv rows, zr[NOUT] = square sum.
    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const uint tg = threadgroup_position_in_grid.x;
    threadgroup float shared_vals[32];
    threadgroup float part[8];
    if (tg == NOUT) {
    {   // MLX row_reduce_looped, 1,024 threads, N_READS 4, on the f32 squares
        const int blocks = KK / 4096, extra = KK - blocks * 4096;
        const device bfloat16_t* in = x + lid * 4;
        float rt = 0.0f;
        for (int b = 0; b < blocks; b++) {
            for (int i = 0; i < 4; i++) { float v = static_cast<float>(in[i]); rt = hold(v * v) + rt; }
            in += 4096;
        }
        if ((int)lid * 4 + 4 <= extra) {
            for (int i = 0; i < 4; i++) { float v = static_cast<float>(in[i]); rt = hold(v * v) + rt; }
        } else {
            for (int i = 0; (int)lid * 4 + i < extra; i++) { float v = static_cast<float>(in[i]); rt = hold(v * v) + rt; }
        }
        float t = 0.0f + rt;
        t = simd_sum(t);
        if (lane == 0) shared_vals[sg] = t;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        t = simd_sum(lid < 32 ? shared_vals[lid] : 0.0f);
        if (lid == 0) zr[NOUT] = t;
    }
    } else {
    if (sg < 8) {
        const uint j = tg, s = sg;
        int bn = (32 * s + lane) * 4;
        float result = 0.0f;
        float v[4];
        for (int i = 0; i < KK / 1024; ++i) {
            for (int t = 0; t < 4; t++) v[t] = static_cast<float>(x[bn + t]);
            const device bfloat16_t* wr = w + (size_t)j * KK + bn;
            for (int t = 0; t < 4; t++) result += wr[t] * v[t];
            bn += 1024;
        }
        for (ushort sn = 16; sn >= 1; sn >>= 1) result += simd_shuffle_down(result, sn);
        if (lane == 0) part[s] = result;
    }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float acc = part[0];
            for (int s = 1; s < 8; s++) acc += part[s];
            zr[tg] = static_cast<float>(static_cast<bfloat16_t>(acc));
        }
    }
"""

_ROUTE3_SRC = r"""
    // grid: NOUT + 1 threadgroups x 1,024 threads. Threadgroup j < NOUT: gemv row j (MLX's 8 simdgroups over K, then in
    // order); threadgroup NOUT: the square sum. Outputs zr[0..NOUT) = bf16-rounded gemv rows, zr[NOUT] = square sum.

    // Prologue: the previous layer's PleNorm (as _PLE4 'norm': row_reduce_looped with HH / 4 threads) on prj, so the
    // streams this route reads are S + bf16(prj * inv * NW) computed in place (the bf16 add of _PLE4 'norm').
    threadgroup float ple_shared[32];
    threadgroup float ple_inv[1];
    {
        const uint plid = thread_position_in_threadgroup.x;
        const uint plane = thread_index_in_simdgroup, psg = simdgroup_index_in_threadgroup;
        constexpr uint NT = HH / 4;
        if (plid < NT) {
            float rt = 0.0f;
            for (int i = 0; i < 4; i++) { float xf = static_cast<float>(prj[plid * 4 + i]); rt = hold(xf * xf) + rt; }
            float t = 0.0f + rt;
            t = simd_sum(t);
            if (plane == 0) ple_shared[psg] = t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (psg == 0) {
            float t = simd_sum(plane < NT / 32 ? ple_shared[plane] : 0.0f);
            if (plane == 0) {
                const float m = t * (1.0f / HH);
                const float a = m + peps[0];
                ple_inv[0] = metal::precise::rsqrt(a);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float pinv = ple_inv[0];
    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const uint tg = threadgroup_position_in_grid.x;
    threadgroup float shared_vals[32];
    threadgroup float part[8];
    if (tg == NOUT) {
        for (uint i = thread_position_in_threadgroup.x; i < 4 * HH; i += 1024) snew[i] = XV(i);
    {   // MLX row_reduce_looped, 1,024 threads, N_READS 4, on the f32 squares
        const int blocks = KK / 4096, extra = KK - blocks * 4096;
        uint in_i = lid * 4;
        float rt = 0.0f;
        for (int b = 0; b < blocks; b++) {
            for (int i = 0; i < 4; i++) { float v = static_cast<float>(XV(in_i + i)); rt = hold(v * v) + rt; }
            in_i += 4096;
        }
        if ((int)lid * 4 + 4 <= extra) {
            for (int i = 0; i < 4; i++) { float v = static_cast<float>(XV(in_i + i)); rt = hold(v * v) + rt; }
        } else {
            for (int i = 0; (int)lid * 4 + i < extra; i++) { float v = static_cast<float>(XV(in_i + i)); rt = hold(v * v) + rt; }
        }
        float t = 0.0f + rt;
        t = simd_sum(t);
        if (lane == 0) shared_vals[sg] = t;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        t = simd_sum(lid < 32 ? shared_vals[lid] : 0.0f);
        if (lid == 0) zr[NOUT] = t;
    }
    } else {
    if (sg < 8) {
        const uint j = tg, s = sg;
        int bn = (32 * s + lane) * 4;
        float result = 0.0f;
        float v[4];
        for (int i = 0; i < KK / 1024; ++i) {
            for (int t = 0; t < 4; t++) v[t] = static_cast<float>(XV(bn + t));
            const device bfloat16_t* wr = w + (size_t)j * KK + bn;
            for (int t = 0; t < 4; t++) result += wr[t] * v[t];
            bn += 1024;
        }
        for (ushort sn = 16; sn >= 1; sn >>= 1) result += simd_shuffle_down(result, sn);
        if (lane == 0) part[s] = result;
    }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float acc = part[0];
            for (int s = 1; s < 8; s++) acc += part[s];
            zr[tg] = static_cast<float>(static_cast<bfloat16_t>(acc));
        }
    }
"""

_AGG2P_SRC = r"""
    // _route_head scale and _route_tail on thread 0 (from zr), then aggregate + RMSNorm as _AGG_NORM_SRC.
    threadgroup float zz[NOUT];
    threadgroup float P[4];
    {
        const uint lid0 = thread_position_in_threadgroup.x;
        if (lid0 < NOUT) {
            const float m = zr[NOUT] * prm[2];
            const float a = m + prm[0];
            const float inv = metal::precise::rsqrt(a);
            zz[lid0] = zr[lid0] * inv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // the route tail of thread 0 in _AGG2_SRC, element by element on separate threads (same expressions)
        threadgroup float ch_t[9], e1_t[12], e_t[16], q_t[16], rho_g[2];
        const device float* H = prm + 24;
        const device float* C = prm + 36;
        if (lid0 < 4) {
            const int i = lid0;
            float m1 = prm[3] * zz[i]; float a1 = m1 + prm[4 + i]; float pr = sigm(a1); pre[i] = pr; P[i] = pr;
            float m2 = prm[8] * zz[4 + i]; float a2 = m2 + prm[9 + i]; float s2 = sigm(a2); post[i] = prm[1] * s2;
        } else if (lid0 < 13) {
            const int k = lid0 - 4;
            float m3 = prm[13] * zz[8 + k]; ch_t[k] = m3 + prm[14 + k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid0 < 12) {
            const int i = lid0 / 3, j = lid0 % 3;
            float acc = 0.0f;
            for (int k = 0; k < 3; k++) acc = fma(H[i * 3 + k], ch_t[k * 3 + j], acc);
            e1_t[i * 3 + j] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid0 < 16) {
            const int i = lid0 / 4, j = lid0 % 4;
            float acc = 0.0f;
            for (int k = 0; k < 3; k++) acc = fma(e1_t[i * 3 + k], H[j * 3 + k], acc);
            e_t[i * 4 + j] = acc;
            float ng = -acc; q_t[i * 4 + j] = ng / C[i * 4 + j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid0 == 0) {
            float mx = -INFINITY;
            for (int t = 0; t < 16; t++) mx = metal::max(mx, q_t[t]);
            rho_g[0] = metal::max(mx, 0.0f);
            rho_g[1] = sigm(prm[23]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid0 < 16) {
            const int i = lid0 / 4, j = lid0 % 4;
            const float gamma = rho_g[1];
            const float omg = 1 - gamma;
            const float d = 1 + rho_g[0];
            float t2 = omg * (i == j ? 1.0f : 0.0f);
            float q = e_t[i * 4 + j] / d;
            float phi = C[i * 4 + j] + q;
            float g = gamma * phi;
            res[i * 4 + j] = t2 + g;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const uint axis_size = HH;
    threadgroup float local_sums[32];
    threadgroup float local_inv[1];
    float tx[4];
    float acc = 0;
    for (int i = 0; i < 4; i++) {
        const uint h = lid * 4 + i;
        float a = 0.0f;
        for (int j = 0; j < 4; j++) a = hold(P[j] * static_cast<float>(X[j * HH + h])) + a;
        tx[i] = static_cast<float>(static_cast<bfloat16_t>(a));
        acc += tx[i] * tx[i];
    }
    acc = simd_sum(acc);
    if (sg == 0) local_sums[lane] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) local_sums[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        acc = simd_sum(local_sums[lane]);
        if (lane == 0) local_inv[0] = metal::precise::rsqrt(acc / axis_size + eps[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < 4; i++) out[lid * 4 + i] = W[lid * 4 + i] * static_cast<bfloat16_t>(tx[i] * local_inv[0]);
"""

_AGG3_SRC = r"""
    // _route_head scale and _route_tail on thread 0 (from zr), then aggregate + RMSNorm as _AGG_NORM_SRC.
    // _AGG2P_SRC's route tail in one stage: each tail thread recomputes the shared inputs it needs (same expressions,
    // same values), and the 16-way max of -e / chart is a simd_max (max is exact in any order).
    threadgroup float P[4];
    {
        const uint lid0 = thread_position_in_threadgroup.x;
        const device float* H = prm + 24;
        const device float* C = prm + 36;
        const float mt = zr[NOUT] * prm[2];
        const float at = mt + prm[0];
        const float inv = metal::precise::rsqrt(at);
        float qv = -INFINITY, ev = 0.0f;
        if (lid0 < 16) {
            const int i = lid0 / 4, j = lid0 % 4;
            float ch[9];
            for (int k = 0; k < 9; k++) { float z = zr[8 + k] * inv; float m3 = prm[13] * z; ch[k] = m3 + prm[14 + k]; }
            float e1r[3];
            for (int jj = 0; jj < 3; jj++) {
                float acc = 0.0f;
                for (int k = 0; k < 3; k++) acc = fma(H[i * 3 + k], ch[k * 3 + jj], acc);
                e1r[jj] = acc;
            }
            float acc = 0.0f;
            for (int k = 0; k < 3; k++) acc = fma(e1r[k], H[j * 3 + k], acc);
            ev = acc;
            float ng = -acc; qv = ng / C[i * 4 + j];
        }
        const float mxv = simd_max(qv);
        if (lid0 < 16) {
            const int i = lid0 / 4, j = lid0 % 4;
            const float rho = metal::max(mxv, 0.0f);
            const float gamma = sigm(prm[23]);
            const float omg = 1 - gamma;
            const float d = 1 + rho;
            float t2 = omg * (i == j ? 1.0f : 0.0f);
            float q = ev / d;
            float phi = C[i * 4 + j] + q;
            float g = gamma * phi;
            res[i * 4 + j] = t2 + g;
        } else if (lid0 < 20) {
            const int i = lid0 - 16;
            float z1 = zr[i] * inv; float m1 = prm[3] * z1; float a1 = m1 + prm[4 + i]; float pr = sigm(a1); pre[i] = pr; P[i] = pr;
            float z2 = zr[4 + i] * inv; float m2 = prm[8] * z2; float a2 = m2 + prm[9 + i]; float s2 = sigm(a2); post[i] = prm[1] * s2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const uint axis_size = HH;
    threadgroup float local_sums[32];
    threadgroup float local_inv[1];
    float tx[4];
    float acc = 0;
    for (int i = 0; i < 4; i++) {
        const uint h = lid * 4 + i;
        float a = 0.0f;
        for (int j = 0; j < 4; j++) a = hold(P[j] * static_cast<float>(X[j * HH + h])) + a;
        tx[i] = static_cast<float>(static_cast<bfloat16_t>(a));
        acc += tx[i] * tx[i];
    }
    acc = simd_sum(acc);
    if (sg == 0) local_sums[lane] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) local_sums[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        acc = simd_sum(local_sums[lane]);
        if (lane == 0) local_inv[0] = metal::precise::rsqrt(acc / axis_size + eps[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < 4; i++) out[lid * 4 + i] = W[lid * 4 + i] * static_cast<bfloat16_t>(tx[i] * local_inv[0]);
"""

_AGG2_SRC = r"""
    // _route_head scale and _route_tail on thread 0 (from zr), then aggregate + RMSNorm as _AGG_NORM_SRC.
    threadgroup float zz[NOUT];
    threadgroup float P[4];
    {
        const uint lid0 = thread_position_in_threadgroup.x;
        if (lid0 < NOUT) {
            const float m = zr[NOUT] * prm[2];
            const float a = m + prm[0];
            const float inv = metal::precise::rsqrt(a);
            zz[lid0] = zr[lid0] * inv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint lid = lid0;
    if (lid == 0) {
            const device float* H = prm + 24;
            const device float* C = prm + 36;
            for (int i = 0; i < 4; i++) {
                float m1 = prm[3] * zz[i]; float a1 = m1 + prm[4 + i]; pre[i] = sigm(a1); P[i] = pre[i];
                float m2 = prm[8] * zz[4 + i]; float a2 = m2 + prm[9 + i]; float s2 = sigm(a2); post[i] = prm[1] * s2;
            }
            float ch[9];
            for (int k = 0; k < 9; k++) { float m3 = prm[13] * zz[8 + k]; ch[k] = m3 + prm[14 + k]; }
            float e1[12];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 3; j++) {
                    float acc = 0.0f;
                    for (int k = 0; k < 3; k++) acc = fma(H[i * 3 + k], ch[k * 3 + j], acc);
                    e1[i * 3 + j] = acc;
                }
            float e[16];
            float mx = -INFINITY;
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) {
                    float acc = 0.0f;
                    for (int k = 0; k < 3; k++) acc = fma(e1[i * 3 + k], H[j * 3 + k], acc);
                    e[i * 4 + j] = acc;
                    float ng = -acc; float q = ng / C[i * 4 + j]; mx = metal::max(mx, q);
                }
            const float rho = metal::max(mx, 0.0f);
            const float gamma = sigm(prm[23]);
            const float omg = 1 - gamma;
            const float d = 1 + rho;
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) {
                    float t2 = omg * (i == j ? 1.0f : 0.0f);
                    float q = e[i * 4 + j] / d;
                    float phi = C[i * 4 + j] + q;
                    float g = gamma * phi;
                    res[i * 4 + j] = t2 + g;
                }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const uint axis_size = HH;
    threadgroup float local_sums[32];
    threadgroup float local_inv[1];
    float tx[4];
    float acc = 0;
    for (int i = 0; i < 4; i++) {
        const uint h = lid * 4 + i;
        float a = 0.0f;
        for (int j = 0; j < 4; j++) a = hold(P[j] * static_cast<float>(X[j * HH + h])) + a;
        tx[i] = static_cast<float>(static_cast<bfloat16_t>(a));
        acc += tx[i] * tx[i];
    }
    acc = simd_sum(acc);
    if (sg == 0) local_sums[lane] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) local_sums[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        acc = simd_sum(local_sums[lane]);
        if (lane == 0) local_inv[0] = metal::precise::rsqrt(acc / axis_size + eps[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < 4; i++) out[lid * 4 + i] = W[lid * 4 + i] * static_cast<bfloat16_t>(tx[i] * local_inv[0]);
"""


@lru_cache(maxsize=None)
def _kernel2(name):
    src, inputs, outputs = {
        'route': (_ROUTE2_SRC, ['x', 'w'], ['zr']),
        'route3': (_ROUTE3_SRC, ['S', 'prj', 'NW', 'peps', 'w'], ['zr', 'snew']),  # uses _XV_MACRO
        'agg': (_AGG2_SRC, ['zr', 'prm', 'X', 'W', 'eps'], ['out', 'pre', 'post', 'res']),
        'aggp': (_AGG2P_SRC, ['zr', 'prm', 'X', 'W', 'eps'], ['out', 'pre', 'post', 'res']),
        'agg3': (_AGG3_SRC, ['zr', 'prm', 'X', 'W', 'eps'], ['out', 'pre', 'post', 'res']),
    }[name]
    return _metal_kernel(name=f'raon1_nano_exact2_{name}', input_names=inputs, output_names=outputs,
                                header=_SIGMOID + (_XV_MACRO if name == 'route3' else ''), source=src,
                                ensure_row_contiguous=True)


# new stream value at flat index i: S + bf16(prj[h] * inv * NW[h]) with the operations of _PLE4 'norm' (no lambda:
# Metal 3.1 compatibility)
_XV_MACRO = '''
#define XV(i) (S[(i)] + static_cast<bfloat16_t>((static_cast<float>(prj[(i) % HH]) * pinv) * static_cast<float>(NW[(i) % HH])))
'''



def _route_agg2(mhc, rows, norm, deferred=None):
    """mhc2: the route gemv spread over one threadgroup per output row, the route tail folded into aggregate + norm.
    deferred = (prj, PleNorm module) of the previous layer's PLE gate (ple_route): the route kernel also finishes that
    gate and returns the new streams, which aggregate and mix then read."""
    weights = cached_array(mhc, '_phi_bf16', lambda: mhc.phi().astype(mx.bfloat16))
    nout, k = weights.shape
    hidden = rows.shape[-1]
    if deferred is not None:
        prj, pnorm = deferred
        peps = cached_array(pnorm, '_eps32', lambda: mx.array([pnorm.eps], mx.float32))
        zr, rows = _kernel2('route3')(inputs=[rows.reshape(-1), prj, pnorm.weight, peps, weights],
                                      template=[('KK', k), ('NOUT', nout), ('HH', hidden)],
                                      grid=(1024 * (nout + 1), 1, 1), threadgroup=(1024, 1, 1),
                                      output_shapes=[(nout + 1,), rows.shape], output_dtypes=[mx.float32, mx.bfloat16])
    else:
        zr = _kernel2('route')(inputs=[rows.reshape(-1), weights], template=[('KK', k), ('NOUT', nout)],
                               grid=(1024 * (nout + 1), 1, 1), threadgroup=(1024, 1, 1), output_shapes=[(nout + 1,)],
                               output_dtypes=[mx.float32])[0]
    eps = cached_array(norm, '_eps32', lambda: mx.array([norm.eps], mx.float32))
    x, pre, post, res = _kernel2('agg3' if 'mhc_ptail2' in _MHC_EXACT else 'aggp' if 'mhc_ptail' in _MHC_EXACT else 'agg')(
        inputs=[zr, _route_params(mhc), rows, norm.weight, eps], template=[('HH', hidden), ('NOUT', nout)],
        grid=(hidden // 4, 1, 1), threadgroup=(hidden // 4, 1, 1),
        output_shapes=[(1, hidden), (1, 4), (1, 4), (1, 4, 4)], output_dtypes=[mx.bfloat16] + [mx.float32] * 3)
    return x, pre, post, res, rows


# --- prefill mHC kernels (released Raon-1 mini mix/aggregate and route square sum, sources copied verbatim) -------
# From KRAFTON/Raon-1-mini-preview-mlx (Hugging Face, revision c57a141), raon_mlx_model.py; see NOTICE.
# They reproduce MLX's batched [4,4]@[4,H] product, the ascending 4-term aggregate, and
# row_reduce_simple (used by MLX only when there are at least 32 rows). Fewer rows keep the MLX ops.
_ROUTE_SUM_SRC = r"""
    constexpr int N_READS = 4, N_WRITES = 4;
    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const uint nsg = TG / 32;
    const int R = dims[0];
    int out_idx = N_WRITES * (int)threadgroup_position_in_grid.y;
    if (out_idx + N_WRITES > R) out_idx = R - N_WRITES;
    threadgroup float shared_vals[32 * N_WRITES];
    float totals[N_WRITES];
    const device bfloat16_t* inputs[N_WRITES];
    inputs[0] = x + (size_t)out_idx * NN + lid * N_READS;
    for (int i = 1; i < N_WRITES; i++) inputs[i] = inputs[i - 1] + NN;
    for (int j = 0; j < N_WRITES; j++) totals[j] = 0.0f;
    const int blocks = NN / (TG * N_READS);
    const int extra = NN - blocks * (TG * N_READS);
    for (int b = 0; b < blocks; b++) {
        for (int j = 0; j < N_WRITES; j++) {
            for (int i = 0; i < N_READS; i++) {
                const float v = static_cast<float>(inputs[j][i]);
                totals[j] = v * v + totals[j];
            }
            inputs[j] += TG * N_READS;
        }
    }
    const int index = lid * N_READS;
    if (index + N_READS <= extra) {
        for (int j = 0; j < N_WRITES; j++) {
            for (int i = 0; i < N_READS; i++) {
                const float v = static_cast<float>(inputs[j][i]);
                totals[j] = v * v + totals[j];
            }
        }
    } else {
        for (int j = 0; j < N_WRITES; j++) {
            for (int i = 0; index + i < extra; i++) {
                const float v = static_cast<float>(inputs[j][i]);
                totals[j] = v * v + totals[j];
            }
        }
    }
    for (int j = 0; j < N_WRITES; j++) totals[j] = simd_sum(totals[j]);
    if (nsg > 1) {
        if (lane == 0) {
            for (int j = 0; j < N_WRITES; j++) shared_vals[sg * N_WRITES + j] = totals[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < N_WRITES; j++) totals[j] = simd_sum((lid < nsg) ? shared_vals[lid * N_WRITES + j] : 0.0f);
    }
    if (lid == 0) {
        for (int j = 0; j < N_WRITES; j++) out[out_idx + j] = totals[j];
    }
"""

_MHC_MIX_SRC = r"""
    uint e = thread_position_in_grid.x;
    uint h = e % HH, i = (e / HH) % 4, n = e / (HH * 4);
    float acc = 0.0f;
    for (int j = 0; j < 4; ++j) acc = fma(R[(n * 4 + i) * 4 + j], (float)X[(n * 4 + j) * HH + h], acc);
    float pb = post[n * 4 + i] * (float)Bt[n * HH + h];
    out[e] = (bfloat16_t)(as_type<float>(as_type<uint>(acc)) + pb);
"""

_MHC_AGG_SRC = r"""
    uint e = thread_position_in_grid.x;
    uint h = e % HH, n = e / HH;
    float acc = P[n * 4] * (float)X[n * 4 * HH + h];
    for (int j = 1; j < 4; ++j) {
        float t = P[n * 4 + j] * (float)X[(n * 4 + j) * HH + h];
        acc = as_type<float>(as_type<uint>(acc)) + t;
    }
    out[e] = (bfloat16_t)acc;
"""


def _reduce_threadgroup(n):
    """MLX threadgroup_size_from_row_size."""
    if n <= 512:
        return 32
    if n <= 1024:
        return 128
    return min(((n + 3) // 4 + 31) // 32 * 32, 1024)


@mx.compile
def _route_head_tail(total, norm, flat, weights, eps):
    inv = mx.rsqrt(total * norm + eps)
    return (flat @ weights.T).astype(mx.float32) * inv


@lru_cache(maxsize=None)
def _pf_kernel(name):
    src, inputs = {'route_sum': (_ROUTE_SUM_SRC, ['x', 'dims']), 'mix': (_MHC_MIX_SRC, ['R', 'X', 'Bt', 'post']),
                   'aggregate': (_MHC_AGG_SRC, ['P', 'X'])}[name]
    return _metal_kernel(name=f'raon1_nano_prompt_{name}', input_names=inputs, output_names=['out'], source=src)


def _pf_route(mhc, stream):
    """MHC.route for prompt rows: the f32 square sum as one kernel (rows >= 32), the rest as the MLX ops."""
    n = mhc.num_streams
    flat = stream.reshape(-1, n * stream.shape[-1])
    rows, width = flat.shape
    if rows < 32 or flat.dtype != mx.bfloat16:
        return mhc.route(stream)
    tg = _reduce_threadgroup(width)
    dims = mx.array([rows], mx.int32)
    total = _pf_kernel('route_sum')(inputs=[flat, dims], template=[('NN', width), ('TG', tg)],
                                    grid=(tg, (rows + 3) // 4, 1), threadgroup=(tg, 1, 1),
                                    output_shapes=[(rows, 1)], output_dtypes=[mx.float32])[0]
    weights = cached_array(mhc, '_phi_bf16', lambda: mhc.phi().astype(mx.bfloat16))
    norm = cached_array(mhc, '_inv_width', lambda: mx.array(1.0 / width, mx.float32))
    eps = cached_array(mhc, '_eps32', lambda: mx.array(mhc.eps, mx.float32))
    z = _route_head_tail(total, norm, flat, weights, eps)
    return _route_tail(
        z, mhc.alpha_pre, mhc.b_pre, mhc.alpha_post, mhc.b_post, mhc.res.alpha_z, mhc.res.b_z, mhc.res.eta,
        mhc._helmert, mhc._chart, mhc._eye,
        cached_array(mhc, '_post_factor32', lambda: mx.array(mhc.post_factor, mx.float32)))


def _pf_aggregate(rows, pre):
    if rows.shape[0] <= 64 or rows.dtype != mx.bfloat16 or pre.dtype != mx.float32:
        return MHC.aggregate(rows, pre)
    n, _, hidden = rows.shape
    return _pf_kernel('aggregate')(inputs=[pre, rows], template=[('HH', hidden)], grid=(n * hidden, 1, 1),
                                   threadgroup=(256, 1, 1), output_shapes=[(n, hidden)], output_dtypes=[mx.bfloat16])[0]


def _pf_mix(rows, branch, post, res):
    if rows.shape[0] <= 64 or rows.dtype != mx.bfloat16 or branch.dtype != mx.bfloat16:
        return MHC.mix(rows, branch, post, res)
    n, streams, hidden = rows.shape
    return _pf_kernel('mix')(inputs=[res, rows, branch, post], template=[('HH', hidden)],
                             grid=(n * streams * hidden, 1, 1), threadgroup=(256, 1, 1),
                             output_shapes=[rows.shape], output_dtypes=[mx.bfloat16])[0]


def _prefill_layer(layer, stream, mask, cache):
    """DecoderLayer.__call__ with the prompt mHC kernels."""
    batch, length, streams, hidden = stream.shape
    rows = stream.reshape(-1, streams, hidden)
    pre, post, residual = _pf_route(layer.self_attn_raon_mhc, stream)
    x = layer.input_layernorm(_pf_aggregate(rows, pre)).reshape(batch, length, hidden)
    branch = layer.self_attn(x, mask, cache).reshape(-1, hidden)
    rows = _pf_mix(rows, branch, post, residual)
    pre, post, residual = _pf_route(layer.mlp_raon_mhc, rows.reshape(stream.shape))
    x = layer.post_attention_layernorm(_pf_aggregate(rows, pre)).reshape(batch, length, hidden)
    branch = layer.mlp(x).reshape(-1, hidden)
    return _pf_mix(rows, branch, post, residual).reshape(stream.shape)


class FusedDecoderLayer(DecoderLayer):
    """DecoderLayer.__call__ for one bf16 decode row with the mHC parts in _MHC_EXACT as exact kernels."""

    def __call__(self, stream, mask=None, cache=None):
        batch, length, streams, hidden = stream.shape
        if length > 1 and 'pf_mhc' in _MHC_EXACT and streams == 4 and mx.default_device() == mx.gpu:
            return _prefill_layer(self, stream, mask, cache)
        if (batch, length, streams) != (1, 1, 4) or stream.dtype != mx.bfloat16 or mx.default_device() != mx.gpu:
            return super().__call__(stream, mask, cache)
        rows = stream.reshape(-1, streams, hidden)

        def route(mhc, rows):
            return _route_exact(mhc, rows) if 'mhc_route' in _MHC_EXACT else mhc.route(rows.reshape(stream.shape))

        def agg_norm(rows, pre, norm):
            if 'mhc_agg' in _MHC_EXACT:
                return _agg_norm_exact(rows, pre, norm).reshape(batch, length, hidden)
            return norm(MHC.aggregate(rows, pre)).reshape(batch, length, hidden)

        def mix(rows, branch, post, res):
            return _mix_exact(rows, branch, post, res) if 'mhc_mix' in _MHC_EXACT else MHC.mix(rows, branch, post, res)

        def route_norm(mhc, rows, norm, deferred=None):
            if 'mhc2' in _MHC_EXACT:
                x, pre, post, res, rows = _route_agg2(mhc, rows, norm, deferred)
                return x.reshape(batch, length, hidden), post, res, rows
            pre, post, res = route(mhc, rows)
            return agg_norm(rows, pre, norm), post, res, rows

        deferred = _DEFER.pop(id(stream), None)
        if deferred is not None:
            assert deferred[0] is stream and 'mhc2' in _MHC_EXACT
            deferred = deferred[1:]
        x, post, res, rows = route_norm(self.self_attn_raon_mhc, rows, self.input_layernorm, deferred)
        branch = self.self_attn(x, mask, cache).reshape(-1, hidden)
        rows = mix(rows, branch, post, res)
        x, post, res, rows = route_norm(self.mlp_raon_mhc, rows, self.post_attention_layernorm)
        branch = self.mlp(x).reshape(-1, hidden)
        if 'mixmean' in _MHC_EXACT and 'ple4' in _MHC_EXACT:
            out, hid = _mix_mean(rows, branch, post, res)
            out = out.reshape(stream.shape)
            _HID.clear()
            _HID[id(out)] = (out, hid)
            return out
        return mix(rows, branch, post, res).reshape(stream.shape)


# --- exact PLE gate decode kernel -------------------------------------------------------------------------------------
# PerLayerInputGate for one decode row in one threadgroup (1,024 threads), each step in MLX 0.32.4's order:
# stream mean (4-term f32 sum from 0, x 0.25, bf16), gate = affine qmv_fast (K 1,536), gelu_approx in f32 -> bf16,
# x layer input (bf16), projection = affine qmv (K 256: load_vector_safe/qdot_safe path), PleNorm as row_reduce_looped
# (384 threads x 4) + rsqrt + weight, then the bf16 add to the 4 streams. MLX's own load_vector/qdot helpers are read
# from the installed MLX headers; intermediates go through device scratch outputs so the helpers read them unchanged.
def _mlx_qdot_helpers():
    import pathlib, hashlib
    src = (pathlib.Path(mx.__file__).parent / 'include/mlx/backend/metal/kernels/quantized.h').read_text()
    start = src.index('template <int bits, int wsize = 8>\ninline constexpr short get_pack_factor')
    end = src.index('\ntemplate <', src.index('inline U qdot_safe('))
    text = src[start:end]
    if hashlib.sha256(text.encode()).hexdigest()[:16] != _QDOT_SHA16:
        raise RuntimeError('MLX quantized.h helpers differ from the version the PLE kernel reproduces')
    return 'static constant constexpr const int SIMD_SIZE = 32;\n' + text


_QDOT_SHA16 = '5fc57168cd16218d'  # sha256 prefix of the helper text in MLX 0.32.4

def _ple_shapes_ok(gate, streams, layer_input):
    try:
        g, p = _qlinear(gate.gate), _qlinear(gate.projection)
    except ValueError:
        return False
    hidden, pd = streams.shape[-1], layer_input.shape[-1]
    return (g[3:], p[3:]) == ((64, 4), (64, 4)) and hidden % 512 == 0 and pd % 64 == 0 and pd % 512 != 0


# Released Raon-1 mini prompt delta-rule kernel (copied verbatim from the release file named above the prefill mHC kernels).
# Vector-gate delta rule in mlx-lm's packed layout (its scalar-gate Dk = 128 kernel): 4 lanes per value row, each lane
# keeps the original lanes' 4-term partials and adds them in the butterfly order of simd_sum (levels 1, 2, 4 in the lane,
# 8 and 16 as the row-group shuffles). The gate is read per state element.
_GD_VEC_SRC = r"""
    constexpr int lanes_per_row = 4;
    constexpr int rows_per_simdgroup = 32 / lanes_per_row;
    constexpr int values_per_lane = Dk / lanes_per_row;
    constexpr int partials_per_lane = values_per_lane / 4;
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    auto lane = thread_index_in_simdgroup;
    auto lane_in_row = lane & (lanes_per_row - 1);
    auto dv_idx = thread_position_in_grid.y * rows_per_simdgroup + lane / lanes_per_row;
    auto q_ = q + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
    auto k_ = k + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
    auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk + lane_in_row * values_per_lane;
    auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
    y += (b_idx * T * Hv + hv_idx) * Dv;
    auto i_state = state_in + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
    float state[values_per_lane];
    for (int i = 0; i < values_per_lane; ++i) state[i] = static_cast<float>(i_state[i]);
    auto beta_ = beta + b_idx * T * Hv;
    for (int t = 0; t < T; ++t) {
      float part[partials_per_lane];
      for (int pb = 0; pb < partials_per_lane; ++pb) {
        float acc = 0.0f;
        for (int i = 0; i < 4; ++i) {
          int e = pb * 4 + i;
          state[e] = state[e] * static_cast<float>(g_[e]);
          acc += state[e] * static_cast<float>(k_[e]);
        }
        part[pb] = acc;
      }
      float kv_mem = ((part[0] + part[1]) + (part[2] + part[3])) + ((part[4] + part[5]) + (part[6] + part[7]));
      kv_mem += simd_shuffle_xor(kv_mem, 1);
      kv_mem += simd_shuffle_xor(kv_mem, 2);
      auto delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * static_cast<float>(beta_[hv_idx]);
      for (int pb = 0; pb < partials_per_lane; ++pb) {
        float acc = 0.0f;
        for (int i = 0; i < 4; ++i) {
          int e = pb * 4 + i;
          state[e] = state[e] + static_cast<float>(k_[e]) * delta;
          acc += state[e] * static_cast<float>(q_[e]);
        }
        part[pb] = acc;
      }
      float out = ((part[0] + part[1]) + (part[2] + part[3])) + ((part[4] + part[5]) + (part[6] + part[7]));
      out += simd_shuffle_xor(out, 1);
      out += simd_shuffle_xor(out, 2);
      if (lane_in_row == 0) y[dv_idx] = static_cast<InT>(out);
      q_ += Hk * Dk;
      k_ += Hk * Dk;
      g_ += Hv * Dk;
      v_ += Hv * Dv;
      y += Hv * Dv;
      beta_ += Hv;
    }
    for (int i = 0; i < values_per_lane; ++i) o_state[i] = static_cast<StT>(state[i]);
"""


def _gated_delta_prompt(q, k, v, g, beta, state, mask=None):
    """gated_delta_kernel for prompt rows; the vector-gate Dk = 128 case through the packed kernel (same bits)."""
    from mlx_lm.models.gated_delta import gated_delta_kernel
    batch, length, hk, dk = k.shape
    hv, dv = v.shape[2:]
    if mask is not None or g.ndim != 4 or dk != 128 or dv % 8 or length == 1 or state.dtype != mx.float32:
        return gated_delta_kernel(q, k, v, g, beta, state, mask)
    return _gd_kernel()(inputs=[q, k, v, g, beta, state, length],
                        template=[("InT", q.dtype), ("StT", state.dtype), ("Dk", dk), ("Dv", dv), ("Hk", hk), ("Hv", hv)],
                        grid=(32, dv // 8, batch * hv), threadgroup=(32, 2, 1),
                        output_shapes=[(batch, length, hv, dv), state.shape], output_dtypes=[q.dtype, state.dtype])


@lru_cache(maxsize=None)
def _gd_kernel():
    return _metal_kernel(name='raon1_nano_prompt_gated_delta_vec', input_names=['q', 'k', 'v', 'g', 'beta', 'state_in', 'T'],
                                output_names=['y', 'state_out'], source=_GD_VEC_SRC)


class FusedRDA(RDA):
    """RDA decode with q|k|v taken as one slice of the merged projection instead of split + concatenate (same values);
    prompt rows with the released vector-gate delta kernel (pf_gd)."""
    def __call__(self, x, mask=None, cache=None):
        if x.shape[1] > 1:
            return self._prompt(x, mask, cache) if 'pf_gd' in _MHC_EXACT else super().__call__(x, mask, cache)
        merged = _merged_decode_parts(self, x) if _RDA_ON else None
        if mask is not None or merged is None or not self._fused_decode(x, cache):
            return super().__call__(x, mask, cache)
        y, sizes = merged
        batch, length, _ = x.shape
        shape = (batch, length, self.num_heads, self.head_dim)
        qkv, gp, fa, braw, delta = (y[..., lo:hi] for lo, hi in zip([0] + sizes[2:], sizes[2:] + [y.shape[-1]]))
        a_exp = cached_array(self, "_a_exp", lambda: mx.exp(self.A_log.astype(mx.float32)))
        dt32 = cached_array(self, "_dt32", lambda: self.dt_bias.astype(mx.float32).reshape(self.num_heads, self.head_dim))
        states = cache.cache
        # The prep kernel reads q|k|v at flat positions [0, 3P) of row b; with one row that is the merged product itself,
        # so it is passed whole (no slice copy).
        folded = _rda_fold_prep(self, y.reshape(1, -1), sizes, states[0], dt32, a_exp) if 'rda_fold' in _RDA_ON and batch == 1 else None
        if folded is not None:
            q, k, v, g, beta, conv_state = folded
        else:
            src = y if batch == 1 and states[0] is not None else qkv  # a missing conv state is sized from the input
            q, k, v, g, beta, conv_state = self._fused_prep(src, states[0], braw, delta, self.f_b_proj(fa), dt32, a_exp)
        state = states[1]
        if state is None:
            state = mx.zeros((batch, self.num_heads, self.head_dim, self.head_dim), dtype=mx.float32)
        gated, state = self._fused_out(q, k, v, g, beta, state, gp.reshape(shape))
        cache.cache = [conv_state, state]
        cache.advance(length)
        _bound_cache_graph(cache, ("lengths", "left_padding"))
        return self.o_proj(gated.reshape(batch, length, self.proj_size))


def _rda_prompt(self, x, mask=None, cache=None):
    """RDA.__call__ for prompt rows (the non-fused branch) with _gated_delta_prompt as the recurrence."""
    batch, length, _ = x.shape
    dtype = x.dtype
    if mask is not None:
        x = mx.where(mask[..., None], x, 0)
    states = [None, None] if cache is None else cache.cache
    lengths = None if cache is None else cache.lengths
    qp, kp, vp, gp, fa, braw, delta = _project(self, _RDA_PARTS, x)
    shape = (batch, length, self.num_heads, self.head_dim)
    a_exp = cached_array(self, "_a_exp", lambda: mx.exp(self.A_log.astype(mx.float32)))
    dt32 = cached_array(self, "_dt32", lambda: self.dt_bias.astype(mx.float32).reshape(self.num_heads, self.head_dim))
    qkv, conv_state = self.qkv_conv1d(mx.concatenate([qp, kp, vp], axis=-1), states[0], mask, lengths)
    q, k, v = mx.split(qkv, 3, axis=-1)
    q, k, v = q.reshape(shape), k.reshape(shape), v.reshape(shape)
    q, k = _rda_qk(q, k, cached_array(self, "_inv_sqrt_d", lambda: mx.array(self.head_dim**-0.5, mx.float32)))
    update_dtype = mx.float32 if dtype == mx.float16 else dtype
    q, k, v = q.astype(update_dtype), k.astype(update_dtype), v.astype(update_dtype)
    decay = self.f_b_proj(fa).reshape(shape)
    v, g, beta = _rda_gates(
        braw, delta, v, decay, dt32, a_exp,
        None if self.lower_bound is None else cached_array(self, "_lower32", lambda: mx.array(self.lower_bound, mx.float32)))
    beta = beta.astype(update_dtype)
    state = states[1]
    if state is None:
        state = mx.zeros((batch, self.num_heads, self.head_dim, self.head_dim), dtype=mx.float32)
    use_kernel = (not self.training and mx.default_device() == mx.gpu and mx.metal.is_available()
                  and self.head_dim >= 32 and self.head_dim % 32 == 0)
    if use_kernel:
        out, state = _gated_delta_prompt(q, k, v, g, beta, state, mask)
    else:
        out, state = gated_delta_ops(q, k, v, g, beta, state, mask)
    gated = _rda_out(out, self.o_norm.weight, gp.reshape(shape), self.eps).astype(dtype)
    if cache is not None:
        cache.cache = [conv_state, state]
        cache.advance(length)
        _bound_cache_graph(cache, ("lengths", "left_padding"))
    return self.o_proj(gated.reshape(batch, length, self.proj_size))


FusedRDA._prompt = _rda_prompt


# --- RDA decode front end folded into the prep kernel (rda_fold) --------------------------------------------------------
# The Raon-1 mini prep kernel reads q|k|v, b and w straight from the merged projection
# (batch 1), and computes the decay projection f_b(fa) itself as MLX's affine qmv_quad (K = 128: a quad of 4 lanes per
# row, 32 values each, load_vector + qdot, quad_sum, bf16) for the head's 128 rows, instead of a separate quantized
# matmul and a concatenate. Same arithmetic per element.
def _rda_fold_source():
    src = _RDA_PREP_SOURCE
    prologue = """
    threadgroup bfloat16_t dec_tg[HEAD_DIM];
    {
        // decay rows h * D .. h * D + D - 1: quad q (lanes 4q..4q+3 of the threadgroup) computes rows q, q + 32, ...
        const uint t = thread_position_in_threadgroup.x;
        const uint quad = t / 4, ql = t % 4;
        constexpr int vpt = FB_K / 4;
        constexpr int in_w = FB_K / 8;            // uint32 words per row (4 bits)
        constexpr int in_g = FB_K / 64;           // groups per row
        constexpr int step = 64 / vpt;
        float x_thread[vpt];
        float sum = load_vector<bfloat16_t, float, vpt, 4>(Y + OFF_FA + ql * vpt, x_thread);
        for (uint rr = quad; rr < HEAD_DIM; rr += HEAD_DIM / 4) {
            const uint row = threadgroup_position_in_grid.x * HEAD_DIM + rr;
            const device uint8_t* wl = (const device uint8_t*)(fbw + row * in_w + ql * (vpt / 8));
            const float sc = fbs[row * in_g + ql / step];
            const float bi = fbb[row * in_g + ql / step];
            float result = 0;
            result += qdot<float, vpt, 4>(wl, x_thread, sc, bi, sum);
            result = quad_sum(result);
            if (ql == 0) dec_tg[rr] = static_cast<bfloat16_t>(result);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""
    reps = [("(float)x[b * 3 * P + c]", "(float)Y[c]"), ("= x[b * 3 * P + c];", "= Y[c];"),
            ("(float)braw[b * NH + h]", "(float)Y[OFF_B + h]"), ("(float)delta[b * NH + h]", "(float)Y[OFF_W + h]"),
            ("(float)decay[b * P + h * D + d]", "(float)dec_tg[d]")]
    for old, new in reps:
        if src.count(old) < 1:
            raise RuntimeError(f'prep kernel source changed: {old}')
        src = src.replace(old, new)
    first = src.index("    const uint b = threadgroup_position_in_grid.y;")
    return src[:first] + prologue + src[first:]


@lru_cache(maxsize=None)
def _rda_fold_kernel():
    return _metal_kernel(
        name='raon1_nano_rda_fold_prep', input_names=['Y', 'state', 'w32', 'fbw', 'fbs', 'fbb', 'dt32', 'a_exp', 'fp'],
        output_names=['q_out', 'k_out', 'v_out', 'g_out', 'beta_out', 'new_state'],
        header=_mlx_qdot_helpers(), source=_rda_fold_source(), ensure_row_contiguous=True)


def _rda_fold_prep(module, y, sizes, state, dt32, a_exp):
    """module._fused_prep for batch 1 with f_b and the q|k|v concatenate folded in; None when it does not apply."""
    try:
        fbw, fbs, fbb, fg, fbits = _qlinear(module.f_b_proj)
    except ValueError:
        return None
    nh, hd, conv = module.num_heads, module.head_dim, module.qkv_conv1d
    fb_k = fbw.shape[1] * 8
    if (fg, fbits) != (64, 4) or fb_k != 128 or hd != 128 or state is None or y.shape[0] != 1:
        return None
    if len(sizes) != 6:  # cumulative ends of q, k, v, g, f_a, b (then rda_w)
        return None
    off_fa, off_b, off_w = sizes[3], sizes[4], sizes[5]
    w32 = cached_array(conv, "_w32", lambda: conv.weight.astype(mx.float32))
    fp = cached_array(module, "_prep_fp", lambda: mx.array([hd**-0.5, module.lower_bound], mx.float32))
    q, k, v, g, beta, new_state = _rda_fold_kernel()(
        inputs=[y.reshape(-1), state, w32.reshape(-1), fbw, fbs, fbb, dt32.reshape(-1), a_exp, fp],
        template=[("HEAD_DIM", hd), ("N_HEADS", nh), ("KSIZE", conv.kernel_size), ("FB_K", fb_k), ("OFF_FA", off_fa),
                  ("OFF_B", off_b), ("OFF_W", off_w)],
        grid=(nh * hd, 1, 1), threadgroup=(hd, 1, 1),
        output_shapes=[(1, nh, hd)] * 4 + [(1, nh), state.shape],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16, mx.float32, mx.bfloat16, y.dtype])
    shape = (1, 1, nh, hd)
    return q.reshape(shape), k.reshape(shape), v.reshape(shape), g.reshape(shape), beta.reshape(1, 1, nh), new_state


def _merged_decode_parts(module, x):
    """_project's merged decode product before the split, with the cumulative split sizes, or None."""
    if module.training or x.size // x.shape[-1] > _DECODE_ROWS:
        return None
    merged = _merged_decode(module, _RDA_PARTS)
    if merged is None or merged[3] is not None:
        return None
    w, sc, b, _, gs, bits, mode, sizes = merged
    return mx.quantized_matmul(x, w, sc, b, transpose=True, group_size=gs, bits=bits, mode=mode), list(sizes)

_PLE4_SRC = {
    'mean': r"""
    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    typedef float U;
    // 1. hidden = bf16(mean over streams)
    for (uint h = thread_position_in_grid.x; h < HH; h += HH) {
        float t = 0.0f;
        for (int j = 0; j < 4; j++) t = static_cast<float>(S[j * HH + h]) + t;
        float m = t * 0.25f;
        hid[h] = static_cast<bfloat16_t>(m);
    }
""",
    'gate': r"""
    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    typedef float U;
    // 2. gate: qmv_fast_impl rows (4 per simdgroup task), K = HH
    {
        constexpr int vpt = 16, block = vpt * 32, step = 64 / vpt;
        const int in_w = HH * 4 / 8, in_g = HH / 64;
        for (int task = threadgroup_position_in_grid.x * 2 + sg; task < PD / 4; task += 1 << 30) {
            const int out_row = task * 4;
            const device uint8_t* ws = (const device uint8_t*)gw + out_row * in_w + lane * 2 * 4;
            const device bfloat16_t* sc = gs + out_row * in_g + lane / step;
            const device bfloat16_t* bi = gb + out_row * in_g + lane / step;
            const device bfloat16_t* xv = hid + lane * vpt;
            U x_thread[vpt];
            U result[4] = {0, 0, 0, 0};
            for (int k = 0; k < HH; k += block) {
                U sum = load_vector<bfloat16_t, U, vpt, 4>(xv, x_thread);
                for (int row = 0; row < 4; row++) {
                    U s = sc[row * in_g]; U b = bi[row * in_g];
                    result[row] += qdot<U, vpt, 4>(ws + row * in_w, x_thread, s, b, sum);
                }
                ws += block * 4 / 8; sc += block / 64; bi += block / 64; xv += block;
            }
            for (int row = 0; row < 4; row++) {
                result[row] = simd_sum(result[row]);
                if (lane == 0) {
                    // gelu_approx (f32) -> bf16, times the layer input (bf16)
                    const float g = static_cast<float>(static_cast<bfloat16_t>(result[row]));
                    const float t1 = 0.5f * g;
                    const float p3 = metal::pow(g, 3.0f);
                    const float t2 = 0.044715f * p3;
                    const float t3 = g + t2;
                    const float t4 = 0.7978845608028654f * t3;
                    const float th = metal::precise::tanh(t4);
                    const float t5 = 1.0f + th;
                    const float ge = t1 * t5;
                    const bfloat16_t a = static_cast<bfloat16_t>(ge);
                    const bfloat16_t li = L[out_row + row];
                    act[out_row + row] = a * li;
                }
            }
        }
    }
""",
    'proj': r"""
    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    typedef float U;
    // 3. projection: qmv_impl rows, K = PD (one safe block)
    {
        constexpr int vpt = 8, block = vpt * 32, step = 64 / vpt;
        const int in_w = PD * 4 / 8, in_g = PD / 64;
        for (int task = threadgroup_position_in_grid.x * 2 + sg; task < HH / 4; task += 1 << 30) {
            const int out_row = task * 4;
            const device uint8_t* ws = (const device uint8_t*)pw + out_row * in_w + lane * 1 * 4;
            const device bfloat16_t* sc = ps + out_row * in_g + lane / step;
            const device bfloat16_t* bi = pb + out_row * in_g + lane / step;
            const device bfloat16_t* xv = act + lane * vpt;
            U x_thread[vpt];
            U result[4] = {0, 0, 0, 0};
            int k = 0;
            for (; k < PD - block; k += block) {
                U sum = load_vector<bfloat16_t, U, vpt, 4>(xv, x_thread);
                for (int row = 0; row < 4; row++) {
                    U s = sc[row * in_g]; U b = bi[row * in_g];
                    result[row] += qdot<U, vpt, 4>(ws + row * in_w, x_thread, s, b, sum);
                }
                ws += block * 4 / 8; sc += block / 64; bi += block / 64; xv += block;
            }
            const int remaining = clamp(static_cast<int>(PD - k - lane * vpt), 0, vpt);
            if (remaining > 0) {
                U sum = load_vector_safe<bfloat16_t, U, vpt, 4>(xv, x_thread, remaining);
                for (int row = 0; row < 4; row++) {
                    U s = sc[row * in_g]; U b = bi[row * in_g];
                    result[row] += qdot_safe<U, vpt, 4>(ws + row * in_w, x_thread, s, b, sum, remaining);
                }
            }
            for (int row = 0; row < 4; row++) {
                result[row] = simd_sum(result[row]);
                if (lane == 0) prj[out_row + row] = static_cast<bfloat16_t>(result[row]);
            }
        }
    }
""",
    'norm': r"""
    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    typedef float U;
    threadgroup float shared_vals[32];
    threadgroup float inv_s[1];
    // 4. PleNorm square sum: row_reduce_looped with HH / 4 threads
    constexpr uint NT = HH / 4;
    float xf[4];
    if (lid < NT) {
        float rt = 0.0f;
        for (int i = 0; i < 4; i++) { xf[i] = static_cast<float>(prj[lid * 4 + i]); rt = hold(xf[i] * xf[i]) + rt; }
        float t = 0.0f + rt;
        t = simd_sum(t);
        if (lane == 0) shared_vals[sg] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float t = simd_sum(lane < NT / 32 ? shared_vals[lane] : 0.0f);
        if (lane == 0) {
            const float m = t * (1.0f / HH);
            const float a = m + eps[0];
            inv_s[0] = metal::precise::rsqrt(a);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 5. delta = bf16(norm * weight), streams + delta
    if (lid < NT) {
        for (int i = 0; i < 4; i++) {
            const uint h = lid * 4 + i;
            const float nrm = xf[i] * inv_s[0];
            const float t = nrm * static_cast<float>(NW[h]);
            const bfloat16_t d = static_cast<bfloat16_t>(t);
            for (int j = 0; j < 4; j++) out[j * HH + h] = S[j * HH + h] + d;
        }
    }
""",
}


@lru_cache(maxsize=None)
def _ple4_kernel(name):
    io = {'mean': (['S'], ['hid']), 'gate': (['hid', 'L', 'gw', 'gs', 'gb'], ['act']),
          'proj': (['act', 'pw', 'ps', 'pb'], ['prj']), 'norm': (['S', 'prj', 'NW', 'eps'], ['out'])}[name]
    return _metal_kernel(name=f'raon1_nano_exact_ple_{name}', input_names=io[0], output_names=io[1],
                                header=_SIGMOID + _mlx_qdot_helpers(), source=_PLE4_SRC[name], ensure_row_contiguous=True)


def _ple4(gate, streams, layer_input):
    """The PLE gate in MLX 0.32.4's order as four kernels; the two products spread over many threadgroups like MLX's qmv."""
    gw, gs, gb, _, _ = _qlinear(gate.gate)
    pw, ps, pb, _, _ = _qlinear(gate.projection)
    hidden, pd = streams.shape[-1], layer_input.shape[-1]
    t = [('HH', hidden), ('PD', pd)]
    eps = cached_array(gate.output_norm, '_eps32', lambda: mx.array([gate.output_norm.eps], mx.float32))
    cached = _HID.pop(id(streams), None)
    if cached is not None and cached[0] is streams:
        hid = cached[1]
    else:
        hid = _ple4_kernel('mean')(inputs=[streams], template=t, grid=(hidden, 1, 1), threadgroup=(256, 1, 1),
                                   output_shapes=[(hidden,)], output_dtypes=[mx.bfloat16])[0]
    act = _ple4_kernel('gate')(inputs=[hid, layer_input, gw, gs, gb], template=t, grid=(32 * (pd // 8), 2, 1),
                               threadgroup=(32, 2, 1), output_shapes=[(pd,)], output_dtypes=[mx.bfloat16])[0]
    prj = _ple4_kernel('proj')(inputs=[act, pw, ps, pb], template=t, grid=(32 * (hidden // 8), 2, 1),
                               threadgroup=(32, 2, 1), output_shapes=[(hidden,)], output_dtypes=[mx.bfloat16])[0]
    if 'ple_route' in _MHC_EXACT and 'mhc2' in _MHC_EXACT and gate.__dict__.get('_defer_ok'):
        _DEFER.clear()
        _DEFER[id(streams)] = (streams, prj, gate.output_norm)  # finished by the next layer's route kernel
        return streams
    return _ple4_kernel('norm')(inputs=[streams, prj, gate.output_norm.weight, eps], template=t,
                                grid=(hidden // 4, 1, 1), threadgroup=(hidden // 4, 1, 1),
                                output_shapes=[streams.shape], output_dtypes=[mx.bfloat16])[0]


# --- MLA decode attention (released Raon-1 mini decode attention, copied verbatim from
# KRAFTON/Raon-1-mini-preview-mlx (Hugging Face, revision c57a141), raon_mlx_model.py; see NOTICE;
# only the version gate and the helpers below are local) ------------------------------------------------
import warnings
from mlx_lm.models.base import scaled_dot_product_attention

_EXACT = {}


_DEFERRED = []  # ple_defer tables with rows still to fill; an eval inside a step must fill them first


def _fill_deferred():
    for table in _DEFERRED:
        table.fill()


def _exact_try(name, new, old):
    """new() on the first use is evaluated at once; if the kernel fails to build or run, `name` uses old() from then on."""
    state = _EXACT.get(name)
    if state is None:
        try:
            out = new()
            _fill_deferred()
            mx.eval(out)
            _EXACT[name] = True
            return out
        except Exception as e:
            warnings.warn(f"exact kernel {name} unavailable ({e}); using MLX operations")
            _EXACT[name] = False
            return old()
    return new() if state else old()


def _constant(cache, key, make):
    value = cache.get(key)
    if value is None:
        value = make()
        _fill_deferred()
        mx.eval(value)
        cache[key] = value
    return value


# --- MLA decode attention ---------------------------------------------------------------------------------------------
# mlx_lm's scaled_dot_product_attention with head dim 512 runs (q * scale) @ k^T + mask -> softmax(precise) -> @ v as
# separate MLX ops; MLX's steel GEMM accumulates each output in float 8x8 fragments over ascending 8-wide k chunks (and
# for split-K shapes sums the partitions in order). The kernels below keep that per-element sequence with one simdgroup
# per 8x8 output tile and many more threadgroups. Where MLX would take a GEMM path not reproduced here (gemv, gemv_wide,
# NAX) or the context is longer than _MLA_EXACT_MAX_L, MLX's own ops run.
_MLA_EXACT_MAX_L = 2048

_MMA_HEADER = r"""
#include <metal_simdgroup_matrix>
inline short2 raon_frag_coord(ushort lane) {        // == MLX BaseMMAFrag<T,8,8>::get_coord: (x = col, y = row)
  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
  return short2{fn, fm};
}
"""

_MMA_SRC = r"""
    // v5.  Runtime dims (no per-L recompile): dims = [M, N, K, KSPLIT, KLAST, BKR].
    //  KPART == 1 (regular steel_gemm_fused): the unaligned K tile [KLAST, KLAST + BKR) is done FIRST (zero-padded to
    //    BKR = the kernel's BK), then [0, KLAST) ascending.  KLAST = (K / BKR) * BKR.
    //  KPART > 1 (steel_gemm_splitk, BK 16): partition p = [p*KSPLIT, (p+1)*KSPLIT); the last one continues to K and
    //    then does the zero-padded tile [KLAST, KLAST + 16) at the END; partials summed 0 + C_0 + C_1 + ... (fp32).
    // One threadgroup per 8x8 output tile, one simdgroup per partition; partials via threadgroup memory.
    threadgroup float part_buf[KPART * 64];
    // v6: the dimension that does not depend on L is a compile-time constant (KFIX for q@k^T, NFIX for p@v)
    const int NN = NFIX ? NFIX : dims[1], KK = KFIX ? KFIX : dims[2];
    const int KSPLIT = dims[3], KLAST = dims[4], BKR = dims[5];
    const uint part = simdgroup_index_in_threadgroup;
    const ushort lane = thread_index_in_simdgroup;
    const int tiles_n = (NN + 7) / 8;
    const int t = (int)threadgroup_position_in_grid.x;
    const int tm = t / tiles_n, tn = t % tiles_n;
    const short2 c = raon_frag_coord(lane);
    const int fm = c.y, fn = c.x;
    simdgroup_matrix<float, 8, 8> Cm = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> Am, Bm;
    const int row = tm * 8 + fm;
    const bool n_full = (tn * 8 + 8 <= NN);
    // chunk sequence for this simdgroup: up to two ranges [r0b, r0e) then [r1b, r1e)
    int r0b, r0e, r1b, r1e;
    const bool k_unaligned = (KLAST < KK);
    if (KPART == 1) {
        if (k_unaligned) { r0b = KLAST; r0e = KLAST + BKR; r1b = 0; r1e = KLAST; }
        else { r0b = 0; r0e = KK; r1b = 0; r1e = 0; }
    } else {
        r0b = (int)part * KSPLIT;
        if ((int)part == KPART - 1) { r0e = KLAST; r1b = KLAST; r1e = k_unaligned ? KLAST + 16 : KLAST; }
        else { r0e = r0b + KSPLIT; r1b = 0; r1e = 0; }
    }
    for (int range = 0; range < 2; ++range) {
      const int kb_ = range == 0 ? r0b : r1b, ke_ = range == 0 ? r0e : r1e;
      for (int k0 = kb_; k0 < ke_; k0 += 8) {
        float a0, a1, b0, b1;
        if (k0 + 8 <= KK && n_full) {            // interior chunk: no bounds checks
            if (SCALE_A) {
                a0 = (float)(bfloat16_t)((float)A[row * KK + k0 + fn] * (float)sc[0]);
                a1 = (float)(bfloat16_t)((float)A[row * KK + k0 + fn + 1] * (float)sc[0]);
            } else { a0 = (float)A[row * KK + k0 + fn]; a1 = (float)A[row * KK + k0 + fn + 1]; }
            if (TRANS_B) { b0 = (float)Bsrc[(tn * 8 + fn) * KK + k0 + fm]; b1 = (float)Bsrc[(tn * 8 + fn + 1) * KK + k0 + fm]; }
            else         { b0 = (float)Bsrc[(k0 + fm) * NN + tn * 8 + fn]; b1 = (float)Bsrc[(k0 + fm) * NN + tn * 8 + fn + 1]; }
        } else {                                  // edge chunk: zero-fill outside K / N (== MLX load_safe)
            const int ka0 = k0 + fn, ka1 = k0 + fn + 1, kb = k0 + fm;
            const int nb0 = tn * 8 + fn, nb1 = tn * 8 + fn + 1;
            a0 = 0.0f; a1 = 0.0f; b0 = 0.0f; b1 = 0.0f;
            if (SCALE_A) {
                if (ka0 < KK) a0 = (float)(bfloat16_t)((float)A[row * KK + ka0] * (float)sc[0]);
                if (ka1 < KK) a1 = (float)(bfloat16_t)((float)A[row * KK + ka1] * (float)sc[0]);
            } else {
                if (ka0 < KK) a0 = (float)A[row * KK + ka0];
                if (ka1 < KK) a1 = (float)A[row * KK + ka1];
            }
            if (kb < KK) {
                if (TRANS_B) { if (nb0 < NN) b0 = (float)Bsrc[nb0 * KK + kb]; if (nb1 < NN) b1 = (float)Bsrc[nb1 * KK + kb]; }
                else         { if (nb0 < NN) b0 = (float)Bsrc[kb * NN + nb0]; if (nb1 < NN) b1 = (float)Bsrc[kb * NN + nb1]; }
            }
        }
        Am.thread_elements()[0] = a0; Am.thread_elements()[1] = a1;
        Bm.thread_elements()[0] = b0; Bm.thread_elements()[1] = b1;
        simdgroup_multiply_accumulate(Cm, Am, Bm, Cm);
      }
    }
    float c0, c1;
    if (KPART == 1) {
        c0 = Cm.thread_elements()[0]; c1 = Cm.thread_elements()[1];
    } else {
        part_buf[part * 64 + lane * 2] = Cm.thread_elements()[0];
        part_buf[part * 64 + lane * 2 + 1] = Cm.thread_elements()[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (part != 0) return;
        c0 = 0.0f; c1 = 0.0f;
        for (int p = 0; p < KPART; ++p) { c0 += part_buf[p * 64 + lane * 2]; c1 += part_buf[p * 64 + lane * 2 + 1]; }
    }
    const int col = tn * 8 + fn;
    if (ADD_MASK) {
        if (col < NN) out[row * NN + col] = (bfloat16_t)((float)(bfloat16_t)c0 + (float)mask[row * NN + col]);
        if (col + 1 < NN) out[row * NN + col + 1] = (bfloat16_t)((float)(bfloat16_t)c1 + (float)mask[row * NN + col + 1]);
    } else {
        if (col < NN) out[row * NN + col] = (bfloat16_t)c0;
        if (col + 1 < NN) out[row * NN + col + 1] = (bfloat16_t)c1;
    }
"""

# the same kernel with the interior k chunks loaded UNR at a time before their MMAs (same order of MMAs)
_MMA_SRC_PIPE = r"""
    // v5.  Runtime dims (no per-L recompile): dims = [M, N, K, KSPLIT, KLAST, BKR].
    //  KPART == 1 (regular steel_gemm_fused): the unaligned K tile [KLAST, KLAST + BKR) is done FIRST (zero-padded to
    //    BKR = the kernel's BK), then [0, KLAST) ascending.  KLAST = (K / BKR) * BKR.
    //  KPART > 1 (steel_gemm_splitk, BK 16): partition p = [p*KSPLIT, (p+1)*KSPLIT); the last one continues to K and
    //    then does the zero-padded tile [KLAST, KLAST + 16) at the END; partials summed 0 + C_0 + C_1 + ... (fp32).
    // One threadgroup per 8x8 output tile, one simdgroup per partition; partials via threadgroup memory.
    threadgroup float part_buf[KPART * 64];
    // v6: the dimension that does not depend on L is a compile-time constant (KFIX for q@k^T, NFIX for p@v)
    const int NN = NFIX ? NFIX : dims[1], KK = KFIX ? KFIX : dims[2];
    const int KSPLIT = dims[3], KLAST = dims[4], BKR = dims[5];
    const uint part = simdgroup_index_in_threadgroup;
    const ushort lane = thread_index_in_simdgroup;
    const int tiles_n = (NN + 7) / 8;
    const int t = (int)threadgroup_position_in_grid.x;
    const int tm = t / tiles_n, tn = t % tiles_n;
    const short2 c = raon_frag_coord(lane);
    const int fm = c.y, fn = c.x;
    simdgroup_matrix<float, 8, 8> Cm = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> Am, Bm;
    const int row = tm * 8 + fm;
    const bool n_full = (tn * 8 + 8 <= NN);
    // chunk sequence for this simdgroup: up to two ranges [r0b, r0e) then [r1b, r1e)
    int r0b, r0e, r1b, r1e;
    const bool k_unaligned = (KLAST < KK);
    if (KPART == 1) {
        if (k_unaligned) { r0b = KLAST; r0e = KLAST + BKR; r1b = 0; r1e = KLAST; }
        else { r0b = 0; r0e = KK; r1b = 0; r1e = 0; }
    } else {
        r0b = (int)part * KSPLIT;
        if ((int)part == KPART - 1) { r0e = KLAST; r1b = KLAST; r1e = k_unaligned ? KLAST + 16 : KLAST; }
        else { r0e = r0b + KSPLIT; r1b = 0; r1e = 0; }
    }
    for (int range = 0; range < 2; ++range) {
      const int kb_ = range == 0 ? r0b : r1b, ke_ = range == 0 ? r0e : r1e;
      int k0 = kb_;
      // v8: interior groups of UNR chunks -- all loads first, then the UNR MMAs in the same ascending order
      if (n_full) {
        for (; k0 + 8 * UNR <= ke_ && k0 + 8 * UNR <= KK; k0 += 8 * UNR) {
          float a0[UNR], a1[UNR], b0[UNR], b1[UNR];
          for (int u = 0; u < UNR; ++u) {
            const int kk = k0 + 8 * u;
            if (SCALE_A) {
                a0[u] = (float)(bfloat16_t)((float)A[row * KK + kk + fn] * (float)sc[0]);
                a1[u] = (float)(bfloat16_t)((float)A[row * KK + kk + fn + 1] * (float)sc[0]);
            } else { a0[u] = (float)A[row * KK + kk + fn]; a1[u] = (float)A[row * KK + kk + fn + 1]; }
            if (TRANS_B) { b0[u] = (float)Bsrc[(tn * 8 + fn) * KK + kk + fm]; b1[u] = (float)Bsrc[(tn * 8 + fn + 1) * KK + kk + fm]; }
            else         { b0[u] = (float)Bsrc[(kk + fm) * NN + tn * 8 + fn]; b1[u] = (float)Bsrc[(kk + fm) * NN + tn * 8 + fn + 1]; }
          }
          for (int u = 0; u < UNR; ++u) {
            Am.thread_elements()[0] = a0[u]; Am.thread_elements()[1] = a1[u];
            Bm.thread_elements()[0] = b0[u]; Bm.thread_elements()[1] = b1[u];
            simdgroup_multiply_accumulate(Cm, Am, Bm, Cm);
          }
        }
      }
      for (; k0 < ke_; k0 += 8) {
        float a0, a1, b0, b1;
        if (k0 + 8 <= KK && n_full) {            // interior chunk: no bounds checks
            if (SCALE_A) {
                a0 = (float)(bfloat16_t)((float)A[row * KK + k0 + fn] * (float)sc[0]);
                a1 = (float)(bfloat16_t)((float)A[row * KK + k0 + fn + 1] * (float)sc[0]);
            } else { a0 = (float)A[row * KK + k0 + fn]; a1 = (float)A[row * KK + k0 + fn + 1]; }
            if (TRANS_B) { b0 = (float)Bsrc[(tn * 8 + fn) * KK + k0 + fm]; b1 = (float)Bsrc[(tn * 8 + fn + 1) * KK + k0 + fm]; }
            else         { b0 = (float)Bsrc[(k0 + fm) * NN + tn * 8 + fn]; b1 = (float)Bsrc[(k0 + fm) * NN + tn * 8 + fn + 1]; }
        } else {                                  // edge chunk: zero-fill outside K / N (== MLX load_safe)
            const int ka0 = k0 + fn, ka1 = k0 + fn + 1, kb = k0 + fm;
            const int nb0 = tn * 8 + fn, nb1 = tn * 8 + fn + 1;
            a0 = 0.0f; a1 = 0.0f; b0 = 0.0f; b1 = 0.0f;
            if (SCALE_A) {
                if (ka0 < KK) a0 = (float)(bfloat16_t)((float)A[row * KK + ka0] * (float)sc[0]);
                if (ka1 < KK) a1 = (float)(bfloat16_t)((float)A[row * KK + ka1] * (float)sc[0]);
            } else {
                if (ka0 < KK) a0 = (float)A[row * KK + ka0];
                if (ka1 < KK) a1 = (float)A[row * KK + ka1];
            }
            if (kb < KK) {
                if (TRANS_B) { if (nb0 < NN) b0 = (float)Bsrc[nb0 * KK + kb]; if (nb1 < NN) b1 = (float)Bsrc[nb1 * KK + kb]; }
                else         { if (nb0 < NN) b0 = (float)Bsrc[kb * NN + nb0]; if (nb1 < NN) b1 = (float)Bsrc[kb * NN + nb1]; }
            }
        }
        Am.thread_elements()[0] = a0; Am.thread_elements()[1] = a1;
        Bm.thread_elements()[0] = b0; Bm.thread_elements()[1] = b1;
        simdgroup_multiply_accumulate(Cm, Am, Bm, Cm);
      }
    }
    float c0, c1;
    if (KPART == 1) {
        c0 = Cm.thread_elements()[0]; c1 = Cm.thread_elements()[1];
    } else {
        part_buf[part * 64 + lane * 2] = Cm.thread_elements()[0];
        part_buf[part * 64 + lane * 2 + 1] = Cm.thread_elements()[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (part != 0) return;
        c0 = 0.0f; c1 = 0.0f;
        for (int p = 0; p < KPART; ++p) { c0 += part_buf[p * 64 + lane * 2]; c1 += part_buf[p * 64 + lane * 2 + 1]; }
    }
    const int col = tn * 8 + fn;
    if (ADD_MASK) {
        if (col < NN) out[row * NN + col] = (bfloat16_t)((float)(bfloat16_t)c0 + (float)mask[row * NN + col]);
        if (col + 1 < NN) out[row * NN + col + 1] = (bfloat16_t)((float)(bfloat16_t)c1 + (float)mask[row * NN + col + 1]);
    } else {
        if (col < NN) out[row * NN + col] = (bfloat16_t)c0;
        if (col + 1 < NN) out[row * NN + col + 1] = (bfloat16_t)c1;
    }
"""

_MMA_KERNELS = {}
_MMA_CONST = {}


def _next_pow2(x):
    n = 1
    while n < x:
        n <<= 1
    return n


def _device_arch():
    arch = _MMA_CONST.get("arch")
    if arch is None:
        arch = mx.device_info().get("architecture", "")
        _MMA_CONST["arch"] = arch
    return arch


def _mlx_gemm_plan(m, n, k):
    """MLX 0.32.4 matmul.cpp dispatch for an un-batched (m x k) @ (k x n) bf16 product: ("splitk", parts, k_per_part),
    ("regular", 1, k), or None for a path not reproduced here."""
    arch = _device_arch()
    gen = int("".join(ch for ch in arch if ch.isdigit())[:2] or 0)
    devc = arch[-1:]
    if min(m, n) == 1:
        return None
    if gen >= 15 and m > 1 and n > 1 and k % 4 == 0 and (m + 4) // 5 <= 3:
        return None
    if gen >= 17:
        return None
    tm, tn, tk = (m + 15) // 16, (n + 15) // 16, k // 16
    if tm * tn <= (2048 if devc in ("s", "d") else 1024) and tk >= 8 and k >= max(m, n):
        parts = min(max(2, _next_pow2(tk // (((m + 31) // 32) * ((n + 31) // 32)))), 32)
        return ("splitk", parts, (tk // parts) * 16)
    return ("regular", 1, k)


def _mlx_regular_bk(trans_b, m, n, k):
    devc = _device_arch()[-1:]
    if devc in ("g", "p"):
        return 32 if trans_b else 16
    if devc == "d":
        if m * n >= (1 << 20) and 2 * max(m, n) > k:
            return 16
        return 32 if trans_b else 16
    return 16


def _mma_gemm(a, b, trans_b, scale=None, mask=None, fix=None, pipe=False):
    """a (M, K) bf16, b (N, K) if trans_b else (K, N) bf16 -> (M, N) bf16, or None where MLX would use another path."""
    m, k = a.shape
    n = b.shape[0] if trans_b else b.shape[1]
    plan = _mlx_gemm_plan(m, n, k)
    if plan is None or m % 8:
        return None
    _, parts, split = plan
    if parts == 1:
        bkr = _mlx_regular_bk(trans_b, m, n, k)
        klast = (k // bkr) * bkr
    else:
        bkr, klast = 16, (k // 16) * 16
    dims = _constant(_MMA_CONST, (m, n, k, split, klast, bkr), lambda: mx.array([m, n, k, split, klast, bkr], mx.int32))
    dummy = _constant(_MMA_CONST, "dummy", lambda: mx.zeros((64,), mx.bfloat16))
    key = (scale is not None, trans_b, mask is not None, pipe)
    kernel = _MMA_KERNELS.get(key)
    if kernel is None:
        kernel = _MMA_KERNELS[key] = _metal_kernel(
            name=f"raon_mla_gemm_{int(key[0])}{int(key[1])}{int(key[2])}{int(key[3])}",
            input_names=["A", "Bsrc", "sc", "mask", "dims"],
            output_names=["out"],
            header=_MMA_HEADER,
            source=_MMA_SRC_PIPE if pipe else _MMA_SRC,
        )
    template = [
        ("KPART", parts),
        ("KFIX", k if fix == "K" else 0),
        ("NFIX", n if fix == "N" else 0),
        ("SCALE_A", int(scale is not None)),
        ("TRANS_B", int(trans_b)),
        ("ADD_MASK", int(mask is not None)),
    ]
    if pipe:
        template.append(("UNR", 8))
    tiles = (m // 8) * ((n + 7) // 8)
    return kernel(
        inputs=[a, b, dummy if scale is None else scale, dummy if mask is None else mask, dims],
        template=template,
        grid=(32 * tiles, parts, 1),
        threadgroup=(32, parts, 1),
        output_shapes=[(m, n)],
        output_dtypes=[mx.bfloat16],
    )[0]


def _mla_exact_attention(q, kv, mask, scale):
    """q (1, H, 1, D), kv (1, 1, L, D), mask (1, H, 1, L) bf16 -> (1, H, 1, D), equal to the MLX ops."""
    _, h, _, d = q.shape
    length = kv.shape[2]
    sc = _constant(_MMA_CONST, ("scale", scale), lambda: mx.array([scale], mx.bfloat16))
    s = _mma_gemm(q.reshape(h, d), kv.reshape(length, d), True, scale=sc, mask=mask.reshape(h, length), fix="K", pipe=True)
    if s is None:
        s = ((q * mx.array(scale, q.dtype)) @ kv.swapaxes(-1, -2) + mask).reshape(h, length)
    p = mx.softmax(s, axis=-1, precise=True)
    o = _mma_gemm(p, kv.reshape(length, d), False, fix="N")
    if o is None:
        o = (p.reshape(1, h, 1, length) @ kv).reshape(h, d)
    return o.reshape(1, h, 1, d)


def _decode_sdpa(queries, keys, values, cache, scale, mask):
    """mlx_lm's scaled_dot_product_attention, through the exact MLA decode kernels where they apply."""
    if (
        mx.__version__ == EXACT_MLX
        and not hasattr(cache, "bits")
        and queries.ndim == 4
        and queries.shape[0] == 1
        and queries.shape[2] == 1
        and keys.shape[1] == 1
        and keys is values
        and queries.shape[-1] == 512
        and keys.shape[-1] == 512
        and queries.dtype == mx.bfloat16
        and keys.dtype == mx.bfloat16
        and isinstance(mask, mx.array)
        and mask.dtype == mx.bfloat16
        and queries.shape[1] % 8 == 0
        and keys.shape[2] <= _MLA_EXACT_MAX_L
        and mask.shape == (1, queries.shape[1], 1, keys.shape[2])
    ):
        return _exact_try(
            "mla_attention",
            lambda: _mla_exact_attention(queries, keys, mask, scale),
            lambda: scaled_dot_product_attention(queries, keys, values, cache=cache, scale=scale, mask=mask),
        )
    return scaled_dot_product_attention(queries, keys, values, cache=cache, scale=scale, mask=mask)


# --- mla_flash (opt-in, changes bits): MLA decode attention with fp32 scores -----------------------------------------
# The default path rounds the rope scores, the scaled query and the summed scores to bf16 between separate MLX ops. Here
# one kernel computes scale * (q_nope . latent + q_rope . rope) as a single 576-wide fp32 dot per (head, key), softmax
# runs in fp32 (MLX precise), and p (rounded to bf16, as the default path's softmax output) @ latent stays MLX's GEMM.
# (A one-pass online-softmax kernel was 3x slower than the default path at 32K keys.)
_MLA_SCORES_SRC = r"""
    // S^T tiles (keys x heads): A = key rows (8 keys x 8 dims, 4-byte loads along a row), B = q^T (8 dims x 8 heads);
    // one simdgroup covers KT key tiles x 4 head tiles.
    constexpr int H = 32, D = 512, R = 64;
    const uint lane = thread_index_in_simdgroup;
    const int L = dims[0];
    const float scale = (float)sc[0];
    const int kt0 = (int)(threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup) * KT;
    if (kt0 * 8 >= L) return;
    const short2 c = raon_frag_coord(lane);
    const int fm = c.y, fn = c.x;
    simdgroup_matrix<float, 8, 8> S[KT][4];
    for (int t = 0; t < KT; ++t) for (int i = 0; i < 4; ++i) S[t][i] = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> Am[KT], Bm;
    for (int k0 = 0; k0 < D + R; k0 += 8) {
        for (int t = 0; t < KT; ++t) {
            const int key = (kt0 + t) * 8 + fm;
            float a0 = 0.0f, a1 = 0.0f;
            if (key < L) {
                if (k0 < D) { a0 = (float)lat[key * D + k0 + fn]; a1 = (float)lat[key * D + k0 + fn + 1]; }
                else { a0 = (float)ext[key * R + k0 - D + fn]; a1 = (float)ext[key * R + k0 - D + fn + 1]; }
            }
            Am[t].thread_elements()[0] = a0; Am[t].thread_elements()[1] = a1;
        }
        for (int i = 0; i < 4; ++i) {
            const int h0 = i * 8 + fn, dk = k0 + fm;
            if (k0 < D) { Bm.thread_elements()[0] = (float)qn[h0 * D + dk]; Bm.thread_elements()[1] = (float)qn[(h0 + 1) * D + dk]; }
            else { Bm.thread_elements()[0] = (float)qr[h0 * R + dk - D]; Bm.thread_elements()[1] = (float)qr[(h0 + 1) * R + dk - D]; }
            for (int t = 0; t < KT; ++t) simdgroup_multiply_accumulate(S[t][i], Am[t], Bm, S[t][i]);
        }
    }
    for (int t = 0; t < KT; ++t) {
        const int key = (kt0 + t) * 8 + fm;
        if (key < L) for (int i = 0; i < 4; ++i) {
            const int h0 = i * 8 + fn;
            out[h0 * L + key] = S[t][i].thread_elements()[0] * scale;
            out[(h0 + 1) * L + key] = S[t][i].thread_elements()[1] * scale;
        }
    }
"""

_MLA_SCORES_KT = 2  # key tiles per simdgroup
_MLA_FLASH_MIN_L = 4096  # shorter contexts keep the exact kernels


@lru_cache(maxsize=None)
def _mla_scores_kernel():
    return _metal_kernel(name='raon_mla_scores_f32', input_names=['qn', 'qr', 'lat', 'ext', 'sc', 'dims'],
                         output_names=['out'], header=_MMA_HEADER, source=_MLA_SCORES_SRC)


def _mla_flash(q_nope, q_rope, latent, extra, scale):
    """q_nope (1, 32, 1, 512), q_rope (1, 32, 1, 64), latent (1, 1, L, 512), extra (1, 1, L, 64), bf16 -> (1, 32, 1, 512)."""
    length = latent.shape[2]
    sc = _constant(_MMA_CONST, ("flash_scale", scale), lambda: mx.array([scale], mx.float32))
    dims = mx.array([length], mx.int32)
    kt = _MLA_SCORES_KT
    s = _mla_scores_kernel()(inputs=[q_nope.reshape(-1), q_rope.reshape(-1), latent.reshape(-1), extra.reshape(-1), sc, dims],
                             template=[('KT', kt)], grid=(128 * ((length + 32 * kt - 1) // (32 * kt)), 1, 1),
                             threadgroup=(128, 1, 1), output_shapes=[(32, length)], output_dtypes=[mx.float32])[0]
    p = mx.softmax(s, axis=-1, precise=True).astype(mx.bfloat16)
    return (p.reshape(1, 32, 1, length) @ latent).astype(mx.bfloat16)


def _mla_decode(self, x, mask=None, cache=None):
    """MLA.__call__ for one decode row (bf16 KV cache) with the exact MLA attention kernels."""
    batch, length, _ = x.shape
    qa, kva, gate = _project(self, ("q_a_proj", "kv_a_proj_with_mqa", "g_proj"), x)
    q = self.q_b_proj(self.q_a_layernorm(qa))
    q = q.reshape(batch, length, self.num_heads, self.nope + self.extra).transpose(0, 2, 1, 3)
    q_nope, q_extra = mx.split(q, [self.nope], axis=-1)
    latent, extra = mx.split(kva, [self.latent_dim], axis=-1)
    latent = self.kv_a_layernorm(latent)[:, None]
    extra = extra[:, None]
    latent, extra = cache.update_and_fetch(latent, extra)
    _bound_cache_graph(cache, ("offset", "left_padding"))
    if ('mla_flash' in _MHC_EXACT and mask is None and latent.shape[2] >= _MLA_FLASH_MIN_L
            and self.num_heads == 32 and latent.shape[-1] == 512 and extra.shape[-1] == 64):
        out = self.unembed_out(_mla_flash(self.embed_q(q_nope), q_extra, latent, extra, self.scale))
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        out = out * mx.sigmoid(gate)
        return self.o_proj(out)
    pe_scores = (q_extra * self.scale) @ extra.swapaxes(-1, -2)
    if mask is not None:
        pe_scores = mx.where(mask, pe_scores, mx.finfo(pe_scores.dtype).min)
    q_nope = self.embed_q(q_nope)
    out = _decode_sdpa(q_nope, latent, latent, cache, self.scale, pe_scores)
    out = self.unembed_out(out)
    out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
    out = out * mx.sigmoid(gate)
    return self.o_proj(out)


class FusedRaon1NanoMLA(Raon1NanoMLA):
    """Raon1NanoMLA.__call__ with prompt options set by install:
    lazy (pf_skiplast, last layer only): no per-block eval calls, so MLX skips the attention when the chunk's output is
        not used (every prompt chunk but the final one), as in Raon-1 mini;
    pe_tail (pf_pe): the rope scores of a block are the full product with only the chunk's own keys recomputed and
        masked (keys before the chunk are visible to every row), instead of a where over [heads, rows, keys].
    Same operations per element in the same order."""
    lazy = False
    pe_tail = False
    exact_decode = False
    group_rows = False
    kv_keep = False

    def _kv_rows(self, cache, latent, length):
        """pf_kvcache: the up-projected keys and values of earlier prompt chunks are kept on the cache (in buffers grown
        4,096 rows at a time) and only the chunk's own rows are projected. Rows of latent @ W do not depend on how many
        rows are multiplied (checked for 1K-32K rows), so the keys and values are the same.
        Costs 16 KB per token and MLA layer until the first decode step."""
        return self._kv_rows_full(cache, latent, length, latent.shape[-2])

    def _kv_rows_full(self, cache, latent, length, total):
        state = cache.__dict__.get('_raon_kv')
        start = state[2] if state is not None and state[2] == total - length else 0
        if start == 0:
            state = None
        new = latent[..., start:, :]
        k_new = new @ self.embed_q.weight
        v_new = new @ self.unembed_out.weight.swapaxes(-1, -2)
        if state is None or state[0].shape[2] < total:
            cap = (total + 4095) // 4096 * 4096
            kb = mx.zeros((1, self.num_heads, cap, k_new.shape[-1]), k_new.dtype)
            vb = mx.zeros((1, self.num_heads, cap, v_new.shape[-1]), v_new.dtype)
            if state is not None:
                kb[:, :, :start] = state[0][:, :, :start]
                vb[:, :, :start] = state[1][:, :, :start]
        else:
            kb, vb = state[0], state[1]
            state = None
            cache.__dict__.pop('_raon_kv', None)  # drop the cache's reference so the update below can be in place
        kb[:, :, start:total] = k_new
        vb[:, :, start:total] = v_new
        cache.__dict__['_raon_kv'] = (kb, vb, total)
        return kb[:, :, :total], vb[:, :, :total]

    def _scores(self, qs, extra, mask, start, end, length):
        pe = qs @ extra.swapaxes(-1, -2)
        if mask is None:
            return pe
        first = extra.shape[-2] - length
        if not self.pe_tail or qs.shape[0] != 1 or first <= 0:
            return mx.where(mask[..., start:end, :], pe, mx.finfo(pe.dtype).min)
        tail = qs @ extra[..., first:, :].swapaxes(-1, -2)
        own = mask[..., start:end, first:]
        pe[..., first:] = mx.where(own, tail, mx.finfo(tail.dtype).min)
        return pe

    def __call__(self, x, mask=None, cache=None):
        if x.shape[1] == 1 and cache is not None:
            cache.__dict__.pop('_raon_kv', None)  # pf_kvcache: prompt done
        if (self.exact_decode and x.shape[:2] == (1, 1) and cache is not None and not hasattr(cache, 'bits')
                and self.use_gate and x.dtype == mx.bfloat16):
            return _mla_decode(self, x, mask, cache)
        if not self.optimized or x.shape[1] <= 128 or hasattr(cache, 'bits'):
            return MLA.__call__(self, x, mask, cache)
        batch, length, _ = x.shape
        qa, kva, gate = _project(self, ('q_a_proj', 'kv_a_proj_with_mqa', 'g_proj'), x)
        q = self.q_b_proj(self.q_a_layernorm(qa)).reshape(
            batch, length, self.num_heads, self.nope + self.extra).transpose(0, 2, 1, 3)
        q_nope, q_extra = mx.split(q, [self.nope], axis=-1)
        latent, extra = mx.split(kva, [self.latent_dim], axis=-1)
        latent, extra = self.kv_a_layernorm(latent)[:, None], extra[:, None]
        if cache is not None:
            latent, extra = cache.update_and_fetch(latent, extra)
            _bound_cache_graph(cache, ('offset', 'left_padding'))
        groups = 4 if latent.shape[-2] >= 16384 and self.num_heads % 4 == 0 else 1
        heads = self.num_heads // groups
        kv_all = self._kv_rows(cache, latent, length) if self.kv_keep and batch == 1 and cache is not None else None
        outputs = []
        for h0 in range(0, self.num_heads, heads):
            h1 = h0 + heads
            if kv_all is not None:
                key, value = kv_all[0][:, h0:h1], kv_all[1][:, h0:h1]
            else:
                key = latent @ self.embed_q.weight[h0:h1]
                value = latent @ self.unembed_out.weight[h0:h1].swapaxes(-1, -2)
            pieces = []
            # pf_grprows: with head groups, query blocks of 128 x groups rows keep the threadgroup count of one call
            # (released Raon-1 mini _PREFILL_GROUP_ROWS); every row's attention is unchanged.
            block = 128 * groups if self.group_rows else 128
            starts = list(range(0, length, block))
            if len(starts) > 1 and length - starts[-1] <= 8:
                starts.pop()
            for index, start in enumerate(starts):
                end = starts[index + 1] if index + 1 < len(starts) else length
                scores = self._scores(q_extra[:, h0:h1, start:end] * self.scale, extra, mask, start, end, length)
                pieces.append(mx.fast.scaled_dot_product_attention(
                    q_nope[:, h0:h1, start:end], key, value, scale=self.scale, mask=scores))
                if not self.lazy:
                    mx.async_eval(pieces[-1])
                    if index:
                        mx.eval(pieces[index - 1])
            output = mx.concatenate(pieces, axis=2)
            if groups > 1 and not self.lazy:
                mx.eval(output)
            outputs.append(output)
        out = mx.concatenate(outputs, axis=1).transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(out * mx.sigmoid(gate))

class FusedModel(Model):
    """Model.__call__ that writes the deferred PLE rows (ple='defer') once the step graph is complete."""
    def __call__(self, inputs, cache=None):
        out = self.model(inputs, cache)
        out = self.model.embed_tokens.as_linear(out) if self.args.tie_word_embeddings else self.lm_head(out)
        _fill_deferred()  # ple_defer: the step graph is complete; write its PLE rows before anyone evaluates it
        return out


class FusedBackbone(Raon1NanoBackbone):
    """Raon1NanoBackbone.__call__ (copied from py) with pf_layer_eval: for prompt chunks the streams are
    evaluated after every layer but the last (whose output a chunk usually discards, see pf_skiplast), so the
    intermediate buffers of one layer are freed before the next layer's graph runs. Same operations and order."""
    def __call__(self, inputs, cache=None):
        h = self.embed_tokens(inputs)
        cache = [None] * len(self.layers) if cache is None else cache
        if len(cache) != len(self.layers):
            raise ValueError("Cache must have one entry per layer")
        linear_mask, full_mask = self._masks(h, cache)
        per_layer = self.per_layer_inputs(inputs, h)
        streams = mx.broadcast_to(h[:, :, None, :], (*h.shape[:2], self.streams, h.shape[-1]))
        fast = (self.args.nano_decode_optimized and inputs.shape[1] == 1 and self.streams == 4
                and h.dtype == mx.bfloat16 and mx.default_device() == mx.gpu)
        layer_eval = 'pf_layer_eval' in _MHC_EXACT and inputs.shape[1] > 1 and not self.training
        last = len(self.layers) - 1
        for index, (layer, state) in enumerate(zip(self.layers, cache)):
            mask = linear_mask if layer.is_linear else full_mask
            streams = self.decode_layer(layer, streams, mask, state) if fast else layer(streams, mask, state)
            streams = layer.per_layer_input_gate(streams, per_layer[:, :, index, :])
            if layer_eval and index < last:
                mx.eval(streams)
        return self.norm(streams.astype(mx.float32).mean(axis=2).astype(h.dtype))


def prefill_chunks(length, low=512, high=2048):
    """Prompt chunk sizes for `length` tokens, the last token left for the first step: half of the tokens done so far, kept
    in [low, high]. MLA re-expands all its latents once per chunk, so long prompts gain from larger chunks, while short ones
    keep the small chunk's lower peak memory. Outputs differ from a fixed chunk in the last bits of bf16 only (gap/quality)."""
    done, chunks = 0, []
    while done < length - 1:
        chunks.append(min(max(low, min(high, done // 2)), length - 1 - done))
        done += chunks[-1]
    return chunks


def prefill(model, ids, cache, low=512, high=2048):
    """Fill `cache` as mlx_lm generate_step does (evaluate the cache state, then clear the buffer cache) with prefill_chunks;
    returns the tokens left for generate_step."""
    done = 0
    for chunk in prefill_chunks(ids.size, low, high):
        model(ids[done:done + chunk][None], cache=cache)
        mx.eval([c.state for c in cache])
        mx.clear_cache()
        done += chunk
    return ids[done:]


def load(path, variants='all', ple='mmap'):
    """Load a Raon-1 nano MLX model with the fused decode/prefill paths installed (the default way to run it).
    ple: 'mmap' (default) keeps the 856 MiB PLE table as no-copy views of the model file's mapping, so only the rows
    that are read become resident and they stay clean; the model file must not be modified while the
    model is in use. Where MLX cannot wrap the mapping without a copy, 'mmap' falls back to 'defer' (rows copied from
    the file each step, decode about 12% slower). 'resident' loads the table into memory as before."""
    from pathlib import Path
    path = Path(path)
    model = load_model(path)
    set_optimized(model, True, enable_decode=False)
    if ple in ('mmap', 'defer'):
        offload_ple_table(model, path / 'model.safetensors', defer=ple == 'defer', mmap=ple == 'mmap')
    elif ple != 'resident':
        raise ValueError(f"ple must be 'mmap', 'defer' or 'resident', not {ple!r}")
    install(model, variants)
    return model


def install(model, variants='off'):
    """Enable the named variants ('off', 'all', or a comma list from VARIANTS) on a loaded Raon-1 nano model."""
    names, drop = set(), set()
    for name in ('' if variants in ('off', None) else variants).replace('+', ',').split(','):
        if name.startswith('-'):  # 'all,-mla_flash': the default set without one part
            drop.add(name[1:])
        elif name:
            names |= set(DEFAULT_ALL) if name == 'all' else set(DEFAULT_ALL) - BIT_CHANGING if name == 'exact' else {name}
    if drop - set(VARIANTS):
        raise ValueError(f'unknown variants {sorted(drop - set(VARIANTS))}')
    names -= drop
    if names - set(VARIANTS):
        raise ValueError(f'unknown variants {sorted(names - set(VARIANTS))}')
    if os.environ.get('RAON1_NANO_FUSED', '1') == '0':
        names = set()
    if mx.__version__ != EXACT_MLX or not mx.metal.is_available():
        names -= _EXACT_NAMES  # the kernels reproduce this MLX version's order only
    for layer in model.layers:
        gate = layer.per_layer_input_gate
        gate.__class__ = FusedPerLayerInputGate if names & {'ple', 'ple4', 'pf_ple'} else PerLayerInputGate
        for mhc in (layer.self_attn_raon_mhc, layer.mlp_raon_mhc):
            mhc.__class__ = FusedMHC if 'route_w' in names else MHC
        if layer.is_linear:
            layer.self_attn.__class__ = FusedRDA if names & {'pf_gd', 'rda_fold'} else RDA
        layer.__class__ = FusedDecoderLayer if names & {'mhc_route', 'mhc_agg', 'mhc_mix', 'mhc2', 'pf_mhc'} else DecoderLayer
    for index, layer in enumerate(model.layers):
        object.__setattr__(layer.per_layer_input_gate, '_defer_ok', index < len(model.layers) - 1)
        attn = layer.self_attn
        if isinstance(attn, Raon1NanoMLA):
            lazy = 'pf_skiplast' in names and index == len(model.layers) - 1
            attn.__class__ = FusedRaon1NanoMLA if lazy or names & {'pf_pe', 'mla_exact', 'pf_grprows', 'pf_kvcache', 'mla_flash'} else Raon1NanoMLA
            if attn.__class__ is FusedRaon1NanoMLA:
                object.__setattr__(attn, 'lazy', lazy)
                object.__setattr__(attn, 'pe_tail', 'pf_pe' in names)
                object.__setattr__(attn, 'exact_decode', bool(names & {'mla_exact', 'mla_flash'}))
                object.__setattr__(attn, 'group_rows', 'pf_grprows' in names)
                object.__setattr__(attn, 'kv_keep', 'pf_kvcache' in names)
    model.__class__ = FusedModel
    model.model.__class__ = FusedBackbone  # same as Raon1NanoBackbone unless pf_layer_eval is on (checked per call)
    _RDA_ON.clear()
    _RDA_ON.update(names & {'rda_fold'})
    _MHC_EXACT.clear()
    _MHC_EXACT.update(names & _EXACT_NAMES)
    return names


# --- memory option: PLE table from the safetensors file -----------------------------------------------------------------
# The per-layer-input table (259,840 x 6,144, 4-bit) is about half the resident weights but is only gathered by token.
# offload_ple_table maps it from the safetensors file (clean file pages, not counted as dirty process memory) and
# gathers rows on the CPU; mx.dequantize then runs on the same rows as nn.QuantizedEmbedding, so the values are equal.
# The gather needs the token values, so a decode step waits for the previous token before building its graph.
class _FileTable:
    def __init__(self, path, prefix, group_size, bits, mode):
        import json, struct
        import numpy as np
        with open(path, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(n))
        def view(name, dtype):
            meta = header[f'{prefix}.{name}']
            start, end = meta['data_offsets']
            return np.memmap(path, dtype=dtype, mode='r', offset=8 + n + start, shape=tuple(meta['shape']))
        self.weight, self.scales, self.biases = view('weight', np.uint32), view('scales', np.uint16), view('biases', np.uint16)
        self.group_size, self.bits, self.mode = group_size, bits, mode
        self.slots, self.turn, self.pending = [], 0, None

    def __call__(self, tokens):
        import numpy as np
        if tokens.size == 1 and self.deferred:
            return self._deferred(tokens)
        ids = np.asarray(tokens)
        bf16 = lambda a: mx.array(np.ascontiguousarray(a[ids])).view(mx.bfloat16)
        return mx.dequantize(mx.array(np.ascontiguousarray(self.weight[ids])), scales=bf16(self.scales),
                             biases=bf16(self.biases), group_size=self.group_size, bits=self.bits, mode=self.mode)

    deferred = False

    def _deferred(self, tokens):
        # ple_file_defer: one-token steps read the rows from evaluated placeholder buffers that fill() writes after
        # the whole step graph is built, so waiting for the token overlaps the previous step's GPU work instead of
        # blocking graph construction. Two buffer sets alternate; when fill() runs for step t+1 the token of step t
        # is evaluated, so step t (which read the other set) is done. Same rows and the same dequantize.
        import numpy as np
        if not self.slots:
            for _ in range(2):
                bufs = [mx.zeros((1, *a.shape[1:]), dtype=t) for a, t in
                        ((self.weight, mx.uint32), (self.scales, mx.uint16), (self.biases, mx.uint16))]
                mx.eval(bufs)
                self.slots.append(bufs)
        bufs = self.slots[self.turn]
        self.turn ^= 1
        self.pending = (tokens, bufs)
        if self not in _DEFERRED:
            _DEFERRED.append(self)
        w, s, b = bufs
        return mx.dequantize(w, scales=s.view(mx.bfloat16), biases=b.view(mx.bfloat16), group_size=self.group_size,
                             bits=self.bits, mode=self.mode).reshape(*tokens.shape, -1)

    def fill(self):
        import numpy as np
        if self.pending is None:
            return
        tokens, bufs = self.pending
        self.pending = None
        i = int(np.asarray(tokens).reshape(-1)[0])
        for buf, src in zip(bufs, (self.weight, self.scales, self.biases)):
            np.array(buf, copy=False)[0] = src[i]


class FilePerLayerInputs(PerLayerInputs):
    def __call__(self, tokens, embeddings):
        import math
        shape = (*tokens.shape, self.layers, self.width)
        projected = self.projection_norm(self.embedding_projection(embeddings).reshape(shape))
        rows = self.__dict__['_file_table'](tokens).reshape(shape) * math.sqrt(self.width)
        return (projected + rows) * (1.0 / math.sqrt(2.0))


def _fill_after_call(self, *args, **kwargs):
    out = super(type(self), self).__call__(*args, **kwargs)
    _fill_deferred()
    return out


def _mmap_tensor(path, header_len, meta, dtype):
    """A tensor of a safetensors file as an mlx array over a read-only file mapping, without a copy: the mapping
    starts on a page boundary (MLX wraps page-aligned host memory as a no-copy Metal buffer, mx.asarray copy=False) and
    the tensor is a slice of it. Pages are read from the file on first touch and stay clean (reclaimable)."""
    import mmap
    import numpy as np
    start, end = meta['data_offsets']
    start += 8 + header_len
    end += 8 + header_len
    page = mmap.PAGESIZE
    base = start // page * page
    length = (end - base + page - 1) // page * page
    with open(path, 'rb') as f:
        size = f.seek(0, 2)
        mm = mmap.mmap(f.fileno(), min(length, size - base), access=mmap.ACCESS_READ, offset=base)
    raw = np.frombuffer(mm, dtype=np.uint8)
    region = mx.asarray(raw, copy=False)
    return region[start - base:end - base].view(dtype).reshape(meta['shape'])


def offload_ple_table(model, path, defer=False, mmap=False):
    """Serve the PLE table from `path` (the model's safetensors file) and drop its device copy. Not reversible.
    mmap=True (ple_mmap): the table stays a QuantizedEmbedding whose arrays are no-copy views of a file mapping, so
    the GPU kernels read it as before (same values) and only touched pages are loaded; otherwise rows are copied from
    the file per call (defer: after the step graph is built)."""
    import gc
    pli = model.model.per_layer_inputs
    table = pli.table
    if not isinstance(table, nn.QuantizedEmbedding):
        raise ValueError('expected a quantized PLE table')
    if mmap:
        import json, struct, warnings
        with open(path, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(n))
        prefix = 'model.per_layer_inputs.table'
        try:
            new = {k: _mmap_tensor(str(path), n, header[f'{prefix}.{k}'], d)
                   for k, d in (('weight', mx.uint32), ('scales', mx.bfloat16), ('biases', mx.bfloat16))}
            for k, v in new.items():
                if v.shape != table[k].shape:
                    raise ValueError(f'PLE table {k} shape {v.shape} != {table[k].shape}')
            mx.eval(list(new.values()))
        except (ValueError, TypeError, RuntimeError) as e:  # no no-copy wrap here (older MLX, no Metal): rows per step
            warnings.warn(f'ple_mmap unavailable ({e}); serving PLE rows from the file per step (ple_defer)')
            return offload_ple_table(model, path, defer=True)
        table.update(new)
        gc.collect()
        mx.clear_cache()
        return
    object.__setattr__(pli, '_file_table', _FileTable(str(path), 'model.per_layer_inputs.table', table.group_size,
                                                      table.bits, table.mode))
    pli.__class__ = FilePerLayerInputs
    if defer:
        pli._file_table.deferred = True
        base = type(model)
        model.__class__ = type('FileFill' + base.__name__, (base,), {'__call__': _fill_after_call})
    del pli['table']
    del table
    gc.collect()
    mx.clear_cache()


# ---- Hugging Face Hub entry points (mlx-lm `model_file`) ----
@classmethod
def _from_dict(cls, params):
    return cls.from_hf(params)


ModelArgs.from_dict = _from_dict


class Model(Model):
    """mlx-lm entry: the Raon-1 nano model with the release fused paths installed after the weights load."""

    def sanitize(self, weights):
        # Hub weights are stored in MLX layout already (merged, renamed, quantized).
        if self.args.raon_weight_format == "mlx":
            return weights
        return super().sanitize(weights)

    def load_weights(self, *args, **kwargs):
        result = super().load_weights(*args, **kwargs)
        set_optimized(self, True, enable_decode=False)
        install(self, self.args.raon_variants)
        return result
