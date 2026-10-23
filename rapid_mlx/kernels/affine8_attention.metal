// Adapted from mlx-serve 191da71de32a65f69ebceeb6cb2e91036ab50b30.
// Modifications: MLX dispatch, boolean masks, uniform fully masked rows, strides.
// See NOTICE-AFFINE8 and LICENSE-AFFINE8*.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

using namespace mpp::tensor_ops;
constexpr int VPW = 32 / BITS;
constexpr int MR = GQA * TQ;
constexpr int M = MPAD; // matmul2d wants rows in multiples of 8; pad rows carry zero queries
constexpr int N = 32;
constexpr int D = DK;
constexpr int NG = DK / GS; // N * NG dequant threads must fit the 256-thread group
const uint tid = thread_position_in_threadgroup.x;
const uint hkv = threadgroup_position_in_grid.y;
const uint split = threadgroup_position_in_grid.z;
const uint nsplit = threadgroups_per_grid.z;
const int Tk = tk[0];
const float scale = scl[0];
const uint mask_bits = (1u << BITS) - 1u;
const int pages = (Tk + N - 1) / N;
const int per = (pages + int(nsplit) - 1) / int(nsplit);
const int pb = int(split) * per;
const int pe = min(pages, pb + per);

threadgroup T tile[N * D];
threadgroup float scores[M * N];
threadgroup float probs[M * N];
threadgroup float rmax[M];
threadgroup float rsum[M];
threadgroup float pscale[M];

if (tid < uint(M)) { rmax[tid] = -INFINITY; rsum[tid] = 0.0f; }
threadgroup_barrier(mem_flags::mem_threadgroup);

constexpr auto qk_desc = matmul2d_descriptor(M, N, D, false, true, false, matmul2d_descriptor::mode::multiply);
constexpr auto pv_desc = matmul2d_descriptor(M, D, N, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<qk_desc, execution_simdgroups<8>> qk;
matmul2d<pv_desc, execution_simdgroups<8>> pv;
auto qt = tensor((device T*)q + (long)hkv * M * D, dextents<int, 2>{D, M}, array<int, 2>{1, D});
auto tt = tensor(tile, dextents<int, 2>{D, N}, array<int, 2>{1, D});
auto st = tensor(scores, dextents<int, 2>{N, M}, array<int, 2>{1, N});
auto pt = tensor(probs, dextents<int, 2>{N, M}, array<int, 2>{1, N});
auto q0 = qt.template slice<D, M>(0, 0);
auto t0 = tt.template slice<D, N>(0, 0);
auto p0 = pt.template slice<N, M>(0, 0);
auto running = pv.template get_destination_cooperative_tensor<decltype(p0), decltype(t0), float>();
for (ushort i = 0; i < running.get_capacity(); ++i)
  if (running.is_valid_element(i)) running[i] = 0.0f;

for (int page = pb; page < pe; ++page) {
  const int tok0 = page * N;
  // K page -> tile: thread d dequantizes element d of the page's 32 rows.
  if (tid < uint(N * NG)) {
    const int n = int(tid) / NG;
    const int g = int(tid) % NG;
    const long r = (long)min(tok0 + n, Tk - 1);
    const float sj = float(ksc[(long)hkv * ksc_strides[1] + r * ksc_strides[2] + g * ksc_strides[3]]);
    const float bj = float(kbi[(long)hkv * kbi_strides[1] + r * kbi_strides[2] + g * kbi_strides[3]]);
    const long wb = (long)hkv * kq_strides[1] + r * kq_strides[2] + (g * GS) / VPW * kq_strides[3];
    for (int wi = 0; wi < GS / VPW; ++wi) {
      const uint w = uint(kq[wb + wi * kq_strides[3]]);
      for (int u = 0; u < VPW; ++u)
        tile[n * D + g * GS + wi * VPW + u] = T(float((w >> (u * BITS)) & mask_bits) * sj + bj);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto page_scores = qk.template get_destination_cooperative_tensor<decltype(q0), decltype(t0), float>();
  qk.run(q0, t0, page_scores);
  page_scores.store(st.template slice<N, M>(0, 0));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < uint(M)) {
    const int m = int(tid);
    const int lim = m < MR ? Tk - TQ + 1 + (m % TQ) : Tk;
    float lmax = -INFINITY;
    for (int n = 0; n < N; ++n)
      if ((tok0 + n < Tk && (HAS_MASK ? (!row_valid[m % TQ] || bool(attn_mask[(m % TQ) * Tk + tok0 + n])) : tok0 + n < lim))) lmax = max(lmax, ((HAS_MASK && !row_valid[m % TQ]) ? 0.0f : scores[m * N + n] * scale));
    const float prev = rmax[m];
    const float next = max(prev, lmax);
    float lsum = 0.0f;
    for (int n = 0; n < N; ++n) {
      const float pr = ((tok0 + n < Tk && (HAS_MASK ? (!row_valid[m % TQ] || bool(attn_mask[(m % TQ) * Tk + tok0 + n])) : tok0 + n < lim))) ? metal::exp(((HAS_MASK && !row_valid[m % TQ]) ? 0.0f : scores[m * N + n] * scale) - next) : 0.0f;
      probs[m * N + n] = pr;
      lsum += pr;
    }
    const float sc = (next == -INFINITY || next == prev) ? 1.0f : metal::exp(prev - next);
    pscale[m] = sc;
    rsum[m] = rsum[m] * sc + lsum;
    rmax[m] = next;
  }
  // V page -> the same tile.
  if (tid < uint(N * NG)) {
    const int n = int(tid) / NG;
    const int g = int(tid) % NG;
    const long r = (long)min(tok0 + n, Tk - 1);
    const float sj = float(vsc[(long)hkv * vsc_strides[1] + r * vsc_strides[2] + g * vsc_strides[3]]);
    const float bj = float(vbi[(long)hkv * vbi_strides[1] + r * vbi_strides[2] + g * vbi_strides[3]]);
    const long wb = (long)hkv * vq_strides[1] + r * vq_strides[2] + (g * GS) / VPW * vq_strides[3];
    for (int wi = 0; wi < GS / VPW; ++wi) {
      const uint w = uint(vq[wb + wi * vq_strides[3]]);
      for (int u = 0; u < VPW; ++u)
        tile[n * D + g * GS + wi * VPW + u] = T(float((w >> (u * BITS)) & mask_bits) * sj + bj);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (ushort i = 0; i < running.get_capacity(); ++i) {
    if (!running.is_valid_element(i)) continue;
    const auto c = running.get_multidimensional_index(i);
    running[i] *= pscale[c[1]];
  }
  pv.run(p0, t0, running);
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
const long slot = (long)hkv * nsplit + split;
auto target = tensor(o_out + slot * M * D, dextents<int, 2>{D, M}, array<int, 2>{1, D});
running.store(target.template slice<D, M>(0, 0));
if (tid < uint(M)) {
  m_out[slot * M + tid] = max(rmax[tid], -1e30f);
  l_out[slot * M + tid] = rsum[tid];
}