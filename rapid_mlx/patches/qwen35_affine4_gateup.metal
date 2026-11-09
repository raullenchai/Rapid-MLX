// Adapted from mlx-serve, MIT License, Copyright (c) 2026 David Dalcu.
// See QWEN35_AFFINE4_LICENSE for the full permission and warranty notice.
const uint lane = thread_index_in_simdgroup;
const int row0 = int(thread_position_in_grid.y) * ROWS;
const uint e = thread_position_in_grid.z;
const uint token = e / TOPK;
constexpr int KW2 = K / 16;
constexpr int KG = K / GS;
const size_t row = size_t(inds[e]) * (2 * N) + row0;
const device uint2* gw = (const device uint2*)wg_q + row * KW2 + lane;
const device uint2* uw = (const device uint2*)wu_q + (row + N) * KW2 + lane;
const device T* gsc = g_scales;
const device T* gbi = g_biases;
const device T* usc = u_scales + N * KG;
const device T* ubi = u_biases + N * KG;
float ag[ROWS] = {0.0f};
float au[ROWS] = {0.0f};
for (int k = 0; k < K; k += 512) {
  const int kk = k + int(lane) * 16;
  if (kk < K) {
    float xr[16];
    float xs = 0.0f;
    for (int i = 0; i < 16; ++i) { xr[i] = float(x[size_t(token) * K + kk + i]); xs += xr[i]; }
    const int gi = kk / GS;
    for (int r = 0; r < ROWS; ++r) {
      const size_t sg = (row + r) * KG + gi;
      ag[r] += float(gsc[sg]) * rapid_affine4_dot16(gw[r * KW2 + k / 16], xr) + float(gbi[sg]) * xs;
      au[r] += float(usc[sg]) * rapid_affine4_dot16(uw[r * KW2 + k / 16], xr) + float(ubi[sg]) * xs;
    }
  }
}
for (int r = 0; r < ROWS; ++r) {
  const float g = simd_sum(ag[r]);
  const float u = simd_sum(au[r]);
  if (lane == 0) {
    const T gt = T(g);
    y[size_t(e) * N + row0 + r] = (gt * sigtab[as_type<ushort>(gt)]) * T(u);
  }
}
