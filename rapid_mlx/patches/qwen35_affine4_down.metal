// Adapted from mlx-serve, MIT License, Copyright (c) 2026 David Dalcu.
// See QWEN35_AFFINE4_LICENSE for the full permission and warranty notice.
const uint lane = thread_index_in_simdgroup;
const int row0 = int(thread_position_in_grid.y) * ROWS;
const uint token = thread_position_in_grid.z;
constexpr int KW2 = I / 16;
constexpr int KG = I / GS;
float acc[ROWS] = {0.0f};
for (int e = 0; e < TOPK; ++e) {
  float projected[ROWS] = {0.0f};
  const size_t row = size_t(inds[token * TOPK + e]) * H + row0;
  const device uint2* dw = (const device uint2*)wd_q + row * KW2 + lane;
  const float sc = float(scores[token * TOPK + e]);
  for (int k = 0; k < I; k += 512) {
    const int kk = k + int(lane) * 16;
    if (kk < I) {
      float xr[16];
      float xs = 0.0f;
      for (int i = 0; i < 16; ++i) { xr[i] = float(act[(size_t(token) * TOPK + e) * I + kk + i]); xs += xr[i]; }
      const int gi = kk / GS;
      for (int r = 0; r < ROWS; ++r) {
        const size_t sg = (row + r) * KG + gi;
        projected[r] += float(d_scales[sg]) * rapid_affine4_dot16(dw[r * KW2 + k / 16], xr) + float(d_biases[sg]) * xs;
      }
    }
  }
  for (int r = 0; r < ROWS; ++r) {
    // Preserve the composed path's rounding before route reduction.
    const T rounded = T(simd_sum(projected[r]));
    const T weighted = T(float(rounded) * sc);
    acc[r] += float(weighted);
  }
}
for (int r = 0; r < ROWS; ++r) {
  const float v = acc[r];
  if (lane == 0) {
    T out = T(v);
    if (SHARED != 0) {
      // The gated shared expert joins as the chain does it: sigmoid in T (MLX's Sigmoid op), the
      // product rounded to T, the sum rounded to T.
      const T gl = glog[token];
      auto sy = 1 / (1 + metal::precise::exp(metal::abs(gl)));
      const T sig = (gl < 0) ? sy : 1 - sy;
      const T gated = T(float(sig) * float(sdown[size_t(token) * H + row0 + r]));
      out = T(float(out) + float(gated));
    }
    y[size_t(token) * H + row0 + r] = out;
  }
}
