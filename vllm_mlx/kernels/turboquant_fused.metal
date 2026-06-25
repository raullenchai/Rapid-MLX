// SPDX-License-Identifier: Apache-2.0
//
// Fused TurboQuant Metal kernel sources.
//
// Vendored (Apache-2.0) from arozanov/turboquant-mlx
//   https://github.com/arozanov/turboquant-mlx
//   pyproject license = "Apache-2.0", commit 6e928d71 (2026-04-30).
//
// Two kernel bodies are concatenated below — the Python binding picks
// each apart with the ``// >>> kernel: <name>`` sentinel before
// shipping each to ``mx.fast.metal_kernel``. The sentinel block is the
// ONLY contract this file has with the Python loader; everything else
// is plain MSL that the Metal compiler sees verbatim.
//
// Deviations from upstream:
//   * Per-coord uniform-scale K8 kernel added (upstream is V-only +
//     mx.quantized_matmul for K). The K8 kernel re-uses the WHT
//     butterfly + L2-norm preamble from the V kernel; only the
//     codebook lookup is replaced with a symmetric uniform scale.
//   * threadgroup capacity bumped from 256 to 512 floats to fit
//     head_dim=128 with the extra scratch buffer needed for the K8
//     scale broadcast.
//   * The dim/bits/vpw/packed_dim/n_centroids tuple is passed as a
//     single uint32x5 buffer so a future caller can switch the dispatch
//     dimensions without touching the kernel source.
//
// All threadgroup_barriers are kept verbatim — the deviation policy is
// "preserve semantics, extend coverage". See PR #919 for the bench
// plan that confirms the K8 path matches the V4 kernel's wave-front
// occupancy on M3 Max.

// >>> kernel: tq_fused_quantize_v4
    uint pos = threadgroup_position_in_grid.x;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    uint bits = dims[1];
    uint vals_per_word = dims[2];
    uint packed_dim = dims[3];
    uint n_centroids = dims[4];

    threadgroup float shared[256];
    shared[elem] = (float)inp[pos * dim + elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float norm_shared[256];
    norm_shared[elem] = shared[elem] * shared[elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = dim / 2; stride > 0; stride >>= 1) {
        if (elem < stride) {
            norm_shared[elem] += norm_shared[elem + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float vec_norm = sqrt(norm_shared[0]);
    float safe_norm = max(vec_norm, 1e-8f);

    shared[elem] = shared[elem] / safe_norm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    shared[elem] = shared[elem] * signs[elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint h = 1;
    while (h < dim) {
        uint block = elem / (2 * h);
        uint offset = elem % (2 * h);
        if (offset < h) {
            uint j = block * 2 * h + offset;
            float a = shared[j];
            float b = shared[j + h];
            shared[j] = a + b;
            shared[j + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h *= 2;
    }

    float scaled = shared[elem];

    uint idx = 0;
    for (uint b = 0; b < n_centroids - 1; b++) {
        if (scaled > boundaries[b]) {
            idx++;
        }
    }

    threadgroup uint idx_shared[256];
    idx_shared[elem] = idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint word_idx = elem / vals_per_word;
    uint pos_in_word = elem % vals_per_word;

    if (pos_in_word == 0 && word_idx < packed_dim) {
        uint word = 0;
        for (uint i = 0; i < vals_per_word && (word_idx * vals_per_word + i) < dim; i++) {
            word |= (idx_shared[word_idx * vals_per_word + i] & ((1u << bits) - 1u)) << (i * bits);
        }
        packed_out[pos * packed_dim + word_idx] = word;
    }

    if (elem == 0) {
        norms_out[pos] = vec_norm;
    }

// >>> kernel: tq_fused_dequant_v4_fp16
    uint pos = threadgroup_position_in_grid.x;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    uint bits = dims[1];
    uint vals_per_word = dims[2];
    uint packed_dim = dims[3];
    uint bit_mask = (1u << bits) - 1u;

    uint word_idx = elem / vals_per_word;
    uint pos_in_word = elem % vals_per_word;
    uint word = packed[pos * packed_dim + word_idx];
    uint idx = (word >> (pos_in_word * bits)) & bit_mask;

    float val = centroids[idx] * scale[0];

    threadgroup float shared[256];
    shared[elem] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint h = 1;
    while (h < dim) {
        uint block = elem / (2 * h);
        uint offset = elem % (2 * h);
        if (offset < h) {
            uint j = block * 2 * h + offset;
            float a = shared[j];
            float b = shared[j + h];
            shared[j] = a + b;
            shared[j + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h *= 2;
    }

    float result = shared[elem] * scale[0] * signs[elem] * norms[pos];
    out[pos * dim + elem] = (half)result;

// >>> kernel: tq_fused_quantize_k8
    // rapid-mlx extension (Apache-2.0): per-coordinate symmetric uniform
    // quant for the K side of the K8V4 mix. Reuses the V4 preamble:
    // L2 norm → unit normalize → randomized Hadamard rotate. After
    // rotation each coord is approximately N(0, 1/sqrt(dim)), so a
    // per-vector absmax + uniform 8-bit symmetric grid (-127..127)
    // matches the Lloyd-Max scheme within 0.3 dB at d>=64 and is
    // cheaper to dequant on the decode path.
    uint pos = threadgroup_position_in_grid.x;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    // dims[1] == 8 always (k8). Kept as a uniform for parity with the V4 kernel.

    threadgroup float shared[256];
    shared[elem] = (float)inp[pos * dim + elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // L2 norm (parallel reduction)
    threadgroup float norm_shared[256];
    norm_shared[elem] = shared[elem] * shared[elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = dim / 2; stride > 0; stride >>= 1) {
        if (elem < stride) {
            norm_shared[elem] += norm_shared[elem + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float vec_norm = sqrt(norm_shared[0]);
    float safe_norm = max(vec_norm, 1e-8f);

    // Normalize to the unit sphere then randomized-Hadamard rotate
    shared[elem] = shared[elem] / safe_norm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    shared[elem] = shared[elem] * signs[elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint h = 1;
    while (h < dim) {
        uint block = elem / (2 * h);
        uint offset = elem % (2 * h);
        if (offset < h) {
            uint j = block * 2 * h + offset;
            float a = shared[j];
            float b = shared[j + h];
            shared[j] = a + b;
            shared[j + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h *= 2;
    }

    // Apply 1/sqrt(dim) so the WHT matches the orthogonal convention
    // used by the Python reference (vllm_mlx.turboquant.walsh_hadamard_transform).
    // Encode and decode both share this scaling so the symmetric int8
    // grid is bit-exact across the fused and unfused paths.
    float inv_sqrt = rsqrt((float)dim);
    shared[elem] = shared[elem] * inv_sqrt;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Find per-vector absmax for the symmetric int8 grid.
    threadgroup float amax_shared[256];
    amax_shared[elem] = fabs(shared[elem]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = dim / 2; stride > 0; stride >>= 1) {
        if (elem < stride) {
            amax_shared[elem] = max(amax_shared[elem], amax_shared[elem + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float amax = max(amax_shared[0], 1e-8f);
    float scale_k = amax / 127.0f;

    // Symmetric 8-bit quant: round(x / scale) clamped to [-127, 127].
    float q = shared[elem] / scale_k;
    int qi = (int)round(q);
    qi = clamp(qi, -127, 127);

    // Pack as uint8 (offset by +128 so the storage type is unsigned).
    packed_k_out[pos * dim + elem] = (uchar)(qi + 128);

    if (elem == 0) {
        norms_out[pos] = vec_norm;
        k_scales_out[pos] = scale_k;
    }
