# SPDX-License-Identifier: MIT
"""Functional singleton residual/RMS seam; reduction matches MLX 0.32.4."""

# Reduction geometry and cast boundaries follow Apple MLX rms_norm.metal,
# Copyright (c) 2024 Apple Inc., MIT; see LICENSE-RMS.
import mlx.core as mx

source = r"""
uint lid=thread_position_in_threadgroup.x;
uint sg=simdgroup_index_in_threadgroup;
uint sl=thread_index_in_simdgroup;
threadgroup float partial[32];
threadgroup float inv[1];
float v[4];float acc=0;
for(int j=0;j<4;j++){
 uint i=lid*4+j;
 T z=i<N ? T(a[i]+b[i]) : T(0);
 v[j]=float(z);
 if(i<N)sum[i]=z;
 acc+=v[j]*v[j];
}
acc=simd_sum(acc);
if(sg==0)partial[sl]=0;
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sl==0)partial[sg]=acc;
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==0){acc=simd_sum(partial[sl]);if(sl==0)inv[0]=metal::precise::rsqrt(acc/float(N)+eps);}
threadgroup_barrier(mem_flags::mem_threadgroup);
for(int j=0;j<4;j++){uint i=lid*4+j;if(i<N)normalized[i]=w[i]*T(v[j]*inv[0]);}
"""
kernel = mx.fast.metal_kernel(
    name="rapid_dense_add_rms",
    input_names=["a", "b", "w", "eps"],
    output_names=["sum", "normalized"],
    source=source,
)


def seam(a, b, norm):
    n = a.shape[-1]
    threads = ((n + 127) // 128) * 32
    return kernel(
        inputs=[a, b, norm.weight, mx.array(norm.eps, mx.float32)],
        template=[("T", a.dtype), ("N", n)],
        grid=(threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[a.shape, a.shape],
        output_dtypes=[a.dtype, a.dtype],
    )
