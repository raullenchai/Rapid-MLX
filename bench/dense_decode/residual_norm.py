# SPDX-License-Identifier: MIT
"""Diagnostic singleton residual/RMS seam; reduction matches MLX 0.32.4."""

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
    name="r4_add_rms",
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


def install(model):
    count = 0
    for _, m in list(model.named_modules()):
        cls = type(m)
        if cls.__name__ not in ("Qwen3_5DecoderLayer", "Lfm2DecoderLayer"):
            continue
        if not getattr(cls, "_r4_seam", False):
            original = cls.__call__
            if cls.__name__ == "Qwen3_5DecoderLayer":

                def call(
                    self,
                    x,
                    mask=None,
                    cache=None,
                    position_ids=None,
                    position_embeddings=None,
                    _orig=original,
                ):
                    if x.size // x.shape[-1] != 1 or x.shape[-1] > 4096:
                        return _orig(
                            self, x, mask, cache, position_ids, position_embeddings
                        )
                    if self.is_linear:
                        r = self.linear_attn(self.input_layernorm(x), mask, cache)
                    else:
                        r = self.self_attn(
                            self.input_layernorm(x),
                            mask=mask,
                            cache=cache,
                            position_ids=position_ids,
                            position_embeddings=position_embeddings,
                        )
                    h, n = seam(x, r, self.post_attention_layernorm)
                    return h + self.mlp(n)
            else:

                def call(self, x, mask=None, cache=None, gdn_sink=None, _orig=original):
                    if x.size // x.shape[-1] != 1 or x.shape[-1] > 4096:
                        return _orig(self, x, mask, cache, gdn_sink)
                    if self.is_attention_layer:
                        r = self.self_attn(
                            self.operator_norm(x), mask=mask, cache=cache
                        )
                    else:
                        r = self.conv(
                            self.operator_norm(x),
                            mask=mask,
                            cache=cache,
                            gdn_sink=gdn_sink,
                        )
                    h, n = seam(x, r, self.ffn_norm)
                    return h + self.feed_forward(n)

            cls.__call__ = call
            cls._r4_seam = True
        count += 1
    return count
