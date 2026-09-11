"""Hyper-Connections — the residual stream is ``hc_mult`` (=4) parallel copies.

V4.1 keeps V4's Sinkhorn machinery but **staggers** the coefficients: the mixes a
sub-layer computes are consumed one sub-layer *later*. From the reference
``Block.forward``:

* ``hc_mixes`` on the attention input yields ``(attn_pre, attn_post, attn_comb)``;
  attention's own ``hc_pre`` uses the ``pre`` produced by the *previous* layer's
  FFN (identity one-hot on copy 0 at the very first layer), while ``attn_post`` /
  ``attn_comb`` are used by attention's ``hc_post`` immediately;
* ``attn_pre`` then collapses the FFN input, whose own mixes hand ``ffn_pre`` to
  the *next* layer — and after the last layer, to the LM head's final collapse.

``comb`` is pushed toward doubly-stochastic by 20 Sinkhorn sweeps.
``hc_post`` computes ``out[k] = post[k]*x + sum_j comb[j,k]*residual[j]`` —
the residual is indexed by the **summed** axis j (comb's first index), a
known bug magnet; broadcasting residual onto k instead stays finite and
plausible but wrong.
"""

from __future__ import annotations

import mlx.core as mx


def split_sinkhorn(
    mixes: mx.array,
    hc_scale: mx.array,
    hc_base: mx.array,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """Split one projection into (pre, post, comb) — transcribed from
    ``hc_split_sinkhorn_kernel``. Layout: first hc entries -> pre, next hc ->
    post, remaining hc*hc -> comb row-major."""
    hc = hc_mult
    m = mixes.astype(mx.float32)
    scale = hc_scale.astype(mx.float32)
    base = hc_base.astype(mx.float32)

    pre = mx.sigmoid(m[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2.0 * mx.sigmoid(m[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])

    comb = m[..., 2 * hc :] * scale[2] + base[2 * hc :]
    comb = comb.reshape(*comb.shape[:-1], hc, hc)
    # row softmax + eps, one column normalization, then (iters-1) full sweeps;
    # the eps sits inside the divisions, as in the kernel
    comb = mx.softmax(comb, axis=-1) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


def hc_mixes(
    x: mx.array,
    hc_fn: mx.array,
    hc_scale: mx.array,
    hc_base: mx.array,
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
):
    """The coefficient projection: RMS-normalize the flattened [b,s,hc*d] stream
    (rsqrt applied AFTER the linear, matching the reference's operation order),
    project, split. Returns (pre, post, comb)."""
    xf = x.reshape(*x.shape[:2], -1).astype(mx.float32)
    rsqrt = mx.rsqrt(mx.mean(mx.square(xf), axis=-1, keepdims=True) + norm_eps)
    mixes = (xf @ hc_fn.astype(mx.float32).T) * rsqrt
    return split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps)


def hc_pre(x: mx.array, pre_mix: mx.array) -> mx.array:
    """Collapse the hc copies into one: [b,s,hc,d] x [b,s,hc] -> [b,s,d], fp32 sum."""
    y = mx.sum(pre_mix[..., None].astype(mx.float32) * x.astype(mx.float32), axis=2)
    return y.astype(x.dtype)


def hc_post(
    x: mx.array, residual: mx.array, post: mx.array, comb: mx.array
) -> mx.array:
    """out[k] = post[k]*x + sum_j comb[j,k]*residual[j].

    x [b,s,d], residual [b,s,hc,d], post [b,s,hc], comb [b,s,hc,hc] -> [b,s,hc,d].
    The residual must sit on the j (summed) axis of the product.
    """
    prod = comb[..., None] * residual[..., :, None, :]  # [b, s, j, k, d]
    out = post[..., None] * x[..., None, :] + mx.sum(prod, axis=2)
    return out.astype(x.dtype)


def make_identity_pre_mix(b: int, s: int, hc_mult: int) -> mx.array:
    """The initial one-hot mix: copy 0 only."""
    m = mx.zeros((b, s, hc_mult), dtype=mx.float32)
    return mx.concatenate([mx.ones((b, s, 1), dtype=mx.float32), m[..., 1:]], axis=-1)
