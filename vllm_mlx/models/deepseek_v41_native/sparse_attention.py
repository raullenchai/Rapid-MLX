"""Gather-based sparse attention — the MLX stand-in for the TileLang kernel.

Semantics from ``sparse_attn_kernel`` in the reference ``kernel.py``:

* ``kv`` is a **single** shared vector per position (MLA): key and value are the
  same tensor;
* an index of ``-1`` means "not visible": its logit is -inf, contributing nothing;
* the learned per-head ``attn_sink`` enters the softmax **denominator only** — it
  carries no value vector, so a head with weak logits attends to nearly nothing;
* a row with no valid index yields an all-zero output (the kernel's finite
  -1e30 max floor).

The kernel keeps its running max over the real logits only and adds
``exp(sink - max)`` at the end; here the sink also competes for the max, which is
mathematically identical (softmax shift invariance) and safer numerically.
"""

from __future__ import annotations

import mlx.core as mx

NEG_INF = -1e30


def _gather_kv(kv: mx.array, idx: mx.array) -> mx.array:
    """kv [b, n, d], idx [b, m, k] -> [b, m, k, d]; row 0 for negative idx."""
    b, n, d = kv.shape
    flat = kv.reshape(b * n, d)
    safe = mx.maximum(idx, 0).astype(mx.int32)
    base = (mx.arange(b, dtype=mx.int32) * n).reshape(b, 1, 1)
    return flat[(safe + base).reshape(-1)].reshape(*idx.shape, d)


def sparse_attn(
    q: mx.array,
    kv: mx.array,
    attn_sink: mx.array,
    topk_idxs: mx.array,
    softmax_scale: float,
    chunk: int = 256,
) -> mx.array:
    """q [b,m,h,d], kv [b,n,d], attn_sink [h], topk_idxs [b,m,k] (-1 = masked)."""
    b, m, h, d = q.shape
    sink = attn_sink.astype(mx.float32).reshape(1, 1, h, 1)

    outs = []
    for start in range(0, m, chunk):
        stop = min(start + chunk, m)
        qc = q[:, start:stop].astype(mx.float32)
        ic = topk_idxs[:, start:stop]
        kvc = _gather_kv(kv, ic).astype(mx.float32)  # [b, c, k, d]

        logits = mx.einsum("bchd,bckd->bchk", qc, kvc) * softmax_scale
        valid = (ic >= 0)[:, :, None, :]
        logits = mx.where(valid, logits, NEG_INF)

        mmax = mx.max(logits, axis=-1, keepdims=True)
        mmax = mx.maximum(mx.maximum(mmax, sink), NEG_INF)
        w = mx.exp(logits - mmax)
        w = mx.where(valid, w, 0.0)
        denom = mx.sum(w, axis=-1, keepdims=True) + mx.exp(sink - mmax)

        o = mx.einsum("bchk,bckd->bchd", w, kvc) / denom
        outs.append(o.astype(q.dtype))

    return mx.concatenate(outs, axis=1) if len(outs) > 1 else outs[0]
