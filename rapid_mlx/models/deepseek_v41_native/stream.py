"""Layer-at-a-time access to the raw release, for divergence ladders and the
streaming quantizer (pattern from glm53-mlx).

A V4.1 layer cannot run in isolation the way a plain transformer layer can:
the hyper-connection stream carries a staggered ``pre_mix`` and the attention
layers share compressed KV / index selections through a ``SharedState``. The
carried state is therefore explicit: :func:`run_layer` takes and returns a
:class:`StreamState` alongside the hidden stream.

Consumer layers reference their kv/index source's caches through the shared
state, so a streaming pass must keep each source layer's ``LayerCache`` alive
until its last consumer has run — :class:`StreamState` owns the full
``ModelCache`` (per-layer caches are small next to the weights: window ring +
compressed latents).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn

from .cache import ModelCache
from .config import ModelArgs
from .convert import sanitize_group
from .hyper_connections import hc_pre, make_identity_pre_mix
from .model import Block, SharedState

_SHARD_CACHE: dict[str, dict[str, mx.array]] = {}


def read_config(src: Path):
    raw = json.load(open(Path(src) / "config.json"))
    return raw, ModelArgs.from_dict(raw)


def shard_map(src: Path) -> dict[str, str]:
    raw = json.load(open(Path(src) / "model.safetensors.index.json"))
    return cast(dict[str, str], raw["weight_map"])


def load_subset(src: Path, smap: dict[str, str], prefix: str) -> dict[str, mx.array]:
    """All tensors whose name starts with ``prefix``, memory-mapped lazily."""
    out = {}
    for k, shard in smap.items():
        if k.startswith(prefix):
            shard_path = str((Path(src) / shard).resolve())
            if shard_path not in _SHARD_CACHE:
                _SHARD_CACHE.clear()
                _SHARD_CACHE[shard_path] = mx.load(shard_path)
            out[k] = _SHARD_CACHE[shard_path][k]
    return out


def build_layer(
    margs: ModelArgs, layer_i: int, src: Path, smap: dict[str, str]
) -> nn.Module:
    """One Block holding the raw release weights for layer ``layer_i``,
    sanitized (dequantized + experts stacked) on the fly."""
    prefix = f"layers.{layer_i}."
    raw = load_subset(src, smap, prefix)
    sane = sanitize_group(sorted(raw), raw)
    # ``wo_a`` is stored flat in the release but represented as MultiLinear
    # so the same module can execute both float and grouped affine weights.
    # Keep the streaming path consistent with the strict full-model loader.
    from .load import reshape_grouped_wo_a

    sane = dict(reshape_grouped_wo_a(sane.items(), margs))
    layer = Block(layer_i, margs)
    layer.load_weights([(k[len(prefix) :], v) for k, v in sane.items()], strict=True)
    return layer


class StreamState:
    """Everything a 40-layer streaming pass carries between layers."""

    def __init__(self, margs: ModelArgs, bsz: int, max_seq_len: int, token_map=None):
        self.margs = margs
        self.cache = ModelCache(margs, bsz, max_seq_len)
        self.shared = SharedState()
        self.pre_mix = None
        self.hashes = None
        self.hasher: Any
        if margs.engram_layer_ids and token_map is not None:
            from .engram import EngramHasher

            self.hasher = EngramHasher(margs, token_map)
        else:
            self.hasher = None

    def begin(self, input_ids_np, h_embedded: mx.array, hc_mult: int) -> mx.array:
        """Start a pass: expand the embedding to hc copies, hash for engram."""
        b, n = input_ids_np.shape
        if self.hasher is not None:
            self.hashes = mx.array(
                self.hasher(input_ids_np, self.cache.offset, self.cache.engram_ids)
            )
        self.pre_mix = make_identity_pre_mix(b, n, hc_mult)
        self.shared = SharedState()
        return mx.broadcast_to(
            h_embedded[:, :, None, :], (b, n, hc_mult, h_embedded.shape[-1])
        )


def run_layer(layer: Block, h: mx.array, state: StreamState) -> mx.array:
    """Run one Block, threading the carried state. Returns the new hc stream."""
    if layer.engram is not None and state.hashes is not None:
        h = layer.engram(h, state.hashes[:, :, layer.engram.layer_hash_index])
    h, state.pre_mix = layer(
        h, state.pre_mix, state.cache.offset, state.cache, state.shared
    )
    return h


def finish(
    h: mx.array, state: StreamState, norm, head_weight: mx.array, seqlen: int
) -> mx.array:
    """Collapse with the last ffn_pre, norm, project to logits; advance offset."""
    h = hc_pre(h, state.pre_mix)
    h = norm(h)
    logits = h.astype(mx.float32) @ head_weight.astype(mx.float32).T
    state.cache.offset += seqlen
    return logits
