"""Engram — n-gram hash lookups added into the residual stream at a few layers.

Layers 1 and 14 of the release each carry a ~384M-row fp8 embedding table
(~100 GB each; together ~40% of the checkpoint). A position is hashed as the
2-, 3- and 4-gram ending there, each split over 8 heads; every (n-gram size,
head) pair owns a disjoint prime-sized bucket range, primes drawn in order from
``engram_vocab_size`` upward and never reused. Hashing runs over a *compressed*
token map (case/accent/whitespace-normalized), so " The"/"the"/"THE" collapse.

The hash: per layer, one odd int64 multiplier per lookback position (seeded
``default_rng(10007 * layer_id)``); the running XOR of ``token * multiplier``
after i steps is the (i+1)-gram hash, taken mod each head's prime. Lookback
stops at the sequence start (and at image spans, which this text-only runtime
never produces); blocked slots read the compressed pad token.

Decode needs the previous 3 compressed ids, so :class:`EngramHasher` keeps a
per-sequence id cache — the model's cache object owns one.

The lookup itself (:class:`Engram`) fetches 24 rows of 256, projects them with
``wkv`` into one key per hyper-connection copy plus one shared value, and gates
the value into the stream by a normalized stream-key dot product pushed through
``sigmoid(signed sqrt)``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import ModelArgs
from .dequant import dequant_fp8_rows


def _isprime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def find_next_prime(start: int, seen: set[int]) -> int:
    c = start + 1
    while not _isprime(c) or c in seen:
        c += 1
    return c


def build_layout(args: ModelArgs):
    """Per-layer primes [n_layers][ngram-1][heads], and per-layer flat offsets.

    Primes are drawn sequentially across layers from ``engram_vocab_size - 1``
    upward, matching ``EngramLayout.from_args`` — so the sum of one layer's 24
    primes must equal its ``engram_num_embeddings`` entry.
    """
    primes, seen = [], set[int]()
    for _ in args.engram_layer_ids:
        per_ngram = []
        for _ in range(args.engram_max_ngram_size - 1):
            sizes, current = [], args.engram_vocab_size - 1
            for _ in range(args.engram_n_heads):
                current = find_next_prime(current, seen)
                seen.add(current)
                sizes.append(current)
            per_ngram.append(sizes)
        primes.append(per_ngram)
    flat = [[p for per_ngram in layer for p in per_ngram] for layer in primes]
    offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
    return np.asarray(primes, dtype=np.int64), np.asarray(offsets, dtype=np.int64)


def compute_hash_multipliers(
    layer_ids, max_ngram_size: int, compressed_vocab_size: int
) -> np.ndarray:
    """One odd multiplier per (layer, lookback), bounded so token*mult fits int64."""
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        g = np.random.default_rng(10007 * layer_id)
        v = g.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(v * 2 + 1)
    return np.stack(rows)


def build_compressed_token_map(tokenizer):
    """Token id -> compressed id, collapsing tokens that normalize alike.

    Transcribed from the reference ``engram.py``; needs the ``tokenizers``
    package and the release tokenizer. Tests inject a map directly instead.
    """
    from tokenizers import Regex, normalizers

    sentinel = ""
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_new: dict = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


class EngramHasher:
    """Maps positions to the hash ids of the n-grams ending there (numpy, int64).

    Stateless over weights; the per-sequence compressed-id history lives in the
    ``ids_cache`` array the caller passes (see ``cache.ModelCache.engram_ids``).
    """

    def __init__(self, args: ModelArgs, token_map):
        self.args = args
        self.max_ngram = args.engram_max_ngram_size
        self.token_map = np.asarray(token_map, dtype=np.int64)
        n_compressed = int(self.token_map.max()) + 1
        if (
            args.engram_compressed_vocab_size
            and n_compressed != args.engram_compressed_vocab_size
        ):
            raise ValueError(
                f"compressed vocab {n_compressed} != config "
                f"{args.engram_compressed_vocab_size}; every hash multiplier is "
                f"derived from it, so a mismatch silently rehashes the table"
            )
        self.pad_id = int(self.token_map[args.engram_pad_id])
        self.primes, self.offsets = build_layout(args)  # [L, G-1, H], [L, cols]
        self.multipliers = compute_hash_multipliers(
            args.engram_layer_ids, self.max_ngram, n_compressed
        )  # [L, G]
        # sanity: bucket layout must match the declared table sizes
        sums = self.primes.reshape(len(args.engram_layer_ids), -1).sum(-1)
        if tuple(int(s) for s in sums) != tuple(args.engram_num_embeddings):
            raise ValueError(
                f"prime sums {sums} != engram_num_embeddings "
                f"{args.engram_num_embeddings}"
            )

    def __call__(
        self, input_ids: np.ndarray, start_pos: int, ids_cache: np.ndarray
    ) -> np.ndarray:
        """input_ids [B, L] -> hash ids [B, L, n_engram_layers, (G-1)*H].

        ``ids_cache`` [B, max_seq] carries compressed ids across chunks; this
        call writes positions [start_pos, start_pos+L) into it.
        """
        input_ids = np.asarray(input_ids)
        batch, seqlen = input_ids.shape
        compressed = self.token_map[input_ids]
        ids_cache[:batch, start_pos : start_pos + seqlen] = compressed

        positions = np.broadcast_to(
            np.arange(start_pos, start_pos + seqlen), (batch, seqlen)
        )
        tokens, blocked = [], np.zeros_like(positions, dtype=bool)
        for shift in range(self.max_ngram):
            src_pos = np.clip(positions - shift, 0, None)
            source = np.take_along_axis(ids_cache[:batch], src_pos, axis=1)
            blocked = blocked | (positions < shift)
            tokens.append(np.where(blocked, self.pad_id, source))
        tokens = np.stack(tokens, axis=-1)  # [B, L, G]

        products = tokens[:, :, None, :] * self.multipliers  # [B, L, nL, G]
        rolling, hashes = products[..., 0], []
        for i in range(1, self.max_ngram):
            rolling = np.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling[..., None] % self.primes[:, i - 1])  # [B, L, nL, H]
        return np.concatenate(hashes, axis=-1) + self.offsets


class EngramEmbedding(nn.Module):
    """The hash table; stays fp8 in memory, rows dequantized on lookup.

    ``weight`` uint8 [rows, d] (fp8 e4m3 bytes) + ``scale`` uint8 [rows, d//32]
    (ue8m0) for the real checkpoint; float ``weight``/``scale`` (fp8 values and
    power-of-two scales as floats) for tests.
    """

    def __init__(self, num_embeddings: int, dim: int, block: int = 32):
        super().__init__()
        self.block = block
        self.weight = mx.zeros((num_embeddings, dim), dtype=mx.uint8)
        self.scale = mx.ones((num_embeddings, dim // block), dtype=mx.uint8)

    def __call__(self, indices: mx.array) -> mx.array:
        v = self.weight[indices]  # [..., d]
        s = self.scale[indices]  # [..., d//block]
        if self.weight.dtype == mx.uint8:
            return dequant_fp8_rows(v, s, self.block)
        vf = v.astype(mx.float32).reshape(
            *v.shape[:-1], v.shape[-1] // self.block, self.block
        )
        return (vf * s.astype(mx.float32)[..., None]).reshape(v.shape)


class QuantizedEngramEmbedding(nn.Module):
    """The hash table in MLX affine quantization (weight/scales/biases), rows
    dequantized on lookup. Used when the build was converted with
    ``engram_bits`` — the native fp8 table alone is 203 GB, which does not
    leave room for the rest of the model on a 512 GB machine."""

    def __init__(
        self, num_embeddings: int, dim: int, group_size: int = 64, bits: int = 4
    ):
        super().__init__()
        self.group_size, self.bits = group_size, bits
        self.weight = mx.zeros((num_embeddings, dim * bits // 32), dtype=mx.uint32)
        self.scales = mx.zeros((num_embeddings, dim // group_size), dtype=mx.bfloat16)
        self.biases = mx.zeros((num_embeddings, dim // group_size), dtype=mx.bfloat16)

    def __call__(self, indices: mx.array) -> mx.array:
        out = mx.dequantize(
            self.weight[indices],
            scales=self.scales[indices],
            biases=self.biases[indices],
            group_size=self.group_size,
            bits=self.bits,
        )
        return out.astype(mx.float32)


class Engram(nn.Module):
    """Gated write of the n-gram lookup into the hc-expanded residual stream."""

    def __init__(self, args: ModelArgs, layer_hash_index: int):
        super().__init__()
        self.layer_hash_index = layer_hash_index
        self.dim = args.dim
        self.hc_mult = args.hc_mult
        self.eps = args.norm_eps
        self.clamp_value = 1e-6

        n_hash_cols = (args.engram_max_ngram_size - 1) * args.engram_n_heads
        self.embed = EngramEmbedding(
            args.engram_num_embeddings[layer_hash_index], args.engram_head_dim
        )
        self.wkv = nn.Linear(
            n_hash_cols * args.engram_head_dim,
            args.dim * (args.hc_mult + 1),
            bias=False,
        )
        self.q_weight = mx.ones((args.hc_mult, args.dim), dtype=mx.float32)
        self.k_weight = mx.ones((args.hc_mult, args.dim), dtype=mx.float32)

    def __call__(self, x: mx.array, hash_ids: mx.array) -> mx.array:
        """x [B, L, hc, dim]; hash_ids [B, L, n_hash_cols]."""
        rows = self.embed(hash_ids)  # [B, L, cols, hd] fp32
        rows = rows.reshape(*rows.shape[:-2], -1).astype(x.dtype)
        kv = self.wkv(rows)  # [B, L, (hc+1)*dim]
        key = kv[..., : self.hc_mult * self.dim].astype(mx.float32)
        value = kv[..., self.hc_mult * self.dim :].astype(mx.float32)
        key = key.reshape(*key.shape[:-1], self.hc_mult, self.dim)

        weight = self.q_weight.astype(mx.float32) * self.k_weight.astype(mx.float32)
        h = x.astype(mx.float32)
        # normalized per (token, hc copy) over dim, NOT jointly over the copies
        rstd = mx.rsqrt(mx.mean(mx.square(h), axis=-1) + self.eps) * mx.rsqrt(
            mx.mean(mx.square(key), axis=-1) + self.eps
        )
        dot = mx.sum(h * weight * key, axis=-1) * rstd * self.dim**-0.5
        # signed sqrt before the sigmoid, matching the training kernel
        mag = mx.sqrt(mx.maximum(mx.abs(dot), self.clamp_value))
        gate = mx.sigmoid(mx.where(dot < 0, -mag, mag))
        out = h + gate[..., None] * value[..., None, :]
        return out.astype(x.dtype)
