"""Instance-local, source-guarded unpadded admission for mlx-lm 0.31.3.

No upstream module or class is patched. Python cache lengths establish the
padding proof at merge; every padding mutation either maintains or invalidates
it. Split/copy/filter lifecycle remains upstream's, including its deep copies
and empty source lists. Only an identity cache filter with zero padding skips
work (upstream's left shift is then also an identity).
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import os
import textwrap
import weakref
from importlib.metadata import PackageNotFoundError, version
from types import FunctionType, MethodType
from typing import Any

# The dependency allows >=0.31.3,<0.32; only this audited release is supported.
_SOURCE_HASHES = {
    "create_causal_mask": "e99d5e475b029fdf6fe2d71a4d45e66d9ccf4304199e42e99c5c92da08f41524",
    "BatchKVCache.merge": "5080e5997d0899e2fa68d6aafced38f37908b002b2f41d53f2adc62052ef84af",
    "BatchKVCache.__init__": "79005036ecdc77b83971e7e9ae04b40e504259b67ce8abd390b00c900749c397",
    "BatchKVCache.prepare": "97f8b6b2d4653bcd5a3f662d6866708c155144fb2cadbddc832ef287141a4ceb",
    "BatchKVCache.finalize": "5f22fa3a902e7ee5803086677881031c02f7330da9018f1652badefabd07dc82",
    "BatchKVCache.trim": "532849df59d7d91f71df2f980d98c8e8e46cc538a36fb36f4245c646bcc65d2b",
    "BatchKVCache.filter": "8706505a6e93b68bc376f4c81584db07af7d958adf7b83d609b3c1e610d6189f",
    "BatchKVCache.extend": "706d054638486a6deaf6fd96f31018970a94951aae904a4688b5314d9278cdf3",
    "BatchKVCache.make_mask": "f832dc6b26a7f2b5d2d9fe997815080c6c7e2495766b593519209b76923d8ef9",
    "BatchKVCache.update_and_fetch": "f7369d07d0a8813e93412407d0573dc9d5ddd0b10710b5c883caa4ef3f62a3ff",
    "BatchKVCache.state.fget": "e7c1426a5a34b5cc4f94679d58b1d5a130a8103b029ea23f098eebfeaf855c56",
    "BatchKVCache.state.fset": "29de9a5816e222ac3d682047e26ff4c824c3f55f5b5326173bb2264f675743e3",
    "KVCache.size": "72d4a28d3858c2fdeb5d83d4faaed9f9037ebe74db526eb7d06fd0652a1fc00a",
    "KVCache.merge": "6cc964f79bc1df50d79969491474d6bf4e7bc596c01265c9e76ed4ed97dd6db8",
    "create_attention_mask": "78e3ed144e774a1e1f39fe8e21a3a6e6b6824805e86a11f71cd78696f4d5a199",
    "PromptProcessingBatch.__init__": "ee2d28798a28924345c42e2031871416d968f6a537c1d9666a2594453014f34e",
    "PromptProcessingBatch._copy": "f8e615486dc810373f1ddd7b64adbd69d33376bb091b19edafeef55629e604a7",
    "PromptProcessingBatch.filter": "d93944a0087ffc72b4b1becc149a4cc9a6a211d96076ebdf07e1ca9c777ce4ea",
    "PromptProcessingBatch.split": "e23eda5b0afc33ba2a439cc30fea882fccb9c56baa556929c9cf2ed059a64062",
    "BatchGenerator._make_batch": "76dba0a9c2477ed64dbd4b72cde2379da3d88a763c1894a10ea26c210d05c499",
}


def _host_zero(padding):
    return padding is None or (
        type(padding) in (list, tuple)
        and all(type(value) is int and value == 0 for value in padding)
    )


class _UnpaddedCache:
    """Mixin used only on exact, freshly merged BatchKVCache objects."""

    _unpadded = False

    def __setattr__(self, name, value):
        if name == "left_padding":
            self._unpadded = False
        super().__setattr__(name, value)

    def make_mask(self, N, return_array=False, **kwargs):  # noqa: N803 - upstream keyword API
        if self._unpadded and not (kwargs.keys() - {"window_size"}):
            return self._plain_mask(
                N, self._idx, return_array, kwargs.get("window_size")
            )
        return super().make_mask(N, return_array=return_array, **kwargs)

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        proven = self._unpadded and _host_zero(left_padding)
        proven = proven and _host_zero(right_padding)
        super().prepare(
            left_padding=left_padding, lengths=lengths, right_padding=right_padding
        )
        self._unpadded = proven

    def finalize(self):
        if self._right_padding is not None:
            self._unpadded = False
        super().finalize()

    def trim(self, n):
        self._unpadded = False
        return super().trim(n)

    def filter(self, batch_indices):
        if (
            self._unpadded
            and type(batch_indices) is list
            and all(type(index) is int for index in batch_indices)
            and batch_indices == list(range(self.offset.shape[0]))
        ):
            return
        self._unpadded = False
        return super().filter(batch_indices)

    def extend(self, other):
        proven = (
            type(other) is type(self)
            and self._unpadded
            and other._unpadded
            and self._idx == other._idx
        )
        super().extend(other)
        self._unpadded = proven


def _clone(function, **overrides):
    """Reuse upstream code with private globals, without a global monkeypatch."""
    return FunctionType(
        function.__code__,
        function.__globals__ | overrides,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )


def _attach(generator, gen, cache):
    class UnpaddedBatchKVCache(_UnpaddedCache, cache.BatchKVCache):
        _plain_mask = staticmethod(cache.create_attention_mask)

    def merge(caches):
        # Same layer merge dispatch as upstream; never intercept non-plain KV.
        merged = []
        for layers in zip(*caches):
            result = layers[0].merge(list(layers))
            if type(result) is cache.BatchKVCache and all(
                type(layer) is cache.KVCache for layer in layers
            ):
                lengths = [layer.size() for layer in layers]
                result.__class__ = UnpaddedBatchKVCache
                result._unpadded = len(set(lengths)) == 1
            merged.append(result)
        return merged

    class PromptBatch(gen.PromptProcessingBatch):
        __init__ = _clone(gen.PromptProcessingBatch.__init__, _merge_caches=merge)

    make_batch = _clone(
        gen.BatchGenerator._make_batch, PromptProcessingBatch=PromptBatch
    )
    generator._make_batch = MethodType(make_batch, weakref.proxy(generator))
    generator._rapid_unpadded_fastpath = True


def install_batchgen_fastpath(generator: Any) -> bool:
    """Opt one ordinary text generator in; silently retain stock on API drift."""
    if os.environ.get("RAPID_MLX_BATCHGEN_FASTPATH") == "0":
        return False
    try:
        if version("mlx-lm") != "0.31.3":
            return False
        gen = importlib.import_module("mlx_lm.generate")
        cache = importlib.import_module("mlx_lm.models.cache")
        if type(generator) is not gen.BatchGenerator:
            return False
        if (
            getattr(generator._make_batch, "__func__", None)
            is not gen.BatchGenerator._make_batch
        ):
            return False
        for path, expected in _SOURCE_HASHES.items():
            target: Any = (
                gen
                if path.startswith(("PromptProcessingBatch.", "BatchGenerator."))
                else cache
            )
            for part in path.split("."):
                target = getattr(target, part)
            if inspect.unwrap(target) is not target:
                return False
            source = textwrap.dedent(inspect.getsource(target))
            if hashlib.sha256(source.encode()).hexdigest() != expected:
                return False
    except (
        PackageNotFoundError,
        ImportError,
        AttributeError,
        OSError,
        TypeError,
        KeyError,
        ValueError,
    ):
        return False
    _attach(generator, gen, cache)
    return True
