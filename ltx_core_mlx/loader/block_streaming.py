"""Block streaming for low-RAM inference on Apple Silicon.

Stream transformer block weights from a memory-mapped safetensors file
into a single shared block module, so peak resident memory stays at
``~1 block`` instead of ``~num_blocks``. MLX-native equivalent of the
upstream ``ltx_core.block_streaming`` package, but radically simpler:

- Apple Silicon has unified memory, so there is no host-to-device copy.
- ``mx.load(path)`` memory-maps safetensors lazily — opening a 20 GB
  file costs ~40 MB RSS until individual arrays are touched.
- MLX has a single command queue, so there are no streams or events to
  coordinate. Per-block ``mx.clear_cache()`` + dropping references is
  sufficient to keep the resident set bounded.

Architecture
------------
- :class:`BlockStreamer` holds the mmap'd weight dict and a key map
  per block. It exposes :meth:`bind`, which loads block ``i``'s weights
  into a caller-provided ``BasicAVTransformerBlock`` instance.
- A pipeline uses a single shared block module and a :class:`BlockStreamer`,
  passing ``block_provider`` to :meth:`LTXModel.__call__` so the iteration
  loop fetches the bound block per index.
- LoRA fusion can be done up-front into the safetensors-derived dict.
  We do not yet support on-the-fly LoRA fusion per block; the upstream
  cost (matmul on GPU per block) is small but the plumbing is.

Memory profile (LTX-2.3 22B bf16)
---------------------------------
Without streaming: ~22 GB resident for the transformer (48 x ~460 MB).
With streaming: ~460 MB for the active block + non-block params (a few
hundred MB) + mmap metadata (~50 MB) ≈ ~1 GB.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

__all__ = ["BlockLoraSource", "BlockStreamer", "StreamingLTXModel"]


class BlockLoraSource:
    """Per-block LoRA A/B matrices indexed by block index.

    Streams LoRA A/B pairs from a memory-mapped safetensors file and
    indexes them by block index so :meth:`BlockStreamer.bind` can fuse
    the matching delta into a streamed block on the fly.

    Args:
        lora_path: Path to LoRA safetensors.
        block_prefix: Prefix the LoRA keys carry **after** optional
            ``sd_ops`` remapping — keys must start with this and be of the
            form ``f"{block_prefix}{idx}.{param}.lora_A.weight"``. NOTE this
            is NOT necessarily the streamer's model-weight prefix:
            ``LTXV_LORA_COMFY_RENAMING_MAP`` produces ``transformer_blocks.``
            (no ``transformer.``), so callers using that map must pass
            ``LTXV_LORA_BLOCK_PREFIX``, not the ``transformer.``-prefixed
            ``BlockStreamer`` prefix. Mixing them silently drops every delta (#52).
        strength: LoRA fusion strength (default 1.0).
        sd_ops: Optional :class:`SDOps` to remap raw safetensors keys
            (e.g. ComfyUI/diffusers → MLX naming via
            ``LTXV_LORA_COMFY_RENAMING_MAP``).
    """

    def __init__(
        self,
        lora_path: str | Path,
        block_prefix: str,
        strength: float = 1.0,
        sd_ops=None,
    ) -> None:
        self.strength = strength
        self.block_prefix = block_prefix
        self._lora_path = str(lora_path)
        self._sd_ops = sd_ops
        self._lora_data = mx.load(self._lora_path)

        # block_idx -> param_name -> {"a": full_key, "b": full_key}
        self._block_keys: dict[int, dict[str, dict[str, str]]] = {}
        for raw_key in self._lora_data:
            model_key = sd_ops.apply_to_key(raw_key) if sd_ops is not None else raw_key
            if model_key is None or not model_key.startswith(block_prefix):
                continue
            rest = model_key[len(block_prefix) :]
            idx_str, _, param_path = rest.partition(".")
            try:
                block_idx = int(idx_str)
            except ValueError:
                continue
            for suffix, slot in ((".lora_A.weight", "a"), (".lora_B.weight", "b")):
                if param_path.endswith(suffix):
                    param_name = param_path[: -len(suffix)]
                    self._block_keys.setdefault(block_idx, {}).setdefault(param_name, {})[slot] = raw_key
                    break

    def has_block(self, block_idx: int) -> bool:
        """Return True iff the LoRA contains at least one matched A/B pair for block_idx."""
        block = self._block_keys.get(block_idx)
        if not block:
            return False
        return any("a" in slots and "b" in slots for slots in block.values())

    def get_block_lora_dict(self, block_idx: int) -> dict[str, mx.array]:
        """Return per-block LoRA dict shaped like ``{param.lora_A.weight: array, ...}``.

        The keys use the param name relative to the block (the same keys
        that :meth:`BlockStreamer.bind` would load). ``apply_loras`` from
        ``ltx_core_mlx.loader.fuse_loras`` consumes this format directly.
        """
        out: dict[str, mx.array] = {}
        block = self._block_keys.get(block_idx, {})
        for param_name, slots in block.items():
            if "a" not in slots or "b" not in slots:
                continue
            out[f"{param_name}.lora_A.weight"] = self._lora_data[slots["a"]]
            out[f"{param_name}.lora_B.weight"] = self._lora_data[slots["b"]]
        return out

    def close(self) -> None:
        self._lora_data = {}
        self._block_keys = {}


class BlockStreamer:
    """Stream transformer block weights from mmap'd safetensors.

    Args:
        weight_paths: One or more ``.safetensors`` paths whose union
            contains every key for the streamed blocks. Loaded via
            :func:`mlx.core.load` which memory-maps the file.
        block_prefix: State-dict key prefix that identifies block
            weights, e.g. ``"transformer.transformer_blocks."``. Keys
            of the form ``f"{block_prefix}{i}.{rest}"`` (where ``i`` is
            an integer) are treated as block ``i``'s weights.

    Notes:
        After construction, :meth:`block_count` returns the number of
        distinct block indices found, and :meth:`block_keys(i)` lists
        the per-block parameter names that will be bound by
        :meth:`bind`. The mmap'd dict is held until :meth:`close` is
        called.
    """

    def __init__(
        self,
        weight_paths: str | Path | Iterable[str | Path],
        block_prefix: str,
    ) -> None:
        if isinstance(weight_paths, str | Path):
            weight_paths = [weight_paths]
        self._weight_paths = [str(p) for p in weight_paths]
        self._block_prefix = block_prefix

        # Merge all safetensors files. mx.load mmaps each one, so cost is
        # roughly metadata-only until individual arrays are touched.
        self._weights = self._reload_dict()

        # Build per-block key map: idx -> list[(full_key, param_name)].
        self._block_key_map: dict[int, list[tuple[str, str]]] = {}
        for full_key in self._weights:
            if not full_key.startswith(self._block_prefix):
                continue
            rest = full_key[len(self._block_prefix) :]
            idx_str, _, param_name = rest.partition(".")
            try:
                block_idx = int(idx_str)
            except ValueError:
                continue
            self._block_key_map.setdefault(block_idx, []).append((full_key, param_name))

        if not self._block_key_map:
            raise ValueError(
                f"BlockStreamer found no keys matching prefix {block_prefix!r} "
                f"in {weight_paths!r}. Check the prefix or the safetensors content."
            )

    @property
    def block_count(self) -> int:
        """Number of distinct block indices discovered in the safetensors."""
        return len(self._block_key_map)

    @property
    def block_prefix(self) -> str:
        return self._block_prefix

    def block_keys(self, idx: int) -> list[str]:
        """Per-block parameter names (without the ``{prefix}{idx}.`` part)."""
        if idx not in self._block_key_map:
            raise KeyError(f"block {idx} not in streamer (have {sorted(self._block_key_map)})")
        return [param_name for _full, param_name in self._block_key_map[idx]]

    def bind(
        self,
        block: nn.Module,
        idx: int,
        evict_previous: int | None = None,
        lora_sources: list[BlockLoraSource] | None = None,
    ) -> None:
        """Load block ``idx``'s weights into ``block`` in-place.

        Internally calls :func:`mlx.nn.Module.load_weights`. After this
        returns, ``block``'s parameters reference the safetensors-mapped
        arrays for index ``idx``. A subsequent :meth:`bind` to a
        different ``idx`` rebinds them, releasing the previous arrays
        from the block's parameter tree.

        Args:
            block: Target ``BasicAVTransformerBlock``-shaped module.
                Its parameter tree must match the keys returned by
                :meth:`block_keys`.
            idx: Block index to load.
            evict_previous: If given, drop the cached array references
                for that block index from the internal weight dict
                before binding. The streamer holds refs to every
                block's weights; without eviction, those refs prevent
                GC even after the bound block is replaced. For
                streaming inference, pass the previously-bound index
                so peak resident memory stays at ~one block.
        """
        if idx not in self._block_key_map:
            raise KeyError(f"block {idx} not in streamer")
        if evict_previous is not None and evict_previous in self._block_key_map:
            for full_key, _ in self._block_key_map[evict_previous]:
                self._weights.pop(full_key, None)
        # Re-mmap if the requested block's keys have already been evicted
        # (typical after a full forward sweep through all 48 blocks).
        sample_key = self._block_key_map[idx][0][0]
        if sample_key not in self._weights:
            self._weights = self._reload_dict()
        weights = [(param_name, self._weights[full_key]) for full_key, param_name in self._block_key_map[idx]]

        if lora_sources:
            weights = self._fuse_lora_into_block(weights, idx, lora_sources)

        block.load_weights(weights, strict=True)

    @staticmethod
    def _fuse_lora_into_block(
        weights: list[tuple[str, mx.array]],
        idx: int,
        lora_sources: list[BlockLoraSource],
    ) -> list[tuple[str, mx.array]]:
        """Apply each source's LoRA delta to the block's weights.

        Reuses :func:`ltx_core_mlx.loader.fuse_loras.apply_loras`, which
        is already quantization-aware (dequantize → fuse → re-quantize
        for q8 / q4 layers). Builds a per-block ``StateDict`` that
        contains only this block's parameters and feeds it through.
        """
        # Lazy-import to avoid loader↔block_streaming cycles at module load.
        from ltx_core_mlx.loader.fuse_loras import apply_loras
        from ltx_core_mlx.loader.primitives import LoraStateDictWithStrength, StateDict

        block_sd_dict: dict[str, mx.array] = dict(weights)
        block_sd = StateDict(sd=block_sd_dict, size=0, dtype=set())

        lora_sd_and_strengths: list[LoraStateDictWithStrength] = []
        for src in lora_sources:
            if not src.has_block(idx):
                continue
            lsd = StateDict(sd=src.get_block_lora_dict(idx), size=0, dtype=set())
            lora_sd_and_strengths.append(LoraStateDictWithStrength(lsd, src.strength))

        if not lora_sd_and_strengths:
            return weights

        fused = apply_loras(block_sd, lora_sd_and_strengths)
        return list(fused.sd.items())

    def _reload_dict(self) -> dict[str, mx.array]:
        """Re-mmap all weight files into a fresh dict."""
        merged: dict[str, mx.array] = {}
        for path in self._weight_paths:
            merged.update(mx.load(path))
        return merged

    def close(self) -> None:
        """Release the mmap'd dict. After this the streamer is unusable."""
        self._weights = {}
        self._block_key_map = {}


class StreamingLTXModel(nn.Module):
    """Drop-in LTXModel replacement that streams transformer blocks.

    Wraps an ``LTXModel`` whose ``transformer_blocks`` list has been
    truncated to a single block, and a :class:`BlockStreamer` over the
    full transformer safetensors. On ``__call__`` it builds a
    block_provider that binds each requested block index into the
    single shared block before the model runs forward.

    Constraints:
        - The wrapped ``LTXModel`` must have exactly one entry in
          ``transformer_blocks``. The pipeline is responsible for
          dropping the other ``num_layers - 1`` blocks before
          constructing this wrapper (otherwise quantization +
          materialization at load time defeats the streaming goal).
        - The wrapped model's ``config.num_layers`` is the iteration
          count seen by the block_provider; the streamer must contain
          that many block indices.

    Args:
        model: Underlying ``LTXModel`` with one block + non-block
            weights already loaded.
        streamer: ``BlockStreamer`` over the transformer safetensors.

    Attribute proxying:
        Unknown attribute reads are forwarded to the wrapped model so
        callers (e.g. ``X0Model``) can read ``self.config`` etc.
    """

    def __init__(
        self,
        model: nn.Module,
        streamer: BlockStreamer,
        lora_sources: list[BlockLoraSource] | None = None,
    ) -> None:
        super().__init__()
        # Register the inner model as a child so its parameters are
        # visible to nn.Module machinery (load_weights, parameters()).
        # Use the alias ``inner`` instead of ``_model`` to avoid the
        # leading-underscore-name shadowing nn.Module would attempt
        # via __getattr__.
        self.inner = model
        shared = model.transformer_blocks[0]
        # Pre-compile the shared block forward. Passing ``inputs=shared``
        # tells mx.compile that the block's parameters can vary between
        # calls, so streamer.bind() rebinds the weights without
        # invalidating the compiled graph. Compiled kernels make each
        # block forward fast enough that 48 sequential mx.eval syncs
        # don't trip the macOS Metal "Impacting Interactivity" watchdog.
        # Empirically this also halves peak Metal (~2.8 GB vs ~4.4 GB).
        compiled = mx.compile(shared, inputs=shared)
        # Store the streamer + shared block + compiled fn in the
        # instance __dict__ directly so they don't show up in
        # parameters().
        object.__setattr__(self, "_streamer", streamer)
        object.__setattr__(self, "_shared_block", shared)
        object.__setattr__(self, "_compiled_block", compiled)
        object.__setattr__(self, "_lora_sources", lora_sources or [])

    def __call__(self, *args, **kwargs):
        # Inject block_provider unless caller already passed one.
        if kwargs.get("block_provider") is None:
            streamer = object.__getattribute__(self, "_streamer")
            shared = object.__getattribute__(self, "_shared_block")
            compiled = object.__getattribute__(self, "_compiled_block")
            lora_sources = object.__getattribute__(self, "_lora_sources")
            prev_idx: list[int | None] = [None]

            # mx.compile can only trace functions that take pytrees of
            # arrays / constants. ``perturbations`` (a custom dataclass)
            # passed through the model's __call__ to each block breaks
            # tracing. Fall back to the eager block when guidance
            # perturbations are active. This loses the compile speedup
            # but the eager block's per-step latency is dominated by
            # attention compute, so the regression is small (~5-10%).
            use_compiled = kwargs.get("perturbations") is None

            def provider(idx: int) -> nn.Module:
                streamer.bind(
                    shared,
                    idx,
                    evict_previous=prev_idx[0],
                    lora_sources=lora_sources or None,
                )
                prev_idx[0] = idx
                return compiled if use_compiled else shared

            kwargs["block_provider"] = provider
        return self.inner(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = super().__getattr__("inner")
            return getattr(inner, name)
