"""State dict key renaming and matching operations.

Ported from ltx-core/src/ltx_core/loader/sd_ops.py
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple, Protocol

import mlx.core as mx


@dataclass(frozen=True, slots=True)
class ContentReplacement:
    """A string replacement to apply to state dict keys."""

    content: str
    replacement: str


@dataclass(frozen=True, slots=True)
class ContentMatching:
    """Prefix/suffix matcher for state dict keys."""

    prefix: str = ""
    suffix: str = ""


class KeyValueOperationResult(NamedTuple):
    """Result of a key-value operation: new key and new value."""

    new_key: str
    new_value: mx.array


class KeyValueOperation(Protocol):
    """Protocol for key-value transformations on state dict entries."""

    def __call__(self, tensor_key: str, tensor_value: mx.array) -> list[KeyValueOperationResult]: ...


@dataclass(frozen=True, slots=True)
class SDKeyValueOperation:
    """A key-value operation bound to a key matcher."""

    key_matcher: ContentMatching
    kv_operation: KeyValueOperation


@dataclass(frozen=True, slots=True)
class SDOps:
    """Immutable class representing state dict key operations.

    Supports key renaming via ContentReplacement, key filtering via
    ContentMatching, and arbitrary key-value transformations via
    SDKeyValueOperation.
    """

    name: str
    mapping: tuple[ContentReplacement | ContentMatching | SDKeyValueOperation, ...] = ()
    allowed_keys: frozenset[str] | None = None

    def with_replacement(self, content: str, replacement: str) -> SDOps:
        """Create a new SDOps with an additional key replacement."""
        new_mapping = (*self.mapping, ContentReplacement(content, replacement))
        return replace(self, mapping=new_mapping)

    def with_matching(self, prefix: str = "", suffix: str = "") -> SDOps:
        """Create a new SDOps with an additional prefix/suffix filter."""
        new_mapping = (*self.mapping, ContentMatching(prefix, suffix))
        return replace(self, mapping=new_mapping)

    def with_additional_allowed_keys(self, keys: frozenset[str]) -> SDOps:
        """Create a new SDOps that only passes keys in *keys* (post-replacement).

        If allowed_keys is already set, the sets are merged via union.
        """
        merged = frozenset(keys) | self.allowed_keys if self.allowed_keys is not None else frozenset(keys)
        return replace(self, allowed_keys=merged)

    def with_kv_operation(
        self,
        operation: KeyValueOperation,
        key_prefix: str = "",
        key_suffix: str = "",
    ) -> SDOps:
        """Create a new SDOps with an additional key-value operation."""
        key_matcher = ContentMatching(key_prefix, key_suffix)
        sd_kv_operation = SDKeyValueOperation(key_matcher, operation)
        new_mapping = (*self.mapping, sd_kv_operation)
        return replace(self, mapping=new_mapping)

    def apply_to_key(self, key: str) -> str | None:
        """Apply key matching and renaming. Returns None if key doesn't match."""
        matchers = [c for c in self.mapping if isinstance(c, ContentMatching)]
        valid = any(key.startswith(f.prefix) and key.endswith(f.suffix) for f in matchers)
        if not valid:
            return None

        for replacement in self.mapping:
            if not isinstance(replacement, ContentReplacement):
                continue
            if replacement.content in key:
                key = key.replace(replacement.content, replacement.replacement)

        if self.allowed_keys is not None and key not in self.allowed_keys:
            return None

        return key

    def apply_to_key_value(self, key: str, value: mx.array) -> list[KeyValueOperationResult]:
        """Apply key-value operations to the given key and value."""
        for operation in self.mapping:
            if not isinstance(operation, SDKeyValueOperation):
                continue
            if key.startswith(operation.key_matcher.prefix) and key.endswith(operation.key_matcher.suffix):
                return operation.kv_operation(key, value)
        return [KeyValueOperationResult(key, value)]


# Predefined SDOps for ComfyUI LoRA key renaming
#
# ComfyUI LoRA keys use diffusers-style naming:
#   diffusion_model.transformer_blocks.N.attn1.to_out.0.lora_{A,B}.weight
#   diffusion_model.transformer_blocks.N.ff.net.0.proj.lora_{A,B}.weight
#   diffusion_model.transformer_blocks.N.ff.net.2.lora_{A,B}.weight
#
# Our model uses:
#   transformer_blocks.N.attn1.to_out.lora_{A,B}.weight
#   transformer_blocks.N.ff.proj_in.lora_{A,B}.weight
#   transformer_blocks.N.ff.proj_out.lora_{A,B}.weight
LTXV_LORA_COMFY_RENAMING_MAP = (
    SDOps("LTXV_LORA_COMFY_PREFIX_MAP")
    .with_matching()
    .with_replacement("diffusion_model.", "")
    .with_replacement(".to_out.0.", ".to_out.")
    .with_replacement(".ff.net.0.proj.", ".ff.proj_in.")
    .with_replacement(".ff.net.2.", ".ff.proj_out.")
    # Audio/joint-block keys: moved here from _remap_lora_keys in ti2vid_two_stages.py
    # so the BlockLoraSource streaming path (which uses this map directly) gets the
    # same remappings as the non-streaming path.
    .with_replacement(".linear_1.", ".linear1.")
    .with_replacement(".linear_2.", ".linear2.")
    .with_replacement("audio_ff.net.0.proj.", "audio_ff.proj_in.")
    .with_replacement("audio_ff.net.2.", "audio_ff.proj_out.")
)

# Block-key prefix that ``LTXV_LORA_COMFY_RENAMING_MAP`` produces.
#
# The map strips ``diffusion_model.`` but never adds ``transformer.``, so a
# renamed key is ``transformer_blocks.N.<param>.lora_{A,B}.weight``. The
# streaming LoRA path (``BlockLoraSource``) keeps a delta only if its renamed
# key starts with the ``block_prefix`` it is given — so every streaming call
# site MUST pass this constant, not the ``transformer.``-prefixed model-weight
# prefix used by ``BlockStreamer`` (whose keys come from ``transformer.safetensors``
# and genuinely carry the ``transformer.`` prefix). Mixing the two silently
# drops every delta, making the LoRA a no-op (#52).
LTXV_LORA_BLOCK_PREFIX = "transformer_blocks."

LTXV_LORA_COMFY_TARGET_MAP = (
    SDOps("LTXV_LORA_COMFY_TARGET_MAP")
    .with_matching()
    .with_replacement("diffusion_model.", "")
    .with_replacement(".to_out.0.", ".to_out.")
    .with_replacement(".ff.net.0.proj.", ".ff.proj_in.")
    .with_replacement(".ff.net.2.", ".ff.proj_out.")
    .with_replacement(".lora_A.weight", ".weight")
    .with_replacement(".lora_B.weight", ".weight")
)
