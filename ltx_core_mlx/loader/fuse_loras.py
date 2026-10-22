"""LoRA weight fusion into model state dicts.

Ported from ltx-core/src/ltx_core/loader/fuse_loras.py

Adapted for MLX:
- Uses mx.array instead of torch.Tensor
- Handles MLX quantized weights (int4/int8 with scales/biases) instead of FP8
- No CUDA-specific paths
"""

from __future__ import annotations

import mlx.core as mx

from ltx_core_mlx.loader.primitives import LoraStateDictWithStrength, StateDict


def apply_loras(
    model_sd: StateDict,
    lora_sd_and_strengths: list[LoraStateDictWithStrength],
    dtype: mx.Dtype | None = None,
    destination_sd: StateDict | None = None,
) -> StateDict:
    """Fuse one or more LoRA weight deltas into a model state dict.

    For each weight key in the model, finds matching lora_A/lora_B pairs
    in the LoRA state dicts, computes delta = B @ A * strength, and adds
    it to the original weight.

    Handles MLX quantized weights (int4/int8): dequantizes, fuses the
    LoRA delta, then re-quantizes.

    Args:
        model_sd: Base model state dict.
        lora_sd_and_strengths: List of (lora_state_dict, strength) pairs.
        dtype: Target dtype for fused weights. If None, uses source dtype.
        destination_sd: Optional existing dict to merge into.

    Returns:
        New StateDict with fused weights.
    """
    sd: dict[str, mx.array] = {}
    if destination_sd is not None:
        sd = dict(destination_sd.sd)

    size = 0
    dtypes: set[mx.Dtype] = set()

    for key, weight in model_sd.sd.items():
        if weight is None:
            continue
        # Skip quantization metadata keys — handled with their weight
        if key.endswith(".scales") or key.endswith(".biases"):
            continue

        target_dtype = dtype if dtype is not None else weight.dtype

        # Check if this weight is quantized (has scales)
        scales_key = f"{key[: -len('.weight')]}.scales" if key.endswith(".weight") else None
        biases_key = f"{key[: -len('.weight')]}.biases" if key.endswith(".weight") else None
        is_quantized = scales_key is not None and scales_key in model_sd.sd

        deltas = _prepare_deltas(lora_sd_and_strengths, key)
        fused = _fuse_deltas(
            deltas,
            weight,
            key,
            target_dtype,
            is_quantized,
            scales_key,
            biases_key,
            model_sd,
        )

        sd.update(fused)
        for tensor in fused.values():
            dtypes.add(tensor.dtype)
            size += tensor.nbytes

    if destination_sd is not None:
        return StateDict(sd=sd, size=size, dtype=dtypes)
    return StateDict(sd=sd, size=size, dtype=dtypes)


def _prepare_deltas(
    lora_sd_and_strengths: list[LoraStateDictWithStrength],
    key: str,
) -> mx.array | None:
    """Compute the combined LoRA delta for a given weight key.

    Looks for matching lora_A.weight and lora_B.weight keys and computes
    delta = sum(B_i @ A_i * strength_i) for all matching LoRAs.

    Args:
        lora_sd_and_strengths: List of (lora_state_dict, strength) pairs.
        key: The model weight key to find LoRA deltas for.

    Returns:
        Combined delta array, or None if no LoRA matches this key.
    """
    deltas = []
    prefix = key[: -len(".weight")] if key.endswith(".weight") else key
    key_a = f"{prefix}.lora_A.weight"
    key_b = f"{prefix}.lora_B.weight"

    for lsd, coef in lora_sd_and_strengths:
        if key_a not in lsd.sd or key_b not in lsd.sd:
            continue
        a = lsd.sd[key_a].astype(mx.float32)
        b = lsd.sd[key_b].astype(mx.float32)
        product = mx.matmul(b * coef, a)
        deltas.append(product)

    if len(deltas) == 0:
        return None
    if len(deltas) == 1:
        return deltas[0]
    return mx.sum(mx.stack(deltas, axis=0), axis=0)


def _fuse_deltas(
    deltas: mx.array | None,
    weight: mx.array,
    key: str,
    target_dtype: mx.Dtype,
    is_quantized: bool,
    scales_key: str | None,
    biases_key: str | None,
    model_sd: StateDict,
) -> dict[str, mx.array]:
    """Fuse LoRA deltas into a weight, handling quantized and non-quantized cases.

    Args:
        deltas: Combined LoRA delta, or None if no LoRA applies.
        weight: Original model weight.
        key: Weight key name.
        target_dtype: Target dtype for the fused weight.
        is_quantized: Whether this weight is int4/int8 quantized.
        scales_key: Key for quantization scales (if quantized).
        biases_key: Key for quantization biases (if quantized).
        model_sd: Full model state dict (for accessing scales/biases).

    Returns:
        Dict of fused weight entries (may include scales/biases).
    """
    if deltas is None:
        # No LoRA for this key — copy original weight
        result = {key: weight.astype(target_dtype)}
        if is_quantized and scales_key:
            result[scales_key] = model_sd.sd[scales_key]
            if biases_key and biases_key in model_sd.sd:
                result[biases_key] = model_sd.sd[biases_key]
        return result

    if is_quantized:
        return _fuse_delta_with_quantized(deltas, weight, key, scales_key, biases_key, model_sd)
    return _fuse_delta_with_float(deltas, weight, key, target_dtype)


def _fuse_delta_with_quantized(
    deltas: mx.array,
    weight: mx.array,
    key: str,
    scales_key: str | None,
    biases_key: str | None,
    model_sd: StateDict,
) -> dict[str, mx.array]:
    """Fuse LoRA delta with a quantized weight.

    Dequantizes the weight, adds the LoRA delta, then re-quantizes.
    MLX quantized weights are stored as (out_features, in_features_packed)
    with separate scales and optional biases per group.

    Args:
        deltas: LoRA delta in float32.
        weight: Quantized weight array.
        key: Weight key name.
        scales_key: Key for quantization scales.
        biases_key: Key for quantization biases.
        model_sd: Full model state dict.

    Returns:
        Dict with re-quantized weight, scales, and biases.
    """
    scales = model_sd.sd[scales_key] if scales_key else None
    biases = model_sd.sd.get(biases_key) if biases_key else None

    # Infer quantization parameters BEFORE dequantizing.
    # MLX packs N-bit values into 32-bit ints: pack_factor = 32 / bits.
    # For a weight (O, packed) and scales (O, num_groups):
    #   real_features = num_groups * group_size
    #   packed_features = real_features / pack_factor
    # So: pack_factor = real_features / packed_features
    #     bits = 32 / pack_factor
    in_features_packed = weight.shape[-1]
    if scales is not None and scales.ndim > 1:
        num_groups = scales.shape[1]
        # Try group_size=64 (default), compute real features, check consistency
        group_size = 64
        in_features_real = num_groups * group_size
        pack_factor = in_features_real / in_features_packed if in_features_packed > 0 else 4
        bits = max(2, min(8, round(32 / pack_factor)))
    else:
        group_size = 64
        bits = 8

    # Dequantize: reconstruct the float weight from quantized representation
    original = mx.dequantize(
        weight,
        scales=scales,
        biases=biases,
        group_size=group_size,
        bits=bits,
    ).astype(mx.float32)

    # Fuse
    new_weight = original + deltas.astype(mx.float32)

    new_quantized, new_scales, new_biases = mx.quantize(new_weight, group_size=group_size, bits=bits)

    result = {key: new_quantized}
    if scales_key:
        result[scales_key] = new_scales
    if biases_key:
        result[biases_key] = new_biases
    return result


def _fuse_delta_with_float(
    deltas: mx.array,
    weight: mx.array,
    key: str,
    target_dtype: mx.Dtype,
) -> dict[str, mx.array]:
    """Fuse LoRA delta with a float (non-quantized) weight.

    Args:
        deltas: LoRA delta.
        weight: Original float weight.
        key: Weight key name.
        target_dtype: Target dtype for the fused weight.

    Returns:
        Dict with the fused weight.
    """
    fused = (weight.astype(mx.float32) + deltas.astype(mx.float32)).astype(target_dtype)
    return {key: fused}
