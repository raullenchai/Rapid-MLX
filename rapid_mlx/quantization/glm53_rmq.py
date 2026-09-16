"""Apple-native mixed-precision planning for GLM-5.3-Flash.

RMQ deliberately emits only MLX affine Q4/Q8 group-64 weights.  The small
format vocabulary keeps generated checkpoints on MLX's native quantized
matmul path.  Plans are also grouped by runtime fusion domain: a sensitivity
override may raise a whole domain, but can never silently split the six KDA
input projections or the experts consumed by one ``SwitchLinear``.

This module plans a conversion. It does not claim that an already-repacked
integer checkpoint can regain information; conversion must start from BF16 or
the official block-FP8 source representation.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.(down_proj|gate_proj|up_proj)\.weight$")

_KDA_FUSED_INPUTS = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.f_a_proj.weight",
    ".self_attn.g_a_proj.weight",
    ".self_attn.b_proj.weight",
)

_FLOAT_DTYPES = frozenset({"BF16", "F16", "F32"})


@dataclass(frozen=True)
class QuantSpec:
    """One MLX-native storage decision."""

    bits: int | None
    group_size: int | None = None
    mode: str | None = None
    reason: str = ""

    @property
    def quantized(self) -> bool:
        return self.bits is not None

    def as_config(self) -> dict[str, int | str]:
        if self.bits is None or self.group_size is None or self.mode is None:
            raise ValueError("floating-point tensors have no quantization config")
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "mode": self.mode,
        }


@dataclass(frozen=True)
class TensorDescriptor:
    name: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def parameters(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class TensorPlan:
    tensor: TensorDescriptor
    spec: QuantSpec
    fusion_domain: str | None


def _q(bits: int, reason: str) -> QuantSpec:
    if bits not in (4, 8):
        raise ValueError(f"RMQ supports only Q4/Q8, got Q{bits}")
    return QuantSpec(bits=bits, group_size=64, mode="affine", reason=reason)


def _fp(reason: str) -> QuantSpec:
    return QuantSpec(bits=None, reason=reason)


def layer_index(name: str) -> int | None:
    match = _LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def module_paths(name: str) -> tuple[str, ...]:
    """Translate one source weight to its post-sanitize mlx-vlm modules."""
    path = name.removesuffix(".weight")
    if path.startswith("model.language_model."):
        path = "language_model.model." + path.removeprefix("model.language_model.")
    elif path == "lm_head":
        path = "language_model.lm_head"
    expert = re.search(r"\.mlp\.experts\.\d+\.(down_proj|gate_proj|up_proj)$", path)
    if expert:
        prefix = path[: expert.start()]
        return (f"{prefix}.mlp.switch_mlp.{expert.group(1)}",)
    if path.endswith(".self_attn.kv_b_proj"):
        prefix = path.removesuffix(".kv_b_proj")
        return (f"{prefix}.embed_q", f"{prefix}.unembed_out")
    for projection in ("f_a_proj", "f_b_proj"):
        old = f".self_attn.{projection}"
        if path.endswith(old):
            path = path.removesuffix(old) + f".self_attn.forget_gate.{projection}"
    return (path,)


def module_path(name: str) -> str:
    """Return the primary post-sanitize module path for compatibility."""
    return module_paths(name)[0]


def mtp_module_paths(name: str, config: Mapping) -> tuple[str, ...]:
    """Return layer-45 paths after extraction into the standalone drafter."""
    idx = layer_index(name)
    text = config.get("text_config") or {}
    n_layers = int(
        text.get("num_hidden_layers") or config.get("num_hidden_layers") or 0
    )
    if idx is None or not n_layers or idx < n_layers:
        return ()
    marker = f".layers.{idx}."
    suffix = name.split(marker, 1)[1].removesuffix(".weight")
    if suffix in ("enorm", "hnorm", "eh_proj"):
        return (suffix,)
    if suffix == "shared_head.norm":
        return ("shared_head_norm",)
    if re.fullmatch(r"mlp\.experts\.\d+\.(down_proj|gate_proj|up_proj)", suffix):
        projection = suffix.rsplit(".", 1)[1]
        return (f"mtp_block.mlp.switch_mlp.{projection}",)
    if suffix in ("mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj"):
        return ("mtp_block.mlp.shared_experts.gate_up_proj",)
    if suffix == "self_attn.kv_b_proj":
        return (
            "mtp_block.self_attn.embed_q",
            "mtp_block.self_attn.unembed_out",
        )
    if suffix in ("self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa"):
        return ("mtp_block.self_attn.qkv_a_proj",)
    return (f"mtp_block.{suffix}",)


def fusion_domain(name: str) -> str | None:
    """Return the runtime unit that must retain homogeneous quantization."""
    idx = layer_index(name)
    if idx is None:
        return None
    if name.endswith(_KDA_FUSED_INPUTS):
        return f"layer:{idx}:kda-fused-input"
    expert = _EXPERT_RE.search(name)
    if expert:
        return f"layer:{idx}:routed:{expert.group(1)}"
    return None


def _score_for(
    name: str,
    domain: str | None,
    sensitivity: Mapping[str, float],
) -> float:
    values = [float(sensitivity.get(name, 0.0))]
    module = module_path(name)
    values.append(float(sensitivity.get(module, 0.0)))
    if domain is not None:
        values.append(float(sensitivity.get(domain, 0.0)))
    score = max(values)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"sensitivity for {name} must be finite and within [0, 1]")
    return score


def _base_spec(tensor: TensorDescriptor, config: Mapping) -> QuantSpec:
    name, shape, dtype = tensor.name, tensor.shape, tensor.dtype.upper()
    text = config.get("text_config") or {}
    n_layers = int(
        text.get("num_hidden_layers") or config.get("num_hidden_layers") or 0
    )
    idx = layer_index(name)

    if dtype not in _FLOAT_DTYPES:
        return _fp("non-floating state or buffer")
    if not name.endswith(".weight") or len(shape) != 2:
        return _fp("non-matrix parameter")
    if shape[-1] % 64:
        return _fp("input width is not divisible by the native group size")
    if name.startswith("model.visual."):
        return _fp("preserve the vision tower in source precision")
    if name.endswith(".mlp.gate.weight"):
        return _fp("router logits remain in source precision")

    # A preserved GLM MTP block is stored immediately after the trunk layers.
    # Its routed experts still dominate its storage and decode cost, so keep
    # those Q4 exactly like the target routed experts.  The always-active MTP
    # glue, attention, and shared expert take the Q8 fidelity floor below.
    # Target/draft precision symmetry matters: independently raising only the
    # drafter can reduce acceptance by removing errors shared with the target.
    if idx is not None and n_layers and idx >= n_layers:
        if _EXPERT_RE.search(name):
            return _q(4, "target-matched MTP routed expert")
        return _q(8, "MTP always-active fidelity floor")

    if name == "lm_head.weight" or name.endswith(".embed_tokens.weight"):
        return _q(8, "token ranking and embedding fidelity")

    if ".self_attn.indexer." in name and name.endswith(
        (".wq_b.weight", ".wk.weight", ".weights_proj.weight")
    ):
        return _q(8, "sparse-index top-k selection fidelity")

    if name.endswith(_KDA_FUSED_INPUTS):
        return _q(8, "KDA recurrent input and fusion domain")
    if any(
        part in name
        for part in (
            ".self_attn.f_b_proj.weight",
            ".self_attn.g_b_proj.weight",
            ".self_attn.o_proj.weight",
            ".self_attn.q_a_proj.weight",
            ".self_attn.q_b_proj.weight",
            ".self_attn.kv_a_proj_with_mqa.weight",
            ".self_attn.kv_b_proj.weight",
        )
    ):
        return _q(8, "attention or recurrent-state fidelity")

    if ".mlp.shared_experts." in name:
        return _q(8, "always-active shared expert")
    if _EXPERT_RE.search(name):
        return _q(4, "bandwidth-dominant routed expert")

    if ".mlp." in name and name.endswith(
        (".gate_proj.weight", ".up_proj.weight", ".down_proj.weight")
    ):
        return _q(8, "dense always-active MLP")
    return _q(4, "native affine baseline")


def _raised_bits(base: int, score: float) -> int:
    if score >= 0.55:
        return 8
    return base


def plan_tensors(
    tensors: Sequence[TensorDescriptor],
    config: Mapping,
    sensitivity: Mapping[str, float] | None = None,
) -> list[TensorPlan]:
    """Build a deterministic RMQ plan, preserving every fusion invariant."""
    sensitivity = sensitivity or {}
    provisional: list[TensorPlan] = []
    domain_bits: dict[str, int] = {}

    for tensor in tensors:
        domain = fusion_domain(tensor.name)
        spec = _base_spec(tensor, config)
        if spec.quantized:
            assert spec.bits is not None
            score = _score_for(tensor.name, domain, sensitivity)
            spec = _q(_raised_bits(spec.bits, score), spec.reason)
            if domain is not None:
                assert spec.bits is not None
                domain_bits[domain] = max(domain_bits.get(domain, 0), spec.bits)
        provisional.append(TensorPlan(tensor, spec, domain))

    result: list[TensorPlan] = []
    for item in provisional:
        if item.spec.quantized and item.fusion_domain is not None:
            bits = domain_bits[item.fusion_domain]
            item = TensorPlan(
                item.tensor,
                _q(bits, item.spec.reason + "; fusion-domain locked"),
                item.fusion_domain,
            )
        result.append(item)
    return result


def projected_storage_bytes(plan: Sequence[TensorPlan]) -> int:
    """Estimate tensor payload bytes, including affine scale/bias tables."""
    total = 0
    for item in plan:
        params = item.tensor.parameters
        if not item.spec.quantized:
            bytes_per = 4 if item.tensor.dtype.upper() == "F32" else 2
            total += params * bytes_per
            continue
        assert item.spec.bits is not None and item.spec.group_size is not None
        bits = item.spec.bits
        groups = params // item.spec.group_size
        total += math.ceil(params * bits / 8) + groups * 4
    return total


def summarize_plan(plan: Sequence[TensorPlan]) -> dict[str, object]:
    by_format: dict[str, dict[str, int]] = {}
    for item in plan:
        label = (
            "fp"
            if not item.spec.quantized
            else f"q{item.spec.bits}-g{item.spec.group_size}"
        )
        bucket = by_format.setdefault(label, {"tensors": 0, "parameters": 0})
        bucket["tensors"] += 1
        bucket["parameters"] += item.tensor.parameters
    return {
        "formats": dict(sorted(by_format.items())),
        "projected_storage_bytes": projected_storage_bytes(plan),
        "fusion_domains": len({p.fusion_domain for p in plan if p.fusion_domain}),
    }


__all__ = [
    "QuantSpec",
    "TensorDescriptor",
    "TensorPlan",
    "fusion_domain",
    "layer_index",
    "module_path",
    "module_paths",
    "mtp_module_paths",
    "plan_tensors",
    "projected_storage_bytes",
    "summarize_plan",
]
