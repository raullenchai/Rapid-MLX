# SPDX-License-Identifier: Apache-2.0
"""Model recipes: structured benchmark data for hardware-aware recommendations.

Each recipe captures:
- Model identity and architecture
- Memory footprint (weights + KV cache growth)
- Measured performance on reference hardware
- Feature support (tool calling, reasoning, vision, TurboQuant)
- Recommended CLI flags
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelRecipe:
    # Identity
    id: str  # short name for CLI lookup
    name: str  # display name
    model_id: str  # HuggingFace model ID
    provider: str
    architecture: str  # "dense", "moe", "moe-hybrid"
    parameter_count: str  # e.g. "35B"
    active_parameters: str | None  # e.g. "3B" for MoE, None for dense
    quantization: str  # e.g. "4bit", "8bit", "mxfp4"

    # Memory
    model_memory_gb: float  # model weights in memory
    kv_per_1k_tokens_mb: float  # KV cache growth per 1K tokens (MB)
    kv_per_1k_turbo_mb: float | None  # with TurboQuant, None if N/A

    # Measured performance (reference: M3 Ultra 256GB, bandwidth 800 GB/s)
    measured_tps: float  # decode tok/s
    measured_ttft_cached_s: float  # cached TTFT in seconds
    bandwidth_efficiency: float  # measured_tps / theoretical_max (0-1)

    # Features
    tool_calling: bool
    tool_parser: str | None  # e.g. "hermes", "mistral", "glm47"
    reasoning: bool
    reasoning_parser: str | None  # e.g. "qwen3"
    vision: bool
    turboquant_compatible: bool  # whether TurboQuant helps this model

    # Flags
    base_args: list[str] = field(default_factory=list)
    notes: str = ""


# All recipes, keyed by short id.
# Performance data from M3 Ultra 256GB (800 GB/s bandwidth).
# fmt: off
MODEL_RECIPES: dict[str, ModelRecipe] = {}

def _r(recipe: ModelRecipe) -> ModelRecipe:
    MODEL_RECIPES[recipe.id] = recipe
    return recipe

# ── Qwen3.5 family ──────────────────────────────────────────────

_r(ModelRecipe(
    id="qwen3.5-4b",
    name="Qwen3.5 4B",
    model_id="mlx-community/Qwen3.5-4B-MLX-4bit",
    provider="Qwen",
    architecture="moe-hybrid",
    parameter_count="4B",
    active_parameters=None,
    quantization="4bit",
    model_memory_gb=2.7,
    kv_per_1k_tokens_mb=24,  # hybrid: ~75% Mamba (no KV) + 25% attention
    kv_per_1k_turbo_mb=4,
    measured_tps=158,
    measured_ttft_cached_s=0.76,
    bandwidth_efficiency=0.53,
    tool_calling=True,
    tool_parser="hermes",
    reasoning=True,
    reasoning_parser="qwen3",
    vision=True,
    turboquant_compatible=True,
    base_args=["--tool-call-parser", "hermes", "--reasoning-parser", "qwen3"],
    notes="Fastest Qwen3.5. Perfect tool calling. Great for constrained devices.",
))

_r(ModelRecipe(
    id="qwen3.5-9b",
    name="Qwen3.5 9B",
    model_id="mlx-community/Qwen3.5-9B-4bit",
    provider="Qwen",
    architecture="moe-hybrid",
    parameter_count="9B",
    active_parameters=None,
    quantization="4bit",
    model_memory_gb=5.1,
    kv_per_1k_tokens_mb=28,
    kv_per_1k_turbo_mb=5,
    measured_tps=109,
    measured_ttft_cached_s=0.145,
    bandwidth_efficiency=0.55,
    tool_calling=True,
    tool_parser="hermes",
    reasoning=True,
    reasoning_parser="qwen3",
    vision=True,
    turboquant_compatible=True,
    base_args=["--tool-call-parser", "hermes", "--reasoning-parser", "qwen3"],
    notes="Best all-around model. 4.2x faster than Ollama. Perfect tool calling.",
))

_r(ModelRecipe(
    id="qwen3.5-27b",
    name="Qwen3.5 27B (Dense)",
    model_id="mlx-community/Qwen3.5-27B-4bit",
    provider="Qwen",
    architecture="dense",
    parameter_count="27B",
    active_parameters=None,
    quantization="4bit",
    model_memory_gb=14.5,
    kv_per_1k_tokens_mb=64,
    kv_per_1k_turbo_mb=9,
    measured_tps=38.6,
    measured_ttft_cached_s=2.757,
    bandwidth_efficiency=0.53,
    tool_calling=True,
    tool_parser="hermes",
    reasoning=True,
    reasoning_parser="qwen3",
    vision=True,
    turboquant_compatible=True,
    base_args=["--tool-call-parser", "hermes", "--reasoning-parser", "qwen3"],
    notes="Dense 27B — slower than MoE 35B-A3B. Use 35B-A3B unless you need dense behavior.",
))

_r(ModelRecipe(
    id="qwen3.5-35b",
    name="Qwen3.5 35B-A3B (MoE)",
    model_id="mlx-community/Qwen3.5-35B-A3B-4bit",
    provider="Qwen",
    architecture="moe-hybrid",
    parameter_count="35B",
    active_parameters="3B",
    quantization="4bit",
    model_memory_gb=20,
    kv_per_1k_tokens_mb=32,
    kv_per_1k_turbo_mb=5,
    measured_tps=82,
    measured_ttft_cached_s=0.191,
    bandwidth_efficiency=0.54,
    tool_calling=True,
    tool_parser="hermes",
    reasoning=True,
    reasoning_parser="qwen3",
    vision=True,
    turboquant_compatible=True,
    base_args=["--tool-call-parser", "hermes", "--reasoning-parser", "qwen3"],
    notes="Best mid-tier: 2x faster than dense 27B. Perfect tool calling. Recommended.",
))

_r(ModelRecipe(
    id="qwen3.5-35b-8bit",
    name="Qwen3.5 35B-A3B (8bit)",
    model_id="mlx-community/Qwen3.5-35B-A3B-8bit",
    provider="Qwen",
    architecture="moe-hybrid",
    parameter_count="35B",
    active_parameters="3B",
    quantization="8bit",
    model_memory_gb=34.8,
    kv_per_1k_tokens_mb=32,
    kv_per_1k_turbo_mb=5,
    measured_tps=82,
    measured_ttft_cached_s=1.335,
    bandwidth_efficiency=0.54,
    tool_calling=True,
    tool_parser="hermes",
    reasoning=True,
    reasoning_parser="qwen3",
    vision=True,
    turboquant_compatible=True,
    base_args=["--tool-call-parser", "hermes", "--reasoning-parser", "qwen3"],
    notes="Higher quality than 4-bit. Needs 64GB+ Mac.",
))

_r(ModelRecipe(
    id="qwen3.5-122b",
    name="Qwen3.5 122B-A10B (8bit)",
    model_id="mlx-community/Qwen3.5-122B-A10B-8bit",
    provider="Qwen",
    architecture="moe-hybrid",
    parameter_count="122B",
    active_parameters="10B",
    quantization="8bit",
    model_memory_gb=121.3,
    kv_per_1k_tokens_mb=48,
    kv_per_1k_turbo_mb=7,
    measured_tps=44,
    measured_ttft_cached_s=2.399,
    bandwidth_efficiency=0.55,
    tool_calling=True,
    tool_parser="hermes",
    reasoning=True,
    reasoning_parser="qwen3",
    vision=True,
    turboquant_compatible=True,
    base_args=["--tool-call-parser", "hermes", "--reasoning-parser", "qwen3"],
    notes="Best intelligence. 121GB model — needs 192GB+ Mac. Slow TTFT.",
))

# ── Qwen3.6 family ─────────────────────────────────────────────

_r(ModelRecipe(
    id="qwen3.6-27b",
    name="Qwen3.6 27B (Dense Hybrid)",
    model_id="mlx-community/Qwen3.6-27B-4bit",
    provider="Qwen",
    architecture="dense",  # 64 layers: 48 DeltaNet + 16 full attention (every 4th)
    parameter_count="27B",
    active_parameters=None,
    quantization="4bit",
    model_memory_gb=14.9,  # measured: 15301 MB peak RAM
    kv_per_1k_tokens_mb=64,  # measured: 155MB base KV — dense with 16 full attn layers
    kv_per_1k_turbo_mb=9,
    measured_tps=36.5,  # measured: 36.5-36.6 tok/s on M3 Ultra 256GB
    measured_ttft_cached_s=3.16,  # measured: 3.16s cached multi-turn
    bandwidth_efficiency=0.53,
    tool_calling=True,
    tool_parser="hermes",
    reasoning=True,
    reasoning_parser="qwen3",
    vision=True,
    turboquant_compatible=True,
    base_args=["--tool-call-parser", "hermes", "--reasoning-parser", "qwen3"],
    notes="Qwen3.6 dense 27B. 262K native context. Better reasoning than 3.5-27B.",
))

_r(ModelRecipe(
    id="qwen3.6-35b",
    name="Qwen3.6 35B-A3B (MoE)",
    model_id="mlx-community/Qwen3.6-35B-A3B-4bit",
    provider="Qwen",
    architecture="moe-hybrid",
    parameter_count="35B",
    active_parameters="3B",
    quantization="4bit",
    model_memory_gb=19,  # measured: 19.0 GB peak RAM
    kv_per_1k_tokens_mb=32,  # same KV heads/dims as Qwen3.5-35B
    kv_per_1k_turbo_mb=5,
    measured_tps=92,  # measured: 91.8-92.4 tok/s on M3 Ultra 256GB
    measured_ttft_cached_s=1.24,  # measured: 1.235s cached multi-turn
    bandwidth_efficiency=0.54,
    tool_calling=True,
    tool_parser="hermes",
    reasoning=True,
    reasoning_parser="qwen3",
    vision=True,
    turboquant_compatible=True,
    base_args=["--tool-call-parser", "hermes", "--reasoning-parser", "qwen3"],
    notes="Qwen3.6: 12% faster than 3.5-35B. Better reasoning + coding. 262K native context.",
))

_r(ModelRecipe(
    id="qwen3.6-35b-6bit",
    name="Qwen3.6 35B-A3B (6bit)",
    model_id="mlx-community/Qwen3.6-35B-A3B-6bit",
    provider="Qwen",
    architecture="moe-hybrid",
    parameter_count="35B",
    active_parameters="3B",
    quantization="6bit",
    model_memory_gb=28,
    kv_per_1k_tokens_mb=32,
    kv_per_1k_turbo_mb=5,
    measured_tps=72,  # estimated: slightly slower due to larger weights
    measured_ttft_cached_s=0.22,
    bandwidth_efficiency=0.54,
    tool_calling=True,
    tool_parser="hermes",
    reasoning=True,
    reasoning_parser="qwen3",
    vision=True,
    turboquant_compatible=True,
    base_args=["--tool-call-parser", "hermes", "--reasoning-parser", "qwen3"],
    notes="Higher quality than 4-bit. Needs 48GB+ Mac.",
))

# ── Qwen3 Coder ─────────────────────────────────────────────────

_r(ModelRecipe(
    id="qwen3-coder-80b",
    name="Qwen3 Coder Next 80B (4bit)",
    model_id="lmstudio-community/Qwen3-Coder-Next-MLX-4bit",
    provider="Qwen",
    architecture="moe",
    parameter_count="80B",
    active_parameters="3B",
    quantization="4bit",
    model_memory_gb=42.2,
    kv_per_1k_tokens_mb=32,
    kv_per_1k_turbo_mb=5,
    measured_tps=74,
    measured_ttft_cached_s=0.099,
    bandwidth_efficiency=0.54,
    tool_calling=True,
    tool_parser="hermes",
    reasoning=False,
    reasoning_parser=None,
    vision=True,
    turboquant_compatible=True,
    base_args=["--tool-call-parser", "hermes"],
    notes="Excellent coding model. 74 tok/s, better than 6bit variant. Needs 64GB+ Mac.",
))

# ── Hermes / Llama ───────────────────────────────────────────────

_r(ModelRecipe(
    id="hermes-3-8b",
    name="Hermes 3 Llama 3.1 8B",
    model_id="mlx-community/Hermes-3-Llama-3.1-8B-4bit",
    provider="Meta/NousResearch",
    architecture="dense",
    parameter_count="8B",
    active_parameters=None,
    quantization="4bit",
    model_memory_gb=4.7,
    kv_per_1k_tokens_mb=64,
    kv_per_1k_turbo_mb=9,
    measured_tps=123,
    measured_ttft_cached_s=0.080,
    bandwidth_efficiency=0.52,
    tool_calling=False,
    tool_parser=None,
    reasoning=False,
    reasoning_parser=None,
    vision=False,
    turboquant_compatible=True,
    base_args=[],
    notes="Fast (123 tok/s) but 0% tool calling. Text generation only.",
))

# ── Phi-4 ────────────────────────────────────────────────────────

_r(ModelRecipe(
    id="phi4-mini-14b",
    name="Phi-4 Mini 14B",
    model_id="lmstudio-community/Phi-4-mini-reasoning-MLX-4bit",
    provider="Microsoft",
    architecture="dense",
    parameter_count="14B",
    active_parameters=None,
    quantization="4bit",
    model_memory_gb=2.4,
    kv_per_1k_tokens_mb=64,
    kv_per_1k_turbo_mb=9,
    measured_tps=174,
    measured_ttft_cached_s=0.101,
    bandwidth_efficiency=0.51,
    tool_calling=False,
    tool_parser=None,
    reasoning=False,
    reasoning_parser=None,
    vision=True,
    turboquant_compatible=True,
    base_args=[],
    notes="Fastest decode (174 tok/s). Only 2.4GB. No tool calling — text only.",
))

# ── Mistral ──────────────────────────────────────────────────────

_r(ModelRecipe(
    id="mistral-small-24b",
    name="Mistral Small 3.2 24B",
    model_id="lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-4bit",
    provider="Mistral",
    architecture="dense",
    parameter_count="24B",
    active_parameters=None,
    quantization="4bit",
    model_memory_gb=12.7,
    kv_per_1k_tokens_mb=64,
    kv_per_1k_turbo_mb=9,
    measured_tps=48,
    measured_ttft_cached_s=0.107,
    bandwidth_efficiency=0.54,
    tool_calling=False,
    tool_parser=None,
    reasoning=False,
    reasoning_parser=None,
    vision=True,
    turboquant_compatible=True,
    base_args=[],
    notes="Vision works. Tool calling broken (template strips tools). 48 tok/s.",
))

_r(ModelRecipe(
    id="devstral-24b",
    name="Devstral Small 2 24B",
    model_id="mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit",
    provider="Mistral",
    architecture="dense",
    parameter_count="24B",
    active_parameters=None,
    quantization="4bit",
    model_memory_gb=12.7,
    kv_per_1k_tokens_mb=64,
    kv_per_1k_turbo_mb=9,
    measured_tps=48,
    measured_ttft_cached_s=0.103,
    bandwidth_efficiency=0.54,
    tool_calling=False,
    tool_parser=None,
    reasoning=False,
    reasoning_parser=None,
    vision=True,
    turboquant_compatible=True,
    base_args=[],
    notes="Code-focused Mistral. Tool calling partial (parser bug). 48 tok/s.",
))

# ── Gemma ────────────────────────────────────────────────────────

_r(ModelRecipe(
    id="gemma3-12b",
    name="Gemma 3 12B QAT",
    model_id="mlx-community/gemma-3-12b-it-qat-4bit",
    provider="Google",
    architecture="dense",
    parameter_count="12B",
    active_parameters=None,
    quantization="4bit",
    model_memory_gb=8.5,
    kv_per_1k_tokens_mb=64,
    kv_per_1k_turbo_mb=9,
    measured_tps=49,
    measured_ttft_cached_s=0.147,
    bandwidth_efficiency=0.51,
    tool_calling=False,
    tool_parser=None,
    reasoning=False,
    reasoning_parser=None,
    vision=False,
    turboquant_compatible=True,
    base_args=[],
    notes="Tool calling broken (template strips tools). Vision broken. 49 tok/s.",
))

# ── GLM ──────────────────────────────────────────────────────────

_r(ModelRecipe(
    id="glm-4.7-flash-9b",
    name="GLM-4.7-Flash 9B (8bit)",
    model_id="mlx-community/GLM-4.7-4bit",
    provider="Zhipu",
    architecture="dense",
    parameter_count="9B",
    active_parameters=None,
    quantization="8bit",
    model_memory_gb=30.1,
    kv_per_1k_tokens_mb=64,
    kv_per_1k_turbo_mb=9,
    measured_tps=60,
    measured_ttft_cached_s=0.110,
    bandwidth_efficiency=0.52,
    tool_calling=True,
    tool_parser="glm47",
    reasoning=False,
    reasoning_parser=None,
    vision=True,
    turboquant_compatible=True,
    base_args=["--tool-call-parser", "glm47"],
    notes="100% tool calling with glm47 parser. 8bit uses 30GB (heavy for 9B).",
))

# ── GPT-OSS ──────────────────────────────────────────────────────

_r(ModelRecipe(
    id="gpt-oss-20b",
    name="GPT-OSS 20B",
    model_id="mlx-community/gpt-oss-20b-MXFP4-Q8",
    provider="ByteDance",
    architecture="dense",
    parameter_count="20B",
    active_parameters=None,
    quantization="mxfp4",
    model_memory_gb=11.8,
    kv_per_1k_tokens_mb=64,
    kv_per_1k_turbo_mb=9,
    measured_tps=123,
    measured_ttft_cached_s=0.112,
    bandwidth_efficiency=0.55,
    tool_calling=False,
    tool_parser=None,
    reasoning=False,
    reasoning_parser=None,
    vision=True,
    turboquant_compatible=True,
    base_args=[],
    notes="Second fastest decode (123 tok/s). 1.56x vs upstream. No tool calling.",
))

# ── GLM-4.5-Air ─────────────────────────────────────────────────

_r(ModelRecipe(
    id="glm-4.5-air",
    name="GLM-4.5-Air (4bit)",
    model_id="GLM-4.5-Air-MLX-4bit",
    provider="Zhipu",
    architecture="moe",
    parameter_count="Unknown",
    active_parameters=None,
    quantization="4bit",
    model_memory_gb=56.4,
    kv_per_1k_tokens_mb=64,
    kv_per_1k_turbo_mb=9,
    measured_tps=46,
    measured_ttft_cached_s=0.108,
    bandwidth_efficiency=0.52,
    tool_calling=True,
    tool_parser="glm47",
    reasoning=False,
    reasoning_parser=None,
    vision=True,
    turboquant_compatible=True,
    base_args=["--tool-call-parser", "glm47"],
    notes="100% tool calling. Long decode broken (infinite thinking). 56GB — needs 64GB+ Mac.",
))

# fmt: on


def get_recipe(model_id: str) -> ModelRecipe | None:
    """Look up a recipe by short id, full HF id, or fuzzy match."""
    # Exact short id
    if model_id in MODEL_RECIPES:
        return MODEL_RECIPES[model_id]

    # Exact HF model_id
    for recipe in MODEL_RECIPES.values():
        if recipe.model_id == model_id:
            return recipe

    # Fuzzy: substring match on id or name
    model_lower = model_id.lower()
    matches = [
        r
        for r in MODEL_RECIPES.values()
        if model_lower in r.id.lower() or model_lower in r.name.lower()
    ]
    if len(matches) == 1:
        return matches[0]

    return None


def search_recipes(query: str = "") -> list[ModelRecipe]:
    """Search recipes by keyword. Empty query returns all."""
    if not query:
        return list(MODEL_RECIPES.values())

    q = query.lower()
    return [
        r
        for r in MODEL_RECIPES.values()
        if q in r.id.lower()
        or q in r.name.lower()
        or q in r.provider.lower()
        or q in r.model_id.lower()
    ]
