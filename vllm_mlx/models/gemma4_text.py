# SPDX-License-Identifier: Apache-2.0
"""
Gemma 4 text-only model loader for the LLM path.

mlx-lm 0.31+ added native ``gemma4`` (used by the 26B / 31B aliases), but
``gemma4_unified`` (the model_type the four ``gemma-4-12b-*`` aliases ship
under) is still not in mlx-lm. This module loads the language model
portion from mlx-vlm and wraps it to be compatible with mlx-lm's
generate_step() interface, enabling:
- Prompt cache (KV reuse across requests)
- DeltaNet state snapshots (if applicable)
- All LLM-path optimizations

The wrapper is thin: it just ensures model(input_ids, cache=cache) returns
a raw logits tensor instead of LanguageModelOutput.

TODO: Remove once mlx-lm adds native ``gemma4_unified`` support (12B variants).
"""

import json
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


def is_gemma4_model(model_path: str | Path) -> bool:
    """Check if the model at the given path is a Gemma 4 model.

    The previous implementation called ``snapshot_download(repo_id)`` to
    populate a local cache before reading ``config.json``. That works,
    but ``snapshot_download`` validates/fetches the ENTIRE model tree
    (all safetensors shards, tokenizer files, generation config), which
    for an 8-bit 35B model is ~35 GB of Xet-protocol revalidation on
    every cold ``rapid-mlx serve`` start. Switch to ``hf_hub_download``
    targeting ``config.json`` directly: a ~5 KB file, validated against
    the existing HF cache. This was the root cause of stress_e2e_bench
    server-boot timeouts on large models in PR #600 validation.
    """
    p = Path(model_path)
    config_path = p / "config.json" if p.is_dir() else None
    if config_path is None or not config_path.exists():
        try:
            from huggingface_hub import hf_hub_download

            config_path = Path(
                hf_hub_download(repo_id=str(model_path), filename="config.json")
            )
        except Exception:
            return False
    if not config_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text())
        model_type = config.get("model_type", "")
        return "gemma4" in model_type
    except Exception:
        return False


class Gemma4TextWrapper(nn.Module):
    """Wraps mlx-vlm's Gemma4 LanguageModel for mlx-lm compatibility.

    mlx-lm's generate_step() expects model(input_ids, cache=cache) -> logits.
    mlx-vlm's LanguageModel returns LanguageModelOutput(logits=...).
    This wrapper extracts .logits so the interface matches.
    """

    def __init__(self, language_model):
        super().__init__()
        self.language_model = language_model
        # Expose config for mlx-lm compatibility
        self.config = language_model.config
        self.model = language_model.model
        self.model_type = getattr(language_model, "model_type", "gemma4")

    def __call__(self, input_ids, cache=None, **kwargs):
        out = self.language_model(input_ids, cache=cache, **kwargs)
        # LanguageModelOutput -> raw logits tensor
        return out.logits if hasattr(out, "logits") else out

    def sanitize(self, weights):
        """Strip language_model. prefix from VLM-format weights."""
        sanitized = {}
        for k, v in weights.items():
            new_key = k
            # Strip top-level "model." wrapper
            if new_key.startswith("model."):
                new_key = new_key[len("model.") :]
            # Strip "language_model." to get bare model weights,
            # then re-add "language_model." for our wrapper structure
            if new_key.startswith("language_model."):
                pass  # keep as-is — our wrapper has .language_model attribute
            elif not any(
                new_key.startswith(p)
                for p in ["vision_tower", "audio_tower", "embed_vision", "embed_audio"]
            ):
                new_key = "language_model." + new_key
            else:
                continue  # skip vision/audio weights
            # Skip rotary embeddings (computed dynamically)
            if "rotary_emb" in new_key:
                continue
            # Skip clipping params (vision-only)
            if any(
                s in new_key
                for s in ["input_max", "input_min", "output_max", "output_min"]
            ):
                continue
            sanitized[new_key] = v
        return sanitized

    def make_cache(self):
        """Delegate to LanguageModel for proper sliding window + full attention cache."""
        return self.language_model.make_cache()

    @property
    def layers(self):
        return self.language_model.layers

    @property
    def head_dim(self):
        return self.language_model.head_dim

    @property
    def n_kv_heads(self):
        return self.language_model.n_kv_heads


def load_gemma4_text(model_path: str | Path, tokenizer_config: dict = None):
    """Load Gemma 4 as a text-only model via the LLM path.

    Returns (model, tokenizer) compatible with mlx-lm's generate_step().
    """
    from mlx_lm.utils import load_tokenizer

    p = Path(model_path)
    if not p.is_dir():
        from huggingface_hub import snapshot_download

        p = Path(snapshot_download(str(model_path)))

    config = json.loads((p / "config.json").read_text())
    text_config = config.get("text_config", config)

    # Build the language model from mlx-vlm. The Gemma 4 model class
    # definitions only live in mlx-vlm; mlx-lm's native gemma4 path
    # exists but misses the ``gemma4_unified`` model_type used by the
    # QAT 8-bit and several upstream checkpoints. On a bare
    # ``pip install rapid-mlx`` (no extras) or a default brew install
    # mlx-vlm is absent — give the user an actionable, size-conscious
    # error instead of a raw ModuleNotFoundError.
    try:
        from mlx_vlm.models.gemma4.config import TextConfig
        from mlx_vlm.models.gemma4.language import LanguageModel
    except ImportError as e:
        raise ImportError(
            "Gemma 4 models require the optional `mlx-vlm` dependency "
            "for the model architecture classes.\n"
            "Install just the Python classes (16 MB, recommended for "
            "text-only use):\n"
            "    pip install --no-deps 'mlx-vlm>=0.6.1'\n"
            "Or the full vision/audio stack (~+450 MB):\n"
            "    pip install 'rapid-mlx[vision]'"
        ) from e

    tc = TextConfig.from_dict(text_config)
    language_model = LanguageModel(tc)

    # Wrap for mlx-lm compatibility
    model = Gemma4TextWrapper(language_model)

    # Apply quantization config if present (converts Linear → QuantizedLinear)
    quant_config = config.get("quantization", config.get("quantization_config"))
    if quant_config:
        default_bits = quant_config.get("bits", 4)
        default_gs = quant_config.get("group_size", 64)

        # Build per-layer override map from config (mixed quantization)
        # Keys like "language_model.model.layers.0.mlp.gate_proj" → {bits:8, group_size:64}
        overrides = {}
        for k, v in quant_config.items():
            if isinstance(v, dict) and "bits" in v:
                overrides[k] = {
                    kk: vv
                    for kk, vv in v.items()
                    if kk in ("bits", "group_size", "mode")
                }

        if overrides:
            logger.info(
                "[gemma4] Mixed quantization: %d-bit default, %d overrides (8-bit MLP)",
                default_bits,
                len(overrides),
            )

            def _class_predicate(path, module):
                if not hasattr(module, "to_quantized"):
                    return False
                # Check per-layer overrides
                # Override keys use "language_model.model.layers..." but nn.quantize
                # sees "model.layers..." (relative to wrapper). Match by suffix.
                for override_path, override_cfg in overrides.items():
                    # Strip common prefixes for matching
                    suffix = override_path.split("language_model.model.")[-1]
                    if path.endswith(suffix):
                        return override_cfg
                return {"bits": default_bits, "group_size": default_gs}

            nn.quantize(model, class_predicate=_class_predicate)
        else:
            logger.info(
                "[gemma4] Applying %d-bit quantization (group_size=%d)",
                default_bits,
                default_gs,
            )
            nn.quantize(model, bits=default_bits, group_size=default_gs)

    # Load weights — sanitize per shard to keep peak memory proportional
    # to the text-model size, not the full multimodal checkpoint.  On
    # Gemma 4 the multimodal shards can be >2× the language-model
    # weight, so loading everything then sanitizing would transiently
    # allocate the entire checkpoint and OOM smaller Macs.  Fixes #123.
    weight_files = sorted(
        f for f in p.glob("*.safetensors") if not f.name.startswith("._")
    )
    if not weight_files:
        raise FileNotFoundError(f"No .safetensors files in {p}")
    sanitized = {}
    for wf in weight_files:
        shard = mx.load(str(wf))
        sanitized.update(model.sanitize(shard))
        del shard
    model.load_weights(list(sanitized.items()), strict=False)

    # Verify weights loaded
    test_param = model.language_model.model.embed_tokens
    if hasattr(test_param, "scales") and mx.all(test_param.scales == 0).item():
        logger.warning(
            "[gemma4] Embedding scales are zero — quantized model may have issues"
        )

    # Load tokenizer
    tokenizer_config = tokenizer_config or {}
    eos_token_ids = config.get("eos_token_id", text_config.get("eos_token_id"))
    tokenizer = load_tokenizer(p, tokenizer_config, eos_token_ids=eos_token_ids)

    logger.info(
        "[gemma4] Loaded text-only model via LLM path (%d layers)", len(model.layers)
    )
    return model, tokenizer
