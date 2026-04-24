# SPDX-License-Identifier: Apache-2.0
"""
Tokenizer utilities with fallback support for non-standard tokenizers.

Some models (e.g., Nemotron) use non-standard tokenizer configurations
that transformers doesn't recognize. This module provides fallback loading
directly from tokenizer.json.
"""

import json
import logging
from pathlib import Path

from .chat_templates import DEFAULT_CHATML_TEMPLATE, NEMOTRON_CHAT_TEMPLATE

logger = logging.getLogger(__name__)

# Models that require tokenizer fallback
FALLBACK_MODELS = [
    "nemotron",
    "NVIDIA-Nemotron",
]


def _needs_tokenizer_fallback(model_name: str) -> bool:
    """Check if model needs tokenizer fallback."""
    model_lower = model_name.lower()
    return any(pattern.lower() in model_lower for pattern in FALLBACK_MODELS)


def load_model_with_fallback(model_name: str, tokenizer_config: dict = None):
    """
    Load model and tokenizer with fallback for non-standard tokenizers.

    Args:
        model_name: HuggingFace model name or local path
        tokenizer_config: Optional tokenizer configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    from mlx_lm import load

    tokenizer_config = tokenizer_config or {}

    # Check if model needs fallback (e.g., Nemotron)
    if _needs_tokenizer_fallback(model_name):
        logger.info(
            f"Model {model_name} requires tokenizer fallback, loading directly..."
        )
        return _load_with_tokenizer_fallback(model_name)

    # Gemma 4: mlx-lm 0.31+ supports it natively. Only use our wrapper
    # for older mlx-lm versions that lack gemma4 model support.
    from ..models.gemma4_text import is_gemma4_model

    if is_gemma4_model(model_name):
        try:
            # Try native mlx-lm load first (0.31+)
            model, tokenizer = load(model_name, tokenizer_config=tokenizer_config)
            logger.info("Gemma 4 loaded natively via mlx-lm")
            return model, tokenizer
        except Exception as e:
            # Fall back to our wrapper for older mlx-lm versions
            # that lack native gemma4 architecture support
            from ..models.gemma4_text import load_gemma4_text

            logger.info(
                f"Gemma 4 native load failed ({e}), "
                "falling back to text-only wrapper (legacy mlx-lm)"
            )
            return load_gemma4_text(model_name, tokenizer_config)

    try:
        model, tokenizer = load(model_name, tokenizer_config=tokenizer_config)
        # mlx_lm.load() succeeds but sanitize() may have silently
        # stripped mtp.* weights.  Check if the config declares MTP
        # layers and the model came back without a .mtp attribute;
        # if so, re-inject from the safetensors on disk.
        _try_inject_mtp_post_load(model, model_name)
        return model, tokenizer
    except ValueError as e:
        # Fallback for models with non-standard tokenizers
        if "TokenizersBackend" in str(e) or "Tokenizer class" in str(e):
            logger.warning(f"Standard tokenizer loading failed, using fallback: {e}")
            return _load_with_tokenizer_fallback(model_name)
        # Fallback for models with extra/missing weights (e.g., vision tower, MTP layers).
        # Retry with strict=False to discard extra weights.
        elif "parameters not in model" in str(e) or (
            "Missing" in str(e) and "parameters" in str(e)
        ):
            logger.warning(
                f"Model has extra/missing parameters (likely VLM / MTP weights), "
                f"retrying with strict=False: {e}"
            )
            return _load_strict_false(model_name, tokenizer_config)
        else:
            raise


def _load_strict_false(model_name: str, tokenizer_config: dict = None):
    """Load model with strict=False to discard extra weights (e.g., vision tower, MTP)."""
    from mlx_lm.utils import load_model, load_tokenizer

    local_path = Path(model_name)
    if local_path.is_dir():
        model_path = local_path
    else:
        from huggingface_hub import snapshot_download

        model_path = Path(snapshot_download(model_name))

    model, config = load_model(model_path, strict=False)
    tokenizer = load_tokenizer(
        model_path,
        tokenizer_config or {},
        eos_token_ids=config.get("eos_token_id", None),
    )
    # Inject MTP support if model has MTP config + weights
    _try_inject_mtp(model, model_path, config)
    return model, tokenizer


def _read_num_mtp_layers(config: dict) -> int:
    """Read num_nextn_predict_layers from config, checking text_config too.

    Multimodal checkpoints (VLM + MTP) store this under text_config,
    while text-only checkpoints put it at the top level.  Fixes #121.
    """
    n = config.get("num_nextn_predict_layers", 0)
    if n == 0:
        n = config.get("text_config", {}).get("num_nextn_predict_layers", 0)
    return n


def _try_inject_mtp(model, model_path, config):
    """Inject MTP support if model has MTP config + weights."""
    num = _read_num_mtp_layers(config)
    if num > 0:
        from ..patches.qwen3_next_mtp import inject_mtp_support

        # inject_mtp_support reads config["num_nextn_predict_layers"]
        # directly.  For VLM checkpoints where the field lives under
        # text_config, surface it to the top level so the injector
        # doesn't skip with "num_nextn_predict_layers=0".
        if config.get("num_nextn_predict_layers", 0) == 0:
            config = {**config, "num_nextn_predict_layers": num}
        inject_mtp_support(model, model_path, config)


def _try_inject_mtp_post_load(model, model_name):
    """Check if MTP weights exist but were stripped by sanitize(), and inject."""
    import json

    from mlx_lm.utils import _download

    model_path = _download(model_name)
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return
    with open(config_path) as f:
        config = json.load(f)
    num_mtp = _read_num_mtp_layers(config)
    if num_mtp > 0 and getattr(model, "mtp", None) is None:
        mtp_file = Path(model_path) / "model-mtp.safetensors"
        if mtp_file.exists():
            logger.info(
                f"[MTP] Found MTP config (layers={num_mtp}) and weights, injecting..."
            )
            _try_inject_mtp(model, model_path, config)
        else:
            logger.info(
                f"[MTP] Config has num_nextn_predict_layers={num_mtp} "
                "but model-mtp.safetensors not found, skipping MTP."
            )


def _load_with_tokenizer_fallback(model_name: str):
    """Load model with fallback tokenizer for non-standard models like Nemotron."""
    from mlx_lm.utils import load_model

    logger.info("Loading with tokenizer fallback...")

    # Get model path - use local path if it exists, otherwise download from Hub
    local_path = Path(model_name)
    if local_path.is_dir():
        model_path = local_path
    else:
        from huggingface_hub import snapshot_download

        model_path = Path(snapshot_download(model_name))

    # Load model
    model, _ = load_model(model_path)

    # Try to load tokenizer from tokenizer.json directly
    tokenizer_json = model_path / "tokenizer.json"
    if tokenizer_json.exists():
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast

        logger.info("Loading tokenizer from tokenizer.json")
        base_tokenizer = Tokenizer.from_file(str(tokenizer_json))

        # Read tokenizer_config.json for special tokens and chat template
        tokenizer_config_path = model_path / "tokenizer_config.json"
        bos_token = "<s>"
        eos_token = "</s>"
        unk_token = "<unk>"
        chat_template = None

        if tokenizer_config_path.exists():
            with open(tokenizer_config_path) as f:
                config = json.load(f)
                bos_token = config.get("bos_token", bos_token)
                eos_token = config.get("eos_token", eos_token)
                unk_token = config.get("unk_token", unk_token)
                chat_template = config.get("chat_template")

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=base_tokenizer,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token="<pad>",
        )

        # Set chat template if available
        if chat_template:
            tokenizer.chat_template = chat_template
            logger.info("Chat template loaded from tokenizer_config.json")
        elif _needs_tokenizer_fallback(model_name):
            # Use official Nemotron chat template with thinking support
            tokenizer.chat_template = NEMOTRON_CHAT_TEMPLATE
            logger.info("Using official Nemotron chat template with thinking support")
        else:
            # Default simple ChatML format for other models
            tokenizer.chat_template = DEFAULT_CHATML_TEMPLATE
            logger.info("Using default ChatML chat template")

        logger.info("Tokenizer loaded via fallback successfully")
        return model, tokenizer
    else:
        raise ValueError(f"No tokenizer.json found in {model_path}")
