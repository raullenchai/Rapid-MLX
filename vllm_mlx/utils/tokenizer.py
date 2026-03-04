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

    try:
        return load(model_name, tokenizer_config=tokenizer_config)
    except ValueError as e:
        # Fallback for models with non-standard tokenizers
        if "TokenizersBackend" in str(e) or "Tokenizer class" in str(e):
            logger.warning(f"Standard tokenizer loading failed, using fallback: {e}")
            return _load_with_tokenizer_fallback(model_name)
        # Fallback for multimodal models loaded as text-only (skip vision weights)
        elif "parameters not in model" in str(e) or ("Missing" in str(e) and "parameters" in str(e)):
            logger.warning(f"Model has extra/missing parameters (likely VLM weights), retrying with strict=False: {e}")
            return _load_non_strict(model_name, tokenizer_config)
        else:
            raise


def _load_non_strict(model_name: str, tokenizer_config: dict = None):
    """Load model with strict=False to skip extra weights (e.g., vision tower)."""
    from mlx_lm.utils import load_model, load_tokenizer

    local_path = Path(model_name)
    if local_path.is_dir():
        model_path = local_path
    else:
        from huggingface_hub import snapshot_download
        model_path = Path(snapshot_download(model_name))

    model, _ = load_model(model_path, strict=False)
    tokenizer = load_tokenizer(model_path, tokenizer_config or {})
    return model, tokenizer


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
