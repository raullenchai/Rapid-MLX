# SPDX-License-Identifier: Apache-2.0
"""
Tokenizer utilities with fallback support for non-standard tokenizers.

Some models (e.g., Nemotron) use non-standard tokenizer configurations
that transformers doesn't recognize. This module provides fallback loading
directly from tokenizer.json.
"""

import importlib.util
import json
import logging
import types
from contextlib import contextmanager
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


def _register_vendored_archs() -> None:
    """Make vendored model architectures visible to mlx-lm's importlib lookup.

    mlx-lm resolves model_type → module via `importlib.import_module(
    f"mlx_lm.models.{model_type}")`. Pre-registering our vendored modules in
    sys.modules under that path lets it find them transparently. Idempotent.
    """
    import sys

    if "mlx_lm.models.deepseek_v4" not in sys.modules:
        try:
            from ..models import deepseek_v4 as _ds_v4

            # setdefault is atomic under the GIL; harmless if a concurrent
            # caller raced ahead (we'd cache the same module either way).
            sys.modules.setdefault("mlx_lm.models.deepseek_v4", _ds_v4)
        except Exception as e:
            logger.debug(f"deepseek_v4 vendored module unavailable: {e}")


# model_types served by vllm_mlx.models.* shims. transformers' AutoConfig /
# PreTrainedConfig won't recognize these, and mlx-lm's load() internally
# uses AutoTokenizer (which routes through AutoConfig). We must skip that
# path entirely for these models and use the lower-level load_model() +
# direct tokenizer.json load instead.
_VENDORED_MODEL_TYPES = {"deepseek_v4"}


def _read_jang_config(model_name: str) -> dict | None:
    """Return jang_config.json if the model declares JANG/JANGTQ weights."""
    try:
        local = Path(model_name)
        if local.is_dir():
            config_path = local / "jang_config.json"
        else:
            from huggingface_hub import hf_hub_download

            config_path = Path(
                hf_hub_download(repo_id=model_name, filename="jang_config.json")
            )
        if not config_path.exists():
            return None
        with open(config_path) as f:
            return json.load(f)
    except Exception as e:
        logger.debug(f"_read_jang_config({model_name}) failed: {e}")
        return None


def _is_jang_model(model_name: str) -> bool:
    return _read_jang_config(model_name) is not None


def _resolve_model_path(model_name: str) -> Path:
    local_path = Path(model_name)
    if local_path.is_dir():
        return local_path

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_name))


def _is_deepseek_v4_path(model_path: Path) -> bool:
    try:
        with open(model_path / "config.json") as f:
            config = json.load(f)
        return config.get("model_type") == "deepseek_v4"
    except Exception as e:
        logger.debug(f"_is_deepseek_v4_path({model_path}) failed: {e}")
        return False


@contextmanager
def _patch_deepseek_v4_jangtq_tokenizer(model_path: Path):
    """Bypass transformers AutoConfig for DSV4 while jang-tools expands EOS ids."""
    if not _is_deepseek_v4_path(model_path):
        yield
        return

    try:
        from transformers import AutoTokenizer, PreTrainedTokenizerFast
    except ImportError:
        yield
        return

    original_from_pretrained = AutoTokenizer.from_pretrained
    resolved_path = model_path.resolve()

    def from_pretrained(name, *args, **kwargs):
        try:
            if Path(name).resolve() == resolved_path:
                tokenizer_json = resolved_path / "tokenizer.json"
                if tokenizer_json.exists():
                    return PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_json))
        except (OSError, RuntimeError):
            pass
        return original_from_pretrained(name, *args, **kwargs)

    AutoTokenizer.from_pretrained = from_pretrained
    try:
        yield
    finally:
        AutoTokenizer.from_pretrained = original_from_pretrained


def _apply_jang_tokenizer_metadata(model_path: Path, tokenizer):
    tokenizer_config_path = model_path / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        return tokenizer

    try:
        with open(tokenizer_config_path) as f:
            tokenizer_config = json.load(f)
    except Exception as e:
        logger.debug(f"Failed to read tokenizer config for {model_path}: {e}")
        return tokenizer

    chat_template = tokenizer_config.get("chat_template")
    if chat_template and not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = chat_template

    for attr, key in (
        ("bos_token", "bos_token"),
        ("eos_token", "eos_token"),
        ("unk_token", "unk_token"),
        ("pad_token", "pad_token"),
    ):
        value = tokenizer_config.get(key)
        if value and not getattr(tokenizer, attr, None):
            try:
                setattr(tokenizer, attr, value)
            except Exception:
                logger.debug(f"Failed to set tokenizer.{attr} for {model_path}")

    if _is_deepseek_v4_path(model_path):
        _apply_deepseek_v4_chat_encoder(model_path, tokenizer)

    return tokenizer


def _apply_deepseek_v4_chat_encoder(model_path: Path, tokenizer):
    encoding_path = model_path / "encoding" / "encoding_dsv4.py"
    if not encoding_path.exists():
        return

    try:
        spec = importlib.util.spec_from_file_location(
            f"encoding_dsv4_{abs(hash(str(model_path.resolve())))}",
            str(encoding_path),
        )
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        logger.debug(f"Failed to load DSV4 chat encoder for {model_path}: {e}")
        return

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=None,
        tools=None,
        reasoning_effort=None,
        **kwargs,
    ):
        prepared = [dict(message) for message in messages]
        if tools:
            if prepared and prepared[0].get("role") in {"system", "developer"}:
                prepared[0] = {**prepared[0], "tools": tools}
            else:
                prepared.insert(0, {"role": "developer", "content": "", "tools": tools})

        # DSV4 JANG bundles declare chat as their default mode.  rapid-mlx's
        # shared helper auto-enables thinking for generic reasoning-capable
        # models, so ignore that auto flag here unless a caller passes an
        # explicit reasoning_effort.
        thinking_mode = "thinking" if reasoning_effort else "chat"
        prompt = module.encode_messages(
            prepared,
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort,
        )
        if not add_generation_prompt:
            for suffix in ("<｜Assistant｜><think>", "<｜Assistant｜></think>"):
                if prompt.endswith(suffix):
                    prompt = prompt[: -len(suffix)]
                    break
        if tokenize:
            return self.encode(prompt, **kwargs)
        return prompt

    tokenizer.apply_chat_template = types.MethodType(apply_chat_template, tokenizer)
    tokenizer._rapid_mlx_direct_generate = True


def _patch_deepseek_v4_jangtq_rope_offset():
    """Allow jang-tools DSV4 RoPE to accept MLX scalar offsets from batching."""
    try:
        from jang_tools.dsv4 import mlx_model
    except ImportError:
        return

    rope_cls = getattr(mlx_model, "DeepseekV4RoPE", None)
    if rope_cls is None or getattr(rope_cls, "_rapid_mlx_offset_patch", False):
        return

    original_call = rope_cls.__call__

    def _as_python_int(value):
        for convert in (int, lambda v: v.item(), lambda v: v.tolist()):
            try:
                converted = convert(value)
                if isinstance(converted, list):
                    converted = converted[0]
                return int(converted)
            except (AttributeError, IndexError, TypeError, ValueError):
                continue
        return value

    def patched_call(self, x, offset=0, inverse=False, positions=None):
        if positions is None and not isinstance(offset, (int, float)):
            offset = _as_python_int(offset)
        return original_call(
            self, x, offset=offset, inverse=inverse, positions=positions
        )

    rope_cls.__call__ = patched_call
    rope_cls._rapid_mlx_offset_patch = True


def _load_jang_model(model_name: str):
    jang_config = _read_jang_config(model_name) or {}
    model_path = _resolve_model_path(model_name)

    if jang_config.get("weight_format") == "mxtq":
        try:
            from jang_tools.load_jangtq import load_jangtq_model
        except ImportError as e:
            raise RuntimeError(
                "JANGTQ/MXTQ model detected, but jang-tools is not installed. "
                'Install the JANG extra with: pip install "rapid-mlx[jang]"'
            ) from e

        logger.info(f"Loading JANGTQ/MXTQ model with jang-tools: {model_path}")
        with _patch_deepseek_v4_jangtq_tokenizer(model_path):
            model, tokenizer = load_jangtq_model(model_path)
        if _is_deepseek_v4_path(model_path):
            _patch_deepseek_v4_jangtq_rope_offset()
        return model, _apply_jang_tokenizer_metadata(model_path, tokenizer)

    try:
        from jang_tools.loader import load_jang_model
    except ImportError as e:
        raise RuntimeError(
            "JANG model detected, but jang-tools is not installed. "
            'Install the JANG extra with: pip install "rapid-mlx[jang]"'
        ) from e

    logger.info(f"Loading JANG model with jang-tools: {model_path}")
    model, tokenizer = load_jang_model(model_path)
    return model, _apply_jang_tokenizer_metadata(model_path, tokenizer)


def _is_vendored_arch_model(model_name: str) -> bool:
    """Return True if model's config.json declares a model_type we vendor."""
    try:
        local = Path(model_name)
        if local.is_dir():
            config_path = local / "config.json"
        else:
            from huggingface_hub import hf_hub_download

            config_path = Path(
                hf_hub_download(repo_id=model_name, filename="config.json")
            )
        if not config_path.exists():
            return False
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg.get("model_type") in _VENDORED_MODEL_TYPES
    except Exception as e:
        logger.debug(f"_is_vendored_arch_model({model_name}) failed: {e}")
        return False


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

    _register_vendored_archs()
    tokenizer_config = tokenizer_config or {}

    if _is_jang_model(model_name):
        return _load_jang_model(model_name)

    # Check if model needs fallback (e.g., Nemotron)
    if _needs_tokenizer_fallback(model_name):
        logger.info(
            f"Model {model_name} requires tokenizer fallback, loading directly..."
        )
        return _load_with_tokenizer_fallback(model_name)

    # Vendored architectures (e.g. deepseek_v4) — transformers' AutoConfig
    # doesn't know about them, so mlx-lm's high-level load() blows up
    # before we get a chance to handle the error. Route directly to the
    # lower-level load_model() + raw tokenizer.json fallback.
    if _is_vendored_arch_model(model_name):
        logger.info(
            f"Model {model_name} uses a vendored architecture, "
            "skipping AutoConfig path and loading directly..."
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
        # Fallback for models with non-standard tokenizers, OR newer model_types
        # transformers' AutoConfig hasn't learned about yet (e.g. deepseek_v4
        # before transformers PR #45643 lands). The vendored arch can still load
        # the weights — we just need to bypass AutoTokenizer.
        if (
            "TokenizersBackend" in str(e)
            or "Tokenizer class" in str(e)
            or "does not recognize this architecture" in str(e)
        ):
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

    return model, tokenizer


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

        # HF convention (transformers >=4.43): chat_template.jinja sits
        # alongside tokenizer_config.json. DeepSeek V4 ships it that way.
        # utf-8-sig strips a UTF-8 BOM if the file was saved with one —
        # jinja2 would otherwise treat \ufeff as part of the template.
        if chat_template is None:
            chat_template_jinja = model_path / "chat_template.jinja"
            if chat_template_jinja.exists():
                chat_template = chat_template_jinja.read_text(encoding="utf-8-sig")
                logger.info("Chat template loaded from chat_template.jinja")

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
