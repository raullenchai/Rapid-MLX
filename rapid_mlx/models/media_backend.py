"""Native media backend registration and checkpoint-based dispatch."""

from pathlib import Path


def native_media_model_type(model_name):
    from ..model_metadata import read_model_metadata

    metadata = read_model_metadata(model_name)
    config = getattr(metadata, "config", None)
    model_type = config.get("model_type") if isinstance(config, dict) else None
    return model_type if model_type in {"lfm3_vl"} else None


def load_native_media(model_path, config):
    if config.get("model_type") != "lfm3_vl":
        raise ValueError("Unsupported native media model")
    from mlx_lm.utils import load_model, load_tokenizer

    from .lfm3_vl import Model, ModelArgs
    from .lfm3_vl_processor import LFM3VLProcessor

    path = Path(model_path)
    model, _ = load_model(path, get_model_classes=lambda config: (Model, ModelArgs))
    tokenizer = load_tokenizer(
        path, tokenizer_config_extra={"trust_remote_code": False}
    )
    return model, LFM3VLProcessor(tokenizer)
