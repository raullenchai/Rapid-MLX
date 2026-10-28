"""Native media backend registration and checkpoint-based dispatch."""

from pathlib import Path


def native_media_model_type(model_name):
    from ..model_metadata import read_model_metadata

    metadata = read_model_metadata(model_name)
    if metadata is None:
        from ..model_aliases import resolve_hub_model

        resolved = resolve_hub_model(model_name)
        if resolved != model_name:
            metadata = read_model_metadata(resolved)
    config = getattr(metadata, "config", None)
    model_type = config.get("model_type") if isinstance(config, dict) else None
    return model_type if model_type in {"lfm3_vl"} else None


def resolve_native_checkpoint(model_name):
    from ..model_aliases import resolve_hub_model, resolve_revision, resolve_subfolder

    local = Path(model_name).expanduser()
    if local.is_dir():
        return local
    revision = resolve_revision(model_name)
    if revision is not None:
        from ..utils.tokenizer import _resolve_subfolder_checkpoint

        return Path(_resolve_subfolder_checkpoint(model_name))
    from huggingface_hub import snapshot_download

    subfolder = resolve_subfolder(model_name)
    prefix = f"{subfolder}/" if subfolder else ""
    patterns = [
        prefix + p
        for p in (
            "*.json",
            "*.jinja",
            "*.txt",
            "*.model",
            "*.tiktoken",
            "*.safetensors",
        )
    ]
    snapshot = Path(
        snapshot_download(resolve_hub_model(model_name), allow_patterns=patterns)
    )
    return snapshot / subfolder if subfolder else snapshot


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
