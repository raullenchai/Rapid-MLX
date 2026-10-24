# SPDX-License-Identifier: Apache-2.0
"""Native Kimi vision runtime with a reviewed, data-only tokenizer."""

import json
from pathlib import Path


def load_kimi_processor(model_path):
    from mlx_vlm.models.kimi_k25.processing_kimi_k25 import (
        KimiK25ImageProcessor,
        KimiK25Processor,
    )
    from mlx_vlm.tokenizer_utils import NaiveStreamingDetokenizer
    from mlx_vlm.utils import StoppingCriteria

    from ..utils.kimi_tokenizer import load_kimi_tokenizer
    from ..utils.tokenizer import (
        RAPID_EXTRA_EOS_ATTR,
        _apply_chat_template_sidecar,
        augment_eos_token_ids_from_generation_config,
    )

    path = Path(model_path)
    config = json.loads((path / "config.json").read_text())
    tokenizer = load_kimi_tokenizer(path)
    _apply_chat_template_sidecar(path, tokenizer)
    augment_eos_token_ids_from_generation_config(tokenizer, str(path))
    image_kwargs = {}
    preprocessor = path / "preprocessor_config.json"
    if preprocessor.is_file():
        media = json.loads(preprocessor.read_text()).get("media_proc_cfg", {})
        for source, target in {
            "patch_size": "patch_size",
            "in_patch_limit": "in_token_limit",
            "image_mean": "image_mean",
            "image_std": "image_std",
            "merge_kernel_size": "merge_kernel_size",
            "patch_limit_on_one_side": "patch_limit_on_one_side",
        }.items():
            if source in media:
                image_kwargs[target] = media[source]
        if isinstance(image_kwargs.get("merge_kernel_size"), int):
            image_kwargs["merge_kernel_size"] = [image_kwargs["merge_kernel_size"]] * 2
    # Same native image implementation and config precedence as mlx-vlm.load.
    image_processor = KimiK25ImageProcessor(config=config, **image_kwargs)
    processor = KimiK25Processor(image_processor=image_processor, tokenizer=tokenizer)
    processor.detokenizer = NaiveStreamingDetokenizer(tokenizer)
    tokenizer.stopping_criteria = StoppingCriteria(
        config.get("eos_token_id", config.get("text_config", {}).get("eos_token_id")),
        tokenizer,
        additional_eos_token_ids=getattr(tokenizer, RAPID_EXTRA_EOS_ATTR, ()),
    )
    return processor


def load_kimi_vlm(model_path, *, adapter_path=None, lazy=False, strict=True, **kwargs):
    from mlx_vlm.utils import apply_lora_layers, load_model

    from ..model_load_errors import typed_mlx_load_boundaries
    from ..utils.model_file_guard import validate_model_code

    path = Path(model_path)
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "kimi_k25":
        raise ValueError("Native Kimi loader requires model_type kimi_k25")
    # This adapter deliberately owns the executable implementation.
    validate_model_code(config, trust_remote_code=False)
    kwargs.pop("trust_remote_code", None)
    with typed_mlx_load_boundaries():
        model = load_model(
            path, lazy=lazy, strict=strict, trust_remote_code=False, **kwargs
        )
    if adapter_path is not None:
        model = apply_lora_layers(model, adapter_path)
        model.eval()
    return model, load_kimi_processor(path)
