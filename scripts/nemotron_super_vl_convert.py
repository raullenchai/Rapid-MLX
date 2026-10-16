"""Offline CPU Super VL text conversion; inference gates require the GPU lock."""

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model, quantize_model, save_config, save_model

from rapid_mlx.model_load_errors import load_model_checked
from rapid_mlx.models.nemotron_h_super import text_config

BASE_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
# Keep seven MoE layers at each boundary at three bits; the interior budget
# needs 26 two-bit layers (42.76 GB weights plus metadata).
TWO_BIT_LAYERS = frozenset(
    (
        17,
        19,
        21,
        23,
        26,
        28,
        30,
        32,
        34,
        37,
        39,
        41,
        43,
        45,
        48,
        50,
        52,
        54,
        56,
        59,
        61,
        63,
        65,
        67,
        70,
        72,
    )
)

ASSETS = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
    "LICENSE",
}


def quantization_predicate(model, bits):
    preserve = model.quant_predicate

    def predicate(path, module):
        if not preserve(path, module):
            return False
        if "embedding" in path:
            return {"bits": 8, "group_size": 64, "mode": "affine"}
        if bits == 3:
            expert = ".switch_mlp." in path
            return {
                "bits": (2 if int(path.split(".")[2]) in TWO_BIT_LAYERS else 3)
                if expert
                else 4,
                "group_size": 128 if expert else 64,
                "mode": "affine",
            }
        return True

    return predicate


def artifact_size(path):
    return sum(
        f.stat().st_size
        for f in path.rglob("*")
        if f.is_file() and ".cache" not in f.relative_to(path).parts
    )


def model_card(bits, size):
    recipe = (
        "4-bit affine, group size 64"
        if bits == 4
        else "mixed: 26 interior MoE layers at 2-bit/group-128, 14 boundary layers at "
        "3-bit/group-128, other eligible projections 4-bit/group-64"
    )
    memory_guidance = (
        "Targets **64 GB** unified\n"
        "  memory for short contexts. Measured MLX allocator peak: **45.903 GB**\n"
        "  for 1,024 prompt tokens plus 64 generated tokens, with prefill chunks of 256.\n"
        "  Measurements use a 256 GiB M3 Ultra; a physical 64 GB Mac has not been tested.\n"
        "  Start at 1K context and leave memory for macOS and other applications."
        if bits == 3
        else "Recommend **128 GB+** unified memory (measured peak 72.7 GB at 1K context "
        "on M3 Ultra; 96 GB Macs may need a raised GPU wired-memory limit and are not "
        "yet verified).\n"
        "  Allow headroom beyond the weights for prefill, caches, macOS and applications."
    )
    return f"""---
license: other
license_name: openmdw-1.1
license_link: https://openmdw.ai/license/1-1/
base_model: {BASE_MODEL}
library_name: mlx
pipeline_tag: text-generation
tags: [mlx, rapid-mlx, nemotron, apple-silicon]
---

# NVIDIA Nemotron 3.5 Super VL 120B-A12B — MLX {bits}bit (text)

Text conversion of [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}).

- Quantization: {recipe}; embeddings 8-bit/group-64. Output head,
  routers and latent projections retain checkpoint precision.
- Artifact size: {size / 1e9:.3f} GB. {memory_guidance}
- Text only: the vision tower, image projector and unused MTP head are omitted.
- Declared context limit: 1,048,576 tokens; this does not promise that the full
  window fits the recommended memory.

## Requirements

Requires Rapid-MLX 0.16.2 or newer. The config retains
`nemotron_h` and opts into Rapid's reviewed precision adapter. Stock loaders
do not apply that adapter; only Rapid-MLX is validated for this conversion.

```sh
rapid-mlx serve rapid-mlx/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-MLX-{bits}bit
```

Reasoning and XML tool calling use the original tokenizer/chat template.
MTP speculative decoding and image input are not enabled.

## License

Original model by NVIDIA under [OpenMDW 1.1](https://openmdw.ai/license/1-1/).
This conversion inherits the original model's license terms.
"""


def shard_path(root, name):
    """Contain checkpoint-controlled paths, including symlinks, in their root."""
    relative = Path(name)
    candidate = root / relative
    if relative.is_absolute() or not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Checkpoint shard path escapes its root: {name}")
    return candidate


def main(argv=None):
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bits", required=True, type=int, choices=(3, 4))
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    config = text_config(json.loads((args.source / "config.json").read_text()))
    index = json.loads((args.source / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    sources = {name: shard_path(args.source, name) for name in shards}
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint download incomplete: {sorted(missing)}")
    start = time.monotonic()
    mx.set_default_device(mx.cpu)
    scratch = Path("/private/tmp/nemotron-super-vl-convert")
    scratch.mkdir(exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{args.output.name}.partial-", dir=args.output.parent)
    )
    try:
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            view = Path(temporary)
            (view / "config.json").write_text(json.dumps(config, indent=2))
            for name, source in sources.items():
                destination = shard_path(view, name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(source.resolve())
            model, config = load_model_checked(load_model, view, lazy=True)
            model, config = quantize_model(
                model,
                config,
                64,
                args.bits,
                quant_predicate=quantization_predicate(model, args.bits),
            )
            save_model(staging, model, donate_model=True)
            save_config(text_config(config), staging / "config.json")
        for name in ASSETS:
            if (args.source / name).is_file():
                if name == "tokenizer_config.json":
                    tokenizer_config = json.loads((args.source / name).read_text())
                    for key in ("auto_map", "processor_class"):
                        tokenizer_config.pop(key, None)
                    (staging / name).write_text(json.dumps(tokenizer_config, indent=2))
                else:
                    shutil.copyfile(args.source / name, staging / name)
        card = staging / "README.md"
        card.write_text(model_card(args.bits, artifact_size(staging)))
        if args.output.exists():
            raise FileExistsError(args.output)
        os.rename(staging, args.output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    result = {
        "bits": args.bits,
        "output": str(args.output),
        "bytes": artifact_size(args.output),
        "wall_seconds": time.monotonic() - start,
    }
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":  # pragma: no cover
    main()
