# SPDX-License-Identifier: Apache-2.0
"""Emit the all-alias/all-method speculative capability report as JSON."""

from __future__ import annotations

import json
import re

from vllm_mlx.model_aliases import list_profiles
from vllm_mlx.spec_decode.capability import REGISTERED_METHODS, assess_method


def _quantization(hf_path: str) -> str:
    lowered = hf_path.lower()
    for label, pattern in (
        ("nvfp4", r"nvfp4"),
        ("mxfp4", r"mxfp4"),
        ("int4", r"int4"),
        ("4bit", r"4[-_]?bit"),
        ("6bit", r"6[-_]?bit"),
        ("8bit", r"8[-_]?bit"),
        ("2bit", r"2[-_]?bit"),
        ("3bit", r"3[-_]?bit"),
        ("fp8", r"fp8"),
        ("bfloat16", r"bfloat16"),
        ("bf16", r"bf16"),
        ("float16", r"float16"),
        ("fp16", r"fp16"),
    ):
        if re.search(rf"(?<![a-z0-9])(?:{pattern})(?![a-z0-9])", lowered):
            return label
    return "not-encoded-in-repo-name"


def build_report() -> dict[str, dict]:
    return {
        alias: {
            "model": {
                "hf_path": profile.hf_path,
                "checkpoint_format": "mlx",
                "quantization": _quantization(profile.hf_path),
                "modality": getattr(profile.modality, "value", profile.modality),
                "is_hybrid": profile.is_hybrid,
                "is_moe": profile.is_moe,
            },
            "methods": {
                method: assess_method(profile, method).to_dict()
                for method in REGISTERED_METHODS
            },
        }
        for alias, profile in sorted(list_profiles().items())
    }


def main() -> None:
    print(json.dumps(build_report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
