# SPDX-License-Identifier: Apache-2.0
"""Snapshot-bound downloads for the pinned F5-TTS implementation."""

from pathlib import Path
from types import FunctionType


def load_f5_checkpoint(model_name: str):
    from f5_tts_mlx.cfm import F5TTS
    from vocos_mlx import Vocos

    from rapid_mlx.model_downloads import snapshot_download

    def resolve(name, patterns):
        path = Path(name).expanduser()
        return (
            path
            if path.is_dir()
            else Path(snapshot_download(name, allow_patterns=patterns))
        )

    def fetch_from_hub(name, quantization_bits=None):
        filename = (
            "model_v1.safetensors"
            if quantization_bits is None
            else f"model_v1_{quantization_bits}b.safetensors"
        )
        return resolve(name, [filename, "duration_v2.safetensors", "*.txt"])

    class PinnedVocos:
        @staticmethod
        def from_pretrained(name):
            return Vocos.from_pretrained(
                str(resolve(name, ["*.yaml", "*.safetensors"]))
            )

    # f5-tts-mlx==0.2.6 hardcodes a nested Vocos repository and its main
    # fetch helper doesn't accept a local path. Bind those two dependencies on
    # a private function copy; other requests and library globals are untouched.
    original = F5TTS.from_pretrained.__func__
    dependencies = dict(original.__globals__)
    dependencies.update(fetch_from_hub=fetch_from_hub, Vocos=PinnedVocos)
    bound = FunctionType(
        original.__code__,
        dependencies,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    bound.__kwdefaults__ = original.__kwdefaults__
    return bound(F5TTS, model_name)
