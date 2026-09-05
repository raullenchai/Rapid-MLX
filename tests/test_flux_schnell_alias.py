# SPDX-License-Identifier: Apache-2.0
"""Public catalog contract for the FLUX.1 schnell image alias."""

from vllm_mlx._download_gate import IMAGE_MODEL_REVISIONS
from vllm_mlx.image.engine import ImageGenerationEngine
from vllm_mlx.model_aliases import list_profiles, resolve_profile
from vllm_mlx.model_sizes import size_bytes
from vllm_mlx.runtime.resident_models import estimate_model_bytes

REPO = "mflux-community/flux-1-schnell-mflux-q4"
REVISION = "bcdbe817ad51175959b2e691e64eca626db30558"


def test_flux_schnell_alias_is_generation_only_mflux_model():
    profile = resolve_profile("flux-schnell")

    assert profile is not None
    assert profile.hf_path == REPO
    assert profile.modality == "image-gen"
    assert profile.min_memory_gb == 16

    engine = ImageGenerationEngine(profile.hf_path)
    assert engine.family == "flux-schnell"
    assert engine.supports_generation is True
    assert engine.supports_editing is False
    assert engine.default_steps == 4
    assert engine._prequantized is True
    assert engine._quantize is None


def test_flux_schnell_weights_are_revision_pinned():
    assert list_profiles()["flux-schnell"].hf_path == REPO
    assert IMAGE_MODEL_REVISIONS[REPO] == REVISION
    assert size_bytes(REPO) == 9_613_040_056
    assert estimate_model_bytes("flux-schnell") == int(9.5 * 1024**3)
