"""Probes for the vendored generation core (step 3a).

Mechanical guarantee: every function/class body in the vendored
``generate/ar.py``, ``generate/common.py``, ``generate/types.py``, and
``sample_utils.py`` is byte-identical to the pinned upstream
``mlx-vlm==0.7.1`` source. The only permitted differences are the
documented import redirects (see the package inventory), which live at
module level and therefore never enter a function's ``getsource``.

Behavioral guarantee: the vendored AR core binds the vendored primitives
(cache, inputs helpers, sampler) so the module graph stays inside the
vendored package except for the documented pinned redirects.
"""

import inspect

import pytest

import rapid_mlx.models.mlx_vlm_vendored as vendored_pkg
import rapid_mlx.models.mlx_vlm_vendored.generate as vendored_generate
import rapid_mlx.models.mlx_vlm_vendored.generate.ar as vendored_ar
import rapid_mlx.models.mlx_vlm_vendored.generate.common as vendored_common
import rapid_mlx.models.mlx_vlm_vendored.generate.types as vendored_types
import rapid_mlx.models.mlx_vlm_vendored.inputs as vendored_inputs
import rapid_mlx.models.mlx_vlm_vendored.sample_utils as vendored_sample_utils

pytest.importorskip("mlx_vlm")

# Functions whose BODY carries a documented redirect hunk (the deviation
# lives on an import line inside the body, so getsource differs by exactly
# those sentinel lines):
# - ``prepare_inputs``: the inputs.py hunks (bytes-path fsdecode; see the
#   inputs.py inventory entry; behavior-tested in
#   tests/test_mlx_vlm_vendored_inputs.py).
# - ``kv_quant_from_legacy``: kv_quant.py's documented lazy ``.turboquant``
#   redirect (see the kv_quant.py inventory entry).
# - ``generate_step`` / ``batch_generate``: ar.py's redirects to the
#   pinned speculative drafters helper and to the vendored
#   ``inputs.process_image``.
_DOCUMENTED_REDIRECT_BODIES = {
    "prepare_inputs",
    "kv_quant_from_legacy",
    "generate_step",
    "batch_generate",
}


def _body_divergences(vendored_module, upstream_module):
    diverged = []
    for name, obj in vars(vendored_module).items():
        if name.startswith("__") or name in _DOCUMENTED_REDIRECT_BODIES:
            continue
        upstream_obj = getattr(upstream_module, name, None)
        if upstream_obj is None:
            continue
        if (inspect.isfunction(obj) or inspect.isclass(obj)) and type(obj) is not type(
            upstream_obj
        ):
            diverged.append(f"{name}: kind mismatch")
            continue
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue
        try:
            vendored_src = inspect.getsource(obj)
            upstream_src = inspect.getsource(upstream_obj)
        except (OSError, TypeError):
            continue
        if vendored_src != upstream_src:
            diverged.append(name)
    return diverged


def test_vendored_ar_bodies_match_upstream():
    from mlx_vlm.generate import ar as upstream_ar

    assert _body_divergences(vendored_ar, upstream_ar) == []


def test_vendored_common_bodies_match_upstream():
    from mlx_vlm.generate import common as upstream_common

    assert _body_divergences(vendored_common, upstream_common) == []


def test_vendored_types_and_sample_utils_bodies_match_upstream():
    from mlx_vlm import sample_utils as upstream_sample_utils
    from mlx_vlm.generate import types as upstream_types

    assert _body_divergences(vendored_types, upstream_types) == []
    assert _body_divergences(vendored_sample_utils, upstream_sample_utils) == []


def test_vendored_ar_binds_vendored_primitives():
    assert vendored_ar.cache is vendored_pkg.cache
    assert vendored_ar.prepare_inputs is vendored_inputs.prepare_inputs
    assert vendored_ar.group_images_by_shape is (vendored_inputs.group_images_by_shape)
    assert vendored_ar.should_add_special_tokens is (
        vendored_inputs.should_add_special_tokens
    )
    assert vendored_ar.make_sampler is vendored_sample_utils.make_sampler
    assert vendored_ar.DEFAULT_KV_GROUP_SIZE is (vendored_common.DEFAULT_KV_GROUP_SIZE)
    assert vendored_ar.GenerateKwargs is vendored_types.GenerateKwargs


def test_generate_shim_exports_text_ar_surface_only():
    """The shim re-exports the vendored surface and nothing from the
    (not-yet-vendored) modality modules — the subset-exports deviation."""
    for name in (
        "BatchGenerator",
        "BatchResponse",
        "BatchStats",
        "PromptProcessingBatch",
        "batch_generate",
        "generate_step",
        "GenerationResult",
        "PromptCacheState",
        "generation_stream",
        "maybe_quantize_kv_cache",
        "wired_limit",
        "GenerateKwargs",
        "ProcessorLike",
    ):
        assert hasattr(vendored_generate, name), name
    # The upstream init eagerly pulls dispatch (generate/stream_generate),
    # image/audio/video/diffusion/edit_image; none is vendored in 3a.
    for name in ("generate", "stream_generate", "generate_image"):
        assert not hasattr(vendored_generate, name), name


def test_generate_step_signature_matches_upstream():
    from mlx_vlm.generate import ar as upstream_ar

    assert (
        inspect.signature(vendored_ar.generate_step).parameters.keys()
        == inspect.signature(upstream_ar.generate_step).parameters.keys()
    )
