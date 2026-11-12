# SPDX-License-Identifier: Apache-2.0
"""Fresh-process MLLM load with inert weights and a standard expert contract."""

import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

# Import the production module first, before the fusion installer or its probe.
from rapid_mlx.models import mllm


def module(name, **attrs):
    value = ModuleType(name)
    value.__dict__.update(attrs)
    sys.modules[name] = value
    return value


mx = module(
    "mlx.core",
    metal=SimpleNamespace(is_available=lambda: True),
    default_device=lambda: "gpu",
    gpu="gpu",
    device_info=lambda: {},
)
module("mlx", core=mx)


class Experts:
    def __call__(self, x, top_k_indices, top_k_weights):
        return "stock"


upstream = module("mlx_vlm.models.gemma4.language", Experts=Experts)
block = Experts()
model = SimpleNamespace(
    named_modules=lambda: [("experts", block)], config=SimpleNamespace()
)
module("mlx_vlm", load=lambda *a, **k: (model, SimpleNamespace()))
module("mlx_vlm.utils", load_config=lambda *a, **k: {"model_type": "gemma4"})
module(
    "rapid_mlx.models.media_backend",
    native_media_model_type=lambda _: None,
    load_native_media=lambda *a: None,
    resolve_native_checkpoint=lambda _: None,
)
for name, installer in (
    ("glm5_next_runtime", "install_glm5_next_runtime_fix"),
    ("glm5_next_processor", "install_glm5_next_processor_patch"),
    ("glm5_next_forget_gate_quant", "install_glm5_next_forget_gate_quant_fix"),
):
    module("rapid_mlx.patches." + name, **{installer: lambda: None})
module(
    "rapid_mlx.utils.tokenizer",
    apply_remote_code_policy=lambda _: ({}, False),
    augment_eos_token_ids_from_generation_config=lambda *a: None,
    repair_byte_level_decoder=lambda *a: None,
)
module(
    "rapid_mlx.utils.model_file_guard",
    validate_model_code=lambda *a, **k: None,
    validate_vlm_processor_policy=lambda *a, **k: None,
    validate_local_model_file=lambda *a: None,
)
module("rapid_mlx.dense_decode_fusion", install_dense_decode_fusion=lambda *a, **k: 0)
# The real gate/up installer sees no compatible SwitchGLU in this tiny model.
module("mlx_lm.models.switch_layers")
mllm._require_mlx_vlm = lambda: None
mllm._resolve_checkpoint = lambda _: sys.argv[1]
os.environ.pop("RAPID_MLX_MOE_DECODE_FUSION", None)
os.environ.pop("RAPID_MLX_MOE_GATE_UP_FUSION", None)
os.environ.pop("RAPID_MLX_BENCHMARK_KNOBS", None)
Path(sys.argv[1], "config.json").write_text('{"model_type":"gemma4"}')
wrapper = mllm.MLXMultimodalLM(sys.argv[1], enable_cache=False)
wrapper.load()

from rapid_mlx.patches.gemma4_moe_decode_fusion import is_installed  # noqa: E402

assert wrapper._loaded
assert is_installed(wrapper.model)
assert getattr(type(block), "_rapid_gemma4_moe_decode_fusion", False)
assert type(block).__bases__ == (upstream.Experts,)
print("OK")
