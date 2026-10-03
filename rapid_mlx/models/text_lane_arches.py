# SPDX-License-Identifier: Apache-2.0
"""Text-lane architecture evidence — pure data, importable without mlx.

Lives apart from the loader modules (``gemma4_text`` imports ``mlx`` in its
module body) so offline lane-routing probes —
:func:`rapid_mlx.api.utils._text_lane_loads_model_type` — can consult these
lists on a base wheel or the no-MLX CI lane without importing a loader. The
loader module re-exports the Gemma 4 tuples, so there is exactly one
definition of each.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Gemma 4 family: arches served by the vendored text loaders.
# ---------------------------------------------------------------------------
# Model_types the Gemma 4 text loader path claims. This is a deliberate
# exact-match allow-list, NOT a ``"gemma4" in model_type`` substring test
# (see #509). The substring check also matched a hypothetical future
# ``gemma4_videogen`` (or ``gemma4_text`` — the inner text sub-config's
# own model_type) and would silently misroute it. Each member is
# classified by :func:`gemma4_family_kind` and routed to the matching
# ``load_gemma4_*`` loader; adding a new supported arch is a one-line
# edit here plus a loader branch, which is the point — routing is
# explicit and unknown arches surface loudly instead of silently riding
# the text path.
#
# - ``gemma4``           : non-unified text arch (26B/31B/e2b/e4b,
#                          ``Gemma4ForConditionalGeneration``).
# - ``gemma4_unified``   : unified arch (the four ``gemma-4-12b-*``
#                          aliases, ``Gemma4UnifiedForConditionalGeneration``).
# - ``gemma4_assistant`` : the ``gemma-4-*-assistant`` aliases
#                          (``Gemma4AssistantForCausalLM``). Its nested
#                          ``text_config`` is a ``gemma4_text`` shape, so
#                          it loads through the SAME non-unified path the
#                          old substring match sent it down. Kept in the
#                          allow-list to preserve that pre-#509 behavior
#                          (dropping it would regress those aliases to the
#                          unsupported native-load path).
_GEMMA4_NONUNIFIED_MODEL_TYPES = ("gemma4", "gemma4_assistant")
_GEMMA4_UNIFIED_MODEL_TYPES = ("gemma4_unified",)
_GEMMA4_FAMILY_MODEL_TYPES = (
    _GEMMA4_NONUNIFIED_MODEL_TYPES + _GEMMA4_UNIFIED_MODEL_TYPES
)

# ---------------------------------------------------------------------------
# mlx-lm arches that load a MULTIMODAL checkpoint's language backbone.
# ---------------------------------------------------------------------------
# The text-degrade probe may only route a MULTIMODAL checkpoint (a config
# with a vision tower / nested text_config) to the ``--no-mllm`` text lane
# when the installed mlx-lm's ``mlx_lm.models.<model_type>`` module actually
# handles that layout: its ModelArgs must consume the nested ``text_config``,
# build the ``language_model`` from it, and its ``sanitize`` must drop the
# vision weights (``vision_tower`` / ``vision_model`` /
# ``multi_modal_projector`` / ``model.visual`` / ``embed_audio``) while
# routing ``language_model.*`` keys. A module that merely exists (``llama``,
# ``qwen3``, …) does NOT prove that — routing a multimodal config there
# bypasses the ``[vision]`` guard and crashes at load with a worse error.
#
# Every entry below was VERIFIED against mlx-lm 0.31.3 by reading the module
# (file:line cite the text_config/language_model construction and the
# vision-dropping sanitize). Adding an arch requires re-reading its module.
#
# - gemma3        models/gemma3.py:17 text_config → :35 language_model;
#                 sanitize :51 pops vision_tower, :53 flattens language_model
# - gemma3n       models/gemma3n.py:48 text_config → :571 language_model;
#                 sanitize :604 pops vision_tower/audio_tower/embed_audio/
#                 embed_vision
# - gemma4        models/gemma4.py:17 text_config → :37 language_model;
#                 sanitize :55-62 drops vision_tower /
#                 multi_modal_projector / audio_tower / embed_vision
# - qwen2_vl      models/qwen2_vl.py:17 text_config → :31 language_model;
#                 sanitize :46 pops vision_tower, :51-52 language_model prefix
# - qwen3_vl      models/qwen3_vl.py:17 text_config → :31 language_model;
#                 sanitize :45 pops vision_tower, :50-51 language_model prefix
# - qwen3_vl_moe  models/qwen3_vl_moe.py:17 text_config → :25 language_model;
#                 sanitize :39-47 remaps language_model.{model,lm_head}
# - qwen3_5       models/qwen3_5.py:358 text_config → :372 language_model;
#                 sanitize :384-390 drops vision_tower / model.visual and
#                 remaps model.language_model → language_model.model
# - qwen3_5_moe   models/qwen3_5_moe.py:12 text_config → language_model;
#                 sanitize :23-30 drops vision_tower / model.visual and
#                 remaps model.language_model → language_model.model
# - mistral3      models/mistral3.py:17 text_config → :29-35 language_model
#                 (ministral3 or llama); sanitize :48-53 pops vision_tower /
#                 multi_modal_projector, flattens language_model
# - pixtral       models/pixtral.py:17 text_config → :31 language_model;
#                 sanitize :45 pops vision_tower
# - llama4        models/llama4.py:44/:48 text_config → :282 language_model;
#                 sanitize :291-296 filters vision_model /
#                 multi_modal_projector keys
# - kimi_vl       models/kimi_vl.py:47 text_config → :75 language_model;
#                 sanitize :84-91 keeps only non-vision_tower /
#                 non-multi_modal_projector keys
# - lfm2-vl       models/lfm2-vl.py:17 text_config → :28 language_model;
#                 sanitize :40-43 pops vision_tower / multi_modal_projector
MLX_LM_MM_BACKBONE_ARCHES = frozenset(
    {
        "gemma3",
        "gemma3n",
        "gemma4",
        "qwen2_vl",
        "qwen3_vl",
        "qwen3_vl_moe",
        "qwen3_5",
        "qwen3_5_moe",
        "mistral3",
        "pixtral",
        "llama4",
        "kimi_vl",
        "lfm2-vl",
    }
)
