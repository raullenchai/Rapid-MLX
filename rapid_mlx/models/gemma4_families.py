# SPDX-License-Identifier: Apache-2.0
"""Gemma 4 family model_type allow-list — pure data, importable without mlx.

Lives apart from :mod:`rapid_mlx.models.gemma4_text` (whose module body needs
``mlx``) so offline lane-routing probes —
:func:`rapid_mlx.api.utils._text_lane_loads_model_type` — can consult the
family list on a base wheel or the no-MLX CI lane without importing the
loader. The loader module re-exports these names, so there is exactly one
definition.
"""

from __future__ import annotations

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
