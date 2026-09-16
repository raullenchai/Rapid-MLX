# SPDX-License-Identifier: Apache-2.0
"""
Vendored Gemma 4 model classes from mlx-vlm 0.6.3.

**Why vendor:** Google publishes Gemma 4 checkpoints under a VLM-family
architecture, so the Python model class lives in `mlx-vlm`. In 0.10.0 we
listed `mlx-vlm` under the `[vision]` extra to keep text-only installs
lean; but a fresh `pip install rapid-mlx==0.10.0 && rapid-mlx serve
gemma-4-12b-4bit` then failed with `ImportError: mlx-vlm dependency
missing`, contradicting our own release notes' "+13 Gemma 4 aliases"
headline. Promoting `mlx-vlm` to a core dep would drag in ~+483 MB of
transitive weight (opencv-python 120 MB, pyarrow 123 MB from datasets,
pandas 70 MB, scipy 98 MB, mlx-audio 17 MB, ...) for text-only users who
never touch Gemma 4 — a bad trade. Vendoring the ~50 KB of Python
source classes we actually need is +200 KB of repo, zero MB of user
install bloat, and stays fully compatible with mlx-lm's cache/attention
plumbing (which is already in core).

**Upstream:** https://github.com/Blaizzy/mlx-vlm/tree/v0.6.3/mlx_vlm/models/gemma4

**Vendored files:**
- `config.py` — verbatim copy of `mlx_vlm/models/gemma4/config.py`
- `language.py` — copy with 2 import redirects:
  - `..base` symbols (LanguageModelOutput, create_attention_mask,
    scaled_dot_product_attention) → this module for LanguageModelOutput,
    `mlx_lm.models.base` for the two attention helpers.
  - `..cache` symbols (KVCache, RotatingKVCache) → `mlx_lm.models.cache`
    (same classes; mlx-vlm re-exports them from mlx-lm anyway).
- `rope_utils.py` — verbatim copy.

**Sync policy:** When Google publishes a new Gemma variant (Gemma 4.1,
Gemma 5, etc.) or mlx-vlm ships a Gemma 4 bug fix, diff the four files
above against upstream and cherry-pick. Typical sync = 30 min every
3-6 months.

**BaseModelConfig** (below) is inlined because it's a trivial
20-line dataclass mixin (`from_dict` + `to_dict`); it's not worth
vendoring `mlx_vlm/models/base.py` for it because that file also
imports PIL and mlx_vlm.turboquant which we don't need for text.

**LanguageModelOutput** is inlined for the same reason.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Dict, List, Optional

import mlx.core as mx


@dataclass
class BaseModelConfig:
    """Dataclass mixin providing from_dict/to_dict. Vendored inline —
    see module docstring for why we don't vendor `mlx_vlm.models.base`
    wholesale (it drags PIL + mlx_vlm.turboquant)."""

    @classmethod
    def from_dict(cls, params):
        if not params:
            return cls()
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class LanguageModelOutput:
    """Return type of Gemma 4 LanguageModel.__call__(). Vendored inline —
    original lives in `mlx_vlm.models.base` alongside a lot of VLM-only
    fields (cross_attention_states, encoder_outputs, gdn_states,
    shared_kv_states). We keep those field names for byte-compat with
    upstream even though the LLM path only reads `.logits`."""

    logits: mx.array
    hidden_states: Optional[List[mx.array]] = None
    cross_attention_states: Optional[List[mx.array]] = None
    encoder_outputs: Optional[List[mx.array]] = None
    gdn_states: Optional[List] = None
    shared_kv_states: Optional[Dict[str, tuple]] = None
