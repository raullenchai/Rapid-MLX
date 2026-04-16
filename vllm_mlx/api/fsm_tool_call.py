# SPDX-License-Identifier: Apache-2.0
"""
FSM-based tool call constrained decoding.

Uses outlines-core's finite state machine to guarantee valid JSON
tool calls during generation.  Replaces the 18 regex-based parsers
with a single FSM that:

1. Lets the model generate freely until it outputs a tool call trigger
2. Switches to constrained mode — only FSM-valid tokens are allowed
3. Guarantees the output is valid JSON matching the tool schema
4. Switches back to free mode after the JSON body is complete

Performance:
- FSM compilation: ~2-8s (once per tool schema, cached by hash)
- Per-token overhead: 0.8 µs (0.004% of a 20ms decode step)
- Precompiled at server startup → zero latency for users
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from outlines_core import Guide, Index, Vocabulary, json_schema

    HAS_OUTLINES_CORE = True
except ImportError:
    HAS_OUTLINES_CORE = False
    Guide = None
    Index = None
    Vocabulary = None
    json_schema = None


# =====================================================================
# Tool call trigger patterns (per parser format)
#
# Each entry maps parser_name → (trigger_suffix, closing_tag).
# - trigger_suffix: text that signals the start of a JSON tool call
# - closing_tag: expected text after the JSON body (for clean extraction)
# =====================================================================

TOOL_CALL_TRIGGERS: dict[str, tuple[str, str]] = {
    # Hermes/Qwen/NousResearch — JSON format
    "hermes": ("<tool_call>\n", "\n</tool_call>"),
    "nous": ("<tool_call>\n", "\n</tool_call>"),
    "qwen": ("<tool_call>\n", "\n</tool_call>"),
    "qwen3": ("<tool_call>\n", "\n</tool_call>"),
    "qwen3_coder": ("<tool_call>\n", "\n</tool_call>"),
    "glm47": ("<tool_call>\n", "\n</tool_call>"),
    "glm4": ("<tool_call>\n", "\n</tool_call>"),
    "granite": ("<tool_call>\n", "\n</tool_call>"),
    "granite3": ("<tool_call>\n", "\n</tool_call>"),
    # Llama — function format
    "llama": ("<function=", "</function>"),
    "llama3": ("<function=", "</function>"),
    "llama4": ("<function=", "</function>"),
    # Functionary
    "functionary": ("<function=", "</function>"),
    "meetkai": ("<function=", "</function>"),
    # Nemotron
    "nemotron": ("<function=", "</function>"),
    "nemotron3": ("<function=", "</function>"),
    # Qwen3 Coder XML
    "qwen3_coder_xml": ("<tool_call>\n", "\n</tool_call>"),
    "qwen3_xml": ("<tool_call>\n", "\n</tool_call>"),
    # Seed OSS
    "seed_oss": ("<seed:tool_call>", "</seed:tool_call>"),
    "seed": ("<seed:tool_call>", "</seed:tool_call>"),
    "gpt_oss": ("<seed:tool_call>", "</seed:tool_call>"),
    "gpt-oss": ("<seed:tool_call>", "</seed:tool_call>"),
    "harmony": ("<seed:tool_call>", "</seed:tool_call>"),
    # MiniMax — XML invoke format (trigger is different)
    "minimax": ("<minimax:tool_call>", "</minimax:tool_call>"),
    "minimax_m2": ("<minimax:tool_call>", "</minimax:tool_call>"),
    # Mistral
    "mistral": ("[TOOL_CALLS]", ""),
    # DeepSeek
    "deepseek": ("<｜tool▁sep｜>", "<｜tool▁call▁end｜>"),
    "deepseek_v3": ("<｜tool▁sep｜>", "<｜tool▁call▁end｜>"),
    "deepseek_r1": ("<｜tool▁sep｜>", "<｜tool▁call▁end｜>"),
    "deepseek_v31": ("<｜tool▁sep｜>", "<｜tool▁call▁end｜>"),
    "deepseek_r1_0528": ("<｜tool▁sep｜>", "<｜tool▁call▁end｜>"),
    # Kimi
    "kimi": ("<|tool_call_argument_begin|>", "<|tool_call_end|>"),
    "kimi_k2": ("<|tool_call_argument_begin|>", "<|tool_call_end|>"),
    "moonshot": ("<|tool_call_argument_begin|>", "<|tool_call_end|>"),
    # Gemma 4
    "gemma4": ("<|tool_call>", "<tool_call|>"),
    "gemma_4": ("<|tool_call>", "<tool_call|>"),
    # xLAM
    "xlam": ("[TOOL_CALLS]", ""),
    # Auto/Generic
    "auto": ("<tool_call>\n", "\n</tool_call>"),
    "generic": ("<tool_call>\n", "\n</tool_call>"),
}


# =====================================================================
# FSM Compilation Cache
# =====================================================================


def _schema_hash(tools: list[dict]) -> str:
    """Stable hash of tool definitions for cache keying."""
    canonical = json.dumps(tools, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _build_tool_call_schema(tools: list[dict]) -> str:
    """Build a JSON schema that matches any valid tool call.

    Produces: {"name": "<one of tool names>", "arguments": {<any object>}}

    This is the "fast path" — constrains JSON structure without
    validating per-tool argument schemas (which would slow compilation).
    """
    tool_names = []
    for tool in tools:
        func = tool.get("function", tool)
        name = func.get("name", "")
        if name:
            tool_names.append(name)

    if not tool_names:
        # Fallback: any string name
        name_schema: dict[str, Any] = {"type": "string"}
    elif len(tool_names) == 1:
        name_schema = {"type": "string", "const": tool_names[0]}
    else:
        name_schema = {"type": "string", "enum": tool_names}

    schema = {
        "type": "object",
        "properties": {
            "name": name_schema,
            "arguments": {"type": "object"},
        },
        "required": ["name", "arguments"],
    }
    return json.dumps(schema)


class FSMToolCallCache:
    """Cache of compiled FSM indices for tool call schemas.

    Compilation is expensive (~2-8s) but the result is reused for all
    requests with the same tool set.  Call ``precompile()`` at server
    startup to hide the cost from users.
    """

    def __init__(self, vocabulary: Any | None = None):
        self._vocabulary = vocabulary
        self._cache: dict[str, tuple[Any, Any]] = {}  # hash → (Index, Guide template)

    def set_vocabulary(self, tokenizer: Any) -> None:
        """Build vocabulary from a HuggingFace tokenizer.

        ``Vocabulary.from_pretrained`` requires a HuggingFace model ID
        (not a local path) because it downloads ``tokenizer.json``
        internally.  If ``name_or_path`` looks like a local path, we
        resolve the original model ID from the HF cache metadata.
        """
        if not HAS_OUTLINES_CORE:
            return
        try:
            import os

            model_id = getattr(tokenizer, "name_or_path", "") or ""

            # If name_or_path is a local path, try to resolve the HF model ID
            if os.sep in model_id or model_id.startswith("/"):
                resolved_id = self._resolve_hf_model_id(model_id)
                if resolved_id:
                    model_id = resolved_id
                else:
                    logger.warning(
                        f"[FSM] Cannot resolve HF model ID from local path: "
                        f"{model_id}. FSM constrained decoding unavailable."
                    )
                    return

            self._vocabulary = Vocabulary.from_pretrained(model_id)
            logger.info(
                f"[FSM] Vocabulary built: {len(self._vocabulary)} tokens "
                f"(model={model_id})"
            )
        except Exception as e:
            logger.warning(f"[FSM] Failed to build vocabulary: {e}")
            self._vocabulary = None

    @staticmethod
    def _resolve_hf_model_id(local_path: str) -> str | None:
        """Try to extract HF model ID from a local snapshot path.

        HF cache layout: .../models--{org}--{repo}/snapshots/{hash}/
        """
        import re

        match = re.search(r"models--([^/]+)--([^/]+)", local_path)
        if match:
            org, repo = match.group(1), match.group(2)
            return f"{org}/{repo}"
        return None

    def precompile(self, tools: list[dict]) -> bool:
        """Precompile FSM for a tool set.  Returns True on success."""
        if not HAS_OUTLINES_CORE or self._vocabulary is None:
            return False

        key = _schema_hash(tools)
        if key in self._cache:
            return True

        schema_str = _build_tool_call_schema(tools)
        try:
            import time

            t0 = time.perf_counter()
            regex = json_schema.build_regex_from_schema(schema_str)
            index = Index(regex, self._vocabulary)
            dt = time.perf_counter() - t0

            self._cache[key] = index
            n_tools = len(
                [t for t in tools if t.get("function", t).get("name")]
            )
            logger.info(
                f"[FSM] Precompiled tool call grammar: {n_tools} tools, "
                f"{len(regex)} char regex, {dt:.1f}s compile time "
                f"(cached as {key})"
            )
            return True
        except Exception as e:
            logger.warning(f"[FSM] Failed to compile tool call grammar: {e}")
            return False

    def get_guide(self, tools: list[dict]) -> Any | None:
        """Get a fresh Guide for the given tools.  Compiles on miss."""
        if not HAS_OUTLINES_CORE or self._vocabulary is None:
            return None

        key = _schema_hash(tools)
        if key not in self._cache:
            if not self.precompile(tools):
                return None

        index = self._cache[key]
        return Guide(index)


# Global cache instance
_fsm_cache = FSMToolCallCache()


def get_fsm_cache() -> FSMToolCallCache:
    """Get the global FSM cache instance."""
    return _fsm_cache


# =====================================================================
# FSM Logits Processor
# =====================================================================


class FSMToolCallProcessor:
    """Logits processor that constrains tool call JSON via FSM.

    Two modes:
    - **Free mode** (default): all tokens allowed, no constraint.
      Model generates text, reasoning, etc. freely.
    - **Constrained mode**: activated when model outputs a tool call
      trigger (e.g., ``<tool_call>\\n``).  Only FSM-valid tokens are
      allowed, guaranteeing valid JSON output.

    The processor tracks recent output text to detect triggers.
    When a trigger is found, it creates a Guide from the cached FSM
    Index and masks invalid tokens until the JSON body is complete.
    """

    def __init__(
        self,
        tokenizer: Any,
        tools: list[dict],
        parser_name: str = "hermes",
        cache: FSMToolCallCache | None = None,
    ):
        self.tokenizer = tokenizer
        self.tools = tools
        self.parser_name = parser_name
        self._cache = cache or _fsm_cache

        # Resolve trigger pattern
        trigger_info = TOOL_CALL_TRIGGERS.get(parser_name)
        self._trigger = trigger_info[0] if trigger_info else None
        self._closing = trigger_info[1] if trigger_info else None

        # State
        self._recent_text = ""
        self._guide: Any | None = None  # Active Guide when in constrained mode
        self._constrained = False
        self._json_depth = 0  # Track brace depth for JSON completion

    def reset(self) -> None:
        """Reset for a new generation."""
        self._recent_text = ""
        self._guide = None
        self._constrained = False
        self._json_depth = 0

    def __call__(self, token_ids: Any, logits: Any) -> Any:
        """Apply FSM constraint to logits.

        In free mode, returns logits unchanged.
        In constrained mode, masks all tokens not allowed by the FSM.
        """
        import mlx.core as mx

        # Decode last token
        if hasattr(token_ids, "tolist"):
            id_list = token_ids.tolist()
        else:
            id_list = list(token_ids)

        if not id_list:
            return logits

        last_tok = id_list[-1]
        last_text = self.tokenizer.decode([last_tok], skip_special_tokens=False)
        self._recent_text += last_text
        if len(self._recent_text) > 500:
            self._recent_text = self._recent_text[-500:]

        # --- Constrained mode: mask invalid tokens ---
        if self._constrained and self._guide is not None:
            # Advance FSM state with the token we just generated
            try:
                self._guide.advance(last_tok)
            except Exception:
                # Token not in FSM vocabulary — deactivate
                logger.debug("[FSM] Token not in vocabulary, deactivating")
                self._constrained = False
                self._guide = None
                return logits

            if self._guide.is_finished():
                # JSON body complete — back to free mode
                self._constrained = False
                self._guide = None
                logger.debug("[FSM] JSON body complete, back to free mode")
                return logits

            # Get allowed tokens and mask logits
            allowed = self._guide.get_tokens()
            if allowed:
                mask = mx.full(logits.shape, -float("inf"))
                allowed_arr = mx.array(allowed)
                if logits.ndim == 2:
                    mask[0, allowed_arr] = 0.0
                else:
                    mask[allowed_arr] = 0.0
                return logits + mask

            return logits

        # --- Free mode: check for trigger ---
        if self._trigger and self._recent_text.endswith(self._trigger):
            guide = self._cache.get_guide(self.tools)
            if guide is not None:
                self._guide = guide
                self._constrained = True
                logger.debug(
                    f"[FSM] Trigger detected: {self._trigger!r} → "
                    "constrained mode"
                )

                # Immediately mask for the NEXT token
                allowed = self._guide.get_tokens()
                if allowed:
                    mask = mx.full(logits.shape, -float("inf"))
                    allowed_arr = mx.array(allowed)
                    if logits.ndim == 2:
                        mask[0, allowed_arr] = 0.0
                    else:
                        mask[allowed_arr] = 0.0
                    return logits + mask

        return logits


# =====================================================================
# Factory
# =====================================================================


def create_fsm_processor(
    parser_name: str,
    tokenizer: Any,
    tools: list[dict] | None = None,
) -> FSMToolCallProcessor | None:
    """Create an FSM tool call processor.

    Returns None if outlines-core is not installed or tools are empty.
    """
    if not HAS_OUTLINES_CORE:
        logger.debug("[FSM] outlines-core not installed, skipping")
        return None

    if not tools:
        return None

    if parser_name not in TOOL_CALL_TRIGGERS:
        logger.debug(f"[FSM] No trigger pattern for parser {parser_name!r}")
        return None

    return FSMToolCallProcessor(
        tokenizer=tokenizer,
        tools=tools,
        parser_name=parser_name,
        cache=_fsm_cache,
    )


def is_fsm_available() -> bool:
    """Check if FSM constrained decoding is available."""
    return HAS_OUTLINES_CORE
