"""Gemma 3 language model wrapper via mlx-lm.

Ported from ltx-core/src/ltx_core/text_encoders/gemma/encoders/base_encoder.py
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


class GemmaLanguageModel(nn.Module):
    """Wrapper around Gemma 3 12B loaded via mlx-lm.

    Uses mlx_lm.load() for native MLX loading.
    Extracts hidden states from ALL layers for multi-layer feature extraction.

    Gemma 3 12B has 48 transformer layers + embedding layer = 49 total
    hidden states (embedding output + 48 layer outputs), each of dim 3840.

    Args:
        model_path: Path to the Gemma 3 MLX weights directory.
    """

    def __init__(self, model_path: str | Path | None = None):
        super().__init__()
        self._model = None
        self._tokenizer = None
        self._model_path = str(model_path) if model_path else None

    def load(self, model_path: str | None = None) -> None:
        """Load the Gemma model via mlx-lm.

        Args:
            model_path: Path or HuggingFace repo ID.
        """
        from mlx_lm import load as mlx_lm_load

        path = model_path or self._model_path
        if path is None:
            raise ValueError("model_path must be provided")

        self._model, self._tokenizer = mlx_lm_load(path)

    def tokenize(self, text: str, max_length: int = 1024) -> tuple[mx.array, mx.array]:
        """Tokenize a text string with left-padding to max_length.

        Reference: LTXVGemmaTokenizer pads to max_length with padding_side="left".
        Returns both token_ids and attention_mask.

        Args:
            text: Input text.
            max_length: Sequence length (padded to this length).

        Returns:
            Tuple of (token_ids, attention_mask), each shape (1, max_length).
            attention_mask: 1 for valid tokens, 0 for padding.
        """
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        tokens = self._tokenizer.encode(text.strip())
        if len(tokens) > max_length:
            tokens = tokens[-max_length:]  # Keep last tokens (left-pad = truncate from left)

        # Left-pad to max_length using the native pad token
        pad_length = max_length - len(tokens)
        pad_token = self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else 0
        padded_tokens = [pad_token] * pad_length + tokens
        attention_mask = [0] * pad_length + [1] * len(tokens)

        return mx.array([padded_tokens]), mx.array([attention_mask])

    @staticmethod
    def _ensure_metal_headroom() -> None:
        """Set Metal cache limit to leave headroom for the GPU watchdog."""
        try:
            mem_limit = mx.device_info()["memory_size"]
            mx.set_cache_limit(int(mem_limit * 0.9))
        except Exception:
            pass

    def get_all_hidden_states(
        self,
        token_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> list[mx.array]:
        """Extract hidden states from ALL layers of the language model.

        Collects the embedding output plus all transformer layer outputs,
        yielding 49 hidden states total for Gemma 3 12B.

        Args:
            token_ids: (B, seq_len) token IDs.
            attention_mask: (B, seq_len) binary mask (1=valid, 0=padding).
                Used to build a causal attention mask that also masks padding.

        Returns:
            List of (B, seq_len, hidden_dim) tensors, one per layer.
            Length = num_transformer_layers + 1 (embedding + layers).
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        self._ensure_metal_headroom()

        # Navigate to the inner model with embed_tokens and layers.
        inner = self._model
        for attr in ("model", "language_model", "model"):
            if hasattr(inner, attr):
                inner = getattr(inner, attr)
            if hasattr(inner, "embed_tokens"):
                break

        if not hasattr(inner, "embed_tokens"):
            raise RuntimeError("Cannot find embed_tokens in the model hierarchy")

        all_hidden_states: list[mx.array] = []

        # Embeddings with Gemma 3 scaling (sqrt(hidden_size))
        h = inner.embed_tokens(token_ids)
        hidden_size = h.shape[-1]
        h = h * mx.array(hidden_size**0.5, dtype=mx.bfloat16).astype(h.dtype)
        all_hidden_states.append(h)

        # Build combined causal + padding mask.
        # Causal mask prevents attending to future tokens; padding mask
        # prevents attending to padding tokens (left-padded input).
        # Must be bfloat16 to match mlx-lm's scaled_dot_product_attention output type.
        T = token_ids.shape[1]
        causal_mask = mx.triu(mx.full((T, T), -1e9, dtype=mx.bfloat16), k=1)
        if attention_mask is not None:
            # attention_mask: (B, T) with 1=valid, 0=padding
            pad_mask = (1 - attention_mask[:, None, None, :].astype(mx.bfloat16)) * -1e9
            combined_mask = causal_mask[None, None, :, :] + pad_mask  # (B, 1, T, T)
        else:
            combined_mask = causal_mask[None, None, :, :]  # (1, 1, T, T)

        # Per-layer eval is necessary on <=48 GB Macs to keep each Metal
        # command buffer below the macOS GPU watchdog deadline. The
        # downstream feature_extractor projection materializes its inputs
        # in one buffer; without per-layer eval, that buffer pulls all 48
        # Gemma layers into a single dispatch that exceeds the macOS GPU
        # watchdog limit (~10 s) under sustained system contention.
        # Originally gated on <=48 GB, but M2 Max 64 GB also crashes
        # (MTLCommandBufferErrorInternal code 14) at production resolutions
        # — the prior 48 GB threshold was validated only at 384x256x9 where
        # the lazy graph fits the watchdog window. Default is now 1 (per-layer)
        # for all devices. LTX2_GEMMA_EVAL_EVERY=0 disables (full lazy graph).
        eval_every = int(os.environ.get("LTX2_GEMMA_EVAL_EVERY", "1"))
        for i, layer in enumerate(inner.layers):
            h = layer(h, mask=combined_mask, cache=None)
            if isinstance(h, tuple):
                h = h[0]
            all_hidden_states.append(h)
            if eval_every and (i + 1) % eval_every == 0:
                mx.eval(h)

        return all_hidden_states

    def encode(self, text: str, max_length: int = 1024) -> mx.array:
        """Tokenize and extract final hidden states in one call.

        Args:
            text: Input text.
            max_length: Padded sequence length.

        Returns:
            Hidden states of shape (1, max_length, hidden_dim).
        """
        token_ids, attention_mask = self.tokenize(text, max_length)
        all_states = self.get_all_hidden_states(token_ids, attention_mask=attention_mask)
        return all_states[-1]

    def encode_all_layers(self, text: str, max_length: int = 1024) -> tuple[list[mx.array], mx.array]:
        """Tokenize and extract ALL layer hidden states.

        Args:
            text: Input text.
            max_length: Padded sequence length.

        Returns:
            Tuple of (hidden_states, attention_mask):
            - hidden_states: list of (1, max_length, hidden_dim) tensors (49 total)
            - attention_mask: (1, max_length) binary mask
        """
        token_ids, attention_mask = self.tokenize(text, max_length)
        hidden_states = self.get_all_hidden_states(token_ids, attention_mask=attention_mask)
        return hidden_states, attention_mask

    # --- Prompt enhancement ---

    def enhance_t2v(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
        seed: int = 10,
    ) -> str:
        """Enhance a text prompt for T2V generation using Gemma.

        Args:
            prompt: Raw user prompt.
            max_new_tokens: Maximum tokens to generate.
            system_prompt: Custom system prompt (uses default T2V prompt if None).
            seed: Random seed for generation.

        Returns:
            Enhanced prompt string.
        """
        system_prompt = system_prompt or self.default_gemma_t2v_system_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"user prompt: {prompt}"},
        ]
        return self._enhance(messages, max_new_tokens=max_new_tokens, seed=seed)

    def enhance_i2v(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
        seed: int = 10,
    ) -> str:
        """Enhance a text prompt for I2V generation using Gemma.

        Note: Unlike the reference implementation, this does not pass the image
        to Gemma (mlx-lm does not support multimodal generation). The system
        prompt still instructs Gemma to write an I2V-style prompt.

        Args:
            prompt: Raw user prompt.
            max_new_tokens: Maximum tokens to generate.
            system_prompt: Custom system prompt (uses default I2V prompt if None).
            seed: Random seed for generation.

        Returns:
            Enhanced prompt string.
        """
        system_prompt = system_prompt or self.default_gemma_i2v_system_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User Raw Input Prompt: {prompt}."},
        ]
        return self._enhance(messages, max_new_tokens=max_new_tokens, seed=seed)

    def _enhance(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 512,
        seed: int = 10,
    ) -> str:
        """Generate an enhanced prompt from chat messages.

        Args:
            messages: Chat-formatted messages (system + user).
            max_new_tokens: Maximum tokens to generate.
            seed: Random seed.

        Returns:
            Generated text.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        # Format as chat template
        chat_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        mx.random.seed(seed)
        sampler = make_sampler(temp=0.7)
        result = mlx_generate(
            model=self._model,
            tokenizer=self._tokenizer,
            prompt=chat_text,
            max_tokens=max_new_tokens,
            sampler=sampler,
            verbose=False,
        )
        return result.strip()

    @functools.cached_property
    def default_gemma_t2v_system_prompt(self) -> str:
        """Load the default T2V system prompt."""
        return _load_system_prompt("gemma_t2v_system_prompt.txt")

    @functools.cached_property
    def default_gemma_i2v_system_prompt(self) -> str:
        """Load the default I2V system prompt."""
        return _load_system_prompt("gemma_i2v_system_prompt.txt")


class Gemma4LanguageModel(GemmaLanguageModel):
    """Gemma 4 12B (LTX-2.5 text encoder) wrapper via mlx-lm.

    Loads a local MLX checkpoint directory produced by
    ``scripts/convert_ltx25_to_mlx.py --step text-encoder``: ``config.json``
    with ``model_type: "gemma4"`` + an embedded ``text_config`` (48 layers,
    hidden 3840, 16 heads x 256, 8 kv heads + 1 global, k_eq_v on the 8
    full-attention layers, rope proportional 0.25@1e6 / default@1e4),
    ``model.safetensors`` (keys ``language_model.model.*`` after sanitize)
    and ``tokenizer.json``/``tokenizer_config.json`` extracted from the
    official single-file.

    Inherits the 49-hidden-state collection (embedding + 48 layers) from
    :class:`GemmaLanguageModel` — gemma4's ``DecoderLayer.__call__`` returns
    a plain array (not a tuple), which the parent handles; the layer call
    ``layer(h, mask=..., cache=None)`` is signature-compatible.
    """

    # LTX-2.5 gemma4-12b-ltx-v1 specs (used for validation on load).
    EXPECTED_LAYERS = 48
    EXPECTED_HIDDEN = 3840

    def load(self, model_path: str | None = None) -> None:
        """Load the Gemma 4 model via mlx-lm and validate the architecture."""
        super().load(model_path)
        self._validate()

    def _validate(self) -> None:
        """Validate the loaded model is gemma4-12b-ltx-v1 (or the expected shape).

        Override ``EXPECTED_LAYERS`` / ``EXPECTED_HIDDEN`` (e.g. in tests with
        a tiny in-memory model) to relax the architecture check.
        """
        model = self._model
        if model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        model_type = getattr(model, "model_type", None)
        if model_type != "gemma4":
            raise ValueError(
                f"Gemma4LanguageModel requires a gemma4 mlx-lm checkpoint "
                f"(config.json model_type='gemma4'), got {model_type!r}"
            )
        # Navigate to the inner text model (same path as get_all_hidden_states).
        inner = model
        for attr in ("model", "language_model", "model"):
            if hasattr(inner, attr):
                inner = getattr(inner, attr)
            if hasattr(inner, "embed_tokens"):
                break
        num_layers = len(getattr(inner, "layers", []))
        hidden = getattr(getattr(inner, "config", None), "hidden_size", None)
        if num_layers != self.EXPECTED_LAYERS:
            raise ValueError(
                f"gemma4-12b-ltx-v1 expected {self.EXPECTED_LAYERS} layers, found {num_layers}"
            )
        if hidden is not None and hidden != self.EXPECTED_HIDDEN:
            raise ValueError(
                f"gemma4-12b-ltx-v1 expected hidden_size {self.EXPECTED_HIDDEN}, found {hidden}"
            )


@functools.lru_cache(maxsize=2)
def _load_system_prompt(prompt_name: str) -> str:
    """Load a system prompt file from the prompts directory."""
    prompt_path = Path(__file__).parent / "prompts" / prompt_name
    with open(prompt_path) as f:
        return f.read()
