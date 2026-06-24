# SPDX-License-Identifier: Apache-2.0
"""
MLX Language Model wrapper.

This module provides a wrapper around mlx-lm for LLM inference,
integrating with vLLM's model execution system.
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class GenerationOutput:
    """Output from text generation."""

    text: str
    tokens: list[int]
    finish_reason: str | None = None
    # When termination was triggered by a client-supplied stop string,
    # the matched sequence.  Propagated to ``RequestOutput.matched_stop``
    # so the Anthropic adapter can emit the spec-required
    # ``stop_reason="stop_sequence"`` + ``stop_sequence="<matched>"``
    # (issue #469).
    matched_stop: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class StreamingOutput:
    """Streaming output chunk."""

    text: str
    token: int
    finished: bool = False
    finish_reason: str | None = None
    prompt_tokens: int = 0
    # See ``GenerationOutput.matched_stop`` (issue #469).
    matched_stop: str | None = None


class MLXLanguageModel:
    """
    Wrapper around mlx-lm for LLM inference.

    This class provides a unified interface for loading and running
    inference on language models using Apple's MLX framework.

    Example:
        >>> model = MLXLanguageModel("mlx-community/Llama-3.2-3B-Instruct-4bit")
        >>> output = model.generate("Hello, how are you?", max_tokens=100)
        >>> print(output.text)
    """

    def __init__(
        self,
        model_name: str,
        tokenizer_name: str | None = None,
        trust_remote_code: bool = False,
        mtp: bool = False,
        mtp_num_draft_tokens: int = 1,
    ):
        """
        Initialize the MLX language model.

        Args:
            model_name: HuggingFace model name or local path
            tokenizer_name: Optional separate tokenizer name
            trust_remote_code: Whether to trust remote code
            mtp: Enable native MTP speculative decoding (model must have MTP head)
            mtp_num_draft_tokens: Draft tokens per speculative MTP step
        """
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.trust_remote_code = trust_remote_code
        self._mtp = mtp
        self._mtp_num_draft_tokens = mtp_num_draft_tokens

        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> None:
        """Load the model and tokenizer."""
        if self._loaded:
            return

        try:
            from ..utils.tokenizer import load_model_with_fallback

            logger.info(f"Loading model: {self.model_name}")

            # Build tokenizer config
            tokenizer_config = {"trust_remote_code": self.trust_remote_code}

            # Qwen3 fix: eos_token changed from <|im_end|> to <|endoftext|>
            # but chat template still uses <|im_end|>, so we need to set it explicitly
            if "qwen3" in self.model_name.lower() or "Qwen3" in self.model_name:
                tokenizer_config["eos_token"] = "<|im_end|>"
                logger.info("Qwen3 detected: setting eos_token to <|im_end|>")

            self.model, self.tokenizer = load_model_with_fallback(
                self.model_name,
                tokenizer_config=tokenizer_config,
            )

            self._loaded = True
            logger.info(f"Model loaded successfully: {self.model_name}")

        except ImportError as err:
            raise ImportError(
                "mlx-lm is required for LLM inference. Install with: pip install mlx-lm"
            ) from err
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _create_sampler(
        self,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
    ):
        """Create a sampler for text generation."""
        from mlx_lm.sample_utils import make_sampler

        return make_sampler(
            temp=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )

    def _create_logits_processors(
        self,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
    ):
        """Create logits processors for penalty-based sampling."""
        from mlx_lm.sample_utils import make_logits_processors

        processors = make_logits_processors(
            repetition_penalty=(
                repetition_penalty if repetition_penalty != 1.0 else None
            ),
            presence_penalty=presence_penalty if presence_penalty != 0.0 else None,
        )
        return processors if processors else None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        stop: list[str] | None = None,
        logits_processors: list | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Top-p (nucleus) sampling parameter
            top_k: Top-k sampling (0 = disabled)
            min_p: Minimum probability threshold
            presence_penalty: Additive penalty for token presence
            repetition_penalty: Multiplicative penalty for repeating tokens
            stop: List of stop sequences
            logits_processors: Optional externally-supplied logits processors
                (e.g. JSON schema constrained decoding).  Merged with built-in
                penalty processors.

        Returns:
            GenerationOutput with generated text and tokens
        """
        if not self._loaded:
            self.load()

        from mlx_lm import generate

        # Create sampler and logits processors with full Unsloth params
        sampler = self._create_sampler(temperature, top_p, top_k, min_p)
        penalty_processors = self._create_logits_processors(
            presence_penalty, repetition_penalty
        )
        # Merge any externally-provided logits_processors with penalty processors
        all_processors = penalty_processors or []
        if logits_processors:
            all_processors = list(logits_processors) + all_processors

        # Generate text
        output_text = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=all_processors if all_processors else None,
            verbose=False,
        )

        # Enforce client-supplied stop strings (mlx_lm.generate has no
        # native ``stop`` support) and surface the matched sequence so the
        # Anthropic adapter can emit ``stop_reason="stop_sequence"`` +
        # ``stop_sequence="<matched>"`` per spec (issue #469).
        matched_stop_value: str | None = None
        if stop:
            best_idx: int | None = None
            for cand in stop:
                if not cand:
                    continue
                idx = output_text.find(cand)
                if idx == -1:
                    continue
                if best_idx is None or idx < best_idx or (
                    idx == best_idx
                    and len(cand) > len(matched_stop_value or "")
                ):
                    best_idx = idx
                    matched_stop_value = cand
            if matched_stop_value is not None and best_idx is not None:
                output_text = output_text[:best_idx]

        # Tokenize output to get token IDs
        tokens = self.tokenizer.encode(output_text)

        # Determine finish reason
        if matched_stop_value is not None:
            finish_reason = "stop"
        elif len(tokens) >= max_tokens:
            finish_reason = "length"
        else:
            finish_reason = "stop"

        return GenerationOutput(
            text=output_text,
            tokens=tokens,
            finish_reason=finish_reason,
            matched_stop=matched_stop_value,
        )

    def stream_generate(
        self,
        prompt: Union[str, "mx.array", list[int]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        stop: list[str] | None = None,
        logits_processors: list | None = None,
        prompt_cache=None,
        **kwargs,
    ) -> Iterator[StreamingOutput]:
        """
        Stream text generation token by token.

        Args:
            prompt: Input prompt text, token array, or token id list
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Top-p (nucleus) sampling parameter
            top_k: Top-k sampling (0 = disabled)
            min_p: Minimum probability threshold
            presence_penalty: Additive penalty for token presence
            repetition_penalty: Multiplicative penalty for repeating tokens
            stop: List of stop sequences
            prompt_cache: Pre-populated KV cache (e.g. from SpecPrefill)

        Yields:
            StreamingOutput for each generated token
        """
        if not self._loaded:
            self.load()

        from mlx_lm import stream_generate

        # Create sampler and logits processors with full Unsloth params
        sampler = self._create_sampler(temperature, top_p, top_k, min_p)
        penalty_processors = self._create_logits_processors(
            presence_penalty, repetition_penalty
        )
        # Merge any externally-provided logits_processors with penalty processors
        all_processors = None
        if penalty_processors or logits_processors:
            all_processors = (logits_processors or []) + (penalty_processors or [])

        # Count prompt tokens once upfront
        if isinstance(prompt, str):
            num_prompt_tokens = len(self.tokenizer.encode(prompt))
        else:
            num_prompt_tokens = len(prompt)

        stop_list = [s for s in (stop or []) if s]
        max_stop_len = max((len(s) for s in stop_list), default=0)
        # Per-request buffers for stop-string enforcement (issue #469).
        # ``cumulative_text`` is the full streamed output so far; we hold
        # back the trailing ``max_stop_len - 1`` chars from emission so a
        # partial prefix of a stop string never leaks into the SSE delta.
        cumulative_text = ""
        published_text_len = 0
        matched_stop_value: str | None = None

        mtp_kwargs = {}
        if self._mtp:
            mtp_kwargs["num_draft_tokens"] = self._mtp_num_draft_tokens
        if prompt_cache is not None:
            mtp_kwargs["prompt_cache"] = prompt_cache

        for token_count, response in enumerate(
            stream_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=all_processors,
                **mtp_kwargs,
            ),
            start=1,
        ):
            # response.text is the new token text (not accumulated)
            raw_new_text = response.text

            if stop_list:
                cumulative_text += raw_new_text
                # Scan the cumulative buffer (bounded by output length;
                # acceptable in practice for non-pathological outputs and
                # stop-list sizes).  Earliest match wins; on tie, longer.
                best_idx: int | None = None
                for stop_seq in stop_list:
                    idx = cumulative_text.find(stop_seq)
                    if idx == -1:
                        continue
                    if best_idx is None or idx < best_idx or (
                        idx == best_idx
                        and len(stop_seq) > len(matched_stop_value or "")
                    ):
                        best_idx = idx
                        matched_stop_value = stop_seq
                if matched_stop_value is not None and best_idx is not None:
                    cumulative_text = cumulative_text[:best_idx]
                    new_text = cumulative_text[published_text_len:]
                    published_text_len = len(cumulative_text)
                    finished = True
                    finish_reason = "stop"
                else:
                    finished = token_count >= max_tokens
                    if finished:
                        # Flush the full safety-window remainder.
                        new_text = cumulative_text[published_text_len:]
                        published_text_len = len(cumulative_text)
                        finish_reason = "length"
                    else:
                        hold_back = max_stop_len - 1
                        safe_end = max(0, len(cumulative_text) - hold_back)
                        if safe_end > published_text_len:
                            new_text = cumulative_text[published_text_len:safe_end]
                            published_text_len = safe_end
                        else:
                            new_text = ""
                        finish_reason = None
            else:
                new_text = raw_new_text
                finished = token_count >= max_tokens
                finish_reason = "length" if finished else None

            yield StreamingOutput(
                text=new_text,
                token=response.token if hasattr(response, "token") else 0,
                finished=finished,
                finish_reason=finish_reason,
                prompt_tokens=num_prompt_tokens,
                matched_stop=matched_stop_value if finished else None,
            )

            if finished:
                break

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list | None = None,
        chat_template_kwargs: dict | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate a chat response.

        Args:
            messages: List of chat messages [{"role": "user", "content": "..."}]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            tools: Optional list of tools for function calling
            **kwargs: Additional generation parameters

        Returns:
            GenerationOutput with the assistant's response
        """
        if not self._loaded:
            self.load()

        # Apply chat template
        if hasattr(self.tokenizer, "apply_chat_template"):
            # Build kwargs for apply_chat_template
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }

            # Add tools if provided and supported
            if tools:
                template_kwargs["tools"] = tools
            if chat_template_kwargs:
                template_kwargs.update(chat_template_kwargs)

            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    **template_kwargs,
                )
            except TypeError:
                # Tokenizer doesn't support all requested template kwargs
                template_kwargs.pop("tools", None)
                for key in (chat_template_kwargs or {}).keys():
                    template_kwargs.pop(key, None)
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    **template_kwargs,
                )
        else:
            # Fallback: simple concatenation
            prompt = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
            prompt += "\nassistant:"

        return self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self._loaded:
            return {"loaded": False, "model_name": self.model_name}

        info = {
            "loaded": True,
            "model_name": self.model_name,
            "tokenizer_name": self.tokenizer_name,
        }

        # Try to get model config
        if hasattr(self.model, "config"):
            config = self.model.config
            info.update(
                {
                    "vocab_size": getattr(config, "vocab_size", None),
                    "hidden_size": getattr(config, "hidden_size", None),
                    "num_layers": getattr(config, "num_hidden_layers", None),
                    "num_heads": getattr(config, "num_attention_heads", None),
                }
            )

        return info

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"<MLXLanguageModel model={self.model_name} status={status}>"
