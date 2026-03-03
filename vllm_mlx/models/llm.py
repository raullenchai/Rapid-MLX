# SPDX-License-Identifier: Apache-2.0
"""
MLX Language Model wrapper.

This module provides a wrapper around mlx-lm for LLM inference,
integrating with vLLM's model execution system.
"""

import copy
import itertools
import logging
import time
from dataclasses import dataclass
from typing import Any, Iterator

import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class GenerationOutput:
    """Output from text generation."""

    text: str
    tokens: list[int]
    finish_reason: str | None = None


@dataclass
class StreamingOutput:
    """Streaming output chunk."""

    text: str
    token: int
    finished: bool = False
    finish_reason: str | None = None
    logprobs: Any = None  # mx.array of shape [vocab_size] from mlx-lm
    prompt_tokens: int = 0


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
        draft_model: str | None = None,
        num_draft_tokens: int = 4,
        prefill_step_size: int = 2048,
        kv_bits: int | None = None,
        kv_group_size: int = 64,
    ):
        """
        Initialize the MLX language model.

        Args:
            model_name: HuggingFace model name or local path
            tokenizer_name: Optional separate tokenizer name
            trust_remote_code: Whether to trust remote code
            draft_model: Optional draft model path for speculative decoding
            num_draft_tokens: Number of tokens to generate speculatively per step
            prefill_step_size: Tokens to process per prefill chunk (default: 2048)
            kv_bits: KV cache quantization bits (None=no quantization, 4 or 8)
            kv_group_size: Group size for KV cache quantization (default: 64)
        """
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.trust_remote_code = trust_remote_code
        self.draft_model_name = draft_model
        self.num_draft_tokens = num_draft_tokens
        self.prefill_step_size = prefill_step_size
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size

        self.model = None
        self.tokenizer = None
        self.draft_model = None
        self._loaded = False

        # Prompt cache for KV reuse across requests
        self._prompt_cache = None
        self._cached_token_ids: list[int] = []
        self._cache_lock = False  # Simple guard against concurrent use

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

            # Load draft model for speculative decoding if specified
            if self.draft_model_name:
                logger.info(
                    f"Loading draft model for speculative decoding: {self.draft_model_name}"
                )
                from mlx_lm import load as mlx_load

                self.draft_model, draft_tokenizer = mlx_load(self.draft_model_name)

                # Validate tokenizer compatibility
                if draft_tokenizer.vocab_size != self.tokenizer.vocab_size:
                    logger.warning(
                        f"Draft model tokenizer vocab size ({draft_tokenizer.vocab_size}) "
                        f"differs from main model ({self.tokenizer.vocab_size}). "
                        "This may reduce speculative decoding effectiveness."
                    )

                logger.info(
                    f"Speculative decoding enabled: draft={self.draft_model_name}, "
                    f"num_draft_tokens={self.num_draft_tokens}"
                )

            self._loaded = True
            logger.info(f"Model loaded successfully: {self.model_name}")

        except ImportError:
            raise ImportError(
                "mlx-lm is required for LLM inference. "
                "Install with: pip install mlx-lm"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _create_sampler(
        self,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """Create a sampler for text generation."""
        from mlx_lm.sample_utils import make_sampler

        return make_sampler(
            temp=temperature,
            top_p=top_p,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        stop: list[str] | None = None,
    ) -> GenerationOutput:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Penalty for repeating tokens
            stop: List of stop sequences

        Returns:
            GenerationOutput with generated text and tokens
        """
        if not self._loaded:
            self.load()

        # Always use stream_generate to collect results.  This ensures
        # special tokens (e.g. Harmony's <|channel|>, <|call|>) are
        # preserved via skip_special_tokens=False decoding, and the
        # prompt cache is properly managed.
        output_text = ""
        for chunk in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        ):
            output_text += chunk.text
            if chunk.finished:
                break

        # Tokenize output to get token IDs
        tokens = self.tokenizer.encode(output_text)

        # Determine finish reason
        finish_reason = "length" if len(tokens) >= max_tokens else "stop"

        return GenerationOutput(
            text=output_text,
            tokens=tokens,
            finish_reason=finish_reason,
        )

    def _find_common_prefix_len(self, new_tokens: list[int]) -> int:
        """Find the length of the common prefix between cached and new tokens."""
        common = 0
        limit = min(len(self._cached_token_ids), len(new_tokens))
        for i in range(limit):
            if self._cached_token_ids[i] != new_tokens[i]:
                break
            common += 1
        return common

    def _save_cache_snapshot(self, token_ids: list[int]) -> None:
        """Save a deep copy of the prompt cache state for future reuse."""
        if self._prompt_cache is None:
            return
        # Store the token IDs that correspond to this cache state
        # The cache itself is the live object — we just track what's in it
        self._cached_token_ids = list(token_ids)

    def _prepare_cache_for_prompt(self, prompt_token_ids: list[int]) -> list[int]:
        """
        Prepare the prompt cache and return only the tokens that need processing.

        If the new prompt shares a prefix with the cached tokens, trim the cache
        to the common prefix and return only the suffix tokens.

        The cache may contain more entries than _cached_token_ids because
        generated tokens from the previous call are also in the cache.
        We must trim based on actual cache offset, not just tracked token count.

        Returns:
            Token IDs that still need to be processed (the non-cached suffix).
        """
        if self._prompt_cache is None:
            # First call — create fresh cache
            from mlx_lm.models.cache import make_prompt_cache
            self._prompt_cache = make_prompt_cache(self.model)
            # When using speculative decoding, mlx-lm expects the prompt_cache
            # to contain layers for both the main model and draft model:
            #   prompt_cache[:len(model.layers)] = main model cache
            #   prompt_cache[len(model.layers):] = draft model cache
            if self.draft_model is not None:
                self._prompt_cache.extend(make_prompt_cache(self.draft_model))
            self._cached_token_ids = []
            return prompt_token_ids

        common_len = self._find_common_prefix_len(prompt_token_ids)

        if common_len == 0:
            # No overlap — reset cache entirely
            for c in self._prompt_cache:
                current = c.offset if hasattr(c, 'offset') else 0
                if current > 0:
                    c.trim(current)
            self._cached_token_ids = []
            return prompt_token_ids

        # Trim cache to common prefix length.
        # Cache offset = prompt_tokens + generated_tokens from last call,
        # so we must trim (cache_offset - common_len), not just
        # (cached_token_ids_len - common_len).
        for c in self._prompt_cache:
            current = c.offset if hasattr(c, 'offset') else 0
            to_trim = current - common_len
            if to_trim > 0:
                c.trim(to_trim)
        self._cached_token_ids = self._cached_token_ids[:common_len]

        # Return only the suffix that needs processing
        suffix = prompt_token_ids[common_len:]
        return suffix

    def estimate_new_tokens(self, prompt: str) -> tuple[int, int]:
        """
        Estimate (total_tokens, new_tokens) without modifying cache state.

        Peeks at the cache overlap to determine how many tokens would need
        prefilling. Used by cloud routing to decide whether to offload.

        Returns:
            (total_tokens, new_tokens) tuple
        """
        if not self._loaded:
            self.load()

        add_special_tokens = (
            self.tokenizer.bos_token is None
            or not prompt.startswith(self.tokenizer.bos_token)
        )
        full_token_ids = self.tokenizer.encode(
            prompt, add_special_tokens=add_special_tokens
        )
        common_len = self._find_common_prefix_len(full_token_ids)
        return len(full_token_ids), len(full_token_ids) - common_len

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        stop: list[str] | None = None,
    ) -> Iterator[StreamingOutput]:
        """
        Stream text generation token by token with KV cache reuse.

        Maintains a persistent prompt cache across calls. When consecutive
        requests share a common prefix (e.g. same system prompt + tools),
        only the new suffix tokens are processed, dramatically reducing
        prefill time.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Penalty for repeating tokens
            stop: List of stop sequences

        Yields:
            StreamingOutput for each generated token
        """
        if not self._loaded:
            self.load()

        import time as _time

        from mlx_lm import stream_generate

        t0 = _time.perf_counter()

        # Tokenize the full prompt
        add_special_tokens = (
            self.tokenizer.bos_token is None
            or not prompt.startswith(self.tokenizer.bos_token)
        )
        full_token_ids = self.tokenizer.encode(
            prompt, add_special_tokens=add_special_tokens
        )

        t_tokenize = _time.perf_counter()

        # Prepare cache and get only the tokens that need processing.
        # Some models (e.g. Qwen3.5-122B-A10B with vision tower weights)
        # can hit broadcast_shapes errors on cache reuse.  Fall back to a
        # fresh cache so the request still succeeds.
        try:
            suffix_tokens = self._prepare_cache_for_prompt(full_token_ids)
        except Exception as cache_err:
            logger.warning(
                "Prompt cache error (%s), resetting cache", cache_err
            )
            self._prompt_cache = None
            self._cached_token_ids = []
            suffix_tokens = self._prepare_cache_for_prompt(full_token_ids)
        prefix_len = len(full_token_ids) - len(suffix_tokens)

        if prefix_len > 0 and len(suffix_tokens) < len(full_token_ids):
            logger.info(
                f"Prompt cache hit: {prefix_len} cached / "
                f"{len(suffix_tokens)} new tokens "
                f"(saved {prefix_len} tokens of prefill)"
            )
        else:
            logger.info(
                f"Prompt cache miss: {len(full_token_ids)} tokens to prefill"
            )

        # Create sampler with parameters
        sampler = self._create_sampler(temperature, top_p)

        token_count = 0
        accumulated_text = ""
        # Track generated token IDs for raw decoding (preserving special tokens).
        # mlx_lm's detokenizer strips special tokens, which breaks parsers
        # (like Harmony) that need control tokens like <|channel|>, <|call|>.
        _generated_ids: list[int] = []
        _prev_raw_text = ""

        # Build generation kwargs
        gen_kwargs = {
            "max_tokens": max_tokens,
            "sampler": sampler,
            "prompt_cache": self._prompt_cache,
            "prefill_step_size": self.prefill_step_size,
        }

        # KV cache quantization reduces memory pressure for long prompts
        if self.kv_bits is not None:
            gen_kwargs["kv_bits"] = self.kv_bits
            gen_kwargs["kv_group_size"] = self.kv_group_size

        # Add draft model for speculative decoding if available
        if self.draft_model is not None:
            gen_kwargs["draft_model"] = self.draft_model
            gen_kwargs["num_draft_tokens"] = self.num_draft_tokens

        # Pass token IDs (not string) so mlx-lm skips re-tokenization.
        # If suffix is empty (exact same prompt), we still need at least 1 token
        # for generate_step. Pop the last token from cache and re-process it.
        if not suffix_tokens:
            if self._prompt_cache and full_token_ids:
                for c in self._prompt_cache:
                    c.trim(1)
                prompt_to_send = full_token_ids[-1:]
            else:
                prompt_to_send = full_token_ids
        else:
            prompt_to_send = suffix_tokens

        t_first_token = None
        cache_saved = False

        def _make_generator():
            return stream_generate(
                self.model,
                self.tokenizer,
                prompt=prompt_to_send,
                **gen_kwargs,
            )

        try:
            gen = _make_generator()
            # Attempt first iteration eagerly so cache errors surface here
            try:
                first_response = next(gen)
            except Exception as gen_err:
                if self._prompt_cache is not None and prefix_len > 0:
                    logger.warning(
                        "Generation failed with cached prompt (%s), "
                        "retrying with fresh cache", gen_err,
                    )
                    # Reset cache and retry with full prompt
                    self._prompt_cache = None
                    self._cached_token_ids = []
                    suffix_tokens = self._prepare_cache_for_prompt(full_token_ids)
                    prompt_to_send = suffix_tokens or full_token_ids
                    gen_kwargs["prompt_cache"] = self._prompt_cache
                    gen = _make_generator()
                    first_response = next(gen)
                else:
                    raise

            for response in itertools.chain([first_response], gen):
                token_count += 1
                if token_count == 1:
                    t_first_token = _time.perf_counter()
                    logger.info(
                        f"TTFT breakdown: tokenize={t_tokenize - t0:.3f}s, "
                        f"prefill+decode={t_first_token - t_tokenize:.3f}s, "
                        f"total={t_first_token - t0:.3f}s "
                        f"(prompt={len(full_token_ids)} tokens, "
                        f"prefilled={len(prompt_to_send)} tokens)"
                    )
                # Decode with skip_special_tokens=False to preserve control
                # tokens (e.g. Harmony's <|channel|>, <|call|>) that tool
                # parsers need. mlx_lm's detokenizer strips these.
                token_id = response.token if hasattr(response, "token") else 0
                _generated_ids.append(token_id)
                _raw_text = self.tokenizer.decode(
                    _generated_ids, skip_special_tokens=False
                )
                new_text = _raw_text[len(_prev_raw_text):]

                # Guard against multi-byte character boundaries: if the
                # tokenizer produced a replacement character (U+FFFD) for
                # an incomplete multi-byte sequence (e.g. emoji split
                # across tokens), hold the delta and re-decode next
                # iteration with more tokens to complete the character.
                if "\ufffd" in new_text:
                    # Don't update _prev_raw_text so next decode
                    # recomputes the delta from the last good position.
                    new_text = ""
                else:
                    _prev_raw_text = _raw_text
                accumulated_text += new_text

                # Check for stop sequences
                should_stop = False
                if stop:
                    for stop_seq in stop:
                        if stop_seq in accumulated_text:
                            should_stop = True
                            break

                # Check if mlx-lm signalled completion (EOS token hit)
                mlx_finished = getattr(response, "finish_reason", None) is not None

                finished = should_stop or token_count >= max_tokens or mlx_finished
                finish_reason = None
                if finished:
                    if should_stop:
                        finish_reason = "stop"
                    elif mlx_finished:
                        finish_reason = getattr(response, "finish_reason", "stop")
                    else:
                        finish_reason = "length"
                    # Save cache BEFORE yielding the finished chunk.
                    # The caller may break/abandon this generator after
                    # receiving the finished chunk, so code after yield
                    # would never execute.
                    self._save_cache_snapshot(full_token_ids)
                    cache_saved = True

                yield StreamingOutput(
                    text=new_text,
                    token=response.token if hasattr(response, "token") else 0,
                    finished=finished,
                    finish_reason=finish_reason,
                    logprobs=getattr(response, "logprobs", None),
                    prompt_tokens=len(full_token_ids),
                )

                if finished:
                    break
        finally:
            # Save cache on any exit (including GeneratorExit from client
            # disconnect) so the next request can reuse the prompt prefix.
            if not cache_saved:
                self._save_cache_snapshot(full_token_ids)

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list | None = None,
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

            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    **template_kwargs,
                )
            except TypeError:
                # Tokenizer doesn't support tools parameter
                del template_kwargs["tools"]
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
