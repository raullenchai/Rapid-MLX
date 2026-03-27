# SPDX-License-Identifier: Apache-2.0
"""
MLX Language Model wrapper.

This module provides a wrapper around mlx-lm for LLM inference,
integrating with vLLM's model execution system.
"""

import itertools
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from ..utils.decode import IncrementalDecoder

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

        # DeltaNet/hybrid cache snapshot for prefix reuse
        self._rnn_state_snapshot: list | None = None  # deep-copied ArraysCache states
        self._snapshot_prefix_ids: list[int] = []     # token IDs at snapshot time
        self._main_cache_len: int = 0  # number of main model cache layers (excl. draft)

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
                "mlx-lm is required for LLM inference. Install with: pip install mlx-lm"
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
        think_budget: int = 0,
        **kwargs,
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
            think_budget: Max tokens inside <think>...</think>; 0 = no limit
            **kwargs: Additional parameters (ignored)

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
            think_budget=think_budget,
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

    def _is_hybrid_cache(self) -> bool:
        """Check if cache has a mix of trimmable + non-trimmable layers (e.g. Qwen3.5)."""
        if self._prompt_cache is None:
            return False
        has_trimmable = any(c.is_trimmable() for c in self._prompt_cache)
        has_non_trimmable = any(not c.is_trimmable() for c in self._prompt_cache)
        return has_trimmable and has_non_trimmable

    def _snapshot_rnn_layers(self, prefix_ids: list[int]) -> None:
        """Deep-copy non-trimmable (ArraysCache) layer states for prefix reuse."""
        import copy

        if self._prompt_cache is None:
            return
        snapshot = []
        for c in self._prompt_cache:
            if not c.is_trimmable():
                snapshot.append(copy.deepcopy(c))
            else:
                snapshot.append(None)  # placeholder — KVCache is trimmed, not snapshotted
        self._rnn_state_snapshot = snapshot
        self._snapshot_prefix_ids = list(prefix_ids)
        logger.info(f"Saved RNN state snapshot for {len(prefix_ids)} token prefix")

    def _prefill_and_snapshot(self, prompt_token_ids: list[int], prefix_len: int) -> list[int]:
        """Create fresh cache, prefill the common prefix, snapshot, return suffix.

        For hybrid caches (Qwen3.5), this enables prefix reuse across requests
        that share the same system prompt but have different user messages.
        The RNN state after processing the prefix is snapshotted so future
        requests can skip re-processing it.

        Args:
            prompt_token_ids: Full prompt token IDs for this request.
            prefix_len: Number of leading tokens that match the previous request.

        Returns:
            Token IDs that still need to be processed (the suffix).
        """
        import mlx.core as mx

        self._prompt_cache = self._make_fresh_cache()
        self._cached_token_ids = []

        if prefix_len <= 0:
            return prompt_token_ids

        # Prefill both main and draft model caches
        self._prefill_cache(mx.array(prompt_token_ids[:prefix_len]))

        # Snapshot the RNN state at the prefix boundary
        self._cached_token_ids = list(prompt_token_ids[:prefix_len])
        self._snapshot_rnn_layers(prompt_token_ids[:prefix_len])

        # Return the suffix
        return prompt_token_ids[prefix_len:]

    def _restore_rnn_layers(self, prompt_token_ids: list[int], common_len: int) -> bool:
        """Restore non-trimmable layers from snapshot and trim trimmable layers.

        The RNN state is restored to snap_len tokens. If common_len > snap_len,
        we must re-run tokens [snap_len:common_len] through the model to bring
        both RNN and KV state into sync at common_len.

        Returns True if restore succeeded.
        """
        import copy

        import mlx.core as mx

        if self._rnn_state_snapshot is None:
            return False
        snap_len = len(self._snapshot_prefix_ids)
        # Snapshot only usable if full snapshot prefix is matched
        if common_len < snap_len:
            return False
        # Verify token match
        if prompt_token_ids[:snap_len] != self._snapshot_prefix_ids:
            return False
        # Restore non-trimmable layers from snapshot
        for i, snap in enumerate(self._rnn_state_snapshot):
            if snap is not None:
                self._prompt_cache[i] = copy.deepcopy(snap)
        # Trim trimmable layers to snap_len (must match RNN state position)
        for c in self._prompt_cache:
            if c.is_trimmable():
                current = c.offset if hasattr(c, "offset") else c.size()
                to_trim = current - snap_len
                if to_trim > 0:
                    c.trim(to_trim)

        # If there are gap tokens between snap_len and common_len,
        # run them through both main and draft models to advance state
        if common_len > snap_len:
            self._prefill_cache(mx.array(prompt_token_ids[snap_len:common_len]))
            # Update snapshot to common_len — better checkpoint for next time
            self._snapshot_rnn_layers(prompt_token_ids[:common_len])

        self._cached_token_ids = list(prompt_token_ids[:common_len])
        logger.info(
            f"Restored RNN snapshot ({snap_len} tok), "
            f"re-ran {common_len - snap_len} gap tokens, "
            f"cache at {common_len}"
        )
        return True

    def _prefill_cache(self, token_ids_array) -> None:
        """Run tokens through both main and draft models to advance cache.

        When speculative decoding is enabled, the prompt cache contains
        layers for both the main model and the draft model.  We must
        prefill both halves so they stay in sync.
        """
        import mlx.core as mx

        step = self.prefill_step_size
        n = len(token_ids_array)
        main_len = getattr(self, "_main_cache_len", len(self._prompt_cache))

        for start in range(0, n, step):
            chunk = token_ids_array[start : start + step]
            self.model(chunk[None], cache=self._prompt_cache[:main_len])
            if self.draft_model is not None and main_len < len(self._prompt_cache):
                self.draft_model(chunk[None], cache=self._prompt_cache[main_len:])

        mx.eval([c.state for c in self._prompt_cache])

    def _make_fresh_cache(self) -> list:
        """Create a fresh prompt cache from the model (and draft model)."""
        from mlx_lm.models.cache import make_prompt_cache

        cache = make_prompt_cache(self.model)
        self._main_cache_len = len(cache)
        if self.draft_model is not None:
            cache.extend(make_prompt_cache(self.draft_model))
        return cache

    def _cache_is_trimmable(self) -> bool:
        """Check if all layers in the prompt cache support trim."""
        from mlx_lm.models.cache import can_trim_prompt_cache

        return can_trim_prompt_cache(self._prompt_cache)

    def _prepare_cache_for_prompt(self, prompt_token_ids: list[int]) -> list[int]:
        """
        Prepare the prompt cache and return only the tokens that need processing.

        If the new prompt shares a prefix with the cached tokens, trim the cache
        to the common prefix and return only the suffix tokens.

        The cache may contain more entries than _cached_token_ids because
        generated tokens from the previous call are also in the cache.
        We must trim based on actual cache offset, not just tracked token count.

        For non-trimmable caches (ArraysCache, CacheList with non-trimmable
        sub-caches), the cache is recreated from scratch since partial trimming
        is not possible.

        Returns:
            Token IDs that still need to be processed (the non-cached suffix).
        """
        if self._prompt_cache is None:
            self._prompt_cache = self._make_fresh_cache()
            self._cached_token_ids = []
            return prompt_token_ids

        common_len = self._find_common_prefix_len(prompt_token_ids)

        if not self._cache_is_trimmable():
            if self._is_hybrid_cache() and common_len > 0:
                # Hybrid cache (e.g. Qwen3.5): mix of trimmable KVCache +
                # non-trimmable ArraysCache.
                #
                # For exact-repeat (common_len == len(prompt)), use
                # common_len - 1 so the last token becomes a suffix.
                # The generic trim(1) exact-repeat path only rolls back
                # trimmable KV layers — non-trimmable RNN layers cannot
                # be trimmed, so the last prompt token would be processed
                # twice in recurrent layers.  Keeping 1 suffix token
                # ensures both RNN and KV process it exactly once.
                effective_len = common_len
                if common_len == len(prompt_token_ids):
                    effective_len = common_len - 1

                # Try to restore from snapshot
                if self._restore_rnn_layers(prompt_token_ids, effective_len):
                    return prompt_token_ids[effective_len:]
                # No usable snapshot — do a prefix-only prefill to build
                # the snapshot for next time.
                return self._prefill_and_snapshot(prompt_token_ids, effective_len)
            # Pure non-trimmable or no overlap — recreate
            self._prompt_cache = self._make_fresh_cache()
            self._cached_token_ids = []
            return prompt_token_ids

        if common_len == 0:
            # No overlap — reset every trimmable layer to offset 0
            for c in self._prompt_cache:
                current = c.offset if hasattr(c, "offset") else c.size()
                if current > 0:
                    c.trim(current)
            self._cached_token_ids = []
            return prompt_token_ids

        # Trim cache to common prefix length.
        # Cache offset = prompt_tokens + generated_tokens from last call,
        # so we must trim (cache_offset - common_len), not just
        # (cached_token_ids_len - common_len).
        # Use .offset when available (KVCache), fall back to .size()
        # for wrappers like CacheList that delegate trim to sub-caches.
        for c in self._prompt_cache:
            current = c.offset if hasattr(c, "offset") else c.size()
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

        For non-trimmable caches (DeltaNet/Mamba), the cache will be
        recreated from scratch on the next call, so new_tokens == total_tokens
        regardless of prefix overlap.

        Returns:
            (total_tokens, new_tokens) tuple
        """
        if not self._loaded:
            self.load()

        add_special_tokens = self.tokenizer.bos_token is None or not prompt.startswith(
            self.tokenizer.bos_token
        )
        full_token_ids = self.tokenizer.encode(
            prompt, add_special_tokens=add_special_tokens
        )

        # Non-trimmable caches get fully recreated — no prefix reuse
        # Exception: hybrid caches with a valid RNN snapshot can reuse prefix
        if self._prompt_cache is not None and not self._cache_is_trimmable():
            if self._is_hybrid_cache() and self._rnn_state_snapshot is not None:
                snap_len = len(self._snapshot_prefix_ids)
                common_len = self._find_common_prefix_len(full_token_ids)
                if common_len >= snap_len:
                    # For exact-repeat, _prepare_cache_for_prompt caps reuse
                    # at common_len - 1 so the last token is reprocessed.
                    effective = common_len
                    if common_len == len(full_token_ids):
                        effective = common_len - 1
                    return len(full_token_ids), len(full_token_ids) - effective
            return len(full_token_ids), len(full_token_ids)

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
        think_budget: int = 0,
        **kwargs,
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
            think_budget: Max tokens inside <think>...</think>; 0 = no limit
            **kwargs: Additional parameters (ignored)

        Yields:
            StreamingOutput for each generated token
        """
        if not self._loaded:
            self.load()

        import time as _time

        from mlx_lm import stream_generate

        t0 = _time.perf_counter()

        # Tokenize the full prompt
        add_special_tokens = self.tokenizer.bos_token is None or not prompt.startswith(
            self.tokenizer.bos_token
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
            logger.warning("Prompt cache error (%s), resetting cache", cache_err)
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
            logger.info(f"Prompt cache miss: {len(full_token_ids)} tokens to prefill")

        # Create sampler with parameters
        sampler = self._create_sampler(temperature, top_p)

        token_count = 0
        accumulated_text = ""
        # Use IncrementalDecoder with skip_special_tokens=False to preserve
        # control tokens (e.g. Harmony's <|channel|>, <|call|>) that tool
        # parsers need. Also handles multi-byte chars (emoji, CJK) safely.
        decoder = IncrementalDecoder(self.tokenizer, skip_special_tokens=False)

        # Think budget tracking: count tokens inside <think>...</think>
        _think_budget = think_budget or 0
        _in_think = False
        _think_tokens = 0
        _think_budget_hit = False
        if _think_budget > 0:
            # Pre-compute the </think> token IDs for budget enforcement
            _end_think_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
            if not _end_think_ids:
                logger.warning("Could not encode '</think>' — think_budget disabled")
                _think_budget = 0

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
                    if c.is_trimmable():
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
                        "retrying with fresh cache",
                        gen_err,
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
                token_id = response.token if hasattr(response, "token") else 0
                new_text = decoder.add_token(token_id)
                accumulated_text += new_text

                # Think budget enforcement: track <think>/</think> boundaries
                if _think_budget > 0 and not _think_budget_hit:
                    if not _in_think and "<think>" in accumulated_text:
                        _in_think = True
                    if _in_think and "</think>" in accumulated_text:
                        _in_think = False  # thinking ended naturally
                    if _in_think:
                        _think_tokens += 1
                        if _think_tokens >= _think_budget:
                            # Budget exceeded — append </think> so reasoning
                            # parsers see a clean boundary, then stop.
                            _think_budget_hit = True
                            _in_think = False
                            close_tag = "</think>"
                            logger.info(
                                f"Think budget reached ({_think_budget} tokens), "
                                f"stopping generation"
                            )
                            yield StreamingOutput(
                                text=new_text + close_tag,
                                token=token_id,
                                finished=True,
                                finish_reason="think_budget",
                            )
                            return

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
