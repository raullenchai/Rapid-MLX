# SPDX-License-Identifier: Apache-2.0
"""
Simple engine for maximum single-user throughput.

This engine wraps mlx-lm directly with zero overhead for optimal
performance when serving a single user at a time.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from ..api.tool_calling import convert_tools_for_template
from ..api.utils import is_mllm_model
from ..utils.chat_template import apply_chat_template as shared_apply_chat_template
from .base import BaseEngine, GenerationOutput

logger = logging.getLogger(__name__)

# Check for guided generation availability
try:
    from ..api.guided import GuidedGenerator, is_guided_available

    HAS_GUIDED = is_guided_available()
except ImportError:
    HAS_GUIDED = False
    GuidedGenerator = None


_MEDIA_TYPES = frozenset(
    {
        "image_url",
        "video_url",
        "audio_url",
        "image",
        "video",
        "audio",
    }
)


def _has_media_content(messages: list) -> bool:
    """Check if any message contains media content (images, video, audio)."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in _MEDIA_TYPES:
                    return True
    return False


class SimpleEngine(BaseEngine):
    """
    Simple engine for direct model calls.

    This engine provides maximum throughput for single-user scenarios
    by calling mlx-lm/mlx-vlm directly without batching overhead.
    """

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = True,
        enable_cache: bool = True,
        force_mllm: bool = False,
        draft_model: str | None = None,
        num_draft_tokens: int = 4,
        mtp: bool = False,
        prefill_step_size: int = 2048,
        kv_bits: int | None = None,
        kv_group_size: int = 64,
        specprefill_enabled: bool = False,
        specprefill_threshold: int = 8192,
        specprefill_keep_pct: float = 0.3,
        specprefill_draft_model: str | None = None,
    ):
        """
        Initialize the simple engine.

        Args:
            model_name: HuggingFace model name or local path
            trust_remote_code: Whether to trust remote code
            enable_cache: Enable VLM cache for multimodal models
            force_mllm: Force loading as MLLM even if not auto-detected
            draft_model: Optional draft model path for speculative decoding
            num_draft_tokens: Number of tokens to generate speculatively per step
            mtp: Enable native MTP speculative decoding (model must have MTP head)
            prefill_step_size: Tokens to process per prefill chunk (default: 2048)
            kv_bits: KV cache quantization bits (None=no quantization, 4 or 8)
            kv_group_size: Group size for KV cache quantization (default: 64)
            specprefill_enabled: Enable SpecPrefill (attention-based sparse prefill)
            specprefill_threshold: Minimum suffix tokens to trigger SpecPrefill
            specprefill_keep_pct: Fraction of tokens to keep (default: 0.3)
            specprefill_draft_model: Path to small draft model for importance scoring
        """
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._enable_cache = enable_cache
        self._is_mllm = force_mllm or is_mllm_model(model_name)
        self._draft_model_name = draft_model
        self._num_draft_tokens = num_draft_tokens
        self._mtp = mtp
        self._prefill_step_size = prefill_step_size
        self._kv_bits = kv_bits
        self._kv_group_size = kv_group_size

        # SpecPrefill config
        self._specprefill_enabled = specprefill_enabled
        self._specprefill_threshold = specprefill_threshold
        self._specprefill_keep_pct = specprefill_keep_pct
        self._specprefill_draft_model_path = specprefill_draft_model

        self._model = None
        self._loaded = False

        # Per-request routing state (MLLM+MTP mode)
        self._text_model = None
        self._text_tokenizer = None

        # SpecPrefill draft model (loaded at start if enabled)
        self._draft_model = None

        # Lock to serialize MLX operations (prevents Metal command buffer conflicts)
        self._generation_lock = asyncio.Lock()

        # System prompt KV cache (reduces repeated prefill across requests)
        self._system_kv_snapshot = None  # List of (keys, values) per backbone layer
        self._system_kv_hash = None  # Hash of system prefix text
        self._system_kv_token_count = 0  # Tokens in cached prefix

    @property
    def model(self):
        """Get the underlying MLXLanguageModel instance."""
        return self._model

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def is_mllm(self) -> bool:
        """Check if this is a multimodal model."""
        return self._is_mllm

    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer."""
        if not self._loaded or self._model is None:
            return None
        if self._is_mllm:
            return getattr(self._model, "processor", None)
        return self._model.tokenizer

    def generate_warmup(self) -> None:
        """Run a minimal generation to compile Metal shaders."""
        if not self._loaded or self._model is None or self._is_mllm:
            return
        try:
            import mlx.core as mx

            model = self._model
            tokenizer = model.tokenizer
            # Encode a short prompt and generate 1 token
            tokens = tokenizer.encode("Hi")
            input_ids = mx.array([tokens])
            # Run one forward pass to trigger shader compilation
            model.model(input_ids)
            mx.eval(mx.zeros(1))
        except Exception:
            pass  # Non-fatal

    async def start(self) -> None:
        """Start the engine (load model if not loaded)."""
        if self._loaded:
            return

        if self._is_mllm:
            from ..models.mllm import MLXMultimodalLM

            self._model = MLXMultimodalLM(
                self._model_name,
                trust_remote_code=self._trust_remote_code,
                enable_cache=self._enable_cache,
            )
            if self._draft_model_name:
                logger.warning("Speculative decoding is not supported with MLLM models")
        else:
            from ..models.llm import MLXLanguageModel

            self._model = MLXLanguageModel(
                self._model_name,
                trust_remote_code=self._trust_remote_code,
                draft_model=self._draft_model_name,
                num_draft_tokens=self._num_draft_tokens,
                mtp=self._mtp,
                prefill_step_size=self._prefill_step_size,
                kv_bits=self._kv_bits,
                kv_group_size=self._kv_group_size,
            )

        self._model.load()
        self._loaded = True

        # Build parallel mlx_lm TextModel for text-only MTP routing
        if self._is_mllm and self._mtp:
            try:
                from ..text_model_from_vlm import build_text_model

                self._text_model = build_text_model(self._model.model, self._model_name)

                if (
                    self._text_model is not None
                    and hasattr(self._text_model, "mtp")
                    and self._text_model.mtp is not None
                ):
                    self._text_tokenizer = self._model.get_tokenizer()

                    # Apply Qwen3.5 eos_token fix (matches MLXLanguageModel.load)
                    if "qwen3" in self._model_name.lower():
                        self._text_tokenizer.eos_token = "<|im_end|>"
                        self._text_tokenizer.eos_token_id = (
                            self._text_tokenizer.convert_tokens_to_ids("<|im_end|>")
                        )

                    logger.info(
                        "MLLM+MTP routing: text-only → mlx_lm TextModel (MTP=True), "
                        "media → mlx_vlm"
                    )
                else:
                    logger.warning(
                        "TextModel built but no MTP — text-only requests won't use MTP"
                    )
                    self._text_model = None

            except Exception as e:
                logger.error("MLLM+MTP routing setup failed: %s", e)
                self._text_model = None
                self._text_tokenizer = None

        # Load SpecPrefill draft model (small model for importance scoring)
        if self._specprefill_enabled and self._specprefill_draft_model_path:
            try:
                from mlx_lm import load as mlx_lm_load

                self._draft_model, _ = mlx_lm_load(self._specprefill_draft_model_path)
                logger.info(
                    "SpecPrefill: draft model loaded (%s), threshold=%d, keep=%.0f%%",
                    self._specprefill_draft_model_path,
                    self._specprefill_threshold,
                    self._specprefill_keep_pct * 100,
                )
            except Exception as e:
                logger.error("SpecPrefill: draft model load failed: %s", e)
                self._draft_model = None

        spec_info = ""
        if self._draft_model_name and not self._is_mllm:
            spec_info = f", speculative={self._draft_model_name}"
        mtp_info = f", MTP={self._mtp}" if self._mtp else ""
        routing = ", routing=per-request" if self._text_model is not None else ""
        specprefill_info = (
            ", SpecPrefill=active" if self._draft_model is not None else ""
        )
        logger.info(
            f"SimpleEngine loaded: {self._model_name} "
            f"(MLLM={self._is_mllm}{spec_info}{mtp_info}{routing}{specprefill_info})"
        )

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        self._model = None
        self._text_model = None
        self._text_tokenizer = None
        self._draft_model = None
        self._loaded = False
        self._system_kv_snapshot = None
        self._system_kv_hash = None
        self._system_kv_token_count = 0
        logger.info("SimpleEngine stopped")

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate a complete response (non-streaming).

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            **kwargs: Additional model-specific parameters

        Returns:
            GenerationOutput with complete text
        """
        if not self._loaded:
            await self.start()

        async with self._generation_lock:
            # Run in thread pool to allow asyncio timeout to work
            output = await asyncio.to_thread(
                self._model.generate,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **kwargs,
            )

            # Return raw text — the server pipeline handles cleaning
            # AFTER tool parsing and reasoning extraction.
            return GenerationOutput(
                text=output.text,
                tokens=getattr(output, "tokens", []),
                prompt_tokens=getattr(output, "prompt_tokens", 0),
                completion_tokens=getattr(
                    output, "completion_tokens", len(getattr(output, "tokens", []))
                ),
                finish_reason=output.finish_reason,
            )

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """
        Stream generation token by token.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        if not self._loaded:
            await self.start()

        # Per-request specprefill overrides (from extra_body)
        specprefill_override = kwargs.pop("specprefill", None)
        specprefill_keep_pct_override = kwargs.pop("specprefill_keep_pct", None)

        # SpecPrefill for non-MLLM models (MLLM+MTP handles it in _stream_generate_text)
        if not self._is_mllm and self._draft_model is not None:
            use_specprefill = True
            if specprefill_override is False:
                use_specprefill = False

            if use_specprefill:
                tokenizer = self._model.tokenizer
                add_special = tokenizer.bos_token is None or not prompt.startswith(
                    tokenizer.bos_token
                )
                tokens_list = tokenizer.encode(prompt, add_special_tokens=add_special)
                n_tokens = len(tokens_list)

                # Threshold check (skip when force-enabled via per-request override)
                if (
                    specprefill_override is not True
                    and n_tokens <= self._specprefill_threshold
                ):
                    use_specprefill = False

                # Upper bound: cap to avoid draft model OOM
                _SPECPREFILL_MAX_TOKENS = 65536
                if use_specprefill and n_tokens > _SPECPREFILL_MAX_TOKENS:
                    logger.warning(
                        "SpecPrefill: prompt %d tokens exceeds max %d, "
                        "falling back to normal path",
                        n_tokens,
                        _SPECPREFILL_MAX_TOKENS,
                    )
                    use_specprefill = False

                if use_specprefill:
                    async for output in self._stream_generate_specprefill(
                        prompt,
                        tokens_list,
                        max_tokens,
                        temperature,
                        top_p,
                        stop=stop,
                        specprefill_keep_pct=specprefill_keep_pct_override,
                        **kwargs,
                    ):
                        yield output
                    return

        async with self._generation_lock:
            accumulated_text = ""
            prompt_tokens = 0
            completion_tokens = 0
            finished = False
            # Cache attribute checks after first chunk
            _has_completion_tokens = None
            _has_text = None

            for chunk in self._model.stream_generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **kwargs,
            ):
                prompt_tokens = getattr(chunk, "prompt_tokens", 0) or prompt_tokens
                # Cache hasattr checks on first iteration
                if _has_completion_tokens is None:
                    _has_completion_tokens = hasattr(chunk, "completion_tokens")
                    _has_text = hasattr(chunk, "text")
                if _has_completion_tokens:
                    completion_tokens = chunk.completion_tokens
                else:
                    completion_tokens += 1
                new_text = chunk.text if _has_text else str(chunk)
                accumulated_text += new_text

                finished = (
                    getattr(chunk, "finished", False) or completion_tokens >= max_tokens
                )
                finish_reason = None
                if finished:
                    finish_reason = getattr(chunk, "finish_reason", "stop")

                # Pass current token ID for logprobs extraction
                current_token = getattr(chunk, "token", None)
                tokens_list = (
                    [current_token]
                    if current_token is not None and current_token != 0
                    else []
                )

                yield GenerationOutput(
                    text=accumulated_text,
                    new_text=new_text,
                    tokens=tokens_list,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    finished=finished,
                    finish_reason=finish_reason,
                    logprobs=getattr(chunk, "logprobs", None),
                    channel=getattr(chunk, "channel", None),
                )

                # Yield to event loop periodically so the server can
                # accept connections, detect disconnects, and process
                # queued requests. Without this, the sync generation
                # loop starves the event loop during decode.
                # Every 8 tokens balances responsiveness vs throughput.
                if completion_tokens % 8 == 0:
                    await asyncio.sleep(0)

                if finished:
                    break

            if not finished:
                yield GenerationOutput(
                    text=accumulated_text,
                    new_text="",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    finished=True,
                    finish_reason=None,
                )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        tools: list[dict] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Chat completion (non-streaming).

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            tools: Optional tool definitions
            images: Optional image URLs/paths
            videos: Optional video URLs/paths
            **kwargs: Additional model-specific parameters

        Returns:
            GenerationOutput with assistant response
        """
        if not self._loaded:
            await self.start()

        # Convert tools for template if provided
        template_tools = convert_tools_for_template(tools) if tools else None

        # Text-only MTP routing — BEFORE the lock because
        # _stream_generate_text() acquires _generation_lock internally.
        if (
            self._is_mllm
            and self._text_model is not None
            and not _has_media_content(messages)
        ):
            logger.info("Text-only request → LLM path (MTP=True) [non-streaming]")
            last_chunk = None
            async for chunk in self._stream_generate_text(
                messages,
                max_tokens,
                temperature,
                top_p,
                stop=stop,
                tools=template_tools,
                **kwargs,
            ):
                last_chunk = chunk
            if last_chunk is not None:
                # _stream_generate_text yields accumulated text, not deltas
                return GenerationOutput(
                    text=last_chunk.text,
                    tokens=[],
                    prompt_tokens=last_chunk.prompt_tokens,
                    completion_tokens=last_chunk.completion_tokens,
                    finish_reason=last_chunk.finish_reason or "stop",
                )
            return GenerationOutput(
                text="", tokens=[], prompt_tokens=0,
                completion_tokens=0, finish_reason="stop",
            )

        async with self._generation_lock:
            if self._is_mllm:
                # For MLLM with media, use the chat method which handles images/videos
                # Run in thread pool to allow asyncio timeout to work
                try:
                    output = await asyncio.to_thread(
                        self._model.chat,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop=stop,
                        tools=template_tools,
                        **kwargs,
                    )
                except Exception as e:
                    logger.error("MLLM chat() failed: %s", e, exc_info=True)
                    raise
                return GenerationOutput(
                    text=output.text,
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    finish_reason=output.finish_reason,
                )
            else:
                # For LLM, build prompt with enable_thinking support,
                # then generate directly.
                enable_thinking_val = kwargs.get("enable_thinking")
                kwargs_copy = kwargs.copy()
                kwargs_copy.pop("enable_thinking", None)
                prompt = self.build_prompt(
                    messages,
                    tools=tools,
                    **(
                        {"enable_thinking": enable_thinking_val}
                        if enable_thinking_val is not None
                        else {}
                    ),
                )
                # Run in thread pool to allow asyncio timeout to work
                output = await asyncio.to_thread(
                    self._model.generate,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    **kwargs_copy,
                )
                # Return raw text — server handles cleaning after
                # tool parsing and reasoning extraction.
                text = output.text

                # Compute prompt tokens from the chat template
                prompt_tokens = getattr(output, "prompt_tokens", 0)
                if not prompt_tokens:
                    tokenizer = self._model.tokenizer
                    if hasattr(tokenizer, "apply_chat_template"):
                        try:
                            prompt_ids = tokenizer.apply_chat_template(
                                messages,
                                tokenize=True,
                                add_generation_prompt=True,
                                tools=template_tools,
                            )
                            prompt_tokens = len(prompt_ids)
                        except TypeError:
                            try:
                                prompt_ids = tokenizer.apply_chat_template(
                                    messages,
                                    tokenize=True,
                                    add_generation_prompt=True,
                                )
                                prompt_tokens = len(prompt_ids)
                            except Exception as e:
                                logger.warning(f"Failed to compute prompt_tokens: {e}")
                                prompt_tokens = 0

                return GenerationOutput(
                    text=text,
                    tokens=output.tokens,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=len(output.tokens),
                    finish_reason=output.finish_reason,
                )

    def build_prompt(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        **kwargs,
    ) -> str:
        """
        Apply chat template to messages and return the prompt string.

        This is the same logic used by stream_chat, extracted for reuse
        (e.g. cloud routing needs the prompt to estimate token counts).
        """
        if not self._loaded:
            raise RuntimeError("Engine not loaded — call start() first")

        if self._is_mllm:
            raise RuntimeError("build_prompt is not supported for MLLM models")

        template_tools = convert_tools_for_template(tools) if tools else None
        enable_thinking = kwargs.get("enable_thinking")

        return shared_apply_chat_template(
            self._model.tokenizer,
            messages,
            tools=template_tools,
            enable_thinking=enable_thinking,
            model_name=self._model_name,
        )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        tools: list[dict] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """
        Stream chat completion token by token.

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            tools: Optional tool definitions
            images: Optional image URLs/paths
            videos: Optional video URLs/paths
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        if not self._loaded:
            await self.start()

        # Convert tools for template
        template_tools = convert_tools_for_template(tools) if tools else None

        # Per-request routing: text-only through mlx_lm with MTP
        if (
            self._is_mllm
            and self._text_model is not None
            and not _has_media_content(messages)
        ):
            logger.info("Text-only request → LLM path (MTP=True)")
            async for chunk in self._stream_generate_text(
                messages,
                max_tokens,
                temperature,
                top_p,
                stop=stop,
                tools=template_tools,
                **kwargs,
            ):
                yield chunk
            return

        # Build prompt using tokenizer
        if self._is_mllm:
            if self._text_model is not None:
                logger.info("Media request → MLLM path")
            # For MLLM, use stream_chat which yields tokens incrementally.
            # Must hold the generation lock to prevent concurrent Metal
            # command buffer conflicts with other generation methods.
            async with self._generation_lock:
                accumulated_text = ""
                token_count = 0

                # Run the synchronous generator in a thread
                # Pop enable_thinking — MLLM models don't support it
                kwargs.pop("enable_thinking", None)
                sync_gen = self._model.stream_chat(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    tools=template_tools,
                    **kwargs,
                )

                while True:
                    try:
                        chunk = await asyncio.to_thread(next, sync_gen)
                    except StopIteration:
                        break
                    except Exception as e:
                        # Some VLM models (e.g. Gemma 4) raise during
                        # generator cleanup after generation completes.
                        # If we already have output, treat as finished.
                        if token_count > 0:
                            logger.warning(
                                "MLLM stream_chat error after %d tokens "
                                "(likely post-generation cleanup): %s",
                                token_count, e,
                            )
                            break
                        raise

                    token_count += 1
                    new_text = chunk.text if hasattr(chunk, "text") else str(chunk)
                    accumulated_text += new_text

                    finished = chunk.finish_reason is not None

                    yield GenerationOutput(
                        text=accumulated_text,
                        new_text=new_text,
                        prompt_tokens=getattr(chunk, "prompt_tokens", 0),
                        completion_tokens=token_count,
                        finished=finished,
                        finish_reason=chunk.finish_reason if finished else None,
                    )

                    if finished:
                        break
            return

        # For LLM, apply chat template and stream
        enable_thinking = kwargs.pop("enable_thinking", None)
        prompt = shared_apply_chat_template(
            self._model.tokenizer,
            messages,
            tools=template_tools,
            enable_thinking=enable_thinking,
            model_name=self._model_name,
        )

        # Stream generate
        async for output in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            **kwargs,
        ):
            yield output

    async def _stream_generate_specprefill(
        self,
        prompt: str,
        tokens: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None = None,
        specprefill_keep_pct: float | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """SpecPrefill path for non-MTP models (Nemotron, GPT-OSS, etc).

        Scores token importance with the draft model, sparse-prefills the target
        model, then generates autoregressively. Falls back to normal generation
        on any error.
        """
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        model = self._model.model
        tokenizer = self._model.tokenizer
        n_tokens = len(tokens)

        async with self._generation_lock:

            def _run_all():
                try:
                    return _run_specprefill()
                except Exception as e:
                    logger.error(
                        "SpecPrefill failed, falling back to normal path: %s", e
                    )
                    return _run_normal()

            def _run_specprefill():
                """Score tokens, sparse prefill, generate autoregressively."""
                import time
                from types import SimpleNamespace

                from ..specprefill import (
                    cleanup_rope,
                    score_tokens,
                    select_chunks,
                    sparse_prefill,
                )

                cache = make_prompt_cache(model)

                try:
                    # Phase 1: Score with draft model
                    t0 = time.monotonic()
                    importance = score_tokens(
                        self._draft_model,
                        tokens,
                        prefill_step_size=self._prefill_step_size,
                    )
                    t_score = time.monotonic() - t0

                    # Phase 2: Select important chunks
                    effective_keep = specprefill_keep_pct or self._specprefill_keep_pct
                    selected = select_chunks(importance, keep_pct=effective_keep)
                    n_selected = selected.shape[0]

                    # Phase 3: Sparse prefill on target model
                    t0 = time.monotonic()
                    logits = sparse_prefill(
                        model,
                        tokens,
                        selected,
                        cache,
                        step_size=self._prefill_step_size,
                    )
                    t_prefill = time.monotonic() - t0

                    logger.info(
                        "SpecPrefill: scored %d tokens in %.1fs, "
                        "sparse prefill %d/%d (keep=%.0f%%) in %.1fs",
                        n_tokens,
                        t_score,
                        n_selected,
                        n_tokens,
                        n_selected / n_tokens * 100,
                        t_prefill,
                    )

                    # Phase 4: Generate (simple autoregressive, no MTP)
                    sampler = make_sampler(temp=temperature, top_p=top_p)
                    eos_id = tokenizer.eos_token_id
                    y = sampler(logits[:, -1, :])
                    mx.eval(y)

                    results = []
                    generated_ids = []
                    prev_decoded = ""

                    for _ in range(max_tokens):
                        tok_id = y.item()
                        generated_ids.append(tok_id)

                        decoded = tokenizer.decode(generated_ids)
                        new_text = decoded[len(prev_decoded) :]
                        prev_decoded = decoded

                        is_eos = tok_id == eos_id
                        results.append(
                            SimpleNamespace(
                                text=new_text,
                                finish_reason="stop" if is_eos else None,
                            )
                        )

                        if is_eos:
                            break

                        logits = model(y.reshape(1, -1), cache=cache)
                        y = sampler(logits[:, -1, :])
                        mx.eval(y)

                    return results

                finally:
                    cleanup_rope(model)

            def _run_normal():
                """Fallback: normal generation without specprefill."""
                from types import SimpleNamespace

                results = []
                for chunk in self._model.stream_generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    **kwargs,
                ):
                    new_text = chunk.text if hasattr(chunk, "text") else str(chunk)
                    results.append(
                        SimpleNamespace(
                            text=new_text,
                            finish_reason=getattr(chunk, "finish_reason", None),
                        )
                    )
                return results

            all_resps = await asyncio.to_thread(_run_all)

        # Yield results as GenerationOutput
        accumulated_text = ""
        token_count = 0
        finished = False
        for i, resp in enumerate(all_resps):
            token_count += 1
            new_text = resp.text
            accumulated_text += new_text

            is_last = i == len(all_resps) - 1
            finished = is_last or token_count >= max_tokens

            yield GenerationOutput(
                text=accumulated_text,
                new_text=new_text,
                prompt_tokens=n_tokens,
                completion_tokens=token_count,
                finished=finished,
                finish_reason=resp.finish_reason or ("stop" if finished else None),
            )

            if finished:
                break

        if not finished:
            yield GenerationOutput(
                text=accumulated_text,
                new_text="",
                prompt_tokens=n_tokens,
                completion_tokens=token_count,
                finished=True,
                finish_reason="length",
            )

    async def _stream_generate_text(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None = None,
        tools: list | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """Text-only generation via mlx_lm TextModel with MTP.

        Used when MLLM+MTP routing is active and the request has no media.
        Runs the full generation in a single thread to maintain Metal safety.

        System prompt KV caching: on the first request, prefills system tokens
        and snapshots backbone KV state. Subsequent requests with the same
        system prompt restore the snapshot and only prefill the suffix tokens.
        """
        import hashlib
        import os

        import mlx.core as mx
        from mlx_lm import stream_generate as mlx_stream_generate
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        # Per-request specprefill overrides (from extra_body)
        specprefill_override = kwargs.pop("specprefill", None)
        specprefill_keep_pct = kwargs.pop("specprefill_keep_pct", None)

        # Read enable_thinking from env (set by runtime_patches, consistent with MLLM path)
        enable_thinking_env = os.environ.get("VLLM_MLX_ENABLE_THINKING", "true")
        enable_thinking = enable_thinking_env.lower() in ("true", "1", "yes")

        # Apply chat template for full prompt
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        }
        if tools:
            template_kwargs["tools"] = tools

        try:
            full_prompt = self._text_tokenizer.apply_chat_template(
                messages, **template_kwargs
            )
        except TypeError:
            # Template doesn't accept tools= or enable_thinking=
            template_kwargs.pop("tools", None)
            template_kwargs.pop("enable_thinking", None)
            full_prompt = self._text_tokenizer.apply_chat_template(
                messages, **template_kwargs
            )

        # Build sampler
        sampler = make_sampler(temp=temperature, top_p=top_p)
        max_tokens = max_tokens or 4096

        # --- System prompt KV caching ---
        backbone_cache = None  # Backbone-only cache (no MTP), used by both paths
        prompt_to_send = full_prompt  # Default: send full prompt text
        cache_hit = False
        system_token_count = 0
        full_token_count = 0
        system_hash = None
        system_tokens = None
        suffix_tokens = None
        full_tokens_list = None

        # Extract system messages for caching
        has_system = any(m.get("role") == "system" for m in messages)

        if has_system and self._text_model is not None:
            # Find system prefix boundary in full prompt text.
            # ChatML format: system section ends where first non-system message begins.
            # Works with tools (rendered inside system section by Qwen templates).
            system_prefix_end = -1
            for marker in ("<|im_start|>user\n", "<|im_start|>assistant\n"):
                idx = full_prompt.find(marker)
                if idx > 0:
                    system_prefix_end = idx
                    break

            if system_prefix_end > 0:
                system_prefix_text = full_prompt[:system_prefix_end]
                system_hash = hashlib.sha256(system_prefix_text.encode()).hexdigest()[
                    :16
                ]

                # Tokenize both (matching stream_generate's tokenization logic)
                tokenizer = self._text_tokenizer
                add_special = tokenizer.bos_token is None or not full_prompt.startswith(
                    tokenizer.bos_token
                )
                full_tokens_list = tokenizer.encode(
                    full_prompt, add_special_tokens=add_special
                )
                full_token_count = len(full_tokens_list)

                system_tokens_list = tokenizer.encode(
                    system_prefix_text, add_special_tokens=add_special
                )
                system_token_count = len(system_tokens_list)

                # Verify system tokens are a proper prefix of full tokens
                prefix_valid = (
                    len(full_tokens_list) > system_token_count
                    and full_tokens_list[:system_token_count] == system_tokens_list
                )

                if prefix_valid:
                    system_tokens = system_tokens_list
                    suffix_tokens = full_tokens_list[system_token_count:]

                    if (
                        system_hash == self._system_kv_hash
                        and self._system_kv_snapshot is not None
                        and system_token_count == self._system_kv_token_count
                    ):
                        # Cache HIT — restore KV state into fresh backbone cache
                        backbone_cache = make_prompt_cache(self._text_model)
                        for i, saved_state in enumerate(self._system_kv_snapshot):
                            backbone_cache[i].state = saved_state

                        prompt_to_send = mx.array(suffix_tokens)
                        cache_hit = True
                        logger.info(
                            "System KV cache HIT: reusing %d cached tokens, "
                            "prefilling %d new tokens (hash=%s)",
                            system_token_count,
                            len(suffix_tokens),
                            system_hash,
                        )
                    else:
                        # Cache MISS — will prefill system tokens and snapshot
                        logger.info(
                            "System KV cache MISS: will prefill %d system tokens, "
                            "%d suffix tokens (hash=%s)",
                            system_token_count,
                            len(suffix_tokens),
                            system_hash,
                        )
                else:
                    logger.debug(
                        "System KV cache: prefix token validation failed, "
                        "using full prompt (%d tokens)",
                        len(full_tokens_list),
                    )
                    system_token_count = 0

        # Determine if SpecPrefill should be used
        # Per-request boolean override: True = force enable, False = force disable
        if specprefill_override is False:
            use_specprefill = False
        elif specprefill_override is True and self._draft_model is not None:
            use_specprefill = True  # Force enable, skip threshold check
        else:
            use_specprefill = self._draft_model is not None

        # For specprefill, ensure we have token IDs (not just prompt text)
        if use_specprefill and suffix_tokens is None and full_tokens_list is None:
            tokenizer = self._text_tokenizer
            add_special = tokenizer.bos_token is None or not full_prompt.startswith(
                tokenizer.bos_token
            )
            full_tokens_list = tokenizer.encode(
                full_prompt, add_special_tokens=add_special
            )
            full_token_count = len(full_tokens_list)

        # Tokens for specprefill: suffix (if system KV) or full prompt
        specprefill_tokens = (
            suffix_tokens if suffix_tokens is not None else full_tokens_list
        )
        specprefill_offset = system_token_count if suffix_tokens is not None else 0

        # Threshold check: only use specprefill on long prompts
        # (skipped when per-request boolean forces enable)
        if (
            use_specprefill
            and specprefill_override is not True
            and (
                specprefill_tokens is None
                or len(specprefill_tokens) <= self._specprefill_threshold
            )
        ):
            use_specprefill = False

        # Upper bound: cap specprefill to avoid draft model OOM on very long prompts
        # 65536 tokens ~ 2GB draft KV cache on Qwen3.5-4B (32KB/token x 8 attn layers)
        _SPECPREFILL_MAX_TOKENS = 65536
        if (
            use_specprefill
            and specprefill_tokens is not None
            and len(specprefill_tokens) > _SPECPREFILL_MAX_TOKENS
        ):
            logger.warning(
                "SpecPrefill: prompt %d tokens exceeds max %d, "
                "falling back to normal path",
                len(specprefill_tokens),
                _SPECPREFILL_MAX_TOKENS,
            )
            use_specprefill = False

        # Run under generation lock, all Metal ops in single thread
        async with self._generation_lock:

            def _run_all():
                nonlocal backbone_cache, prompt_to_send

                model = self._text_model

                # Cache MISS with valid prefix: prefill system tokens and snapshot
                if (
                    not cache_hit
                    and system_token_count > 0
                    and system_tokens is not None
                    and suffix_tokens is not None
                ):
                    mc = make_prompt_cache(model)
                    sys_arr = mx.array(system_tokens)

                    # Prefill system tokens in chunks (matching generate_step)
                    step = self._prefill_step_size
                    while sys_arr.size > step:
                        model(sys_arr[:step][None], cache=mc)
                        mx.eval([c.state for c in mc])
                        sys_arr = sys_arr[step:]
                        mx.clear_cache()
                    if sys_arr.size > 0:
                        model(sys_arr[None], cache=mc)
                        mx.eval([c.state for c in mc])

                    # Snapshot backbone cache (immutable mx.arrays, safe to reuse)
                    snapshot = [c.state for c in mc]
                    mx.eval([s for pair in snapshot for s in pair])

                    self._system_kv_snapshot = snapshot
                    self._system_kv_hash = system_hash
                    self._system_kv_token_count = system_token_count

                    backbone_cache = mc
                    prompt_to_send = mx.array(suffix_tokens)
                    logger.info(
                        "System KV cache: stored %d-token snapshot (%.1f MB), "
                        "prefilling %d remaining",
                        system_token_count,
                        sum(c.nbytes for c in mc) / 1e6,
                        len(suffix_tokens),
                    )

                # --- SpecPrefill path (with fallback to normal on failure) ---
                if use_specprefill:
                    try:
                        return _run_specprefill(model, backbone_cache)
                    except Exception as e:
                        logger.error(
                            "SpecPrefill failed, falling back to normal MTP path: %s",
                            e,
                        )
                        # Discard potentially corrupted cache
                        backbone_cache = None
                        prompt_to_send = full_prompt

                # --- Normal path (MTP via mlx_lm stream_generate) ---
                prompt_cache = None
                if backbone_cache is not None:
                    # Add MTP cache on top of backbone
                    if hasattr(model, "make_mtp_cache"):
                        mtp_cache = model.make_mtp_cache()
                        prompt_cache = backbone_cache + mtp_cache
                    else:
                        prompt_cache = backbone_cache

                results = []
                gen_kwargs = dict(
                    max_tokens=max_tokens,
                    sampler=sampler,
                    mtp=True,
                    prefill_step_size=self._prefill_step_size,
                )
                if prompt_cache is not None:
                    gen_kwargs["prompt_cache"] = prompt_cache

                for resp in mlx_stream_generate(
                    model,
                    self._text_tokenizer,
                    prompt=prompt_to_send,
                    **gen_kwargs,
                ):
                    results.append(resp)
                return results

            def _run_specprefill(model, bc):
                """Score tokens, sparse prefill, generate without MTP."""
                from types import SimpleNamespace

                from ..specprefill import (
                    cleanup_rope,
                    score_tokens,
                    select_chunks,
                    sparse_prefill,
                )

                # Create backbone cache if not already from system KV
                if bc is None:
                    bc = make_prompt_cache(model)

                try:
                    # Phase 1: Score with draft model
                    import time

                    t0 = time.monotonic()
                    importance = score_tokens(
                        self._draft_model,
                        specprefill_tokens,
                        prefill_step_size=self._prefill_step_size,
                    )
                    t_score = time.monotonic() - t0

                    # Phase 2: Select important chunks
                    effective_keep = specprefill_keep_pct or self._specprefill_keep_pct
                    selected = select_chunks(importance, keep_pct=effective_keep)
                    n_selected = selected.shape[0]
                    n_total = len(specprefill_tokens)

                    # Phase 3: Sparse prefill on target model
                    t0 = time.monotonic()
                    logits = sparse_prefill(
                        model,
                        specprefill_tokens,
                        selected,
                        bc,
                        step_size=self._prefill_step_size,
                        position_offset=specprefill_offset,
                    )
                    t_prefill = time.monotonic() - t0

                    logger.info(
                        "SpecPrefill: scored %d tokens in %.1fs, "
                        "sparse prefill %d/%d (keep=%.0f%%) in %.1fs "
                        "(offset=%d, effective_keep=%.2f)",
                        n_total,
                        t_score,
                        n_selected,
                        n_total,
                        n_selected / n_total * 100,
                        t_prefill,
                        specprefill_offset,
                        effective_keep,
                    )

                    # Phase 4: Generate (simple autoregressive, no MTP)
                    eos_id = self._text_tokenizer.eos_token_id
                    y = sampler(logits[:, -1, :])
                    mx.eval(y)

                    results = []
                    generated_ids = []
                    prev_decoded = ""

                    for _ in range(max_tokens):
                        tok_id = y.item()
                        generated_ids.append(tok_id)

                        # Incremental text decode
                        decoded = self._text_tokenizer.decode(generated_ids)
                        new_text = decoded[len(prev_decoded) :]
                        prev_decoded = decoded

                        is_eos = tok_id == eos_id
                        results.append(
                            SimpleNamespace(
                                text=new_text,
                                finish_reason="stop" if is_eos else None,
                            )
                        )

                        if is_eos:
                            break

                        # Next token
                        logits = model(y.reshape(1, -1), cache=bc)
                        y = sampler(logits[:, -1, :])
                        mx.eval(y)

                    return results

                finally:
                    cleanup_rope(model)

            all_resps = await asyncio.to_thread(_run_all)

        # Yield results as GenerationOutput
        accumulated_text = ""
        token_count = 0
        finished = False
        for i, resp in enumerate(all_resps):
            token_count += 1
            new_text = resp.text if hasattr(resp, "text") else str(resp)
            accumulated_text += new_text

            # Check stop sequences (mlx_lm doesn't handle these natively)
            stop_hit = False
            if stop:
                for stop_seq in stop:
                    idx = accumulated_text.find(stop_seq)
                    if idx != -1:
                        # Trim both accumulated and new_text so SSE streams
                        # never emit the stop sequence or anything after it.
                        overshoot = len(accumulated_text) - idx
                        accumulated_text = accumulated_text[:idx]
                        new_text = new_text[: max(0, len(new_text) - overshoot)]
                        stop_hit = True
                        break

            is_last = i == len(all_resps) - 1
            finished = stop_hit or is_last or token_count >= max_tokens

            yield GenerationOutput(
                text=accumulated_text,
                new_text=new_text,
                prompt_tokens=full_token_count or 0,
                completion_tokens=token_count,
                finished=finished,
                finish_reason=getattr(resp, "finish_reason", None)
                or ("stop" if finished else None),
            )

            if finished:
                break

        if not finished:
            yield GenerationOutput(
                text=accumulated_text,
                new_text="",
                prompt_tokens=full_token_count or 0,
                completion_tokens=token_count,
                finished=True,
                finish_reason="length",
            )

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        stats = {
            "engine_type": "simple",
            "model_name": self._model_name,
            "is_mllm": self._is_mllm,
            "loaded": self._loaded,
        }

        # SpecPrefill stats
        if self._draft_model is not None:
            stats["specprefill"] = {
                "enabled": True,
                "draft_model": self._specprefill_draft_model_path,
                "threshold": self._specprefill_threshold,
                "keep_pct": self._specprefill_keep_pct,
            }

        # System KV cache stats
        if self._system_kv_snapshot is not None:
            cache_bytes = 0
            for entry in self._system_kv_snapshot:
                if isinstance(entry, tuple) and len(entry) == 2:
                    cache_bytes += entry[0].nbytes + entry[1].nbytes
                elif isinstance(entry, list):
                    cache_bytes += sum(a.nbytes for a in entry if a is not None)
            stats["system_kv_cache"] = {
                "tokens": self._system_kv_token_count,
                "hash": self._system_kv_hash,
                "memory_mb": round(cache_bytes / 1e6, 1),
            }

        # Include Metal memory stats
        try:
            import mlx.core as mx

            if mx.metal.is_available():
                stats["metal_active_memory_gb"] = round(mx.get_active_memory() / 1e9, 2)
                stats["metal_peak_memory_gb"] = round(mx.get_peak_memory() / 1e9, 2)
                stats["metal_cache_memory_gb"] = round(mx.get_cache_memory() / 1e9, 2)
        except Exception:
            pass

        return stats

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics (for MLLM models)."""
        if self._is_mllm and self._model is not None:
            return self._model.get_cache_stats()
        return None

    async def _inject_shared_model(
        self,
        model,
        tokenizer,
    ) -> None:
        """
        Inject a pre-loaded shared model instead of loading a new one.

        This is used by HybridEngine to share a single model instance
        between SimpleEngine and BatchedEngine, saving ~44GB of RAM.

        Args:
            model: Pre-loaded MLX model
            tokenizer: Pre-loaded tokenizer
        """
        from ..models.llm import MLXLanguageModel

        # Create MLXLanguageModel wrapper without loading
        self._model = MLXLanguageModel.__new__(MLXLanguageModel)
        self._model.model_name = self._model_name
        self._model.tokenizer_name = self._model_name
        self._model.trust_remote_code = self._trust_remote_code
        self._model.draft_model_name = self._draft_model_name
        self._model.num_draft_tokens = self._num_draft_tokens
        self._model.prefill_step_size = self._prefill_step_size
        self._model.kv_bits = self._kv_bits
        self._model.kv_group_size = self._kv_group_size
        self._model._prompt_cache = None
        self._model._cached_token_ids = []
        self._model._cache_lock = asyncio.Lock()
        self._model.model = model
        self._model.tokenizer = tokenizer
        self._model.draft_model = None
        self._model._loaded = True

        # Load draft model separately if specified
        if self._draft_model_name:
            from mlx_lm import load as mlx_load

            logger.info(
                f"Loading draft model for speculative decoding: {self._draft_model_name}"
            )
            try:
                self._model.draft_model, draft_tokenizer = mlx_load(
                    self._draft_model_name
                )
            except Exception as e:
                logger.error(f"Failed to load draft model: {e}")
                self._model.draft_model = None
                raise

            # Validate tokenizer compatibility
            if draft_tokenizer.vocab_size != tokenizer.vocab_size:
                logger.warning(
                    f"Draft model tokenizer vocab size ({draft_tokenizer.vocab_size}) "
                    f"differs from main model ({tokenizer.vocab_size}). "
                    "This may reduce speculative decoding effectiveness."
                )

            logger.info(
                f"Speculative decoding enabled: draft={self._draft_model_name}, "
                f"num_draft_tokens={self._num_draft_tokens}"
            )

        self._loaded = True
        logger.info(f"SimpleEngine injected with shared model: {self._model_name}")

    @property
    def supports_guided_generation(self) -> bool:
        """Check if guided generation is available."""
        return HAS_GUIDED and not self._is_mllm

    async def generate_with_schema(
        self,
        messages: list[dict[str, Any]],
        json_schema: dict[str, Any],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate JSON output constrained to a schema using guided decoding.

        This method uses outlines for constrained generation to guarantee
        the output is valid JSON matching the specified schema.

        Args:
            messages: List of chat messages
            json_schema: JSON schema to constrain output
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            **kwargs: Additional parameters

        Returns:
            GenerationOutput with JSON text matching the schema
        """
        if not self.supports_guided_generation:
            raise RuntimeError(
                "Guided generation not available. "
                "Install with: pip install 'rapid-mlx[guided]'"
            )

        if not self._loaded:
            await self.start()

        # Build prompt from messages
        tokenizer = self._model.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            prompt += "\nassistant:"

        async with self._generation_lock:
            # Run guided generation in thread pool
            result = await asyncio.to_thread(
                self._run_guided_generation,
                prompt=prompt,
                json_schema=json_schema,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            if result is None:
                # Fallback to regular generation INLINE (not via self.generate()
                # which would re-acquire _generation_lock and deadlock —
                # asyncio.Lock is not reentrant).
                logger.warning(
                    "Guided generation failed, falling back to regular generation"
                )
                output = await asyncio.to_thread(
                    self._model.generate,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **kwargs,
                )
                return GenerationOutput(
                    text=output.text,
                    tokens=getattr(output, "tokens", []),
                    prompt_tokens=getattr(output, "prompt_tokens", 0),
                    completion_tokens=len(getattr(output, "tokens", [])),
                    finish_reason=output.finish_reason,
                )

            # Tokenize for completion count
            tokens = tokenizer.encode(result)

            return GenerationOutput(
                text=result,
                tokens=tokens,
                prompt_tokens=len(tokenizer.encode(prompt)),
                completion_tokens=len(tokens),
                finish_reason="stop",
            )

    def _run_guided_generation(
        self,
        prompt: str,
        json_schema: dict[str, Any],
        max_tokens: int,
        temperature: float,
    ) -> str | None:
        """
        Run guided generation synchronously (called from thread pool).

        Args:
            prompt: Input prompt
            json_schema: JSON schema
            max_tokens: Maximum tokens
            temperature: Sampling temperature

        Returns:
            JSON string or None if failed
        """
        try:
            generator = GuidedGenerator(self._model.model, self._model.tokenizer)
            return generator.generate_json(
                prompt=prompt,
                json_schema=json_schema,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            logger.error(f"Guided generation error: {e}")
            return None
