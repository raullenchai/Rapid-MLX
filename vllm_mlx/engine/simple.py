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
from ..api.utils import clean_output_text, is_mllm_model
from .base import BaseEngine, GenerationOutput

logger = logging.getLogger(__name__)


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
        mtp: bool = False,
        prefill_step_size: int = 2048,
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
            mtp: Enable native MTP speculative decoding (model must have MTP head)
            prefill_step_size: Chunk size for prompt prefill processing (default: 2048)
            specprefill_enabled: Enable SpecPrefill (attention-based sparse prefill)
            specprefill_threshold: Minimum suffix tokens to trigger SpecPrefill
            specprefill_keep_pct: Fraction of tokens to keep (default: 0.3)
            specprefill_draft_model: Path to small draft model for importance scoring
        """
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._enable_cache = enable_cache
        self._is_mllm = force_mllm or is_mllm_model(model_name)
        self._mtp = mtp
        self._prefill_step_size = prefill_step_size

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
        else:
            from ..models.llm import MLXLanguageModel

            self._model = MLXLanguageModel(
                self._model_name,
                trust_remote_code=self._trust_remote_code,
                mtp=self._mtp,
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

        mtp_info = f", MTP={self._mtp}" if self._mtp else ""
        routing = ", routing=per-request" if self._text_model is not None else ""
        specprefill_info = (
            ", SpecPrefill=active" if self._draft_model is not None else ""
        )
        logger.info(
            f"SimpleEngine loaded: {self._model_name} "
            f"(MLLM={self._is_mllm}{mtp_info}{routing}{specprefill_info})"
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

            # Clean output text
            text = clean_output_text(output.text)

            return GenerationOutput(
                text=text,
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

            for chunk in self._model.stream_generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **kwargs,
            ):
                prompt_tokens = (
                    chunk.prompt_tokens
                    if hasattr(chunk, "prompt_tokens")
                    else prompt_tokens
                )
                completion_tokens += 1
                new_text = chunk.text if hasattr(chunk, "text") else str(chunk)
                accumulated_text += new_text

                finished = (
                    getattr(chunk, "finished", False) or completion_tokens >= max_tokens
                )
                finish_reason = None
                if finished:
                    finish_reason = getattr(chunk, "finish_reason", "stop")

                yield GenerationOutput(
                    text=accumulated_text,
                    new_text=new_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    finished=finished,
                    finish_reason=finish_reason,
                )

                if finished:
                    break

            if not finished:
                if prompt_tokens == 0:
                    prompt_tokens = len(self._model.tokenizer.encode(prompt))
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

        async with self._generation_lock:
            if self._is_mllm:
                # For MLLM, use the chat method which handles images/videos
                # Run in thread pool to allow asyncio timeout to work
                output = await asyncio.to_thread(
                    self._model.chat,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    tools=template_tools,
                    **kwargs,
                )
                text = clean_output_text(output.text)
                return GenerationOutput(
                    text=text,
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    finish_reason=output.finish_reason,
                )
            else:
                # For LLM, use the chat method
                # Run in thread pool to allow asyncio timeout to work
                output = await asyncio.to_thread(
                    self._model.chat,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    tools=template_tools,
                    **kwargs,
                )
                text = clean_output_text(output.text)
                return GenerationOutput(
                    text=text,
                    tokens=output.tokens,
                    completion_tokens=len(output.tokens),
                    finish_reason=output.finish_reason,
                )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
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
            # Must hold _generation_lock to prevent concurrent Metal access
            # (e.g. OpenCode sends title + main request simultaneously).
            async with self._generation_lock:
                accumulated_text = ""
                token_count = 0

                # Run stream_chat in thread pool since it's synchronous
                def run_stream():
                    return list(
                        self._model.stream_chat(
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            tools=template_tools,
                            **kwargs,
                        )
                    )

                chunks = await asyncio.to_thread(run_stream)

                for chunk in chunks:
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
        tokenizer = self._model.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            # Disable thinking mode for coder models since it interferes
            # with tool call parsing (tags leak as raw text).
            enable_thinking = "coder" not in self._model_name.lower()
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": enable_thinking,
            }
            if template_tools:
                template_kwargs["tools"] = template_tools

            try:
                prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
            except TypeError:
                # Some templates don't support all kwargs
                for key in ["tools", "enable_thinking"]:
                    if key in template_kwargs:
                        del template_kwargs[key]
                prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            prompt += "\nassistant:"

        # Stream generate
        async for output in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
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

            is_last = i == len(all_resps) - 1
            finished = is_last or token_count >= max_tokens

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
