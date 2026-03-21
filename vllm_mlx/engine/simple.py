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
        prefill_step_size: int = 2048,
        kv_bits: int | None = None,
        kv_group_size: int = 64,
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
            prefill_step_size: Tokens to process per prefill chunk (default: 2048)
            kv_bits: KV cache quantization bits (None=no quantization, 4 or 8)
            kv_group_size: Group size for KV cache quantization (default: 64)
        """
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._enable_cache = enable_cache
        self._is_mllm = force_mllm or is_mllm_model(model_name)
        self._draft_model_name = draft_model
        self._num_draft_tokens = num_draft_tokens
        self._prefill_step_size = prefill_step_size
        self._kv_bits = kv_bits
        self._kv_group_size = kv_group_size
        self._model = None
        self._loaded = False

        # Lock to serialize MLX operations (prevents Metal command buffer conflicts)
        self._generation_lock = asyncio.Lock()

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
                prefill_step_size=self._prefill_step_size,
                kv_bits=self._kv_bits,
                kv_group_size=self._kv_group_size,
            )

        self._model.load()
        self._loaded = True

        spec_info = ""
        if self._draft_model_name and not self._is_mllm:
            spec_info = f", speculative={self._draft_model_name}"
        logger.info(
            f"SimpleEngine loaded: {self._model_name} (MLLM={self._is_mllm}{spec_info})"
        )

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        self._model = None
        self._loaded = False
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

        async with self._generation_lock:
            if self._is_mllm:
                # For MLLM, use the chat method which handles images/videos
                # Run in thread pool to allow asyncio timeout to work
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

        # Build prompt using tokenizer
        if self._is_mllm:
            # For MLLM, use stream_chat which yields tokens incrementally.
            # Must hold the generation lock to prevent concurrent Metal
            # command buffer conflicts with other generation methods.
            async with self._generation_lock:
                accumulated_text = ""
                token_count = 0

                # Run the synchronous generator in a thread
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
        enable_thinking = kwargs.get("enable_thinking")
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
            **kwargs,
        ):
            yield output

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        stats = {
            "engine_type": "simple",
            "model_name": self._model_name,
            "is_mllm": self._is_mllm,
            "loaded": self._loaded,
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
