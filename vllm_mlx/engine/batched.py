# SPDX-License-Identifier: Apache-2.0
"""
Batched engine for continuous batching with multiple concurrent users.

This engine wraps AsyncEngineCore to provide continuous batching
for better throughput when serving multiple concurrent requests.

For MLLM models, all requests (text-only and multimodal) are routed through
the MLLMScheduler, which handles vision encoding and batched generation via
MLLMBatchGenerator. MLLM models only initialise the MLLM scheduler (not the
LLM engine), so text-only requests must also be routed through it.
"""

import logging
from collections.abc import AsyncIterator
from typing import Any

from ..api.tool_calling import convert_tools_for_template
from ..api.utils import clean_output_text, extract_multimodal_content, is_mllm_model
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


def _extract_media_from_messages(messages: list[dict[str, Any]]) -> tuple:
    """
    Extract images and videos from OpenAI-format messages.

    Returns:
        Tuple of (has_media, images_list, videos_list)
    """
    images = []
    videos = []

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        for item in content:
            # Handle Pydantic models
            if hasattr(item, "model_dump"):
                item = item.model_dump(exclude_none=True)
            elif hasattr(item, "dict"):
                item = {k: v for k, v in item.dict().items() if v is not None}

            if not isinstance(item, dict):
                continue

            item_type = item.get("type", "")

            if item_type == "image_url":
                img_url = item.get("image_url", {})
                if isinstance(img_url, str):
                    images.append(img_url)
                elif isinstance(img_url, dict):
                    url = img_url.get("url", "")
                    if url:
                        images.append(url)

            elif item_type == "image":
                img = item.get("image") or item.get("url", "")
                if img:
                    images.append(img)

            elif item_type == "video_url":
                vid_url = item.get("video_url", {})
                if isinstance(vid_url, str):
                    videos.append(vid_url)
                elif isinstance(vid_url, dict):
                    url = vid_url.get("url", "")
                    if url:
                        videos.append(url)

            elif item_type == "video":
                vid = item.get("video") or item.get("url", "")
                if vid:
                    videos.append(vid)

    has_media = bool(images or videos)
    return has_media, images, videos


class MLLMModelWrapper:
    """
    Wrapper for MLLM models to make them compatible with BatchGenerator.

    BatchGenerator expects model output to be subscriptable (logits array),
    but MLLM models return LanguageModelOutput objects. This wrapper extracts
    the logits from the output.

    Also handles Gemma 3's required pixel_values argument by injecting None
    for text-only requests.
    """

    def __init__(self, model):
        self._model = model
        # Detect if this is a Gemma 3 model (requires pixel_values as positional arg)
        self._is_gemma3 = (
            hasattr(model, "model_type")
            and "gemma3" in str(getattr(model, "model_type", "")).lower()
        )

    def __call__(self, *args, **kwargs):
        """Call the model and extract logits from LanguageModelOutput."""
        # Gemma 3 requires pixel_values as a positional argument, unlike Qwen
        # which makes it optional. Inject pixel_values=None for text-only requests.
        if self._is_gemma3 and "pixel_values" not in kwargs:
            kwargs["pixel_values"] = None

        output = self._model(*args, **kwargs)
        # If output has logits attribute, return just the logits
        if hasattr(output, "logits"):
            return output.logits
        return output

    def __getattr__(self, name):
        """Forward all other attributes to the wrapped model."""
        return getattr(self._model, name)


class BatchedEngine(BaseEngine):
    """
    Batched engine for continuous batching.

    This engine provides better throughput when serving multiple
    concurrent users by batching requests together.

    For MLLM (multimodal) models, this engine uses MLLMScheduler
    which handles images and videos alongside text generation.
    """

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = True,
        scheduler_config: Any | None = None,
        stream_interval: int = 1,
        force_mllm: bool = False,
        gpu_memory_utilization: float = 0.90,
    ):
        """
        Initialize the batched engine.

        Args:
            model_name: HuggingFace model name or local path
            trust_remote_code: Whether to trust remote code
            scheduler_config: Optional scheduler configuration
            stream_interval: Tokens to batch before streaming (1=every token)
            force_mllm: Force loading as MLLM even if not auto-detected
            gpu_memory_utilization: Fraction of device memory for Metal allocation
                limit and emergency threshold (0.0-1.0, default 0.90)
        """
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._scheduler_config = scheduler_config
        self._stream_interval = stream_interval
        self._gpu_memory_utilization = gpu_memory_utilization
        self._is_mllm = force_mllm or is_mllm_model(model_name)
        self._tool_logits_processor_factory = None

        self._model = None
        self._processor = None  # For MLLM
        self._tokenizer = None  # For LLM
        self._engine = None  # AsyncEngineCore for LLM
        self._mllm_scheduler = None  # MLLMScheduler for MLLM
        self._mllm_instance = None  # MLXMultimodalLM instance
        self._loaded = False
        self._engine_started = False  # Track if engine loop is running

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
        if self._is_mllm and self._processor:
            return getattr(self._processor, "tokenizer", self._processor)
        return self._tokenizer

    def generate_warmup(self) -> None:
        """Run a minimal forward pass to compile Metal shaders."""
        if not self._loaded or self._model is None or self._is_mllm:
            return
        try:
            import mlx.core as mx

            tokens = self._tokenizer.encode("Hi")
            input_ids = mx.array([tokens])
            self._model(input_ids)
            mx.eval(mx.zeros(1))
        except Exception:
            pass  # Non-fatal

    async def start(self) -> None:
        """Start the engine (load model if not loaded)."""
        if self._loaded:
            return

        if self._is_mllm:
            await self._start_mllm()
        else:
            await self._start_llm()

        self._loaded = True
        logger.info(f"BatchedEngine loaded: {self._model_name} (mllm={self._is_mllm})")

    async def _start_mllm(self) -> None:
        """Start the MLLM engine with MLLMScheduler (continuous batching)."""
        from ..mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig
        from ..models.mllm import MLXMultimodalLM

        # Load the MLLM model
        self._mllm_instance = MLXMultimodalLM(
            self._model_name,
            trust_remote_code=self._trust_remote_code,
        )
        self._mllm_instance.load()

        self._model = self._mllm_instance.model
        self._processor = self._mllm_instance.processor

        # Create MLLM scheduler config with batch generator support
        if self._scheduler_config and hasattr(self._scheduler_config, "max_num_seqs"):
            max_num_seqs = self._scheduler_config.max_num_seqs
        else:
            max_num_seqs = 16  # Default for continuous batching

        # Get batch sizes from config if available
        prefill_batch_size = getattr(self._scheduler_config, "prefill_batch_size", 4)
        completion_batch_size = getattr(
            self._scheduler_config, "completion_batch_size", 16
        )

        mllm_config = MLLMSchedulerConfig(
            max_num_seqs=max_num_seqs,
            prefill_batch_size=prefill_batch_size,
            completion_batch_size=completion_batch_size,
            enable_vision_cache=True,
            vision_cache_size=100,
        )

        # Create and start MLLM scheduler
        self._mllm_scheduler = MLLMScheduler(
            model=self._model,
            processor=self._processor,
            config=mllm_config,
        )
        await self._mllm_scheduler.start()

        logger.info(
            f"MLLM Scheduler started with continuous batching: "
            f"max_num_seqs={max_num_seqs}, prefill_batch={prefill_batch_size}, "
            f"completion_batch={completion_batch_size}"
        )

    async def _start_llm(self) -> None:
        """Start the LLM engine with AsyncEngineCore."""
        from ..engine_core import AsyncEngineCore, EngineConfig
        from ..scheduler import SchedulerConfig
        from ..utils.tokenizer import load_model_with_fallback

        # Build tokenizer config
        tokenizer_config = {"trust_remote_code": self._trust_remote_code}

        # Qwen3 fix
        if "qwen3" in self._model_name.lower() or "Qwen3" in self._model_name:
            tokenizer_config["eos_token"] = "<|im_end|>"

        self._model, self._tokenizer = load_model_with_fallback(
            self._model_name,
            tokenizer_config=tokenizer_config,
        )

        # Validate MTP support if enabled
        if self._scheduler_config and self._scheduler_config.enable_mtp:
            from ..patches.qwen3_next_mtp import validate_mtp_support

            if validate_mtp_support(self._model):
                logger.info("[MTP] Model validated for MTP speculative decoding")
            else:
                logger.warning(
                    "[MTP] MTP validation failed — --enable-mtp will be ignored. "
                    "See warnings above for details."
                )

        # Set Metal memory limits to make allocation failures graceful
        # instead of fatal Metal command buffer errors (SIGABRT)
        try:
            import mlx.core as mx

            if mx.metal.is_available():
                device_info = mx.device_info()
                max_recommended = device_info.get(
                    "max_recommended_working_set_size",
                    device_info.get("memory_size", 0),
                )
                if max_recommended > 0:
                    soft_limit = int(max_recommended * self._gpu_memory_utilization)
                    mx.set_memory_limit(soft_limit)
                    mx.set_cache_limit(32 * 1024 * 1024 * 1024)  # 32GB
                    pct = self._gpu_memory_utilization * 100
                    logger.info(
                        f"Metal memory limits set: "
                        f"allocation_limit={soft_limit / 1e9:.1f}GB "
                        f"({pct:.0f}% of {max_recommended / 1e9:.1f}GB), "
                        f"cache_limit=32GB"
                    )
        except Exception as e:
            logger.warning(f"Failed to set Metal memory limits: {e}")

        # Create engine config
        scheduler_config = self._scheduler_config or SchedulerConfig()
        engine_config = EngineConfig(
            model_name=self._model_name,
            scheduler_config=scheduler_config,
            stream_interval=self._stream_interval,
            gpu_memory_utilization=self._gpu_memory_utilization,
            tool_logits_processor_factory=self._tool_logits_processor_factory,
        )

        # Create async engine
        self._engine = AsyncEngineCore(
            model=self._model,
            tokenizer=self._tokenizer,
            config=engine_config,
        )

        await self._engine.engine.start()
        self._engine_started = True

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        if self._mllm_scheduler:
            await self._mllm_scheduler.stop()
            self._mllm_scheduler = None

        if self._engine:
            await self._engine.stop()
            self._engine.engine.close()
            self._engine = None

        self._model = None
        self._tokenizer = None
        self._processor = None
        self._mllm_instance = None
        self._loaded = False
        self._engine_started = False
        logger.info("BatchedEngine stopped")

    def _apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        num_images: int = 0,
        enable_thinking: bool | None = None,
    ) -> str:
        """Apply chat template to messages.

        Uses the processor's (or tokenizer's) apply_chat_template with the
        full message list so that system prompts and conversation history
        are preserved.

        Args:
            messages: Chat messages in OpenAI format.
            tools: Converted tool definitions for template.
            num_images: Number of images (triggers MLLM message preparation).
            enable_thinking: Whether to enable thinking mode (None = auto).
        """
        # Choose the best template applicator.
        # For MLLM models, the processor handles special vision tokens.
        # For text-only models, the tokenizer is sufficient.
        template_applicator = None
        if (
            self._is_mllm
            and self._processor
            and hasattr(self._processor, "apply_chat_template")
        ):
            template_applicator = self._processor
        elif hasattr(self.tokenizer, "apply_chat_template"):
            template_applicator = self.tokenizer

        # Convert OpenAI image_url content parts to HuggingFace format
        # so the processor can insert the correct vision placeholder tokens.
        if self._is_mllm and num_images > 0:
            messages = self._prepare_mllm_messages(messages)

        # If no suitable applicator was found, pass self.tokenizer anyway;
        # the shared function will fall back to plain-text formatting when
        # apply_chat_template is missing.
        applicator = template_applicator or self.tokenizer
        return shared_apply_chat_template(
            applicator,
            messages,
            tools=tools,
            enable_thinking=enable_thinking,
            model_name=self._model_name,
        )

    @staticmethod
    def _prepare_mllm_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert OpenAI-style image_url content to HuggingFace format.

        The OpenAI API uses ``{"type": "image_url", "image_url": {"url": ...}}``
        while HuggingFace processors expect ``{"type": "image"}``.

        Args:
            messages: List of chat messages in OpenAI format. Each message is a
                dict with at least ``role`` and ``content`` keys.

        Returns:
            A new list of messages with ``image_url`` parts replaced by
            ``{"type": "image"}`` entries for the HuggingFace processor.
        """
        prepared = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, list):
                new_content = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        new_content.append({"type": "image"})
                    elif isinstance(part, (dict, str)):
                        new_content.append(part)
                    # skip non-dict/non-str parts to avoid passing unexpected types
                prepared.append({**msg, "content": new_content})
            else:
                prepared.append(msg)
        return prepared

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
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
            images: Optional image URLs/paths (for MLLM)
            videos: Optional video URLs/paths (for MLLM)
            **kwargs: Additional model-specific parameters

        Returns:
            GenerationOutput with complete text
        """
        if not self._loaded:
            await self.start()

        if self._is_mllm and self._mllm_scheduler:
            # Use MLLM scheduler for all requests when model is multimodal.
            # MLLM models only initialise the _mllm_scheduler (not _engine),
            # so text-only requests must also be routed here.
            output = await self._mllm_scheduler.generate(
                prompt=prompt,
                images=images,
                videos=videos,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                video_fps=kwargs.pop("video_fps", None),
                video_max_frames=kwargs.pop("video_max_frames", None),
            )

            return GenerationOutput(
                text=clean_output_text(output.output_text),
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.completion_tokens,
                finish_reason=output.finish_reason,
            )

        # Use LLM engine for text-only (non-MLLM models)
        from ..request import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
        )

        output = await self._engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
        )

        text = clean_output_text(output.output_text)

        return GenerationOutput(
            text=text,
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
            finish_reason=output.finish_reason,
        )

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
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
            images: Optional image URLs/paths (for MLLM)
            videos: Optional video URLs/paths (for MLLM)
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        if not self._loaded:
            await self.start()

        if self._is_mllm and self._mllm_scheduler:
            # Use MLLM scheduler for all streaming when model is multimodal
            request_id = await self._mllm_scheduler.add_request_async(
                prompt=prompt,
                images=images,
                videos=videos,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                video_fps=kwargs.pop("video_fps", None),
                video_max_frames=kwargs.pop("video_max_frames", None),
            )

            async for output in self._mllm_scheduler.stream_outputs(request_id):
                yield GenerationOutput(
                    text=clean_output_text(output.output_text),
                    new_text=output.new_text,
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    finished=output.finished,
                    finish_reason=output.finish_reason,
                )
            return

        # Use LLM engine for text-only
        from ..request import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
        )

        prefix_boundary = kwargs.pop("prefix_boundary", 0)
        request_id = await self._engine.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
            prefix_boundary=prefix_boundary,
        )

        async for output in self._engine.stream_outputs(request_id):
            text = clean_output_text(output.output_text)

            yield GenerationOutput(
                text=text,
                new_text=output.new_text,
                tokens=output.new_token_ids,
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.completion_tokens,
                finished=output.finished,
                finish_reason=output.finish_reason,
                logprobs=output.logprobs,
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

        For MLLM models, all requests (including text-only) are routed through
        the MLLMScheduler for vision-aware batched generation.
        For non-MLLM models, uses the LLM engine with BatchGenerator.

        Args:
            messages: List of chat messages (OpenAI format)
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

        # Extract images/videos from messages (OpenAI multimodal format)
        # Note: We only use extracted media here, messages are already processed by server
        _, extracted_images, extracted_videos = extract_multimodal_content(messages)
        all_images = (images or []) + extracted_images
        all_videos = (videos or []) + extracted_videos

        # Extract enable_thinking before passing kwargs downstream
        enable_thinking = kwargs.pop("enable_thinking", None)

        # Convert tools for template
        template_tools = convert_tools_for_template(tools) if tools else None

        # Apply chat template
        prompt = self._apply_chat_template(
            messages,
            template_tools,
            num_images=len(all_images),
            enable_thinking=enable_thinking,
        )

        return await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            images=all_images if all_images else None,
            videos=all_videos if all_videos else None,
            **kwargs,
        )

    def _compute_prefix_boundary(
        self, messages: list[dict[str, Any]], tools: list[dict] | None = None
    ) -> int:
        """Compute token count for the shared prefix across message variations.

        Uses a two-tokenization approach: tokenize the full prompt twice
        (once as-is, once with the last user message replaced by a dummy)
        and find the longest common prefix (LCP).  This gives the exact
        boundary where different user suffixes diverge, avoiding template
        discrepancies (e.g. Qwen3 <think> markers on last assistant).
        """
        # Find index of last user message
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx is None or last_user_idx == 0:
            return 0
        try:
            template_tools = convert_tools_for_template(tools) if tools else None

            # Tokenize the real prompt
            real_prompt = self._apply_chat_template(messages, template_tools)

            # Build a dummy variant with different last user content
            dummy_messages = list(messages)
            dummy_messages[last_user_idx] = {
                **messages[last_user_idx],
                "content": "XXXXXXXXXX",
            }
            dummy_prompt = self._apply_chat_template(dummy_messages, template_tools)

            tokenizer = self.tokenizer
            if hasattr(tokenizer, "tokenizer"):
                tokenizer = tokenizer.tokenizer

            real_tokens = tokenizer.encode(real_prompt)
            dummy_tokens = tokenizer.encode(dummy_prompt)

            # Find LCP — the point where the two diverge is the boundary
            lcp = 0
            for j in range(min(len(real_tokens), len(dummy_tokens))):
                if real_tokens[j] != dummy_tokens[j]:
                    break
                lcp = j + 1

            return lcp
        except Exception:
            return 0

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

        For MLLM models, all requests (including text-only) are streamed through
        the MLLMScheduler for vision-aware batched generation.
        For non-MLLM models, uses the LLM engine with BatchGenerator.

        Args:
            messages: List of chat messages (OpenAI format)
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

        # Extract images/videos from messages (OpenAI multimodal format)
        # Note: We only use extracted media here, messages are already processed by server
        _, extracted_images, extracted_videos = extract_multimodal_content(messages)
        all_images = (images or []) + extracted_images
        all_videos = (videos or []) + extracted_videos

        # Extract enable_thinking before passing kwargs downstream
        enable_thinking = kwargs.pop("enable_thinking", None)

        # Convert tools for template
        template_tools = convert_tools_for_template(tools) if tools else None

        # Apply chat template
        prompt = self._apply_chat_template(
            messages,
            template_tools,
            num_images=len(all_images),
            enable_thinking=enable_thinking,
        )

        # Compute prefix boundary for cache
        prefix_boundary = self._compute_prefix_boundary(messages, tools)
        if prefix_boundary > 0:
            kwargs["prefix_boundary"] = prefix_boundary

        async for output in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            images=all_images if all_images else None,
            videos=all_videos if all_videos else None,
            **kwargs,
        ):
            yield output

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        stats = {
            "engine_type": "batched",
            "model_name": self._model_name,
            "is_mllm": self._is_mllm,
            "loaded": self._loaded,
            "stream_interval": self._stream_interval,
        }

        if self._mllm_scheduler:
            mllm_stats = self._mllm_scheduler.get_stats()
            stats["mllm_scheduler"] = mllm_stats
            # Promote Metal memory stats to top-level for /v1/status
            for key in (
                "metal_active_memory_gb",
                "metal_peak_memory_gb",
                "metal_cache_memory_gb",
            ):
                if key in mllm_stats:
                    stats[key] = mllm_stats[key]
        elif self._engine:
            stats.update(self._engine.get_stats())

        return stats

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics."""
        if self._mllm_scheduler and self._mllm_scheduler.vision_cache:
            return self._mllm_scheduler.vision_cache.get_stats()
        elif self._engine:
            return self._engine.get_cache_stats()
        return None

    def save_cache_to_disk(self, cache_dir: str) -> bool:
        """Save prefix cache to disk for persistence across restarts."""
        if self._engine:
            return self._engine.save_cache_to_disk(cache_dir)
        return False

    def load_cache_from_disk(self, cache_dir: str) -> int:
        """Load prefix cache from disk. Returns number of entries loaded."""
        if self._engine:
            return self._engine.load_cache_from_disk(cache_dir)
        return 0

    # ------------------------------------------------------------------
    # Guided generation (JSON schema constrained decoding via outlines)
    # ------------------------------------------------------------------

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
        """Generate JSON output constrained to a schema using guided decoding.

        Uses outlines for constrained generation to guarantee the output is
        valid JSON matching the specified schema.  Runs synchronously in a
        thread pool to avoid blocking the event loop.
        """
        import asyncio

        if not self.supports_guided_generation:
            raise RuntimeError(
                "Guided generation not available. "
                "Install with: pip install 'rapid-mlx[guided]'"
            )

        if not self._loaded:
            await self.start()

        # Build prompt from messages
        tokenizer = self.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            prompt += "\nassistant:"

        # Run guided generation in thread pool (outlines is synchronous)
        result = await asyncio.to_thread(
            self._run_guided_generation,
            prompt=prompt,
            json_schema=json_schema,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if result is None:
            # Fallback to standard generation
            logger.warning(
                "Guided generation failed, falling back to regular generation"
            )
            return await self.chat(messages=messages, max_tokens=max_tokens, **kwargs)

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
        """Run guided generation synchronously (called from thread pool)."""
        try:
            model = self._model
            tokenizer = self._tokenizer
            if self._is_mllm:
                return None
            generator = GuidedGenerator(model, tokenizer)
            return generator.generate_json(
                prompt=prompt,
                json_schema=json_schema,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            logger.error(f"Guided generation error: {e}")
            return None

    async def _inject_shared_model(
        self,
        model,
        tokenizer,
        start_engine: bool = True,
    ) -> None:
        """
        Inject a pre-loaded shared model instead of loading a new one.

        This is used to inject a pre-loaded model instance.

        Args:
            model: Pre-loaded MLX model
            tokenizer: Pre-loaded tokenizer
            start_engine: Whether to start the engine loop immediately.
                         Set to False for HybridEngine (lazy start on first use).
        """
        from ..engine_core import AsyncEngineCore, EngineConfig
        from ..scheduler import SchedulerConfig

        self._model = model
        self._tokenizer = tokenizer

        # Create engine config
        scheduler_config = self._scheduler_config or SchedulerConfig()
        engine_config = EngineConfig(
            model_name=self._model_name,
            scheduler_config=scheduler_config,
            stream_interval=self._stream_interval,
            tool_logits_processor_factory=self._tool_logits_processor_factory,
        )

        # Create async engine with shared model
        self._engine = AsyncEngineCore(
            model=self._model,
            tokenizer=self._tokenizer,
            config=engine_config,
        )

        # Only start engine loop if requested (HybridEngine starts lazily)
        if start_engine:
            await self._engine.engine.start()

        self._loaded = True
        self._engine_started = start_engine
        logger.info(
            f"BatchedEngine injected with shared model: {self._model_name} (started={start_engine})"
        )
