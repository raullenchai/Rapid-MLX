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

import asyncio
import inspect
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from ..api.tool_calling import convert_tools_for_template
from ..api.utils import clean_output_text, extract_multimodal_content, is_mllm_model
from .base import (
    BaseEngine,
    GenerationOutput,
    cleanup_startup_cancellation,
    run_blocking_startup_work,
)
from .chat_template_safety import normalize_messages_for_chat_template

logger = logging.getLogger(__name__)


def _normalize_tool_call_arguments_for_template(messages: list[dict]) -> list[dict]:
    """Normalize OpenAI tool-call replay for templates expecting mappings."""
    return normalize_messages_for_chat_template(messages)


def _extract_media_from_messages(messages: list[dict[str, Any]]) -> tuple:
    """
    Extract images, videos, and audio from OpenAI-format messages.

    Returns:
        Tuple of (has_media, images_list, videos_list, audios_list)
    """
    images = []
    videos = []
    audios = []

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

            elif item_type == "audio_url":
                audio_url = item.get("audio_url", {})
                if isinstance(audio_url, str):
                    audios.append(audio_url)
                elif isinstance(audio_url, dict):
                    url = audio_url.get("url", "")
                    if url:
                        audios.append(url)

            elif item_type == "audio":
                audio = item.get("audio") or item.get("url", "")
                if audio:
                    audios.append(audio)

    has_media = bool(images or videos or audios)
    return has_media, images, videos, audios


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
        trust_remote_code: bool = False,
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
        self._created_at = time.time()
        self._trust_remote_code = trust_remote_code
        self._scheduler_config = scheduler_config
        self._stream_interval = stream_interval
        self._gpu_memory_utilization = gpu_memory_utilization
        self._is_mllm = force_mllm or is_mllm_model(model_name)

        self._model = None
        self._processor = None  # For MLLM
        self._tokenizer = None  # For LLM
        self._engine = None  # AsyncEngineCore for LLM
        self._mllm_scheduler = None  # MLLMScheduler for MLLM
        self._mllm_instance = None  # MLXMultimodalLM instance
        self._loaded = False

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

    def prepare_for_start(self) -> None:
        """Load heavyweight model state off the serving event loop."""
        if self._model is not None:
            return

        if self._is_mllm:
            self._prepare_mllm_model()
        else:
            self._prepare_llm_model()

    async def start(self) -> None:
        """Start the engine (load model if not loaded)."""
        if self._loaded:
            return

        try:
            if self._model is None:
                if self._uses_default_prepare_for_start():
                    # Load inline on the event-loop thread so mlx-lm's
                    # generation_stream (created at module import on this
                    # thread) and model weights are owned by the same thread
                    # that drives scheduler.step (issue #407).
                    self.prepare_for_start()
                else:
                    # Test doubles and custom overrides may block; run them via
                    # the shared cancellation-safe thread helper.
                    await run_blocking_startup_work(self.prepare_for_start)

            if self._is_mllm:
                await self._start_mllm()
            else:
                await self._start_llm()

            self._loaded = True
            logger.info(
                f"BatchedEngine loaded: {self._model_name} (mllm={self._is_mllm})"
            )
        except asyncio.CancelledError:
            await cleanup_startup_cancellation(self.stop)
            raise

    def _uses_default_prepare_for_start(self) -> bool:
        """Return True when prepare_for_start is the class implementation."""
        method = getattr(self.prepare_for_start, "__func__", None)
        return method is BatchedEngine.prepare_for_start

    def _prepare_mllm_model(self) -> None:
        """Load the MLLM model before scheduler startup."""
        from ..models.mllm import MLXMultimodalLM

        max_kv_size = getattr(self._scheduler_config, "max_kv_size", 0)
        self._mllm_instance = MLXMultimodalLM(
            self._model_name,
            trust_remote_code=self._trust_remote_code,
            max_kv_size=max_kv_size,
        )
        self._mllm_instance.load()
        self._model = self._mllm_instance.model
        self._processor = self._mllm_instance.processor

        # Set Metal memory limits (same as LLM path)
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

        # Inject MTP support if enabled
        if self._scheduler_config and self._scheduler_config.enable_mtp:
            self._inject_mtp_mllm()

    async def _start_mllm(self) -> None:
        """Start the MLLM engine with MLLMScheduler (continuous batching)."""
        from ..mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig

        if self._model is None or self._processor is None:
            self._prepare_mllm_model()

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

        cache_memory_mb = getattr(self._scheduler_config, "cache_memory_mb", None)
        max_kv_size = getattr(self._scheduler_config, "max_kv_size", 0)
        enable_prefix_cache = getattr(
            self._scheduler_config, "enable_prefix_cache", True
        )
        use_memory_aware_cache = getattr(
            self._scheduler_config, "use_memory_aware_cache", True
        )
        prefix_cache_memory_mb = getattr(
            self._scheduler_config, "cache_memory_mb", None
        )
        enable_mtp = (
            self._scheduler_config.enable_mtp if self._scheduler_config else False
        )
        mtp_num_draft = getattr(self._scheduler_config, "mtp_num_draft_tokens", 1)
        kv_quant = getattr(self._scheduler_config, "kv_cache_quantization", False)
        kv_bits = getattr(self._scheduler_config, "kv_cache_quantization_bits", 8)
        kv_group_size = getattr(
            self._scheduler_config, "kv_cache_quantization_group_size", 64
        )

        chunked_prefill_tokens = getattr(
            self._scheduler_config, "chunked_prefill_tokens", 0
        )

        prefill_step_size = getattr(
            self._scheduler_config, "mllm_prefill_step_size", None
        )
        if prefill_step_size is None:
            prefill_step_size = getattr(
                self._scheduler_config, "prefill_step_size", None
            )
        mllm_extra = {}
        if prefill_step_size is not None:
            mllm_extra["prefill_step_size"] = prefill_step_size
        mllm_config = MLLMSchedulerConfig(
            max_num_seqs=max_num_seqs,
            prefill_batch_size=prefill_batch_size,
            completion_batch_size=completion_batch_size,
            enable_vision_cache=True,
            vision_cache_size=100,
            cache_memory_mb=cache_memory_mb,
            enable_prefix_cache=enable_prefix_cache,
            use_memory_aware_cache=use_memory_aware_cache,
            prefix_cache_memory_mb=prefix_cache_memory_mb,
            enable_mtp=enable_mtp,
            mtp_num_draft_tokens=mtp_num_draft,
            kv_cache_quantization=kv_quant,
            kv_cache_quantization_bits=kv_bits,
            kv_cache_quantization_group_size=kv_group_size,
            chunked_prefill_tokens=chunked_prefill_tokens,
            max_kv_size=max_kv_size,
            **mllm_extra,
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
            f"completion_batch={completion_batch_size}, "
            f"prefill_step_size={mllm_config.prefill_step_size}"
        )

    def _inject_mtp_mllm(self) -> None:
        """Inject MTP weights into the MLLM model's language_model."""
        import json
        from pathlib import Path

        from mlx_lm.utils import _download

        model = self._model
        model_path = Path(_download(self._model_name))
        config_path = model_path / "config.json"
        if not config_path.exists():
            logger.warning("[MTP-MLLM] No config.json found, skipping MTP")
            return

        with open(config_path) as f:
            config = json.load(f)

        text_config = config.get("text_config", config)
        num_mtp = text_config.get("mtp_num_hidden_layers", 0)
        if num_mtp == 0:
            num_mtp = text_config.get(
                "num_nextn_predict_layers",
                config.get("num_nextn_predict_layers", 0),
            )
        if num_mtp == 0:
            logger.info("[MTP-MLLM] No MTP layers in config, skipping")
            return

        # Navigate to text model
        text_model = model
        if hasattr(model, "language_model"):
            text_model = model.language_model
        if getattr(text_model, "mtp", None) is not None:
            logger.info("[MTP-MLLM] Model already has MTP, skipping injection")
            return

        model_type = text_config.get("model_type", config.get("model_type", ""))
        if "qwen3_5" in model_type:
            from ..patches.qwen3_5_mtp import inject_mtp_support

            ok = inject_mtp_support(model, model_path, config)
            if ok:
                logger.info("[MTP-MLLM] Qwen3.5 MTP injected successfully")
            else:
                logger.warning("[MTP-MLLM] Qwen3.5 MTP injection failed")
        else:
            logger.info(f"[MTP-MLLM] MTP not supported for model_type={model_type}")

    def _prepare_llm_model(self) -> None:
        """Load the LLM model/tokenizer before engine loop startup."""
        from ..utils.tokenizer import load_model_with_fallback

        if self._model is not None and self._tokenizer is not None:
            return

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
            from ..patches.qwen3_5_mtp import validate_mtp_support as validate_35
            from ..patches.qwen3_next_mtp import validate_mtp_support

            if validate_mtp_support(self._model) or validate_35(self._model):
                logger.info("[MTP] Model validated for MTP speculative decoding")
            else:
                logger.warning(
                    "[MTP] MTP validation failed — --enable-mtp will be ignored. "
                    "See warnings above for details."
                )

        self._configure_metal_memory_limits()

    def _configure_metal_memory_limits(self) -> None:
        """Make MLX allocation failures graceful during startup."""
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

    async def _start_llm(self) -> None:
        """Start the LLM engine with AsyncEngineCore."""
        from ..engine_core import AsyncEngineCore, EngineConfig
        from ..scheduler import SchedulerConfig

        if self._model is None or self._tokenizer is None:
            self._prepare_llm_model()

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

        # Create engine config
        scheduler_config = self._scheduler_config or SchedulerConfig()
        engine_config = EngineConfig(
            model_name=self._model_name,
            scheduler_config=scheduler_config,
            stream_interval=self._stream_interval,
            gpu_memory_utilization=self._gpu_memory_utilization,
        )

        # Create async engine
        self._engine = AsyncEngineCore(
            model=self._model,
            tokenizer=self._tokenizer,
            config=engine_config,
        )

        await self._engine.engine.start()

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
        logger.info("BatchedEngine stopped")

    def _apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        num_images: int = 0,
        num_audios: int = 0,
        chat_template_kwargs: dict[str, Any] | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        """Apply chat template to messages.

        Uses the processor's (or tokenizer's) apply_chat_template with the
        full message list so that system prompts and conversation history
        are preserved. The previous implementation extracted only the last
        user message text via mlx_vlm.prompt_utils.apply_chat_template,
        which dropped system prompts and all prior turns.
        """
        messages = _normalize_tool_call_arguments_for_template(messages)

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

        if template_applicator is not None:
            # Convert OpenAI image_url content parts to HuggingFace format
            # so the processor can insert the correct vision placeholder tokens.
            if self._is_mllm and (num_images > 0 or num_audios > 0):
                messages = self._prepare_mllm_messages(messages)

            # Per-request enable_thinking override; default: True unless coder model.
            if enable_thinking is None:
                enable_thinking = "coder" not in self._model_name.lower()
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": enable_thinking,
            }
            if chat_template_kwargs:
                template_kwargs.update(chat_template_kwargs)
            if tools and "tools" not in template_kwargs:
                template_kwargs["tools"] = tools

            tokenizer_applicator = None
            tokenizer = self.tokenizer
            if template_applicator is not tokenizer and hasattr(
                tokenizer, "apply_chat_template"
            ):
                tokenizer_applicator = tokenizer

            try:
                return template_applicator.apply_chat_template(
                    messages, **template_kwargs
                )
            except ValueError as e:
                # Some HF processors define apply_chat_template but do not carry
                # a template (e.g. Gemma-3 processor). Retry on tokenizer.
                if (
                    tokenizer_applicator is not None
                    and "does not have a chat template" in str(e)
                ):
                    return tokenizer_applicator.apply_chat_template(
                        messages, **template_kwargs
                    )
                raise
            except TypeError as e:
                # Some templates don't accept extra kwargs; retry without them.
                logger.debug(f"Chat template TypeError, retrying without extras: {e}")
                for key in [
                    "tools",
                    "enable_thinking",
                    *(chat_template_kwargs or {}).keys(),
                ]:
                    template_kwargs.pop(key, None)
                return template_applicator.apply_chat_template(
                    messages, **template_kwargs
                )
        else:
            # Fallback for models without apply_chat_template
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            return prompt + "\nassistant:"

    @staticmethod
    def _prepare_mllm_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert OpenAI-style multimodal content to HuggingFace format.

        The OpenAI API uses ``{"type": "image_url", "image_url": {"url": ...}}``
        and ``{"type": "audio_url", "audio_url": {"url": ...}}`` while
        HuggingFace processors expect ``{"type": "image"}`` / ``{"type": "audio"}``.

        Args:
            messages: List of chat messages in OpenAI format. Each message is a
                dict with at least ``role`` and ``content`` keys.

        Returns:
            A new list of messages with ``image_url`` / ``audio_url`` parts
            replaced by ``{"type": "image"}`` / ``{"type": "audio"}`` entries
            for the HuggingFace processor.
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
                    elif isinstance(part, dict) and part.get("type") == "audio_url":
                        new_content.append({"type": "audio"})
                    elif isinstance(part, (dict | str)):
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
        audio: list[str] | None = None,
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
            audio: Optional audio URLs/paths (for MLLM)
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
                audio=audio,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=kwargs.pop("top_k", 0),
                min_p=kwargs.pop("min_p", 0.0),
                presence_penalty=kwargs.pop("presence_penalty", 0.0),
                repetition_penalty=kwargs.pop("repetition_penalty", 1.0),
                stop=stop or [],
                logits_processors=kwargs.pop("logits_processors", None),
            )

            return GenerationOutput(
                text=clean_output_text(output.output_text),
                tokens=output.output_token_ids,
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.completion_tokens,
                finish_reason=output.finish_reason,
                mtp_drafts=output.mtp_drafts,
                mtp_accepted=output.mtp_accepted,
                matched_stop=getattr(output, "matched_stop", None),
            )

        # Use LLM engine for text-only (non-MLLM models)
        from ..request import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=kwargs.pop("top_k", 0),
            min_p=kwargs.pop("min_p", 0.0),
            presence_penalty=kwargs.pop("presence_penalty", 0.0),
            repetition_penalty=kwargs.pop("repetition_penalty", 1.0),
            stop=stop or [],
            logits_processors=kwargs.pop("logits_processors", None),
        )

        output = await self._engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
        )

        text = clean_output_text(output.output_text)

        return GenerationOutput(
            text=text,
            tokens=output.output_token_ids,
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
            finish_reason=output.finish_reason,
            matched_stop=getattr(output, "matched_stop", None),
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
        audio: list[str] | None = None,
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
            audio: Optional audio URLs/paths (for MLLM)
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
                audio=audio,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=kwargs.pop("top_k", 0),
                min_p=kwargs.pop("min_p", 0.0),
                presence_penalty=kwargs.pop("presence_penalty", 0.0),
                repetition_penalty=kwargs.pop("repetition_penalty", 1.0),
                stop=stop or [],
                logits_processors=kwargs.pop("logits_processors", None),
            )

            async for output in self._mllm_scheduler.stream_outputs(request_id):
                yield GenerationOutput(
                    text=clean_output_text(output.output_text),
                    new_text=output.new_text,
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    finished=output.finished,
                    finish_reason=output.finish_reason,
                    mtp_drafts=output.mtp_drafts,
                    mtp_accepted=output.mtp_accepted,
                    matched_stop=getattr(output, "matched_stop", None),
                )
            return

        # Use LLM engine for text-only
        from ..request import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=kwargs.pop("top_k", 0),
            min_p=kwargs.pop("min_p", 0.0),
            presence_penalty=kwargs.pop("presence_penalty", 0.0),
            repetition_penalty=kwargs.pop("repetition_penalty", 1.0),
            stop=stop or [],
            logits_processors=kwargs.pop("logits_processors", None),
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
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.completion_tokens,
                finished=output.finished,
                finish_reason=output.finish_reason,
                matched_stop=getattr(output, "matched_stop", None),
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

        # Extract images/videos/audio from messages (OpenAI multimodal format)
        # Note: We only use extracted media here, messages are already processed by server
        _, extracted_images, extracted_videos, extracted_audios = (
            extract_multimodal_content(messages)
        )
        all_images = (images or []) + extracted_images
        all_videos = (videos or []) + extracted_videos
        all_audios = extracted_audios

        # Convert tools for template
        template_tools = convert_tools_for_template(tools) if tools else None
        chat_template_kwargs = dict(kwargs.pop("chat_template_kwargs", {}) or {})

        # Per-request enable_thinking override
        enable_thinking = kwargs.pop("enable_thinking", None)

        # Apply chat template
        prompt = self._apply_chat_template(
            messages,
            template_tools,
            num_images=len(all_images),
            num_audios=len(all_audios),
            chat_template_kwargs=chat_template_kwargs,
            enable_thinking=enable_thinking,
        )

        return await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            images=all_images if all_images else None,
            videos=all_videos if all_videos else None,
            audio=all_audios if all_audios else None,
            **kwargs,
        )

    def _compute_prefix_boundary(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
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
            real_prompt = self._apply_chat_template(
                messages,
                template_tools,
                chat_template_kwargs=chat_template_kwargs,
            )

            # Build a dummy variant with different last user content
            dummy_messages = list(messages)
            dummy_messages[last_user_idx] = {
                **messages[last_user_idx],
                "content": "XXXXXXXXXX",
            }
            dummy_prompt = self._apply_chat_template(
                dummy_messages,
                template_tools,
                chat_template_kwargs=chat_template_kwargs,
            )

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

        # Extract images/videos/audio from messages (OpenAI multimodal format)
        # Note: We only use extracted media here, messages are already processed by server
        _, extracted_images, extracted_videos, extracted_audios = (
            extract_multimodal_content(messages)
        )
        all_images = (images or []) + extracted_images
        all_videos = (videos or []) + extracted_videos
        all_audios = extracted_audios

        # Convert tools for template
        template_tools = convert_tools_for_template(tools) if tools else None
        chat_template_kwargs = dict(kwargs.pop("chat_template_kwargs", {}) or {})

        # Per-request enable_thinking override
        enable_thinking = kwargs.pop("enable_thinking", None)

        # Apply chat template
        prompt = self._apply_chat_template(
            messages,
            template_tools,
            num_images=len(all_images),
            num_audios=len(all_audios),
            chat_template_kwargs=chat_template_kwargs,
            enable_thinking=enable_thinking,
        )

        # Compute prefix boundary for cache
        prefix_boundary = self._compute_prefix_boundary(
            messages,
            tools,
            chat_template_kwargs=chat_template_kwargs,
        )
        if prefix_boundary > 0:
            kwargs["prefix_boundary"] = prefix_boundary

        async for output in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            images=all_images if all_images else None,
            videos=all_videos if all_videos else None,
            audio=all_audios if all_audios else None,
            **kwargs,
        ):
            yield output

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        stats = {
            "engine_type": "batched",
            "model_name": self._model_name,
            "uptime_seconds": time.time() - self._created_at,
            "is_mllm": self._is_mllm,
            "loaded": self._loaded,
            "stream_interval": self._stream_interval,
        }

        if self._mllm_scheduler:
            mllm_stats = self._mllm_scheduler.get_stats()
            stats["mllm_scheduler"] = mllm_stats
            # Promote stats to top-level for /v1/status and monitoring
            for key in (
                "running",
                "num_running",
                "num_waiting",
                "num_requests_processed",
                "total_prompt_tokens",
                "total_completion_tokens",
                "metal_active_memory_gb",
                "metal_peak_memory_gb",
                "metal_cache_memory_gb",
                "memory_aware_cache",
                "paged_cache",
                "prefix_cache",
                "batch_generator",
                "mtp",
                "requests",
            ):
                if key in mllm_stats:
                    stats[key] = mllm_stats[key]
            # MLLM engine is always "running" once loaded
            if "running" not in stats:
                stats["running"] = self._loaded
        elif self._engine:
            stats.update(self._engine.get_stats())

        return stats

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics."""
        if self._mllm_scheduler and self._mllm_scheduler.batch_generator:
            return {
                "prefix_cache": self._mllm_scheduler.batch_generator.get_prefix_cache_stats(),
                "vision_embedding_cache": self._mllm_scheduler.batch_generator.get_vision_cache_stats(),
            }
        elif self._engine:
            return self._engine.get_cache_stats()
        return None

    def clear_runtime_caches(self) -> dict[str, Any] | None:
        """Clear engine-managed runtime caches."""
        if self._mllm_scheduler is not None:
            return self._mllm_scheduler.clear_runtime_caches()
        if self._engine is not None:
            return self._engine.clear_runtime_caches()
        return None

    async def abort_request(self, request_id: str) -> bool:
        """Abort an active or queued batched request by request ID."""
        if self._mllm_scheduler is not None:
            return self._mllm_scheduler.abort_request(request_id)
        if self._engine is not None and hasattr(self._engine, "abort_request"):
            result = self._engine.abort_request(request_id)
            if inspect.isawaitable(result):
                return await result
            return result
        return False

    def save_cache_to_disk(self, cache_dir: str) -> bool:
        """Save prefix cache to disk for persistence across restarts."""
        if self._mllm_scheduler and self._mllm_scheduler.batch_generator:
            pc = self._mllm_scheduler.batch_generator.prefix_cache
            if pc is not None:
                return pc.save_to_disk(cache_dir)
        if self._engine:
            return self._engine.save_cache_to_disk(cache_dir)
        return False

    def load_cache_from_disk(self, cache_dir: str) -> int:
        """Load prefix cache from disk. Returns number of entries loaded."""
        if self._mllm_scheduler:
            self._mllm_scheduler._ensure_batch_generator()
            pc = self._mllm_scheduler.batch_generator.prefix_cache
            if pc is not None:
                return pc.load_from_disk(cache_dir)
        if self._engine:
            return self._engine.load_cache_from_disk(cache_dir)
        return 0

    def clear_prefix_cache(self) -> None:
        """Clear the in-memory prefix cache. Used by bench-serve for clean
        cold-start measurements between configurations."""
        if self._mllm_scheduler and self._mllm_scheduler.batch_generator:
            pc = self._mllm_scheduler.batch_generator.prefix_cache
            if pc is not None and hasattr(pc, "clear"):
                pc.clear()
                return
        if self._engine and hasattr(self._engine, "clear_prefix_cache"):
            self._engine.clear_prefix_cache()
