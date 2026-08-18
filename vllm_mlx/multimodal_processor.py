# SPDX-License-Identifier: Apache-2.0
"""
Multimodal processor for VLM continuous batching.

This module handles preprocessing of multimodal inputs (images, videos)
for use with the continuous batching scheduler. It extracts processed
inputs that can be batched together efficiently.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from .models.mllm import (
    DEFAULT_FPS,
    MAX_FRAMES,
    extract_video_frames_smart,
    process_image_input,
    process_video_input,
    save_frames_to_temp,
)

logger = logging.getLogger(__name__)


@dataclass
class ProcessedMultimodalInput:
    """
    Container for processed multimodal inputs ready for batching.

    Attributes:
        input_ids: Tokenized text with image/video tokens (mx.array)
        pixel_values: Processed image tensors (mx.array)
        attention_mask: Attention mask for the input (mx.array)
        image_grid_thw: Grid info for Qwen-VL models (mx.array)
        num_images: Number of images in this input
        num_tokens: Number of tokens in input_ids
        extra_kwargs: Additional model-specific kwargs
    """

    input_ids: mx.array
    pixel_values: mx.array | None = None
    attention_mask: mx.array | None = None
    image_grid_thw: mx.array | None = None
    num_images: int = 0
    num_tokens: int = 0
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


class MultimodalProcessor:
    """
    Processor for preparing multimodal inputs for VLM batching.

    This class wraps mlx_vlm's prepare_inputs function and provides
    a clean interface for the scheduler to preprocess requests.

    Example:
        >>> processor = MultimodalProcessor(model, vlm_processor)
        >>> processed = processor.process(
        ...     prompt="What's in this image?",
        ...     images=["photo.jpg"]
        ... )
        >>> # processed.input_ids, processed.pixel_values ready for batching
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        config: Any | None = None,
    ):
        """
        Initialize the multimodal processor.

        Args:
            model: The VLM model (for config access)
            processor: The VLM processor (tokenizer + image processor)
            config: Optional model config
        """
        self.model = model
        self.processor = processor
        self.config = config or getattr(model, "config", None)

        # Get tokenizer from processor
        self.tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )

        # Get image token index if available
        self.image_token_index = (
            getattr(self.config, "image_token_index", None) if self.config else None
        )

    def process(
        self,
        prompt: str,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        video_fps: float = DEFAULT_FPS,
        video_max_frames: int = MAX_FRAMES,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> ProcessedMultimodalInput:
        """
        Process multimodal inputs for batching.

        Args:
            prompt: Text prompt (already formatted with chat template)
            images: List of image paths, URLs, or base64 strings
            videos: List of video inputs
            video_fps: FPS for video frame extraction
            video_max_frames: Max frames per video
            add_special_tokens: Whether to add special tokens
            **kwargs: Additional model-specific parameters

        Returns:
            ProcessedMultimodalInput with all processed tensors
        """
        from mlx_vlm.utils import prepare_inputs

        # Process raw images
        # Image-fetch failures used to be swallowed with logger.warning + continue,
        # leaving the request to proceed with an empty image list. The model would
        # then either emit an empty completion (case in #457) or hallucinate plausible
        # content from no input. Raise to surface a clean HTTP 400 at the route layer
        # — mirrors the pattern already used by mllm_batch_generator.py:485.
        all_images = []
        if images:
            for img in images:
                try:
                    path = process_image_input(img)
                except Exception as e:
                    raise ValueError(f"Failed to process image: {e}") from e
                all_images.append(path)

        # Extract frames from videos — same semantics as images above.
        if videos:
            for video in videos:
                try:
                    video_path = process_video_input(video)
                    frames = extract_video_frames_smart(
                        video_path,
                        fps=video_fps,
                        max_frames=video_max_frames,
                    )
                    frame_paths = save_frames_to_temp(frames)
                except Exception as e:
                    raise ValueError(f"Failed to process video: {e}") from e
                all_images.extend(frame_paths)
                logger.debug(f"Extracted {len(frame_paths)} frames from video")

        # Determine add_special_tokens based on model type
        if self.config and self.config.model_type in ["gemma3", "gemma3n"]:
            add_special_tokens = not hasattr(self.processor, "chat_template")

        # Prepare inputs using mlx_vlm
        inputs = prepare_inputs(
            self.processor,
            images=all_images if all_images else None,
            prompts=prompt,
            image_token_index=self.image_token_index,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )

        # Extract processed tensors
        input_ids = inputs.get("input_ids")
        pixel_values = inputs.get("pixel_values")
        attention_mask = inputs.get("attention_mask")

        # Extract model-specific kwargs
        extra_kwargs = {
            k: v
            for k, v in inputs.items()
            if k not in ["input_ids", "pixel_values", "attention_mask"]
        }

        # Get image_grid_thw for Qwen-VL models
        image_grid_thw = extra_kwargs.pop("image_grid_thw", None)

        return ProcessedMultimodalInput(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            num_images=len(all_images),
            num_tokens=input_ids.size if input_ids is not None else 0,
            extra_kwargs=extra_kwargs,
        )

    def process_for_request(
        self,
        prompt: str,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Process inputs and return a dict suitable for Request fields.

        This is a convenience method that returns the processed data
        in a format that can be directly assigned to Request fields.

        Args:
            prompt: Text prompt
            images: List of image inputs
            videos: List of video inputs
            **kwargs: Additional parameters

        Returns:
            Dict with keys matching Request multimodal fields
        """
        processed = self.process(prompt, images, videos, **kwargs)

        return {
            "prompt_token_ids": (
                processed.input_ids.tolist()
                if processed.input_ids is not None
                else None
            ),
            "num_prompt_tokens": processed.num_tokens,
            "pixel_values": processed.pixel_values,
            "attention_mask": processed.attention_mask,
            "image_grid_thw": processed.image_grid_thw,
            "multimodal_kwargs": processed.extra_kwargs,
            "is_multimodal": processed.num_images > 0,
        }

    def batch_pixel_values(
        self,
        pixel_values_list: list[mx.array | None],
    ) -> mx.array | None:
        """
        Batch multiple pixel_values tensors together.

        For VLM batching, we need to concatenate pixel values from
        multiple requests. This handles the case where some requests
        may not have images.

        Args:
            pixel_values_list: List of pixel_values from multiple requests

        Returns:
            Batched pixel_values or None if no images
        """
        # Filter out None values
        valid_pixels = [p for p in pixel_values_list if p is not None]

        if not valid_pixels:
            return None

        # Concatenate along batch dimension
        try:
            return mx.concatenate(valid_pixels, axis=0)
        except Exception as e:
            logger.warning(f"Failed to batch pixel_values: {e}")
            # Fall back to returning first valid
            return valid_pixels[0] if valid_pixels else None

    def batch_image_grid_thw(
        self,
        grid_thw_list: list[mx.array | None],
    ) -> mx.array | None:
        """
        Batch multiple image_grid_thw tensors together.

        Args:
            grid_thw_list: List of image_grid_thw from multiple requests

        Returns:
            Batched image_grid_thw or None
        """
        valid_grids = [g for g in grid_thw_list if g is not None]

        if not valid_grids:
            return None

        try:
            return mx.concatenate(valid_grids, axis=0)
        except Exception as e:
            logger.warning(f"Failed to batch image_grid_thw: {e}")
            return valid_grids[0] if valid_grids else None

    def prepare_for_batch(
        self,
        processed_inputs: list[ProcessedMultimodalInput],
    ) -> tuple[mx.array, dict[str, Any], list[int]]:
        """
        Prepare multiple processed inputs for batch generation.

        This method takes a list of ProcessedMultimodalInput objects and
        combines them into batched tensors suitable for the MLLMBatchGenerator.

        Args:
            processed_inputs: List of ProcessedMultimodalInput from process()

        Returns:
            Tuple of:
            - input_ids: Left-padded input tokens [batch_size, max_seq_len]
            - batch_kwargs: Dict with batched pixel_values, attention_mask, etc.
            - padding_amounts: List of padding amounts for each request
        """
        if not processed_inputs:
            return mx.array([]), {}, []

        # Get all input_ids and compute padding
        input_ids_list = [p.input_ids for p in processed_inputs]
        lengths = [ids.size if ids is not None else 0 for ids in input_ids_list]
        max_length = max(lengths) if lengths else 0
        padding_amounts = [max_length - seq_len for seq_len in lengths]

        # Left-pad input_ids
        padded_ids = []
        for ids, pad_amount in zip(input_ids_list, padding_amounts):
            if ids is None:
                padded_ids.append([0] * max_length)
            else:
                tokens = ids.tolist() if hasattr(ids, "tolist") else list(ids)
                padded_ids.append([0] * pad_amount + tokens)

        input_ids = mx.array(padded_ids)

        # Batch pixel values
        pixel_values = self.batch_pixel_values(
            [p.pixel_values for p in processed_inputs]
        )

        # Batch image_grid_thw
        image_grid_thw = self.batch_image_grid_thw(
            [p.image_grid_thw for p in processed_inputs]
        )

        # Batch attention masks
        attention_masks = [p.attention_mask for p in processed_inputs]
        valid_masks = [m for m in attention_masks if m is not None]
        batched_attention_mask = None
        if valid_masks:
            try:
                # Pad attention masks to match input_ids shape
                padded_masks = []
                for mask, pad_amount in zip(attention_masks, padding_amounts):
                    if mask is None:
                        padded_masks.append(mx.ones((max_length,)))
                    else:
                        mask_flat = mask.reshape(-1)
                        pad = mx.zeros((pad_amount,))
                        padded_masks.append(mx.concatenate([pad, mask_flat]))
                batched_attention_mask = mx.stack(padded_masks)
            except Exception as e:
                logger.warning(f"Failed to batch attention masks: {e}")

        # Merge extra_kwargs (take first non-empty set)
        merged_extra = {}
        for p in processed_inputs:
            if p.extra_kwargs:
                merged_extra.update(p.extra_kwargs)
                break

        batch_kwargs = {
            "pixel_values": pixel_values,
            "attention_mask": batched_attention_mask,
            "image_grid_thw": image_grid_thw,
            **merged_extra,
        }

        # Remove None values
        batch_kwargs = {k: v for k, v in batch_kwargs.items() if v is not None}

        return input_ids, batch_kwargs, padding_amounts

    def extract_vision_embeddings(
        self,
        pixel_values: mx.array,
        image_grid_thw: mx.array | None = None,
    ) -> mx.array:
        """
        Extract vision embeddings from pixel values.

        This runs the vision encoder part of the VLM to get embeddings
        that can be cached and reused.

        Args:
            pixel_values: Processed image tensors
            image_grid_thw: Optional grid info for Qwen-VL models

        Returns:
            Vision embeddings tensor
        """
        if not hasattr(self.model, "vision_tower") and not hasattr(
            self.model, "vision_model"
        ):
            raise ValueError("Model does not have a vision encoder")

        # Get the vision encoder
        vision_encoder = getattr(self.model, "vision_tower", None)
        if vision_encoder is None:
            vision_encoder = getattr(self.model, "vision_model", None)

        if vision_encoder is None:
            raise ValueError("Could not find vision encoder in model")

        # Run vision encoding
        if image_grid_thw is not None:
            # Qwen-VL style with grid info
            try:
                embeddings = vision_encoder(pixel_values, grid_thw=image_grid_thw)
            except TypeError:
                embeddings = vision_encoder(pixel_values)
        else:
            embeddings = vision_encoder(pixel_values)

        return embeddings

    def compute_vision_hash(
        self,
        pixel_values: mx.array,
    ) -> str:
        """
        Compute a hash for pixel values for caching purposes.

        Args:
            pixel_values: Processed image tensors

        Returns:
            Hash string for the vision inputs
        """
        import hashlib

        # Use shape and a sample of values for hashing
        shape_str = str(pixel_values.shape)
        sample_values = pixel_values.reshape(-1)[:100].tolist()
        hash_input = f"{shape_str}_{sample_values}"

        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]


def create_mllm_prompt_cache(model: Any, max_kv_size: int | None = None) -> Any:
    """
    Create a prompt cache for VLM generation.

    This wraps mlx_vlm's cache creation for the language model part.

    Args:
        model: The VLM model
        max_kv_size: Optional maximum KV cache size

    Returns:
        Prompt cache object
    """
    from mlx_vlm.models import cache

    return cache.make_prompt_cache(
        model.language_model,
        max_kv_size=max_kv_size,
    )
