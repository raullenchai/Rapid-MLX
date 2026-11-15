"""Pure host layout math from the official reference implementation.

The reference's ``reference_noise_aug`` is 0.999, not 0: its clock runs
from noisy t=0 towards clean t=1. Text rows follow the generated timestep.
"""

from __future__ import annotations

import math

import numpy as np

CANVAS_MULTIPLE = 32
VAE_RATIO = 16
PATCH_SIZE = (1, 2, 2)
MIN_GENERATION_AREA = 2048 * 2048
MIN_AREA_BOUNDS = (1024, 4096 * 4096)
REFERENCE_SHORT_EDGE_BOUNDS = (1024, 2048)
VISION_SHORT_EDGE_BOUNDS = (1024, 1024)
MAX_PROMPT_TOKENS = 4096
MAX_REFERENCES = 16
NUM_STEPS = 6
REFERENCE_TIMESTEP = 0.999
SIGMAS = (
    1.0,
    0.9675236940383911,
    0.8875485062599182,
    0.75,
    0.5327739119529724,
    0.23200887441635132,
    0.0,
)
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)


def validate_request(
    width: int,
    height: int,
    min_generation_area: int = MIN_GENERATION_AREA,
    num_inference_steps: int = NUM_STEPS,
    guidance_scale: float = 1.0,
    negative_prompt: str | None = None,
    num_references: int = 0,
) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if not MIN_AREA_BOUNDS[0] <= min_generation_area <= MIN_AREA_BOUNDS[1]:
        raise ValueError("min_generation_area must be within [1024, 16777216]")
    if num_inference_steps != NUM_STEPS:
        raise ValueError("MiniMax-H3-Image steps must be 6")
    if guidance_scale != 1.0:
        raise ValueError("MiniMax-H3-Image guidance must be 1")
    if negative_prompt:
        raise ValueError("MiniMax-H3-Image does not support a negative prompt")
    if not 0 <= num_references <= MAX_REFERENCES:
        raise ValueError("MiniMax-H3-Image supports at most 16 references")


def validate_canvas(height: int, width: int, multiple: int = CANVAS_MULTIPLE) -> None:
    if min(height, width) <= 0 or height % multiple or width % multiple:
        raise ValueError(f"canvas size must be positive multiples of {multiple}")


def get_image_canvas_size(
    height: int,
    width: int,
    min_generation_area: int = MIN_GENERATION_AREA,
    multiple: int = CANVAS_MULTIPLE,
) -> tuple[int, int]:
    validate_request(width, height, min_generation_area)
    scale = max(1.0, math.sqrt(min_generation_area / (width * height)))
    return (
        max(multiple, round(height * scale / multiple) * multiple),
        max(multiple, round(width * scale / multiple) * multiple),
    )


def bound_short_edge(
    width: int,
    height: int,
    min_short_edge: int,
    max_short_edge: int,
    multiple: int = CANVAS_MULTIPLE,
) -> tuple[int, int]:
    if (
        min(width, height, multiple, min_short_edge) <= 0
        or max_short_edge < min_short_edge
    ):
        raise ValueError(
            "dimensions and short-edge bounds must be positive and ordered"
        )
    short_edge = min(width, height)
    if min_short_edge <= short_edge <= max_short_edge:
        return width, height
    scale = (
        min_short_edge if short_edge < min_short_edge else max_short_edge
    ) / short_edge
    return (
        max(multiple, round(width * scale / multiple) * multiple),
        max(multiple, round(height * scale / multiple) * multiple),
    )


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> np.ndarray:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    return np.linspace(left, left + ratio, dim // patch, endpoint=False) * 32


def _frame_position_grid(
    latent_height: int, latent_width: int, patch_h: int, patch_w: int
) -> tuple[np.ndarray, np.ndarray]:
    sqrt_area = np.sqrt(latent_height * latent_width)
    h = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    w = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    return np.stack(
        [g.reshape(-1) for g in np.meshgrid(h, w, indexing="ij")], axis=-1
    ), w


def build_image_packed_sequence(
    text_token_tags: np.ndarray,
    reference_grids: list[tuple[int, int]],
    latent_height: int,
    latent_width: int,
    patch_size: tuple[int, int, int] = PATCH_SIZE,
    image_tag: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    text_token_tags = np.asarray(text_token_tags, dtype=np.int64)
    if text_token_tags.ndim != 1 or not np.isin(text_token_tags, [0, 1]).all():
        raise ValueError("text_token_tags must be a vector of image/text tags 0 or 1")
    if len(reference_grids) > MAX_REFERENCES:
        raise ValueError("at most 16 references are supported")
    _, ph, pw = patch_size
    grids = [*reference_grids, (latent_height, latent_width)]
    if patch_size[0] != 1 or min(ph, pw) <= 0:
        raise ValueError(
            "image patch must have temporal size 1 and positive spatial sizes"
        )
    if any(min(h, w) <= 0 or h % ph or w % pw for h, w in grids):
        raise ValueError(
            "latent dimensions must be positive multiples of the patch size"
        )
    n = len(text_token_tags)
    counts = [(h // ph) * (w // pw) for h, w in grids]
    positions = np.zeros((n + sum(counts), 3), dtype=np.float64)
    positions[:n, 0] = np.arange(n)
    cursor = n
    for i, ((h, w), count) in enumerate(zip(grids, counts, strict=True)):
        positions[cursor : cursor + count, 0] = n + i
        positions[cursor : cursor + count, 1:] = _frame_position_grid(h, w, ph, pw)[0]
        cursor += count
    tags = np.concatenate(
        [text_token_tags, np.full(sum(counts), image_tag, dtype=np.int64)]
    )
    return positions, tags, np.arange(n, cursor), np.arange(n), sum(counts[:-1])


def row_timestep_plan(
    sequence_length: int,
    image_indices: np.ndarray,
    num_reference_rows: int,
    sigmas: tuple[float, ...] = SIGMAS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    plan = []
    # The pipeline builds t, reconstructs sigma, then the scheduler rebuilds t.
    timesteps = np.asarray([1.0 - sigma for sigma in sigmas[:-1]], dtype=np.float32)
    timesteps = np.float32(1.0) - (np.float32(1.0) - timesteps)
    for t in timesteps:
        rows = np.full(sequence_length, t, dtype=np.float32)
        rows[image_indices[:num_reference_rows]] = REFERENCE_TIMESTEP
        values, indices = np.unique(rows, return_inverse=True)
        plan.append((values, indices.astype(np.int32)))
    return plan
