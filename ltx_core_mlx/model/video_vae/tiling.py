"""Video VAE tiling system for memory-efficient encoding/decoding.

Splits large video tensors into overlapping tiles, processes each through the VAE
independently, then blends overlapping regions using trapezoidal masks.

Ported from ltx-core/src/ltx_core/model/video_vae/tiling.py
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import mlx.core as mx


def compute_trapezoidal_mask_1d(
    length: int,
    ramp_left: int,
    ramp_right: int,
    left_starts_from_0: bool = False,
) -> mx.array:
    """Generate a 1D trapezoidal blending mask with linear ramps.

    Args:
        length: Output length of the mask.
        ramp_left: Fade-in length on the left.
        ramp_right: Fade-out length on the right.
        left_starts_from_0: Whether the ramp starts from 0 or first non-zero value.
            Useful for temporal tiles where the first tile is causal.

    Returns:
        A 1D array of shape (length,) with values in [0, 1].
    """
    if length <= 0:
        raise ValueError("Mask length must be positive.")

    ramp_left = max(0, min(ramp_left, length))
    ramp_right = max(0, min(ramp_right, length))

    mask = mx.ones(length)

    if ramp_left > 0:
        interval_length = ramp_left + 1 if left_starts_from_0 else ramp_left + 2
        fade_in = mx.linspace(0.0, 1.0, interval_length)[:-1]
        if not left_starts_from_0:
            fade_in = fade_in[1:]
        mask = mx.concatenate([mask[:ramp_left] * fade_in, mask[ramp_left:]])

    if ramp_right > 0:
        fade_out = mx.linspace(1.0, 0.0, ramp_right + 2)[1:-1]
        mask = mx.concatenate([mask[:-ramp_right], mask[-ramp_right:] * fade_out])

    return mx.clip(mask, 0.0, 1.0)


def compute_rectangular_mask_1d(
    length: int,
    left_ramp: int,
    right_ramp: int,
) -> mx.array:
    """Generate a 1D rectangular (pulse) mask.

    Args:
        length: Output length of the mask.
        left_ramp: Number of elements at the start to set to 0.
        right_ramp: Number of elements at the end to set to 0.

    Returns:
        A 1D array of shape (length,) with values 0 or 1.
    """
    if length <= 0:
        raise ValueError("Mask length must be positive.")

    mask = mx.ones(length)
    if left_ramp > 0:
        mask = mx.concatenate([mx.zeros(left_ramp), mask[left_ramp:]])
    if right_ramp > 0:
        mask = mx.concatenate([mask[:-right_ramp], mx.zeros(right_ramp)])
    return mask


@dataclass(frozen=True)
class SpatialTilingConfig:
    """Configuration for dividing each frame into spatial tiles with optional overlap.

    Args:
        tile_size_in_pixels: Size of each tile in pixels. Must be >= 64 and divisible by 32.
        tile_overlap_in_pixels: Overlap between tiles in pixels. Must be divisible by 32.
    """

    tile_size_in_pixels: int
    tile_overlap_in_pixels: int = 0

    def __post_init__(self) -> None:
        if self.tile_size_in_pixels < 64:
            raise ValueError(f"tile_size_in_pixels must be at least 64, got {self.tile_size_in_pixels}")
        if self.tile_size_in_pixels % 32 != 0:
            raise ValueError(f"tile_size_in_pixels must be divisible by 32, got {self.tile_size_in_pixels}")
        if self.tile_overlap_in_pixels % 32 != 0:
            raise ValueError(f"tile_overlap_in_pixels must be divisible by 32, got {self.tile_overlap_in_pixels}")
        if self.tile_overlap_in_pixels >= self.tile_size_in_pixels:
            raise ValueError(
                f"Overlap must be less than tile size, got {self.tile_overlap_in_pixels} and {self.tile_size_in_pixels}"
            )


@dataclass(frozen=True)
class TemporalTilingConfig:
    """Configuration for dividing a video into temporal tiles with optional overlap.

    Args:
        tile_size_in_frames: Number of frames per tile. Must be >= 16 and divisible by 8.
        tile_overlap_in_frames: Overlapping frames between tiles. Must be divisible by 8.
    """

    tile_size_in_frames: int
    tile_overlap_in_frames: int = 0

    def __post_init__(self) -> None:
        if self.tile_size_in_frames < 16:
            raise ValueError(f"tile_size_in_frames must be at least 16, got {self.tile_size_in_frames}")
        if self.tile_size_in_frames % 8 != 0:
            raise ValueError(f"tile_size_in_frames must be divisible by 8, got {self.tile_size_in_frames}")
        if self.tile_overlap_in_frames % 8 != 0:
            raise ValueError(f"tile_overlap_in_frames must be divisible by 8, got {self.tile_overlap_in_frames}")
        if self.tile_overlap_in_frames >= self.tile_size_in_frames:
            raise ValueError(
                f"Overlap must be less than tile size, got {self.tile_overlap_in_frames} and {self.tile_size_in_frames}"
            )


@dataclass(frozen=True)
class TilingConfig:
    """Combined spatial + temporal tiling configuration.

    Attributes:
        spatial_config: Configuration for spatial dimension tiling.
        temporal_config: Configuration for temporal dimension tiling.
    """

    spatial_config: SpatialTilingConfig | None = None
    temporal_config: TemporalTilingConfig | None = None

    @classmethod
    def default(cls) -> TilingConfig:
        """Create a default tiling config (768x768 spatial, 80-frame temporal).

        Defaults bumped to 768/80 in upstream PR #212 (was 512/64) to reduce
        the number of tile boundaries on production resolutions.
        """
        return cls(
            spatial_config=SpatialTilingConfig(tile_size_in_pixels=768, tile_overlap_in_pixels=64),
            temporal_config=TemporalTilingConfig(tile_size_in_frames=80, tile_overlap_in_frames=24),
        )


@dataclass(frozen=True)
class DimensionIntervals:
    """How a single dimension is split into overlapping intervals.

    Each list has length N (number of intervals). The i-th element describes the i-th interval.

    Attributes:
        starts: Start index of each interval (inclusive).
        ends: End index of each interval (exclusive).
        left_ramps: Left blend ramp length per interval.
        right_ramps: Right blend ramp length per interval.
    """

    starts: list[int]
    ends: list[int]
    left_ramps: list[int]
    right_ramps: list[int]


# Type aliases for split and mapping operations.
SplitOperation = Callable[[int], DimensionIntervals]
MappingOperation = Callable[[DimensionIntervals], tuple[list[slice], list[mx.array | None]]]


def default_split_operation(length: int) -> DimensionIntervals:
    """No-op split: single interval covering the full dimension."""
    return DimensionIntervals(starts=[0], ends=[length], left_ramps=[0], right_ramps=[0])


DEFAULT_SPLIT_OPERATION: SplitOperation = default_split_operation


def default_mapping_operation(
    _intervals: DimensionIntervals,
) -> tuple[list[slice], list[mx.array | None]]:
    """No-op mapping: full slice, no mask."""
    return [slice(0, None)], [None]


DEFAULT_MAPPING_OPERATION: MappingOperation = default_mapping_operation


class Tile(NamedTuple):
    """A single tile with input/output coordinates and blend masks.

    Attributes:
        in_coords: Slices for cutting the tile from the INPUT tensor.
        out_coords: Slices for placing the tile's OUTPUT in the reconstructed tensor.
        masks_1d: Per-dimension 1D masks in output units for blending.
    """

    in_coords: tuple[slice, ...]
    out_coords: tuple[slice, ...]
    masks_1d: tuple[mx.array | None, ...]

    @property
    def blend_mask(self) -> mx.array:
        """Create an N-D blending mask from per-dimension 1D masks."""
        num_dims = len(self.out_coords)
        per_dimension_masks: list[mx.array] = []

        for dim_idx in range(num_dims):
            mask_1d = self.masks_1d[dim_idx]
            view_shape = [1] * num_dims
            if mask_1d is None:
                view_shape[dim_idx] = 1
                per_dimension_masks.append(mx.ones(1).reshape(*view_shape))
                continue

            view_shape[dim_idx] = mask_1d.shape[0]
            per_dimension_masks.append(mask_1d.reshape(*view_shape))

        combined_mask = per_dimension_masks[0]
        for mask in per_dimension_masks[1:]:
            combined_mask = combined_mask * mask

        return combined_mask


def create_tiles_from_intervals_and_mappers(
    original_shape: tuple[int, ...],
    dimension_intervals: tuple[DimensionIntervals, ...],
    mappers: list[MappingOperation],
) -> list[Tile]:
    """Create tiles from pre-computed intervals and mapping operations.

    Args:
        original_shape: Shape of the tensor being tiled.
        dimension_intervals: Per-dimension intervals.
        mappers: Per-dimension mapping operations.

    Returns:
        List of Tile objects.
    """
    full_dim_input_slices = []
    full_dim_output_slices = []
    full_dim_masks_1d = []

    for axis_index in range(len(original_shape)):
        dim_intervals = dimension_intervals[axis_index]
        input_slices = [slice(s, e) for s, e in zip(dim_intervals.starts, dim_intervals.ends, strict=True)]
        output_slices, masks_1d = mappers[axis_index](dim_intervals)
        full_dim_input_slices.append(input_slices)
        full_dim_output_slices.append(output_slices)
        full_dim_masks_1d.append(masks_1d)

    tiles = []
    tile_in_coords = list(itertools.product(*full_dim_input_slices))
    tile_out_coords = list(itertools.product(*full_dim_output_slices))
    tile_mask_1ds = list(itertools.product(*full_dim_masks_1d))

    for in_coord, out_coord, mask_1d in zip(tile_in_coords, tile_out_coords, tile_mask_1ds, strict=True):
        tiles.append(Tile(in_coords=in_coord, out_coords=out_coord, masks_1d=mask_1d))

    return tiles


def create_tiles(
    tensor_shape: tuple[int, ...],
    splitters: list[SplitOperation],
    mappers: list[MappingOperation],
) -> list[Tile]:
    """Create tiles for a tensor with the given shape.

    Args:
        tensor_shape: Shape of the tensor to tile.
        splitters: Per-dimension split operations.
        mappers: Per-dimension mapping operations.

    Returns:
        List of Tile objects.
    """
    if len(splitters) != len(tensor_shape):
        raise ValueError(
            f"Number of splitters must match tensor dimensions, got {len(splitters)} and {len(tensor_shape)}"
        )
    if len(mappers) != len(tensor_shape):
        raise ValueError(f"Number of mappers must match tensor dimensions, got {len(mappers)} and {len(tensor_shape)}")
    intervals = tuple(splitter(length) for splitter, length in zip(splitters, tensor_shape, strict=True))
    return create_tiles_from_intervals_and_mappers(tensor_shape, intervals, mappers)


# ---------------------------------------------------------------------------
# Split operations
# ---------------------------------------------------------------------------


def split_with_symmetric_overlaps(size: int, overlap: int) -> SplitOperation:
    """Create a split operation that divides a dimension into overlapping tiles.

    Args:
        size: Tile size.
        overlap: Overlap between adjacent tiles.

    Returns:
        A SplitOperation callable.
    """

    def split(dimension_size: int) -> DimensionIntervals:
        if dimension_size <= size:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        amount = (dimension_size + size - 2 * overlap - 1) // (size - overlap)
        starts = [i * (size - overlap) for i in range(amount)]
        ends = [start + size for start in starts]
        ends[-1] = dimension_size
        left_ramps = [0] + [overlap] * (amount - 1)
        right_ramps = [overlap] * (amount - 1) + [0]
        return DimensionIntervals(starts=starts, ends=ends, left_ramps=left_ramps, right_ramps=right_ramps)

    return split


def split_temporal_latents(size: int, overlap: int) -> SplitOperation:
    """Split temporal axis in latent space with causal handling.

    Non-first tiles are shifted back by 1 and their left ramp extended by 1
    to maintain causal temporal continuity.

    Args:
        size: Tile size in latent steps.
        overlap: Overlap between tiles in latent steps.

    Returns:
        A SplitOperation callable.
    """
    non_causal_split = split_with_symmetric_overlaps(size, overlap)

    def split(dimension_size: int) -> DimensionIntervals:
        if dimension_size <= size:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        intervals = non_causal_split(dimension_size)

        starts = intervals.starts
        starts[1:] = [s - 1 for s in starts[1:]]

        left_ramps = intervals.left_ramps
        left_ramps[1:] = [r + 1 for r in left_ramps[1:]]

        return replace(intervals, starts=starts, left_ramps=left_ramps)

    return split


def split_temporal_frames(tile_size_frames: int, overlap_frames: int) -> SplitOperation:
    """Split temporal axis in video frame space into overlapping tiles.

    Args:
        tile_size_frames: Tile length in frames.
        overlap_frames: Overlap between consecutive tiles in frames.

    Returns:
        A SplitOperation callable.
    """
    non_causal_split = split_with_symmetric_overlaps(tile_size_frames, overlap_frames)

    def split(dimension_size: int) -> DimensionIntervals:
        if dimension_size <= tile_size_frames:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        intervals = non_causal_split(dimension_size)
        ends = intervals.ends
        ends[:-1] = [e + 1 for e in ends[:-1]]
        right_ramps = [0] * len(intervals.right_ramps)
        return replace(intervals, ends=ends, right_ramps=right_ramps)

    return split


# ---------------------------------------------------------------------------
# Mapping operations
# ---------------------------------------------------------------------------


def make_mapping_operation(
    map_func: Callable[[int, int, int, int, int], tuple[slice, mx.array | None]],
    scale: int,
) -> MappingOperation:
    """Create a mapping operation from a per-interval mapping function.

    Args:
        map_func: Function mapping (start, end, left_ramp, right_ramp, scale) to (slice, mask).
        scale: Scale factor passed to map_func.

    Returns:
        A MappingOperation callable.
    """

    def map_op(intervals: DimensionIntervals) -> tuple[list[slice], list[mx.array | None]]:
        output_slices: list[slice] = []
        masks_1d: list[mx.array | None] = []
        for i in range(len(intervals.starts)):
            output_slice, mask_1d = map_func(
                intervals.starts[i],
                intervals.ends[i],
                intervals.left_ramps[i],
                intervals.right_ramps[i],
                scale,
            )
            output_slices.append(output_slice)
            masks_1d.append(mask_1d)
        return output_slices, masks_1d

    return map_op


def map_temporal_interval_to_frame(
    begin: int,
    end: int,
    left_ramp: int,
    right_ramp: int,
    scale: int,
) -> tuple[slice, mx.array]:
    """Map temporal interval in latent space to video frame space.

    Args:
        begin: Start position in latent space.
        end: End position in latent space.
        left_ramp: Left ramp size in latent space.
        right_ramp: Right ramp size in latent space.
        scale: Temporal scale factor.

    Returns:
        Tuple of (output_slice, blend_mask).
    """
    start = begin * scale
    stop = 1 + (end - 1) * scale

    left_ramp_frames = 0 if left_ramp == 0 else 1 + (left_ramp - 1) * scale
    right_ramp_frames = right_ramp * scale

    mask_1d = compute_trapezoidal_mask_1d(stop - start, left_ramp_frames, right_ramp_frames, True)
    return slice(start, stop), mask_1d


def map_temporal_interval_to_latent(
    begin: int,
    end: int,
    left_ramp: int,
    right_ramp: int,
    scale: int,
) -> tuple[slice, mx.array]:
    """Map temporal interval in video frame space to latent space.

    Args:
        begin: Start position in video frame space.
        end: End position in video frame space.
        left_ramp: Left ramp size in video frame space.
        right_ramp: Right ramp size in video frame space.
        scale: Temporal scale factor.

    Returns:
        Tuple of (output_slice, blend_mask).
    """
    start = begin // scale
    stop = (end - 1) // scale + 1

    left_ramp_latents = 0 if left_ramp == 0 else 1 + (left_ramp - 1) // scale
    right_ramp_latents = right_ramp // scale

    if right_ramp_latents != 0:
        raise ValueError("For tiled encoding, temporal tiles are expected to have a right ramp equal to 0")

    mask_1d = compute_rectangular_mask_1d(stop - start, left_ramp_latents, right_ramp_latents)
    return slice(start, stop), mask_1d


def map_spatial_interval_to_pixel(
    begin: int,
    end: int,
    left_ramp: int,
    right_ramp: int,
    scale: int,
) -> tuple[slice, mx.array]:
    """Map spatial interval in latent space to pixel space.

    Args:
        begin: Start position in latent space.
        end: End position in latent space.
        left_ramp: Left ramp size in latent space.
        right_ramp: Right ramp size in latent space.
        scale: Spatial scale factor.

    Returns:
        Tuple of (output_slice, blend_mask).
    """
    start = begin * scale
    stop = end * scale
    mask_1d = compute_trapezoidal_mask_1d(stop - start, left_ramp * scale, right_ramp * scale, False)
    return slice(start, stop), mask_1d


def map_spatial_interval_to_latent(
    begin: int,
    end: int,
    left_ramp: int,
    right_ramp: int,
    scale: int,
) -> tuple[slice, mx.array]:
    """Map spatial interval in pixel space to latent space.

    Args:
        begin: Start position in pixel space.
        end: End position in pixel space.
        left_ramp: Left ramp size in pixel space.
        right_ramp: Right ramp size in pixel space.
        scale: Spatial scale factor.

    Returns:
        Tuple of (output_slice, blend_mask).
    """
    start = begin // scale
    stop = end // scale
    left_ramp_latent = max(0, left_ramp // scale - 1)
    right_ramp_latent = 0 if right_ramp == 0 else 1

    mask_1d = compute_rectangular_mask_1d(stop - start, left_ramp_latent, right_ramp_latent)
    return slice(start, stop), mask_1d


# ---------------------------------------------------------------------------
# High-level tile preparation helpers
# ---------------------------------------------------------------------------

# Video scale factors: temporal=8, spatial_height=32, spatial_width=32
_SCALE_TIME = 8
_SCALE_HEIGHT = 32
_SCALE_WIDTH = 32

_MINIMUM_SPATIAL_OVERLAP_PX = 64
_MINIMUM_TEMPORAL_OVERLAP_FRAMES = 16


def prepare_tiles_for_decoding(
    latent_shape: tuple[int, ...],
    tiling_config: TilingConfig | None = None,
) -> list[Tile]:
    """Prepare tiles for VAE decoding (latent -> pixels).

    Operates on latent tensor shape (B, C, F', H', W') in PyTorch layout.

    Args:
        latent_shape: Shape of the latent tensor (B, C, F', H', W').
        tiling_config: Tiling configuration.

    Returns:
        List of tiles.
    """
    ndim = len(latent_shape)
    splitters: list[SplitOperation] = [DEFAULT_SPLIT_OPERATION] * ndim
    mappers: list[MappingOperation] = [DEFAULT_MAPPING_OPERATION] * ndim

    if tiling_config is not None and tiling_config.spatial_config is not None:
        cfg = tiling_config.spatial_config
        long_side = max(latent_shape[3], latent_shape[4])

        def _enable_spatial_axis(axis_idx: int, factor: int) -> None:
            tile_size = cfg.tile_size_in_pixels // factor
            overlap = cfg.tile_overlap_in_pixels // factor
            axis_length = latent_shape[axis_idx]
            lower_threshold = max(2, overlap + 1)
            adjusted_size = max(lower_threshold, round(tile_size * axis_length / long_side))
            splitters[axis_idx] = split_with_symmetric_overlaps(adjusted_size, overlap)
            mappers[axis_idx] = make_mapping_operation(map_spatial_interval_to_pixel, scale=factor)

        _enable_spatial_axis(3, _SCALE_HEIGHT)
        _enable_spatial_axis(4, _SCALE_WIDTH)

    if tiling_config is not None and tiling_config.temporal_config is not None:
        cfg = tiling_config.temporal_config
        tile_size = cfg.tile_size_in_frames // _SCALE_TIME
        overlap = cfg.tile_overlap_in_frames // _SCALE_TIME
        splitters[2] = split_temporal_latents(tile_size, overlap)
        mappers[2] = make_mapping_operation(map_temporal_interval_to_frame, scale=_SCALE_TIME)

    return create_tiles(latent_shape, splitters, mappers)


def prepare_tiles_for_encoding(
    video_shape: tuple[int, ...],
    tiling_config: TilingConfig | None = None,
) -> list[Tile]:
    """Prepare tiles for VAE encoding (pixels -> latent).

    Operates on video tensor shape (B, 3, F, H, W) in PyTorch layout.

    Args:
        video_shape: Shape of the video tensor (B, 3, F, H, W).
        tiling_config: Tiling configuration.

    Returns:
        List of tiles.
    """
    ndim = len(video_shape)
    splitters: list[SplitOperation] = [DEFAULT_SPLIT_OPERATION] * ndim
    mappers: list[MappingOperation] = [DEFAULT_MAPPING_OPERATION] * ndim

    if tiling_config is not None and tiling_config.spatial_config is not None:
        cfg = tiling_config.spatial_config
        tile_size_px = cfg.tile_size_in_pixels
        overlap_px = cfg.tile_overlap_in_pixels

        if overlap_px < _MINIMUM_SPATIAL_OVERLAP_PX:
            overlap_px = _MINIMUM_SPATIAL_OVERLAP_PX

        # Height (axis 3) and Width (axis 4)
        splitters[3] = split_with_symmetric_overlaps(tile_size_px, overlap_px)
        mappers[3] = make_mapping_operation(map_spatial_interval_to_latent, scale=_SCALE_HEIGHT)

        splitters[4] = split_with_symmetric_overlaps(tile_size_px, overlap_px)
        mappers[4] = make_mapping_operation(map_spatial_interval_to_latent, scale=_SCALE_WIDTH)

    if tiling_config is not None and tiling_config.temporal_config is not None:
        cfg = tiling_config.temporal_config
        tile_size_frames = cfg.tile_size_in_frames
        overlap_frames = cfg.tile_overlap_in_frames

        if overlap_frames < _MINIMUM_TEMPORAL_OVERLAP_FRAMES:
            overlap_frames = _MINIMUM_TEMPORAL_OVERLAP_FRAMES

        splitters[2] = split_temporal_frames(tile_size_frames, overlap_frames)
        mappers[2] = make_mapping_operation(map_temporal_interval_to_latent, scale=_SCALE_TIME)

    return create_tiles(video_shape, splitters, mappers)


# ---------------------------------------------------------------------------
# Token-grid tiling primitives (used by VideoModalityTiler)
# ---------------------------------------------------------------------------
# These are MLX-native ports of the upstream `split_by_size`,
# `split_by_count`, `identity_mapping_operation`, `DimensionTilingConfig`,
# and `TileCountConfig` from `ltx_core.tiling`. They operate on the
# patchified token grid (F, H, W) consumed by the DiT, not the VAE pixel
# grid. The grid units match `compute_video_latent_shape` output.
#
# Naming: ``split_by_size`` is an alias for the existing
# ``split_with_symmetric_overlaps``. ``split_by_count`` delegates to it
# after computing the implied tile size.


# Alias that matches upstream naming.
split_by_size = split_with_symmetric_overlaps


def split_by_count(num_tiles: int, overlap: int = 0) -> SplitOperation:
    """Split a dimension into ``num_tiles`` tiles with ``overlap`` shared elements.

    Computes the tile size as ``(dim_size + overlap * (num_tiles - 1)) // num_tiles``
    so that ``num_tiles`` tiles of that size with the given overlap cover the
    dimension evenly. When the total isn't evenly divisible by ``num_tiles``,
    the first ``remainder`` tiles each absorb one extra unit.

    Args:
        num_tiles: Number of tiles. Must be >= 1.
        overlap: Overlap between adjacent tiles. Must be >= 0 and
            less than the computed tile size.

    Returns:
        A SplitOperation callable.
    """
    if num_tiles < 1:
        raise ValueError(f"num_tiles must be >= 1, got {num_tiles}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")

    def split(dim_size: int) -> DimensionIntervals:
        if num_tiles > dim_size:
            raise ValueError(
                f"num_tiles ({num_tiles}) exceeds dim_size ({dim_size}). Cannot assign at least 1 unit per tile."
            )
        if num_tiles == 1:
            return DEFAULT_SPLIT_OPERATION(dim_size)

        total = dim_size + overlap * (num_tiles - 1)
        tile_size = total // num_tiles
        remainder = total % num_tiles

        # Use split_by_size on the rounded-down dimension, then absorb
        # the ``remainder`` extra units into the first ``remainder`` tiles.
        base = split_by_size(tile_size, overlap)(dim_size - remainder)

        starts: list[int] = []
        ends: list[int] = []
        for i, (s, e) in enumerate(zip(base.starts, base.ends, strict=True)):
            shift = min(i, remainder)
            grow = 1 if i < remainder else 0
            starts.append(s + shift)
            ends.append(e + shift + grow)

        return DimensionIntervals(
            starts=starts,
            ends=ends,
            left_ramps=list(base.left_ramps),
            right_ramps=list(base.right_ramps),
        )

    return split


def identity_mapping_operation(
    intervals: DimensionIntervals,
) -> tuple[list[slice], list[mx.array | None]]:
    """Each interval maps to the same output range with a trapezoidal blend mask.

    Mirrors upstream ``ltx_core.tiling.identity_mapping_operation``. Used
    when tile coordinates in the input grid match coordinates in the
    output grid (the typical case for token-level modality tiling).
    """
    out_slices: list[slice] = []
    masks: list[mx.array | None] = []
    for s, e, lr, rr in zip(intervals.starts, intervals.ends, intervals.left_ramps, intervals.right_ramps, strict=True):
        out_slices.append(slice(s, e))
        masks.append(compute_trapezoidal_mask_1d(e - s, lr, rr))
    return out_slices, masks


@dataclass(frozen=True)
class DimensionTilingConfig:
    """Tiling parameters for a single dimension of the patchified token grid.

    Attributes:
        num_tiles: Number of tiles along this dimension.
        overlap: Overlap between adjacent tiles, in token-grid units.
    """

    num_tiles: int = 1
    overlap: int = 0

    def __post_init__(self) -> None:
        if self.num_tiles < 1:
            raise ValueError(f"num_tiles must be >= 1, got {self.num_tiles}")
        if self.overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {self.overlap}")


@dataclass(frozen=True)
class TileCountConfig:
    """Token-grid tiling layout for a (F, H, W) shape.

    Specifies tile counts per dimension (as opposed to tile sizes used
    by the pixel-grid VAE :class:`TilingConfig`). Used by
    :class:`~ltx_core_mlx.components.modality_tiling.VideoModalityTiler`
    to split the DiT input tokens into spatial/temporal regions.
    """

    frames: DimensionTilingConfig = field(default_factory=DimensionTilingConfig)
    height: DimensionTilingConfig = field(default_factory=DimensionTilingConfig)
    width: DimensionTilingConfig = field(default_factory=DimensionTilingConfig)
