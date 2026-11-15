"""Hand-derived fixtures for the official reference implementation's layout math."""

import numpy as np
import pytest

from rapid_mlx.image.minimax_h3_image.layout import (
    _frame_position_grid,
    _spatial_position_grid,
    bound_short_edge,
    build_image_packed_sequence,
    get_image_canvas_size,
    row_timestep_plan,
    validate_canvas,
    validate_request,
)


def test_canvas_sizes():
    assert get_image_canvas_size(1024, 1024) == (2048, 2048)
    assert get_image_canvas_size(4096, 4096) == (4096, 4096)
    # sqrt(4194304 / 1572864) = sqrt(8/3): raw sides 1672.185, 2508.277.
    assert get_image_canvas_size(1024, 1536) == (1664, 2496)
    assert get_image_canvas_size(1024, 1536, 1024) == (1024, 1536)
    assert get_image_canvas_size(33, 65, 1024) == (32, 64)


@pytest.mark.parametrize(
    "size,bounds,expected",
    [
        ((512, 768), (1024, 2048), (1024, 1536)),
        ((3072, 4096), (1024, 2048), (2048, 2720)),
        ((1201, 1803), (1024, 2048), (1201, 1803)),
        ((2048, 3072), (1024, 1024), (1024, 1536)),
        ((1024, 2048), (1024, 1024), (1024, 2048)),
    ],
)
def test_bound_short_edge(size, bounds, expected):
    assert bound_short_edge(*size, *bounds, 32) == expected


def test_square_and_aspect_normalized_grid():
    np.testing.assert_array_equal(_spatial_position_grid(8, 2, 8), [0, 8, 16, 24])
    # area=64, sqrt=8; h ratio=1/2 -> left=1/4; w ratio=2 -> left=-1/2.
    grid, width = _frame_position_grid(4, 16, 2, 2)
    np.testing.assert_array_equal(width, [-16, -8, 0, 8, 16, 24, 32, 40])
    np.testing.assert_array_equal(grid[:8, 0], [8] * 8)
    np.testing.assert_array_equal(grid[8:, 0], [16] * 8)
    assert grid.dtype == np.float64


def test_t2i_sequence():
    pos, tags, images, text, refs = build_image_packed_sequence(np.ones(3), [], 4, 4)
    assert pos.shape == (7, 3)
    assert pos.dtype == np.float64
    assert refs == 0
    np.testing.assert_array_equal(tags, [1, 1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(images, [3, 4, 5, 6])
    np.testing.assert_array_equal(text, [0, 1, 2])
    np.testing.assert_array_equal(
        pos,
        [
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [3, 0, 0],
            [3, 0, 16],
            [3, 16, 0],
            [3, 16, 16],
        ],
    )


def test_two_reference_edit_sequence():
    pos, tags, images, text, refs = build_image_packed_sequence(
        np.array([1, 0, 0, 1]), [(2, 2), (4, 4)], 4, 4
    )
    assert refs == 5
    assert pos.shape == (13, 3)
    np.testing.assert_array_equal(pos[:, 0], [0, 1, 2, 3, 4, 5, 5, 5, 5, 6, 6, 6, 6])
    np.testing.assert_array_equal(
        pos[4:],
        [
            [4, 0, 0],
            [5, 0, 0],
            [5, 0, 16],
            [5, 16, 0],
            [5, 16, 16],
            [6, 0, 0],
            [6, 0, 16],
            [6, 16, 0],
            [6, 16, 16],
        ],
    )
    np.testing.assert_array_equal(tags, [1, 0, 0, 1] + [0] * 9)
    np.testing.assert_array_equal(text, [0, 1, 2, 3])
    np.testing.assert_array_equal(images, list(range(4, 13)))


def test_row_timestep_plan():
    plan = row_timestep_plan(7, np.arange(3, 7), 2)
    assert len(plan) == 6
    # All text rows (including vision text rows) follow the target clock.
    np.testing.assert_array_equal(plan[0][0], np.array([0, 0.999], np.float32))
    np.testing.assert_array_equal(plan[0][1], [0, 0, 0, 1, 1, 0, 0])
    np.testing.assert_allclose(
        plan[-1][0], [0.7679911255836487, 0.999], rtol=0, atol=3e-8
    )
    single = row_timestep_plan(7, np.arange(3, 7), 0)
    np.testing.assert_array_equal(single[0][0], [0])
    np.testing.assert_array_equal(single[0][1], np.zeros(7))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"width": 0}, "positive"),
        ({"height": -2}, "positive"),
        ({"min_generation_area": 0}, "min_generation_area"),
        ({"min_generation_area": 16777217}, "min_generation_area"),
        ({"num_inference_steps": 5}, "steps must be 6"),
        ({"guidance_scale": 2}, "guidance must be 1"),
        ({"negative_prompt": "bad"}, "negative prompt"),
        ({"num_references": 17}, "16 references"),
    ],
)
def test_validation_errors(kwargs, match):
    args = {"width": 1024, "height": 1024, **kwargs}
    with pytest.raises(ValueError, match=match):
        validate_request(**args)


def test_invalid_canvas_and_grid():
    with pytest.raises(ValueError, match="multiples of 32"):
        validate_canvas(1023, 1024)
    with pytest.raises(ValueError, match="patch size"):
        build_image_packed_sequence(np.ones(2), [], 3, 4)
    with pytest.raises(ValueError, match="tags"):
        build_image_packed_sequence(np.array([2]), [], 4, 4)
