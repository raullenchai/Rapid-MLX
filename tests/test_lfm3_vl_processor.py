import numpy as np
import pytest
from PIL import Image

from rapid_mlx.models.lfm3_vl_processor import (
    IMAGE_TOKEN_ID,
    VIDEO_TOKEN_ID,
    expand_image_tokens,
    preprocess_images,
    smart_resize,
)


def test_patch_order_and_temporal_repetition():
    a = np.zeros((32, 32, 3), dtype=np.uint8)
    a[:16, :16] = 10
    a[:16, 16:] = 20
    a[16:, :16] = 30
    a[16:, 16:] = 40
    patches, grid = preprocess_images([Image.fromarray(a)], min_pixels=1024, max_pixels=1024)
    assert grid.tolist() == [[1, 2, 2]]
    assert patches.shape == (4, 1536)
    for patch, value in zip(patches, (10, 20, 30, 40)):
        np.testing.assert_allclose(patch, value / 127.5 - 1, atol=1e-6)


def test_image_expansion_and_validation():
    assert expand_image_tokens([1, IMAGE_TOKEN_ID, 2], [(1, 4, 4)]) == [1] + [IMAGE_TOKEN_ID] * 4 + [2]
    with pytest.raises(ValueError, match="More image markers"):
        expand_image_tokens([IMAGE_TOKEN_ID], [])
    with pytest.raises(ValueError, match="More images"):
        expand_image_tokens([], [(1, 2, 2)])
    with pytest.raises(ValueError, match="Video"):
        expand_image_tokens([VIDEO_TOKEN_ID], [])


def test_resize_and_budget():
    assert smart_resize(256, 256) == (256, 256)
    assert smart_resize(512, 1024) == (512, 1024)
    with pytest.raises(ValueError, match="aspect ratio"):
        smart_resize(1, 201)
