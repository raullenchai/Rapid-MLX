# SPDX-License-Identifier: Apache-2.0
"""
Tests for MLLM (Multimodal Language Model) KV cache functionality.

These tests verify the MLLMCacheManager for caching KV states
when the same image/video + prompt combination is requested.
"""

import os
import tempfile
from pathlib import Path

import pytest

from rapid_mlx.mllm_cache import (
    MLLMCacheStats,
    MLLMPrefixCacheEntry,
    MLLMPrefixCacheManager,
    compute_image_hash,
    compute_images_hash,
)

# Aliases for test compatibility
MLLMCacheEntry = MLLMPrefixCacheEntry
MLLMCacheManager = MLLMPrefixCacheManager


class TestMLLMCacheStats:
    """Tests for MLLMCacheStats class."""

    def test_initial_stats(self):
        """Test initial statistics are zero."""
        stats = MLLMCacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.tokens_saved == 0
        assert stats.image_cache_hits == 0
        assert stats.total_queries == 0
        assert stats.evictions == 0
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = MLLMCacheStats(hits=3, misses=7, total_queries=10)
        assert stats.hit_rate == 0.3

    def test_hit_rate_zero_queries(self):
        """Test hit rate with zero queries."""
        stats = MLLMCacheStats()
        assert stats.hit_rate == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = MLLMCacheStats(
            hits=5,
            misses=5,
            tokens_saved=100,
            image_cache_hits=3,
            total_queries=10,
            evictions=2,
        )
        d = stats.to_dict()
        assert d["hits"] == 5
        assert d["misses"] == 5
        assert d["hit_rate"] == 0.5
        assert d["tokens_saved"] == 100
        assert d["image_cache_hits"] == 3
        assert d["total_queries"] == 10
        assert d["evictions"] == 2


class TestImageHashing:
    """Tests for image hashing functions."""

    def test_compute_image_hash_file(self):
        """Test hashing a real image file."""
        # Create a temp file with some content
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake image content for testing")
            temp_path = f.name

        try:
            hash1 = compute_image_hash(temp_path)
            hash2 = compute_image_hash(temp_path)

            # Same file should give same hash
            assert hash1 == hash2
            assert len(hash1) == 16  # First 16 chars of SHA256
        finally:
            os.unlink(temp_path)

    def test_compute_image_hash_different_content(self):
        """Test that different content gives different hash."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1:
            f1.write(b"content 1")
            path1 = f1.name

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
            f2.write(b"content 2")
            path2 = f2.name

        try:
            hash1 = compute_image_hash(path1)
            hash2 = compute_image_hash(path2)
            assert hash1 != hash2
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_compute_image_hash_url(self):
        """Test hashing a URL (non-existent file path)."""
        url = "https://example.com/image.jpg"
        hash1 = compute_image_hash(url)
        hash2 = compute_image_hash(url)

        assert hash1 == hash2
        assert len(hash1) == 16

    def test_compute_images_hash_empty(self):
        """Test hashing empty image list."""
        result = compute_images_hash([])
        assert result == "no_images"

    def test_compute_images_hash_single(self):
        """Test hashing single image."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"test content")
            path = f.name

        try:
            hash_single = compute_images_hash([path])
            assert len(hash_single) == 16
        finally:
            os.unlink(path)

    def test_compute_images_hash_multiple(self):
        """Test hashing multiple images."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1:
            f1.write(b"image 1")
            path1 = f1.name

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
            f2.write(b"image 2")
            path2 = f2.name

        try:
            # Order matters: image sequence affects vision embeddings/KV state
            hash_a = compute_images_hash([path1, path2])
            hash_b = compute_images_hash([path2, path1])
            assert hash_a != hash_b, (
                "Different image orders must produce different hashes"
            )
            # Same order is deterministic
            hash_c = compute_images_hash([path1, path2])
            assert hash_a == hash_c
        finally:
            os.unlink(path1)
            os.unlink(path2)


class TestMLLMCacheEntry:
    """Tests for MLLMPrefixCacheEntry class."""

    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        entry = MLLMPrefixCacheEntry(
            image_hash="abc123",
            prompt_hash="def456",
            kv_cache=["mock_kv_cache"],
            prompt_tokens=50,
        )
        assert entry.kv_cache == ["mock_kv_cache"]
        assert entry.image_hash == "abc123"
        assert entry.prompt_hash == "def456"
        assert entry.prompt_tokens == 50
        assert entry.hit_count == 0

    def test_cache_entry_hit_count_increment(self):
        """Test incrementing hit count."""
        entry = MLLMPrefixCacheEntry(
            image_hash="xyz",
            prompt_hash="abc",
            kv_cache=["cache"],
            prompt_tokens=10,
        )
        entry.hit_count += 1
        assert entry.hit_count == 1


class TestMLLMCacheManager:
    """Tests for MLLMCacheManager class."""

    @pytest.fixture
    def cache_manager(self):
        """Create a cache manager with default settings."""
        return MLLMCacheManager(max_entries=10)

    @pytest.fixture
    def temp_image(self):
        """Create a temporary image file."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"test image content")
            path = f.name
        yield path
        os.unlink(path)

    def test_initialization(self):
        """Test cache manager initialization."""
        manager = MLLMCacheManager(max_entries=50)
        assert manager.max_size == 50
        assert len(manager) == 0

    def test_fetch_empty_cache(self, cache_manager):
        """Test fetching from empty cache returns miss."""
        cache, hit = cache_manager.fetch_cache(["image.jpg"], "Describe this")

        assert cache is None
        assert hit is False
        assert cache_manager.stats.misses == 1
        assert cache_manager.stats.hits == 0

    def test_store_and_fetch_exact_match(self, cache_manager, temp_image):
        """Test storing and fetching exact match."""
        images = [temp_image]
        prompt = "Describe this image"
        mock_cache = ["kv_layer_1", "kv_layer_2"]

        # Store cache
        cache_manager.store_cache(images, prompt, mock_cache, num_tokens=100)
        assert len(cache_manager) == 1

        # Fetch exact match
        cache, hit = cache_manager.fetch_cache(images, prompt)

        assert cache is not None
        assert hit is True
        assert cache_manager.stats.hits == 1
        assert cache_manager.stats.tokens_saved == 100

    def test_different_prompt_different_cache(self, cache_manager, temp_image):
        """Test that different prompts get different cache entries."""
        images = [temp_image]

        # Store with first prompt
        cache_manager.store_cache(images, "Describe this", ["cache1"], num_tokens=50)

        # Fetch with different prompt
        cache, hit = cache_manager.fetch_cache(images, "What is in this image?")

        assert cache is None
        assert hit is False
        assert cache_manager.stats.misses == 1

    def test_different_image_different_cache(self, cache_manager):
        """Test that different images get different cache entries."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1:
            f1.write(b"image 1 content")
            img1 = f1.name

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
            f2.write(b"image 2 content")
            img2 = f2.name

        try:
            prompt = "Describe this"

            # Store with first image
            cache_manager.store_cache([img1], prompt, ["cache1"], num_tokens=50)

            # Fetch with different image - should miss
            cache, hit = cache_manager.fetch_cache([img2], prompt)
            assert cache is None
            assert hit is False
        finally:
            os.unlink(img1)
            os.unlink(img2)

    def test_video_cache_key(self, cache_manager):
        """Test that video parameters affect cache key."""
        video_source_1 = "video:test.mp4:fps2.0:max32"
        video_source_2 = "video:test.mp4:fps1.0:max64"
        prompt = "Describe this video"

        # Store with first video params
        cache_manager.store_cache([video_source_1], prompt, ["cache1"], num_tokens=100)

        # Fetch with same params - should hit
        cache, hit = cache_manager.fetch_cache([video_source_1], prompt)
        assert hit is True

        # Fetch with different params - should miss
        cache, hit = cache_manager.fetch_cache([video_source_2], prompt)
        assert hit is False

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        manager = MLLMCacheManager(max_entries=3)

        # Fill cache
        manager.store_cache(["img1.jpg"], "prompt1", ["cache1"])
        manager.store_cache(["img2.jpg"], "prompt2", ["cache2"])
        manager.store_cache(["img3.jpg"], "prompt3", ["cache3"])
        assert len(manager) == 3

        # Add one more - should evict oldest
        manager.store_cache(["img4.jpg"], "prompt4", ["cache4"])
        assert len(manager) == 3
        assert manager.stats.evictions == 1

        # img1 should be evicted
        cache, hit = manager.fetch_cache(["img1.jpg"], "prompt1")
        assert cache is None
        assert hit is False

    def test_lru_touch_on_access(self):
        """Test that accessing a cache updates LRU order."""
        manager = MLLMCacheManager(max_entries=3)

        # Fill cache
        manager.store_cache(["img1.jpg"], "p1", ["cache1"])
        manager.store_cache(["img2.jpg"], "p2", ["cache2"])
        manager.store_cache(["img3.jpg"], "p3", ["cache3"])

        # Access img1 to make it most recently used
        manager.fetch_cache(["img1.jpg"], "p1")

        # Add new entry - should evict img2 (oldest untouched)
        manager.store_cache(["img4.jpg"], "p4", ["cache4"])

        # img1 should still be there
        cache, hit = manager.fetch_cache(["img1.jpg"], "p1")
        assert hit is True

        # img2 should be evicted
        cache, hit = manager.fetch_cache(["img2.jpg"], "p2")
        assert hit is False

    def test_store_empty_cache(self, cache_manager):
        """Test that empty cache is not stored."""
        cache_manager.store_cache(["img.jpg"], "prompt", [])
        assert len(cache_manager) == 0

    def test_store_none_cache(self, cache_manager):
        """Test that None cache is not stored."""
        cache_manager.store_cache(["img.jpg"], "prompt", None)
        assert len(cache_manager) == 0

    def test_get_stats(self, cache_manager, temp_image):
        """Test getting statistics."""
        # Generate some activity
        cache_manager.store_cache([temp_image], "Describe", ["cache1"], num_tokens=50)
        cache_manager.fetch_cache([temp_image], "Describe")  # Hit
        cache_manager.fetch_cache(["other.jpg"], "Describe")  # Miss

        stats = cache_manager.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["total_queries"] == 2
        assert stats["image_cache_hits"] == 1

    def test_reset_stats(self, cache_manager):
        """Test resetting statistics."""
        cache_manager.stats.hits = 10
        cache_manager.stats.misses = 5
        cache_manager.reset_stats()

        assert cache_manager.stats.hits == 0
        assert cache_manager.stats.misses == 0

    def test_clear(self, cache_manager, temp_image):
        """Test clearing the cache."""
        cache_manager.store_cache([temp_image], "p1", ["cache1"])
        cache_manager.store_cache(["img2.jpg"], "p2", ["cache2"])
        assert len(cache_manager) == 2

        cache_manager.clear()
        assert len(cache_manager) == 0

        # Stats should also be reset
        assert cache_manager.stats.hits == 0

    def test_cache_returns_deep_copy(self, cache_manager, temp_image):
        """Test that fetched cache returns a deep copy to prevent mutation corruption."""
        original = [[1, 2, 3]]
        cache_manager.store_cache([temp_image], "prompt", original)

        cache, hit = cache_manager.fetch_cache([temp_image], "prompt")
        assert hit is True
        assert cache is not None
        assert cache[0] == [1, 2, 3]

        # Each fetch returns an independent deep copy
        cache2, _ = cache_manager.fetch_cache([temp_image], "prompt")
        assert cache2 is not cache  # Different objects

    def test_repr(self, cache_manager):
        """Test string representation."""
        repr_str = repr(cache_manager)
        assert "MLLMPrefixCacheManager" in repr_str
        assert "entries=0" in repr_str
        assert "memory=" in repr_str

    def test_multi_image_cache(self, cache_manager):
        """Test caching with multiple images."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1:
            f1.write(b"img1")
            img1 = f1.name

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
            f2.write(b"img2")
            img2 = f2.name

        try:
            prompt = "Compare these images"
            images = [img1, img2]

            # Store
            cache_manager.store_cache(images, prompt, ["multi_cache"], num_tokens=200)

            # Fetch same images in same order — should hit
            cache, hit = cache_manager.fetch_cache(images, prompt)
            assert hit is True

            # Fetch same images in different order — must MISS
            # (image order affects vision embeddings and KV state)
            cache, hit = cache_manager.fetch_cache([img2, img1], prompt)
            assert hit is False, "Different image order should produce cache miss"
        finally:
            os.unlink(img1)
            os.unlink(img2)


class TestMLXMultimodalLMCache:
    """Tests for cache integration with MLXMultimodalLM."""

    def test_mllm_cache_enabled_by_default(self):
        """Test that cache is enabled by default in MLXMultimodalLM."""
        from rapid_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM("test-model")
        assert model.enable_cache is True
        assert model._cache_manager is not None

    def test_mllm_cache_disabled(self):
        """Test disabling cache in MLXMultimodalLM."""
        from rapid_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM("test-model", enable_cache=False)
        assert model.enable_cache is False
        assert model._cache_manager is None

    def test_mllm_cache_custom_size(self):
        """Test custom cache size in MLXMultimodalLM."""
        from rapid_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM("test-model", cache_size=100)
        assert model._cache_manager.max_size == 100

    def test_mllm_get_cache_stats_disabled(self):
        """Test get_cache_stats when cache is disabled."""
        from rapid_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM("test-model", enable_cache=False)
        stats = model.get_cache_stats()
        assert stats["enabled"] is False

    def test_mllm_get_cache_stats_enabled(self):
        """Test get_cache_stats when cache is enabled."""
        from rapid_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM("test-model", enable_cache=True)
        stats = model.get_cache_stats()
        assert stats["enabled"] is True
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "cache_entries" in stats
        assert "max_entries" in stats

    def test_mllm_clear_cache(self):
        """Test clearing cache in MLXMultimodalLM."""
        from rapid_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM("test-model", enable_cache=True)

        # Add some entries manually
        model._cache_manager.store_cache(["img.jpg"], "prompt", ["cache"])
        assert len(model._cache_manager) == 1

        # Clear
        model.clear_cache()
        assert len(model._cache_manager) == 0


if __name__ == "__main__":
    # Verbose MLLM cache test with real model
    # Uses mlx-vlm directly (no transformers processor bugs)
    import time
    from pathlib import Path

    VLM_MODEL = "mlx-community/Qwen3-VL-4B-Instruct-3bit"

    def print_header(title):
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)

    def print_subheader(title):
        print("\n" + "-" * 70)
        print(f"  {title}")
        print("-" * 70)

    def print_table(headers, rows, indent=4):
        """Print a formatted table."""
        pad = " " * indent
        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))

        # Print header
        header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        separator = "-+-".join("-" * w for w in col_widths)
        print(f"{pad}{header_line}")
        print(f"{pad}{separator}")

        # Print rows
        for row in rows:
            row_line = " | ".join(
                str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)
            )
            print(f"{pad}{row_line}")

    def print_cache_stats_table(manager, title="Cache Statistics"):
        """Print cache stats as a table."""
        stats = manager.get_stats()
        print(f"\n    {title}:")
        print_table(
            ["Metric", "Value"],
            [
                ["Hits", stats["hits"]],
                ["Misses", stats["misses"]],
                ["Hit Rate", f"{stats['hit_rate'] * 100:.1f}%"],
                ["Tokens Saved", stats["tokens_saved"]],
                ["Image/Video Hits", stats["image_cache_hits"]],
                ["Evictions", stats["evictions"]],
            ],
        )

    def run_mllm_cache_test():
        """
        Test MLLM cache with real model's KV cache and real images/videos.
        """
        from huggingface_hub import snapshot_download
        from mlx_vlm.models import cache as vlm_cache
        from mlx_vlm.utils import load_config, load_model

        from rapid_mlx.benchmark import (
            MLLM_TEST_IMAGE_URLS,
            VLM_TEST_VIDEO_URLS,
            download_test_image,
            download_video,
            get_video_info,
        )

        print_header("MLLM KV CACHE TEST")
        print(f"\n  Model: {VLM_MODEL}")
        print(
            "  Test: Verify KV cache reuse for repeated image/video + prompt combinations"
        )
        print("  Expected behavior:")
        print("    - Same image + same prompt → cache HIT")
        print("    - Same image + different prompt → cache MISS")
        print("    - Different image + same prompt → cache MISS")
        print("    - Same video + same fps/max_frames → cache HIT")
        print("    - Same video + different fps/max_frames → cache MISS")

        # ============================================================
        # SETUP: Load Model and Create KV Cache
        # ============================================================
        print_subheader("SETUP: Loading Model")
        print(f"    Downloading: {VLM_MODEL}")
        model_path = Path(
            snapshot_download(VLM_MODEL, allow_patterns=["*.safetensors", "*.json"])
        )

        load_start = time.perf_counter()
        model = load_model(model_path)
        config = load_config(model_path)
        load_time = time.perf_counter() - load_start
        print(f"    Model loaded in {load_time:.2f}s")
        print(f"    Model type: {config.get('model_type', 'unknown')}")

        print("\n    Creating KV cache from model.language_model...")
        real_kv_cache = vlm_cache.make_prompt_cache(model.language_model)
        print(
            f"    KV cache: {len(real_kv_cache)} layers of {type(real_kv_cache[0]).__name__}"
        )

        # ============================================================
        # SETUP: Download Test Images
        # ============================================================
        print_subheader("SETUP: Downloading Test Images")
        image_paths = []
        resized_image_entries = []
        base_image = None
        import tempfile

        from PIL import Image

        for idx, url in enumerate(MLLM_TEST_IMAGE_URLS, start=1):
            try:
                test_image = download_test_image(url)
                temp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                temp_path = temp_img.name
                temp_img.close()
                test_image.save(temp_path, "JPEG")
                image_paths.append(temp_path)
                if base_image is None:
                    base_image = test_image.copy()
                print(f"    Image {idx}: {test_image.size[0]}x{test_image.size[1]}")
            except Exception as exc:
                print(f"    Image {idx}: FAILED ({exc})")
        if not image_paths:
            raise RuntimeError("No test images could be downloaded.")

        if base_image is not None:
            print("\n    Creating resized variants for cache key testing...")
            resize_sizes = [(224, 224), (336, 336), (512, 512), (768, 768)]
            for width, height in resize_sizes:
                resized = base_image.resize((width, height), Image.Resampling.LANCZOS)
                temp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                temp_path = temp_img.name
                temp_img.close()
                resized.save(temp_path, "JPEG")
                resized_image_entries.append((temp_path, width, height))
                print(f"    Resized: {width}x{height}")

        # ============================================================
        # SETUP: Download Test Videos
        # ============================================================
        print_subheader("SETUP: Downloading Test Videos")
        video_paths = []
        for idx, url in enumerate(VLM_TEST_VIDEO_URLS, start=1):
            try:
                path = download_video(url)
                video_paths.append(path)
                video_info = get_video_info(path)
                print(
                    f"    Video {idx}: {video_info['width']}x{video_info['height']}, "
                    f"{video_info['duration']:.1f}s @ {video_info['fps']:.1f}fps"
                )
            except Exception as exc:
                print(f"    Video {idx}: FAILED ({exc})")
        if not video_paths:
            raise RuntimeError("No test videos could be downloaded.")

        primary_image_path = image_paths[0]
        primary_video_path = video_paths[0]

        # Initialize MLLM Cache Manager
        cache_manager = MLLMCacheManager(max_entries=50)

        # Collect test results for summary table
        test_results = []

        # ============================================================
        # TEST 1: Image Cache - Same image, same prompt should HIT
        # ============================================================
        print_subheader("TEST 1: Image Cache - Basic Hit/Miss")
        test_prompt = "Describe this image in detail"
        print(f"    Image: {primary_image_path}")
        print(f'    Prompt: "{test_prompt}"')

        # Test table for this section
        test1_rows = []

        # First request - miss
        t_start = time.perf_counter()
        cached, hit = cache_manager.fetch_cache([primary_image_path], test_prompt)
        t1 = (time.perf_counter() - t_start) * 1000
        test1_rows.append(
            [
                "1a",
                "First request (new)",
                "MISS",
                "HIT" if hit else "MISS",
                f"{t1:.2f}ms",
                "PASS" if not hit else "FAIL",
            ]
        )
        assert not hit, "Expected cache miss"

        # Store cache
        cache_manager.store_cache(
            [primary_image_path], test_prompt, real_kv_cache, num_tokens=500
        )

        # Second request - hit
        t_start = time.perf_counter()
        cached, hit = cache_manager.fetch_cache([primary_image_path], test_prompt)
        t2 = (time.perf_counter() - t_start) * 1000
        test1_rows.append(
            [
                "1b",
                "Same image+prompt",
                "HIT",
                "HIT" if hit else "MISS",
                f"{t2:.2f}ms",
                "PASS" if hit else "FAIL",
            ]
        )
        assert hit, "Expected cache hit"

        # Different prompt - miss
        t_start = time.perf_counter()
        cached, hit = cache_manager.fetch_cache(
            [primary_image_path], "What colors are in this image?"
        )
        t3 = (time.perf_counter() - t_start) * 1000
        test1_rows.append(
            [
                "1c",
                "Same image, diff prompt",
                "MISS",
                "HIT" if hit else "MISS",
                f"{t3:.2f}ms",
                "PASS" if not hit else "FAIL",
            ]
        )
        assert not hit, "Expected cache miss for different prompt"

        print("\n    Results:")
        print_table(
            ["Step", "Description", "Expected", "Actual", "Time", "Status"], test1_rows
        )
        test_results.extend(test1_rows)
        print_cache_stats_table(cache_manager)

        # ============================================================
        # TEST 2: Different Images Have Different Cache Keys
        # ============================================================
        if len(image_paths) > 1:
            print_subheader("TEST 2: Different Images = Different Cache Keys")
            test2_rows = []
            for idx, image_path in enumerate(image_paths[1:], start=2):
                extra_prompt = f"Describe image {idx}"
                t_start = time.perf_counter()
                cached, hit = cache_manager.fetch_cache([image_path], extra_prompt)
                t_ms = (time.perf_counter() - t_start) * 1000
                test2_rows.append(
                    [
                        f"2.{idx}a",
                        f"Image {idx} first",
                        "MISS",
                        "HIT" if hit else "MISS",
                        f"{t_ms:.2f}ms",
                        "PASS" if not hit else "FAIL",
                    ]
                )
                assert not hit
                cache_manager.store_cache(
                    [image_path], extra_prompt, real_kv_cache, num_tokens=300
                )
                t_start = time.perf_counter()
                cached, hit = cache_manager.fetch_cache([image_path], extra_prompt)
                t_ms = (time.perf_counter() - t_start) * 1000
                test2_rows.append(
                    [
                        f"2.{idx}b",
                        f"Image {idx} cached",
                        "HIT",
                        "HIT" if hit else "MISS",
                        f"{t_ms:.2f}ms",
                        "PASS" if hit else "FAIL",
                    ]
                )
                assert hit
            print("\n    Results:")
            print_table(
                ["Step", "Description", "Expected", "Actual", "Time", "Status"],
                test2_rows,
            )
            test_results.extend(test2_rows)
            print_cache_stats_table(cache_manager)

        # ============================================================
        # TEST 3: Resized Images Have Different Cache Keys (content hash differs)
        # ============================================================
        if resized_image_entries:
            print_subheader("TEST 3: Resized Images = Different Cache Keys")
            print("    (Cache uses content hash, so different sizes = different keys)")
            test3_rows = []
            for idx, (image_path, width, height) in enumerate(
                resized_image_entries, start=1
            ):
                extra_prompt = f"Describe this {width}x{height} image"
                t_start = time.perf_counter()
                cached, hit = cache_manager.fetch_cache([image_path], extra_prompt)
                t_ms = (time.perf_counter() - t_start) * 1000
                test3_rows.append(
                    [
                        f"3.{idx}a",
                        f"{width}x{height} first",
                        "MISS",
                        "HIT" if hit else "MISS",
                        f"{t_ms:.2f}ms",
                        "PASS" if not hit else "FAIL",
                    ]
                )
                assert not hit
                cache_manager.store_cache(
                    [image_path], extra_prompt, real_kv_cache, num_tokens=200
                )
                t_start = time.perf_counter()
                cached, hit = cache_manager.fetch_cache([image_path], extra_prompt)
                t_ms = (time.perf_counter() - t_start) * 1000
                test3_rows.append(
                    [
                        f"3.{idx}b",
                        f"{width}x{height} cached",
                        "HIT",
                        "HIT" if hit else "MISS",
                        f"{t_ms:.2f}ms",
                        "PASS" if hit else "FAIL",
                    ]
                )
                assert hit
            print("\n    Results:")
            print_table(
                ["Step", "Description", "Expected", "Actual", "Time", "Status"],
                test3_rows,
            )
            test_results.extend(test3_rows)
            print_cache_stats_table(cache_manager)

        # ============================================================
        # TEST 4: Video Cache - fps and max_frames affect cache key
        # ============================================================
        print_subheader("TEST 4: Video Cache - fps/max_frames in Cache Key")
        video_fps = 2.0
        video_max_frames = 16
        video_key = f"video:{primary_video_path}:fps{video_fps}:max{video_max_frames}"
        video_prompt = "Describe what happens in this video"

        print(f"    Config: fps={video_fps}, max_frames={video_max_frames}")
        test4_rows = []

        # First request - miss
        t_start = time.perf_counter()
        cached, hit = cache_manager.fetch_cache([video_key], video_prompt)
        t_ms = (time.perf_counter() - t_start) * 1000
        test4_rows.append(
            [
                "4a",
                "Video first request",
                "MISS",
                "HIT" if hit else "MISS",
                f"{t_ms:.2f}ms",
                "PASS" if not hit else "FAIL",
            ]
        )
        assert not hit

        # Store cache
        cache_manager.store_cache(
            [video_key], video_prompt, real_kv_cache, num_tokens=800
        )

        # Same params - hit
        t_start = time.perf_counter()
        cached, hit = cache_manager.fetch_cache([video_key], video_prompt)
        t_ms = (time.perf_counter() - t_start) * 1000
        test4_rows.append(
            [
                "4b",
                "Same video+params",
                "HIT",
                "HIT" if hit else "MISS",
                f"{t_ms:.2f}ms",
                "PASS" if hit else "FAIL",
            ]
        )
        assert hit

        # Different fps - miss (important for video!)
        video_key_diff_fps = f"video:{primary_video_path}:fps4.0:max{video_max_frames}"
        t_start = time.perf_counter()
        cached, hit = cache_manager.fetch_cache([video_key_diff_fps], video_prompt)
        t_ms = (time.perf_counter() - t_start) * 1000
        test4_rows.append(
            [
                "4c",
                "Different fps (4.0)",
                "MISS",
                "HIT" if hit else "MISS",
                f"{t_ms:.2f}ms",
                "PASS" if not hit else "FAIL",
            ]
        )
        assert not hit

        # Different max_frames - miss
        video_key_diff_frames = f"video:{primary_video_path}:fps{video_fps}:max32"
        t_start = time.perf_counter()
        cached, hit = cache_manager.fetch_cache([video_key_diff_frames], video_prompt)
        t_ms = (time.perf_counter() - t_start) * 1000
        test4_rows.append(
            [
                "4d",
                "Different max_frames (32)",
                "MISS",
                "HIT" if hit else "MISS",
                f"{t_ms:.2f}ms",
                "PASS" if not hit else "FAIL",
            ]
        )
        assert not hit

        # Multiple fps/max_frames combinations
        video_configs = [(0.5, 2), (1.0, 4), (2.0, 8), (4.0, 16)]
        for fps_value, max_frames in video_configs:
            extra_key = f"video:{primary_video_path}:fps{fps_value}:max{max_frames}"
            extra_prompt = f"Describe video at {fps_value} fps"
            t_start = time.perf_counter()
            cached, hit = cache_manager.fetch_cache([extra_key], extra_prompt)
            t_ms = (time.perf_counter() - t_start) * 1000
            test4_rows.append(
                [
                    f"4.{fps_value}a",
                    f"fps={fps_value} first",
                    "MISS",
                    "HIT" if hit else "MISS",
                    f"{t_ms:.2f}ms",
                    "PASS" if not hit else "FAIL",
                ]
            )
            assert not hit
            cache_manager.store_cache(
                [extra_key],
                extra_prompt,
                real_kv_cache,
                num_tokens=600 + int(fps_value * 100) + max_frames,
            )
            t_start = time.perf_counter()
            cached, hit = cache_manager.fetch_cache([extra_key], extra_prompt)
            t_ms = (time.perf_counter() - t_start) * 1000
            test4_rows.append(
                [
                    f"4.{fps_value}b",
                    f"fps={fps_value} cached",
                    "HIT",
                    "HIT" if hit else "MISS",
                    f"{t_ms:.2f}ms",
                    "PASS" if hit else "FAIL",
                ]
            )
            assert hit

        print("\n    Results:")
        print_table(
            ["Step", "Description", "Expected", "Actual", "Time", "Status"], test4_rows
        )
        test_results.extend(test4_rows)
        print_cache_stats_table(cache_manager)

        # Extra videos
        if len(video_paths) > 1:
            print_subheader("TEST 5: Additional Videos")
            test5_rows = []
            for idx, path in enumerate(video_paths[1:], start=2):
                extra_video_key = f"video:{path}:fps{video_fps}:max{video_max_frames}"
                extra_prompt = f"Describe video {idx}"
                t_start = time.perf_counter()
                cached, hit = cache_manager.fetch_cache([extra_video_key], extra_prompt)
                t_ms = (time.perf_counter() - t_start) * 1000
                test5_rows.append(
                    [
                        f"5.{idx}a",
                        f"Video {idx} first",
                        "MISS",
                        "HIT" if hit else "MISS",
                        f"{t_ms:.2f}ms",
                        "PASS" if not hit else "FAIL",
                    ]
                )
                assert not hit
                cache_manager.store_cache(
                    [extra_video_key],
                    extra_prompt,
                    real_kv_cache,
                    num_tokens=700 + idx * 10,
                )
                t_start = time.perf_counter()
                cached, hit = cache_manager.fetch_cache([extra_video_key], extra_prompt)
                t_ms = (time.perf_counter() - t_start) * 1000
                test5_rows.append(
                    [
                        f"5.{idx}b",
                        f"Video {idx} cached",
                        "HIT",
                        "HIT" if hit else "MISS",
                        f"{t_ms:.2f}ms",
                        "PASS" if hit else "FAIL",
                    ]
                )
                assert hit
            print("\n    Results:")
            print_table(
                ["Step", "Description", "Expected", "Actual", "Time", "Status"],
                test5_rows,
            )
            test_results.extend(test5_rows)
            print_cache_stats_table(cache_manager)

        # ============================================================
        # TEST 6: LRU Eviction
        # ============================================================
        print_subheader("TEST 6: LRU Eviction Policy")
        small_cache = MLLMCacheManager(max_entries=2)
        small_cache.store_cache(["img1.jpg"], "p1", real_kv_cache)
        small_cache.store_cache(["img2.jpg"], "p2", real_kv_cache)
        print(f"    Cache capacity: 2 entries (currently {len(small_cache)}/2)")

        # Access img1 to make it recently used
        small_cache.fetch_cache(["img1.jpg"], "p1")
        print("    Touched img1 to make it recently used")

        # Add new entry - should evict img2
        small_cache.store_cache(["img3.jpg"], "p3", real_kv_cache)

        test6_rows = []
        t_start = time.perf_counter()
        _, hit = small_cache.fetch_cache(["img2.jpg"], "p2")
        t_ms = (time.perf_counter() - t_start) * 1000
        test6_rows.append(
            [
                "6a",
                "img2 (oldest, evicted)",
                "MISS",
                "HIT" if hit else "MISS",
                f"{t_ms:.2f}ms",
                "PASS" if not hit else "FAIL",
            ]
        )
        assert not hit

        t_start = time.perf_counter()
        _, hit = small_cache.fetch_cache(["img1.jpg"], "p1")
        t_ms = (time.perf_counter() - t_start) * 1000
        test6_rows.append(
            [
                "6b",
                "img1 (recently used)",
                "HIT",
                "HIT" if hit else "MISS",
                f"{t_ms:.2f}ms",
                "PASS" if hit else "FAIL",
            ]
        )
        assert hit

        t_start = time.perf_counter()
        _, hit = small_cache.fetch_cache(["img3.jpg"], "p3")
        t_ms = (time.perf_counter() - t_start) * 1000
        test6_rows.append(
            [
                "6c",
                "img3 (newest)",
                "HIT",
                "HIT" if hit else "MISS",
                f"{t_ms:.2f}ms",
                "PASS" if hit else "FAIL",
            ]
        )
        assert hit

        print("\n    Results:")
        print_table(
            ["Step", "Description", "Expected", "Actual", "Time", "Status"], test6_rows
        )
        print(f"\n    Evictions: {small_cache.stats.evictions}")

        # ============================================================
        # FINAL SUMMARY
        # ============================================================
        print_header("TEST RESULTS SUMMARY")

        stats = cache_manager.get_stats()
        print("\n    Final Cache Statistics:")
        print_table(
            ["Metric", "Value"],
            [
                ["Total Hits", stats["hits"]],
                ["Total Misses", stats["misses"]],
                ["Hit Rate", f"{stats['hit_rate'] * 100:.1f}%"],
                ["Tokens Saved", stats["tokens_saved"]],
                ["Image/Video Hits", stats["image_cache_hits"]],
                ["Evictions", stats["evictions"]],
            ],
        )
        print("\n" + "=" * 70)
        print("  [OK] ALL TESTS PASSED - MLLM cache working correctly")
        print("=" * 70)

        # Cleanup temp files
        for path in image_paths:
            try:
                os.unlink(path)
            except OSError:
                pass
        for path, _, _ in resized_image_entries:
            try:
                os.unlink(path)
            except OSError:
                pass
        for path in video_paths:
            try:
                os.unlink(path)
            except OSError:
                pass

    run_mllm_cache_test()
