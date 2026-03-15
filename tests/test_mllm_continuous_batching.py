# SPDX-License-Identifier: Apache-2.0
"""
Tests for MLLM (Multimodal Language Model) continuous batching.

These tests verify that the MLLM batch generator and scheduler work correctly
for batching multiple multimodal requests together.

Test Cases:
- Single MLLM request works correctly
- 2, 4, 8 concurrent requests with batching
- Vision cache hits/misses
- Streaming with batching
- Mixed text-only and multimodal requests
"""

import base64
import os
import tempfile
from unittest.mock import MagicMock

import pytest

# Skip all tests if MLX is not available
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


# Test image (small PNG)
TEST_IMAGE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


def create_test_image(path: str, size: tuple = (32, 32)) -> str:
    """Create a test image file."""
    try:
        import numpy as np
        from PIL import Image

        img = Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8))
        img.save(path)
        return path
    except ImportError:
        # Fallback: write a minimal valid PNG
        png_data = base64.b64decode(TEST_IMAGE_B64)
        with open(path, "wb") as f:
            f.write(png_data)
        return path


class TestMLLMBatchRequest:
    """Tests for MLLMBatchRequest dataclass."""

    def test_create_request(self):
        """Test creating a basic request."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        req = MLLMBatchRequest(
            uid=0,
            request_id="test-1",
            prompt="What's in this image?",
            images=["test.jpg"],
            max_tokens=100,
        )

        assert req.uid == 0
        assert req.request_id == "test-1"
        assert req.prompt == "What's in this image?"
        assert req.images == ["test.jpg"]
        assert req.max_tokens == 100
        assert req.num_tokens == 0
        assert req.vision_encoded is False

    def test_request_defaults(self):
        """Test default values."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        req = MLLMBatchRequest(
            uid=1,
            request_id="test-2",
            prompt="Hello",
        )

        assert req.images is None
        assert req.videos is None
        assert req.max_tokens == 256
        assert req.temperature == 0.7
        assert req.top_p == 0.9
        assert req.output_tokens == []


class TestMLLMBatchResponse:
    """Tests for MLLMBatchResponse dataclass."""

    def test_create_response(self):
        """Test creating a response."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchResponse

        logprobs = mx.array([0.1, 0.2, 0.3])

        resp = MLLMBatchResponse(
            uid=0,
            request_id="test-1",
            token=42,
            logprobs=logprobs,
            finish_reason=None,
        )

        assert resp.uid == 0
        assert resp.request_id == "test-1"
        assert resp.token == 42
        assert resp.finish_reason is None

    def test_finished_response(self):
        """Test response with finish reason."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchResponse

        resp = MLLMBatchResponse(
            uid=0,
            request_id="test-1",
            token=2,  # EOS
            logprobs=mx.array([0.1]),
            finish_reason="stop",
        )

        assert resp.finish_reason == "stop"


class TestMLLMBatch:
    """Tests for MLLMBatch class."""

    def test_batch_length(self):
        """Test batch length calculation."""
        from vllm_mlx.mllm_batch_generator import MLLMBatch, MLLMBatchRequest

        requests = [
            MLLMBatchRequest(uid=i, request_id=f"req-{i}", prompt=f"prompt {i}")
            for i in range(3)
        ]

        batch = MLLMBatch(
            uids=[0, 1, 2],
            request_ids=["req-0", "req-1", "req-2"],
            y=mx.array([100, 200, 300]),
            logprobs=[mx.array([0.1]), mx.array([0.2]), mx.array([0.3])],
            max_tokens=[100, 100, 100],
            num_tokens=[0, 0, 0],
            cache=[],
            requests=requests,
        )

        assert len(batch) == 3

    def test_batch_filter(self):
        """Test filtering a batch."""
        from vllm_mlx.mllm_batch_generator import MLLMBatch, MLLMBatchRequest

        requests = [
            MLLMBatchRequest(uid=i, request_id=f"req-{i}", prompt=f"prompt {i}")
            for i in range(4)
        ]

        batch = MLLMBatch(
            uids=[0, 1, 2, 3],
            request_ids=["req-0", "req-1", "req-2", "req-3"],
            y=mx.array([100, 200, 300, 400]),
            logprobs=[
                mx.array([0.1]),
                mx.array([0.2]),
                mx.array([0.3]),
                mx.array([0.4]),
            ],
            max_tokens=[100, 100, 100, 100],
            num_tokens=[0, 0, 0, 0],
            cache=[],
            requests=requests,
        )

        # Keep only indices 1 and 3
        batch.filter([1, 3])

        assert len(batch) == 2
        assert batch.uids == [1, 3]
        assert batch.request_ids == ["req-1", "req-3"]


class TestMLLMBatchStats:
    """Tests for MLLMBatchStats."""

    def test_stats_initialization(self):
        """Test stats initialization."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchStats

        stats = MLLMBatchStats()

        assert stats.prompt_tokens == 0
        assert stats.generation_tokens == 0
        assert stats.prompt_time == 0
        assert stats.generation_time == 0
        assert stats.num_images_processed == 0

    def test_tps_calculation(self):
        """Test tokens per second calculation."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchStats

        stats = MLLMBatchStats()
        stats.prompt_tokens = 100
        stats.prompt_time = 2.0
        stats.generation_tokens = 50
        stats.generation_time = 1.0

        assert stats.prompt_tps == 50.0
        assert stats.generation_tps == 50.0

    def test_tps_zero_time(self):
        """Test TPS with zero time."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchStats

        stats = MLLMBatchStats()

        assert stats.prompt_tps == 0
        assert stats.generation_tps == 0


class TestMLLMSchedulerConfig:
    """Tests for MLLMSchedulerConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from vllm_mlx.mllm_scheduler import MLLMSchedulerConfig

        config = MLLMSchedulerConfig()

        assert config.max_num_seqs == 16
        # prefill_batch_size set equal to max_num_seqs to avoid batch extend issues
        assert config.prefill_batch_size == 16
        assert config.completion_batch_size == 16
        assert config.enable_vision_cache is True
        assert config.vision_cache_size == 100

    def test_custom_config(self):
        """Test custom configuration."""
        from vllm_mlx.mllm_scheduler import MLLMSchedulerConfig

        config = MLLMSchedulerConfig(
            max_num_seqs=8,
            prefill_batch_size=2,
            completion_batch_size=8,
            enable_vision_cache=False,
        )

        assert config.max_num_seqs == 8
        assert config.prefill_batch_size == 2
        assert config.completion_batch_size == 8
        assert config.enable_vision_cache is False


class TestMLLMRequest:
    """Tests for MLLMRequest dataclass."""

    def test_create_request(self):
        """Test creating an MLLM request."""
        from vllm_mlx.mllm_scheduler import MLLMRequest
        from vllm_mlx.request import RequestStatus

        req = MLLMRequest(
            request_id="req-1",
            prompt="Describe this image",
            images=["image.jpg"],
        )

        assert req.request_id == "req-1"
        assert req.prompt == "Describe this image"
        assert req.images == ["image.jpg"]
        assert req.status == RequestStatus.WAITING
        assert req.output_text == ""


class TestMLLMSchedulerOutput:
    """Tests for MLLMSchedulerOutput."""

    def test_empty_output(self):
        """Test empty scheduler output."""
        from vllm_mlx.mllm_scheduler import MLLMSchedulerOutput

        output = MLLMSchedulerOutput()

        assert output.scheduled_request_ids == []
        assert output.num_scheduled_tokens == 0
        assert output.finished_request_ids == set()
        assert output.outputs == []
        assert output.has_work is False


class TestMultimodalProcessorBatch:
    """Tests for MultimodalProcessor batch methods."""

    def test_batch_pixel_values_empty(self):
        """Test batching empty pixel values."""
        from vllm_mlx.multimodal_processor import MultimodalProcessor

        # Create mock processor
        mock_model = MagicMock()
        mock_processor = MagicMock()

        processor = MultimodalProcessor(mock_model, mock_processor)

        result = processor.batch_pixel_values([None, None])
        assert result is None

    def test_batch_pixel_values_single(self):
        """Test batching single pixel value."""
        from vllm_mlx.multimodal_processor import MultimodalProcessor

        mock_model = MagicMock()
        mock_processor = MagicMock()

        processor = MultimodalProcessor(mock_model, mock_processor)

        pixels = mx.ones((1, 3, 32, 32))
        result = processor.batch_pixel_values([pixels])

        assert result is not None
        assert result.shape == (1, 3, 32, 32)

    def test_batch_pixel_values_multiple(self):
        """Test batching multiple pixel values."""
        from vllm_mlx.multimodal_processor import MultimodalProcessor

        mock_model = MagicMock()
        mock_processor = MagicMock()

        processor = MultimodalProcessor(mock_model, mock_processor)

        pixels1 = mx.ones((1, 3, 32, 32))
        pixels2 = mx.ones((1, 3, 32, 32)) * 2

        result = processor.batch_pixel_values([pixels1, pixels2])

        assert result is not None
        assert result.shape == (2, 3, 32, 32)

    def test_batch_image_grid_thw(self):
        """Test batching image grid thw."""
        from vllm_mlx.multimodal_processor import MultimodalProcessor

        mock_model = MagicMock()
        mock_processor = MagicMock()

        processor = MultimodalProcessor(mock_model, mock_processor)

        grid1 = mx.array([[1, 4, 4]])
        grid2 = mx.array([[1, 8, 8]])

        result = processor.batch_image_grid_thw([grid1, grid2])

        assert result is not None
        assert result.shape[0] == 2

    def test_prepare_for_batch(self):
        """Test prepare_for_batch method."""
        from vllm_mlx.multimodal_processor import (
            MultimodalProcessor,
            ProcessedMultimodalInput,
        )

        mock_model = MagicMock()
        mock_processor = MagicMock()

        processor = MultimodalProcessor(mock_model, mock_processor)

        # Create processed inputs
        inputs = [
            ProcessedMultimodalInput(
                input_ids=mx.array([1, 2, 3]),
                pixel_values=mx.ones((1, 3, 32, 32)),
                num_images=1,
                num_tokens=3,
            ),
            ProcessedMultimodalInput(
                input_ids=mx.array([4, 5, 6, 7, 8]),
                pixel_values=mx.ones((1, 3, 32, 32)),
                num_images=1,
                num_tokens=5,
            ),
        ]

        input_ids, batch_kwargs, padding = processor.prepare_for_batch(inputs)

        # Check left-padding
        assert input_ids.shape == (2, 5)  # max length is 5
        assert padding == [2, 0]  # first input needs 2 padding

    def test_compute_vision_hash(self):
        """Test vision hash computation."""
        from vllm_mlx.multimodal_processor import MultimodalProcessor

        mock_model = MagicMock()
        mock_processor = MagicMock()

        processor = MultimodalProcessor(mock_model, mock_processor)

        pixels = mx.ones((1, 3, 32, 32))
        hash1 = processor.compute_vision_hash(pixels)
        hash2 = processor.compute_vision_hash(pixels)

        # Same input should give same hash
        assert hash1 == hash2
        assert len(hash1) == 16  # SHA256 truncated to 16 chars


class TestVisionCache:
    """Tests for VLM cache functionality."""

    def test_cache_creation(self):
        """Test VLM cache creation."""
        from vllm_mlx.mllm_cache import MLLMCacheManager

        cache = MLLMCacheManager(max_entries=10)

        assert len(cache) == 0
        assert cache.max_size == 10

    def test_cache_miss(self):
        """Test cache miss."""
        from vllm_mlx.mllm_cache import MLLMCacheManager

        cache = MLLMCacheManager()

        result, hit = cache.fetch_cache(["image.jpg"], "prompt")

        assert result is None
        assert hit is False
        assert cache.stats.misses == 1

    def test_cache_store_and_fetch(self):
        """Test storing and fetching from cache."""
        from vllm_mlx.mllm_cache import MLLMCacheManager

        cache = MLLMCacheManager()

        # Store cache
        test_cache = [{"key": "value"}]
        cache.store_cache(["image.jpg"], "prompt", test_cache, num_tokens=100)

        # Fetch cache
        result, hit = cache.fetch_cache(["image.jpg"], "prompt")

        assert result is not None
        assert hit is True
        assert cache.stats.hits == 1
        assert cache.stats.tokens_saved == 100

    def test_cache_eviction(self):
        """Test cache eviction when full."""
        from vllm_mlx.mllm_cache import MLLMCacheManager

        cache = MLLMCacheManager(max_entries=2)

        # Fill cache
        cache.store_cache(["img1.jpg"], "prompt1", [1], num_tokens=10)
        cache.store_cache(["img2.jpg"], "prompt2", [2], num_tokens=20)

        assert len(cache) == 2

        # Add one more (should evict oldest)
        cache.store_cache(["img3.jpg"], "prompt3", [3], num_tokens=30)

        assert len(cache) == 2
        assert cache.stats.evictions == 1

        # img1 should be evicted
        _, hit = cache.fetch_cache(["img1.jpg"], "prompt1")
        assert hit is False


class TestMLLMSchedulerStopSequences:
    """Regression tests for stop sequence forwarding (PR #21)."""

    def test_mllm_request_carries_stop(self):
        """MLLMRequest should carry text-based stop sequences."""
        from vllm_mlx.mllm_scheduler import MLLMRequest

        req = MLLMRequest(
            request_id="test-stop",
            prompt="Hello",
            stop=["###", "\n\n"],
        )
        assert req.stop == ["###", "\n\n"]

    def test_mllm_request_default_stop_empty(self):
        """MLLMRequest.stop should default to empty list."""
        from vllm_mlx.mllm_scheduler import MLLMRequest

        req = MLLMRequest(request_id="test-default", prompt="Hello")
        assert req.stop == []

    def test_process_batch_responses_stop_string(self):
        """_process_batch_responses should finish request when stop string found."""
        from vllm_mlx.mllm_scheduler import (
            MLLMRequest,
            MLLMScheduler,
            MLLMSchedulerConfig,
        )
        from vllm_mlx.mllm_batch_generator import MLLMBatchResponse
        from vllm_mlx.request import SamplingParams

        # Create scheduler with mocks
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_tokenizer = MagicMock()
        mock_processor.tokenizer = mock_tokenizer

        # Simulate tokenizer decoding:
        # Token 10 -> "Hello", Token 20 -> " world", Token 30 -> "###end"
        mock_tokenizer.decode.side_effect = lambda ids: {
            (10,): "Hello",
            (20,): " world",
            (30,): "###end",
            (10, 20): "Hello world",
            (10, 20, 30): "Hello world###end",
        }.get(tuple(ids), "")

        config = MLLMSchedulerConfig()
        scheduler = MLLMScheduler(mock_model, mock_processor, config)

        # Create a request with stop sequences
        request = MLLMRequest(
            request_id="req-1",
            prompt="Say hello",
            sampling_params=SamplingParams(max_tokens=100),
            stop=["###"],
        )
        request.output_tokens = [10, 20]  # Already has "Hello world"
        request.num_output_tokens = 2

        scheduler.running["req-1"] = request
        scheduler.uid_to_request_id[0] = "req-1"

        # Process a response with token 30 (contains "###")
        response = MLLMBatchResponse(
            uid=0,
            request_id="req-1",
            token=30,
            logprobs=mx.array([0.1]),
            finish_reason=None,  # BatchGenerator didn't detect stop
        )

        outputs, finished_ids = scheduler._process_batch_responses([response])

        assert "req-1" in finished_ids
        assert outputs[0].finished is True
        assert outputs[0].finish_reason == "stop"
        # Output text should be trimmed at stop string
        assert outputs[0].output_text == "Hello world"

    def test_add_request_forwards_stop(self):
        """add_request should store stop sequences on the MLLMRequest."""
        from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig

        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_processor.tokenizer = MagicMock()

        scheduler = MLLMScheduler(mock_model, mock_processor, MLLMSchedulerConfig())

        req_id = scheduler.add_request(
            prompt="test",
            stop=["<|end|>", "STOP"],
        )

        request = scheduler.requests[req_id]
        assert request.stop == ["<|end|>", "STOP"]


class TestDeferredAbortWaitingDeque:
    """Regression tests for deferred abort cleaning up waiting deque (PR #21)."""

    def test_do_abort_removes_waiting_when_request_none(self):
        """_do_abort_request should remove from waiting even if request already cleaned."""
        from vllm_mlx.scheduler import Scheduler, SchedulerConfig
        from vllm_mlx.request import Request, RequestStatus, SamplingParams

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]

        config = SchedulerConfig()
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=config,
        )

        # Manually add a request to the waiting deque
        request = Request(
            request_id="test-abort",
            prompt="hello",
            sampling_params=SamplingParams(),
            prompt_token_ids=[1, 2, 3],
            num_prompt_tokens=3,
        )
        request.status = RequestStatus.WAITING
        scheduler.waiting.append(request)
        # Do NOT add to scheduler.requests — simulates _cleanup_request
        # having already popped it

        assert len(scheduler.waiting) == 1

        # Call _do_abort_request — request is None in self.requests
        scheduler._do_abort_request("test-abort")

        # Waiting deque should be empty now
        assert len(scheduler.waiting) == 0
        assert "test-abort" in scheduler.finished_req_ids


# Integration tests (require model loading)
@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("RUN_SLOW_TESTS"), reason="Slow tests disabled")
class TestMLLMSchedulerIntegration:
    """Integration tests for MLLMScheduler with real models."""

    @pytest.fixture
    def test_image_path(self):
        """Create a test image."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            path = create_test_image(f.name)
            yield path
            os.unlink(path)

    async def test_single_request(self, test_image_path):
        """Test single MLLM request."""
        from mlx_vlm import load

        from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig

        # Load a small model
        model, processor = load("mlx-community/Qwen3-VL-4B-Instruct-3bit")

        config = MLLMSchedulerConfig(max_num_seqs=4)
        scheduler = MLLMScheduler(model, processor, config)

        await scheduler.start()

        try:
            request_id = scheduler.add_request(
                prompt="What's in this image?",
                images=[test_image_path],
                max_tokens=50,
            )

            # Run until complete
            while scheduler.has_requests():
                output = scheduler.step()
                if request_id in output.finished_request_ids:
                    break

            # Check result
            request = scheduler.get_request(request_id)
            assert request is not None
            assert len(request.output_tokens) > 0

        finally:
            await scheduler.stop()

    async def test_concurrent_requests(self, test_image_path):
        """Test multiple concurrent MLLM requests."""
        from mlx_vlm import load

        from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig

        model, processor = load("mlx-community/Qwen3-VL-4B-Instruct-3bit")

        config = MLLMSchedulerConfig(max_num_seqs=4)
        scheduler = MLLMScheduler(model, processor, config)

        await scheduler.start()

        try:
            # Add multiple requests
            request_ids = []
            for i in range(4):
                req_id = scheduler.add_request(
                    prompt=f"Describe image {i}",
                    images=[test_image_path],
                    max_tokens=30,
                )
                request_ids.append(req_id)

            # Run until all complete
            finished = set()
            while len(finished) < len(request_ids):
                output = scheduler.step()
                finished.update(output.finished_request_ids)

            # Check all completed
            assert len(finished) == 4

            # Check stats show batching
            stats = scheduler.get_stats()
            assert stats["num_requests_processed"] == 4

        finally:
            await scheduler.stop()

    async def test_streaming(self, test_image_path):
        """Test streaming MLLM generation."""
        from mlx_vlm import load

        from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig

        model, processor = load("mlx-community/Qwen3-VL-4B-Instruct-3bit")

        config = MLLMSchedulerConfig()
        scheduler = MLLMScheduler(model, processor, config)

        await scheduler.start()

        try:
            request_id = await scheduler.add_request_async(
                prompt="Describe this image briefly",
                images=[test_image_path],
                max_tokens=30,
            )

            tokens_received = 0
            async for output in scheduler.stream_outputs(request_id):
                tokens_received += len(output.new_token_ids)
                if output.finished:
                    break

            assert tokens_received > 0

        finally:
            await scheduler.stop()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
