# SPDX-License-Identifier: Apache-2.0
"""
Performance Benchmark for vllm-mlx.

Measures key performance metrics for LLM and MLLM (Multimodal Language Model) inference:
- Time to First Token (TTFT)
- Time Per Output Token (TPOT)
- Tokens Per Second (TPS) - both input processing and output generation
- End-to-End Latency
- Throughput
- Memory Usage (process and MLX cache)
- MLLM: Image resolution performance
- MLLM: Video frame count performance

Usage:
    # LLM benchmark
    python -m vllm_mlx.benchmark --model mlx-community/Llama-3.2-1B-Instruct-4bit
    python -m vllm_mlx.benchmark --model mlx-community/Llama-3.2-3B-Instruct-4bit --prompts 10 --max-tokens 256

    # MLLM image benchmark (auto-detected or use --mllm flag)
    python -m vllm_mlx.benchmark --model mlx-community/Qwen3-VL-4B-Instruct-3bit
    python -m vllm_mlx.benchmark --model mlx-community/Qwen3-VL-4B-Instruct-3bit --mllm --quick

    # MLLM video benchmark
    python -m vllm_mlx.benchmark --model mlx-community/Qwen3-VL-4B-Instruct-3bit --video
    python -m vllm_mlx.benchmark --model mlx-community/Qwen3-VL-4B-Instruct-3bit --video --video-url https://example.com/video.mp4
"""

# Disable tokenizers parallelism warning (must be before importing transformers)
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import base64
import io
import json
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None  # Only needed for video benchmarks
import numpy as np
import requests
from PIL import Image
from tabulate import tabulate

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# =============================================================================
# Resource Monitoring (Memory)
# =============================================================================


@dataclass
class ResourceMetrics:
    """Resource usage metrics during benchmark."""

    # Memory metrics (in GB)
    process_memory_gb: float = 0.0
    mlx_cache_gb: float = 0.0
    mlx_peak_memory_gb: float = 0.0
    system_memory_used_gb: float = 0.0
    system_memory_total_gb: float = 0.0


def reset_mlx_peak_memory():
    """Reset MLX peak memory counter."""
    if not HAS_MLX:
        return

    try:
        # Use new API (mx.*) if available, fallback to deprecated (mx.metal.*)
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        else:
            mx.metal.reset_peak_memory()
    except Exception:
        pass


def get_mlx_memory_info(reset_peak: bool = True) -> dict:
    """Get MLX memory usage information.

    Args:
        reset_peak: If True, reset peak memory counter after reading.
    """
    if not HAS_MLX:
        return {}

    try:
        # Use new API (mx.*) if available, fallback to deprecated (mx.metal.*)
        if hasattr(mx, "get_cache_memory"):
            cache_memory = mx.get_cache_memory()
            peak_memory = mx.get_peak_memory()
            active_memory = (
                mx.get_active_memory() if hasattr(mx, "get_active_memory") else 0
            )
        else:
            # Fallback for older MLX versions
            cache_memory = mx.metal.get_cache_memory()
            peak_memory = mx.metal.get_peak_memory()
            active_memory = (
                mx.metal.get_active_memory()
                if hasattr(mx.metal, "get_active_memory")
                else 0
            )

        info = {
            "cache_memory_gb": cache_memory / (1024**3),
            "peak_memory_gb": peak_memory / (1024**3),
            "active_memory_gb": active_memory / (1024**3),
        }
        # Reset peak for next measurement if requested
        if reset_peak:
            reset_mlx_peak_memory()
        return info
    except Exception:
        return {}


def get_process_memory() -> float:
    """Get current process memory usage in GB."""
    if not HAS_PSUTIL:
        return 0.0

    try:
        process = psutil.Process()
        return process.memory_info().rss / (1024**3)
    except Exception:
        return 0.0


def get_system_memory() -> tuple[float, float]:
    """Get system memory (used, total) in GB."""
    if not HAS_PSUTIL:
        return 0.0, 0.0

    try:
        mem = psutil.virtual_memory()
        return mem.used / (1024**3), mem.total / (1024**3)
    except Exception:
        return 0.0, 0.0


class ResourceMonitor:
    """Monitor system resources during benchmark runs."""

    def __init__(self):
        self.samples: list[ResourceMetrics] = []
        self._start_time: float = 0
        self._start_memory: float = 0

    def start(self):
        """Start monitoring."""
        self._start_time = time.perf_counter()
        self._start_memory = get_process_memory()

        # Reset MLX peak memory
        reset_mlx_peak_memory()

    def sample(self) -> ResourceMetrics:
        """Take a resource sample."""
        mlx_info = get_mlx_memory_info()
        sys_used, sys_total = get_system_memory()

        metrics = ResourceMetrics(
            process_memory_gb=get_process_memory(),
            mlx_cache_gb=mlx_info.get("cache_memory_gb", 0.0),
            mlx_peak_memory_gb=mlx_info.get("peak_memory_gb", 0.0),
            system_memory_used_gb=sys_used,
            system_memory_total_gb=sys_total,
        )

        self.samples.append(metrics)
        return metrics

    def get_summary(self) -> ResourceMetrics:
        """Get summary of all samples."""
        if not self.samples:
            return ResourceMetrics()

        # Get peak values
        peak_process = max(s.process_memory_gb for s in self.samples)
        peak_mlx = max(s.mlx_peak_memory_gb for s in self.samples)
        peak_mlx_cache = max(s.mlx_cache_gb for s in self.samples)

        # Get latest system memory
        latest = self.samples[-1]

        return ResourceMetrics(
            process_memory_gb=peak_process,
            mlx_cache_gb=peak_mlx_cache,
            mlx_peak_memory_gb=peak_mlx,
            system_memory_used_gb=latest.system_memory_used_gb,
            system_memory_total_gb=latest.system_memory_total_gb,
        )


# =============================================================================
# Video Benchmark Configuration
# =============================================================================

# Sample video URLs for testing (free, no copyright)
VIDEO_SAMPLE_URLS = {
    "bunny_10s": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4",
    "bunny_240p": "https://docs.evostream.com/sample_content/assets/bunny.mp4",
    "sintel_720p": "https://docs.evostream.com/sample_content/assets/sintel1m720p.mp4",
}

DEFAULT_VIDEO_URL = VIDEO_SAMPLE_URLS["bunny_10s"]
VLM_TEST_VIDEO_URLS = [
    VIDEO_SAMPLE_URLS["bunny_10s"],
    VIDEO_SAMPLE_URLS["bunny_240p"],
]


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    prompt: str
    prompt_tokens: int
    generated_tokens: int

    # Timing metrics (in seconds)
    ttft: float  # Time to First Token
    total_time: float  # End-to-End Latency

    # Derived metrics
    tpot: float = 0.0  # Time Per Output Token
    generation_tps: float = 0.0  # Output tokens per second
    processing_tps: float = 0.0  # Input tokens per second (prompt processing)

    def __post_init__(self):
        if self.generated_tokens > 1:
            # TPOT excludes the first token (which is measured by TTFT)
            generation_time = self.total_time - self.ttft
            self.tpot = (
                generation_time / (self.generated_tokens - 1)
                if self.generated_tokens > 1
                else 0
            )
            self.generation_tps = (
                (self.generated_tokens - 1) / generation_time
                if generation_time > 0
                else 0
            )

        # Processing TPS: how fast the prompt was processed (tokens / TTFT)
        if self.ttft > 0:
            self.processing_tps = self.prompt_tokens / self.ttft


@dataclass
class BenchmarkSummary:
    """Summary statistics across all benchmark runs."""

    model_name: str
    num_runs: int
    total_prompt_tokens: int
    total_generated_tokens: int
    total_time: float

    # TTFT stats (in seconds)
    ttft_mean: float
    ttft_min: float
    ttft_max: float
    ttft_p50: float
    ttft_p95: float

    # TPOT stats (in seconds)
    tpot_mean: float
    tpot_min: float
    tpot_max: float

    # TPS stats
    generation_tps_mean: float
    generation_tps_max: float
    processing_tps_mean: float

    # End-to-end stats (in seconds)
    latency_mean: float
    latency_min: float
    latency_max: float
    latency_p50: float
    latency_p95: float

    # Overall throughput
    total_throughput_tps: float  # Total tokens (input + output) per second
    requests_per_second: float

    # Hardware info
    hardware_chip: str = ""
    hardware_memory_gb: float = 0.0
    hardware_bandwidth_gbs: float = 0.0

    # Resource metrics
    resources: ResourceMetrics = field(default_factory=ResourceMetrics)


def calculate_percentile(data: list, percentile: float) -> float:
    """Calculate percentile from a list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = int(len(sorted_data) * percentile / 100)
    index = min(index, len(sorted_data) - 1)
    return sorted_data[index]


def benchmark_single_prompt(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> BenchmarkResult | None:
    """
    Benchmark a single prompt with detailed timing.

    Args:
        model: The loaded MLX model
        tokenizer: The tokenizer
        prompt: The prompt to benchmark
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        BenchmarkResult with timing metrics
    """
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    try:
        # Tokenize the prompt to count tokens
        prompt_tokens = tokenizer.encode(prompt)
        prompt_token_count = len(prompt_tokens)

        # Create sampler
        sampler = make_sampler(temp=temperature)

        # Start timing
        start_time = time.perf_counter()
        ttft = None
        token_count = 0

        # Generate tokens using stream_generate
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            if ttft is None:
                ttft = time.perf_counter() - start_time
            token_count += 1

        total_time = time.perf_counter() - start_time

        if ttft is None:
            ttft = total_time

        return BenchmarkResult(
            prompt=prompt[:50] + "..." if len(prompt) > 50 else prompt,
            prompt_tokens=prompt_token_count,
            generated_tokens=token_count,
            ttft=ttft,
            total_time=total_time,
        )

    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback

        traceback.print_exc()
        return None


def run_benchmark(
    model_name: str,
    num_prompts: int = 5,
    max_tokens: int = 256,
    temperature: float = 0.7,
    warmup_runs: int = 1,
) -> BenchmarkSummary | None:
    """
    Run the full benchmark suite.

    Args:
        model_name: HuggingFace model name or local path
        num_prompts: Number of prompts to test
        max_tokens: Maximum tokens per generation
        temperature: Sampling temperature
        warmup_runs: Number of warmup runs before measuring

    Returns:
        BenchmarkSummary with aggregate statistics
    """
    from vllm_mlx.optimizations import detect_hardware
    from vllm_mlx.utils.tokenizer import load_model_with_fallback

    # Detect hardware
    hw = detect_hardware()

    # Test prompts of varying lengths: 3 short, 3 medium, 4 long
    prompts = [
        # Short prompts (~5-15 tokens) - 3 prompts
        "Hello, how are you?",
        "What is 2+2?",
        "Say hello in Spanish.",
        # Medium prompts (~30-60 tokens) - 3 prompts
        "What is the capital of France and why is it historically significant? Include some interesting facts about the city.",
        "Write a Python function to calculate fibonacci numbers using memoization. Explain how it works.",
        "Explain the difference between a list and a tuple in Python. When should you use each one?",
        # Long prompts (100+ tokens) - 4 prompts
        """Explain quantum computing in comprehensive detail. You should cover all of the following topics thoroughly:
        1. What are qubits and how do they fundamentally differ from classical bits in traditional computing?
        2. What is quantum superposition and how does it enable parallel computation?
        3. What is quantum entanglement and why is it crucial for quantum algorithms?
        4. What are the most promising potential applications of quantum computing in cryptography, drug discovery, and optimization?
        5. What are the current hardware limitations, error correction challenges, and decoherence problems?
        6. Compare the approaches of IBM, Google, and other major players in quantum computing research.""",
        """Write a comprehensive and detailed guide to building a production-ready REST API with Python Flask. Include all of the following sections with code examples:
        1. Setting up the project structure with blueprints, configuration management, and environment variables
        2. Creating routes and endpoints following RESTful conventions with proper HTTP methods
        3. Handling JSON requests and responses with validation using marshmallow or pydantic
        4. Adding authentication and authorization with JWT tokens and role-based access control
        5. Implementing error handling best practices with custom exception handlers
        6. Writing comprehensive tests with pytest including unit tests and integration tests
        7. Setting up logging, monitoring, and API documentation with Swagger/OpenAPI""",
        """Describe the complete process of photosynthesis in plants with scientific detail. Your explanation should cover:
        1. The light-dependent reactions that occur in the thylakoid membrane, including photosystems I and II
        2. The electron transport chain and chemiosmosis for ATP synthesis
        3. The Calvin cycle (light-independent reactions) and the process of carbon fixation by RuBisCO
        4. The role of chlorophyll a, chlorophyll b, and accessory pigments like carotenoids
        5. How environmental factors like light intensity, CO2 concentration, and temperature affect photosynthesis rate
        6. The importance of photosynthesis for life on Earth and its role in the carbon cycle
        7. C3, C4, and CAM photosynthesis adaptations in different plant species""",
        """You are a senior software architect with 15 years of experience. Design a complete microservices architecture for a large-scale e-commerce platform that handles millions of users. Your design should include:
        1. Service breakdown with detailed responsibilities: User service (authentication, profiles, preferences), Product catalog service (search, filtering, recommendations), Inventory service (stock management, warehouses), Order service (cart, checkout, order history), Payment service (multiple providers, refunds), Notification service (email, SMS, push)
        2. Database choices for each service with justification (PostgreSQL vs MongoDB vs Redis)
        3. Inter-service communication patterns: synchronous REST/gRPC vs asynchronous message queues
        4. API gateway design with rate limiting, authentication, and request routing
        5. Caching strategy with Redis for sessions, product data, and search results
        6. Message queue architecture with RabbitMQ or Kafka for event-driven communication
        7. Kubernetes deployment with horizontal pod autoscaling, health checks, and rolling updates
        8. CI/CD pipeline with GitHub Actions, testing stages, and blue-green deployments""",
    ]

    # Use only the requested number of prompts
    test_prompts = (prompts * ((num_prompts // len(prompts)) + 1))[:num_prompts]

    print(f"\n{'=' * 60}")
    print("vllm-mlx Performance Benchmark")
    print(f"{'=' * 60}")

    # Hardware info table
    hw_table = [
        ["Model", model_name],
        ["Hardware", f"{hw.chip_name} ({hw.total_memory_gb:.0f} GB)"],
        ["Memory Bandwidth", f"{hw.memory_bandwidth_gbs} GB/s"],
        ["GPU Cores", hw.gpu_cores],
        ["Prompts", num_prompts],
        ["Max Tokens", max_tokens],
        ["Temperature", temperature],
    ]
    print(tabulate(hw_table, tablefmt="plain"))
    print(f"{'=' * 60}\n")

    # Initialize resource monitor
    monitor = ResourceMonitor()
    monitor.start()

    # Load model
    print(f"Loading model: {model_name}...")
    load_start = time.perf_counter()
    model, tokenizer = load_model_with_fallback(model_name)
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.2f}s\n")

    # Show prompt length distribution
    prompt_lengths = [len(tokenizer.encode(p)) for p in test_prompts]
    short = sum(1 for length in prompt_lengths if length < 20)
    medium = sum(1 for length in prompt_lengths if 20 <= length < 100)
    long_p = sum(1 for length in prompt_lengths if length >= 100)
    print("Prompt Distribution:")
    dist_data = [
        ["Short (<20 tokens)", short],
        ["Medium (20-100)", medium],
        ["Long (100+)", long_p],
        ["Total input tokens", sum(prompt_lengths)],
    ]
    print(tabulate(dist_data, tablefmt="plain"))
    print()

    # Warmup runs
    if warmup_runs > 0:
        print(f"Running {warmup_runs} warmup run(s)...")
        for i in range(warmup_runs):
            benchmark_single_prompt(
                model, tokenizer, "Hello, how are you?", max_tokens=20
            )

        # Show model memory after warmup (MLX uses lazy evaluation)
        mlx_info = get_mlx_memory_info(reset_peak=True)
        if mlx_info and mlx_info.get("peak_memory_gb", 0) > 0:
            print(f"Model memory: {mlx_info['peak_memory_gb']:.2f} GB (MLX peak)")
        print("Warmup complete.\n")

    # Main benchmark runs
    results: list[BenchmarkResult] = []
    overall_start = time.perf_counter()

    run_data = []
    for i, prompt in enumerate(test_prompts, 1):
        result = benchmark_single_prompt(
            model, tokenizer, prompt, max_tokens, temperature
        )

        if result:
            results.append(result)
            run_data.append(
                [
                    i,
                    result.prompt_tokens,
                    result.generated_tokens,
                    f"{result.ttft * 1000:.1f}",
                    f"{result.generation_tps:.1f}",
                ]
            )
            # Sample resources after each run
            monitor.sample()

    overall_time = time.perf_counter() - overall_start

    # Print per-run results table
    print("Per-Run Results:")
    print(
        tabulate(
            run_data,
            headers=["Run", "Input", "Output", "TTFT (ms)", "Gen TPS"],
            tablefmt="simple",
        )
    )
    print()

    if not results:
        print("No successful benchmark runs!")
        return None

    # Calculate summary statistics
    ttfts = [r.ttft for r in results]
    tpots = [r.tpot for r in results if r.tpot > 0]
    latencies = [r.total_time for r in results]
    gen_tps = [r.generation_tps for r in results if r.generation_tps > 0]
    proc_tps = [r.processing_tps for r in results if r.processing_tps > 0]

    total_prompt_tokens = sum(r.prompt_tokens for r in results)
    total_generated_tokens = sum(r.generated_tokens for r in results)
    total_tokens = total_prompt_tokens + total_generated_tokens

    summary = BenchmarkSummary(
        model_name=model_name,
        num_runs=len(results),
        total_prompt_tokens=total_prompt_tokens,
        total_generated_tokens=total_generated_tokens,
        total_time=overall_time,
        ttft_mean=statistics.mean(ttfts),
        ttft_min=min(ttfts),
        ttft_max=max(ttfts),
        ttft_p50=calculate_percentile(ttfts, 50),
        ttft_p95=calculate_percentile(ttfts, 95),
        tpot_mean=statistics.mean(tpots) if tpots else 0,
        tpot_min=min(tpots) if tpots else 0,
        tpot_max=max(tpots) if tpots else 0,
        generation_tps_mean=statistics.mean(gen_tps) if gen_tps else 0,
        generation_tps_max=max(gen_tps) if gen_tps else 0,
        processing_tps_mean=statistics.mean(proc_tps) if proc_tps else 0,
        latency_mean=statistics.mean(latencies),
        latency_min=min(latencies),
        latency_max=max(latencies),
        latency_p50=calculate_percentile(latencies, 50),
        latency_p95=calculate_percentile(latencies, 95),
        total_throughput_tps=total_tokens / overall_time,
        requests_per_second=len(results) / overall_time,
        hardware_chip=hw.chip_name,
        hardware_memory_gb=hw.total_memory_gb,
        hardware_bandwidth_gbs=hw.memory_bandwidth_gbs,
        resources=monitor.get_summary(),
    )

    return summary


# =============================================================================
# MLLM Benchmark Functions
# =============================================================================

# MLLM model detection patterns
MLLM_PATTERNS = [
    "-VL-",
    "-VL/",
    "VL-",
    "llava",
    "LLaVA",
    "idefics",
    "Idefics",
    "paligemma",
    "PaliGemma",
    "pixtral",
    "Pixtral",
    "molmo",
    "Molmo",
    "phi3-vision",
    "phi-3-vision",
    "cogvlm",
    "CogVLM",
    "internvl",
    "InternVL",
    "deepseek-vl",
    "DeepSeek-VL",
]

# Test image URL (Yellow Labrador from Wikimedia Commons)
MLLM_TEST_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg"
MLLM_TEST_IMAGE_URLS = [
    MLLM_TEST_IMAGE_URL,
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/640px-PNG_transparency_demonstration_1.png",
]


def is_mllm_model(model_name: str) -> bool:
    """Check if model name indicates a multimodal language model."""
    model_lower = model_name.lower()
    for pattern in MLLM_PATTERNS:
        if pattern.lower() in model_lower:
            return True
    return False


@dataclass
class MLLMBenchmarkResult:
    """Result from a single MLLM benchmark run."""

    resolution: str
    width: int
    height: int
    pixels: int
    time_seconds: float
    tokens_generated: int
    tokens_per_second: float
    response_preview: str
    # Resource metrics for this run
    memory_gb: float = 0.0
    mlx_memory_gb: float = 0.0


def download_test_image(url: str, timeout: int = 30) -> Image.Image:
    """Download image from URL and return PIL Image."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    response = requests.get(url, timeout=timeout, headers=headers)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content))


def resize_image(img: Image.Image, width: int, height: int) -> Image.Image:
    """Resize image to specified dimensions."""
    return img.resize((width, height), Image.Resampling.LANCZOS)


def image_to_base64(img: Image.Image, format: str = "JPEG") -> str:
    """Convert PIL Image to base64 data URL."""
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    buffer = io.BytesIO()
    img.save(buffer, format=format, quality=85)
    b64 = base64.b64encode(buffer.getvalue()).decode()
    mime = "image/jpeg" if format == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


def benchmark_mllm_resolution(
    model,
    processor,
    config,
    base_image: Image.Image,
    width: int,
    height: int,
    max_tokens: int = 256,
    warmup: bool = False,
) -> MLLMBenchmarkResult:
    """Run MLLM benchmark for a specific resolution."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    # Reset MLX peak memory before this run
    reset_mlx_peak_memory()

    # Resize image
    img = resize_image(base_image, width, height)

    # Save to temp file for mlx_vlm
    import tempfile

    temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(temp_file.name, "JPEG", quality=85)
    image_path = temp_file.name

    resolution_name = f"{width}x{height}"
    pixels = width * height

    if not warmup:
        print(f"  {resolution_name:>10} | {pixels:>12,} |", end=" ", flush=True)

    # Apply chat template
    prompt = "What animal is in this image? Describe it briefly."
    try:
        formatted_prompt = apply_chat_template(
            processor,
            config,
            prompt,
            num_images=1,
        )
    except Exception:
        formatted_prompt = prompt

    # Generate
    start_time = time.perf_counter()
    result = generate(
        model,
        processor,
        formatted_prompt,
        [image_path],
        max_tokens=max_tokens,
        temp=0.7,
        verbose=False,
    )
    elapsed = time.perf_counter() - start_time

    # Extract text
    if hasattr(result, "text"):
        text = result.text
        tokens = getattr(result, "generation_tokens", len(text.split()))
    else:
        text = str(result)
        tokens = len(text.split())

    tps = tokens / elapsed if elapsed > 0 else 0

    # Get memory metrics
    mlx_info = get_mlx_memory_info()
    process_mem = get_process_memory()

    if not warmup:
        mem_str = f"{mlx_info.get('peak_memory_gb', 0):.1f} GB" if mlx_info else "-"
        print(f"{elapsed:>6.2f}s | {tokens:>6} | {tps:>10.1f} tok/s | {mem_str}")

    # Cleanup
    import os

    os.unlink(image_path)

    return MLLMBenchmarkResult(
        resolution=resolution_name,
        width=width,
        height=height,
        pixels=pixels,
        time_seconds=elapsed,
        tokens_generated=tokens,
        tokens_per_second=tps,
        response_preview=text[:150] + "..." if len(text) > 150 else text,
        memory_gb=process_mem,
        mlx_memory_gb=mlx_info.get("peak_memory_gb", 0.0),
    )


def run_mllm_benchmark(
    model_name: str,
    quick: bool = False,
    max_tokens: int = 256,
    warmup_runs: int = 1,
) -> list[MLLMBenchmarkResult]:
    """
    Run MLLM benchmark across multiple image resolutions.

    Args:
        model_name: HuggingFace model name
        quick: If True, test only 4 resolutions
        max_tokens: Max tokens to generate
        warmup_runs: Number of warmup runs

    Returns:
        List of MLLMBenchmarkResult
    """
    from mlx_vlm import load
    from mlx_vlm.utils import load_config

    from vllm_mlx.optimizations import detect_hardware

    # Detect hardware
    hw = detect_hardware()

    # Define resolutions
    if quick:
        resolutions = [
            (224, 224),
            (448, 448),
            (768, 768),
            (1024, 1024),
        ]
    else:
        resolutions = [
            (224, 224),
            (336, 336),
            (448, 448),
            (512, 512),
            (672, 672),
            (768, 768),
            (896, 896),
            (1024, 1024),
            (1280, 720),
            (1920, 1080),
        ]

    print(f"\n{'=' * 70}")
    print("vllm-mlx MLLM Performance Benchmark")
    print(f"{'=' * 70}")

    # Info table
    info_table = [
        ["Model", model_name],
        ["Hardware", f"{hw.chip_name} ({hw.total_memory_gb:.0f} GB)"],
        ["Test Image", "Yellow Labrador (Wikimedia Commons)"],
        ["Resolutions", len(resolutions)],
        ["Max Tokens", max_tokens],
    ]
    print(tabulate(info_table, tablefmt="plain"))
    print(f"{'=' * 70}\n")

    # Load model
    print(f"Loading MLLM model: {model_name}...")
    load_start = time.perf_counter()
    model, processor = load(model_name)
    config = load_config(model_name)
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.2f}s\n")

    # Download test image
    print("Downloading test image...")
    try:
        base_image = download_test_image(MLLM_TEST_IMAGE_URL)
        print(f"  Original size: {base_image.size[0]}x{base_image.size[1]}\n")
    except Exception as e:
        print(f"Error downloading image: {e}")
        return []

    # Warmup
    if warmup_runs > 0:
        print(f"Running {warmup_runs} warmup run(s)...")
        for _ in range(warmup_runs):
            benchmark_mllm_resolution(
                model, processor, config, base_image, 224, 224, max_tokens, warmup=True
            )

        # Show model memory after warmup (MLX uses lazy evaluation)
        mlx_info = get_mlx_memory_info(reset_peak=True)
        if mlx_info and mlx_info.get("peak_memory_gb", 0) > 0:
            print(f"Model memory: {mlx_info['peak_memory_gb']:.2f} GB (MLX peak)")
        print("Warmup complete.\n")

    # Run benchmarks
    print("-" * 80)
    print(
        f"  {'Resolution':>10} | {'Pixels':>12} | {'Time':>6} | {'Tokens':>6} | {'Speed':>14} | {'Memory':>8}"
    )
    print("-" * 80)

    results = []
    for width, height in resolutions:
        try:
            result = benchmark_mllm_resolution(
                model, processor, config, base_image, width, height, max_tokens
            )
            results.append(result)
        except Exception as e:
            print(f"  Error at {width}x{height}: {e}")

    return results


def print_mllm_summary(results: list[MLLMBenchmarkResult], model_name: str):
    """Print MLLM benchmark summary."""
    if not results:
        print("No results to display.")
        return

    print(f"\n{'=' * 80}")
    print("MLLM BENCHMARK RESULTS")
    print(f"{'=' * 80}\n")

    # Results table
    table_data = []
    for r in results:
        table_data.append(
            [
                r.resolution,
                f"{r.pixels:,}",
                f"{r.time_seconds:.2f}s",
                r.tokens_generated,
                f"{r.tokens_per_second:.1f}",
                (
                    f"{r.pixels / r.time_seconds / 1000:.1f}K"
                    if r.time_seconds > 0
                    else "N/A"
                ),
                f"{r.mlx_memory_gb:.2f}" if r.mlx_memory_gb > 0 else "-",
            ]
        )

    headers = [
        "Resolution",
        "Pixels",
        "Time",
        "Tokens",
        "Tok/s",
        "Pixels/s",
        "Mem (GB)",
    ]
    print(tabulate(table_data, headers=headers, tablefmt="simple"))

    # Summary stats
    total_time = sum(r.time_seconds for r in results)
    total_tokens = sum(r.tokens_generated for r in results)
    avg_tps = total_tokens / total_time if total_time > 0 else 0
    peak_memory = max(r.mlx_memory_gb for r in results) if results else 0

    print("-" * 80)
    print(f"Total Time:      {total_time:.2f}s")
    print(f"Total Tokens:    {total_tokens}")
    print(f"Average Tok/s:   {avg_tps:.1f}")
    if peak_memory > 0:
        print(f"Peak Memory:     {peak_memory:.2f} GB")

    fastest = min(results, key=lambda r: r.time_seconds)
    slowest = max(results, key=lambda r: r.time_seconds)

    print(f"\nFastest:  {fastest.resolution} ({fastest.time_seconds:.2f}s)")
    print(f"Slowest:  {slowest.resolution} ({slowest.time_seconds:.2f}s)")
    print(f"Slowdown: {slowest.time_seconds / fastest.time_seconds:.1f}x")
    print(f"{'=' * 80}")


# =============================================================================
# Video Benchmark Functions
# =============================================================================


@dataclass
class VideoBenchmarkResult:
    """Result from a single video benchmark run."""

    config_name: str
    fps: float
    max_frames: int
    frames_extracted: int
    video_duration: float
    time_seconds: float
    prompt_tokens: int
    completion_tokens: int
    tokens_per_second: float
    response_preview: str
    # Resource metrics
    memory_gb: float = 0.0
    mlx_memory_gb: float = 0.0


def create_test_video(
    duration: float = 10.0,
    fps: float = 30.0,
    width: int = 640,
    height: int = 480,
) -> str:
    """Create a synthetic test video with colored frames and text."""
    if cv2 is None:
        raise ImportError(
            "opencv-python is required for video benchmarks. "
            "Install with: pip install 'rapid-mlx[vision]'"
        )
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_file.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_file.name, fourcc, fps, (width, height))

    total_frames = int(duration * fps)

    scenes = [
        ((255, 0, 0), "Blue Scene"),
        ((0, 255, 0), "Green Scene"),
        ((0, 0, 255), "Red Scene"),
        ((255, 255, 0), "Cyan Scene"),
        ((255, 0, 255), "Magenta Scene"),
        ((0, 255, 255), "Yellow Scene"),
    ]

    frames_per_scene = total_frames // len(scenes)

    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        scene_idx = min(i // frames_per_scene, len(scenes) - 1)
        color, scene_name = scenes[scene_idx]
        frame[:] = color

        cv2.putText(
            frame,
            scene_name,
            (width // 4, height // 2 - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            3,
        )
        cv2.putText(
            frame,
            f"Frame {i}/{total_frames}",
            (width // 4, height // 2 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        out.write(frame)

    out.release()
    return temp_file.name


def download_video(url: str, timeout: int = 120) -> str:
    """Download video from URL and return local path."""
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

    print(f"  Downloading video from: {url[:60]}...")
    response = requests.get(url, timeout=timeout, headers=headers, stream=True)
    response.raise_for_status()

    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    for chunk in response.iter_content(chunk_size=8192):
        temp_file.write(chunk)
    temp_file.close()

    file_size = Path(temp_file.name).stat().st_size
    print(f"  Downloaded: {file_size / 1024 / 1024:.1f} MB")

    return temp_file.name


def get_video_info(video_path: str) -> dict:
    """Get information about a video file."""
    if cv2 is None:
        raise ImportError(
            "opencv-python is required for video benchmarks. "
            "Install with: pip install 'rapid-mlx[vision]'"
        )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Cannot open video"}

    info = {
        "path": video_path,
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0

    cap.release()
    return info


def benchmark_video_config(
    model,
    video_path: str,
    fps: float,
    max_frames: int,
    config_name: str,
    video_info: dict,
    max_tokens: int = 150,
    warmup: bool = False,
) -> VideoBenchmarkResult:
    """Run a single video benchmark configuration."""

    # Reset MLX peak memory before this run
    reset_mlx_peak_memory()

    if not warmup:
        print(f"  {config_name:>25} |", end=" ", flush=True)

    start_time = time.perf_counter()

    output = model.generate(
        prompt="Describe what happens in this video. What do you see?",
        videos=[video_path],
        video_fps=fps,
        video_max_frames=max_frames,
        max_tokens=max_tokens,
        temperature=0.7,
    )

    elapsed = time.perf_counter() - start_time

    prompt_tokens = output.prompt_tokens
    completion_tokens = output.completion_tokens
    tps = completion_tokens / elapsed if elapsed > 0 else 0

    # Estimate frames extracted
    duration = video_info["duration"]
    frames_from_fps = int(duration * fps)
    frames_extracted = min(frames_from_fps, max_frames, video_info["total_frames"])

    # Get memory metrics
    mlx_info = get_mlx_memory_info()
    process_mem = get_process_memory()

    if not warmup:
        mem_str = f"{mlx_info.get('peak_memory_gb', 0):.1f} GB" if mlx_info else "-"
        print(
            f"{frames_extracted:>2} frames | {elapsed:>5.2f}s | {completion_tokens:>3} tok | {tps:>6.1f} tok/s | {mem_str}"
        )

    return VideoBenchmarkResult(
        config_name=config_name,
        fps=fps,
        max_frames=max_frames,
        frames_extracted=frames_extracted,
        video_duration=duration,
        time_seconds=elapsed,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tokens_per_second=tps,
        response_preview=(
            output.text[:100] + "..." if len(output.text) > 100 else output.text
        ),
        memory_gb=process_mem,
        mlx_memory_gb=mlx_info.get("peak_memory_gb", 0.0),
    )


def run_video_benchmark(
    model_name: str,
    video_url: str = None,
    video_path: str = None,
    quick: bool = False,
    max_tokens: int = 150,
    warmup_runs: int = 1,
) -> list[VideoBenchmarkResult]:
    """
    Run video benchmark across multiple frame configurations.

    Args:
        model_name: HuggingFace MLLM model name
        video_url: URL to download video from
        video_path: Local video file path
        quick: If True, test only 3 configurations
        max_tokens: Max tokens to generate
        warmup_runs: Number of warmup runs

    Returns:
        List of VideoBenchmarkResult
    """
    from vllm_mlx.models.mllm import MLXMultimodalLM
    from vllm_mlx.optimizations import detect_hardware

    # Detect hardware
    hw = detect_hardware()

    # Define configurations: (name, fps, max_frames)
    if quick:
        configs = [
            ("4 frames @ 1fps", 1.0, 4),
            ("8 frames @ 2fps", 2.0, 8),
            ("16 frames @ 2fps", 2.0, 16),
        ]
    else:
        configs = [
            # Varying frame counts (from 2 to 64 frames)
            # Note: 96+ frames causes GPU timeout on most hardware
            ("2 frames @ 0.5fps", 0.5, 2),
            ("4 frames @ 1fps", 1.0, 4),
            ("6 frames @ 1fps", 1.0, 6),
            ("8 frames @ 2fps", 2.0, 8),
            ("12 frames @ 2fps", 2.0, 12),
            ("16 frames @ 2fps", 2.0, 16),
            ("24 frames @ 4fps", 4.0, 24),
            ("32 frames @ 4fps", 4.0, 32),
            ("48 frames @ 8fps", 8.0, 48),
            ("64 frames @ 8fps", 8.0, 64),
        ]

    print(f"\n{'=' * 70}")
    print("vllm-mlx Video Performance Benchmark")
    print(f"{'=' * 70}")

    # Info table
    info_table = [
        ["Model", model_name],
        ["Hardware", f"{hw.chip_name} ({hw.total_memory_gb:.0f} GB)"],
        ["Configurations", len(configs)],
        ["Max Tokens", max_tokens],
    ]
    print(tabulate(info_table, tablefmt="plain"))
    print(f"{'=' * 70}\n")

    # Load model
    print(f"Loading MLLM model: {model_name}...")
    load_start = time.perf_counter()
    model = MLXMultimodalLM(model_name)
    model.load()
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.2f}s\n")

    # Get or create video
    if video_path and Path(video_path).exists():
        print(f"Using local video: {video_path}")
    elif video_url:
        video_path = download_video(video_url)
    else:
        print("Downloading default test video (Big Buck Bunny 10s)...")
        video_path = download_video(DEFAULT_VIDEO_URL)

    video_info = get_video_info(video_path)
    print(
        f"\nVideo: {video_info['width']}x{video_info['height']}, "
        f"{video_info['duration']:.1f}s, {video_info['fps']:.1f} fps, "
        f"{video_info['total_frames']} frames\n"
    )

    # Warmup
    if warmup_runs > 0:
        print(f"Running {warmup_runs} warmup run(s)...")
        for _ in range(warmup_runs):
            benchmark_video_config(
                model, video_path, 1.0, 4, "warmup", video_info, max_tokens, warmup=True
            )

        # Show model memory after warmup (MLX uses lazy evaluation)
        mlx_info = get_mlx_memory_info(reset_peak=True)
        if mlx_info and mlx_info.get("peak_memory_gb", 0) > 0:
            print(f"Model memory: {mlx_info['peak_memory_gb']:.2f} GB (MLX peak)")
        print("Warmup complete.\n")

    # Run benchmarks
    print("-" * 85)
    print(
        f"  {'Configuration':>25} | {'Frames':>6} | {'Time':>6} | {'Tokens':>4} | {'Speed':>10} | {'Memory':>8}"
    )
    print("-" * 85)

    results = []
    for config_name, fps, max_frames in configs:
        try:
            result = benchmark_video_config(
                model, video_path, fps, max_frames, config_name, video_info, max_tokens
            )
            results.append(result)
        except Exception as e:
            print(f"  Error with {config_name}: {e}")

    return results


def print_video_summary(results: list[VideoBenchmarkResult], model_name: str):
    """Print video benchmark summary."""
    if not results:
        print("No results to display.")
        return

    print(f"\n{'=' * 85}")
    print("VIDEO BENCHMARK RESULTS")
    print(f"{'=' * 85}\n")

    # Results table
    table_data = []
    for r in sorted(results, key=lambda x: x.frames_extracted):
        table_data.append(
            [
                r.config_name,
                r.frames_extracted,
                f"{r.time_seconds:.2f}s",
                r.completion_tokens,
                f"{r.tokens_per_second:.1f}",
                f"{r.mlx_memory_gb:.2f}" if r.mlx_memory_gb > 0 else "-",
            ]
        )

    headers = ["Configuration", "Frames", "Time", "Tokens", "Tok/s", "Mem (GB)"]
    print(tabulate(table_data, headers=headers, tablefmt="simple"))

    # Summary stats
    total_time = sum(r.time_seconds for r in results)
    total_tokens = sum(r.completion_tokens for r in results)
    avg_tps = total_tokens / total_time if total_time > 0 else 0
    peak_memory = max(r.mlx_memory_gb for r in results) if results else 0

    print("-" * 85)
    print(f"Total Time:      {total_time:.2f}s")
    print(f"Total Tokens:    {total_tokens}")
    print(f"Average Tok/s:   {avg_tps:.1f}")
    if peak_memory > 0:
        print(f"Peak Memory:     {peak_memory:.2f} GB")

    fastest = min(results, key=lambda r: r.time_seconds)
    slowest = max(results, key=lambda r: r.time_seconds)
    most_frames = max(results, key=lambda r: r.frames_extracted)

    print(
        f"\nFastest:     {fastest.config_name} ({fastest.time_seconds:.2f}s, {fastest.tokens_per_second:.1f} tok/s)"
    )
    print(
        f"Slowest:     {slowest.config_name} ({slowest.time_seconds:.2f}s, {slowest.tokens_per_second:.1f} tok/s)"
    )
    print(
        f"Most Frames: {most_frames.config_name} ({most_frames.frames_extracted} frames)"
    )
    print(f"{'=' * 85}")


# =============================================================================
# LLM Benchmark Summary
# =============================================================================


def print_summary(summary: BenchmarkSummary):
    """Print a formatted summary of benchmark results using tabulate."""
    print(f"\n{'=' * 60}")
    print("BENCHMARK RESULTS")
    print(f"{'=' * 60}\n")

    # Overview table
    overview_data = [
        ["Model", summary.model_name],
        ["Hardware", f"{summary.hardware_chip} ({summary.hardware_memory_gb:.0f} GB)"],
        ["Total Runs", summary.num_runs],
        ["Input Tokens", f"{summary.total_prompt_tokens:,}"],
        ["Output Tokens", f"{summary.total_generated_tokens:,}"],
        ["Total Time", f"{summary.total_time:.2f}s"],
    ]
    print(tabulate(overview_data, tablefmt="plain"))
    print()

    # Main metrics table
    metrics_data = [
        [
            "TTFT (Time to First Token)",
            f"{summary.ttft_mean * 1000:.1f} ms",
            f"{summary.ttft_p95 * 1000:.1f} ms",
        ],
        [
            "TPOT (Time Per Output Token)",
            f"{summary.tpot_mean * 1000:.2f} ms",
            f"{summary.tpot_max * 1000:.2f} ms",
        ],
        [
            "Generation Speed",
            f"{summary.generation_tps_mean:.1f} tok/s",
            f"{summary.generation_tps_max:.1f} tok/s",
        ],
        ["Processing Speed", f"{summary.processing_tps_mean:.1f} tok/s", "-"],
        [
            "Latency (per request)",
            f"{summary.latency_mean:.2f}s",
            f"{summary.latency_p95:.2f}s",
        ],
    ]
    print("Performance Metrics:")
    print(
        tabulate(
            metrics_data,
            headers=["Metric", "Mean", "P95/Max"],
            tablefmt="simple",
        )
    )
    print()

    # Throughput table
    throughput_data = [
        ["Total Throughput", f"{summary.total_throughput_tps:.1f} tok/s"],
        ["Requests/Second", f"{summary.requests_per_second:.2f} req/s"],
    ]
    print("Throughput:")
    print(tabulate(throughput_data, tablefmt="plain"))
    print()

    # Resource metrics
    res = summary.resources
    if res.process_memory_gb > 0 or res.mlx_peak_memory_gb > 0:
        print("Resource Usage:")
        resource_data = []

        if res.process_memory_gb > 0:
            resource_data.append(
                ["Process Memory (peak)", f"{res.process_memory_gb:.2f} GB"]
            )

        if res.mlx_peak_memory_gb > 0:
            resource_data.append(
                ["MLX Peak Memory", f"{res.mlx_peak_memory_gb:.2f} GB"]
            )

        if res.mlx_cache_gb > 0:
            resource_data.append(["MLX Cache Memory", f"{res.mlx_cache_gb:.2f} GB"])

        if res.system_memory_total_gb > 0:
            used_pct = (res.system_memory_used_gb / res.system_memory_total_gb) * 100
            resource_data.append(
                [
                    "System Memory",
                    f"{res.system_memory_used_gb:.1f} / {res.system_memory_total_gb:.0f} GB ({used_pct:.0f}%)",
                ]
            )

        print(tabulate(resource_data, tablefmt="plain"))
        print()

    print(f"{'=' * 60}")


def main():
    """Run the benchmark."""
    parser = argparse.ArgumentParser(
        description="vllm-mlx Performance Benchmark (LLM, MLLM Image & Video)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # LLM benchmark
    vllm-mlx-bench --model mlx-community/Llama-3.2-1B-Instruct-4bit
    vllm-mlx-bench --model mlx-community/Llama-3.2-3B-Instruct-4bit --prompts 10

    # MLLM image benchmark (auto-detected)
    vllm-mlx-bench --model mlx-community/Qwen3-VL-4B-Instruct-3bit
    vllm-mlx-bench --model mlx-community/Qwen3-VL-4B-Instruct-3bit --quick

    # MLLM video benchmark
    vllm-mlx-bench --model mlx-community/Qwen3-VL-4B-Instruct-3bit --video
    vllm-mlx-bench --model mlx-community/Qwen3-VL-4B-Instruct-3bit --video --quick
    vllm-mlx-bench --model mlx-community/Qwen3-VL-4B-Instruct-3bit --video --video-url https://example.com/video.mp4

    # Force MLLM mode
    vllm-mlx-bench --model custom-vision-model --mllm
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (HuggingFace model name or local path)",
    )
    parser.add_argument(
        "--prompts",
        type=int,
        default=5,
        help="Number of prompts to benchmark for LLM (default: 5)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per prompt (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup runs (default: 1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for JSON results",
    )
    parser.add_argument(
        "--mllm",
        action="store_true",
        help="Force MLLM benchmark mode (auto-detected by default)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick benchmark with fewer configurations",
    )
    # Video benchmark arguments
    parser.add_argument(
        "--video",
        action="store_true",
        help="Run video benchmark instead of image benchmark (for MLLM models)",
    )
    parser.add_argument(
        "--video-url",
        type=str,
        default=None,
        help="URL of video to use for benchmark (default: Big Buck Bunny 10s)",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Local path to video file for benchmark",
    )

    args = parser.parse_args()

    # Determine if MLLM model
    run_mllm = args.mllm or is_mllm_model(args.model)

    if args.video:
        # Video Benchmark
        results = run_video_benchmark(
            model_name=args.model,
            video_url=args.video_url,
            video_path=args.video_path,
            quick=args.quick,
            max_tokens=args.max_tokens,
            warmup_runs=args.warmup,
        )

        if results:
            print_video_summary(results, args.model)

            # Save to JSON if requested
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(
                        {
                            "type": "video",
                            "model": args.model,
                            "test_video": args.video_url
                            or args.video_path
                            or "Big Buck Bunny 10s",
                            "results": [
                                {
                                    "config_name": r.config_name,
                                    "fps": r.fps,
                                    "max_frames": r.max_frames,
                                    "frames_extracted": r.frames_extracted,
                                    "video_duration": r.video_duration,
                                    "time_seconds": r.time_seconds,
                                    "prompt_tokens": r.prompt_tokens,
                                    "completion_tokens": r.completion_tokens,
                                    "tokens_per_second": r.tokens_per_second,
                                    "response_preview": r.response_preview,
                                }
                                for r in results
                            ],
                        },
                        f,
                        indent=2,
                    )
                print(f"\nResults saved to: {args.output}")

    elif run_mllm:
        # MLLM Image Benchmark
        results = run_mllm_benchmark(
            model_name=args.model,
            quick=args.quick,
            max_tokens=args.max_tokens,
            warmup_runs=args.warmup,
        )

        if results:
            print_mllm_summary(results, args.model)

            # Save to JSON if requested
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(
                        {
                            "type": "mllm_image",
                            "model": args.model,
                            "test_image": "Yellow Labrador (Wikimedia Commons)",
                            "results": [
                                {
                                    "resolution": r.resolution,
                                    "width": r.width,
                                    "height": r.height,
                                    "pixels": r.pixels,
                                    "time_seconds": r.time_seconds,
                                    "tokens_generated": r.tokens_generated,
                                    "tokens_per_second": r.tokens_per_second,
                                    "response_preview": r.response_preview,
                                }
                                for r in results
                            ],
                        },
                        f,
                        indent=2,
                    )
                print(f"\nResults saved to: {args.output}")
    else:
        # LLM Benchmark
        summary = run_benchmark(
            model_name=args.model,
            num_prompts=args.prompts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            warmup_runs=args.warmup,
        )

        if summary:
            print_summary(summary)

            # Save to JSON if requested
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(
                        {
                            "type": "llm",
                            "model": summary.model_name,
                            "hardware": {
                                "chip": summary.hardware_chip,
                                "memory_gb": summary.hardware_memory_gb,
                                "bandwidth_gbs": summary.hardware_bandwidth_gbs,
                            },
                            "num_runs": summary.num_runs,
                            "total_prompt_tokens": summary.total_prompt_tokens,
                            "total_generated_tokens": summary.total_generated_tokens,
                            "total_time_seconds": summary.total_time,
                            "ttft_ms": {
                                "mean": summary.ttft_mean * 1000,
                                "min": summary.ttft_min * 1000,
                                "max": summary.ttft_max * 1000,
                                "p50": summary.ttft_p50 * 1000,
                                "p95": summary.ttft_p95 * 1000,
                            },
                            "tpot_ms": {
                                "mean": summary.tpot_mean * 1000,
                                "min": summary.tpot_min * 1000,
                                "max": summary.tpot_max * 1000,
                            },
                            "tokens_per_second": {
                                "generation_mean": summary.generation_tps_mean,
                                "generation_max": summary.generation_tps_max,
                                "processing_mean": summary.processing_tps_mean,
                                "total_throughput": summary.total_throughput_tps,
                            },
                            "latency_seconds": {
                                "mean": summary.latency_mean,
                                "min": summary.latency_min,
                                "max": summary.latency_max,
                                "p50": summary.latency_p50,
                                "p95": summary.latency_p95,
                            },
                            "requests_per_second": summary.requests_per_second,
                        },
                        f,
                        indent=2,
                    )
                print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
