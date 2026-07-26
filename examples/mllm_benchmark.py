#!/usr/bin/env python3
"""
MLLM Benchmark Script for rapid-mlx

Tests Multimodal Language Models with real images of dogs from Wikimedia Commons
at different resolutions and measures performance metrics.

Usage:
    # Start the MLLM server first:
    python -m vllm_mlx.server --model mlx-community/Qwen3-VL-4B-Instruct-3bit --port 8000

    # Run benchmark:
    python examples/mllm_benchmark.py

    # Or specify server URL:
    python examples/mllm_benchmark.py --server-url http://localhost:8000
"""

import argparse
import base64
import io
import json
import time
from dataclasses import dataclass

import requests
from PIL import Image
from tabulate import tabulate

# Test images from Wikimedia Commons - Dogs!
# Using different dog images at various original resolutions
TEST_IMAGES = {
    "golden_retriever": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Golden_Retriever_Dukedestiny01_dbread_loose.jpg/1280px-Golden_Retriever_Dukedestiny01_dread_loose.jpg",
        "description": "Golden Retriever",
        "fallback": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Golden_Retriever_Dukedestiny01_dread_loose.jpg/800px-Golden_Retriever_Dukedestiny01_dread_loose.jpg",
    },
    "german_shepherd": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/German_Shepherd_-_DSC_0346_%2810096362833%29.jpg/1280px-German_Shepherd_-_DSC_0346_%2810096362833%29.jpg",
        "description": "German Shepherd",
    },
    "labrador": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg",
        "description": "Yellow Labrador",
    },
    "beagle": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Beagle_600.jpg/1200px-Beagle_600.jpg",
        "description": "Beagle",
    },
    "husky": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/800px-Camponotus_flavomarginatus_ant.jpg",
        # Better husky image
        "fallback": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Husky_IMG_0921.jpg/800px-Husky_IMG_0921.jpg",
        "description": "Siberian Husky",
    },
}

# Primary test image - a cute dog photo
PRIMARY_DOG_IMAGE = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg"


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    resolution: str
    width: int
    height: int
    pixels: int
    time_seconds: float
    tokens_generated: int
    tokens_per_second: float
    response_preview: str


def download_image(url: str, timeout: int = 30) -> Image.Image:
    """Download image from URL and return PIL Image."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, timeout=timeout, headers=headers)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content))


def resize_image(img: Image.Image, width: int, height: int) -> Image.Image:
    """Resize image to specified dimensions."""
    return img.resize((width, height), Image.Resampling.LANCZOS)


def image_to_base64(img: Image.Image, format: str = "JPEG") -> str:
    """Convert PIL Image to base64 data URL."""
    # Convert RGBA to RGB if needed
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


def run_mllm_request(
    server_url: str,
    image_b64: str,
    prompt: str = "Describe this image in detail. What do you see?",
    max_tokens: int = 256,
    model: str = "default",
) -> tuple[str, float, int]:
    """
    Send an MLLM request to the server.

    Returns:
        (response_text, time_seconds, tokens_generated)
    """
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_b64}},
                ],
            }
        ],
        "max_tokens": max_tokens,
    }

    start_time = time.perf_counter()

    response = requests.post(
        f"{server_url}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )

    elapsed = time.perf_counter() - start_time

    response.raise_for_status()
    data = response.json()

    text = data["choices"][0]["message"]["content"]
    tokens = data.get("usage", {}).get("completion_tokens", len(text.split()))

    return text, elapsed, tokens


def benchmark_resolution(
    server_url: str,
    base_image: Image.Image,
    width: int,
    height: int,
    model: str,
    warmup: bool = False,
) -> BenchmarkResult:
    """Run benchmark for a specific resolution."""

    # Resize image to target resolution
    img = resize_image(base_image, width, height)
    img_b64 = image_to_base64(img)

    resolution_name = f"{width}x{height}"
    pixels = width * height

    if not warmup:
        print(
            f"  Testing {resolution_name:>10} ({pixels:>10,} pixels)...",
            end=" ",
            flush=True,
        )

    # Run request
    text, elapsed, tokens = run_mllm_request(
        server_url=server_url,
        image_b64=img_b64,
        prompt="What animal is in this image? Describe it briefly.",
        model=model,
    )

    tps = tokens / elapsed if elapsed > 0 else 0

    if not warmup:
        print(f"{elapsed:>6.2f}s | {tokens:>3} tokens | {tps:>6.1f} tok/s")

    return BenchmarkResult(
        resolution=resolution_name,
        width=width,
        height=height,
        pixels=pixels,
        time_seconds=elapsed,
        tokens_generated=tokens,
        tokens_per_second=tps,
        response_preview=text[:150] + "..." if len(text) > 150 else text,
    )


def run_benchmark(
    server_url: str = "http://localhost:8000",
    resolutions: list[tuple[int, int]] = None,
    warmup_runs: int = 1,
    image_url: str = None,
) -> list[BenchmarkResult]:
    """
    Run full MLLM benchmark across multiple resolutions.

    Args:
        server_url: URL of the rapid-mlx server
        resolutions: List of (width, height) tuples to test
        warmup_runs: Number of warmup runs before measuring
        image_url: URL of image to use (default: dog from Wikimedia)

    Returns:
        List of BenchmarkResult objects
    """

    # Default resolutions to test (common MLLM input sizes)
    if resolutions is None:
        resolutions = [
            (224, 224),  # Tiny - common MLLM input size
            (336, 336),  # Small - LLaVA default
            (448, 448),  # Medium - Qwen-VL default
            (512, 512),  # Standard
            (672, 672),  # Large
            (768, 768),  # HD-ish
            (896, 896),  # Higher
            (1024, 1024),  # Full HD square
            (1280, 720),  # 720p landscape
            (1920, 1080),  # 1080p landscape
        ]

    # Check server health
    print(f"Connecting to server at {server_url}...")
    try:
        health = requests.get(f"{server_url}/health", timeout=10)
        health.raise_for_status()
        health_data = health.json()
        model_name = health_data.get("model_name", "unknown")
        model_type = health_data.get("model_type", "unknown")
        print(f"Server healthy: {model_name} ({model_type})")
    except Exception as e:
        print(f"Error connecting to server: {e}")
        print("\nMake sure the MLLM server is running:")
        print(
            "  python -m vllm_mlx.server --model mlx-community/Qwen3-VL-4B-Instruct-3bit --port 8000"
        )
        return []

    if model_type not in ("mllm", "vlm"):
        print(f"\nWarning: Server is running a {model_type} model, not an MLLM!")
        print("Please start with an MLLM model like Qwen3-VL or LLaVA")
        return []

    # Download base image
    image_url = image_url or PRIMARY_DOG_IMAGE
    print("\nDownloading test image (dog)...")
    print(f"  URL: {image_url}")

    try:
        base_image = download_image(image_url)
        print(f"  Original size: {base_image.size[0]}x{base_image.size[1]}")
    except Exception as e:
        print(f"Error downloading image: {e}")
        return []

    # Warmup runs
    if warmup_runs > 0:
        print(f"\nRunning {warmup_runs} warmup run(s)...")
        for i in range(warmup_runs):
            benchmark_resolution(
                server_url, base_image, 224, 224, model_name, warmup=True
            )
        print("Warmup complete.")

    # Run benchmarks
    print("\n" + "=" * 70)
    print("MLLM BENCHMARK - Image Resolution Performance")
    print("=" * 70)
    print(f"Model:       {model_name}")
    print("Test Image:  Dog (Yellow Labrador)")
    print(f"Resolutions: {len(resolutions)}")
    print("-" * 70)
    print(
        f"  {'Resolution':>10} | {'Pixels':>12} | {'Time':>7} | {'Tokens':>6} | {'Speed':>10}"
    )
    print("-" * 70)

    results = []
    for width, height in resolutions:
        try:
            result = benchmark_resolution(
                server_url, base_image, width, height, model_name
            )
            results.append(result)
        except Exception as e:
            print(f"  Error at {width}x{height}: {e}")

    return results


def print_results(results: list[BenchmarkResult]):
    """Print benchmark results in a nice table."""

    if not results:
        print("No results to display.")
        return

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)

    # Prepare table data
    table_data = []
    for r in results:
        table_data.append(
            [
                r.resolution,
                f"{r.pixels:,}",
                f"{r.time_seconds:.2f}s",
                r.tokens_generated,
                f"{r.tokens_per_second:.1f}",
                f"{r.pixels / r.time_seconds / 1000:.1f}K"
                if r.time_seconds > 0
                else "N/A",
            ]
        )

    headers = ["Resolution", "Pixels", "Time", "Tokens", "Tok/s", "Pixels/s"]
    print(tabulate(table_data, headers=headers, tablefmt="simple"))

    # Summary stats
    total_time = sum(r.time_seconds for r in results)
    total_tokens = sum(r.tokens_generated for r in results)
    avg_tps = total_tokens / total_time if total_time > 0 else 0

    print("-" * 70)
    print(f"Total Time:      {total_time:.2f}s")
    print(f"Total Tokens:    {total_tokens}")
    print(f"Average Tok/s:   {avg_tps:.1f}")

    # Find best/worst
    fastest = min(results, key=lambda r: r.time_seconds)
    slowest = max(results, key=lambda r: r.time_seconds)

    print(f"\nFastest:  {fastest.resolution} ({fastest.time_seconds:.2f}s)")
    print(f"Slowest:  {slowest.resolution} ({slowest.time_seconds:.2f}s)")
    print(
        f"Slowdown: {slowest.time_seconds / fastest.time_seconds:.1f}x from smallest to largest"
    )


def save_results(results: list[BenchmarkResult], output_path: str):
    """Save benchmark results to JSON file."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_image": "Yellow Labrador from Wikimedia Commons",
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
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="MLLM Benchmark for rapid-mlx - Tests with dog images at different resolutions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start the server first:
    python -m vllm_mlx.server --model mlx-community/Qwen3-VL-4B-Instruct-3bit --port 8000

    # Basic benchmark
    python examples/mllm_benchmark.py

    # Custom server URL
    python examples/mllm_benchmark.py --server-url http://localhost:8001

    # Save results to file
    python examples/mllm_benchmark.py --output mllm_benchmark_results.json

    # Quick test with fewer resolutions
    python examples/mllm_benchmark.py --quick
        """,
    )

    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="URL of the rapid-mlx server",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup runs (default: 1)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with fewer resolutions",
    )

    args = parser.parse_args()

    # Define resolutions
    if args.quick:
        resolutions = [
            (224, 224),
            (448, 448),
            (768, 768),
            (1024, 1024),
        ]
    else:
        resolutions = None  # Use defaults

    # Run benchmark
    results = run_benchmark(
        server_url=args.server_url,
        resolutions=resolutions,
        warmup_runs=args.warmup,
    )

    # Print results
    print_results(results)

    # Save if requested
    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
