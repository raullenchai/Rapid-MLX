#!/usr/bin/env python3
"""
Video Benchmark Script for rapid-mlx

Tests Vision-Language Models with video at different configurations
(FPS, frame count, resolution) and measures performance metrics.

Usage:
    # Direct API benchmark (no server needed):
    python examples/video_benchmark.py --model mlx-community/Qwen3-VL-4B-Instruct-3bit

    # With video URL:
    python examples/video_benchmark.py --video-url https://example.com/video.mp4

    # Quick test:
    python examples/video_benchmark.py --quick
"""

import argparse
import json
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Sample video URLs for testing (free, no copyright)
# Sources: file-examples.com, docs.evostream.com
SAMPLE_VIDEOS = {
    # From file-examples.com - 30 second videos at different resolutions
    "480x270_30s": {
        "url": "https://file-examples.com/storage/feb05093c66764aa00cbc58/2017/04/file_example_MP4_480_1_5MG.mp4",
        "description": "Sample video 480x270 (30s)",
        "resolution": "480x270",
        "duration": 30,
    },
    "640x360_30s": {
        "url": "https://file-examples.com/storage/feb05093c66764aa00cbc58/2017/04/file_example_MP4_640_3MG.mp4",
        "description": "Sample video 640x360 (30s)",
        "resolution": "640x360",
        "duration": 30,
    },
    "1280x720_30s": {
        "url": "https://file-examples.com/storage/feb05093c66764aa00cbc58/2017/04/file_example_MP4_1280_10MG.mp4",
        "description": "Sample video 1280x720 HD (30s)",
        "resolution": "1280x720",
        "duration": 30,
    },
    # From evostream - various content
    "bunny_240p": {
        "url": "https://docs.evostream.com/sample_content/assets/bunny.mp4",
        "description": "Big Buck Bunny 240p",
        "resolution": "424x240",
        "duration": 60,
    },
    "sintel_720p": {
        "url": "https://docs.evostream.com/sample_content/assets/sintel1m720p.mp4",
        "description": "Sintel Trailer 720p",
        "resolution": "1280x720",
        "duration": 60,
    },
    # From test-videos.co.uk
    "big_buck_bunny_10s": {
        "url": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4",
        "description": "Big Buck Bunny (10s, 360p)",
        "resolution": "640x360",
        "duration": 10,
    },
}


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


def create_test_video(
    duration: float = 5.0,
    fps: float = 30.0,
    width: int = 640,
    height: int = 480,
) -> str:
    """
    Create a synthetic test video with colored frames and text.

    Args:
        duration: Video duration in seconds
        fps: Frames per second
        width: Video width
        height: Video height

    Returns:
        Path to created video file
    """
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_file.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_file.name, fourcc, fps, (width, height))

    total_frames = int(duration * fps)

    # Different scenes with colors
    scenes = [
        ((255, 0, 0), "Blue Scene"),  # Blue (BGR)
        ((0, 255, 0), "Green Scene"),  # Green
        ((0, 0, 255), "Red Scene"),  # Red
        ((255, 255, 0), "Cyan Scene"),  # Cyan
        ((255, 0, 255), "Magenta Scene"),  # Magenta
        ((0, 255, 255), "Yellow Scene"),  # Yellow
    ]

    frames_per_scene = total_frames // len(scenes)

    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        scene_idx = min(i // frames_per_scene, len(scenes) - 1)
        color, scene_name = scenes[scene_idx]
        frame[:] = color

        # Add scene name text
        cv2.putText(
            frame,
            scene_name,
            (width // 4, height // 2 - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            3,
        )

        # Add frame counter
        cv2.putText(
            frame,
            f"Frame {i}/{total_frames}",
            (width // 4, height // 2 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        # Add timestamp
        timestamp = i / fps
        cv2.putText(
            frame,
            f"Time: {timestamp:.1f}s",
            (width // 4, height // 2 + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 200, 200),
            2,
        )

        out.write(frame)

    out.release()
    return temp_file.name


def download_video(url: str, timeout: int = 120) -> str:
    """Download video from URL."""
    import requests

    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

    logger.info(f"Downloading video from: {url}")
    response = requests.get(url, timeout=timeout, headers=headers, stream=True)
    response.raise_for_status()

    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    for chunk in response.iter_content(chunk_size=8192):
        temp_file.write(chunk)
    temp_file.close()

    file_size = Path(temp_file.name).stat().st_size
    logger.info(f"Downloaded: {file_size / 1024 / 1024:.1f} MB")

    return temp_file.name


def get_video_info(video_path: str) -> dict:
    """Get information about a video file."""
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


def run_video_benchmark(
    model,
    video_path: str,
    fps: float,
    max_frames: int,
    config_name: str,
    warmup: bool = False,
) -> VideoBenchmarkResult:
    """Run a single video benchmark configuration."""

    video_info = get_video_info(video_path)

    if not warmup:
        print(
            f"  {config_name:>20} | fps={fps:<4} max_frames={max_frames:<3} |",
            end=" ",
            flush=True,
        )

    start_time = time.perf_counter()

    output = model.generate(
        prompt="Describe what happens in this video. What do you see?",
        videos=[video_path],
        video_fps=fps,
        video_max_frames=max_frames,
        max_tokens=150,
        temperature=0.7,
    )

    elapsed = time.perf_counter() - start_time

    prompt_tokens = output.prompt_tokens
    completion_tokens = output.completion_tokens
    tps = completion_tokens / elapsed if elapsed > 0 else 0

    # Count actual frames extracted (approximation)
    duration = video_info["duration"]
    frames_from_fps = int(duration * fps)
    frames_extracted = min(frames_from_fps, max_frames, video_info["total_frames"])

    if not warmup:
        print(
            f"{elapsed:>5.2f}s | {frames_extracted:>2} frames | {completion_tokens:>3} tok | {tps:>5.1f} tok/s"
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
        response_preview=output.text[:100] + "..."
        if len(output.text) > 100
        else output.text,
    )


def run_benchmark(
    model_name: str,
    video_path: str = None,
    video_url: str = None,
    video_duration: float = 10.0,
    warmup_runs: int = 1,
    quick: bool = False,
) -> list[VideoBenchmarkResult]:
    """
    Run full video benchmark across multiple configurations.

    Args:
        model_name: VLM model to use
        video_path: Local video file path
        video_url: URL to download video from
        video_duration: Duration for synthetic video
        warmup_runs: Number of warmup runs
        quick: Run quick benchmark with fewer configs

    Returns:
        List of VideoBenchmarkResult objects
    """
    from vllm_mlx.models.vlm import MLXVisionLanguageModel

    # Load model
    print(f"\nLoading model: {model_name}")
    start = time.time()
    model = MLXVisionLanguageModel(model_name)
    model.load()
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")

    # Get or create video
    if video_path and Path(video_path).exists():
        print(f"\nUsing local video: {video_path}")
    elif video_url:
        video_path = download_video(video_url)
    else:
        print(f"\nCreating synthetic test video ({video_duration}s)...")
        video_path = create_test_video(duration=video_duration)

    video_info = get_video_info(video_path)
    print(
        f"Video: {video_info['width']}x{video_info['height']}, "
        f"{video_info['duration']:.1f}s, {video_info['fps']:.1f} fps, "
        f"{video_info['total_frames']} frames"
    )

    # Define benchmark configurations
    if quick:
        configs = [
            ("4 frames @ 1fps", 1.0, 4),
            ("8 frames @ 2fps", 2.0, 8),
            ("16 frames @ 2fps", 2.0, 16),
        ]
    else:
        configs = [
            # Varying FPS with same max_frames
            ("2 frames @ 0.5fps", 0.5, 2),
            ("4 frames @ 1fps", 1.0, 4),
            ("8 frames @ 2fps", 2.0, 8),
            ("16 frames @ 4fps", 4.0, 16),
            ("32 frames @ 4fps", 4.0, 32),
            # Varying max_frames with same FPS
            ("4 frames @ 2fps", 2.0, 4),
            ("8 frames @ 2fps", 2.0, 8),
            ("12 frames @ 2fps", 2.0, 12),
            ("16 frames @ 2fps", 2.0, 16),
            ("24 frames @ 2fps", 2.0, 24),
            # High density
            ("32 frames @ 8fps", 8.0, 32),
            ("48 frames @ 8fps", 8.0, 48),
        ]

    # Warmup
    if warmup_runs > 0:
        print(f"\nRunning {warmup_runs} warmup run(s)...")
        for _ in range(warmup_runs):
            run_video_benchmark(model, video_path, 1.0, 4, "warmup", warmup=True)
        print("Warmup complete.")

    # Run benchmarks
    print("\n" + "=" * 80)
    print("VIDEO BENCHMARK - Frame Count & FPS Performance")
    print("=" * 80)
    print(f"Model:          {model_name}")
    print(f"Video Duration: {video_info['duration']:.1f}s")
    print(f"Video Size:     {video_info['width']}x{video_info['height']}")
    print("-" * 80)
    print(
        f"  {'Configuration':>20} | {'Params':<22} | {'Time':>6} | {'Frames':>6} | {'Tokens':>4} | {'Speed':>9}"
    )
    print("-" * 80)

    results = []
    for config_name, fps, max_frames in configs:
        try:
            result = run_video_benchmark(
                model, video_path, fps, max_frames, config_name
            )
            results.append(result)
        except Exception as e:
            print(f"  Error with {config_name}: {e}")

    return results


def print_results(results: list[VideoBenchmarkResult]):
    """Print benchmark results in a nice table."""
    from tabulate import tabulate

    if not results:
        print("No results to display.")
        return

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    # Table by frame count
    print("\n### By Frame Count ###")
    table_data = []
    for r in sorted(results, key=lambda x: x.frames_extracted):
        table_data.append(
            [
                r.config_name,
                r.frames_extracted,
                f"{r.fps}",
                f"{r.time_seconds:.2f}s",
                r.completion_tokens,
                f"{r.tokens_per_second:.1f}",
            ]
        )

    headers = ["Config", "Frames", "FPS", "Time", "Tokens", "Tok/s"]
    print(tabulate(table_data, headers=headers, tablefmt="simple"))

    # Summary stats
    total_time = sum(r.time_seconds for r in results)
    total_tokens = sum(r.completion_tokens for r in results)
    avg_tps = total_tokens / total_time if total_time > 0 else 0

    print("-" * 80)
    print(f"Total Time:      {total_time:.2f}s")
    print(f"Total Tokens:    {total_tokens}")
    print(f"Average Tok/s:   {avg_tps:.1f}")

    # Find best/worst
    fastest = min(results, key=lambda r: r.time_seconds)
    slowest = max(results, key=lambda r: r.time_seconds)
    most_frames = max(results, key=lambda r: r.frames_extracted)

    print(f"\nFastest:     {fastest.config_name} ({fastest.time_seconds:.2f}s)")
    print(f"Slowest:     {slowest.config_name} ({slowest.time_seconds:.2f}s)")
    print(
        f"Most Frames: {most_frames.config_name} ({most_frames.frames_extracted} frames)"
    )

    # Frames vs Speed analysis
    print("\n### Frames vs Speed Analysis ###")
    frame_groups = {}
    for r in results:
        key = r.frames_extracted
        if key not in frame_groups:
            frame_groups[key] = []
        frame_groups[key].append(r)

    analysis_data = []
    for frames in sorted(frame_groups.keys()):
        group = frame_groups[frames]
        avg_time = sum(r.time_seconds for r in group) / len(group)
        avg_tps = sum(r.tokens_per_second for r in group) / len(group)
        analysis_data.append([frames, f"{avg_time:.2f}s", f"{avg_tps:.1f}"])

    print(
        tabulate(
            analysis_data,
            headers=["Frames", "Avg Time", "Avg Tok/s"],
            tablefmt="simple",
        )
    )

    # Sample response
    print("\n" + "-" * 80)
    print("Sample Response (first config):")
    print(f'  "{results[0].response_preview}"')


def save_results(
    results: list[VideoBenchmarkResult], output_path: str, model_name: str
):
    """Save benchmark results to JSON file."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "benchmark_type": "video",
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
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Video Benchmark for rapid-mlx VLM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic benchmark with default model
    python examples/video_benchmark.py

    # Specify model
    python examples/video_benchmark.py --model mlx-community/Qwen3-VL-8B-Instruct-4bit

    # Use video from URL
    python examples/video_benchmark.py --video-url https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4

    # Quick test
    python examples/video_benchmark.py --quick

    # Save results
    python examples/video_benchmark.py --output video_benchmark.json
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen3-VL-4B-Instruct-3bit",
        help="VLM model to use",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to local video file",
    )
    parser.add_argument(
        "--video-url",
        type=str,
        default=None,
        help="URL to download video from",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration of synthetic test video (seconds)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup runs",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with fewer configurations",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    # Run benchmark
    results = run_benchmark(
        model_name=args.model,
        video_path=args.video,
        video_url=args.video_url,
        video_duration=args.duration,
        warmup_runs=args.warmup,
        quick=args.quick,
    )

    # Print results
    print_results(results)

    # Save if requested
    if args.output:
        save_results(results, args.output, args.model)


if __name__ == "__main__":
    main()
