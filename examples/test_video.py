#!/usr/bin/env python3
"""
Test script for VLM video support in rapid-mlx.

This script tests video understanding capabilities by:
1. Downloading a sample video (or using a local one)
2. Loading a VLM model that supports video
3. Running inference on the video

Usage:
    python examples/test_video.py
    python examples/test_video.py --video /path/to/video.mp4
    python examples/test_video.py --model mlx-community/Qwen3-VL-8B-Instruct-4bit
"""

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def download_sample_video() -> str:
    """Download a sample video for testing."""
    import requests

    # Short sample video from Pexels (free to use)
    # This is a ~5 second video of nature
    video_url = "https://www.pexels.com/download/video/3571264/"

    logger.info("Downloading sample video...")

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
        response = requests.get(video_url, timeout=60, headers=headers, stream=True)
        response.raise_for_status()

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        temp_file.close()

        logger.info(f"Video saved to: {temp_file.name}")
        return temp_file.name

    except Exception as e:
        logger.error(f"Failed to download video: {e}")
        logger.info("Please provide a local video with --video flag")
        sys.exit(1)


def create_test_video() -> str:
    """Create a simple test video using OpenCV if download fails."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.error("OpenCV is required: pip install opencv-python")
        sys.exit(1)

    logger.info("Creating synthetic test video...")

    # Create a simple video with colored frames
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_file.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_file.name, fourcc, 30.0, (640, 480))

    colors = [
        (255, 0, 0),  # Blue
        (0, 255, 0),  # Green
        (0, 0, 255),  # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
    ]

    # Create 5 seconds of video (150 frames at 30fps)
    for i in range(150):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        color_idx = i // 30  # Change color every second
        frame[:] = colors[color_idx % len(colors)]

        # Add text showing frame number
        cv2.putText(
            frame,
            f"Frame {i}",
            (250, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            3,
        )
        out.write(frame)

    out.release()
    logger.info(f"Test video created: {temp_file.name}")
    return temp_file.name


def get_video_info(video_path: str) -> dict:
    """Get information about a video file."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Cannot open video"}

    info = {
        "path": video_path,
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration_seconds"] = (
        info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
    )

    cap.release()
    return info


def test_frame_extraction(video_path: str):
    """Test video frame extraction."""
    from vllm_mlx.models.vlm import extract_video_frames_smart

    logger.info("\n=== Testing Frame Extraction ===")

    video_info = get_video_info(video_path)
    logger.info(f"Video info: {video_info}")

    # Test with different FPS settings
    for fps in [1.0, 2.0, 4.0]:
        start = time.time()
        frames = extract_video_frames_smart(video_path, fps=fps, max_frames=32)
        elapsed = time.time() - start

        logger.info(f"  FPS={fps}: Extracted {len(frames)} frames in {elapsed:.2f}s")
        logger.info(f"    Frame shape: {frames[0].shape if frames else 'N/A'}")


def test_video_generation(video_path: str, model_name: str):
    """Test video understanding with VLM."""
    from vllm_mlx.models.vlm import MLXVisionLanguageModel

    logger.info("\n=== Testing Video Generation ===")
    logger.info(f"Model: {model_name}")
    logger.info(f"Video: {video_path}")

    # Load model
    logger.info("Loading model...")
    start = time.time()
    model = MLXVisionLanguageModel(model_name)
    model.load()
    load_time = time.time() - start
    logger.info(f"Model loaded in {load_time:.2f}s")

    # Test describe_video convenience method
    logger.info("\n--- Test 1: describe_video() ---")
    start = time.time()
    description = model.describe_video(
        video=video_path,
        prompt="Describe what happens in this video in detail.",
        fps=2.0,
        max_frames=16,
        max_tokens=256,
    )
    elapsed = time.time() - start

    logger.info(f"Time: {elapsed:.2f}s")
    logger.info(f"Description:\n{description}")

    # Test generate() with videos parameter
    logger.info("\n--- Test 2: generate() with videos ---")
    start = time.time()
    output = model.generate(
        prompt="What actions or movements can you see in this video?",
        videos=[video_path],
        video_fps=1.0,
        video_max_frames=8,
        max_tokens=200,
    )
    elapsed = time.time() - start

    logger.info(f"Time: {elapsed:.2f}s")
    logger.info(
        f"Tokens: prompt={output.prompt_tokens}, completion={output.completion_tokens}"
    )
    logger.info(f"Response:\n{output.text}")

    # Test chat() with video (OpenAI format)
    logger.info("\n--- Test 3: chat() with video (OpenAI format) ---")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": "Summarize this video in one sentence."},
            ],
        }
    ]

    start = time.time()
    output = model.chat(
        messages=messages,
        max_tokens=100,
        video_fps=2.0,
        video_max_frames=12,
    )
    elapsed = time.time() - start

    logger.info(f"Time: {elapsed:.2f}s")
    logger.info(f"Response:\n{output.text}")

    return model  # Return model for URL test


def test_video_url(model, video_url: str):
    """Test video from URL."""
    logger.info("\n=== Testing Video from URL ===")
    logger.info(f"URL: {video_url}")

    # Test with video_url format (OpenAI style)
    logger.info("\n--- Test: video_url format (OpenAI style) ---")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_url}},
                {
                    "type": "text",
                    "text": "What is happening in this video? Describe briefly.",
                },
            ],
        }
    ]

    start = time.time()
    output = model.chat(
        messages=messages,
        max_tokens=150,
        video_fps=1.0,
        video_max_frames=8,
    )
    elapsed = time.time() - start

    logger.info(f"Time: {elapsed:.2f}s")
    logger.info(f"Response:\n{output.text}")

    # Test with generate() using URL directly
    logger.info("\n--- Test: generate() with URL ---")
    start = time.time()
    output = model.generate(
        prompt="Describe what you see in this video.",
        videos=[video_url],
        video_fps=1.0,
        video_max_frames=6,
        max_tokens=100,
    )
    elapsed = time.time() - start

    logger.info(f"Time: {elapsed:.2f}s")
    logger.info(f"Response:\n{output.text}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Test VLM video support")
    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file (will create test video if not provided)",
    )
    parser.add_argument(
        "--video-url", type=str, help="URL to a video file to test URL support"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen3-VL-4B-Instruct-3bit",
        help="VLM model to use",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only test frame extraction (no model loading)",
    )
    parser.add_argument(
        "--create-test-video",
        action="store_true",
        help="Create synthetic test video instead of downloading",
    )
    parser.add_argument(
        "--url-only",
        action="store_true",
        help="Only test video URL support (requires --video-url)",
    )

    args = parser.parse_args()

    # Get or create video
    if args.video:
        video_path = args.video
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            sys.exit(1)
    elif args.create_test_video:
        video_path = create_test_video()
    else:
        # Try to download, fall back to creating
        try:
            video_path = download_sample_video()
        except Exception:
            video_path = create_test_video()

    # Run tests
    logger.info(f"\n{'=' * 50}")
    logger.info("VLM Video Test")
    logger.info(f"{'=' * 50}")

    model = None

    if not args.url_only:
        # Test frame extraction
        test_frame_extraction(video_path)

        # Test model generation (unless extract-only)
        if not args.extract_only:
            model = test_video_generation(video_path, args.model)

    # Test video URL if provided
    if args.video_url:
        if model is None:
            from vllm_mlx.models.vlm import MLXVisionLanguageModel

            logger.info(f"\nLoading model for URL test: {args.model}")
            model = MLXVisionLanguageModel(args.model)
            model.load()

        test_video_url(model, args.video_url)

    logger.info("\n✅ Video tests completed!")


if __name__ == "__main__":
    main()
