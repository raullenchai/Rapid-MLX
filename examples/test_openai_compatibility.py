# SPDX-License-Identifier: Apache-2.0
"""
OpenAI API Compatibility Test Script for rapid-mlx.

This script tests the OpenAI API compatibility of the rapid-mlx server.
It tests both the direct HTTP API and the official OpenAI Python client.

Usage:
    # First start the server:
    rapid-mlx --model mlx-community/Qwen3-VL-4B-Instruct-3bit --port 8000

    # Then run this script:
    python examples/test_openai_compatibility.py

    # With a different server URL:
    python examples/test_openai_compatibility.py --server-url http://localhost:9000

    # Test only specific endpoints:
    python examples/test_openai_compatibility.py --test-image --test-video
"""

import argparse
import base64
import sys
from pathlib import Path

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_header(text: str):
    """Print a section header."""
    print(f"\n{BLUE}{BOLD}{'=' * 60}{RESET}")
    print(f"{BLUE}{BOLD}{text}{RESET}")
    print(f"{BLUE}{BOLD}{'=' * 60}{RESET}\n")


def print_test(name: str, passed: bool, message: str = ""):
    """Print test result."""
    status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    print(f"  [{status}] {name}")
    if message:
        print(f"        {message}")


def print_warning(text: str):
    """Print a warning message."""
    print(f"{YELLOW}WARNING: {text}{RESET}")


def create_test_image() -> tuple[str, bytes]:
    """Create a simple test image and return (path, bytes)."""
    try:
        import io

        from PIL import Image

        # Create a simple 100x100 red square image
        img = Image.new("RGB", (100, 100), color="red")

        # Save to bytes
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()

        # Save to temp file
        import tempfile

        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_file.write(img_bytes)
        temp_file.close()

        return temp_file.name, img_bytes

    except ImportError:
        print_warning("Pillow not installed. Using a minimal PNG.")
        # Minimal 1x1 red PNG
        minimal_png = bytes(
            [
                0x89,
                0x50,
                0x4E,
                0x47,
                0x0D,
                0x0A,
                0x1A,
                0x0A,  # PNG signature
                0x00,
                0x00,
                0x00,
                0x0D,
                0x49,
                0x48,
                0x44,
                0x52,  # IHDR chunk
                0x00,
                0x00,
                0x00,
                0x01,
                0x00,
                0x00,
                0x00,
                0x01,  # 1x1 dimensions
                0x08,
                0x02,
                0x00,
                0x00,
                0x00,
                0x90,
                0x77,
                0x53,  # bit depth, color type
                0xDE,
                0x00,
                0x00,
                0x00,
                0x0C,
                0x49,
                0x44,
                0x41,  # IDAT chunk
                0x54,
                0x08,
                0xD7,
                0x63,
                0xF8,
                0xFF,
                0xFF,
                0x3F,  # compressed data
                0x00,
                0x05,
                0xFE,
                0x02,
                0xFE,
                0xDC,
                0xCC,
                0x59,  #
                0xE7,
                0x00,
                0x00,
                0x00,
                0x00,
                0x49,
                0x45,
                0x4E,  # IEND chunk
                0x44,
                0xAE,
                0x42,
                0x60,
                0x82,
            ]
        )
        import tempfile

        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_file.write(minimal_png)
        temp_file.close()
        return temp_file.name, minimal_png


def test_health_endpoint(server_url: str) -> bool:
    """Test the /health endpoint."""
    import requests

    try:
        response = requests.get(f"{server_url}/health", timeout=10)
        return response.status_code == 200
    except Exception as e:
        print_warning(f"Health check failed: {e}")
        return False


def test_models_endpoint(server_url: str) -> bool:
    """Test the /v1/models endpoint."""
    import requests

    try:
        response = requests.get(f"{server_url}/v1/models", timeout=10)
        if response.status_code != 200:
            return False

        data = response.json()
        # Should have "data" key with list of models
        return "data" in data and isinstance(data["data"], list)
    except Exception as e:
        print_warning(f"Models endpoint failed: {e}")
        return False


def test_chat_completions_http(server_url: str) -> tuple[bool, str]:
    """Test /v1/chat/completions with direct HTTP."""
    import requests

    try:
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {"role": "user", "content": "Say 'Hello' and nothing else."}
                ],
                "max_tokens": 50,
                "temperature": 0.1,
            },
            timeout=60,
        )

        if response.status_code != 200:
            return False, f"Status code: {response.status_code}"

        data = response.json()

        # Validate response structure
        if "choices" not in data:
            return False, "Missing 'choices' in response"

        if len(data["choices"]) == 0:
            return False, "Empty choices array"

        choice = data["choices"][0]
        if "message" not in choice:
            return False, "Missing 'message' in choice"

        if "content" not in choice["message"]:
            return False, "Missing 'content' in message"

        content = choice["message"]["content"]
        return True, f"Response: {content[:50]}..."

    except Exception as e:
        return False, str(e)


def test_chat_completions_openai(server_url: str) -> tuple[bool, str]:
    """Test /v1/chat/completions with OpenAI Python client."""
    try:
        from openai import OpenAI
    except ImportError:
        return False, "OpenAI package not installed. Run: pip install openai"

    try:
        client = OpenAI(
            base_url=f"{server_url}/v1",
            api_key="not-needed",  # rapid-mlx doesn't require API key
        )

        response = client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": "Say 'World' and nothing else."}],
            max_tokens=50,
            temperature=0.1,
        )

        content = response.choices[0].message.content
        return True, f"Response: {content[:50]}..."

    except Exception as e:
        return False, str(e)


def test_completions_endpoint(server_url: str) -> tuple[bool, str]:
    """Test /v1/completions endpoint (legacy)."""
    import requests

    try:
        response = requests.post(
            f"{server_url}/v1/completions",
            json={
                "model": "default",
                "prompt": "The capital of France is",
                "max_tokens": 20,
                "temperature": 0.1,
            },
            timeout=60,
        )

        if response.status_code != 200:
            return False, f"Status code: {response.status_code}"

        data = response.json()

        if "choices" not in data:
            return False, "Missing 'choices' in response"

        if len(data["choices"]) == 0:
            return False, "Empty choices array"

        text = data["choices"][0].get("text", "")
        return True, f"Response: {text[:50]}..."

    except Exception as e:
        return False, str(e)


def test_image_chat_http(server_url: str) -> tuple[bool, str]:
    """Test multimodal image chat with direct HTTP."""
    import requests

    # Create test image
    image_path, image_bytes = create_test_image()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    try:
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What color is this image? Answer in one word.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 50,
                "temperature": 0.1,
            },
            timeout=120,
        )

        if response.status_code != 200:
            return (
                False,
                f"Status code: {response.status_code}, Body: {response.text[:100]}",
            )

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return True, f"Response: {content[:50]}..."

    except Exception as e:
        return False, str(e)
    finally:
        # Clean up temp file
        Path(image_path).unlink(missing_ok=True)


def test_image_chat_openai(server_url: str) -> tuple[bool, str]:
    """Test multimodal image chat with OpenAI client."""
    try:
        from openai import OpenAI
    except ImportError:
        return False, "OpenAI package not installed"

    # Create test image
    image_path, image_bytes = create_test_image()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    try:
        client = OpenAI(
            base_url=f"{server_url}/v1",
            api_key="not-needed",
        )

        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What color is this image? Answer briefly.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=50,
            temperature=0.1,
        )

        content = response.choices[0].message.content
        return True, f"Response: {content[:50]}..."

    except Exception as e:
        return False, str(e)
    finally:
        Path(image_path).unlink(missing_ok=True)


def test_image_url_http(server_url: str) -> tuple[bool, str]:
    """Test image from URL."""
    import requests

    # Use a public test image
    test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/200px-PNG_transparency_demonstration_1.png"

    try:
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image briefly."},
                            {"type": "image_url", "image_url": {"url": test_image_url}},
                        ],
                    }
                ],
                "max_tokens": 100,
                "temperature": 0.7,
            },
            timeout=120,
        )

        if response.status_code != 200:
            return False, f"Status code: {response.status_code}"

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return True, f"Response: {content[:80]}..."

    except Exception as e:
        return False, str(e)


def test_streaming_chat(server_url: str) -> tuple[bool, str]:
    """Test streaming chat completions."""
    import requests

    try:
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "Count from 1 to 5."}],
                "max_tokens": 50,
                "temperature": 0.1,
                "stream": True,
            },
            timeout=60,
            stream=True,
        )

        if response.status_code != 200:
            return False, f"Status code: {response.status_code}"

        chunks = []
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    chunks.append(data)

        if len(chunks) == 0:
            return False, "No streaming chunks received"

        return True, f"Received {len(chunks)} streaming chunks"

    except Exception as e:
        return False, str(e)


def create_test_video() -> tuple[str, bytes]:
    """
    Create a simple test video with colored frames.

    Returns (path, bytes) of a minimal MP4 video.
    """
    try:
        import tempfile

        import cv2
        import numpy as np

        # Create a simple video with 3 colored frames (red, green, blue)
        colors = [
            (0, 0, 255),  # Red (BGR)
            (0, 255, 0),  # Green (BGR)
            (255, 0, 0),  # Blue (BGR)
        ]

        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_path = temp_file.name
        temp_file.close()

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_path, fourcc, 1.0, (100, 100))

        for color in colors:
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            frame[:] = color
            out.write(frame)

        out.release()

        # Read back the bytes
        with open(temp_path, "rb") as f:
            video_bytes = f.read()

        return temp_path, video_bytes

    except ImportError:
        print_warning("OpenCV not installed. Skipping video test.")
        return None, None


def test_video_chat_http(server_url: str) -> tuple[bool, str]:
    """Test multimodal video chat with direct HTTP."""
    import requests

    # Create test video
    video_path, video_bytes = create_test_video()
    if video_path is None:
        return False, "Could not create test video (OpenCV required)"

    video_base64 = base64.b64encode(video_bytes).decode("utf-8")

    try:
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What colors appear in this video? List them briefly.",
                            },
                            {
                                "type": "video_url",
                                "video_url": {
                                    "url": f"data:video/mp4;base64,{video_base64}"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 100,
                "temperature": 0.3,
            },
            timeout=180,
        )

        if response.status_code != 200:
            return (
                False,
                f"Status code: {response.status_code}, Body: {response.text[:100]}",
            )

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return True, f"Response: {content[:80]}..."

    except Exception as e:
        return False, str(e)
    finally:
        # Clean up temp file
        if video_path:
            Path(video_path).unlink(missing_ok=True)


def test_video_chat_openai(server_url: str) -> tuple[bool, str]:
    """Test multimodal video chat with OpenAI client."""
    try:
        from openai import OpenAI
    except ImportError:
        return False, "OpenAI package not installed"

    # Create test video
    video_path, video_bytes = create_test_video()
    if video_path is None:
        return False, "Could not create test video (OpenCV required)"

    video_base64 = base64.b64encode(video_bytes).decode("utf-8")

    try:
        client = OpenAI(
            base_url=f"{server_url}/v1",
            api_key="not-needed",
        )

        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe what you see in this video briefly.",
                        },
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": f"data:video/mp4;base64,{video_base64}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=100,
            temperature=0.3,
        )

        content = response.choices[0].message.content
        return True, f"Response: {content[:80]}..."

    except Exception as e:
        return False, str(e)
    finally:
        if video_path:
            Path(video_path).unlink(missing_ok=True)


def test_video_url_http(server_url: str) -> tuple[bool, str]:
    """Test video from URL."""
    import requests

    # Use a public test video (Big Buck Bunny - small clip)
    test_video_url = "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"

    try:
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe what happens in this video briefly.",
                            },
                            {"type": "video_url", "video_url": {"url": test_video_url}},
                        ],
                    }
                ],
                "max_tokens": 150,
                "temperature": 0.7,
            },
            timeout=180,
        )

        if response.status_code != 200:
            return False, f"Status code: {response.status_code}"

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return True, f"Response: {content[:80]}..."

    except Exception as e:
        return False, str(e)


def run_all_tests(server_url: str, test_image: bool = True, test_video: bool = True):
    """Run all compatibility tests."""
    results = {"passed": 0, "failed": 0}

    def record(passed: bool):
        if passed:
            results["passed"] += 1
        else:
            results["failed"] += 1

    print_header("rapid-mlx OpenAI API Compatibility Tests")
    print(f"Server URL: {server_url}\n")

    # Basic endpoint tests
    print_header("1. Basic Endpoints")

    passed = test_health_endpoint(server_url)
    print_test("/health endpoint", passed)
    record(passed)

    passed = test_models_endpoint(server_url)
    print_test("/v1/models endpoint", passed)
    record(passed)

    # Chat completions tests
    print_header("2. Chat Completions - Text Only (/v1/chat/completions)")

    passed, msg = test_chat_completions_http(server_url)
    print_test("Direct HTTP request", passed, msg)
    record(passed)

    passed, msg = test_chat_completions_openai(server_url)
    print_test("OpenAI Python client", passed, msg)
    record(passed)

    # Legacy completions test
    print_header("3. Legacy Completions (/v1/completions)")

    passed, msg = test_completions_endpoint(server_url)
    print_test("Direct HTTP request", passed, msg)
    record(passed)

    # Streaming test
    print_header("4. Streaming")

    passed, msg = test_streaming_chat(server_url)
    print_test("Streaming chat completions", passed, msg)
    record(passed)

    # Multimodal image tests
    if test_image:
        print_header("5. Multimodal - Images")

        passed, msg = test_image_chat_http(server_url)
        print_test("Base64 image (HTTP)", passed, msg)
        record(passed)

        passed, msg = test_image_chat_openai(server_url)
        print_test("Base64 image (OpenAI client)", passed, msg)
        record(passed)

        passed, msg = test_image_url_http(server_url)
        print_test("Image from URL", passed, msg)
        record(passed)

    # Multimodal video tests
    if test_video:
        print_header("6. Multimodal - Video")

        passed, msg = test_video_chat_http(server_url)
        print_test("Base64 video (HTTP)", passed, msg)
        record(passed)

        passed, msg = test_video_chat_openai(server_url)
        print_test("Base64 video (OpenAI client)", passed, msg)
        record(passed)

        passed, msg = test_video_url_http(server_url)
        print_test("Video from URL", passed, msg)
        record(passed)

    # Summary
    print_header("Test Summary")

    total = results["passed"] + results["failed"]
    print(f"  Total tests: {total}")
    print(f"  {GREEN}Passed: {results['passed']}{RESET}")
    print(f"  {RED}Failed: {results['failed']}{RESET}")

    if results["failed"] == 0:
        print(f"\n{GREEN}{BOLD}All tests passed! API is OpenAI-compatible.{RESET}")
        return 0
    else:
        print(f"\n{RED}{BOLD}Some tests failed. Check the output above.{RESET}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Test OpenAI API compatibility of rapid-mlx server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test with default server (localhost:8000)
    python test_openai_compatibility.py

    # Test with custom server
    python test_openai_compatibility.py --server-url http://myserver:9000

    # Skip image tests (for text-only models)
    python test_openai_compatibility.py --no-image
        """,
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="URL of the rapid-mlx server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--no-image",
        action="store_true",
        help="Skip image tests",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip video tests",
    )
    args = parser.parse_args()

    # Check if server is reachable
    print(f"Checking server at {args.server_url}...")
    if not test_health_endpoint(args.server_url):
        print(f"{RED}ERROR: Cannot connect to server at {args.server_url}{RESET}")
        print("Make sure the rapid-mlx server is running:")
        print("  rapid-mlx --model mlx-community/Qwen3-VL-4B-Instruct-3bit --port 8000")
        sys.exit(1)

    print(f"{GREEN}Server is reachable!{RESET}")

    return run_all_tests(
        server_url=args.server_url,
        test_image=not args.no_image,
        test_video=not args.no_video,
    )


if __name__ == "__main__":
    sys.exit(main())
