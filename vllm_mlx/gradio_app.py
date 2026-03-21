# SPDX-License-Identifier: Apache-2.0
"""
Gradio Chatbot Interface for vllm-mlx.

A multimodal chat interface that connects to the vllm-mlx server
and supports text, images, and video files.

Usage:
    # First start the server with a multimodal model:
    vllm-mlx --model mlx-community/Qwen3-VL-4B-Instruct-3bit --port 8000

    # Then run this app:
    vllm-mlx-chat

    # Or with custom settings:
    vllm-mlx-chat --server-url http://localhost:8000 --port 7860
"""

import argparse
import base64
from pathlib import Path

try:
    import gradio as gr
except ImportError:
    print("Error: gradio is required for the chat UI.")
    print("Install it with: pip install 'rapid-mlx[chat]'")
    raise SystemExit(1)
import requests


def encode_file_to_base64(file_path: str) -> tuple[str, str]:
    """
    Encode a file to base64 data URL.

    Returns:
        Tuple of (data_url, media_type) where media_type is 'image' or 'video'
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    # Image MIME types
    image_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }

    # Video MIME types
    video_types = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
    }

    if suffix in image_types:
        mime_type = image_types[suffix]
        media_type = "image"
    elif suffix in video_types:
        mime_type = video_types[suffix]
        media_type = "video"
    else:
        # Default to image/jpeg for unknown
        mime_type = "image/jpeg"
        media_type = "image"

    with open(file_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{data}", media_type


def build_message_content(text: str, files: list[str] | None = None) -> list | str:
    """
    Build OpenAI-compatible message content with text and optional files.

    Args:
        text: The text message
        files: Optional list of file paths (images or videos)

    Returns:
        Content in OpenAI multimodal format
    """
    if not files:
        return text

    content = []

    # Add text part first
    if text:
        content.append({"type": "text", "text": text})

    # Add files
    for file_path in files:
        data_url, media_type = encode_file_to_base64(file_path)

        if media_type == "image":
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        elif media_type == "video":
            content.append({"type": "video_url", "video_url": {"url": data_url}})

    return content if content else text


def create_chat_function(server_url: str, max_tokens: int, temperature: float):
    """
    Create the chat function for Gradio ChatInterface.

    Args:
        server_url: URL of the vllm-mlx server
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Chat function compatible with gr.ChatInterface
    """
    # Store media from previous messages to include in context
    media_cache = {}  # Maps message index to list of file data URLs

    def chat(message: dict, history: list) -> str:
        """
        Process a multimodal message and return response.

        Args:
            message: Dict with 'text' and optional 'files' keys
            history: List of previous messages

        Returns:
            Assistant response text
        """
        nonlocal media_cache

        # Extract text and files from message
        text = message.get("text", "") if isinstance(message, dict) else message
        files = message.get("files", []) if isinstance(message, dict) else []

        # Debug output
        import sys

        print(f"[Chat] Text: {text!r}", flush=True)
        print(f"[Chat] Files: {files}", flush=True)
        print(f"[Chat] History length: {len(history)}", flush=True)
        sys.stdout.flush()

        # Build messages list for API
        messages = []

        # Process history - keep media content for multimodal context
        for i, msg in enumerate(history):
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")

                # Check if this message had media cached
                if i in media_cache and role == "user":
                    # Rebuild multimodal content with cached media
                    if isinstance(content, str):
                        rebuilt_content = [{"type": "text", "text": content}]
                    elif isinstance(content, list):
                        # Extract just text parts
                        text_parts = [
                            p.get("text", "")
                            for p in content
                            if isinstance(p, dict) and p.get("type") == "text"
                        ]
                        rebuilt_content = (
                            [{"type": "text", "text": " ".join(text_parts)}]
                            if text_parts
                            else []
                        )
                    else:
                        rebuilt_content = [{"type": "text", "text": str(content)}]

                    # Add cached media
                    for media_item in media_cache[i]:
                        rebuilt_content.append(media_item)

                    messages.append({"role": role, "content": rebuilt_content})
                else:
                    # No media, just text
                    if isinstance(content, list):
                        text_parts = [
                            p.get("text", "")
                            for p in content
                            if isinstance(p, dict) and p.get("type") == "text"
                        ]
                        content = " ".join(text_parts)
                    elif isinstance(content, dict):
                        content = content.get("text", str(content))
                    messages.append({"role": role, "content": content})

        # Build current message content and cache media
        current_content = build_message_content(text, files if files else None)
        messages.append({"role": "user", "content": current_content})

        # Cache media for this message (will be at index len(history) after this turn)
        if files:
            current_idx = len(history)
            media_items = []
            for file_path in files:
                data_url, media_type = encode_file_to_base64(file_path)
                if media_type == "image":
                    media_items.append(
                        {"type": "image_url", "image_url": {"url": data_url}}
                    )
                elif media_type == "video":
                    media_items.append(
                        {"type": "video_url", "video_url": {"url": data_url}}
                    )
            if media_items:
                media_cache[current_idx] = media_items
                print(
                    f"[Chat] Cached {len(media_items)} media items for message {current_idx}",
                    flush=True,
                )

        # Debug
        print(f"[Chat] Sending {len(messages)} messages to server")
        if isinstance(current_content, list):
            print(f"[Chat] Content types: {[c.get('type') for c in current_content]}")

        # Send request to server
        try:
            response = requests.post(
                f"{server_url}/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=600,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to server. Make sure vllm-mlx is running."
        except requests.exceptions.Timeout:
            return "Error: Timeout - server took too long to respond."
        except Exception as e:
            return f"Error: {str(e)}"

    return chat


def main():
    """Run the Gradio app."""
    parser = argparse.ArgumentParser(
        description="Gradio Multimodal Chat Interface for vllm-mlx",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start with default settings
    vllm-mlx-chat

    # Connect to a different server
    vllm-mlx-chat --server-url http://localhost:9000

    # Create a public share link
    vllm-mlx-chat --share

Note: Make sure the vllm-mlx server is running with a multimodal model:
    vllm-mlx --model mlx-community/Qwen3-VL-4B-Instruct-3bit --port 8000
        """,
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="URL of the vllm-mlx server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for Gradio interface (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate (default: 2048)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Use text-only mode (no image/video support, faster for LLM-only models)",
    )
    args = parser.parse_args()

    print(f"Connecting to vllm-mlx server at: {args.server_url}")
    print(f"Starting Gradio interface on port: {args.port}")

    if args.text_only:
        print("Mode: Text-only (no multimodal support)")

        # Create text-only chat function
        def text_chat(message: str, history: list) -> str:
            """Process a text-only message."""
            messages = []
            for msg in history:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        text_parts = [
                            p.get("text", "")
                            for p in content
                            if isinstance(p, dict) and p.get("type") == "text"
                        ]
                        content = " ".join(text_parts)
                    messages.append({"role": role, "content": content})

            messages.append({"role": "user", "content": message})

            try:
                response = requests.post(
                    f"{args.server_url}/v1/chat/completions",
                    json={
                        "model": "default",
                        "messages": messages,
                        "max_tokens": args.max_tokens,
                        "temperature": args.temperature,
                    },
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
            except requests.exceptions.ConnectionError:
                return "Error: Cannot connect to server. Make sure vllm-mlx is running."
            except requests.exceptions.Timeout:
                return "Error: Timeout - server took too long to respond."
            except Exception as e:
                return f"Error: {str(e)}"

        demo = gr.ChatInterface(
            fn=text_chat,
            title="vllm-mlx Text Chat",
            description="Fast text-only chat with LLM models on Apple Silicon.",
            examples=[
                "Hello, who are you?",
                "Explain quantum computing in simple terms.",
                "Write a haiku about programming.",
            ],
        )
    else:
        print("Mode: Multimodal (text, image, and video)")

        # Create chat function
        chat_fn = create_chat_function(
            server_url=args.server_url,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        # Create ChatInterface with multimodal support
        demo = gr.ChatInterface(
            fn=chat_fn,
            title="vllm-mlx Multimodal Chat",
            description="Chat with vision-language models on Apple Silicon. Upload images or videos!",
            multimodal=True,
            textbox=gr.MultimodalTextbox(
                file_types=["image", "video"],
                file_count="multiple",
                placeholder="Type a message or upload an image/video...",
            ),
        )

    demo.launch(
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
