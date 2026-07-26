#!/usr/bin/env python3
"""
Demo: OpenAI API - Image Analysis

Shows how to use rapid-mlx with the OpenAI Python SDK for image understanding.

Usage:
    1. Start the server with a VLM model:
       rapid-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit --port 8000

    2. Run this script:
       python examples/demo_openai_image.py
"""

import base64

from openai import OpenAI

# Connect to rapid-mlx server
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

print("=" * 60)
print("OpenAI API Demo - Image Analysis")
print("=" * 60)

# 1. Image from URL
print("\n1. Analyze Image from URL")
print("-" * 40)
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
print(f"Image URL: {image_url}")
print("Question: What animal is in this image?")

response = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What animal is in this image? Describe it briefly.",
                },
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ],
    max_tokens=150,
)
print(f"Answer: {response.choices[0].message.content}")

# 2. Another image from URL
print("\n2. Describe a Scene")
print("-" * 40)
scene_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/Empire_State_Building_%28aerial_view%29.jpg/800px-Empire_State_Building_%28aerial_view%29.jpg"
print(f"Image URL: {scene_url}")
print("Question: What famous building is this?")

response = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What famous building is shown in this image? Where is it located?",
                },
                {"type": "image_url", "image_url": {"url": scene_url}},
            ],
        }
    ],
    max_tokens=150,
)
print(f"Answer: {response.choices[0].message.content}")

# 3. Base64 encoded image (creating a simple test image)
print("\n3. Analyze Base64 Encoded Image")
print("-" * 40)

# Create a simple red square image for testing
try:
    import io

    from PIL import Image

    # Create a simple 100x100 red image
    img = Image.new("RGB", (100, 100), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    print("Created: 100x100 red square image")
    print("Question: What color is this image?")

    response = client.chat.completions.create(
        model="default",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ],
        max_tokens=50,
    )
    print(f"Answer: {response.choices[0].message.content}")
except ImportError:
    print("Skipped (PIL not available)")

# 4. Ask follow-up question about same image
print("\n4. Follow-up Questions (Multi-turn with Image)")
print("-" * 40)
food_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Good_Food_Display_-_NCI_Visuals_Online.jpg/800px-Good_Food_Display_-_NCI_Visuals_Online.jpg"
print(f"Image URL: {food_url}")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What foods do you see in this image?"},
            {"type": "image_url", "image_url": {"url": food_url}},
        ],
    }
]

response = client.chat.completions.create(
    model="default", messages=messages, max_tokens=150
)
print("Q1: What foods do you see in this image?")
print(f"A1: {response.choices[0].message.content}")

# Follow-up (note: image context may not persist in all models)
messages.append({"role": "assistant", "content": response.choices[0].message.content})
messages.append({"role": "user", "content": "Which of these foods are fruits?"})

response = client.chat.completions.create(
    model="default", messages=messages, max_tokens=100
)
print("\nQ2: Which of these foods are fruits?")
print(f"A2: {response.choices[0].message.content}")

print("\n" + "=" * 60)
print("Demo complete!")
print("=" * 60)
