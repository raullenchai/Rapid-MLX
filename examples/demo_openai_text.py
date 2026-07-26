#!/usr/bin/env python3
"""
Demo: OpenAI API - Text Chat

Shows how to use rapid-mlx with the OpenAI Python SDK for text-only chat.

Usage:
    1. Start the server with any model:
       rapid-mlx --model mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000

    2. Run this script:
       python examples/demo_openai_text.py
"""

from openai import OpenAI

# Connect to rapid-mlx server
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

print("=" * 60)
print("OpenAI API Demo - Text Chat")
print("=" * 60)

# 1. Simple chat completion
print("\n1. Simple Chat Completion")
print("-" * 40)
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello, who are you?"}],
    max_tokens=100,
)
print("User: Hello, who are you?")
print(f"Assistant: {response.choices[0].message.content}")

# 2. Chat with system message
print("\n2. Chat with System Message")
print("-" * 40)
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": "You are a pirate. Respond in pirate speak."},
        {"role": "user", "content": "What is the weather like today?"},
    ],
    max_tokens=100,
)
print("System: You are a pirate. Respond in pirate speak.")
print("User: What is the weather like today?")
print(f"Assistant: {response.choices[0].message.content}")

# 3. Streaming response
print("\n3. Streaming Response")
print("-" * 40)
print("User: Tell me a short joke")
print("Assistant: ", end="")
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Tell me a short joke"}],
    max_tokens=150,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print("\n")

# 4. Multi-turn conversation
print("4. Multi-turn Conversation")
print("-" * 40)
messages = [{"role": "user", "content": "What is 2 + 2?"}]
response = client.chat.completions.create(
    model="default", messages=messages, max_tokens=50
)
print("User: What is 2 + 2?")
print(f"Assistant: {response.choices[0].message.content}")

# Continue the conversation
messages.append({"role": "assistant", "content": response.choices[0].message.content})
messages.append({"role": "user", "content": "Now multiply that by 10"})
response = client.chat.completions.create(
    model="default", messages=messages, max_tokens=50
)
print("\nUser: Now multiply that by 10")
print(f"Assistant: {response.choices[0].message.content}")

# 5. With temperature control
print("\n5. Temperature Control (Creative vs Deterministic)")
print("-" * 40)
prompt = "Complete this sentence: The robot walked into the"

# Low temperature (more deterministic)
response_low = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=30,
    temperature=0.1,
)
print(f"Temperature 0.1: {response_low.choices[0].message.content}")

# High temperature (more creative)
response_high = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=30,
    temperature=1.0,
)
print(f"Temperature 1.0: {response_high.choices[0].message.content}")

print("\n" + "=" * 60)
print("Demo complete!")
print("=" * 60)
