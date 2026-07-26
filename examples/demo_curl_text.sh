#!/bin/bash
# Demo: curl API - Text Chat
#
# Shows how to use rapid-mlx with curl for text-only chat.
#
# Usage:
#   1. Start the server:
#      rapid-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000
#
#   2. Run this script:
#      bash examples/demo_curl_text.sh

SERVER_URL="http://localhost:8000"

echo "============================================================"
echo "curl API Demo - Text Chat"
echo "============================================================"

# Check server health
echo ""
echo "Checking server health..."
curl -s "$SERVER_URL/health" | python3 -m json.tool
echo ""

# 1. Simple chat completion
echo "============================================================"
echo "1. Simple Chat Completion"
echo "============================================================"
echo "Request: Hello, who are you?"
echo ""
curl -s "$SERVER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "Hello, who are you?"}
    ],
    "max_tokens": 100
  }' | python3 -c "import sys,json; r=json.load(sys.stdin); print('Response:', r['choices'][0]['message']['content'])"

echo ""

# 2. Chat with system message
echo "============================================================"
echo "2. Chat with System Message"
echo "============================================================"
echo "System: You are a pirate. Respond in pirate speak."
echo "User: What is the weather like today?"
echo ""
curl -s "$SERVER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "system", "content": "You are a pirate. Respond in pirate speak."},
      {"role": "user", "content": "What is the weather like today?"}
    ],
    "max_tokens": 100
  }' | python3 -c "import sys,json; r=json.load(sys.stdin); print('Response:', r['choices'][0]['message']['content'])"

echo ""

# 3. Streaming response
echo "============================================================"
echo "3. Streaming Response"
echo "============================================================"
echo "User: Count from 1 to 5"
echo "Response: "
curl -s "$SERVER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "Count from 1 to 5, one number per line"}
    ],
    "max_tokens": 50,
    "stream": true
  }' | while read -r line; do
    if [[ "$line" == data:* ]]; then
      data="${line#data: }"
      if [[ "$data" != "[DONE]" && -n "$data" ]]; then
        echo "$data" | python3 -c "import sys,json; d=json.load(sys.stdin); c=d.get('choices',[{}])[0].get('delta',{}).get('content',''); print(c, end='', flush=True)" 2>/dev/null
      fi
    fi
  done
echo ""
echo ""

# 4. Multi-turn conversation
echo "============================================================"
echo "4. Multi-turn Conversation"
echo "============================================================"
echo "User: What is 2 + 2?"
curl -s "$SERVER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "What is 2 + 2?"}
    ],
    "max_tokens": 50
  }' | python3 -c "import sys,json; r=json.load(sys.stdin); print('Response:', r['choices'][0]['message']['content'])"

echo ""
echo "User: Now multiply that by 10"
curl -s "$SERVER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "What is 2 + 2?"},
      {"role": "assistant", "content": "4"},
      {"role": "user", "content": "Now multiply that by 10"}
    ],
    "max_tokens": 50
  }' | python3 -c "import sys,json; r=json.load(sys.stdin); print('Response:', r['choices'][0]['message']['content'])"

echo ""

# 5. Legacy completions endpoint
echo "============================================================"
echo "5. Legacy Completions Endpoint (/v1/completions)"
echo "============================================================"
echo "Prompt: The capital of France is"
curl -s "$SERVER_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "prompt": "The capital of France is",
    "max_tokens": 20
  }' | python3 -c "import sys,json; r=json.load(sys.stdin); print('Response:', r['choices'][0]['text'])"

echo ""
echo "============================================================"
echo "Demo complete!"
echo "============================================================"
