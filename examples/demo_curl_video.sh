#!/bin/bash
# Demo: curl API - Video Analysis
#
# Shows how to use rapid-mlx with curl for video understanding.
#
# Usage:
#   1. Start the server with a VLM model:
#      rapid-mlx --model mlx-community/Qwen3-VL-4B-Instruct-3bit --port 8000
#
#   2. Run this script:
#      bash examples/demo_curl_video.sh

SERVER_URL="http://localhost:8000"

echo "============================================================"
echo "curl API Demo - Video Analysis"
echo "============================================================"

# Check server health
echo ""
echo "Checking server health..."
curl -s "$SERVER_URL/health" | python3 -m json.tool
echo ""

# 1. Video from URL
echo "============================================================"
echo "1. Analyze Video from URL"
echo "============================================================"
echo "Video: Big Buck Bunny (10 seconds)"
echo "Question: What is happening in this video?"
echo ""
curl -s "$SERVER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What is happening in this video? Describe the scene briefly."},
        {"type": "video_url", "video_url": {"url": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"}}
      ]
    }],
    "max_tokens": 200
  }' | python3 -c "import sys,json; r=json.load(sys.stdin); print('Response:', r['choices'][0]['message']['content'])"

echo ""

# 2. Different video
echo "============================================================"
echo "2. Analyze Jellyfish Video"
echo "============================================================"
echo "Video: Jellyfish (10 seconds)"
echo "Question: What do you see in this video?"
echo ""
curl -s "$SERVER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What do you see in this video? Describe the colors and movement."},
        {"type": "video_url", "video_url": {"url": "https://test-videos.co.uk/vids/jellyfish/mp4/h264/360/Jellyfish_360_10s_1MB.mp4"}}
      ]
    }],
    "max_tokens": 200
  }' | python3 -c "import sys,json; r=json.load(sys.stdin); print('Response:', r['choices'][0]['message']['content'])"

echo ""

# 3. Specific questions about video
echo "============================================================"
echo "3. Specific Questions About Video"
echo "============================================================"
echo "Video: Big Buck Bunny"
echo "Question: What colors are prominent?"
echo ""
curl -s "$SERVER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What colors are most prominent in this video?"},
        {"type": "video_url", "video_url": {"url": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"}}
      ]
    }],
    "max_tokens": 100
  }' | python3 -c "import sys,json; r=json.load(sys.stdin); print('Response:', r['choices'][0]['message']['content'])"

echo ""

# 4. Is it animated?
echo "============================================================"
echo "4. Determine Video Type"
echo "============================================================"
echo "Question: Is this an animated or live-action video?"
echo ""
curl -s "$SERVER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Is this an animated or live-action video? How can you tell?"},
        {"type": "video_url", "video_url": {"url": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"}}
      ]
    }],
    "max_tokens": 150
  }' | python3 -c "import sys,json; r=json.load(sys.stdin); print('Response:', r['choices'][0]['message']['content'])"

echo ""
echo "============================================================"
echo "Demo complete!"
echo "============================================================"
