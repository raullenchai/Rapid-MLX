#!/bin/bash
# Demo: curl API - Image Analysis
#
# Shows how to use rapid-mlx with curl for image understanding.
#
# Usage:
#   1. Start the server with a VLM model:
#      rapid-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit --port 8000
#
#   2. Run this script:
#      bash examples/demo_curl_image.sh

SERVER_URL="http://localhost:8000"

echo "============================================================"
echo "curl API Demo - Image Analysis"
echo "============================================================"

# Check server health
echo ""
echo "Checking server health..."
curl -s "$SERVER_URL/health" | python3 -m json.tool
echo ""

# 1. Image from URL
echo "============================================================"
echo "1. Analyze Image from URL"
echo "============================================================"
echo "Image: Cat photo from Wikipedia"
echo "Question: What animal is in this image?"
echo ""
curl -s "$SERVER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What animal is in this image? Describe it briefly."},
        {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"}}
      ]
    }],
    "max_tokens": 150
  }' | python3 -c "import sys,json; r=json.load(sys.stdin); print('Response:', r['choices'][0]['message']['content'])"

echo ""

# 2. Another image
echo "============================================================"
echo "2. Describe a Landmark"
echo "============================================================"
echo "Image: Empire State Building from Wikipedia"
echo "Question: What famous building is this?"
echo ""
curl -s "$SERVER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What famous building is shown in this image? Where is it located?"},
        {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/Empire_State_Building_%28aerial_view%29.jpg/800px-Empire_State_Building_%28aerial_view%29.jpg"}}
      ]
    }],
    "max_tokens": 150
  }' | python3 -c "import sys,json; r=json.load(sys.stdin); print('Response:', r['choices'][0]['message']['content'])"

echo ""

# 3. Base64 encoded image
echo "============================================================"
echo "3. Analyze Base64 Encoded Image"
echo "============================================================"
echo "Creating a simple red 10x10 PNG image..."

# Create a simple red PNG using Python (smallest valid PNG)
BASE64_IMAGE=$(python3 -c "
import base64
from PIL import Image
import io
img = Image.new('RGB', (10, 10), color='red')
buffer = io.BytesIO()
img.save(buffer, format='PNG')
print(base64.b64encode(buffer.getvalue()).decode('utf-8'))
")

echo "Question: What color is this image?"
echo ""
curl -s "$SERVER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"default\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"text\", \"text\": \"What color is this image?\"},
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,$BASE64_IMAGE\"}}
      ]
    }],
    \"max_tokens\": 50
  }" | python3 -c "import sys,json; r=json.load(sys.stdin); print('Response:', r['choices'][0]['message']['content'])"

echo ""

# 4. Food image analysis
echo "============================================================"
echo "4. Analyze Food Image"
echo "============================================================"
echo "Image: Food display from Wikipedia"
echo "Question: What foods do you see?"
echo ""
curl -s "$SERVER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What foods do you see in this image? List them."},
        {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Good_Food_Display_-_NCI_Visuals_Online.jpg/800px-Good_Food_Display_-_NCI_Visuals_Online.jpg"}}
      ]
    }],
    "max_tokens": 200
  }' | python3 -c "import sys,json; r=json.load(sys.stdin); print('Response:', r['choices'][0]['message']['content'])"

echo ""
echo "============================================================"
echo "Demo complete!"
echo "============================================================"
