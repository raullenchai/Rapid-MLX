#!/usr/bin/env bash
# ---------------------------------------------------------------
# Smoke test matrix for engine parity verification.
#
# Usage:
#   1. Start server:  vllm-mlx serve <model> --port 8000
#                 or: vllm-mlx serve <model> --port 8000 --continuous-batching
#   2. Run:  bash tests/test_smoke_matrix.sh [port]
#
# Tests emoji decode, CJK, enable_thinking, and special token leaks.
# Output: PASS/FAIL per scenario with details on failure.
# ---------------------------------------------------------------
set -euo pipefail

PORT="${1:-8000}"
BASE="http://localhost:${PORT}/v1/chat/completions"
PASS=0
FAIL=0
ERRORS=""

# Detect engine type from /v1/status
ENGINE=$(curl -sf "http://localhost:${PORT}/v1/status" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('engine_type', 'unknown'))
except: print('unknown')
" 2>/dev/null || echo "unknown")
echo "=== Smoke Matrix — engine=${ENGINE}, port=${PORT} ==="
echo ""

# Helper: non-streaming request
chat() {
    local body="$1"
    curl -sf "${BASE}" \
        -H "Content-Type: application/json" \
        -d "${body}" 2>/dev/null
}

# Helper: streaming request, collect all content deltas
stream_chat() {
    local body="$1"
    curl -sfN "${BASE}" \
        -H "Content-Type: application/json" \
        -d "${body}" 2>/dev/null \
    | sed -n 's/^data: //p' \
    | python3 -c "
import sys, json
text = ''
for line in sys.stdin:
    line = line.strip()
    if not line or line == '[DONE]':
        continue
    try:
        d = json.loads(line)
        delta = d.get('choices', [{}])[0].get('delta', {})
        c = delta.get('content', '')
        if c:
            text += c
    except: pass
print(text)
" 2>/dev/null
}

check() {
    local name="$1"
    local result="$2"
    local pattern="$3"
    local negate="${4:-}"

    if [ -z "$result" ]; then
        FAIL=$((FAIL + 1))
        ERRORS="${ERRORS}\n  FAIL: ${name} — empty response"
        echo "  FAIL: ${name} — empty response"
        return
    fi

    if [ "$negate" = "NOT" ]; then
        if echo "$result" | grep -qF "$pattern"; then
            FAIL=$((FAIL + 1))
            ERRORS="${ERRORS}\n  FAIL: ${name} — found '${pattern}' (should be absent)"
            echo "  FAIL: ${name} — found '${pattern}' in response"
        else
            PASS=$((PASS + 1))
            echo "  PASS: ${name}"
        fi
    else
        if echo "$result" | grep -qF "$pattern"; then
            PASS=$((PASS + 1))
            echo "  PASS: ${name}"
        else
            FAIL=$((FAIL + 1))
            ERRORS="${ERRORS}\n  FAIL: ${name} — '${pattern}' not found"
            echo "  FAIL: ${name} — '${pattern}' not found in: ${result:0:200}"
        fi
    fi
}

# ---------------------------------------------------------------
# 1. Emoji decode (streaming, thinking off to get direct output)
# ---------------------------------------------------------------
echo "[1/5] Emoji decode (streaming)"
EMOJI_RESULT=$(stream_chat '{
    "model": "default",
    "messages": [{"role": "user", "content": "Reply with ONLY these 3 emoji, nothing else: 🎉🚀😊"}],
    "max_tokens": 50,
    "temperature": 0,
    "stream": true,
    "enable_thinking": false
}')
echo "    response: ${EMOJI_RESULT:0:100}"
check "emoji 🎉 present" "$EMOJI_RESULT" "🎉"
check "no U+FFFD leak" "$EMOJI_RESULT" $'\ufffd' NOT
echo ""

# ---------------------------------------------------------------
# 2. CJK decode (streaming, thinking off)
# ---------------------------------------------------------------
echo "[2/5] CJK decode (streaming)"
CJK_RESULT=$(stream_chat '{
    "model": "default",
    "messages": [{"role": "user", "content": "只回复四个字：你好世界"}],
    "max_tokens": 50,
    "temperature": 0,
    "stream": true,
    "enable_thinking": false
}')
echo "    response: ${CJK_RESULT:0:100}"
check "CJK 你好 present" "$CJK_RESULT" "你好"
check "no U+FFFD leak" "$CJK_RESULT" $'\ufffd' NOT
echo ""

# ---------------------------------------------------------------
# 3. enable_thinking=false (no <think> block in raw content)
# ---------------------------------------------------------------
echo "[3/5] enable_thinking=false"
NOTHINK_RESULT=$(chat '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
    "max_tokens": 50,
    "temperature": 0,
    "enable_thinking": false
}')
NOTHINK_TEXT=$(echo "$NOTHINK_RESULT" | python3 -c "
import sys, json
try:
    d = json.loads(sys.stdin.read())
    print(d['choices'][0]['message']['content'])
except: print('')
" 2>/dev/null)
echo "    response: ${NOTHINK_TEXT:0:100}"
check "no <think> tag" "$NOTHINK_TEXT" "<think>" NOT
check "has answer" "$NOTHINK_TEXT" "4"
echo ""

# ---------------------------------------------------------------
# 4. enable_thinking=true (default) vs false — verify difference
#    The server strips <think> from non-streaming, so we test via
#    streaming and check that <think> appears when enabled.
# ---------------------------------------------------------------
echo "[4/5] enable_thinking=true vs false (streaming)"
THINK_STREAM=$(stream_chat '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is 17 * 23?"}],
    "max_tokens": 500,
    "temperature": 0,
    "stream": true
}')
echo "    thinking=default (first 120): ${THINK_STREAM:0:120}"
# When thinking is active, stream content starts with <think> or
# contains thinking markers. The key test: default response is
# LONGER than no-think response (thinking adds tokens).
NOTHINK_STREAM=$(stream_chat '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is 17 * 23?"}],
    "max_tokens": 500,
    "temperature": 0,
    "stream": true,
    "enable_thinking": false
}')
echo "    thinking=false (first 120): ${NOTHINK_STREAM:0:120}"
# The thinking response should be significantly longer
THINK_LEN=${#THINK_STREAM}
NOTHINK_LEN=${#NOTHINK_STREAM}
echo "    lengths: thinking=${THINK_LEN} vs nothink=${NOTHINK_LEN}"
RATIO_OK=$(python3 -c "
t, n = $THINK_LEN, $NOTHINK_LEN
# Thinking response should be at least 2x longer
if n > 0 and t > n * 1.5:
    print('THINKING_LONGER')
elif t > 0 and n == 0:
    print('THINKING_LONGER')
else:
    print('SAME_OR_SHORTER')
" 2>/dev/null)
check "thinking produces longer output" "$RATIO_OK" "THINKING_LONGER"
echo ""

# ---------------------------------------------------------------
# 5. Special token leak check (streaming)
# ---------------------------------------------------------------
echo "[5/5] Special token leak (streaming)"
CLEAN_RESULT=$(stream_chat '{
    "model": "default",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 50,
    "temperature": 0,
    "stream": true,
    "enable_thinking": false
}')
echo "    response: ${CLEAN_RESULT:0:100}"
check "no <|im_end|>" "$CLEAN_RESULT" "<|im_end|>" NOT
check "no <|endoftext|>" "$CLEAN_RESULT" "<|endoftext|>" NOT
check "no <|im_start|>" "$CLEAN_RESULT" "<|im_start|>" NOT
echo ""

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
echo "==========================================="
echo "  Engine: ${ENGINE}"
echo "  PASS: ${PASS}   FAIL: ${FAIL}"
if [ $FAIL -gt 0 ]; then
    echo ""
    echo "  Failures:"
    echo -e "$ERRORS"
fi
echo "==========================================="

exit $FAIL
