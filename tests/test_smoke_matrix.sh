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

# Helper: streaming request, collect all content + reasoning_content deltas.
# reasoning_content is included so length-based comparisons (e.g. test 4)
# correctly measure thinking output that the reasoning parser routes there.
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
    except json.JSONDecodeError:
        continue
    choices = d.get('choices') or []
    if not choices:
        continue
    delta = choices[0].get('delta', {}) if isinstance(choices[0], dict) else {}
    text += delta.get('content', '') or ''
    text += delta.get('reasoning_content', '') or ''
print(text)
" 2>/dev/null
}

# Helper: streaming request, return content and reasoning_content lengths
# separately on a single line as "<content_len>|<reasoning_len>". Used by
# test 4 to verify the reasoning parser is actually splitting thinking
# tokens into ``reasoning_content`` — combined length is unreliable because
# some models compensate for disabled thinking by writing longer answers
# in ``content`` (e.g. qwen3.5-4b-4bit on simple math drops thinking ratio
# below the previous 1.5x heuristic).
stream_chat_split() {
    local body="$1"
    curl -sfN "${BASE}" \
        -H "Content-Type: application/json" \
        -d "${body}" 2>/dev/null \
    | sed -n 's/^data: //p' \
    | python3 -c "
import sys, json
content = reasoning = ''
for line in sys.stdin:
    line = line.strip()
    if not line or line == '[DONE]':
        continue
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        continue
    choices = d.get('choices') or []
    if not choices:
        continue
    delta = choices[0].get('delta', {}) if isinstance(choices[0], dict) else {}
    content += delta.get('content', '') or ''
    reasoning += delta.get('reasoning_content', '') or ''
print(f'{len(content)}|{len(reasoning)}')
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
# 4. enable_thinking=true (default) vs false — verify the reasoning
#    parser actually separates thinking tokens into reasoning_content.
#
#    Earlier versions of this test compared combined content+reasoning
#    length and required thinking-on to be 1.5x longer than thinking-off.
#    That's unreliable: when thinking is disabled, smaller models often
#    compensate by writing a longer step-by-step answer in ``content``,
#    which collapses the ratio below 1.5x even though the toggle is
#    working correctly. The deterministic signal is the split itself:
#    thinking-on should produce reasoning_content deltas; thinking-off
#    must not.
# ---------------------------------------------------------------
echo "[4/5] enable_thinking=true vs false (streaming, parser split)"
THINK_SPLIT=$(stream_chat_split '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is 17 * 23?"}],
    "max_tokens": 500,
    "temperature": 0,
    "stream": true
}')
THINK_CONTENT_LEN=${THINK_SPLIT%|*}
THINK_REASONING_LEN=${THINK_SPLIT#*|}
echo "    thinking=default: content=${THINK_CONTENT_LEN} reasoning=${THINK_REASONING_LEN}"

NOTHINK_SPLIT=$(stream_chat_split '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is 17 * 23?"}],
    "max_tokens": 500,
    "temperature": 0,
    "stream": true,
    "enable_thinking": false
}')
NOTHINK_CONTENT_LEN=${NOTHINK_SPLIT%|*}
NOTHINK_REASONING_LEN=${NOTHINK_SPLIT#*|}
echo "    thinking=false:   content=${NOTHINK_CONTENT_LEN} reasoning=${NOTHINK_REASONING_LEN}"

SPLIT_OK=$(python3 -c "
tc, tr = $THINK_CONTENT_LEN, $THINK_REASONING_LEN
nc, nr = $NOTHINK_CONTENT_LEN, $NOTHINK_REASONING_LEN
# Thinking on: parser must route some thinking tokens into
# reasoning_content. The absolute floor is intentionally small (10
# chars) so the test passes even when a model thinks tersely.
# Thinking off: reasoning_content MUST be empty — anything non-zero
# means the chat template's enable_thinking=false directive was ignored
# and the parser saw a <think> block it shouldn't have.
if tr >= 10 and nr == 0 and (tc + tr) > 0 and (nc + nr) > 0:
    print('THINKING_SPLIT_OK')
else:
    print('SPLIT_BROKEN')
" 2>/dev/null)
check "thinking toggle splits reasoning_content correctly" "$SPLIT_OK" "THINKING_SPLIT_OK"
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
