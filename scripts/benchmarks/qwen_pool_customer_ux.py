import concurrent.futures
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

KEY = Path("/run/secrets/qsp_key").read_text().strip()
N = int(os.environ.get("QWEN_UX_CONCURRENCY", "8"))
OUT = Path(os.environ.get("QWEN_UX_OUT", "/work/results.json"))
PROMPTS = [
    "A shop sells a notebook and pen for $1.10 total. The notebook costs $1.00 more than the pen. Give the pen price and one short equation; answer in English.",
    "Write a Python function `dedupe(items)` that removes duplicates from a list while preserving order, including unhashable values. Keep the answer under 90 words.",
]
GATE = threading.Barrier(N)


def measure(i):
    prompt = PROMPTS[i % 2]
    payload = {
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 160,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        "https://api.quicksilverpro.io/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": "Bearer " + KEY,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "Rapid-MLX-qwen-customer-ux/1.0",
        },
    )
    GATE.wait()
    start = time.monotonic()
    result = {
        "index": i,
        "prompt": i % 2,
        "id": None,
        "http": None,
        "first_event_s": None,
        "first_text_s": None,
        "total_s": None,
        "event_count": 0,
        "content": "",
        "reasoning_chars": 0,
        "usage": None,
        "finish_reason": None,
    }
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            result["http"] = response.status
            for raw in response:
                if not raw.startswith(b"data: "):
                    continue
                body = raw[6:].strip()
                if body == b"[DONE]":
                    break
                try:
                    event = json.loads(body)
                except ValueError:
                    continue
                now = time.monotonic()
                result["event_count"] += 1
                if result["first_event_s"] is None:
                    result["first_event_s"] = round(now - start, 3)
                if result["id"] is None:
                    result["id"] = event.get("id")
                if event.get("usage"):
                    result["usage"] = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    content = delta.get("content") or ""
                    if content and result["first_text_s"] is None:
                        result["first_text_s"] = round(now - start, 3)
                    result["content"] += content
                    result["reasoning_chars"] += len(
                        delta.get("reasoning_content") or ""
                    )
                    if choice.get("finish_reason"):
                        result["finish_reason"] = choice["finish_reason"]
    except urllib.error.HTTPError as exc:
        result["http"] = exc.code
        result["error"] = exc.read(512).decode(errors="replace")
    except Exception as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
    result["total_s"] = round(time.monotonic() - start, 3)
    return result


with concurrent.futures.ThreadPoolExecutor(max_workers=N) as executor:
    results = sorted(executor.map(measure, range(N)), key=lambda x: x["index"])
OUT.write_text(json.dumps(results, indent=2))
print(
    json.dumps(
        [
            {k: v for k, v in r.items() if k not in ("content", "error")}
            for r in results
        ],
        indent=2,
    )
)
