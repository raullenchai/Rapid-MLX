"""OpenAI-compatible streaming load client (engine-agnostic; same prompt for every server).

usage: srvbench.py BASE_URL MODEL LABEL OUT.jsonl --prompt-file F --concurrency N [--max-tokens 256] [--rounds 2]

Per request: TTFT, decode tok/s after the first content token. Per round: aggregate
completion tok/s over the wall time from first submit to last finish, and aggregate
decode tok/s over the window in which all streams were decoding.
"""
import argparse
import http.client
import json
import statistics
import threading
import time
import urllib.parse

ap = argparse.ArgumentParser()
ap.add_argument("base")
ap.add_argument("model")
ap.add_argument("label")
ap.add_argument("out")
ap.add_argument("--prompt-file", required=True)
ap.add_argument("--concurrency", type=int, default=1)
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--warmup", type=int, default=1)
ap.add_argument("--api-key", default=None)
a = ap.parse_args()

PROMPT = open(a.prompt_file).read()
u = urllib.parse.urlparse(a.base)


def one(i, rnd, res):
    body = {
        "model": a.model,
        "messages": [{"role": "user", "content": f"[{rnd}.{i}] " + PROMPT}],
        "max_tokens": a.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    hdr = {"Content-Type": "application/json"}
    if a.api_key:
        hdr["Authorization"] = "Bearer " + a.api_key
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=3600)
    t0 = time.perf_counter()
    c.request("POST", "/v1/chat/completions", json.dumps(body), hdr)
    r = c.getresponse()
    first = last = None
    chunks = 0
    usage = None
    buf = b""
    while True:
        line = r.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        try:
            ev = json.loads(data)
        except ValueError:
            continue
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                now = time.perf_counter()
                if first is None:
                    first = now
                last = now
                chunks += 1
    end = time.perf_counter()
    ntok = (usage or {}).get("completion_tokens") or chunks
    res[i] = dict(status=r.status, t0=t0, first=first, last=last or end, end=end, tokens=ntok, chunks=chunks,
                  prompt_tokens=(usage or {}).get("prompt_tokens"))


def round_(rnd):
    res = [None] * a.concurrency
    th = [threading.Thread(target=one, args=(i, rnd, res)) for i in range(a.concurrency)]
    start = time.perf_counter()
    [t.start() for t in th]
    [t.join() for t in th]
    wall = time.perf_counter() - start
    return res, wall


for w in range(a.warmup):
    round_(f"w{w}")
out = open(a.out, "a")
for rnd in range(a.rounds):
    res, wall = round_(rnd)
    ok = [r for r in res if r and r["status"] == 200 and r["first"]]
    per = [(r["tokens"] - 1) / (r["last"] - r["first"]) for r in ok if r["tokens"] > 1 and r["last"] > r["first"]]
    ttft = sorted(r["first"] - r["t0"] for r in ok)
    total = sum(r["tokens"] for r in ok)
    # overlap window: all streams past first token and none finished
    lo = max(r["first"] for r in ok) if ok else 0
    hi = min(r["last"] for r in ok) if ok else 0
    row = dict(label=a.label, concurrency=a.concurrency, round=rnd, ok=len(ok), prompt_tokens=[r["prompt_tokens"] for r in ok][:1],
               tokens=total, wall_s=round(wall, 2), aggregate_tok_s=round(total / wall, 2),
               per_stream_decode_tok_s_median=round(statistics.median(per), 2) if per else None,
               sum_stream_decode_tok_s=round(sum(per), 2) if per else None,
               ttft_p50_s=round(ttft[len(ttft) // 2], 2) if ttft else None, ttft_max_s=round(ttft[-1], 2) if ttft else None,
               overlap_s=round(hi - lo, 2))
    print(json.dumps(row), flush=True)
    out.write(json.dumps(row) + "\n")
    out.flush()
