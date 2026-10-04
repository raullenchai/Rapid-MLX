#!/usr/bin/env python3
"""MTP concurrency client: aggregate + per-request decode tok/s at concurrency 1/2/4.

Used by 2026-10-02-mtp-concurrency-m4-pro.md.

Same tok/s definition as bench_compare.py: decode tps = (engine-reported completion_tokens - 1)
/ (t_last - t_first_token) over SSE chunks carrying non-empty content/reasoning; aggregate =
sum(completion_tokens) / wall-clock around the N concurrent requests. 2 reps per level.
Stdlib only. --thinking default (no chat_template_kwargs sent).
"""

from __future__ import annotations

import argparse
import http.client
import json
import statistics
import threading
import time
import urllib.parse

TOPICS = [
    "the history of the printing press",
    "how tides work",
    "the life cycle of a star",
    "the invention of the telephone",
]


def stream_once(host, port, model, prompt, max_tokens, timeout=600) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        return _stream_on(conn, body)
    finally:
        conn.close()


def _stream_on(conn, body) -> dict:
    t0 = time.perf_counter()
    conn.request(
        "POST",
        "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    if resp.status != 200:
        detail = resp.read()[:800].decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {resp.status}: {detail}")
    t_first = t_last = None
    usage = None
    finish = None
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        if obj.get("usage"):
            usage = obj["usage"]
        for ch in obj.get("choices") or []:
            d = ch.get("delta") or {}
            if (
                (d.get("content") or "")
                + (d.get("reasoning_content") or "")
                + (d.get("reasoning") or "")
            ):
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    t_end = time.perf_counter()
    ct = (usage or {}).get("completion_tokens")
    if not isinstance(ct, int) or ct < 1:
        # Aggregate tok/s is defined on engine-reported counts only.
        raise RuntimeError(f"stream reported no completion_tokens (usage={usage})")
    rec = {
        "ttft_s": (t_first - t0) if t_first is not None else (t_end - t0),
        "total_s": t_end - t0,
        "gen_window_s": (t_last - t_first) if (t_first and t_last) else None,
        "completion_tokens": ct,
        "finish_reason": finish,
        "usage": usage,
    }
    if rec["gen_window_s"] and ct > 1:
        rec["decode_tps"] = (ct - 1) / rec["gen_window_s"]
    else:
        rec["decode_tps"] = None
    return rec


def conc_level(host, port, model, n, max_tokens=256) -> dict:
    prompts = [
        f"Write a detailed 600-word explainer on {TOPICS[i % 4]}." for i in range(n)
    ]
    results = [None] * n
    errors = [None] * n

    def worker(i):
        try:
            results[i] = stream_once(host, port, model, prompts[i], max_tokens)
        except BaseException as exc:  # surfaced after join, not lost in a thread
            errors[i] = exc

    ths = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    t0 = time.perf_counter()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0
    for exc in errors:
        if exc is not None:
            raise exc
    toks = [r["completion_tokens"] for r in results]
    agg = sum(toks) / wall
    return {
        "concurrency": n,
        "wall_s": round(wall, 3),
        "completion_tokens": toks,
        "aggregate_tps": round(agg, 2),
        "per_request_decode_tps": [
            round(r["decode_tps"], 2) if r.get("decode_tps") else None for r in results
        ],
        "per_request_ttft_s": [round(r["ttft_s"], 3) for r in results],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    u = urllib.parse.urlparse(a.base_url)
    if u.hostname not in ("127.0.0.1", "localhost"):
        raise SystemExit("perf_mtp_concurrency: loopback base URLs only")
    if u.scheme != "http" or u.path.rstrip("/") not in ("", "/v1"):
        # Requests always go to http://<host>:<port>/v1/chat/completions.
        raise SystemExit("perf_mtp_concurrency: expected http://<loopback>:<port>/v1")
    stream_once(u.hostname, u.port, a.model, "Say hello.", 16)  # warmup
    levels = []
    for n in (1, 2, 4):
        for _rep in range(2):
            lv = conc_level(u.hostname, u.port, a.model, n)
            lv["rep"] = _rep + 1
            levels.append(lv)
            print(json.dumps(lv))
    summary = {}
    for n in (1, 2, 4):
        aggs = [lv["aggregate_tps"] for lv in levels if lv["concurrency"] == n]
        summary[f"conc{n}"] = {
            "median_aggregate_tps": statistics.median(aggs),
            "reps": aggs,
        }
    with open(a.out, "w", encoding="utf-8") as output:
        json.dump(
            {
                "model": a.model,
                "levels": levels,
                "summary": summary,
                "method": "aggregate = sum(engine-reported completion_tokens)/wall; per-request = (ct-1)/(t_last-t_first); 2 reps",
                "finished_unix": time.time(),
            },
            output,
            indent=1,
        )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
