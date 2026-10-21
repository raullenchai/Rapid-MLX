"""Summarize window results: in-process A/B vs main, and server metrics vs the same-machine reference.

usage: summarize.py results.jsonl server.jsonl
Reference rows in server.jsonl carry labels starting with "ref-" (written by refbench.sh via srvbench.py).
"""
import json
import statistics
import sys
from collections import defaultdict

res = [json.loads(l) for l in open(sys.argv[1]) if l.strip()] if len(sys.argv) > 1 else []
srv = []
try:
    srv = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
except (IndexError, FileNotFoundError):
    pass

dec = defaultdict(list)
for r in res:
    if r.get("phase") == "decode":
        dec[(r["label"].rsplit("-", 1)[0], r["ctx"])].append(r["step_ms"])
base = {ctx: statistics.median(v) for (n, ctx), v in dec.items() if n == "main"}
print("== in-process decode (median step ms over reps; tok/s = 1000/step)")
for (n, ctx), v in sorted(dec.items(), key=lambda kv: (kv[0][1], statistics.median(kv[1]))):
    m = statistics.median(v)
    b = base.get(ctx)
    print(f"{n:>10} ctx={ctx:>5}  {m:7.2f} ms  {1000 / m:6.2f} tok/s  reps={len(v)}" + (f"  vs main {b / m:5.2f}x" if b else ""))
print("== other phases")
for r in res:
    if r.get("phase") in ("prefill_raw", "stream_prefill", "draft", "batched", "ttft_stream"):
        keep = {k: v for k, v in r.items() if k not in ("src", "device", "note")}
        print("  ", json.dumps(keep))
print("== server (same 1008-token prompt; median over rounds)")
agg = defaultdict(list)
for r in srv:
    agg[(r["label"], r["concurrency"])].append(r)
for (lab, c), rows in sorted(agg.items()):
    a = statistics.median(x["aggregate_tok_s"] for x in rows)
    s = [x["per_stream_decode_tok_s_median"] for x in rows if x["per_stream_decode_tok_s_median"]]
    t = [x["ttft_p50_s"] for x in rows if x["ttft_p50_s"] is not None]
    print(f"{lab:>16} c={c}  aggregate {a:7.2f} tok/s  per-stream decode {statistics.median(s) if s else float('nan'):7.2f}"
          f"  ttft p50 {statistics.median(t) if t else float('nan'):6.2f}s  prompt_tokens={rows[0].get('prompt_tokens')}")
