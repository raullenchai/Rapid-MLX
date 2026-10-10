import concurrent.futures as cf
import json
import sys
import threading
import time
from probe import ROOT, clear, messages, request, status

label = sys.argv[1]
out = {"label": label, "runs": []}
for concurrency in [1, 2, 4]:
    prompts = [messages(4096, f"warm-session-{i}") for i in range(concurrency)]
    for rep in range(3):
        clear()
        populate = [request(p, 8) for p in prompts]
        barrier = threading.Barrier(concurrency)
        with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
            started = time.perf_counter()
            fs = [pool.submit(request, p, 96, barrier) for p in prompts]
            rows = [f.result() for f in fs]
            wall = time.perf_counter() - started
        item = dict(
            concurrency=concurrency,
            repeat=rep,
            populate=populate,
            rows=rows,
            wall_s=wall,
            output_tps=sum(r["usage"]["completion_tokens"] for r in rows) / wall,
        )
        out["runs"].append(item)
        print(
            json.dumps(
                dict(
                    label=label,
                    concurrency=concurrency,
                    repeat=rep,
                    ttft_ms=[round(x["ttft_ms"], 1) for x in rows],
                    cached=[
                        x["usage"]
                        .get("prompt_tokens_details", {})
                        .get("cached_tokens", 0)
                        for x in rows
                    ],
                    output_tps=item["output_tps"],
                )
            ),
            flush=True,
        )
        (ROOT / (label + "-warm.json")).write_text(json.dumps(out, indent=2))
out["status"] = status()
(ROOT / (label + "-warm.json")).write_text(json.dumps(out, indent=2))
