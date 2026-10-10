import json
import os
import statistics as st
from pathlib import Path

root = Path(os.environ.get("AGENT_PROBE_OUTPUT_DIR", Path(__file__).parent))
result = {}
for label in ["prefill-2048", "prefill-512", "prefill-2048-recheck"]:
    file = root / (label + ".json")
    if not file.exists():
        continue
    d = json.loads(file.read_text())
    summary = {}
    cache = [r for r in d["runs"] if r["kind"] == "cache"]
    if cache:
        summary["cache"] = {
            key: st.median(r["rows"][i]["ttft_ms"] for r in cache)
            for i, key in enumerate(["cold_ttft_ms", "exact_ttft_ms", "append_ttft_ms"])
        }
        summary["cache"]["exact_output_equal_count"] = sum(
            r["rows"][0]["output_sha256"] == r["rows"][1]["output_sha256"]
            for r in cache
        )
    summary["cold_concurrency"] = {}
    for n in [1, 2, 4]:
        runs = [
            r for r in d["runs"] if r["kind"] == "concurrency" and r["concurrency"] == n
        ]
        if runs:
            summary["cold_concurrency"][n] = dict(
                ttft_ms=st.median(x["ttft_ms"] for r in runs for x in r["rows"]),
                output_tps=st.median(r["output_tps"] for r in runs),
                mean_tpot_ms=st.median(
                    x["mean_tpot_ms"] for r in runs for x in r["rows"]
                ),
            )
    runs = [r for r in d["runs"] if r["kind"] == "contention"]
    if runs:
        summary["contention"] = dict(
            short_ttft_ms=st.median(r["rows"][1]["ttft_ms"] for r in runs),
            short_solo_ttft_ms=st.median(r["short_solo"]["ttft_ms"] for r in runs),
            long_ttft_ms=st.median(r["rows"][0]["ttft_ms"] for r in runs),
            short_max_gap_ms=st.median(r["rows"][1]["sse_gap_max_ms"] for r in runs),
            output_tps=st.median(r["output_tps"] for r in runs),
            short_output_equal_count=sum(
                r["short_solo"]["output_sha256"] == r["rows"][1]["output_sha256"]
                for r in runs
            ),
        )
    warm = root / (label + "-warm.json")
    if warm.exists():
        wd = json.loads(warm.read_text())
        summary["warm_concurrency"] = {}
        for n in [1, 2, 4]:
            runs = [r for r in wd["runs"] if r["concurrency"] == n]
            if runs:
                summary["warm_concurrency"][n] = dict(
                    ttft_ms=st.median(x["ttft_ms"] for r in runs for x in r["rows"]),
                    output_tps=st.median(r["output_tps"] for r in runs),
                    mean_tpot_ms=st.median(
                        x["mean_tpot_ms"] for r in runs for x in r["rows"]
                    ),
                    cached_tokens=[
                        x["usage"]
                        .get("prompt_tokens_details", {})
                        .get("cached_tokens", 0)
                        for r in runs
                        for x in r["rows"]
                    ],
                )
    summary["metal"] = d.get("final_status", {}).get("metal", {})
    multi = root / (label + "-multiturn.json")
    if multi.exists():
        md = json.loads(multi.read_text())
        cold = [rows[0] for rows in md["sessions"]]
        hot = [x for rows in md["sessions"] for x in rows[1:]]
        summary["multiturn"] = dict(
            sessions=len(md["sessions"]),
            warm_turns=len(hot),
            warm_hits=sum(
                x["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0) > 0
                for x in hot
            ),
            cold_ttft_ms=st.median(x["ttft_ms"] for x in cold),
            warm_ttft_ms=st.median(x["ttft_ms"] for x in hot),
            warm_cached_fraction=st.median(
                x["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
                / x["usage"]["prompt_tokens"]
                for x in hot
            ),
        )
    result[label] = summary
print(json.dumps(result, indent=2))
(root / "summary.json").write_text(json.dumps(result, indent=2))
