# Telemetry v2 inference-path overhead

Measured on 2026-09-21 on macOS 26.5.2, Apple M3 Ultra, arm64, with Python
3.12.14 from the validation environment.

The benchmark timed 500 sequential first-bucket crossings. Each sample did one
real SQLite `store.record()` against an isolated temporary `HOME`, followed by
one real registry/envelope `track()` call with an injected in-memory sender.
The official-build stamp, consent result, version (`0.15.1`), and Apple Silicon
platform facts were injected exactly as in the loopback emitter test. This is a
deliberately pessimistic workload: production calls `track()` only at the
logarithmic bucket crossings, while `store.record()` runs once per completed
request.

Result: `n=500 p50_ms=0.5933 p95_ms=0.7474 max_ms=2.8442`.

The p50 is below the T8 budget of 1 ms. All six call sites are beside the
existing terminal v1 request emission, after non-streaming generation has
completed or after the streaming terminal marker has been yielded and the
generator resumes. The measurement excludes model generation and response
serialization; it isolates the additive local telemetry work.

Reproduce by timing this body with `time.perf_counter_ns()` for 500 unique keys
under an explicit temporary `HOME`:

```python
crossing = store.record(f"bench|model-{i}|/v1/chat/completions|cursor|ok")
if crossing:
    track.track(
        "inference_bucket_reached",
        {
            "model": "<custom>",
            "endpoint": "/v1/chat/completions",
            "caller": "cursor",
            "result": "ok",
            "count_bucket": crossing.bucket,
            "bucket_source": crossing.bucket_source,
        },
    )
```
