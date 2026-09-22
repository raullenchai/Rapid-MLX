# Telemetry v2 inference-path overhead

Measured on 2026-09-21 on macOS 26.5.2, Apple M3 Ultra, arm64, with Python
3.12.14 from the validation environment.

The worker benchmark timed 500 sequential calls to the real worker half of
`emit_completed_request()` after 50 warmups. Each sample included model,
endpoint, caller and result normalization; one real SQLite `store.record()`;
the real registry/envelope `track()` path at bucket crossings with an injected
in-memory sender; and the successful request's real `emit_active_day()` claim.
The database lived under an isolated explicit temporary `HOME`. The
official-build stamp, consent result, version (`0.15.1`), and Apple Silicon
platform facts were injected exactly as in the loopback emitter test.

Worker result: `n=500 p50_ms=0.8443 p95_ms=1.2525 max_ms=1.8033`.

The response-path benchmark timed 1,000 calls to the public
`emit_completed_request()` with its real build/consent gate and default-executor
submission. The worker was replaced with an in-memory completion latch so the
measurement isolates work that remains on the event loop; the test suite
separately holds the real SQLite write lock in a second process and requires the
request-side call to return within 50 ms.

On-loop result: `n=1000 p50_us=3.62 p95_us=18.04 max_us=1597.42`. The maximum
includes executor cold-start; steady-state p50 is 3.62 microseconds. SQLite and
active-day work are fire-and-forget and cannot delay the response coroutine.

The worker p50 remains below the T8 1 ms budget, while the request coroutine now
pays only executor submission. Successful emits remain after response
serialization or the streaming terminal marker; generation errors use the
separate `failed` counter and client disconnects emit nothing.

Reproduce the worker measurement by timing this body with
`time.perf_counter_ns()` under an explicit temporary `HOME` and the injected
official-build/consent/context fixtures described above:

```python
inference._record_completed_request(
    model="<custom>",
    endpoint="/v1/chat/completions",
    caller_agent="cursor/1.0",
    caller_client=None,
    result="ok",
)
```

## Counter cardinality

The closed registry currently has 8 endpoint values (including `other`), 21
caller values, and 2 result values: 336 worst-case counter keys per model.
`store.MAX_KEYS = 12_000` therefore holds every combination for 35 complete
models (`35 × 336 = 11,760`); the 36th model is where a fully saturated
worst-case installation begins exhausting new keys. The cap was raised from
2,000 because that allowed only 5 complete models. Even 12,000 rows remain a
small local SQLite database, and existing keys continue counting at the cap.
