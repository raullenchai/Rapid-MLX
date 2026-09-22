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

Worker result after the per-process active-day memo was added:
`n=500 p50_ms=0.5171 p95_ms=0.6275 max_ms=2.0481`.

The response-path benchmark timed 1,000 calls to the public
`emit_completed_request()` while the then-current dedicated telemetry worker performed
the corresponding SQLite update. Each iteration timed only the public call,
then waited outside the timed region for that real worker item to finish before
starting the next sample. No latch or replacement worker was used. The test
suite separately holds the real SQLite write lock in a second process and
requires both the request-side call and an unrelated `asyncio.to_thread()` call
to return within 50 ms.

On-loop result: `n=1000 p50_us=4.29 p95_us=5.50 max_us=8.42`. This historical
measurement used executor admission/submission with an idle queue; review round
3 replaced that executor with a lazy daemon thread so interpreter shutdown can
never join a SQLite-blocked telemetry worker. SQLite, active-day, registry, and
PostHog work still run outside asyncio's shared default executor. At most 64
items wait in the queue in addition to the item currently running; further
items are dropped immediately. At process exit queued items are dropped without
draining or joining, and `fork()` resets the child to a fresh queue and worker
state.

The worker p50 remains below the T8 1 ms budget, while the request coroutine
pays the live consent gate plus bounded queue admission.
Successful emits remain after response
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

The closed registry currently has 8 endpoint values (including `other`), 26
caller values, and 2 result values: 416 worst-case counter keys per model.
`store.MAX_KEYS = 12_000` therefore holds every combination for 28 complete
models (`28 × 416 = 11,648`); the 29th model is where a fully saturated
worst-case installation begins exhausting new keys. The cap was raised from
2,000 because that allowed only 5 complete models. Even 12,000 rows remain a
small local SQLite database, and existing keys continue counting at the cap.
