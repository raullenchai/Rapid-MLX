# TensorFold lane concurrency: rebased qualification

**Verdict: Qwen passes the full rebased matrix and HTTP contracts; Nemotron passes the requested rebase parity smoke. Serial remains the default.**

Owner: Vector. Requalified on 2026-10-16, shared Mac Studio M3 Ultra with 256 GiB unified memory, macOS 26.5.2 build 25F84. Serving commit `02940049e5899f9716f8f140edb54ddec8cf4455`, rebased onto `origin/main` `a1b3b1d21323540eee59d7d610b802afd4d97ae6`. The evidence commit changes documentation only. This report replaces all earlier measurements, including the earlier lone-request, calibration, and HTTP tables; those earlier numbers do not qualify the current Qwen prefill behavior.

TensorFold 0.6.6, immutable runtime revision `cb2ebf0540f42604e2759b2ddef497861e928248`; MLX 0.32.3, MLX-LM 0.31.3, Transformers 5.15.1, Python 3.12.13. No runtime edits, vendoring, checkpoint downloads, push, PR, deployment, or external messages.

## Rebase and compatibility

All eight local commits replayed. The two expected conflicts were resolved by preserving the message-boundary prefill behavior from #4626: tokenizer-derived markers, `PrefillPlan(LaneEngine.prefill_step, openers, 256, assistant)`, and `engine_factory=functools.partial(LaneEngine, prefill_plan=plan)`. The existing test still instantiates that factory and checks every plan field, now for serial and concurrent construction. The opt-in adds `lanes`, `parallel`, and concurrent memory admission alongside the factory. The other six commits replayed unchanged. #4641's pool-worker capability refusals remain intact.

`RAPID_MLX_TENSORFOLD_LANES` accepts 1, 2, 4, or 8, default 1. Qualified opt-ins use a single runtime engine/scheduler with shared rounds and per-stream cache state. Weights are shared. Other profiles refuse concurrent opt-in before weight loading. `BackendExecutionPolicy(capacity, executor)` uses the backend-owned executor and admission capacity; absent the policy, the shared runner keeps its existing serial lock/executor and deadline behavior.

The existing concurrent cancellation registry, iterator finalization, and deferred reservation cleanup remain unchanged. Default serial timeout/shutdown has a separately characterized provider-worker leak in the existing regression suite; this requalification does not claim to fix or qualify serial process shutdown.

## Measurement contract

Eight distinct prompts (four code, four prose), greedy temperature 0, 256 generated native token IDs per request, N-sized waves totaling eight requests per cell. Qwen runs N=1,2,4,8 with three repeats; Nemotron runs only N=1,8 with one repeat as a rebase smoke. Every request is compared against the same prompt's first solo token-ID sequence, including serial repeats. Checkpoints are cleared on the idle scheduler before each cold cell. Cold means prompt-state cold: weights and kernel calibration are already loaded, and startup is excluded. Model construction, sampling, provider rendering, stopping, prefill plans, and runtime scheduler are unchanged.

Rounds/token is measured runtime rounds divided by generated tokens; aggregate tok/s includes prefill, all waves, provider drainage, and wave scheduling. TTFT starts after rendering the chat-template string and measures the first native token, including provider tokenization, prefill and runtime queue time. HTTP SSE timing below starts before the network request and measures first visible content. Peak memory uses the MLX API, reset before each workload. The Qwen serial baseline reads it at completion; later runs also sample it every 5 ms and retain the maximum, because the family runtime can reset the counter during prefill. The sampled peak can miss a transient between samples; it is not host RSS or total load-time peak. Throughput varies with shared-host load and fresh runtime calibration; no universal speed-up or lone-request regression bound is asserted.

## Qwen3.8-27B

| N | Repeats | Rounds/token | Aggregate tok/s | TTFT ms | MLX peak GiB | Mean / peak streams per round | Rounds per repeat | Token-ID parity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 0.3213–0.3213 | 64.65–64.71 | 246.51–260.28 | 15.92–16.28 | 1.00–1.00 / 1 | 658, 658, 658 | pass |
| 2 | 3 | 0.1953–0.1973 | 93.59–94.92 | 246.85–630.76 | 16.12–18.52 | 1.82–1.83 / 2 | 400, 404, 404 | pass |
| 4 | 3 | 0.1255–0.1304 | 113.53–117.18 | 246.80–1192.44 | 16.81–18.55 | 3.43–3.56 / 4 | 257, 267, 262 | pass |
| 8 | 3 | 0.0752–0.0811 | 117.28–123.90 | 252.15–2667.56 | 18.14–20.35 | 6.25–6.67 / 8 | 154, 163, 166 | pass |

## Nemotron 3.5 Lightning

| N | Repeats | Rounds/token | Aggregate tok/s | TTFT ms | MLX peak GiB | Mean / peak streams per round | Rounds per repeat | Token-ID parity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.5156 | 222.42 | 74.81–94.78 | 18.28 | 1.00 / 1 | 1056 | pass |
| 8 | 1 | 0.1216 | 328.90 | 104.04–812.21 | 20.71 | 6.38 / 8 | 249 | pass |

Core user messages, rendered with `enable_thinking=False` and `add_generation_prompt=True`:

1. Write a complete Python LRU cache implementation with type annotations, a detailed explanation, and at least five unit test examples. Explain every method.
2. Write a complete Rust thread pool with graceful shutdown, error handling, and several usage examples. Explain the design in detail.
3. Write a complete SQL schema for a library with loans, reservations, indexes, constraints, and ten example queries. Explain all choices.
4. Write a complete JavaScript event emitter with once listeners, removal, wildcard events, and a thorough test suite. Explain the implementation.
5. Write a detailed essay of at least 1500 words explaining how tides work, including lunar and solar effects and coastal geography.
6. Write a detailed story of at least 1500 words about a botanist discovering a forgotten garden on a mountain. Include dialogue and vivid descriptions.
7. Write a detailed guide of at least 1500 words to organizing a community science fair, with schedules, budgets, accessibility, and safety planning.
8. Write a detailed essay of at least 1500 words on the history of navigation, from stars and compasses to modern satellites. Include examples.

Continuation seeds use prompts 1 and 6 with a newline, `Additional reference notes:`, another newline, followed by 32 newline-separated records for j=0..31: `Record {j}: the workshop team will review item {j}, test its behavior carefully, and document the result.` Each seed answer is appended as an assistant message, then the user asks `Continue with another detailed section of at least 1000 words.` This is a new reproducible fixture; the previous scratch prompts were not retained and their old numbers are excluded.

## Kept prompt state and cancellation

Two distinct conversations are seeded immediately before continuing, with a sub-2048-token reference context long enough to exercise the message boundary. Both continuations are submitted together, including for lanes=1 (where they queue). Each warm result matches a checkpoint-cleared cold result. Lanes=8 also matches lanes=1 for seed and continuation native IDs.

| Conversation | Restored tokens, N=1 | Restored tokens, N=8 | Cold restored tokens, both | Token-ID parity |
| --- | --- | --- | --- | --- |
| 1 | 850 | 850 | 0 | pass |
| 2 | 854 | 854 | 0 | pass |

In-process cancel-one-of-eight verifies the cancelled prefix against solo and the seven surviving complete token sequences; the serial cancellation control also passes. No cancellation workaround or runtime patch is applied.

## Real HTTP and shutdown

Actual `rapid-mlx serve` CLI entrypoint, lanes=8, verified-free loopback port 18738, through the same model-run lock. One pass per HTTP case. Audit records preserve original generated IDs; returned HTTP text is compared separately. The same HTTP histories recur across modes with the existing cache enabled; actual restored-token counts remain in the audit. These are transport-path observations, distinct from the cold matrix, and do not establish a paired cold-run speed-up.

| Case | Requests | Elapsed s | Aggregate tok/s | Visible SSE TTFT ms | Result |
| --- | --- | --- | --- | --- | --- |
| solo | 8 | 33.522 | 61.09 | not observed | pass |
| nonstream | 8 | 17.527 | 116.85 | not observed | pass |
| stream | 8 | 16.274 | 125.84 | 285.43–2478.02 | pass |
| cancel_and_overflow | 8 | 15.025 | 119.40 | 257.30–2455.49 | pass |
| continue_seed | 2 | 11.201 | 45.71 | not observed | pass |
| continued_reference | 2 | 10.154 | 50.42 | not observed | pass |
| continued | 2 | 6.137 | 83.43 | not observed | pass |

All complete streaming/non-streaming token arrays and HTTP text match the HTTP solo references; those reference native IDs also match lanes=1 in-process for the identical core messages. Cancel-one-of-eight leaves seven full matching answers and one matching partial prefix. The ninth simultaneous request returns HTTP 503 with `Retry-After: 1` and `at_capacity`. Continued HTTP conversations match solo HTTP native IDs and visible text for the identical public message history, and both solo and concurrent requests restore kept state. The separate in-process comparison above verifies identical restored counts between lanes=1 and lanes=8.

The timeout probe returns HTTP 504 for a 0.75-second request deadline, with no terminal audit yet (worker still active). SIGTERM to the task-owned server process closes the server in 0.259 seconds. The final audit records 49 generated tokens, proving termination during generation; the server process has exited and the port is free. Only the task-owned process is signalled.

The HTTP client's initial final assertion used a bind without `SO_REUSEADDR`, an insufficient listener check after TCP shutdown. The server had already exited with no owned survivors. Independent verification finds no listener, gets `ECONNREFUSED`, and successfully binds with socket reuse; it also rechecks all captured token/text arrays and 503/504 contracts. `http-port-probe-original/` preserves the original script and result, and `http-port-verification.json` records the correction. The shutdown figure measures process exit, not the later port check. No serving code or measured token/timing value was changed. The reproduction harness now uses socket reuse for its port probe.

## Pool and Desktop follow-up

With opt-in unset, pool registration still advertises exactly one Qwen27 slot. Its alias-specific default and ceiling are 1; it injects `--max-num-seqs 1`. The backend still admits one HTTP request with no execution policy. Desktop still selects the accelerated alias and applies its 48 GiB physical-memory floor plus model availability checks. No pool/Desktop files changed.

A follow-up needs one effective qualified capacity shared by pool registration, explicit flag validation, child launch/environment, runtime lanes and HTTP admission, plus Desktop configuration/status. An inherited lanes opt-in does not raise the pool advertisement today. The current standalone backend takes capacity from lanes rather than `--max-num-seqs`; align those contracts deliberately before enabling pool concurrency. Preserve default one and reject capacity disagreement. Pixel must carry an explicit Desktop opt-in through spawning and display the effective capacity. Validate N-specific memory headroom on floor hardware before changing eligibility: a 256 GiB host run does not qualify eight lanes on a 48 GiB machine. Atlas owns that product/rollout decision; Vector owns backend/pool alignment and Harbor packaged-runtime validation.

## Checks and limits

pytest 9.1.1: 645 passed, 1 skipped in 15.53 seconds: `tests/test_tensorfold_*`, `tests/test_dflash_integration.py`, `tests/test_benchmark_nemotron_prefix.py`, `tests/test_share_quicksilver.py`, and `tests/test_share_cli.py`. The skip is the explicitly disabled optional DFlash real-model E2E, which requires different checkpoints. The cached test extras run includes MLX-VLM 0.7.2 and MLX 0.32.3; the separate TensorFold model measurements use their unchanged minimal environment. The earlier minimal-environment test run passed 624 tests with 22 dependency/opt-in skips. Ruff 0.17.1 check and format check pass for all TensorFold modules/tests plus the shared server (12 files). `git diff --check` passes. Rebase range-diff contains only the two conflict resolutions described above.

Not remeasured: paired lone-request latency, loading/RSS memory, maximum contexts, sampled decoding, forced OOM/admission pressure, eight long continuations, restart/disk restoration, other families, or serial process shutdown. Removed historical tables make no claims for this source. This change remains standalone opt-in only.

## Reproduction and artifacts

All harnesses, venv, logs and raw native token arrays are under `/private/tmp/tensorfold-lane-concurrency-pr/`. `probe-api.py` is the serial baseline observer and `probe.py` adds peak-memory sampling for subsequent cells; both record source commit, script SHA256, runtime VCS provenance, dependency versions, complete IDs, per-request timing, cached-token counts, and per-round stats. `controller.py` runs the matrix sequentially through the shared lock and then `http_check.py`; after a lock timeout, `resume.py` runs only the unfinished cells under one outer 64 GiB lease (14400-second lock timeout), with each child still passing its own model-run memory check; the HTTP harness starts its own real CLI child through that lock. `http-audit.jsonl` records native token IDs without re-encoding text. `REPORT.md` is the local handoff.

```sh
P=/private/tmp/tensorfold-lane-concurrency-pr
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=. TMPDIR="$P" RUFF_CACHE_DIR="$P/ruff-cache"
"$P/venv/bin/python" scripts/large-model-run.py --working-set-gb 64 --lock-timeout 7200 -- \
  "$P/venv/bin/python" "$P/probe.py" --profile qwen27 --lanes 1 --output "$P/qwen27-n1.json"
# Repeat Qwen with --lanes 2/4/8 --baseline "$P/qwen27-n1.json" (three repeats).
# Nemotron uses --working-set-gb 40, --profile nemotron, --repeats 1, N=1/8.
"$P/venv/bin/python" "$P/http_check.py"
"$P/test-venv/bin/python" -m pytest tests/test_tensorfold_* tests/test_dflash_integration.py \
  tests/test_benchmark_nemotron_prefix.py tests/test_share_quicksilver.py tests/test_share_cli.py \
  -q -o cache_dir="$P/pytest-cache" --basetemp="$P/tests-tmp"
"$P/venv/bin/python" -m ruff check rapid_mlx/speculative/tensorfold*.py \
  rapid_mlx/speculative/dflash/server.py tests/test_tensorfold_*.py
"$P/venv/bin/python" -m ruff format --check rapid_mlx/speculative/tensorfold*.py \
  rapid_mlx/speculative/dflash/server.py tests/test_tensorfold_*.py
```

All model snapshots resolve with `local_files_only=True`. Targets remain Qwen revision `70ae7fac63274ff2eac54152031433374cb80f2f`, drafter `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`, and Nemotron `d9d758fb83953437f7263256b0d96157e2a348b8`. No runtime or checkpoint substitutions.
