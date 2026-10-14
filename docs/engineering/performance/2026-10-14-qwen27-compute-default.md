# Qwen27 Compute / Earn default and packaged acceptance

Date: 2026-10-14 UTC. Owner: Vector. No production release was performed.

## Result and configuration

Compute / Earn's Qwen3.8 27B catalog entry now selects
`qwen3.8-27b-tensorfold` in both the CLI and Desktop. The legacy pool spelling
`qwen3.8-27b-4bit` selects the same default. Ordinary local model selection is
unchanged. The message-aware in-memory prefix plan from #4626 is automatic;
pool mode continues to disable disk snapshots. The node advertises one slot,
matching the qualified HTTP backend's admission limit. Desktop enforces the
qualified 48 GiB memory floor, including when no catalog metadata is available.

The full Desktop build includes TensorFold 0.6.6 at the adapter's immutable
source revision and MLX 0.32.3. The original main build lacked TensorFold.
An intermediate package then exposed a real startup failure: source hoisting
deleted `tensorfold/drafters/vendor/z_lab_dflash/model_mlx.py`, which the runtime
loads explicitly by filename. TensorFold source files are now preserved, and
the post-trim build smoke invokes that vendored loader before signing.

TensorFold remains a text backend. In pool mode, unsupported capabilities now
return 503 so QuickSilver can use a compatible provider. The unmodified final
build was observed returning 503 for a valid forced weather-tool request;
the public API returned 200 with `get_weather({"city":"San Francisco"})`,
`finish_reason=tool_calls` and usage. The observation launcher logged only
request digests, IDs and response statuses; it did not rewrite responses.
Standalone serving retains 400 for unsupported features. Shared malformed
request validation, including empty messages and unsupported `n>1`, also
retains 400. This change does not implement local TensorFold tool decoding.

## Environment and measurements

Same M4 Pro worker as the preceding comparison: 12 CPU cores, 16 GPU cores,
48 GiB, macOS 26.5.1. Models were copied from existing qualified Studio
snapshots into the worker's normal HF cache; no external model downloads.

- Base: `7ac7ad58bc8cabe609c4c5ee737707b5ff8f07bc`, including #4626.
- Initial packaged engine: `b598a0ccf9eecc8e1f42aae55d4035af34b605c9`.
- Final tested code: `c6e1567414b2f5a9fdaff369dbc740d69102fe23`.
- TensorFold: `cb2ebf0540f42604e2759b2ddef497861e928248`; MLX/Metal 0.32.3.
- Target: `Vontra/Qwen3.8-27B-MLX-4bit` at
  `70ae7fac63274ff2eac54152031433374cb80f2f`.
- Drafter: `z-lab/Qwen3.8-27B-DFlash2` at
  `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`, runtime-packed to four bits.

TTFT means first nonempty SSE content at the client. The workload uses a short
opening, an answer of `4`, then 140 repetitions of `alpha beta gamma delta. `
and another instruction to answer only `4`. Repeat that request exactly, then
append an assistant answer and a short follow-up. Each wave changes an early
part of the long turn. Temperature is zero and the output cap is 32 tokens.
Startup, runtime calibration and warm-up are excluded.

| Client / build | Cold long turn | Exact repeat | Follow-up |
| --- | ---: | ---: | ---: |
| Initial packaged loopback, three-wave medians | 7.506 s | 0.214 s | 0.351 s |
| Initial QuickSilver public path, three-wave medians | 8.200 s | 0.805 s | 0.833 s |
| Final QuickSilver public path, three-wave medians | 9.036 s | 1.595 s | 1.547 s |

All 14 initial loopback calls passed (including warm-up, short turns and a
nonstream call). All nine final public wave calls returned `4`, DONE and usage,
and matched fresh relay request records on the test node. Matching excludes
IDs observed before each call so a previous repeat cannot provide false proof
of current routing. The final build's additional confirmation measured
8.659 / 0.811 / 1.068 seconds for cold/repeat/follow-up. An immediate loopback
repeat on that final node measured 0.272 seconds. Public delay therefore
includes substantial variability outside the local inference path; this test
does not isolate its network, relay and gateway components.

One tool probe timed out before any observation on the test node (34.121 s).
The identical request succeeded on retry in 3.176 s through another provider.
Keep that sample as a service-level uncertainty rather than attributing it to
this worker or presenting an unconditional reliability claim.

Stopping the task node left the known conversation available publicly:
200, DONE, usage and the correct answer in 1.538 s, with TTFT 1.473 s.
Restarting reused the same registration without the provider account key.
After restart, the previously cached long turn was cold again (TTFT 8.469 s);
its repeat was 0.779 s. In-memory reuse does not survive process restart.

The [distilled fixture](fixtures/2026-10-14-qwen27-compute-default.json) retains
latency samples and protocol results without credentials, host addresses,
prompt text or relay IDs. Public `usage.cost` was present on successful calls;
contributor earnings were not verified because the read credential was unavailable.

## Reproduction and remaining acceptance

Build the full app from the candidate source, with no source overlay:

```sh
SKIP_BUNDLED_MODEL=1 bash apps/rapid-mac/scripts/build.sh
```

For loopback measurement, use the built bundle's `rapid-mlx` entrypoint:

```sh
rapid-mlx serve qwen3.8-27b-tensorfold --host 127.0.0.1 --port 18763 \
  --disable-disk-caches --no-thinking --served-model-name qwen3.8-27b
```

For pool acceptance, use the default catalog spelling without speculative
flags. Provide the account key over stdin or the existing interactive prompt:

```sh
rapid-mlx share qwen3.8-27b --quicksilver --worker acceptance-node
```

The initial Desktop refresh reused the already built unchanged engine while
updating its GUI memory floor. The final code above was rebuilt as a complete
app and sidecar; native pool refusals were verified from that sealed artifact.
Ad-hoc strict code-sign verification passed on Studio and the borrowed worker.
Desktop launched with isolated preferences and auto-start off. Its 12 Share
Compute tests cover the actual default spawn arguments, stdin-only provider
key transport, registration binding and 32/48 GiB eligibility. GUI button
automation was unavailable due Accessibility permissions; a real GUI join was
not asserted.

Focused Python checks: 379 passed. Adjacent TensorFold and sidecar-smoke
contracts: 186 passed. Ruff, formatting and shell syntax passed. These are
proportional feature checks, not a claim of a completed release matrix.

Cold prefill remains roughly 7.5 seconds locally for this workload. Public
warm latency varies, a transport timeout was observed, contributor earnings
remain unverified, and the shared MLX bundle pin still needs release-level
family acceptance. Harbor owns signed/notarized candidate acceptance; Atlas
and the QuickSilver owner own rollout and cold-request routing. Cancellation
propagation remains outside scope, as requested.

Task processes, the worker app, copied models, metadata audit and node
credentials were removed; both task ports were closed and the lease released.
The Studio local app remains available for review.
