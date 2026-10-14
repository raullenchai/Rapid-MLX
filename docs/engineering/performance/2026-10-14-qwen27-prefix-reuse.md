# Qwen 27B TensorFold in-memory prefix reuse

Date: 2026-10-14 UTC. Owner: Vector. Scope: the qualified Qwen3.8-27B
TensorFold backend; no production deployment or pool policy changes.

## Root cause

The adapter correctly computed a 748-token reusable history for a 755-token
prompt. However, it constructed `ChatApp` without the message-aware engine
factory used by the runtime's serving path. The default `LaneEngine` plan is
a plain 2048-token grid. Its only chunk start for this prompt was zero, so
the scheduler floored the 748-token checkpoint candidate to zero and saved
nothing. Subsequent requests found an empty store and repeated all prefill.

Temporary metadata-only diagnostics on the real model confirmed the chain:

| Stage | Release backend | Patched backend |
| --- | --- | --- |
| Input/history length | 755 / 748 | 755 / 748 |
| Prompt chunk starts | `[0]` | `[0, 748]` |
| Checkpoint requested on cold fill | none | 748 |
| Saved checkpoint | none | 748 |
| Cache restored on repeat | 0 | 748 |

The fix supplies a `LaneEngine` factory with a `PrefillPlan` derived from the
loaded tokenizer's message markers, a 2048-token maximum chunk and a
256-token minimum cut spacing. This follows the already implemented
TensorFold serving/family adapter pattern. It preserves exact recurrent-state
boundaries instead of truncating GDN state. `snapshot_dir=None` remains in
place: all reuse here is in memory; prompt snapshots stay disabled.

## Measurements

Same M4 Pro worker as the earlier acceptance: 12 CPU cores, 16 GPU cores,
48 GiB, macOS 26.5.1. Same physical target/drafter snapshots in the standard
HF cache, copied from the existing qualified Studio cache. Servers ran alone
over loopback HTTP on port 18761, with one request at a time and thinking off.

- Rapid code baseline: signed Desktop 0.16.1 sidecar,
  `df1f4db1fc0548ec97c6376739a76f6330e9b000`.
- Optional runtime: TensorFold 0.6.6,
  `cb2ebf0540f42604e2759b2ddef497861e928248`; MLX/Metal 0.32.3.
- Target: `Vontra/Qwen3.8-27B-MLX-4bit`,
  `70ae7fac63274ff2eac54152031433374cb80f2f`.
- Drafter: `z-lab/Qwen3.8-27B-DFlash2`,
  `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`, runtime-packed to 4 bits.
- Patched run used the signed sidecar code and external qualified runtime,
  replacing only `tensorfold_qwen27.py` through an external test loader.
  Neither the signed bundle nor installed production runtime was edited.

TTFT is time to the first nonempty SSE content at the local client. Medians
below use three measured samples per row, with startup and kernel warmup
discarded. The baseline samples come from the immediately preceding same-host
benchmark; a new instrumented baseline independently reproduced the failure
(repeat 7.304 s, follow-up 7.611 s).

| Input | Release median TTFT | Fixed median TTFT |
| --- | ---: | ---: |
| Short, 28 tokens | 0.346 s | 0.335 s |
| New long prefix, 755 tokens | 7.415 s | 7.501 s |
| Identical 755-token input repeated | 7.298 s | **0.213 s** |
| Follow-up, 779 tokens | 7.603 s | **0.352 s** |

Each wave changes the first user message near its beginning, then sends the
long turn, repeats it unchanged, and appends the answer and a short follow-up.
The long reference contains 140 repetitions of `alpha beta gamma delta. `
followed by a request to answer only `4`. Temperature is zero, generation is
capped at 32 tokens, and streaming usage is required. All 12 fixed latency
requests returned `4`, DONE and usage. Cache matches for repeats/follow-ups
explicitly restored 748 tokens; new conversations missed, as expected.

## Correctness and validation

Two additional cold/repeat pairs generated 128 tokens explaining photosynthesis:
greedy sampling and temperature 0.7 with seed 1234. For both pairs, the cached
and uncached requests produced exactly identical native token-ID hashes and
visible-text hashes, with DONE and usage. The test-only generated-token audit
was limited to synthetic loopback traffic and removed afterward.

The [distilled measurement fixture](fixtures/2026-10-14-qwen27-prefix-reuse.json)
contains all latency samples, checkpoint metadata, and correctness hashes;
no prompt texts, credentials or generated token sequences are included.

Focused Qwen backend/HTTP tests: 42 passed. Adjacent TensorFold family,
GLM contract and capture tests: 152 passed. Ruff checks and formatting pass.
The load regression now executes the supplied engine factory and verifies
that tokenizer-derived reply markers reach the plan while disk snapshots
remain disabled. No cancellation propagation change was made.

Reproduction uses the same release's qualified optional runtime and alias:

```sh
rapid-mlx serve qwen3.8-27b-tensorfold --host 127.0.0.1 --port 18761 \
  --max-num-seqs 1 --disable-disk-caches --no-thinking --served-model-name qwen27
```

Submit the synthetic three-message workload described above to
`/v1/chat/completions` with model `qwen27`; repeat unchanged and then append a
follow-up. Use a fresh opening per wave and exclude startup from TTFT.

## Remaining boundary

This fixes repeated/multi-turn prefill. A genuinely uncached approximately
750-token first turn still needs about 7.5 seconds locally. Node restart,
cache eviction or routing to a different node can make a turn cold again.
The result is not a claim that all QuickSilver requests now complete prefill
in less than one second. Pool admission/routing and the packaged TensorFold
runtime still need their own acceptance before community rollout. The change
was not deployed and the patched QuickSilver public path was not tested here.

The test processes, task-owned model copies, external runtime/app bundles and
synthetic audit were removed; the worker was cleaned and its lease released.
