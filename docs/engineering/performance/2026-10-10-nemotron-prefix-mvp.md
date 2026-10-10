# Nemotron long-chat prefix reuse qualification

Owner: Vector. Date: 2026-10-10. Scope: the experimental
`nemotron-3.5-lightning-tensorfold` provider, greedy text requests, one request
at a time. This qualification adds no serving change. Conversation boundaries
already reach the provider through #4308.

## Contract

The probe supplies a deterministic inventory of 650 service workers as the first
user message, yielding about 19.4k prompt tokens. It exercises the initial turn,
two continuations, an edited last question, regeneration, a branch from an
earlier turn, and a different inventory. The edited and branched requests keep
the earlier messages unchanged. The changed inventory checks that a previous
conversation's state does not alter a new request's answer.

Each cache mode runs in a fresh process with the same qualified checkpoint and
runtime. Cache-off sets the isolated scheduler's checkpoint store to `None`
before any request. It neither reads nor retains prompt checkpoints. This is a
benchmark control, not a new public server flag.

Comparison requires identical request hashes, prompt lengths, generated token
IDs and answers for all seven cases. Every cache-off request must report zero
cached tokens; the cache-on initial request must also be cold. Each of the five
resumed cases must restore at least 95% of its prompt. Each resumed input
prompt must contain at least as many tokens as the initial input prompt.
Checkpoint boundaries can leave a few initial tokens to recompute. A tiny shared system prefix or truncated history
cannot qualify. Invalid counts, empty visible answers (including EOS-only
outputs), incomplete cases, different source fingerprints and different
hardware fail qualification. Checkpoint/runtime identities must match the
qualified profile and the probe fingerprint must match the actual script,
even if both artifacts agree on an incorrect value.

The probe records its own SHA-256 before loading the model and rejects dirty
tracked serving sources. Comparison resolves the recorded commit and requires its `rapid_mlx` Git tree
to match the current clean serving tree; documentation-only commits remain
compatible with retained evidence. The serving commit, runtime versions, checkpoint
revision, CPU and RAM accompany each artifact. The initial MVP probe can be
untracked because its exact bytes are independently fingerprinted.

## Measurements and limits

Hardware: Apple M3 Ultra, 256 GiB, macOS 26.5.2; Python 3.12.14,
MLX 0.32.3, TensorFold 0.6.6 at
`cb2ebf0540f42604e2759b2ddef497861e928248`. Checkpoint:
`TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit` at
`d9d758fb83953437f7263256b0d96157e2a348b8`, reused from the shared local
Hugging Face cache. The host had other workloads and approximately 8.83 GiB
of pre-existing used swap; it remained allocated during the runs.

This is a cache-on versus cache-off comparison of existing serving behavior,
not a before/after optimization claim. Timing is descriptive rather than a
pass/fail speed threshold on a shared host. Provider TTFT measures the first
generated token; it does not include HTTP transport and can precede visible
text when the detokenizer buffers a fragment. Prefill uses the scheduler's
start and prefilled timestamps. Cached usage comes from the provider's terminal
output, not intermediate token events, whose cached count defaults to zero.

The retained [qualification artifact](fixtures/nemotron-prefix-mvp-2026-10-10.json) contains complete generated token sequences and request hashes. All seven pairs matched. The five resumed cases restored 19,365 tokens each (over 99.5% of their prompts).

| Case | Cache-on prefill | Cache-off prefill | Cache-on token TTFT |
| --- | ---: | ---: | ---: |
| initial | 16.695 s | 18.130 s | 16.729 s |
| continue | 0.288 s | 17.214 s | 0.321 s |
| continue_again | 0.403 s | 18.929 s | 0.437 s |
| edit | 0.295 s | 20.259 s | 0.328 s |
| regenerate | 0.290 s | 14.641 s | 0.324 s |
| branch | 0.293 s | 17.023 s | 0.326 s |
| different_inventory | 0.245 s | 0.281 s | 0.245 s |

The qualification does not establish sampled, concurrent, tool-bearing,
cross-restart, ordinary-backend or arbitrary-length behavior. The accelerated
profile continues to reject tools. No native runtime migration or kernel
change is included.

## Reproduce

Use an isolated environment with the qualified runtime and MLX versions above,
plus this repository's dependencies. The complete checkpoint must already be
in the default HF cache. Run from the repository root:

```sh
mkdir -p /private/tmp/tensorfold-absorb-nemotron-mvp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.
python scripts/large-model-run.py --working-set-gb 32 --lock-timeout 60 -- \
  python scripts/benchmark_nemotron_prefix.py --cache on \
  --output /private/tmp/tensorfold-absorb-nemotron-mvp/on.json
python scripts/large-model-run.py --working-set-gb 32 --lock-timeout 60 -- \
  python scripts/benchmark_nemotron_prefix.py --cache off \
  --output /private/tmp/tensorfold-absorb-nemotron-mvp/off.json
python scripts/benchmark_nemotron_prefix.py --compare \
  /private/tmp/tensorfold-absorb-nemotron-mvp/on.json \
  /private/tmp/tensorfold-absorb-nemotron-mvp/off.json
```

The wrapper serializes model loads and checks available RAM under the host lock.
The probe closes its backend on request failures. Supervise model runs with a
bounded execution window (480 seconds per mode was used for this qualification).
If a scheduler stalls, interrupt the owning wrapper, which forwards termination
to its command group, and verify the owned processes exited before another run.
Do not stop unrelated services or remove model caches to make a run fit.

Regression tests exercise mismatched prompts and outputs, inadequate and invalid
cached counts, contaminated cold controls, incomplete responses and provenance
mismatches. The tests also verify that instrumentation restores scheduler
submission after a request failure and reject stale terminal results from an earlier request.
