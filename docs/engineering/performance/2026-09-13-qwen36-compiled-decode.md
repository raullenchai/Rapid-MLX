# Qwen3.6-35B-A3B compiled decode replay qualification

Date: 2026-09-13

## Decision

Enable request-private whole-step compiled replay for the exact Qwen3.6
35B-A3B ordinary text architecture on mlx-lm 0.31.3. The optimization applies
automatically to batch-one decode with BF16 KV cache. A second concurrent
request converts the fixed-shape cache back to the stock continuous-batching
representation before joining. MTP, other speculative methods, live KV
quantization, shared multimodal wrappers, unqualified dependency versions, and
unknown model geometries remain on their existing paths.

The fixed KV slabs are bounded at 16,384 tokens and use the qualified capacity
ladder `1023, 1024, 2048, 4096, 8192, 16384`. The adjacent 1023/1024 buckets
preserve the attention-kernel boundary used by the growing stock cache. An
initial position with fewer than 32 tokens left in its bucket declines replay
for that request, avoiding repeated trace cost on short completions near a
growth boundary. Set `RAPID_MLX_COMPILED_DECODE=0` before process start to opt
out for diagnosis.

## Environment

- Mac Studio, Apple M3 Ultra, 256 GB unified memory
- macOS 26.5.2 (25F84)
- Python 3.12.13
- MLX 0.32.2; mlx-lm 0.31.3
- `mlx-community/Qwen3.6-35B-A3B-4bit@38740b847e4cb78f352aba30aa41c76e08e6eb46`
- Existing default Hugging Face cache; no model download or cache override
- Greedy decoding, thinking disabled, 128 output tokens
- Existing Rapid gate/up, router, GDN input, GDN decode, and eager-layer
  optimizations enabled in both arms

## Paired performance and exactness

The model was loaded once. Both modes were warmed, then each of coding,
arithmetic reasoning, creative writing, strict JSON, and tool-argument prompts
ran two alternating pairs. All 10 pairs produced identical token SHA-256
digests.

| Metric | Stock graph construction | Compiled replay |
| --- | ---: | ---: |
| Median decode | 124.55 tok/s | 143.38 tok/s |
| Median paired speedup | 1.00x | **1.151x (+15.1%)** |
| Positive pairs |  | 10 / 10 |
| Exact token pairs |  | 10 / 10 |

Every compiled run recorded one trace, 128 submissions, 128 completion-backed
receipts, zero pending calls, and no poison event. Peak MLX memory for the full
campaign was 18.76 GiB.

A separate real HTTP server pair used the same 256-token coding request,
`--no-spec-decode --no-mllm --disable-prefix-cache`, and a fresh process per
arm. The response body was identical and contained the requested typed
implementation and tests:

| Server mode | Decode throughput |
| --- | ---: |
| Replay disabled | 101.7 tok/s |
| Replay enabled | 121.6 tok/s |

This is a **19.6%** end-to-end decode-throughput improvement on that request.

## Lifecycle gates

- Started one compiled request, admitted a second request while the first was
  active, completed both after stock batch promotion, then completed a fresh
  singleton recovery request.
- Closed a 4,096-token streaming response after its first chunks. The deferred
  abort removed the request, and the next request returned exactly `RECOVERED`
  in 0.14 seconds.
- Unit contracts exercise one-trace replay, explicit cache-state threading,
  completion receipts, transactional conversion, B1-to-B2 promotion, later B1
  reattachment, and one-attempt-only decline behavior.

## Rejected and bounded experiments

- Plain runtime-compiled sigmoid changes BF16/FP32 bytes because its fused
  exponential differs from the eager Metal primitive. It caused a real token
  divergence and was rejected. A custom precise sigmoid-plus-multiply
  primitive restores eager bytes while remaining opaque to the outer compile.
- A 1,018-token prompt followed by a 24-token completion crossed the 1023 and
  1024 cache boundaries. It remained token-exact but paid three traces and ran
  2.7% slower. Initial attachment now declines inside the final 32 positions of
  a bucket, once per request.
- Shared multimodal wrappers retain lane-local MRoPE values in Python. Replaying
  a traced wrapper would skip that state exchange, so those wrappers fail
  closed rather than risking cross-request state contamination.
- Compiled replay and MTP are not combined in this change. The existing MTP
  default and UI controls are unchanged; comparative routing policy is a
  separate decision so this PR does not widen from an execution primitive into
  product-policy changes.

## Reproduction

Use the already-cached immutable snapshot path; the harness refuses a remote
model identifier:

```bash
PYTHONPATH=. python scripts/benchmark_qwen36_compiled_decode.py \
  --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46 \
  --rounds 2 --max-tokens 128
```

The process exits nonzero on any token mismatch or if more than one of the ten
pairs fails to improve.
