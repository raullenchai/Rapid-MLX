# Qwen3.5-family continuous MTP qualification

Date: 2026-08-31
Host: Mac Studio, M3 Ultra, 256 GB unified memory
Code base: stacked on continuous-MTP foundation PR #2842 and Qwen dense
adapter PR #2854

## Decision

Continuous MTP is qualified per concrete target artifact. Sharing the
`qwen3_5` model type, MTP tensor ABI, and cache layout is necessary but not
sufficient evidence that batched target verification preserves task results.
The catalog therefore records `verified`, `blocked`, or `unknown`; ordinary
MTP remains available for every existing alias regardless of this tier.

When no speculative configuration is supplied, a `verified` artifact now
selects its declared MTP preset and the continuous scheduler automatically.
`--no-spec-decode` remains the user-facing opt-out for MTP, while an explicit
`"continuous_batching": false` keeps MTP enabled on its ordinary scheduler.
`blocked` and `unknown` artifacts remain unchanged unless an operator
explicitly forces an experiment.

An unverified artifact fails closed when a user explicitly requests
`continuous_batching=true`. `--force-spec-decode` remains an explicit operator
override for controlled experiments.

Continuous MTP also requires an unquantized BF16 KV cache for transactional
trim/restore. A verified alias's automatic cache-compression default yields to
that method requirement. Explicit `--kv-cache-turboquant`, legacy cache
quantization, or `--kv-cache-dtype int4|int8` requests fail before model load
with an actionable error instead of silently falling back to ordinary MTP.

## Design precedent

Production speculative schedulers integrate draft and target verification into
the request scheduler rather than maintaining a separate single-request mode.
They still gate models on explicit implementations, checkpoint metadata, draft
depth, quantization, cache ownership, and verifier compatibility. Rapid-MLX
retains those runtime gates. The additional catalog tier records the
artifact-level evidence that structure cannot prove: stable task outcomes under
mixed batching and a measured concurrent throughput win.

## Methodology

- one model resident; the task-owned server was stopped between conditions
- prefix cache disabled
- performance requests used temperature 0 with thinking disabled
- performance: four deterministic lane-specific 603-token prompts, 128
  completion tokens, four simultaneous requests, three cohorts per condition
- correctness: 48 task-distinct requests in 12 mixed four-request cohorts
- correctness categories: coding (including generated-project execution),
  JSON Schema, forced/automatic tool calls with argument validation, English
  and Chinese reasoning, creative writing, open chat, multi-turn state,
  stop sequences, and 2K/8K/32K context recall
- both OpenAI-compatible and Anthropic-compatible routes
- temperature 0; thinking used the user-facing default budget where supported
- qualification requires zero ordinary-pass/continuous-fail task regressions,
  no cross-lane state contamination, coherent inspection of every changed
  output, and a positive measured throughput change

The checked-in client is `bench/bench_continuous_mtp_server.py`. Example:

```bash
python3.12 bench/bench_continuous_mtp_server.py \
  --label continuous \
  --model "$TARGET_MODEL" \
  --base-url http://127.0.0.1:8475/v1 \
  --runs 3 --concurrency 4 --max-tokens 128 \
  --baseline-json legacy.json
```

The two server conditions differed only in the continuous scheduler fields:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3.12 -m vllm_mlx.cli serve \
  "$TARGET_MODEL" --host 127.0.0.1 --port 8475 \
  --max-num-seqs 4 --max-concurrent-requests 4 \
  --disable-prefix-cache --no-thinking --force-spec-decode \
  --speculative-config \
  "{\"method\":\"mtp\",\"model\":\"$MTP_MODEL\",\"num_speculative_tokens\":2,\"disable_auto_k\":true,\"continuous_batching\":false,\"allow_dynamic_membership\":false}"
```

For the continuous condition, set `continuous_batching` to `true`.
`allow_dynamic_membership` remained `false` so both conditions used fixed
cohorts.

## Results and disposition

| Target artifact | Target revision | MTP revision | Ordinary MTP aggregate | Continuous MTP aggregate | Change | Identical text | Task regressions | Tier |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3.5-4B MLX 4-bit | `32f3e8ec` | `ab6f59bc` | 106.49 tok/s | 139.30 tok/s | +30.8% | 44/48 | 0 | verified |
| Qwen3.5-9B MLX 4-bit | `8b2b98c0` | `222dfd2c` | 74.34 tok/s | 92.07 tok/s | +23.9% | 46/48 | 0 | verified |
| Qwen3.6-27B MLX 4-bit | `c000ac2c` | `83795d54` | 28.37 tok/s | 32.37 tok/s | +14.1% | 45/48 | 0 | verified |
| Qwen3.8-27B MLX 4-bit MTP | `aa985c29` | self-contained | 25.82 tok/s | 32.51 tok/s | +25.9% | 46/48 | 0 | verified |

Hash differences were localized rather than accepted blindly. For the three
verified artifacts, every changed response was a coherent alternative that
preserved its task contract. Changes were limited to equivalent code spelling,
creative prose, and open-chat wording. Structured JSON, tool names and
arguments, long-context answers, stop behavior, and generated-project results
did not regress. A 9B creative-writing response initially hit the artificial
384-token battery cap; ordinary and continuous conditions both passed when
rerun with 512 tokens, so the cap was not treated as product evidence.

The changed 4B Chinese reasoning response ended mid-sentence after the
reasoning parser reported an incomplete trace. The ordinary response had the
same incomplete-trace condition under the same generation budget, and both
conditions produced the correct task answer; this is therefore recorded as a
shared reasoning-budget limitation rather than a continuous-scheduler
regression. With zero task regressions across the battery, the artifact is
promoted alongside the other three. Other quantizations and model sizes remain
`unknown` until the same task-level gate is run on their exact artifacts.

Peak active/peak MLX memory observed for Qwen3.6-27B was approximately
17.7/19.6 GB for four continuous lanes and 16.3/17.2 GB for ordinary MTP.

Qualification runs selected BF16, installed the continuous coordinator, and
completed every eligible cohort through the `continuous_planned` route. Normal
The four verified aliases now select MTP and that route by default across CLI,
Server, and Desktop. Unknown aliases retain their prior behavior. Every
task-owned server was stopped after its paired condition.
