# Qwen4 prompt-lookup qualification

## Outcome

Prompt-lookup drafting (PLD) is qualified for greedy native-MTP requests on
`rapid-mlx/Qwen3.8-Flash-Next-4bit` with a conservative model-specific policy:

- minimum matching suffix: 16 tokens
- maximum matching suffix: 64 tokens
- maximum proposal: 8 tokens
- sampled requests: ordinary MTP (PLD disabled)

The policy is captured with the immutable prompt at request start. Every
proposal is verified by the target model, and only the accepted prefix is
committed to the target and MTP caches. Multi-token verification records each
QSA raw-ring boundary so rollback remains atomic across the model's four-token
index-compression groups. Ordinary K=1 MTP does not pay this snapshot cost.

The PLD adaptation and high-overlap optimization direction were contributed by
Pierre Lamy in PR #2809. This qualification narrows that contribution to the
measured, lossless Qwen4 route.

## Environment

- Mac Studio (Mac15,14), Apple M3 Ultra, 256 GB unified memory
- macOS 26.5.2 (25F84)
- Rapid-MLX base `01739b38b0143d6e70cfe6642a5c9d857c5e98ed`
- artifact revision `dcf657e4acda2aae72da99cde65b6c491cd96998`
- text-only serving, BF16 KV cache, native MTP fixed at K=1
- thinking disabled, temperature 0, request cache cleared before every run

The server command was identical between variants except for the PLD process
override used to measure the off baseline:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3.12 -m vllm_mlx.cli serve MODEL_SNAPSHOT \
  --served-model-name qwen3.8-flash-next-4bit \
  --host 127.0.0.1 --port 8465 --no-thinking --no-mllm \
  --speculative-config '{"method":"mtp","disable_auto_k":true}'
```

For the baseline only, `RAPID_MLX_MTP_PROMPT_LOOKUP=0` was added. The PLD
variant used `min_ngram=16`, `max_ngram=64`, and `max_tokens=8`.

The five-scenario measurement is reproducible with:

```bash
python3.12 bench/bench_qwen4_prompt_lookup.py \
  --label mtp-only-or-pld \
  --runs 3 \
  --output /private/tmp/qwen4-prompt-lookup.json
```

## Decode results

Each value is the median of three cold-request runs. Completion lengths were
804–959 tokens depending on the scenario.

| Scenario | MTP only (tok/s) | MTP + PLD (tok/s) | Speedup | Output parity |
| --- | ---: | ---: | ---: | --- |
| Exact code copy | 46.75 | 98.74 | 2.11x | SHA-256 identical, 3/3 |
| One-line code edit | 41.04 | 94.12 | 2.29x | SHA-256 identical, 3/3 |
| JSON manifest copy | 46.42 | 99.40 | 2.14x | SHA-256 identical, 3/3 |
| Chinese document copy | 46.40 | 92.74 | 2.00x | SHA-256 identical, 3/3 |
| Multi-turn code edit | 45.92 | 93.80 | 2.04x | SHA-256 identical, 3/3 |

The production-default run used no PLD environment variables. It proposed
15,680 tokens in 1,960 eight-token windows and accepted 15,388 (98.1%). MLX
active memory remained approximately 104.7–105.6 GB; the process-reported peak
was 148.1 GB.

## Correctness evidence

- The 45-case Flash-Next battery retained the baseline pass vector: 42/45.
  Long context (8K/32K), JSON schema, tool calls, Chinese, multi-turn, stop
  sequences, and both protocol routes passed.
- The three existing failures were reproduced without PLD: one model math
  answer, one overly narrow code scorer, and a project harness that invokes an
  unavailable `python` executable instead of `python3`.
- The five high-overlap scenarios produced identical output hashes with PLD on
  and off for every measured run (15/15).
- Model-free tests cover full acceptance, partial rejection, target/MTP cache
  alignment, request-scoped prompt history, explicit opt-out, sampled-request
  exclusion, cancellation during target verification, transaction restoration
  across a QSA compression boundary, and zero QSA snapshot overhead for
  ordinary K=1 MTP.

## Why the old defaults were rejected

An 8–10-token lookup suffix is too ambiguous in repetitive code and structured
documents. Token replay found frequent wrong-first-token proposals and the
real model skipped repeated lines under that policy. At 16–64 tokens with an
8-token proposal, exact-copy and JSON traces had no wrong-first-token proposal.
Expected disagreements at an edited code line or changing Chinese sequence
number were rejected by the target and rolled back without changing output.

## Scope boundary

This evidence does not qualify sampled/non-greedy PLD, other model families, or
continuous multi-request speculation. Those routes remain on their existing
decoders until they have independent correctness and performance evidence.
