# DeepSeek V4.1 Flash 40 tok/s investigation

## Status

Paused on 2026-09-12 to prioritize Qwen3.6-35B-A3B. No candidate from this
investigation is release-ready and no performance claim below should be exposed
as product behavior without a fresh quality qualification.

The best reproducible single-domain result is 39.0--39.3 decode tok/s on a code
prompt. The best quality-conservative four-domain result is approximately 28.5
weighted tok/s. A 32.77 weighted tok/s confidence setting was observed, but it
changed the greedy trajectory relative to autoregressive decoding on all four
prompts and is rejected as a product default.

## Environment and artifacts

- Apple M3 Ultra, 256 GB unified memory
- Python 3.12.14, MLX 0.32.2
- Target: `DeepSeek-V4.1-Flash-native-2bit-reap12_5`
- Full draft head: `DeepSeek-V4.1-Flash-DSpark-MLX-4bit`
- Fixed suite: code, reasoning, JSON-only structured output, and Chinese
- 128 generated tokens per prompt, two repeats unless EOS ended earlier
- Peak active memory for the REAP q2 target plus full q4 head: 222.10 GB

Representative command:

```bash
python3.12 scripts/qualify_deepseek_v41_dspark_suite.py \
  --target /path/to/DeepSeek-V4.1-Flash-native-2bit-reap12_5 \
  --overlay /path/to/DeepSeek-V4.1-Flash-DSpark-MLX-4bit \
  --tokens 128 --repeats 2 --verify-k 5 --skip-ar \
  --confidence-threshold 0.35
```

## Results retained

Lazy evaluation made the full 8.01 GB q4 DSpark head fit beside the target,
invalidating the earlier capacity-only rejection. Max-K5 with a 0.35 confidence
floor measured:

```text
code:        39.0--39.3 tok/s, 3.414 accepted/block
reasoning:   approximately 22.7 tok/s
structured: approximately 32.1 tok/s
Chinese:    approximately 18.4 tok/s, early EOS
weighted:   approximately 28.5 tok/s
peak:       222.10 GB
```

Keeping the five-position Markov recurrence on device and converting to a host
list once, rather than calling `.item()` at every position, preserved proposals
and improved the four-domain weighted result by about 0.7%.

A confidence sweep produced the following execution-shape results:

```text
threshold 0.20: 23.12 weighted tok/s
threshold 0.25: 32.77 weighted tok/s
threshold 0.30: 30.47 weighted tok/s
threshold 0.35: 28.56 weighted tok/s
threshold 0.40: 30.41 weighted tok/s
threshold 0.45: 26.99 weighted tok/s
threshold 0.50: 30.09 weighted tok/s
```

The apparent 0.25 winner failed a same-process AR comparison: 0/4 prompts had
an identical token trajectory and the earliest divergence was token 4. The
2-bit target is numerically sensitive to verification batch shape, and several
outputs were repetitive. Do not quote 32.77 tok/s as quality-safe throughput.

## Rejected paths

- Fixed K5: 24.68 weighted tok/s, slower than K4 globally.
- Raising only target layers 37--39 to q4: 31.94--32.14 code tok/s and 2.688
  accepted/block, both worse than the q2 target baseline.
- Unpruned 384-expert q2 target with Engram SSD offload: 32.1--34.4 warm code
  tok/s and at most 2.765 accepted/block.
- Exact grouped target execution: less than 1% end-to-end improvement.
- Gate/up fusion: exceeded safe 256 GB residency and was not faster.
- SDPA substitution: regressed attention as context grew.
- Compiling the complete Markov recurrence: no steady draft-time reduction.
- Temperature and draft-temperature sweeps: no useful acceptance frontier.
- Vocabulary calibration, hidden LoRA, and logit LoRA: offline top-k gains did
  not translate to on-policy block survival; the full-q4 logit LoRA left
  acceptance unchanged at 3.414 and reduced throughput to 38.98 tok/s.
- REAP tap-layer q4 overlay and the unpruned q2 target should not be used as
  speed candidates.

## Storage state at pause

The default Hugging Face cache was expanded to a 650 GB quota volume. A proposed
REAP50 mixed 4/8-bit target requires about 255 GiB of text weights, but its
download failed when another concurrent GLM-5.3 cache grew to approximately
184 GiB. At pause, HFCache had about 43 GiB free and the warm tier about 77 GiB
free. No usable REAP50 checkpoint was retained by the failed download.

The complete 223 GiB local cache entry is a 2-bit MLX model; it cannot serve as
a source for a genuine 3-bit requantization. The official high-precision source
is approximately 475 GiB and only three source shards were retained on the warm
tier from the tap-layer experiment.

## Resume points

1. Train the q4 DSpark head itself on target on-policy trajectories with
   consecutive four-token block survival as the selection metric. Do not use
   held-out per-token top-k accuracy as the primary gate.
2. If at least 255 GiB becomes available in the default HF cache, qualify the
   REAP50 mixed 4/8-bit target directly with Engram SSD offload before attempting
   a new q3 conversion. Require multi-domain quality, repeat stability, peak
   memory, and AR/speculative trajectory analysis.

Do not reopen the rejected microkernel or runtime-adapter paths without new
evidence that changes their measured bottleneck.
