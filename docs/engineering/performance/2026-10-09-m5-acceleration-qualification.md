# Existing acceleration paths on M5 Max

Date: 2026-10-09. This is a qualification campaign, not a change to runtime
defaults. Engine source is fixed at
`5c843788007f91431321fb6953688a2fe44ca10b`; the new harnesses accompany this
report. All model runs were serial on the M5.

## Environment and artifacts

- Mac Studio, Apple M5 Max, 36 GB unified memory, macOS 27.0.1 (26A434).
- Python 3.12.13; MLX/Metal 0.32.3; mlx-lm 0.31.3; mlx-vlm 0.7.2;
  Transformers 5.15.1; NumPy 2.5.3.
- Target: `mlx-community/Qwen3.6-35B-A3B-4bit` at
  `38740b847e4cb78f352aba30aa41c76e08e6eb46`.
- MTP head: `mlx-community/Qwen3.6-35B-A3B-MTP-4bit` at
  `0295b81421bf4d0fccca9a7c0fcfb1418dda3516`.
- Both immutable snapshots were copied from an existing cache into the test
  machine's default HF cache (20.9 GB combined). No alternate model cache,
  public benchmark upload, telemetry, or production service was used.
- Scratch and child servers were isolated; HTTP servers bound loopback only.
  Existing machine environments and product flags were left unchanged.

Raw observations are in [fixtures/m5-2026-10-09](fixtures/m5-2026-10-09/).
Only filesystem paths were normalized in these JSON artifacts; measurements,
hashes and receipts are retained. The fixture provenance records the exact
runtime versions and Metal device properties.

This model was chosen because compiled replay explicitly qualifies its
architecture. These results do not qualify every dense model, every MTP
backend, vision workloads, sampled decoding, concurrency, or longer contexts.

## Compiled replay: exact, with no material throughput gain

The existing compiled-decode harness alternates modes across rounds over
coding, reasoning, prose, JSON and tool-argument prompts. Both modes use the
same gate/up, router, GDN input, fused GDN decode and attention-precision
patches. It compares fixed-length greedy token trajectories; it is not an HTTP
or EOS-termination benchmark.

| Output steps | Pairs | Stock median | Replay median | Median paired speedup | Exact pairs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 15 | 148.05 tok/s | 147.26 tok/s | 0.9947x | 15/15 |
| 512 | 10 | 147.08 tok/s | 147.36 tok/s | 1.0018x | 10/10 |

All receipts have one trace, the requested number of submissions and
completions, zero pending calls and no poison event. Peak active MLX memory
was 18.84 GiB in the 128-step campaign and 18.76 GiB in the 512-step campaign.
The short campaign failed the existing performance gate; the longer campaign
passed its positive-pair gate but improved only 0.18%. Neither supports a
material speed claim on this machine.

Exactness here is **incremental replay versus the same fused eager model**.
It does not establish equivalence between that fused model and an unfused
baseline; the separate GDN test below failed that stronger condition.

An exploratory M3 throughput control was interrupted and excluded after a
separate inference service appeared during measurement. Do not use its
partial samples for a hardware speed comparison.

## Fused GDN decode: speed benefit, exactness gate failed

The existing real-model harness warmed both modes, alternated six 128-token
coding pairs, and compared five task trajectories at up to 256 tokens.
Thirty GDN layers enrolled; gate/up, input-projection fusion and MoE routing
were held constant.

- Median paired decode speedup: **1.0469x**, positive in 6/6 pairs.
- Stock/fused throughput coefficients of variation: 0.00674/0.00176.
- Exact quality trajectories: **0/5**. The harness exited nonzero.

A separate stock-driven real-weight diagnostic compared three decode calls
on every enrolled layer, using copies of the same incoming cache for fused
and stock executions. Of 90 comparisons, three recurrent-state results and
one layer output differed; all convolution states matched. The layer-output
maximum absolute difference was `0.0001220703125`; observed recurrent-state
differences in this diagnostic were about `5.34e-6` to `1.13e-5`.

The same source, model revision, MLX 0.32.3, mlx-lm 0.31.3, Transformers
5.15.1 and NumPy 2.5.3 also reproduced differences on an M3 Ultra: one layer
output and six recurrent-state comparisons differed. This is not evidence of
an M5-only defect. It establishes that the small synthetic installation probe
does not cover all real-weight states in the measured runtime combination.
It does not establish which release introduced the difference or that the
resulting text is unusable. The exact-output claim remains unqualified.

## Blocked GDN prefill and its combination with fused decode

The new prefill harness keeps one model resident, creates a fresh prompt
cache per generation, warms every arm for every length and reverses arm order
on alternating rounds. Three rounds use 128 output tokens and 2048-token
prefill chunks. The blocked-kernel calls are counted, preventing a silent
stock fallback from being reported as a candidate measurement.

| Prompt tokens | Stock prompt throughput | Blocked prompt throughput | Paired speedup |
| ---: | ---: | ---: | ---: |
| 512 | 3062.8 tok/s | 3199.2 tok/s | 1.0425x |
| 2048 | 4232.2 tok/s | 4577.7 tok/s | 1.0799x |
| 8192 | 4268.3 tok/s | 4627.8 tok/s | 1.0848x |

Blocked prefill alone left decode throughput essentially unchanged. Adding
fused GDN decode improved decode by about 4.3–4.7% relative to the common
stock arm, with similar prefill throughput. Peak active memory across this
matrix was 20.16 GiB.

The generated token digests differed from stock in all nine blocked-prefill
pairs and all nine combined pairs, although each arm repeated its own digest
across rounds. Blocked prefill's existing numerical contract permits small
FP32 accumulation/rounding differences; its focused numerical tests passed.
That is different from a bitwise or exact-generated-token contract. These
results support a measured prefill benefit under that numerical contract,
not a lossless whole-generation claim. The combined arm also inherits the
separate fused-decode exactness failure.

## Hybrid prefix checkpoints: beneficial incremental control

The HTTP matrix tested checkpoints off/on with stock/blocked prefill. Fused
GDN decode, replay, host prompt caching and speculation were disabled to
isolate these effects. Each of three deterministic approximately 6.5K-token
documents was edited near its end, middle and start. Before every sequence,
the reusable cache was explicitly cleared; the original document seeded the
cache, the edited document ran warm, and the edited document ran again after
another clear as a cold reference. Each response used greedy decoding and a
32-token cap.

| Prefill | Edit | Checkpoints off: warm TTFT | On: warm TTFT | Median paired gain | Resumed tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| Stock | Late | 1.622 s | 0.714 s | 2.274x | 4096 |
| Stock | Middle | 1.613 s | 1.165 s | 1.384x | 2048 |
| Blocked | Late | 1.510 s | 0.676 s | 2.235x | 4096 |
| Blocked | Middle | 1.501 s | 1.090 s | 1.375x | 2048 |

Head edits reported zero cached tokens and approximately 1x gain. Server
logs independently confirmed checkpoint snaps to 2048/4096; benefits were not
inferred only from latency.

All **18 checkpoint-on/off output comparisons** matched for the same prefill
mode, document and request history. This supports incremental checkpoint
compatibility in the measured matrix. HTTP exposes decoded content/reasoning,
so these are byte-output comparisons, not direct token-ID comparisons.
The joined pairs are retained in `checkpoint-incremental-comparisons.json`;
each is derived from the matching warm rows in `checkpoint-matrix.json`.

The stronger warm-versus-cleared-cold gate passed only **26/36** comparisons.
Differences also occurred with checkpoints disabled and zero cached tokens;
therefore they cannot be attributed to the added checkpoint feature from
this experiment. The harness correctly exited nonzero. Full cold/warm
equivalence remains unqualified and needs a separate control investigation.
The four server arms ran serially, not in randomized cross-process order.

### Follow-up: shared-prefix segmentation control

A checkpoint-off, stock-prefill diagnostic repeated the first late-edit case,
adding a second cleared-cold request. All edited requests reported zero cached
tokens. The two cold outputs matched, while the warm output differed. Server
logs showed that the warm request additionally snapshotted a shared prefix at
5568 tokens; the cold requests did not. A zero-token cache hit therefore does
not imply identical prefill segmentation.

In a separate child process, disabling only `Scheduler._shared_prefix_local_split`
with a diagnostic monkeypatch made warm, cold and repeated-cold outputs match
the original cold digest. Their TTFTs were 1.603, 1.606 and 1.606 seconds.
Receipts are in `checkpoint-baseline-control.json` and
`checkpoint-no-shared-split-control.json`. Each executed one late-edit row;
the configured round count is not the number of completed repetitions.

This supports shared-prefix prefill segmentation as an explanation for this
particular mismatch, rather than checkpoint restoration. It does not qualify
the other nine mismatches, establish a numerical root cause, or justify
disabling shared-prefix snapshots in production. The diagnostic launcher also
imports the scheduler before entering the CLI; an import-order control and a
broader matrix remain necessary before proposing a runtime change.

## MTP

The existing continuous-MTP harness completed 15 pairs over five prompts,
three runs and 192-token caps. The model and sidecar used the immutable
revisions above; measured acceptance was 65.33%.

- Pooled ordinary/MTP decode: **135.00/183.33 tok/s**, or **1.358x**.
- Exact token pairs: **0/15**; the required lossless gate exited with code 2.
- A token audit of the first prompt found the first difference at zero-based
  token index **177**. An ordinary → MTP → ordinary control reproduced the
  ordinary tokens exactly, excluding simple persistent contamination as the
  cause of that particular mismatch.

This harness reloads the model and always runs ordinary before MTP within a
pair; it does not counterbalance mode order. The first run includes cold
kernel effects. Treat the speed figure as exploratory, not approval to enable
MTP or claim lossless equivalence on the measured configuration.

The current vendored native-MTP loader failed before generation in a fresh
process: `mlx_vlm.models.qwen3_5_mtp` lacked `TextConfig`. The architecture
binding creates a shim with `Model`/`ModelConfig`, but the configuration loader
also needs nested configuration exports. This is a loading/integration
blocker, not a measured GPU performance failure.

The optional `--preload-canonical` benchmark mode primes that shim with the
upstream drafter package's exports before installing the vendored classes.
It is diagnostic only and does not change the normal product loader. Its
separate resident-model campaign warmed both modes, alternated pair order
over three rounds and five prompts, and used a 192-token cap with normal EOS
termination. It produced **15/15 exact token pairs**, a **1.4008x median
paired decode speedup**, and 13/15 positive pairs. Peak active MLX memory was
19.76 GiB. Short EOS-terminated outputs are included, so this is a workload
median rather than a fixed-length throughput claim. The raw observations are
in `native-mtp-preloaded.json`. Normal-start qualification remains blocked;
the passing diagnostic makes the loader repair the first follow-up priority.

## Validation and follow-up ownership

The focused suite passed **130 tests** for fused GDN, prefill, hybrid
checkpoints and compiled replay. Six additional benchmark receipt tests
passed; they prevent duplicated terminal streaming tokens or incomplete
receipts from corrupting native-MTP token comparisons. Passing unit/numerical
tests does not supersede the failed real-model exactness gates above.

Measured combinations are blocked prefill + fused GDN decode, blocked
prefill + hybrid checkpoints, and replay + its required precision/fusion
stack. Replay + MTP is outside the existing qualification policy and was not
forced through it. No sampled, concurrent or vision combinations were tested.

Vector owns the next actions:

1. Repair the fresh-process native-drafter configuration export contract and
   add a real loader-boundary regression gate before rerunning native MTP.
2. Strengthen fused GDN's exactness probe with real-weight recurrent states;
   investigate the observed state/output mismatch before certifying an exact
   contract for this runtime combination.
3. Trace the continuous-MTP mismatch around token 177, then rerun strict
   token checks over the full matrix with a warmed, counterbalanced harness.
4. Investigate warm/cold output differences with checkpoints off as the
   negative control; preserve the passing incremental checkpoint evidence.

Atlas owns any change to hardware routing or default-policy claims. This
campaign changes neither. There is no deployment or release action.

## Reproduction

Use the matching dependency versions, an isolated checkout and existing local
snapshots. All commands run from the checkout root. The following shell
variables identify existing cache paths; they do not relocate the cache.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RAPID_MLX_TELEMETRY=0
export PYTHONPATH=.
target="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46"
head="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-MTP-4bit/snapshots/0295b81421bf4d0fccca9a7c0fcfb1418dda3516"
out=/private/tmp/Pierre-m5-qualification
mkdir -p "$out"

python scripts/benchmark_qwen36_compiled_decode.py --model "$target" --rounds 3 --max-tokens 128
python scripts/benchmark_qwen36_compiled_decode.py --model "$target" --rounds 2 --max-tokens 512
python scripts/benchmark_qwen35_fused_gdn_decode.py --model "$target" --pairs 6 --max-tokens 128 --quality-max-tokens 256
python scripts/benchmark_m5_prefill_matrix.py --model "$target" --output "$out/prefill-matrix.json"
python scripts/benchmark_m5_checkpoint_matrix.py --model "$target" --output "$out/checkpoint-matrix.json"
python bench/bench_spec_decode_mtp.py --model "$target" --mtp-sidecar "$head" --runs 3 --prompts 5 --max-tokens 192 --require-lossless
python scripts/benchmark_m5_native_mtp.py --target "$target" --drafter "$head" --output "$out/native-mtp.json"
python scripts/benchmark_m5_native_mtp.py --preload-canonical --target "$target" --drafter "$head" --output "$out/native-mtp-preloaded.json"
python scripts/diagnose_m5_fused_gdn.py --model "$target" --output "$out/gdn-diagnostic.json"
python scripts/diagnose_m5_continuous_mtp.py --model "$target" --drafter "$head" --output "$out/continuous-mtp-diagnostic.json"
```

Run commands serially; several are expected to exit nonzero while the
reported gates remain unresolved. Scratch may be removed by normal retention
policy. The report, harnesses and normalized raw observations are durable in
Git; the cached snapshots remain available for follow-up runs.
