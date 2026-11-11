# Reproducible MTP depth costs

Greedy MTP requests previously obtained their round-cost inputs from the first
process-local calibration. The process froze those inputs, and each request
maintained its own acceptance controller. Across process restarts, timing noise
still changed the expected-value comparisons and therefore verify widths.
Width-dependent floating-point rounding then changed some greedy argmax results.

The tuning profile now supplies `mtp_forward_costs_ms`. The installed MTP adapter
qualifies geometry and actual expert quantization using the existing depth-cell
facts. It slices measured costs through the resolved, checkpoint-clamped cap.
The same round-cost and measured copy-verify formulas consume either source.
Missing or insufficient curves use the existing process-local timing fallback.

## Measurement protocol

The source base was `e299a09c53642f23d73cfe04d034be4b53c18bf6`. Patched measurements
use that base with the depth-cost implementation and profile data in this change.
Weights and MTP head remained in the existing Hugging Face cache, with downloads
disabled. The model was `mlx-community/Qwen3.6-35B-A3B-4bit`, primary revision
`38740b847e4cb78f352aba30aa41c76e08e6eb46`, and the matching 4-bit MTP head revision
`0295b81421bf4d0fccca9a7c0fcfb1418dda3516`.

Each arm used five fresh server processes, concurrency one, a warmup request,
and two identical greedy requests with 564 template-expanded prompt tokens and
256 generated tokens. Prefix and optional disk caches were disabled. Plain
arms used `--no-spec-decode`. MTP arms used the cached head through
`--speculative-config`. Startup's internal warmup and the explicit warmup were
excluded from measured request captures. Target widths, scheduled depths and
output-token hashes were captured without printing tokens. Decode throughput
uses streamed output between its first and last content chunks. Tokens per
forward is 255 decode tokens divided by the generator's target-forward count,
excluding the scheduler's first generated token.

GPU campaigns held `/private/tmp/harbor-prvalidate.lock` with blocking acquisition
and verified the held inode against the path. M5 work also held and heartbeated
its MachineControl lease. Only campaign-owned servers were stopped.

For each shipped curve, take the median step milliseconds across five fresh
calibrations and multiply it by each width's median paired width/step ratio.
The draft entry uses the median paired draft/step ratio. Each calibration itself
uses the established two warmup and 21 measured round-robin passes on fresh
scratch caches. The resulting immutable costs preserve the existing normalized
round overhead of 0.25.

## Explicit hardware cells

The M3 Ultra measurement uses 60 GPU cores and 256 GiB, macOS 26.5.2 and MLX
0.32.4. The M5 Max uses 40 GPU cores and 36 GiB, macOS 27.0.1 and MLX 0.32.3.
Both use MLX-LM 0.31.3 and MLX-VLM 0.7.2. Profile selectors pin generation/tier,
exact memory, software, affine expert geometry `(2048, 512, 256, 8, 4, 64)`,
MTP mode, width one and batch one. They explicitly include all core variants,
including unknown core counts, because MLX omits that fact on the measured
runtime. The measured core counts are retained as provenance.

## Results

Both chips completed five fresh-server repetitions per condition.

| Chip | Arm | Median [min, max] tok/s | Output digests | Depth sequences | Tokens/forward |
| --- | --- | --- | --- | --- | --- |
| M3 Ultra | Main MTP | 158.07 [155.67, 160.11] | 5 | 5 | 1.66667, 1.70000, 1.73469, 1.86131, 1.94656 |
| M3 Ultra | Profile-cost MTP | 155.31 [64.38, 159.08] | 1 | 1 | 1.74658 |
| M3 Ultra | Main plain | 177.58 [176.31, 178.31] | 1 | 1 | — |
| M3 Ultra | Patched plain | 177.11 [176.60, 177.42] | 1 | 1 | — |
| M5 Max | Main MTP | 172.56 [172.21, 176.26] | 2 | 4 | 1.71141, 1.82143 |
| M5 Max | Profile-cost MTP | 172.46 [172.40, 172.77] | 1 | 1 | 1.82143 |
| M5 Max | Main plain | 167.83 [167.64, 168.06] | 1 | 1 | — |
| M5 Max | Patched plain | 168.05 [167.76, 168.20] | 1 | 1 | — |

MTP median delta is -1.75% on M3 and
-0.06% on M5. Each fixed condition has one
output-token digest, one chosen-depth sequence and one consumed-width sequence.
Every same-server replay matches. Profile runs performed zero live calibrations.
M3's additional explicit K=3 run also matches its default condition exactly.
M5's full throughput range narrows from 4.05 to 0.37 tok/s.

M3's fixed sample includes 64.38 tok/s, with identical depths and output tokens.
The same process had a slower warmup and a 74.48 tok/s replay. In-request GPU-clock
and OS-scheduling traces were not captured, so its cause is unverified. All five
samples are included. This campaign establishes depth/output repeatability under
the stated cache/concurrency condition. M3 wall-clock spread increased in the
complete sample.

Frozen measured values are as follows.

- M3 Ultra milliseconds are `[6.800500007, 9.010624999, 10.513812041, 12.027729643, 13.952790992, 0.992714634]`.
- M5 Max milliseconds are `[6.717833690, 8.564853409, 10.033190635, 11.409948580, 13.100452610, 1.061060239]`.

Each dense-control condition also has five fresh-server repetitions.

| Chip | Model | Main median [min, max] tok/s | Patched median [min, max] tok/s | Median delta | Tokens/forward |
| --- | --- | --- | --- | --- | --- |
| M3 Ultra | 4B | 156.57 [155.89, 159.65] | 159.89 [158.30, 160.51] | +2.12% | 1.60377 |
| M3 Ultra | 9B | 112.38 [111.83, 114.49] | 118.96 [116.96, 119.90] | +5.86% | 1.61392 |
| M5 Max | 4B | 142.61 [141.83, 142.68] | 143.26 [142.51, 144.14] | +0.46% | 1.55488 |
| M5 Max | 9B | 99.00 [98.22, 99.40] | 99.22 [98.01, 99.67] | +0.22% | 1.63462 |

Both dense controls used live-calibration fallback cells. Their output digests,
chosen depths, consumed widths and tokens/forward match across conditions. These
are phase-separated shared-host regression checks. CPU validation overlapped
parts of the campaign, and admission records retain host load.

Raw private evidence is under `/private/tmp/rapid-mlx-gap-round20/`, including
per-run calibration timings, complete chosen-depth/consumed-width sequences,
output digests, source snapshots, hardware metadata, full samples and logs.
`REPORT.md` records exclusions, the M3 outlier and final validation receipts.

## Interpretation

The shared timing/acceptance controller remains an observer for greedy requests.
The request-local controller starts its acceptance EMA and exploration counters
fresh for every request. Its decisions use the shipped constant costs after
each round. The copy gate is also request-local and consumes the fixed measured
verify curve. The shipped costs remove startup timing from these decisions.

Profile-selected depth still adapts to the request's own acceptance. Fixed-depth
settings retain their existing path. Different cache states, concurrent serving
lanes or different numerical kernels can change outputs and require separate
qualification. The live fallback for unmeasured cells can still vary across
process restarts.
