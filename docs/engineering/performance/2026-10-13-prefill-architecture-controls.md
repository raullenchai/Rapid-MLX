# Gemma 4 and Nemotron prefill controls (2026-10-13)

Owner: Vector. This checks two additional architectures requested after the
cross-hardware Qwen measurements. PR #4584 remains a draft. The tested production
source is unchanged: baseline `bec47abba980d48432c8b5917ca682dd0a5758de`, candidate
`32fdd92c880a8fc1341e37e604c8a568ef558ebb`.

## Result and applicability

Medians in seconds; changes are descriptive observations, not attributed benefits.

| Model / metric | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| Gemma 4 26B-A4B, overlapping short TTFT | 28.973043 | 28.986537 | +0.047% |
| Gemma 4 26B-A4B, short singleton TTFT | 0.545255 | 0.545591 | +0.062% |
| Gemma 4 26B-A4B, overlapping long total | 28.378953 | 28.383166 | +0.015% |
| Gemma 4 26B-A4B, long singleton total | 28.401822 | 28.419463 | +0.062% |
| Nemotron 3.5 Lightning 30B-A3B, overlapping short TTFT | 21.062956 | 21.005809 | -0.271% |
| Nemotron 3.5 Lightning 30B-A3B, short singleton TTFT | 0.566758 | 0.566763 | +0.001% |
| Nemotron 3.5 Lightning 30B-A3B, overlapping long total | 38.195506 | 38.122794 | -0.190% |
| Nemotron 3.5 Lightning 30B-A3B, long singleton total | 25.542859 | 25.509301 | -0.131% |

These controls do not establish a speedup for either architecture. The
cooperative-prefill setting is enabled only after the existing qualified Qwen
shared-native bridge starts. Gemma's `gemma4` and Nemotron's `nemotron_h` model
types do not pass that gate. In this campaign, Gemma requests use the unshared
MLLM scheduler with cooperative prefill disabled; Nemotron requests use the
native scheduler. The native scheduler source hash is identical across both
arms. No activation or routing override was introduced for the experiment.

The earlier improvement therefore generalizes across tested hardware and dense
Qwen sizes, but it is not an architecture-wide optimization. Gemma would require
separate analysis of its serving/scheduler/cache contracts. Nemotron's observed
native-path contention requires its own diagnosis. Enabling a Qwen-specific
bridge for either model without that work is not supported by these results.

## Frozen experiment and correctness

M4 Pro, 48 GiB, macOS 26.5.1; Python 3.12.13; MLX 0.32.3,
mlx-lm 0.31.3, mlx-vlm 0.7.2, transformers 5.15.1, NumPy 2.4.4,
httpx 0.28.1. The worker was leased for the task; process and memory/swap samples
were retained and inspected. Three preexisting Python processes were present,
with up to 19% sampled CPU and about 508 MiB maximum RSS per process. Sampled
swap use grew from 449.50M to 2612.88M. This is not certified quiet-host latency
qualification or evidence of zero memory pressure; micro-differences are not
attributed to the PR. Existing services were left untouched. Model copies were task-local transfers from existing Studio
snapshots, with no external model downloads. Processes, models and scratch were
cleaned before the lease was released; existing caches/environments were kept.

Both arms run the same frozen default-route HTTP harness, SHA-256
`7ff73a02c9c3b790a70d127cbbf996dfa8d24592e5b6b6d3be155313392d34e7`.
Each cell alternates baseline/candidate then candidate/baseline across four
fresh children. Each child warms up and runs a cold long singleton plus two
repetitions of cold short singleton and overlapping long/short requests.
Configured prefill is 2048, prefill batch limit 8; speculation, disk caches and
downloads are disabled, temperature zero, thinking disabled and maximum output
32 tokens. These are explicit controls, not a claim that every launch parameter
is a default. Gemma's existing MLLM budget resolves to 8192 while its text forward
cap remains 2048. Nemotron uses the native 2048 setting.

Long raw prompt budget is 16,384 tokens and short is 256, with a four-digit
retrieval code inserted before the question. Actual templated lengths are
recorded in usage and vary by tokenizer. Require exact `7319`/`9283` answers,
stop/[DONE], positive correct usage, cold caches and short arrival before long
first content. No answer-format gate was relaxed. Each cell has eight overlap
waves, 28 distinct measured streams and four warmups; the long singleton receipt
is reused by two row references and is not two independent measurements.

All 16 overlap waves, 56 distinct measured streams and eight warmups pass the
strict HTTP checks. Every Gemma child records 38 MLLM events, zero native events,
and no cooperative or suspended continuation. Every Nemotron child records 82
native events, zero MLLM events and no cooperative or suspended continuation.
Actual long/short prompt usage is 16,423/295 tokens for Gemma and 16,425/297 for
Nemotron. Source hashes bind each child to the stated clean baseline/candidate;
all seven transferred weight-shard hashes match the cached upstream LFS digests.
The validator independently recomputes raw receipt medians and checks the
archive's inner hashes.

No suspended-continuation cancellation or cache-reuse lifecycle pass is claimed
for these architectures by this campaign. Cold-cache streaming correctness,
worker idleness and clean server shutdown are checked. The Qwen-specific
lifecycle results in the earlier report cannot be transferred to other models.

## Models and evidence

- `mlx-community/gemma-4-26b-a4b-it-4bit`, snapshot
  `0d77464eeb233a2da68ebf9d7dc4edaac7db956d`.
- `mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit`, snapshot
  `55ac8c89261109b36c04371cd3f479a4594208c8`.

The standard Nemotron MLX checkpoint is tested here. The separate TensorFold/MTP
checkpoint and runtime are not covered. Gemma's other sizes/quantizations,
images/audio/video, speculation, other hardware and load shapes remain untested.
The small controlled samples do not establish statistical equivalence or a
population latency guarantee; trace/source evidence establishes that the
acceleration path is disabled.

Evidence: [architecture controls archive](fixtures/default-prefill-yield/2026-10-13-architecture-controls.tar.gz)
and [outer manifest](fixtures/default-prefill-yield/2026-10-13-architecture-controls.manifest.json).
The archive has 28 members, 73,854 bytes and SHA-256
`e839fa6e696e9fab6d986b354bb6d08952088c34fb938e97f096d6c745041f8f`.
It includes the frozen harness, validation script, HTTP results, server logs,
scheduler traces, sanitized process/memory samples and model/source manifests.
It contains no weights or credentials. Generated home paths are normalized;
process samples retain time, CPU, RSS and command basenames. Source bytes remain
unchanged. The validator runs without model loading from a repository containing
both source commits:

```sh
mkdir -p /private/tmp/rapid-mlx-family-validation
tar -xzf docs/engineering/performance/fixtures/default-prefill-yield/2026-10-13-architecture-controls.tar.gz -C /private/tmp/rapid-mlx-family-validation
python3 /private/tmp/rapid-mlx-family-validation/validate_family_controls.py /private/tmp/rapid-mlx-family-validation
```

To repeat inference on a leased compatible worker, install the pinned versions
above, prepare clean checkouts of both commits, and reuse the archived
`benchmark_default_prefill.py` with `--help`. Run each named snapshot separately
with `--arms baseline2048 candidate2048 --repeat 2 --cycles 2 --workloads retrieval`,
the same explicit generation controls and a task-local `--output` JSON path.
The frozen harness expects the baseline checkout at
`/private/tmp/rapid-mlx-default-prefill/base` and candidate at
`/private/tmp/rapid-mlx-default-prefill/worktree`; place the harness and
`bench_metadata.py` together and run with the candidate checkout as the working
directory. Supply `--model` with an existing local snapshot whose parent retains
the checkpoint name and `--port` with a free local port. Preserve source/model manifests and host
samples; require all correctness checks before reading latency. The results
record the complete argument namespace. Repeating the experiment requires
existing model snapshots and a compatible Metal host.

The [cross-hardware Qwen report](2026-10-13-prefill-hardware-model-matrix.md)
contains the qualified dense-path gains and remaining default-sized gaps.

The inference implementation is unchanged from the earlier regression suite and
independent code review. This extension changes measurement documentation only.
At this campaign checkpoint, the failed M4/2048 benefit gate and untested
27B/default-sized cell kept the PR draft. The subsequent
[27B/default-sized dogfood report](2026-10-13-qwen27-default-prefill-dogfood.md)
closes that missing cell and records the narrower integration benefit claim;
the original M4 failure remains a failure. No software
merge or release action is included.
