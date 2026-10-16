# Nemotron 3.5 Super VL text qualification

Owner: Vector. Local qualification only; no upload or release authorization.
Architecture and pending gates: [decision](../decisions/2026-10-15-nemotron-super-vl.md).

## Environment and workload

Studio: M3 Ultra, 28 CPU / 60 GPU cores, 256 GiB unified memory, macOS 26.5.2.
Python 3.12.14, Rapid-MLX 0.16.1 checkout `07764546c`, based on `7118aaf72`;
MLX 0.32.4, mlx-lm 0.31.3, mlx-vlm 0.7.2, Transformers 5.15.1.
The BF16 source revision is `a1ccf94d69377bf82c4db39798aac28447d9c16f`.
All 88 source files pass content-hash verification; all 64 source weight
shards and all 14 converted shards pass safetensors extent checks.

The generator benchmark loads through Rapid's precision-adapted text loader.
It repeats tokenized `The quick brown fox jumps over the lazy dog. ` to exactly
1,024 raw prompt tokens, then greedily decodes 64 tokens, with prefill chunks
of 256. The four-bit repetitions use the resident model after eight short smoke prompts.
The mixed benchmark runs in a fresh process after one short France-capital
warmup, using the same two-repetition 1K/64-token workload.
All inference and GPU tests hold `lockf -k /var/tmp/rapid-mlx-large-model.lock`.
The offline converter uses CPU execution and does not need this GPU lock.
Each repetition creates a fresh inference cache. Artifacts reside on RTL-2T.

## Four-bit result

The conversion takes 321.02 seconds and yields 4.609 effective bits per weight.
The output is `/Volumes/RTL-2T/scratch-archive/rapid-mlx/super-vl-120b/text-4bit-g64`.
Weights total 69,523,298,304 bytes. The complete artifact is 69,540,668,652 bytes;
the download-size generator's file filter counts **69,540,654,086 bytes**.
The latter excludes README and the Jinja chat template. Config and tokenizer
metadata contain no remote-code selectors or image-processor class.

| Repetition | Prompt tokens/s | Decode tokens/s | MLX allocator peak (GB) |
| --- | ---: | ---: | ---: |
| 1 | 466.12 | 48.56 | 72.680 |
| 2 | 467.59 | 48.39 | 72.680 |

Eight greedy smoke prompts cover factual recall, arithmetic, code, translation,
light colors, a sequence, explanation and name recall. Examples: `Paris`, `391`,
`Buenos días.`, primary light colors red/green/blue, and `Your name is Ada.`
Sampling is limited to 16 greedy tokens per prompt (or EOS); this is a smoke
check, not a quality benchmark. Raw EOS tokens in the direct model evaluation
are retained as token-level evidence. HTTP token filtering is checked by the separate serving gates below.

## BF16 comparison and HTTP gates

The evaluation-only paged BF16 reference matches the tiny resident BF16 model
bit-exactly for full and cached logits. The full checkpoint's eight-prompt
reference completes in 640.59 seconds. It uses exact source tensors and native
expert math without materializing the 241 GB text checkpoint on the GPU.
The comparison measures first-token logits and up to 16 independently greedy
tokens per prompt, with thinking disabled.

| Prompt | Same greedy first token | First-token top-10 overlap | Matching greedy prefix |
| --- | --- | ---: | ---: |
| France capital | No | 8/10 | 0 |
| 17 × 23 | Yes | 9/10 | 4 |
| Python addition | No | 9/10 | 0 |
| Spanish greeting | Yes | 8/10 | 5 |
| Primary light colors | Yes | 10/10 | 7 |
| Sequence continuation | Yes | 10/10 | 16 |
| Sky explanation | Yes | 10/10 | 16 |
| Ada recall | Yes | 10/10 | 6 |

Strict greedy first-token agreement is **6/8**; top-10 overlap totals **74/80**
(92.5%). Both first-token mismatches select another token tied at the BF16
maximum, giving **8/8 tie-aware first-token agreement**. Actual greedy IDs are
compared rather than the first entry of an argsorted top-10 list. Five of eight
sampled sequences match exactly. Other differences are wording or formatting:
`Paris` versus `The capital of France is Paris.`, a different code preamble,
and bold emphasis around primary colors. The clipped code sample does not
establish complete-function correctness. This is not independent full-model
NVIDIA-runtime parity or a quality benchmark.

All **seven HTTP requests pass**, with automatic parser selection:

- Multi-turn: `Your name is Ada, and your favorite color is teal.`
- Thinking off: visible answers with no reasoning content or control tokens.
- Thinking on: separate nonempty reasoning and `17 × 23 = **391**` as content.
- Non-stream and stream: `get_weather({"city":"Paris"})`, with `tool_calls`
  finish reasons and valid JSON arguments.
- Both tool-result follow-ups: `The current weather in Paris is sunny with a
  temperature of 18°C.`

No added special tokens, thinking delimiters or XML tool markers leak into
visible content or reasoning. The server shuts down, clears its caches and
releases port 8479. Its SIGTERM thread dump is the normal signal-observability
hook, followed by graceful engine/application shutdown; the harness exits zero.

The final **95 focused tests** and **3,774 catalog/README tests** pass. Ruff and
format checks pass. Changed-line coverage against
`origin/vector/nemotron-35-bolt` is **141/141 lines, 100%**. The branch also
includes the committed optional-top-k validation guard from `d0ee60130`; it does
not change this model's valid top-22 configuration or the measured norm math.

## Mixed low-bit result for the 64 GB budget

The initial session deferred mixed conversion. The continuation converts on CPU
in **267.35 seconds**, without acquiring the GPU lock or downloading weights.
The output is `text-3bit-mixed` under the same artifact root. Weights occupy
**42,746,861,568 bytes**; the generator-filtered catalog footprint is
**42,764,296,516 bytes**. The complete artifact is **42,764,311,538 bytes**
after finalizing the model card. Nine shards pass extent, contiguous-offset, index and weight-total
audits. The model card is excluded from the catalog footprint.

The recipe is affine: **26 interior MoE layers at 2-bit/group-128**, **14 boundary
MoE layers at 3-bit/group-128**, 192 other projections at 4-bit/group-64, and
one embedding at 8-bit/group-64. The output head, routers and latent projections
retain checkpoint precision. Two-bit layer indices are 17, 19, 21, 23, 26, 28,
30, 32, 34, 37, 39, 41, 43, 45, 48, 50, 52, 54, 56, 59, 61, 63, 65, 67, 70
and 72. Seven MoE layers at each model boundary retain three bits. Per-module
quantization entries in `config.json` are authoritative.

Mostly three-bit experts cannot meet this budget: all three-bit experts with
the protected tensors would occupy about 51.925 GB. At least 23 two-bit expert
layers are needed to get below 44 GB. The selected 26 leave additional room for
runtime allocation. The repository's `3bit` suffix describes a mixed artifact;
most expert layers in this budgeted artifact are two-bit.

The same eight BF16 comparison prompts give **6/8 strict first-token agreement**,
**7/8 tie-aware agreement**, **68/80 top-10 overlap**, and **4/8 identical sampled
greedy sequences** (up to 16 tokens). Samples remain coherent: `Paris`, `391`,
`Buenos días.`, primary light colors, and Ada recall. The code answer is clipped
in its preamble; this does not validate a complete function. This limited smoke
comparison does not establish broad accuracy after aggressive quantization.

All **seven HTTP gates pass** with automatic parser selection. The model
recalls `Ada` and `teal`, separates 191 characters of reasoning from
`17 × 23 = **391**`, calls `get_weather({"city":"Paris"})` in both stream modes,
and reports sunny/18°C from both tool-result follow-ups. No control tokens leak.
The serving harness shuts down its own server cleanly and releases port 8480.

| Repetition | Prompt tokens/s | Decode tokens/s | MLX allocator peak (GB) |
| --- | ---: | ---: | ---: |
| 1 | 458.47 | 50.06 | 45.903106266 |
| 2 | 459.28 | 49.93 | 45.903368468 |

The **45.903 GB** peak passes the requested **46 GB** allocator budget at
1,024 prompt tokens plus 64 generated tokens. The artifact targets **64 GB**
unified memory at short contexts, with prefill chunks of 256. Start at 1K and
leave memory for macOS and applications. This measurement was made on the
256 GiB M3 Ultra, not a physical 64 GB Mac; it does not qualify long contexts.
Allocator peak excludes memory outside MLX. Four-bit guidance remains **128 GB+**;
a 96 GB Mac may need a raised GPU wired-memory limit and is not physically
qualified.

The initial mixed comparison held the lock. A fresh priority #4679 validator
queued while the first serving harness loaded, so only our own incomplete
harness was interrupted and cleaned up. After roughly 63 minutes of waiting,
the remaining gates resumed under the lock. No foreign process was stopped.
The continuation environment is unchanged from above; parent merge
`acb5e990a` incorporates precision-boundary comments/tests without inference
math changes. Conversion recipe and focused regressions are recorded in
`6a866e188`. The focused suite passes **97 tests**; changed-line coverage against
`origin/vector/nemotron-35-bolt` is **142/142, 100%**, with Ruff/format checks
passing. The **3,848 catalog/README tests pass** on CPU. Local qualification
only; no push, upload or release.

## Reproduction and evidence

Evidence and evaluation-only harnesses are archived at
`/Volumes/RTL-2T/scratch-archive/rapid-mlx/super-vl-120b/evidence/`.
Use this checkout with the task-local Python environment and offline source cache.

```sh
output_root=/Volumes/RTL-2T/scratch-archive/rapid-mlx/super-vl-120b
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 lockf -k \
  /var/tmp/rapid-mlx-large-model.lock .venv/bin/python \
  "$output_root/evidence/evaluate.py" \
  --model "$output_root/text-4bit-g64" --name 4bit --tokens 16 --bench
```

The harness writes results to `/private/tmp/nemotron-super-vl-evidence/`.
`eval-4bit.json` contains both benchmark repetitions and eight token/logit traces;
`artifact-4bit.json` records the exact catalog file set. `source-hashes.json`
records all 88 validated source hashes. Performance numbers are specific to
this hardware, software, model recipe and workload.

The corresponding reference and HTTP gates use the archived harnesses:

```sh
source_snapshot="$HOME/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16/snapshots/a1ccf94d69377bf82c4db39798aac28447d9c16f"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 lockf -k \
  /var/tmp/rapid-mlx-large-model.lock .venv/bin/python \
  "$output_root/evidence/evaluate.py" \
  --model "$source_snapshot" --name bf16 --tokens 16 < /dev/null
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 lockf -k \
  /var/tmp/rapid-mlx-large-model.lock .venv/bin/python \
  "$output_root/evidence/serve.py" \
  --model "$output_root/text-4bit-g64" --name 4bit --port 8479
.venv/bin/python "$output_root/evidence/compare.py"
```

The HTTP harness starts a loopback server with `--text-only`, disabled disk
and prefix caches, prefill step size 256 and a 768-token response cap. It
checks name/color recall across turns, reasoning on/off, XML tool calls and
tool-result follow-ups in both stream modes, finish reasons, valid JSON
arguments and absence of control tokens in visible content/reasoning. It
terminates only the server process group it started. The BF16 harness rejects
checkpoint custom code and uses the local fast tokenizer fallback.

The mixed continuation uses the same archived reference and serving harness:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 lockf -k \
  /var/tmp/rapid-mlx-large-model.lock .venv/bin/python \
  "$output_root/evidence/evaluate.py" \
  --model "$output_root/text-3bit-mixed" --name 3bit --tokens 16
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 lockf -k \
  /var/tmp/rapid-mlx-large-model.lock .venv/bin/python \
  "$output_root/evidence/serve.py" \
  --model "$output_root/text-3bit-mixed" --name 3bit --port 8480
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 lockf -k \
  /var/tmp/rapid-mlx-large-model.lock .venv/bin/python \
  "$output_root/evidence/evaluate.py" \
  --model "$output_root/text-3bit-mixed" --name 3bit-perf \
  --prompt-index 0 --tokens 16 --bench
.venv/bin/python "$output_root/evidence/compare.py"
```

`eval-3bit.json` and `parity-3bit.json` contain the eight-prompt comparison;
`serve-transcripts-3bit.json` contains all seven HTTP responses;
`eval-3bit-perf-0.json` contains the 1K measurements. `artifact-3bit.json`
records the exact generator-filtered file set and weight extent audit.
