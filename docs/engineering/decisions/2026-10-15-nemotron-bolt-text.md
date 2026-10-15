# Nemotron 3.5 Bolt text backbone

Owner: Vector. Phase A adds the local text architecture and unit contracts.
Phase B below qualifies local conversions and real-checkpoint behavior.
Publication and model catalog entries remain with the coordinator.
No model weights are required for the unit contracts.

## Config and checkpoint boundary

Converted text artifacts must have `model_type: nemotron_h_puzzle` at the root.
`rapid_mlx.models.nemotron_h_puzzle.text_config(config)` extracts the original
omni checkpoint's `llm_config` and removes its remote-code `auto_map`. Write that
returned dictionary as the conversion output's `config.json`, preserving the
checkpoint's tokenizer and chat template files separately.

This conversion boundary keeps text artifacts on the existing text-loader path
without adding an omni routing exception or advertising vision support. Direct
serving of the original `nemotron_h_omni` artifact is intentionally unsupported.
Architecture registration alone does not make the standalone conversion CLI
import Rapid's modules: conversion must explicitly call
`rapid_mlx.utils.tokenizer._register_vendored_archs()` before model construction.

The sanitizer accepts `language_model.backbone.*` and `language_model.lm_head.*`,
removes the language prefix, drops `vision_model.*`, `mlp1.*`, and MTP tensors,
and also accepts the standalone `model.*` backbone naming. It transposes original
convolution weights and stacks individual experts into the local switch layout.
Already sanitized text tensors pass through unchanged.

Each MoE block supplies its own intermediate size, expert count, and top-k.
The root expert count is only a fallback for older uniform checkpoints. The
language configuration's encoded infinite time-step bound is normalized before
execution. RMS normalization and router logits use FP32; the SSM's softplus
retains the projected activation dtype before FP32 state-space computation.
Attention, convolution, and MoE forwards reuse the installed dependency's blocks.
The local SSM adapter preserves those dtype boundaries without patching global
runtime functions. Masked or length-aware decode uses the attention-form SSM path.

The quantization predicate preserves `lm_head`, routers, and both latent
projections. Four-bit and mixed three-bit quality and size are not yet verified.
The checkpoint template uses the existing XML tool-call grammar and
`enable_thinking` prefixes; template compatibility is a source inspection finding,
not evidence of real-model generation quality.

## Verification and next phase

Phase A was validated on an Apple M5 Max Mac (36 GB, macOS 27.0.1)
with task-local CPython 3.12.15, MLX 0.32.4, mlx-lm 0.31.3, pytest 9.1.1,
coverage 7.16.2, and diff-cover 8.0.3. No model weights were downloaded.
The focused suite passed all 68 tests (13 Bailing, 13 Bolt, 5 registration/loader,
20 parser, and 17 CI coverage-roster contracts). Ruff check and format check
passed. Changed-line coverage against `origin/main` at
`6fb35fbb24c5b8bb151d4fe2a7d7858adfc6d146` was 100%: all 297 changed
executable production lines covered, zero missing.

The Bolt contracts now live in `tests/test_nemotron_h_puzzle_vendored.py`,
explicitly listed in the Apple CI lane. A separate architecture file follows
the established vendored-model test pattern and avoids coupling Bolt's config,
registration state, and numerical contracts to the unrelated Bailing backbone.
They cover heterogeneous config parsing, omni extraction, sanitization,
float32/BF16 four-layer forwards and cached decode, single-kind schedules,
norm/router math, quantization boundaries, SSM chunking and length-aware state,
and model registration. The full-size config contract allocates no model tensors.
Registration state is restored after each test; random weights use seed 0.

The initial M5 default-precision run had three numerical failures across the
focused suite: Bailing full-prefill/cache parity, Bolt float32 full-prefill/cache
parity, and Bolt SSM full/chunk parity. The full-float32 rerun passed.

The Bolt float32 comparisons now account for TF32's 10 fraction bits without
requiring a process environment setting. They bound both RMS and peak error by
`8 * 2**-10` times the corresponding reference scale. The eight-rounding-unit
budget accommodates composed matmuls selecting GEMM versus GEMV kernels; scaling
by tensor RMS/peak avoids unstable elementwise relative errors at cancellation
zeros. Exact expert-index and per-layer weight-shape assertions remain strict,
as does router score normalization/scaling. Negative controls inject a 1% output
head scaling error, omit router scaling, and remove the SSM residual. All must
be detected. BF16 tolerances are unchanged.

Installed MLX 0.32.4 caches `MLX_ENABLE_TF32` in a function-local C++ static in
`mlx/include/mlx/utils.h`; the Python core and Metal API stubs expose no precision
switch. The existing `test_muse_glimmer_model.py` precedent selects full fp32 in
a fresh subprocess to compare cache semantics at equal precision. These Bolt
contracts instead retain the caller's precision policy to cover the default GPU
path, using the bounded comparisons and negative controls above.

The test portability follow-up was validated on an Apple M4 Pro Mac, 48 GiB,
macOS 26.5.1, CPython 3.12.12, MLX 0.32.4, mlx-lm 0.31.3. Both the unset default
policy and `MLX_ENABLE_TF32=0` passed all 68 focused tests, including the negative
controls. Each run covered all 297 changed production lines against the Phase A
base and all 23 executable test lines changed by this follow-up against
`921021257`. Ruff and diff checks passed. These follow-up runs establish M4
behavior; M5 execution of the revised comparisons remains unverified.

From the repository root, create a Python 3.12 `.venv` and install
`-e '.[dev]'` plus `diff-cover==8.0.3`. Run the following there:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/pytest \
  tests/test_bailing_hybrid_model.py \
  tests/test_nemotron_h_puzzle_vendored.py \
  tests/test_gpt_oss_puzzle_vendored.py \
  tests/test_nemotron_tool_parser_fail_open.py \
  tests/test_ci_apple_coverage_union.py \
  --cov=rapid_mlx --cov-report=xml:coverage.xml -q
.venv/bin/ruff check rapid_mlx/models/nemotron_h_puzzle.py \
  rapid_mlx/utils/tokenizer.py tests/test_bailing_hybrid_model.py \
  tests/test_nemotron_h_puzzle_vendored.py
.venv/bin/ruff format --check rapid_mlx/models/nemotron_h_puzzle.py \
  rapid_mlx/utils/tokenizer.py tests/test_bailing_hybrid_model.py \
  tests/test_nemotron_h_puzzle_vendored.py
```

Repeat the pytest command with `MLX_ENABLE_TF32=0` for full-float32 parity.

Set `COVERAGE_XML` to the generated `coverage.xml` path. From the source
worktree, mirror CI's diff-cover gate:

```sh
.venv/bin/python -m diff_cover.diff_cover_tool "$COVERAGE_XML" \
  --compare-branch origin/main --show-uncovered --fail-under 100
```

CI normally combines Linux and Apple coverage first; this focused Apple run
alone covers every changed executable production line.

Original Phase A handoff (superseded by Phase B below):

- Download and convert the original BF16 checkpoint in the standard shared cache.
- Prove four-bit/group-64 loading and evaluate a mixed three-bit 32 GB variant;
  record actual artifact sizes and the precise conversion recipe.
- Compare reference logits/tokens, check multi-turn chat and thinking on/off,
  and exercise streaming and non-streaming tool calls through the server.
- Measure prefill and decode on the converted artifact with exact commands.
- Add aliases and model sizes only after a local artifact works and the
  coordinator supplies the publication repository ID.

Phase A makes no real-checkpoint parity, quality, memory, or throughput claim.

## Phase B: M3 Ultra qualification (2026-10-15)

Conversion uses MLX's CPU device. No downloads, cache cleanup, publication,
push, or catalog changes are made in this phase.

Environment: M3 Ultra, 256 GiB unified memory, macOS 26.5.2, Python 3.12.14,
MLX 0.32.4, mlx-lm 0.31.3, Transformers 5.15.1, PyTorch 2.14.1.
Source: the existing shared-cache snapshot of
`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16`, commit
`bca0a8cb74200a2eb7ca56eb8428b821c829bd6e`.
For reproduction, set `MODEL_DIR` to the converted four-bit artifact directory
and `OUT_DIR` to a user-supplied evidence output directory. Historical raw
evidence filenames below identify local records; these files are not shipped
with the repository.

### Conversion evidence

| Artifact | Exact bytes, including tokenizer/config | GB (decimal) | GiB | Wall time |
| --- | ---: | ---: | ---: | ---: |
| `text-4bit-g64` | 43,274,345,550 | 43.274 | 40.302 | 201.405 s |
| `text-3bit-g64` (mixed recipe below) | 31,670,620,592 | 31.671 | 29.496 | 157.841 s |

The text backbone has 74,042,705,408 parameters. Artifact tensor payloads are
43,256,967,168 and 31,653,163,008 bytes respectively. Both conversions use
`_register_vendored_archs()`, `load_model(view, lazy=True)`,
`quantize_model`, `save_model(..., donate_model=True)`, and `save_config`.
The `bf16-text-view` directory contains a `text_config(original_config)`
config and symlinks to original cached files; it does not duplicate weights.
The original tokenizer files, chat-template sidecar, and generation config are
copied unchanged into the quantized artifacts. No remote model code is used
for conversion or MLX execution.

Exact recipes (affine quantization, all exclusions first delegated to the
vendored model's `quant_predicate`):

- Four-bit: eligible weights 4-bit/group-64; `backbone.embeddings` 8-bit/group-64.
- Mixed: expert `switch_mlp.fc1` and `fc2` weights in layers
  **17, 28, 39, 50, 61, 72** use 2-bit/group-128; experts in the other
  34 MoE layers use 3-bit/group-128. Other eligible projections use
  4-bit/group-64; embeddings use 8-bit/group-64. The artifact directory's
  `g64` suffix describes the global default; per-module config overrides are
  authoritative. This is a mixed approximately-three-bit recipe, not uniform
  three-bit/group-64.
- Both retain the head, router gates, and both latent projections in BF16,
  and leave non-quantizable tensors unchanged. An index audit finds no
  quantization scales for these protected tensors. Mixed config has 68
  expert projection modules at 3/128, 12 at 2/128, 192 other modules at 4/64,
  and one embedding at 8/64. Effective total precision is 4.674 and 3.420
  bits/parameter respectively.

An attempted group-256 recipe failed before writing shards: MLX 0.32.4
supports affine groups 32, 64, and 128. Uniform 3-bit/group-128 experts with
these other precision boundaries were estimated at 32.922 GB, so the six
interior 2-bit layers meet the strict decimal 32 GB target. Quality and runtime
memory require execution evidence; disk fit alone does not qualify a 48 GB Mac.

Free space on the conversion output filesystem was checked before each
conversion. Initial free space was 123 GiB; the mixed conversion started with 84 GiB available. After both
artifacts, 53 GiB remained. An 8-bit artifact and a full BF16 copy were avoided.

Conversion workload specification: run offline with `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`, select `mx.set_default_device(mx.cpu)`, and use the
registration, lazy loading, quantization, and save APIs listed above for each
recipe. The original conversion helper is not shipped in the repository;
this is a workload specification rather than an executable conversion command.

### Runtime fixes and reference boundary

Converted paths initially resolved no automatic profile. Local commit
`3260250b0` recognizes the explicit `nemotron_h_puzzle` text architecture:
`tool_call_parser=nemotron`, `reasoning_parser=qwen3`, hybrid/MoE true,
standard speculative decoding disabled. The shared recurrence check now
recognizes that architecture before loading weights, so recurrent prefix-cache
admission has the same safety signal. Generic template inference cannot replace
the dedicated tool parser. No model-name regex or alias entry was added.

Validation: all 360 tests in `tests/test_nemotron_bolt_metadata.py` and
`tests/test_model_auto_config.py` pass in 39.24 s; Ruff and format checks pass.
The vendored-model test file was unchanged in this phase.

The cached HF reference imports `mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn`
unconditionally and raises `ImportError` without it. `mamba_ssm` is absent on
this Apple host. The full HF reference comparison is therefore infeasible in
the installed environment; no claim of HF/MLX full-model parity is made.
The fallback is eight-prompt quantized-versus-MLX-BF16 top-k and greedy-prefix
comparison using Rapid's loader, not an independent validation of model math.

Initial parity harness attempts stopped after loading because Transformers 5
returned `BatchEncoding` from `apply_chat_template`; the evaluation must extract
`input_ids`. No model implementation change was needed for that API difference.
A first server attempt exposed a missing `nemotron_h_puzzle` entry in
`byom.preflight.RAPID_MODEL_TYPES`. Local commit `c5fb48e11` adds it; all 116
preflight tests pass in 0.95 s, including the existing registration-inventory
contract. Serving then passes with preflight enabled.

### Proposed catalog handoff (not applied)

After qualification and assignment of publication repository IDs, propose
`nemotron-3.5-bolt-74b-a9b-4bit` and
`nemotron-3.5-bolt-74b-a9b-mixed-3bit`. Each would declare
`modality: text`, `is_text_only: true`, `tool_call_parser: nemotron`,
`reasoning_parser: qwen3`, `is_hybrid: true`, `is_hybrid_explicit: true`,
`is_moe: true`, and `supports_spec_decode: false`, with `hf_path` set to the
coordinator-supplied publication ID. Do not infer a tested minimum-memory
requirement from disk size. Catalog admission remains gated on the execution
results below and coordinator disposition.

The current size-manifest rule counts weights and JSON/tokenizer files but
excludes `.jinja`: proposed `model_sizes.json` values for these exact local
payloads are **43,274,333,036** and **31,670,608,078** bytes, keyed by the eventual
publication IDs. Whole-directory totals in the conversion table additionally
include the 12,514-byte template. Neither catalog file has been edited.

### Parity smoke set

All three artifacts load through `load_model_with_fallback`; BF16 loads
148,085,469,184 bytes of language weights from the symlink view. Thinking is
explicitly off for this comparison. Next-token logits are collected from a
single uncached full-prompt forward. Independently, greedy generation uses a
fresh normal model cache, temperature zero, prefill step 256, and a 32-token cap.
Finite-precision full-prefill and cached-generation paths can choose different
first tokens near a tie; the two measurements must not be conflated.
Top-k agreement is set intersection divided by k. Greedy prefixes below count
matching leading generated token IDs, including a final EOS when emitted.

| Quantization vs MLX-BF16 | Next-token top-1 | Mean top-5 overlap | Mean top-10 overlap | Exact greedy outputs | Mean matching prefix |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4-bit | 8/8 | 92.5% | 92.5% | 5/8 | 8.375 tokens |
| Mixed approximately 3-bit | 6/8 | 90.0% | 87.5% | 6/8 | 9.125 tokens |

| Prompt (exact text except code formatting) | BF16 output tokens | 4-bit prefix | Mixed prefix |
| --- | ---: | ---: | ---: |
| What is the capital of France? Answer briefly. | 2 | 0 | 2 |
| Calculate 17 times 23. Answer with the number only. | 4 | 4 | 4 |
| Write a Python function that adds two integers. | 32 | 32 | 32 |
| Translate to Spanish: Good morning. | 5 | 5 | 5 |
| Name three primary colors of light. | 18 | 18 | 18 |
| Continue the sequence: 2, 4, 8, 16, | 3 | 0 | 3 |
| In one sentence, explain why the sky is blue. | 32 | 2 | 9 |
| My name is Ada. What is my name? Answer briefly. | 6 | 6 | 0 |

Differing generations were coherent wording differences: BF16's capital answer
was `Paris`, while 4-bit used `The capital of France is Paris.`; BF16's sequence
answer was `32`, while 4-bit explained the powers-of-two pattern. Mixed used
`Ada` where BF16 used `Your name is Ada.` Both quantizations returned `391`,
`Buenos días.`, RGB primary colors, and a correct addition function. This small
smoke set does not establish broad accuracy or favor mixed quantization over
4-bit. The HF reference boundary above remains unresolved.

### OpenAI serving and parser evidence

The four-bit artifact was served at `127.0.0.1:18035`, with neither parser
specified on the command line. Startup logs select `nemotron` tool parsing,
`qwen3` reasoning parsing, and hybrid safety defaults. Preflight is enabled.
All seven `/v1/chat/completions` requests succeeded with HTTP 200, temperature
zero, maximum 768 generated tokens, and the original model-owned template.

Short transcripts:

- Thinking off, turn 1: user says their name is Ada and favorite color teal;
  assistant: `Got it, Ada. I've noted that your favorite color is teal.`
- Turn 2: `What is my name and favorite color?` →
  `Your name is Ada and your favorite color is teal.`
- Thinking on: `What is 17 times 23?` → content `17 × 23 = **391**`, with
  nonempty reasoning separated into `reasoning_content`.
- Both non-streaming and SSE: `Use get_weather to check the current weather
  in Paris.` → one structured `get_weather` call with arguments
  `{"city":"Paris"}` and finish reason `tool_calls`.
- After the synthetic tool response (Paris, 18 C, sunny), both transports return
  `The current weather in Paris is sunny with a temperature of 18°C.`

The harness checks assembled content and reasoning against every special token
marked in `tokenizer.json`, plus think/tool markup. No special tokens, think
tags, or raw tool envelope leaked. SSE call fragments reconstruct to the same
name and valid JSON arguments as non-streaming. These seven serving requests
qualify the four-bit artifact; mixed serving itself was not tested. The weather
response is synthetic test data, not a live weather lookup.
The harness stopped only its own server process group. The SIGTERM thread dump
in the log is intentional signal observability, followed by clean engine and
application shutdown; port 18035 is free afterward.

### Prefill and decode evidence

Standalone `mlx_lm.stream_generate` through Rapid's loader, batch one,
temperature zero, BF16 cache, prefill step 256, no prefix reuse or speculative
model. Prompts are exactly 1,024 or 8,192 token IDs, formed by repeating/trimming
the tokenizer encoding of `The quick brown fox jumps over the lazy dog. `.
Each run generates 64 tokens. Two sequential measurements per prompt length
are retained, not selected by best result; no separate warmup is excluded.
Rates are the generator's reported prompt and generation rates.

| Artifact | Prompt tokens | Prefill tok/s, runs 1 / 2 | Decode tok/s, runs 1 / 2 | Peak MLX GB |
| --- | ---: | ---: | ---: | ---: |
| 4-bit | 1,024 | 527.82 / 535.72 | 33.53 / 32.16 | 46.189 |
| 4-bit | 8,192 | 537.56 / 491.28 | 31.51 / 30.55 | 46.247 |
| Mixed approximately 3-bit | 1,024 | 435.46 / 464.29 | 23.71 / 23.89 | 34.585 |
| Mixed approximately 3-bit | 8,192 | 505.89 / 432.97 | 25.84 / 31.50 | 34.644 |

The smaller recipe trades throughput for storage and allocation savings on this
host. Its maximum observed standalone allocation is 32.264 GiB, consistent
with the 48 GB target's headroom, but an M4 Pro/48 GB machine has not been tested
and allocator peak is not total system RAM usage. Do not transfer these M3 Ultra
rates to M4 Pro.

This is a shared machine, not an isolated performance baseline. Existing
operator servers on 7659/7660 and the validator's 8451 server were left alone.
Start-of-run one-minute system load averages were 7.00 (mixed), 7.72 (BF16), and
6.46 (4-bit); process/CPU/RSS snapshots are saved with the evidence. Source
conversion had finished before these runs. The variation between repeats is
material; no speedup claim is made.

### Execution workload and evidence files

The original evaluation and serving harnesses are not shipped in the repository.
Reproduce the parity smoke set and standalone benchmarks using the prompts,
loader, cache, token limits, repetitions, and comparison metrics specified above.
Run offline with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; save results
under `$OUT_DIR`, along with software versions and workload parameters.

The serving workload uses the following repository command from the worktree
root with a `.venv`. Set `MODEL_DIR` to the four-bit artifact directory. Wait
for health, issue the seven requests specified above, retain response bodies
and SSE chunks, then stop the server:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  RAPID_MLX_TELEMETRY=0 DO_NOT_TRACK=1 \
  .venv/bin/python -m rapid_mlx.cli serve \
  "$MODEL_DIR" --host 127.0.0.1 --port 18035 \
  --text-only --disable-disk-caches --disable-prefix-cache \
  --prefill-step-size 256 --max-tokens 768
```

Raw results: `conversion-{4,3}bit.json`, `eval-{bf16,4bit,3bit}.json`,
`summary.json`, `serve-transcripts.json` (requests, response bodies, and SSE
chunks), `serve-4bit.log`, `concurrent-load-{bf16,4bit,3bit}.txt`, and
`commands.json`. Successful evaluation wall times including load were 60.40 s
(BF16), 58.92 s (4-bit, including benchmarks), and 68.18 s (mixed, including
benchmarks), excluding lock wait.

Final disk check after serving: approximately **38.3 GiB free**
(41,163,239,424 bytes at the final evidence checkpoint; `disk-final.json`). The OS reports roughly
14 GiB allocated swap. No scratch artifacts or HF cache entries were deleted.
No push or upload occurred. Phase B's local qualification is complete within the
explicit HF-reference and M4-hardware limits. Atlas/coordinator owns publication
ID assignment, catalog admission, and any broader quality or actual M4 testing.

## Round 1 review correction and catalog admission

The original text-only scope and non-goals are unchanged. Catalog admission
uses the publication repository IDs below. Converted directories remained
read-only inputs; no conversion or model-directory modification is part of
this round.

The right-padded SSM regression compares a 333-position batch against independent
sequences of lengths (256, 300), (0, 256), and (255, 257), with a nonzero incoming
state and slow decay. This exercises a 256-token boundary, an exhausted sequence,
and a partial final chunk, both with and without an explicit mask. Valid
outputs and final recurrent cache states must
match; zero valid tokens must preserve the incoming state exactly.
The top-one router regression uses zero logits and checks that normalized weights
equal the routed scaling factor, while the unnormalized control retains the
sigmoid probability times that factor.

The installed shared `mlx_lm.models.ssm.ssm_attn` in mlx-lm 0.31.3 contains the
same unmasked incoming-state cumulative decay and `lengths < 0` preservation
condition copied into this vendored adapter. The ordinary `nemotron_h` mixer
imports its `ssm_update`, so existing models using that shared path are also
exposed to this right-padding defect when an incoming state or an earlier chunk
exists. This is a pre-existing dependency defect, not a new effect on those
models. **Follow-up owner: Vector** — reproduce against the shared dependency,
fix it separately with batched cache regression coverage, and qualify other
Nemotron-H models. This PR changes only the Bolt adapter.

Catalog entries:

| Alias | Hugging Face repository | Manifest payload bytes |
| --- | --- | ---: |
| `nemotron-3.5-bolt` | `rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit` | 43,274,333,036 |
| `nemotron-3.5-bolt-3bit` | `rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-3bit` | 31,670,608,078 |

The manifest records `scripts/gen_model_sizes.py` output: generator-counted
weight and JSON/tokenizer files, excluding `README.md` and `.jinja`, consistent
with the download-size estimator. These are not whole-directory totals.
The complete local artifacts total 43,274,347,300 and 31,670,622,410 bytes,
including `README.md` (1,750 / 1,818 bytes) and `chat_template.jinja` (12,514
bytes each). The three-bit
repository uses the mixed recipe already documented in Phase B. Both aliases
select native tool/reasoning parsers, text-only hybrid/MoE safety, and disabled
standard speculative decoding. Neither asserts a tested minimum Mac memory.

### Round 1 verification and handoff

The regressions were written and run against the unchanged implementation
before applying fixes. Exact result: **7 failed, 1 passed, 13 deselected**.
The normalized top-one test observed 2.5 instead of 5.0; all six padded-state
variants failed. Raw output: `$OUT_DIR/round1-red.log`.
The local fix derives the chunk mask from remaining lengths, applies that mask
to incoming-state decay, and preserves incoming state when lengths are zero
or negative. Deriving the mask also prevents padded state contributions when
lengths are supplied without a separate mask. The top-one override normalizes
before scaling, retaining the reference epsilon and unnormalized behavior.

An isolated reproduction against the installed shared SSM (zero input, unit
incoming state, 333 positions, timestep logits -5) returned final state values
`[0.179222, 0.106863, 0.106863]` for lengths `[0, 256, 300]`, rather than
`[1.000000, 0.179222, 0.133373]`. This confirms the separate shared-path follow-up;
no non-Bolt code was patched.

Final validation on this worktree:

- **4,419 passed** in 67.88 s: Bolt, existing vendored/registration/tokenizer
  contracts, preflight, metadata, aliases, sizes, atomic catalog, CLI and serving
  snapshots, parser contracts, and Apple coverage-roster checks.
- **21/21 Bolt tests passed** with `MLX_ENABLE_TF32=0` in 2.51 s, including exact
  BF16 normalization, cached decode, and the new regressions.
- Ruff check and format check passed for all changed Python files.
- Changed-line coverage versus `origin/main` at
  `6fb35fbb24c5b8bb151d4fe2a7d7858adfc6d146`: **314/314, 100%, zero missing**.
- CLI and serving snapshots were regenerated and remained byte-identical.
  Packaged catalog-schema checks passed; atomic catalog snapshots are built
  dynamically from the alias and size sources and need no checked-in rewrite.

Historical verification logs are `round1-validation.log`, `round1-fp32.log`,
`round1-coverage.xml`, and `round1-diff-coverage.log`. The original verification
helper is not shipped in the repository. Reproduce the validation using the
focused test groups and checks listed above, saving logs under `$OUT_DIR`.
The converted directories remained read-only inputs.

[The decode investigation](../performance/2026-10-15-nemotron-bolt-decode.md)
records the layer profile, bytes-read bound, machine activity, and follow-ups.
A quiet baseline measured 51–52 tok/s; compiling the existing FP32 normalization
math locally produced 56.29 / 56.50 / 56.50 tok/s in the final implementation,
about 9–10% faster than the matched baselines. MoE is the largest measured
layer category, followed by Mamba; Python graph construction does not dominate.
Larger expert/SSM kernel or graph-capture changes remain a Vector follow-up.

Vector owns the shared-path correctness and deeper performance follow-ups above.
