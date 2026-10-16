# Nemotron 3.5 Super VL text support

Owner: Vector. Host: Studio, M3 Ultra (28 CPU / 60 GPU cores), 256 GiB
unified memory.
Branch: `vector/nemotron-35-super-vl`, stacked on the Bolt text support branch.
Local conversion and validation only; no push, PR, upload or release execution.

## Conversion and runtime boundary

The source is `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`, revision
`a1ccf94d69377bf82c4db39798aac28447d9c16f`. The coordinator restored the
original download. On resume, all **88 files** and **64 indexed weight shards**
are present in the standard Hugging Face cache. Every safetensors shard's
header-declared final tensor extent matches its physical file size.
All 88 cached files also match their Hugging Face blob content hashes
(SHA-256 for LFS blobs and Git blob SHA-1 otherwise); the complete hash pass
took 125.59 seconds. The snapshot totals 248,937,280,067 bytes; the index declares
248,906,833,960 tensor bytes. No second weight download was started by this task.

The text converter hoists `llm_config`, derives the 88-layer count, removes
remote-code selectors, and normalizes the encoded default infinite time-step
bound by omitting it. The artifact retains `model_type: nemotron_h`, the native
language tensor layout, tokenizer files and chat template. The opt-in
`rapid_mlx_nemotron_h: true` selects a local precision adapter through the
existing checked loader's architecture callback. Ordinary Nemotron artifacts
and explicit caller architecture callbacks retain their existing loading path.

The adapter reuses the existing Bolt sanitizer, native-dtype SSM softplus,
FP32 state-space math, FP32 router logits, latent MoE, attention and cache
implementations. Super adds its language normalization boundary: FP32
normalization, activation cast back to the input dtype, then weight scaling.
Its gated Mamba normalization also gates and reduces in FP32. The checkpoint's
omni source delegates language construction to the Transformers implementation;
the source's declared version is 5.12.1. This source inspection establishes the
intended fused-path boundaries, not independent full-checkpoint NVIDIA parity.

The uniform model uses 40 Mamba, 40 MoE and eight attention layers, 512 routed
experts with top-22 selection, expert width 2688 and latent width 1024. Routing
uses sigmoid scores, correction bias for expert selection, normalization of
original selected scores and scaling by five. Shared experts have width 5376.
Attention has 32 query heads and two KV heads, dimension 128, without rotary
position encoding. The declared position limit is 1,048,576; qualification is
limited to the measured workloads below.

Sanitization strips `language_model.`, drops the vision tower, `mlp1.*`, the
additional `vision_projector.*` final norm and `language_model.mtp.*`, transposes
original convolution weights and stacks routed experts. No existing Rapid
consumer supports this Nemotron-H MTP head; its configuration is also removed.
MTP speculative decoding remains disabled.

The converted metadata selects the existing `nemotron` XML tool parser and
`qwen3` implicit-thinking parser. The supplied reasoning parser uses `<think>`
and `</think>` and treats all generated text as content when the template
prefills `<think></think>` for `enable_thinking=false`. No new parser protocol
is introduced. Optional source-parser replay/force-nonempty behavior is not a
new serving API promise.

## Reproduction

Create the task-local Python 3.12 environment and install `-e '.[dev]'`.
Inference and GPU model tests use
`lockf -k /var/tmp/rapid-mlx-large-model.lock`. Offline conversion explicitly
selects the CPU and does not require the GPU lock. The complete 88-file source
snapshot has been verified; the converter additionally rejects any missing
indexed weight shard.

```sh
source_snapshot="$HOME/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16/snapshots/a1ccf94d69377bf82c4db39798aac28447d9c16f"
output_root=/Volumes/RTL-2T/scratch-archive/rapid-mlx/super-vl-120b
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
  scripts/nemotron_super_vl_convert.py --source "$source_snapshot" \
  --output "$output_root/text-4bit-g64" --bits 4
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
  scripts/nemotron_super_vl_convert.py --source "$source_snapshot" \
  --output "$output_root/text-3bit-mixed" --bits 3
```

The four-bit recipe uses affine/group-64 eligible projections and eight-bit
embeddings. The budgeted mixed recipe uses 26 interior MoE layers at
two-bit/group-128 and 14 boundary MoE layers at three-bit/group-128, other
eligible projections at four-bit/group-64, and eight-bit embeddings. Both
preserve the output head, router and latent projections at source precision.
The converter writes the model card and reports every output file, including
README, excluding `.cache` directories. Catalog sizing separately uses
`rapid_mlx._download_gate._is_weight_file`, the same filename filter used by
`scripts/gen_model_sizes.py`; it excludes README, LICENSE and the Jinja template.
Model cards require Rapid-MLX PR #4679 and follow-up. Four-bit guidance is
128 GB+; mixed guidance must come from its measured 1K peak, with the physical
qualification hardware identified.

Intended publication repositories, for human upload only:

- `rapid-mlx/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-MLX-4bit`
- `rapid-mlx/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-MLX-3bit`

## Vision disposition

The installed omni implementation has compatible RADIO, pixel-shuffle and
projector concepts, but Super adds a trained final LayerNorm on RADIO features
before pixel shuffle when the language model has MTP. The two tensors are
`vision_projector.vision_final_layernorm.{weight,bias}`. The installed omni
implementation does not consume or apply this module. Language config also
requires the derived layer count and numeric time-step bound. A direct type
rename is insufficient; image support requires an adapted projector and
language precision path plus real-image validation. Text artifacts deliberately
contain no image tower or image processor assets. Full multimodal loading
and image qualification remain pending.

## Evidence status

Based on Bolt `d0ee60130`, including the current-chunk SSM gather bound,
generator-aligned catalog expectations and optional-top-k validation guard. All 88 source files pass content
hashes. Conversion succeeds in 321.02 seconds; the text artifact loads and
completes eight greedy smoke prompts plus two 1K generator benchmarks.

Four-bit output:
`/Volumes/RTL-2T/scratch-archive/rapid-mlx/super-vl-120b/text-4bit-g64`.
It contains 69,540,668,652 bytes. The generator-compatible catalog footprint is
69,540,654,086 bytes. Fourteen converted shards pass extent checks; runtime
weights total 69,523,298,304 bytes. The directory includes its OpenMDW 1.1
model card, Rapid-only requirement and memory guidance.

The compiled norm implementation uses the same free-function binding pattern
as the shared precision adapter. Directly decorating an instance method did
not bind `self` with the installed MLX API; the tiny tensor contracts caught
this before qualification. The corrected norms pass the complete 53-test
Super/shared-model/parser suite. The converter also removes unused null model
selectors and tokenizer processor selectors from saved text metadata.

The evaluation-only paged BF16 reference passes its tiny full/cached resident
comparison bit-exactly. Full-checkpoint comparison completes on eight prompts:
strict greedy first-token agreement **6/8**, tie-aware agreement **8/8**,
first-token top-10 overlap **74/80**, and **5/8** identical sampled greedy
sequences (up to 16 tokens). The two first-token differences choose another
BF16-tied maximum. Remaining differences are wording or bold formatting.
This establishes a short quantization comparison, not independent NVIDIA
runtime parity or complete-function code correctness.

All seven HTTP serving requests pass: name/color recall across turns, reasoning
on/off, XML tool calls and tool-result follow-ups in stream and non-stream
responses, proper finish reasons, and no leaked control tokens. Examples:
`Your name is Ada, and your favorite color is teal.`, `17 × 23 = **391**`,
`get_weather({"city":"Paris"})`, and a sunny/18°C follow-up. The harness exits
zero and its own server stops cleanly; no other process is stopped by this task.

Direct generator throughput is 466.12–467.59 prompt tokens/s and 48.39–48.56
decode tokens/s for 1,024 prompt / 64 generation tokens, with a 72.68 GB MLX allocator
peak. [Environment, commands, parity table and evidence](../performance/2026-10-15-nemotron-super-vl-text.md).

The four-bit catalog alias `nemotron-3.5-super-vl`, exact filtered footprint,
README totals (273 aliases) and model table pass **3,774 tests**. The final
Super/shared-model/parser/metadata suite passes **95 tests**. Ruff, format and
diff checks pass. Changed-line coverage against
`origin/vector/nemotron-35-bolt` is **141/141 lines, 100%**.

The unrelated PR #4599 validator held the shared lock until approximately
19:50 UTC. After a brief conversion/sampling window, unrelated PR #4672 and
then already-queued PR #4682 held the lock until coordinator intervention.
The final regressions, full BF16 reference and HTTP serving then completed.
Roughly 70 minutes of the original time budget went to lock waits. The task
never bypassed the lock or stopped a foreign process. Mixed three-bit was
deferred before conversion to preserve the primary four-bit gates; its output
directory and catalog entry did not exist at that checkpoint. The continuation
results below supersede that mixed-artifact status. Four-bit qualification is local only;
there was no push, upload or release action.

Run focused tensor and conversion gates:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 lockf -k \
  /var/tmp/rapid-mlx-large-model.lock .venv/bin/pytest \
  tests/test_nemotron_h_super.py \
  tests/test_nemotron_h_puzzle_vendored.py \
  tests/test_nemotron_tool_parser_fail_open.py \
  tests/test_nemotron_bolt_metadata.py tests/test_model_metadata.py \
  tests/test_nemotron_super_metadata.py \
  --basetemp=/private/tmp/nemotron-super-vl-tests \
  --cov=rapid_mlx --cov=scripts --cov-report=xml:coverage.xml -q
.venv/bin/python -m diff_cover.diff_cover_tool coverage.xml \
  --compare-branch origin/vector/nemotron-35-bolt --show-uncovered --fail-under 100
```

Use the documented conversion commands only after all 88 snapshot files exist.
Eight parity prompts cover factual recall, arithmetic, code, translation, light
colors, a sequence, a brief explanation and user-name recall. Compare first-token
top-1/top-10 and greedy token sequences against MLX BF16. The evaluation-only
BF16 paging implementation must first pass its tiny resident-model comparison;
it copies exact selected source experts into the native switch MLP to avoid
materializing a roughly 241 GB contiguous model. Reference prompts run together
inside the exclusive window; every invocation holds the shared lock. Serving and
1K performance scripts are prepared in the private evidence directory.

The continuation produces the mixed artifact at
`/Volumes/RTL-2T/scratch-archive/rapid-mlx/super-vl-120b/text-3bit-mixed` on CPU
in 267.35 seconds. Its weights total 42,746,861,568 bytes and generator-filtered
catalog footprint is 42,764,296,516 bytes. All nine shard extents and protected
precision boundaries pass inspection. Keeping all experts at three bits would
produce about 51.925 GB; the requested disk and allocator budgets require most
expert layers at two bits. The selected 26 interior layers leave seven MoE
layers at each boundary at three bits. Per-module config records this mix.

The locked eight-prompt comparison passes the requested minimum of 5/8 strict
first-token matches: **6/8 strict**, **7/8 tie-aware**, **68/80 top-10 overlap**,
and **4/8 identical sampled sequences**. All **seven HTTP serving gates pass**,
including reasoning separation and streaming/non-streaming XML tools with
follow-ups. The 1K/64-token benchmark reaches **458.47–459.28 prompt tokens/s**,
**49.93–50.06 decode tokens/s**, and a maximum **45.903368468 GB MLX allocator
peak**, below the requested 46 GB limit. The final artifact is
**42,764,311,538 bytes**, including the measured-memory model card.

The artifact targets 64 GB unified memory at short contexts; start at 1K with
prefill chunks of 256 and leave memory for macOS and applications. Measurements
use a 256 GiB M3 Ultra, so a physical 64 GB Mac and long contexts remain
unqualified. The recipe and tensor/conversion tests are committed as
`6a866e188`; the parent branch's later precision-boundary comments/regression
are merged as `acb5e990a` without changing inference math.

The alias `nemotron-3.5-super-vl-3bit` selects the intended `MLX-3bit` repository,
native parsers, explicit recurrence, text-only capability and no speculative
decoding. The catalog uses the exact **42,764,296,516-byte** generator-filtered
footprint. README totals are **208 text / 274 total aliases**, and the model
table identifies mixed 2/3-bit precision explicitly. The original 4-bit alias
remains unchanged.

The focused suite passes **97 tests**. Ruff and format checks pass. Changed-line
coverage against `origin/vector/nemotron-35-bolt` is **142/142, 100%**. All
**3,848 catalog/README contract tests pass** on CPU, independently of the GPU lock;
all inference and tensor tests held the lock. After yielding only our own
initial serving harness, the gates waited approximately 63 minutes for the
priority #4679 validator, then completed under the shared lock.

Vector owns this completed local text qualification. No push, upload or release
was performed. Vision remains a separate image-input follow-up; the human
retains publication and release authorization.

Private operational scratch is under `/private/tmp/nemotron-super-vl-evidence/`;
completed measurements and harnesses are archived under
`/Volumes/RTL-2T/scratch-archive/rapid-mlx/super-vl-120b/evidence/`.
Role start FYIs were attempted through Orca; Atlas, Pixel and Echo named
mailboxes were unavailable. Harbor's live mailbox received the start FYI.
Atlas/coordinator owns release authorization and the human owns publication.
