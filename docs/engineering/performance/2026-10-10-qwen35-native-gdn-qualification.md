# Qwen3.5-family native GDN state qualification

Issue #4448 adds long-history and cross-checkpoint evidence beyond the original
rounding witness. The qualification command uses the original native layer as
its oracle and the production fused adapter as candidate. It captures the real
normalized output immediately before the output projection, projected output,
convolution cache and FP32 recurrent state. Both recurrent caches persist across
decode steps; neither is reset to the other's result after comparison.

Each trajectory starts with fresh native prefill and a cloned shadow cache. The
first arm determines the full model's subsequent hidden inputs and greedy tokens.
Both stock-first and fused-first trajectories run, with full evaluation barriers
between arms. Every tensor comparison includes storage-bit hashes, differing
count and index hash. On mismatch, tensors and the full differing-index arrays
are retained, and the trajectory fails. The harness also rejects silent stock
fallback, non-finite tensors and shape/dtype disagreement.

A repeated factual-record history exercises actual model prefill and generated
continuations. This is numerical text-model dogfood, not a semantic-quality,
HTTP serving, multimodal, or universal correctness qualification. No timing or
performance claim is made. The tested production adapter's probe chooses the
threadgroup geometry; this run does not force otherwise unselected geometries.

## Environment and source

- Apple M3 Ultra, macOS 26.5.2, Python 3.12.14.
- MLX and Metal 0.32.3, native model package 0.31.3, NumPy 2.4.4.
- Clean source commit: `ed25e6fda` (source hashes in the evidence inventory).
- Stock system MLX 0.32.2 is kept unchanged. Its failed admission uses native
  decode; that runtime is not counted as a passing fused qualification.

## Reproduce

Use an environment with the versions above and already cached full checkpoints.
The command does not download model weights or contact an existing service.

```bash
python -m scripts.qualify_qwen35_fused_gdn \
  --model "$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec" \
  --model "$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46" \
  --histories 128 8192 --steps 256 \
  --output /private/tmp/validate-qwen35-4448/evidence-new-run
```

## Verification boundaries

The command takes a cooperative lock excluding other copies of this harness and
executes arms serially. Other Studio services remain running, so hardware-wide
GPU exclusivity is explicitly false in the inventory. Hardware-exclusive evidence
and vision-wrapper qualification remain Vector/Harbor follow-up under #4448;
this report does not close those requirements. Existing admission and defaults
are unchanged because no newly admitted defect has been established.

The regression suite exercises a real native GDN layer. It covers a one-token
prefill tail, failed prefill cleanup, completed-step accounting, production
fallback and a state-only corruption whose normalized output remains identical.
The latter fails qualification and retains tensor and index witnesses. A mutation
restoring shape-only prefill detection makes the 257-token regression fail;
restoring the explicit prefill phase passes.

## Results

Both checkpoints enrolled 30 GDN layers and selected threadgroup Y=32. Each of
four trajectories per model completed all 256 decode steps. All 61,440 layer
comparison rows passed all four tensor comparisons, with zero differing storage
elements. Generated token sequences agreed between both orders for each history.

| Checkpoint | Immutable revision | Histories | Orders | Layer rows | Result |
| --- | --- | --- | --- | ---: | --- |
| Qwen3.5-35B-A3B-4bit | `1e20fd8d42056f870933bf98ca6211024744f7ec` | 128, 8192 | stock/fused, fused/stock | 30,720 | exact |
| Qwen3.6-35B-A3B-4bit | `38740b847e4cb78f352aba30aa41c76e08e6eb46` | 128, 8192 | stock/fused, fused/stock | 30,720 | exact |

[Machine-readable provenance and trajectory digests](2026-10-10-qwen35-native-gdn-qualification.json)
records source/runtime identities and the hashes of the complete gzip row streams.
Raw inventory, every row and generated continuations are retained on Studio under
`/Volumes/RTL-2T/scratch-archive/rapid-mlx-4448/ed25e6fda/`; the reproduction command
creates the same artifact structure. The result is bound to this source, runtime,
geometry and checkpoint matrix, with the execution limitation described above.

81 focused tests passed, including 19 qualification contracts. Ruff check and
format passed. Independent adversarial findings were fixed and re-reviewed to
LGTM: prefill-tail/error accounting, malformed-return fallback detection,
failed-row retention, shard completeness, and typed operational receipts.
Four tensor/return failure injections fail on the earlier harness and pass on
the fix. The same numerical matrix passed five complete runs; the final run
reproduced every earlier tensor hash.

Every referenced weight shard is streamed through SHA-256 and checked against
its cached content-addressed blob name before loading. Config/index hashes and
verified shard hashes are retained in the JSON provenance. Same-size corrupted
bytes and unverified ordinary weight files fail qualification. This identifies
local cached bytes; remote attestation of a cache owner's revision labels is
outside the command's offline contract.

Row-stream SHA-256 values identify exact gzip artifact bytes, including their
container headers. Reproducibility checks compare row identities and individual
tensor hashes; equal gzip artifact hashes across reruns are not claimed.

The remote review host could not authenticate; that attempt is not counted as a
passing review. Independent local review returned LGTM, and the complete PR
validation pipeline retains its separate Codex review gate.

Lock contention and lock-file access errors produce typed failing reports with
`exact=false` before any model runs. Real contention and denied-open regression
tests pass; independent review of this final fix returned LGTM. The clean source
above reproduced every numerical tensor hash from the earlier complete runs.
