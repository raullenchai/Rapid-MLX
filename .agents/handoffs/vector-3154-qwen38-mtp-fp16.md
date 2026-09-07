# Vector handoff: Qwen3.8 27B MTP FP16 sibling

- Owner: Vector
- Issue: #3154
- Branch: `vector/3154-qwen38-mtp-fp16`
- Worktree: `/private/tmp/vector-3154-qwen38-mtp-fp16`
- Published model: `rapid-mlx/Qwen3.8-27B-4bit-MTP-fp16-MLX`
- Published revision: `53542c8c8c41261bfde18871197daac4e4fdb74f`

## Goal and scope

Give M1/M2 users an explicit FP16-storage sibling of the existing Qwen3.8 27B
4-bit MTP checkpoint. Preserve the regular alias and all engine behavior. No
automatic hardware routing, engine math change, default switch, or unrelated
model/catalog work belongs in this PR.

## Verified facts

- Converted 1,705 BF16 tensors to FP16 and preserved 506 non-BF16 tensors
  value-for-value across three target shards and the MTP sidecar.
- Read-back verification covered tensor names, shapes, dtypes, values, and
  safetensors metadata. The sidecar checksum was regenerated.
- The public artifact is 16.3 GB and contains 18 expected repository files.
- Rapid-MLX loaded all 31/31 MTP tensors from the sibling on M3 Ultra.
- Two of three fixed greedy FP16/BF16 prompts were byte-identical; the third
  differed by one semantically equivalent phrase, so no universal identity
  claim is made.
- FP16 MTP output was byte-identical to FP16 autoregressive output for the
  paired smoke prompt. Four simultaneous HTTP requests completed.
- Reporter evidence on M2 Max: a 2,726-token cold prefill improved from 133 to
  178 tok/s (+33.8%, 1.34x) when only non-quantized storage changed BF16→FP16.

## Reference-first check (private)

Reviewed vLLM and SGLang checkpoint conversion/loading precedents first, then
MLX-LM's dtype conversion and atomic shard-save path, and the repository's
existing Qwen3.8 streaming converter. Adopted a deterministic offline artifact
conversion with explicit source revision, per-shard read-back, a success
manifest written last, and publication kept outside the converter. Rapid-MLX
diverges only by preserving its colocated MTP sidecar and validating the
extensionless blob layout used by the local Hugging Face cache.

## Adversarial review record

1. Artifact input review found that cache snapshot symlinks resolve to
   extensionless blobs. Fixed loading by explicitly selecting safetensors and
   added a regression test.
2. Integrity review found that copying the source sidecar checksum left stale
   metadata. Fixed by regenerating SHA-256 after conversion and asserting it.
3. Numerical review rejected the original byte-identity assumption after one
   of three M3 prompts diverged at a normal FP16/BF16 boundary. Model card,
   performance report, and PR evidence state the measured 2/3 result.
4. Boundary review added fail-closed existing/nested output checks, indexed
   path-escape rejection, required-sidecar enforcement, and preflight disk
   capacity validation. The converter has no upload path.
5. Alias review kept the BF16 alias unchanged, declined silent chip routing,
   and deliberately did not copy DFlash revision pins because they bind the
   original target revision.
6. Full-unit review caught the new repository missing from the checked-in size
   manifest. Added the exact Hub `used_storage` value (16,313,463,520 bytes)
   and passed the size-manifest and adjacent image-precision suites. The other
   initial full-unit failure was a local validation environment missing the
   declared image extra; installing that extra made the unchanged test pass.

## M2 Pro follow-up

The mini had another managed inference model resident and 1,914 MiB swap in
use, so this task declined to load Qwen or publish a contaminated A/B. Both
fixed checkpoints are being placed in the policy-mandated default Hugging Face
cache for later #3156 work; no other cache or model was removed. A future
paired measurement requires an owner-provided idle, low-swap window. The PR
uses the supplied M2 Max result with explicit attribution in the meantime.
