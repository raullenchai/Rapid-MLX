# Qwen3.8 serialized-MLLM fused-GDN canary

## Decision

Ship this optimization as an internal, default-off canary only. It is not an
alias default and does not change the Qwen3.6 MoE path. Enable it before
process start with:

```bash
RAPID_MLX_QWEN38_MLLM_FUSED_GDN=1 rapid-mlx serve \
  /path/to/the/canonical/aa985c29ff5b334cbfdcbbc787d47e66e9d9e456/snapshot \
  --mllm
```

The path must be the canonical local Hugging Face snapshot for
`rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX` at revision
`aa985c29ff5b334cbfdcbbc787d47e66e9d9e456`. B0 artifact truth and its opaque
verified-target mint are mandatory; a repo name, mutable revision, copied
directory, different quantization, ABI drift, or failed real-weight probe
leaves stock mlx-vlm active. Unset the variable (or set it to `0`) to roll
back. The canary status is exposed in engine statistics as
`qwen38_mllm_fused_gdn_canary` with `requested`, `qualified`, `active`, and a
closed `fallback_reason`.

## Frozen evidence

- Experiment source: `1c1c6573ab0353aa94bfb2b04e18208d5ddf2466`
  (`experiment/qwen38-mllm-fused-gdn`)
- Methodology SHA-256:
  `5eec2e6dbcd67e5b9ef2df856880eef468ea5394c7d43a3fbad62150394bb5b3`
- Raw receipt: external experiment artifact, identified by the SHA below. Its
  decision fields and safety evidence are summarized here so this document
  does not depend on a machine-private scratch path.
- Receipt SHA-256:
  `7fb5e03a8b672cb93943ff33238076b3ebad0814edf04e9059dbb1ecc14de5a2`
- Host: M3 Ultra, 256 GB unified memory; mlx-vlm 0.7.1 and rapid-mlx 0.15.0
- Runtime lock: mlx 0.32.2, mlx-lm 0.31.3, mlx-vlm 0.7.1. The measured
  `rapid_mlx/kernels/qwen4_fused_gdn_decode.py` source SHA-256 is
  `73deff77202039597b77cb75cd4e91dd91dfc05575acdca53a1b2937031b2db8`.
- Six prompt strata, ABBA/BAAB, two samples per arm, 256 fixed completion
  tokens: median wall-clock speedup **+7.80%** and median decode speedup
  **+8.12%**. All six strata were positive; paired wall-ratio CV was 0.135%
  and paired decode-ratio CV was 0.108%.

The receipt's raw aggregate result is **false** and must remain reported as
false. Its only failed aggregate check was the combined media semantic gate.
Independent adjudication found the VLM differential itself passed: stock
before, fused candidate, and stock after produced identical image token and
text hashes; every candidate GDN layer had the exact completion-plus-one hit
count; stock had zero candidate hits. The middle text output also matched the
earlier eager experiment exactly. The aggregate was invalid because it mixed
this differential with a new `"100" in text` assertion on a response that hit
its 64-token cap. The command also used the semantically matching custom image
prompt `What animal is shown? Reply with one short sentence.`, but the old
harness did not record that argument. This is a semantic-checker/provenance
false negative, not permission to rewrite the raw receipt as passing.

To reproduce the frozen experiment, check out the experiment commit and run
its benchmark in the pinned environment, entirely offline, with the same
canonical snapshot and image:

```bash
SCRATCH=/private/tmp/qwen38-fused-gdn-repro
mkdir -p "$SCRATCH"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/benchmark_qwen38_mllm_fused_gdn.py \
  --model /path/to/models--rapid-mlx--Qwen3.8-27B-4bit-MTP-MLX/snapshots/aa985c29ff5b334cbfdcbbc787d47e66e9d9e456 \
  --image-path apps/rapid-mac/Sources/Rapid/Resources/cheetah.png \
  --image-prompt 'What animal is shown? Reply with one short sentence.' \
  --image-expect cheetah \
  --output "$SCRATCH/qwen38-fused-gdn-receipt.json"
```

## Production safety contract

The canary reuses the one loaded VLM and its existing model-owner executor. It
does not load a sidecar or duplicate target weights. Admission requires the
exact ordered 64-layer/48-GDN layout, weight and cache geometry, source hashes,
mlx-vlm 0.7.1 ABI, plain non-speculative two-slot `ArraysCache`, and a
32-step real-weight bit-exact proof of output, convolution cache, recurrent
cache, metadata, and exact instance hits. Python errors before cache mutation
fall back to stock. Once cache commit begins, failures propagate and the
request cache is discarded; stock is never replayed. Measured forwards do not
add per-layer synchronization. Stop/reload restores the exact original class
method before the shared executor shuts down.

Only the M3 Ultra result is qualified. The target user hardware classes at
48 GB and 64 GB have not been measured, so this evidence is insufficient for
default-on behavior.
