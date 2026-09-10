# Vector handoff — Qwen3.8 27B Abliterated support

## Ownership and scope

- Receiving roles: Atlas for integration; Vector for any speculative follow-up
- Owner/host: Vector, Studio
- Branch: `feat/qwen38-27b-abliterated`
- Worktree: `/private/tmp/rapid-qwen38-27b-abliterated`
- Goal: expose the evidence-best abliterated Qwen3.8 27B MLX checkpoint through
  the shared Server/Desktop catalog, with exact artifact selection and honest
  capability gates.
- Non-goals: changing Smart/Fast defaults, endorsing model safety or accuracy,
  enabling an unmatched speculative drafter, or changing public APIs.

## Verified facts

- Alias `qwen3.8-27b-abliterated-4bit` selects only `oQ4e/` from
  `windowsxp811203/Qwen3.8-27B-Abliterated-MLX-MTP`. Qualification used the
  immutable revision `5b6802378702c89de48c990b29f3e55a3c84a2e3`; the current
  general alias schema does not pin target revisions.
- The selected 18-file artifact is 16,998,765,834 bytes. The normal
  Hugging Face cache was used and no sibling quantization was downloaded.
- Real Rapid-MLX inference passed coherent text, required tool calling, image
  input, four-way concurrency, streaming cancellation, and immediate recovery.
- Measured single-machine decode was 20.9 tok/s for the text smoke, 12.1 tok/s
  for the tool call, and 15.6 tok/s for the image response. These prompts differ
  and are not an A/B throughput comparison.
- The product path applied its `qwen3_5_norm_shift` repair to 161 gains. This
  addresses the conversion-layout issue that makes the publisher's stock
  mlx-lm 0.31.3 test produce invalid output.
- Vision loaded with mlx-vlm 0.6.17, the release-pinned version. A system
  installation at 0.6.16 correctly failed the dependency preflight rather than
  serving an unqualified combination.
- The alias is experimental, is not a recommendation, and keeps speculative
  decoding and MTP defaults disabled.

## Risk and next action

The repository's `drafter/` directory is a matching MTP head modified alongside
the abliterated trunk, but current alias metadata cannot pin a drafter subfolder
inside the target repository. Vector may propose a separate, scope-limited
change for `mtp_draft_subfolder` resolution. It must prove subtree-only fetch,
revision pinning, target/drafter compatibility, acceptance, correctness, and
net latency before Atlas enables the capability. The base-Qwen DFlash2/DSpark
drafter must not be substituted.

The reproducible evidence and commands are recorded in
`docs/engineering/performance/2026-09-09-qwen38-27b-abliterated-qualification.md`.
