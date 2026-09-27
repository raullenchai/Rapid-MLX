# Qwen Image 2.1 low-memory handoff

- Receiving role: Atlas for runtime/checkpoint integration; Vector for physical
  8 GB and 16 GB qualification.
- Branch: `atlas/qwen21-lowmem-experiment` from `origin/main` at `e68bb534`.
- Goal: make Qwen Image 2.1 usable on 8 GB and 16 GB Macs in both Server and
  GUI while preserving the previous path as `qwen-image-2.1-bf16`.
- Verified facts: a full MLX q4 mflux checkpoint was produced from the pinned
  official revision. It is 8.9 GB on disk. Its measured peak was 4.68 GiB for
  512-square/40-step generation and 5.14 GiB for 1024-square/four-step
  generation under MLX 8 GiB limits. Reloaded output matched the pre-save q4
  output byte for byte. Two visual prompt comparisons, including exact text and
  spatial relations, found no obvious semantic regression against the bf16
  encoder.
- Constraint: mflux 0.20.0 marks the Qwen3-VL encoder `skip_quantization=True`,
  and Rapid-MLX currently rejects quantized Qwen text encoders. Both contracts
  must change narrowly for the reviewed Qwen Image 2.1 full-q4 pack.
- Product integration: the default alias points to the public pinned
  `mlx-community/Qwen-Image-2.1-mflux-q4` revision
  `746a58556820933a2df5c75887a2570f1ad200c0`; the engine verifies and loads
  its q4 encoder, materializes prompt embeddings, and evicts components. The
  GUI catalog exposes both low-memory and bf16 aliases.
- Exact Server candidate: branch `efa24060` passed default generation and
  img2img against the pinned uploaded artifact. Generation was HTTP 200 at
  512-square/40 steps with 4.65 GiB peak RSS; img2img was HTTP 200 at the
  derived 1024-square canvas with 5.06 GiB peak RSS.
- Risk: testing ran on a 256 GB Mac Studio with MLX memory limits, not a
  physical 8 GB Mac. Quality evidence covers only two prompts.
- Next action: run the exact Server generation and img2img paths, then qualify
  the candidate on physical 8 GB and 16 GB Macs.
