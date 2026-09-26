# Qwen Image 2.1 low-memory handoff

- Receiving role: Atlas for runtime/checkpoint integration; Vector for physical
  8 GB and 16 GB qualification.
- Branch: `atlas/qwen21-lowmem-experiment` from `origin/main` at `e68bb534`.
- Goal: make Qwen Image 2.1 usable on 8 GB and 16 GB Macs without changing the
  existing q8 alias until the low-memory artifact is qualified.
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
- Risk: testing ran on a 256 GB Mac Studio with MLX memory limits, not a
  physical 8 GB Mac. Quality evidence covers only two prompts. The generated
  checkpoint remains local to the Studio lab and is not a publishable artifact.
- Next action: implement pinned full-q4 checkpoint loading and prompt
  materialization in the image engine, add hermetic loader guards, then run the
  documented suite on physical 8 GB and 16 GB Macs before changing catalog
  memory claims.
