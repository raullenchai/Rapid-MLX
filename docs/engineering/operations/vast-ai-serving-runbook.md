# Operation: Serving Marvin on rented NVIDIA GPUs (Vast.ai)

Status: PREP — nothing here spends money or deploys. Serving/renting requires
explicit human authorization (repo safety rules). This document is the
checklist so the actual session is ~1 hour, not a research project.

## The one hard constraint

**MLX does not run on NVIDIA.** Our training/eval stack (mlx-lm) is Apple
silicon only. "Rent a Vast 4090 and serve Marvin" therefore means moving the
BASE + LoRA into a CUDA-capable runtime — the reference target is
**llama.cpp `llama-server`** (CUDA build) fronted by our thin `/v1/classify`
proxy. vLLM is the fallback; Mac-cloud (MacStadium / AWS mac2) is the
no-conversion escape hatch.

## Paths (decision tree)

| Path | Base weights | Adapter | Risk | Est. decision latency* |
| --- | --- | --- | --- | --- |
| **A. llama.cpp CUDA** (primary) | GGUF quant of Ternary-Bonsai-27B (Q4_K_M fits 24 GB; ternary i2_s if upstream ships BitNet-style weights) | MLX LoRA → PEFT → `convert_lora_to_gguf` | quant conversion + LoRA conversion are the two unknowns | ~100–250 ms |
| B. vLLM | same GGUF/HF | merge LoRA into base, serve merged | BitNet-class ternary support immature | ~50–150 ms |
| C. Mac cloud (no Vast) | current MLX 2-bit, unchanged | unchanged (zero conversion) | cost, not tech | ~600–900 ms |

\* estimates to verify in smoke test; Jev p50 is 190 ms hosted.

**Accuracy contract:** every published number (94.44% ± 1.08) was measured on
the 2-bit MLX quant. Any new quant/merge MUST re-run the 192-item held-out
readout before any claim repeats. The letter-prob readout is reproducible on
llama.cpp via next-token logprobs of the candidate letters (proxy implemented
in `serve/classify_proxy.py`, backend `llama_cpp`).

## Vast.ai session plan (~1 h, ≈ $0.40)

1. **Rent**: RTX 4090 24 GB (or L40S), PyTorch 2.x + CUDA 12 template,
   1 port open. Interruptible pricing fine for smoke tests.
2. **Convert** (on the box, from upstream HF weights — never from our
   quantized snapshot):
   - `python convert_hf_to_gguf.py <bonsai-27b-hf> --outfile bonsai-27b-Q4K.gguf`
     then `llama-quantize` if needed.
   - Adapter: `python convert_lora_to_gguf.py` requires PEFT format —
     run `serve/export_lora_peft.py` (converts mlx-lm LoRA safetensors →
     PEFT layout, preserving adapter⇄template metadata below).
3. **Serve**: `llama-server -m bonsai-27b-Q4K.gguf --lora lora.gguf
   --port 8080 -ngl 99 -c 4096`.
4. **Verify (gate)**:
   - `python serve/classify_proxy.py --backend llama_cpp --base http://localhost:8080 --selfcheck`
   - Re-run 192-item eval through the proxy; PASS = within 1.5 pts of
     94.44% (quant tolerance) and p50 < 400 ms.
5. **Only then**: keep instance alive behind the proxy with API key;
   record instance id, cost/hr, and snapshot id in ops log. Teardown plan:
   destroy instance; weights + adapter live in object storage (HF repo),
   nothing stateful on the box.

## The adapter⇄template contract travels with the adapter

Any serving path must carry (from `bench/marvins_garden/README.md`):
- `think_mode`: `enabled` for the v15c lineage (training-matched template) —
  stored in `serve/adapter_manifest.json` next to the weights.
- prompt rendering: `render.render_prompt(...)` byte-identical to training.
- candidates: letter menu A..H; readout = softmax over letter tokens.

## Cost model (2026-09 spot prices, verify at booking)

- RTX 4090 24 GB: ~$0.28–0.40/hr → ~$250–350/month always-on.
- A100 80GB (only if bf16 merge is required): ~$1.30–1.60/hr.
- Mac cloud M2 Ultra: ~$0.80–1.20/hr, zero conversion.

## Open questions (to resolve on the rented box, listed honestly)

1. Does upstream `prism-ml/Ternary-Bonsai-27B` ship bf16 safetensors (needed
   for GGUF conversion)? Which repo/revision is canonical?
2. llama.cpp ternary (i2_s) support for this architecture — if the arch is
   BitNet-b1.58-like, conversion may be direct; otherwise Q4_K_M fallback
   (accuracy gate above decides).
3. MLX-LoRA → GGUF-LoRA conversion fidelity (rank-16, 24 layers — small, so
   byte-level review is cheap).
4. p50 latency on 4090 with ~600-token prompts (prefill-bound).

## Safety

- Production serving = release decision: needs human authorization.
- API keys live in env only; the demo proxy refuses to boot with keys in argv.
- No customer data in this demo path; logs are prompts+probs only, retention off.
