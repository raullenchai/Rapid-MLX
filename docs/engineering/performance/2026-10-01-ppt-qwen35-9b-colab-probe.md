# Qwen3.5-9B Parallel Power Tempering: Colab feasibility probe

Date: 2026-10-01 (Pacific). Owner: Vector, with Atlas handling product decisions.

## Scope

[Explore Broadly, Reason Sharply](https://arxiv.org/abs/2609.38104) proposes Parallel Power Tempering (PPT). As of this probe, the paper links no author implementation. Its Appendix B specifies the local Metropolis–Hastings (MH) proposal and adjacent replica exchange; Appendix G lists the experiment configuration. We implemented those moves independently in [the standalone probe](../../../bench/repro_ppt_qwen35_9b_colab.py). This is a mechanics and cost check, **not** a replication of the paper's Qwen3.5-9B benchmark table.

## Environment and protocol

- Google Colab A100-SXM4-40GB, PyTorch 2.11.0+cu130, Transformers 5.17.0.
- Unquantized `Qwen/Qwen3.5-9B` in bfloat16, checkpoint revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a8b`.
- First five `openai/gsm8k` test rows (offset 0), retrieved from the Hugging Face datasets-server API. Prompt: original question followed by `Solve concisely and put only the final numeric answer in \boxed{}.`; Qwen chat template, thinking disabled.
- Baseline: temperature 1, top-p 1, top-k disabled, one completion. PPT: three powers `{1.25, 1.4, 1.6}`, one fixed-horizon local MH round per chain, then one ordered adjacent swap sweep, returning the highest-power chain. Completion horizon 512. Sampling seeds: `1000 + row_index` for the baseline, `2000 + row_index` for PPT.
- Scoring: last boxed integer, or an output that is entirely an integer, compared to the GSM8K final integer after removing commas. No verifier or best-of-N selector.
- The implementation calls Transformers `generate` separately for each proposal and scores generated tokens from the returned raw logits. It does not batch replicas or reuse KV across proposals. Its elapsed times should not be compared with the paper's optimized vLLM/H100 times.
- The baseline uses the same score-recording path, including full-vocabulary log-probability and normalizer calculations it does not need for ordinary inference. The time ratio measures these two instrumented paths; it is not a production request latency ratio.

The exact paper setup for 9B is substantially larger: AIME 24&25 uses three chains with powers `{1.25, 1.4, 1.6}`, horizon **65,536**, and five local MH updates; GPQA and LiveCodeBench use horizon **131,072** and other powers. On AIME 2024 item 0, ordinary sampling reached the 1,024-token cap without a final answer. A short-cap AIME score would be misleading.

## Results

| GSM8K row | Gold | Baseline final | PPT final | Baseline s | PPT s | PPT local proposals | Accepted swaps |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 18 | 18 | 18 | 7.74 | 21.27 | 1 | 0 |
| 1 | 3 | 3 | 2.5 | 4.25 | 10.86 | 1 | 0 |
| 2 | 70000 | 70000 | 70000 | 16.24 | 123.77 | 3 | 2 |
| 3 | 540 | 810 | 540 | 0.77 | 10.01 | 0 | 2 |
| 4 | 20 | 20 | 20 | 0.23 | 25.76 | 1 | 2 |

Final-number accuracy: **4/5 for both**. Summed per-item instrumented time: **29.23 seconds baseline vs 191.67 seconds PPT (6.56×)**. Measured peak PyTorch allocated GPU memory was about 17.7–18.0 GiB per item, including the 17.53 GiB model weights.

On row 3, before swaps the highest-power chain predicted 2520; the other two predicted 540. No local proposals occurred because every uniformly chosen restart position was after a chain's EOS. The two accepted swaps moved a correct trace to the output chain. Row 1 moved the other way: the PPT output was `2.5`, versus the baseline's correct 3. Five items cannot estimate an accuracy difference. Short completions also leave most of the 512 restart positions after EOS, so one MH round gives little exploration.

## Reproduce

From a checkout with `google-colab-cli` authenticated to an account with an A100 entitlement:

```bash
colab run --gpu A100 -s ppt-qwen9b bench/repro_ppt_qwen35_9b_colab.py \
  --offset 0 --limit 5 --horizon 512 --rounds 1 --powers 1.25 1.4 1.6 \
  > ppt-gsm8k-results.jsonl
```

`colab run` releases its new VM after the script finishes. The script prints a configuration record and one JSON record per item with full outputs, token counts, timing, and accepted moves. It does not change Rapid-MLX inference code. The probe session was stopped; an unrelated existing Colab session was untouched.

A fresh A100 session also ran the standalone script with `--limit 1` successfully and self-terminated. It reproduced row 0's answers (18 for both methods); the independent run measured 9.65 seconds baseline and 22.19 seconds PPT. Exact latency varies by session. Transformers reported that `causal_conv1d` and `flash-linear-attention` were absent, so their fallback kernels add overhead.

## Next evidence needed

For a claim about the paper's reported 9B improvement, obtain author code or validate this implementation against it, then run the original AIME, GPQA, and LiveCodeBench sets with the Appendix G horizons, prompt templates, grading, and a compute-matched single-chain plus an uncoupled-chain control. GPQA Diamond requires dataset access approval. Record exact checkpoint/dataset revisions, seeds, per-item outputs, generated-token budgets, elapsed time, and peak memory. Keep an A100 result separate from the paper's H100 numbers.
