# Qwen3.8-Flash-Next-4bit vs Qwen3.8-27B-4bit — sampled standard evals (2026-08-27)

Both runs complete on 2026-08-27 (Flash-Next 14:36–15:45Z, 68.5 min; 27B 15:46–16:30Z, 44.5 min).

## Setup (identical for both models)
- Harness: EleutherAI lm-evaluation-harness `lm_eval` 0.4.12, model type `local-chat-completions`, `--apply_chat_template`, `num_concurrent=1`, `tokenized_requests=False`, `temperature=0`, `--seed 1234`, `--limit N` = first N docs (no shuffle in these task configs).
- Server: PyPI rapid-mlx 0.13.1 (source 819db667), `HF_HUB_OFFLINE=1 rapid-mlx serve <alias> --host 127.0.0.1 --port 8464 --no-thinking` on Mac Studio M3 Ultra 256 GB, one model at a time.
- Thinking OFF for both (`--no-thinking` → `enable_thinking=false` in the chat template; 0 of all logged samples contain `<think>`).
- Artifacts: `rapid-mlx/Qwen3.8-Flash-Next-4bit` @ dcf657e4 (PLE q4-g32 / gates q8-g64 / rest q4-g64); `rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX` @ aa985c29 (affine q4-g64; speculative decoding OFF in this run).
- Files in this directory: `RUNLOG.md` (every command with timestamps), `results/<model>/<task>/` (harness results JSON + per-example samples JSONL), `rescore_gsm8k.py` (supplementary GSM8K re-score), `tasks/humaneval_chat/` (chat-safe HumanEval task definition). Paths inside RUNLOG/results refer to the scratch directory used on the day (`/private/tmp/atlas-flash-eval`).

## Headline table (harness numbers)

| Task | N | Flash-Next-4bit | 27B-4bit | Δ |
|---|---:|---:|---:|---:|
| MMLU-Redux 2.0, generative 0-shot (4 per subject × 57 subjects) | 228 | **86.8** ±2.1 | 83.3 ±2.3 | +3.5 |
| HumanEval instruct, pass@1 (chat-safe variant) | 100 | 96.0 ±2.0 | **98.0** ±1.4 | −2.0 |
| GSM8K 0-shot CoT, flexible-extract | 100 | **81.0** ±3.9 | 80.0 ±4.0 | +1.0 |
| GSM8K same samples, bold-aware re-score (supplementary) | 100 | **96.0** | 94.0 | +2.0 |
| IFEval prompt-level strict | 100 | **84.0** ±3.7 | 82.0 ±3.9 | +2.0 |
| IFEval instruction-level strict | 163 inst | **89.0** | 88.3 | +0.6 |
| IFEval prompt-level loose | 100 | **88.0** ±3.3 | 84.0 ±3.7 | +4.0 |
| IFEval instruction-level loose | 163 inst | **92.6** | 90.2 | +2.5 |

Read: on this sample the 4-bit Flash-Next matches or edges the 4-bit dense 27B on knowledge (MMLU-Redux), math (GSM8K) and instruction following (IFEval), and trails by 2 problems on HumanEval. Every Δ is inside the sampling error (±2–4 points at N=100), so the honest claim is "on par with the 27B sibling under an identical harness", not "better".

## Detail

**MMLU-Redux** (61 subject rows incl. groups): Flash 37 subjects at 4/4, 11 at 3/4; 27B 33 at 4/4, 12 at 3/4. Both weakest on college_mathematics (1/4), abstract_algebra, econometrics, professional_accounting, global_facts (2/4). Flash additionally 2/4 on college_physics, high_school_chemistry, college_medicine, human_aging; 27B additionally 2/4 on electrical_engineering, elementary_mathematics, high_school_physics, high_school_statistics, moral_scenarios, professional_law. Generative letter-answer format, `max_gen_toks=64`, ~1 s/example.

**HumanEval**: Flash misses HumanEval/32, /76, /84, /93; 27B misses /32, /93 (both share the same two). Stock `humaneval_instruct` scores 0 over chat completions because its inherited completion-style stop strings (`\ndef`, `\n#` …) truncate a chat reply at line 1 — a local variant `humaneval_instruct_chat` (same dataset, prompt and pass@1 metric; `until: []` + fenced-code-block extraction) was used for BOTH models. `HF_ALLOW_CODE_EVAL=1`, `max_gen_toks=1024`.

**GSM8K**: harness `flexible-extract` takes the LAST number in the reply; the models answer in bold and then restate context ("**$64** for the 16 glasses" → extracts 16; "**$26.00**" → 26.00 ≠ 26). Inspection of Flash's 19 harness misses: 15 are this extraction artifact, 4 real. A bold-aware re-score of the identical saved samples (prefer the last bold number, strip $ , and trailing .00) gives Flash 96.0 / 27B 94.0. `strict-match` is 0.0 for both (expects the literal "The answer is N." — meaningless for chat models). ~15–25 s/example.

**IFEval** per-instruction strict accuracy (Flash / 27B): change_case 18/19 · 19/19; combination 10/11 · 9/11; detectable_content 8/9 · 8/9; detectable_format 28/29 · 27/29; keywords 32/39 · 31/39; language 3/4 · 4/4; length_constraints 26/29 · 25/29; punctuation 10/12 · 11/12; startend 10/11 · 10/11. Keyword constraints are the common weak spot for both.

## Caveats (must accompany any publication)
1. Sampled: first N examples per task (N=100; MMLU-Redux 4/subject = 228). ± is the harness stderr; differences of 1–4 points are not significant.
2. Non-thinking mode, temperature 0, single run, greedy — upstream's published numbers are bf16 with thinking ON and are NOT comparable.
3. Two harness adaptations, applied identically to both models: chat-safe HumanEval variant; GSM8K bold-aware re-score reported beside (never instead of) the harness number.
4. 27B reference ran without speculative decoding; quality is unaffected by MTP anyway (lossless contract), speed is.
5. Both models are 4-bit; there is no bf16 baseline here (Flash bf16 = 335 GiB does not fit the 256 GB box). Quantization fidelity vs bf16 remains unmeasured — the comparison isolates "Flash-Next-4bit vs dense-27B-4bit", not "4bit vs bf16".

## Publication
- Published on the model card: https://huggingface.co/rapid-mlx/Qwen3.8-Flash-Next-4bit (section "Sampled standard evals").
- Suggested wording: "Under an identical harness (non-thinking, temp 0, first-100 sampling), the 4-bit Flash-Next scores on par with the 4-bit dense Qwen3.8-27B: MMLU-Redux 86.8 vs 83.3, HumanEval 96 vs 98, IFEval strict 84 vs 82, GSM8K 96 vs 94 (answer-aware scoring)."
