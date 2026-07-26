# rapid-mlx Model Scorecard

*Auto-generated on 2026-03-05 00:32 UTC*

> **Tested on**: Apple M3 Ultra (256GB)
>
> **Methodology**: All suites use `enable_thinking: false`. Cache cleared between suites. See [README](README.md) for details.

## Comparison Table

| Model | Quant | RAM | TTFT | Decode (s) | Decode (l) | Tools | Coding | Reasoning | General | Avg | Date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Devstral-Small-2-4bit | 4bit | 13.4 GB | 291ms | 22.5 t/s | 47.2 t/s | 17% | 90% | 70% | 70% | 62% | 2026-03-04 |
| GLM-4.5-Air-4bit | 4bit | 60.3 GB | 708ms | 27.5 t/s | 53.6 t/s | 73% | 90% | 70% | 80% | 78% | 2026-03-04 |
| GLM-4.7-Flash-8bit | 8bit | 31.9 GB | 362ms | 33.4 t/s | 57.2 t/s | 73% | 100% | 90% | 50% | 78% | 2026-03-04 |
| GPT-OSS-20B-mxfp4-q8 | mxfp4-q8 | 12.1 GB | 339ms | 85.7 t/s | 124.2 t/s | 80% | 20% | 60% | 90% | 62% | 2026-03-05 |
| Hermes-3-Llama-3.1-8B-4bit | 4bit | 4.6 GB | 152ms | 69.3 t/s | 122.9 t/s | 17% | 20% | 30% | 40% | 27% | 2026-03-04 |
| MiniMax-M2.5-4bit | 4bit | 128.9 GB | 1.3s | 46.2 t/s | 49.9 t/s | 87% | 10% | 80% | 90% | 67% | 2026-03-04 |
| Mistral-Small-3.2-4bit | 4bit | 13.4 GB | 1.1s | 27.6 t/s | 47.2 t/s | 17% | 80% | 60% | 60% | 54% | 2026-03-04 |
| Qwen3-0.6B-4bit | 4bit | 0.4 GB | 94ms | 78.3 t/s | 364.7 t/s | 30% | 20% | 20% | 30% | 25% | 2026-03-04 |
| Qwen3-Coder-Next-4bit | 4bit | 44.9 GB | 473ms | 41.5 t/s | 73.5 t/s | 90% | 90% | 70% | 70% | 80% | 2026-03-04 |
| Qwen3-Coder-Next-6bit | 6bit | 64.8 GB | 642ms | 34.6 t/s | 65.6 t/s | 87% | 90% | 80% | 70% | 82% | 2026-03-04 |
| Qwen3.5-122B-A10B-8bit | 8bit | 129.8 GB | 1.3s | 19.4 t/s | 42.7 t/s | 87% | 90% | 90% | 90% | 89% | 2026-03-04 |
| Qwen3.5-122B-A10B-mxfp4 | mxfp4 | 65.0 GB | 714ms | 26.3 t/s | 57 t/s | 90% | 90% | 80% | 90% | 88% | 2026-03-04 |
| Qwen3.5-27B-4bit | 4bit | 15.3 GB | 453ms | 17.7 t/s | 37.7 t/s | 83% | 90% | 50% | 80% | 76% | 2026-03-04 |
| Qwen3.5-35B-A3B-4bit | 4bit | 19.6 GB | 322ms | 31.7 t/s | 95.2 t/s | 87% | 90% | 50% | 70% | 74% | 2026-03-04 |
| Qwen3.5-35B-A3B-8bit | 8bit | 36.9 GB | 456ms | 32.4 t/s | 80 t/s | 90% | 90% | 80% | 80% | 85% | 2026-03-04 |
| Qwen3.5-4B-4bit | 4bit | 2.4 GB | 196ms | 43 t/s | 157.6 t/s | 73% | 50% | 50% | 50% | 56% | 2026-03-04 |
| Qwen3.5-9B-4bit | 4bit | 5.1 GB | 228ms | 35.4 t/s | 106.4 t/s | 83% | 70% | 60% | 70% | 71% | 2026-03-04 |

## Details

### Devstral-Small-2-4bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-04
- **TTFT**: cold=291ms, warm=106ms
- **Decode**: short=22.5 t/s, long=47.2 t/s
- **RAM**: active=13.4 GB, peak=13.4 GB
- **Tool Calling**: 17% (5/30)
- **Coding**: 90% (9/10)
- **Reasoning**: 70% (7/10)
- **General**: 70% (7/10)
- **Eval time**: 323.4s

### GLM-4.5-Air-4bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: glm47
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser glm47`
- **Date**: 2026-03-04
- **TTFT**: cold=708ms, warm=107ms
- **Decode**: short=27.5 t/s, long=53.6 t/s
- **RAM**: active=60.3 GB, peak=60.3 GB
- **Tool Calling**: 73% (22/30)
- **Coding**: 90% (9/10)
- **Reasoning**: 70% (7/10)
- **General**: 80% (8/10)
- **Eval time**: 305.5s

### GLM-4.7-Flash-8bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: glm47
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser glm47`
- **Date**: 2026-03-04
- **TTFT**: cold=362ms, warm=113ms
- **Decode**: short=33.4 t/s, long=57.2 t/s
- **RAM**: active=31.9 GB, peak=31.9 GB
- **Tool Calling**: 73% (22/30)
- **Coding**: 100% (10/10)
- **Reasoning**: 90% (9/10)
- **General**: 50% (5/10)
- **Eval time**: 230.6s

### GPT-OSS-20B-mxfp4-q8

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: harmony
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser harmony`
- **Date**: 2026-03-05
- **TTFT**: cold=339ms, warm=112ms
- **Decode**: short=85.7 t/s, long=124.2 t/s
- **RAM**: active=12.1 GB, peak=12.6 GB
- **Tool Calling**: 80% (24/30)
- **Coding**: 20% (2/10)
- **Reasoning**: 60% (6/10)
- **General**: 90% (9/10)
- **Eval time**: 197.6s

### Hermes-3-Llama-3.1-8B-4bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-04
- **TTFT**: cold=152ms, warm=72ms
- **Decode**: short=69.3 t/s, long=122.9 t/s
- **RAM**: active=4.6 GB, peak=4.7 GB
- **Tool Calling**: 17% (5/30)
- **Coding**: 20% (2/10)
- **Reasoning**: 30% (3/10)
- **General**: 40% (4/10)
- **Eval time**: 111.3s

### MiniMax-M2.5-4bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: minimax
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser minimax`
- **Date**: 2026-03-04
- **TTFT**: cold=1.3s, warm=136ms
- **Decode**: short=46.2 t/s, long=49.9 t/s
- **RAM**: active=128.9 GB, peak=128.9 GB
- **Tool Calling**: 87% (26/30)
- **Coding**: 10% (1/10)
- **Reasoning**: 80% (8/10)
- **General**: 90% (9/10)
- **Eval time**: 610.7s

### Mistral-Small-3.2-4bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-04
- **TTFT**: cold=1.1s, warm=104ms
- **Decode**: short=27.6 t/s, long=47.2 t/s
- **RAM**: active=13.4 GB, peak=13.7 GB
- **Tool Calling**: 17% (5/30)
- **Coding**: 80% (8/10)
- **Reasoning**: 60% (6/10)
- **General**: 60% (6/10)
- **Eval time**: 369.5s

### Qwen3-0.6B-4bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-04
- **TTFT**: cold=94ms, warm=73ms
- **Decode**: short=78.3 t/s, long=364.7 t/s
- **RAM**: active=0.4 GB, peak=0.4 GB
- **Tool Calling**: 30% (9/30)
- **Coding**: 20% (2/10)
- **Reasoning**: 20% (2/10)
- **General**: 30% (3/10)
- **Eval time**: 38.8s

### Qwen3-Coder-Next-4bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-04
- **TTFT**: cold=473ms, warm=27ms
- **Decode**: short=41.5 t/s, long=73.5 t/s
- **RAM**: active=44.9 GB, peak=45.0 GB
- **Tool Calling**: 90% (27/30)
- **Coding**: 90% (9/10)
- **Reasoning**: 70% (7/10)
- **General**: 70% (7/10)
- **Eval time**: 218.9s

### Qwen3-Coder-Next-6bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-04
- **TTFT**: cold=642ms, warm=29ms
- **Decode**: short=34.6 t/s, long=65.6 t/s
- **RAM**: active=64.8 GB, peak=64.8 GB
- **Tool Calling**: 87% (26/30)
- **Coding**: 90% (9/10)
- **Reasoning**: 80% (8/10)
- **General**: 70% (7/10)
- **Eval time**: 250.8s

### Qwen3.5-122B-A10B-8bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-04
- **TTFT**: cold=1.3s, warm=32ms
- **Decode**: short=19.4 t/s, long=42.7 t/s
- **RAM**: active=129.8 GB, peak=129.9 GB
- **Tool Calling**: 87% (26/30)
- **Coding**: 90% (9/10)
- **Reasoning**: 90% (9/10)
- **General**: 90% (9/10)
- **Eval time**: 342.5s

### Qwen3.5-122B-A10B-mxfp4

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-04
- **TTFT**: cold=714ms, warm=27ms
- **Decode**: short=26.3 t/s, long=57 t/s
- **RAM**: active=65.0 GB, peak=65.1 GB
- **Tool Calling**: 90% (27/30)
- **Coding**: 90% (9/10)
- **Reasoning**: 80% (8/10)
- **General**: 90% (9/10)
- **Eval time**: 261.5s

### Qwen3.5-27B-4bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-04
- **TTFT**: cold=453ms, warm=29ms
- **Decode**: short=17.7 t/s, long=37.7 t/s
- **RAM**: active=15.3 GB, peak=15.4 GB
- **Tool Calling**: 83% (25/30)
- **Coding**: 90% (9/10)
- **Reasoning**: 50% (5/10)
- **General**: 80% (8/10)
- **Eval time**: 451.7s

### Qwen3.5-35B-A3B-4bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-04
- **TTFT**: cold=322ms, warm=33ms
- **Decode**: short=31.7 t/s, long=95.2 t/s
- **RAM**: active=19.6 GB, peak=19.6 GB
- **Tool Calling**: 87% (26/30)
- **Coding**: 90% (9/10)
- **Reasoning**: 50% (5/10)
- **General**: 70% (7/10)
- **Eval time**: 168.2s

### Qwen3.5-35B-A3B-8bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-04
- **TTFT**: cold=456ms, warm=30ms
- **Decode**: short=32.4 t/s, long=80 t/s
- **RAM**: active=36.9 GB, peak=36.9 GB
- **Tool Calling**: 90% (27/30)
- **Coding**: 90% (9/10)
- **Reasoning**: 80% (8/10)
- **General**: 80% (8/10)
- **Eval time**: 186.0s

### Qwen3.5-4B-4bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-04
- **TTFT**: cold=196ms, warm=29ms
- **Decode**: short=43 t/s, long=157.6 t/s
- **RAM**: active=2.4 GB, peak=2.5 GB
- **Tool Calling**: 73% (22/30)
- **Coding**: 50% (5/10)
- **Reasoning**: 50% (5/10)
- **General**: 50% (5/10)
- **Eval time**: 111.8s

### Qwen3.5-9B-4bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-04
- **TTFT**: cold=228ms, warm=28ms
- **Decode**: short=35.4 t/s, long=106.4 t/s
- **RAM**: active=5.1 GB, peak=5.2 GB
- **Tool Calling**: 83% (25/30)
- **Coding**: 70% (7/10)
- **Reasoning**: 60% (6/10)
- **General**: 70% (7/10)
- **Eval time**: 179.8s

---

## How to Add Your Results

1. Start rapid-mlx with your model: `rapid-mlx serve <model> --port 8000`
2. Run the eval: `python evals/run_eval.py --model "<model-name>" --quantization <quant>`
3. Your results are saved to `evals/results/<model>.json`
4. Regenerate this table: `python evals/generate_scorecard.py`
5. Submit a PR with your JSON file!

