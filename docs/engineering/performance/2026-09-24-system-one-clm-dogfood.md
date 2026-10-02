# System One CLM-8B server dogfood

Date: 2026-09-24

## Result

The native CLM integration converted the official PyTorch head, loaded the
original BF16 Qwen3-8B encoder, and completed authenticated requests through
`/v1/systemone` and `/v1/rank`. CPU selection, vector caching, clean shutdown,
and restart from the converted artifact worked.

The official README example selected `billing` with probability `0.9721`; the
published upstream example reports `0.9388`. A tides ranking put the Moon's
gravitational pull first with probability `0.9987`. A credential-phishing
question returned `P(true)=0.7052`, and a backup precaution returned
`P(true)=0.9804`.

The small qualitative screen also exposed weak zero-shot decisions. An
application-crash ticket selected `billing` over `technical` (0.5062 versus
0.3982), and an SQL-injection repair ranking put retrying the query ahead of a
parameterized query (0.6958 versus 0.2586). Adding an irrelevant sales option
to a duplicate-charge routing question also changed the winner from billing
to sales. This is a product-path dogfood screen, not a benchmark. It supports
an experimental CLM server claim, but not a broad accuracy or calibrated
probability claim.

Dogfood found and fixed two launch-path defects:

- the top-level CLI treated `clm-latest` as a generative-model alias and
  rejected the documented command before System One dispatch;
- `--device cpu` was silently ignored for CLM, leaving MLX on its default GPU.

## Environment

- Machine: Apple M3 Ultra, 256 GB unified memory
- macOS: 26.5.2
- Python: 3.12.13
- MLX: 0.32.2
- Rapid-MLX base revision: `576555d18`
- Encoder: `Qwen/Qwen3-8B`, BF16
- Encoder revision: `b968826d9c46dd6066d109eabc6255188de91218`
- Head: `Contrastive-LM/CLM-v0.1-8B`, `CLM_v0.1-8B.pt`
- Head revision: `87655cb835bd76fd66c2da78e1e3709f7fa11a94`
- Upstream CLM source used for comparison: `7956937c58ed5839c06ddc4dc6b6b61c3a3e4094`
- Device: CPU, to avoid contending with an unrelated active GPU workload
- Encoder payload: 15 GiB in the standard Hugging Face cache
- Converted head: 72 MiB safetensors plus JSON configuration
- Peak RSS in the direct cold-load probe: 16,296 MiB

The 32 GB Mac mini was inspected but not used. It had only 27 GiB free disk
and was already serving a 27B model with a 25 GB resident-memory limit. No
process was stopped and no cache was deleted.

## Conversion and numerical parity

The converter accepted the released checkpoint without key or shape changes.
For three real Qwen3 hidden-state vectors, the converted MLX projection heads
were compared directly with the upstream PyTorch `HeadPair`:

| Comparison | Maximum absolute error | Mean absolute error |
| --- | ---: | ---: |
| State projection | 1.49e-7 | 1.21e-8 |
| Action projection | 8.94e-8 | 1.01e-8 |

The two billing-example logits were `30.7411079, 27.1910839` in PyTorch and
`30.7411098, 27.1910839` in MLX. This validates checkpoint conversion and head
math for identical encoder hidden states. It does not replace an end-to-end
comparison against the upstream vLLM pooling server.

## Timing

All request times are loopback wall times on CPU. Cache hits contain no encoder
work; the cache is process-local and starts empty after restart.

| Operation | Observed time |
| --- | ---: |
| Backend construction, direct probe | 5.57 s |
| First three-option direct request | 3.04 s |
| Identical direct cache hit | 0.30 ms |
| First official three-question HTTP request | 3.42 s |
| Identical HTTP cache hit, server time | 0.66 ms |
| Cold three-candidate tides rank | 1.24 s |
| Restart to `/health` ready | 7.24 s |
| First three-question request after restart | 5.52 s |

For the repeated official request, `usage.input_tokens` fell from 98 to zero
while `requested_tokens` remained 98 and every answer remained identical.

## Reproduction

Downloads used the configured shared Hugging Face cache. No alternate
`HF_HOME`, `cache_dir`, or local model directory was created.

```bash
hf download Contrastive-LM/CLM-v0.1-8B CLM_v0.1-8B.pt \
  --revision 87655cb835bd76fd66c2da78e1e3709f7fa11a94
hf download Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218

rapid-mlx-convert-clm-head /path/to/CLM_v0.1-8B.pt \
  /private/tmp/rapid-mlx-clm-dogfood/head

rapid-mlx system-one clm-latest \
  --backend clm \
  --encoder /path/to/Qwen3-8B/snapshot \
  --head /private/tmp/rapid-mlx-clm-dogfood/head \
  --device cpu \
  --api-key dogfood-secret \
  --port 18710
```

## Limits found

- CPU cache misses take roughly one encoder pass per fresh state or candidate;
  the current implementation deliberately uses padding-free serial passes.
- Warm repeated decisions are sub-millisecond because projected state and
  action vectors are cached separately.
- The released zero-shot head can be confidently wrong on straightforward
  software and routing decisions. Candidate wording and candidate-set changes
  can materially alter the winner.
- Projection-head parity is measured. Full end-to-end probability drift versus
  the upstream vLLM last-token pooling runtime remains to be measured on an
  NVIDIA host.
