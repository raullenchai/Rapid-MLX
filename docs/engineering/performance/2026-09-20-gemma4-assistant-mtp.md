# Gemma 4 26B-A4B assistant-sidecar MTP qualification

Date: 2026-09-20

## Decision

Ship Gemma 4 assistant-sidecar MTP as an explicit, batch-size-one opt-in. Do
not automatically select an assistant and do not change ordinary multimodal
inference. Automatic draft-depth selection remains recommended because fixed
`K=3` is workload-dependent rather than a universal speedup.

Targets that use cross-layer K/V sharing remain fail-closed. Their cache holds
producer slots rather than one slot per decoder layer, and this qualification
does not establish the layer-to-producer mapping required by the assistant.

## Environment

- Mac mini, Apple M4 Pro, 48 GB unified memory
- macOS 26.5.1
- PR revision: `4e2355b739bf1cf59c2f1ac5ce92027517498fa9`
- Target: `unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit@ea6005b2a9b3dda91bcb26cb94a6ddf3a2eea4df`
- Assistant: `mlx-community/gemma-4-26B-A4B-it-assistant-bf16@cda74908f1dbe7d3dbd3030e66576a7d4094144f`
- Greedy decoding, thinking disabled, chat template enabled
- Eight coding, reasoning, structured-output, and prose prompts
- 160 generated tokens per arm, except an earlier stop token
- One loaded target/assistant pair shared by stock, `K=0`, and `K=3` arms
- Existing default Hugging Face cache; no cache override or model copy

## Correctness and stability gate

All 16 `K=0`/`K=3` arms completed. There was no fallback, stream corruption,
rollback failure, or process crash. `K=0` matched stock autoregressive output
on all eight prompts.

`K=3` matched `K=0` token-for-token on two prompts. Of the six first
divergences, five selected a target/non-draft token. One selected a draft token
that the batched target forward also chose. This is consistent with the
documented quantized numerical fork between single-token target decoding and
multi-token target verification. It is not an unverified-draft path: every
accepted proposal was checked by the target.

The output-equality observation is therefore diagnostic, not a product claim.
The shipping contract is target verification, complete output, intact rollback
state, and fail-closed unsupported shapes.

## Performance

Pooled decode timings exclude the first generated token from both token and
time totals.

| Mode | Decode tokens | Decode seconds | Decode throughput | Relative |
|---|---:|---:|---:|---:|
| Same-generator `K=0` | 1,155 | 19.954 | 57.88 tok/s | 1.000x |
| Fixed `K=3` | 1,155 | 17.208 | 67.12 tok/s | 1.160x |

Fixed `K=3` attempted 1,185 drafts and accepted 766, an aggregate 64.64%
acceptance rate. Individual prompts ranged from 50.41 to 86.14 tok/s under
`K=3`; the corresponding `K=0` measurements were approximately 57-58 tok/s.
High-acceptance coding and structured tasks improved substantially, while two
low-acceptance prose/reasoning tasks slowed down. This variance is why the
fixed depth remains a validation/operator control and automatic depth selection
is the serving recommendation.

## Supported server smoke

The supported HTTP server booted both pinned artifacts, attached the Gemma 4
dispatcher, and returned HTTP 200 for both `K=0` and `K=3`. The same short
Polish prompt completed with 39 generated tokens at 36.7 tok/s (`K=0`) and
37.6 tok/s (`K=3`). The `K=3` run recorded 15 verify calls and accepted
10, 7, and 5 drafts at depths one through three. Ordinary outer-wrapper
inference retained its existing call contract.

### Sliding-window rollover follow-up

Gemma 4 uses a 1,024-token rotating attention window on most decoder layers.
The initial qualification covered short prompts; a follow-up gate exercised
rejected MTP blocks after that ring had wrapped. The target cache now retains a
small rollback buffer outside the visible attention window, while cache layouts
that cannot prove rollback safety park at `K=0` before drafting.

On the same M4 Pro 48 GB host, with fixed `K=3`, prefix cache disabled, and the
same pinned target and assistant:

- a 9,269-token prompt plus 21 completion tokens returned HTTP 200; 18 drafts
  were attempted and 14 accepted, proving four rejected drafts rolled back
  after the sliding-window boundary;
- a 2,020-token prompt completed the full 700-token output budget with HTTP
  200; 525 drafts were attempted and 524 accepted;
- neither request logged a rollback-preflight failure, generator abort, or
  HTTP 503, and the server remained healthy after both requests.

The second prompt intentionally requested a long deterministic integer list;
it is a stability workload, not a quality or universal throughput claim.

## Reproduction

From the exact PR revision above, with both pinned artifacts present in the
default cache:

```bash
python bench/repro_mtp_forced_k_parity.py \
  --model unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit \
  --sidecar mlx-community/gemma-4-26B-A4B-it-assistant-bf16 \
  --k-values 0,3 \
  --chat-template \
  --max-tokens 160 \
  --output /private/tmp/gemma4-mtp-report.json
```

The JSON report records token hashes, first-divergence classification, decode
time, decode throughput, draft attempts, acceptances, verify calls, completion,
and termination for each prompt. The qualification artifact SHA-256 was
`d18192089d78213c36bfab32837880e2bf9df968cb6c275f46a7fd94de960d19`.

## Limitations

- This qualification covers batch size one only.
- It covers targets with one cache slot per decoder layer; shared-K/V E2B/E4B
  layouts remain ineligible.
- It does not qualify continuous batching, sampled decoding, automatic model
  recommendation, or automatic assistant discovery.
- Fixed `K=3` can regress low-acceptance workloads; the aggregate improvement
  must not be presented as a per-request guarantee.
- The assistant consumes additional unified memory and was qualified on a
  48 GB machine, not on lower-memory Macs.
