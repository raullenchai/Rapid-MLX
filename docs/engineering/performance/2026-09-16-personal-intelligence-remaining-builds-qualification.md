# Personal Intelligence remaining-build qualification

Date: 2026-09-16

## Scope and rule

This run completes the remaining September Personal Intelligence matrix. A
build is product-qualified only when the exact public identity, backing
artifact, live tool parser, and harness profile pass all five behavior cases
across seeds `11`, `22`, and `33` (15/15). A borderline model is never admitted
because it nearly passes; family resemblance, parser capability, and a lower
quantization's receipt are not evidence.

All runs used the same Mac Studio checkout at source revision
`54c72e437e33d98643ff449197468f01a0f9337c`, port `18950`, and the reproduction
metadata embedded in each JSON receipt. MiniCPM5-2B Q8 also required a
small qualification-table fix: the short alias must be treated as a known
MiniCPM catalog identity so that the public alias resolves to the exact Q8
record instead of falling back to the conservative local profile.

## Qualified builds

| Model | Qualification ID | Parser | Receipt |
|---|---|---|---|
| Qwen3.5-4B Q8 | `qwen3.5-4b-q8-v1` | `hermes` | 15/15 |
| Qwen3.5-9B Q8 | `qwen3.5-9b-q8-v1` | `hermes` | 15/15 |
| Qwen3.6-35B Q4 | `qwen3.6-35b-q4-v1` | `qwen3_coder_xml` | 15/15 |
| Qwen3.6-27B Q4 | `qwen3.6-27b-q4-v1` | `qwen3_coder_xml` | 15/15 |
| Qwen3.8-27B upstream Q4 | `qwen3.8-27b-upstream-q4-v1` | `qwen3_coder_xml` | 15/15 |
| Qwen3.8-27B Rapid MTP Q4 | `qwen3.8-27b-rapid-mtp-q4-v1` | `qwen3_coder_xml` | 15/15 |
| Qwen3.8-27B mixed 3.5 BPW | `qwen3.8-27b-mixed-3.5bpw-v1` | `qwen3_coder_xml` | 15/15 |
| Qwen3.8-27B FP16 MTP | `qwen3.8-27b-fp16-mtp-v1` | `qwen3_coder_xml` | 15/15 |
| Bonsai 27B 2-bit | `bonsai-27b-2bit-v1` | `hermes` | 15/15 |
| Ling 3.0 Tiny Q4 | `ling-3.0-tiny-4bit-v1` | `glm47` | 15/15 |
| MiniCPM5-2B Q8 | `minicpm5-2b-q8-v1` | `minicpm` | 15/15 |

The Qwen3.8 upstream checkpoint must be served with both
`--enable-auto-tool-choice` and `--tool-call-parser qwen3_coder_xml`. With the
parser flag alone, the tool parser remains disabled and the model card
correctly stays unqualified.

Evidence files:

- `reports/benchmarks/personal-intelligence-qwen3.5-4b-8bit.json`
- `reports/benchmarks/personal-intelligence-qwen3.5-9b-8bit.json`
- `reports/benchmarks/personal-intelligence-qwen3.6-35b-4bit.json`
- `reports/benchmarks/personal-intelligence-qwen3.6-27b-4bit.json`
- `reports/benchmarks/personal-intelligence-qwen3.8-27b-upstream-q4.json`
- `reports/benchmarks/personal-intelligence-qwen3.8-27b-4bit.json`
- `reports/benchmarks/personal-intelligence-qwen3.8-27b-mixed-3.5bpw.json`
- `reports/benchmarks/personal-intelligence-qwen3.8-27b-4bit-fp16.json`
- `reports/benchmarks/personal-intelligence-bonsai-27b-2bit.json`
- `reports/benchmarks/personal-intelligence-ling-3.0-tiny-4bit.json`
- `reports/benchmarks/personal-intelligence-minicpm5-2b-8bit.json`

## Failed builds, held out

Two builds repeatedly passed 14/15 and remain disabled. Their committed
receipts preserve the full behavior results, but the model-card identity fields
and qualification check were annotated after the run because those runs
temporarily observed candidate records that this PR deliberately does not
ship. Each annotated receipt carries a `review_note` explaining that edit.

| Model | Receipt | Stable failure |
|---|---|---|
| Qwen3-Coder 30B Q4 | `reports/benchmarks/personal-intelligence-qwen3-coder-30b-4bit.json` | Seed 33 emits `Rrapid-MLX` in the source URL for the open-ended search/browse case. Seed 22 repeated the same typo on the retry; seed 11 passed. |
| GPT-OSS 20B MXFP4 Q8 | `reports/benchmarks/personal-intelligence-gpt-oss-20b.json` | Seed 33 ends the open-ended search/browse case with a Harmony citation (`【2†source】`) instead of the required source URL. Seeds 11 and 22 passed. |

Both failures are model-output instability on the non-“only” search/browse
task, not parser registration or harness-profile mismatches. Their records are
deliberately absent from `PERSONAL_INTELLIGENCE_QUALIFICATIONS`; adding a
record before a 15/15 exact-build receipt would break the fail-closed product
contract.

## Follow-up

Qwen3-Coder 30B and GPT-OSS 20B need another full canonical matrix after the
upstream model or serving prompt behavior changes. Any future retry must use
the same exact identity/backing/parser/profile checks and cannot reuse these
14/15 receipts as qualification evidence.
