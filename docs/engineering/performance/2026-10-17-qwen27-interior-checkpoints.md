# Qwen27 interior prefill checkpoints

For `qwen3.8-27b-tensorfold`, lanes=1, retain at most one optional
interior checkpoint in a long final message. Editing text after that cut can
resume prefill there instead of processing the whole message again.

## Mechanism and scope

V1 keeps the reviewed `b3c948294` policy and lowers final-message eligibility
from two prefill steps to one (2048 tokens). The existing 2048-token prefill
grid is unchanged. The latest eligible grid cut before the native history
boundary is added only when native selections and store capacity allow it.
Short final messages below 2048 tokens retain ordinary behavior. Lanes>1 use
ordinary boundary checkpoints; no new flags or scheduler policy are introduced.

The optional state remains expendable: release it before native slot/byte
pressure or an incompatible conversation. Promote it only when native
selection without that optional hit would retain the same prefix, using
request history/shared-prefix context registered until stream cleanup.
Cancellation preserves runtime progress snapshots. The hook does not re-match
the store to change the scheduler's resume point or memory admission.

Two retention divergences are accepted, and are not repaired:

- **Resume-skip:** native selections between main's ordinary hit and the
  optional resumed cut are not computed when resuming from that cut.
- **LRU/last_prompt:** main's ordinary hit is not touched on an optional resume,
  so its eviction order and last-prompt metadata can differ later.

Both affect which checkpoints are retained later, never output tokens.
They can change future cache reuse under pressure. This is not a claim of
identical cache inventories for arbitrary request streams.

## Measured evidence

M3 Ultra (28 CPU / 60 GPU cores, 256 GiB, macOS 26.5.2): clean accepted
late-edit HTTP TTFT, seconds; edit near 88% of the final message:

| Final message | Main | V1 | Reused tokens in V1 |
| --- | ---: | ---: | ---: |
| ~3k | 8.98 | 3.12 | 2048 |
| ~4k | 11.81 | 5.97 | 2048 |
| ~6k | 17.90 | 6.02 | 4096 |
| ~8k | 23.88 | 5.84 | 6144 |

The ~1k/~2k late edits are unchanged (no cut). Cold, follow-up and ~4k
mid-edit reuse/behavior are unchanged; timings are comparable except the
first ~1k cold request includes compilation noise. Exact greedy output-token
parity: **57/57** accepted comparisons, including baseline self-comparisons.
Cumulative MLX peak: **20.12 → 20.38 GiB**. Maximum retained checkpoint
observations: 3 → 4 states, 1486.9 → 2117.7 MiB (including native states).

These are partial repeats: **main 1, V1 2**, rather than three clean runs each.
Accepted V1 late-edit repeats deviate by <0.5% from their medians. Contaminated
whole-server attempts were excluded after host checks; accepted requests had
quiet pre/post checks and no slow-round logs. The incomplete balanced sweep
limits confidence; this is one deterministic fixture family. The baseline was
`d84585a7c`, V1 was `b3c948294` plus the threshold change, so this compares
historical commits rather than a same-commit threshold ablation.

M4 Pro (48 GiB, macOS 26.5.1): ~8k late-edit median **113.8 → 24.8 s**,
0 → 6144 reused tokens, exact parity across three paired repeats. That run
used `946e82d596` versus `b3c948294`, before threshold lowering. A foreign VM
contended with inference; absolute timings are indicative only. The earlier
~4k fixture was below the old two-step threshold and gained no reuse.

A 1024-grid variant was rejected: it breaks exact token parity even on cold
requests. Do not lower the grid or prefill step.

## Reproduction

Evidence receipts are under `/private/tmp/tensorfold-qwen27-cuts-pilot/`:
`sweep3/REPORT.md`, `sweep3/method-receipt.json`, `sweep3/parity.json`,
`mzr001/REPORT.md` and `sweep/REPORT.md`. These are local scratch artifacts,
not permanent repository fixtures. The method receipt preserves the clean
sweep's harness adaptations; `sweep/build_workload.py` builds the original
16 cases and the clean sweep adds ~8k cold/late/follow-up from that builder.
Its workload SHA-256 is
`e0710c6cf8950771ee8444f196201c84fffe3fff79774e413704ccb784e43844`.
To repeat the historical sweep, restore the receipt's scripts/workload beside
`MAIN`/`V1` checkouts and independent `venv-MAIN`/`venv-V1` environments, then
run the preserved harness for each variant/repeat:

```sh
unset XDG_CACHE_HOME
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python http_probe_sweep.py MAIN on 0
python http_probe_sweep.py V1 on 0
```

Use fresh servers, alternating main/V1 order, evict the checkpoint store before
each size, and keep fixed assistant text in follow-ups. The clean harness
serializes launches with `lockf -k /var/tmp/rapid-mlx-large-model.lock`, rejects
foreign VM/model contention, and excludes an entire repeat on failed pre/post
host checks. The equivalent server command is:

```sh
lockf -k /var/tmp/rapid-mlx-large-model.lock \
  .venv/bin/rapid-mlx serve qwen3.8-27b-tensorfold \
  --host 127.0.0.1 --port 18800 --max-num-seqs 1 \
  --disable-disk-caches --disable-model-downloads --no-preflight \
  --context-length 16384 --served-model-name probe --reasoning-parser qwen3
```

Requests: streaming chat completions, temperature=0, max_tokens=64,
`chat_template_kwargs={"enable_thinking": false}`; TTFT ends at the first
nonempty content/reasoning delta and excludes startup. Lanes=1, store slots=8,
budget=16 GiB. M4 used up to 200 output tokens.

M3 environment: Python 3.12.14, Rapid-MLX 0.16.3, MLX 0.32.3,
MLX-LM 0.31.3, Transformers 5.15.1, huggingface-hub 1.33.0;
M4 used Python 3.12.15 with the same packages. Both used TensorFold 0.6.6 at
`cb2ebf0540f42604e2759b2ddef497861e928248`, target
`Vontra/Qwen3.8-27B-MLX-4bit` revision
`70ae7fac63274ff2eac54152031433374cb80f2f`, drafter
`z-lab/Qwen3.8-27B-DFlash2` revision
`50307d4c4cde6860d4eee73e2547cd786fe8e8a4`. Resolve weights from the default
HF cache; no alternate cache is required.

Focused model-free contracts cover the one-step threshold, native selection
and promotion, scheduler resume preservation, lane gating, context registry
cleanup, cancellation, and eviction safety. Run in an isolated Python 3.12
venv with `.[test]`, Ruff and the qualified runtime installed:

```sh
.venv/bin/ruff check rapid_mlx/speculative/tensorfold_qwen27*.py tests/test_tensorfold_qwen27*.py
.venv/bin/ruff format --check rapid_mlx/speculative/tensorfold_qwen27*.py tests/test_tensorfold_qwen27*.py
.venv/bin/python -m pytest tests/test_tensorfold_qwen27*.py
```

Branch convergence: 87 focused Qwen27 tests pass; Ruff check/format pass.
Changed-lines coverage against fresh main is 100% (96 executable lines, none
missing), with no coverage exemptions.

No new loaded-model measurements were made during branch convergence. Existing
parity evidence does not qualify sampling, other model pairs, multiple lanes,
or arbitrary long-running pressure workloads.
