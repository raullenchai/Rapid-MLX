# Hybrid checkpoint qualification on M5 Max

This campaign measures the existing recurrent-state checkpoint feature for
long-document edits. It adds a reproducible HTTP benchmark and distinct
incremental and cold-equivalence gates. It changes no serving code or defaults.

## Environment and scope

- Date: 2026-10-09. Apple M5 Max, 36 GiB unified memory, macOS 27.0.1 (26A434).
- Tested checkout: `fcbd6d83dc9648bef9a6afc36af04e586372e3ef`, based on
  `e44e9598a065962f91f9a13b43959de7c564877f`; clean worktree during measurement.
- Python 3.12.13; MLX 0.32.3; MLX-LM 0.31.3; native vision runtime 0.7.2;
  Transformers 5.15.1; NumPy 2.5.3; HTTPX 0.28.1.
- Model: `mlx-community/Qwen3.6-35B-A3B-4bit`, immutable snapshot
  `38740b847e4cb78f352aba30aa41c76e08e6eb46`, reused from the default HF cache.
- Text-only loopback service; one child process at a time, no concurrent,
  sampled, vision, disk-cache or speculative workloads. No weight downloads.

The benchmark pins 2048-token prefill chunks and checkpoint stride, compares
maximum checkpoint counts 0 and 4, and tests stock and blocked GDN prefill
separately. Fused GDN decode, compiled decode, host prompt caching and
speculation are disabled. Optional disk caches and automatic prefix-cache
restore/save are disabled for the child servers.

## Workload and contracts

Three seeded documents contain about 6.5K prompt tokens each. Edits replace
section 60, 35 or 0 in a 70-section document. Each case executes:

1. Clear the reusable prefix cache and generate from the original document.
2. Generate from the edited document with that request history (warm).
3. Clear the cache and generate from the edited document again (cold).

Every measured response uses greedy decoding and a 32-token output cap.
The warmup may terminate at EOS; measured responses must expose 32 completion
tokens, `length`, usage and `[DONE]`. Output comparisons hash the decoded
content/reasoning pair and also compare prompt/completion counts and finish
reason. These are HTTP byte-output comparisons, not direct token-ID checks.

Four serial arms run stock off/on, then blocked on/off. This reverses
checkpoint order between prefill modes. Each mode uses one fixed process order;
there is no within-mode counterbalancing or randomized process order, so thermal
or process-order effects may bias these observed ratios. Repeat each mode in
both orders before treating the ratios as a controlled causal speedup estimate.
The summary joins cases by prefill mode, document round and edit location;
missing/duplicate rows, incomplete streams, invalid timings or usage, failed
server arms and unexpected cached-token counts cannot qualify a campaign.

Two contracts are explicit:

- **Incremental:** seed, warm and cold outputs each match between checkpoint
  on/off for the same prefill mode and history. Late/middle edits report
  4096/2048 resumed tokens with checkpoints, versus zero without them. Head
  edits, seed requests and cleared-cold requests report zero. Each of the
  four late/middle groups must meet the configured median paired speedup.
- **Cold:** requires the incremental contract and matching warm/cold output
  within every arm. This stronger contract is the CLI default.

`--contract incremental` selects the narrower condition explicitly; its
summary still reports every cold-equivalence failure. Neither contract
compares stock versus blocked output or certifies answer quality.

## Results

| Prefill | Edit | Checkpoints off: warm TTFT | On: warm TTFT | Median paired gain | Resumed tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| Stock | Late | 1.671 s | 0.763 s | 2.1888x | 4096 |
| Stock | Middle | 1.662 s | 1.215 s | 1.3669x | 2048 |
| Blocked | Late | 1.560 s | 0.725 s | 2.1519x | 4096 |
| Blocked | Middle | 1.549 s | 1.139 s | 1.3603x | 2048 |

All 12 late/middle pairs improved. Head edits reported zero cached tokens
and approximately 1x gain (stock 0.9998x, blocked 0.9995x).

The **18/18 checkpoint-on/off warm comparisons matched**; their corresponding
seed and cold comparisons also matched (54/54 phase comparisons). Expected
cache receipts passed for all 36 rows. Retained child-log excerpts confirm
six snaps to 4096 and six to 2048 across the checkpoint-on arms; blocked
prefill installation and stock disablement were also logged.

The historical summary passed its 1.1x threshold in every late/middle
group under its fixed-order rules. The current harness rejects performance
qualification of this schema-1 matrix because it lacks both orders per mode. The stronger cold contract **failed: only 26/36 warm/cold comparisons
matched**, including failures with checkpoints disabled and zero resumed
tokens. Re-evaluating the same receipts under the cold contract returns
`passed: false`. These results describe the incremental observations under this
request history; they do not establish a controlled causal gain or lossless
cold/warm equivalence.

The measured matrix and recomputed summary are retained in
`fixtures/m5-checkpoints-2026-10-09/`. Filesystem paths are normalized; hashes,
timings, usage and tested-source provenance are preserved.

## Segmentation control

A separate historical-source diagnostic at
`5c843788007f91431321fb6953688a2fe44ca10b` investigated warm/cold drift with
checkpoints disabled. It used the same documents, edit cases and cache-clear
sequence, adding a second cleared-cold request per row. Both diagnostic
launchers import `Scheduler` before entering the CLI, controlling import
order. The intervention alone replaces `_shared_prefix_local_split` so it
clears `shared_prefix_snapshot_at` and returns `None`.

| Prefill | Shared-prefix split | Warm/cold exact | Cold/repeated cold exact |
| --- | --- | ---: | ---: |
| Stock | Normal | 6/9 | 9/9 |
| Stock | Disabled in diagnostic child | 9/9 | 9/9 |
| Blocked | Disabled in diagnostic child | 9/9 | 9/9 |
| Blocked | Normal | 7/9 | 9/9 |

All edited requests in these controls reported zero cached tokens. The
normal warm path can still split prefill to snapshot a future shared prefix;
the first late-edit stock case recorded a snapshot at 5568 tokens. Zero
cached tokens therefore does not imply the same prefill segmentation as a
cleared-cold execution. The full 36-row control is retained in
`segmentation-control.json`, including child launch commands.

These controls support segmentation as the cause of the five unique
checkpoint-off warm/cold mismatches in this historical matrix. They do not
identify a faulty arithmetic operation, establish acceptable quality, or
justify disabling shared-prefix snapshots in production. The intervention
also removes future reusable snapshots. It is a diagnostic, not a proposed
runtime change or cold-equivalence certification of the current checkout.

## Reproduction

Use the tested checkout and dependency versions above, with the immutable
model already in the default cache. Choose an unused loopback port and a new,
empty campaign directory. The script owns and reaps only its child servers;
an occupied port fails before it sends HTTP requests.

```sh
python scripts/benchmark_hybrid_checkpoints.py \
  --model "$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46" \
  --output /private/tmp/Pierre-checkpoint-repeat/receipts.json \
  --rounds 3 --port 8637 --contract incremental --min-speedup 1.1
```

Omit `--contract incremental` to require the stronger cold contract. Changing
the port does not relax any qualification condition. The workload and exact
4096/2048/0 cache receipts are specific to this model, prompt generator and
chunk configuration; another model may correctly fail these expectations.

To re-evaluate stored receipts without running a model:

```python
import json
from scripts.benchmark_hybrid_checkpoints import summarize

with open("receipts.json") as stream:
    result = json.load(stream)
print(summarize(result))
result["contract"] = "cold"
print(summarize(result))
```

For the historical segmentation diagnostic, use the recorded engine commit,
the same prompt generator and checkpoint maximum 0 in both prefill modes.
The fixture's `arms[].command` records each exact launcher and serving flags.
Use the sequence above followed by another clear and identical cold request;
compare all three edited outputs. The disabled launcher uses only:

```python
Scheduler._shared_prefix_local_split = lambda self, request, pending: setattr(
    request, "shared_prefix_snapshot_at", 0
)
```

## Validation and ownership

The benchmark's focused receipt/qualification suite passed 29 tests locally,
including re-evaluation of the recorded real-model matrix. On M5 the initial
28 benchmark cases plus hybrid-state and GDN-prefill numerical suites passed
97 tests before adding that stored-matrix test. Ruff lint/format and diff
whitespace checks passed. No serving implementation changed.
A deliberate in-memory mutation replaced the final cold-equivalence clause
`and (contract == "incremental" or cold_exact == len(rows))` with `and True`.
`test_incremental_contract_does_not_hide_cold_drift` then failed at
`assert not summarize(result)["passed"]`, proving that this false-green
regression is detected. No source file was modified by the mutation run.

Vector owns this benchmark and further segmentation investigation. Atlas
owns any later runtime/default-policy decision. This report qualifies one
model and workload on one M5; it makes no M3/M4 comparison or release claim.


The normalized `*.evidence.txt` excerpts were extracted from the original
child logs after rechecking their SHA256 against `server-evidence.json`.
They retain prefill install/disable messages, checkpoint resume positions and
shutdown completion. The matrix's added `controlled_env` fields are reconstructed
from the historical launcher source, as their origin field states; they were
not originally emitted by that run. Future runs persist this allowlist directly
and fail qualification if the server's effective prefill evidence disagrees.
Ambient environment variables and credentials are never serialized.

## Within-mode counterbalance and evidence boundary

The current schema-2 harness runs eight fresh server arms, in blocks stock
off/on, blocked on/off, blocked off/on, stock on/off. Both checkpoint orders
use the same document seeds and clear/seed/warm/clear/cold sequence within
each prefill mode. Each row carries its order identity; missing orders,
incorrect arm execution sequence, mixed identities and duplicate rows fail
closed. Every late/middle group must clear the threshold in each order before
any pooled estimate can qualify. Head edits remain a zero-reuse control.
The prefill block order is reversed in the second half; this is a serial
order-balanced design, not randomized thermal equivalence.

Schema-1 fixtures retain their original raw numbers and historical summary.
Recomputing with the current harness retains output/cold comparisons but
returns `order_balanced: false`, `performance_passed: false` and `passed: false`.
The original summary is historical evidence, not current certification.

Schema-2 qualification also checks the exact canonical text-only launch
arguments: prefix cache enabled, eight hybrid entries, 2048-token chunks,
loopback binding and speculation/vision disabled. Controlled environment
flags and effective prefill install/disable logs are checked independently.
Actual HTTP cache receipts establish the observed 4096/2048/0 reuse, not an
independent attestation of the maximum number of retained checkpoints.

Source/model/dependency hashes and commands are reproducibility receipts.
They cannot authenticate an author-supplied matrix or expected manifest.
Trusted execution attestation requires a separate verifier and is outside
this opt-in benchmark. HOME retains existing user configuration and the
default model cache; the minimal environment prevents ambient environment
overrides but does not isolate the filesystem or authenticate execution.
The new exclusive-leased M5 campaign completed on 2026-10-10,
09:11:39–09:17:07 UTC, at clean `fe09916eed82a163ac0bd1be7b6a5904f280a5f0`.
The harness SHA256 is
`912a191df00f3239f757588978ccc3ef37c1a6bfc7771b8c90d51cffc4df245b`;
model and dependency versions match the environment above. Two document seeds
were measured in every order, with all 48 rows retained. Subsequent changes only
validate provenance completeness and port bounds; they do not change the
launcher, request sequence or inference implementation.

| Prefill | Edit | Off/on median | On/off median | Order-balanced median |
| --- | --- | ---: | ---: | ---: |
| Stock | Late | 2.1888x | 2.1902x | 2.1902x |
| Stock | Middle | 1.3703x | 1.3658x | 1.3674x |
| Blocked | Late | 2.1498x | 2.1491x | 2.1491x |
| Blocked | Middle | 1.3580x | 1.3583x | 1.3583x |

All 16 late/middle pairs improved. All 24 checkpoint-on/off comparisons
matched in each of seed, warm and cold (72/72 phase comparisons). Both orders
passed the 1.1x per-group threshold. Head edits have zero reuse and order-balanced
medians 0.9998x stock / 0.9992x blocked. The independent cold contract still
**fails: 36/48 warm/cold pairs match**; no rows or failures were excluded.
These results qualify only the incremental contract for this measured workload.
Two seeds per order are a bounded qualification sample, not a confidence interval
or a universal performance claim. Temperature/power traces were not collected;
we cannot claim thermal equivalence.

`counterbalanced-matrix.json` retains all actual receipts and recomputed summaries.
`counterbalanced-evidence.json` binds original child-log hashes to normalized
prefill/resume/shutdown excerpts. Re-evaluation with the current validator passes
only the explicitly selected incremental contract and fails the cold contract.
The same reproduction command now emits schema 2 and eight arms; `--rounds 2`
reproduces this campaign's sample count.
