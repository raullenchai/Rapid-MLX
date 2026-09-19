# Vendor-mllm PR stack handoff (2a / 2b-1 / 2b-2)

Receiving role: Atlas

Owner: Atlas

Host: Raullens-Mac-Studio (macOS 26.5.2)

Branches: `refactor/vendor-mllm-primitives` (#3545) ← `refactor/vendor-mllm-apc-support` (#3554) ← `refactor/vendor-mllm-apc-engine` (#3558, worktree `~/orca/workspaces/rapid-mlx/vendor-mllm-apc-engine`)

Iron rules: never merge or enqueue (stack waits for manual queueing); evidence JSON must not contain model paths / user data / raw App traces; never bare `git stash` on worktrees sharing a stash (use a temp WIP commit).

## Verified facts

- #3558 head `d90ea5c8a`. Diff vs `refactor/vendor-mllm-apc-support` = **199,975 bytes**; codex cap is 200,000, so ~25 bytes headroom. Any net-add must be funded by an equal deletion elsewhere in the diff.
- Provenance (machine-verified): vendored `apc.py` vs `/opt/homebrew/lib/python3.12/site-packages/mlx_vlm/apc.py` (0.7.1) differs by exactly **18 sentinel lines = 15 `.models*` + 3 `.turboquant`** import redirects; upstream sha256 `5b2b940852f11f34f7b4daf627bc31fc701f8abffc72d40189bc3e5ac57f878c`. The earlier "16 .models*" figure (handoff + PR body + `__init__.py` note) was a miscount; corrected in `d90ea5c8a` (zero byte delta).
- `full_unit` red set on this host is exactly **17 tests, byte-identical to `/tmp/base-fails.txt`** (verified for #3558 r1 and #3554 r5; env-related, pre-existing per #3545/#3554 bodies). `scorecard.verdict()` is strict (any fail → DO NOT MERGE), so letter-level MERGE-SAFE is unreachable here until the 17 baseline reds are fixed at root. **Loop bar: codex 0 BLOCKING + lint/targeted green + full_unit == baseline set.**
- codex reads **only** PR metadata + PR body + diff (`steps/codex_review.py` prompt build; PR comments are never fetched). Refutations must go into the PR body via `gh pr edit`. The body disposition works: r1's 2 BLOCKING were re-tiered to NIT by r2 after the provenance ledger landed.
- Concurrent `pr_validate` runs of the same PR are isolated (per-run artifact dir + per-run detached temp worktree) but waste CPU; #3554's loop (background subagent, SendMessage or read its output file — do not spawn a duplicate) overlapped my r2. Wait for the previous run to exit before starting the next.
- `stress_e2e_bench` gates on blast=high; #3558 is medium, so bench is skipped and there is no bench-conflict with other runs — but check `pgrep -fl scripts.pr_validate` before starting anyway.

## Loop state

- #3558 — CONVERGED at r3 (`~/pr_validate-3558-r3.log`): codex 0 blocking; lint/targeted green; full_unit = 17 baseline + 1 documented load flake (sidecar smoke 5s timeout under concurrent load; passed in r1/r2). Body-disposition ledger + PR comment 5736088251 carry the evidence. Head `d90ea5c8a`.
- #3563 (2b-3, branch `refactor/vendor-mllm-apc-engine-dual-ns`, worktree `vendor-mllm-apc-engine-dualns`, head `7ebafeccd`) — CONVERGED at r1 (`~/pr_validate-3563-r1.log`): codex 0 blocking on first pass; full_unit = 17 baseline + 2 port-collision flakes (serve-command e2e tests hit `Address already in use` on port 8000, held by an unrelated long-lived process). PR comment 5736752372. Targeted-tests step skipped (diff→test mapping quirk on stacked PRs) — locally verified 23 engine/support + 196 lane/adapters green; mypy pinned-env zero growth. Scope: dual-namespace helpers + `_ns` faithful restore + resolver widening + 3 upstream-bugfix hunks with repro tests; inventory now 14 redirects + helpers + bugfixes (sentinels = 22). Non-goals documented in PR body (make_prompt_cache / quantize/batch constructors = step 3).
- #3566 (2c, branch `refactor/vendor-mllm-apc-inputs`, worktree `vendor-mllm-apc-engine-dualns`, head `26854466c`) — loop r1→r6, near-converged. codex PASS at r2 and r6; between those it mined the 830-line verbatim region one finding per round (video_sampling hook seam ×2, cv2 capture leak ×2, bytes-path mangling ×1) — all resolved as **four documented hunks** (dual-ns normalization in `processor_video_sampling`; capture-release try/finally incl. failed opens in `load_video`; `os.fsdecode` for bytes video paths in `prepare_inputs`; `with`-closed URL response in `load_audio`), each repro-tested, all recorded in the package inventory; those four functions are excluded from the byte-identical probe and behavior-tested instead. r6 codex PASS but bench `[INCONCLUSIVE]` under concurrent 3571 load; **r7 CONVERGED on a quiet machine** (`~/pr_validate-3566-r7.log`): codex/lint/targeted PASS, bench PASS (4×3, 3 pre-existing), full_unit = 17 baseline + the 2 port-8000 flakes (byte-identical failure set vs #3563). PR comment carries the final scorecard. Do not enqueue until #3563 merges. Lesson: for a big verbatim region, run the resource-hygiene sweep PROACTIVELY instead of refuting codex one finding per round — fixing the whole class at once converges the loop in one round.
- Letter-level verdicts stay DO NOT MERGE (strict verdict() × the 17 baseline reds) — accepted stack state; nothing enqueued, per iron rule.

## Gotchas (each paid for once)

- Bash cwd resets to `iotex-core` between commands — `cd` to the target worktree every time.
- `git checkout -- <file>` restores from the **index**; undoing staged edits needs `git checkout HEAD -- <file>`.
- mypy only under `/private/tmp/mypy-pinned-venv/bin/python` (homebrew python3.11 mypy resolves stale mlx_vlm 0.6.16 → phantom errors).
- pytest via `vendor-mllm-primitives/.venv/bin/python`, run from the target worktree root.
- `mx.clear_streams()` on the main thread poisons the process-default stream — tests must call it from a worker thread.
- codex false-positive rate is high; verify against code + upstream before editing. Prior evidence-backed refutations: `from_legacy(3.5,'uniform')` does not raise; `merge` is upstream's own text.
- `gh pr edit` must run from inside a repo worktree (fails from `/tmp`).
- Baseline mypy additions for this stack live at the end of `config/mypy-error-baseline.txt`: `apc.py 37` (2b-2) and `cache.py 34` (2a's debt, landed here so the stack stays green; #3545 must NOT add it again).

## Queue (in order)

1. ~~Finish the #3558 codex loop~~ — done (r3, codex 0 blocking).
2. ~~2b-3: engine dual-namespace widening + engine-body deviation candidates~~ — done, #3563 converged at r1.
3. **2c**: DONE as #3566 (see loop state; r7 bench re-run pending a quiet machine). prepare_inputs closure vendored (region sha256 in inventory), lane callsites redirected (`multimodal_processor.py:121`, `mllm_batch_generator.py:3040`), Qwen36TextArraysCache parent swapped, prepare_inputs stubs flipped to the vendored module. After convergence: step 3 (make_prompt_cache / quantize constructors) per design doc.
4. Task #11: after #3551 merges, revalidate #3545 (its baseline entry is already satisfied via #3558 — nothing to add there).
5. Cleanup MZR-3: `~/rapid-mlx`, qualification artifacts, HF downloads, `/tmp` logs.

## Risks

- 25-byte diff headroom: even a one-line addition can push codex's 200 KB truncation at a file boundary and silently shrink the reviewed surface.
- The 17 baseline reds are env-related; if the set ever differs from `/tmp/base-fails.txt`, diff the FAILED lines before acting — a superset means this PR regressed something (or hit an environment collision: port 8000 is held by a foreign long-lived process on this host and the `test_legacy_prefix_cache_flag_warnings` serve e2e tests bind it — expect 2 extra failures when it is occupied; `test_sidecar_vision_smoke` has a hard-coded 5.0s startup budget and flakes under concurrent full_unit load).
- #3554's codex loop runs on this host from a separate session; stagger `full_unit` runs to keep timings comparable.
