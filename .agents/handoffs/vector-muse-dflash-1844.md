# Vector handoff — Muse-Glimmer 8-bit DFlash (#1844)

## 2026-09-07 — starting

- Start FYI fallback for Atlas, Pixel, Harbor, Echo, and ds0731 (Orca role
  messaging unavailable): Vector owns branch `vector/1844-muse-dflash`,
  worktree `/private/tmp/vector-muse-dflash-1844`, based on
  `origin/main@9113b5d8d` on Local Mac.
- Intention: expose a revision-pinned, curated Muse-Glimmer 30B 8-bit target
  plus its published assistant through Rapid's existing single-user DFlash
  server, so users on sufficiently large Apple Silicon systems can opt into
  the independently reported roughly 2.1–2.3x decode improvement.
- Scope: alias/catalog/capacity data, exact target/drafter/runtime identity,
  DFlash eligibility contracts, focused tests, and real-weight server
  dogfood. Non-goals: 4-bit DFlash, changing the dedicated DFlash server,
  batched DFlash, upstream runtime changes, new UI, release, or deployment.
- Private reference check: the required assistant model, target hidden-state
  capture, rollback, and serial server loop landed upstream and shipped in
  mlx-vlm 0.6.13; Rapid pins 0.6.17. Rapid's existing curated DFlash aliases
  establish revision pinning, algorithm identity, fail-closed eligibility,
  cancellation, and response-contract patterns. The change will extend those
  patterns rather than add another runtime.
- Verification plan: alias/schema/download/model-size tests; stream and
  non-stream route contracts; exact cached-artifact boot; same-host plain vs
  DFlash throughput and output evidence; cancellation/recovery; scope-locked
  adversarial self-review; exact-head PR validation and managed queue. No
  Spark or reviewer sub-agent.

## 2026-09-07 — implementation and dogfood complete

- Added `muse-glimmer-30b-8bit` with a 48 GB memory floor, exact target and
  assistant revisions, and fail-closed `dflash` identity. The 4-bit alias
  remains ineligible.
- Real-weight dogfood on an M3 Ultra / 256 GB host with mlx-vlm 0.6.17:
  31.1 GiB target plus published assistant loaded through the public alias;
  `/healthz` returned both exact revisions and `algorithm=dflash`.
- Three-run, 256-token qualification: code workloads improved 1.89x–2.10x,
  code median 2.00x; general chat improved 1.83x. The pair passed both the
  1.30x code gate and 1.00x non-code floor. Post-request RSS was 36.9 GiB.
- Stream cancellation after 1.5 seconds / 4,537 response bytes released the
  serial lease; an immediate recovery request completed in about 1.0 second.
- Scope-locked adversarial self-review (three rounds): tightened the contract
  from assistant-only allowance to an exact target+assistant pair; corrected
  the README catalog total; final round found no in-scope blocker.
- Verification: focused suite 3,929 passed / 2 skipped; full suite 23,047
  passed / 238 skipped / 6 xfailed / 1 xpassed. Seven Bonsai runtime tests
  fail identically on clean `origin/main@9113b5d8d` because the installed
  image runtime's `TilingConfig` API differs; no changed file intersects that
  failure. The one initial README count failure was caused by this alias and
  was fixed before the focused rerun.
- Durable benchmark evidence:
  `docs/engineering/performance/muse-glimmer-30b-dflash-qualification.md`.
- Completion FYI fallback for Atlas, Pixel, Harbor, Echo, and ds0731: PR
  #3211 (`7ab445bc7`) contains the qualified implementation and evidence. No
  release or deployment action is authorized; no follow-up owner is required
  for this PR beyond managed-queue observation.

## 2026-09-07 — PR validation

- Validator review round 1 requested an explicit eligibility-report assertion.
  `check()` already raises on rejection (and returns `None` on success), but
  the test now also pins `reasons == ()` and `recommendation == "verified"`.
- Validator review round 2 found no blocking issues. Description, supply-chain,
  lint, and 3,443 targeted tests passed.
- Full unit under the repository-required mflux 0.19.1 completed with 23,033
  passed and four doctor/extras failures. All four reproduce byte-for-byte on
  clean `origin/main@9113b5d8d` with the same venv; they arise from existing
  subprocess optional-extra row indexing and do not intersect this diff.
- Final exact-head validation may explicitly skip only `full_unit`, because
  that gate has already run in full and its complete failure set has been
  reproduced against the merge base. No product test failure is waived.
