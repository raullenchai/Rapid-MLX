# Vector handoff: bounded native AX collection

- Owner: Vector
- Branch/worktree: `vector/cua-native-ax-bounded`, `/private/tmp/harbor-desk-cua-native-ax-bounded`
- Base: target-revalidation stack `7036ff81eff44926d9545964048b9e0f7dc42435`
- Goal: return prioritized native-app snapshots within the existing 20-second watchdog even when a dense dynamic table tail is slow.
- Root cause: `ax_driver.collect` retried every app up to four times waiting for `AXWebArea`. Native apps never expose that role, so a complete Activity Monitor walk was repeated four times with three 1.5-second sleeps. A roughly 9-second first pass could therefore exceed the watchdog on a later pass.
- Design: native apps collect once. Chromium-family bundle IDs explicitly opt into the existing four-attempt lazy web-content retry. The 600-target/depth bounds remain unchanged. The watchdog receives a shared append-only target prefix; if a later AX call blocks after priority controls were fully appended, it returns a copied, center-complete prefix marked `truncated`, while a timeout before any complete target remains typed `ax_unavailable`.
- Safety: entries are appended only after secure-field redaction and complete target construction. Timeout fallback copies dictionaries while retaining exact live AX refs; action-time target/window revalidation remains in force. Large sibling collections keep the no-prescan behavior.
- Reference check: the existing Rapid Chromium-only manual accessibility activation establishes the same capability boundary. Retry is now coupled to that bundle family instead of guessing from the presence or absence of web targets in native apps.
- Verification: native collection test proves one walk/no sleep; explicit Chromium test reaches `AXWebArea` on retry; timeout test blocks after an actionable priority target and proves copied live mapping, computed center, and partial status; secure/priority regressions remain green.
- Risk: a daemon AX worker can still outlive a timeout, as before. Partial fallback is possible only after a fully appended target exists and is clearly marked truncated. Safari remains single-pass until evidence demonstrates it needs a platform-specific lazy web retry.
