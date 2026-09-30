# Share Compute Desktop official integration

- Receiving roles: Atlas for live gateway acceptance; Harbor for signed release validation.
- Branch / PR: `atlas/share-compute-official`, `raullenchai/Rapid-MLX#3923`.
- Source: `loriz-art/Rapid-MLX#2`, UI commit `1f3d2e0ba`, cherry-picked onto official `main` with Lori's authorship retained.

## Verified

- The UI contribution includes Share, Live Pool, Credits, a Keychain-backed `qsprk-` read key, the pool-summary and ledger clients, and focused desktop tests.
- The current-main memory floor remains enforced when the UI offers models to Share or Live Pool.
- `cd apps/rapid-mac && swift test --filter ShareCompute` passed 125 tests in 7 suites on the Studio Mac after the Keychain failure-path fixes.
- An independent local reviewer found four actionable state defects. The PR now keeps failed Keychain saves/removals visible, labels restore as requested until readiness is known, and disables the Live Pool action when upstream availability is unreported.
- `python3.12 -m pytest tests/test_rapid_mac_ax_identifiers.py -q` passed 69 tests.
- `GET https://pay.quicksilverpro.io/v1/pool/summary` returned the expected top-level fields on 2026-09-30, but showed zero connected and ready nodes at that time.

## Open acceptance work

- Atlas: with a connected, ready node and an authorized test account, send a real gateway inference request; check the response and an attributable ledger window or credit. Verify `next_cursor` pagination against actual rows and observe service errors or rate limiting without exceeding the published budget. The local tests cover these client states, not production routing or settlement.
- Harbor: validate the exact release candidate after signing and notarization. This Studio keychain had zero valid code-signing identities on 2026-09-30, so source tests cannot stand in for artifact validation. Production release still needs human authorization.
- Atlas: re-run independent PR review. The `spark2` review runner could not start Codex because its login token refresh returned HTTP 401 on 2026-09-30.

No claim that a ready node has served a real request or earned a credit is established by this PR.
