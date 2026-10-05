# Engineering decisions

Record decisions that constrain future work. Include context, decision, evidence,
alternatives considered, consequences, owner, and date.

## Decisions

| Date | Decision | Status |
| --- | --- | --- |
| 2026-10-05 | [CUA approvals belong to the brain, not the hands](2026-10-05-cua-approvals-belong-to-the-brain.md) | Accepted |
| 2026-09-27 | [Agent-session prefix reuse on 16-32 GB Macs](2026-09-27-agent-session-prefix-cache.md) | Accepted |
| 2026-09-24 | [Native System One server boundary](2026-09-24-system-one-server.md) | Accepted for implementation |
| 2026-09-15 | [Community Benchmark read-API seam](2026-09-15-community-benchmark-read-api-seam.md) | Implemented; aggregate gaps documented |
| 2026-09-13 | [Rapid agent runtime](2026-09-13-rapid-agent-runtime.md) | Accepted for P0 implementation |
| 2026-09-08 | [Share Compute Desktop boundary](2026-09-08-share-compute-desktop-boundary.md) | Implemented behind the experimental gate |
| 2026-09-05 | [Always-on service configuration and transactional apply](2026-09-05-always-on-service-configuration.md) | Accepted for implementation |
| 2026-08-31 | [Community benchmark wire contract v1](2026-08-31-community-benchmark-wire-contract.md) | Accepted contract; producer and ingestion migration deferred |
| 2026-08-31 | [Community Benchmark local workspace](2026-08-31-community-benchmark-local-workspace.md) | Accepted for internal beta |
| 2026-08-31 | [Atomic product model catalog](2026-08-31-atomic-product-model-catalog.md) | Accepted for shadow/dual-read implementation |
| 2026-08-25 | [Assistant replacement and dictation coexistence](2026-08-25-assistant-replacement-and-dictation-coexistence.md) | Accepted for the 0.13.1 implementation track |
| 2026-08-22 | [Model management and performance decision SSOT](2026-08-22-model-management-performance-ssot.md) | Accepted direction; incremental rollout |

Keep accepted decisions while their implementations or compatibility boundaries
remain live, even when no other document links to them. When a later decision
replaces one, retain the older record with a superseded status and a link to its
successor. Draft investigations that never became architecture may be removed
after their durable conclusions move into code, tests, or a current decision.
