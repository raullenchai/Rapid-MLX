# Engineering decisions

Record decisions that constrain future work. Include context, decision, evidence,
alternatives considered, consequences, owner, and date.

## Decisions

| Date | Decision | Status |
| --- | --- | --- |
| 2026-09-27 | [Agent-session prefix reuse on 16-32 GB Macs](2026-09-27-agent-session-prefix-cache.md) | Accepted |
| 2026-09-13 | [Rapid agent runtime](2026-09-13-rapid-agent-runtime.md) | Accepted for P0 implementation |
| 2026-08-31 | [Community benchmark wire contract v1](2026-08-31-community-benchmark-wire-contract.md) | Accepted contract; producer and ingestion migration deferred |
| 2026-08-31 | [Community Benchmark local workspace](2026-08-31-community-benchmark-local-workspace.md) | Accepted for internal beta |
| 2026-08-22 | [Model management and performance decision SSOT](2026-08-22-model-management-performance-ssot.md) | Accepted direction; incremental rollout |

Superseded decision records are removed once no live code, tests, CI, or docs
reference them; retrieve older ones from git history. A decision document stays
listed here only while it still constrains current work (each entry above is
referenced from live docs or CI-enforced architecture material).
