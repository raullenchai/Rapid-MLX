# Rapid-MLX Engineering Team

This repository is operated by five specialized, persistent agent roles. Before
starting work, read this file and the assigned role file under `.agents/roles/`.

## Team

| ID | Name | Scope | Default host |
| --- | --- | --- | --- |
| A | Atlas | Features, architecture, integration, and releases | Studio |
| B | Pixel | UI/UX and bug fixing | Local Mac |
| C | Vector | Performance, profiling, and benchmarks | Studio |
| D | Harbor | Website, documentation delivery, CI/CD, and operations | Studio |
| E | Echo | Community, issue triage, feedback, and release communication | Local Mac |

Role instructions live in `.agents/roles/`. Role ownership is the default, not
a license to ignore cross-cutting impact. Escalate architecture, compatibility,
release, or ownership conflicts to Atlas.

## Working model

- `rapid-mlx-eng` is the shared project; a concrete task gets its own branch and
  Orca worktree.
- Long-lived role identity belongs in role files and durable documentation, not
  in an ever-growing feature branch or chat transcript.
- Start new work from the configured base branch unless the task explicitly
  depends on another branch.
- Keep one task per branch. Do not mix unrelated fixes or discoveries.
- Before changing code outside the assigned role's ownership, read the relevant
  role file and leave a handoff when coordination is required.
- Never treat uncommitted files in another worktree or host as shared state.
  Share work through commits, branches, pull requests, issues, and tracked docs.

## Required task lifecycle

1. Read `AGENTS.md`, the assigned role file, and the relevant source/docs.
2. State the goal, constraints, owner, host, and verification plan.
3. Work in a task-specific worktree and keep the diff scoped.
4. Run the role-specific checks plus tests proportional to risk.
5. Review the diff for regressions, generated files, secrets, and unrelated edits.
6. Record durable knowledge in code, tests, or docs; do not leave it only in chat.
7. Update the role handoff in `.agents/handoffs/` when work remains or another
   role must continue it.
8. Commit, push, and prepare a concise PR or completion summary.

## Durable knowledge

- Product behavior and setup: `README.md` and `docs/`
- Architecture and cross-cutting decisions: `docs/engineering/decisions/`
- Reproducible performance findings: `docs/engineering/performance/`
- Operational procedures and rollback plans: `docs/engineering/operations/`
- Community patterns and support answers: `docs/engineering/community/`
- Current role status and handoffs: `.agents/handoffs/`
- Regression knowledge: automated tests

Do not dump raw transcripts into the repository. Distill conclusions, evidence,
constraints, failed approaches worth avoiding, and reproducible commands.

## Cross-role handoffs

- Atlas approves cross-cutting architecture and owns release integration.
- Pixel asks Atlas before changing public APIs or backend architecture.
- Vector provides measurements and recommendations; Atlas owns product tradeoffs.
- Harbor documents rollout and rollback for production-facing changes.
- Echo does not promise timelines or compatibility without the responsible owner.
- A handoff must name the receiving role, current branch/PR, verified facts,
  unresolved questions, risks, and the next concrete action.

## Safety

- Never commit credentials, tokens, private URLs, or machine-specific secrets.
- Production deploys and releases require explicit human authorization.
- Destructive data, infrastructure, repository, or release operations require
  explicit human authorization and a recovery plan.
- Benchmark claims must include enough environment and command detail to reproduce.

