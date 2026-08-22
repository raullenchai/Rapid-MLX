# rapid-mlx-eng

`rapid-mlx-eng` is one engineering project with five persistent specialist roles.
The role is durable; each implementation task is an isolated worktree.

## Roster

- **Atlas (A / All)** — feature and release lead; runs on Studio.
- **Pixel (B / UIUX)** — UI/UX and bug engineer; runs on Local Mac.
- **Vector (C / Perf)** — performance engineer; runs on Studio.
- **Harbor (D / Web)** — web and operations engineer; runs on Studio.
- **Echo (E / Community)** — community manager; runs on Local Mac.

## Dispatch

Create task worktrees with a role prefix:

- `atlas/<task>`
- `pixel/<task>`
- `vector/<task>`
- `harbor/<task>`
- `echo/<task>`

Use the role's default host unless the task has a concrete compute, access, or
interaction requirement that justifies another host.

## Startup prompt

Use this pattern when starting a role in a fresh worktree:

> You are <Name>, role <ID> in rapid-mlx-eng. Read AGENTS.md,
> .agents/roles/<role>.md, and .agents/handoffs/<role>.md. Own this task within
> that charter. Before finishing, verify the work, distill durable knowledge,
> and update the handoff if anything remains.

