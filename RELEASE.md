# Releasing Rapid-MLX

How a release actually happens, what gates it, and what to do when things go
wrong. If you only read one thing: **you cut a release by merging a
`chore: bump version to X.Y.Z` commit to `main`** — everything else is
automated.

## TL;DR — cut a release

1. Bump `version` in `pyproject.toml` to `X.Y.Z`.
2. Open a PR whose commit subject is exactly `chore: bump version to X.Y.Z`
   (GitHub's `(#N)` squash suffix is fine), and merge it to `main`.
3. That's it. The pipeline below tags `vX.Y.Z`, publishes a GitHub Release,
   PyPI, and Homebrew — **after** the Tier-1 agent gate passes.

Nothing else is manual. There is no separate "publish" button.

## The pipeline

`.github/workflows/auto-release.yml` runs on every push to `main` and is three
jobs:

```
push to main
   │
   ▼
┌─ detect (GitHub-hosted, ~2s) ─────────────────────────────────┐
│  Is the commit subject "chore: bump version to X.Y.Z"?        │
│  pyproject matches? tag not already present?                  │
│  → outputs should_release / version. Non-bump pushes stop here.│
└───────────────────────────────────────────────────────────────┘
   │ should_release == true
   ▼
┌─ tier1-agent-gate (SELF-HOSTED, Apple Silicon "Studio", ~10m) ┐
│  Build the exact release source into a fresh venv, then run   │
│  tests/integrations/agent_smoke.sh: boot rapid-mlx serve and  │
│  drive Claude Code / Codex / Hermes / Aider through a real     │
│  end-to-end edit. Exit non-zero if any of the 4 regresses.    │
└───────────────────────────────────────────────────────────────┘
   │ gate passed
   ▼
┌─ release (GitHub-hosted) ─────────────────────────────────────┐
│  Create tag vX.Y.Z + GitHub Release (changelog + contributors)│
│  using RELEASE_PAT → fires the `release: published` event.     │
└───────────────────────────────────────────────────────────────┘
   │
   ▼
publish.yml → PyPI    →    Homebrew core autobump → `brew` users
```

`release` `needs: [detect, tier1-agent-gate]`, so **a version bump cannot tag
or publish unless the gate passes.** This mirrors how Apple's own MLX gates its
PyPI publish on a self-hosted-Metal `test_wheel` job.

## Why a self-hosted runner

The gate re-verifies that our four **Tier-1 (flagship) agents** — Claude Code,
Codex, Hermes, Aider — actually work end-to-end against a real local model on
Apple Silicon, on the *current* client binaries. GitHub-hosted runners cannot do
this: no Metal, no cached weights (a boot model is ~40–70 GB), and the agent
CLIs aren't installed. So the gate runs on a **self-hosted Apple-Silicon runner
(the "Studio", an M3 Ultra)**.

> **Two meanings of "Tier-1", don't conflate:** *Tier-1 model families*
> (Qwen 3.6 / Gemma 4 / DeepSeek / gpt-oss / Hy3) and *Tier-1 agents* (the 4
> flagship). The gate is about the **agents**. See the agent-tier docs on
> rapidmlx.com/docs/matrix and `tests/integrations/README.md`.

### The runner

- Registered on the Studio as a launchd service
  (`actions.runner.raullenchai-Rapid-MLX.studio-m3-ultra`), labels
  `[self-hosted, macOS, ARM64, rapidmlx-studio]`. It auto-restarts on crash or
  reboot, and maintains an **outbound** long-poll to GitHub (no inbound ports,
  no VPN) — GitHub never connects *into* the Studio; the runner pulls jobs.
- **Security:** the self-hosted job runs only on push to protected `main` /
  maintainer `workflow_dispatch`, never on `pull_request`, and fork-PR workflow
  approval is set to *all external contributors* — so fork code can never reach
  the runner. Never add a `pull_request` trigger to a `rapidmlx-studio` job.
- Check it's online: `gh api repos/raullenchai/Rapid-MLX/actions/runners`.
- Manual / on-demand run of the gate (no release):
  `gh workflow run agent-gate.yml -R raullenchai/Rapid-MLX`.

## When the gate fails (a Tier-1 agent regressed)

The `release` job is skipped; the version stays committed-but-unpublished. In
the failed `tier1-agent-gate` job log you'll see which agent (`claude-code` /
`codex-cli` / `hermes` / `aider`) reported `FAIL`.

1. First rule out a *model-strength* artifact — a weak model that fakes success
   is not an integration bug. The gate uses `qwen3.6-35b-8bit` (≥8-bit on
   purpose; never 4-bit, which confounds the two).
2. Localize with a wire replay: proxy the real client once, diff our SSE stream
   against what it expects, and pin the offending event/field.
3. Fix the integration (engine PR), then re-run: `gh run rerun <run-id>` or push
   again. Once green, the release proceeds automatically.

## Studio down / emergency release (break-glass)

If the Studio is **offline** when a version bump lands, the `tier1-agent-gate`
job has no runner to run on: it **queues** (up to GitHub's 24 h limit, then
fails), and the release stalls. This is intentional — *no verification, no
release*.

**Normal recovery:** bring the Studio back (it's a launchd service — usually
just power/network), then `gh run rerun <auto-release-run-id>`. Nothing is lost,
only delayed.

**Emergency override** (Studio can't be fixed and the release truly can't wait):
run auto-release manually with the gate bypassed —

```bash
gh workflow run auto-release.yml -R raullenchai/Rapid-MLX \
  -f force_version=X.Y.Z \
  -f reason="Studio offline, security fix must ship"
```

This skips the gate and releases `vX.Y.Z` (it still checks pyproject matches and
the tag is new). The bypass is **audited**: the actor + reason are logged and
stamped into the GitHub Release notes as an "⚠️ Emergency release" banner. Use
it sparingly — it ships a version whose agent integrations were *not* verified.

## Related docs

- `tests/integrations/agent_smoke.sh` — the gate script (canonical).
- `tests/integrations/README.md` — the full agent × model integration matrix.
- `landing/runbooks/agent-release-verification.md` (rapidmlx.com repo) — the
  operational runbook + the freshness-stamp bump for the website.
- `.github/workflows/publish.yml` — PyPI publish on `release: published`.
