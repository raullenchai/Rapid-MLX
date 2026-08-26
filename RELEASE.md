# Releasing Rapid-MLX

How a release actually happens, what gates it, and what to do when things go
wrong. If you only read one thing: **you cut a release by merging a
`chore: bump version to X.Y.Z` commit to `main`** — automation then does the
work and presents one **reviewer approval** (the only manual transaction step)
before anything is tagged or published.

This file is the **canonical** release-flow reference (as
`scripts/pr_validate/README.md` is for the PR-validation pipeline). The
extended operational guide, `docs/development/releasing.md`, defers to it.

## TL;DR — cut a release

1. Bump `version` in `pyproject.toml` to `X.Y.Z`.
2. In the same PR, `git mv docs/release-notes/unreleased.md
   docs/release-notes/vX.Y.Z.md` (optional — see "Release notes" below).
3. Open a PR whose commit subject is exactly `chore: bump version to X.Y.Z`
   (GitHub's `(#N)` squash suffix is fine), and merge it to `main`.
4. Merging starts the pipeline. After the Desktop candidate validates the
   exact commit, the live main head and release-blocker evidence are gathered,
   and the protected `rapid-mac-tag` deployment requests approval, a **reviewer
   inspects the exact SHA / main-head / blocker evidence** shown there and
   approves it. Automation then tags `vX.Y.Z`, publishes a GitHub Release,
   PyPI, Homebrew, and the Desktop DMG/updater feeds.

There is **no separate manual tag or publish command** — no button, no
hand-run script. The reviewer's approval at the protected environment gate is
the **only manual transaction step** a release requires; everything else is
automated.

## Desktop (Rapid-MLX Desktop) RC tags — validated before claimed

The engine release also drives the Desktop app: a `rapid-mac-vX.Y.Z[-rcN]` tag
is created so `rapid-mac-release.yml` builds/signs/notarises the DMG. Per
[#2301](https://github.com/raullenchai/Rapid-MLX/issues/2301), that immutable
tag is claimed **only at the exact commit whose desktop candidate passed the
signed/notarised/DMG-validated lane, and which is still the live `main` head**:

- `auto-release.yml` runs a `desktop-candidate-gate` (macos-15) that builds and
  validates the exact release commit **before** the tag claim, producing a
  Desktop manifest bound to the source SHA, embedded app versions and DMG digest.
- The tag claim runs inside the **`rapid-mac-tag`** environment (required
  reviewer, `prevent_self_review=false`, deployment-branch policy exactly
  `main`), after the pre-approval `release-prep` job has printed the exact
  validated SHA + live main head + live release-blocker evidence.
- Release-blocker evidence and the live main-head identity are re-queried
  **immediately before** the immutable claim. This is a **freshness/cutoff
  guard, not a transaction** — GitHub exposes no single atomic op across Issues,
  `refs/heads/main`, and the tag POST (the POST is atomic only for tag
  identity). It establishes the release cutoff right before the claim. Changes
  **observed at the cutoff** abort the claim; changes *after* the cutoff are not
  observable before the POST — they are post-cut, handled as a follow-up/next
  RC once detected, and cannot retroactively invalidate the exact validated
  artifact SHA the tag is bound to.
- **Operational freeze:** once `release-prep` evidence is ready, **hold `main`
  merges through environment approval and the tag claim** (sole owning reviewer
  coordinates this). If a blocker change is **detected before** the claim,
  abort and use the normal retry route at the new head; a change after the
  cutoff may be unobservable until post-claim, at which point it is handled as
  a next-RC/release incident.
- An RC needing correction is **superseded by the next RC** on its own validated
  commit; an existing RC tag is never moved, force-pushed, or deleted.

On the bump PR only PF-3 runs — a fail-closed read-back that the
`rapid-mac-tag` environment exists and is protected (required reviewer,
`prevent_self_review=false`, deployment policy exactly `main`), so an
unprotected/drifted environment is a NO-GO before any release rather than a
surprise on the day. The **live** blocker evidence and live main-head identity
gates (PF-4) run *after merge*: in `release-prep` before approval and again
immediately pre-tag. Full ordering and the break-glass path:
`docs/development/releasing.md` and `apps/rapid-mac/RELEASING.md`.

## Release notes

`scripts/build_release_notes.sh` builds the GitHub Release body. It reads
**`docs/release-notes/vX.Y.Z.md` out of the commit being tagged**; if that file
exists it becomes the top of the notes verbatim (prose, `## Highlights`,
benchmark tables, caveats) and the auto commit list is appended below it under a
collapsed `<details>`. If it's absent the release publishes exactly as before —
a flat commit list. **A release is never blocked on prose.**

Write the prose as you land the work, in `docs/release-notes/unreleased.md`, and
rename it in the version-bump PR. Full guidance and a template:
`docs/release-notes/README.md`.

Two invariants the workflow enforces, so notes can never describe a tree other
than the one being tagged:

- The release commit is resolved **once** (from `github.sha`, asserted equal to
  the checked-out `HEAD`) and that one SHA is used for the notes, the ancestry
  assert, and `--target`.
- The baseline is the nearest **ancestor** tag (`git describe`), not the highest
  version string in the repo — those differ whenever a newer tag exists that the
  release commit does not descend from.
- If the tag already exists at a *different* commit, the job **fails loudly**
  rather than publishing: `gh release create` reuses an existing tag and
  silently ignores `--target`.

Run the offline tests with `./tests/release/test_build_release_notes.sh`.

## The pipeline

`.github/workflows/auto-release.yml` runs on every push to `main`. On a version
bump it fans out, waits for two independent gates, gathers pre-approval
evidence, asks a human to approve the exact SHA, then claims the desktop tag and
creates the engine release:

```
push to main (subject "chore: bump version to X.Y.Z")
   │
   ▼
detect (GitHub-hosted, ~2s): is it a bump? does pyproject agree?
   is the release missing? → outputs should_release / version.
   Non-bump pushes stop here.
   │ should_release == true
   ├───────────────────────────────►  (PARALLEL — independent, both need only detect)
   ▼                                    │
tier1-agent-gate (self-hosted "Studio") │    desktop-candidate-gate (macos-15)
Build exact source; run agent_smoke.sh  │    Build + sign + notarise + DMG-validate
driving Claude Code / Codex / Hermes /  │    the app at the EXACT release commit;
Aider / DeepSeek Harness.  Exit non-zero│    emit a Desktop manifest binding source
if any of the 5 regresses.              │    SHA + embedded versions + DMG digest
                                        │    (no tag, no release, no updater pointer)
   │  BOTH must pass (tier1 force-bypassable)        │
   ▼                                                    ▼
release-prep (pre-approval evidence, no env): resolve the exact release SHA,
verify it equals the accepted desktop-candidate SHA, verify the LIVE main head
still equals it, gather LIVE release-blocker evidence vs the waiver file, and
print all of it — this exact SHA is what a reviewer approves.
   │
   ▼
release (environment: rapid-mac-tag — HUMAN APPROVAL on the printed SHA):
   re-verify live rapid-mac-tag protection, re-query live blockers + main head
   (TOCTOU), then tag the desktop app at the exact validated SHA.
   │
   ▼
rapid-mac-v* tag fires rapid-mac-release.yml: re-runs the SAME shared
desktop-releasable validation on the tagged commit and re-verifies the tag
binding before upload. It mirrors the immutable DMG/Sparkle assets (as a
GitHub prerelease for an RC); ONLY a non-RC stable release publishes the
mutable appcast/latest.json updater pointers.
   │ exact tagged run succeeded + published non-empty DMG + tag recheck
   ▼
release creates the engine tag vX.Y.Z + GitHub Release (RELEASE_PAT)
→ publish.yml to PyPI.
```

The two gates run **in parallel** (they share nothing — different runners,
different checks) and `release-prep` `needs` BOTH, so **a canonical normal
release cannot tag or publish unless *both* pass**: the Tier-1 engine gate and
the signed Desktop candidate at the exact commit.

Exact invariant: canonical normal releases require the Tier-1 gate; the audited
**emergency dispatch may bypass only the Tier-1 gate** (see below); the signed
Desktop candidate, the exact-SHA binding, the live release-blocker / main-head
gates, and the protected `rapid-mac-tag` environment approval are **never
bypassed by either supported auto-release route** (normal or emergency). (Manual
`rapid-mac-v*` tag creation is separately unsupported — `apps/rapid-mac/RELEASING.md` D2.)

## Why a self-hosted runner

The gate re-verifies that our five **Tier-1 (flagship) agents** — Claude Code,
Codex, Hermes, Aider, DeepSeek Harness (`dsh`) — actually work end-to-end
against a real local model on
Apple Silicon, on the *current* client binaries. GitHub-hosted runners cannot do
this: no Metal, no cached weights (a boot model is ~40–70 GB), and the agent
CLIs aren't installed. So the gate runs on a **self-hosted Apple-Silicon runner
(the "Studio", an M3 Ultra)**.

> **Two meanings of "Tier-1", don't conflate:** *Tier-1 model families*
> (Qwen 3.6 / Gemma 4 / DeepSeek / gpt-oss / Hy3) and *Tier-1 agents* (the 5
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
`codex-cli` / `hermes` / `aider` / `dsh`) reported `FAIL`.

1. First rule out a *model-strength* artifact — a weak model that fakes success
   is not an integration bug. The gate uses `qwen3.6-35b-8bit` (≥8-bit on
   purpose; never 4-bit, which confounds the two).
2. Localize with a wire replay: proxy the real client once, diff our SSE stream
   against what it expects, and pin the offending event/field.
3. Fix the integration (engine PR), then re-run: `gh run rerun <run-id>` or push
   again. Once green, the release proceeds through the signed Desktop candidate
   and `release-prep` (live main head + blocker evidence), then waits at the
   protected `rapid-mac-tag` gate for the reviewer's approval before anything is
   tagged or published.

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

The force bypasses **only the Tier-1 agent gate** (the Studio is what's down).
Every release-safety gate still applies and is mandatory: the Desktop candidate
must still pass the **signed/notarised/DMG-validated lane** at the exact
candidate SHA, the **live `main` head** must still equal the validated
candidate, the **live release-blocker set** must still be zero or explicitly
waived for this version, the **protected `rapid-mac-tag` environment approval**
is still required (and **admin bypass remains forbidden** for it). So a forced
release ships a version whose *agent integrations* were not re-verified at gate
time — the packaging and identity gates are unchanged. The bypass is
**audited**: the actor + reason are logged and stamped into the GitHub Release
notes as an "⚠️ Emergency release" banner. Use it sparingly.

## Normal retry after main drift (no bypass)

A different, everyday stall: the release aborted because `main` advanced past
the validated candidate just before the tag claim (the pre-tag TOCTOU check —
the candidate is no longer the live head). Nothing is broken; the ordering guard
is doing its job. To proceed you want a **normal** re-run of the full chain at
the *current* head — not the emergency bypass.

```bash
gh workflow run auto-release.yml -R raullenchai/Rapid-MLX \
  -f retry_version=X.Y.Z \
  -f reason="main advanced past the validated candidate; re-validating at new head"
```

`retry_version` runs on `main`, must equal `pyproject.toml`, and is **mutually
exclusive with `force_version`**. Unlike the emergency path it bypasses
**nothing**: `should_release=true` with `force=false`, so the Tier-1 gate, the
signed Desktop candidate at the new head, the live blocker/main-head evidence,
and the protected `rapid-mac-tag` approval are all re-required at the current
`main` head.

If the Desktop tag already exists at the accepted SHA, the tag claim is an
idempotent no-op, not publication evidence. The release job waits (bounded) for
an exact successful `rapid-mac-release.yml` run on that tag and a published,
non-empty canonical DMG, then re-resolves the immutable tag before it creates
the engine Release. A missing/failed tagged run, missing artifact, API/auth
failure, timeout, or SHA mismatch stops the engine half. Recover by rerunning
the failed exact tagged workflow (or explicitly dispatching it at
`--ref rapid-mac-vX.Y.Z[-rcN]`) and then rerunning auto-release; never move or
delete the tag.

## Related docs

- `docs/release-notes/README.md` — how curated release notes are written.
- `scripts/build_release_notes.sh` + `tests/release/test_build_release_notes.sh`
  — the notes builder and its offline tests.
- `tests/integrations/agent_smoke.sh` — the gate script (canonical).
- `tests/integrations/README.md` — the full agent × model integration matrix.
- `landing/runbooks/agent-release-verification.md` (rapidmlx.com repo) — the
  operational runbook + the freshness-stamp bump for the website.
- `.github/workflows/publish.yml` — PyPI publish on `release: published`.
