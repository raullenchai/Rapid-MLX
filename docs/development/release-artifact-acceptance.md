# Release artifact acceptance

This document is the source of truth for the final quality gate between a
version-bump candidate and publication. It answers a narrower question than
the ordinary test suite: **does the exact wheel that will be uploaded behave
correctly when real users drive it through supported models, agents, and SDKs?**

`make release-smoke` remains useful, but import-only smoke cannot answer that
question. The artifact acceptance matrix must start the installed release
wheel from an empty directory and drive it over HTTP.

## Rollout status

This repository currently has no registered self-hosted release runner. This
change therefore ships the promotion workflow as **dispatch-only and dry-run
by default**: leave `publish` unchecked to validate a PR/tag candidate without
changing PyPI, GitHub Releases, or Homebrew. The existing release event path
has still been hardened with a manifest, least privilege, post-upload sdist
hash check, and immutable action pins; it is not yet made to wait forever for
a runner that does not exist.

Do not enable `publish` until the [activation transaction](#activation-transaction)
is complete. That is intentional: having both the legacy `publish.yml` and
the new promotion workflow upload the same immutable version would turn a
successful release into a duplicate-upload failure.

## Target promotion flow

```text
protected release PR on main
  -> build wheel + sdist once
  -> twine check + SHA-256 release manifest
  -> clean-room install/import/CLI smoke of both artifacts
  -> artifact matrix: installed wheel + real model + agent/framework cells
  -> PyPI Trusted Publishing (OIDC only in this job)
  -> download PyPI files; compare release manifest hashes and verify attestation
  -> publish GitHub Release; promote the verified sdist to Homebrew
```

The release PR merge is the single-maintainer approval point. The `pypi`
environment remains part of the PyPI Trusted Publisher identity and is limited
to the protected release source, but it does **not** need a second-person
reviewer.

## What must be tested

### 1. Both distribution formats

The clean-room smoke runs each candidate `rapid_mlx-*.whl` and
`rapid_mlx-*.tar.gz` in its own venv. It imports every base entrypoint module
and ensures the package was not shadowed by the checkout. The command is:

```bash
python scripts/release_smoke.py --dist-dir candidate/dist
```

The sdist check catches omitted package data and build-backend errors; the
wheel is the artifact used by the live matrix.

### 2. Agent and framework matrix from the installed wheel

The matrix is one server boot per family. A test run sets both
`RAPID_MLX_MATRIX_STRICT=1` and `RAPID_MLX_MATRIX_NO_SKIPS=1`:

- wrong model / unavailable server is a failure, not a skip;
- a missing SDK, Docker daemon, Aider executable, or other ordinary
  prerequisite is a failure, not a skip;
- a documented `strict=True` xfail remains an exception, and an unexpected
  pass is a failure.

The current feasible set is four families:

| Family | Candidate alias | Agent cells | Framework cells |
| --- | --- | ---: | ---: |
| Qwen | `qwen3.5-4b-4bit` | 11 | 3 |
| Gemma 4 | `gemma-4-12b-4bit` | 11 | 3 |
| DeepSeek | `deepseek-r1-32b-4bit` | 11 | 3 |
| gpt-oss | `gpt-oss-20b-mxfp4-q8` | 11 | 3 |
| **Total** | | **44** | **12** |

So the normal release gate contains **56 cells**. The 44-cell shorthand means
only the agent half; it must not be used to omit the 12 framework cells.

There are currently ten narrow, expected strict-xfails in those four families:
nine DeepSeek-R1-Distill tool-call cells and the gpt-oss × OpenHands format
mismatch. Their reason and upstream issue must remain in
`tests/integrations/conftest.py`; no broad `skip`, `xfail`, or `|| true` is
allowed in the release lane.

Hy3 adds 14 more cells (11 agent + 3 framework), but its 166 GB model requires
a dedicated Ultra host. Every release still runs its offline parser contract.
Any change touching Hy3, shared server loading, routing, tokenization, or
streaming additionally requires the real Hy3 matrix on that host before
promotion.

## Runner contract

`release-artifact-matrix.yml` is dispatch-only until a runner with these
labels exists:

```text
self-hosted, macOS, ARM64, rapid-mlx-release
```

It must be a dedicated Apple-Silicon release machine, not a developer's daily
workstation and not an operator host. The OpenHands cell deliberately mounts a
Docker socket into a pinned container; that makes an isolated daemon and
ephemeral workspace non-negotiable. Install or pre-provision:

- Python 3.12 and a current GitHub Actions runner;
- Docker Desktop/daemon, plus `docker info` access for the runner user;
- `coreutils` (`gtimeout` on macOS), used by the Aider harness;
- enough local disk for the four model caches plus a 100 GB safety floor.

The runner receives no PyPI token, OIDC permission, release-write token, or
Homebrew credential. It is only allowed to read the candidate artifact and
download public model data.

## Running a candidate

From GitHub Actions, dispatch **Release artifact acceptance matrix**, select
the exact release-PR SHA/ref, and keep the four default families. Diagnostic
runs may select a non-empty unique subset, but a `publish=true` run rejects
anything other than all four families exactly once. The workflow:

1. builds wheel and sdist once on GitHub-hosted Linux;
2. runs `twine check` and writes `release-manifest.json` with source SHA,
   version, size, and SHA-256 for both artifacts;
3. sends that artifact to the isolated macOS runner;
4. verifies the manifest before each matrix shard;
5. performs the clean-room smoke and runs the strict family matrix against
   the installed wheel.

For local runner diagnosis, use the same artifact rather than `pip install .`:

```bash
python scripts/release_manifest.py verify \
  --dist-dir candidate/dist --manifest candidate/release-manifest.json
python scripts/release_smoke.py --dist-dir candidate/dist
python scripts/release_artifact_matrix.py \
  --dist-dir candidate/dist --family qwen36 --port 18000
```

Pass `--keep-venv` to preserve the isolated venv and server log after a
failure. Never re-run the matrix from the checkout to diagnose a distribution
failure; doing so changes the thing being tested.

## Publication rules

After the runner is registered, artifact acceptance becomes a required release
check. The publishing job may consume only the artifact whose manifest was
accepted. It gets `id-token: write`; build, test, attestation, and Homebrew
jobs do not. The PyPI package is then downloaded again and its wheel/sdist
hashes must equal the manifest before GitHub Release or Homebrew promotion.

PyPI's automatic Trusted-Publisher attestation proves who uploaded each file.
GitHub artifact provenance and an SBOM are added alongside the manifest as the
next hardening step; they do not replace the live matrix.

## Activation transaction

Perform the following once a runner meeting the contract above is online. It
is a short, intentional repository-settings change; it needs no second
maintainer approval, no PyPI API token, and no release secret added to the
runner.

1. Register the dedicated runner with all four labels shown in the runner
   contract, then dispatch this workflow once with `publish=false` for a
   tagged candidate. Confirm all four shards pass.
2. In GitHub **Settings → Environments → `pypi`**, keep required reviewers
   empty. Set its deployment branch/tag policy to selected tags `v*`. A manual
   workflow run that publishes must itself be started from that same `vX.Y.Z`
   tag, not from `main`.
3. In PyPI’s Trusted Publisher settings, replace the current legacy workflow
   identity with repository `raullenchai/Rapid-MLX`, workflow
   `release-artifact-matrix.yml`, and environment `pypi`. Trusted Publishing
   then grants a short-lived token only to this workflow’s publish job.
4. In the same reviewed migration PR, retire the legacy
   `release: published → publish.yml` uploader and change auto-release to
   create a draft GitHub Release. The promotion workflow publishes that draft
   only after PyPI verification and Homebrew dispatch. Do these two changes
   together: keeping both uploaders active would attempt two uploads of the
   same immutable version.
5. Make the artifact-acceptance workflow a required release check/ruleset
   condition for the protected release PR. The condition is the matrix run
   against the release tag, not an arbitrary branch run.

After that migration, start the workflow from `vX.Y.Z`, set `ref` to exactly
`vX.Y.Z`, and set `publish=true`. The workflow itself rejects a publication
unless its input ref, workflow ref, and checked-out source SHA all point to
the tag matching `pyproject.toml`’s version.
The PyPI job is the sole job with `id-token: write`. The next job re-downloads
both PyPI files, checks their hash and size against the candidate manifest,
and verifies PyPI’s Sigstore publish attestations with
`pypi-attestations` before Homebrew receives the sdist URL and SHA-256.
