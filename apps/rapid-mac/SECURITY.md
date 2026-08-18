# Security Policy

We take security in Rapid-MLX Desktop seriously. The app touches your local
filesystem, your shell environment, your Keychain, and the network, and
runs as a signed + notarised macOS binary that auto-updates — every
surface matters.

## Reporting a vulnerability

**Do not file a public GitHub issue for security reports.**

Email **security@rapidmlx.com** with:

- A description of the vulnerability
- Steps to reproduce (a minimal Swift / shell snippet is ideal)
- Affected version (`Rapid-MLX Desktop → About` shows the build number, or
  `defaults read "/Applications/Rapid-MLX Desktop.app/Contents/Info" CFBundleShortVersionString`)
- Whether you would like public attribution

We aim to acknowledge reports within **3 business days** and ship a fix
or workaround in the next release cycle (typically 1-2 weeks for high
severity, 4 weeks for low severity).

## Supported versions

The latest minor release on the
[releases page](https://github.com/raullenchai/Rapid-MLX/releases/latest)
is the only version receiving security updates. We do not backport
fixes to older versions; please upgrade.

## Scope

In scope:

- The shipped `Rapid-MLX Desktop.app` binary
- The Cloudflare Workers under `*.rapidmlx.com` that the app talks to
  (telemetry, update checks)
- How Rapid-MLX Desktop spawns, sandboxes, and communicates with the
  `rapid-mlx` subprocess (`ServerLocator`, `ServerManager`, the
  loopback HTTP client) — independent of which `rapid-mlx` binary is
  resolved at launch. The bundled-sidecar slot itself will ship once
  Phase 5 (CI release integration) lands; until then the resolved
  binary comes from `$PATH` / Homebrew / pipx, and PRIVACY.md
  documents the slot order users can audit.

Out of scope:

- Vulnerabilities in upstream open-source dependencies — please report
  those to the upstream project (we will pull fixes once released)
- Vulnerabilities in user-installed `rapid-mlx` distributions
  (Homebrew tap, pipx, hand-built fork). We will help triage but the
  fix belongs in the upstream project at
  [github.com/raullenchai/Rapid-MLX](https://github.com/raullenchai/Rapid-MLX).
- Local privilege escalations that require root or full disk access
- Vulnerabilities in third-party search providers (Brave, Tavily) the
  user opts into for `web_search`

## Recognition

We list contributors who responsibly disclose issues, with their
permission, in the release notes for the fix.
