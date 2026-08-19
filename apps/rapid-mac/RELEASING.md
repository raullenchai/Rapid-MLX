# Releasing the rapid-mac app

How to produce a signed + notarised `Rapid-MLX Desktop.app` / `.dmg` — both
the local dogfood build and the tag-triggered CI release — and exactly which
Apple credentials **you** must provision first.

Everything here is env-var / GitHub-Actions-secret driven. No secret value
lives in this repo; the scripts and workflow only reference secrets by name
and read key files by path. **Only you can create the Apple credentials** —
they are tied to your Apple Developer account.

> **Layout note (monorepo):** the Swift app is at `apps/rapid-mac/`; the
> `rapid-mlx` engine is the repository ROOT. The sidecar bundled into the app
> is pip-installed from that engine at the repo root (no git submodule). App
> release tags are prefixed **`rapid-mac-v`** so they never collide with the
> engine's own `v*` tags. The CI workflow lives at the repo root
> `.github/workflows/rapid-mac-release.yml` (GitHub only runs workflows from
> the repo root).

---

## Two lanes

Two lanes, split by **who the build is for**. The frequent case (dogfood)
runs on your Mac for $0; the rare, user-facing case (public) runs on GitHub CI,
which owns signing secrets and the distribution layer (auto-update feeds).

```
Rapid-MLX Desktop release — who is this build for?
│
├─ "just me / a tester"  →  DOGFOOD · local  (most builds, $0 CI)
│      scripts/release-local.sh
│      → a signed, testable .app/DMG on your Mac; no tag, no GitHub Release
│
└─ "all users"           →  PUBLIC · GitHub CI  (a few / month)
       push a rapid-mac-v* tag
       → rapid-mac-release.yml: build → sign → notarise → GitHub Release,
         plus the Sparkle appcast / latest.json auto-update feeds
```

A bare `v*` tag is the **engine's** release scheme, not the app's; the desktop
lane is gated on the `rapid-mac-v*` prefix so the two never collide in this
monorepo.

| | Dogfood (local) | Public (CI) |
|---|---|---|
| Command | `scripts/release-local.sh` | push a `rapid-mac-v*` tag |
| Runs on | your Mac ($0 CI) | `macos-15` GitHub runner |
| Build → sign → notarise → DMG | yes | yes |
| Attaches DMG to a GitHub Release | no | yes |
| Optional CDN mirror | no | yes (if configured) |

Dividing line: local = *"produce a signed app I can install and test."*
CI = *"ship it to everyone via a GitHub Release."*

---

## Part A — credentials only YOU can create (Apple account)

These require signing into your Apple Developer / App Store Connect account.
The scripts never create, print, or commit any of them.

### A1. Developer ID Application certificate (code signing)

1. In **Xcode → Settings → Accounts**, or at
   <https://developer.apple.com/account/resources/certificates>, create a
   **Developer ID Application** certificate. This signs the `.app`/`.dmg` for
   distribution outside the App Store.
2. It lands in your **login keychain**. Verify:
   ```bash
   security find-identity -v -p codesigning | grep "Developer ID Application"
   ```
3. **For CI:** export it (with its private key) to a `.p12`:
   Keychain Access → right-click the identity → *Export* → `.p12`, set an
   export password. Then base64-encode it for the GitHub secret:
   ```bash
   base64 -i DeveloperID.p12 | pbcopy    # → MACOS_DEVID_CERT_P12_BASE64
   ```
   The export password → `MACOS_DEVID_CERT_PASSWORD`.

### A2. App Store Connect API key (notarisation)

1. At <https://appstoreconnect.apple.com/access/integrations/api>, create an
   **App Store Connect API key** with the **Developer** role. Download the
   `AuthKey_<KEYID>.p8` (**one-time download** — Apple never shows it again).
2. Note two identifiers from that page:
   - **Key ID** — the `<KEYID>` in `AuthKey_<KEYID>.p8` → `AC_API_KEY_ID`
   - **Issuer ID** — the UUID at the top of the Keys page → `AC_API_ISSUER_ID`
3. **For CI:** base64-encode the `.p8`:
   ```bash
   base64 -i AuthKey_<KEYID>.p8 | pbcopy   # → AC_API_KEY_P8_BASE64
   ```

### A3. Apple Team ID

Your 10-character Team ID from
<https://developer.apple.com/account> (Membership details) → `APPLE_TEAM_ID`.

---

## Part B — GitHub Actions secrets (for the CI lane)

Add these under **Settings → Secrets and variables → Actions**. The workflow
`.github/workflows/rapid-mac-release.yml` references every one of them by name
only.

### Required Apple secrets (6)

| Secret name | Value (from Part A) |
|---|---|
| `MACOS_DEVID_CERT_P12_BASE64` | base64 of the Developer ID Application `.p12` (A1) |
| `MACOS_DEVID_CERT_PASSWORD` | the `.p12` export password (A1) |
| `APPLE_TEAM_ID` | 10-char Apple Team ID (A3) |
| `AC_API_KEY_ID` | App Store Connect key id (A2) |
| `AC_API_ISSUER_ID` | App Store Connect issuer id (A2) |
| `AC_API_KEY_P8_BASE64` | base64 of `AuthKey_<KEYID>.p8` (A2) |

### Required Sparkle signing key

Generate this once on a trusted Mac after `swift package resolve`:

```bash
cd apps/rapid-mac
.build/artifacts/sparkle/Sparkle/bin/generate_keys --account rapid-mlx
.build/artifacts/sparkle/Sparkle/bin/generate_keys --account rapid-mlx \
  -x ~/Desktop/rapid-mlx-sparkle-private-key
```

The first command prints the public key. Add it as the Actions **variable**
`SPARKLE_PUBLIC_ED_KEY`. Add the exact contents of the exported file as the
Actions **secret** `SPARKLE_ED_PRIVATE_KEY`, then move the exported file into
the team's protected credential store and remove the Desktop copy. Losing this
private key breaks the automatic-update chain for every installed version;
rotating it requires a signed transition release.

### Required for tagged releases — updater fallback publishing

| Name | Kind | Value |
|---|---|---|
| `CLOUDFLARE_API_TOKEN` | secret | token scoped to *Workers R2 Storage: Edit* and *Zone Cache Purge* |
| `CLOUDFLARE_ACCOUNT_ID` | secret | the Cloudflare account id wrangler operates against |
| `CLOUDFLARE_ZONE_ID` | secret | zone id owning `dl.rapidmlx.com`, used for single-file cache purge |
| `RAPID_MAC_DIST_R2_BUCKET` | **variable** | `rapid-desktop-dist` |
| `RAPID_MAC_DIST_CDN_BASE` | **variable** | `https://dl.rapidmlx.com` |
| `SPARKLE_ED_PRIVATE_KEY` | secret | exported Sparkle Ed25519 private key |
| `SPARKLE_PUBLIC_ED_KEY` | **variable** | matching `SUPublicEDKey` value |

The release jobs fail a tagged release if these are absent. They upload the
content-addressed DMG and Sparkle ZIP first, then queue every tag's monotonic
`latest.json` + `appcast.xml` publication, so neither updater can advertise a
missing artifact or roll back when releases overlap. The bucket/CDN and public
key values are
**config, not credentials**, so they go in *Variables*, not *Secrets*.

### Dropped

- `SENTRY_DSN` — **removed.** Sentry is being taken out of the app, so this
  secret (present in the old desktop workflow) is no longer plumbed here.
  Do not add it.

---

## Part C — local `~/.rapid-release.env` (for the dogfood lane)

For a **notarised** local build, `scripts/release-local.sh` reads notary
identifiers from `~/.rapid-release.env` and reads the `.p8` by path (Apple's
`notarytool` reads the key, not the script). This file holds **identifiers
only — never a private key.**

```bash
cp scripts/release-local.env.example ~/.rapid-release.env
chmod 600 ~/.rapid-release.env      # sourced as shell — keep owner-only
```

Then fill in (`~/.rapid-release.env`):

| Variable | Value |
|---|---|
| `AC_API_KEY_ID` | the `<KEYID>` in `AuthKey_<KEYID>.p8` (A2) |
| `AC_API_KEY_PATH` | path to the `.p8` you placed yourself (see below) |
| `AC_API_ISSUER_ID` | App Store Connect issuer id (A2) |
| `CODESIGN_IDENTITY` | *optional* — pin a specific Developer ID; blank auto-detects |

Place the `.p8` yourself (read by `notarytool`, never by the scripts):

```bash
mkdir -p ~/.appstoreconnect/private_keys
mv ~/Downloads/AuthKey_<KEYID>.p8 ~/.appstoreconnect/private_keys/
chmod 600 ~/.appstoreconnect/private_keys/AuthKey_<KEYID>.p8
```

`~/.rapid-release.env`, `*.p8`, `*.p12` are all git-ignored — never commit them.

---

## Part D — command flow

### D1. Local dogfood build

```bash
cd apps/rapid-mac
scripts/release-local.sh --check     # report signing/notary setup, build nothing
scripts/release-local.sh             # build a signed DMG at build/rapid-mlx-desktop.dmg
```

Signing degrades gracefully by what's configured:

- **Developer ID + notary key** → notarised DMG, safe to hand to a tester.
- **Developer ID, no notary key** → Developer-ID-signed but un-notarised
  (installs via right-click → Open).
- **No Developer ID** → ad-hoc signed — runs on *your* Mac only.

No tag, no GitHub Release, no CI. Costs $0.

### D2. Public CI release (tag-triggered)

```bash
# 1. Bump apps/rapid-mac/Resources/Info.plist:
#      CFBundleShortVersionString = X.Y.Z
#      CFBundleVersion = a strictly increasing positive integer
#    Then add a "## [X.Y.Z]" section to apps/rapid-mac/CHANGELOG.md, on main.
# 2. Cut it (guarded — preflights CHANGELOG/plist/tag, then pushes the tag):
cd apps/rapid-mac
scripts/release-local.sh --publish rapid-mac-v0.11.0
```

`--publish` does **not** build locally. It preflights (stable
`rapid-mac-vX.Y.Z` tag, CHANGELOG entry present, tag == plist version, monotonic
`CFBundleVersion`, local
`main` == the release remote's `main`, tag is new and strictly newer than the latest
rapid-mac release, and the root workflow triggers on the tag) then pushes the
tag. `.github/workflows/rapid-mac-release.yml` then builds → signs →
notarises → generates an EdDSA-signed Sparkle ZIP/appcast → size-gates →
publishes both updater feeds → attaches `rapid-mlx-desktop.dmg` to the GitHub
Release. The script auto-detects the remote whose URL is
`raullenchai/Rapid-MLX`; set `RAPID_RELEASE_REMOTE=<name>` to select it
explicitly. (You can equally `git push <release-remote> rapid-mac-v0.11.0` by hand;
`--publish` only adds the guardrails.)

Watch it:
```bash
gh run watch $(gh run list --workflow=rapid-mac-release.yml --limit=1 --json databaseId -q ".[0].databaseId")
```

---

## Who does what

| Step | Only you (Apple account) | Automated |
|---|---|---|
| Create Developer ID cert (A1) | ✅ | |
| Create App Store Connect API key (A2) | ✅ | |
| Export `.p12` / base64-encode keys | ✅ | |
| Add GitHub Actions secrets (Part B) | ✅ | |
| Place `.p8` + write `~/.rapid-release.env` (Part C) | ✅ | |
| Build / sign / notarise / staple / DMG | | ✅ scripts + CI |
| Attach DMG to the GitHub Release | | ✅ CI |

First installation is distributed as a direct `.dmg` download from GitHub
Releases; subsequent signed releases are also published as Sparkle ZIPs for
automatic update. There is no Homebrew cask. The `rapid-mlx` name is reserved
for the engine (`pip install rapid-mlx` and `brew install rapid-mlx`, the latter
already in homebrew-core); the desktop app never claims it.

## Open TODO(owner) items

- **Release repo owner is `raullenchai`** (`raullenchai/Rapid-MLX`).
- **Updater fallback credentials** (`CLOUDFLARE_API_TOKEN` /
  `CLOUDFLARE_ACCOUNT_ID` / `CLOUDFLARE_ZONE_ID`) must remain configured and scoped to the
  `rapid-desktop-dist` bucket. Tagged releases fail closed if publishing the
  mirrored DMG and `latest.json` is unavailable.
