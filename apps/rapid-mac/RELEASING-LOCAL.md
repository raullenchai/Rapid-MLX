# Releasing Rapid-MLX Desktop: local dogfood + CI for public

Two lanes, split by **who the build is for**. The expensive, frequent case
(dogfood) runs on your Mac for $0; the rare, user-facing case (public) runs
on GitHub CI, which owns the distribution layer (auto-update + landing page).

```
                  Rapid-MLX Desktop release
                              │
                 ┌────────────┴─────────────┐
                 │  who is this build for?   │
                 └────────────┬─────────────┘
        "just me / a tester"          "all users"
                 │                          │
    ┌────────────▼───────────┐   ┌──────────▼─────────────┐
    │  DOGFOOD · local        │   │  PUBLIC · GitHub CI     │
    │  ~most builds           │   │  a few / month          │
    │  scripts/release-local  │   │  push a v* tag          │
    │  $0 CI                  │   │  → release.yml runs      │
    └────────────────────────┘   └────────────────────────┘
      makes a signed, testable       distributes it + writes
      .app/DMG on your Mac           latest.json (auto-update)
```

**Dividing line:** local = *"produce a signed app I can install and test."*
CI = *"ship it to everyone and let existing users auto-update."*

## What runs where

| Phase | Step | Dogfood (local) | Public (CI) |
|-------|------|:---:|:---:|
| Build | `swift build` → .app + bundled sidecar | ✅ your Mac | ✅ runner |
| Sign | codesign (Developer ID, or ad-hoc for local-only) | ✅ | ✅ |
| Notarise | notarytool + staple (.app & .dmg) | ✅ if `.p8` set up | ✅ CI secret |
| Package | `dmg.sh` + `validate-dmg.sh` | ✅ | ✅ |
| Size gates | 500 MB app cap / +50 MB DMG delta | ⬜ eyeball `du -sh` | ✅ enforced |
| Distribute | install locally / hand to a tester | ✅ manual | — |
| Distribute | GitHub Release with the DMG | ⬜ | ✅ |
| Distribute | slim bootstrapper DMG (new-user first run) | ⬜ | ✅ |
| Distribute | sidecar + quickstart-model tarballs → R2 | ⬜ | ✅ |
| Distribute | mirror DMG → `dl.rapidmlx.com` | ⬜ | ✅ |
| Distribute | write `latest.json` (in-app auto-update) | ⬜ | ✅ |

The top half (build/sign/notarise/package) is identical on both lanes — local
just moves it off the CI macOS runner onto your Mac. The bottom half (the
user-facing distribution layer, including the `latest.json` manifest the
in-app updater polls) stays in CI, where it runs in a clean, reproducible
environment with secrets injected per-run.

## Lane 1 — dogfood (the default, ~all the time)

```bash
scripts/release-local.sh            # build a signed DMG at build/rapid-mlx-desktop.dmg
scripts/release-local.sh --check    # verify signing/notary setup, build nothing
```

- No git tag, no GitHub Release, no CI, no R2. Costs $0.
- Signing degrades by what's set up, in three steps:
  - **Developer ID + notary key** (below) → notarised DMG, safe to hand to a tester.
  - **Developer ID, no notary key** → Developer-ID-signed but *un-notarised*
    (a warning is printed; installs with a right-click → Open, or a first-launch
    Gatekeeper prompt).
  - **No Developer ID** → ad-hoc signed — fine on your own Mac (right-click →
    Open), not Gatekeeper-valid on any other machine.
  A configured-but-missing key path warns and downgrades to un-notarised
  rather than aborting, so dogfood always produces a DMG.

## Lane 2 — public release (rare, reaches users)

```bash
# 1. Bump Resources/Info.plist + add the "## [X.Y.Z]" CHANGELOG.md section, on main.
# 2. Cut it:
scripts/release-local.sh --publish v0.10.4
```

`--publish` **does not build locally**. It preflights (stable `vX.Y.Z` tag —
no prerelease suffix, since the updater has no prerelease channel; CHANGELOG
entry exists; tag matches the plist version; local main == origin/main; tag is
new *and* strictly newer than the latest release) and then pushes the tag. `.github/workflows/release.yml` takes over and does the full
pipeline: build → sign → notarise → size gates → GitHub Release → slim DMG +
tarballs → mirror to `dl.rapidmlx.com` → `latest.json`. Existing users see the
update within ~6h. (You can equally just `git push origin v0.10.4` by hand;
`--publish` only adds the guardrails.)

`release.yml` is unchanged by this runbook — pushing a `v*` tag is still what
triggers a public release. The only discipline is: **push a tag only when you
mean to ship to users; dogfood with `release-local.sh` otherwise.**

## One-time setup (for notarised / public builds — do this yourself)

These scripts do not create, copy, print, or commit the private key. You
place the `.p8`; it is read by Apple's `notarytool` during notarisation, and
the scripts only pass its path. `~/.rapid-release.env` is sourced as shell,
so keep it owner-only (`chmod 600`) and never commit it.

1. **Developer ID Application** cert + key in your **login** keychain (double-
   click the `.p12`). Verify:
   ```bash
   security find-identity -v -p codesigning | grep "Developer ID Application"
   ```
2. **App Store Connect notary key**, placed by you:
   ```bash
   mkdir -p ~/.appstoreconnect/private_keys
   mv ~/Downloads/AuthKey_XXXX.p8 ~/.appstoreconnect/private_keys/
   chmod 600 ~/.appstoreconnect/private_keys/AuthKey_XXXX.p8
   ```
3. `~/.rapid-release.env` — copy `scripts/release-local.env.example`, then fill
   in your `AC_API_KEY_ID` (the `<KEYID>` in `AuthKey_<KEYID>.p8`, in both the
   id and the path lines) and `AC_API_ISSUER_ID` (App Store Connect → Users and
   Access → Integrations → Issuer ID). Never commit it.

CI's public-release path uses the repo's own Actions secrets, not this file.
