# Homebrew Cask distribution

`rapid-mlx.rb` is the Homebrew Cask manifest for the signed +
notarised `.dmg` attached to each GitHub Release cut by
`.github/workflows/rapid-mac-release.yml` (tag prefix `rapid-mac-v*`,
asset name `rapid-mlx-desktop.dmg`).

> **Note:** the release/tap repo owner is `raullenchai`
> (`raullenchai/Rapid-MLX`), used throughout the commands below and in
> `rapid-mlx.rb`.

## End-user install

Once the tap is published, users install with:

```bash
brew tap raullenchai/tap
brew install --cask rapid-mlx
```

## Publishing the tap (one-time setup)

Homebrew expects custom taps in a repo named `homebrew-<tapname>`:

1. Create a new public GitHub repo at `raullenchai/homebrew-tap`.
2. Copy `Casks/rapid-mlx.rb` from this repo to `Casks/rapid-mlx.rb`
   in `raullenchai/homebrew-tap`.
3. Push. End users can now `brew tap raullenchai/tap` (Homebrew expands
   that to the `homebrew-tap` repo).

You can keep the canonical copy in this repo (`Casks/rapid-mlx.rb`)
and have a release-time script copy it to the tap repo, OR make
`homebrew-tap` the only home and remove this file. The repo-local
copy here is the convenient development surface.

## Updating on each release

After cutting a new tag (e.g. `rapid-mac-v0.11.0`):

```bash
set -euo pipefail
VERSION=0.11.0
OWNER=raullenchai

# 1. Pull the new DMG SHA from the GitHub Release
SHA=$(curl -fsSL "https://github.com/${OWNER}/Rapid-MLX/releases/download/rapid-mac-v${VERSION}/rapid-mlx-desktop.dmg" \
    | shasum -a 256 | awk '{print $1}')

# 2. Bump the cask
sed -i.bak \
    -e "s/version \"[^\"]*\"/version \"${VERSION}\"/" \
    -e "s/sha256 \"[^\"]*\"/sha256 \"${SHA}\"/" \
    Casks/rapid-mlx.rb
rm Casks/rapid-mlx.rb.bak

# 3. Commit + push to BOTH this repo and the tap repo
```

`brew bump-cask-pr` from the Homebrew CLI can automate steps 1+2 plus
opening a PR when targeting the main `homebrew-cask` repo. For a
custom tap, the sed shape above is the simplest.

## Submitting to the main `homebrew-cask` repo

Homebrew's main `homebrew-cask` repo requires notability thresholds
(75+ stars, 30+ forks, 30+ watchers for regular submissions; 225 / 90 /
90 for self-submissions per
[Acceptable Casks](https://docs.brew.sh/Acceptable-Casks)). Once the
release repo clears those, run `brew create --cask rapid-mlx`
against the latest DMG and open a PR to `Homebrew/homebrew-cask`. Until
then the custom tap is the right distribution channel.

## Cask invariants

- `verified:` pins the GitHub repo path so a future hostname change
  doesn't silently swap the source.
- `livecheck strategy: :github_latest` + the `rapid-mac-v<semver>`
  regex makes `brew livecheck rapid-mlx` detect new **app** releases
  without being confused by the engine's own releases in the same
  monorepo.
- `depends_on macos: :sonoma` because the SwiftUI surface relies on
  macOS 14+ APIs (`NavigationSplitView`, observation-tracking macros).
- `auto_updates true` because the app's built-in updater handles
  upgrades; Homebrew should not fight it.
