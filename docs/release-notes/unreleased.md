<!-- Scratch space for the next release's notes. Rename to vX.Y.Z.md in the
     version-bump PR. See README.md in this directory. -->

<!-- One or two sentences: what is this release about? -->

## Highlights

**<Named thing>** — what changed, and why a user should care. Link the merged PR.

## Fixes

- **Quick Look in Local Search no longer crashes the app**: choosing Quick Look on a search result (or pressing Space) opens the preview reliably instead of occasionally quitting the app.

- **Fresh installs no longer break image generation**: a new release of the imaging library (Pillow 13.0.0) fails to import on supported Python versions, so a fresh install could pull it and the image runtime would fail to load. Rapid-MLX now keeps Pillow below 13 until a compatible release is available.
- **Desktop no longer quits when you open What's New after an update**: clicking "See what's new" on the post-update banner opens the release notes and dismisses the banner instead of crashing the app. The "Star on GitHub" prompt had the same defect and is fixed too.
