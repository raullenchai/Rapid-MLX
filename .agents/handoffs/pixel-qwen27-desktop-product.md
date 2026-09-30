# Pixel handoff — Qwen 27B accelerated Desktop flow

- Owner: Pixel
- Branch: `pixel/qwen27-desktop-product`
- Base/dependency: `vector/qwen27-tensorfold-api` / backend product branch
- Host: Local Mac

## Verified facts

- The existing model picker already owns catalog selection, model download,
  startup progress, and ordinary-mode launch; no new setup page is needed.
- Settings → Performance already owns per-alias acceleration opt in/out,
  restart-required state, and live active/pending/unavailable status.
- The qualified accelerated profile is text-only, one request at a time, and
  reports tools, media, and grammar as unsupported. Ordinary mode remains the
  remedy for those capabilities. Only 48 GB has qualification evidence; the
  Desktop makes no 32 GB or universal speed claim.
- Desktop now parses an exact DFlash companion/backend preset from the catalog,
  emits the opaque `--speculative-config`, decodes live backend and unsupported
  feature fields, and names the ordinary-mode remedy when tools are selected.

## Reference check

- Orca/native Mac pattern retained: model-scoped settings, progressive
  disclosure, one switch, visible restart state, and inline status by the
  composer.
- Open WebUI, Jan, Cherry Studio, and LM Studio flows were reviewed for local
  model setup. The adopted pattern is selection from the normal model library,
  visible fit/download state, and advanced settings attached to the selected
  model. Separate raw paths and engine-specific JSON were rejected because the
  catalog already provides an exact alias and companion contract.

## Verification

- `swift test --disable-sandbox --filter ServerModelProfileTests` — 47 passed.
- `swift test --disable-sandbox --filter DownloadCatalogHardeningTests` — 22
  tests across the matched suites passed.
- Native Swift package build succeeded. No model was downloaded or launched.

## Remaining dependency

Vector must land the atomic catalog fields and live profile keys consumed here,
plus actionable startup failure markers for missing runtime/model states. Rebase
this branch onto that commit before review; then capture the actual Settings and
composer states with the bundled sidecar.
