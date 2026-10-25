# Bundled runtime dependencies

Public requirements use compatibility ranges; the Desktop CPython 3.12 runtime
uses `apps/rapid-mac/scripts/sidecar-constraints.txt` and
`sidecar-requirements.lock`. The lock covers the complete index dependency graph
for the base package plus `audio-desktop` and `computer-use`. Every index package
has an exact version and SHA-256 artifact hashes. Builds require wheels and target
macOS 14 arm64, independent of the build host's OS version.

The application source/candidate wheel and the separately commit-pinned runtime
are installed with `--no-deps` after the locked index closure. Their source
provenance is managed separately; the index lock does not authenticate those
sources or isolated build-tool dependencies. The existing distribution-coherence
and runtime import gates still run before packaging.

## Reviewed versions and updates

The initial lock retains the existing Desktop pins, including torch 2.14.1 and
its matching torchvision. It selects tiktoken 0.13.0 (the tokenizer regression
baseline), safetensors 0.8.0, and MCP's 1.x compatibility line. Public torch
requirements retain the restricted-loading safety floor of 2.10.0 or higher.
All other direct base dependencies have a next-major compatibility ceiling
(next-minor for pre-1.0 packages). Existing narrower bounds remain unchanged.

To update, review upstream release notes and dependency metadata, change the
selected versions in the constraints, and regenerate from the repository root:

```sh
MACOSX_DEPLOYMENT_TARGET=14.0 uv pip compile pyproject.toml \
  --extra audio-desktop --extra computer-use \
  --constraint apps/rapid-mac/scripts/sidecar-constraints.txt \
  --python-version 3.12 --python-platform aarch64-apple-darwin \
  --generate-hashes --no-header --no-annotate \
  --output-file apps/rapid-mac/scripts/sidecar-requirements.lock
```

Review every changed version/hash and synchronize new transitive pins into the
constraints. Hashes are captured from the package index at review time; subsequent
builds compare downloaded bytes to the committed values. Keep the immutable
runtime's declared dependencies compatible with this closure when updating its
source pin. Run distribution/lock tests, tokenizer regressions, focused Swift
sidecar tests and the Desktop build. A lock bump must not lower the torch safety
floor. Missing wheels, dependency conflicts, missing pins and wrong hashes must
fail instead of falling back to an unbounded install.
