# Release artifact acceptance

Follow the [current release procedure](../../RELEASE.md) to cut a release. A
version-bump merge starts `auto-release.yml`: exact-commit CI, a real-model agent
gate, and the signed Desktop candidate gate run before the protected release
approval. Publication then runs through `publish.yml`.

## Inspect a candidate distribution

The repository also contains a dispatch-only
[release artifact matrix](../../.github/workflows/release-artifact-matrix.yml).
It builds a wheel and source distribution once, records their hashes, and runs
its model and client checks against the installed wheel. Its runner requirements
are defined in the workflow. This workflow is separate from the normal
`auto-release.yml` publication path.

For a local packaging check, run the clean-room smoke:

```bash
make release-smoke
```

To check distributions already built in `dist/`, run
`python3 scripts/release_smoke.py --dist-dir dist`. The smoke checks import and
packaging. It does not replace the real-model gates
in the [release procedure](../../RELEASE.md). Use the exact candidate files and
commit when comparing results.

The matrix's family selection comes from
[`scripts/release_fleet.json`](../../scripts/release_fleet.json). Check that
manifest and the workflow for current families, prerequisites, and dispatch
inputs before running it.
