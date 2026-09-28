<!-- Scratch space for the next release's notes. Rename to vX.Y.Z.md in the
     version-bump PR. See README.md in this directory. -->

<!-- One or two sentences: what is this release about? -->

## Highlights

**Privacy-safe capability rejection detail** — `capability_rejected` can now
include an optional closed `reject_reason` for structured-output and
context-length rejections. The field contains only registry-approved values
and never includes error messages or other free-form text.

**<Named thing>** — what changed, and why a user should care. Include the
numbers if there are numbers, and the caveat if there is a caveat. ([#1234](https://github.com/raullenchai/Rapid-MLX/pull/1234))

| Context | Prefill tok/s | Decode tok/s |
| ------: | ------------: | -----------: |
|      1K |               |              |
|    128K |               |              |

| Workload | Off | On | Change | Acceptance |
| -------- | --: | -: | -----: | ---------: |
| Code     |     |    |        |            |
| Prose    |     |    |        |            |
