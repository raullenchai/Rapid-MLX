<!-- Scratch space for the next release's notes. Rename to vX.Y.Z.md in the
     version-bump PR. See README.md in this directory. -->

<!-- One or two sentences: what is this release about? -->

## Highlights

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

## Model catalog

- `qwopus-27b-8bit` is retired: its upstream repository is no longer
  accessible, so a fresh pull could only fail. Use `qwopus-27b-4bit` (the
  4-bit build of the same model from the same publisher).
- `qwen3.8-27b-abliterated-4bit` now pulls from
  `windowsxp811203/Qwen3.8-27B-Abliterated-MLX-oQ4e-mtp`, where the publisher
  moved the identical oQ4e build after removing the old multi-build repository.
