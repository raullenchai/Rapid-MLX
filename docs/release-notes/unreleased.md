<!-- Scratch space for the next release's notes. Rename to vX.Y.Z.md in the
     version-bump PR. See README.md in this directory. -->

<!-- One or two sentences: what is this release about? -->

## Highlights

**<Named thing>** — what changed, and why a user should care. Link the merged PR.

- **Buffered tool-text parity**: buffered `/v1/responses` and `/v1/messages` replies now return tool-call-like text unchanged when no tools are offered or `tool_choice` is `none`, matching streaming replies. Fixes #4901.
