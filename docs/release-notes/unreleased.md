<!-- Scratch space for the next release's notes. Rename to vX.Y.Z.md in the
     version-bump PR. See README.md in this directory. -->

<!-- One or two sentences: what is this release about? -->

## Highlights

**<Named thing>** — what changed, and why a user should care. Link the merged PR.

- **Buffered tool-text parity**: buffered `/v1/responses` and `/v1/messages` replies now return tool-call-like text unchanged when no tools are offered or `tool_choice` is `none`, matching streaming replies. Fixes #4901.
- **Image edit auth first**: with `--api-key` set, `/v1/images/edits` now answers an unauthenticated request with 401 before the upload size limit or multipart parsing runs, so it no longer reveals 413/400/422 for request shape. Fixes #4850.
- **Chinese Desktop assistant identity**: Desktop's default assistant instruction now includes a Chinese line, so Chinese greetings describe what the assistant can write, translate, and explain instead of listing only its tools, and Chinese questions answered with a tool are more often answered in Chinese. Fixes #4907.
- **Interrupted replies stay visible**: switching chats while a reply is being written now keeps the partial reply marked as stopped with Retry instead of letting it vanish; earlier interrupted replies are restored on launch, and generated chat titles still arrive after you switch away. Fixes #4933, #4934.
