<!-- Scratch space for the next release's notes. Rename to vX.Y.Z.md in the
     version-bump PR. See README.md in this directory. -->

<!-- One or two sentences: what is this release about? -->

## Highlights

**<Named thing>** — what changed, and why a user should care. Link the merged PR.

- **Buffered tool-text parity**: buffered `/v1/responses` and `/v1/messages` replies now return tool-call-like text unchanged when no tools are offered or `tool_choice` is `none`, matching streaming replies. Fixes #4901.
- **Image edit auth first**: with `--api-key` set, `/v1/images/edits` now answers an unauthenticated request with 401 before the upload size limit or multipart parsing runs, so it no longer reveals 413/400/422 for request shape. Fixes #4850.
- **Chinese Desktop assistant identity**: Desktop's default assistant instruction now includes a Chinese line, so Chinese greetings describe what the assistant can write, translate, and explain instead of listing only its tools, and Chinese questions answered with a tool are more often answered in Chinese. Fixes #4907.
- **Interrupted replies stay visible**: switching chats while a reply is being written now keeps the partial reply marked as stopped with Retry instead of letting it vanish; earlier interrupted replies are restored on launch, and generated chat titles still arrive after you switch away. Fixes #4933, #4934.
- **TensorFold timeout release**: on the default serial TensorFold path, a non-streaming request that times out now cancels its generation, and server shutdown cancels in-flight jobs, so later requests and process exit no longer hang behind a timed-out job. Fixes #4848.
- **Large tool lists on native-XML models**: with `tool_choice: "auto"`, models whose chat template writes tool calls in native XML (such as the Qwen3.5 family) no longer return HTTP 400 for tool lists over the grammar limits (256 tools, 64 KiB or depth 32). The request runs on the normal tool-call parsing path, up to 1024 tools, 256 KiB or depth 64. Fixes #4918.
- **Calmer date and weather answers**: asking for today's date or the time no longer shows the "may be a guess" caution, since the app already gives the model the current date and time; asking for weather in a state or region now shows the tool step as a quiet "needs a more specific place" note instead of a red failure card above the model's which-city question. Real tool failures still show the red card. Fixes #4935, #4936.
- **Chat titles and suggestions follow the user**: Desktop now cues conversation titles from the script of the first message, so Chinese chats get Chinese sidebar titles. A title with no Chinese, Japanese or Korean characters for a CJK chat, or a mostly-CJK title for any other chat, is discarded in favor of the opening words. Follow-up suggestions are prompted from the user's side, and suggestions that read as the assistant offering help, such as "Do you need…?" or "还需要我…吗？", are no longer shown. Fixes #4937, #4938.
