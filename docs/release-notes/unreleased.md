<!-- Scratch space for the next release's notes. Rename to vX.Y.Z.md in the
     version-bump PR. See README.md in this directory. -->

<!-- One or two sentences: what is this release about? -->

## Highlights

**<Named thing>** — what changed, and why a user should care. Link the merged PR.

- **Local web pages work from `localhost` and `127.0.0.1` alike**: a page served from `http://localhost:<port>` can call the server on the same port even when it is addressed as `127.0.0.1` or `[::1]`; other origins stay blocked.

- **Smaller Desktop download**: the bundled runtime no longer ships build-only headers, test suites, or Intel code slices, trimming about 14 MB from the DMG, about 110 MB and 11,000 files from the installed app.

- **Local Search works on stock Macs**: the default index location and index folders under `/tmp` or `/var` no longer fail with a 503 because of standard macOS deny ACLs or directory aliases. Unsafe ancestor permissions still fail with a message naming the affected path.
- **Faster opt-in MTP cohorts under load**: continuous self-MTP now forms a cohort only when at most 3 requests are live and never grows one past 3 lanes; with 4 or more concurrent requests, the rest use plain batched decode, which measured 23–24% faster than the previous cohort for short prompts and 6–13% faster at 8k–16k-token prompts.

- **Desktop no longer quits when you open a generated video**: clicking a finished video's preview now plays it in the app instead of crashing Rapid-MLX Desktop.
- **TensorFold lane setting no longer ignored**: setting `RAPID_MLX_TENSORFOLD_LANES` above 1, or to an invalid value, on a model that is not served through a qualified TensorFold profile now stops `rapid-mlx serve` at startup with an error instead of being silently ignored.

- **`/v1/responses` honors `max_tokens`**: it is treated as `max_output_tokens` instead of being ignored (which allowed unbounded replies); conflicting values return 400.

- **Cleaner tool-call replies**: unfinished tool payloads stay out of visible replies when arguments contain literal tool markers or a forced tool call is returned.
- **`agents codex --test` uses your saved server address**: after `rapid-mlx agents codex --setup --base-url <url>`, `rapid-mlx agents codex --test` now tests that saved address instead of always probing localhost:8000.
- **XML tool-call examples stay visible**: fenced code-block examples remain content even when they name an available tool, including in streamed replies; real calls after a closed fence still execute.
- **XML tool-call examples with CR line endings stay visible**: fenced examples written with bare CR or CRLF line endings remain content instead of running as tool calls, in streamed and non-streamed replies.
- **Leaner Desktop app bundle**: the app now ships only the math font it renders equations with, trimming about 6 MB from the installed app.
- **MTP defaults work with copied model caches**: aliases can resolve a sole cached snapshot without `refs/main`; unavailable or ambiguous configs now explain why plain decode is used.
- **Packaged license notices**: installed packages now include the license and notice files for the vendored speculative-decoding (MTP) code and the MiniCPM-V 4.7 vision model code.
- **Faster text decode on multimodal Qwen MoE models**: text-only requests on the default vision-capable lane of multimodal Qwen3.6-35B-A3B-class MoE checkpoints now keep single-request caches and receive MoE gate+up fusion. They no longer run slower than `--no-mllm`.
- **Oversized request errors reach clients**: requests whose bodies exceed the size cap now get an HTTP 413 JSON error naming the cap and `--max-request-bytes` instead of a connection reset while the upload is still in progress.
