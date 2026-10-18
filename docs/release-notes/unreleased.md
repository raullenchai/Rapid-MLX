<!-- Scratch space for the next release's notes. Rename to vX.Y.Z.md in the
     version-bump PR. See README.md in this directory. -->

<!-- One or two sentences: what is this release about? -->

## Highlights

- **Cleaner Desktop answers after tool use**: chats that reach their tool budget now send the final-answer instruction as the last message, and show an answer or a clear fallback instead of raw tool-call markup. Fixes #4871.
- **Start & Send**: when the chosen chat model is downloaded but stopped, Send starts it and sends the draft once it is ready; the draft stays put if startup is cancelled or fails. Fixes #4854.
- **Desktop current-info answers recover more often**: chat retries an unreadable tool request once, then explains when live information could not be checked. Weather lookup also retries unresolved simplified Chinese place names in traditional script.
- **Chat drafts stay with their conversation**: unsent text is saved locally across app restarts, with a separate draft for New Chat.
- **Desktop file paste**: pasting a file into the chat composer now attaches it, or explains which file types are supported, instead of inserting its local path.
- **Fewer unrelated Desktop tool calls**: plain writing and coding chats set off far fewer unrelated web searches and page visits, and no longer call a document reader when nothing is attached. Desktop offers the document reader only when the conversation has an attached file, and the per-message time note now shows only the local date and time, without the time-zone region. Fixes #4872.
- **Reliable automatic tool calls**: fixed empty replies when Qwen models use native XML tool calls; Desktop now gives accurate retry guidance when an empty reply stops before its token limit.
- **Image endpoint authentication**: image endpoints now require the API key when `--api-key` is set.
- **Personal Intelligence removed**: Desktop now uses the ordinary Chat path. `/v1/models` cards no longer include `personal_intelligence_profile` or `personal_intelligence_qualification`; `/v1/agent/runs` remains available with its existing runtime and client execution mode. Local file and command tools have been removed from Chat. Built-in tools are Web Search, Browse Web Page, Weather, and Read Attached Document again, plus any connectors you add. Files you created are not touched. Privacy copy now explains online tool data sharing.

- **LFM2.5-24B-A2B alias**: `rapid-mlx serve lfm2.5-24b-a2b-4bit` loads LiquidAI's official 4-bit MLX build with tool calling and reasoning parsing preconfigured.

- **Faster and bounded retrieval requests**: jina-reranker-v3 scores large requests faster. Rerank requests now default to 1000 documents and embeddings to 2048 inputs; excess returns HTTP 400 (override or disable with `--rerank-max-documents` / `--embedding-max-inputs`). Empty rerank documents or instruction return 400. A full retrieval queue returns 503 with `Retry-After: 1`.

- Faster text-only decode on the default vision-capable lane for Qwen3.6-35B-A3B-class multimodal MoE checkpoints.

- Vision-model servers now restore text-only prompt-prefix caches across restarts; image-bearing prefixes are never persisted.

**<Named thing>** — what changed, and why a user should care. Link the merged PR.

- **Local web pages work from `localhost` and `127.0.0.1` alike**: a page served from `http://localhost:<port>` can call the server on the same port even when it is addressed as `127.0.0.1` or `[::1]`; other origins stay blocked.

- **Smaller Desktop download**: the bundled runtime no longer ships build-only headers, test suites, or Intel code slices, trimming about 14 MB from the DMG, about 110 MB and 11,000 files from the installed app.

- **Local Search works on stock Macs**: the default index location and index folders under `/tmp` or `/var` no longer fail with a 503 because of standard macOS deny ACLs or directory aliases. Unsafe ancestor permissions still fail with a message naming the affected path.
- **Long prompts prefill concurrently when memory allows**: the scheduler no longer over-estimates prefill memory for long prompts (about 300 GB estimated vs about 45 GB measured for a 32k-token prompt on a 35B MoE hybrid model), so concurrent 32k-token requests are no longer forced to prefill one at a time.
- **Faster opt-in MTP cohorts under load**: continuous self-MTP now forms a cohort only when at most 3 requests are live and never grows one past 3 lanes; with 4 or more concurrent requests, the rest use plain batched decode, which measured 23–24% faster than the previous cohort for short prompts and 6–13% faster at 8k–16k-token prompts.

- **Neutral Desktop time context**: Desktop no longer labels each message with “This message was sent …”; the model sees a neutral local-time annotation, so email drafting requests are no longer answered as if the email had already been sent, and current-time answers use the latest message’s time. Fixes #4859.
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
- **Multimodal Qwen MoE short blocks match the unfused model exactly**: with MoE gate+up fusion on the vision-capable lane, 2–8-token blocks (prefill tails and speculative verify) now produce byte-identical output to the unfused model, without extra weight copies.
- **Oversized request errors reach clients**: when a request body exceeds the size cap, the server now reads and discards the rest of the upload (about 64 MiB at most, within 10 s) and then sends an HTTP 413 JSON error naming the cap and `--max-request-bytes`, so clients still uploading get the error instead of a connection reset.
- **Accurate health probes after idle unload**: `/healthz` now reports `model_loaded: false` while an idle-unloaded primary model is in standby, matching `/metrics`, and the server logs unload and on-demand reload transitions.
- **Document reranking with `/v1/rerank`**: start the server with `--rerank-model <hf-id>` to score and sort documents against a query (Qwen3-Reranker, mxbai-rerank-v2, jina-reranker-v3, BGE reranker v2 Gemma and XLM-RoBERTa cross-encoders such as bge-reranker-v2-m3). It runs alongside a chat model, an embedding model, or on its own. Embedding and rerank requests now run off the request loop, so other requests and health checks keep responding while they compute. Contributed by @weizenspreu.
