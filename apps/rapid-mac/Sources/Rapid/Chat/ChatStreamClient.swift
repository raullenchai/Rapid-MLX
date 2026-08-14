import Foundation

/// Streaming client for the local rapid-mlx ``/v1/chat/completions``
/// endpoint. We deliberately re-implement this (rather than pulling in
/// an OpenAI SDK) because:
///
/// 1. The dependency surface needs to stay tiny — Tauri v0.1's only
///    Swift port goal was "zero non-Apple deps" and we keep that.
/// 2. mlx-lm hybrid models emit ``reasoning_content`` deltas which most
///    third-party SDKs don't surface in their typed structs (they only
///    decode the OpenAI-canonical ``content`` field). Owning the parser
///    lets us route reasoning into its own UI lane.
///
/// The client is single-shot: one ``send(...)`` per assistant turn. To
/// stop mid-stream the caller cancels the returned ``Task``; the SSE
/// loop sees ``Task.isCancelled`` between line reads and tears down the
/// underlying ``URLSessionDataTask``.
struct ChatStreamClient {
    /// Per-line byte cap for the SSE reader. A misbehaving or
    /// malicious server can stream bytes without ever emitting a
    /// newline, and the default `URLSession.AsyncBytes.lines`
    /// buffers a `String` internally until a `\n` arrives — an
    /// unbounded server-side write turns into an unbounded
    /// client-side allocation (OOM). 1 MiB is well above any
    /// realistic chunk: a 50K-token completion fits in ~150KB of
    /// JSON, an error envelope is rarely >2KB. Reaching the cap is
    /// a strong signal the peer is malicious or broken, not slow,
    /// so we surface a transport error rather than truncating.
    /// Audit P1 (`ChatStreamClient.send():150`).
    static let maxSSELineBytes: Int = 1 << 20

    /// Base URL of the local rapid-mlx server. Defaults to
    /// ``Self.defaultBaseURL`` (derived from
    /// ``PortSweep.defaultPort``). ``ChatViewModel`` re-targets this
    /// onto ``ServerManager.activePort`` before every send via
    /// ``Self.loopbackURL(port:)`` so a PortAllocator fallback off
    /// the default port still reaches the live child.
    var baseURL: URL

    /// Per-request INACTIVITY deadline — the max time we wait for the
    /// *next* byte from the server, reset every time data arrives (this
    /// is what ``URLRequest.timeoutInterval`` means for a streaming
    /// ``bytes(for:)`` task, NOT a total-response cap). The session's
    /// ``timeoutIntervalForResource`` is the total-response cap; a long
    /// reasoning trace that keeps emitting tokens resets THIS timer on
    /// every delta, so it is never clipped by it.
    ///
    /// The only legitimate silent window against a LOCAL loopback model
    /// server is the first token's prefill (before any byte arrives);
    /// once tokens start, inter-token gaps are milliseconds. Even a cold
    /// prefill of a normal chat prompt on the largest supported models
    /// stays well under this bound (a 27B on a 4K-token prompt tops out
    /// ~80 s). 180 s leaves generous headroom for that while still
    /// catching a genuinely wedged/silent server — the mid-session
    /// model-switch hang users hit — in ~3 min instead of the prior 10.
    ///
    /// Pre-fix this was 600 s (a remote-API-shaped value): a server that
    /// went silent looked like a permanent hang, and the user force-quit
    /// long before it fired. At 180 s the timeout surfaces as an
    /// actionable, retryable failure row (``FailureDiagnoser`` maps
    /// ``NSURLErrorTimedOut`` → ``.requestFailed`` + a Retry action)
    /// instead of a dead, error-less spinner.
    var requestTimeout: TimeInterval = 180

    /// What the SSE loop emits to the caller. Each event is delivered on
    /// the main actor so callers can mutate ``@Observable`` state without
    /// hopping.
    enum Event: Sendable {
        /// The first generated token of the turn has arrived, on whichever
        /// lane carried it. Emitted exactly once, before the payload event
        /// that triggered it, and never for a turn that generates nothing.
        ///
        /// This exists because time-to-first-token is a property of the
        /// STREAM, not of any one lane, and only this client sees all
        /// three. A caller stamping the clock in its own `.content` and
        /// `.reasoning` handlers misses a turn whose first output is a
        /// tool-call fragment — it would then time the later prose instead
        /// and report a decode window that excludes real generation,
        /// inflating the rate.
        ///
        /// Deliberately NOT routed through the coalescer: it marks an
        /// instant, not a payload, and batching it would move the very
        /// measurement it exists to take. For the same reason the instant
        /// travels *in* the event, sampled where the delta was recognised,
        /// rather than being read by the handler once it is finally
        /// scheduled.
        case firstToken(ContinuousClock.Instant)
        /// A delta to the visible ``content`` lane.
        case content(String)
        /// A delta to the hybrid-thinking ``reasoning_content`` lane.
        case reasoning(String)
        /// Final set of tool calls produced this turn. Emitted exactly
        /// once just before ``.finished(reason: "tool_calls")`` so the
        /// caller can capture the finalised calls and run them.
        case toolCalls([ToolCall])
        /// v0.4.13: terminal ``usage`` block from a server that honours
        /// ``stream_options.include_usage``. Per OpenAI spec this
        /// arrives as the last chunk before ``[DONE]``. Servers that
        /// don't honour the option simply never emit this event —
        /// the caller falls back to the v0.4.12 char-count estimate.
        case usage(promptTokens: Int, completionTokens: Int)
        /// Server reported ``finish_reason`` and the stream ended cleanly.
        case finished(reason: String?)
    }

    /// Request shape for a single chat turn. We don't accept the whole
    /// OpenAI surface — just the knobs the desktop UI surfaces.
    ///
    /// Sampling knob choices:
    ///
    ///   * ``repetitionPenalty: 1.1`` — non-zero is mandatory on
    ///     small fine-tuned hybrid models (Qwopus 9B/27B, etc.)
    ///     which otherwise degenerate into endless "智能X: a, b, c"
    ///     style list cycles without ever emitting EOS, blowing
    ///     past ``maxTokens``. The 1.1 value is what BCG ships for
    ///     the same reason. Verified against Qwopus repro.
    ///   * ``frequencyPenalty: 0.0`` / ``presencePenalty: 0.0`` —
    ///     OFF by default as of v0.4.1. The pre-v0.4.1 build
    ///     defaulted these to ``1.0`` / ``0.5`` as belt-and-
    ///     suspenders against repetition, but live testing on
    ///     ``qwen3.5-4b`` showed the combined penalty was strong
    ///     enough to push small 4-bit models into emoji /
    ///     non-Latin Unicode space (the model gets "punished" for
    ///     re-using any token it has emitted, so it picks
    ///     ever-rarer ones to dodge the penalty until the output
    ///     looks like vocabulary garbage). ``repetition_penalty``
    ///     alone catches the same loop pathology without the
    ///     scatter side-effect, so we drop fp/pp to neutral and
    ///     let advanced users re-enable them per-call if they ever
    ///     need to. Reference: ChatGPT-Desktop / Claude Desktop
    ///     ship fp=0, pp=0 defaults for the same reason.
    ///   * ``maxTokens: 4096`` — pre-v0.4.1 used 2048 which a
    ///     small hybrid model could blow through inside the
    ///     reasoning block alone, surfacing the dreaded
    ///     "Reached max_tokens before any output" error on a
    ///     one-line user question. 4096 gives reasoning models
    ///     enough headroom to emit a normal-length final answer
    ///     without changing the per-turn cost ceiling
    ///     dramatically.
    struct Request: Sendable {
        let alias: String
        let messages: [Wire.Message]
        let temperature: Double
        let topP: Double
        let maxTokens: Int
        let repetitionPenalty: Double
        let frequencyPenalty: Double
        let presencePenalty: Double
        let tools: [ToolDefinition]?
        /// #161: when ``false`` the wire body carries
        /// ``chat_template_kwargs: {enable_thinking: false}`` so the
        /// chat template skips the hybrid ``<think>...</think>``
        /// block. Non-hybrid models ignore the kwarg.
        let enableThinking: Bool
        /// #141: when non-nil, the wire body emits
        /// ``"tool_choice": {"type":"function","function":{"name":<forcedTool>}}``
        /// instead of the default ``"auto"`` string. The chat view
        /// sets this when a send originates from an empty-state
        /// capability chip whose CTA promises a specific tool
        /// (Search the web → web_search, Weather → weather, etc.) —
        /// cross-model probe found qwen3.6-35b misrouting the
        /// chip's prompt to ``get_datetime`` ~50% of the time
        /// without the explicit bias. Free-typed prompts leave this
        /// nil and keep ``tool_choice=auto``.
        let forcedTool: String?

        init(
            alias: String,
            messages: [ChatMessage],
            temperature: Double = 0.7,
            topP: Double = 0.95,
            maxTokens: Int = 4096,
            repetitionPenalty: Double = 1.1,
            frequencyPenalty: Double = 0.0,
            presencePenalty: Double = 0.0,
            tools: [ToolDefinition]? = nil,
            enableThinking: Bool = false,
            forcedTool: String? = nil
        ) {
            self.alias = alias
            let includeImages = ModelBrandStyle.supportsImageInput(forAlias: alias)
            self.messages = messages.map {
                Wire.Message(from: $0, includeImages: includeImages)
            }
            self.temperature = temperature
            self.topP = topP
            self.maxTokens = maxTokens
            self.repetitionPenalty = repetitionPenalty
            self.frequencyPenalty = frequencyPenalty
            self.presencePenalty = presencePenalty
            self.tools = tools
            self.enableThinking = enableThinking
            self.forcedTool = forcedTool
        }
    }

    /// Override hook for tests — let a fake URLProtocol session
    /// intercept the request so we can assert wire-body contents
    /// without booting a real HTTP server. Production code path
    /// uses ``Self.sharedSession``.
    var injectedSession: URLSession?

    /// Process-wide ephemeral session reused across every
    /// production call. Ephemeral config (no on-disk cache, no
    /// cookie store). Held statically so consecutive turns +
    /// tool-loop iterations reuse the same HTTP/2 connection instead
    /// of paying a fresh TLS handshake each time.
    /// [codex audit r1 ChatStreamClient.swift:175]
    ///
    /// Two distinct timeouts (see ``requestTimeout`` for the full
    /// rationale):
    ///   * ``timeoutIntervalForRequest`` — INACTIVITY between packets,
    ///     reset on each received byte. 180 s bounds the max silent
    ///     window (prefill / a wedged server) without clipping an
    ///     actively-streaming response. Individual sends override this
    ///     via ``req.timeoutInterval`` = ``requestTimeout``; kept in
    ///     sync here so a caller that reuses this session without a
    ///     per-request override still gets the bounded policy.
    ///   * ``timeoutIntervalForResource`` — TOTAL wall-clock cap for the
    ///     whole streamed response (10 min). A response still actively
    ///     streaming past 10 min IS terminated here — this is the
    ///     absolute worst-case bound, not "unlimited". 10 min of
    ///     continuous local generation is far beyond any normal chat
    ///     turn, so in practice the 180 s inactivity timer (silence, not
    ///     length) is what fires; this cap only backstops a pathological
    ///     never-ending stream.
    static let sharedSession: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 180
        config.timeoutIntervalForResource = 600
        return URLSession(configuration: config)
    }()

    /// Build a loopback URL on the given port. Single helper used by
    /// both ``defaultBaseURL`` AND ``ChatViewModel``'s re-target
    /// site, so scheme + host are written exactly once in the app.
    /// Codex r1 BLOCKING — the prior shape left
    /// `"http://127.0.0.1:\(server.activePort)"` duplicated in
    /// ChatViewModel, so a future scheme/host change (e.g. switch to
    /// `localhost`) would silently drift between the two sites.
    static func loopbackURL(port: Int) -> URL {
        // `URL(string:)` for a literal "http://127.0.0.1:<int>" can
        // never fail; force-unwrap is the right shape here.
        URL(string: "http://127.0.0.1:\(port)")!
    }

    /// The chat-completions endpoint for a port-only base URL.
    ///
    /// Written exactly once, for the same reason `loopbackURL` is: the base
    /// URL every caller holds is `http://127.0.0.1:<port>` with NO path, so
    /// each one has to remember to add `v1/`. One caller (since removed) did
    /// not, POSTed to `/chat/completions`, got a 404 from the engine and
    /// reported its own failure as the model's — a shipped feature that could
    /// never once have worked (#1668). Anything that talks to the local
    /// engine's chat endpoint goes through here.
    static func chatCompletionsURL(base: URL) -> URL {
        base.appendingPathComponent("v1/chat/completions")
    }

    /// Default startup-time base URL — pulled from
    /// `PortSweep.defaultPort` (the single source of truth) so the
    /// literal port doesn't live in two places. ChatViewModel
    /// re-targets this onto `ServerManager.activePort` BEFORE the
    /// first request, so this default is only the bootstrapping
    /// guess for callers that construct the client without a live
    /// `ServerManager` (e.g. TestDriver). Audit P1.
    static let defaultBaseURL: URL = loopbackURL(port: PortSweep.defaultPort)

    /// URLError codes that are safe to replay BEFORE any token
    /// bytes have arrived. Server-side mutation has not yet
    /// happened, the assistant placeholder is empty, and the
    /// usual cause is a brief restart window (model swap,
    /// PortAllocator fallback). Once the SSE loop has started
    /// consuming bytes we never retry — the assistant row may
    /// hold partial content and the server has no resume.
    ///
    /// Conservative set:
    /// * `.cannotConnectToHost` — TCP refused. Usual cause on
    ///   loopback is "server restarting after model swap" or
    ///   "PortAllocator just shifted ports".
    /// * `.networkConnectionLost` — connection died after the
    ///   handshake but before bytes arrived. Rare on loopback;
    ///   common on WiFi.
    /// * `.dnsLookupFailed` — DNS hiccup. Doesn't apply to
    ///   `127.0.0.1` but harmless to include for non-default
    ///   `baseURL` callers (TestDriver, future remote-tunnel).
    /// Explicitly NOT retried:
    /// * `.timedOut` — the request already got its full inactivity
    ///   budget (``requestTimeout``); a timeout means the server is
    ///   genuinely hung, so retrying just compounds the wait. Surface
    ///   it as an actionable, retryable failure instead.
    /// * `.notConnectedToInternet` — the user's WiFi is off; a
    ///   200ms retry won't fix that and just delays the error
    ///   surface.
    /// * `.cancelled` — never retry user-initiated cancels.
    static let retryableURLErrorCodes: Set<URLError.Code> = [
        .cannotConnectToHost,
        .networkConnectionLost,
        .dnsLookupFailed,
    ]

    /// Audit P1 (`ChatStreamClient — no retry on transient
    /// network errors`). One pre-stream retry on a brief delay
    /// when `session.bytes(for:)` throws a code from
    /// `retryableURLErrorCodes`. Returns the SAME tuple shape as
    /// the underlying call, so the SSE loop downstream is
    /// unchanged.
    ///
    /// Bounded — one retry max, then the original error
    /// propagates. Worst-case added latency is one short sleep
    /// (~250 ms). Idempotent in spirit because the SSE chat
    /// completion has not started any server-side decoding yet
    /// when this fires.
    ///
    /// Test seam: ``retryDelay`` is overridable so unit tests
    /// don't pay the wall-clock cost.
    static let defaultRetryDelay: Duration = .milliseconds(250)

    static func openBytesWithRetry(
        session: URLSession,
        request: URLRequest,
        retryDelay: Duration = defaultRetryDelay
    ) async throws -> (URLSession.AsyncBytes, URLResponse) {
        do {
            return try await session.bytes(for: request)
        } catch let urlError as URLError where
            retryableURLErrorCodes.contains(urlError.code) {
            // Sleep first so a tight refused-connection retry
            // doesn't immediately hit the same server-restart
            // window. `Task.sleep` honours cancellation — if the
            // user hits Stop during the delay we throw
            // CancellationError, which the caller already
            // distinguishes from a transport error.
            try await Task.sleep(for: retryDelay)
            return try await session.bytes(for: request)
        }
    }

    init(
        baseURL: URL = ChatStreamClient.defaultBaseURL,
        session: URLSession? = nil
    ) {
        self.baseURL = baseURL
        self.injectedSession = session
    }

    /// Open a streaming chat completion. ``onEvent`` is called on the
    /// main actor for every parsed delta and once at the end with
    /// ``.finished``. Throws ``ChatStreamError`` on transport / parse
    /// failure or ``CancellationError`` if the surrounding ``Task`` was
    /// cancelled mid-stream.
    func send(
        _ request: Request,
        bearerToken: String? = nil,
        onEvent: @escaping @MainActor (Event) -> Void
    ) async throws {
        let url = Self.chatCompletionsURL(base: baseURL)
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        // #17 desktop-half: per-launch bearer secret. ChatViewModel
        // passes ``server.activeBearer`` here; the embedded
        // rapid-mlx checks the matching ``RAPID_MLX_API_KEY`` env.
        // Anything else hitting :<port>/v1/chat/completions without
        // this header lands on 401.
        if let bearerToken, !bearerToken.isEmpty {
            req.setValue("Bearer \(bearerToken)", forHTTPHeaderField: "Authorization")
        }
        req.timeoutInterval = requestTimeout

        let body = Wire.ChatCompletionRequest(
            model: request.alias,
            messages: request.messages,
            stream: true,
            temperature: request.temperature,
            top_p: request.topP,
            max_tokens: request.maxTokens,
            repetition_penalty: request.repetitionPenalty,
            frequency_penalty: request.frequencyPenalty,
            presence_penalty: request.presencePenalty,
            tools: (request.tools?.isEmpty == false) ? request.tools : nil,
            tool_choice: Wire.ToolChoice.resolve(
                hasTools: request.tools?.isEmpty == false,
                forcedTool: request.forcedTool
            ),
            stream_options: .init(include_usage: true),
            // #161: only emit the kwarg when thinking is OFF. Sending
            // it when ON would be a no-op for hybrid models (the
            // chat-template default is thinking ON), and for
            // non-hybrid models the kwarg is silently ignored by
            // their Jinja chat template either way.
            chat_template_kwargs: request.enableThinking
                ? nil
                : .init(enable_thinking: false)
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = []
        req.httpBody = try encoder.encode(body)

        // URLSession.shared inherits app-level timeouts which can be
        // shorter than we want; we keep a process-wide shared
        // ephemeral session with the right timeout policy so HTTP/2
        // connection + TLS handshake survive across tool-loop
        // iterations and consecutive user turns. Test rigs inject
        // their own session via ``injectedSession``.
        //
        // Codex audit r1 (ChatStreamClient.swift:175): the previous
        // shape constructed and invalidated a fresh URLSession per
        // call, defeating connection reuse.
        let session: URLSession = injectedSession ?? Self.sharedSession

        // Audit P1 — pre-stream transient retry. If the connection
        // dies BEFORE any token bytes arrive (server is restarting
        // after a model swap, PortAllocator just shifted from :8000
        // to :8001 and the route table hasn't caught up, brief
        // loopback flake during a model swap mid-send), one short
        // retry typically masks the blip invisibly. Mid-stream
        // retries are unsafe — the assistant placeholder may
        // already hold partial bytes and the server has no resume
        // semantic, so we ONLY retry the initial `session.bytes`
        // dispatch.
        let (bytes, response) = try await Self.openBytesWithRetry(
            session: session,
            request: req
        )
        guard let http = response as? HTTPURLResponse else {
            throw ChatStreamError.transport("non-HTTP response")
        }
        guard (200..<300).contains(http.statusCode) else {
            // Drain whatever the server sent so we can surface its error
            // text in the UI. Most rapid-mlx 4xx/5xx replies are JSON, but
            // we treat the body as opaque text for the error message.
            var text = ""
            for try await line in bytes.lines {
                text += line + "\n"
                if text.count > 4096 { break }
            }
            throw ChatStreamError.httpStatus(http.statusCode, text.trimmingCharacters(in: .whitespacesAndNewlines))
        }

        // SSE framing: each event is one or more ``field: value`` lines
        // terminated by a blank line. We only care about ``data:`` lines.
        // For chat completions, every data line is one chunk's JSON
        // (rapid-mlx emits one event per chunk) so we don't need to
        // accumulate multi-line data fields.
        let decoder = JSONDecoder()
        // Tool calls are partitioned across many SSE chunks; the
        // first carries id + name, subsequent chunks append fragments
        // of arguments. Accumulate here and emit one final
        // ``.toolCalls`` event when finish_reason: "tool_calls" arrives.
        var toolAcc = ToolCallAccumulator()
        // #896: did the stream produce a terminal event before EOF?
        // ``[DONE]`` or any chunk with ``finish_reason`` flips this
        // to true. If we exit the loop with it still false, the
        // backend died mid-response and we throw rather than
        // silently emit ``.finished(nil)``.
        var sawTerminalEvent = false
        // v0.4.13: when ``stream_options.include_usage`` is on, the
        // OpenAI spec emits the usage chunk AFTER finish_reason and
        // BEFORE [DONE]. Returning early on finish_reason (the
        // v0.4.12 behaviour) skipped the usage chunk entirely.
        // Instead we now stash the reason and keep reading until
        // [DONE] or EOF — emitting ``.finished`` exactly once, at
        // the very end, after any usage event has already landed.
        // Callers that listen for ``.usage`` therefore always see
        // it BEFORE the terminal ``.finished``. ``finalizedTools``
        // guards against finalising the tool-call accumulator twice
        // if more chunks land after the finish_reason chunk
        // (shouldn't happen, but spec-tolerant beats silent loss).
        var capturedFinishReason: String?
        var finalizedTools = false
        /// One-shot latch for ``Event/firstToken``.
        var sawFirstGeneratedDelta = false
        // Audit P1 — SSE delta coalescing. Per-delta MainActor.run on
        // a fast stream (M3 Ultra rapid-mlx can decode 200+ tok/s)
        // burned a main-actor hop per token. Coalesce content/reasoning
        // deltas into a single hop per window so a 500-token response
        // that previously cost ~500 hops collapses to ~30 — while the
        // first delta of each type still surfaces immediately so the
        // typing indicator appears without perceptible delay.
        //
        // The window is 16 ms (one display frame at 60 Hz) and widens
        // with the length already sent, to 250 ms — see
        // ``SSEDeltaCoalescer/currentWindowNs()`` for why a fixed
        // window leaves the turn quadratic (#1743).
        //
        // Invariants the coalescer preserves:
        //   * First content/reasoning delta flushes BEFORE any
        //     accumulation (zero perceived first-token latency).
        //   * Any non-delta event (usage, tool_calls, finish_reason,
        //     [DONE]) flushes pending buffers BEFORE itself so callers
        //     never see a stale tail after a terminal.
        //   * Reasoning and content buffers are independent — a
        //     reasoning-only delta doesn't drain pending content and
        //     vice versa.
        //
        // Held as a class so the captured mutation across `await`
        // boundaries doesn't trip Swift 6 strict concurrency.
        let coalescer = SSEDeltaCoalescer()

        // Codex r1 BLOCKING-1: every throw path out of the SSE loop
        // (cancellation, line-too-long, mid-stream error envelope)
        // must drain pending coalesced text BEFORE rethrowing so the
        // caller's placeholder keeps the partial reply the server
        // already sent. The do/catch wraps the entire loop body;
        // the cleanup path emits whatever's pending then rethrows.
        do {
        for try await line in bytes.boundedLines(maxLineBytes: Self.maxSSELineBytes) {
            try Task.checkCancellation()
            // Heartbeats, comments, headers: skip.
            guard line.hasPrefix("data:") else { continue }
            let payload = line.dropFirst("data:".count).trimmingCharacters(in: .whitespaces)
            if payload == "[DONE]" {
                sawTerminalEvent = true
                // Coalescer invariant: drain pending content/reasoning
                // BEFORE the terminal .finished so the caller never
                // sees a stale tail after the end-of-stream marker.
                await coalescer.flush(onEvent: onEvent)
                let reason = capturedFinishReason
                await MainActor.run { onEvent(.finished(reason: reason)) }
                return
            }
            guard let payloadData = payload.data(using: .utf8) else { continue }
            let chunk: Wire.StreamChunk
            do {
                chunk = try decoder.decode(Wire.StreamChunk.self, from: payloadData)
            } catch {
                // Codex audit r1 (ChatStreamClient.swift:245): rapid-mlx
                // and other OpenAI-compatible servers signal mid-stream
                // failures with a ``{"error": {"message": "..."}}``
                // envelope rather than a closed HTTP connection. The
                // previous shape decoded that into a parse error and
                // silently skipped — the UI saw a clean empty
                // completion. Try the error envelope BEFORE giving up
                // so the user sees the real reason.
                if let env = try? decoder.decode(Wire.ErrorEnvelope.self, from: payloadData),
                   let message = env.error.message, !message.isEmpty {
                    throw ChatStreamError.transport(message)
                }
                // Tolerate genuinely malformed lines — the spec says we
                // MUST ignore unparseable events.
                continue
            }
            // v0.4.13: the OpenAI streaming-usage extension emits one
            // extra chunk with ``choices: []`` and ``usage`` populated.
            // Hand it to the caller and continue — the real terminal
            // event is still the ``[DONE]`` sentinel that follows it.
            if let usage = chunk.usage {
                // Coalescer invariant: usage is a non-delta event; flush
                // pending content/reasoning first so callers never see a
                // stale tail after the usage callback.
                await coalescer.flush(onEvent: onEvent)
                await MainActor.run {
                    onEvent(.usage(
                        promptTokens: usage.prompt_tokens,
                        completionTokens: usage.completion_tokens
                    ))
                }
            }
            for choice in chunk.choices {
                if let delta = choice.delta {
                    let generated = !(delta.reasoning_content ?? "").isEmpty
                        || !(delta.content ?? "").isEmpty
                        || !(delta.tool_calls ?? []).isEmpty
                    if generated, !sawFirstGeneratedDelta {
                        sawFirstGeneratedDelta = true
                        // Sampled HERE, in the parse, and carried across —
                        // NOT read on the far side of the hop below.
                        // `MainActor.run` queues behind whatever else the UI
                        // is doing, and the stream keeps filling URLSession's
                        // buffer while it waits. A stamp taken after the hop
                        // charges that delay to prefill instead of decode,
                        // shrinking the decode window and inflating the rate:
                        // the same direction of error, from the same cause,
                        // that this event was added to remove.
                        let at = ContinuousClock.now
                        await MainActor.run { onEvent(.firstToken(at)) }
                    }
                    if let r = delta.reasoning_content, !r.isEmpty {
                        await coalescer.appendReasoning(r, onEvent: onEvent)
                    }
                    if let c = delta.content, !c.isEmpty {
                        await coalescer.appendContent(c, onEvent: onEvent)
                    }
                    if let deltas = delta.tool_calls {
                        for d in deltas { toolAcc.accept(d) }
                    }
                }
                if let reason = choice.finish_reason {
                    sawTerminalEvent = true
                    capturedFinishReason = reason
                    if reason == "tool_calls" && !finalizedTools {
                        // Coalescer invariant: drain pending content/
                        // reasoning before the .toolCalls callback so
                        // the caller sees ``.content … .toolCalls`` in
                        // the original server order.
                        await coalescer.flush(onEvent: onEvent)
                        let calls = toolAcc.finalize()
                        finalizedTools = true
                        if !calls.isEmpty {
                            await MainActor.run { onEvent(.toolCalls(calls)) }
                        }
                    }
                    // Don't ``return`` here — keep reading so the
                    // optional usage chunk + [DONE] can land. If
                    // the server only emits finish_reason and EOFs
                    // (some non-OpenAI servers do this), the
                    // ``sawTerminalEvent`` gate at the bottom keeps
                    // us from misclassifying it as a crash, and the
                    // bottom-of-loop ``.finished`` emit covers the
                    // caller contract.
                }
            }
        }
        // #896 + v0.4.13: the stream EOF'd. Three cases:
        //   1. EOF after a finish_reason chunk but no [DONE] sentinel
        //      (some non-OpenAI servers) — synthesise the .finished
        //      event here so the caller still sees a terminal event,
        //      and DO NOT throw (the response was clean).
        //   2. EOF with neither finish_reason nor [DONE] — the
        //      backend died mid-response. Throw streamTruncated so
        //      the catch path in ``runOneStream`` marks the
        //      in-flight assistant message ``.failed`` and the
        //      crash banner has somewhere to point.
        //   3. (Reached only when [DONE] arrived: we already
        //      returned above.)
        if sawTerminalEvent {
            // Coalescer invariant: EOF-after-finish path must also
            // drain pending text before the terminal .finished.
            await coalescer.flush(onEvent: onEvent)
            await MainActor.run { onEvent(.finished(reason: capturedFinishReason)) }
            return
        }
        // EOF mid-stream: drain whatever the caller has accumulated
        // so the failure UI shows the tail of partial output (the
        // catch path in ``runOneStream`` marks the placeholder
        // ``.failed`` but keeps the prefix it received).
        await coalescer.flush(onEvent: onEvent)
        throw ChatStreamError.streamTruncated
        } catch {
            // Codex r1 BLOCKING-1: cancellation / boundedLines /
            // error-envelope throw paths land here. Drain pending
            // coalesced text BEFORE the rethrow so the caller's
            // assistant placeholder keeps the partial reply it
            // already received (the existing catch in
            // ``ChatViewModel.runOneStream`` then marks it
            // ``.failed`` but the surfaced prefix is preserved).
            await coalescer.flush(onEvent: onEvent)
            throw error
        }
    }
}

/// `AsyncSequence` wrapper that yields newline-terminated `String`
/// chunks from any underlying byte stream, while capping the number
/// of bytes that may accumulate before a `\n` is seen. The standard
/// `URLSession.AsyncBytes.lines` is unbounded — a peer that sends a
/// gigabyte without a newline forces a gigabyte allocation. This
/// wrapper throws `ChatStreamError.transport(...)` the moment the
/// buffer crosses `maxLineBytes` so a malicious server can't OOM
/// the renderer.
///
/// CRLF is normalised: a trailing `\r` immediately before the `\n`
/// is stripped before the `String` is yielded, matching the
/// behaviour of `AsyncLineSequence`.
///
/// EOF behaviour mirrors `AsyncLineSequence`: a non-empty tail
/// without a trailing `\n` is yielded once before the iterator
/// returns `nil`. The cap still applies to the tail.
struct BoundedLinesSequence<Base: AsyncSequence>: AsyncSequence where Base.Element == UInt8 {
    typealias Element = String

    let base: Base
    let maxLineBytes: Int

    struct AsyncIterator: AsyncIteratorProtocol {
        var iterator: Base.AsyncIterator
        let maxLineBytes: Int
        var buffer: [UInt8] = []

        mutating func next() async throws -> String? {
            // `keepingCapacity: true` so the geometric-growth buffer
            // (up to ~2× maxLineBytes peak) is reused across lines —
            // avoids per-line realloc churn on long streams. The
            // trade-off is a constant ~2 MiB-per-idle-connection
            // resident, acceptable for a single in-flight SSE loop.
            buffer.removeAll(keepingCapacity: true)
            while let byte = try await iterator.next() {
                if byte == 0x0A {
                    if buffer.last == 0x0D { buffer.removeLast() }
                    return String(decoding: buffer, as: UTF8.self)
                }
                buffer.append(byte)
                if buffer.count > maxLineBytes {
                    throw ChatStreamError.transport(
                        "SSE line exceeded \(maxLineBytes)-byte cap; closing stream to prevent OOM"
                    )
                }
                // Codex r1 NIT-3: a hostile server feeding 1 MiB
                // byte-by-byte holds this loop without yielding to
                // the surrounding `for try await` consumer, which
                // means the consumer's `Task.checkCancellation()`
                // doesn't fire until throw / line break. Probe
                // cancellation periodically (every 4 KiB) so a
                // cancelled `Task` tears down the connection
                // promptly even mid-line.
                if buffer.count & 0xFFF == 0 {
                    try Task.checkCancellation()
                }
            }
            if buffer.isEmpty { return nil }
            // Tail without a `\n`: the in-loop cap is the invariant
            // that bounds `buffer.count` at this point, so no second
            // cap check is needed here. `keepingCapacity: false`
            // because the iterator is about to return `nil` next call
            // — release the backing store eagerly so a long-lived
            // owning `Task` doesn't hold ~2 MiB after the stream is
            // done.
            let tail = String(decoding: buffer, as: UTF8.self)
            buffer.removeAll(keepingCapacity: false)
            return tail
        }
    }

    func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(iterator: base.makeAsyncIterator(), maxLineBytes: maxLineBytes)
    }
}

extension AsyncSequence where Element == UInt8 {
    /// Apply a per-line byte cap to a raw byte stream. See
    /// `BoundedLinesSequence` for rationale and semantics.
    func boundedLines(maxLineBytes: Int) -> BoundedLinesSequence<Self> {
        BoundedLinesSequence(base: self, maxLineBytes: maxLineBytes)
    }
}

enum ChatStreamError: LocalizedError {
    case transport(String)
    case httpStatus(Int, String)
    /// The SSE stream closed before any terminal event (either an
    /// OpenAI-style ``[DONE]`` sentinel or a chunk carrying
    /// ``finish_reason``) was observed. v0.4.4 silently treated
    /// this as success — but the only realistic cause is the
    /// backend dying mid-response (rapid-mlx SIGKILL'd by the
    /// kernel for OOM, the user, or a model-load crash). #896:
    /// surface it as a transport failure so the assistant
    /// message is marked ``.failed`` and the crash banner has
    /// somewhere to point.
    case streamTruncated

    var errorDescription: String? {
        switch self {
        case .transport(let m): return m
        case .httpStatus(let code, let body):
            if body.isEmpty { return "rapid-mlx returned HTTP \(code)" }
            return "rapid-mlx returned HTTP \(code): \(body)"
        case .streamTruncated:
            return "rapid-mlx closed the stream mid-response (likely a crash). Restart the server and resend."
        }
    }
}

/// On-the-wire shapes. Kept inside ``ChatStreamClient`` so the rest of
/// the app deals with the public ``ChatMessage`` type and the SSE event
/// enum, not these Codable details.
enum Wire {
    struct ChatCompletionRequest: Encodable {
        let model: String
        let messages: [Message]
        let stream: Bool
        let temperature: Double
        let top_p: Double
        let max_tokens: Int
        // Repetition / frequency / presence penalties — defaults
        // chosen to break degenerate token-loop cycles on small
        // hybrid fine-tunes (Qwopus 9B/27B). rapid-mlx server
        // accepts all three independently; setting to 0 / 1.0
        // disables each.
        let repetition_penalty: Double
        let frequency_penalty: Double
        let presence_penalty: Double
        // ``tools`` and ``tool_choice`` are only included when the
        // caller actually has tools to expose. We deliberately skip
        // ``parallel_tool_calls`` — the rapid-mlx server defaults it
        // to ``true`` and the UI is happy to render N parallel
        // calls as N chips.
        let tools: [ToolDefinition]?
        let tool_choice: ToolChoice?
        // v0.4.13: opt into the OpenAI streaming-usage extension.
        // Per spec, when ``stream`` is true AND
        // ``stream_options.include_usage`` is true, the server emits
        // one extra terminal chunk where ``choices`` is empty and
        // ``usage`` carries ``prompt_tokens`` + ``completion_tokens``
        // + ``total_tokens``. rapid-mlx supports this; servers that
        // don't recognise the field treat it as unknown JSON and
        // ignore it (validated against mlx-lm + OpenAI-API).
        let stream_options: StreamOptions
        /// #161: passes ``{"enable_thinking": false}`` through the
        /// Jinja chat template so hybrid models (Qwen 3 / 3.5 / 3.6,
        /// GLM 4.7, Qwopus) skip the ``<think>...</think>`` block
        /// and emit the answer directly. Non-hybrid templates ignore
        /// unknown variables. Encoded only when non-nil so the wire
        /// body stays clean when thinking is opt-in.
        let chat_template_kwargs: ChatTemplateKwargs?

        struct StreamOptions: Encodable {
            let include_usage: Bool
        }

        struct ChatTemplateKwargs: Encodable, Sendable {
            let enable_thinking: Bool
        }
    }

    /// OpenAI ``tool_choice`` field: string (``"auto"`` / ``"none"`` /
    /// ``"required"``) OR a typed-function object that pins the next
    /// turn to one specific tool. #141: the empty-state capability
    /// chips ("Search the web", "Weather", …) populate the typed
    /// form so cross-model probes (qwen3.6-35b in particular) can't
    /// silently route the chip's promised intent through the wrong
    /// tool. Free-typed prompts keep the ``"auto"`` string form.
    enum ToolChoice: Encodable, Sendable, Equatable {
        case auto
        case function(name: String)

        private struct FunctionForm: Encodable {
            let type: String
            let function: NamePayload
            struct NamePayload: Encodable {
                let name: String
            }
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.singleValueContainer()
            switch self {
            case .auto:
                try c.encode("auto")
            case .function(let name):
                try c.encode(FunctionForm(
                    type: "function",
                    function: .init(name: name)
                ))
            }
        }

        /// Pick the wire ``tool_choice`` value for the next chat
        /// turn. Centralises the "tools present + caller forced a
        /// specific tool" decision so the encode site and the unit
        /// tests both lean on the same helper.
        ///
        /// Contract:
        ///   * No tools advertised → ``nil`` (the field is omitted).
        ///   * Tools advertised + ``forcedTool`` nil → ``.auto``.
        ///   * Tools advertised + ``forcedTool`` non-empty →
        ///     ``.function(name:)`` with that name.
        ///   * ``forcedTool`` an empty/whitespace string degrades to
        ///     ``.auto`` rather than emitting ``function.name: ""`` —
        ///     a server-side validator would reject the bad shape
        ///     and break the send.
        static func resolve(hasTools: Bool, forcedTool: String?) -> ToolChoice? {
            guard hasTools else { return nil }
            if let forcedTool, !forcedTool.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return .function(name: forcedTool)
            }
            return .auto
        }
    }

    /// One message on the wire. ``content`` follows the OpenAI shape
    /// where the field accepts either a plain string OR an array of
    /// typed parts (text + image_url) — we pick at encode time based
    /// on whether the source message carries image attachments.
    struct Message: Encodable {
        let role: String
        let content: MessageContent
        let tool_calls: [ToolCall]?
        let tool_call_id: String?

        init(from message: ChatMessage, includeImages: Bool = true) {
            self.role = message.role.rawValue
            let modelContent = message.modelContent
            if message.imageAttachments.isEmpty || !includeImages {
                self.content = .text(modelContent)
            } else {
                var parts: [ContentPart] = []
                if !modelContent.isEmpty {
                    parts.append(.init(type: "text", text: modelContent, image_url: nil))
                }
                parts.append(contentsOf: message.imageAttachments.map {
                    .init(type: "image_url", text: nil, image_url: .init(url: $0.dataURL))
                })
                self.content = .parts(parts)
            }
            self.tool_calls = (message.toolCalls?.isEmpty == false) ? message.toolCalls : nil
            self.tool_call_id = message.toolCallID
        }

        enum CodingKeys: String, CodingKey {
            case role, content, tool_calls, tool_call_id
        }
    }

    /// Sum type encoded either as a string (legacy plain content) or
    /// as a JSON array of typed parts (multimodal). OpenAI accepts
    /// both shapes interchangeably.
    enum MessageContent: Encodable {
        case text(String)
        case parts([ContentPart])

        func encode(to encoder: Encoder) throws {
            var c = encoder.singleValueContainer()
            switch self {
            case .text(let s): try c.encode(s)
            case .parts(let arr): try c.encode(arr)
            }
        }
    }

    struct ContentPart: Encodable {
        let type: String
        let text: String?
        let image_url: ImageURL?

        struct ImageURL: Encodable {
            let url: String
        }
    }

    struct StreamChunk: Decodable {
        let choices: [Choice]
        /// v0.4.13: present on the terminal usage chunk emitted by
        /// servers that honour ``stream_options.include_usage``. Per
        /// OpenAI spec this is the last chunk before ``[DONE]`` and
        /// has ``choices: []`` plus a populated ``usage`` block.
        /// Older / non-conforming servers never emit this — leaving
        /// the field ``nil`` everywhere, which the caller handles
        /// gracefully (falls back to the char-count estimate).
        let usage: Usage?
    }

    struct Choice: Decodable {
        /// Codex audit r1 (ChatStreamClient.swift:583): some servers
        /// emit a terminal chunk that carries ONLY ``finish_reason``
        /// with the ``delta`` field omitted (e.g. ``{"index":0,
        /// "finish_reason":"stop"}``). The pre-audit shape made
        /// ``delta`` required, so the JSONDecoder threw and the
        /// whole chunk was dropped — and the terminal reason with
        /// it. Making delta optional lets the finish_reason path
        /// fire independently of the per-token delta path.
        let delta: Delta?
        let finish_reason: String?
    }

    struct Delta: Decodable {
        let content: String?
        let reasoning_content: String?
        let tool_calls: [ToolCallDelta]?
    }

    /// Codex audit r1 (ChatStreamClient.swift:245): error envelope
    /// shape emitted by rapid-mlx / OpenAI-compatible servers in
    /// the SSE body when a mid-stream failure occurs. Decoded as a
    /// fallback when ``StreamChunk`` rejects the payload so the UI
    /// surfaces the real reason instead of a silent empty bubble.
    struct ErrorEnvelope: Decodable {
        let error: ErrorBody
        struct ErrorBody: Decodable {
            let message: String?
            let type: String?
            let code: String?
        }
    }

    struct Usage: Decodable {
        let prompt_tokens: Int
        let completion_tokens: Int
        // ``total_tokens`` is on the wire but we don't surface it —
        // the chat UI shows prompt + completion separately if at
        // all, and total = prompt + completion is recoverable from
        // those when needed.
    }
}

// MARK: - SSE delta coalescer (audit P1)

/// Reduces ``MainActor.run`` traffic on the SSE hot path.
///
/// Per-token MainActor hops on a fast stream (M3 Ultra rapid-mlx
/// at 200+ tok/s) burned a hop per emitted delta. This coalescer
/// accumulates ``content`` and ``reasoning_content`` deltas inside
/// a coalescing window and emits ONE MainActor callback per window
/// — collapsing a 500-token stream from ~500 hops to ~30.
///
/// The window starts at 16 ms (one display frame at 60 Hz) and, past
/// ``adaptiveThresholdChars``, widens in proportion to the text already
/// flushed, up to ``maxWindowNs``. A fixed window would leave the turn
/// quadratic: each repaint re-parses the whole message, so an O(length)
/// parse at a fixed rate costs O(length²) over the turn (#1743).
///
/// Held as a reference type so the captured mutable state survives
/// the ``await`` boundary the SSE loop crosses on every line
/// without tripping Swift 6 strict-concurrency captures.
///
/// Contract (codex r1 hardened):
///   * ``appendContent`` / ``appendReasoning`` push text into a
///     SINGLE ordered queue keyed by kind, merging into the
///     trailing entry when the kind matches. The first call of
///     each kind surfaces IMMEDIATELY (first-token visibility —
///     the typing indicator must not wait a frame). Subsequent
///     calls flush once the current window (see
///     ``currentWindowNs()``) has elapsed.
///   * ``flush`` emits the pending queue IN ORDER, so a
///     content→reasoning interleave preserves the server's wire
///     order — there is no cross-kind reordering.
///   * Callers MUST invoke ``flush`` before any non-delta event
///     (usage, tool_calls, finish_reason, [DONE]) AND before
///     throwing on cancellation or transport error. The send()
///     scope guarantees the latter via a `defer` block.
///   * The window is opportunistic (checked on each append), not
///     timed, and its length is adaptive — 16 ms until the message
///     passes ``adaptiveThresholdChars``, then proportional to the
///     text so far up to ``maxWindowNs`` (#1743): if a delta burst stops mid-window,
///     the trailing text is held until the next event of any
///     kind or until send()'s defer flush. Real backends emit
///     a continuous stream of deltas terminated by
///     `finish_reason` + `[DONE]`, so this corner case only
///     surfaces on a hung/aborted backend — where the defer
///     flush in send()'s catch path still drains the tail.
///
/// Annotated ``@unchecked Sendable`` because the SSE loop is the
/// sole owner — each call to ``send`` constructs its own instance
/// and never hands it off across tasks.
final class SSEDeltaCoalescer: @unchecked Sendable {
    /// 16 ms = 1 frame at 60 Hz. Tight enough that the user
    /// perceives a smooth typing effect; loose enough to amortise
    /// MainActor hop cost.
    ///
    /// This is the FLOOR, not the whole story — see ``currentWindowNs``.
    private static let coalesceWindowNs: UInt64 = 16_000_000

    /// Above this many flushed characters the window starts to widen.
    /// Chosen so an ordinary reply never changes behaviour at all: the
    /// median chat answer is a few hundred characters and the long tail
    /// of "write me a function" answers still lands under 2 000.
    private static let adaptiveThresholdChars: Int = 2_000

    /// Ceiling on the widened window.
    ///
    /// **2026-08 (#1843): 250 ms → 16 ms**, which collapses the widening curve
    /// entirely — the window is now flat at `coalesceWindowNs` for every
    /// message length. The ramp machinery is kept rather than deleted so the
    /// behaviour can be restored by changing one number if a profile ever
    /// justifies it again.
    ///
    /// The ramp existed to contain `LaTeXMarkdownView` re-parsing the whole
    /// accumulated message through MarkdownUI on every flush — O(length²)
    /// across a turn, which is what pinned the main thread in #1743. That
    /// cost is bounded, not removed: the TextKit 2 path still re-parses the
    /// full buffer, but on its own 100 ms debounce rather than at the SSE
    /// flush rate, and the renderer appends without reflowing settled text.
    /// Measured on the new path a compile is linear and cheap —
    ///
    ///     2 000 chars →  1.5 ms      12 000 chars →  7.9 ms
    ///     6 000 chars →  4.0 ms      24 000 chars → 15.0 ms
    ///
    /// — so the flush rate is no longer coupled to message length at all.
    ///
    /// Keeping any ramp was visible in use. At the 250 ms cap a long answer
    /// updated four times a second; at a 60 ms cap, ~17 times. Both read as
    /// text arriving in slabs, and both make the word-by-word fade look like
    /// it is chasing the text, because words enter its queue in bursts
    /// proportional to the window. A flat 16 ms is what the native-chat
    /// prototype this render layer came from has always used, and is the
    /// cadence its streaming was tuned against.
    private static let maxWindowNs: UInt64 = 16_000_000

    /// Total characters handed to the UI so far this turn.
    private var flushedCharacters: Int = 0

    /// How long to coalesce before the next flush, given how much text the
    /// message already holds.
    ///
    /// The widening logic below is currently INERT: `maxWindowNs` equals
    /// `coalesceWindowNs` (see the decision note on `maxWindowNs`), so this
    /// always returns the 16 ms floor. The ramp exists — and these comments
    /// document it — because a fixed 16 ms cadence made `LaTeXMarkdownView`
    /// re-parse the whole accumulated message on every flush: parsing is
    /// O(length), so the cost across a turn was O(length²), and a long
    /// answer saturated the main thread (#1743). The TextKit 2 streaming
    /// path no longer has that coupling — it compiles on its own 100 ms
    /// debounce — so the ramp was flattened.
    private func currentWindowNs() -> UInt64 {
        guard flushedCharacters > Self.adaptiveThresholdChars else {
            return Self.coalesceWindowNs
        }
        // Multiply BEFORE dividing so the growth is actually linear. Dividing
        // first floors the ratio to an integer, which makes the window a
        // staircase — flat from 2 000 to 3 999 characters and then jumping
        // straight from 16 ms to 32 ms — rather than the smooth ramp the
        // comment above promises.
        //
        // `characters` is clamped to the value that already yields the cap, so
        // the multiplication cannot overflow no matter how long the message
        // gets, and no masking operators are needed.
        //
        // Derive that clamp by solving for it, not by scaling the window
        // ratio. With the former 250 ms `maxWindowNs`, 250/16 floors to 15,
        // capping at 30 000 characters (a 240 ms window) — `maxWindowNs`
        // would never be reached and the ceiling the older comments here
        // reference would not exist. The exact answer is 31 250. (250e6 * 2000
        // fits in UInt64 with ~7 orders of magnitude to spare, so the
        // numerator here is safe.) With the current flattened value the clamp
        // is 2 000 characters and the window never leaves the floor.
        let capCharacters = Int(
            Self.maxWindowNs * UInt64(Self.adaptiveThresholdChars) / Self.coalesceWindowNs
        )
        let characters = UInt64(min(flushedCharacters, capCharacters))
        let widened = Self.coalesceWindowNs * characters / UInt64(Self.adaptiveThresholdChars)
        return min(max(widened, Self.coalesceWindowNs), Self.maxWindowNs)
    }

    /// Codex r2 BLOCKING (unbounded queue): force-flush when the
    /// pending queue would exceed this many segments. An adversarial
    /// or malformed stream alternating content/reasoning could
    /// otherwise enqueue one new segment per delta without ever
    /// crossing the 16 ms window. The cap is permissive enough to
    /// let realistic interleaves (model produces reasoning then
    /// switches to content then switches back once or twice) coalesce
    /// into one flush, but tight enough that a runaway stream gets
    /// drained promptly.
    private static let maxPendingSegments: Int = 32

    /// How many times ``flush`` has applied a batch on the main actor — one
    /// per flush, by construction.
    ///
    /// Exposed because the emitted event stream cannot distinguish one hop
    /// carrying N segments from N hops carrying one each, and that difference
    /// is the whole point: it is what decides whether the view rebuilds once
    /// or N times per flush. A test asserting only on events would pass just
    /// as happily against the per-segment version this replaced.
    private(set) var mainActorApplications: Int = 0

    enum Kind { case content, reasoning }

    private struct PendingSegment {
        let kind: Kind
        var text: String
    }

    private var pending: [PendingSegment] = []
    private var contentEverFlushed: Bool = false
    private var reasoningEverFlushed: Bool = false
    private var lastFlushNs: UInt64 = 0

    private func nowNs() -> UInt64 { DispatchTime.now().uptimeNanoseconds }

    /// Append into the trailing segment when the kind matches, else
    /// push a new segment. Codex r2 BLOCKING (merge-on-append cost):
    /// mutate the trailing element in-place via direct subscript
    /// access on a `var` instead of copying into a `var tail` and
    /// writing it back — the latter forces a struct copy on every
    /// adjacent same-kind append, which for a long content run
    /// degrades to O(N²) string copying.
    private func appendSegment(kind: Kind, text: String) {
        if text.isEmpty { return }
        let lastIdx = pending.count - 1
        if lastIdx >= 0, pending[lastIdx].kind == kind {
            pending[lastIdx].text += text
        } else {
            pending.append(PendingSegment(kind: kind, text: text))
        }
    }

    func appendContent(
        _ c: String,
        onEvent: @escaping @MainActor (ChatStreamClient.Event) -> Void
    ) async {
        // Codex r1 NIT: empty-string upstream guard (the SSE loop
        // already filters, but the public surface must be safe).
        guard !c.isEmpty else { return }
        appendSegment(kind: .content, text: c)
        let elapsed = nowNs() &- lastFlushNs
        // Codex r2 BLOCKING: cap pending segments to bound queue
        // growth on adversarial alternating-kind streams.
        let overCap = pending.count > Self.maxPendingSegments
        if !contentEverFlushed || elapsed >= currentWindowNs() || overCap {
            await flush(onEvent: onEvent)
        }
    }

    func appendReasoning(
        _ r: String,
        onEvent: @escaping @MainActor (ChatStreamClient.Event) -> Void
    ) async {
        guard !r.isEmpty else { return }
        appendSegment(kind: .reasoning, text: r)
        let elapsed = nowNs() &- lastFlushNs
        let overCap = pending.count > Self.maxPendingSegments
        if !reasoningEverFlushed || elapsed >= currentWindowNs() || overCap {
            await flush(onEvent: onEvent)
        }
    }

    func flush(
        onEvent: @escaping @MainActor (ChatStreamClient.Event) -> Void
    ) async {
        guard !pending.isEmpty else { return }
        let toEmit = pending
        pending.removeAll(keepingCapacity: true)
        // Count `toEmit`, NOT `pending` — `pending` was just emptied, so
        // counting it leaves the total at zero forever and the adaptive
        // window silently never widens.
        // Saturating, not wrapping: a wrapped total would silently drop the
        // window back to its 16 ms floor, which is the bug this bounds.
        let batch = toEmit.reduce(0) { $0 + $1.text.count }
        flushedCharacters = (flushedCharacters > Int.max - batch)
            ? Int.max
            : flushedCharacters + batch
        lastFlushNs = nowNs()
        for segment in toEmit {
            switch segment.kind {
            case .content: contentEverFlushed = true
            case .reasoning: reasoningEverFlushed = true
            }
        }
        // ONE hop for the whole batch, in wire order.
        //
        // Hopping per segment defeats the point of coalescing on an
        // alternating content/reasoning stream: the `overCap` force-flush
        // fires at 33 segments regardless of the adaptive window, and if each
        // segment then gets its own MainActor mutation the view rebuilds —
        // and re-parses the whole message — about once per delta again, which
        // is exactly the #1743 failure the window is meant to bound. Applying
        // them in a single hop keeps the queue bound AND the repaint bound,
        // and order is preserved because the closure runs the array in
        // sequence.
        await MainActor.run {
            // Counted INSIDE the hop, not beside it. Incrementing once per
            // flush would measure flushes, and the thing worth measuring is
            // main-actor applications: restore the per-segment version and a
            // flush-side counter still reports 1, so the test that watches it
            // would stay green through the exact regression it exists for.
            self.mainActorApplications += 1
            for segment in toEmit {
                switch segment.kind {
                case .content: onEvent(.content(segment.text))
                case .reasoning: onEvent(.reasoning(segment.text))
                }
            }
        }
    }
}
