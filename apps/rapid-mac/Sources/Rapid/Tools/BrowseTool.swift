import Foundation

/// `browse` — fetch a web page and return it as readable Markdown.
///
/// Design mirrors the other action tools: the model proposes, the USER approves
/// (``BrowseApprovalStore``), and the fetch is confined — only `http`/`https`,
/// every hop SSRF-checked (``BrowseSSRFGuard``), a byte cap, a timeout, and a
/// bounded redirect chain. The page is extracted to Markdown
/// (``HTMLToMarkdown``), the FULL body is cached (``BrowseContentCache``), and
/// the tool returns a budgeted first slice plus a `next_offset` cursor so the
/// model can page through a long document without re-fetching.
enum BrowseTool {
    /// Characters of rendered Markdown returned per call (the "15k budget").
    static let charBudget = 15_000
    /// Max response bytes we will download for one page.
    static let maxResponseBytes = 2 * 1024 * 1024
    /// Hard wall-clock ceiling for a single hop's fetch (seconds). Unlike
    /// URLSession's idle timeout — which resets on every byte and lets a server
    /// dribble one byte per interval to hold a hop open forever — this bounds the
    /// TOTAL time one hop may take and cancels it when it expires. Applied per
    /// hop (see ``fetchFollowingRedirects``).
    static let requestTimeout: TimeInterval = 12
    /// URLSession's own per-task resource backstop (seconds), a little above the
    /// wall-clock ceiling so our ``withDeadline`` is the primary control and this
    /// is defence-in-depth.
    static let resourceTimeout: TimeInterval = 20
    /// Max redirect hops we follow (each is SSRF-validated before we connect).
    static let maxRedirects = 5

    static let definition = ToolDefinition(
        name: "browse",
        description: "Fetch a web page (http/https) and return its readable content as Markdown. Use this to read articles, documentation, or any URL the user shares or you find via web_search. Long pages are paginated: the result includes 'next_offset' and 'has_more' — call browse again with that 'offset' to read the next part. Fetching from the network requires user approval; fresh cached pages are returned without prompting. Set refresh=true to fetch the latest version.",
        parameters: .object([
            "type": .string("object"),
            "properties": .object([
                "url": .object([
                    "type": .string("string"),
                    "description": .string("The absolute http(s) URL to fetch, e.g. 'https://example.com/article'.")
                ]),
                "offset": .object([
                    "type": .string("integer"),
                    "description": .string("Character offset for pagination. Omit (or 0) for the start of the page; pass the 'next_offset' from a previous browse call to continue reading the same URL.")
                ]),
                "refresh": .object([
                    "type": .string("boolean"),
                    "description": .string("Set true to bypass any cached copy and fetch the latest page. A network fetch requires user approval.")
                ])
            ]),
            "required": .array([.string("url")])
        ])
    )

    struct Args: Decodable {
        let url: String
        let offset: Int?
        let refresh: Bool?
    }

    /// The user answered "Don't allow" at an approval prompt.
    ///
    /// A distinct type, not a sentence: the decline can surface from deep
    /// inside ``fetchFollowingRedirects`` (a redirect that leaves the approved
    /// origin re-prompts), and the only honest way for ``run`` to tell it apart
    /// from a real fetch error is for the throw site to SAY which it is.
    /// Matching on the wording instead would misfire the day someone rewords
    /// this string — or the day a fetched page happens to contain it.
    ///
    /// ``message`` is the model-facing wire text. It stays plain and factual;
    /// what the USER sees is ``FailureDiagnosis/Kind/userDeclined``.
    struct ApprovalDeclined: LocalizedError, Equatable {
        let message: String
        /// Belt-and-braces: ``errorResult`` unwraps this type before anything
        /// reads `localizedDescription`, but a future caller that doesn't would
        /// otherwise hand the model Foundation's "The operation couldn't be
        /// completed." placeholder instead of what actually happened.
        var errorDescription: String? { message }
    }

    static func run(
        arguments: String,
        approval: BrowseApprovalStore,
        cache: BrowseContentCache = .shared
    ) async -> ToolCallResult {
        let tool = "browse"
        guard let data = arguments.data(using: .utf8),
              let args = try? JSONDecoder().decode(Args.self, from: data) else {
            return err(tool, "could not parse arguments JSON")
        }
        let rawURL = args.url.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !rawURL.isEmpty else { return err(tool, "empty url") }
        let offset = max(0, args.offset ?? 0)

        // Syntactic gate BEFORE any network or prompt: scheme allowlist, a host,
        // and IP-literal range check (no DNS). A blocked URL is rejected without
        // bothering the user with an approval it would only fail.
        guard let url = URL(string: rawURL), let scheme = url.scheme?.lowercased() else {
            return err(tool, "not a valid absolute URL")
        }
        guard BrowseSSRFGuard.allowedSchemes.contains(scheme) else {
            return err(tool, "scheme '\(scheme)' is not allowed (only http/https)")
        }
        guard let host = url.host, !host.isEmpty else {
            return err(tool, "URL has no host")
        }
        let bareHost = host.hasPrefix("[") && host.hasSuffix("]") ? String(host.dropFirst().dropLast()) : host
        if let literal = ParsedIP(bareHost), literal.isBlocked {
            return err(tool, "host '\(host)' is a private/loopback address (\(literal.canonical)) and cannot be browsed")
        }

        // Fresh cache hit → serve WITHOUT prompting or fetching. The cache is
        // consulted BEFORE the approval gate: a page we already have (this
        // session, an earlier page's fetch, or persisted to disk from a
        // previous launch) is served straight back with zero network and no
        // prompt while it remains inside the TTL. This covers both paging
        // (offset > 0) and a re-read at offset 0. Expired entries and explicit
        // refreshes fall through to the approval gate because they open a new
        // request to the host.
        if args.refresh != true, let cached = cache.get(rawURL) {
            return sliceResult(
                tool: tool,
                rawURL: rawURL,
                entry: cached,
                offset: offset,
                bytesFetched: nil,
                cacheExpiresAt: cache.expirationDate(for: cached)
            )
        }

        switch await approval.requestApproval(url: rawURL, host: host) {
        case .deny:
            return declined(tool, ApprovalDeclined(message: "the user did not approve browsing \(host)"))
        case .unavailable:
            // Nobody was asked, so this is NOT the user's decision and must not
            // be reported as one.
            return err(tool, approvalUnavailableMessage(host: host))
        case .allowOnce:
            break
        }
        // Approval can take arbitrarily long (it waits on the user). If the tool
        // call was cancelled while the sheet was up, don't now open a socket.
        if Task.isCancelled { return err(tool, "browse was cancelled") }

        do {
            // Redirects that leave the approved origin get a fresh approval — the
            // security model is "the user approved THIS host", and a server must
            // not be able to silently bounce the fetch to an unseen destination.
            let fetched = try await fetchFollowingRedirects(startURL: url) { redirectURL in
                let rHost = redirectURL.host ?? redirectURL.absoluteString
                return await approval.requestApproval(url: redirectURL.absoluteString, host: rHost)
            }
            let rendered = await renderMarkdown(from: fetched)
            let entry = BrowseContentCache.Entry(
                title: rendered.title,
                markdown: rendered.markdown,
                finalURL: fetched.finalURL.absoluteString
            )
            cache.put(rawURL, entry: entry)
            return sliceResult(
                tool: tool,
                rawURL: rawURL,
                entry: entry,
                offset: offset,
                bytesFetched: fetched.data.count,
                cacheExpiresAt: cache.expirationDate(for: entry)
            )
        } catch {
            return errorResult(tool: tool, error: error)
        }
    }

    /// How the redirect gate's answer ends the fetch, as a value: `nil` to
    /// carry on, otherwise the error to throw.
    ///
    /// Pure and separate because the branch is otherwise unreachable from a
    /// test — a redirect needs a live server, and the SSRF guard (correctly)
    /// refuses the loopback address a local one would have. This is the exact
    /// function the fetch loop calls, so the decline/abort distinction is
    /// covered by the code that ships rather than by a lookalike.
    static func redirectGateError(
        _ decision: BrowseApprovalStore.Decision,
        destination: URL
    ) -> Error? {
        let host = destination.host ?? "the destination"
        switch decision {
        case .allowOnce:
            return nil
        case .deny:
            return ApprovalDeclined(message: "the user did not approve the redirect to \(host)")
        case .unavailable:
            // Never asked, so never declined — a plain failure.
            return simpleError(approvalUnavailableMessage(host: host))
        }
    }

    /// Wire text for "the approval prompt could not be put to the user". Kept
    /// in one place so the initial gate and the redirect gate say the same
    /// thing, and so neither is mistaken for the user's own answer.
    static func approvalUnavailableMessage(host: String) -> String {
        "the approval prompt for \(host) could not be shown (the request was cancelled, "
            + "or another approval is already open)"
    }

    /// Turn a thrown fetch error into the result the model and the transcript
    /// both read. Kept as one pure function (rather than a chain of `catch`
    /// clauses) so the decline-vs-failure decision is testable without a
    /// network round-trip — the redirect decline can only be reached mid-fetch.
    static func errorResult(tool: String, error: Error) -> ToolCallResult {
        if let declinedByUser = error as? ApprovalDeclined {
            return declined(tool, declinedByUser)
        }
        if let rejection = error as? BrowseSSRFGuard.Rejection {
            return err(tool, rejection.message)
        }
        return err(tool, error.localizedDescription)
    }

    // MARK: - Fetch (manual redirect following, SSRF-checked per hop)

    struct Fetched {
        let finalURL: URL
        let data: Data
        let mime: String?
        let charset: String?
    }

    /// Follow redirects by hand so every hop is SSRF-validated before a socket
    /// opens. `startURL` is assumed already user-approved; `approveRedirect` is
    /// consulted only when a redirect leaves an already-approved origin, and
    /// anything but ``BrowseApprovalStore/Decision/allowOnce`` aborts the
    /// fetch — see ``redirectGateError`` for which abort is which.
    static func fetchFollowingRedirects(
        startURL: URL,
        approveRedirect: (URL) async -> BrowseApprovalStore.Decision
    ) async throws -> Fetched {
        var current = startURL
        var approvedOrigins: Set<String> = [origin(of: startURL)]
        var hop = 0
        while true {
            // Validate (incl. DNS) BEFORE connecting to this hop's host.
            try await BrowseSSRFGuard.validate(current)

            var req = URLRequest(url: current)
            req.timeoutInterval = requestTimeout
            req.setValue(userAgent, forHTTPHeaderField: "User-Agent")
            req.setValue("text/html,application/xhtml+xml,text/plain,application/json;q=0.9,*/*;q=0.8",
                         forHTTPHeaderField: "Accept")
            // Bind to a `let` so the @Sendable deadline closure captures an
            // immutable value (URLRequest is a Sendable value type).
            let request = req
            // Hard wall-clock ceiling per hop — see ``requestTimeout``.
            let raw = try await withDeadline(requestTimeout) { try await cappedBytes(request) }
            guard let http = raw.response as? HTTPURLResponse else {
                throw simpleError("no HTTP response from \(current.host ?? "host")")
            }
            if (300..<400).contains(http.statusCode), let loc = http.value(forHTTPHeaderField: "Location") {
                hop += 1
                guard hop <= maxRedirects else { throw simpleError("too many redirects (> \(maxRedirects))") }
                guard let next = URL(string: loc, relativeTo: current)?.absoluteURL else {
                    throw simpleError("redirect to an invalid URL")
                }
                // A redirect that crosses the approved origin needs a fresh OK —
                // otherwise a trusted host could bounce the fetch to an unseen one.
                let nextOrigin = origin(of: next)
                if !approvedOrigins.contains(nextOrigin) {
                    let decision = await approveRedirect(next)
                    if let refusal = redirectGateError(decision, destination: next) { throw refusal }
                    if Task.isCancelled { throw simpleError("browse was cancelled") }
                    approvedOrigins.insert(nextOrigin)
                }
                current = next
                continue
            }
            guard (200..<300).contains(http.statusCode) else {
                throw simpleError("HTTP \(http.statusCode) from \(current.host ?? "host")")
            }
            let (mime, charset) = parseContentType(http.value(forHTTPHeaderField: "Content-Type"))
            return Fetched(finalURL: http.url ?? current, data: raw.data, mime: mime, charset: charset)
        }
    }

    /// Normalised scheme://host:port so a redirect that only changes the path is
    /// treated as same-origin (no re-prompt) while a host/scheme/port change is
    /// cross-origin (re-prompt). Default ports are made explicit so
    /// `http://h` and `http://h:80` compare equal.
    static func origin(of url: URL) -> String {
        let scheme = url.scheme?.lowercased() ?? ""
        let host = url.host?.lowercased() ?? ""
        let port = url.port ?? (scheme == "https" ? 443 : 80)
        return "\(scheme)://\(host):\(port)"
    }

    /// Run `op` with a hard wall-clock ceiling: if it hasn't finished within
    /// `seconds`, the timer wins, `op`'s task is cancelled, and we throw. Used to
    /// turn URLSession's resettable idle timeout into a real per-hop deadline.
    static func withDeadline<T: Sendable>(
        _ seconds: TimeInterval,
        _ op: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        try await withThrowingTaskGroup(of: T.self) { group in
            group.addTask { try await op() }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
                throw simpleError("hop exceeded \(Int(seconds))s deadline")
            }
            defer { group.cancelAll() }
            guard let result = try await group.next() else {
                throw simpleError("hop produced no result")
            }
            return result
        }
    }

    /// Dedicated, isolated session for browsing. Ephemeral + cookies and cache
    /// fully OFF: a public-content fetch must NOT accumulate or replay cookies
    /// (which could send one site's session token to another, or hand
    /// authenticated content to the model) and must not persist fetched pages to
    /// disk. The session delegate blocks automatic redirects so ``BrowseTool``
    /// follows them itself, SSRF-checking every hop before a socket opens.
    static let session: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.httpCookieStorage = nil
        config.httpShouldSetCookies = false
        config.httpCookieAcceptPolicy = .never
        config.urlCache = nil
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        config.timeoutIntervalForRequest = requestTimeout
        config.timeoutIntervalForResource = resourceTimeout
        return URLSession(configuration: config, delegate: BrowseNoRedirectDelegate.shared, delegateQueue: nil)
    }()

    /// A fetched response body + its `URLResponse`, boxed as `Sendable` so it can
    /// cross the ``withDeadline`` task-group boundary. `URLResponse` is safe to
    /// hand across here: it is fully materialised and we only read it.
    struct RawResponse: @unchecked Sendable {
        let data: Data
        let response: URLResponse
    }

    /// Stream a request body, aborting once it crosses ``maxResponseBytes``.
    static func cappedBytes(_ request: URLRequest) async throws -> RawResponse {
        let (stream, response) = try await session.bytes(for: request)
        var data = Data()
        data.reserveCapacity(min(maxResponseBytes, 256 * 1024))
        for try await byte in stream {
            if data.count >= maxResponseBytes {
                throw simpleError("page exceeded \(maxResponseBytes / (1024 * 1024)) MB cap")
            }
            data.append(byte)
        }
        return RawResponse(data: data, response: response)
    }

    // MARK: - Render

    /// Decode + render off the main actor. The registry runs tools on
    /// `@MainActor`, and decoding a large body plus HTML tokenizing is CPU-bound
    /// (bounded, but not free on a 2 MB page or adversarial markup) — doing it
    /// inline would block the UI, so it is hopped onto a detached task.
    static func renderMarkdown(from fetched: Fetched) async -> HTMLToMarkdown.Result {
        let mime = (fetched.mime ?? "").lowercased()
        let data = fetched.data
        let charset = fetched.charset
        let byteCount = data.count
        return await Task.detached(priority: .userInitiated) {
            let text = decode(data, charset: charset)
            // HTML / XHTML → readability extraction. text/plain, JSON, and
            // anything else textual → return as-is (still budgeted + cached
            // downstream). A binary type is near-useless as text; surface a note.
            if mime.contains("html") || mime.contains("xml") {
                return HTMLToMarkdown.extract(text, baseURL: fetched.finalURL)
            }
            if mime.isEmpty || mime.hasPrefix("text/") || mime.contains("json") {
                return HTMLToMarkdown.Result(title: nil, markdown: text)
            }
            return HTMLToMarkdown.Result(
                title: nil,
                markdown: "[browse: content type '\(mime)' is not text; \(byteCount) bytes not shown]"
            )
        }.value
    }

    /// Decode bytes to a String, honouring a declared charset, then UTF-8, then
    /// a lossy fallback (never fails — readability tolerates a few replacements).
    static func decode(_ data: Data, charset: String?) -> String {
        if let cs = charset?.lowercased() {
            let enc: String.Encoding?
            switch cs {
            case "utf-8", "utf8": enc = .utf8
            case "iso-8859-1", "latin1", "iso8859-1": enc = .isoLatin1
            case "windows-1252", "cp1252": enc = .windowsCP1252
            case "utf-16", "utf16": enc = .utf16
            case "ascii", "us-ascii": enc = .ascii
            default: enc = nil
            }
            if let enc, let s = String(data: data, encoding: enc) { return s }
        }
        if let s = String(data: data, encoding: .utf8) { return s }
        return String(decoding: data, as: UTF8.self)   // lossy, always succeeds
    }

    // MARK: - Slice / paginate

    static func sliceResult(
        tool: String,
        rawURL: String,
        entry: BrowseContentCache.Entry,
        offset: Int,
        bytesFetched: Int?,
        cacheExpiresAt: Date? = nil
    ) -> ToolCallResult {
        let full = entry.markdown
        let total = entry.count
        let start = min(max(0, offset), total)
        var end = min(start + charBudget, total)
        // Snap the cut back to a line boundary when there's more to come, so a
        // page doesn't end mid-line. Keep `end` (and thus next_offset) exactly
        // at the emitted length — no off-by-one in the cursor the model reuses.
        if end < total {
            let lower = max(start, end - 500)
            if let nl = lastNewline(in: entry, from: lower, to: end) { end = nl + 1 }
        }
        let sliceStart = entry.index(atCharacterOffset: start)
        let sliceEnd = entry.index(atCharacterOffset: end)
        let content = String(full[sliceStart..<sliceEnd])
        let hasMore = end < total

        var payload: [String: Any] = [
            "url": rawURL,
            "content": content,
            "offset": start,
            "total_chars": total,
            "has_more": hasMore,
            "cache_hit": bytesFetched == nil,
            "fetched_at": entry.fetchedAt.ISO8601Format(),
        ]
        if let cacheExpiresAt {
            payload["cache_expires_at"] = cacheExpiresAt.ISO8601Format()
        }
        if let title = entry.title, !title.isEmpty { payload["title"] = title }
        if entry.finalURL != rawURL { payload["final_url"] = entry.finalURL }
        if let b = bytesFetched { payload["bytes_fetched"] = b }
        if hasMore {
            payload["next_offset"] = end
            payload["note"] = "Showing characters \(start)–\(end) of \(total). Call browse again with offset=\(end) to continue."
        } else if start >= total && total > 0 {
            payload["note"] = "offset \(start) is at or past the end of the \(total)-character page."
        }
        return ToolCallResult(toolCallID: "", content: jsonString(payload), isError: false)
    }

    /// Index of the last "\n" in `full` within [from, to), or nil.
    private static func lastNewline(
        in entry: BrowseContentCache.Entry,
        from: Int,
        to: Int
    ) -> Int? {
        guard from < to else { return nil }
        let full = entry.markdown
        let lo = entry.index(atCharacterOffset: from)
        let hi = entry.index(atCharacterOffset: to)
        var found: Int? = nil
        var idx = lo
        var pos = from
        while idx < hi {
            if full[idx] == "\n" { found = pos }
            idx = full.index(after: idx)
            pos += 1
        }
        return found
    }

    // MARK: - Helpers

    static let userAgent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) RapidMLX/1.0 Safari/605.1.15"

    static func parseContentType(_ header: String?) -> (mime: String?, charset: String?) {
        guard let header, !header.isEmpty else { return (nil, nil) }
        let parts = header.split(separator: ";").map { $0.trimmingCharacters(in: .whitespaces) }
        let mime = parts.first?.lowercased()
        var charset: String? = nil
        for p in parts.dropFirst() where p.lowercased().hasPrefix("charset=") {
            charset = String(p.dropFirst("charset=".count)).trimmingCharacters(in: CharacterSet(charactersIn: "\"' "))
        }
        return (mime, charset)
    }

    private static func err(_ tool: String, _ message: String) -> ToolCallResult {
        ToolCallResult(toolCallID: "", content: "\(tool) error: \(message)", isError: true)
    }

    /// A fetch the USER turned down. The wire content stays identical in shape
    /// to any other unsuccessful result — the model still needs to be told, in
    /// plain words, that it has no page and why — but the result carries an
    /// explicit ``FailureDiagnosis/Kind/userDeclined`` so the transcript can
    /// render a deliberate choice as the ordinary outcome it is instead of
    /// reporting a malfunction and blaming the user's input for it.
    private static func declined(_ tool: String, _ error: ApprovalDeclined) -> ToolCallResult {
        ToolCallResult(
            toolCallID: "",
            content: "\(tool) error: \(error.message)",
            isError: true,
            failureKind: .userDeclined
        )
    }

    private static func simpleError(_ message: String) -> NSError {
        NSError(domain: "RapidBrowse", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }

    static func jsonString(_ payload: [String: Any]) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys]),
              let s = String(data: data, encoding: .utf8) else {
            return "{\"error\":\"failed to encode browse result\"}"
        }
        return s
    }
}

/// Per-task delegate that refuses every automatic redirect so ``BrowseTool``
/// can follow them by hand, SSRF-validating each target before it connects.
final class BrowseNoRedirectDelegate: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    static let shared = BrowseNoRedirectDelegate()
    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        completionHandler(nil)   // never auto-follow
    }
}
