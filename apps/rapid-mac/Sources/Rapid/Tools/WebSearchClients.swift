import Foundation

/// Codex audit batch 6 finding (WebSearchClients.swift:31, P2):
/// an API key pasted with a CR/LF or other ASCII control byte
/// would survive the outer ``trimmingCharacters(in: .whitespacesAndNewlines)``
/// at the storage layer (which trims only LEADING/TRAILING
/// whitespace) and reach ``URLRequest.setValue(_:forHTTPHeaderField:)``.
/// URLRequest does NOT validate header values for CRLF — it will
/// happily produce a request whose serialised header bytes
/// contain ``X-Subscription-Token: bravekey<CR><LF>X-Evil: foo``.
/// Defensive check: refuse to build the request when the key
/// contains any control byte.
private func headerSafeKey(_ apiKey: String) -> String? {
    let trimmed = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return nil }
    for scalar in trimmed.unicodeScalars {
        // Reject all C0 control bytes (0x00-0x1F) and DEL (0x7F).
        if scalar.value < 0x20 || scalar.value == 0x7F { return nil }
    }
    return trimmed
}

/// Codex audit batch 6 finding (WebSearchTool.swift:137 / WeatherTool.swift:105, P2/P3):
/// every provider call read the full response body via
/// ``URLSession.shared.data(for:)``. A misbehaving (or hostile)
/// upstream that returns a multi-GB response would balloon the
/// app's memory before any JSON parse runs. ``cappedData`` streams
/// from ``URLSession.bytes`` and aborts the moment the body
/// crosses the limit. 1 MB is generous for any of the listed
/// search backends (a 6-result Brave/Tavily payload is < 100 KB
/// in practice). Throws on cap-exceeded so the caller surfaces a
/// clear error rather than silently truncated JSON.
func cappedData(
    for request: URLRequest,
    byteCap: Int = 1_048_576,
    deadline: TimeInterval = 20
) async throws -> (Data, URLResponse) {
    // ``URLRequest.timeoutInterval`` is an INACTIVITY timer: it resets on every
    // byte received, so an upstream that dribbles one byte every few seconds
    // resets it forever and holds the tool call (and its chat turn) open. Race
    // the streamed read against a hard wall-clock deadline; the byte stream is
    // cancellable, so the losing task is actually stopped.
    try await withThrowingTaskGroup(of: (Data, URLResponse).self) { group in
        group.addTask { try await streamCappedData(for: request, byteCap: byteCap) }
        group.addTask {
            try await Task.sleep(nanoseconds: UInt64(deadline * 1_000_000_000))
            throw NSError(
                domain: "RapidWebSearch",
                code: 408,
                userInfo: [NSLocalizedDescriptionKey: "request exceeded \(Int(deadline))s deadline"]
            )
        }
        defer { group.cancelAll() }
        guard let result = try await group.next() else {
            throw CancellationError()
        }
        return result
    }
}

private func streamCappedData(
    for request: URLRequest,
    byteCap: Int
) async throws -> (Data, URLResponse) {
    let (stream, response) = try await URLSession.shared.bytes(for: request)
    var data = Data()
    data.reserveCapacity(min(byteCap, 64 * 1024))
    for try await byte in stream {
        if data.count >= byteCap {
            throw NSError(
                domain: "RapidWebSearch",
                code: 413,
                userInfo: [NSLocalizedDescriptionKey: "response exceeded \(DownloadProgress.formatBytes(Int64(byteCap))) cap"]
            )
        }
        data.append(byte)
    }
    return (data, response)
}

/// Brave Search API client. The free tier serves 2 000 queries/month
/// and the request shape is a plain GET with a single header carrying
/// the subscription key.
///
/// We deliberately request a small ``count`` (matches WebSearchTool's
/// existing 6-result cap) because the model only quotes the first
/// few results and every extra result costs the user a query slot.
enum BraveSearchClient {
    static let endpoint = "https://api.search.brave.com/res/v1/web/search"
    static let timeout: TimeInterval = 15

    /// Builds a fully-formed URLRequest for the Brave Search API.
    /// Extracted so tests can pin the URL / header / body shape
    /// without spinning up URLSession. Returns ``nil`` when the
    /// API key contains a control character (codex audit batch 6,
    /// P2 — CRLF in a pasted key would inject a header).
    static func buildRequest(query: String, apiKey: String, count: Int) -> URLRequest? {
        guard let cleanKey = headerSafeKey(apiKey) else { return nil }
        var components = URLComponents(string: endpoint)
        components?.queryItems = [
            URLQueryItem(name: "q", value: query),
            URLQueryItem(name: "count", value: String(count)),
            URLQueryItem(name: "safesearch", value: "moderate"),
        ]
        guard let url = components?.url else { return nil }
        var req = URLRequest(url: url)
        req.httpMethod = "GET"
        req.timeoutInterval = timeout
        // Brave's required header. ``Accept: application/json``
        // makes them respond with the structured payload (the
        // default is the HTML SERP, which we'd have to scrape).
        req.setValue(cleanKey, forHTTPHeaderField: "X-Subscription-Token")
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        return req
    }

    /// Parse Brave's JSON response into the engine-agnostic
    /// ``WebSearchTool.Result`` shape so the rest of the pipeline
    /// doesn't need to know which backend ran the query.
    ///
    /// Brave returns ``{ "web": { "results": [{ "title", "url",
    /// "description" }] } }``. Brave legitimately omits the entire `web`
    /// vertical when it has no web hits, so absent `web` is an empty success;
    /// a present vertical still requires a valid results array so schema drift
    /// is distinguishable from emptiness. Fields within a result remain
    /// optional: a missing title is fine when the URL is present.
    static func parseResults(_ data: Data, cap: Int) -> [WebSearchTool.Result]? {
        struct Envelope: Decodable {
            struct Web: Decodable {
                let results: [Item]
            }
            struct Item: Decodable {
                let title: String?
                let url: String?
                let description: String?
            }
            let web: Web?
        }
        guard let env = try? JSONDecoder().decode(Envelope.self, from: data) else {
            return nil
        }
        let items = env.web?.results ?? []
        var out: [WebSearchTool.Result] = []
        for item in items {
            if out.count >= cap { break }
            let url = item.url ?? ""
            // Filter out non-http(s) hits — same safety gate as
            // the DDG path. Brave is well-behaved here but we
            // defend at the boundary anyway.
            guard WebSearchTool.isSafeHttpURL(url) else { continue }
            out.append(WebSearchTool.Result(
                title: item.title ?? "",
                url: url,
                snippet: item.description ?? ""
            ))
        }
        return out
    }
}

/// Tavily Search API client. The free tier serves 1 000 queries/month.
/// Request shape is a POST with the key inside the JSON body —
/// Tavily explicitly does not accept the key in a header.
enum TavilySearchClient {
    static let endpoint = "https://api.tavily.com/search"
    static let timeout: TimeInterval = 15

    /// Builds a fully-formed URLRequest for the Tavily Search API.
    /// The key lives in the JSON body, ``api_key``.
    ///
    /// ``search_depth: "basic"`` is the cheap tier (1 credit per
    /// query); ``"advanced"`` costs 2 credits and runs LLM
    /// re-ranking server-side. We keep basic by default — the
    /// model is local and quoting results, the extra re-ranking
    /// is mostly redundant.
    static func buildRequest(query: String, apiKey: String, maxResults: Int) -> URLRequest? {
        // Tavily takes the key in the JSON body rather than a
        // header, so CRLF can't inject. But we still want to
        // reject control bytes in case a paste introduced them
        // by accident — JSON body containing a literal newline
        // breaks the upstream parser anyway.
        guard let cleanKey = headerSafeKey(apiKey) else { return nil }
        guard let url = URL(string: endpoint) else { return nil }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.timeoutInterval = timeout
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        let body: [String: Any] = [
            "api_key": cleanKey,
            "query": query,
            "max_results": maxResults,
            "search_depth": "basic",
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: body) else { return nil }
        req.httpBody = data
        return req
    }

    /// Parse Tavily's JSON response. The schema is
    /// ``{ "results": [{ "title", "url", "content" }] }``.
    /// ``content`` is the snippet — Tavily ships a pre-summarised
    /// extract rather than a raw description, which usually reads
    /// better than DDG's HTML scrape but occasionally truncates a
    /// useful fact. We don't second-guess.
    static func parseResults(_ data: Data, cap: Int) -> [WebSearchTool.Result]? {
        struct Envelope: Decodable {
            struct Item: Decodable {
                let title: String?
                let url: String?
                let content: String?
            }
            let results: [Item]
        }
        guard let env = try? JSONDecoder().decode(Envelope.self, from: data) else {
            return nil
        }
        let items = env.results
        var out: [WebSearchTool.Result] = []
        for item in items {
            if out.count >= cap { break }
            let url = item.url ?? ""
            guard WebSearchTool.isSafeHttpURL(url) else { continue }
            out.append(WebSearchTool.Result(
                title: item.title ?? "",
                url: url,
                snippet: item.content ?? ""
            ))
        }
        return out
    }
}

/// Keenable client (#2041) — the zero-setup default backend.
///
/// Keenable exposes two surfaces and we use both:
///
///   * **Keyless** — its public MCP endpoint accepts a bare
///     JSON-RPC ``tools/call`` POST with no account, no key and no
///     session handshake (verified live 2026-08-18: a cold
///     ``tools/call`` with neither ``initialize`` nor a session
///     header answers 200). Shared pool, 1 000 requests/hour per
///     IP. This is NOT an MCP-client integration — it's one plain
///     HTTPS POST; we never speak the rest of the protocol.
///   * **Keyed** — ``POST /v1/search`` REST with an ``X-API-Key``
///     header. Requires a (free) key; lifts the hourly cap and
///     meters against the org's monthly credit allowance.
///
/// The keyless response arrives as a JSON-RPC envelope whose
/// ``result.content[0].text`` is a plain-text block list
/// (``Title:`` / ``URL:`` / ``Snippets:`` separated by ``---``);
/// the keyed response is structured JSON. Both funnel into the
/// engine-agnostic ``WebSearchTool.Result`` shape.
enum KeenableSearchClient {
    static let mcpEndpoint = "https://api.keenable.ai/mcp"
    static let restEndpoint = "https://api.keenable.ai/v1/search"
    static let timeout: TimeInterval = 15

    /// One JSON-RPC ``tools/call`` against the public MCP endpoint.
    /// The ``id`` is fixed: we send exactly one request per HTTP
    /// call, so there is nothing to correlate.
    static func buildKeylessRequest(query: String, snippetMaxLength: Int) -> URLRequest? {
        guard let url = URL(string: mcpEndpoint) else { return nil }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.timeoutInterval = timeout
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        // The MCP transport may answer either plain JSON or an SSE
        // frame; advertise both so the server picks freely and
        // ``parseKeylessResults`` handles either.
        req.setValue("application/json, text/event-stream", forHTTPHeaderField: "Accept")
        let body: [String: Any] = [
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": [
                "name": "search_web_pages",
                "arguments": [
                    "query": query,
                    // Server-side schema minimum is 180; we pass our
                    // display cap so the upstream doesn't ship chars
                    // we'd truncate anyway.
                    "snippet_max_length": max(180, snippetMaxLength),
                ] as [String: Any],
            ] as [String: Any],
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: body) else { return nil }
        req.httpBody = data
        return req
    }

    /// Keyed REST call. The key rides in ``X-API-Key`` — same
    /// control-byte defence as every other header-borne key.
    static func buildKeyedRequest(query: String, apiKey: String, snippetMaxLength: Int) -> URLRequest? {
        guard let cleanKey = headerSafeKey(apiKey) else { return nil }
        guard let url = URL(string: restEndpoint) else { return nil }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.timeoutInterval = timeout
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        req.setValue(cleanKey, forHTTPHeaderField: "X-API-Key")
        let body: [String: Any] = [
            "query": query,
            "snippet_max_length": max(180, snippetMaxLength),
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: body) else { return nil }
        req.httpBody = data
        return req
    }

    /// Unwrap the JSON-RPC envelope (optionally SSE-framed) and parse
    /// the tool text. ``nil`` means "malformed / error envelope" —
    /// the caller degrades to the DuckDuckGo backstop; an empty array
    /// is a genuine zero-hit search.
    static func parseKeylessResults(_ data: Data, cap: Int) -> [WebSearchTool.Result]? {
        struct Envelope: Decodable {
            struct ResultBody: Decodable {
                struct ContentItem: Decodable {
                    let type: String?
                    let text: String?
                }
                let content: [ContentItem]?
                let isError: Bool?
            }
            struct RPCError: Decodable { let message: String? }
            let result: ResultBody?
            let error: RPCError?
        }
        var payload = data
        // SSE frame: take the last ``data:`` line — the transport
        // streams at most one JSON-RPC response per request here.
        if let text = String(data: data, encoding: .utf8),
           text.hasPrefix("event:") || text.hasPrefix("data:") || text.contains("\ndata:") {
            let lines = text.split(separator: "\n", omittingEmptySubsequences: true)
            if let last = lines.last(where: { $0.hasPrefix("data:") }) {
                let json = last.dropFirst("data:".count).trimmingCharacters(in: .whitespaces)
                payload = Data(json.utf8)
            }
        }
        guard let env = try? JSONDecoder().decode(Envelope.self, from: payload) else { return nil }
        guard env.error == nil, let result = env.result, result.isError != true else { return nil }
        guard let text = result.content?.first(where: { $0.text != nil })?.text else { return nil }
        return parseTextBlocks(text, cap: cap)
    }

    /// Parse the keyless tool text: blocks separated by ``---``
    /// lines, each carrying ``Title:`` / ``URL:`` headers and a
    /// free-text body after a ``Snippets:`` line. ``Published:`` /
    /// ``Acquired:`` metadata lines are dropped — the model reads
    /// title + URL + snippet, same as every other backend.
    static func parseTextBlocks(_ text: String, cap: Int) -> [WebSearchTool.Result] {
        var out: [WebSearchTool.Result] = []
        for rawBlock in text.components(separatedBy: "\n---\n") {
            if out.count >= cap { break }
            var title = ""
            var url = ""
            var snippetLines: [String] = []
            var inSnippets = false
            for line in rawBlock.split(separator: "\n", omittingEmptySubsequences: false) {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                if !inSnippets {
                    if trimmed.hasPrefix("Title: ") {
                        title = String(trimmed.dropFirst("Title: ".count))
                    } else if trimmed.hasPrefix("URL: ") {
                        url = String(trimmed.dropFirst("URL: ".count))
                    } else if trimmed == "Snippets:" {
                        inSnippets = true
                    }
                    // ``Published:`` / ``Acquired:`` fall through untouched.
                } else if !trimmed.isEmpty {
                    // Codex r1: a metadata line emitted AFTER the
                    // snippet body must not leak into the model-visible
                    // snippet text.
                    let isMetadata = ["Title: ", "URL: ", "Published: ", "Acquired: "]
                        .contains { trimmed.hasPrefix($0) }
                    if !isMetadata { snippetLines.append(trimmed) }
                }
            }
            // Same boundary gate as every backend: only http(s)
            // destinations reach the model.
            guard WebSearchTool.isSafeHttpURL(url) else { continue }
            out.append(WebSearchTool.Result(
                title: title,
                url: url,
                snippet: snippetLines.joined(separator: " ")
            ))
        }
        return out
    }

    /// Keyed REST response: ``{ "query", "results": [{ "title",
    /// "url", "description", "snippet" }] }``. ``snippet`` carries
    /// the query-relevant highlight; ``description`` is the static
    /// page description — prefer the former, fall back to the
    /// latter.
    static func parseKeyedResults(_ data: Data, cap: Int) -> [WebSearchTool.Result]? {
        struct Envelope: Decodable {
            struct Item: Decodable {
                let title: String?
                let url: String?
                let description: String?
                let snippet: String?
            }
            let results: [Item]
        }
        guard let env = try? JSONDecoder().decode(Envelope.self, from: data) else { return nil }
        var out: [WebSearchTool.Result] = []
        for item in env.results {
            if out.count >= cap { break }
            let url = item.url ?? ""
            guard WebSearchTool.isSafeHttpURL(url) else { continue }
            out.append(WebSearchTool.Result(
                title: item.title ?? "",
                url: url,
                snippet: item.snippet ?? item.description ?? ""
            ))
        }
        return out
    }
}

/// Parallel Search client (#2042) — the recommended keyed backend.
/// ``POST /v1/search`` with the key in ``x-api-key``. We pin
/// ``mode: "advanced"`` (the provider default and its
/// strongest-measured tier) rather than inheriting a server-side
/// default that could drift, and bound the response to our display
/// budget via ``advanced_settings`` so extra excerpt characters are
/// never fetched just to be truncated locally.
enum ParallelSearchClient {
    static let endpoint = "https://api.parallel.ai/v1/search"
    static let timeout: TimeInterval = 15

    static func buildRequest(
        query: String,
        apiKey: String,
        maxResults: Int,
        maxCharsPerResult: Int
    ) -> URLRequest? {
        guard let cleanKey = headerSafeKey(apiKey) else { return nil }
        guard let url = URL(string: endpoint) else { return nil }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.timeoutInterval = timeout
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        req.setValue(cleanKey, forHTTPHeaderField: "x-api-key")
        // ``search_queries`` is the required field (an array);
        // ``objective`` additionally carries the natural-language
        // intent, which Parallel uses for ranking. Our tool has one
        // query string, so it fills both.
        let body: [String: Any] = [
            "objective": query,
            "search_queries": [query],
            "mode": "advanced",
            "advanced_settings": [
                "max_results": maxResults,
                "excerpt_settings": [
                    "max_chars_per_result": maxCharsPerResult
                ] as [String: Any],
            ] as [String: Any],
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: body) else { return nil }
        req.httpBody = data
        return req
    }

    /// Response: ``{ "results": [{ "url", "title", "excerpts": [str] }] }``.
    /// ``excerpts`` is an array of markdown-formatted passages;
    /// join them — ``formatOutput`` applies the display cap.
    static func parseResults(_ data: Data, cap: Int) -> [WebSearchTool.Result]? {
        struct Envelope: Decodable {
            struct Item: Decodable {
                let url: String?
                let title: String?
                let excerpts: [String]?
            }
            let results: [Item]
        }
        guard let env = try? JSONDecoder().decode(Envelope.self, from: data) else { return nil }
        var out: [WebSearchTool.Result] = []
        for item in env.results {
            if out.count >= cap { break }
            let url = item.url ?? ""
            guard WebSearchTool.isSafeHttpURL(url) else { continue }
            let snippet = (item.excerpts ?? [])
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
                .joined(separator: " … ")
            out.append(WebSearchTool.Result(
                title: item.title ?? "",
                url: url,
                snippet: snippet
            ))
        }
        return out
    }
}
