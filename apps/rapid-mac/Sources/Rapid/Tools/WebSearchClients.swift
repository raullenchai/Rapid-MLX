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
                userInfo: [NSLocalizedDescriptionKey: "response exceeded \(byteCap / 1024) KB cap"]
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
    /// "description" }] } }``. We tolerate missing/empty fields
    /// gracefully — a result row that's missing a title is still
    /// useful if the URL is present.
    static func parseResults(_ data: Data, cap: Int) -> [WebSearchTool.Result] {
        struct Envelope: Decodable {
            struct Web: Decodable {
                let results: [Item]?
            }
            struct Item: Decodable {
                let title: String?
                let url: String?
                let description: String?
            }
            let web: Web?
        }
        guard let env = try? JSONDecoder().decode(Envelope.self, from: data) else {
            return []
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
    static func parseResults(_ data: Data, cap: Int) -> [WebSearchTool.Result] {
        struct Envelope: Decodable {
            struct Item: Decodable {
                let title: String?
                let url: String?
                let content: String?
            }
            let results: [Item]?
        }
        guard let env = try? JSONDecoder().decode(Envelope.self, from: data) else {
            return []
        }
        let items = env.results ?? []
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
