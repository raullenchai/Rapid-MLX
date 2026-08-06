import Foundation

/// Cheap web search via DuckDuckGo's HTML-only endpoint. No API key,
/// no JS engine — we just GET the HTML page and pull the visible
/// result links + snippets with regex. Brittle by design but the
/// alternative (Google CSE / Bing API / SerpAPI) all require paid
/// keys and an account, which Rapid's privacy-first stance doesn't
/// want to ship with.
///
/// Returns the top N results as a plain-text bulleted list the
/// model can quote from. No raw HTML, no link tracking — just
/// title + URL + snippet per result.
enum WebSearchTool {
    static let definition = ToolDefinition(
        name: "web_search",
        description: "Search the web and get the top results (title + URL + snippet). Use this when the user asks about current events, recent news, or facts that might have changed since the model was trained.",
        parameters: .object([
            "type": .string("object"),
            "properties": .object([
                "query": .object([
                    "type": .string("string"),
                    "description": .string("Search query in natural language.")
                ])
            ]),
            "required": .array([.string("query")])
        ])
    )

    static let resultCap: Int = 6
    static let snippetCharCap: Int = 240
    static let totalOutputCharCap: Int = 4096

    struct Args: Decodable {
        let query: String
    }

    /// Compatibility shim — the legacy single-arg form runs against
    /// DuckDuckGo. Used by tests and any caller that hasn't
    /// migrated to the provider-aware overload yet.
    static func run(arguments: String) async -> ToolCallResult {
        await run(arguments: arguments, provider: .duckduckgo, apiKey: nil)
    }

    /// v0.4.41 provider-aware entry point. Dispatches on the
    /// configured ``WebSearchProvider``:
    ///
    ///   * ``.duckduckgo`` — HTML scrape, no key
    ///   * ``.brave`` — JSON API, ``X-Subscription-Token`` header
    ///   * ``.tavily`` — JSON API, key in body
    ///
    /// When a paid provider is configured but the user hasn't
    /// pasted a key yet, we silently fall back to DuckDuckGo with
    /// a one-line hint appended to the model-visible result so the
    /// assistant can tell the user what's going on (and so the
    /// fallback isn't completely invisible). This matches the
    /// "no broken state" promise the rest of the app makes — the
    /// search keeps working even if the API key slot is empty.
    static func run(
        arguments: String,
        provider: WebSearchProvider,
        apiKey: String?
    ) async -> ToolCallResult {
        let toolName = "web_search"
        guard let data = arguments.data(using: .utf8),
              let args = try? JSONDecoder().decode(Args.self, from: data) else {
            return ToolCallResult(toolCallID: "", content: "\(toolName) error: could not parse arguments JSON", isError: true)
        }
        let q = args.query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else {
            return ToolCallResult(toolCallID: "", content: "\(toolName) error: empty query", isError: true)
        }
        var effectiveProvider = provider
        var fallbackNote: String? = nil
        if provider.requiresKey, apiKey == nil {
            effectiveProvider = .duckduckgo
            fallbackNote = "Note: \(provider.displayName) is selected but no API key is set — falling back to DuckDuckGo. Open Settings → Tools → Web search to paste a key."
        }
        switch effectiveProvider {
        case .duckduckgo:
            return await runDuckDuckGo(query: q, fallbackNote: fallbackNote)
        case .brave:
            return await runBrave(query: q, apiKey: apiKey ?? "")
        case .tavily:
            return await runTavily(query: q, apiKey: apiKey ?? "")
        }
    }

    /// Shared formatting: takes a list of provider-agnostic
    /// ``Result`` rows and renders the model-visible bulleted
    /// payload. Lifted out so every backend produces an
    /// identical wire shape, which keeps the tool description
    /// honest regardless of which provider answered.
    static func formatOutput(
        query: String,
        provider: WebSearchProvider,
        results: [Result],
        fallbackNote: String? = nil
    ) -> ToolCallResult {
        if results.isEmpty {
            var content = "web_search: no results found for \"\(query)\""
            if let note = fallbackNote { content += "\n\n" + note }
            return ToolCallResult(toolCallID: "", content: content, isError: false)
        }
        let bullets = results.enumerated().map { i, r -> String in
            let t = r.title.isEmpty ? r.url : r.title
            var snippet = r.snippet
            if snippet.count > snippetCharCap {
                snippet = String(snippet.prefix(snippetCharCap)) + "…"
            }
            return "\(i + 1). \(t)\n   \(r.url)\n   \(snippet)"
        }
        var content = "Web search via \(provider.displayName): \"\(query)\" — \(results.count) results\n\n" + bullets.joined(separator: "\n\n")
        if let note = fallbackNote {
            content += "\n\n" + note
        }
        if content.count > totalOutputCharCap {
            content = String(content.prefix(totalOutputCharCap)) + "\n…(truncated)"
        }
        return ToolCallResult(toolCallID: "", content: content, isError: false)
    }

    // MARK: - Per-provider runners

    /// Build DuckDuckGo's HTML-endpoint URL for a raw query.
    ///
    /// Uses ``URLComponents`` + ``URLQueryItem`` rather than hand-splicing a
    /// percent-encoded string: ``.urlQueryAllowed`` deliberately leaves the
    /// sub-delimiters ``& = + #`` unescaped, so a query that merely contains an
    /// ``&`` (``"cats & dogs"``) or ``#`` would otherwise inject extra query
    /// parameters or truncate the query into a fragment. ``URLQueryItem``
    /// percent-encodes the value as a single opaque field.
    static func duckDuckGoSearchURL(query q: String) -> URL? {
        var components = URLComponents()
        components.scheme = "https"
        components.host = "html.duckduckgo.com"
        components.path = "/html/"
        components.queryItems = [URLQueryItem(name: "q", value: q)]
        // URLQueryItem leaves a literal ``+`` unescaped, and a form-decoding
        // endpoint reads ``+`` as a space — so ``C++`` would arrive as ``C  ``.
        // Escape it explicitly. (Spaces are already ``%20`` here, so every
        // remaining ``+`` is a real plus from the query text.)
        components.percentEncodedQuery = components.percentEncodedQuery?
            .replacingOccurrences(of: "+", with: "%2B")
        return components.url
    }

    static func runDuckDuckGo(query q: String, fallbackNote: String?) async -> ToolCallResult {
        let toolName = "web_search"
        // DDG's HTML endpoint expects ``q=`` URL-encoded.
        guard let url = duckDuckGoSearchURL(query: q) else {
            return ToolCallResult(toolCallID: "", content: "\(toolName) error: could not build search URL", isError: true)
        }
        var req = URLRequest(url: url)
        // Without a UA header DDG redirects to a landing page; the
        // Safari UA is innocuous and stable.
        req.setValue("Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15", forHTTPHeaderField: "User-Agent")
        req.timeoutInterval = 15
        do {
            let (data, response) = try await cappedData(for: req)
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                let code = (response as? HTTPURLResponse)?.statusCode ?? -1
                return ToolCallResult(toolCallID: "", content: "\(toolName) error: DuckDuckGo returned HTTP \(code)", isError: true)
            }
            guard let html = String(data: data, encoding: .utf8) else {
                return ToolCallResult(toolCallID: "", content: "\(toolName) error: non-UTF8 response", isError: true)
            }
            // DDG silently rate-limits repeat callers from a single
            // IP/UA: the HTTP response is still 200 but the body is
            // the "Unfortunately, bots use DuckDuckGo too." modal
            // instead of result blocks. Without an explicit check
            // ``parseDDGHTML`` returns an empty list and the model
            // hears "no results found" — indistinguishable from a
            // query that genuinely matched nothing. Surface the
            // block so the assistant can either back off or tell
            // the user to configure a paid backend.
            if detectDDGAntiBot(html) {
                return ToolCallResult(
                    toolCallID: "",
                    content: "\(toolName) error: DuckDuckGo blocked this request (anti-bot rate limit). Either wait a few minutes, or configure a paid backend in Settings → Tools → Web search (Brave: 2000 queries/month free; Tavily: 1000 queries/month free).",
                    isError: true
                )
            }
            let results = parseDDGHTML(html, cap: resultCap)
            return formatOutput(query: q, provider: .duckduckgo, results: results, fallbackNote: fallbackNote)
        } catch {
            return ToolCallResult(toolCallID: "", content: "\(toolName) error: \(error.localizedDescription)", isError: true)
        }
    }

    static func runBrave(query q: String, apiKey: String) async -> ToolCallResult {
        let toolName = "web_search"
        guard let req = BraveSearchClient.buildRequest(query: q, apiKey: apiKey, count: resultCap) else {
            return ToolCallResult(toolCallID: "", content: "\(toolName) error: could not build Brave request", isError: true)
        }
        do {
            let (data, response) = try await cappedData(for: req)
            guard let http = response as? HTTPURLResponse else {
                return ToolCallResult(toolCallID: "", content: "\(toolName) error: Brave returned no HTTP response", isError: true)
            }
            switch http.statusCode {
            case 200..<300:
                let results = BraveSearchClient.parseResults(data, cap: resultCap)
                return formatOutput(query: q, provider: .brave, results: results)
            case 401, 403:
                // Account-level error — surface clearly so the
                // user knows to revisit their key.
                return ToolCallResult(
                    toolCallID: "",
                    content: "\(toolName) error: Brave rejected the API key (HTTP \(http.statusCode)). Re-paste it in Settings → Tools → Web search.",
                    isError: true
                )
            case 429:
                return ToolCallResult(
                    toolCallID: "",
                    content: "\(toolName) error: Brave free-tier limit hit (HTTP 429). Wait a few minutes or upgrade your plan.",
                    isError: true
                )
            default:
                return ToolCallResult(toolCallID: "", content: "\(toolName) error: Brave returned HTTP \(http.statusCode)", isError: true)
            }
        } catch {
            return ToolCallResult(toolCallID: "", content: "\(toolName) error: \(error.localizedDescription)", isError: true)
        }
    }

    static func runTavily(query q: String, apiKey: String) async -> ToolCallResult {
        let toolName = "web_search"
        guard let req = TavilySearchClient.buildRequest(query: q, apiKey: apiKey, maxResults: resultCap) else {
            return ToolCallResult(toolCallID: "", content: "\(toolName) error: could not build Tavily request", isError: true)
        }
        do {
            let (data, response) = try await cappedData(for: req)
            guard let http = response as? HTTPURLResponse else {
                return ToolCallResult(toolCallID: "", content: "\(toolName) error: Tavily returned no HTTP response", isError: true)
            }
            switch http.statusCode {
            case 200..<300:
                let results = TavilySearchClient.parseResults(data, cap: resultCap)
                return formatOutput(query: q, provider: .tavily, results: results)
            case 401, 403:
                return ToolCallResult(
                    toolCallID: "",
                    content: "\(toolName) error: Tavily rejected the API key (HTTP \(http.statusCode)). Re-paste it in Settings → Tools → Web search.",
                    isError: true
                )
            case 429:
                return ToolCallResult(
                    toolCallID: "",
                    content: "\(toolName) error: Tavily free-tier limit hit (HTTP 429). Wait a few minutes or upgrade your plan.",
                    isError: true
                )
            default:
                return ToolCallResult(toolCallID: "", content: "\(toolName) error: Tavily returned HTTP \(http.statusCode)", isError: true)
            }
        } catch {
            return ToolCallResult(toolCallID: "", content: "\(toolName) error: \(error.localizedDescription)", isError: true)
        }
    }

    struct Result: Equatable, Sendable {
        let title: String
        let url: String
        let snippet: String
    }

    /// True when ``html`` is DuckDuckGo's anti-bot challenge page
    /// rather than a real results page. DDG returns these with a
    /// normal HTTP 200, so without this check the empty result list
    /// is indistinguishable from a query that genuinely matched
    /// nothing.
    ///
    /// Detection uses a two-step gate to avoid false-positives on
    /// queries that *describe* the bot-challenge UI (e.g. a search
    /// for ``"anomaly-modal class"`` or ``"ddg cc=botnet"``):
    ///
    /// 1. **No ``result__body`` *as a class token*** — every real
    ///    results page ships at least one ``<div>`` whose class
    ///    list contains ``result__body``. The challenge page ships
    ///    zero. The check anchors on ``class="`` so that an echoed
    ///    query containing the literal text ``result__body`` (which
    ///    DDG copies back into a form ``value=""`` attribute on the
    ///    challenge page) does not satisfy the guard. Codex round-2
    ///    P3 called this out: a bare substring check would let an
    ///    adversarial query bypass detection and revert to "no
    ///    results found" UX.
    /// 2. **A challenge marker is present** — either
    ///    ``anomaly-modal`` (the BEM class root the challenge modal
    ///    uses on its container, title, and form children,
    ///    ``anomaly-modal__title`` etc.) or ``cc=botnet`` (the
    ///    classification token DDG embeds in the challenge form's
    ///    action URL). Bare-token matching survives the same kind
    ///    of class-chain reordering that broke the ``result__body``
    ///    parser on 2026-06-09; ``cc=botnet`` is kept as a fallback
    ///    in case the modal class is renamed but the wire-level
    ///    classification stays put.
    ///
    /// Codex round-1 P2 (PR #184) called out the original
    /// substring-only form misclassifying a legitimate results
    /// page whose echoed query / title / snippet happened to
    /// contain one of the marker strings. The class-attribute gate
    /// closes that hole — real result pages always have at least
    /// one block, so the detector cannot fire when the parser would
    /// have found hits.
    static func detectDDGAntiBot(_ html: String) -> Bool {
        // Cheap structural guard: a real results page always ships
        // result blocks whose class list contains ``result__body``.
        // We require the token to appear inside a ``class="…"``
        // attribute so that an echoed query value of the same
        // string (codex r2 P3) doesn't satisfy the guard.
        if containsResultBodyClassToken(html) { return false }
        return html.contains("anomaly-modal") || html.contains("cc=botnet")
    }

    /// True when ``html`` contains a ``class="…"`` attribute whose
    /// value list includes the ``result__body`` token. Used by
    /// ``detectDDGAntiBot`` to tell a real results page (which has
    /// the token in markup) from a challenge page that merely
    /// echoes the user's query string back into a form input.
    ///
    /// Scans both single- and double-quoted class attributes since
    /// DDG's HTML has historically mixed quoting on the same page.
    /// The check is intentionally simple — DDG only writes the
    /// class list in well-formed quoted attributes; if that ever
    /// changes the existing ``parseDDGHTML`` regression suite will
    /// fire first and we can revisit.
    static func containsResultBodyClassToken(_ html: String) -> Bool {
        let needle = "result__body"
        var idx = html.startIndex
        while let r = html.range(of: needle, range: idx..<html.endIndex) {
            idx = r.upperBound
            // Look backward for the opening ``class=`` attribute on the same
            // tag. ``<`` bounds the search so we don't pull in a class
            // attribute from an earlier element.
            let scope = html[html.startIndex..<r.lowerBound]
            guard let openTag = scope.range(of: "<", options: .backwards),
                  let classAttr = scope.range(
                      of: "class=",
                      options: .backwards,
                      range: openTag.upperBound..<scope.endIndex
                  )
            else { continue }
            // No closing ``>`` between ``class=`` and the token: it must
            // still be inside the same tag, not a following sibling.
            if html[classAttr.upperBound..<r.lowerBound].contains(">") { continue }
            // The token must live inside the QUOTED value that opens right
            // after ``class=``. Without this, ``result__body`` echoed in a
            // DIFFERENT attribute of the same tag (e.g.
            // ``data-x="result__body"`` on an anti-bot page) would sneak past
            // and mask the block as an ordinary empty result set.
            guard classAttr.upperBound < html.endIndex else { continue }
            let quote = html[classAttr.upperBound]
            guard quote == "\"" || quote == "'" else { continue }
            let valueStart = html.index(after: classAttr.upperBound)
            guard let closeQuote = html.range(
                of: String(quote),
                range: valueStart..<html.endIndex
            ) else { continue }
            guard r.lowerBound >= valueStart, r.upperBound <= closeQuote.lowerBound else { continue }
            // Whitespace/quote boundaries so ``result__bodyx`` doesn't match a
            // partial token inside the class value.
            let isBoundary: (Character) -> Bool = { $0 == " " || $0 == "\t" || $0 == "\n" || $0 == "\r" }
            let before = r.lowerBound == valueStart ? " " : html[html.index(before: r.lowerBound)]
            let after = r.upperBound == closeQuote.lowerBound ? " " : html[r.upperBound]
            if isBoundary(before) && isBoundary(after) { return true }
        }
        return false
    }

    /// Best-effort parser for DuckDuckGo's HTML-only results page.
    /// The endpoint serves one ``<div class="result">`` block per
    /// hit; inside each block we find ``<a class="result__a"``
    /// (title + link) and ``<a class="result__snippet"`` (snippet).
    static func parseDDGHTML(_ html: String, cap: Int) -> [Result] {
        // Split into per-result blocks. The HTML is permissive and
        // server-rendered so this is robust enough for v0.3.
        //
        // ``result__body`` is the marker token DDG has used on the
        // per-hit container ever since the HTML-only endpoint
        // launched. The v0.3 build pinned the marker to the
        // *prefix* form ``class="result__body`` because the class
        // chain back then was ``class="result__body links_main"``.
        // DDG has since reordered the chain to
        // ``class="links_main links_deep result__body"`` (observed
        // 2026-06-09), which made the prefix-anchored split match
        // zero times and silently turned every search into "no
        // results found." Anchoring on the bare token catches both
        // orderings; the extractor below still scans backward for
        // the enclosing ``<a>`` tag, so the block boundary itself
        // only needs to land *inside* the result container.
        let blockMarker = "result__body"
        var blocks = html.components(separatedBy: blockMarker)
        if blocks.count <= 1 { return [] }
        blocks.removeFirst()  // pre-content header

        var out: [Result] = []
        for block in blocks {
            if out.count >= cap { break }
            let title = extractText(in: block, between: "class=\"result__a\"", and: "</a>")
            let urlHref = extractAttr(in: block, attr: "href", after: "class=\"result__a\"")
            let snippet = extractText(in: block, between: "class=\"result__snippet\"", and: "</a>")
            // Codex round-2 finding: the previous form fell back to
            // ``urlHref`` whenever ``ddgRedirectExtract`` returned
            // nil — and ``ddgRedirectExtract`` returns nil for any
            // ``uddg=`` value whose scheme is not http(s). The raw
            // href could still be a ``javascript:``/``data:`` URL,
            // which the model would happily surface and the user
            // might click. Validate the final URL scheme before
            // including the result.
            let decoded = ddgRedirectExtract(urlHref) ?? urlHref
            guard isSafeHttpURL(decoded) else { continue }
            let cleanedTitle = stripTags(title).trimmingCharacters(in: .whitespacesAndNewlines)
            let cleanedSnippet = stripTags(snippet).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !cleanedTitle.isEmpty || !decoded.isEmpty else { continue }
            out.append(Result(title: cleanedTitle, url: decoded, snippet: cleanedSnippet))
        }
        return out
    }

    /// DDG wraps every result href as ``/l/?uddg=<url-encoded>``;
    /// pull the wrapped URL out so the model sees the real domain.
    ///
    /// Returns ``nil`` for non-http(s) destinations. DDG's HTML
    /// surface has historically been used to smuggle ``javascript:``
    /// and ``data:`` URIs into result lists, and the model will
    /// happily surface those for the user to click. Restricting to
    /// http/https keeps the assistant from turning into a phishing
    /// vector.
    static func ddgRedirectExtract(_ raw: String) -> String? {
        // Look for "uddg=" then take the rest until "&" or end.
        guard let r = raw.range(of: "uddg=") else { return nil }
        let after = raw[r.upperBound...]
        let cut = after.firstIndex(of: "&") ?? after.endIndex
        let encoded = String(after[..<cut])
        let decoded = encoded.removingPercentEncoding ?? encoded
        return isSafeHttpURL(decoded) ? decoded : nil
    }

    /// True only when ``raw`` parses as an http or https URL. Used
    /// at both the wrapped-URL extraction site AND the post-fallback
    /// gate so a raw ``javascript:``/``data:``/``file:`` href that
    /// sneaks past one path is still rejected by the other.
    static func isSafeHttpURL(_ raw: String) -> Bool {
        guard let scheme = URLComponents(string: raw)?.scheme?.lowercased() else { return false }
        return scheme == "http" || scheme == "https"
    }

    /// Pull the substring between ``start`` (passed past the
    /// opening tag, e.g. ``class="..."``) and the next ``end``.
    static func extractText(in block: String, between start: String, and end: String) -> String {
        guard let s = block.range(of: start) else { return "" }
        // Skip to the end of the opening tag.
        guard let openClose = block[s.upperBound...].firstIndex(of: ">") else { return "" }
        let textStart = block.index(after: openClose)
        guard let e = block[textStart...].range(of: end) else { return "" }
        return String(block[textStart..<e.lowerBound])
    }

    static func extractAttr(in block: String, attr: String, after marker: String) -> String {
        guard let m = block.range(of: marker) else { return "" }
        // Look back from the marker to find the enclosing ``<a ...>``
        // tag's ``href=`` attribute.
        let leading = block[..<m.upperBound]
        guard let aTagStart = leading.range(of: "<a ", options: .backwards) else { return "" }
        let scope = block[aTagStart.upperBound...]
        // Within the tag, find the attribute.
        let attrToken = "\(attr)=\""
        guard let attrR = scope.range(of: attrToken) else { return "" }
        let valueStart = attrR.upperBound
        guard let q = scope[valueStart...].firstIndex(of: "\"") else { return "" }
        return String(scope[valueStart..<q])
    }

    static func stripTags(_ html: String) -> String {
        // Quick-and-dirty tag stripping; ``</b>`` and ``<b>`` come
        // through on bolded query terms. We don't need a real parser.
        var out = ""
        var inside = false
        for ch in html {
            if ch == "<" { inside = true; continue }
            if ch == ">" { inside = false; continue }
            if !inside { out.append(ch) }
        }
        return decodeHTMLEntities(out)
    }

    static func decodeHTMLEntities(_ s: String) -> String {
        // Just the half-dozen entities DDG actually emits — full HTML
        // entity decoding is overkill for what's effectively title +
        // snippet text.
        s.replacingOccurrences(of: "&amp;", with: "&")
            .replacingOccurrences(of: "&lt;", with: "<")
            .replacingOccurrences(of: "&gt;", with: ">")
            .replacingOccurrences(of: "&quot;", with: "\"")
            .replacingOccurrences(of: "&#39;", with: "'")
            .replacingOccurrences(of: "&apos;", with: "'")
            .replacingOccurrences(of: "&nbsp;", with: " ")
    }
}
