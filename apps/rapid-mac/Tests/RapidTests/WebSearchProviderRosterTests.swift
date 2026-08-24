import Foundation
import Testing
@testable import Rapid

// Tests for the #2040–#2043 web-search roster overhaul: the
// descriptor table, the Keenable keyless default + fallback chain,
// the Parallel client, and the Brave billing-honesty copy.

@Suite("Web-search provider descriptor table")
struct WebSearchProviderDescriptorTests {
    @Test("Roster order is the Settings radio order: default first, recommended second, backstop last")
    func rosterOrder() {
        #expect(WebSearchProvider.allCases.first == .keenable)
        #expect(WebSearchProvider.allCases[1] == .parallel)
        #expect(WebSearchProvider.allCases.last == .duckduckgo)
    }

    @Test("Keychain accounts are unique and present exactly for the key-accepting providers")
    func keychainAccounts() {
        let accounts = WebSearchProvider.allCases.compactMap(\.keychainAccount)
        #expect(Set(accounts).count == accounts.count, "two providers sharing an account would leak one key into the other's calls")
        for provider in WebSearchProvider.allCases {
            #expect((provider.keychainAccount != nil) == provider.acceptsKey)
        }
    }

    @Test("Every key-requiring provider also accepts a key")
    func requiresImpliesAccepts() {
        for provider in WebSearchProvider.allCases where provider.requiresKey {
            #expect(provider.acceptsKey)
        }
    }

    @Test("Keenable is keyless-capable but key-accepting; DuckDuckGo is neither")
    func keylessMatrix() {
        #expect(!WebSearchProvider.keenable.requiresKey)
        #expect(WebSearchProvider.keenable.acceptsKey)
        #expect(!WebSearchProvider.duckduckgo.requiresKey)
        #expect(!WebSearchProvider.duckduckgo.acceptsKey)
    }

    @Test("Every key-accepting provider links a key dashboard")
    func dashboardsPresent() {
        for provider in WebSearchProvider.allCases where provider.acceptsKey {
            #expect(provider.keyDashboardURL != nil, "\(provider.rawValue) has a key field but no way to mint a key")
        }
    }

    @Test("Brave's subtitle discloses the card + auto-billing and never claims 'free' (#2043)")
    func braveBillingHonesty() {
        let subtitle = WebSearchProvider.brave.subtitle
        #expect(subtitle.localizedCaseInsensitiveContains("card"))
        #expect(subtitle.localizedCaseInsensitiveContains("auto-billed"))
        #expect(!subtitle.localizedCaseInsensitiveContains("free"))
    }

    @Test("Parallel's subtitle carries the recommendation")
    func parallelRecommended() {
        #expect(WebSearchProvider.parallel.subtitle.localizedCaseInsensitiveContains("recommended"))
    }

    @Test("Raw values survive: a pre-roster stored choice still decodes")
    func legacyRawValuesDecode() {
        // These strings are persisted in UserDefaults on existing
        // installs; renaming a case would silently reset users to
        // the default.
        #expect(WebSearchProvider(rawValue: "duckduckgo") == .duckduckgo)
        #expect(WebSearchProvider(rawValue: "brave") == .brave)
        #expect(WebSearchProvider(rawValue: "tavily") == .tavily)
    }
}

@MainActor
@Suite("Web-search config: default, promotion, usability")
struct WebSearchConfigRosterTests {
    private func freshDefaults() -> UserDefaults {
        let name = "web-search-roster-tests-\(UUID().uuidString)"
        let d = UserDefaults(suiteName: name)!
        d.removePersistentDomain(forName: name)
        return d
    }

    @Test("A fresh install defaults to Keenable (#2041)")
    func freshDefaultIsKeenable() {
        let config = WebSearchConfig(defaults: freshDefaults(), keychain: RosterInMemoryKeychain())
        #expect(config.provider == .keenable)
    }

    @Test("An explicit pre-existing choice survives the default flip")
    func persistedChoiceWins() {
        let defaults = freshDefaults()
        defaults.set("duckduckgo", forKey: "rapid.webSearch.provider")
        let config = WebSearchConfig(defaults: defaults, keychain: RosterInMemoryKeychain())
        #expect(config.provider == .duckduckgo)
    }

    @Test("A corrupted stored value falls back to the Keenable default")
    func corruptedValueFallsBack() {
        let defaults = freshDefaults()
        defaults.set("altavista", forKey: "rapid.webSearch.provider")
        let config = WebSearchConfig(defaults: defaults, keychain: RosterInMemoryKeychain())
        #expect(config.provider == .keenable)
    }

    @Test("Pasting a keyed-provider key promotes from either keyless backend", arguments: [WebSearchProvider.keenable, .duckduckgo])
    func autoPromoteFromKeyless(start: WebSearchProvider) {
        let config = WebSearchConfig(defaults: freshDefaults(), keychain: RosterInMemoryKeychain())
        config.provider = start
        #expect(config.setAPIKey("pk-123", for: .parallel))
        #expect(config.provider == .parallel)
    }

    @Test("An explicit keyed choice is never overridden by another provider's key")
    func explicitKeyedChoiceWins() {
        let config = WebSearchConfig(defaults: freshDefaults(), keychain: RosterInMemoryKeychain())
        config.provider = .tavily
        #expect(config.setAPIKey("BSA-abc", for: .brave))
        #expect(config.provider == .tavily)
    }

    @Test("A Keenable key is stored in place without any promotion dance")
    func keenableKeyStaysPut() {
        let config = WebSearchConfig(defaults: freshDefaults(), keychain: RosterInMemoryKeychain())
        #expect(config.provider == .keenable)
        #expect(config.setAPIKey("keen_xyz", for: .keenable))
        #expect(config.provider == .keenable)
        #expect(config.apiKey(for: .keenable) == "keen_xyz")
    }

    @Test("Keyless providers are always usable; key-requiring ones only with a key")
    func usability() {
        let config = WebSearchConfig(defaults: freshDefaults(), keychain: RosterInMemoryKeychain())
        config.provider = .keenable
        #expect(config.currentProviderUsable)
        config.provider = .parallel
        #expect(!config.currentProviderUsable)
        #expect(config.setAPIKey("pk-123", for: .parallel))
        #expect(config.currentProviderUsable)
    }
}

@Suite("Web-search dispatch plan (fallback chain, no network)")
struct WebSearchDispatchPlanTests {
    @Test("A key-requiring provider without a key degrades to Keenable with a note", arguments: [WebSearchProvider.parallel, .tavily, .brave])
    func keylessFallsBackToKeenable(provider: WebSearchProvider) {
        let plan = WebSearchTool.dispatchPlan(provider: provider, hasKey: false)
        #expect(plan.effective == .keenable)
        let note = plan.fallbackNote ?? ""
        #expect(note.contains(provider.displayName))
        #expect(note.contains("Keenable"))
        #expect(note.contains("Settings → Tools"))
    }

    @Test("Keyless providers run as themselves with no note", arguments: [WebSearchProvider.keenable, .duckduckgo])
    func keylessRunsDirect(provider: WebSearchProvider) {
        let plan = WebSearchTool.dispatchPlan(provider: provider, hasKey: false)
        #expect(plan.effective == provider)
        #expect(plan.fallbackNote == nil)
    }

    @Test("A keyed provider with its key runs as itself")
    func keyedRunsDirect() {
        let plan = WebSearchTool.dispatchPlan(provider: .parallel, hasKey: true)
        #expect(plan.effective == .parallel)
        #expect(plan.fallbackNote == nil)
    }
}

@Suite("Keenable client request/response contracts")
struct KeenableClientTests {
    // Captured live 2026-08-18 from the public MCP endpoint (trimmed).
    static let keylessText = """
        Title: Apple M3 Ultra — 96–512 GB VRAM: Which LLMs Can It Run?
        URL: https://canitrun.dev/gpus/m3-ultra
        Published: 2026-08-01
        Acquired: 2026-08-04
        Snippets:
        Apple M3 Ultra ships in 96–512 GB unified-memory configurations at 819 GB/s.

        ---

        Title: Mac Studio M3 Ultra: What AI Models Can It Run?
        URL: https://modelpiper.com/mac-studio/m3-ultra
        Acquired: 2026-08-01
        Snippets:
        Chip Apple M3 Ultra CPU cores 28 or 32
        Memory bandwidth 819 GB/s
        Published: 2026-07-24

        ---

        Title: Block with no link is skipped
        Snippets:
        Body without a URL line.

        ---

        Title: Unsafe scheme is skipped
        URL: javascript:alert(1)
        Snippets:
        Nope.
        """

    private static func rpcEnvelope(text: String) -> Data {
        let obj: [String: Any] = [
            "jsonrpc": "2.0", "id": 1,
            "result": ["content": [["type": "text", "text": text]]],
        ]
        return try! JSONSerialization.data(withJSONObject: obj)
    }

    private static func response(statusCode: Int) -> HTTPURLResponse {
        HTTPURLResponse(
            url: URL(string: KeenableSearchClient.restEndpoint)!,
            statusCode: statusCode,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "application/json"]
        )!
    }

    @Test(
        "Keyed HTTP account failures stamp their producer-owned diagnosis",
        arguments: [
            (401, FailureDiagnosis.Kind.webSearchKeyRejected),
            (403, FailureDiagnosis.Kind.webSearchKeyRejected),
            (402, FailureDiagnosis.Kind.webSearchKeyQuotaExceeded),
            (429, FailureDiagnosis.Kind.webSearchKeyRateLimited),
        ]
    )
    func keyedAccountFailureKind(statusCode: Int, expected: FailureDiagnosis.Kind) async {
        let transport: WebSearchTool.KeenableTransport = { request in
            #expect(request.value(forHTTPHeaderField: "X-API-Key") == "keen_valid")
            return (Data("{}".utf8), Self.response(statusCode: statusCode))
        }
        let result = await WebSearchTool.runKeenable(
            query: "current news",
            apiKey: "keen_valid",
            fallbackNote: nil,
            transport: transport
        )
        #expect(result.isError)
        #expect(result.failureKind == expected)
    }

    @Test("A malformed stored key points to key settings without touching transport")
    func malformedKeyFailureKind() async {
        let transport: WebSearchTool.KeenableTransport = { _ in
            Issue.record("Malformed key must fail before transport")
            throw CancellationError()
        }
        let result = await WebSearchTool.runKeenable(
            query: "current news",
            apiKey: "keen_valid\r\nX-Evil: 1",
            fallbackNote: nil,
            transport: transport
        )
        #expect(result.isError)
        #expect(result.failureKind == .webSearchKeyRejected)
    }

    @Test("Keyless request is a single JSON-RPC tools/call POST with the snippet cap clamped to the server minimum")
    func keylessRequestShape() throws {
        let req = try #require(KeenableSearchClient.buildKeylessRequest(query: "hello world", snippetMaxLength: 100))
        #expect(req.url?.absoluteString == KeenableSearchClient.mcpEndpoint)
        #expect(req.httpMethod == "POST")
        #expect(req.value(forHTTPHeaderField: "Accept")?.contains("text/event-stream") == true)
        let body = try #require(req.httpBody)
        let obj = try #require(try JSONSerialization.jsonObject(with: body) as? [String: Any])
        #expect(obj["jsonrpc"] as? String == "2.0")
        #expect(obj["method"] as? String == "tools/call")
        let params = try #require(obj["params"] as? [String: Any])
        #expect(params["name"] as? String == "search_web_pages")
        let args = try #require(params["arguments"] as? [String: Any])
        #expect(args["query"] as? String == "hello world")
        // Schema minimum is 180 — a smaller local cap must clamp up,
        // not send an invalid value.
        #expect(args["snippet_max_length"] as? Int == 180)
    }

    @Test("Keyed request hits the REST endpoint with the key in X-API-Key")
    func keyedRequestShape() throws {
        let req = try #require(KeenableSearchClient.buildKeyedRequest(query: "q", apiKey: "keen_abc", snippetMaxLength: 240))
        #expect(req.url?.absoluteString == KeenableSearchClient.restEndpoint)
        #expect(req.value(forHTTPHeaderField: "X-API-Key") == "keen_abc")
        let obj = try JSONSerialization.jsonObject(with: req.httpBody!) as? [String: Any]
        #expect(obj?["query"] as? String == "q")
        #expect(obj?["snippet_max_length"] as? Int == 240)
    }

    @Test("A key with a control byte refuses to build (header injection)")
    func keyedRequestRejectsControlBytes() {
        #expect(KeenableSearchClient.buildKeyedRequest(query: "q", apiKey: "keen\r\nX-Evil: 1", snippetMaxLength: 240) == nil)
    }

    @Test("Keyless text blocks parse into title + URL + joined snippet, skipping linkless and unsafe blocks")
    func textBlockParsing() {
        let results = KeenableSearchClient.parseTextBlocks(Self.keylessText, cap: 6)
        #expect(results.count == 2)
        #expect(results[0].title.hasPrefix("Apple M3 Ultra"))
        #expect(results[0].url == "https://canitrun.dev/gpus/m3-ultra")
        #expect(results[0].snippet.contains("819 GB/s"))
        // Multi-line snippet bodies join into one line, and metadata
        // lines never leak into the snippet — whether they appear
        // BEFORE the Snippets: header or trail AFTER the body (the
        // fixture carries a trailing "Published:" line).
        #expect(results[1].snippet == "Chip Apple M3 Ultra CPU cores 28 or 32 Memory bandwidth 819 GB/s")
        #expect(!results[0].snippet.contains("Acquired"))
        #expect(!results[1].snippet.contains("Published"))
    }

    @Test("The result cap truncates the block list")
    func textBlockCap() {
        let results = KeenableSearchClient.parseTextBlocks(Self.keylessText, cap: 1)
        #expect(results.count == 1)
    }

    @Test("A plain JSON-RPC envelope parses end-to-end")
    func keylessEnvelopeParses() {
        let results = KeenableSearchClient.parseKeylessResults(Self.rpcEnvelope(text: Self.keylessText), cap: 6)
        #expect(results?.count == 2)
    }

    @Test("An SSE-framed envelope parses identically")
    func keylessSSEEnvelopeParses() {
        let json = String(data: Self.rpcEnvelope(text: Self.keylessText), encoding: .utf8)!
        let sse = "event: message\ndata: \(json)\n\n"
        let results = KeenableSearchClient.parseKeylessResults(Data(sse.utf8), cap: 6)
        #expect(results?.count == 2)
    }

    @Test("A JSON-RPC error envelope is nil (degrade), not an empty success")
    func keylessErrorEnvelopeIsNil() {
        let err = try! JSONSerialization.data(withJSONObject: [
            "jsonrpc": "2.0", "id": 1,
            "error": ["code": -32000, "message": "boom"],
        ])
        #expect(KeenableSearchClient.parseKeylessResults(err, cap: 6) == nil)
    }

    @Test("A tool-level isError result is nil (degrade)")
    func keylessToolErrorIsNil() {
        let err = try! JSONSerialization.data(withJSONObject: [
            "jsonrpc": "2.0", "id": 1,
            "result": ["isError": true, "content": [["type": "text", "text": "rate limited"]]],
        ])
        #expect(KeenableSearchClient.parseKeylessResults(err, cap: 6) == nil)
    }

    @Test("Malformed keyless bytes are nil, a resultless text is an empty success")
    func keylessMalformedVsEmpty() {
        #expect(KeenableSearchClient.parseKeylessResults(Data("not json".utf8), cap: 6) == nil)
        let empty = KeenableSearchClient.parseKeylessResults(Self.rpcEnvelope(text: "No results."), cap: 6)
        #expect(empty?.isEmpty == true)
    }

    @Test("Keyed REST results prefer the query snippet over the static description")
    func keyedParsePrefersSnippet() {
        let data = try! JSONSerialization.data(withJSONObject: [
            "query": "q",
            "results": [
                ["title": "A", "url": "https://a.example/", "description": "static", "snippet": "relevant"],
                ["title": "B", "url": "https://b.example/", "description": "static only"],
                ["title": "C", "url": "file:///etc/passwd", "snippet": "unsafe"],
            ],
        ])
        let results = KeenableSearchClient.parseKeyedResults(data, cap: 6)
        #expect(results?.count == 2)
        #expect(results?[0].snippet == "relevant")
        #expect(results?[1].snippet == "static only")
    }

    @Test("Malformed keyed JSON is nil, not an empty success")
    func keyedMalformedIsNil() {
        #expect(KeenableSearchClient.parseKeyedResults(Data("{}".utf8), cap: 6) == nil)
    }
}

@Suite("Parallel client request/response contracts")
struct ParallelClientTests {
    @Test("Request pins endpoint, auth header, mode and the local display budgets")
    func requestShape() throws {
        let req = try #require(ParallelSearchClient.buildRequest(
            query: "swift concurrency",
            apiKey: "pk-1",
            maxResults: 6,
            maxCharsPerResult: 240
        ))
        #expect(req.url?.absoluteString == ParallelSearchClient.endpoint)
        #expect(req.httpMethod == "POST")
        #expect(req.value(forHTTPHeaderField: "x-api-key") == "pk-1")
        let obj = try #require(try JSONSerialization.jsonObject(with: req.httpBody!) as? [String: Any])
        #expect(obj["objective"] as? String == "swift concurrency")
        #expect(obj["search_queries"] as? [String] == ["swift concurrency"])
        // Pinned, not inherited: the server default could drift.
        #expect(obj["mode"] as? String == "advanced")
        let advanced = try #require(obj["advanced_settings"] as? [String: Any])
        #expect(advanced["max_results"] as? Int == 6)
        let excerpts = try #require(advanced["excerpt_settings"] as? [String: Any])
        #expect(excerpts["max_chars_per_result"] as? Int == 240)
    }

    @Test("A key with a control byte refuses to build")
    func rejectsControlBytes() {
        #expect(ParallelSearchClient.buildRequest(query: "q", apiKey: "pk\r\nX: y", maxResults: 6, maxCharsPerResult: 240) == nil)
    }

    @Test("Excerpts join into one snippet; unsafe URLs are filtered; empty results decode as empty")
    func responseParsing() {
        let data = try! JSONSerialization.data(withJSONObject: [
            "search_id": "s1",
            "results": [
                ["url": "https://a.example/", "title": "A", "excerpts": ["first passage", "  second  "]],
                ["url": "https://b.example/", "title": "B", "excerpts": []],
                ["url": "javascript:alert(1)", "title": "evil", "excerpts": ["x"]],
            ],
        ])
        let results = ParallelSearchClient.parseResults(data, cap: 6)
        #expect(results?.count == 2)
        #expect(results?[0].snippet == "first passage … second")
        #expect(results?[1].snippet == "")

        let empty = try! JSONSerialization.data(withJSONObject: ["search_id": "s2", "results": [] as [Any]])
        #expect(ParallelSearchClient.parseResults(empty, cap: 6)?.isEmpty == true)
        #expect(ParallelSearchClient.parseResults(Data("nope".utf8), cap: 6) == nil)
    }
}

@MainActor
@Suite("Settings web-search captions")
struct WebSearchSettingsCaptionTests {
    @Test("A key-requiring provider's empty slot names the Keenable fallback")
    func requiredKeyCaption() {
        let caption = SettingsToolsPanel.noKeyCaption(for: .parallel)
        #expect(caption.contains("Keenable"))
    }

    @Test("Keenable's empty slot says it works without a key")
    func optionalKeyCaption() {
        let caption = SettingsToolsPanel.noKeyCaption(for: .keenable)
        #expect(caption.contains("works without one"))
        #expect(caption.contains("Keenable"))
    }
}

/// Local in-memory keychain — mirrors the private one in
/// ``BuiltinToolsTests`` (private types don't cross files).
private final class RosterInMemoryKeychain: KeychainStoring, @unchecked Sendable {
    private var store: [String: String] = [:]
    private let lock = NSLock()

    func read(account: String) -> String? {
        lock.lock(); defer { lock.unlock() }
        return store[account]
    }

    func write(account: String, secret: String) -> Bool {
        lock.lock(); defer { lock.unlock() }
        store[account] = secret
        return true
    }

    func delete(account: String) -> Bool {
        lock.lock(); defer { lock.unlock() }
        store.removeValue(forKey: account)
        return true
    }
}
