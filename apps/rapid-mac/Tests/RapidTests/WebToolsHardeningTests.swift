import Foundation
import Testing
@testable import Rapid

/// Regression coverage for the browse/search hardening pass on top of the
/// built-in web tools. These modules carry security- and correctness-critical
/// logic that previously had no direct unit tests.
@Suite("Web tools hardening")
struct WebToolsHardeningTests {
    // MARK: - #1535 provider schema failures

    @Test("Brave distinguishes malformed JSON from a valid empty result set")
    func braveParseFailureIsNotEmptySuccess() {
        #expect(BraveSearchClient.parseResults(Data(#"{"web":{"results":[]}}"#.utf8), cap: 6) == [])
        #expect(BraveSearchClient.parseResults(Data(#"{}"#.utf8), cap: 6) == [])
        #expect(BraveSearchClient.parseResults(Data(#"{"web":{"unexpected":[]}}"#.utf8), cap: 6) == nil)
        #expect(BraveSearchClient.parseResults(Data("not-json".utf8), cap: 6) == nil)
    }

    @Test("Tavily distinguishes malformed JSON from a valid empty result set")
    func tavilyParseFailureIsNotEmptySuccess() {
        #expect(TavilySearchClient.parseResults(Data(#"{"results":[]}"#.utf8), cap: 6) == [])
        #expect(TavilySearchClient.parseResults(Data(#"{"unexpected":[]}"#.utf8), cap: 6) == nil)
        #expect(TavilySearchClient.parseResults(Data("not-json".utf8), cap: 6) == nil)
    }

    // MARK: - #1535 relative browse destinations

    @Test("HTML links and images resolve against the fetched page's final URL")
    func relativeDestinationsResolveAgainstFinalURL() throws {
        let base = try #require(URL(string: "https://example.com/articles/entry.html"))
        let result = HTMLToMarkdown.extract(
            #"<main><a href="../docs/guide">Guide</a><img src="/assets/chart.png" alt="Chart"></main>"#,
            baseURL: base
        )
        #expect(result.markdown.contains("[Guide](https://example.com/docs/guide)"))
        #expect(result.markdown.contains("![Chart](https://example.com/assets/chart.png)"))
    }

    @Test("Resolving relative destinations never revives an unsafe scheme")
    func relativeResolutionPreservesSchemeSafety() throws {
        let base = try #require(URL(string: "https://example.com/articles/entry.html"))
        let result = HTMLToMarkdown.extract(
            #"<main><a href="javascript:alert(1)">bad</a><a href="next">good</a></main>"#,
            baseURL: base
        )
        #expect(!result.markdown.contains("javascript:"))
        #expect(result.markdown.contains("bad"))
        #expect(result.markdown.contains("[good](https://example.com/articles/next)"))
    }

    // MARK: - #4 DuckDuckGo query encoding

    @Test("A query with & and # is one opaque q value, not injected parameters")
    func duckDuckGoURLEscapesSubDelimiters() throws {
        let url = try #require(WebSearchTool.duckDuckGoSearchURL(query: "cats & dogs #1"))
        // Exactly one query item named q carrying the whole string — no stray
        // parameters split off an unescaped &, no fragment split off a #.
        let comps = try #require(URLComponents(url: url, resolvingAgainstBaseURL: false))
        #expect(comps.queryItems?.count == 1)
        #expect(comps.queryItems?.first?.name == "q")
        #expect(comps.queryItems?.first?.value == "cats & dogs #1")
        #expect(url.fragment == nil)
        #expect(url.host == "html.duckduckgo.com")
        // The raw string must not contain a bare & or # in the query.
        let raw = url.absoluteString
        #expect(!raw.contains("q=cats & dogs"))
        #expect(raw.contains("html.duckduckgo.com/html/?q="))
    }

    @Test("A benign query still round-trips unchanged")
    func duckDuckGoURLBenignQuery() throws {
        let url = try #require(WebSearchTool.duckDuckGoSearchURL(query: "swift concurrency"))
        let comps = try #require(URLComponents(url: url, resolvingAgainstBaseURL: false))
        #expect(comps.queryItems?.first?.value == "swift concurrency")
    }

    @Test("A literal + survives as %2B, not a form-decoded space")
    func duckDuckGoURLEscapesPlus() throws {
        let url = try #require(WebSearchTool.duckDuckGoSearchURL(query: "C++ vs Rust"))
        // Raw wire form must carry %2B so a form-decoding endpoint keeps the
        // pluses rather than reading them as spaces ("C  vs Rust").
        #expect(url.absoluteString.contains("C%2B%2B"))
        #expect(!url.absoluteString.contains("q=C++"))
        // And it still decodes back to the exact query.
        let comps = try #require(URLComponents(url: url, resolvingAgainstBaseURL: false))
        #expect(comps.queryItems?.first?.value == "C++ vs Rust")
    }

    // MARK: - #5 anti-bot class-token boundary

    @Test("Token inside the class value is detected, both class orderings")
    func classTokenDetectedInsideClassValue() {
        #expect(WebSearchTool.containsResultBodyClassToken(
            #"<div class="result__body links_main">x</div>"#))
        #expect(WebSearchTool.containsResultBodyClassToken(
            #"<div class="links_main links_deep result__body">x</div>"#))
    }

    @Test("Token echoed in a NON-class attribute does not count")
    func classTokenRejectedInOtherAttribute() {
        // An anti-bot page can carry the string anywhere; only the class value
        // proves a real result container.
        #expect(!WebSearchTool.containsResultBodyClassToken(
            #"<div class="notice" data-x="result__body">blocked</div>"#))
        #expect(!WebSearchTool.containsResultBodyClassToken(
            #"<span title="result__body"></span>"#))
    }

    @Test("A partial token inside the class value does not count")
    func classTokenRejectsPartialMatch() {
        #expect(!WebSearchTool.containsResultBodyClassToken(
            #"<div class="result__bodyish">x</div>"#))
    }

    // MARK: - #2 bounded DNS resolve

    @Test("resolve returns via the deadline-guarded path for a local host")
    func resolveResolvesLocalHost() async throws {
        // ``localhost`` resolves from /etc/hosts with no network, so this
        // exercises the fire-and-forget lookup + single-resume bridge
        // deterministically. It must return the loopback address(es), and the
        // guard must treat them all as blocked.
        let ips = try await BrowseSSRFGuard.resolve("localhost")
        #expect(!ips.isEmpty)
        #expect(ips.allSatisfy { $0.isBlocked })
    }

    @Test("resolve yields empty for an NXDOMAIN reserved TLD")
    func resolveEmptyForInvalidTLD() async throws {
        // RFC 6761 guarantees ``.invalid`` never resolves; getaddrinfo returns
        // an error → empty list (→ caller treats as unresolvable) rather than a
        // hang or crash.
        let ips = try await BrowseSSRFGuard.resolve("nonexistent-host.invalid")
        #expect(ips.isEmpty)
    }

    // MARK: - Weather geocoding overflow

    @Test("An Int.max population in untrusted geocoding JSON doesn't trap")
    func weatherGeocodingPopulationOverflowIsSafe() {
        // Two same-named hits force the population-margin check; a runner-up
        // near Int.max would trap on `max(1, pop) * 5`. The overflow-reporting
        // multiply must treat the (unreachable) threshold as ambiguous and
        // return nil instead of crashing the process on hostile JSON.
        let json = Data(#"""
        {"results":[
          {"name":"Springfield","latitude":1,"longitude":2,"population":9223372036854775807},
          {"name":"Springfield","latitude":3,"longitude":4,"population":9223372036854775807}
        ]}
        """#.utf8)
        let hit = WeatherTool.parseGeocodingResponse(json, location: "Springfield")
        #expect(hit == nil)
    }
}
