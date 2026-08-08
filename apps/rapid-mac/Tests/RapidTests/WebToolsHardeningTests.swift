import Foundation
import Testing
@testable import Rapid

/// Regression coverage for the browse/search hardening pass on top of the
/// built-in web tools. These modules carry security- and correctness-critical
/// logic that previously had no direct unit tests.
@Suite("Web tools hardening")
struct WebToolsHardeningTests {
    // MARK: - Relative-date query grounding

    @Test("Last-week news query gets the previous complete calendar week")
    func lastWeekQueryGetsConcreteDates() throws {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = try #require(TimeZone(secondsFromGMT: 0))
        let now = try #require(calendar.date(from: DateComponents(
            year: 2026, month: 8, day: 8, hour: 19
        )))

        let prepared = WebSearchTool.preparedQuery(
            "What's a major news story from the last week?",
            now: now,
            calendar: calendar
        )

        #expect(prepared.contains("2026-07-26 through 2026-08-01"))
        #expect(prepared.hasPrefix("What's a major news story from the last week?"))
    }

    @Test("Chinese last-week query gets the same explicit date window")
    func chineseLastWeekQueryGetsConcreteDates() throws {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = try #require(TimeZone(secondsFromGMT: 0))
        let now = try #require(calendar.date(from: DateComponents(
            year: 2026, month: 8, day: 8
        )))

        let prepared = WebSearchTool.preparedQuery(
            "上周有什么重大新闻？",
            now: now,
            calendar: calendar
        )
        #expect(prepared.contains("2026-07-26 through 2026-08-01"))
    }

    @Test("Last weekend is not broadened into a week-long query")
    func lastWeekendIsNotRewritten() {
        let query = "What happened last weekend?"
        #expect(WebSearchTool.preparedQuery(query) == query)
    }

    @Test("A historical last-week-of phrase is not treated as relative")
    func historicalLastWeekOfIsNotRewritten() {
        let query = "What happened in the last week of July 2020?"
        #expect(WebSearchTool.preparedQuery(query) == query)
    }

    @Test("Chinese last weekend is not broadened into a week-long query")
    func chineseLastWeekendIsNotRewritten() {
        let query = "上周末有什么活动？"
        #expect(WebSearchTool.preparedQuery(query) == query)
    }

    @Test("The Chinese week-before-last phrase is not shifted forward")
    func chineseWeekBeforeLastIsNotRewritten() {
        let query = "上上周有什么新闻？"
        #expect(WebSearchTool.preparedQuery(query) == query)
    }

    @Test("A standalone Chinese last-week phrase still matches beside last-weekend")
    func standaloneChineseLastWeekStillMatches() throws {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = try #require(TimeZone(secondsFromGMT: 0))
        let now = try #require(calendar.date(from: DateComponents(
            year: 2026, month: 8, day: 8
        )))
        let prepared = WebSearchTool.preparedQuery(
            "总结上周新闻，不包括上周末",
            now: now,
            calendar: calendar
        )
        #expect(prepared.contains("2026-07-26 through 2026-08-01"))
    }

    @Test("Date constraint stays Gregorian for a non-Gregorian user calendar")
    func dateConstraintUsesGregorianYear() throws {
        var calendar = Calendar(identifier: .buddhist)
        calendar.timeZone = try #require(TimeZone(secondsFromGMT: 0))
        let now = try #require(ISO8601DateFormatter().date(from: "2026-08-08T12:00:00Z"))
        let prepared = WebSearchTool.preparedQuery(
            "major news last week",
            now: now,
            calendar: calendar
        )
        #expect(prepared.contains("2026-"))
        #expect(!prepared.contains("2569-"))
    }

    @Test("Timeless query is not rewritten")
    func timelessQueryPassesThrough() {
        let query = "Swift concurrency actor isolation"
        #expect(WebSearchTool.preparedQuery(query) == query)
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
