import Foundation
import Testing
@testable import Rapid

/// Contracts for the DuckDuckGo throttle path.
///
/// Background (measured 2026-08-05 against
/// ``https://html.duckduckgo.com/html/?q=…``): the first request of a session
/// returns HTTP 200 with ten ``result__a`` blocks; every request after it
/// returns HTTP **202** with a ~14 KB page carrying no result blocks — five
/// different queries, both a plain Safari UA and the tool's own UA. So the
/// endpoint throttles per IP, and 202-with-no-results is its fingerprint.
///
/// Before this pass, 202 sailed through the ``(200..<300)`` success guard, the
/// parser found nothing, and the user got "Web search couldn't finish. Check
/// its settings, then try again." — pointing at a Settings page where nothing
/// is wrong. These tests pin the classification and the copy on both sides of
/// the boundary: what the user reads, and what the model reads.
@Suite("Web search throttle")
struct WebSearchThrottleTests {

    /// One real result block, in the class ordering DDG ships today.
    private static let resultsPage = """
        <html><body>
        <div class="links_main links_deep result__body">
          <a class="result__a" href="/l/?uddg=https%3A%2F%2Fwww.swift.org%2F">Swift.org</a>
          <a class="result__snippet">Swift is a general-purpose programming language.</a>
        </div>
        </body></html>
        """

    /// The measured throttle body: a normal-looking page with no result blocks
    /// and — importantly — none of the anti-bot markers the old detector
    /// keyed on. Status code is the only signal available here.
    private static let throttledPage = """
        <html><head><title>DuckDuckGo</title></head><body>
        <form action="/html/" method="post"><input name="q" value="current stable version of Swift"></form>
        <div class="header">DuckDuckGo</div>
        </body></html>
        """

    /// The older shape: HTTP 200 carrying the challenge modal.
    private static let antiBotPage = """
        <html><body>
        <div class="anomaly-modal__title">Unfortunately, bots use DuckDuckGo too.</div>
        <form action="/html/?cc=botnet"><input name="q" value="swift"></form>
        </body></html>
        """

    // MARK: - Status-code classification

    @Test("HTTP 202 with a non-results body is a throttle, not a success")
    func status202WithoutResultsIsThrottled() {
        // The regression this whole change exists for: 202 passed the 2xx
        // guard, parsed to zero results, and the user was told to go check
        // settings that were already correct.
        #expect(WebSearchTool.isDuckDuckGoThrottled(
            statusCode: 202,
            html: Self.throttledPage
        ))
    }

    @Test("HTTP 202 that still carries result blocks is NOT a throttle")
    func status202WithResultsIsNotThrottled() {
        // The signature is "202 with a non-results body", not 202 alone. A body
        // with hits in it is a body we can parse, whatever the status line says.
        #expect(!WebSearchTool.isDuckDuckGoThrottled(
            statusCode: 202,
            html: Self.resultsPage
        ))
    }

    @Test("A plain 200 results page is never a throttle")
    func status200WithResultsIsNotThrottled() {
        #expect(!WebSearchTool.isDuckDuckGoThrottled(
            statusCode: 200,
            html: Self.resultsPage
        ))
    }

    @Test("A 200 with zero hits and no challenge markers stays 'no results'")
    func status200EmptyIsNotThrottled() {
        // A query that genuinely matched nothing must keep reading as an empty
        // result set — mislabelling it a throttle would send the user off to
        // buy an API key for a search that simply has no answers.
        #expect(!WebSearchTool.isDuckDuckGoThrottled(
            statusCode: 200,
            html: "<html><body><div class=\"no-results\">No results.</div></body></html>"
        ))
    }

    @Test("Explicit rate-limit and block statuses are throttles too", arguments: [429, 403])
    func rateLimitStatusesAreThrottled(code: Int) {
        // These used to fall out of the non-2xx branch as a bare
        // "DuckDuckGo returned HTTP 429", which is the same situation for the
        // user as the 202 soft-throttle.
        #expect(WebSearchTool.isDuckDuckGoThrottled(statusCode: code, html: Self.throttledPage))
    }

    @Test("The 200 anti-bot modal is still detected")
    func antiBotModalStillDetected() {
        #expect(WebSearchTool.isDuckDuckGoThrottled(statusCode: 200, html: Self.antiBotPage))
        #expect(WebSearchTool.detectDDGAntiBot(Self.antiBotPage))
    }

    // MARK: - Model-visible payload

    @Test("The throttle result is stamped with the rate-limited kind, not sniffed")
    func throttleResultCarriesFailureKind() {
        let result = WebSearchTool.duckDuckGoThrottledResult()
        #expect(result.isError)
        #expect(result.failureKind == .webSearchRateLimited)
    }

    @Test("The model-visible text names the backend as the limit, not the capability")
    func throttleContentDoesNotReadAsMissingCapability() {
        // This string is what reaches the model as the ``role: "tool"`` body —
        // the UI diagnosis never does. The old text ("DuckDuckGo blocked this
        // request (anti-bot rate limit)") left room for a small model to answer
        // "I don't have the ability to browse the web", contradicting the tool
        // card the user is looking at. Naming the tool as enabled closes that.
        let content = WebSearchTool.duckDuckGoThrottleContent
        #expect(content.contains("web_search tool is enabled and working"))
        #expect(content.contains("rate-limited"))
        #expect(content.contains("Brave Search"))
        #expect(content.contains("Tavily"))
        #expect(content.contains("Settings → Tools"))
    }

    @Test("A pending fallback note survives the throttle path")
    func throttleResultKeepsFallbackNote() {
        // "Brave is selected but no key is set" matters MORE once DDG has
        // throttled — the old anti-bot branch dropped the note on the floor.
        let note = "Note: Brave Search is selected but no API key is set."
        let result = WebSearchTool.duckDuckGoThrottledResult(fallbackNote: note)
        #expect(result.content.contains(note))
        #expect(result.content.contains(WebSearchTool.duckDuckGoThrottleContent))
    }

    // MARK: - User-visible diagnosis

    @Test("The user-visible message names DuckDuckGo and the real remedy")
    func diagnosisCopyIsActionable() {
        let diagnosis = FailureDiagnoser.diagnosis(for: .webSearchRateLimited)
        #expect(diagnosis.message.contains("DuckDuckGo"))
        #expect(diagnosis.message.contains("Brave Search"))
        #expect(diagnosis.message.contains("Tavily"))
        #expect(diagnosis.message.contains("Settings → Tools"))
        // The dead end this replaced. Settings are not the problem when the
        // free backend throttles, so the copy must not send the user there to
        // "check" anything.
        #expect(!diagnosis.message.contains("Check its settings"))
    }

    @Test("The rate-limited diagnosis offers the Settings deep-link, not Retry")
    func diagnosisActionOpensWebSearchSettings() {
        // Retry re-runs straight into the same per-IP throttle.
        #expect(FailureDiagnoser.diagnosis(for: .webSearchRateLimited).action == .openWebSearchSettings)
    }

    // MARK: - Inline tool-card button

    @Test("The rate-limited card offers its button when Settings is reachable")
    func inlineActionOfferedWhenRoutable() {
        let diagnosis = FailureDiagnoser.diagnosis(for: .webSearchRateLimited)
        #expect(FailureDiagnosis.inlineToolCardAction(
            for: diagnosis,
            canRouteToSettings: true
        ) == .openWebSearchSettings)
    }

    @Test("No router, no button — an inert button is the bug being fixed")
    func inlineActionSuppressedWithoutRouter() {
        // The card resolves ``SettingsRouter`` optionally so it renders in
        // hosts that never injected one. Those hosts also have no Settings
        // window to open, so the button must be ABSENT, not merely harmless.
        let diagnosis = FailureDiagnoser.diagnosis(for: .webSearchRateLimited)
        #expect(FailureDiagnosis.inlineToolCardAction(
            for: diagnosis,
            canRouteToSettings: false
        ) == nil)
    }

    @Test("Retry-shaped diagnoses get no inline button")
    func inlineActionIgnoresRetry() {
        // Retry has to rewind the chat turn; the assistant row above the card
        // owns that. Rendering it here would be a second, weaker Retry.
        for kind in [FailureDiagnosis.Kind.webSearchUnavailable, .webSearchOffline, .toolFailed] {
            #expect(FailureDiagnosis.inlineToolCardAction(
                for: FailureDiagnoser.diagnosis(for: kind),
                canRouteToSettings: true
            ) == nil)
        }
    }

    @Test("A still-running or successful tool call gets no inline button")
    func inlineActionNilWithoutDiagnosis() {
        #expect(FailureDiagnosis.inlineToolCardAction(for: nil, canRouteToSettings: true) == nil)
    }

    @Test("Other web_search failures keep their existing copy")
    func unrelatedWebSearchCopyUnchanged() {
        let unavailable = FailureDiagnoser.diagnosis(for: .webSearchUnavailable)
        #expect(unavailable.message == "Web search couldn't finish. Check its settings, then try again.")
        #expect(unavailable.action == .retry)
    }

    // MARK: - Fallback classification (restored transcripts)

    @Test("A restored throttle row without a stored kind still classifies")
    func classifierRecognisesThrottleContent() {
        let kind = FailureDiagnoser.toolFailureKind(
            toolName: "web_search",
            content: WebSearchTool.duckDuckGoThrottleContent,
            isError: true
        )
        #expect(kind == .webSearchRateLimited)
    }

    @Test("The legacy anti-bot wording classifies as rate-limited too")
    func classifierRecognisesLegacyAntiBotContent() {
        let kind = FailureDiagnoser.toolFailureKind(
            toolName: "web_search",
            content: "web_search error: DuckDuckGo blocked this request (anti-bot rate limit).",
            isError: true
        )
        #expect(kind == .webSearchRateLimited)
    }

    @Test("A Brave quota error is NOT relabelled as a DuckDuckGo throttle")
    func classifierLeavesKeyedBackendQuotaAlone() {
        // Telling a Brave user to switch to Brave would be the same dead end
        // in a different costume.
        let kind = FailureDiagnoser.toolFailureKind(
            toolName: "web_search",
            content: "web_search error: Brave free-tier limit hit (HTTP 429). Wait a few minutes or upgrade your plan.",
            isError: true
        )
        #expect(kind == .webSearchUnavailable)
    }

    @Test("An offline web_search failure still classifies as offline")
    func classifierKeepsOfflineBranch() {
        let kind = FailureDiagnoser.toolFailureKind(
            toolName: "web_search",
            content: "web_search error: The Internet connection appears to be offline.",
            isError: true
        )
        #expect(kind == .webSearchOffline)
    }

    // MARK: - Settings copy

    @Test("The DuckDuckGo picker caption no longer claims it works out of the box")
    func duckDuckGoSubtitleIsHonest() {
        let subtitle = WebSearchProvider.duckduckgo.subtitle
        #expect(!subtitle.lowercased().contains("works out of the box"))
        #expect(subtitle.contains("No key required"))
        #expect(subtitle.lowercased().contains("throttled"))
    }

    @Test("The keyed backends keep their signup-cost captions")
    func keyedSubtitlesUnchanged() {
        #expect(WebSearchProvider.brave.subtitle.contains("Brave Search API key"))
        #expect(WebSearchProvider.tavily.subtitle.contains("Tavily API key"))
    }

    @Test("Key links use each provider's live dashboard host")
    func keyDashboardLinksAreCurrent() {
        #expect(
            WebSearchProvider.brave.keyDashboardURL?.absoluteString
                == "https://api-dashboard.search.brave.com/app/keys"
        )
        #expect(
            WebSearchProvider.tavily.keyDashboardURL?.absoluteString
                == "https://app.tavily.com/home"
        )
        #expect(WebSearchProvider.duckduckgo.keyDashboardURL == nil)
    }
}
