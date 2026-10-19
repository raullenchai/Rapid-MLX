import Foundation
import Testing
@testable import Rapid

@Suite("BrowseRedirect same-site approval")
struct BrowseRedirectSameSiteTests {
    @Test(arguments: [
        ("https://nasa.gov/x", "https://www.nasa.gov/x", true),
        ("https://www.nasa.gov", "https://nasa.gov", true),
        ("http://nasa.gov", "https://www.nasa.gov", true),
        ("http://nasa.gov", "https://nasa.gov", true),
        ("http://nasa.gov:80", "https://nasa.gov:443", true),
        ("https://NASA.GOV.", "https://WWW.NASA.GOV.", true),
        ("https://WWW.NASA.GOV.", "https://nasa.gov", true),
        ("https://nasa.gov:443", "https://nasa.gov", true),
        ("https://nasa.gov", "https://www.nasa.gov:443", true),
        ("http://nasa.gov", "http://www.nasa.gov:80", true),
        ("https://nasa.gov:8443", "https://www.nasa.gov:8443", true),
        ("https://1.2.3.4", "https://1.2.3.4:443/x", true),
        ("https://[2606:4700::1111]", "https://[2606:4700::1111]/x", true),
        ("https://nasa.gov", "http://nasa.gov", false),
        ("https://nasa.gov", "http://www.nasa.gov", false),
        ("https://example.com", "https://evil.com", false),
        ("https://nasa.gov", "https://news.nasa.gov", false),
        ("https://nasa.gov", "https://nasa.gov.evil.com", false),
        ("https://nasa.gov", "https://evilnasa.gov", false),
        ("https://nasa.gov", "https://www2.nasa.gov", false),
        ("https://www.www.nasa.gov", "https://nasa.gov", false),
        ("https://nasa.gov", "https://www.www.nasa.gov", false),
        ("https://nasa.gov", "https://nasa.org", false),
        ("https://nasa.gov", "https://nasa.gov:8443", false),
        ("http://nasa.gov:8080", "https://nasa.gov", false),
        ("http://nasa.gov", "https://nasa.gov:8443", false),
        ("http://nasa.gov", "http://nasa.gov:8080", false),
        ("http://nasa.gov", "https://nasa.gov:80", false),
        ("https://www.com", "https://com", false),
        ("https://com", "https://www.com", false),
        ("https://nasa.gov", "https://com", false),
        ("https://www", "https://www", true),
        ("https://com", "https://com", true),
        ("https://1.2.3.4", "https://www.1.2.3.4", false),
        ("https://www.1.2.3.4", "https://1.2.3.4", false),
        ("https://1.2.3.4", "https://1.2.3.5", false),
        ("https://[2606:4700::1111]", "https://[2606:4700::1001]", false),
        ("https://nasa.gov..", "https://nasa.gov", false),
        ("/relative", "https://nasa.gov", false),
        ("https://nasa.gov", "/relative", false),
    ])
    func sameSite(source: String, destination: String, expected: Bool) throws {
        #expect(BrowseTool.isSameSiteRedirect(
            from: try #require(URL(string: source)), to: try #require(URL(string: destination))
        ) == expected)
    }

    @Test func promptDecisionUsesEveryApprovedSite() throws {
        let nasa = try #require(URL(string: "https://nasa.gov"))
        let example = try #require(URL(string: "https://example.com"))
        let sibling = try #require(URL(string: "https://www.example.com/page"))
        #expect(BrowseTool.redirectNeedsApproval(nasa, from: nasa, approved: []))
        #expect(!BrowseTool.redirectNeedsApproval(nasa, from: nasa, approved: [nasa]))
        #expect(BrowseTool.redirectNeedsApproval(example, from: nasa, approved: [nasa]))
        #expect(BrowseTool.redirectNeedsApproval(sibling, from: nasa, approved: [nasa]))
        #expect(!BrowseTool.redirectNeedsApproval(sibling, from: nasa, approved: [nasa, example]))
        #expect(!BrowseTool.redirectNeedsApproval(nasa, from: nasa, approved: [nasa, example]))
        #expect(BrowseTool.redirectNeedsApproval(
            try #require(URL(string: "http://www.example.com")), from: example, approved: [nasa, example]
        ))
        // https -> http always asks, even back to the approved http start URL.
        let httpStart = try #require(URL(string: "http://nasa.gov/a"))
        let upgraded = try #require(URL(string: "https://www.nasa.gov/a"))
        #expect(!BrowseTool.redirectNeedsApproval(upgraded, from: httpStart, approved: [httpStart]))
        #expect(BrowseTool.redirectNeedsApproval(httpStart, from: upgraded, approved: [httpStart]))
        #expect(!BrowseTool.redirectNeedsApproval(httpStart, from: httpStart, approved: [httpStart]))
    }

    @MainActor
    @Test func approvalPublishesRedirectContext() async throws {
        let suite = TestDefaultsScope.mintSuiteName(prefix: "rapid-browse-redirect-")
        defer { TestDefaultsScope.cleanup(suiteNames: [suite]) }
        let store = BrowseApprovalStore(defaults: try #require(UserDefaults(suiteName: suite)))
        async let redirectDecision = store.requestApproval(
            url: "https://example.com/page", host: "example.com", redirectedFrom: "nasa.gov"
        )
        #expect(await store.waitUntilPendingRequest())
        let redirected = try #require(store.pendingRequest)
        #expect(redirected.redirectedFrom == "nasa.gov")
        store.answer(.allowOnce)
        #expect(await redirectDecision == .allowOnce)

        async let initialDecision = store.requestApproval(url: "https://example.com/page", host: "example.com")
        #expect(await store.waitUntilPendingRequest())
        let initial = try #require(store.pendingRequest)
        #expect(initial.redirectedFrom == nil)
        #expect(initial != redirected)
        store.answer(.deny)
        #expect(await initialDecision == .deny)
    }

    @Test func loopApprovesOnlyTheCrossSiteHop() async throws {
        let start = try #require(URL(string: "https://nasa.gov/start"))
        let redirects = [
            "https://nasa.gov/start": "https://www.nasa.gov/page",
            "https://www.nasa.gov/page": "https://example.com/page",
            "https://example.com/page": "https://www.example.com/page",
            "https://www.example.com/page": "https://nasa.gov/end",
        ]
        var validated: [URL] = []
        var prompted: [URL] = []
        let result = try await BrowseTool.fetchFollowingRedirects(
            startURL: start,
            resolve: { url in
                validated.append(url)
                return [try #require(ParsedIP("1.2.3.4"))]
            },
            fetch: { url, _, _ in
                let location = redirects[url.absoluteString]
                return (Data(), try #require(HTTPURLResponse(
                    url: url, statusCode: location == nil ? 200 : 301,
                    httpVersion: nil, headerFields: location.map { ["Location": $0] }
                )))
            },
            approveRedirect: { url, fromHost in
                prompted.append(url)
                #expect(fromHost == "www.nasa.gov")
                return .allowOnce
            }
        )
        #expect(prompted.map(\.absoluteString) == ["https://example.com/page"])
        #expect(validated.count == 5)
        #expect(validated.last == result.finalURL)
        #expect(result.finalURL.absoluteString == "https://nasa.gov/end")
    }

    @Test func sameSiteRedirectStillRequiresSSRFValidation() async throws {
        let start = try #require(URL(string: "https://nasa.gov"))
        let blocked = try #require(URL(string: "https://www.nasa.gov"))
        var validated: [URL] = []
        do {
            _ = try await BrowseTool.fetchFollowingRedirects(
                startURL: start,
                resolve: { url in
                    validated.append(url)
                    if url == blocked {
                        throw NSError(domain: "BrowseRedirectTests", code: 1, userInfo: [
                            NSLocalizedDescriptionKey: "blocked redirect address"
                        ])
                    }
                    return [try #require(ParsedIP("1.2.3.4"))]
                },
                fetch: { url, _, _ in
                    #expect(url == start)
                    return (Data(), try #require(HTTPURLResponse(
                        url: url, statusCode: 301, httpVersion: nil,
                        headerFields: ["Location": blocked.absoluteString]
                    )))
                },
                approveRedirect: { _, _ in
                    Issue.record("Same-site redirect must not prompt")
                    return .deny
                }
            )
            Issue.record("A failed SSRF check must stop the fetch")
        } catch {
            #expect(error.localizedDescription == "blocked redirect address")
        }
        #expect(validated == [start, blocked])
    }

    @Test(arguments: [BrowseApprovalStore.Decision.deny, .unavailable])
    func loopRejectsUnapprovedRedirect(decision: BrowseApprovalStore.Decision) async throws {
        let start = try #require(URL(string: "/relative"))
        do {
            _ = try await BrowseTool.fetchFollowingRedirects(
                startURL: start,
                resolve: { _ in [try #require(ParsedIP("1.2.3.4"))] },
                fetch: { _, _, _ in
                    (Data(), try #require(HTTPURLResponse(
                        url: URL(string: "https://nasa.gov")!, statusCode: 302,
                        httpVersion: nil, headerFields: ["Location": "https://example.com"]
                    )))
                },
                approveRedirect: { _, fromHost in
                    #expect(fromHost == "/relative")
                    return decision
                }
            )
            Issue.record("Unapproved redirect must stop the fetch")
        } catch {
            if decision == .deny {
                let declined = try #require(error as? BrowseTool.ApprovalDeclined)
                #expect(declined.message == "the user did not approve the redirect to example.com")
            } else {
                #expect(!(error is BrowseTool.ApprovalDeclined))
                #expect(error.localizedDescription
                    == BrowseTool.approvalUnavailableMessage(host: "example.com"))
            }
        }
    }

    @MainActor
    @Test(arguments: ["https://example.com/page", "file:///page"])
    func runForwardsRedirectContext(destination: String) async throws {
        let suite = TestDefaultsScope.mintSuiteName(prefix: "rapid-browse-redirect-run-")
        defer { TestDefaultsScope.cleanup(suiteNames: [suite]) }
        let store = BrowseApprovalStore(defaults: try #require(UserDefaults(suiteName: suite)))
        let url = try #require(URL(string: destination))
        let task = Task {
            await BrowseTool.run(
                arguments: #"{"url":"https://nasa.gov/start","refresh":true}"#,
                approval: store,
                cache: BrowseContentCache(diskDirectory: nil),
                resolve: { _ in [try #require(ParsedIP("1.2.3.4"))] },
                fetch: { hop, _, _ in
                    let isInitial = hop.host == "nasa.gov"
                    return (Data("page".utf8), try #require(HTTPURLResponse(
                        url: hop, statusCode: isInitial ? 301 : 200, httpVersion: nil,
                        headerFields: isInitial ? ["Location": destination] : ["Content-Type": "text/plain"]
                    )))
                }
            )
        }
        #expect(await store.waitUntilPendingRequest())
        #expect(store.pendingRequest?.redirectedFrom == nil)
        store.answer(.allowOnce)
        #expect(await store.waitUntilPendingRequest())
        #expect(store.pendingRequest?.redirectedFrom == "nasa.gov")
        #expect(store.pendingRequest?.host == (url.host ?? destination))
        store.answer(.allowOnce)
        #expect(!(await task.value).isError)
    }

    @MainActor
    @Test func defaultFetchStillValidatesSSRF() async throws {
        let suite = TestDefaultsScope.mintSuiteName(prefix: "rapid-browse-redirect-ssrf-")
        defer { TestDefaultsScope.cleanup(suiteNames: [suite]) }
        let store = BrowseApprovalStore(defaults: try #require(UserDefaults(suiteName: suite)))
        store.mode = .autoApproveAll
        let result = await BrowseTool.run(
            arguments: #"{"url":"http://localhost/","refresh":true}"#,
            approval: store, cache: BrowseContentCache(diskDirectory: nil)
        )
        #expect(result.isError)
        do {
            _ = try await BrowseTool.fetchFollowingRedirects(
                startURL: try #require(URL(string: "http://localhost/")),
                approveRedirect: { @Sendable _, _ in
                    Issue.record("Blocked host must fail before any redirect")
                    return .deny
                }
            )
            Issue.record("The default resolver must enforce SSRF validation")
        } catch {
            let rejection = try #require(error as? BrowseSSRFGuard.Rejection)
            #expect(rejection.message.contains("localhost"))
        }
    }

    @MainActor
    @Test func defaultTransportRejectsMalformedURLBeforeConnecting() async throws {
        let url = try #require(URL(string: "/relative"))
        let suite = TestDefaultsScope.mintSuiteName(prefix: "rapid-browse-redirect-transport-")
        defer { TestDefaultsScope.cleanup(suiteNames: [suite]) }
        let store = BrowseApprovalStore(defaults: try #require(UserDefaults(suiteName: suite)))
        store.mode = .autoApproveAll
        let result = await BrowseTool.run(
            arguments: #"{"url":"https://nasa.gov:65536/start","refresh":true}"#,
            approval: store, cache: BrowseContentCache(diskDirectory: nil),
            resolve: { _ in [try #require(ParsedIP("1.2.3.4"))] }
        )
        #expect(result.isError)
        do {
            _ = try await BrowseTool.fetchFollowingRedirects(
                startURL: url,
                resolve: { @Sendable _ in [try #require(ParsedIP("1.2.3.4"))] },
                approveRedirect: { @Sendable _, _ in
                    Issue.record("Malformed URL must fail before any redirect")
                    return .deny
                }
            )
            Issue.record("Malformed URL must be rejected")
        } catch {
            #expect(error.localizedDescription == "URL has no scheme")
        }
    }
}
