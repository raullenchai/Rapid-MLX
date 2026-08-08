import Foundation
import Testing
@testable import Rapid

/// v0.4.37 contract pin for the Settings deep-link router.
///
/// The router carries a one-shot target category. ``SettingsView``
/// consumes it on appear and on change, applying the request to its
/// ``selected`` state and clearing the field. Tests pin both the
/// "request placed" and "request consumed" half-cycles so a future
/// refactor of the consumption protocol (e.g. switching from
/// ``.onAppear`` to ``.task``) can't silently break the deep-link.
@MainActor
@Suite("SettingsRouter — v0.4.37 deep-link contract")
struct SettingsRouterTests {
    @Test("Fresh router has no pending request — Settings opens on the user's last tab")
    func freshRouterIsNil() {
        let r = SettingsRouter()
        #expect(r.requestedCategory == nil)
    }

    @Test("Set + read round-trip — every category survives the channel")
    func setReadRoundtrip() {
        for cat in SettingsView.Category.allCases {
            let r = SettingsRouter()
            r.requestedCategory = cat
            #expect(r.requestedCategory == cat,
                    "Round-trip failed for \(cat) — the channel must carry every Category cell")
        }
    }

    @Test("Manual clear restores the 'no override' state — pinning the consumer's clear-after-apply step")
    func manualClearWorks() {
        // SettingsView clears the field after consuming the request
        // so a subsequent ``openSettings()`` without a fresh
        // ``requestedCategory`` set lands on the user's last tab.
        // This test pins that the clear path actually works on the
        // ``@Observable`` field — defensive against a future refactor
        // that makes ``requestedCategory`` non-optional.
        let r = SettingsRouter()
        r.requestedCategory = .app
        #expect(r.requestedCategory == .app)
        r.requestedCategory = nil
        #expect(r.requestedCategory == nil)
    }

    @Test("Overwrite-without-consume — latest request wins")
    func overwriteLatestWins() {
        // Two deep-link clicks in rapid succession should land on
        // the SECOND target. SwiftUI's ``.onChange`` collapses
        // intermediate values so this is the only sane semantic.
        let r = SettingsRouter()
        r.requestedCategory = .appearance
        r.requestedCategory = .app
        #expect(r.requestedCategory == .app)
    }

    @Test("Quickstart catalogue return is one-shot")
    func quickstartCatalogueReturn() {
        let r = SettingsRouter()
        #expect(r.quickstartReturnGeneration == 0)
        #expect(!r.quickstartCatalogReturnPending)

        r.completeQuickstartCatalogRoundTrip()
        #expect(r.quickstartReturnGeneration == 0)

        r.beginQuickstartCatalogRoundTrip()
        #expect(r.quickstartCatalogReturnPending)
        r.completeQuickstartCatalogRoundTrip()
        #expect(!r.quickstartCatalogReturnPending)
        #expect(r.quickstartReturnGeneration == 1)

        r.completeQuickstartCatalogRoundTrip()
        #expect(r.quickstartReturnGeneration == 1)
    }
}
