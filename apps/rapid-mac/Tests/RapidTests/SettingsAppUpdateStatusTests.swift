import Foundation
import Testing
@testable import Rapid

/// Pin contract for the Settings → App panel's update-status
/// resolver (`#191`). The naive "if availableUpdate == nil → up to
/// date" shape that landed in the first cut lied during two real
/// states: (1) first poll still in flight at app launch, (2) the
/// release worker just errored. Both produce ``availableUpdate ==
/// nil`` even though we have not established that the local
/// version is current.
///
/// Codex r1 P2 (Settings → App update gating) caught this. The
/// status resolver below carries the truth table; this suite
/// exercises every branch.
@MainActor
@Suite("SettingsView.resolveAppUpdateStatus — Settings → App update status truth table (#191)")
struct SettingsAppUpdateStatusTests {

    private func release(version: String) -> UpdateChecker.Release {
        UpdateChecker.Release(
            schemaVersion: 1,
            version: version,
            tagName: "v\(version)",
            htmlURL: "https://github.com/machinefi/rapid-desktop/releases/tag/v\(version)",
            notes: "",
            publishedAt: "2026-06-15T00:00:00Z",
            dmgURL: nil
        )
    }

    @Test("availableUpdate non-nil → .available — always wins, even mid-check")
    func availableUpdateAlwaysWins() {
        let s = SettingsView.resolveAppUpdateStatus(
            currentVersion: "0.6.6",
            availableUpdate: release(version: "0.6.7"),
            latest: release(version: "0.6.7"),
            checking: true,                   // a re-check is in flight
            lastCheckedAt: Date(),
            lastError: nil
        )
        #expect(s == .available(version: "0.6.7"))
    }

    @Test("First poll in flight (no completed check yet) → .checking, NOT .upToDate")
    func firstPollInFlightIsChecking() {
        // The bug: pre-fix the resolver returned .upToDate(currentVersion)
        // here because availableUpdate was nil. The app had not yet
        // talked to the release worker, so claiming "you're on the
        // latest" is a fiction. Pin .checking instead.
        let s = SettingsView.resolveAppUpdateStatus(
            currentVersion: "0.6.6",
            availableUpdate: nil,
            latest: nil,
            checking: true,
            lastCheckedAt: nil,                // never completed
            lastError: nil
        )
        #expect(s == .checking)
    }

    @Test("No check has completed AND none is in flight → .unknown — surfaces lastError when present")
    func noCheckIsUnknown() {
        let s = SettingsView.resolveAppUpdateStatus(
            currentVersion: "0.6.6",
            availableUpdate: nil,
            latest: nil,
            checking: false,
            lastCheckedAt: nil,
            lastError: "transport error: offline"
        )
        #expect(s == .unknown(reason: "transport error: offline"))
    }

    @Test("Successful check resolved a release AND no upgrade pending → .upToDate")
    func successfulCheckIsUpToDate() {
        let s = SettingsView.resolveAppUpdateStatus(
            currentVersion: "0.6.7",
            availableUpdate: nil,                  // strictly newer? no
            latest: release(version: "0.6.7"),
            checking: false,
            lastCheckedAt: Date(),
            lastError: nil
        )
        #expect(s == .upToDate(version: "0.6.7"))
    }

    @Test("Check completed but latest is nil — worker errored → .unknown, NOT .upToDate")
    func completedButLatestNilIsUnknown() {
        // The second bug shape: lastCheckedAt is populated (a check
        // attempt ran) but ``latest`` is still nil because the worker
        // returned 5xx / a decoder error / payload was rejected by
        // the validation gate. Pre-fix this fell through to the
        // green check. Pin .unknown with the surfaced error message.
        let s = SettingsView.resolveAppUpdateStatus(
            currentVersion: "0.6.6",
            availableUpdate: nil,
            latest: nil,
            checking: false,
            lastCheckedAt: Date(),
            lastError: "update server returned HTTP 503"
        )
        #expect(s == .unknown(reason: "update server returned HTTP 503"))
    }

    @Test("Recheck in flight on top of a previously-resolved up-to-date state stays .upToDate")
    func rerunOverUpToDateStaysUpToDate() {
        // Subtle case: a successful check landed earlier (lastCheckedAt
        // set, latest set), then the user mashed Recheck. ``checking``
        // is now true but the previous result is still authoritative —
        // mirror what System Settings → Software Update does (it keeps
        // the green check while the spinner runs).
        let s = SettingsView.resolveAppUpdateStatus(
            currentVersion: "0.6.7",
            availableUpdate: nil,
            latest: release(version: "0.6.7"),
            checking: true,
            lastCheckedAt: Date(),
            lastError: nil
        )
        #expect(s == .upToDate(version: "0.6.7"))
    }
}
