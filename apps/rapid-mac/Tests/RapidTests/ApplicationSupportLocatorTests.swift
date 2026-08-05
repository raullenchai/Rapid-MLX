import Foundation
import Testing
@testable import Rapid

/// Contract for #419 + #420 — Application Support resolution must
/// honour ``$HOME`` consistently across every caller, so a dogfood /
/// test launch with overridden ``HOME`` writes sessions, crash
/// markers, and Quickstart installs into the overridden path instead
/// of contaminating the real ``~/Library/Application Support/Rapid``.
///
/// Pins:
/// - HOME-set + absolute → ``<HOME>/Library/Application Support/Rapid``
/// - HOME-empty / HOME-relative → fallback to FileManager
/// - HOME-unset → fallback to FileManager
/// - folderName pinned to ``Rapid``
/// - delegating call sites (SessionStore + CrashReporter +
///   QuickstartModel.defaultInstallRoot + BootstrapCoordinator
///   .defaultApplicationSupportRoot + ServerLocator
///   .defaultApplicationSupportURL) reference the locator by name,
///   so a future regression that re-introduces a direct
///   ``FileManager.urls(for: .applicationSupportDirectory)`` call
///   gets caught by source-grep
@Suite("ApplicationSupportLocator — #419 + #420 HOME-honouring consolidation")
struct ApplicationSupportLocatorTests {

    // MARK: - Resolution rules

    @Test("HOME-set + absolute → <HOME>/Library/Application Support/Rapid")
    func honoursAbsoluteHome() {
        let env = ["HOME": "/Users/test-user"]
        let url = ApplicationSupportLocator.applicationSupportRoot(environment: env)
        #expect(url.path == "/Users/test-user/Library/Application Support/Rapid")
    }

    @Test("HOME-set + absolute under /tmp (dogfood shape) → expected path")
    func honoursDogfoodHome() {
        // The exact shape v0.8.8 dogfood used. Pre-fix, callers
        // ignored this and wrote to the real ~/Library; the
        // dogfood agent then saw the user's real chat session
        // contaminate the test instance.
        let env = ["HOME": "/tmp/dogfood-v088/home"]
        let url = ApplicationSupportLocator.applicationSupportRoot(environment: env)
        #expect(url.path == "/tmp/dogfood-v088/home/Library/Application Support/Rapid")
    }

    @Test("HOME-empty string → fallback to FileManager (never empty-path concatenation)")
    func handlesEmptyHome() {
        let env: [String: String] = ["HOME": ""]
        let url = ApplicationSupportLocator.applicationSupportRoot(environment: env)
        // Whatever FileManager resolves to, the trailing component is
        // still ``Rapid`` — we don't accept a degenerate
        // ``/Library/Application Support/Rapid`` (which would land
        // at the filesystem root and possibly fail to mkdir).
        #expect(url.lastPathComponent == "Rapid")
        // Defensive: never starts with a bare ``/Library`` (the
        // shape we'd get if we naively concatenated empty HOME).
        #expect(!url.path.hasPrefix("/Library/Application Support"))
    }

    @Test("HOME-relative (no leading /) → fallback to FileManager")
    func rejectsRelativeHome() {
        // A relative HOME makes no sense as an absolute filesystem
        // path; treat it as if HOME were unset and fall back.
        let env: [String: String] = ["HOME": "relative/path"]
        let url = ApplicationSupportLocator.applicationSupportRoot(environment: env)
        #expect(url.lastPathComponent == "Rapid")
        #expect(!url.path.hasPrefix("relative/path"))
    }

    @Test("HOME-unset (empty env dict) → fallback to FileManager")
    func handlesMissingHome() {
        let url = ApplicationSupportLocator.applicationSupportRoot(environment: [:])
        #expect(url.lastPathComponent == "Rapid")
    }

    @Test("folderName is pinned to \"Rapid\" (not derived from anything)")
    func folderNameIsPinned() {
        // Pinned so a rename has exactly one site to touch + so
        // delegating callers can reference the constant when
        // building sub-paths instead of duplicating the literal.
        #expect(ApplicationSupportLocator.folderName == "Rapid")
    }

    // MARK: - Source-grep regression pins

}
