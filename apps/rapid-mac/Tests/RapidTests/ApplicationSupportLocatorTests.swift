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


    /// Detect any new direct call to
    /// ``FileManager.urls(for: .applicationSupportDirectory)`` /
    /// ``FileManager.url(for: .applicationSupportDirectory)`` outside
    /// the locator itself. Such a call leaks ``$HOME`` overrides and
    /// re-introduces #419 / #420. Auditing here at test time
    /// because Swift has no idiomatic way to forbid a specific API at
    /// the type system level.
    @Test("No direct FileManager.applicationSupportDirectory calls outside the locator")
    func noDirectApplicationSupportDirectoryCalls() throws {
        let sourcesRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources", isDirectory: true)
        let fm = FileManager.default
        let enumerator = fm.enumerator(at: sourcesRoot, includingPropertiesForKeys: nil)
        var offenders: [String] = []
        // Files allowed to mention the API:
        //   1. ApplicationSupportLocator.swift — the one place we
        //      legitimately consult FileManager's resolution as the
        //      defensive fallback.
        //   2. The pre-existing BootstrapCoordinator.swift comment
        //      block referencing the historical path lives nearby
        //      — exact path checked by name so a rename surfaces
        //      explicitly.
        let allowedFiles: Set<String> = [
            "ApplicationSupportLocator.swift",
        ]
        while let url = enumerator?.nextObject() as? URL {
            guard url.pathExtension == "swift" else { continue }
            if allowedFiles.contains(url.lastPathComponent) { continue }
            let content = try String(contentsOf: url, encoding: .utf8)
            // The exact two API shapes that bypass HOME:
            //   FileManager.default.urls(for: .applicationSupportDirectory
            //   FileManager.default.url(for: .applicationSupportDirectory
            // Strip doc-comments + line comments first so a doc
            // string that NAMES the API for explanation (like the
            // one in QuickstartModel.defaultInstallRoot pointing
            // at the historical bug) doesn't false-positive.
            // Strip line-comments AND any line that opens a block
            // comment (codex r1 NIT — a future `/* FileManager
            // .default.urls(...) */` block would otherwise false-
            // positive). The block-comment heuristic is conservative:
            // any line containing `/*` is skipped, which would also
            // skip a line that legitimately calls the API inside an
            // inline `/* … */` — fine for our purposes (we want to
            // err on the side of NOT false-positiving, given the
            // test runs on every commit).
            let nonCommentLines = content.split(separator: "\n", omittingEmptySubsequences: false)
                .filter { line in
                    let trimmed = line.drop { $0 == " " || $0 == "\t" }
                    return !trimmed.hasPrefix("///")
                        && !trimmed.hasPrefix("//")
                        && !trimmed.hasPrefix("*")
                        && !trimmed.hasPrefix("/*")
                        && !line.contains("/*")
                }
                .joined(separator: "\n")
            // Codex r1 MINOR: catch extra whitespace before `(` (a
            // future ``FileManager.default.urls   ( for:`` shape
            // would otherwise sneak past).
            if nonCommentLines.range(of: #"urls\s*\(\s*for:\s*\.applicationSupportDirectory"#,
                                     options: .regularExpression) != nil
                || nonCommentLines.range(of: #"url\s*\(\s*for:\s*\.applicationSupportDirectory"#,
                                         options: .regularExpression) != nil {
                offenders.append(url.lastPathComponent)
            }
        }
        // Consolidation tail closed in PR #423 — every direct
        // FileManager.applicationSupportDirectory call site now
        // routes through ApplicationSupportLocator. A new entry
        // in this set is a real regression, not a tracked debt.
        // Consolidation fully complete — every direct
        // FileManager.applicationSupportDirectory call site routes
        // through ApplicationSupportLocator. A new file appearing
        // here is a real regression.
        let knownLeftovers: Set<String> = []
        let unexpected = Set(offenders).subtracting(knownLeftovers)
        #expect(unexpected.isEmpty,
                "New file(s) bypass ApplicationSupportLocator: \(unexpected). Route through ApplicationSupportLocator.applicationSupportRoot() instead — see #419/#420 for why.")
    }

}
