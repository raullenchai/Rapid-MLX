import Foundation
import Testing
@testable import Rapid

/// Rapid-MLX Desktop 0.16.2 — clicking "See what's new" on the
/// post-upgrade banner crashed with `EXC_BREAKPOINT` on
/// `com.apple.launchservices.open-queue`:
///
/// ```
/// _dispatch_assert_queue_fail
/// _swift_task_checkIsolatedSwift
/// closure #1 in … WhatsNewBanner.body.getter
/// static OpenURLAction._defaultAction(env:)
/// _NSWorkspaceHandleLSOpenResult
/// ```
///
/// `OpenURLAction` invokes its completion from LaunchServices'
/// open-queue, not the main thread. The `Rapid` target builds in Swift 6
/// language mode, so a completion written as a closure literal inside a
/// SwiftUI `body` inherits the view's `@MainActor` isolation and the
/// compiler inserts an executor check at the closure's ENTRY. An earlier
/// fix opened the closure body with `Task { @MainActor in … }`; that hop
/// never runs, because the entry check traps first.
///
/// The invariant pinned here: every `openURL` completion in
/// `Sources/Rapid` is built by ``OpenURLCompletion``, which is formed in a
/// nonisolated context, and no `openURL` call takes a closure literal.
/// The runtime test proves the helper is callable off the main thread and
/// still delivers its action on the main actor, only on acceptance.
@Suite("openURL completions are safe to invoke off the main thread")
struct OpenURLCompletionMainActorTests {

    /// Package root (`apps/rapid-mac`), derived from ``#filePath`` so the
    /// test runs from any cwd (swift test, Xcode, CI).
    private static var sourceRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // RapidTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // apps/rapid-mac (package root)
    }

    /// The only accepted spelling of an `openURL` completion argument, in
    /// canonical (whitespace-free) form.
    private static let requiredCompletion = "completion:OpenURLCompletion.onMainActorIfAccepted{"

    // MARK: - Runtime

    @MainActor
    private final class Recorder {
        var calls: [Bool] = []
    }

    /// Builds the completion on the main actor, as a view `body` does, then
    /// invokes it the way LaunchServices does — from a background queue —
    /// and checks the action lands on the main thread, once, only for the
    /// accepted open. The entry trap itself only fires when an isolated
    /// closure crosses into the SDK's pre-concurrency API, which a unit test
    /// cannot drive without launching another app; the source guards below
    /// pin that no call site hands `openURL` such a closure.
    @MainActor
    @Test("Completion can be invoked off the main thread and acts on the main actor only when accepted")
    func completionIsSafeOffMainThread() async throws {
        let recorder = Recorder()
        let completion = OpenURLCompletion.onMainActorIfAccepted { [recorder] in
            recorder.calls.append(Thread.isMainThread)
        }

        let invokedOffMain = await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let offMain = !Thread.isMainThread
                completion(false)
                completion(true)
                continuation.resume(returning: offMain)
            }
        }
        #expect(invokedOffMain)

        let deadline = Date().addingTimeInterval(5)
        while recorder.calls.isEmpty, Date() < deadline {
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        // Let any (wrongly) scheduled work for the declined call land too.
        try await Task.sleep(nanoseconds: 50_000_000)

        let calls = recorder.calls
        #expect(calls == [true], "Expected exactly one action, on the main thread, for the accepted open; got \(calls)")
    }

    // MARK: - Tree-wide gate

    @Test("Every openURL completion in Sources is built by OpenURLCompletion")
    func openURLCompletionsUseTheHelper() throws {
        let sites = try Self.openURLCompletionSites()

        // Fail closed if the scanner found nothing: two completion sites
        // exist today (WhatsNewBanner, GitHubStarPrompt).
        #expect(
            sites.count >= 2,
            "Scanner found \(sites.count) openURL completion site(s) in Sources/Rapid, expected at least the two known ones (WhatsNewBanner, GitHubStarPrompt). Either the sweep broke or the call sites moved."
        )

        for site in sites {
            #expect(
                site.usesHelper,
                """
                \(site.file) (openURL call #\(site.occurrence)) — pass the \
                completion as `completion: OpenURLCompletion.onMainActorIfAccepted { … }`. \
                A closure literal in a view inherits @MainActor isolation and \
                traps on entry when LaunchServices calls it off the main thread \
                (EXC_BREAKPOINT, Rapid-MLX Desktop 0.16.2 "See what's new" \
                crash). Call: \(site.raw)
                """
            )
        }
    }

    // MARK: - Named sites

    /// The two sites that carry a completion, pinned by name so a
    /// regression reads as "WhatsNewBanner lost its safe completion".
    @Test("Known openURL completion sites route their mutation through the helper")
    func knownSitesUseTheHelper() throws {
        let pinned = [
            ("Sources/Rapid/UI/WhatsNewBanner.swift", "installTracker.dismissUpgradeNotice()"),
            ("Sources/Rapid/UI/GitHubStarPrompt.swift", "prompt.repositoryOpened()"),
        ]
        for (path, mutation) in pinned {
            let canonical = try Self.canonicalSource(at: path)
            let owning = Self.openURLCompletionSites(inCanonical: canonical, file: path)
                .filter { $0.raw.contains(Self.strip(mutation)) }

            #expect(
                owning.count == 1,
                "\(path) should contain exactly one openURL completion that calls \(mutation); found \(owning.count). If the call moved, move this pin with it."
            )
            for site in owning {
                #expect(site.usesHelper, "\(path) — \(mutation) must run inside OpenURLCompletion.onMainActorIfAccepted. Call: \(site.raw)")
            }
        }
    }

    @Test("Scanner rejects closure literals and fails closed on ambiguous arguments")
    func scannerCoverage() {
        func sites(_ source: String) -> [CallSite] {
            Self.openURLCompletionSites(
                inCanonical: SourceGuardSupport.canonicalSource(source, literals: .erase),
                file: "fixture.swift"
            )
        }

        let helper = sites("openURL (url, completion: OpenURLCompletion.onMainActorIfAccepted { [t] in t.done() })")
        #expect(helper.count == 1)
        #expect(helper.first?.usesHelper == true)

        // The 0.16.2 shape: trailing closure that hops first. Still traps.
        let trailing = sites("openURL(url) { accepted in Task { @MainActor in guard accepted else { return } } }")
        #expect(trailing.count == 1)
        #expect(trailing.first?.usesHelper == false)

        let labelled = sites("openURL(url, completion: { accepted in Task { @MainActor in } })")
        #expect(labelled.first?.usesHelper == false)

        let otherHelper = sites("openURL(url, completion: makeCompletion { })")
        #expect(otherHelper.first?.usesHelper == false)

        // Helper wrapped around trailing content still fails.
        let helperPlusTrailing = sites(
            "openURL(url, completion: OpenURLCompletion.onMainActorIfAccepted { }) { accepted in }"
        )
        #expect(helperPlusTrailing.first?.usesHelper == false)

        let ambiguous = sites("openURL(url / divisor) { accepted in }")
        #expect(ambiguous.count == 1)
        #expect(ambiguous.first?.usesHelper == false)
        #expect(ambiguous.first?.raw.hasPrefix("<unscannable:") == true)

        // A plain open without a completion is not a completion site.
        #expect(sites("openURL(url)").isEmpty)
        // An identifier that merely ends in openURL is not a call to it.
        #expect(sites("myopenURL(url) { }").isEmpty)
    }

    // MARK: - Scanner

    private struct CallSite {
        /// Source file name.
        var file: String
        /// 1-based index of this `openURL(` occurrence within the file.
        var occurrence: Int
        /// The canonical call text (argument list plus any trailing closure).
        var raw: String
        /// Whether the completion is built by ``OpenURLCompletion`` and the
        /// call takes no trailing closure.
        var usesHelper: Bool
    }

    /// Every `openURL` call with a completion under `Sources/Rapid`.
    private static func openURLCompletionSites() throws -> [CallSite] {
        let root = sourceRoot.appendingPathComponent("Sources/Rapid")
        let enumerator = try #require(
            FileManager.default.enumerator(
                at: root,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles]
            ),
            "Could not enumerate Sources/Rapid — directory missing?"
        )

        var sites: [CallSite] = []
        for case let url as URL in enumerator where url.pathExtension == "swift" {
            let source = try String(contentsOf: url, encoding: .utf8)
            sites += openURLCompletionSites(
                inCanonical: SourceGuardSupport.canonicalSource(source, literals: .erase),
                file: url.lastPathComponent
            )
        }
        return sites
    }

    private static func canonicalSource(at relativePath: String) throws -> String {
        let url = sourceRoot.appendingPathComponent(relativePath)
        let source = try String(contentsOf: url, encoding: .utf8)
        return SourceGuardSupport.canonicalSource(source, literals: .erase)
    }

    /// Walk canonical source and return one entry per `openURL(…)` call
    /// that carries a completion — a `completion:` argument, a closure in
    /// the argument list, or a trailing closure. `openURL(url)` alone is
    /// skipped. Anything the scanner cannot parse is reported as a failing
    /// site rather than skipped.
    private static func openURLCompletionSites(
        inCanonical canonical: String,
        file: String
    ) -> [CallSite] {
        var sites: [CallSite] = []
        var cursor = canonical.startIndex
        var occurrence = 0

        while let call = canonical.range(
            of: "openURL(", range: cursor..<canonical.endIndex
        ) {
            cursor = call.upperBound
            if call.lowerBound > canonical.startIndex {
                let previous = canonical[canonical.index(before: call.lowerBound)]
                if previous.isLetter || previous.isNumber || previous == "_" {
                    continue
                }
            }
            occurrence += 1

            let openParen = canonical.index(before: call.upperBound)
            guard let closeParen = endOfParenGroup(canonical, openAt: openParen) else {
                sites.append(CallSite(
                    file: file, occurrence: occurrence,
                    raw: "<unscannable: openURL argument list contains syntax the source guard refuses to parse>",
                    usesHelper: false
                ))
                continue
            }
            let arguments = String(canonical[canonical.index(after: openParen)..<closeParen])
            let afterCall = canonical.index(after: closeParen)
            let hasTrailingClosure = afterCall < canonical.endIndex && canonical[afterCall] == "{"
            let hasCompletionArgument = arguments.contains("completion:") || arguments.contains("{")
            guard hasTrailingClosure || hasCompletionArgument else { continue }

            var usesHelper = false
            if !hasTrailingClosure, let label = arguments.range(of: "completion:") {
                let completion = arguments[label.lowerBound...]
                // The helper call must be the entire completion argument:
                // one balanced `{ … }` block straight after the helper name
                // and nothing following it.
                if completion.hasPrefix(requiredCompletion) {
                    let brace = arguments.index(label.lowerBound, offsetBy: requiredCompletion.count - 1)
                    if let block = SourceGuardSupport.balancedBlock(in: arguments, openingBraceAt: brace) {
                        let end = arguments.index(brace, offsetBy: block.count)
                        usesHelper = end == arguments.endIndex
                    }
                }
            }
            var raw = arguments
            if hasTrailingClosure {
                raw += " trailing: "
                raw += SourceGuardSupport.balancedBlock(in: canonical, openingBraceAt: afterCall)
                    ?? "<unscannable closure>"
            }
            sites.append(CallSite(
                file: file,
                occurrence: occurrence,
                raw: raw,
                usesHelper: usesHelper
            ))
        }
        return sites
    }

    /// Index of the `)` closing the group opened at `start`. Fails closed on
    /// `/` because a bare regex literal and a division share that syntax.
    private static func endOfParenGroup(
        _ text: String,
        openAt start: String.Index
    ) -> String.Index? {
        guard start < text.endIndex, text[start] == "(" else { return nil }
        var depth = 0
        var index = start
        while index < text.endIndex {
            if text[index] == "/" { return nil }
            if text[index] == "(" { depth += 1 }
            if text[index] == ")" {
                depth -= 1
                if depth == 0 { return index }
            }
            index = text.index(after: index)
        }
        return nil
    }

    /// Canonicalise a needle the same way the haystack was.
    private static func strip(_ needle: String) -> String {
        SourceGuardSupport.canonicalSource(needle, literals: .erase)
    }
}
