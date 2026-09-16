import Foundation
import Testing
@testable import Rapid

/// Rapid-MLX Desktop 0.14.2 — clicking "See what's new" on the
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
/// `OpenURLAction`'s completion handler is invoked by `NSWorkspace`
/// from LaunchServices' open-queue, NOT from the main thread. Both
/// completion bodies touched a `@MainActor @Observable` model
/// (``InstallTracker``, ``GitHubStarPromptCoordinator``) inline, so
/// the Swift runtime's isolation check tripped and trapped. The
/// package builds in `.swiftLanguageMode(.v5)` (see `Package.swift`),
/// so the compiler is silent about it — which is exactly why this
/// needs a source-level tripwire rather than a compiler upgrade note.
///
/// The invariant pinned here is deliberately stricter than "don't
/// touch main-actor state off the main thread": **every `openURL(…)`
/// trailing-closure body must open with a main-actor hop**, before
/// reading anything. Hopping first makes the whole body isolated by
/// construction, so a later edit that inserts a statement at the top
/// of the completion cannot silently reintroduce the crash. A rule
/// about the *first* statement is also the only version of this a
/// text gate can check honestly.
///
/// Shape follows ``ToolUseCapabilitySourceGuardTests`` and
/// ``CapabilityChipRenderGateSourceGuardTests`` — canonicalise the
/// source (comments, whitespace and string literals gone) then scan
/// for the required / forbidden shapes, and fail closed whenever the
/// scanner cannot prove what it is looking at.
@Suite("openURL completion handlers hop to the main actor before touching state")
struct OpenURLCompletionMainActorTests {

    /// Package root (`apps/rapid-mac`), derived from ``#filePath`` so the
    /// test runs from any cwd (swift test, Xcode, CI).
    private static var sourceRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // RapidTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // apps/rapid-mac (package root)
    }

    /// Canonical spellings of "get onto the main actor".
    ///
    /// `MainActor.assumeIsolated` is deliberately absent: it asserts
    /// rather than hops, so on the open-queue it traps with the very
    /// signature this guard exists to prevent.
    private static let mainActorHops = [
        "Task{@MainActorin",
        "DispatchQueue.main.async{",
    ]

    // MARK: - Tree-wide gate

    /// Every `openURL(…) { … }` call site in `Sources/Rapid` must open
    /// its completion with a main-actor hop.
    @Test("Every openURL trailing closure in Sources opens with a main-actor hop")
    func openURLCompletionsHopFirst() throws {
        let sites = try Self.openURLCompletionSites()

        // Fail closed if the scanner found nothing: a silently-empty
        // sweep is the classic way a source guard rots into a no-op
        // (a rename of `openURL`, a canonicaliser change, a moved
        // source tree). Two sites are known to exist today.
        #expect(
            sites.count >= 2,
            "Scanner found \(sites.count) openURL trailing-closure call site(s) in Sources/Rapid, expected at least the two known ones (WhatsNewBanner, GitHubStarPrompt). Either the sweep broke or the call sites moved — this guard is not measuring anything until that is resolved."
        )

        for site in sites {
            #expect(
                site.hopsFirst,
                """
                \(site.file) (openURL call #\(site.occurrence)) — the body must \
                START with a main-actor hop (`Task { @MainActor in … }` or \
                `DispatchQueue.main.async { … }`). OpenURLAction invokes it on \
                LaunchServices' open-queue, so any @MainActor state touched \
                inline traps at runtime (EXC_BREAKPOINT, Rapid-MLX Desktop \
                0.14.2 "See what's new" crash). Swift 5 language mode will not \
                flag it for you. Closure body: \(site.raw)
                """
            )
        }
    }

    // MARK: - Named sites

    /// The two sites the 0.14.2 crash report and its twin came from,
    /// pinned by name so a regression reads as "WhatsNewBanner lost
    /// its hop" rather than "one of N call sites failed".
    @Test("Known openURL completion sites keep their hop")
    func knownSitesHopBeforeTouchingState() throws {
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
                #expect(
                    site.hopsFirst,
                    "\(path) — \(mutation) must run inside a main-actor hop that OPENS the openURL completion body, not inline on LaunchServices' open-queue. Closure body: \(site.raw)"
                )
                let acceptedGuard = "guardacceptedelse{return}"
                let strippedMutation = Self.strip(mutation)
                let actorBody = site.mainActorBody ?? ""
                let guardRange = actorBody.range(of: acceptedGuard)
                let mutationRange = actorBody.range(of: strippedMutation)
                let acceptanceGuardsMutation = guardRange.flatMap { guardRange in
                    mutationRange.map { guardRange.lowerBound < $0.lowerBound }
                } ?? false
                #expect(
                    acceptanceGuardsMutation,
                    "\(path) — the accepted result must guard \(mutation). A declined openURL request must leave prompt/banner state unchanged. Closure body: \(site.raw)"
                )
            }
        }
    }

    /// `MainActor.assumeIsolated` inside an `openURL` completion would
    /// look like a fix and crash identically — it asserts the current
    /// executor instead of switching to it. Called out separately so
    /// the failure explains itself.
    @Test("No openURL completion uses MainActor.assumeIsolated")
    func openURLCompletionsDoNotAssumeIsolation() throws {
        for site in try Self.openURLCompletionSites() {
            #expect(
                !site.raw.contains("MainActor.assumeIsolated"),
                "\(site.file) (openURL call #\(site.occurrence)) — `MainActor.assumeIsolated` ASSERTS that the caller is already on the main actor; the openURL completion never is. It traps with the same EXC_BREAKPOINT as the unfixed code. Use `Task { @MainActor in … }`."
            )
        }
    }

    @Test("Scanner accepts legal call spacing and fails closed on ambiguous arguments")
    func scannerCoverage() {
        let spaced = SourceGuardSupport.canonicalSource(
            "openURL (url) { accepted in Task { @MainActor in guard accepted else { return } } }",
            literals: .erase
        )
        let spacedSites = Self.openURLCompletionSites(inCanonical: spaced, file: "fixture.swift")
        #expect(spacedSites.count == 1)
        #expect(spacedSites.first?.hopsFirst == true)

        let ambiguous = SourceGuardSupport.canonicalSource(
            "openURL(url / divisor) { accepted in Task { @MainActor in } }",
            literals: .erase
        )
        let ambiguousSites = Self.openURLCompletionSites(
            inCanonical: ambiguous,
            file: "fixture.swift"
        )
        #expect(ambiguousSites.count == 1)
        #expect(ambiguousSites.first?.hopsFirst == false)
        #expect(ambiguousSites.first?.raw.hasPrefix("<unscannable:") == true)

        let falsePrefix = Self.openURLCompletionSites(
            inCanonical: "openURL(url){spinTask{@MainActorin}}",
            file: "fixture.swift"
        )
        #expect(falsePrefix.first?.hopsFirst == false)

        let falseGlobalActor = Self.openURLCompletionSites(
            inCanonical: "openURL(url){acceptedinTask{@MainActorFooin}}",
            file: "fixture.swift"
        )
        #expect(falseGlobalActor.first?.hopsFirst == false)

        let escapedMutation = Self.openURLCompletionSites(
            inCanonical:
                "openURL(url){acceptedinTask{@MainActoringuardacceptedelse{return}}prompt.repositoryOpened()}",
            file: "fixture.swift"
        )
        #expect(escapedMutation.first?.hopsFirst == false)

        let parenthesizedCompletion = Self.openURLCompletionSites(
            inCanonical:
                "openURL(url,completion:{acceptedinTask{@MainActorin}})",
            file: "fixture.swift"
        )
        #expect(parenthesizedCompletion.first?.hopsFirst == false)
    }

    // MARK: - Scanner

    private struct CallSite {
        /// Source file name.
        var file: String
        /// 1-based index of this `openURL(` occurrence within the file.
        /// Whitespace is gone by the time the scan runs, so a line
        /// number is not recoverable; this is enough to tell two sites
        /// in one file apart, and `raw` carries the text besides.
        var occurrence: Int
        /// The canonical closure body, parameter clause included.
        var raw: String
        /// Whether the first statement of the body is a main-actor hop.
        var hopsFirst: Bool
        /// The complete opening hop block. A non-nil value also proves
        /// that the outer completion contains no statements after it.
        var mainActorBody: String?
    }

    /// Every `openURL(…) { … }` site under `Sources/Rapid`.
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

    /// Walk canonical source and return one entry per `openURL(…)`
    /// call that carries a trailing closure. Calls without one
    /// (`openURL(url)`) are not completion sites and are skipped.
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

            // `openURL(` — the `(` is the last character of the match.
            let openParen = canonical.index(before: call.upperBound)
            guard let closeParen = endOfParenGroup(canonical, openAt: openParen) else {
                let unscannable =
                    "<unscannable: openURL argument list contains syntax the source guard refuses to parse>"
                sites.append(CallSite(
                    file: file, occurrence: occurrence, raw: unscannable,
                    hopsFirst: false, mainActorBody: nil
                ))
                continue
            }
            let afterCall = canonical.index(after: closeParen)
            guard afterCall < canonical.endIndex, canonical[afterCall] == "{" else {
                let arguments = canonical[canonical.index(after: openParen)..<closeParen]
                if arguments.contains("completion:") || arguments.contains("{") {
                    let unsupported =
                        "<unsupported: use openURL's trailing-completion form so the actor guard can prove the hop>"
                    sites.append(CallSite(
                        file: file, occurrence: occurrence, raw: unsupported,
                        hopsFirst: false, mainActorBody: nil
                    ))
                }
                continue  // no trailing closure: nothing to schedule wrongly
            }
            // `balancedBlock` fails closed on unresolved `/` (regex vs
            // division). A nil here means "cannot prove what this
            // closure contains", so report it as a body that trips the
            // hop assertion rather than skipping it quietly.
            guard let block = SourceGuardSupport.balancedBlock(
                in: canonical, openingBraceAt: afterCall
            ) else {
                let unscannable =
                    "<unscannable: closure contains slash syntax the source guard refuses to parse>"
                sites.append(CallSite(
                    file: file, occurrence: occurrence, raw: unscannable,
                    hopsFirst: false, mainActorBody: nil
                ))
                continue
            }

            let inner = String(block.dropFirst().dropLast())
            let actorBody = openingMainActorHopBody(inner)
            sites.append(CallSite(
                file: file,
                occurrence: occurrence,
                raw: inner,
                hopsFirst: actorBody != nil,
                mainActorBody: actorBody
            ))
        }
        return sites
    }

    /// Whether the closure body's first statement is one of
    /// ``mainActorHops``.
    ///
    /// The hop must be the first thing after the optional `accepted in`
    /// parameter clause. This gate deliberately requires that spelling:
    /// after whitespace canonicalisation, accepting any identifier ending
    /// in `in` would mistake `spinTask` for `accepted in Task`.
    /// Anything else in front of the hop — a `guard`, a call, an
    /// assignment, a `;` — is a statement executing off the main
    /// actor, so the site fails. Only the FIRST occurrence of a hop
    /// counts: a hop that follows a state touch is exactly the bug.
    ///
    /// The hop block must also consume the rest of the outer completion;
    /// otherwise a later state mutation could escape back off actor.
    private static func openingMainActorHopBody(_ inner: String) -> String? {
        // OpenURLAction's completion has one Bool parameter. Requiring the
        // explicit spelling `accepted in` keeps this source guard token-safe:
        // whitespace-free canonical source cannot otherwise distinguish a
        // parameter clause from an identifier ending in "in" (`spinTask`).
        let acceptedClause = "acceptedin"
        let executable = inner.hasPrefix(acceptedClause)
            ? String(inner.dropFirst(acceptedClause.count))
            : inner

        for hop in mainActorHops {
            guard executable.hasPrefix(hop),
                  let openingBrace = executable.firstIndex(of: "{"),
                  let block = SourceGuardSupport.balancedBlock(
                      in: executable, openingBraceAt: openingBrace
                  )
            else { continue }

            let afterBlock = executable.index(openingBrace, offsetBy: block.count)
            guard afterBlock == executable.endIndex else { continue }
            return block
        }
        return nil
    }

    /// Index of the `)` closing the group opened at `start`.
    ///
    /// Fails closed on `/` for the same reason
    /// ``SourceGuardSupport/balancedBlock(in:openingBraceAt:)`` does:
    /// a bare regex literal and a division share that syntax, and
    /// guessing lets a paren inside `/[)]/ ` end the scan early.
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

    /// Canonicalise a needle the same way the haystack was, so
    /// `prompt.repositoryOpened()` written with spaces still matches.
    private static func strip(_ needle: String) -> String {
        SourceGuardSupport.canonicalSource(needle, literals: .erase)
    }
}
