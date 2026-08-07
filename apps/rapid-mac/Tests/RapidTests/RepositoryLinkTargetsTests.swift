import Foundation
import Testing

/// Static integrity check for the in-repo GitHub links the app hard-codes.
///
/// Two shipped links opened nothing. "Open-source credits" pointed at
/// `blob/main/THIRD_PARTY.md` — the repository ROOT — while the file has
/// always lived one directory down under `apps/rapid-mac/`, so it 404'd. And
/// the "Privacy policy" link (Settings, and again in the About window) pointed
/// at `rapidmlx.com/privacy`, a page that has never been published, while a
/// 259-line `apps/rapid-mac/PRIVACY.md` sat in the repo. Nothing in the build
/// could notice either one: a `URL(string:)` literal is valid regardless of
/// whether the far end resolves.
///
/// This suite closes that gap **without touching the network**. It reads the
/// working tree only: every `https://github.com/raullenchai/Rapid-MLX/blob|tree|raw/<ref>/<path>`
/// URL appearing anywhere under `Sources/Rapid/**` is mapped back to a
/// repo-relative path, which must exist in this checkout. A file moved,
/// renamed, or deleted without updating the link fails here, at `swift test`,
/// instead of in a user's browser — and so does a new link that guesses the
/// wrong directory, which is the exact mistake that shipped.
///
/// Scope and its limits, stated plainly so this is not mistaken for more than
/// it is: this proves the path exists **on the branch under test**. It cannot
/// prove the file has been pushed to `main` on GitHub, and it cannot say
/// anything about URLs pointing OUTSIDE the repository (`rapidmlx.com/...`,
/// model cards, …) — those would require a network call, which this suite
/// deliberately never makes. The remedy for an off-repo dead link is to point
/// it at something in-repo, which is what both privacy links now do.
@Suite("In-repo GitHub link targets")
struct RepositoryLinkTargetsTests {
    /// Repository root, derived from this file's own compile-time location:
    /// `<root>/apps/rapid-mac/Tests/RapidTests/<this file>`.
    private static var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // RapidTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // rapid-mac
            .deletingLastPathComponent()  // apps
            .deletingLastPathComponent()  // <root>
    }

    private static var sourcesRoot: URL {
        repositoryRoot
            .appendingPathComponent("apps/rapid-mac/Sources/Rapid", isDirectory: true)
    }

    /// `blob` / `tree` / `raw` all address a path inside the repo; only
    /// bare `https://github.com/raullenchai/Rapid-MLX` (the project landing
    /// page, used by the About panel) addresses no file and is skipped.
    ///
    /// Computed rather than a stored `static let`: `NSRegularExpression` is not
    /// `Sendable` on every SDK this package builds against, and a stored static
    /// of a non-`Sendable` type is a hard error under the Swift 6 language mode.
    private static func linkPattern() throws -> NSRegularExpression {
        try NSRegularExpression(
            pattern: #"https://github\.com/raullenchai/Rapid-MLX/(?:blob|tree|raw)/[^/\s"'()]+/([^\s"'()<>]+)"#
        )
    }

    private struct RepoLink {
        let path: String
        let sourceFile: String
    }

    /// Every in-repo GitHub path referenced from a Swift source file, with the
    /// file that references it (so a failure names the call site to fix).
    private static func repoLinks() throws -> [RepoLink] {
        let fm = FileManager.default
        let pattern = try linkPattern()
        var links: [RepoLink] = []

        guard let walker = fm.enumerator(
            at: sourcesRoot,
            includingPropertiesForKeys: [.isRegularFileKey]
        ) else {
            return links
        }

        for case let url as URL in walker where url.pathExtension == "swift" {
            let text = try String(contentsOf: url, encoding: .utf8)
            let range = NSRange(text.startIndex..<text.endIndex, in: text)
            for match in pattern.matches(in: text, range: range) {
                guard let captured = Range(match.range(at: 1), in: text) else { continue }
                // Strip a `#fragment` / `?query` — neither is part of the path
                // GitHub resolves against the tree.
                var path = String(text[captured])
                if let cut = path.firstIndex(where: { $0 == "#" || $0 == "?" }) {
                    path = String(path[path.startIndex..<cut])
                }
                path = path.removingPercentEncoding ?? path
                links.append(
                    RepoLink(path: path, sourceFile: url.lastPathComponent)
                )
            }
        }
        return links
    }

    /// Guard on the guard: if `#filePath` arithmetic or the source layout ever
    /// changes, the scan below would find nothing and pass vacuously. Fail
    /// loudly instead.
    @Test("the source tree the scan reads is actually there")
    func sourceTreeIsReachable() {
        var isDirectory: ObjCBool = false
        let exists = FileManager.default.fileExists(
            atPath: Self.sourcesRoot.path,
            isDirectory: &isDirectory
        )
        #expect(exists && isDirectory.boolValue,
                "expected a Sources/Rapid directory at \(Self.sourcesRoot.path)")
    }

    @Test("every hard-coded in-repo link points at a file that exists")
    func everyInRepoLinkResolves() throws {
        let links = try Self.repoLinks()

        // The scan must find something — an over-tightened regex that matches
        // nothing must not read as "all links are fine".
        #expect(!links.isEmpty, "found no in-repo GitHub links to verify")

        for link in links {
            let target = Self.repositoryRoot.appendingPathComponent(link.path)
            #expect(
                FileManager.default.fileExists(atPath: target.path),
                """
                \(link.sourceFile) links to \(link.path), which does not exist \
                in this checkout. The app would open a GitHub 404. Point the \
                link at the real path — most likely the file lives under \
                apps/rapid-mac/ rather than the repository root, which is \
                exactly the mistake this test exists to catch. Do not delete \
                the link if the app owes the user the document it promises.
                """
            )
        }
    }

    /// Every user-facing link, pinned to the file that renders it AND the
    /// document it must reach.
    ///
    /// Per-surface, not an aggregate set: the privacy policy is linked from two
    /// different places, and a set-membership check stays green when one of
    /// them regresses to an off-repo 404 while the other stays correct — which
    /// is exactly the state this PR found the app in. Pinning `(source file,
    /// path)` makes each surface fail on its own.
    ///
    /// Paths, not just "some link exists": the shipped bug was a link naming
    /// the WRONG directory for a file that did exist.
    @Test(
        "each surface links to the document it promises",
        arguments: [
            // Settings → Privacy renders all three of these.
            ("SettingsView.swift", "apps/rapid-mac/PRIVACY.md"),
            ("SettingsView.swift", "LICENSE"),
            ("SettingsView.swift", "apps/rapid-mac/THIRD_PARTY.md"),
            // The About window's "Privacy" link — the second dead privacy link.
            ("AboutPanel.swift", "apps/rapid-mac/PRIVACY.md"),
        ]
    )
    func eachSurfaceLinksToItsDocument(sourceFile: String, path: String) throws {
        let links = try Self.repoLinks()
        #expect(
            links.contains { $0.sourceFile == sourceFile && $0.path == path },
            "\(sourceFile) no longer links to \(path)"
        )
        let target = Self.repositoryRoot.appendingPathComponent(path)
        #expect(
            FileManager.default.fileExists(atPath: target.path),
            "\(path) is missing from the checkout"
        )
    }

    /// The repository root holds a `LICENSE` but **not** `THIRD_PARTY.md` or
    /// `PRIVACY.md` — those are app documents under `apps/rapid-mac/`. Asserted
    /// so nobody "fixes" a future root-path link by creating a second, diverging
    /// copy at the root instead of correcting the link.
    @Test("the app documents live under apps/rapid-mac, not at the root")
    func appDocumentsAreNotDuplicatedAtTheRoot() {
        let fm = FileManager.default
        for name in ["THIRD_PARTY.md", "PRIVACY.md"] {
            #expect(
                !fm.fileExists(
                    atPath: Self.repositoryRoot.appendingPathComponent(name).path
                ),
                """
                \(name) exists at the repository root as well as under \
                apps/rapid-mac/. Two attribution/privacy documents drift apart; \
                keep the app's copy and point links at it.
                """
            )
        }
    }
}
