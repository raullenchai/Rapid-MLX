import Foundation
import Testing

@testable import Rapid

/// Pins the voice channel: one Discord invite, reachable from the menu bar.
///
/// Two failure modes this exists to catch, both silent at build time:
///
/// 1. **A second invite.** A literal typed at a new call site creates a
///    second community that nobody monitors. The constant is the only
///    copy on the Swift side, and it must equal the invite the
///    repository already publishes in `README.md` — which this suite
///    reads from the working tree rather than trusting from memory.
/// 2. **A menu item that quietly stops opening it.** A SwiftUI
///    `.commands` block is not reachable from a unit test, so the wiring
///    is verified by reading `RapidApp.swift` the way
///    `RepositoryLinkTargetsTests` reads the whole source tree: the Help
///    menu must still carry the item, and it must open the shared
///    constant rather than a literal of its own.
///
/// Deliberately no network call: this proves the app opens the invite the
/// project publishes, not that Discord is up.
@Suite("Community feedback link")
struct FeedbackLinkTests {
    private static var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // RapidTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // rapid-mac
            .deletingLastPathComponent()  // apps
            .deletingLastPathComponent()  // <root>
    }

    private static func sourceText(_ relativePath: String) throws -> String {
        try String(
            contentsOf: repositoryRoot.appendingPathComponent(relativePath),
            encoding: .utf8
        )
    }

    @Test("the invite constant parses into a URL")
    func inviteIsAValidURL() {
        #expect(CommunityLinks.discordInvite.absoluteString
            == CommunityLinks.discordInviteURLString)
        #expect(CommunityLinks.discordInvite.host == "discord.gg")
        #expect(CommunityLinks.discordInvite.scheme == "https")
    }

    @Test("the invite is the one the repository publishes")
    func inviteMatchesTheReadme() throws {
        let readme = try Self.sourceText("README.md")
        #expect(
            readme.contains(CommunityLinks.discordInviteURLString),
            """
            CommunityLinks.discordInviteURLString is \
            \(CommunityLinks.discordInviteURLString), which appears nowhere in \
            README.md. Either the invite was rotated in one place only, or the \
            app now points at a second community. Make them the same link.
            """
        )
    }

    @Test("the Help menu still opens it, through the shared constant")
    func helpMenuOpensTheInvite() throws {
        let app = try Self.sourceText("apps/rapid-mac/Sources/Rapid/RapidApp.swift")
        #expect(app.contains("CommandGroup(after: .help)"),
                "the Help menu group that carries the feedback item is gone")
        #expect(app.contains("Tell Us What You Want…"),
                "the Help menu no longer offers the feedback item")
        #expect(
            app.contains("NSWorkspace.shared.open(CommunityLinks.discordInvite)"),
            """
            the Help menu item no longer opens CommunityLinks.discordInvite. \
            A literal URL typed here is exactly how a second invite gets \
            shipped; open the shared constant.
            """
        )
    }

    /// Guard on the guard: if the `#filePath` arithmetic or the source
    /// layout changes, the reads above would throw or scan nothing.
    @Test("the files this suite reads are actually there")
    func scannedFilesExist() {
        let fm = FileManager.default
        for path in ["README.md", "apps/rapid-mac/Sources/Rapid/RapidApp.swift"] {
            #expect(
                fm.fileExists(
                    atPath: Self.repositoryRoot.appendingPathComponent(path).path
                ),
                "expected \(path) under \(Self.repositoryRoot.path)"
            )
        }
    }
}
