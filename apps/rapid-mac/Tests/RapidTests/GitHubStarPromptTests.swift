import Foundation
import Testing
@testable import Rapid

@Suite("GitHub star onboarding prompt")
struct GitHubStarPromptTests {
    private static func source(_ name: String) throws -> String {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try String(
            contentsOf: root.appendingPathComponent("Sources/Rapid/UI/\(name)"),
            encoding: .utf8
        )
    }

    @Test("Repository link targets the canonical Rapid-MLX project")
    func canonicalRepositoryURL() {
        #expect(GitHubCommunity.repositoryURL.absoluteString ==
                "https://github.com/raullenchai/Rapid-MLX")
    }

    @Test("Star entry stays in Chat and onboarding never covers the composer")
    func productionWiring() throws {
        let chat = try Self.source("ChatView.swift")
        let content = try Self.source("ContentView.swift")

        #expect(chat.contains("GitHubStarButton()"))
        #expect(!content.contains("OnboardingCompletePrompt"))
        #expect(!content.contains("showOnboardingCompletePrompt"))
    }
}
