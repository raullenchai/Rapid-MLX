import Foundation
import Testing
@testable import Rapid

@Suite("GitHub star surfaces")
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

    @Test("The value-moment card follows the approved compact hierarchy")
    func valueMomentVisualContract() throws {
        let card = try Self.source("GitHubStarPrompt.swift")
        let content = try Self.source("ContentView.swift")
        let snapshot = try Self.source("../DevSnapshot.swift")

        #expect(card.contains("Enjoying Rapid-MLX?"))
        #expect(card.contains("Rapid-MLX is open source."))
        #expect(card.contains(".frame(width: 360)"))
        #expect(card.contains(".frame(width: 84, height: RapidTheme.ControlHeight.medium)"))
        #expect(card.contains(".padding(14)"))
        #expect(content.contains(".padding(.trailing, 16)"))
        #expect(content.contains(".padding(.bottom, 40)"))
        #expect(snapshot.contains("github-star-value-moment.png"))
    }

    @Test("The card is nonmodal, focus-neutral, and fully addressable")
    func interactionContract() throws {
        let card = try Self.source("GitHubStarPrompt.swift")
        let coordinator = try Self.source("GitHubStarPromptCoordinator.swift")

        for identifier in ["Card", "Open", "Later", "Close"] {
            #expect(card.contains("GitHub.Star.ValueMoment.\(identifier)"))
        }
        #expect(!card.contains("@FocusState"))
        #expect(!card.contains(".keyboardShortcut"))
        #expect(!card.contains(".isModal"))
        #expect(coordinator.contains("NSEvent.addLocalMonitorForEvents(matching: [.keyDown])"))
        #expect(!coordinator.contains("URLSession"), "eligibility must not probe GitHub or the network")
    }
}
