import SwiftUI

enum GitHubCommunity {
    static let repositoryURL = URL(string: "https://github.com/raullenchai/Rapid-MLX")!
    /// Retained so re-onboarding can clear the preference written by older
    /// builds, even though completion no longer presents an overlay.
    static let didShowOnboardingPromptKey = "Rapid.didShowOnboardingGitHubStarPrompt"
}

struct GitHubStarButton: View {
    var onOpen: () -> Void = {}
    var accessibilityIdentifier = "GitHub.Star.EmptyState"

    @Environment(\.openURL) private var openURL

    var body: some View {
        Button {
            onOpen()
            openURL(GitHubCommunity.repositoryURL)
        } label: {
            Label("Star on GitHub", systemImage: "star")
                .font(.system(size: 12, weight: .medium))
                .padding(.horizontal, 14)
                .padding(.vertical, 7)
        }
        .buttonStyle(.plain)
        .foregroundStyle(RapidTheme.brandPrimaryDeep)
        .background(
            Capsule(style: .continuous)
                .fill(RapidTheme.brandPrimaryTint)
        )
        .overlay(
            Capsule(style: .continuous)
                .stroke(RapidTheme.brandPrimary.opacity(0.55), lineWidth: 1)
        )
        .contentShape(Capsule(style: .continuous))
        .accessibilityIdentifier(accessibilityIdentifier)
        .accessibilityHint("Opens the Rapid-MLX repository in your browser")
    }
}
