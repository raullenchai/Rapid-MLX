import SwiftUI

enum GitHubCommunity {
    static let repositoryURL = URL(string: "https://github.com/raullenchai/Rapid-MLX")!
    static let feedbackBugReportURL = URL(
        string: "https://github.com/raullenchai/Rapid-MLX/issues/new?template=desktop_bug.yml"
    )!
    static let feedbackFeatureRequestURL = URL(
        string: "https://github.com/raullenchai/Rapid-MLX/issues/new?template=feature_request.yml"
    )!
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

/// A quiet, nonmodal value-moment card. It sits above the workspace instead
/// of reflowing it, takes no focus, and remains until the user chooses.
struct GitHubStarPromptCard: View {
    @Environment(GitHubStarPromptCoordinator.self) private var prompt
    @Environment(\.openURL) private var openURL

    var body: some View {
        HStack(alignment: .top, spacing: RapidTheme.Space.sm) {
            Image(systemName: "star.fill")
                .font(.system(size: 17, weight: .semibold))
                .foregroundStyle(RapidTheme.amberDeep)
                .frame(width: 22, height: 22)
                .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
                VStack(alignment: .leading, spacing: RapidTheme.Space.xs) {
                    Text("Enjoying Rapid-MLX?")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(RapidTheme.textPrimary)

                    Text("Rapid-MLX is open source. If it helped today, a GitHub star helps other developers find it.")
                        .font(.system(size: 14))
                        .foregroundStyle(RapidTheme.textSecondary)
                        .lineSpacing(2)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(.trailing, 28)

                HStack(spacing: RapidTheme.Space.sm) {
                    Button {
                        openURL(GitHubCommunity.repositoryURL) { accepted in
                            guard accepted else { return }
                            prompt.repositoryOpened()
                        }
                    } label: {
                        HStack(spacing: RapidTheme.Space.xs) {
                            Text("Open GitHub")
                            Image(systemName: "arrow.up.right")
                                .font(.system(size: 10, weight: .semibold))
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(RapidPrimaryButtonStyle(
                        expands: true,
                        height: RapidTheme.ControlHeight.medium,
                        font: .system(size: 14, weight: .medium)
                    ))
                    .frame(maxWidth: .infinity, minHeight: RapidTheme.ControlHeight.medium)
                    .accessibilityHint("Opens the Rapid-MLX repository in your browser")
                    .accessibilityIdentifier("GitHub.Star.ValueMoment.Open")

                    Button("Later") { prompt.deferPrompt() }
                        .buttonStyle(RapidSecondaryButtonStyle(
                            height: RapidTheme.ControlHeight.medium,
                            font: .system(size: 14, weight: .medium)
                        ))
                        .frame(width: 84, height: RapidTheme.ControlHeight.medium)
                        .accessibilityIdentifier("GitHub.Star.ValueMoment.Later")

                    Menu("Feedback") {
                        Button("Bug report") {
                            openURL(GitHubCommunity.feedbackBugReportURL)
                        }
                        .accessibilityIdentifier("GitHub.Star.ValueMoment.BugReport")
                        Button("Feature request") {
                            openURL(GitHubCommunity.feedbackFeatureRequestURL)
                        }
                        .accessibilityIdentifier("GitHub.Star.ValueMoment.FeatureRequest")
                    }
                    .buttonStyle(RapidSecondaryButtonStyle(
                        height: RapidTheme.ControlHeight.medium,
                        font: .system(size: 14, weight: .medium)
                    ))
                    .frame(width: 84, height: RapidTheme.ControlHeight.medium)
                    .accessibilityIdentifier("GitHub.Star.ValueMoment.Feedback")
                }
            }

        }
        .padding(14)
        .frame(width: 360)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(RapidTheme.card)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(RapidTheme.hairline.opacity(0.8), lineWidth: 1)
        )
        .overlay(alignment: .topTrailing) {
            Button { prompt.deferPrompt() } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 11, weight: .semibold))
                    .frame(width: 28, height: 28)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .foregroundStyle(RapidTheme.textSecondary)
            .help("Later")
            .accessibilityLabel("Show the GitHub invitation later")
            .accessibilityIdentifier("GitHub.Star.ValueMoment.Close")
            .padding(10)
        }
        .shadow(color: Color.black.opacity(0.08), radius: 10, y: 4)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("GitHub.Star.ValueMoment.Card")
    }
}
