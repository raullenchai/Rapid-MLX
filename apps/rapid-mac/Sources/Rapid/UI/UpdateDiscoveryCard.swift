import AppKit
import SwiftUI

/// Lightweight discovery surface for an update already resolved by
/// ``UpdateChecker``. Phase one deliberately stops at the hand-off boundary:
/// Sparkle's standard user driver remains the sole owner of download,
/// verification, installation, and all UI after the primary action is pressed.
struct UpdateDiscoveryCard: View {
    let release: UpdateChecker.Release
    let sparkleEnabled: Bool
    let sparkleCanCheck: Bool
    let releaseURL: URL?
    let onUpdate: () -> Void
    let onDismiss: () -> Void

    @discardableResult
    static func openManualDownload(
        _ url: URL,
        using opener: (URL) -> Bool,
        onOpened: () -> Void
    ) -> Bool {
        guard opener(url) else { return false }
        onOpened()
        return true
    }

    private let restartNotice = "Restarting to install stops active generations and local API sessions."

    var body: some View {
        HStack(spacing: RapidTheme.Space.sm) {
            Image(systemName: "arrow.down.circle")
                .foregroundStyle(Color.accentColor)
                .accessibilityHidden(true)
            Text("Rapid-MLX v\(release.version) is available")
                .font(RapidFont.bodyEmphasis)
                .foregroundStyle(RapidTheme.textPrimary)
                .lineLimit(1)
                .layoutPriority(1)
            Text(restartNotice)
                .font(RapidFont.caption)
                .foregroundStyle(RapidTheme.textSecondary)
                .lineLimit(1)
                .frame(minWidth: 0, maxWidth: .infinity, alignment: .leading)
                .layoutPriority(-1)
                .help(restartNotice)
            if let releaseURL {
                Link("Release notes", destination: releaseURL)
                    .font(RapidFont.caption)
                    .fixedSize()
                    .accessibilityIdentifier("UpdateCard.ReleaseNotes")
            }
            if sparkleEnabled {
                Button(action: onUpdate) {
                    HStack(spacing: RapidTheme.Space.xs) {
                        if !sparkleCanCheck {
                            ProgressView().controlSize(.mini)
                        }
                        Text(sparkleCanCheck ? "Update" : "Update in progress…")
                    }
                }
                .buttonStyle(.rapidPrimaryCompact)
                .controlSize(.small)
                .fixedSize()
                .disabled(!sparkleCanCheck)
                .accessibilityIdentifier("UpdateCard.Update")
            } else if let releaseURL {
                Button {
                    Self.openManualDownload(
                        releaseURL,
                        using: NSWorkspace.shared.open,
                        onOpened: onDismiss
                    )
                } label: {
                    Text("Download")
                }
                .buttonStyle(.rapidPrimaryCompact)
                .controlSize(.small)
                .fixedSize()
                .accessibilityIdentifier("UpdateCard.ManualDownload")
            }
            Button(action: onDismiss) {
                Image(systemName: "xmark")
                    .font(.system(size: 11, weight: .semibold))
                    .frame(width: 28, height: 28)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .foregroundStyle(RapidTheme.textSecondary)
            .fixedSize()
            .help("Dismiss this update")
            .accessibilityLabel("Dismiss update v\(release.version)")
            .accessibilityIdentifier("UpdateCard.Dismiss")
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 6)
        .background(.bar)
        .overlay(Divider(), alignment: .bottom)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Rapid-MLX update v\(release.version) available")
        .accessibilityIdentifier("UpdateCard")
        .transition(.opacity)
    }
}
