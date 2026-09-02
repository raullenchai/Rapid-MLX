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

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
            HStack(alignment: .top, spacing: RapidTheme.Space.sm) {
                VStack(alignment: .leading, spacing: RapidTheme.Space.xs) {
                    Text("Update Available")
                        .font(RapidFont.bodyEmphasis)
                        .foregroundStyle(RapidTheme.textPrimary)
                    Text("Rapid-MLX v\(release.version) is ready.")
                        .font(RapidFont.body)
                        .foregroundStyle(RapidTheme.textPrimary)
                }

                Spacer(minLength: RapidTheme.Space.sm)

                Button(action: onDismiss) {
                    Image(systemName: "xmark")
                        .font(.system(size: 11, weight: .semibold))
                        .frame(width: 28, height: 28)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .foregroundStyle(RapidTheme.textSecondary)
                .frame(minWidth: 44, minHeight: 44)
                .contentShape(Rectangle())
                .help("Dismiss this update")
                .accessibilityLabel("Dismiss update v\(release.version)")
                .accessibilityIdentifier("UpdateCard.Dismiss")
                .padding(.top, -8)
                .padding(.trailing, -8)
            }

            Text("Download in the background. Restarting to install will stop active generations and local API sessions.")
                .font(RapidFont.caption)
                .foregroundStyle(RapidTheme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)

            if let releaseURL {
                Link("View release notes", destination: releaseURL)
                    .font(RapidFont.caption)
                    .accessibilityIdentifier("UpdateCard.ReleaseNotes")
            }

            if sparkleEnabled {
                Button(action: onUpdate) {
                    HStack(spacing: RapidTheme.Space.sm) {
                        if !sparkleCanCheck {
                            ProgressView()
                                .controlSize(.small)
                        }
                        Text(sparkleCanCheck ? "Update Rapid-MLX" : "Update in progress…")
                            .frame(maxWidth: .infinity)
                    }
                }
                .buttonStyle(.rapidPrimary)
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
                    Label("Download from release page", systemImage: "arrow.up.right.square")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.rapidPrimary)
                .accessibilityIdentifier("UpdateCard.ManualDownload")
            }
        }
        .padding(RapidTheme.Space.lg)
        .frame(width: 360, alignment: .leading)
        .background(RapidTheme.card)
        .clipShape(RoundedRectangle(cornerRadius: RapidTheme.cardRadius, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: RapidTheme.cardRadius, style: .continuous)
                .strokeBorder(RapidTheme.hairline, lineWidth: 1)
        }
        .shadow(color: Color.black.opacity(0.14), radius: 18, y: 8)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Rapid-MLX update v\(release.version) available")
        .accessibilityIdentifier("UpdateCard")
        .transition(reduceMotion ? .opacity : .move(edge: .trailing).combined(with: .opacity))
    }
}
