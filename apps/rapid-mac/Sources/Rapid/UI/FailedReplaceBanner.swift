import SwiftUI

/// Sticky banner shown above the model picker when
/// ``InstallTracker.failedReplaceDetected == true``. Surfaces the
/// "Finder Replace into /Applications silently failed because Rapid-MLX
/// Desktop was running" footgun so the user understands why an upgrade
/// attempt didn't take. See ``InstallTracker`` for the detection
/// signal and ``rapid-desktop#251`` for the live-incident context.
///
/// Two CTAs:
///   * **Check for updates** — hands off to Sparkle, which downloads,
///     verifies, and installs on quit, so the user can finish the upgrade
///     without re-mounting the DMG by hand. Disabled on unsigned builds
///     where Sparkle carries no public key.
///   * **Dismiss** — clears the banner for this session. Persistence
///     was already advanced on construction, so the next launch only
///     re-fires if ANOTHER failed Finder Replace happens between now
///     and then.
struct FailedReplaceBanner: View {
    @Environment(InstallTracker.self) private var installTracker
    @Environment(SparkleUpdateController.self) private var sparkleUpdater

    var body: some View {
        if installTracker.failedReplaceDetected {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                    .font(.system(size: 14, weight: .semibold))
                    .padding(.top, 1)
                VStack(alignment: .leading, spacing: 4) {
                    // Codex r1 nit (a11y), r2 BLOCKING follow-up:
                    // combine ONLY the headline + body into a single
                    // VoiceOver element so the screen reader announces
                    // the safety-net message as one unit (instead of
                    // two disconnected leaf labels). The CTA HStack
                    // lives as a SIBLING in the same VStack so the two
                    // Buttons stay independently focusable. An earlier
                    // shape applied `.combine` to the outer container,
                    // which would have absorbed the buttons into the
                    // banner element.
                    VStack(alignment: .leading, spacing: 4) {
                        Text("An update attempt didn't take")
                            .scaledSystemFont(13, weight: .semibold)
                        Text(
                            "Rapid-MLX is still running v\(installTracker.currentVersion). " +
                            "macOS Finder can't replace files inside a running .app, so a drag-to-/Applications " +
                            "Replace silently leaves the old build in place. Check for updates " +
                            "(the updater handles the quit + swap automatically), or quit and re-run the installer."
                        )
                        .scaledSystemFont(12)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                    }
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel(
                        "An update attempt didn't take. " +
                        "Rapid-MLX is still running v\(installTracker.currentVersion). " +
                        "macOS Finder can't replace files inside a running .app. " +
                        "Check for updates, or quit and re-run the installer."
                    )
                    .accessibilityAddTraits(.isHeader)
                    HStack(spacing: 8) {
                        Button("Check for updates") {
                            sparkleUpdater.checkForUpdates()
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                        .disabled(!sparkleUpdater.isEnabled)
                        .accessibilityIdentifier("FailedReplace.OpenUpdate")
                        Button("Dismiss") {
                            installTracker.dismiss()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .accessibilityIdentifier("FailedReplace.Dismiss")
                    }
                    .padding(.top, 2)
                }
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(Color.orange.opacity(0.10))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .strokeBorder(Color.orange.opacity(0.30), lineWidth: 1)
            )
            .padding(.horizontal, 12)
            .padding(.top, 8)
            .accessibilityIdentifier("FailedReplaceBanner")
        }
    }
}
