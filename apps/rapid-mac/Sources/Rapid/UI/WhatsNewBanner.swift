import SwiftUI

/// Sticky banner shown once after the app updates itself, above the model
/// picker, when ``InstallTracker.upgradedFrom`` is set.
///
/// Sparkle's default path is deliberately silent: it downloads in the
/// background, installs on quit, and relaunches into the new build with no
/// prompt at any point. That is the right behaviour for the update itself —
/// nobody wants a modal on launch — but it also means the app never tells
/// the user that anything changed. Crossing 0.13.1 → 0.14.1 in the 0.14.1
/// dogfood added Computer Use, Share Compute, PDF analysis, model unload and
/// six image models, and the window came back byte-identical apart from the
/// number in the version pill.
///
/// So: one line naming both versions, one link to that release's notes, one
/// Dismiss. No modal, no feature tour, and nothing that has to be clicked
/// before the user can type.
///
/// It cannot collide with ``FailedReplaceBanner``: that one requires the
/// version to have stayed PUT across the launch, this one requires it to
/// have moved forward.
struct WhatsNewBanner: View {
    @Environment(InstallTracker.self) private var installTracker
    @Environment(\.openURL) private var openURL

    /// The release page for a desktop version. Tags are published as
    /// ``rapid-mac-v<version>`` (see RELEASING.md), and GitHub renders the
    /// release body — the same notes the appcast carries.
    static func releaseNotesURL(for version: String) -> URL? {
        let trimmed = version.trimmingCharacters(in: .whitespacesAndNewlines)
        let tag = trimmed.hasPrefix("v") ? String(trimmed.dropFirst()) : trimmed
        // Only a plain dotted-numeric version becomes a tag URL — no link
        // beats a 404, and only released versions have a release page.
        // ``strictNumericParts`` is not the gate here: it drops a
        // pre-release suffix, so "0.14.1-dev" would pass it and then point
        // at a tag that was never published.
        guard tag.range(of: #"^[0-9]+(\.[0-9]+)*$"#, options: .regularExpression) != nil
        else { return nil }
        return URL(string: "https://github.com/raullenchai/Rapid-MLX/releases/tag/rapid-mac-v\(tag)")
    }

    var body: some View {
        // `FailedReplaceBanner` wins when both fire. They are no longer
        // mutually exclusive by construction: the upgrade notice is sticky
        // until acknowledged, so a launch that upgraded and a LATER launch
        // whose Finder Replace failed can both be true at once — and stacking
        // "Updated to v0.14.2" on top of "your update didn't install" would
        // contradict itself. The stale-bundle warning is the one the user has
        // to act on, so it stands alone.
        if let from = installTracker.upgradedFrom, !installTracker.failedReplaceDetected {
            let current = installTracker.currentVersion
            let headline = "Updated to v\(current)"
            let detail = "You were on v\(from). The release notes list what changed."
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: "sparkles")
                    .foregroundStyle(RapidTheme.brand)
                    .font(.system(size: 14, weight: .semibold))
                    .padding(.top, 1)
                VStack(alignment: .leading, spacing: 4) {
                    // Headline + detail combine into ONE VoiceOver element so
                    // the update is announced as a sentence; the buttons stay
                    // siblings so each remains focusable. Same shape as
                    // ``FailedReplaceBanner`` — see the note there.
                    VStack(alignment: .leading, spacing: 4) {
                        Text(headline)
                            .scaledSystemFont(13, weight: .semibold)
                        Text(detail)
                            .scaledSystemFont(12)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                            .textSelection(.enabled)
                    }
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel("\(headline). \(detail)")
                    .accessibilityAddTraits(.isHeader)
                    HStack(spacing: 8) {
                        if let url = Self.releaseNotesURL(for: current) {
                            Button("See what's new") {
                                // Dismiss only once the browser actually took
                                // the URL. `openURL` can be declined (no
                                // handler, a policy block), and dismissing
                                // regardless would retire the only route to the
                                // notes for a click that opened nothing — the
                                // banner cannot come back, because
                                // `lastSeenVersion` already rolled forward.
                                // Same shape as `GitHubStarPrompt`.
                                openURL(url) { accepted in
                                    guard accepted else { return }
                                    installTracker.dismissUpgradeNotice()
                                }
                            }
                            .buttonStyle(.borderedProminent)
                            .controlSize(.small)
                            .accessibilityIdentifier("WhatsNew.OpenNotes")
                        }
                        Button("Dismiss") {
                            installTracker.dismissUpgradeNotice()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .accessibilityIdentifier("WhatsNew.Dismiss")
                    }
                    .padding(.top, 2)
                }
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(RapidTheme.brand.opacity(0.10))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .strokeBorder(RapidTheme.brand.opacity(0.30), lineWidth: 1)
            )
            .padding(.horizontal, 12)
            .padding(.top, 8)
            // `.contain`, not the default: an identifier on a plain container
            // ABSORBS its descendants' identifiers, so both buttons came back
            // from an AX dump as `WhatsNewBanner` and neither could be pressed
            // by name. ``CampaignBanner`` is the pattern being matched here.
            .accessibilityElement(children: .contain)
            .accessibilityIdentifier("WhatsNewBanner")
        }
    }
}
