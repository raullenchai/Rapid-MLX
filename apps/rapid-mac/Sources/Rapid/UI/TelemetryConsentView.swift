import SwiftUI

/// First-run disclosure for the shared desktop + embedded-engine
/// telemetry pipeline. The full-window presenter blocks the workspace
/// until the user makes an explicit choice without creating an AppKit
/// modal sheet that would also block normal application termination.
struct TelemetryConsentView: View {
    let onDecision: (Bool) -> Void
    @FocusState private var primaryActionFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack(spacing: 12) {
                Image(systemName: "chart.bar.xaxis")
                    .font(.system(size: 26, weight: .medium))
                    .foregroundStyle(RapidTheme.brand)
                    .frame(width: 40, height: 40)
                    .background(RapidTheme.brandTint, in: RoundedRectangle(cornerRadius: 8))

                VStack(alignment: .leading, spacing: 3) {
                    Text("Help improve Rapid-MLX")
                        .font(.title2.weight(.semibold))
                    Text("Anonymous usage data is off until you choose to share it.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
            }

            VStack(alignment: .leading, spacing: 12) {
                disclosureRow(
                    icon: "desktopcomputer",
                    text: "App and engine versions, macOS version, chip family, and memory tier"
                )
                disclosureRow(
                    icon: "gauge.with.dots.needle.50percent",
                    text: "Public model names, feature names, coarse performance, redacted crash diagnostics, and error categories"
                )
                disclosureRow(
                    icon: "hand.raised.fill",
                    text: "Never prompts, responses, attachments, API keys, account details, device serials, or unredacted user paths"
                )
            }

            Text("The app and its embedded engine share one random identifier so this Mac is counted once. Change this choice anytime in Settings > Privacy.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            HStack(spacing: 10) {
                Spacer()
                Button("Don't share", role: .cancel) {
                    onDecision(false)
                }
                .accessibilityIdentifier("TelemetryConsent.DontShare")
                Button("Share anonymous data") {
                    onDecision(true)
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .focused($primaryActionFocused)
                .accessibilityIdentifier("TelemetryConsent.Share")
            }
        }
        .padding(24)
        .frame(width: 500)
        .onAppear { primaryActionFocused = true }
    }

    private func disclosureRow(icon: String, text: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: icon)
                .font(.system(size: 15, weight: .medium))
                .foregroundStyle(.secondary)
                .frame(width: 20, height: 20)
            Text(text)
                .font(.callout)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
