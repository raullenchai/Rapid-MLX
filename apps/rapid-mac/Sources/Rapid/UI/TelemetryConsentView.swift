import SwiftUI

/// Non-modal invitation shown only after Rapid has delivered useful output.
/// It deliberately does not take focus from the composer or the result the
/// user just received.
struct DeferredTelemetryConsentBanner: View {
    @Environment(DeferredTelemetryConsentCoordinator.self) private var consent

    var body: some View {
        HStack(spacing: RapidTheme.Space.md) {
            Image(systemName: "chart.bar.xaxis")
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(RapidTheme.brand)
                .frame(width: 32, height: 32)
                .background(RapidTheme.brandTint, in: RoundedRectangle(cornerRadius: RapidTheme.Radius.row))

            VStack(alignment: .leading, spacing: RapidTheme.Space.xxs) {
                Text("Help improve Rapid by sharing anonymous usage data?")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(RapidTheme.textPrimary)
                Text("For each first successful text chat reply, dictation, or generated image, only the milestone name and “Desktop” are sent. This version does not send a vision-reply milestone. The collector derives a country code but never stores your IP. Prompts, responses, attachments, and API keys are never collected. Change this anytime in Settings → Privacy.")
                    .font(.system(size: 12))
                    .foregroundStyle(RapidTheme.textSecondary)
                    .lineLimit(4)
            }

            Spacer(minLength: RapidTheme.Space.md)

            Button("No thanks", role: .cancel) { consent.decline() }
                .buttonStyle(.bordered)
                .accessibilityIdentifier("TelemetryConsent.PostValue.Decline")

            Button("Share") { consent.share() }
                .buttonStyle(.rapidPrimaryCompact)
                .accessibilityIdentifier("TelemetryConsent.PostValue.Share")

            Button { consent.close() } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 11, weight: .semibold))
                    .frame(width: 24, height: 24)
            }
            .buttonStyle(.plain)
            .help("Dismiss")
            .accessibilityLabel("Dismiss telemetry invitation")
            .accessibilityIdentifier("TelemetryConsent.PostValue.Close")
        }
        .padding(.horizontal, RapidTheme.Space.lg)
        .padding(.vertical, RapidTheme.Space.sm)
        .background(RapidTheme.surfaceRaised)
        .overlay(alignment: .bottom) { Rectangle().fill(RapidTheme.hairline).frame(height: 1) }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("TelemetryConsent.PostValueBanner")
    }
}
