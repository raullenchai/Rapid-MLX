import SwiftUI

/// Non-modal launch disclosure for telemetry v2. This is a notice with one
/// acknowledgement, not a yes/no consent prompt; Settings remains the durable
/// opt-out surface.
struct TelemetryNoticeBanner: View {
    @Environment(TelemetryNoticeCoordinator.self) private var notice

    var body: some View {
        HStack(spacing: RapidTheme.Space.md) {
            Image(systemName: "chart.bar.xaxis")
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(RapidTheme.brand)
                .frame(width: 32, height: 32)
                .background(
                    RapidTheme.brandTint,
                    in: RoundedRectangle(cornerRadius: RapidTheme.Radius.row)
                )

            VStack(alignment: .leading, spacing: RapidTheme.Space.xxs) {
                Text("Anonymous telemetry is now on by default")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(RapidTheme.textPrimary)
                Text("This includes installs that previously turned telemetry off. Rapid sends metadata-only events: the app's to rapidmlx.com's telemetry service, the bundled engine's to PostHog Cloud (US)—never your IP or a per-person profile; the app's collector keeps only a coarse country code. Rapid never sends prompts, responses, file paths, or API key values. Nothing is sent before this notice appears. Turn it off in Settings → Privacy, with `rapid-mlx telemetry off`, `RAPID_MLX_TELEMETRY=0`, or `DO_NOT_TRACK=1`. Learn more: https://rapidmlx.com/docs/telemetry")
                    .font(.system(size: 12))
                    .foregroundStyle(RapidTheme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
            }

            Spacer(minLength: RapidTheme.Space.md)

            Button("Got it") { notice.acknowledge() }
                .buttonStyle(.rapidPrimaryCompact)
                .accessibilityIdentifier("TelemetryNotice.Acknowledge")
        }
        .padding(.horizontal, RapidTheme.Space.lg)
        .padding(.vertical, RapidTheme.Space.sm)
        .background(RapidTheme.surfaceRaised)
        .overlay(alignment: .bottom) {
            Rectangle().fill(RapidTheme.hairline).frame(height: 1)
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("TelemetryNotice.Banner")
        .onAppear { notice.noticeDidAppear() }
    }
}
