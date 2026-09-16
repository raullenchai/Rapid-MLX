import SwiftUI

/// A single, optically balanced four-point mark for Personal Intelligence.
///
/// This is intentionally drawn rather than borrowed from the generic
/// ``sparkles`` symbol: the three-glyph cluster becomes muddy at composer size
/// and makes the control look like an undifferentiated "AI" button.
struct PersonalIntelligenceGlyph: Shape {
    func path(in rect: CGRect) -> Path {
        let center = CGPoint(x: rect.midX, y: rect.midY)
        let halfWidth = rect.width / 2
        let halfHeight = rect.height / 2
        let shoulderX = halfWidth * 0.24
        let shoulderY = halfHeight * 0.24

        var path = Path()
        path.move(to: CGPoint(x: center.x, y: rect.minY))
        path.addCurve(
            to: CGPoint(x: rect.maxX, y: center.y),
            control1: CGPoint(x: center.x + shoulderX, y: center.y - shoulderY),
            control2: CGPoint(x: center.x + shoulderX, y: center.y - shoulderY)
        )
        path.addCurve(
            to: CGPoint(x: center.x, y: rect.maxY),
            control1: CGPoint(x: center.x + shoulderX, y: center.y + shoulderY),
            control2: CGPoint(x: center.x + shoulderX, y: center.y + shoulderY)
        )
        path.addCurve(
            to: CGPoint(x: rect.minX, y: center.y),
            control1: CGPoint(x: center.x - shoulderX, y: center.y + shoulderY),
            control2: CGPoint(x: center.x - shoulderX, y: center.y + shoulderY)
        )
        path.addCurve(
            to: CGPoint(x: center.x, y: rect.minY),
            control1: CGPoint(x: center.x - shoulderX, y: center.y - shoulderY),
            control2: CGPoint(x: center.x - shoulderX, y: center.y - shoulderY)
        )
        path.closeSubpath()
        return path
    }
}

struct PersonalIntelligencePopover: View {
    let modelAlias: String
    let modelSupported: Bool
    let showsIntroductionActions: Bool
    let onNotNow: () -> Void
    let onTurnOn: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
            VStack(alignment: .leading, spacing: RapidTheme.Space.xs) {
                Text("Personal Intelligence")
                    .font(.headline)
                if modelSupported {
                    Text("Use your Mac’s tools and local context to get things done.")
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                } else {
                    Text("Personal Intelligence hasn’t been tuned for \(modelAlias) yet. Regular chat is still available.")
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            if modelSupported {
                VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
                    promise("doc.text", "Reads only what you choose")
                    promise("checkmark.shield", "Asks before making changes")
                    promise("desktopcomputer", "Runs locally by default")
                }
            }

            if modelSupported && showsIntroductionActions {
                Divider()
                HStack(spacing: RapidTheme.Space.sm) {
                    Button("Not now", action: onNotNow)
                        .buttonStyle(.bordered)
                        .accessibilityIdentifier("ChatView.PersonalIntelligence.NotNow")
                    Button("Turn on", action: onTurnOn)
                        .buttonStyle(.borderedProminent)
                        .accessibilityIdentifier("ChatView.PersonalIntelligence.TurnOn")
                }
                .frame(maxWidth: .infinity, alignment: .trailing)
            }
        }
        .padding(RapidTheme.Space.lg)
        .frame(width: 340)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("ChatView.PersonalIntelligence.Popover")
    }

    private func promise(_ symbol: String, _ text: String) -> some View {
        Label(text, systemImage: symbol)
            .font(.callout)
    }
}
