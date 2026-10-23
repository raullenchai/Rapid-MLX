import SwiftUI

enum NewsletterLink {
    static let url = URL(string: "https://rapidmlx.com/newsletter?src=desktop")!
}

struct NewsletterPromptCard: View {
    @Environment(PromptScheduler.self) private var prompt
    @Environment(\.openURL) private var openURL

    var body: some View {
        HStack(alignment: .top, spacing: RapidTheme.Space.sm) {
            Image(systemName: "envelope.fill")
                .font(.system(size: 17, weight: .semibold))
                .foregroundStyle(RapidTheme.amberDeep)
                .frame(width: 22, height: 22)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
                Text("New models for your Mac, the day they drop")
                    .font(.rapid(size: 14, weight: .semibold))
                    .foregroundStyle(RapidTheme.textPrimary)
                Text("A short newsletter: new models, what fits your RAM, one command to run them.")
                    .font(.rapid(size: 14))
                    .foregroundStyle(RapidTheme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                HStack(spacing: RapidTheme.Space.sm) {
                    Button("Subscribe") {
                        openURL(NewsletterLink.url)
                        prompt.completeNewsletter()
                    }
                    .accessibilityIdentifier("Newsletter.ValueMoment.Subscribe")
                    .buttonStyle(RapidPrimaryButtonStyle(expands: true, height: RapidTheme.ControlHeight.medium,
                                                         font: .rapid(size: 14, weight: .medium)))
                    Button("Not now") { prompt.completeNewsletter() }
                        .accessibilityIdentifier("Newsletter.ValueMoment.Dismiss")
                        .buttonStyle(RapidSecondaryButtonStyle(expands: true, height: RapidTheme.ControlHeight.medium,
                                                               font: .rapid(size: 14, weight: .medium)))
                }
            }
        }
        .padding(14)
        .frame(width: 360)
        .background(RoundedRectangle(cornerRadius: 12).fill(RapidTheme.card))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(RapidTheme.hairline.opacity(0.8)))
        .shadow(color: Color.black.opacity(0.08), radius: 10, y: 4)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("Newsletter.ValueMoment.Card")
    }
}
