import SwiftUI

/// The incognito control on the New Chat page. Pressing it again leaves
/// temporary mode.
struct TemporaryChatToggle: View {
    let isOn: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: isOn ? "eye.slash.fill" : "eye.slash")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(isOn ? RapidTheme.brandAmber : Color.secondary)
                .frame(width: 28, height: 28)
                .background(Circle().fill(Color.primary.opacity(isOn ? 0.14 : 0.06)))
        }
        .buttonStyle(.plain)
        .help(isOn
            ? String(localized: "Turn off temporary chat")
            : String(localized: "Turn on temporary chat"))
        .accessibilityLabel("Temporary chat")
        .accessibilityValue(isOn ? String(localized: "On") : String(localized: "Off"))
        .accessibilityAddTraits(.isToggle)
        .accessibilityIdentifier("Chat.NewChat.TemporaryToggle")
    }
}

/// Warns before a used temporary chat is thrown away, and offers the app's
/// existing export formats first.
struct TemporaryChatExitDialog: ViewModifier {
    @Bindable var controller: TemporaryChatController

    func body(content: Content) -> some View {
        content.alert(
            "Discard this temporary chat?",
            isPresented: $controller.isConfirmingExit
        ) {
            Button("Export Markdown…") { controller.export(.markdown) }
            Button("Export JSON…") { controller.export(.json) }
            Button("Discard Chat", role: .destructive) { controller.confirmDiscard() }
            Button("Cancel", role: .cancel) { controller.cancelExit() }
        } message: {
            Text("Temporary chats aren't saved. Turning this off deletes the conversation. Export it first to keep a copy.")
        }
    }
}
