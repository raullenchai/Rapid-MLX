import SwiftUI

/// Shared multiline editor treatment for global and per-conversation
/// instructions. The parent owns save semantics; this view owns only the field.
struct InstructionTextEditor: View {
    @Binding var text: String
    let placeholder: String
    let height: CGFloat
    let accessibilityIdentifier: String
    var autoFocus: Bool = false

    @FocusState private var focused: Bool

    var body: some View {
        ZStack(alignment: .topLeading) {
            if text.isEmpty {
                Text(placeholder)
                    .font(RapidFont.body)
                    .foregroundStyle(RapidTheme.textTertiary)
                    .padding(.horizontal, RapidTheme.Space.md)
                    .padding(.vertical, RapidTheme.Space.sm + 1)
                    .allowsHitTesting(false)
            }
            TextEditor(text: Binding(
                get: { text },
                set: { text = CustomInstructionsConfig.limited($0) }
            ))
                .accessibilityIdentifier(accessibilityIdentifier)
                .font(RapidFont.body)
                .scrollContentBackground(.hidden)
                .focused($focused)
                .padding(RapidTheme.Space.xs)
        }
        .frame(height: height)
        .background(
            RoundedRectangle(cornerRadius: RapidTheme.Radius.input, style: .continuous)
                .fill(RapidTheme.surfaceCode)
        )
        .overlay(
            RoundedRectangle(cornerRadius: RapidTheme.Radius.input, style: .continuous)
                .strokeBorder(RapidTheme.hairlineStrong, lineWidth: 1)
        )
        .contentShape(
            RoundedRectangle(cornerRadius: RapidTheme.Radius.input, style: .continuous)
        )
        .onTapGesture { focused = true }
        .overlay(alignment: .bottomTrailing) {
            Text("\(text.count) / \(CustomInstructionsConfig.maximumLength)")
                .font(RapidFont.caption)
                .foregroundStyle(RapidTheme.textTertiary)
                .padding(RapidTheme.Space.sm)
                .accessibilityIdentifier("\(accessibilityIdentifier).Count")
        }
        .task {
            guard autoFocus else { return }
            await Task.yield()
            guard !Task.isCancelled else { return }
            focused = true
        }
    }
}

/// Flat settings section for a single large editor. It deliberately avoids a
/// grouped card: the editor already defines its own surface and nesting it in
/// another bordered box adds visual weight without adding structure.
struct InstructionEditorSection<Content: View>: View {
    let title: String
    let subtitle: String
    let clearEnabled: Bool
    let onClear: () -> Void
    @ViewBuilder let content: Content

    init(
        _ title: String,
        subtitle: String,
        clearEnabled: Bool,
        onClear: @escaping () -> Void,
        @ViewBuilder content: () -> Content
    ) {
        self.title = title
        self.subtitle = subtitle
        self.clearEnabled = clearEnabled
        self.onClear = onClear
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
            SectionHeader(title, subtitle: subtitle, emphasis: .section) {
                QuietIconButton(
                    symbol: "trash",
                    label: "Clear global instructions",
                    action: onClear
                )
                .disabled(!clearEnabled)
                .accessibilityIdentifier("Settings.Instructions.Clear")
            }
            content
        }
    }
}

struct ConversationInstructionsPopover: View {
    @Binding var draft: String
    let onSave: (String) -> Void
    let onCancel: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.lg) {
            VStack(alignment: .leading, spacing: RapidTheme.Space.xxs) {
                Text("Conversation Instructions")
                    .font(RapidFont.sectionTitle)
                    .foregroundStyle(RapidTheme.textPrimary)
                Text("Applied only to this conversation, after global instructions.")
                    .font(RapidFont.caption)
                    .foregroundStyle(RapidTheme.textSecondary)
            }
            InstructionTextEditor(
                text: $draft,
                placeholder: "Add instructions for this conversation.",
                height: 160,
                accessibilityIdentifier: "ChatView.ConversationInstructions.Editor",
                autoFocus: true
            )
            HStack(spacing: RapidTheme.Space.sm) {
                if !draft.isEmpty {
                    Button {
                        draft = ""
                    } label: {
                        Image(systemName: "trash")
                    }
                    .buttonStyle(.rapidSecondaryCompact)
                    .help("Clear instructions")
                    .accessibilityLabel("Clear instructions")
                    .accessibilityIdentifier("ChatView.ConversationInstructions.Clear")
                }
                Spacer(minLength: 0)
                Button("Cancel", action: onCancel)
                    .buttonStyle(.rapidSecondaryCompact)
                    .keyboardShortcut(.cancelAction)
                    .accessibilityIdentifier("ChatView.ConversationInstructions.Cancel")
                Button("Save") { onSave(draft) }
                    .buttonStyle(.rapidPrimaryCompact)
                    .keyboardShortcut(.defaultAction)
                    .accessibilityIdentifier("ChatView.ConversationInstructions.Save")
            }
        }
        .padding(RapidTheme.Space.xl)
        .frame(width: 440)
        .background(RapidTheme.surfaceOverlay)
    }
}
