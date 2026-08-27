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
            Text("\(text.count) of \(CustomInstructionsConfig.maximumLength) characters")
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
                    label: "Clear global system prompt",
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
    let global: String
    let onSave: (String) -> Void
    let onCancel: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.lg) {
            VStack(alignment: .leading, spacing: RapidTheme.Space.xxs) {
                Text("Conversation System Prompt")
                    .font(RapidFont.sectionTitle)
                    .foregroundStyle(RapidTheme.textPrimary)
                Text("Sent only with this conversation. If it conflicts with the global default, this prompt wins.")
                    .font(RapidFont.caption)
                    .foregroundStyle(RapidTheme.textSecondary)
            }
            InstructionTextEditor(
                text: $draft,
                placeholder: "Add a system prompt for this conversation.",
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
                    .help("Clear conversation system prompt")
                    .accessibilityLabel("Clear conversation system prompt")
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
            EffectiveSystemPromptDisclosure(
                global: global,
                conversation: draft,
                accessibilityIdentifier: "ChatView.SystemPrompt.EffectivePreview"
            )
        }
        .padding(RapidTheme.Space.xl)
        .frame(width: 440)
        .background(RapidTheme.surfaceOverlay)
    }
}

/// Progressive disclosure for the exact Desktop-authored base prompt. The
/// server or request path may append tool and attachment context at send time,
/// which the explanatory copy states instead of pretending the preview owns
/// those dynamic layers.
struct EffectiveSystemPromptDisclosure: View {
    let global: String
    let conversation: String
    let accessibilityIdentifier: String

    @State private var expanded = false

    /// One clock sample for the preview. Keeping this pure lets the rollover
    /// contract prove the displayed prompt uses the same date-context helper
    /// as request assembly without waiting for wall-clock time in a UI test.
    nonisolated static func prompt(
        at now: Date,
        calendar: Calendar,
        global: String,
        conversation: String
    ) -> String {
        ChatViewModel.effectiveSystemPrompt(
            dateContext: ChatViewModel.currentDateTimeContext(
                now: now,
                calendar: calendar
            ),
            global: global,
            conversation: conversation
        )
    }

    var body: some View {
        DisclosureGroup("Effective System Prompt", isExpanded: $expanded) {
            VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
                Text("Preview includes current automatic context. Tool and attachment context may be added when you send.")
                    .font(RapidFont.caption)
                    .foregroundStyle(RapidTheme.textSecondary)
                TimelineView(.periodic(from: .now, by: 60)) { context in
                    ScrollView {
                        Text(Self.prompt(
                            at: context.date,
                            calendar: .autoupdatingCurrent,
                            global: global,
                            conversation: conversation
                        ))
                            .font(RapidFont.code)
                            .foregroundStyle(RapidTheme.textPrimary)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(RapidTheme.Space.md)
                            .accessibilityIdentifier(accessibilityIdentifier)
                    }
                    .frame(maxHeight: 180)
                    .background(
                        RoundedRectangle(
                            cornerRadius: RapidTheme.Radius.input,
                            style: .continuous
                        )
                        .fill(RapidTheme.surfaceCode)
                    )
                }
            }
            .padding(.top, RapidTheme.Space.sm)
        }
        .font(RapidFont.body)
        .accessibilityIdentifier("\(accessibilityIdentifier).Disclosure")
    }
}
