import SwiftUI

/// Modal "Select text…" surface for an assistant message.
///
/// Why this exists: the completed-message body renders through
/// MarkdownUI, which emits every block-level element (paragraph,
/// list item, heading) as its own SwiftUI view — and
/// ``textSelection(.enabled)`` cannot span views, so dragging in the
/// transcript always dies at the block edge (2026-07 dogfood: "I can
/// only ever select one line at a time"). This sheet renders the
/// same sanitised text as ONE selectable ``Text``, where selection
/// crosses lines, paragraphs and list items freely.
///
/// Deliberately a stopgap: the real fix — native cross-block
/// selection inline in the transcript — is the P2 "Text selection"
/// entry in ``docs/plans/v1-prod-readiness-gaps.md`` and needs a
/// rendering rework (a single AttributedString/NSTextView surface
/// reimplementing the ``.rapidChat`` theme and the LaTeX
/// segmentation). Until then this delivers the actual user need:
/// copying an arbitrary passage.
struct SelectTextSheet: View {
    let text: String
    @Environment(\.dismiss) private var dismiss

    /// What the sheet shows for a message body: the display-
    /// sanitised text — the same pipeline the transcript renders and
    /// the Copy button writes, so bidi/control scalars never reach
    /// this surface either (F-10-4 family). Static + pure so tests
    /// pin the contract without mounting the sheet.
    static func selectableText(for content: String) -> String {
        ChatTextSanitizer.sanitizeForDisplay(content)
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Select text")
                        .font(.headline)
                    Text("Selection here crosses paragraphs — ⌘C copies it.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("Done") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                    .accessibilityIdentifier("SelectText.Done")
            }
            .padding(16)
            Divider()
            ScrollView {
                // ONE Text on purpose — cross-block selection is the
                // whole point of this sheet. Type mirrors the chat
                // body (15 pt serif, 5 pt leading) so a passage reads
                // the same here as where the user just saw it.
                Text(text)
                    .scaledSystemFont(15, design: .serif)
                    .lineSpacing(5)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(16)
            }
        }
        .frame(minWidth: 440, idealWidth: 560, minHeight: 320, idealHeight: 480)
    }
}
