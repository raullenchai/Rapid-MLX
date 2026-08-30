import SwiftUI

/// The three questions offered under the last answer.
///
/// ## Why it holds its height while it is empty
///
/// This is the whole reason the type has a `reservedHeight` at all.
///
/// ``TranscriptScrollPositionProbe`` follows the bottom of the transcript by
/// watching the document frame: any growth while the reader is pinned pulls
/// the view down. During a stream that is right, and there is a release valve
/// for the case where the answer outgrows the viewport — but the valve is
/// gated on `isStreaming`, which is already false by the time a settled-turn
/// footer could appear. Chips arriving a second after the answer settles
/// would therefore yank a pinned reader with nothing to stop them, and the
/// same growth flips `isPinnedToBottom` false for a frame, which pops the
/// jump-to-bottom button and pops it back.
///
/// So the rail mounts at its final height the instant the turn settles — in
/// the same layout pass that adds the stats caption and the action row — and
/// the chips fade in *inside* that reservation. There is one document growth
/// per turn, at the moment the reader already expects one. Filling the rail
/// changes no height, posts no frame notification, and therefore cannot move
/// anybody. The hazard is removed by construction rather than guarded
/// against, which is why the probe needs no changes at all.
///
/// The height is a constant because the rail can never wrap: one line, one
/// horizontal scroller, `.lineLimit(1)` on every chip. That is also why the
/// row scrolls sideways rather than flowing — a flowing layout would make the
/// height depend on the text, and the text comes from a model.
struct FollowUpSuggestionRail: View {

    let state: ChatViewModel.FollowUpState
    let isEnabled: Bool
    let disabledTooltip: String
    let onSelect: (String) -> Void

    /// One control plus the gap above it. Fixed — see the type comment.
    static let reservedHeight: CGFloat =
        RapidTheme.ControlHeight.small + RapidTheme.Space.md

    var body: some View {
        ZStack(alignment: .leading) {
            // Holds the space open in every state, including `.idle`, so the
            // rail's arrival and departure are the only layout events.
            Color.clear
            if case .ready(let questions) = state {
                chips(questions)
            }
        }
        .frame(height: Self.reservedHeight, alignment: .bottom)
        .rapidAnimation(RapidMotion.quick, value: state)
    }

    private func chips(_ questions: [String]) -> some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: RapidTheme.Space.sm) {
                ForEach(Array(questions.enumerated()), id: \.offset) { index, question in
                    Button {
                        onSelect(question)
                    } label: {
                        Text(question)
                            .font(.caption)
                            .lineLimit(1)
                            .padding(.horizontal, RapidTheme.Space.md)
                            .frame(height: RapidTheme.ControlHeight.small)
                            // Deliberately `card`, not `brandPrimary`: the
                            // composer's send disc is this surface's one amber
                            // moment, and three glowing chips would take that
                            // away from it.
                            .background(RapidTheme.card)
                            .clipShape(Capsule())
                            .overlay(
                                Capsule().stroke(RapidTheme.hairline, lineWidth: 1)
                            )
                    }
                    .buttonStyle(.plain)
                    .disabled(!isEnabled)
                    .help(isEnabled ? question : disabledTooltip)
                    // Keyed on the index, never the text: the label is model
                    // output, and an identifier that moves with it is not a
                    // hook. Deliberately NOT prefixed `ChatView.Message.` —
                    // `gui-golden-flows.sh`'s `transcript_only` scopes its
                    // slice to that prefix, so a chip carrying it would drag
                    // model-authored text into a frozen snapshot.
                    .accessibilityIdentifier("ChatView.FollowUp.\(index)")
                }
            }
        }
        .transition(.opacity)
    }
}
