import SwiftUI

/// The "jump to newest" control that appears when the reader has scrolled
/// away from the bottom.
///
/// Ported from native-chat. Two jobs in one element: the arrow says *there is
/// newer content below*, and the ring around it says *more is still
/// arriving*. Scrolled up, the typing dot at the end of the reply is off
/// screen, so without the ring the reader cannot tell a finished answer from
/// one still being written.
///
/// Rapid drives transcript follow-mode from AppKit
/// (``TranscriptScrollPositionProbe``) rather than a `ScrollViewReader`, so
/// this button has no scrolling of its own: flipping `isPinnedToBottom` back
/// to `true` re-enters the probe through `updateNSView`, whose `attach`
/// scrolls to the bottom. One owner for positioning, as the probe's comment
/// requires.
struct JumpToBottomButton: View {
    var isStreaming: Bool
    var action: () -> Void

    @State private var ringPhase: Double = 0
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private let diameter: CGFloat = 32

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle()
                    .fill(RapidTheme.card)
                    .overlay {
                        Circle().stroke(RapidTheme.hairline, lineWidth: 1)
                    }
                    .shadow(color: .black.opacity(0.12), radius: 6, y: 2)

                if isStreaming {
                    // A quarter-circle sweep reads as motion at any size; a
                    // near-full ring looks static while it turns.
                    Circle()
                        .trim(from: 0, to: 0.25)
                        .stroke(
                            RapidTheme.brandAmber,
                            style: StrokeStyle(lineWidth: 2, lineCap: .round)
                        )
                        .frame(width: diameter - 3, height: diameter - 3)
                        .rotationEffect(.degrees(ringPhase))
                }

                Image(systemName: "arrow.down")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Color.primary)
            }
            .frame(width: diameter, height: diameter)
        }
        .buttonStyle(.plain)
        .accessibilityLabel(
            isStreaming
                ? "Jump to latest, still generating"
                : "Jump to latest"
        )
        .accessibilityIdentifier("Transcript.JumpToBottom")
        .onAppear { startRingIfNeeded() }
        .onChange(of: isStreaming) { _, streaming in
            if streaming {
                startRingIfNeeded()
            } else {
                // Stop where it is rather than snapping to 0 — a jump back to
                // the start reads as a second, unrelated event right as the
                // answer completes.
                withAnimation(.none) {
                    ringPhase = ringPhase.truncatingRemainder(dividingBy: 360)
                }
            }
        }
    }

    private func startRingIfNeeded() {
        guard isStreaming, !reduceMotion else { return }
        withAnimation(.linear(duration: 1).repeatForever(autoreverses: false)) {
            ringPhase += 360
        }
    }
}
