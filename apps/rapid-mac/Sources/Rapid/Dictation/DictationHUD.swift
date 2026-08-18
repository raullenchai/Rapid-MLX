import AppKit
import SwiftUI

/// The floating capsule shown while dictating. This is the only interface most
/// dictations ever surface — the Audio tab is configuration, this is the product.
@MainActor
final class DictationHUD {
    enum Phase: Equatable {
        /// Only seen on a cold microphone. With the recorder kept warm between
        /// dictations this phase is usually skipped entirely.
        case starting
        case recording(seconds: TimeInterval, level: Float)
        case transcribing
    }

    private var panel: NSPanel?
    private let model = HUDModel()

    func show(_ phase: Phase) {
        model.phase = phase
        guard panel == nil else { return }

        let hosting = NSHostingView(rootView: DictationHUDView(model: model))
        hosting.frame = NSRect(x: 0, y: 0, width: 210, height: 48)

        let panel = NSPanel(
            contentRect: hosting.frame,
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        panel.contentView = hosting
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = true
        panel.level = .statusBar
        panel.ignoresMouseEvents = true
        panel.hidesOnDeactivate = false
        // Dictation happens in other apps, so the HUD must survive Space
        // switches and never steal focus from the app receiving the text.
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        panel.isFloatingPanel = true

        reposition(panel)
        panel.orderFrontRegardless()
        self.panel = panel
    }

    func update(_ phase: Phase) {
        model.phase = phase
        if panel == nil { show(phase) }
    }

    func hide() {
        panel?.orderOut(nil)
        panel = nil
    }

    private func reposition(_ panel: NSPanel) {
        guard let screen = NSScreen.main else { return }
        let frame = screen.visibleFrame
        let size = panel.frame.size
        panel.setFrameOrigin(
            NSPoint(
                x: frame.midX - size.width / 2,
                y: frame.minY + 120
            )
        )
    }
}

@MainActor
@Observable
private final class HUDModel {
    var phase: DictationHUD.Phase = .starting
}

private struct DictationHUDView: View {
    @Bindable var model: HUDModel
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        HStack(spacing: 11) {
            indicator
            content
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 12)
        .background(
            Capsule().fill(Color(nsColor: .init(white: 0.09, alpha: 0.95)))
        )
        .overlay(
            Capsule().strokeBorder(Color.white.opacity(0.09), lineWidth: 1)
        )
        .fixedSize()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var indicator: some View {
        Circle()
            .fill(dotColor)
            .frame(width: 9, height: 9)
    }

    @ViewBuilder
    private var content: some View {
        switch model.phase {
        case .starting:
            Text("Starting…")
                .font(.system(size: 13, design: .monospaced))
                .foregroundStyle(Color.white.opacity(0.9))
        case .recording(let seconds, let level):
            HStack(spacing: 10) {
                LevelMeter(level: level, animated: !reduceMotion)
                Text(Self.timeLabel(seconds))
                    .font(.system(size: 13, design: .monospaced))
                    .monospacedDigit()
                    .foregroundStyle(Color.white.opacity(0.92))
            }
        case .transcribing:
            Text("Transcribing…")
                .font(.system(size: 13, design: .monospaced))
                .foregroundStyle(Color.white.opacity(0.9))
        }
    }

    private var dotColor: Color {
        switch model.phase {
        case .starting: return Color(red: 0.88, green: 0.64, blue: 0.24)
        case .recording: return Color(red: 0.95, green: 0.25, blue: 0.29)
        case .transcribing: return Color(red: 0.35, green: 0.62, blue: 0.86)
        }
    }

    private static func timeLabel(_ seconds: TimeInterval) -> String {
        seconds < 60
            ? String(format: "%.1fs", seconds)
            : String(format: "%d:%02d", Int(seconds) / 60, Int(seconds) % 60)
    }
}

/// Live input level. Beyond looking alive, it answers the one question a static
/// dot cannot: whether the microphone actually picking you up is the one you
/// meant to use.
private struct LevelMeter: View {
    let level: Float
    let animated: Bool

    private static let bars = 6

    var body: some View {
        HStack(alignment: .bottom, spacing: 2.5) {
            ForEach(0..<Self.bars, id: \.self) { index in
                Capsule()
                    .fill(Color(red: 0.95, green: 0.25, blue: 0.29))
                    .frame(width: 2.5, height: height(for: index))
            }
        }
        .frame(height: 16, alignment: .bottom)
        .animation(animated ? .easeOut(duration: 0.08) : nil, value: level)
    }

    private func height(for index: Int) -> CGFloat {
        // Speech energy is logarithmic; a linear bar barely moves at
        // conversational volume.
        let normalized = CGFloat(min(1, max(0, level)))
        let boosted = pow(normalized, 0.45)
        let phase = CGFloat(index) / CGFloat(Self.bars - 1)
        let shape = 0.55 + 0.45 * sin(phase * .pi)
        return max(3, boosted * 15 * shape)
    }
}
