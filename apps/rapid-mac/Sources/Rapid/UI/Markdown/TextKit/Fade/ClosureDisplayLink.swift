import AppKit
import QuartzCore

/// A `CADisplayLink` wrapper that calls a closure each frame.
///
/// Created via `NSView.displayLink(target:selector:)` rather than
/// `CADisplayLink(target:selector:)`. The view-bound variant binds to the
/// display the view is *actually on*, which is the difference between smooth
/// and stuttering when a 120Hz and a 60Hz monitor are both attached.
///
/// The callback receives `targetTimestamp` — the moment the frame will be
/// shown — not `CACurrentMediaTime()`. Animating against "now" is one frame
/// behind by construction and jitters under load.
@MainActor
final class ClosureDisplayLink {

    private var link: CADisplayLink?
    private var onTick: ((CFTimeInterval) -> Void)?

    var isRunning: Bool { link != nil }

    func start(
        in view: NSView,
        preferredFrameRate: Float,
        minimumFrameRate: Float,
        onTick: @escaping (CFTimeInterval) -> Void
    ) {
        guard link == nil else { return }
        self.onTick = onTick
        let link = view.displayLink(target: self, selector: #selector(step(_:)))
        // A range rather than a fixed rate: the compositor can drop to the
        // minimum while nothing is fading instead of burning frames.
        link.preferredFrameRateRange = CAFrameRateRange(
            minimum: minimumFrameRate,
            maximum: preferredFrameRate,
            preferred: preferredFrameRate
        )
        link.add(to: .main, forMode: .common)
        self.link = link
    }

    func stop() {
        link?.invalidate()
        link = nil
        onTick = nil
    }

    @objc private func step(_ link: CADisplayLink) {
        onTick?(link.targetTimestamp)
    }

    // No `deinit` cleanup: `CADisplayLink` is not `Sendable`, so a nonisolated
    // deinit cannot touch it. Callers own the lifecycle and call `stop()` —
    // the animator does so in `reset()` and when its parts drain.
}
