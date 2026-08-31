import AppKit
import QuartzCore
import SwiftUI

/// Delivers frame callbacks in sync with the display containing this view.
///
/// A view display link only fires while the view's window sits on an active
/// display. If the window is off every display — restored onto a disconnected
/// monitor, dragged out of the visible frame, or hosted off-screen by an
/// in-process test stage — the link stays silent and buffered streaming text
/// would never reveal. A main-runloop timer stands in for exactly that
/// condition so the presentation pipeline drains wherever the window lives.
struct StreamingPresentationDisplayLink: NSViewRepresentable {
    let isActive: Bool
    let onFrame: @MainActor (TimeInterval) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(isActive: isActive, onFrame: onFrame)
    }

    func makeNSView(context: Context) -> NSView {
        let view = HostView(frame: .zero)
        view.coordinator = context.coordinator
        context.coordinator.attach(to: view)
        return view
    }

    func updateNSView(_ view: NSView, context: Context) {
        context.coordinator.onFrame = onFrame
        context.coordinator.setActive(isActive)
        context.coordinator.attach(to: view)
    }

    static func dismantleNSView(_ view: NSView, coordinator: Coordinator) {
        coordinator.invalidate()
    }

    /// Tells the coordinator when the hosting conditions that decide between
    /// the display link and the timer fallback may have changed.
    final class HostView: NSView {
        weak var coordinator: Coordinator?

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            coordinator?.hostingConditionsChanged()
        }
    }

    @MainActor
    final class Coordinator: NSObject {
        /// Matches the common display cadence; the presentation buffer paces
        /// by wall-clock duration, so a display running at another rate only
        /// changes callback granularity, not reveal speed.
        private static let fallbackInterval: TimeInterval = 1.0 / 60.0

        var onFrame: @MainActor (TimeInterval) -> Void
        private(set) var isActive: Bool
        private var displayLink: CADisplayLink?
        private var fallbackTimer: Timer?
        private weak var view: NSView?
        private var screenObserver: NSObjectProtocol?

        var isDisplayLinkPaused: Bool { displayLink?.isPaused ?? !isActive }

        init(
            isActive: Bool,
            onFrame: @escaping @MainActor (TimeInterval) -> Void
        ) {
            self.isActive = isActive
            self.onFrame = onFrame
        }

        func attach(to view: NSView) {
            if self.view !== view {
                self.view = view
                observeScreenChanges()
            }
            defer { reconcileFallback() }
            guard displayLink == nil else { return }
            let link = view.displayLink(
                target: self,
                selector: #selector(displayLinkDidFire(_:))
            )
            link.isPaused = !isActive
            link.add(to: .main, forMode: .common)
            displayLink = link
        }

        func setActive(_ isActive: Bool) {
            self.isActive = isActive
            displayLink?.isPaused = !isActive
            reconcileFallback()
        }

        func hostingConditionsChanged() {
            observeScreenChanges()
            reconcileFallback()
        }

        func invalidate() {
            displayLink?.invalidate()
            displayLink = nil
            fallbackTimer?.invalidate()
            fallbackTimer = nil
            if let screenObserver {
                NotificationCenter.default.removeObserver(screenObserver)
                self.screenObserver = nil
            }
        }

        @objc private func displayLinkDidFire(_ link: CADisplayLink) {
            let duration = link.duration > 0
                ? link.duration
                : link.targetTimestamp - link.timestamp
            onFrame(duration)
        }

        private var needsFallback: Bool {
            isActive && view?.window?.screen == nil
        }

        private func reconcileFallback() {
            guard needsFallback else {
                fallbackTimer?.invalidate()
                fallbackTimer = nil
                lastFallbackFire = nil
                return
            }
            guard fallbackTimer == nil else { return }
            let timer = Timer(
                timeInterval: Self.fallbackInterval,
                repeats: true
            ) { [weak self] _ in
                MainActor.assumeIsolated {
                    self?.fallbackTimerDidFire()
                }
            }
            RunLoop.main.add(timer, forMode: .common)
            fallbackTimer = timer
            lastFallbackFire = nil
        }

        private var lastFallbackFire: TimeInterval?

        private func fallbackTimerDidFire() {
            guard needsFallback else {
                reconcileFallback()
                return
            }
            // Report the measured gap, not the nominal interval: a busy main
            // run loop delays or coalesces timer firings, and the buffer
            // paces reveal by the durations it is told, so under-reporting
            // would reveal text slower than wall-clock.
            let now = ProcessInfo.processInfo.systemUptime
            let duration = lastFallbackFire.map { now - $0 } ?? Self.fallbackInterval
            lastFallbackFire = now
            onFrame(duration)
        }

        private func observeScreenChanges() {
            if let screenObserver {
                NotificationCenter.default.removeObserver(screenObserver)
                self.screenObserver = nil
            }
            guard let window = view?.window else { return }
            screenObserver = NotificationCenter.default.addObserver(
                forName: NSWindow.didChangeScreenNotification,
                object: window,
                queue: .main
            ) { [weak self] _ in
                MainActor.assumeIsolated {
                    self?.hostingConditionsChanged()
                }
            }
        }
    }
}
