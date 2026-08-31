import AppKit
import SwiftUI
import Testing

/// In-process stage for golden-flow tests: mounts a real SwiftUI view in a
/// window the user never sees and drives it through the same accessibility
/// surface the out-of-process AX golden flows use — find by identifier,
/// press, set value, wait for tree state — without launching the app,
/// taking the OS foreground, or needing a TCC Accessibility grant.
///
/// Recipe (each ingredient is load-bearing; discovered empirically on
/// macOS 26 during the golden-flow sink spike):
///
/// 1. The window sits at far off-screen coordinates and is ordered **back**,
///    never front and never key. Ordering it in (vs. never showing it) is
///    what lets AppKit present real `.popover` and `.sheet` windows, which
///    several journeys assert on. It still steals no focus and paints no
///    pixels on any display.
/// 2. A bare hosting view — even inside a window — reports zero AX children:
///    SwiftUI materializes its `AccessibilityNode` tree only when it
///    believes an assistive client is attached. The
///    `accessibilityEnhancedUserInterface` toggle simulates that client
///    (it is the flag VoiceOver sets over the AX wire).
/// 3. The materialized nodes are `SwiftUI.AccessibilityNode` instances,
///    which implement the AX getters/actions but are not KVC-compliant and
///    do not conform to `NSAccessibilityProtocol`; they must be driven via
///    `responds(to:)`/`perform(_:)`.
///
/// Sheets and popovers arrive as additional windows (`SheetPresentationWindow`,
/// `_NSPopoverWindow`), so every query walks all of the app's windows, not
/// just the stage's own.
@MainActor
final class GoldenStage {
    /// One (identifier, role, text) triple per AX node — the same currency
    /// the bash flows' `see_main` JSON captures per element.
    struct Node {
        let id: String
        let role: String
        let text: String
    }

    struct StageError: Error, CustomStringConvertible {
        let description: String
    }

    private let window: NSWindow
    private let host: NSView

    /// `accessibilityEnhancedUserInterface` is process-global, so stages
    /// reference-count it: the first live stage records the pre-existing
    /// value and turns it on, the last one restores what it found. Without
    /// this every stage would leave the whole test process permanently in
    /// assistive-client mode.
    private static var liveStageCount = 0
    private static var enhancedUIWasEnabled = false

    /// Default per-wait budget. Generous relative to the tens of
    /// milliseconds a settled tree usually takes, tiny relative to the
    /// bash flows' multi-second polling loops.
    static let defaultTimeout: TimeInterval = 10

    init<Content: View>(_ content: Content, size: CGSize = CGSize(width: 1024, height: 700)) {
        let host = NSHostingView(rootView: content)
        host.frame = CGRect(origin: .zero, size: size)
        // Off-screen on both axes so a popover clamped toward the screen on
        // one axis (observed: x is clamped, y is not) still never enters any
        // display's visible frame.
        let window = NSWindow(
            contentRect: CGRect(origin: CGPoint(x: -8000, y: -8000), size: size),
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        window.contentView = host
        window.isReleasedWhenClosed = false
        window.orderBack(nil)
        self.window = window
        self.host = host

        if Self.liveStageCount == 0 {
            let current = NSApplication.shared
                .value(forKey: "accessibilityEnhancedUserInterface") as? Bool
            Self.enhancedUIWasEnabled = current ?? false
            NSApplication.shared.setValue(
                true, forKey: "accessibilityEnhancedUserInterface"
            )
        }
        Self.liveStageCount += 1
        host.layoutSubtreeIfNeeded()
        Self.turnRunLoop()
    }

    deinit {
        let window = self.window
        Task { @MainActor in
            Self.liveStageCount -= 1
            if Self.liveStageCount == 0, !Self.enhancedUIWasEnabled {
                NSApplication.shared.setValue(
                    false, forKey: "accessibilityEnhancedUserInterface"
                )
            }
            window.orderOut(nil)
            window.close()
        }
    }

    // MARK: - Tree access

    /// The stage's window plus every presentation window it spawned —
    /// sheets attach via ``NSWindow/sheetParent`` and popovers via
    /// ``NSWindow/parent``, both possibly nested (a sheet's popover). Scoped
    /// this way so concurrent stages in one test process can never satisfy
    /// each other's queries through same-named identifiers.
    private func stageWindows() -> [NSWindow] {
        var scope: Set<ObjectIdentifier> = [ObjectIdentifier(window)]
        var result: [NSWindow] = [window]
        var grew = true
        while grew {
            grew = false
            for candidate in NSApplication.shared.windows {
                guard !scope.contains(ObjectIdentifier(candidate)) else { continue }
                let attachedTo = candidate.sheetParent ?? candidate.parent
                if let attachedTo, scope.contains(ObjectIdentifier(attachedTo)) {
                    scope.insert(ObjectIdentifier(candidate))
                    result.append(candidate)
                    grew = true
                }
            }
        }
        return result
    }

    /// Depth-first (id, role, text) triples across the stage's windows, so
    /// sheet and popover content participates in assertions exactly like
    /// main-window content does in the bash flows' full-tree walks.
    func tree() -> [Node] {
        var nodes: [Node] = []
        for window in stageWindows() {
            Self.walk(window, into: &nodes)
        }
        return nodes
    }

    /// Every text fragment on the stage, for substring assertions
    /// (`assert_tree_text` in the bash harness).
    func treeText() -> String {
        tree().map(\.text).filter { !$0.isEmpty }.joined(separator: "\n")
    }

    func identifiers() -> [String] {
        tree().map(\.id).filter { !$0.isEmpty }
    }

    /// First identifier matching a prefix — how the bash flows discover
    /// per-message identifiers with dynamic UUID suffixes.
    func identifier(withPrefix prefix: String, last: Bool = false) -> String? {
        let matches = identifiers().filter { $0.hasPrefix(prefix) }
        return last ? matches.last : matches.first
    }

    // MARK: - Driving

    func press(_ identifier: String) throws {
        guard let node = findNode(identifier) else {
            throw StageError(description: "press: no AX node with identifier \(identifier)")
        }
        let sel = NSSelectorFromString("accessibilityPerformPress")
        guard node.responds(to: sel) else {
            throw StageError(description: "press: \(identifier) does not support press")
        }
        // `accessibilityPerformPress` returns BOOL. `perform(_:)` is only
        // defined for object-returning selectors, so call through a typed
        // IMP instead of relying on the scalar riding home in a pointer.
        typealias PressIMP = @convention(c) (NSObject, Selector) -> Bool
        let imp = unsafeBitCast(node.method(for: sel), to: PressIMP.self)
        _ = imp(node, sel)
        Self.turnRunLoop()
    }

    /// AX value replacement — the analog of the AX driver's `set-value`
    /// (which the bash flows use for composer and editor text entry).
    func setValue(_ value: String, for identifier: String) throws {
        guard let node = findNode(identifier) else {
            throw StageError(description: "setValue: no AX node with identifier \(identifier)")
        }
        let sel = NSSelectorFromString("setAccessibilityValue:")
        guard node.responds(to: sel) else {
            throw StageError(description: "setValue: \(identifier) does not accept a value")
        }
        // Void-returning — same typed-IMP treatment as `press(_:)`.
        typealias SetValueIMP = @convention(c) (NSObject, Selector, NSString) -> Void
        let imp = unsafeBitCast(node.method(for: sel), to: SetValueIMP.self)
        imp(node, sel, value as NSString)
        Self.turnRunLoop()
    }

    /// The user's Esc key, delivered to the most recently attached stage
    /// window. This is how a user dismisses a transient popover or sheet;
    /// an AX press on the popover's anchor cannot stand in for it, because
    /// the transient-dismissal half of that gesture happens inside AppKit's
    /// event routing, before any button action runs.
    func pressEscape() {
        guard let target = stageWindows().last else { return }
        let escape = NSEvent.keyEvent(
            with: .keyDown,
            location: .zero,
            modifierFlags: [],
            timestamp: ProcessInfo.processInfo.systemUptime,
            windowNumber: target.windowNumber,
            context: nil,
            characters: "\u{1b}",
            charactersIgnoringModifiers: "\u{1b}",
            isARepeat: false,
            keyCode: 53
        )
        if let escape {
            target.sendEvent(escape)
        }
        Self.turnRunLoop()
    }

    func value(of identifier: String) -> String? {
        guard let node = findNode(identifier) else { return nil }
        let sel = NSSelectorFromString("accessibilityValue")
        guard node.responds(to: sel), let result = node.perform(sel) else { return nil }
        return result.takeUnretainedValue() as? String
    }

    // MARK: - Scrolling

    /// The stage's first scroll position as a 0 (top) … 1 (bottom) fraction —
    /// the same currency as the AX driver's scroll-bar value that the bash
    /// flows asserted on. Nil while nothing on the stage scrolls.
    func scrollFraction() -> Double? {
        guard let scrollView = firstScrollView(),
              let document = scrollView.documentView else { return nil }
        let clip = scrollView.contentView
        let travel = document.bounds.height - clip.bounds.height
        guard travel > 0 else { return nil }
        let fromTop = (clip.bounds.minY - document.bounds.minY) / travel
        let clamped = max(0, min(1, fromTop))
        return document.isFlipped ? clamped : 1 - clamped
    }

    /// Programmatic analog of the AX driver's `set-scroll-value`. The scroll
    /// arrives unbracketed by live-scroll notifications — exactly what a
    /// legacy mouse wheel produces — which the transcript's follow-mode probe
    /// deliberately treats as user intent.
    func setScrollFraction(_ fraction: Double) throws {
        guard let scrollView = firstScrollView(),
              let document = scrollView.documentView else {
            throw StageError(description: "setScrollFraction: no scroll view on the stage")
        }
        let clip = scrollView.contentView
        let travel = document.bounds.height - clip.bounds.height
        guard travel > 0 else {
            throw StageError(description: "setScrollFraction: stage content does not scroll")
        }
        let fromTop = document.isFlipped ? fraction : 1 - fraction
        let proposed = NSRect(
            origin: NSPoint(
                x: clip.bounds.minX,
                y: document.bounds.minY + travel * fromTop
            ),
            size: clip.bounds.size
        )
        clip.scroll(to: clip.constrainBoundsRect(proposed).origin)
        scrollView.reflectScrolledClipView(clip)
        Self.turnRunLoop()
    }

    private func firstScrollView() -> NSScrollView? {
        for window in stageWindows() {
            guard let root = window.contentView else { continue }
            if let hit = Self.firstScrollView(in: root) { return hit }
        }
        return nil
    }

    private static func firstScrollView(in view: NSView) -> NSScrollView? {
        if let scrollView = view as? NSScrollView { return scrollView }
        for subview in view.subviews {
            if let hit = firstScrollView(in: subview) { return hit }
        }
        return nil
    }

    // MARK: - Waiting

    /// Pump the main runloop until `condition` holds. Async so in-flight
    /// main-actor work (streaming callbacks, @Published mutations) can
    /// interleave with the polling instead of deadlocking behind it.
    func wait(
        for what: String,
        timeout: TimeInterval = GoldenStage.defaultTimeout,
        until condition: () -> Bool
    ) async throws {
        let deadline = Date(timeIntervalSinceNow: timeout)
        while Date() < deadline {
            if condition() { return }
            // Suspend rather than pump: an awaited sleep releases the main
            // actor, so the process's own main runloop turns (timers, AppKit
            // presentation) while other suites' main-actor work proceeds
            // undisturbed. A nested `RunLoop.main.run` here measurably
            // altered concurrent timing-sensitive tests' batching cadence
            // when the package test suite runs parallel. The sleep's
            // cancellation error propagates so a cancelled test stops
            // polling instead of riding out the full timeout.
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        if condition() { return }
        throw StageError(description: "timed out waiting for \(what)")
    }

    func waitForIdentifier(
        _ identifier: String,
        timeout: TimeInterval = GoldenStage.defaultTimeout
    ) async throws {
        try await wait(for: "identifier \(identifier)", timeout: timeout) {
            findNode(identifier) != nil
        }
    }

    func waitForIdentifierGone(
        _ identifier: String,
        timeout: TimeInterval = GoldenStage.defaultTimeout
    ) async throws {
        try await wait(for: "identifier \(identifier) to disappear", timeout: timeout) {
            findNode(identifier) == nil
        }
    }

    func waitForText(
        _ text: String,
        timeout: TimeInterval = GoldenStage.defaultTimeout
    ) async throws {
        try await wait(for: "tree text containing \(text)", timeout: timeout) {
            treeText().contains(text)
        }
    }

    // MARK: - Internals

    private func findNode(_ identifier: String) -> NSObject? {
        for window in stageWindows() {
            if let hit = Self.find(window, id: identifier) { return hit }
        }
        return nil
    }

    private static func turnRunLoop(_ seconds: TimeInterval = 0.05) {
        RunLoop.main.run(until: Date(timeIntervalSinceNow: seconds))
    }

    private static func string(_ obj: NSObject, _ selector: String) -> String {
        let sel = NSSelectorFromString(selector)
        guard obj.responds(to: sel), let result = obj.perform(sel) else { return "" }
        return (result.takeUnretainedValue() as? String) ?? ""
    }

    private static func children(_ obj: NSObject) -> [NSObject] {
        let sel = NSSelectorFromString("accessibilityChildren")
        guard obj.responds(to: sel), let result = obj.perform(sel) else { return [] }
        return (result.takeUnretainedValue() as? [NSObject]) ?? []
    }

    /// AX children plus, for AppKit views, their subviews. SwiftUI's node
    /// tree omits the content of some `NSViewRepresentable`s (observed: the
    /// TextKit streaming-markdown body — its action buttons appear, its
    /// text does not), while the real AX server the bash flows talk to
    /// walks the NSView hierarchy and sees it. Merging both, deduplicated,
    /// restores parity with what assistive clients actually read.
    private static func branches(_ obj: NSObject, visited: inout Set<ObjectIdentifier>) -> [NSObject] {
        var next: [NSObject] = []
        for child in children(obj) where visited.insert(ObjectIdentifier(child)).inserted {
            next.append(child)
        }
        if let view = obj as? NSView {
            for subview in view.subviews where visited.insert(ObjectIdentifier(subview)).inserted {
                next.append(subview)
            }
        }
        return next
    }

    private static func find(_ obj: NSObject, id target: String) -> NSObject? {
        var visited: Set<ObjectIdentifier> = [ObjectIdentifier(obj)]
        return find(obj, id: target, visited: &visited)
    }

    private static func find(
        _ obj: NSObject,
        id target: String,
        visited: inout Set<ObjectIdentifier>
    ) -> NSObject? {
        if string(obj, "accessibilityIdentifier") == target { return obj }
        for child in branches(obj, visited: &visited) {
            if let hit = find(child, id: target, visited: &visited) { return hit }
        }
        return nil
    }

    private static func walk(_ obj: NSObject, into out: inout [Node]) {
        var visited: Set<ObjectIdentifier> = [ObjectIdentifier(obj)]
        walk(obj, into: &out, visited: &visited)
    }

    private static func walk(
        _ obj: NSObject,
        into out: inout [Node],
        visited: inout Set<ObjectIdentifier>
    ) {
        let id = string(obj, "accessibilityIdentifier")
        let role = string(obj, "accessibilityRole")
        var text = ""
        let valueSel = NSSelectorFromString("accessibilityValue")
        if obj.responds(to: valueSel), let value = obj.perform(valueSel) {
            let raw = value.takeUnretainedValue()
            text = (raw as? String) ?? (raw as? NSAttributedString)?.string ?? ""
        }
        if text.isEmpty { text = string(obj, "accessibilityLabel") }
        if text.isEmpty { text = string(obj, "accessibilityTitle") }
        if !id.isEmpty || !text.isEmpty {
            out.append(Node(id: id, role: role, text: text))
        }
        for child in branches(obj, visited: &visited) {
            walk(child, into: &out, visited: &visited)
        }
    }
}
