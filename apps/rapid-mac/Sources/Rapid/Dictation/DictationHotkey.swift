import AppKit
import ApplicationServices

/// Global hotkey for dictation, built on a `CGEventTap`.
///
/// Why not `RegisterEventHotKey`/`NSEvent.addGlobalMonitorForEvents`: Carbon's
/// hotkey API can only bind *modifier + key* combinations, and it cannot express
/// "the user tapped a modifier on its own". Dictation wants a bare modifier tap
/// so the gesture never competes with an app's own shortcuts, which leaves a
/// `flagsChanged` tap as the only mechanism.
///
/// Right Command is the only offered key. Left Command is deliberately absent:
/// it is pressed as part of ⌘C, ⌘V, ⌘Tab and friends dozens of times an hour, and
/// no amount of debouncing makes "tapped it alone" reliable enough to arm a
/// microphone with. The right-hand key is essentially unused by muscle memory.
@MainActor
final class DictationHotkey {
    /// Which bare modifier arms dictation.
    enum Trigger: String, CaseIterable, Identifiable, Sendable {
        case rightCommand
        case rightOption

        var id: String { rawValue }

        var label: String {
            switch self {
            case .rightCommand: return "Right ⌘"
            case .rightOption: return "Right ⌥"
            }
        }

        /// Virtual keycodes are side-specific; the modifier *flag* is not.
        var keyCode: Int64 {
            switch self {
            case .rightCommand: return 54   // kVK_RightCommand
            case .rightOption: return 61    // kVK_RightOption
            }
        }

        var flag: CGEventFlags {
            switch self {
            case .rightCommand: return .maskCommand
            case .rightOption: return .maskAlternate
            }
        }
    }

    /// A tap is only a tap if the modifier went down and back up quickly with no
    /// other key in between. Holding longer than this means the user is using it
    /// as a real modifier and we must stay out of the way.
    /// `nonisolated` because the event-tap callback reads it off the main actor.
    private nonisolated static let maxTapDuration: CFTimeInterval = 0.4

    /// Mutable state touched from the event-tap callback.
    ///
    /// `@unchecked Sendable` is sound here because every access happens on the
    /// single run-loop thread that services the tap: `CGEvent` callbacks for one
    /// tap are delivered serially, and `suspended`/`trigger` are only written
    /// from the main thread while the tap is stopped or between events (both are
    /// single-word stores, and a stale read costs at most one missed tap).
    private final class TapState: @unchecked Sendable {
        var trigger: Trigger = .rightCommand
        var downAt: CFTimeInterval?
        var sawOtherKey = false
        var suspended = false
        var onTap: (@Sendable () -> Void)?
    }

    private let state = TapState()
    private var tap: CFMachPort?
    private var source: CFRunLoopSource?

    private(set) var isRunning = false

    var trigger: Trigger {
        get { state.trigger }
        set { state.trigger = newValue }
    }

    /// Suspend while synthesising ⌘V so the injection cannot re-trigger us.
    var isSuspended: Bool {
        get { state.suspended }
        set { state.suspended = newValue }
    }

    /// Called on the main actor when a bare trigger tap is recognised.
    var onTap: (() -> Void)?

    // MARK: - Permission

    /// Whether the process may install a listening event tap.
    static var hasAccessibilityPermission: Bool { AXIsProcessTrusted() }

    /// Ask the system to show the Accessibility prompt. Returns the trust state
    /// *at call time* — macOS grants asynchronously, and the app must be
    /// re-launched or re-checked afterwards, so callers should poll rather than
    /// treat `false` as final.
    @discardableResult
    static func requestAccessibilityPermission() -> Bool {
        // The SDK exposes `kAXTrustedCheckOptionPrompt` as a mutable global,
        // which Swift 6 rejects as shared mutable state. Its value is a stable
        // documented constant, so spell it directly.
        let key = "AXTrustedCheckOptionPrompt" as CFString
        return AXIsProcessTrustedWithOptions([key: true] as CFDictionary)
    }

    /// Opens the exact Accessibility pane. The prompt above only appears once per
    /// app version, so returning users need a direct route.
    static func openAccessibilitySettings() {
        guard let url = URL(
            string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
        ) else { return }
        NSWorkspace.shared.open(url)
    }

    // MARK: - Lifecycle

    /// Installs the tap. Returns `false` when Accessibility permission is
    /// missing — `CGEvent.tapCreate` returns nil in that case rather than
    /// failing loudly.
    @discardableResult
    func start() -> Bool {
        guard !isRunning else { return true }
        guard Self.hasAccessibilityPermission else { return false }

        state.onTap = { [weak self] in
            Task { @MainActor in self?.onTap?() }
        }
        state.downAt = nil
        state.sawOtherKey = false

        let mask = (1 << CGEventType.flagsChanged.rawValue)
            | (1 << CGEventType.keyDown.rawValue)

        guard let tap = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .headInsertEventTap,
            // Listen-only: dictation must never swallow or delay a keystroke.
            options: .listenOnly,
            eventsOfInterest: CGEventMask(mask),
            callback: { _, type, event, refcon in
                guard let refcon else { return Unmanaged.passUnretained(event) }
                let state = Unmanaged<TapState>.fromOpaque(refcon).takeUnretainedValue()
                DictationHotkey.handle(type: type, event: event, state: state)
                return Unmanaged.passUnretained(event)
            },
            userInfo: Unmanaged.passUnretained(state).toOpaque()
        ) else {
            return false
        }

        let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
        CFRunLoopAddSource(CFRunLoopGetMain(), source, .commonModes)
        CGEvent.tapEnable(tap: tap, enable: true)

        self.tap = tap
        self.source = source
        isRunning = true
        return true
    }

    func stop() {
        if let tap {
            CGEvent.tapEnable(tap: tap, enable: false)
            CFMachPortInvalidate(tap)
        }
        if let source {
            CFRunLoopRemoveSource(CFRunLoopGetMain(), source, .commonModes)
        }
        tap = nil
        source = nil
        state.onTap = nil
        state.downAt = nil
        isRunning = false
    }

    /// The system disables a tap that takes too long to respond, and there is no
    /// notification when it does — callers re-arm defensively.
    func reEnableIfDisabled() {
        guard let tap, isRunning, !CGEvent.tapIsEnabled(tap: tap) else { return }
        CGEvent.tapEnable(tap: tap, enable: true)
    }

    // MARK: - Event handling

    /// Runs on the event-tap thread. Keep it allocation-free and fast: a slow
    /// callback gets the whole tap disabled by the system.
    private nonisolated static func handle(
        type: CGEventType,
        event: CGEvent,
        state: TapState
    ) {
        if type == .keyDown {
            // Any real keystroke while the modifier is held means it is being
            // used as a modifier (⌘C, ⌘Tab…), not tapped on its own.
            if state.downAt != nil { state.sawOtherKey = true }
            return
        }

        guard type == .flagsChanged else { return }
        guard event.getIntegerValueField(.keyboardEventKeycode) == state.trigger.keyCode else {
            // Another modifier changing while the trigger is held is still a
            // chord, even though it produces `flagsChanged` rather than the
            // `keyDown` event handled above. Without this, a quick Right ⌘ +
            // Shift gesture could arm the microphone when Right ⌘ was released.
            if state.downAt != nil { state.sawOtherKey = true }
            return
        }

        let isDown = event.flags.contains(state.trigger.flag)
        let now = CACurrentMediaTime()

        guard !isDown else {
            state.downAt = now
            state.sawOtherKey = false
            return
        }

        let heldFor = state.downAt.map { now - $0 } ?? .greatestFiniteMagnitude
        let wasCleanTap = !state.sawOtherKey && heldFor < DictationHotkey.maxTapDuration
        state.downAt = nil
        guard wasCleanTap, !state.suspended else { return }
        state.onTap?()
    }
}
