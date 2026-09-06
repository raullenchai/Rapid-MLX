import ApplicationServices
import CoreGraphics

/// Accessibility is the public macOS boundary that identifies the focused
/// window inside an otherwise-frontmost process. It does not expose a public
/// CGWindow number, so Computer Use binds its frame to exactly one CGWindow;
/// ambiguous same-frame windows fail closed.
enum MacOSComputerUseWindowIdentity {
    static func focusedWindowFrame(processIdentifier: pid_t) -> CGRect? {
        let application = AXUIElementCreateApplication(processIdentifier)
        var focusedValue: CFTypeRef?
        guard AXUIElementCopyAttributeValue(
            application,
            kAXFocusedWindowAttribute as CFString,
            &focusedValue
        ) == .success,
            let focusedValue,
            CFGetTypeID(focusedValue) == AXUIElementGetTypeID()
        else { return nil }

        let focused = unsafeDowncast(focusedValue, to: AXUIElement.self)
        var positionValue: CFTypeRef?
        var sizeValue: CFTypeRef?
        guard AXUIElementCopyAttributeValue(
            focused,
            kAXPositionAttribute as CFString,
            &positionValue
        ) == .success,
            AXUIElementCopyAttributeValue(
                focused,
                kAXSizeAttribute as CFString,
                &sizeValue
            ) == .success,
            let positionValue,
            let sizeValue,
            CFGetTypeID(positionValue) == AXValueGetTypeID(),
            CFGetTypeID(sizeValue) == AXValueGetTypeID()
        else { return nil }

        var origin = CGPoint.zero
        var size = CGSize.zero
        guard AXValueGetValue(
            unsafeDowncast(positionValue, to: AXValue.self),
            .cgPoint,
            &origin
        ),
            AXValueGetValue(
                unsafeDowncast(sizeValue, to: AXValue.self),
                .cgSize,
                &size
            )
        else { return nil }
        let frame = CGRect(origin: origin, size: size)
        return frame.width > 0 && frame.height > 0 ? frame : nil
    }

    static func framesMatch(_ lhs: CGRect, _ rhs: CGRect) -> Bool {
        let tolerance = 0.5
        return abs(lhs.origin.x - rhs.origin.x) <= tolerance
            && abs(lhs.origin.y - rhs.origin.y) <= tolerance
            && abs(lhs.width - rhs.width) <= tolerance
            && abs(lhs.height - rhs.height) <= tolerance
    }
}
