import ApplicationServices
import CoreGraphics
import Security

/// Accessibility is the public macOS boundary that identifies the focused
/// window inside an otherwise-frontmost process. It does not expose a public
/// CGWindow number, so Computer Use binds its frame to exactly one CGWindow;
/// ambiguous same-frame windows fail closed.
enum MacOSComputerUseWindowIdentity {
    struct WindowRecord: Equatable {
        let identifier: String
        let ownerPID: pid_t
        let ownerName: String?
        let name: String?
        let layer: Int
        let frame: CGRect
        let alpha: Double
    }

    /// Returns the first visible window in Core Graphics front-to-back order
    /// whose bounds contain the global point. Conservatively treating any
    /// visible overlay as an occluder prevents a global click from reaching a
    /// window other than the one the user selected.
    static func topmostWindowIdentifier(at point: CGPoint) -> String? {
        guard let dictionaries = CGWindowListCopyWindowInfo(
            [.optionOnScreenOnly, .excludeDesktopElements],
            kCGNullWindowID
        ) as? [[CFString: Any]]
        else { return nil }
        let records = dictionaries.compactMap(windowRecord)
        let trustedDockPIDs = Set(records.compactMap { record -> pid_t? in
            guard record.ownerName == "Dock",
                  isTrustedDockProcess(record.ownerPID)
            else { return nil }
            return record.ownerPID
        })
        return topmostWindowIdentifier(
            at: point,
            records: records,
            activeDisplayFrames: activeDisplayFrames(),
            trustedDockPIDs: trustedDockPIDs
        )
    }

    static func topmostWindowIdentifier(
        at point: CGPoint,
        records: [WindowRecord],
        activeDisplayFrames: [CGRect],
        trustedDockPIDs: Set<pid_t>
    ) -> String? {
        for record in records {
            guard record.frame.contains(point), record.alpha > 0 else { continue }
            if isNonInteractiveDockBackdrop(
                record,
                activeDisplayFrames: activeDisplayFrames,
                trustedDockPIDs: trustedDockPIDs
            ) {
                continue
            }
            return record.identifier
        }
        return nil
    }

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

    static func targetsMatch(
        _ lhs: WorkflowInteractionTarget,
        _ rhs: WorkflowInteractionTarget
    ) -> Bool {
        guard lhs.bundleIdentifier == rhs.bundleIdentifier,
              lhs.processIdentifier == rhs.processIdentifier,
              lhs.processLaunchDate == rhs.processLaunchDate,
              lhs.windowIdentifier == rhs.windowIdentifier
        else { return false }
        return framesMatch(lhs.windowFrame.cgRect, rhs.windowFrame.cgRect)
    }

    private static func windowRecord(
        _ dictionary: [CFString: Any]
    ) -> WindowRecord? {
        guard let number = dictionary[kCGWindowNumber] as? NSNumber,
              let ownerPID = dictionary[kCGWindowOwnerPID] as? NSNumber,
              let bounds = dictionary[kCGWindowBounds] as? [String: NSNumber],
              let frame = CGRect(dictionaryRepresentation: bounds as CFDictionary),
              let layer = dictionary[kCGWindowLayer] as? NSNumber
        else { return nil }
        return WindowRecord(
            identifier: String(number.uint32Value),
            ownerPID: ownerPID.int32Value,
            ownerName: dictionary[kCGWindowOwnerName] as? String,
            name: dictionary[kCGWindowName] as? String,
            layer: layer.intValue,
            frame: frame,
            alpha: (dictionary[kCGWindowAlpha] as? NSNumber)?.doubleValue ?? 1
        )
    }

    private static func activeDisplayFrames() -> [CGRect] {
        var count: UInt32 = 0
        guard CGGetActiveDisplayList(0, nil, &count) == .success, count > 0 else {
            return []
        }
        var identifiers = [CGDirectDisplayID](repeating: 0, count: Int(count))
        guard CGGetActiveDisplayList(count, &identifiers, &count) == .success else {
            return []
        }
        return identifiers.prefix(Int(count)).map(CGDisplayBounds)
    }

    private static func isTrustedDockProcess(_ processIdentifier: pid_t) -> Bool {
        let attributes = [kSecGuestAttributePid: NSNumber(value: processIdentifier)]
            as CFDictionary
        var code: SecCode?
        guard SecCodeCopyGuestWithAttributes(nil, attributes, [], &code) == errSecSuccess,
              let code
        else { return false }

        var requirement: SecRequirement?
        guard SecRequirementCreateWithString(
            "anchor apple and identifier \"com.apple.dock\"" as CFString,
            [],
            &requirement
        ) == errSecSuccess,
            let requirement
        else { return false }
        return SecCodeCheckValidity(code, [], requirement) == errSecSuccess
    }

    private static func isNonInteractiveDockBackdrop(
        _ record: WindowRecord,
        activeDisplayFrames: [CGRect],
        trustedDockPIDs: Set<pid_t>
    ) -> Bool {
        guard trustedDockPIDs.contains(record.ownerPID),
              record.ownerName == "Dock",
              record.name == "Dock",
              record.layer == 20
        else { return false }
        return activeDisplayFrames.contains { framesMatch($0, record.frame) }
    }
}

private extension WorkflowWindowFrame {
    var cgRect: CGRect {
        CGRect(x: x, y: y, width: width, height: height)
    }
}
