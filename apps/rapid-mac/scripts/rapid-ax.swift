#!/usr/bin/env swift
// Minimal native Accessibility driver for deterministic Rapid GUI journeys.
// It deliberately exposes only semantic operations: dump, press and set-value.
import ApplicationServices
import Foundation

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(("rapid-ax: \(message)\n").data(using: .utf8)!)
    exit(1)
}

guard CommandLine.arguments.count >= 3,
      let pid = pid_t(CommandLine.arguments[2]) else {
    fail("usage: rapid-ax <dump|press|set-value> <pid> [identifier] [value]")
}

let command = CommandLine.arguments[1]
let application = AXUIElementCreateApplication(pid)
var visited = Set<CFHashCode>()
var records = [[String: Any]]()
var match: AXUIElement?
// The window list is NOT a by-product of the tree walk, because every way the
// walk can come up short is silent: it skips a root child whose AXRole read
// failed, drops a title it could not read, and stops dead at the record cap.
// Each shortens the list, and no caller can tell a shortened list from a
// window that closed — which is exactly how a golden flow waiting for a window
// to DISAPPEAR takes a transient AX failure as proof that it did. So enumerate
// the root's children once, up front, and say plainly whether that enumeration
// can be trusted. `complete: false` is not an error; it means "ask again".
var windowTitles = [String]()
var windowListComplete = true
var windowElements = [AXUIElement]()
let wanted = CommandLine.arguments.count > 3 ? CommandLine.arguments[3] : nil

func attribute(_ element: AXUIElement, _ name: CFString) -> AnyObject? {
    var value: CFTypeRef?
    guard AXUIElementCopyAttributeValue(element, name, &value) == .success else { return nil }
    return value
}

func string(_ element: AXUIElement, _ name: CFString) -> String? {
    attribute(element, name) as? String
}

func jsonValue(_ value: AnyObject?) -> Any? {
    switch value {
    case let text as String: return text
    case let number as NSNumber: return number
    default: return nil
    }
}

func point(_ element: AXUIElement, _ name: CFString) -> CGPoint? {
    guard let wrapped = attribute(element, name), CFGetTypeID(wrapped) == AXValueGetTypeID() else { return nil }
    var result = CGPoint.zero
    guard AXValueGetValue(wrapped as! AXValue, .cgPoint, &result) else { return nil }
    return result
}

func size(_ element: AXUIElement, _ name: CFString) -> CGSize? {
    guard let wrapped = attribute(element, name), CFGetTypeID(wrapped) == AXValueGetTypeID() else { return nil }
    var result = CGSize.zero
    guard AXValueGetValue(wrapped as! AXValue, .cgSize, &result) else { return nil }
    return result
}

func walk(_ element: AXUIElement, depth: Int) {
    guard depth <= 40, records.count < 12_000 else { return }
    let identity = CFHash(element)
    guard visited.insert(identity).inserted else { return }

    let identifier = string(element, kAXIdentifierAttribute as CFString)
    var record: [String: Any] = ["depth": depth]
    if let identifier { record["identifier"] = identifier }
    if let role = string(element, kAXRoleAttribute as CFString) { record["role"] = role }
    if let subrole = string(element, kAXSubroleAttribute as CFString), !subrole.isEmpty {
        record["subrole"] = subrole
    }
    // Structural baselines diff enabled/disabled transitions, so the dump has
    // to carry the state even though the journeys themselves press by
    // identifier. AXEnabled is absent on containers; only record the boolean
    // when the element actually publishes it.
    if let enabled = attribute(element, kAXEnabledAttribute as CFString) as? NSNumber {
        record["enabled"] = enabled.boolValue
    }
    // Which of several equal-looking things is the CHOSEN one. Without it a
    // flow can see that the model cards exist and are enabled but not which
    // one the user picked, so "the wizard silently discarded your selection"
    // is invisible to every assertion the harness can make (#1653). Same rule
    // as AXEnabled: absent on most elements, recorded only when published.
    if let selected = attribute(element, kAXSelectedAttribute as CFString) as? NSNumber {
        record["selected"] = selected.boolValue
    }
    if let title = string(element, kAXTitleAttribute as CFString), !title.isEmpty { record["title"] = title }
    if let description = string(element, kAXDescriptionAttribute as CFString), !description.isEmpty { record["description"] = description }
    if let help = string(element, kAXHelpAttribute as CFString), !help.isEmpty { record["help"] = help }
    if let value = jsonValue(attribute(element, kAXValueAttribute as CFString)) { record["value"] = value }
    if let origin = point(element, kAXPositionAttribute as CFString),
       let extent = size(element, kAXSizeAttribute as CFString) {
        record["bounds"] = [
            "x": origin.x, "y": origin.y,
            "width": extent.width, "height": extent.height
        ]
    }
    records.append(record)

    if match == nil, identifier == wanted { match = element }
    // At depth 0 the children ARE the windows enumerated below, reused rather
    // than read again: a second AXChildren read can return a different set, and
    // then `ui_elements` and the `windows` list this dump vouches for would
    // disagree about which windows exist.
    //
    // That enumeration also drops the global menu bar, which the application
    // root owns as well. Traversing it captures unrelated macOS Recent Items in
    // artifacts and adds thousands of irrelevant nodes; golden flows only need
    // app windows, and sheets and popovers stay descendants of those.
    let children: [AXUIElement]
    if depth == 0 {
        children = windowElements
    } else {
        guard let kids = attribute(element, kAXChildrenAttribute as CFString) as? [AXUIElement] else { return }
        children = kids
    }
    for child in children {
        walk(child, depth: depth + 1)
    }
}

// Enumerate the windows before walking, so the walk can be filtered against
// the result. Every read that fails here marks the list incomplete instead of
// quietly shortening it; a window we cannot name is still a window we saw.
if let rootChildren = attribute(application, kAXChildrenAttribute as CFString) as? [AXUIElement] {
    for child in rootChildren {
        guard let role = string(child, kAXRoleAttribute as CFString) else {
            // We could not even establish whether this child is a window.
            windowListComplete = false
            continue
        }
        guard role == kAXWindowRole as String else { continue }
        // Recorded even when the title will not read: a window we cannot name
        // is still a window, and dropping it here would shorten the tree too.
        windowElements.append(child)
        guard let title = string(child, kAXTitleAttribute as CFString) else {
            windowListComplete = false
            continue
        }
        windowTitles.append(title)
    }
} else {
    windowListComplete = false
}

walk(application, depth: 0)

if command == "dump" {
    let payload: [String: Any] = [
        "success": true,
        "data": [
            "pid": pid,
            "ui_elements": records,
            // The authority for "is window X open?" — see the note above. Callers
            // must treat `complete: false` as "could not observe", never as an
            // answer in either direction.
            "windows": ["titles": windowTitles, "complete": windowListComplete]
        ]
    ]
    let data = try! JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
    exit(0)
}

guard let identifier = wanted, let target = match else {
    fail("identifier not found: \(wanted ?? "<missing>")")
}

switch command {
case "press":
    let result = AXUIElementPerformAction(target, kAXPressAction as CFString)
    guard result == .success else { fail("AXPress \(identifier) failed: \(result.rawValue)") }
case "set-value":
    guard CommandLine.arguments.count > 4 else { fail("set-value requires a value") }
    let value = CommandLine.arguments[4] as CFString
    let focusResult = AXUIElementSetAttributeValue(target, kAXFocusedAttribute as CFString, kCFBooleanTrue)
    guard focusResult == .success else { fail("focus \(identifier) failed: \(focusResult.rawValue)") }
    let result = AXUIElementSetAttributeValue(target, kAXValueAttribute as CFString, value)
    guard result == .success else { fail("set value \(identifier) failed: \(result.rawValue)") }
default:
    fail("unknown command: \(command)")
}

print("{\"success\":true,\"identifier\":\"\(identifier)\",\"action\":\"\(command)\"}")
