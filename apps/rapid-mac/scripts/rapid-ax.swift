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
    guard let children = attribute(element, kAXChildrenAttribute as CFString) as? [AXUIElement] else { return }
    for child in children {
        // The application root also owns the global menu bar. Traversing it
        // captures unrelated macOS Recent Items in test artifacts and adds
        // thousands of irrelevant nodes. Golden flows only need app windows;
        // sheets and popovers remain descendants of those windows.
        if depth == 0, string(child, kAXRoleAttribute as CFString) != kAXWindowRole as String {
            continue
        }
        walk(child, depth: depth + 1)
    }
}

walk(application, depth: 0)

if command == "dump" {
    let payload: [String: Any] = [
        "success": true,
        "data": ["pid": pid, "ui_elements": records]
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
