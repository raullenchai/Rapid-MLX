#!/usr/bin/env swift
// Minimal native Accessibility driver for deterministic Rapid GUI journeys.
// It deliberately exposes only semantic operations: dump, press, set-value,
// paste-file, and closing a named native window through its AXCloseButton.
import AppKit
import ApplicationServices
import Foundation

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(("rapid-ax: \(message)\n").data(using: .utf8)!)
    exit(1)
}

// `trust` answers the question every other command silently depends on: is
// THIS process allowed to read another process's accessibility tree?
//
// Without it, a missing Accessibility grant is indistinguishable from a
// product bug. `AXUIElementCopyAttributeValue` just fails, `dump` returns a
// tree containing nothing but the application root, and the caller times out
// on "main window did not appear" — which accuses the app of never opening a
// window when the truth is that we were never allowed to look. That is the
// same trap the window-list comment below describes, and it is the single
// most likely way an unattended (CI) run of the golden flows would misreport
// itself.
//
// `AXIsProcessTrusted()` is the gate the AX APIs actually consult, but on its
// own it is only the system's opinion about us. When a target pid is supplied
// the check also performs a real cross-process read, so a grant that exists on
// paper yet does not work in practice still fails HERE, naming the permission,
// instead of surfacing minutes later as a phantom UI regression.
if CommandLine.arguments.count >= 2, CommandLine.arguments[1] == "trust" {
    let trusted = AXIsProcessTrusted()
    var payload: [String: Any] = ["trusted": trusted]
    var readSucceeded = true

    // A LOCKED screen produces the exact symptom this command exists to
    // prevent being misread. Measured while writing this: with the screen
    // locked, Accessibility is still granted and the Dock's AX tree still
    // reads fine, but no application can present a window, so every golden
    // flow dies on "main window did not appear" — indistinguishable from the
    // app being broken. The permission check alone does NOT see it.
    //
    // Complementary signals, deliberately: reading another process's tree is
    // positive evidence that a GUI session exists at all; the lock bit is the
    // one negative signal that evidence cannot carry, because the Dock is
    // perfectly readable behind a lock screen.
    //
    // Only an explicit `true` fails. An unreadable session dictionary is
    // recorded and left to the cross-process read above to adjudicate, rather
    // than inventing a verdict from a failed read.
    let session = CGSessionCopyCurrentDictionary() as? [String: Any]
    let screenLocked = (session?["CGSSessionScreenIsLocked"] as? NSNumber)?.boolValue
    payload["session_readable"] = session != nil
    payload["screen_locked"] = screenLocked ?? false
    if let onConsole = (session?["kCGSSessionOnConsoleKey"] as? NSNumber)?.boolValue {
        payload["on_console"] = onConsole
    }

    if CommandLine.arguments.count >= 3 {
        guard let target = pid_t(CommandLine.arguments[2]) else {
            fail("trust: target must be a pid")
        }
        var value: CFTypeRef?
        let result = AXUIElementCopyAttributeValue(
            AXUIElementCreateApplication(target),
            kAXChildrenAttribute as CFString,
            &value
        )
        // `.noValue` / `.attributeUnsupported` mean the read itself WORKED and
        // the target simply publishes no children. Only an outright failure is
        // evidence that we were refused.
        readSucceeded =
            result == .success || result == .noValue || result == .attributeUnsupported
        payload["target_pid"] = Int(target)
        payload["target_read"] = readSucceeded
        payload["target_read_error"] = Int(result.rawValue)
    }

    payload["success"] = trusted && readSucceeded && !(screenLocked ?? false)
    let data = try! JSONSerialization.data(
        withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))

    if screenLocked == true {
        fail(
            "the screen is locked (CGSSessionScreenIsLocked). Accessibility is "
            + "fine and other processes' trees still read, but no app can "
            + "present a window while locked, so every flow would fail with "
            + "\"main window did not appear\" and look like an app regression. "
            + "Unlock the screen and re-run."
        )
    }
    if !trusted {
        fail(
            "this process is NOT trusted for Accessibility "
            + "(AXIsProcessTrusted() == false). Every AX read will fail, and the "
            + "golden flows would report a missing window instead of a missing "
            + "permission. Grant Accessibility to the controlling process "
            + "(System Settings > Privacy & Security > Accessibility)."
        )
    }
    if !readSucceeded {
        fail(
            "Accessibility is granted but reading another process's AX tree "
            + "still failed. The grant is not effective for this process tree."
        )
    }
    exit(0)
}

guard CommandLine.arguments.count >= 3,
      let pid = pid_t(CommandLine.arguments[2]) else {
    fail("usage: rapid-ax <dump|press|set-scroll-value|increment|decrement|set-value|paste-file|set-window-size|close-window|trust> <pid> [identifier-or-window-title] [value]")
}

let command = CommandLine.arguments[1]
let application = AXUIElementCreateApplication(pid)

// A semantic action should model an actual user interacting with the app.
// On unattended runners AXPress can return success for a background SwiftUI
// window without dispatching the Button closure. Bring the target app forward
// before reading its tree, then resolve the post-activation elements so the
// action never holds a reference across the activation/layout transition.
if command != "dump" {
    guard let running = NSRunningApplication(processIdentifier: pid) else {
        fail("target application is no longer running")
    }
    running.activate(options: [.activateAllWindows])
    usleep(100_000)
}

var visited = Set<AXUIElement>()
var records = [[String: Any]]()
var match: AXUIElement?
// A negative assertion over ui_elements is valid only if the descendant walk
// reached every child. Caps and AX read failures are observations of
// "unknown", not proof that an element is absent.
var elementWalkComplete = true
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
    guard depth <= 40, records.count < 12_000 else {
        elementWalkComplete = false
        return
    }
    guard visited.insert(element).inserted else { return }

    let identifier = string(element, kAXIdentifierAttribute as CFString)
    let role = string(element, kAXRoleAttribute as CFString)
    // Action commands only need one element. Building a complete 12k-node
    // dump after finding it leaves SwiftUI several seconds to replace the
    // backing accessibility object; AXPress then receives a stale reference
    // and fails with invalidUIElement/cannotComplete. Stop at the match. Dump
    // still walks the complete tree because negative assertions depend on it.
    if command == "set-scroll-value", match == nil,
       role == kAXScrollBarRole as String {
        match = element
        return
    }
    if command != "dump", match == nil, identifier == wanted {
        match = element
        return
    }
    var record: [String: Any] = ["depth": depth]
    if let identifier { record["identifier"] = identifier }
    if let role { record["role"] = role }
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
        var value: CFTypeRef?
        let result = AXUIElementCopyAttributeValue(
            element, kAXChildrenAttribute as CFString, &value)
        if result == .attributeUnsupported || result == .noValue {
            children = []
        } else if result == .success, let kids = value as? [AXUIElement] {
            children = kids
        } else {
            elementWalkComplete = false
            return
        }
    }
    for child in children {
        walk(child, depth: depth + 1)
        if command != "dump", match != nil { break }
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
elementWalkComplete = elementWalkComplete && windowListComplete

if command == "close-window" {
    guard let wanted else { fail("close-window requires a window title") }
    guard let window = windowElements.first(where: {
        string($0, kAXTitleAttribute as CFString) == wanted
    }) else {
        fail("window not found: \(wanted)")
    }
    guard let closeButton = attribute(window, kAXCloseButtonAttribute as CFString) else {
        fail("window has no AXCloseButton: \(wanted)")
    }
    let result = AXUIElementPerformAction(closeButton as! AXUIElement, kAXPressAction as CFString)
    guard result == .success else { fail("AXPress close window \(wanted) failed: \(result.rawValue)") }
    print("{\"success\":true,\"window\":\"\(wanted)\",\"action\":\"close-window\"}")
    exit(0)
}

if command == "set-window-size" {
    guard let wanted else { fail("set-window-size requires a window title") }
    guard CommandLine.arguments.count > 4 else {
        fail("set-window-size requires WIDTHxHEIGHT")
    }
    let parts = CommandLine.arguments[4].split(separator: "x", maxSplits: 1)
    guard parts.count == 2,
          let width = Double(parts[0]),
          let height = Double(parts[1]),
          width > 0, height > 0 else {
        fail("set-window-size requires positive WIDTHxHEIGHT")
    }
    guard let window = windowElements.first(where: {
        string($0, kAXTitleAttribute as CFString) == wanted
    }) else {
        fail("window not found: \(wanted)")
    }
    var requested = CGSize(width: width, height: height)
    guard let value = AXValueCreate(.cgSize, &requested) else {
        fail("could not encode requested window size")
    }
    let result = AXUIElementSetAttributeValue(
        window, kAXSizeAttribute as CFString, value
    )
    guard result == .success else {
        fail("setting window size failed: \(result.rawValue)")
    }
    usleep(300_000)
    guard let actual = size(window, kAXSizeAttribute as CFString) else {
        fail("window size could not be read after resize")
    }
    let payload: [String: Any] = [
        "success": true,
        "window": wanted,
        "requested": ["width": width, "height": height],
        "actual": ["width": actual.width, "height": actual.height],
    ]
    let data = try! JSONSerialization.data(
        withJSONObject: payload, options: [.prettyPrinted, .sortedKeys]
    )
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
    exit(0)
}

if command == "dump" {
    let payload: [String: Any] = [
        "success": true,
        "data": [
            "pid": pid,
            "ui_elements": records,
            "walk": [
                "complete": elementWalkComplete,
                "record_count": records.count
            ],
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
case "set-scroll-value":
    guard CommandLine.arguments.count > 3,
          let value = Double(CommandLine.arguments[3]),
          (0.0...1.0).contains(value)
    else { fail("set-scroll-value requires a normalized value from 0 through 1") }
    let result = AXUIElementSetAttributeValue(
        target, kAXValueAttribute as CFString, NSNumber(value: value))
    guard result == .success else {
        fail("setting the first visible scroll bar failed: \(result.rawValue)")
    }
    usleep(150_000)
case "increment":
    let result = AXUIElementPerformAction(target, kAXIncrementAction as CFString)
    guard result == .success else { fail("AXIncrement \(identifier) failed: \(result.rawValue)") }
case "decrement":
    let result = AXUIElementPerformAction(target, kAXDecrementAction as CFString)
    guard result == .success else { fail("AXDecrement \(identifier) failed: \(result.rawValue)") }
case "set-value":
    guard CommandLine.arguments.count > 4 else { fail("set-value requires a value") }
    let value = CommandLine.arguments[4] as CFString
    let focusResult = AXUIElementSetAttributeValue(target, kAXFocusedAttribute as CFString, kCFBooleanTrue)
    guard focusResult == .success else { fail("focus \(identifier) failed: \(focusResult.rawValue)") }
    let result = AXUIElementSetAttributeValue(target, kAXValueAttribute as CFString, value)
    guard result == .success else { fail("set value \(identifier) failed: \(result.rawValue)") }
case "paste-file":
    guard CommandLine.arguments.count > 4 else { fail("paste-file requires a path") }
    let url = URL(fileURLWithPath: CommandLine.arguments[4])
    guard FileManager.default.fileExists(atPath: url.path) else {
        fail("paste-file path does not exist: \(url.path)")
    }
    let pasteboard = NSPasteboard.general
    pasteboard.clearContents()
    guard pasteboard.writeObjects([url as NSURL]) else {
        fail("could not write file URL to the pasteboard")
    }
    guard let running = NSRunningApplication(processIdentifier: pid) else {
        fail("target application is no longer running")
    }
    running.activate(options: [.activateAllWindows])
    // Activating the app can replace its first responder. Make activation
    // real first, then focus the compose field immediately before posting the
    // shortcut so Command-V cannot land in this short-lived driver instead.
    usleep(150_000)
    let focusResult = AXUIElementSetAttributeValue(
        target, kAXFocusedAttribute as CFString, kCFBooleanTrue)
    guard focusResult == .success else {
        fail("focus \(identifier) failed: \(focusResult.rawValue)")
    }
    usleep(50_000)
    guard let down = CGEvent(keyboardEventSource: nil, virtualKey: 9, keyDown: true),
          let up = CGEvent(keyboardEventSource: nil, virtualKey: 9, keyDown: false)
    else { fail("could not create Command-V events") }
    down.flags = .maskCommand
    up.flags = .maskCommand
    down.post(tap: .cghidEventTap)
    up.post(tap: .cghidEventTap)
    usleep(150_000)
default:
    fail("unknown command: \(command)")
}

print("{\"success\":true,\"identifier\":\"\(identifier)\",\"action\":\"\(command)\"}")
