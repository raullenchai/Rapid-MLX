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
let maxDepth = 40
let recordCap = 12_000
// Hash-bucketed, but membership is decided by CFEqual. A bare hash set treats
// two distinct elements that happen to collide as the same object and drops the
// second one's whole subtree — silently, and with `complete: true`, which is the
// exact failure this dump is supposed to have stopped being able to report.
var visited = [CFHashCode: [AXUIElement]]()
var records = [[String: Any]]()
var match: AXUIElement?
// The same contract as `windows.complete`, for the element array. A caller that
// asserts something is ABSENT is reading `ui_elements` as an inventory, and the
// walk has three silent ways to come up short of one: a child list it cannot
// read, the depth cap, and the record cap. Each removes a subtree while leaving
// `success: true`, so `length == 0` is satisfied by never having looked. Say so
// instead. `complete: false` is not an error; it means "this dump cannot answer
// a question about absence".
//
// The scope is the app's WINDOW forest, deliberately: the walk starts from the
// root's window children and never enters the global menu bar, which would drag
// unrelated macOS Recent Items into every artifact. `complete` therefore means
// "every window and descendant was observed", not "every AX element the process
// owns" — a menu-bar item is out of scope, not missing. It also cannot be an
// atomic snapshot: a window opened after its parent's AXChildren was read is
// simply not in this dump. Callers proving absence must have settled the UI
// first; completeness is a floor, not a substitute for waiting.
//
// The sharpest case: with the screen LOCKED, every read succeeds and every app
// reports zero windows. `complete: true` is then honest — the accessibility API
// really is showing us the whole of what this session has — and still useless
// for proving a control is gone, because everything is. Measured on a locked
// Mac: `records: 1`, `windows.titles: []`, both `complete: true`. No flag in
// this dump can separate that from an app that closed its windows; what does is
// asserting positively that the thing you expect IS there before concluding
// anything about what is not.
var walkComplete = true
var walkUnreadableChildren = 0
var walkLastChildrenError: AXError?
var walkUnreadableFields = 0
var walkUnobservableElements = 0
var walkLastFieldError: AXError?
var walkHitDepthCap = false
var walkHitRecordCap = false
var walkRootReasons = [String]()
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

// A leaf and an element we failed to interrogate are the same shape to the
// caller — both yield no children — but only one of them is an observation.
// Most elements simply do not publish AXChildren; `attributeUnsupported` and
// `noValue` are that, and nothing is missing from the dump because of them.
// Every other error means there may be a subtree here that this dump does not
// contain, and no assertion over `ui_elements` can rule anything out.
enum ChildrenRead {
    case children([AXUIElement])
    case leaf
    case unreadable(AXError)
}

func children(_ element: AXUIElement) -> ChildrenRead {
    var value: CFTypeRef?
    let status = AXUIElementCopyAttributeValue(element, kAXChildrenAttribute as CFString, &value)
    switch status {
    case .success:
        // Success with a payload we cannot use is still a subtree we did not
        // walk, so it counts against completeness rather than as a leaf.
        guard let kids = value as? [AXUIElement] else { return .unreadable(.failure) }
        return .children(kids)
    case .attributeUnsupported, .noValue:
        return .leaf
    default:
        return .unreadable(status)
    }
}

/// The four attributes the absence assertions actually search
/// (`identifier`, `value`, `title`, `description`). A read that FAILED on one
/// of them omits it from the record, and a filter testing that field then finds
/// nothing — the same "absent, or never looked?" ambiguity as a missing
/// subtree, one level down. The AXError is the only way to tell an attribute
/// the element does not publish from one we could not obtain.
enum SearchableRead {
    /// Nothing searchable here: either the element does not publish the
    /// attribute, or it published something that cannot spell an alias name
    /// (`AXValue` carries CFRange and CGPoint as well as numbers). Both are
    /// observations, not gaps.
    case absent
    case text(String)
    case number(NSNumber)
    case unreadable(AXError)
}

func searchable(_ element: AXUIElement, _ name: CFString) -> SearchableRead {
    var value: CFTypeRef?
    let status = AXUIElementCopyAttributeValue(element, name, &value)
    switch status {
    case .success:
        if let text = value as? String { return .text(text) }
        if let number = value as? NSNumber { return .number(number) }
        return .absent
    case .attributeUnsupported, .noValue:
        return .absent
    default:
        return .unreadable(status)
    }
}

/// Read a searchable attribute, noting the label when the read FAILED (as
/// opposed to the element simply not publishing one). `nil` means "nothing to
/// record" in either case; the caller decides what a failure costs.
func searchableText(
    _ element: AXUIElement, _ name: CFString, _ label: String, failed: inout [String]
) -> String? {
    switch searchable(element, name) {
    case .text(let text): return text
    case .absent, .number: return nil
    case .unreadable(let error):
        failed.append(label)
        walkLastFieldError = error
        return nil
    }
}

/// True the first time this exact element is seen. Equality is `CFEqual`, not
/// hash equality — see `visited`.
func visit(_ element: AXUIElement) -> Bool {
    let key = CFHash(element)
    let bucket = visited[key] ?? []
    if bucket.contains(where: { CFEqual($0, element) }) { return false }
    visited[key] = bucket + [element]
    return true
}

func walk(_ element: AXUIElement, depth: Int) {
    guard depth <= maxDepth else {
        walkHitDepthCap = true
        walkComplete = false
        return
    }
    guard records.count < recordCap else {
        walkHitRecordCap = true
        walkComplete = false
        return
    }
    guard visit(element) else { return }

    // Which of the SEARCHED attributes failed to read on this element, as
    // opposed to simply not being published. Collected per element because the
    // cost depends on what else the element carries.
    var failedFields = [String]()
    let identifier = searchableText(
        element, kAXIdentifierAttribute as CFString, "identifier", failed: &failedFields)
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
    if let title = searchableText(
        element, kAXTitleAttribute as CFString, "title", failed: &failedFields),
        !title.isEmpty
    {
        record["title"] = title
    }
    if let description = searchableText(
        element, kAXDescriptionAttribute as CFString, "description", failed: &failedFields),
        !description.isEmpty
    {
        record["description"] = description
    }
    // `help` is not one of the searched fields, so a failed read here costs
    // the dump nothing it claims to vouch for.
    if let help = string(element, kAXHelpAttribute as CFString), !help.isEmpty { record["help"] = help }
    switch searchable(element, kAXValueAttribute as CFString) {
    case .text(let text): record["value"] = text
    case .number(let number): record["value"] = number
    case .absent: break
    case .unreadable(let error):
        failedFields.append("value")
        walkLastFieldError = error
    }

    // A failed read costs the dump its completeness only when it leaves the
    // element with NOTHING searchable — that is the element that could be
    // hiding the string an assertion is looking for. One whose title would not
    // read but whose identifier did is still found by every filter that tests
    // identifiers, and condemning the whole dump for it makes the signal
    // unsatisfiable: measured on this app, five of seventy-seven dumps carry
    // one such read failure, in the same panels every run.
    if !failedFields.isEmpty {
        record["unreadable"] = failedFields
        walkUnreadableFields += 1
        let searchableTextPresent = ["identifier", "title", "description", "value"]
            .contains { record[$0] != nil }
        if !searchableTextPresent {
            walkUnobservableElements += 1
            walkComplete = false
        }
    }
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
    let descendants: [AXUIElement]
    if depth == 0 {
        descendants = windowElements
    } else {
        switch children(element) {
        case .children(let kids):
            descendants = kids
        case .leaf:
            return
        case .unreadable(let error):
            walkUnreadableChildren += 1
            walkLastChildrenError = error
            walkComplete = false
            return
        }
    }
    for child in descendants {
        walk(child, depth: depth + 1)
    }
}

// Enumerate the windows before walking, so the walk can be filtered against
// the result. Every read that fails here marks the list incomplete instead of
// quietly shortening it; a window we cannot name is still a window we saw.
// The walk starts from this list, so a gap here is a gap in `ui_elements` too —
// with one exception, called out below.
if let rootChildren = attribute(application, kAXChildrenAttribute as CFString) as? [AXUIElement] {
    for child in rootChildren {
        guard let role = string(child, kAXRoleAttribute as CFString) else {
            // We could not even establish whether this child is a window, so it
            // is missing from both the window list and the element tree.
            windowListComplete = false
            walkComplete = false
            walkRootReasons.append("a top-level child's AXRole could not be read, so its subtree was skipped")
            continue
        }
        guard role == kAXWindowRole as String else { continue }
        // Recorded even when the title will not read: a window we cannot name
        // is still a window, and dropping it here would shorten the tree too.
        // This is the exception — the element tree is whole, only the window
        // list is short, so `walkComplete` is deliberately left alone.
        windowElements.append(child)
        guard let title = string(child, kAXTitleAttribute as CFString) else {
            windowListComplete = false
            continue
        }
        windowTitles.append(title)
    }
} else {
    windowListComplete = false
    walkComplete = false
    walkRootReasons.append("the application's AXChildren could not be read, so no window was walked")
}

walk(application, depth: 0)

// Why the element array cannot be trusted as an inventory, in words, so the
// harness log says what went wrong instead of only that something did. Empty
// exactly when `walkComplete` is true.
var walkReasons = walkRootReasons
if walkUnreadableChildren > 0 {
    let code = walkLastChildrenError.map { String($0.rawValue) } ?? "unknown"
    walkReasons.append(
        "AXChildren was unreadable on \(walkUnreadableChildren) element(s) (last AXError \(code))")
}
if walkUnobservableElements > 0 {
    let code = walkLastFieldError.map { String($0.rawValue) } ?? "unknown"
    walkReasons.append(
        "\(walkUnobservableElements) element(s) carry no searchable text because every "
            + "attribute that could have held it failed to read (last AXError \(code)) — "
            + "each could be hiding anything")
}
if walkHitDepthCap { walkReasons.append("the depth cap of \(maxDepth) was reached") }
if walkHitRecordCap { walkReasons.append("the record cap of \(recordCap) was reached") }

if command == "dump" {
    let payload: [String: Any] = [
        "success": true,
        "data": [
            "pid": pid,
            "ui_elements": records,
            // Does `ui_elements` cover everything in scope? Only a caller
            // asking whether something is ABSENT needs this; a match is
            // self-proving, but "no match" from a clipped walk is not an
            // observation at all. `scope` is named rather than implied so
            // `complete` cannot be read as a claim about the menu bar, which
            // this walk deliberately never enters.
            "walk": [
                "complete": walkComplete,
                "scope": "window-forest",
                "reasons": walkReasons,
                // Informational: elements where a searched attribute would not
                // read but something else identified them anyway. Not a gap —
                // recorded so the artifact can say which, since `reasons` only
                // ever explains why `complete` is false.
                "elements_with_unreadable_fields": walkUnreadableFields
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
