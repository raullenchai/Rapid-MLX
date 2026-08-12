#!/usr/bin/env swift
// Minimal native Accessibility driver for deterministic Rapid GUI journeys.
// It deliberately exposes only semantic operations — dump, press, set-value,
// and closing a named native window through its AXCloseButton — plus two
// CGEvent keyboard commands, `key` and `type`, for the handful of surfaces
// that publish no AX identifiers at all (the standard file picker).
import ApplicationServices
import CoreGraphics
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
    fail("usage: rapid-ax <dump|press|set-value|close-window|key|type|keypanel|typepanel|trust> <pid> [identifier-or-window-title|combo|text] [value]")
}

let command = CommandLine.arguments[1]

// ---- CGEvent keyboard injection --------------------------------------
//
// The standard file picker is a native NSOpenPanel: its controls publish no
// `kAXIdentifierAttribute`, so neither `press` nor `set-value` can reach it.
// `key` and `type` fall back to synthesizing keyboard events into the target
// process's window server session — the same channel a human's keystrokes
// take — which the panel understands and the AX-only commands cannot touch.
//
// That channel does NOT reach a *modal* panel. An NSOpenPanel opened with
// `runModal()` runs in its own nested event loop, and `CGEvent.postToPid`
// injects into the app's normal event stream, which a busy modal simply never
// processes — measured: Cmd+Shift+G posted to the pid never opens the panel's
// "Go to Folder" sheet. The panel only reacts to events posted to the *HID
// session tap* (`.cghidEventTap`), which arrive through the window server at
// whatever window is key. `keypanel`/`typepanel` are the session-targeted
// variants used for exactly this stage; they are safe because a modal panel
// is by construction the only thing that can receive input while it is up.
//
// The target pid is still required: it is what `postToPid` uses to route the
// events, and it keeps the commands honest about WHICH app is being driven
// (a golden flow names a pid, never "the frontmost app", so it cannot grab
// whatever window the operator happens to have focused).

// Where a synthesized keyboard event is delivered.
enum PostTarget {
    // Route through `CGEvent.postToPid` into the named app's ordinary event
    // stream. Right for every SwiftUI surface Rapid owns; wrong for a modal
    // panel, whose nested run loop never sees these.
    case pid
    // Route through `CGEvent.post(.cghidEventTap)` to the HID session tap, so
    // the event lands on whatever window is key. Reserved for native modal
    // panels, which postToPid cannot reach.
    case session
}

/// Post a keyboard event with the chosen target.
func post(_ event: CGEvent, _ target: PostTarget, _ pid: pid_t) {
    switch target {
    case .pid: event.postToPid(pid)
    case .session: event.post(tap: .cghidEventTap)
    }
}

func keyCodeFor(_ name: String) -> CGKeyCode? {
    // Physical keycodes only for the *control* keys — the keys that are the
    // same physical key regardless of keyboard layout (Return is Return, Esc
    // is Esc, an arrow is an arrow). These are safe to post as a keycode
    // because layout does not change what physical key they are.
    //
    // Letters, digits and punctuation are deliberately NOT here: a US-ANSI
    // keycode for `g` is a DIFFERENT key on Dvorak / AZERTY / Colemak, and a
    // menu key-equivalent (e.g. NSOpenPanel's Go to Folder = Cmd+Shift+G)
    // matches on the *character* produced after layout interpretation. Those
    // keys are handled by ``characterFor``/``postKeyChar`` via
    // ``keyboardSetUnicodeString``, exactly like ``type`` does, so the same
    // chord opens the same menu item on any host layout.
    let map: [String: CGKeyCode] = [
        "return": 36, "enter": 76, "tab": 48, "space": 49,
        "escape": 53, "delete": 51, "backspace": 51,
        "up": 126, "down": 125, "left": 123, "right": 124,
        "home": 115, "end": 119, "pageup": 116, "pagedown": 121,
        "cmd": 55, "command": 55, "shift": 56, "option": 58, "alt": 58,
        "control": 59, "ctrl": 59, "fn": 63,
    ]
    return map[name.lowercased()]
}

// Control-key names that are resolved to physical keycodes (layout-neutral).
// Anything else is a *character* key and is resolved by character.
let controlKeyNames: Set<String> = [
    "return", "enter", "tab", "space", "escape", "delete", "backspace",
    "up", "down", "left", "right", "home", "end", "pageup", "pagedown",
    "fn",
]

func isControlKey(_ name: String) -> Bool {
    controlKeyNames.contains(name.lowercased())
}

// The shifted glyph each US-ANSI digit / punctuation key produces. Used to
// derive what character a `cmd+shift+x`-style chord should emit so the target
// matches the menu key-equivalent by character, on any keyboard layout.
let shiftedGlyphs: [String: Character] = [
    "1": "!", "2": "@", "3": "#", "4": "$", "5": "%", "6": "^", "7": "&",
    "8": "*", "9": "(", "0": ")",
    "-": "_", "=": "+", "[": "{", "]": "}", ";": ":", "'": "\"",
    ",": "<", ".": ">", "/": "?", "\\": "|", "`": "~",
]

// Resolve a bare character key (letter / digit / punctuation) to the Unicode
// character it produces for a given shift state. `key` keeps combos
// lowercase, so a letter's upper/shifted form is derived here.
func characterFor(_ key: String, shifted: Bool) -> Character? {
    let k = key.lowercased()
    if k.count == 1, let c = k.first, c.isLetter {
        return shifted ? Character(String(c).uppercased()) : c
    }
    if k.count == 1, let c = k.first, c.isNumber {
        return shifted ? (shiftedGlyphs[k] ?? c) : c
    }
    if shifted, let glyph = shiftedGlyphs[k] {
        return glyph
    }
    if k.count == 1 {
        return k.first
    }
    return nil
}

func modifiersFrom(_ flags: Set<String>) -> CGEventFlags {
    var result: CGEventFlags = []
    if flags.contains("cmd") || flags.contains("command") { result.insert(.maskCommand) }
    if flags.contains("shift") { result.insert(.maskShift) }
    if flags.contains("option") || flags.contains("alt") { result.insert(.maskAlternate) }
    if flags.contains("control") || flags.contains("ctrl") { result.insert(.maskControl) }
    return result
}

func postKey(_ pid: pid_t, _ target: PostTarget, _ keyCode: CGKeyCode, _ modifier: CGEventFlags) {
    // Key down while holding the modifiers, then key up, each with the
    // modifiers still held so the panel sees a coherent chord.
    let down = CGEvent(keyboardEventSource: nil, virtualKey: keyCode, keyDown: true)
    down?.flags = modifier
    if let down { post(down, target, pid) }
    let up = CGEvent(keyboardEventSource: nil, virtualKey: keyCode, keyDown: false)
    up?.flags = modifier
    if let up { post(up, target, pid) }
    usleep(30_000)
}

func postCombination(_ pid: pid_t, _ target: PostTarget, _ combo: String) {
    // A `+`-joined combo: "cmd+shift+g". At least one bare key is required.
    let parts = combo.split(separator: "+").map { String($0).lowercased() }
    guard let last = parts.last else {
        fail("key: unrecognized key in combo '\(combo)'")
    }
    let modifiers = modifiersFrom(Set(parts.dropLast()))
    if isControlKey(last) {
        guard let keyCode = keyCodeFor(last) else {
            fail("key: unrecognized control key '\(last)' in combo '\(combo)'")
        }
        postKey(pid, target, keyCode, modifiers)
    } else {
        let shifted = modifiers.contains(.maskShift)
        // Which Unicode character to type depends on HOW the chord reaches the
        // target:
        //   * ``key`` (pid-targeted) goes through the app's ordinary event
        //     pipeline, which re-interprets the produced glyph — so post the
        //     SHIFTED character ("G" for cmd+shift+g), as a layout-safe
        //     stand-in for the physical key.
        //   * ``keypanel`` (session-tap) posts straight into a native modal's
        //     key-equivalent matching, which compares against the stored base
        //     character "g" WITH the shift modifier bit set. Posting the
        //     shifted "G" there never matches (measured: the Go to Folder
        //     sheet never opens), so post the BASE character while holding
        //     shift — exactly what a human's physical shift+g produces.
        let char = (target == .session)
            ? characterFor(last, shifted: false)
            : characterFor(last, shifted: shifted)
        guard let char else {
            fail("key: unrecognized key '\(last)' in combo '\(combo)'")
        }
        postKeyChar(pid, target, char, modifiers)
    }
}

func postKeyChar(_ pid: pid_t, _ target: PostTarget, _ char: Character, _ modifier: CGEventFlags) {
    // Character chord: attach the character via ``keyboardSetUnicodeString``
    // so the target sees that exact character regardless of the active layout,
    // while the modifiers (cmd/option/control, and shift for a forced-uppercase
    // combo) are still held on both the down and up events.
    let down = CGEvent(keyboardEventSource: nil, virtualKey: 0, keyDown: true)
    let up = CGEvent(keyboardEventSource: nil, virtualKey: 0, keyDown: false)
    let units = Array(String(char).utf16)
    units.withUnsafeBufferPointer { buf in
        down?.keyboardSetUnicodeString(stringLength: buf.count, unicodeString: buf.baseAddress)
        up?.keyboardSetUnicodeString(stringLength: buf.count, unicodeString: buf.baseAddress)
    }
    down?.flags = modifier
    up?.flags = modifier
    if let down { post(down, target, pid) }
    if let up { post(up, target, pid) }
    usleep(30_000)
}

func typeText(_ pid: pid_t, _ target: PostTarget, _ text: String) {
    // `type` types literal text, so every character is attached as an explicit
    // Unicode string to the event via ``keyboardSetUnicodeString``; the system
    // sends that exact text through regardless of the active layout, so any
    // Unicode character is supported and the fixture path can never be mangled
    // by a Dvorak / AZERTY / any non-US host keyboard. (The `key` command posts
    // physical keycodes only for layout-neutral control keys like Return and
    // Escape; its character chords also inject Unicode — see
    // ``characterFor``/``postKeyChar`` — so both commands are layout-safe.)
    for ch in text {
        let down = CGEvent(keyboardEventSource: nil, virtualKey: 0, keyDown: true)
        let up = CGEvent(keyboardEventSource: nil, virtualKey: 0, keyDown: false)
        let units = Array(String(ch).utf16)
        units.withUnsafeBufferPointer { buf in
            down?.keyboardSetUnicodeString(stringLength: buf.count, unicodeString: buf.baseAddress)
            up?.keyboardSetUnicodeString(stringLength: buf.count, unicodeString: buf.baseAddress)
        }
        if let down { post(down, target, pid) }
        if let up { post(up, target, pid) }
        usleep(15_000)
    }
}

if command == "key" {
    guard CommandLine.arguments.count > 3 else {
        fail("key requires a combo, e.g. cmd+shift+g or return")
    }
    postCombination(pid, .pid, CommandLine.arguments[3])
    print("{\"success\":true,\"action\":\"key\",\"combo\":\"\(CommandLine.arguments[3])\"}")
    exit(0)
}

if command == "type" {
    guard CommandLine.arguments.count > 3 else {
        fail("type requires text to type")
    }
    typeText(pid, .pid, CommandLine.arguments[3])
    print("{\"success\":true,\"action\":\"type\",\"text\":\"\(CommandLine.arguments[3])\"}")
    exit(0)
}

if command == "keypanel" {
    guard CommandLine.arguments.count > 3 else {
        fail("keypanel requires a combo, e.g. cmd+shift+g or return")
    }
    postCombination(pid, .session, CommandLine.arguments[3])
    print("{\"success\":true,\"action\":\"keypanel\",\"combo\":\"\(CommandLine.arguments[3])\"}")
    exit(0)
}

if command == "typepanel" {
    guard CommandLine.arguments.count > 3 else {
        fail("typepanel requires text to type")
    }
    typeText(pid, .session, CommandLine.arguments[3])
    print("{\"success\":true,\"action\":\"typepanel\",\"text\":\"\(CommandLine.arguments[3])\"}")
    exit(0)
}

let application = AXUIElementCreateApplication(pid)
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
