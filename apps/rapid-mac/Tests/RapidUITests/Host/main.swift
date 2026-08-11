import AppKit

// Xcode requires a target application for an UI-testing bundle. The tests
// deliberately launch the already-built production bundle by identifier; this
// inert host exists only as XCTest's runner and never stands in for Rapid.
let app = NSApplication.shared
app.setActivationPolicy(.accessory)
app.terminate(nil)
