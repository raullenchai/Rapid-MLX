// window-id.swift — print the CGWindowID of a process's main window.
//
// `screencapture -R<rect>` grabs whatever is on screen at that rectangle,
// which is the frontmost window — not necessarily the one under review. A
// screenshot harness that cannot bring its app to the front (no Accessibility
// permission) therefore has to capture BY WINDOW, and `screencapture -l`
// needs a CGWindowID.
//
// The usual way to resolve one is PyObjC, which the system python3 no longer
// ships. This is the same query in ~20 lines of Swift, compiled on demand by
// share-compute-shots.sh.
//
// Usage: window-id <pid>   → prints the id, or exits 1 if there is no window.
import CoreGraphics
import Foundation

guard CommandLine.arguments.count > 1,
      let pid = Int(CommandLine.arguments[1]) else {
    FileHandle.standardError.write(Data("usage: window-id <pid>\n".utf8))
    exit(2)
}

guard let windows = CGWindowListCopyWindowInfo(
    [.optionOnScreenOnly, .excludeDesktopElements],
    kCGNullWindowID
) as? [[String: Any]] else {
    exit(1)
}

// The app's largest layer-0 window is its document window; menu-bar extras and
// popovers sit on other layers or are much smaller.
let candidates = windows.compactMap { window -> (id: CGWindowID, area: CGFloat)? in
    guard window[kCGWindowOwnerPID as String] as? Int == pid,
          window[kCGWindowLayer as String] as? Int == 0,
          let id = window[kCGWindowNumber as String] as? CGWindowID,
          let bounds = window[kCGWindowBounds as String] as? [String: CGFloat],
          let width = bounds["Width"], let height = bounds["Height"],
          width > 400
    else { return nil }
    return (id, width * height)
}

guard let best = candidates.max(by: { $0.area < $1.area }) else { exit(1) }
print(best.id)
