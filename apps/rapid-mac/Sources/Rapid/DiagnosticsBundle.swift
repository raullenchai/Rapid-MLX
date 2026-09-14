import AppKit
import Foundation

/// One-click "Export diagnostics…" support bundle.
///
/// A resident menu-bar app is installed by non-technical users who
/// won't read logs; when something goes wrong the fastest path to a
/// fix is a single button that hands us everything we need. This
/// assembles a plain-text report — app version, machine, sidecar
/// state, and the recent (already-scrubbed) log tail — and lets the
/// user save it to share.
///
/// Privacy: every free-text line runs back through ``LogScrubber``
/// so tokens / auth headers can't ride along, and the machine section
/// carries only the same non-identifying facts telemetry already
/// reports (chip, RAM, macOS) — never a username, hostname, or path.
enum DiagnosticsBundle {

    /// Build the report text from live diagnostics.
    @MainActor
    static func makeReport(server: ServerManager) -> String {
        let hw = MacHardware.detect()
        let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String
            ?? String(localized: "unknown")
        let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String
            ?? String(localized: "unknown")
        let os = ProcessInfo.processInfo.operatingSystemVersionString

        var out = ""
        func line(_ s: String) { out += s + "\n" }

        line(String(localized: "Rapid-MLX diagnostics"))
        line("=====================")
        line(String(localized: "Generated: \(ISO8601DateFormatter().string(from: Date()))"))
        line("")
        line(String(localized: "App"))
        line(String(localized: "  version:  \(version) (build \(build))"))
        line(String(localized: "  bundle:   \(Bundle.main.bundleIdentifier ?? "unknown")"))
        line("")
        line(String(localized: "Machine"))
        line(String(localized: "  chip:     \(hw.brandString)"))
        line(String(localized: "  ram:      \(String(format: "%.1f", hw.physicalRAMGB)) GB"))
        line(String(localized: "  macOS:    \(os)"))
        line("")
        line(String(localized: "Server"))
        line(String(localized: "  state:    \(describe(server.state))"))
        line(String(localized: "  serving:  \(server.servingAlias ?? "—")"))
        line(String(localized: "  binary:   \(binaryDescription(server.binaryPath))"))
        line("")
        line(String(localized: "Recent log (scrubbed, last \(logTailCount) lines)"))
        line("------------------------------------------------")
        let tail = server.logLines.suffix(logTailCount)
        if tail.isEmpty {
            line(String(localized: "  (no log output yet)"))
        } else {
            for l in tail {
                // logLines are scrubbed at capture; scrub again so this
                // path is safe regardless of the capture site.
                line("  " + LogScrubber.scrub(l))
            }
        }
        return out
    }

    /// Present a save panel and write the report. Reveals the saved
    /// file in Finder on success. No-ops cleanly if the user cancels.
    @MainActor
    static func exportViaSavePanel(server: ServerManager) {
        let report = makeReport(server: server)
        let panel = NSSavePanel()
        panel.title = String(localized: "Export Rapid-MLX Diagnostics")
        panel.nameFieldStringValue = defaultFilename()
        panel.allowedContentTypes = [.plainText]
        panel.isExtensionHidden = false
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            do {
                try report.data(using: .utf8)?.write(to: url)
                NSWorkspace.shared.activateFileViewerSelecting([url])
            } catch {
                let alert = NSAlert()
                alert.messageText = String(localized: "Couldn't save diagnostics")
                alert.informativeText = error.localizedDescription
                alert.alertStyle = .warning
                alert.runModal()
            }
        }
    }

    // MARK: - Helpers

    static let logTailCount = 300

    private static func defaultFilename() -> String {
        let stamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        return "rapid-mlx-diagnostics-\(stamp).txt"
    }

    private static func describe(_ state: ServerState) -> String {
        switch state {
        case .idle: return String(localized: "idle")
        case .starting(let a): return String(localized: "starting (\(a))")
        case .ready(let a): return String(localized: "ready (\(a))")
        case .stopped: return String(localized: "stopped")
        case .missing: return String(localized: "missing (no rapid-mlx binary found)")
        case .crashed(let a, let msg): return String(localized: "crashed (\(a)): \(LogScrubber.scrub(msg))")
        }
    }

    /// Report only whether the sidecar binary was located, not its full
    /// path — the path can carry the username and install location.
    private static func binaryDescription(_ url: URL?) -> String {
        guard let url else { return String(localized: "not found") }
        return String(localized: "found (\(url.lastPathComponent))")
    }
}
