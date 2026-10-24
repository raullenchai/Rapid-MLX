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
/// Privacy: the complete report redacts the home directory and username,
/// and runs back through ``LogScrubber``
/// so tokens / auth headers can't ride along, and the machine section
/// carries only the same non-identifying facts telemetry already
/// reports (chip, RAM, macOS) — never a username, hostname, or path.
enum DiagnosticsBundle {

    /// Build the report text from live diagnostics.
    @MainActor
    static func makeReport(
        server: ServerManager,
        homeDirectory: String = NSHomeDirectory(),
        username: String = NSUserName()
    ) -> String {
        let hw = MacHardware.detect()
        let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "unknown"
        let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "unknown"
        let os = ProcessInfo.processInfo.operatingSystemVersionString

        var out = ""
        func line(_ s: String) { out += s + "\n" }

        line("Rapid-MLX diagnostics")
        line("=====================")
        line("Generated: \(ISO8601DateFormatter().string(from: Date()))")
        line("")
        line("App")
        line("  version:  \(version) (build \(build))")
        line("  bundle:   \(Bundle.main.bundleIdentifier ?? "unknown")")
        line("")
        line("Machine")
        line("  chip:     \(hw.brandString)")
        line("  ram:      \(String(format: "%.1f", hw.physicalRAMGB)) GB")
        line("  macOS:    \(os)")
        line("")
        line("Server")
        line("  state:    \(describe(server.state))")
        line("  serving:  \(server.servingAlias ?? "—")")
        line("  binary:   \(binaryDescription(server.binaryPath))")
        line("")
        line("Recent log (scrubbed, last \(logTailCount) lines)")
        line("------------------------------------------------")
        let tail = server.logLines.suffix(logTailCount)
        if tail.isEmpty {
            line("  (no log output yet)")
        } else {
            for l in tail {
                // logLines are scrubbed at capture; scrub again so this
                // path is safe regardless of the capture site.
                line("  " + LogScrubber.scrub(l))
            }
        }
        return redact(out, homeDirectory: homeDirectory, username: username)
    }

    /// Present a save panel and write the report. Reveals the saved
    /// file in Finder on success. No-ops cleanly if the user cancels.
    @MainActor
    static func exportViaSavePanel(server: ServerManager) {
        let report = makeReport(server: server)
        let panel = NSSavePanel()
        panel.title = "Export Rapid-MLX Diagnostics"
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
                alert.messageText = "Couldn't save diagnostics"
                alert.informativeText = error.localizedDescription
                alert.alertStyle = .warning
                alert.runModal()
            }
        }
    }

    // MARK: - Helpers

    /// Scrub the complete export, including metadata, state, environment/log
    /// text and crash messages. Keep path suffixes useful for troubleshooting.
    /// Literal replacement also handles account names containing regex syntax.
    static func redact(_ report: String, homeDirectory: String, username: String) -> String {
        let home = homeDirectory.hasSuffix("/") ? String(homeDirectory.dropLast()) : homeDirectory
        func identityPattern(_ value: String) -> String {
            // Accept mixtures of literal and percent-encoded characters,
            // including encoded separators and lower-case hex digits.
            value.unicodeScalars.map { scalar in
                let literal = NSRegularExpression.escapedPattern(for: String(scalar))
                let encoded = [String(scalar), String(scalar).lowercased(), String(scalar).uppercased()]
                    .map { $0.utf8.map { String(format: "%%%02X", $0) }.joined() }
                    .joined(separator: "|")
                let escapedCharacter = #"|\\"# + literal
                return "(?:" + literal + "|(?i:" + encoded + ")" + escapedCharacter + ")"
            }.joined()
        }
        var patterns: [String] = []
        // Match delimiters in their literal and encoded forms alike.
        let delimiters = "/ \t\r\n\"'<>:,;)]}?!#&="
            .map { identityPattern(String($0)) }.joined(separator: "|")
        let ending = "(?:$|" + delimiters + ")"
        if !home.isEmpty && home != "/" {
            // A home is a complete path component, not a prefix of another
            // account's directory. A final period may terminate prose.
            let boundary = "(?=" + ending + "|" + identityPattern(".") + "(?=" + ending + "))"
            patterns.append(identityPattern(home) + boundary)
        }
        if !username.isEmpty {
            let account = identityPattern(username)
            // Account names of any length can be ordinary words ("a", "model").
            // Limit matching to identity fields and path/quoted components.
            let field = #"(\b(?i:USER|LOGNAME|USERNAME|account|serving|alias)\s*[=:]\s*["']?|\b(?i:user|account)\s+)"#
            patterns.append(field + account + #"(?![\p{L}\p{N}_])"#)
            let filenameBoundary = username.count > 2
                ? "(?:" + ending + "|" + identityPattern("-") + "|" + identityPattern("_") + ")"
                : ending
            patterns.append(#"(?:(?<=/)|(?<=%2f))"# + account + "(?=" + filenameBoundary + ")")
            patterns.append(#"(?<=[("'])"# + account + #"(?![\p{L}\p{N}_])"#)
        }
        // Replace in one pass so a short account name cannot rewrite the
        // placeholder inserted for a home path (or another identity variant).
        guard !patterns.isEmpty else { return LogScrubber.scrub(report) }
        let pattern = patterns.joined(separator: "|")
        let replacement = !username.isEmpty ? "$1<redacted>" : "<redacted>"
        let result = report.replacingOccurrences(of: pattern, with: replacement, options: [.regularExpression, .caseInsensitive])
        return LogScrubber.scrub(result)
    }

    static let logTailCount = 300

    private static func defaultFilename() -> String {
        let stamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        return "rapid-mlx-diagnostics-\(stamp).txt"
    }

    private static func describe(_ state: ServerState) -> String {
        switch state {
        case .idle: return "idle"
        case .starting(let a): return "starting (\(a))"
        case .ready(let a): return "ready (\(a))"
        case .stopped: return "stopped"
        case .missing: return "missing (no rapid-mlx binary found)"
        case .crashed(let a, let msg): return "crashed (\(a)): \(LogScrubber.scrub(msg))"
        }
    }

    /// Report only whether the sidecar binary was located, not its full
    /// path — the path can carry the username and install location.
    private static func binaryDescription(_ url: URL?) -> String {
        guard let url else { return "not found" }
        return "found (\(url.lastPathComponent))"
    }
}
