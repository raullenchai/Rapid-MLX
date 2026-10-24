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
        username: String = NSUserName(),
        fullName: String = NSFullUserName()
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
        return redact(out, homeDirectory: homeDirectory, username: username, fullName: fullName)
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

    /// Owner policy: redact home paths anywhere, including backup/escaped
    /// paths. Known account names of at least three characters are redacted
    /// as whole tokens, even when they are also ordinary words. Unknown names
    /// in free text cannot be identified reliably and remain a residual risk.
    static func redact(
        _ report: String, homeDirectory: String, username: String, fullName: String = ""
    ) -> String {
        func identityPattern(_ value: String) -> String {
            // Match literal, shell/JSON-escaped, and percent-encoded text,
            // including mixed encodings and mixed-case account paths.
            value.unicodeScalars.map { scalar in
                let literal = NSRegularExpression.escapedPattern(for: String(scalar))
                let encoded = [String(scalar), String(scalar).lowercased(), String(scalar).uppercased()]
                    .map { $0.utf8.map { String(format: "%%%02X", $0) }.joined() }
                    .joined(separator: "|")
                let jsonPattern = [String(scalar), String(scalar).lowercased(), String(scalar).uppercased()]
                    .map { value in
                        let escaped = value.utf16.map { String(format: "\\u%04X", $0) }.joined()
                        return NSRegularExpression.escapedPattern(for: escaped)
                    }.joined(separator: "|")
                return "(?:" + literal + "|(?i:" + encoded + ")|" + jsonPattern + #"|\\"# + literal + ")"
            }.joined()
        }
        func replace(_ text: String, pattern: String, with placeholder: String) -> String {
            text.replacingOccurrences(of: pattern, with: placeholder,
                                      options: [.regularExpression, .caseInsensitive])
        }

        let home = homeDirectory.hasSuffix("/") ? String(homeDirectory.dropLast()) : homeDirectory
        var paths: [String] = []
        if !home.isEmpty && home != "/" { paths.append(identityPattern(home) + #"(?![\p{L}\p{N}_.+-])"#) }
        // Also cover other accounts, embedded backup paths, and file:// URLs.
        // Encoded/escaped separators terminate the account component.
        let account = #"(?:\\(?!u002f)[^/\r\n]|%(?!2f|09|0a|0d|3f|23|26|3d)[0-9a-f]{2}|[^/\s\\%"'<>:,;)\]}?!#&=])+"#
        paths.append(identityPattern("/Users/") + account)
        // A literal ~/ path is already anonymous; ~account/ is not.
        paths.append(identityPattern("~") + account + "(?=" + identityPattern("/") + ")")
        var result = replace(report, pattern: paths.joined(separator: "|"), with: "~")

        let names = [username, fullName].filter { $0.count >= 3 }.sorted { $0.count > $1.count }
        if !names.isEmpty {
            let tokens = names.map {
                #"(?<![\p{L}\p{N}_])"# + identityPattern($0) + #"(?![\p{L}\p{N}_])"#
            }.joined(separator: "|")
            result = replace(result, pattern: tokens, with: "<redacted>")
        }
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
