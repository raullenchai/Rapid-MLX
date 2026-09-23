import Foundation

/// Closed, user-safe contract emitted by deterministic engine preflights.
/// Raw sidecar text is never carried beyond the parser.
struct SidecarStartupFailure: Equatable, Sendable {
    static let markerPrefix = "RAPID_MLX_STARTUP_FAILURE:"

    enum Reason: String, Equatable, Sendable {
        case runtimeExtraMissing = "runtime_extra_missing"
        case runtimeDependencyMissing = "runtime_dependency_missing"
        case pythonVersionUnsupported = "python_version_unsupported"
        case runtimeIncompatible = "runtime_incompatible"
        case runtimeBroken = "runtime_broken"
    }

    enum Extra: String, Equatable, Sendable {
        case video
        case vision
        case audio
        case image

        var displayName: String {
            rawValue.capitalized
        }
    }

    enum RecoveryAction: Equatable, Sendable {
        case openStartupLog
    }

    let reason: Reason
    let extra: Extra

    var message: String {
        switch reason {
        case .runtimeExtraMissing:
            return "The installed engine doesn't include \(extra.displayName) support. Open Startup Log for installation details."
        case .runtimeDependencyMissing:
            return "\(extra.displayName) support is missing a required runtime dependency. Open Startup Log for setup details."
        case .pythonVersionUnsupported:
            return "The installed engine's Python version can't run \(extra.displayName) models. Open Startup Log for the required version."
        case .runtimeIncompatible:
            return "The installed \(extra.displayName) runtime isn't compatible with this engine. Open Startup Log for repair details."
        case .runtimeBroken:
            return "The installed \(extra.displayName) runtime couldn't load. Open Startup Log for repair details."
        }
    }

    var action: RecoveryAction { .openStartupLog }

    fileprivate static func parse(line: String) -> SidecarStartupFailure? {
        let fields = line.split(separator: " ", omittingEmptySubsequences: true)
        guard fields.count == 3,
              fields[0] == Substring(markerPrefix),
              let reason = Reason(rawValue: String(fields[1])),
              fields[2].hasPrefix("extra="),
              let extra = Extra(rawValue: String(fields[2].dropFirst("extra=".count)))
        else { return nil }
        guard line == "\(markerPrefix) \(reason.rawValue) extra=\(extra.rawValue)" else {
            return nil
        }
        return SidecarStartupFailure(reason: reason, extra: extra)
    }
}

/// Per-child collector. Production feeds only the owned sidecar's stderr and
/// seals it at readiness, making model output and chat text ineligible even if
/// they contain a byte-for-byte marker.
final class SidecarStartupFailureCapture: @unchecked Sendable {
    enum Source: Equatable, Sendable {
        case sidecarStderr
        case sidecarStdout
        case modelOutput
        case chatMessage
    }

    private let lock = NSLock()
    private var pending = Data()
    private var storedFailure: SidecarStartupFailure?
    private var accepting = true

    func ingest(_ data: Data, source: Source) {
        guard source == .sidecarStderr, !data.isEmpty else { return }
        lock.lock()
        defer { lock.unlock() }
        guard accepting else { return }
        pending.append(data)
        while let newline = pending.firstIndex(of: 0x0A) {
            var lineData = pending[..<newline]
            pending.removeSubrange(...newline)
            if lineData.last == 0x0D {
                lineData = lineData.dropLast()
            }
            guard storedFailure == nil,
                  let line = String(data: lineData, encoding: .utf8),
                  let parsed = SidecarStartupFailure.parse(line: line)
            else { continue }
            storedFailure = parsed
        }
        // The marker contract is a short single line. Bound an unterminated
        // fragment so arbitrary stderr cannot turn this collector into a log.
        if pending.count > 512 {
            pending.removeAll(keepingCapacity: true)
        }
    }

    func sealAtReadiness() {
        lock.lock()
        accepting = false
        pending.removeAll(keepingCapacity: false)
        lock.unlock()
    }

    var failure: SidecarStartupFailure? {
        lock.lock()
        defer { lock.unlock() }
        return storedFailure
    }
}
