import Foundation

/// Closed, user-safe contract emitted by deterministic engine preflights.
/// Raw sidecar text is never carried beyond the parser.
struct SidecarStartupFailure: Equatable, Sendable {
    // Hyphens keep this protocol tag outside the RAPID_MLX_* env-var namespace.
    // The longest canonical marker remains 66 bytes, well below the 512-byte cap.
    static let markerPrefix = "RAPID-MLX-STARTUP-FAILURE:"

    enum Reason: String, Equatable, Sendable {
        case runtimeExtraMissing = "runtime_extra_missing"
        case runtimeDependencyMissing = "runtime_dependency_missing"
        case pythonVersionUnsupported = "python_version_unsupported"
        case runtimeIncompatible = "runtime_incompatible"
        case runtimeBroken = "runtime_broken"
        case modelNotFound = "model_not_found"
        case modelGated = "model_gated"
        case hubOffline = "hub_offline"
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
    let extra: Extra?

    var action: RecoveryAction { .openStartupLog }

    fileprivate static func parse(line: String) -> SidecarStartupFailure? {
        let fields = line.split(separator: " ", omittingEmptySubsequences: true)
        guard fields.count >= 2,
              fields[0] == Substring(markerPrefix),
              let reason = Reason(rawValue: String(fields[1]))
        else { return nil }

        switch reason {
        case .modelNotFound, .modelGated, .hubOffline:
            guard fields.count == 2,
                  line == "\(markerPrefix) \(reason.rawValue)"
            else { return nil }
            return SidecarStartupFailure(reason: reason, extra: nil)
        case .runtimeExtraMissing, .runtimeDependencyMissing,
             .pythonVersionUnsupported, .runtimeIncompatible, .runtimeBroken:
            guard fields.count == 3,
                  fields[2].hasPrefix("extra="),
                  let extra = Extra(rawValue: String(fields[2].dropFirst("extra=".count))),
                  line == "\(markerPrefix) \(reason.rawValue) extra=\(extra.rawValue)"
            else { return nil }
            return SidecarStartupFailure(reason: reason, extra: extra)
        }
    }
}

/// Per-child lifecycle gate. Production feeds only the owned sidecar's stderr,
/// then records health success under the same lock used to snapshot termination.
/// Model output, chat text, and anything observed after readiness are ineligible.
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
    private var discardingCurrentLine = false
    private var readyObserved = false

    struct TerminationSnapshot: Equatable, Sendable {
        let failure: SidecarStartupFailure?
        let readyObserved: Bool
    }

    func ingest(_ data: Data, source: Source) {
        guard source == .sidecarStderr, !data.isEmpty else { return }
        lock.lock()
        defer { lock.unlock() }
        guard !readyObserved else { return }

        for byte in data {
            if discardingCurrentLine {
                if byte == 0x0A {
                    discardingCurrentLine = false
                }
                continue
            }

            if byte == 0x0A {
                parsePendingLine()
                pending.removeAll(keepingCapacity: true)
                continue
            }

            pending.append(byte)
            if pending.count > 512 {
                pending.removeAll(keepingCapacity: true)
                discardingCurrentLine = true
            }
        }
    }

    /// Called directly from the URLSession completion before its async
    /// continuation is resumed. A 2xx transition seals capture atomically with
    /// recording readiness, so termination cannot snapshot an intermediate
    /// "marker accepted, not ready" state.
    @discardableResult
    func recordHealthResponse(statusCode: Int?) -> Bool {
        let succeeded = statusCode.map { (200..<300).contains($0) } ?? false
        guard succeeded else { return false }

        lock.lock()
        readyObserved = true
        pending.removeAll(keepingCapacity: false)
        storedFailure = nil
        lock.unlock()
        return true
    }

    func snapshotAtTermination() -> TerminationSnapshot {
        lock.lock()
        defer { lock.unlock() }
        return TerminationSnapshot(
            failure: readyObserved ? nil : storedFailure,
            readyObserved: readyObserved
        )
    }

    var failure: SidecarStartupFailure? {
        lock.lock()
        defer { lock.unlock() }
        return readyObserved ? nil : storedFailure
    }

    private func parsePendingLine() {
        var lineData = pending[...]
        if lineData.last == 0x0D {
            lineData = lineData.dropLast()
        }
        guard storedFailure == nil,
              let line = String(data: lineData, encoding: .utf8),
              let parsed = SidecarStartupFailure.parse(line: line)
        else { return }
        storedFailure = parsed
    }
}
