import SwiftUI

/// Bottom-status memory pill. Shows used / total RAM with an
/// Activity-Monitor-style coloured dot so the user can read system
/// memory pressure at a glance — added in v0.4.40 after the user
/// flagged that desktop Macs running local models routinely hit
/// memory pressure and the UI gave them no signal.
///
/// Refresh cadence: 3 s. Fast enough that the pill responds during a
/// model load (the moment most users care), slow enough that the
/// number doesn't flicker on every minor allocation. A view-bound task
/// samples off-main and cancels when the footer unmounts.
///
/// #546: the numeric readout stays on the fixed `.font(.system(size:))`
/// rail (NOT `.scaledSystemFont`) on purpose. This is ambient live
/// telemetry in the fixed-height bottom status bar — the same platform
/// convention Activity Monitor's menu-bar readout and iStat follow —
/// where growing the glyph with Dynamic Type would clip the bar rather
/// than aid reading. Task-critical content (the chat transcript) scales;
/// this chrome does not. The Dynamic-Type source guard allowlists this
/// file for that reason.
struct MemoryPill: View {
    /// Manual refresh cadence. Surfaced so a future settings knob
    /// (or a stress-test fixture) can override the default without
    /// touching the view body.
    var refreshInterval: TimeInterval
    private let sample: @Sendable () -> MemoryProbe.Snapshot?
    private let snapshotDidPublish: @MainActor @Sendable (MemoryProbe.Snapshot?) -> Void
    @State private var snapshot: MemoryProbe.Snapshot?

    init(
        refreshInterval: TimeInterval = 3,
        sample: @escaping @Sendable () -> MemoryProbe.Snapshot? = {
            MemoryProbe.snapshot()
        },
        snapshotDidPublish: @escaping @MainActor @Sendable (MemoryProbe.Snapshot?) -> Void = { _ in }
    ) {
        self.refreshInterval = refreshInterval
        self.sample = sample
        self.snapshotDidPublish = snapshotDidPublish
        _snapshot = State(initialValue: sample())
    }

    var body: some View {
        content
            .task(id: refreshInterval) {
                await refreshLoop()
            }
    }

    @ViewBuilder
    private var content: some View {
        if let snap = snapshot {
            let pressure = MemoryProbe.Pressure.classify(ratio: snap.usedRatio)
            HStack(spacing: 5) {
                Circle()
                    .fill(color(for: pressure))
                    .frame(width: 6, height: 6)
                Text(MemoryProbe.formatLabel(snap))
                    .font(.rapid(size: 10, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            .help(tooltip(snap: snap, pressure: pressure))
            // Collapse the HStack into a single VoiceOver element so the
            // override label below replaces the children's text — without
            // this, VoiceOver reads the override AND the visible
            // "memory: 24G / 64G" Text, double-announcing the numbers.
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(accessibilityLabel(snap: snap, pressure: pressure))
        } else {
            // Probe failed — show a calm hyphen instead of zeros so
            // the user reads "we couldn't measure" rather than "you
            // have no memory."
            HStack(spacing: 5) {
                Circle()
                    .fill(Color.secondary.opacity(0.4))
                    .frame(width: 6, height: 6)
                Text("memory: —")
                    .font(.rapid(size: 10, design: .monospaced))
                    .foregroundStyle(.tertiary)
            }
            .help("Memory probe failed — host_statistics64 returned an error.")
        }
    }

    /// Activity-Monitor-style colour mapping. Static so the test
    /// suite can pin the threshold→colour contract directly.
    static func color(for pressure: MemoryProbe.Pressure) -> Color {
        switch pressure {
        case .normal:   return .green
        case .warning:  return .yellow
        case .critical: return .red
        }
    }

    private func color(for pressure: MemoryProbe.Pressure) -> Color {
        Self.color(for: pressure)
    }

    @MainActor
    private func refreshLoop() async {
        while !Task.isCancelled {
            let next = await SystemProbeSampler.sample(sample)
            guard !Task.isCancelled else { return }
            snapshot = next
            snapshotDidPublish(next)
            do {
                try await Task.sleep(for: .seconds(max(0.1, refreshInterval)))
            } catch {
                return
            }
        }
    }

    static func tooltip(snap: MemoryProbe.Snapshot, pressure: MemoryProbe.Pressure) -> String {
        let used = DownloadProgress.formatMemory(Int64(clamping: snap.usedBytes))
        let free = DownloadProgress.formatMemory(Int64(clamping: snap.freeBytes))
        let pct = Int((snap.usedRatio * 100).rounded())
        let header: String
        switch pressure {
        case .normal:   header = String(localized: "Memory: comfortable")
        case .warning:  header = String(localized: "Memory: tight")
        case .critical: header = String(localized: "Memory: critical — model loads may fail or trigger swap")
        }
        return String(localized: "\(header)\n\(used) in use (\(pct)%), \(free) free.")
    }

    private func tooltip(snap: MemoryProbe.Snapshot, pressure: MemoryProbe.Pressure) -> String {
        Self.tooltip(snap: snap, pressure: pressure)
    }

    static func accessibilityLabel(snap: MemoryProbe.Snapshot, pressure: MemoryProbe.Pressure) -> String {
        let used = DownloadProgress.formatMemory(Int64(clamping: snap.usedBytes))
        let total = MemoryProbe.formatTotal(snap.totalBytes)
        let state: String
        switch pressure {
        case .normal:   state = String(localized: "normal")
        case .warning:  state = String(localized: "tight")
        case .critical: state = String(localized: "critical")
        }
        return String(localized: "Memory \(state): \(used) used out of \(total)")
    }

    private func accessibilityLabel(snap: MemoryProbe.Snapshot, pressure: MemoryProbe.Pressure) -> String {
        Self.accessibilityLabel(snap: snap, pressure: pressure)
    }
}
