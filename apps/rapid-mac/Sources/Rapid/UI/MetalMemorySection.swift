import SwiftUI

/// Poll only while the Performance panel is mounted. Reject in-flight results
/// from previous launches, models, ports, or credentials before publishing.
struct MetalMemorySection: View {
    @Environment(ServerManager.self) private var server
    @State private var sampledRuntime: MemoryOverviewSession?
    @State private var snapshot: MetalMemorySnapshot?
    @State private var system: MemoryProbe.Snapshot?
    @State private var history = MemoryOverviewHistory()

    private var runtime: MemoryOverviewSession {
        MemoryOverviewSession(sessionID: server.activeServerSessionID, port: server.activePort,
                              bearer: server.activeBearer, model: server.servingAlias)
    }

    var body: some View {
        let current = runtime
        let visible = sampledRuntime == current ? snapshot : nil
        let mac = sampledRuntime == current ? system : nil
        let samples = history.session == current ? history.samples : []
        VStack(alignment: .leading, spacing: RapidTheme.Space.xl) {
            macSection(mac)
            engineSection(visible, mac: mac)
            prefixCacheSection(visible?.prefixCache)
            requestsSection(visible)
            trendSection(samples)
        }
        .task(id: current) {
            snapshot = nil
            system = nil
            history.reset(for: current)
            sampledRuntime = current
            let client = MetalMemoryClient()
            let clock = ContinuousClock()
            while !Task.isCancelled {
                let deadline = clock.now.advanced(by: .seconds(3))
                let nextSystem = await SystemProbeSampler.sample { MemoryProbe.snapshot() }
                var nextMetal: MetalMemorySnapshot?
                if current.sessionID != nil, let bearer = current.bearer {
                    nextMetal = await client.fetch(port: current.port, bearer: bearer)
                }
                guard !Task.isCancelled, runtime == current else { return }
                system = nextSystem
                snapshot = nextMetal
                history.append(date: Date(), system: nextSystem, metal: nextMetal, session: current)
                do { try await clock.sleep(until: deadline) }
                catch { return }
            }
        }
    }

    private func macSection(_ mac: MemoryProbe.Snapshot?) -> some View {
        SettingsSection(String(localized: "This Mac")) {
            VStack(alignment: .leading, spacing: 10) {
                LabeledContent("Total RAM", value: MemoryOverviewFormat.bytes(mac?.totalBytes))
                    .accessibilityIdentifier("Settings.Performance.SystemMemory.Total")
                LabeledContent("Memory used", value: MemoryOverviewFormat.bytes(mac?.usedBytes))
                    .accessibilityIdentifier("Settings.Performance.SystemMemory.Used")
                MemoryUsageBar(
                    fractions: systemFractions(mac), colors: [.blue, .orange, .purple],
                    label: String(localized: "This Mac memory usage"), value: systemSummary(mac)
                )
                .accessibilityIdentifier("Settings.Performance.SystemMemory.Bar")
                memoryRow(String(localized: "App memory"), bytes: mac?.appBytes, color: .blue,
                          identifier: "Settings.Performance.SystemMemory.App")
                memoryRow(String(localized: "Wired memory"), bytes: mac?.wiredBytes, color: .orange,
                          identifier: "Settings.Performance.SystemMemory.Wired")
                memoryRow(String(localized: "Compressed memory"), bytes: mac?.compressedBytes, color: .purple,
                          identifier: "Settings.Performance.SystemMemory.Compressed")
                memoryRow(String(localized: "Not in use"), bytes: mac?.freeBytes, color: .secondary,
                          identifier: "Settings.Performance.SystemMemory.NotInUse")
                Text("Used = app + wired + compressed. Not in use = total − used, including reclaimable memory; it is not a count of physical free pages. All values are decimal GB; this is usage, not memory pressure. A dash means no current measurement.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .accessibilityIdentifier("Settings.Performance.SystemMemory")
    }

    private func engineSection(_ metal: MetalMemorySnapshot?, mac: MemoryProbe.Snapshot?) -> some View {
        let share = MemoryOverviewFormat.engineShare(activeGB: metal?.active, totalBytes: mac?.totalBytes)
        return SettingsSection(String(localized: "Inference engine (Metal)")) {
            VStack(alignment: .leading, spacing: 10) {
                LabeledContent("Active allocations", value: MetalMemorySnapshot.label(metal?.active))
                    .accessibilityIdentifier("Settings.Performance.MetalMemory.Active")
                LabeledContent("Allocator cache", value: MetalMemorySnapshot.label(metal?.cache))
                    .accessibilityIdentifier("Settings.Performance.MetalMemory.Cache")
                LabeledContent("Peak allocations", value: MetalMemorySnapshot.label(metal?.peak))
                    .accessibilityIdentifier("Settings.Performance.MetalMemory.Peak")
                LabeledContent("Metal active / total RAM", value: MemoryOverviewFormat.percent(share))
                    .accessibilityIdentifier("Settings.Performance.MetalMemory.Share")
                MemoryUsageBar(fractions: share.map { [min($0, 1)] }, colors: [.teal],
                               label: String(localized: "Metal active / total RAM"),
                               value: MemoryOverviewFormat.percent(share))
                    .accessibilityIdentifier("Settings.Performance.MetalMemory.Bar")
                Text("Live counters for the inference engine, in decimal GB. These are not separate totals for model weights or the prefix cache, and should not be added to system memory use. A dash means the engine has no current measurement.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .accessibilityIdentifier("Settings.Performance.MetalMemory")
    }

    private func prefixCacheSection(_ cache: PrefixCacheSnapshot?) -> some View {
        let value = PrefixCacheSnapshot.label(cache)
        return SettingsSection(String(localized: "Prefix cache")) {
            VStack(alignment: .leading, spacing: 10) {
                LabeledContent("Current / capacity", value: value)
                    .accessibilityElement(children: .ignore)
                    .accessibilityLabel(String(localized: "Reusable prompt cache budget"))
                    .accessibilityValue(Text(value))
                    .accessibilityIdentifier("Settings.Performance.PrefixCache.Usage")
                MemoryUsageBar(fractions: cache?.utilization.map { [min($0, 1)] }, colors: [.indigo],
                               label: String(localized: "Reusable prompt cache budget"), value: value)
                    .accessibilityIdentifier("Settings.Performance.PrefixCache.Bar")
                Text("Reusable prompt cache budget. This overlaps engine memory and is not added to Mac or Metal totals. A dash means this cache mode reports no current usage or capacity.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .accessibilityIdentifier("Settings.Performance.PrefixCache")
    }

    private func requestsSection(_ snapshot: MetalMemorySnapshot?) -> some View {
        SettingsSection(String(localized: "Requests")) {
            VStack(alignment: .leading, spacing: 10) {
                LabeledContent("Running", value: MemoryOverviewFormat.requestCount(snapshot?.numRunning))
                    .accessibilityElement(children: .ignore)
                    .accessibilityLabel(String(localized: "Running requests"))
                    .accessibilityValue(Text(MemoryOverviewFormat.requestCount(snapshot?.numRunning)))
                    .accessibilityIdentifier("Settings.Performance.Requests.Running")
                LabeledContent("Waiting", value: MemoryOverviewFormat.requestCount(snapshot?.numWaiting))
                    .accessibilityElement(children: .ignore)
                    .accessibilityLabel(String(localized: "Waiting requests"))
                    .accessibilityValue(Text(MemoryOverviewFormat.requestCount(snapshot?.numWaiting)))
                    .accessibilityIdentifier("Settings.Performance.Requests.Waiting")
            }
        }
        .accessibilityIdentifier("Settings.Performance.Requests")
    }

    private func trendSection(_ samples: [MemoryOverviewHistory.Sample]) -> some View {
        SettingsSection(String(localized: "Recent memory usage")) {
            VStack(alignment: .leading, spacing: 10) {
                Text("Up to 5 minutes, sampled every 3 seconds while this panel is visible. Gaps mean no measurement.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                MemoryTrendChart(samples: samples)
                LabeledContent("System used (solid)", value: MetalMemorySnapshot.label(samples.last?.systemUsedGB))
                    .foregroundStyle(.blue)
                    .accessibilityIdentifier("Settings.Performance.MemoryTrend.System")
                LabeledContent("Metal active (dashed)", value: MetalMemorySnapshot.label(samples.last?.metalActiveGB))
                    .foregroundStyle(.teal)
                    .accessibilityIdentifier("Settings.Performance.MemoryTrend.Metal")
            }
        }
        .accessibilityIdentifier("Settings.Performance.MemoryTrend")
    }

    private func memoryRow(_ title: String, bytes: UInt64?, color: Color, identifier: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Circle().fill(color).frame(width: 8, height: 8).accessibilityHidden(true)
            LabeledContent(title, value: MemoryOverviewFormat.bytes(bytes))
        }
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier(identifier)
    }

    private func systemFractions(_ mac: MemoryProbe.Snapshot?) -> [Double]? {
        guard let mac, mac.totalBytes > 0, let app = mac.appBytes,
              let wired = mac.wiredBytes, let compressed = mac.compressedBytes else { return nil }
        return [app, wired, compressed].map { Double($0) / Double(mac.totalBytes) }
    }

    private func systemSummary(_ mac: MemoryProbe.Snapshot?) -> String {
        guard let mac else { return "—" }
        return String(localized: "Total: \(MemoryOverviewFormat.bytes(mac.totalBytes)); used: \(MemoryOverviewFormat.bytes(mac.usedBytes)); app: \(MemoryOverviewFormat.bytes(mac.appBytes)); wired: \(MemoryOverviewFormat.bytes(mac.wiredBytes)); compressed: \(MemoryOverviewFormat.bytes(mac.compressedBytes)); not in use: \(MemoryOverviewFormat.bytes(mac.freeBytes))")
    }
}

private struct MemoryUsageBar: View {
    let fractions: [Double]?
    let colors: [Color]
    let label: String
    let value: String

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                Rectangle().fill(Color.secondary.opacity(0.15))
                if let fractions {
                    HStack(spacing: 0) {
                        ForEach(fractions.indices, id: \.self) { index in
                            Rectangle().fill(colors[index])
                                .frame(width: geometry.size.width * fractions[index])
                        }
                    }
                } else {
                    Text("—").frame(maxWidth: .infinity)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 4))
        }
        .frame(height: 18)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text(label))
        .accessibilityValue(Text(value))
    }
}

/// Shared decimal-GB axis. Solid/dashed strokes and textual legends distinguish
/// series without color. Never connect across missing readings or long pauses.
private struct MemoryTrendChart: View {
    let samples: [MemoryOverviewHistory.Sample]

    private var hasValues: Bool {
        samples.contains { $0.systemUsedGB != nil || $0.metalActiveGB != nil }
    }

    private var maximum: Double {
        max(1, samples.flatMap { [$0.systemUsedGB, $0.metalActiveGB].compactMap { $0 } }.max() ?? 1)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(MetalMemorySnapshot.label(hasValues ? maximum : nil)).font(.caption).foregroundStyle(.secondary)
            GeometryReader { geometry in
                ZStack {
                    Rectangle().fill(Color.secondary.opacity(0.06))
                    path(size: geometry.size, value: { $0.systemUsedGB })
                        .stroke(Color.blue, style: StrokeStyle(lineWidth: 2))
                    path(size: geometry.size, value: { $0.metalActiveGB })
                        .stroke(Color.teal, style: StrokeStyle(lineWidth: 2, dash: [5, 3]))
                    if !hasValues {
                        Text("—")
                    }
                }
            }
            .frame(height: 110)
            Text(MetalMemorySnapshot.label(hasValues ? 0 : nil))
                .font(.caption)
                .foregroundStyle(.secondary)
            HStack {
                Text("5 min ago")
                Spacer()
                Text("Now")
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(String(localized: "Recent memory usage"))
        .accessibilityValue(Text(accessibilitySummary))
        .accessibilityIdentifier("Settings.Performance.MemoryTrend.Chart")
    }

    private var accessibilitySummary: String {
        let system = summary(values: samples.map(\.systemUsedGB))
        let metal = summary(values: samples.map(\.metalActiveGB))
        return String(localized: "System used (solid): \(system). Metal active (dashed): \(metal).")
    }

    private func summary(values: [Double?]) -> String {
        let valid = values.compactMap { $0 }
        let latest = MetalMemorySnapshot.label(values.last.flatMap { $0 })
        let low = MetalMemorySnapshot.label(valid.min())
        let high = MetalMemorySnapshot.label(valid.max())
        return String(localized: "Latest: \(latest); low: \(low); high: \(high)")
    }

    private func path(size: CGSize, value: (MemoryOverviewHistory.Sample) -> Double?) -> Path {
        var path = Path()
        guard let end = samples.last?.date else { return path }
        let ceiling = maximum
        var previous: Date?
        for sample in samples {
            guard let gb = value(sample) else { previous = nil; continue }
            let x = size.width * max(0, 1 - end.timeIntervalSince(sample.date) / MemoryOverviewHistory.window)
            let point = CGPoint(x: x, y: size.height * (1 - gb / ceiling))
            if let previous, sample.date.timeIntervalSince(previous) <= 6 {
                path.addLine(to: point)
            } else {
                // A tiny mark makes a single first reading visible too.
                path.move(to: CGPoint(x: max(0, x - 1), y: point.y))
                path.addLine(to: point)
            }
            previous = sample.date
        }
        return path
    }
}
