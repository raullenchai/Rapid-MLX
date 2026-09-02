import Darwin
import Foundation
import SwiftUI

struct CommunityBenchmarkModel: Identifiable, Hashable {
    let entry: ModelEntry
    let task: ModelTask
    let protocolName: String
    let isFocus: Bool
    let estimatedMemoryGib: Int?
    let memoryFit: String

    var id: String { entry.alias }

    static let focusAliases: Set<String> = [
        "qwen3.8-27b-4bit", "qwen3.5-9b-4bit", "gemma-4-e4b-4bit",
        "flux2-klein-4b", "z-image-turbo", "qwen-image",
        "wan2.2-ti2v-5b-q8"
    ]
    static let registeredWanAliases: Set<String> = [
        "wan2.2-t2v-a14b-bf16", "wan2.2-ti2v-5b-bf16", "wan2.2-ti2v-5b-q8"
    ]

    static func models(
        from catalog: [ModelEntry],
        metadata: [String: CommunityBenchmarkCatalogModel] = [:]
    ) -> [Self] {
        catalog.compactMap { entry in
            let catalogModel = metadata[entry.alias]
            if !metadata.isEmpty, catalogModel == nil { return nil }
            let task: ModelTask?
            if entry.taskTypes.contains(.imageGeneration),
               entry.operationModes.contains(.textToImage) {
                task = .imageGeneration
            } else if entry.taskTypes.contains(.videoGeneration),
                      entry.operationModes.contains(.textToVideo),
                      registeredWanAliases.contains(entry.alias) {
                task = .videoGeneration
            } else if entry.taskTypes.contains(.textGeneration) {
                task = .textGeneration
            } else if entry.taskTypes.isEmpty {
                // A pre-atomic Desktop row has no capability evidence. Text
                // keeps its historical fallback, but diffusion/video rows
                // must not advertise a registered protocol based on `kind`
                // alone: the shared CLI may reject their operation/family.
                switch entry.kind {
                case .image, .video: task = nil
                case .chat: task = .textGeneration
                case .audio: task = nil
                }
            } else {
                task = nil
            }
            guard let task else { return nil }
            let protocolVersion = catalogModel?.protocolVersion ?? {
                switch task {
                case .textGeneration: 2
                case .imageGeneration, .videoGeneration: 1
                default: 1
                }
            }()
            let protocolName: String
            switch task {
            case .imageGeneration: protocolName = "Rapid Image Speed v\(protocolVersion)"
            case .videoGeneration: protocolName = "Rapid Video Speed v\(protocolVersion)"
            case .textGeneration: protocolName = "Rapid Community Speed v\(protocolVersion)"
            default: return nil
            }
            return Self(
                entry: entry,
                task: task,
                protocolName: protocolName,
                isFocus: catalogModel?.focus ?? focusAliases.contains(entry.alias),
                estimatedMemoryGib: catalogModel?.estimatedMemoryGib,
                memoryFit: catalogModel?.memoryFit ?? "unknown"
            )
        }
        .sorted {
            if $0.isFocus != $1.isFocus { return $0.isFocus }
            if $0.entry.cached != $1.entry.cached { return $0.entry.cached }
            return $0.entry.alias.localizedStandardCompare($1.entry.alias) == .orderedAscending
        }
    }

    static func reconciledSelection(current: String, models: [Self]) -> String {
        if models.contains(where: { $0.entry.alias == current }) { return current }
        return models.first?.entry.alias ?? ""
    }
}

struct CommunityBenchmarkCatalogModel: Decodable, Sendable {
    let alias: String
    let focus: Bool
    let estimatedMemoryGib: Int?
    let memoryFit: String
    let protocolVersion: Int?

    enum CodingKeys: String, CodingKey {
        case alias, focus
        case estimatedMemoryGib = "estimated_memory_gib"
        case memoryFit = "memory_fit"
        case protocolVersion = "protocol_version"
    }
}

private struct CommunityBenchmarkCatalogEnvelope: Decodable {
    let models: [CommunityBenchmarkCatalogModel]
}

private struct CommunityBenchmarkResults: Decodable {
    let runs: [CommunityBenchmarkResult]
}

private struct CommunityBenchmarkResult: Decodable, Identifiable {
    struct Workload: Decodable { let taskType: String
        enum CodingKeys: String, CodingKey { case taskType = "task_type" }
    }
    struct Outcome: Decodable { let status: String }
    struct Measurement: Decodable {
        let caseID: String
        let totalDurationMS: Double
        enum CodingKeys: String, CodingKey {
            case caseID = "case_id"
            case totalDurationMS = "total_duration_ms"
        }
    }
    struct Model: Decodable {
        struct Component: Decodable {
            struct Source: Decodable { let repoID: String?
                enum CodingKeys: String, CodingKey { case repoID = "repo_id" }
            }
            let source: Source
        }
        let components: [Component]
    }
    let id: String
    let completedAt: String
    let workload: Workload
    let outcome: Outcome
    let measurements: [Measurement]?
    let model: Model

    enum CodingKeys: String, CodingKey {
        case id = "run_id"
        case completedAt = "completed_at"
        case workload, outcome, measurements, model
    }

    var duration: String? {
        guard let values = measurements?.map(\.totalDurationMS), !values.isEmpty else { return nil }
        let average = values.reduce(0, +) / Double(values.count)
        let spansCases = Set(measurements?.map(\.caseID) ?? []).count > 1
        let value = average >= 1_000
            ? String(format: "%.1f s avg", average / 1_000)
            : String(format: "%.0f ms avg", average)
        return spansCases ? "\(value) across cases" : value
    }

    var repoID: String { model.components.first?.source.repoID ?? "Local model" }
}

final class BenchmarkProcessBox: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    private var child: ProcessGroupChild?

    func start(
        binary: URL,
        arguments: [String],
        standardOutput: Pipe,
        standardError: Pipe
    ) throws -> ProcessGroupChild {
        lock.lock()
        defer { lock.unlock() }
        if cancelled { throw CancellationError() }
        let spawned = try ProcessGroupChild.spawn(
            executableURL: binary,
            arguments: arguments,
            standardInput: .nullDevice,
            standardOutput: standardOutput,
            standardError: standardError
        )
        child = spawned
        return spawned
    }

    func cancel() {
        lock.lock()
        cancelled = true
        let runningChild = child
        lock.unlock()
        // Wake the detached waiter immediately. It owns TERM/KILL escalation
        // and final liveness confirmation, so this callback never blocks
        // AppKit while the server reservation remains held by `startRun`.
        runningChild?.signalProcessGroup(SIGTERM)
    }

    func waitForCompletion(_ child: ProcessGroupChild) -> pid_t? {
        defer { clearTrackedChild(child) }
        while child.isRunning {
            lock.lock()
            let shouldCancel = cancelled
            lock.unlock()
            if shouldCancel { return Self.terminateAndReap(child) }
            // Also drives ProcessGroupChild's non-blocking waitpid fallback
            // when its dispatch exit source is delayed on a saturated host.
            _ = child.isProcessGroupAlive
            Thread.sleep(forTimeInterval: 0.01)
        }
        // A crashed/cancelled CLI can exit before one of its serve descendants.
        // Never return control (and release the memory reservation) with any
        // member of the benchmark process group still alive.
        if child.isProcessGroupAlive {
            return Self.terminateAndReap(child)
        }
        return nil
    }

    private func clearTrackedChild(_ completedChild: ProcessGroupChild) {
        lock.lock()
        if child === completedChild { child = nil }
        lock.unlock()
    }

    internal var _testHasTrackedChild: Bool {
        lock.lock()
        defer { lock.unlock() }
        return child != nil
    }

    private static func terminateAndReap(_ child: ProcessGroupChild) -> pid_t? {
        let exited = boundedTermination(
            isAlive: { child.isProcessGroupAlive },
            signal: { child.signalProcessGroup($0) },
            termGrace: 2,
            killGrace: 1
        )
        return exited ? nil : child.processGroupID
    }

    static func boundedTermination(
        isAlive: () -> Bool,
        signal: (Int32) -> Void,
        termGrace: TimeInterval,
        killGrace: TimeInterval,
        now: () -> Date = Date.init,
        sleep: (TimeInterval) -> Void = Thread.sleep(forTimeInterval:)
    ) -> Bool {
        signal(SIGTERM)
        let termDeadline = now().addingTimeInterval(termGrace)
        while isAlive(), now() < termDeadline { sleep(0.01) }
        if isAlive() {
            signal(SIGKILL)
            let killDeadline = now().addingTimeInterval(killGrace)
            while isAlive(), now() < killDeadline { sleep(0.01) }
        }
        return !isAlive()
    }
}

enum CommunityBenchmarkCommand {
    struct Failure: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    private enum RunOutcome {
        case output(Data)
        case deferredReap(pid_t)
    }

    private struct PipeCapture: Sendable {
        let data: Data
        let truncated: Bool
    }

    private static let maxStdoutBytes = 8 * 1_024 * 1_024
    private static let maxStderrBytes = 256 * 1_024
    private static let pipeChunkBytes = 64 * 1_024

    static func benchmarkRunArguments(alias: String) -> [String] {
        [
            "benchmark", "run", alias, "--json",
            "--inherit-process-group",
        ]
    }

    static func benchmarkResultsArguments(limit: Int = 8) -> [String] {
        ["benchmark", "results", "--limit", String(limit), "--json"]
    }

    @MainActor
    static func run(
        binary: URL,
        arguments: [String],
        onDeferredReap: ((pid_t) -> Void)? = nil
    ) async throws -> Data {
        let box = BenchmarkProcessBox()
        return try await withTaskCancellationHandler {
            let outcome: RunOutcome
            do {
                outcome = try await Task.detached(priority: .userInitiated) {
                    let stdout = Pipe()
                    let stderr = Pipe()
                    let child = try box.start(
                        binary: binary,
                        arguments: arguments,
                        standardOutput: stdout,
                        standardError: stderr
                    )
                    let outputTask = Task.detached {
                        readBoundedPipe(
                            stdout.fileHandleForReading,
                            maxBytes: maxStdoutBytes,
                            retainTail: false
                        )
                    }
                    let errorTask = Task.detached {
                        readBoundedPipe(
                            stderr.fileHandleForReading,
                            maxBytes: maxStderrBytes,
                            retainTail: true
                        )
                    }
                    // The child owns duplicated write descriptors after spawn.
                    // Drop the parent's copies so both readers observe EOF when
                    // the process group exits.
                    try? stdout.fileHandleForWriting.close()
                    try? stderr.fileHandleForWriting.close()
                    defer {
                        // A read error on either stream must not strand the
                        // sibling detached reader or its descriptor. Closing
                        // first wakes a blocking read; cancellation then
                        // prevents any remaining detached work from escaping
                        // this command invocation.
                        try? stdout.fileHandleForReading.close()
                        try? stderr.fileHandleForReading.close()
                        outputTask.cancel()
                        errorTask.cancel()
                    }
                    if let processGroupID = box.waitForCompletion(child) {
                        return RunOutcome.deferredReap(processGroupID)
                    }
                    let output = await outputTask.value
                    let errorCapture = await errorTask.value
                    guard child.terminationStatus == 0 else {
                        let detail = String(data: errorCapture.data, encoding: .utf8)?
                            .trimmingCharacters(in: .whitespacesAndNewlines)
                        let message = detail.flatMap { $0.isEmpty ? nil : $0 }
                            ?? "Benchmark exited with code \(child.terminationStatus)."
                        throw Failure(message: message)
                    }
                    guard !output.truncated else {
                        throw Failure(
                            message: "Benchmark output exceeded the 8 MiB safety limit."
                        )
                    }
                    return RunOutcome.output(output.data)
                }.value
            } catch {
                try Task.checkCancellation()
                throw error
            }
            guard case let .output(data) = outcome else {
                if case let .deferredReap(processGroupID) = outcome {
                    if let onDeferredReap {
                        onDeferredReap(processGroupID)
                    } else {
                        ProcessGroupChild.reapProcessGroupInBackground(
                            processGroupID: processGroupID
                        )
                    }
                }
                throw CancellationError()
            }
            try Task.checkCancellation()
            return data
        } onCancel: {
            box.cancel()
        }
    }

    private static func readBoundedPipe(
        _ handle: FileHandle,
        maxBytes: Int,
        retainTail: Bool
    ) -> PipeCapture {
        var data = Data()
        var truncated = false
        while true {
            let chunk: Data?
            do {
                chunk = try handle.read(upToCount: pipeChunkBytes)
            } catch {
                break
            }
            guard let chunk, !chunk.isEmpty else { break }
            if chunk.count >= maxBytes {
                truncated = truncated || !data.isEmpty || chunk.count > maxBytes
                data = retainTail ? Data(chunk.suffix(maxBytes)) : Data(chunk.prefix(maxBytes))
                continue
            }
            let overflow = data.count + chunk.count - maxBytes
            if overflow > 0 {
                truncated = true
                if retainTail {
                    data.removeFirst(overflow)
                    data.append(chunk)
                } else {
                    data.append(chunk.prefix(maxBytes - data.count))
                }
            } else if retainTail || data.count < maxBytes {
                data.append(chunk)
            }
        }
        return PipeCapture(data: data, truncated: truncated)
    }

    internal static func _testReadBoundedPipe(
        _ handle: FileHandle,
        maxBytes: Int,
        retainTail: Bool
    ) -> (data: Data, truncated: Bool) {
        let capture = readBoundedPipe(
            handle, maxBytes: maxBytes, retainTail: retainTail
        )
        return (capture.data, capture.truncated)
    }
}

struct CommunityBenchmarkView: View {
    let catalog: [ModelEntry]
    let binary: URL?
    let prepareServer: () async throws -> UUID
    let releaseServer: (UUID) -> Void
    let retainServerDuringDeferredReap: (pid_t) -> Void

    @State private var selectedAlias = ""
    @State private var results: [CommunityBenchmarkResult] = []
    @State private var isRunning = false
    @State private var errorMessage: String?
    @State private var runTask: Task<Void, Never>?
    @State private var benchmarkMetadata: [String: CommunityBenchmarkCatalogModel] = [:]
    @State private var benchmarkCLIAvailable = false

    private var models: [CommunityBenchmarkModel] {
        CommunityBenchmarkModel.models(from: catalog, metadata: benchmarkMetadata)
    }

    private var selected: CommunityBenchmarkModel? {
        models.first { $0.entry.alias == selectedAlias }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                header
                setupCard
                recentResults
            }
            .frame(maxWidth: 760, alignment: .leading)
            .padding(32)
            .frame(maxWidth: .infinity, alignment: .top)
        }
        .background(RapidTheme.surfaceCanvas)
        .task {
            if selectedAlias.isEmpty { selectedAlias = models.first?.entry.alias ?? "" }
            await refreshBenchmarkCatalog()
            await refreshResults()
        }
        .onDisappear {
            // `runTask` is intentionally unstructured so the button owns it;
            // navigation must explicitly cancel it before the only Stop
            // control disappears. ServerManager keeps the lease until the
            // subprocess tree has actually been reaped.
            runTask?.cancel()
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Community Benchmark")
                .font(.system(size: 28, weight: .semibold))
            Text("Measure any supported model on this Mac. Results stay local unless you choose to share them later.")
                .foregroundStyle(.secondary)
        }
    }

    private var setupCard: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Run a benchmark").font(.headline)
            Picker("Model", selection: $selectedAlias) {
                ForEach(models) { model in
                    Text("\(model.isFocus ? "★ " : "")\(model.entry.alias)")
                        .tag(model.entry.alias)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityIdentifier("CommunityBenchmark.ModelPicker")

            if let selected {
                HStack(spacing: 10) {
                    Label(selected.protocolName, systemImage: "gauge.with.dots.needle.50percent")
                    if selected.entry.cached {
                        Text("Downloaded").foregroundStyle(.green)
                    } else {
                        Text("Download required").foregroundStyle(.secondary)
                    }
                    if let memory = selected.estimatedMemoryGib {
                        Text(memoryCopy(memory, fit: selected.memoryFit))
                            .foregroundStyle(selected.memoryFit == "does_not_fit" ? .orange : .secondary)
                    }
                }
                .font(.callout)
                Text(protocolDescription(selected.task))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if let errorMessage {
                Text(errorMessage).font(.callout).foregroundStyle(.red)
            }

            HStack {
                Button(isRunning ? "Stop" : "Run locally") {
                    isRunning ? runTask?.cancel() : startRun()
                }
                .buttonStyle(.borderedProminent)
                .accessibilityIdentifier("CommunityBenchmark.RunOrStop")
                .disabled(
                    !isRunning
                        && (selected == nil || binary == nil || !benchmarkCLIAvailable)
                )
                if isRunning {
                    ProgressView().controlSize(.small)
                    Text("The active server will stop while this model is measured.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
        }
        .padding(20)
        .background(RapidTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 14))
    }

    private var recentResults: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Recent local results").font(.headline)
            if results.isEmpty {
                Text("No benchmarks yet. Your first result will appear here.")
                    .foregroundStyle(.secondary)
                    .padding(.vertical, 12)
            } else {
                ForEach(results.prefix(8)) { result in
                    HStack {
                        VStack(alignment: .leading, spacing: 3) {
                            Text(alias(for: result.repoID)).fontWeight(.medium)
                            Text("\(result.workload.taskType.replacingOccurrences(of: "_", with: " ")) · \(result.completedAt)")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        VStack(alignment: .trailing, spacing: 3) {
                            Text(result.duration ?? result.outcome.status.capitalized)
                            Text("Local").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    .padding(.vertical, 8)
                    Divider()
                }
            }
        }
    }

    private func protocolDescription(_ task: ModelTask) -> String {
        switch task {
        case .imageGeneration: return "1 warmup + 1 measured 1024×1024 render · fixed prompt, seed and 20 steps"
        case .videoGeneration: return "1 measured 832×480, 81-frame render · fixed prompt and seed"
        default: return "Two fixed token workloads · 1 warmup + 5 measured rounds each · concurrency 1"
        }
    }

    private func alias(for repoID: String) -> String {
        catalog.first { $0.hfRepo == repoID }?.alias ?? repoID
    }

    private func memoryCopy(_ memory: Int, fit: String) -> String {
        fit == "does_not_fit" ? "Needs about \(memory) GB" : "About \(memory) GB"
    }

    private func startRun() {
        guard benchmarkCLIAvailable, let selected, let binary else { return }
        errorMessage = nil
        isRunning = true
        runTask = Task {
            var acquiredReservation = false
            do {
                let reservation = try await prepareServer()
                acquiredReservation = true
                defer { releaseServer(reservation) }
                try Task.checkCancellation()
                _ = try await CommunityBenchmarkCommand.run(
                    binary: binary,
                    arguments: CommunityBenchmarkCommand.benchmarkRunArguments(
                        alias: selected.entry.alias
                    ),
                    onDeferredReap: retainServerDuringDeferredReap
                )
                await refreshResults()
            } catch is CancellationError {
                errorMessage = acquiredReservation
                    ? "Benchmark stopped. No incomplete result was shared."
                    : "Benchmark request stopped before it started."
            } catch {
                errorMessage = error.localizedDescription
            }
            isRunning = false
            runTask = nil
        }
    }

    private func refreshResults() async {
        guard benchmarkCLIAvailable, let binary else { return }
        do {
            let data = try await CommunityBenchmarkCommand.run(
                binary: binary,
                arguments: CommunityBenchmarkCommand.benchmarkResultsArguments()
            )
            results = try JSONDecoder().decode(CommunityBenchmarkResults.self, from: data).runs
        } catch {
            if results.isEmpty { errorMessage = "Couldn’t read local results: \(error.localizedDescription)" }
        }
    }

    private func refreshBenchmarkCatalog() async {
        guard let binary else {
            benchmarkCLIAvailable = false
            errorMessage = "Community Benchmark needs the bundled rapid-mlx runtime. Restart Rapid, then try again."
            return
        }
        let memory = max(1, Int(MacHardware.detect().physicalRAMGB.rounded()))
        do {
            let data = try await CommunityBenchmarkCommand.run(
                binary: binary,
                arguments: ["benchmark", "catalog", "--memory-gib", String(memory), "--json"]
            )
            let envelope = try JSONDecoder().decode(
                CommunityBenchmarkCatalogEnvelope.self, from: data
            )
            var metadata: [String: CommunityBenchmarkCatalogModel] = [:]
            for model in envelope.models {
                guard metadata.updateValue(model, forKey: model.alias) == nil else {
                    throw CommunityBenchmarkCommand.Failure(
                        message: "Benchmark catalog contains duplicate alias \(model.alias)."
                    )
                }
            }
            benchmarkCLIAvailable = true
            benchmarkMetadata = metadata
            selectedAlias = CommunityBenchmarkModel.reconciledSelection(
                current: selectedAlias,
                models: models
            )
        } catch {
            benchmarkCLIAvailable = false
            benchmarkMetadata = [:]
            errorMessage = "Community Benchmark needs a current rapid-mlx runtime. Update or restart Rapid, then try again."
        }
    }
}
