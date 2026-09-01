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
                switch entry.kind {
                case .image: task = .imageGeneration
                case .video: task = .videoGeneration
                case .chat: task = .textGeneration
                case .audio: task = nil
                }
            } else {
                task = nil
            }
            guard let task else { return nil }
            let protocolName: String
            switch task {
            case .imageGeneration: protocolName = "Rapid Image Speed v1"
            case .videoGeneration: protocolName = "Rapid Video Speed v1"
            case .textGeneration: protocolName = "Rapid Community Speed v1"
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
}

struct CommunityBenchmarkCatalogModel: Decodable, Sendable {
    let alias: String
    let focus: Bool
    let estimatedMemoryGib: Int?
    let memoryFit: String

    enum CodingKeys: String, CodingKey {
        case alias, focus
        case estimatedMemoryGib = "estimated_memory_gib"
        case memoryFit = "memory_fit"
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
    struct Measurement: Decodable { let totalDurationMS: Double
        enum CodingKeys: String, CodingKey { case totalDurationMS = "total_duration_ms" }
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
        return average >= 1_000
            ? String(format: "%.1f s avg", average / 1_000)
            : String(format: "%.0f ms avg", average)
    }

    var repoID: String { model.components.first?.source.repoID ?? "Local model" }
}

private final class BenchmarkProcessBox: @unchecked Sendable {
    let process = Process()
    private let lock = NSLock()
    private var cancelled = false

    func start() throws {
        lock.lock()
        defer { lock.unlock() }
        if cancelled { throw CancellationError() }
        try process.run()
    }

    func cancel() {
        lock.lock()
        defer { lock.unlock() }
        cancelled = true
        if process.isRunning { process.terminate() }
    }
}

private enum CommunityBenchmarkCommand {
    struct Failure: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    static func run(binary: URL, arguments: [String]) async throws -> Data {
        let box = BenchmarkProcessBox()
        return try await withTaskCancellationHandler {
            let data = try await Task.detached(priority: .userInitiated) {
                let identifier = UUID().uuidString
                let root = FileManager.default.temporaryDirectory
                    .appendingPathComponent("rapid-community-benchmark-\(identifier)", isDirectory: true)
                try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
                defer { try? FileManager.default.removeItem(at: root) }
                let stdoutURL = root.appendingPathComponent("stdout")
                let stderrURL = root.appendingPathComponent("stderr")
                FileManager.default.createFile(atPath: stdoutURL.path, contents: nil)
                FileManager.default.createFile(atPath: stderrURL.path, contents: nil)
                let stdout = try FileHandle(forWritingTo: stdoutURL)
                let stderr = try FileHandle(forWritingTo: stderrURL)
                defer {
                    try? stdout.close()
                    try? stderr.close()
                }
                box.process.executableURL = binary
                box.process.arguments = arguments
                box.process.standardOutput = stdout
                box.process.standardError = stderr
                box.process.standardInput = FileHandle.nullDevice
                try box.start()
                box.process.waitUntilExit()
                try stdout.synchronize()
                try stderr.synchronize()
                let output = try Data(contentsOf: stdoutURL)
                guard box.process.terminationStatus == 0 else {
                    let errorData = try Data(contentsOf: stderrURL)
                    let detail = String(data: errorData, encoding: .utf8)?
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    let message = detail.flatMap { $0.isEmpty ? nil : $0 }
                        ?? "Benchmark exited with code \(box.process.terminationStatus)."
                    throw Failure(message: message)
                }
                return output
            }.value
            try Task.checkCancellation()
            return data
        } onCancel: {
            box.cancel()
        }
    }
}

struct CommunityBenchmarkView: View {
    let catalog: [ModelEntry]
    let binary: URL?
    let prepareServer: () async -> Void
    let releaseServer: () -> Void

    @State private var selectedAlias = ""
    @State private var results: [CommunityBenchmarkResult] = []
    @State private var isRunning = false
    @State private var errorMessage: String?
    @State private var runTask: Task<Void, Never>?
    @State private var benchmarkMetadata: [String: CommunityBenchmarkCatalogModel] = [:]

    private var models: [CommunityBenchmarkModel] {
        CommunityBenchmarkModel.models(from: catalog, metadata: benchmarkMetadata)
    }

    private var selected: CommunityBenchmarkModel? {
        models.first { $0.entry.alias == selectedAlias } ?? models.first
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
                .disabled(!isRunning && (selected == nil || binary == nil))
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
        guard let selected, let binary else { return }
        errorMessage = nil
        isRunning = true
        runTask = Task {
            await prepareServer()
            defer { releaseServer() }
            guard !Task.isCancelled else {
                isRunning = false
                runTask = nil
                return
            }
            do {
                _ = try await CommunityBenchmarkCommand.run(
                    binary: binary,
                    arguments: ["benchmark", "run", selected.entry.alias, "--json"]
                )
                await refreshResults()
            } catch is CancellationError {
                errorMessage = "Benchmark stopped. No incomplete result was shared."
            } catch {
                errorMessage = error.localizedDescription
            }
            isRunning = false
            runTask = nil
        }
    }

    private func refreshResults() async {
        guard let binary else { return }
        do {
            let data = try await CommunityBenchmarkCommand.run(
                binary: binary, arguments: ["benchmark", "results", "--json"]
            )
            results = try JSONDecoder().decode(CommunityBenchmarkResults.self, from: data).runs
        } catch {
            if results.isEmpty { errorMessage = "Couldn’t read local results: \(error.localizedDescription)" }
        }
    }

    private func refreshBenchmarkCatalog() async {
        guard let binary else { return }
        let memory = max(1, Int(MacHardware.detect().physicalRAMGB.rounded()))
        do {
            let data = try await CommunityBenchmarkCommand.run(
                binary: binary,
                arguments: ["benchmark", "catalog", "--memory-gib", String(memory), "--json"]
            )
            let envelope = try JSONDecoder().decode(
                CommunityBenchmarkCatalogEnvelope.self, from: data
            )
            benchmarkMetadata = Dictionary(
                uniqueKeysWithValues: envelope.models.map { ($0.alias, $0) }
            )
            if selected == nil { selectedAlias = models.first?.entry.alias ?? "" }
        } catch {
            // The existing atomic Desktop catalog remains a safe fallback. A
            // sidecar from before this feature simply lacks the richer plan.
        }
    }
}
