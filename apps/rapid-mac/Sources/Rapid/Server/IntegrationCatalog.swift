import Foundation

struct IntegrationTarget: Codable, Identifiable, Equatable, Sendable {
    enum Kind: String, Codable, Sendable {
        case configWriter = "config_writer"
        case adapterProfile = "adapter_profile"
    }

    let id: String
    let name: String
    let kind: Kind
    let configPath: String?

    enum CodingKeys: String, CodingKey {
        case id, name, kind
        case configPath = "config_path"
    }
}

enum IntegrationCatalog {
    static func decode(_ data: Data) throws -> [IntegrationTarget] {
        try JSONDecoder().decode([IntegrationTarget].self, from: data)
    }

    /// Rows the Launch page leads with, in this order.
    ///
    /// The registry's own order is "config writers first, then adapter
    /// profiles", which is an implementation detail of how the sidecar
    /// assembles the list — it put Cline at the top, a client most users do
    /// not have, and buried Codex at position six behind LangChain. Neither
    /// position reflects how likely the row is to work.
    ///
    /// These two are pinned because they are the two the page can genuinely
    /// finish: paste the command and the client is running against the local
    /// server, nothing written, nothing left behind. Most of the rest either
    /// only print instructions or leave a config carrying a bearer that stops
    /// working at the next restart — worth showing, not worth leading with.
    static let leadingIntegrations = ["claude-code", "codex"]

    /// Applies ``leadingIntegrations`` and disturbs nothing else.
    ///
    /// Sorted on the original index as a tiebreaker rather than with a bare
    /// comparator. `sorted(by:)` is documented as not guaranteed stable, so
    /// unpinned rows could shuffle between renders of the same list. Today's
    /// implementation does preserve input order at this size — removing the
    /// tiebreaker does not change the result — so this is insurance against a
    /// standard-library change, not a fix for observed behaviour.
    static func displayOrdered(_ targets: [IntegrationTarget]) -> [IntegrationTarget] {
        func rank(_ id: String) -> Int {
            leadingIntegrations.firstIndex(of: id) ?? leadingIntegrations.count
        }
        return targets.enumerated()
            .sorted { lhs, rhs in
                let left = rank(lhs.element.id)
                let right = rank(rhs.element.id)
                return left == right ? lhs.offset < rhs.offset : left < right
            }
            .map(\.element)
    }

    static func load(binary: URL? = ServerLocator.find()) async -> [IntegrationTarget] {
        guard let binary else { return [] }
        return await Task.detached(priority: .utility) {
            let process = Process()
            process.executableURL = binary
            process.arguments = ["launch", "list", "--json"]
            var environment = ProcessInfo.processInfo.environment
            environment["RAPID_MLX_TELEMETRY"] = "0"
            process.environment = environment
            let output = Pipe()
            process.standardOutput = output
            process.standardError = FileHandle.nullDevice
            do {
                try process.run()
                let data = output.fileHandleForReading.readDataToEndOfFile()
                process.waitUntilExit()
                guard process.terminationStatus == 0 else { return [] }
                return (try? decode(data)) ?? []
            } catch {
                return []
            }
        }.value
    }
}
